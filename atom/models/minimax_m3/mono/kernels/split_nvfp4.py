# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4's split-K sparse attention over an NVFP4 K/V cache (single-buffer pages), one
256-key partition x 16 heads a task, f16 MFMA over the exact dequantized products.

The page (128 tokens, ``KV_PAGE_BYTES`` = 18432 B at one kv head) is aiter's
``minimax_m3_sparse_pa_nvfp4`` one (``common.py`` here: ``nvfp4_k_store`` documents it),
and a wave's work is that kernel's wave: 32 keys, the quarter ``q`` of a block,

  S^T = K Q^T   A = K tile (row r of tile st is key 8 (r // 4) + 4 st + r % 4 of the
                quarter), B = Q^T (column = query head); lane (n = lane % 16, g =
                lane // 16) holds keys 8 g + 4 st + e (st < 2, e < 4) of head n;
  O^T += V^T P^T  A = V^T tile dt (row n' is dim 8 n' + dt), B = P^T taken unchanged
                from the S^T accumulators (lane (n, g) supplies keys 8 g .. 8 g + 7).

K and V elements are cvt(e2m1) x the e4m3 scale in f16: exact (<= 6 significant
bits), so nothing is rounded before the MFMA; Q goes bf16 -> f16 and P is f16.
Scores are in the natural-log domain like the fp8 split's (``sm_scale * k_scale *
dot``), a task puts the partition's normalized output, max and sum the merge stage
takes. The task's page list is the fp8 split's (``blk``: 16 page-16 ids, then the
context): page-16 id p is block p // 8, keys 16 (p % 8) of it, so a wave's two
entries are one quarter."""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.minimax_m3.mono.config import HEAD_DIM
from atom.models.minimax_m3.mono.kernels.common import KV_PAGE_BYTES
from atom.models.minimax_m3.mono.kernels.stages.shared import NEG
from atom.models.minimax_m3.mono.layout import N_SPLIT, O_K, SPLIT_KEYS, H
from atom.mono.device.ops import (
    hw_exp2,
    kernel_symbol,
    launcher,
    memrealtime,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES

PPW = SPLIT_KEYS // WAVES // 16  # page-16 entries a wave
KPW = SPLIT_KEYS // WAVES  # keys a wave (32)
_K_SCALES = 8192  # a head's K scale region (Hkv = 1)
_V_DATA = 9216
_V_SCALES = 9216 + 8192
_CURRENT_STREAM = fx.Stream(None)
_V = fx.Vector


def _cvt_fp4x2_f16(word, sel):
    return _V(
        rocdl.cvt_scalef32_pk_f16_fp4(
            _V.make_type(2, fx.Float16),
            as_ir_value(fx.Int32(word)),
            as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _cvt_fp8x2_f16(word, hi):
    return _V(
        rocdl.cvt_scalef32_pk_f16_fp8(
            _V.make_type(2, fx.Float16),
            as_ir_value(fx.Int32(word)),
            as_ir_value(fx.Float32(1.0)),
            hi,
        )
    )


def _mfma_f16(a4, b4, acc):
    a = _V(a4).bitcast(fx.Float16)
    b = _V(b4).bitcast(fx.Float16)
    return _V(
        rocdl.mfma_f32_16x16x32_f16(_V.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0])
    )


def _pk_f16(a, b):
    """Two f32 -> a dword of two f16 (RNE), ``a`` low."""
    return (
        _V.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
        .to(fx.Float16)
        .bitcast(fx.Int32)[0]
    )


_NEG_BITS = struct.unpack("i", struct.pack("f", NEG))[0]


def _masked(x, left):
    """``x``, or NEG where ``left`` < 0 (the key is past the context): a bit select
    by the sign of ``left``, so a NaN ``x`` of a stale key is replaced too."""
    dead = fx.Int32(left) >> 31
    bits = x.bitcast(fx.Int32)
    return (bits ^ ((bits ^ fx.Int32(_NEG_BITS)) & dead)).bitcast(fx.Float32)


@traced
def load_quarter(r_page, w, n, g, cm=0):
    """This lane's raw dwords of quarter ``w`` of the page ``r_page`` (a buffer
    resource of its 18432 B): K data of its two key rows (dims 32 g .. + 31), their
    K scale dword, V data (dword dt = 8 keys 8 g .. + 7 of dim 8 n + dt) and the V
    scale bytes of those keys' group n // 2 (``cm``: the loads' cache modifier,
    ``CM_DEV`` for a page this launch's K1 wrote). ->  (kd[2][4], ks, vd[8], vs[2])."""
    kd = []
    for st in range_constexpr(2):
        row = w * KPW + 8 * (n // 4) + 4 * st + n % 4
        kd.append(
            _V(
                bo.buffer_load(
                    r_page,
                    (row * 64 + g * 16) // 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=cm,
                )
            )
        )
    ks = fx.Int32(
        bo.buffer_load(
            r_page,
            (_K_SCALES + ((w * 4 + g) * 16 + n) * 4) // 4,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=cm,
        )
    )
    vbase = (_V_DATA + ((w * 4 + g) * 128 + 8 * n) * 4) // 4
    vd = [
        _V(
            bo.buffer_load(
                r_page, vbase + 4 * h, vec_width=4, dtype=T.i32, cache_modifier=cm
            )
        )
        for h in range_constexpr(2)
    ]
    vs = _V(
        bo.buffer_load(
            r_page,
            (_V_SCALES + (n // 2) * 128 + w * KPW + 8 * g) // 4,
            vec_width=2,
            dtype=T.i32,
            cache_modifier=cm,
        )
    )
    return kd, ks, vd, vs


@traced
def dequant_quarter(raw, key_mask):
    """K fragments ``kf[st][s]`` (A operands of S^T: dims 32 g + 8 s .. + 7) and V^T
    fragments ``vf[dt]`` in f16, exact products. ``key_mask[i]``: the mask dword
    (0xFFFF per live key) of V^T operand dword i (keys 8 g + 2 i, + 1): a key past
    the context may hold stale bytes (a NaN scale), its P is 0 but 0 x NaN is not."""
    kd, ks, vd, vs = raw
    kf = []
    for st in range_constexpr(2):
        ksc = _cvt_fp8x2_f16(ks, st == 1)  # [group 2 g, 2 g + 1] of row st
        row = []
        for s in range_constexpr(4):
            sv = _V.from_elements([ksc[s // 2], ksc[s // 2]], fx.Float16)
            row.append(
                _V.from_elements(
                    [
                        (_cvt_fp4x2_f16(kd[st][s], i) * sv).bitcast(fx.Int32)[0]
                        for i in range_constexpr(4)
                    ],
                    fx.Int32,
                )
            )
        kf.append(row)
    scales = []
    for h in range_constexpr(2):
        scales += [_cvt_fp8x2_f16(vs[h], False), _cvt_fp8x2_f16(vs[h], True)]
    words = [
        vd[0][0],
        vd[0][1],
        vd[0][2],
        vd[0][3],
        vd[1][0],
        vd[1][1],
        vd[1][2],
        vd[1][3],
    ]
    # V data dword dt: a byte = two consecutive keys of dim 8 n + dt (nibble 0 the even)
    vf = []
    for dt in range_constexpr(8):
        vf.append(
            _V.from_elements(
                [
                    (
                        (_cvt_fp4x2_f16(words[dt], i) * scales[i]).bitcast(fx.Int32)[0]
                        & key_mask[i]
                    )
                    for i in range_constexpr(4)
                ],
                fx.Int32,
            )
        )
    return kf, vf


@traced
def split_nvfp4_task(
    tid, lane, wave, tok, t, blk, q, kv_addr, k_gs, v_gs, sm_scale, hst, opart, n_ctx,
    put_o, put_m, put_l, cm=0, stamp=None,
):  # fmt: skip
    """Task (token ``tok``, partition ``t``): the partition's output, max and sum.
    ``blk``: LDS ptr of its 16 page-16 ids; ``q``: Int64 address of the bf16 query
    rows [token][O_K]; ``kv_addr``: the cache's Int64 base; ``k_gs`` / ``v_gs``: the
    layer's global scales (f32); ``hst`` / ``opart``: LDS as the fp8 split's;
    ``put_o(e0, ov)``, ``put_m(head, m)``, ``put_l(head, l)``; ``cm``: the cache
    modifier of the page and query loads; ``stamp(k)``: the K4 timeline's points 12 -
    15 (the dequantized tiles ready, the max exchanged, the PV done, the partition
    summed). Runs under barriers every thread of the CTA reaches."""
    l16 = lane % 16
    g4 = lane // 16
    c = fx.Float32(sm_scale) * k_gs
    # this wave's quarter: its first entry's block and quarter, scalars
    p16 = uniform(fx.ptr_load(blk + wave * PPW))
    block = p16 // 8
    quarter = (p16 % 8) // 2
    r_page = rsrc(kv_addr + fx.Int64(block) * KV_PAGE_BYTES, KV_PAGE_BYTES)
    raw = load_quarter(r_page, quarter, l16, g4, cm)
    # Q^T fragments: head l16, dims 32 g + 8 s .. + 7, bf16 -> f16
    qb = []
    r_q = rsrc(q)
    for s in range_constexpr(4):
        w4 = _V(
            bo.buffer_load(
                r_q,
                (tok * O_K + l16 * HEAD_DIM + 32 * g4 + 8 * s) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=cm,
            )
        )
        qb.append(
            w4.bitcast(fx.BFloat16).to(fx.Float32).to(fx.Float16).bitcast(fx.Int32)
        )
    key0 = t * SPLIT_KEYS + wave * KPW
    # Masks by integer arithmetic, not compares: a compare's result is a pair of
    # scalar registers, and the layer kernel has none to spare (with eight of them
    # live the build needed a private segment). ``nl``: this lane's keys of its 8 in
    # the context; a V operand dword of two keys has 16 bits a live key.
    nl = n_ctx - (key0 + 8 * g4)
    key_mask = [
        fx.Int32(
            (
                (fx.Int64(1) << (fx.Int64(fx.min(fx.max(nl - 2 * i, 0), 2)) * 16))
                - fx.Int64(1)
            )
            & fx.Int64(0xFFFFFFFF)
        )
        for i in range(4)
    ]
    kf, vf = dequant_quarter(raw, key_mask)
    if stamp is not None:
        stamp(12)
    sc = []
    for st in range_constexpr(2):
        acc = _V.filled(4, 0.0, fx.Float32)
        for s in range_constexpr(4):
            acc = _mfma_f16(kf[st][s], qb[s], acc)
        sc.append([_masked(acc[e] * c, nl - 1 - (4 * st + e)) for e in range(4)])
    # head l16's max over this wave's 32 keys: its 8, then the four lane groups
    m = sc[0][0]
    for st in range_constexpr(2):
        for e in range_constexpr(4):
            m = fx.max(m, sc[st][e])
    m = fx.max(m, xshfl(m, 16))
    m = fx.max(m, xshfl(m, 32))
    if g4 == 0:
        fx.ptr_store(m, hst + (wave * H + l16))
    gpu.barrier()
    if stamp is not None:
        stamp(13)
    mx = fx.ptr_load(hst + l16)
    for w in range_constexpr(1, WAVES):
        mx = fx.max(mx, fx.ptr_load(hst + (w * H + l16)))
    p = [[hw_exp2((sc[st][e] - mx) * LOG2E) for e in range(4)] for st in range(2)]
    ls = p[0][0]
    for st in range_constexpr(2):
        for e in range_constexpr(4):
            if (st, e) != (0, 0):
                ls = ls + p[st][e]
    ls = ls + xshfl(ls, 16)
    ls = ls + xshfl(ls, 32)
    if g4 == 0:
        fx.ptr_store(ls, hst + (WAVES * H + wave * H + l16))
    pt = _V.from_elements(
        [_pk_f16(p[st][2 * h], p[st][2 * h + 1]) for st in range(2) for h in range(2)],
        fx.Int32,
    )
    # O^T tile dt: lane (n, g) rows 4 g + e = dims 32 g + 8 e + dt of head n
    for e in range_constexpr(4):
        outs = []
        for dt in range_constexpr(8):
            cv = _mfma_f16(vf[dt], pt, _V.filled(4, 0.0, fx.Float32))
            outs.append(cv[e])
        base = opart + (wave * O_K + l16 * HEAD_DIM + 32 * g4 + 8 * e)
        fx.ptr_store(_V.from_elements(outs[:4], fx.Float32), base)
        fx.ptr_store(_V.from_elements(outs[4:], fx.Float32), base + 4)
    if tid < H:
        fx.ptr_store(mx, hst + (2 * WAVES * H + tid))  # tid < 16: wave 0's head max
    gpu.barrier()
    if stamp is not None:
        stamp(14)
    # the partition: sum the waves' partials, normalized by its sum, the v scale
    e0 = tid * 4
    head = e0 // HEAD_DIM
    lsum = fx.ptr_load(hst + (WAVES * H + head))
    for w in range_constexpr(1, WAVES):
        lsum = lsum + fx.ptr_load(hst + (WAVES * H + w * H + head))
    inv_l = 1.0 / (lsum > 0.0).select(lsum, fx.Float32(1.0))
    ov = []
    for k in range_constexpr(4):
        acc = fx.ptr_load(opart + (e0 + k))
        for w in range_constexpr(1, WAVES):
            acc = acc + fx.ptr_load(opart + (w * O_K + e0 + k))
        ov.append((acc * inv_l) * v_gs)
    if stamp is not None:
        stamp(15)
    put_o(e0, ov)
    if tid < H:
        put_m(tid, fx.ptr_load(hst + (2 * WAVES * H + tid)))
        lt = fx.ptr_load(hst + (WAVES * H + tid))
        for w in range_constexpr(1, WAVES):
            lt = lt + fx.ptr_load(hst + (WAVES * H + w * H + tid))
        put_l(tid, lt)
    gpu.barrier()


def build_split_nvfp4_kernel(sm_scale: float, tasks: int, timed: bool = False):
    """``@flyc.jit`` launcher of ``tasks`` split tasks alone, one CTA each (K4's
    harness): task ts = (token ts // N_SPLIT, partition ts % N_SPLIT); ``tables``
    [tasks][32] i32 (16 page-16 ids, n_ctx at 16), ``q`` bf16 [token][O_K], the cache,
    the f32 global scales [k, v]; writes ``sp_o`` bf16 [ts][O_K], ``sp_m`` / ``sp_l``
    f32 [ts][H]. ``timed``: ``sp_l`` is followed by an i32 [tasks] of the task's
    ``s_memrealtime`` ticks (100 MHz), first load of its page list to its last put."""
    assert 1 <= tasks <= BLOCKS

    @fx.struct
    class Smem:
        hst: fx.Array[fx.Float32, 3 * WAVES * H + WAVES, 16]
        opart: fx.Array[fx.Float32, WAVES * O_K, 16]
        blk: fx.Array[fx.Int32, 32, 16]

    kernel_name = kernel_symbol(
        "minimax_m3_split_nvfp4", s=tasks, sm=int(sm_scale * 1e6), t=int(timed)
    )

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def split_kernel(
        tables: Int64, q: Int64, kv: Int64, gs: Int64, sp_o: Int64, sp_m: Int64,
        sp_l: Int64, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        ts = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        blk = lds.blk.ptr
        if tid < 17:
            fx.ptr_store(
                fx.Int32(
                    bo.buffer_load(
                        rsrc(tables), ts * 32 + tid, vec_width=1, dtype=T.i32
                    )
                ),
                blk + tid,
            )
        gpu.barrier()
        t0 = memrealtime()
        n_ctx = uniform(fx.ptr_load(blk + 16))
        kg = fx.Float32(bo.buffer_load(rsrc(gs), 0, vec_width=1, dtype=T.f32))
        vg = fx.Float32(bo.buffer_load(rsrc(gs), 1, vec_width=1, dtype=T.f32))

        def put_o(e0, ov):
            words = []
            for j in range_constexpr(2):
                words.append(
                    _V.from_elements([ov[2 * j], ov[2 * j + 1]], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
            bo.buffer_store(
                _V.from_elements(words, fx.Int32), rsrc(sp_o), (ts * O_K + e0) // 2
            )

        def put_m(head, m):
            bo.buffer_store(m, rsrc(sp_m), ts * H + head)

        def put_l(head, lt):
            bo.buffer_store(lt, rsrc(sp_l), ts * H + head)

        split_nvfp4_task(
            tid, lane, wave, ts // N_SPLIT, ts % N_SPLIT, blk, q, kv, kg, vg,
            sm_scale, lds.hst.ptr, lds.opart.ptr, n_ctx, put_o, put_m, put_l,
        )  # fmt: skip
        if const_expr(timed):  # noqa: SIM102
            if tid == 0:
                bo.buffer_store(fx.Int32(memrealtime() - t0), rsrc(tl), ts)

    @launcher(kernel_name)
    def launch_split(
        tables: Int64, q: Int64, kv: Int64, gs: Int64, sp_o: Int64, sp_m: Int64,
        sp_l: Int64, tl: Int64, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = (sm_scale, tasks)  # build parameters
        split_kernel(tables, q, kv, gs, sp_o, sp_m, sp_l, tl).launch(
            grid=(tasks,), block=(THREADS,), stream=stream
        )

    return launch_split
