# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Megakernel M2a (``ATOM_MONO_MLA_POST=1``): an MLA layer's attention tail as
the first stage of its own mono K2 launch -- no new launch, no grid barrier.

Without it, between ``kn_mla_reduce`` and the MLA layer's K2, four kernels run:
the v-up batched GEMM (aiter triton ``_batched_gemm_a8w8_a_per_token_group_
prequant_w_per_batched_tensor_quant``), ``g_proj`` (FlyDSL a8w8 GEMM, ptpc FP8),
``sigmoid`` and ``mul``. They produce K2's ``core``:

    core[t, 128 h + v] = bf16( av[t, h, v] * bf16(sigmoid(gate[t, 128 h + v])) )
    av   = bf16( b_scale * sum_c fma-chain( dot_c(q8(olat[t, h, c]), W_V[h, v, c]), a_scale[t, h, c] ) )
    gate = bf16( (sum_k x8[t, k] * Wg8[n, k]) * xs[t] * ws[n] )

with ``olat`` the MLA decode's latent output [S, 12, 512] bf16, ``q8`` the
triton kernel's per-token 128-group FP8 quant (scale = max|a| / 448), ``x8`` /
``xs`` the input RMSNorm's fused per-token FP8 activation, ``Wg8`` / ``ws``
g_proj's preshuffled per-token FP8 plane.

Placement (``mla_post_plan.CTA_RANGES``): 96 tasks of 16 core columns (one
g_proj row tile of K = 7168 and one 16-row tile of one head's W_V, K = 512) on
the CTAs with the least other work queued behind them; every CTA issues its
o_proj weights first. Each task publishes a tagged pair in ``mpdone`` after its
device-scope core stores (``publish``, the K1 XRDY pattern). Every CTA's o_proj
-- which needs every core column, exactly as it needed the previous kernel's
whole output -- polls the 96 pairs one thread each (one round trip after the
last lands) and reads ``core`` at device scope. That is the only hand-off, and
it replaces a kernel boundary that already existed: there is no grid-wide
barrier.

Numerics: the elementwise tail (sigmoid with libm ``expf`` and IEEE division,
the bf16 roundings, the product) is the torch kernels' bit for bit given equal
``gate`` and ``av``. The two GEMMs accumulate the same exact FP8 products in a
different order (bf16 MFMAs of the exactly widened FP8 operands, wave-chunked K)
than the triton / FlyDSL kernels: fp32-ulp differences, rarely one bf16 ulp of
``gate`` / ``av``. The v-up quantization itself (amax, scale, reciprocal, RNE
E4M3) is the triton kernel's operation for operation, NaN handling included
(a NaN is ignored by the amax and the clamp).
"""

import struct

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import arith as farith
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.kernels.stages.mla_post_plan import (  # noqa: F401
    CTA_RANGES,
    G_GROUPS,
    GROUP_COLS,
    HEADS,
    HIDDEN,
    KCH,
    LORA,
    MAX_ROWS,
    PROJ,
    PT,
    ROWS,
    V_CH,
    VH,
    XPAD,
    lds_words,
    scratch_pairs,
    slabs,
    task_cta,
    task_rows,
    w8_dword,
)
from atom.mono.device import gemv
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fp8x8_bf16_pk,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import THREADS, WAVES


def _fmaxnum(a, b):
    """max that ignores a NaN operand (``arith.maxnumf``), as triton's
    ``tl.maximum`` / ``tl.max`` / ``tl.clamp`` do by default; ``fx.max`` propagates
    NaN (``arith.maximumf``)."""
    return fx.Float32(
        farith.maxnumf(
            farith.as_ir_value(fx.Float32(a)), farith.as_ir_value(fx.Float32(b))
        )
    )


def _fminnum(a, b):
    """min that ignores a NaN operand (``arith.minnumf``); see ``_fmaxnum``."""
    return fx.Float32(
        farith.minnumf(
            farith.as_ir_value(fx.Float32(a)), farith.as_ir_value(fx.Float32(b))
        )
    )


def _bf16_rne(x: float) -> float:
    """``x`` as f32, then rounded to bf16 (nearest even), as a float."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


assert (ROWS, KCH, GROUP_COLS) == (gemv.ROWS, gemv.KCH, gemv.KGROUP)
GCH = HIDDEN // gemv.KCH  # 56 g_proj K chunks
G_PER = GCH // WAVES  # 7 a wave
FP8_MAX = 448.0
INV_FP8_MAX = 1.0 / FP8_MAX  # the triton kernel's one_over_DTYPE_MAX (f32-rounded)
# tl.maximum(bf16 amax, 1e-10): the floor rounded to bf16 as triton promotes it
AMAX_FLOOR = _bf16_rne(1e-10)

REGIONS = ("mpdone",)

ARGS = ("olat", "olat_sr", "w_v", "w_v_s", "xg", "xg_s", "w_g", "w_g_s")


def task_of(bid):
    """This CTA's post task, or -1 (traced; ``mla_post_plan.cta_task``)."""
    j = fx.Int32(-1)
    for cta0, j0, n in CTA_RANGES:
        j = ((bid >= cta0) & (bid < cta0 + n)).select(bid - cta0 + j0, j)
    return j


# ---- FP8 weight operands: AITER's (16, 16) preshuffle of a [N, K] E4M3 plane,
# read as the bf16 GEMV's operands (``gemv.chunk_ops``' layout, 8 fp8 a lane)
def token_tiles(s):
    """``[(t0, n)]``: the 16-token tiles of ``s`` rows, one at s <= 16."""
    return [(t0, min(ROWS, s - t0)) for t0 in range(0, s, ROWS)]


def w8_operand(op):
    """A ``w8_loads`` operand (two dwords of E4M3) -> the bf16 operand."""
    return fp8x8_bf16_pk([op[0], op[1]])


def w8_scale(s_ptr, row):
    """Weight row ``row``'s fp32 scale."""
    return fx.Float32(bo.buffer_load(rsrc(s_ptr), row, vec_width=1, dtype=T.f32))


def w8_loads(lane, wave, w, row0, k):
    """``gemv.loads`` of the preshuffled E4M3 plane (``w8_dword``): this wave's
    chunks of weight rows ``row0`` .. + 16, K = ``k``, two dwords an operand."""
    per, n = gemv.chunks(k)
    r_w = rsrc(w)
    out = []
    for i in range(per):
        ch = fx.min(wave + WAVES * i, n - 1)
        base = w8_dword(row0, lane, k, ch * KCH)
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 128 * q,
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                for q in range(4)
            ]
        )
    return out


def _f32(v):
    return fx.Float32(v)


@traced
def post_task(c, j):
    """Task ``j``: core columns 16 j .. 16 j + 15 for every token -> ``core``
    (device-scope stores), then its ``mpdone`` pair."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    xl = c["post_xl"]
    plan = c["post_slabs"]
    g = lane // ROWS
    row0 = j * ROWS
    h = j // (VH // ROWS)
    vr0 = (j % (VH // ROWS)) * ROWS
    # ---- weights first: g_proj's 16 rows (7 chunks a wave), W_V's 16 rows
    gops = w8_loads(lane, wave, a["w_g"], row0, HIDDEN)
    vch = fx.min(wave, V_CH - 1)
    vrow = h * VH + vr0 + lane % ROWS
    vbase = (vrow * LORA + vch * gemv.KCH + g * 8) // 4
    r_v = rsrc(a["w_v"])
    vops = [
        fx.Vector(
            bo.buffer_load(
                r_v, vbase + 8 * q, vec_width=2, dtype=T.i32, cache_modifier=CM_NT
            )
        )
        for q in range(4)
    ]
    # ---- the latent rows (bf16, written by the previous kernel): plain loads
    tiles = token_tiles(s)
    ts = [t0 + fx.min(lane % ROWS, nt - 1) for t0, nt in tiles]
    r_o = rsrc(a["olat"])
    avals = [
        [
            fx.Vector(
                bo.buffer_load(
                    r_o,
                    (ts[jt] * a["olat_sr"] + h * LORA + vch * gemv.KCH + g * 8 + 32 * q)
                    // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for q in range(4)
        ]
        for jt in range(len(tiles))
    ]
    # ---- g_proj: x (fp8) a slab at a time into LDS, widened exactly to bf16
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in tiles]
    xrow = max(g1 - g0 for g0, g1 in plan) * GROUP_COLS + XPAD
    for sl in range_constexpr(len(plan)):
        g0 = plan[sl][0]
        g1 = plan[sl][1]
        if const_expr(sl):
            gpu.barrier()
        gemv.rows_to_lds(
            tid,
            a["xg"],
            (g1 - g0) * GROUP_COLS // 2,
            s,
            xl,
            xrow // 2,
            0,
            ldk=HIDDEN // 2,
            k0=g0 * GROUP_COLS // 2,
        )
        for i in range_constexpr(g0, g1):
            kk = ((i - g0) * WAVES + wave) * gemv.KCH + g * 8
            for q in range_constexpr(4):
                wa = w8_operand(gops[i][q])
                for jt in range_constexpr(len(tiles)):
                    d = fx.Vector(
                        fx.ptr_load(
                            xl + (ts[jt] * xrow + kk + 32 * q) // 4,
                            result_type=fx.Vector.make_type(2, fx.Int32),
                        )
                    )
                    xb = fp8x8_bf16_pk([d[0], d[1]])
                    accs[jt] = mfma_bf16(wa, xb, accs[jt])
    # ---- v-up: the triton kernel's per-token 128-group FP8 quant, in registers.
    # Lanes lane % 16 == t of the 4 lane groups hold token t's 128 values of
    # this wave's chunk; their amax is the group's. The amax and the clamp ignore
    # a NaN as triton's do: the MLA decode can return whole NaN latent columns,
    # which the production v-up turns into finite values, and a propagated NaN
    # amax would make every value of the group (so the whole core) NaN.
    vaccs = []
    vscs = []
    for jt in range_constexpr(len(tiles)):
        vals = []
        for q in range_constexpr(4):
            for e in range_constexpr(4):
                w = avals[jt][q][e]
                vals += [bf_lo(w), bf_hi(w)]
        lm = abs(vals[0])
        for e in range_constexpr(1, 32):
            lm = _fmaxnum(lm, abs(vals[e]))
        am = butterfly(lm, (16, 32), _fmaxnum)
        m = _fmaxnum(am, _f32(AMAX_FLOOR))
        sc = m * _f32(INV_FP8_MAX)
        rcp = _f32(1.0) / sc
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for q in range_constexpr(4):
            qv = [
                _fminnum(_fmaxnum(vals[8 * q + e] * rcp, _f32(-FP8_MAX)), _f32(FP8_MAX))
                for e in range(8)
            ]
            d0 = fp8_pack4(qv[0], qv[1], qv[2], qv[3])
            d1 = fp8_pack4(qv[4], qv[5], qv[6], qv[7])
            acc = mfma_bf16(w8_operand(vops[q]), fp8x8_bf16_pk([d0, d1]), acc)
        vaccs.append(acc)
        vscs.append(sc)
    # ---- epilogue a token tile at a time
    xs_r = rsrc(a["xg_s"])
    bsc = fx.Float32(bo.buffer_load(rsrc(a["w_v_s"]), 0, vec_width=1, dtype=T.f32))
    r_core = rsrc(c["args"]["core"])
    for jt in range_constexpr(len(tiles)):
        u0 = tiles[jt][0]
        n = tiles[jt][1]
        # v-up: waves 0 .. 3 the chunk dots, their scales past them in red
        if wave < V_CH:
            fx.ptr_store(vaccs[jt], red + (wave * 64 + lane) * 4)
            if g == 0:
                fx.ptr_store(
                    vscs[jt], red + (V_CH * 64 * 4 + wave * ROWS + lane % ROWS)
                )
        gpu.barrier()
        if tid < n * ROWS:
            tt = tid // ROWS
            r = tid % ROWS
            tot = _f32(0.0)
            for cc in range_constexpr(V_CH):
                dv = fx.ptr_load(red + ((cc * 64 + 16 * (r // 4) + tt) * 4 + r % 4))
                sv = fx.ptr_load(red + (V_CH * 64 * 4 + cc * ROWS + tt))
                tot = _f32(fmath.fma(dv, sv, tot))
            av = bf16_round(tot * bsc)
            fx.ptr_store(av.bitcast(fx.Int32), xl + (tt * ROWS + r))
        gpu.barrier()
        # g_proj: every wave's partial
        fx.ptr_store(accs[jt], red + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < n * (ROWS // 2):
            tt = tid // (ROWS // 2)
            rp = tid % (ROWS // 2)
            t = u0 + tt
            xs = fx.Float32(bo.buffer_load(xs_r, t, vec_width=1, dtype=T.f32))
            outs = []
            for e in range_constexpr(2):
                r = 2 * rp + e
                gsum = row_sum(red, r, tt)
                gate = bf16_round((gsum * xs) * w8_scale(a["w_g_s"], row0 + r))
                # torch's bf16 sigmoid: opmath 1 / (1 + expf(-g)), rounded to bf16
                sg = bf16_round(_f32(1.0) / (_f32(1.0) + fmath.exp(-gate)))
                av = fx.ptr_load(xl + (tt * ROWS + r)).bitcast(fx.Float32)
                outs.append(av * sg)
            bo.buffer_store(
                bf16_pair(outs[0], outs[1]).bitcast(fx.Int32),
                r_core,
                (t * PROJ + row0 + 2 * rp) // 2,
                cache_modifier=CM_DEV,
            )
        gpu.barrier()
    publish(c["put"], c["mpdone"], j, fx.Int32(1), tid == 0)


@traced
def wait_core(c):
    """Every post task's ``mpdone`` pair (one thread a task, one round trip)."""
    if c["tid"] < PT:
        c["poll"]([(c["mpdone"], c["tid"], 1)])
    gpu.barrier()


@traced
def load_core_dev(c, t0, n, ok, lds_pad):
    """Core rows t0 .. t0 + n (S x ``ok`` bf16, written by this launch's post
    tasks) -> LDS rows 0 .., at device scope (past the per-XCD L2)."""
    tid = c["tid"]
    words = n * ok // 8
    row = ok + lds_pad
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["core"]),
                (t0 * (ok // 8) + e) * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        t = e // (ok // 8)
        k = e % (ok // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * row + k) // 2)
    gpu.barrier()
