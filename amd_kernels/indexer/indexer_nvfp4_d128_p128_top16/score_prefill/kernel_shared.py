# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Prefill index scorer over the NVFP4 index pages with one dequantized block per workgroup, FlyDSL (gfx950): per
(row, block) the max over the row's causal tokens of q . k with the init (1e30) / local (1e29) sentinels, on either
route: exact (keys the e2m1 x e4m3 product in f16, ``v_mfma_f32_32x32x16_f16``) or fp8 (keys
e4m3_rn_satfinite(e2m1 x scale), ``v_mfma_f32_32x32x64_f8f6f4``). The exact route runs it at every size, the fp8 route
on small work (``kernel.py`` is faster on large work).

H index heads (1, 2 or 4) share the one index-K head: a query row is a (token, head) pair, row = token * H + head, so
one dequantized block serves every head. A workgroup is 4 waves x 32 MQ query rows (a 128- or 256-row tile) x a
segment of its causal window.

Per block, wave w dequantizes token tile w (tokens 32 w .. +31) into the A fragments in LDS (lane (r, h): token 32 w +
r; exact k-step s = dims 64 h + 8 s .. +7 through ``v_cvt_scalef32_pk_f16_fp4`` and ``v_pk_mul_f16`` by the group
scale; fp8 k-step s = dims 64 h + 32 s .. +31, those f16 keys clamped to +-448 and converted to e4m3), double buffered
with one barrier per block. Its raw words come from global memory one block ahead, and the next block's dequant is
interleaved with this block's MFMAs. Every wave multiplies all four token tiles against its MQ 32-row query tiles (Q in
VGPRs: e4m3, or converted to f16 exactly). Blocks before the wave's first query position skip the causal mask, in a
loop of their own.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _mlir_ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..score_prefill.kernel import (
    BLK,
    PAGE_BYTES,
    PAGE_CODES,
    SENTINEL_INIT,
    SENTINEL_LOCAL,
    D,
    _as_f32,
    _bits,
    _global_i32_ptr,
    _maxf_nn,
    _permlane_swap,
    page_lookup,
)

NW = 4  # waves per workgroup
MQ_CHOICES = (1, 2)  # 32-row query tiles per wave: 128- or 256-row workgroup tiles
NT = BLK // 32  # 32-token tiles per block (tile w dequantized by wave w)
RING = 4  # fragment reads in flight ahead of the MFMA


def _f32(x):
    return fx.as_ir_value(fx.Float32(x))


def _cvt_fp4x2_f16(word, sel):
    """Byte ``sel`` of ``word`` (two e2m1, low nibble first) -> 2 x f16, exact."""
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp4(
            Vec.make_type(2, fx.Float16), fx.as_ir_value(word), _f32(1.0), sel
        )
    )


def _cvt_fp8x2_f16(word, hi):
    """Half ``hi`` of ``word`` (two e4m3) -> 2 x f16, exact."""
    return Vec(
        _rocdl_ops.cvt_scalef32_pk_f16_fp8(
            Vec.make_type(2, fx.Float16), fx.as_ir_value(word), _f32(1.0), hi
        )
    )


def _pack_f16x8(parts):
    """Four 2 x f16 -> the 8 x f16 operand of the f16 16x16x32 MFMA."""
    return Vec.from_elements(
        [fx.Int32(p.bitcast(fx.Int32)[0]) for p in parts], fx.Int32
    ).bitcast(fx.Float16)


def _sat448_fp8x2(x):
    """2 x f16 clamped to +-448 -> 2 x e4m3 (the hardware convert is not saturating)"""
    fm = _mlir_ir.Attribute.parse("#arith.fastmath<nnan>")
    hi = fx.as_ir_value(fx.Vector.filled(2, 448.0, fx.Float16))
    lo = fx.as_ir_value(fx.Vector.filled(2, -448.0, fx.Float16))
    v = _arith.MaxNumFOp(fx.as_ir_value(x), lo, fastmath=fm).result
    return _arith.MinNumFOp(v, hi, fastmath=fm).result


def _fp8_pair(w, sv):
    """Code dword w (8 e2m1, dims 8 apart) x splat scale sv -> 2 dwords of e4m3"""
    out = []
    for h in range_constexpr(2):
        p = fx.as_ir_value(fx.Vector.filled(2, 0, fx.Int16))
        for k in range_constexpr(2):
            x = _sat448_fp8x2(_cvt_fp4x2_f16(w, 2 * h + k) * sv)
            p = _rocdl_ops.cvt_scalef32_pk_fp8_f16(
                Vec.make_type(2, fx.Int16), p, x, _f32(1.0), k == 1
            )
        out.append(fx.Int32(Vec(p).bitcast(fx.Int32)[0]))
    return out


def _mfma_f16_32x32x16(a8, b8, acc16):
    """``v_mfma_f32_32x32x16_f16``: lane (r, h) gets D[8 (i // 4) + 4 h + i % 4][r]"""
    return Vec(
        _rocdl_ops.mfma_f32_32x32x16_f16_(
            Vec.make_type(16, fx.Float32),
            fx.as_ir_value(a8),
            fx.as_ir_value(b8),
            fx.as_ir_value(acc16),
            0,
            0,
            _mlir_ir.Attribute.parse("#rocdl<mfma_perm_b none>"),
        )
    )


def _mfma_fp8_32x32x64(a8, b8, acc16):
    """``v_mfma_f32_32x32x64_f8f6f4`` (e4m3 x e4m3), same lane layout as ``_mfma_f16_32x32x16``"""
    zero = fx.as_ir_value(fx.Int32(0))
    e4m3 = _mlir_ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    opsel0 = _mlir_ir.IntegerAttr.get(_mlir_ir.IntegerType.get_signless(32), 0)
    return Vec(
        _rocdl_ops.mfma_scale_f32_32x32x64_f8f6f4_(
            Vec.make_type(16, fx.Float32),
            fx.as_ir_value(a8),
            fx.as_ir_value(b8),
            fx.as_ir_value(acc16),
            e4m3,
            e4m3,
            opsel0,
            zero,
            opsel0,
            zero,
        )
    )


def _xlane_max32(x):
    """max(x[L], x[L ^ 32])"""
    a, b = _permlane_swap(_rocdl.permlane32_swap, _bits(x), _bits(x))
    return _maxf_nn(_as_f32(a), _as_f32(b))


def tile_q(mq: int) -> int:
    """Query rows per workgroup"""
    return NW * 32 * mq


def compile_score_prefill_shared(heads: int, mq: int, fp8: bool):
    """One kernel per index-head count (1, 2 or 4), 32-row query tiles per wave ``mq`` and route (``fp8`` keys, else
    exact f16); all sizes are runtime arguments."""
    assert heads in (1, 2, 4) and mq in MQ_CHOICES, (heads, mq)
    H, MQ = heads, mq
    ROWS_PER_WAVE = 32 * MQ
    TILE_Q = tile_q(MQ)
    KS = 2 if fp8 else D // 16  # MFMA k-steps per block row (K 64 / 16)
    FL = 32 if fp8 else 16  # fragment bytes per lane
    FRAG_BYTES = NT * KS * 64 * FL  # one block's A fragments (16 / 32 KB)
    mfma = "fp8" if fp8 else "f16"
    name = f"nvfp4_{mfma}mfma_index_score_prefill_shared_d128_p128_tq{TILE_Q}_h{H}"

    @fx.struct
    class Shared:
        frag: fx.Array[fx.Uint8, 2 * FRAG_BYTES, 16]  # A fragments, 2 blocks

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # idx_q [total_q, H, 128] e4m3, rows contiguous
        arg_kv: fx.Int64,  # NVFP4 index pages [num_blocks, 128, 72] u8
        arg_score: fx.Int64,  # score [H, total_q, S] fp32 (head stride i32_hs)
        arg_bt: fx.Int64,  # block table [batch, >= max_block] i32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1] i32
        arg_seq: fx.Int64,  # seq_lens [batch] i32
        i64_page_stride: fx.Int64,  # bytes between pages
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,  # query tiles per request
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i64_bt_bytes: fx.Int64,  # block table storage size (unused: the launch signature of every prefill scorer)
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_hs: fx.Int32,  # score head stride (elements)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        r32, h = lane % 32, lane // 32
        n_wg = i32_qt * i32_nseg * i32_nb
        wid = pid
        q_tile = wid % i32_qt
        seg = (wid // i32_qt) % i32_nseg
        b = wid // (i32_qt * i32_nseg)

        bc = (wid < n_wg).select(b, fx.Int32(0))  # padded workgroups read request 0
        cu = _global_i32_ptr(arg_cu)
        seq_start = fx.Int32(cu[bc])
        q_len = fx.Int32(cu[bc + 1]) - seq_start
        seq_len = fx.Int32(_global_i32_ptr(arg_seq)[bc])
        prefix = seq_len - q_len  # rows are aligned to the end of the request
        q_rows = q_len * H  # request-local query rows: token * H + head
        row0 = q_tile * TILE_Q
        if (wid < n_wg) & (row0 < q_rows):
            # causal window of this query tile, split into segments
            hi_pos = prefix + (row0 + TILE_Q) // H
            hi = (hi_pos < seq_len).select(hi_pos, seq_len)
            nblk = (hi + (BLK - 1)) // BLK
            seg_len = (nblk + i32_nseg - 1) // i32_nseg
            blk0 = seg * seg_len
            blk1e = blk0 + seg_len
            blk1 = (blk1e < nblk).select(blk1e, nblk)
            if blk0 < blk1:
                qr = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_total_q) * (H * D)
                )
                sr = buffer_ops.create_buffer_resource_from_addr(
                    arg_score,
                    num_records_bytes=(
                        fx.Int64(i32_hs) * (H - 1)
                        + fx.Int64(i32_total_q) * fx.Int64(i32_S)
                    )
                    * 4,
                )
                # past the segment: its last block
                page_of = page_lookup(arg_bt, b * i32_bt_stride, blk1)

                tok = wave * 32 + r32  # this lane's token of the wave's tile

                def load_raw(blk):
                    """Token tok's codes of dims 64 h .. +63 (two 16 B loads) and
                    its group scales 4 h .. 4 h + 3 (one dword) of block blk
                    (past the segment: no memory access, zeros)"""
                    ok = blk < blk1
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_kv + fx.Int64(page_of(blk)) * i64_page_stride,
                        num_records_bytes=PAGE_BYTES,
                    )
                    w = []
                    for i in range_constexpr(2):
                        v = Vec(
                            buffer_ops.buffer_load(
                                kr,
                                (tok * 64 + h * 32 + i * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=ok,
                            )
                        )
                        w += [fx.Int32(v[e]) for e in range_constexpr(4)]
                    sc = fx.Int32(
                        buffer_ops.buffer_load(
                            kr,
                            (PAGE_CODES + tok * 8 + h * 4) // 4,
                            vec_width=1,
                            dtype=fx.Int32,
                            mask=ok,
                        )
                    )
                    return w + [sc]

                frag128 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.frag.ptr),
                        fx.make_layout(2 * FRAG_BYTES // 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

                def frag_slot(buf, n, s, half=0):
                    """16 B slot of this lane's fragment (token tile n, k-step s;
                    fp8: its 16 B half) in buffer buf"""
                    return (
                        buf * (FRAG_BYTES // 16)
                        + ((n * KS + s) * 64 + lane) * (FL // 16)
                        + half
                    )

                def store16(words, slot):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(Vec.from_elements(words, fx.Int32))
                    fx.copy(atom16, r, fx.slice(frag128, (None, slot)))

                def load16(slot):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(atom16, fx.slice(frag128, (None, slot)), r)
                    return [fx.Int32(x) for x in Vec(r.load())]

                def dequant_step(raw, buf, s):
                    """k-step s of this wave's token tile -> buffer buf. Exact: dims
                    64 h + 8 s .. +7 = code dword s, group scale 4 h + s // 2; fp8:
                    dims 64 h + 32 s .. +31 = code dwords 4 s .. +3, groups 4 h +
                    2 s, + 1"""
                    if const_expr(fp8):
                        scl = _cvt_fp8x2_f16(raw[8], s == 1)
                        sv = [
                            Vec.from_elements([scl[g], scl[g]], fx.Float16)
                            for g in (0, 1)
                        ]
                        k = []
                        for e in range_constexpr(4):
                            k += _fp8_pair(raw[4 * s + e], sv[e // 2])
                        store16(k[:4], frag_slot(buf, wave, s, 0))
                        store16(k[4:], frag_slot(buf, wave, s, 1))
                    else:
                        scl = _cvt_fp8x2_f16(raw[8], s >= 4)
                        sc = scl[(s // 2) % 2]
                        sv = Vec.from_elements([sc, sc], fx.Float16)
                        f = _pack_f16x8(
                            [_cvt_fp4x2_f16(raw[s], k) * sv for k in range(4)]
                        )
                        store16(
                            [fx.Int32(x) for x in f.bitcast(fx.Int32)],
                            frag_slot(buf, wave, s),
                        )

                def read_frag(buf, n, s):
                    if const_expr(fp8):
                        return Vec.from_elements(
                            load16(frag_slot(buf, n, s, 0))
                            + load16(frag_slot(buf, n, s, 1)),
                            fx.Int32,
                        )
                    return Vec.from_elements(
                        load16(frag_slot(buf, n, s)), fx.Int32
                    ).bitcast(fx.Float16)

                mfma = _mfma_fp8_32x32x64 if fp8 else _mfma_f16_32x32x16

                # this wave's query rows (request-local): row0 + wave*64 + 32 m + r32 = token * H + head; rows past
                # q_len * H are zero and never stored
                rows = [row0 + wave * ROWS_PER_WAVE + 32 * m + r32 for m in range(MQ)]
                qpos = [prefix + rw // H for rw in rows]  # absolute query positions
                # Q B operands: lane (r32, h), dims 64 h .. +63
                qf = []
                for m in range_constexpr(MQ):
                    qrow = (seq_start * H + rows[m]) * D
                    d = []
                    for i in range_constexpr(4):
                        v = Vec(
                            buffer_ops.buffer_load(
                                qr,
                                (qrow + h * 64 + i * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=rows[m] < q_rows,
                            )
                        )
                        d += [fx.Int32(v[e]) for e in range_constexpr(4)]
                    if const_expr(fp8):  # k-step s = dims 64 h + 32 s .. +31
                        qf.append(
                            [
                                Vec.from_elements(d[8 * s : 8 * s + 8], fx.Int32)
                                for s in range(KS)
                            ]
                        )
                    else:  # k-step s = dims 64 h + 8 s .. +7, exactly to f16
                        qf.append(
                            [
                                _pack_f16x8(
                                    [
                                        _cvt_fp8x2_f16(d[2 * s], False),
                                        _cvt_fp8x2_f16(d[2 * s], True),
                                        _cvt_fp8x2_f16(d[2 * s + 1], False),
                                        _cvt_fp8x2_f16(d[2 * s + 1], True),
                                    ]
                                )
                                for s in range(KS)
                            ]
                        )

                zero16 = fx.Vector.filled(16, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                sent_init = fx.Float32(SENTINEL_INIT)
                sent_local = fx.Float32(SENTINEL_LOCAL)
                st_tok = [rw // H for rw in rows]
                st_out = [
                    (rw % H) * i32_hs + (seq_start + t) * i32_S
                    for rw, t in zip(rows, st_tok)
                ]
                st_nb = [qp // BLK + 1 for qp in qpos]  # the row's causal blocks

                # the wave's first query position: blocks ending at or before it
                # need no causal mask
                qpos_min = prefix + (row0 + wave * ROWS_PER_WAVE) // H

                def max16(acc):
                    v = acc[0]
                    for i in range_constexpr(1, 16):
                        v = _maxf_nn(v, acc[i])
                    return v

                def max16_masked(acc, lim):
                    v = neg_inf
                    for i in range_constexpr(16):
                        x = (lim >= 8 * (i // 4) + i % 4).select(acc[i], neg_inf)
                        v = _maxf_nn(v, x)
                    return v

                def block(blk, buf, raw_next, masked):
                    """Scores of block blk (fragments in buffer buf) for this wave's
                    rows (``masked``: the causal mask, a block on the diagonal);
                    dequantizes raw_next into the other buffer meanwhile"""
                    runs = [neg_inf] * MQ
                    jobs = [(n, s) for n in range(NT) for s in range(KS)]
                    ring = [None] * len(jobs)
                    for j in range_constexpr(RING - 1):
                        ring[j] = read_frag(buf, *jobs[j])
                    for n in range_constexpr(NT):
                        accs = [zero16] * MQ
                        for s in range_constexpr(KS):
                            j = n * KS + s
                            if const_expr(j + RING - 1 < len(jobs)):
                                ring[j + RING - 1] = read_frag(buf, *jobs[j + RING - 1])
                            if const_expr(j % NT == 0):
                                dequant_step(raw_next, 1 - buf, j // NT)
                            for m in range_constexpr(MQ):
                                accs[m] = mfma(ring[j], qf[m][s], accs[m])
                        base = blk * BLK + 32 * n + 4 * h
                        for m in range_constexpr(MQ):
                            if const_expr(masked):
                                mx = max16_masked(accs[m], qpos[m] - base)
                            else:
                                mx = max16(accs[m])
                            runs[m] = _maxf_nn(runs[m], mx)
                    for m in range_constexpr(MQ):
                        v = _xlane_max32(runs[m])
                        is_local = (blk >= st_nb[m] - i32_local) & (blk < st_nb[m])
                        is_init = (blk < i32_init) & (blk < st_nb[m])
                        v = is_local.select(sent_local, is_init.select(sent_init, v))
                        buffer_ops.buffer_store(
                            v,
                            sr,
                            st_out[m] + blk,
                            mask=(rows[m] < q_rows) & (h == 0),
                        )

                # prologue: block blk0's fragments in buffer 0, blk0 + 1's raw
                raw0 = load_raw(blk0)
                for s in range_constexpr(KS):
                    dequant_step(raw0, 0, s)
                r0, r1, r2, r3, r4, r5, r6, r7, r8 = load_raw(blk0 + 1)
                fx.gpu.barrier()

                # blocks below dsplit end at or before the wave's first query position (no causal mask); two loops,
                # not a branch per block (one straight-line copy each keeps the wave at <= 256 registers). The waves
                # split at different blocks but pass the same barriers.
                dfirst = (qpos_min + 1) // BLK  # first block with a token > qpos_min
                dsplit = (dfirst < blk0).select(blk0, dfirst)
                dsplit = (dsplit > blk1).select(blk1, dsplit)
                blk = blk0
                buf = fx.Int32(0)
                while blk < dsplit:
                    cur = [r0, r1, r2, r3, r4, r5, r6, r7, r8]
                    r0, r1, r2, r3, r4, r5, r6, r7, r8 = load_raw(blk + 2)
                    block(blk, buf, cur, False)
                    # publish the next block's fragments; every wave is done
                    # reading this buffer before it is overwritten
                    fx.gpu.barrier()
                    blk = blk + 1
                    buf = 1 - buf
                while blk < blk1:
                    cur = [r0, r1, r2, r3, r4, r5, r6, r7, r8]
                    r0, r1, r2, r3, r4, r5, r6, r7, r8 = load_raw(blk + 2)
                    block(blk, buf, cur, True)
                    fx.gpu.barrier()
                    blk = blk + 1
                    buf = 1 - buf

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_score,
            arg_bt,
            arg_cu,
            arg_seq,
            i64_page_stride,
            i32_bt_stride,
            i32_S,
            i32_nseg,
            i32_qt,
            i32_total_q,
            i32_nb,
            i64_bt_bytes,
            i32_init,
            i32_local,
            i32_hs,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
