# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gumbel block partials of one DSpark Markov draft step: base + bias -> bf16 cache row, Gumbel noise, block argmax.

Grid (n rows, 196 / G); 256 threads; a workgroup does G consecutive 1024-column blocks of one row, thread t holds
columns 4t .. 4t + 3 of each block (the layout whose reduction tree the argmax mirrors, see ``reference.py``).
The row's temperature picks one of three straight-line bodies (uniform branch): T 0 or an invalid row (no noise),
T 1 (noise, no division), any other T (IEEE fp32 division, then noise).
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import NB, V

F, I = fx.Float32, fx.Int32
NEG_INF = float("-inf")
_C = [float(struct.unpack("<f", struct.pack("<f", 1.0 / k))[0]) for k in range(1, 9)]


def _fbits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


_LN2_HI, _LN2_LO = _fbits(0x3F317217), _fbits(0x3377D1CF)


def _i32(x):
    return x - (1 << 32) if x >= (1 << 31) else x


def _log_normal(x):
    """Triton's ``tl.log`` (LLVM's llvm.log.f32 expansion on gfx950) for finite x >= 2^-126: the input scaling and
    the infinity select of the expansion are dead there, so they are left out (same bits)."""
    y = fx.math.log2(x, fastmath=fx.FastMathFlags.afn)
    r0 = y * F(_LN2_HI)
    f0 = fx.math.fma(y, F(_LN2_HI), r0 * F(-1.0))
    f1 = fx.math.fma(y, F(_LN2_LO), f0)
    return fx.math.fma(y, F(_LN2_HI), f1)


def _lshr(x, s):
    return (x >> s) & ((1 << (32 - s)) - 1)


def _rotl(x, s):
    return (x << s) | _lshr(x, 32 - s)


def _mix(h, key, key_c1=None):
    """murmur3's mix; ``key_c1`` = key * 0xCC9E2D51 (mod 2^32) when the caller has it already."""
    key = key * _i32(0xCC9E2D51) if key_c1 is None else key_c1
    key = _rotl(key, 15)
    key = key * _i32(0x1B873593)
    h = h ^ key
    h = _rotl(h, 13)
    return (h << 2) + h + _i32(0xE6546B64)  # h * 5: a shift-add, not a quarter-rate v_mul_lo_u32


def _fmix(h):
    h = h ^ _lshr(h, 16)
    h = h * _i32(0x85EBCA6B)
    h = h ^ _lshr(h, 13)
    h = h * _i32(0xC2B2AE35)
    return h ^ _lshr(h, 16)


def _noise(h0, col_c1):
    """fp32 Gumbel noise of the vocab id whose product with 0xCC9E2D51 is ``col_c1`` (murmur3_uniform32, then
    -log(-_log1p_neg_stable(u)))."""
    r = _fmix(_mix(h0, None, col_c1) ^ 16)
    u = _lshr(r, 16).to(F) * F(1.52587890625e-05) + ((r & 0xFFFF).to(F) + F(0.5)) * F(2.3283064365386963e-10)
    p = F(_C[7])
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = fx.math.fma(u, p, F(_C[k - 1]))
    series = (u * F(-1.0)) * p
    one_m = F(1.0) - u
    direct = _log_normal((one_m > F(5.960464477539063e-08)).select(one_m, F(5.960464477539063e-08)))
    l1 = (u < F(0.25)).select(series, direct)
    return _log_normal(l1 * F(-1.0)) * F(-1.0)


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return I(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full))


def _swap(op, v):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False)
    return I(_llvm.extractvalue(T.i32, res, [0])), I(_llvm.extractvalue(T.i32, res, [1]))


def _xor_lane(v, m, upper):
    """Lane ``l ^ m`` of the wave as an i32; ``upper``: this lane's bit m set (m = 16, 32)."""
    if m == 1:
        return _dpp(v, 0xB1)  # quad_perm [1, 0, 3, 2]
    if m == 2:
        return _dpp(v, 0x4E)  # quad_perm [2, 3, 0, 1]
    if m == 4:  # row_shr:4 into banks 1, 3, then row_shl:4 into banks 0, 2
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp(v, 0x128)  # row_ror:8
    a, b = _swap(_rocdl_ops.permlane16_swap if m == 16 else _rocdl_ops.permlane32_swap, v)
    return upper.select(a, b)


def _comb(v1, i1, v2, i2):
    """Triton's argmax combine (tie_break_left): keep (v1, i1) iff v1 > v2 or (v1 == v2 and i1 < i2)."""
    keep = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return keep.select(v1, v2), keep.select(i1, i2)


def _ge(v1, i1, v2, i2):
    """The in-thread pair step (i1 < i2 known): keep (v1, i1) iff v1 >= v2."""
    keep = v1 >= v2
    return keep.select(v1, v2), keep.select(i1, i2)


def compile_gumbel_partials(blocks_per_wg: int):
    G = blocks_per_wg
    assert NB % G == 0
    name = f"gumbel_partials_bias_bf16_v200064_g{G}"

    @fx.struct
    class Shared:
        mx: fx.Array[fx.Int32, 4 * G]
        ix: fx.Array[fx.Int32, 4 * G]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # bf16 [n, >= V] base rows (row stride i32_sx elements)
        i32_sx: fx.Int32,
        arg_bias: fx.Int64,  # bf16 [*, V] bias rows
        arg_brows: fx.Int64,  # int64 [n]: bias row of each row
        arg_idx: fx.Int64,  # int32 [n]: request slot, < 0 = invalid row
        arg_temp: fx.Int64,  # fp32 [slots]
        arg_seed: fx.Int64,  # int64 [slots]
        arg_pos: fx.Int64,  # int64 [n]
        i32_pos_add: fx.Int32,  # added to the low word of pos: 2^30 (_DRAFT_NOISE_SALT) + the caller's offset
        arg_cache: fx.Int64,  # bf16 [slots, steps, V], contiguous
        i32_cs0: fx.Int32,  # cache stride 0 / V (= steps)
        i32_step: fx.Int32,  # draft step = cache column
        arg_lmax: fx.Int64,  # fp32 [n, NB]
        arg_larg: fx.Int64,  # int64 [n, NB]
    ):
        tx, row, wgy = fx.thread_idx.x, fx.block_idx.x, fx.block_idx.y
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        sr = buffer_ops.create_buffer_resource_from_addr(arg_idx + fx.Int64(row) * 4, num_records_bytes=4)
        slot = I(buffer_ops.buffer_load(sr, 0, vec_width=1, dtype=I))
        valid = slot >= 0
        slot0 = valid.select(slot, I(0))
        tr = buffer_ops.create_buffer_resource_from_addr(arg_temp + fx.Int64(slot0) * 4, num_records_bytes=4)
        temp = valid.select(I(buffer_ops.buffer_load(tr, 0, vec_width=1, dtype=I)).bitcast(F), F(0.0))
        sdr = buffer_ops.create_buffer_resource_from_addr(arg_seed + fx.Int64(slot0) * 8, num_records_bytes=8)
        sd = Vec(buffer_ops.buffer_load(sdr, 0, vec_width=2, dtype=I))
        seed_lo = valid.select(I(sd[0]), I(0))
        seed_hi = valid.select(I(sd[1]), I(0)) ^ i32_step  # spec_step_seed: seed ^ (step << 32)
        pr = buffer_ops.create_buffer_resource_from_addr(arg_pos + fx.Int64(row) * 8, num_records_bytes=8)
        pos_lo = I(buffer_ops.buffer_load(pr, 0, vec_width=1, dtype=I)) + i32_pos_add  # low word of pos + salt
        brr = buffer_ops.create_buffer_resource_from_addr(arg_brows + fx.Int64(row) * 8, num_records_bytes=8)
        brow = I(buffer_ops.buffer_load(brr, 0, vec_width=1, dtype=I))
        h0 = _mix(_mix(_mix(I(0), seed_lo), seed_hi), pos_lo)
        blk0 = wgy * G
        c_wg = blk0 * 1024
        n_wg = ((V - c_wg) < 1024 * G).select(V - c_wg, I(1024 * G))
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x + (fx.Int64(row) * fx.Int64(i32_sx) + fx.Int64(c_wg)) * 2, num_records_bytes=n_wg * 2)
        br = buffer_ops.create_buffer_resource_from_addr(
            arg_bias + (fx.Int64(brow) * V + fx.Int64(c_wg)) * 2, num_records_bytes=n_wg * 2)
        cr = buffer_ops.create_buffer_resource_from_addr(
            arg_cache + ((fx.Int64(slot0) * fx.Int64(i32_cs0) + fx.Int64(i32_step)) * V + fx.Int64(c_wg)) * 2,
            num_records_bytes=valid.select(n_wg * 2, I(0)))
        upper16 = (lane & 16) != 0
        upper32 = (lane & 32) != 0
        is_t0 = tx == 0
        lmr = buffer_ops.create_buffer_resource_from_addr(
            arg_lmax + (fx.Int64(row) * NB + fx.Int64(blk0)) * 4, num_records_bytes=is_t0.select(I(4 * G), I(0)))
        lar = buffer_ops.create_buffer_resource_from_addr(
            arg_larg + (fx.Int64(row) * NB + fx.Int64(blk0)) * 8, num_records_bytes=is_t0.select(I(8 * G), I(0)))
        xw = [Vec(buffer_ops.buffer_load(xr, g * 512 + tx * 2, vec_width=2, dtype=I)) for g in range_constexpr(G)]
        bw = [Vec(buffer_ops.buffer_load(br, g * 512 + tx * 2, vec_width=2, dtype=I)) for g in range_constexpr(G)]
        # bf16(base + bias) per column, the cache store, the column ids and the in-range flags
        # columns >= V load 0 (buffer bounds) and their cache stores drop; their values are set to -inf below
        ys, cols, oks, kc1 = [], [], [], []
        col_t = c_wg + tx * 4
        c1_t = col_t * _i32(0xCC9E2D51)
        for g in range_constexpr(G):
            sums = []
            for k in range_constexpr(4):
                w_x, w_b = I(xw[g][k // 2]), I(bw[g][k // 2])
                xb = (w_x << 16) if k % 2 == 0 else (w_x & -65536)
                bb = (w_b << 16) if k % 2 == 0 else (w_b & -65536)
                col = col_t + (g * 1024 + k)
                sums.append(xb.bitcast(F) + bb.bitcast(F))
                cols.append(col)
                oks.append(col < V)
                kc1.append(c1_t + _i32(((g * 1024 + k) * 0xCC9E2D51) & 0xFFFFFFFF))
            pk = [I(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(sums[2 * i]), fx.as_ir_value(sums[2 * i + 1])))
                  for i in range_constexpr(2)]
            buffer_ops.buffer_store(Vec.from_elements(pk, I), cr, g * 512 + tx * 2)
            for k in range_constexpr(4):
                ys.append(((pk[k // 2] << 16) if k % 2 == 0 else (pk[k // 2] & -65536)).bitcast(F))

        def body(noisy, divide):
            for g in range_constexpr(G):
                vv = []
                for k in range_constexpr(4):
                    j = 4 * g + k
                    v = ys[j] / temp if divide else ys[j]
                    if noisy:
                        v = v + _noise(h0, kc1[j])
                    vv.append(oks[j].select(v, F(NEG_INF)))
                cg = cols[4 * g:4 * g + 4]
                p01 = _ge(vv[0], cg[0], vv[1], cg[1])
                p23 = _ge(vv[2], cg[2], vv[3], cg[3])
                av, ai = _comb(p01[0], p01[1], p23[0], p23[1])
                for m in (32, 16, 8, 4, 2, 1):
                    up = upper32 if m == 32 else upper16
                    ov = _xor_lane(av.bitcast(I), m, up).bitcast(F)
                    oi = _xor_lane(ai, m, up)
                    av, ai = _comb(av, ai, ov, oi)
                # Triton keeps lane 63's wave result
                wv = I(fx.rocdl.readlane(T.i32, fx.as_ir_value(av.bitcast(I)), fx.as_ir_value(I(63))))
                wi = I(fx.rocdl.readlane(T.i32, fx.as_ir_value(ai), fx.as_ir_value(I(63))))
                smem.mx.ptr[g * 4 + wave] = wv
                smem.ix.ptr[g * 4 + wave] = wi
            fx.gpu.barrier()
            ms, cs = [], []
            for g in range_constexpr(G):
                w = [(I(smem.mx.ptr[g * 4 + q]).bitcast(F), I(smem.ix.ptr[g * 4 + q])) for q in range_constexpr(4)]
                a0 = _comb(w[0][0], w[0][1], w[2][0], w[2][1])
                a1 = _comb(w[1][0], w[1][1], w[3][0], w[3][1])
                m, ci = _comb(a0[0], a0[1], a1[0], a1[1])
                ms.append(m.bitcast(I))
                cs.extend([ci, I(0)])
            buffer_ops.buffer_store(Vec.from_elements(ms, I) if G > 1 else ms[0], lmr, 0)
            for q in range_constexpr((2 * G + 3) // 4):
                part = cs[4 * q:4 * q + 4]
                buffer_ops.buffer_store(Vec.from_elements(part, I), lar, 4 * q)

        if temp == F(0.0):
            body(False, False)
        else:
            if temp == F(1.0):
                body(True, False)
            else:
                body(True, True)

    @flyc.jit
    def launch_gumbel_partials_bias(
        arg_x: fx.Int64, i32_sx: fx.Int32, arg_bias: fx.Int64, arg_brows: fx.Int64, arg_idx: fx.Int64,
        arg_temp: fx.Int64, arg_seed: fx.Int64, arg_pos: fx.Int64, i32_pos_add: fx.Int32, arg_cache: fx.Int64,
        i32_cs0: fx.Int32, i32_step: fx.Int32, arg_lmax: fx.Int64, arg_larg: fx.Int64, i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_x, i32_sx, arg_bias, arg_brows, arg_idx, arg_temp, arg_seed, arg_pos, i32_pos_add, arg_cache,
               i32_cs0, i32_step, arg_lmax, arg_larg).launch(grid=(fx.Int64(i32_n), NB // G, 1), block=(256, 1, 1),
                                                    stream=stream)

    return launch_gumbel_partials_bias
