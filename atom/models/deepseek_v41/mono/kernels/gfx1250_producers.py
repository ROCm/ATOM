# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Verified K1 producer compute bodies, using caller-owned LDS."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from flydsl.expr import range_constexpr
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from atom.models.deepseek_v41.mono.kernels.gfx1250_norm_wqkv import (
    HIDDEN,
    _load_bf8,
    _weighted_bf16,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_qkv import (
    Q_RANK,
    KV_DIM,
    QKV_DIM,
    ROPE,
    EPS,
    _dual_total,
)
from atom.models.deepseek_v41.mono.kernels.mx import code_round, fp8_value
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, pow2, rcp_pow2
from atom.mono.device.ops import (
    bf_hi,
    bf_lo,
    bf16_round,
    div_rn,
    fp8_pack4,
    hw_rsq,
    readlane,
    traced,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_mhc import _typed


def _mhc_weighted_bf16(value, weight):
    # The Triton delayed mHC seam rounds the weight product to FP32 before
    # converting to BF16. An opaque mul prevents mixed-BF16 contraction.
    product = fx.Float32(
        llvm.inline_asm(
            T.f32,
            [value.ir_value(), weight.ir_value()],
            "v_mul_f32 $0, $1, $2",
            "=v,v,v",
            has_side_effects=False,
        )
    )
    return bf16_round(product)


@traced
def hidden_norm_device(
    x,
    norm_weight,
    normed,
    quantized,
    codes,
    lds,
    tid,
    row,
    rows,
    mhc_style=False,
    partial=None,
):
    lane, wave = tid % 32, tid // 32
    ld32, st32 = make_lds_copy_ops(32)
    rx = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(x), num_records_bytes=rows * HIDDEN * 2
    )
    rw = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(norm_weight), num_records_bytes=HIDDEN * 2
    )
    rn = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(normed), num_records_bytes=rows * HIDDEN * 2
    )
    rq = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(quantized), num_records_bytes=rows * HIDDEN
    )
    rs = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(codes), num_records_bytes=rows * HIDDEN // 32
    )
    cols = [tid * 8 + 2048 * ch for ch in range(3)]
    weights = [_load_bf8(rw, fx.min(cols[ch], HIDDEN - 8)) for ch in range(3)]
    loaded = [
        _load_bf8(rx, row * HIDDEN + fx.min(cols[ch], HIDDEN - 8)) for ch in range(3)
    ]
    xs = [
        [(cols[ch] < HIDDEN).select(v, fx.Float32(0.0)) for v in loaded[ch]]
        for ch in range(3)
    ]
    if const_expr(mhc_style and partial is not None):
        pm = _typed(partial, fx.Float32)
        pieces = [
            fx.ptr_load(pm + (row * 160 + lane + i * 32) * 26 + 25) for i in range(5)
        ]
        # Match native NUM_KSPLIT160's register tree and final lane31 reduce.
        total = ((pieces[0] + pieces[2]) + pieces[4]) + (pieces[1] + pieces[3])
        for off in (8, 4, 2, 1, 16):
            total = total + total.shuffle_xor(off, 32)
        total = readlane(total, 31)
    else:
        accum = fx.Float32(0.0)
        for ch in range_constexpr(3):
            for j in range_constexpr(8):
                accum = accum + xs[ch][j] * xs[ch][j]
        for off in range_constexpr(5):
            accum = accum + accum.shuffle_xor(1 << off, 32)
        if lane == 0:
            st32(lds, wave * 4, Vec.from_elements([accum.bitcast(fx.Int32)]))
        fx.gpu.barrier()
        partials = [fx.Int32(ld32(lds, i * 4)[0]).bitcast(fx.Float32) for i in range(8)]
        pairs = [partials[2 * i] + partials[2 * i + 1] for i in range(4)]
        total = (pairs[0] + pairs[1]) + (pairs[2] + pairs[3])
    rstd = hw_rsq(fx.fma(total, fx.Float32(1.0 / HIDDEN), fx.Float32(1e-20)))
    for ch in range_constexpr(3):
        if const_expr(mhc_style):
            ys = [
                _mhc_weighted_bf16(xs[ch][j] * rstd, weights[ch][j]) for j in range(8)
            ]
        else:
            ys = [_weighted_bf16(xs[ch][j] * rstd, weights[ch][j]) for j in range(8)]
        amax = fx.Float32(0.0)
        for y in ys:
            amax = fx.max(amax, abs(y))
        amax = fx.max(amax, amax.shuffle_xor(1, 32))
        amax = fx.max(amax, amax.shuffle_xor(2, 32))
        code = code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX))
        mul = rcp_pow2(code)
        if cols[ch] < HIDDEN:
            col = row * HIDDEN + cols[ch]
            for j in range_constexpr(8):
                bo.buffer_store(ys[j].to(fx.BFloat16), rn, col + j)
            for j in range_constexpr(2):
                bo.buffer_store(
                    fp8_pack4(*[clamp_fp8(ys[4 * j + k] * mul) for k in range(4)]),
                    rq,
                    col // 4 + j,
                )
            if tid % 4 == 0:
                bo.buffer_store(
                    code.to(fx.Int8), rs, row * (HIDDEN // 32) + cols[ch] // 32
                )

    fx.gpu.barrier()


@traced
def qkv_norm_device(
    qkv,
    qnw,
    kvnw,
    cos,
    sin,
    qout,
    qsout,
    kvout,
    lds,
    tid,
    row,
    rows,
    q_only=False,
    kv_only=False,
):
    """Dual producer by default; K1 may schedule Q and KV independently."""
    lane, wave = tid % 32, tid // 32
    ld32, st32 = make_lds_copy_ops(32)
    rx = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(qkv), num_records_bytes=rows * QKV_DIM * 2
    )
    rq = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(qnw), num_records_bytes=Q_RANK * 2
    )
    rk = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(kvnw), num_records_bytes=KV_DIM * 2
    )
    rc = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(cos), num_records_bytes=rows * ROPE
    )
    rs = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(sin), num_records_bytes=rows * ROPE
    )
    qo = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(qout), num_records_bytes=rows * Q_RANK
    )
    qso = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(qsout), num_records_bytes=rows * Q_RANK // 32
    )
    ko = bo.create_buffer_resource_from_addr(
        fx.ptrtoint(kvout), num_records_bytes=rows * KV_DIM * 2
    )
    if const_expr(not kv_only):
        col = tid * 8
        qc = fx.min(col, Q_RANK - 8)
        qvals = _load_bf8(rx, row * QKV_DIM + qc)
        xq = [(col < Q_RANK).select(v, fx.Float32(0.0)) for v in qvals]
        qw = _load_bf8(rq, qc)
    if const_expr(not q_only):
        kw = fx.Int32(bo.buffer_load(rk, tid, vec_width=1, dtype=T.i32))
        kv = fx.Int32(
            bo.buffer_load(
                rx, (row * QKV_DIM + Q_RANK) // 2 + tid, vec_width=1, dtype=T.i32
            )
        )
        xk = [bf_lo(kv), bf_hi(kv)]
        weights = [bf_lo(kw), bf_hi(kw)]
        phase = tid - (KV_DIM - ROPE) // 2
        table_col = fx.max(phase, 0)
        table_index = row * (ROPE // 2) + table_col
        cw = fx.Int32(bo.buffer_load(rc, table_index // 2, vec_width=1, dtype=T.i32))
        sw = fx.Int32(bo.buffer_load(rs, table_index // 2, vec_width=1, dtype=T.i32))
        cosine = (phase >= 0).select(
            (table_index % 2 == 0).select(bf_lo(cw), bf_hi(cw)), fx.Float32(1.0)
        )
        sine = (phase >= 0).select(
            (table_index % 2 == 0).select(bf_lo(sw), bf_hi(sw)), fx.Float32(0.0)
        )

    # Initialize both values for FlyDSL's dynamic lane branch, including
    # compile-time specializations that do not compute one of the norms.
    sq, sk = fx.Float32(0.0), fx.Float32(0.0)
    if const_expr(not kv_only):
        sq = xq[0] * xq[0]
        for j in range_constexpr(1, 8):
            sq = sq + xq[j] * xq[j]
    if const_expr(not q_only):
        sk = xk[0] * xk[0] + xk[1] * xk[1]
    for offset in (8, 4, 2, 1, 16):
        if const_expr(not kv_only):
            sq = sq + sq.shuffle_xor(offset, 32)
        if const_expr(not q_only):
            sk = sk + sk.shuffle_xor(offset, 32)
    if lane == 31:
        if const_expr(not kv_only):
            st32(lds, wave * 4, Vec.from_elements([sq.bitcast(fx.Int32)]))
        if const_expr(not q_only):
            st32(lds, 32 + wave * 4, Vec.from_elements([sk.bitcast(fx.Int32)]))
    fx.gpu.barrier()
    if const_expr(not kv_only):
        qp = [fx.Int32(ld32(lds, i * 4)[0]).bitcast(fx.Float32) for i in range(8)]
        nq = hw_rsq(
            div_rn(_dual_total(qp), fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
            + fx.Float32(EPS)
        )
        # Q deliberately stays FP32 until MXFP8 packing.
        ys = [(xq[j] * nq) * qw[j] for j in range(8)]
        amax = fx.Float32(0.0)
        for y in ys:
            amax = fx.max(amax, abs(y))
        for offset in (1, 2):
            amax = fx.max(amax, amax.shuffle_xor(offset, 32))
        code = code_round(amax)
        mul = pow2(254 - code)
        if col < Q_RANK:
            bo.buffer_store(
                fp8_pack4(*[ys[j] * mul for j in range(4)]),
                qo,
                (row * Q_RANK + col) // 4,
            )
            bo.buffer_store(
                fp8_pack4(*[ys[j + 4] * mul for j in range(4)]),
                qo,
                (row * Q_RANK + col) // 4 + 1,
            )
            if tid % 4 == 0:
                bo.buffer_store(code.to(fx.Int8), qso, row * (Q_RANK // 32) + col // 32)
    if const_expr(not q_only):
        kp = [fx.Int32(ld32(lds, 32 + i * 4)[0]).bitcast(fx.Float32) for i in range(8)]
        nk = hw_rsq(
            div_rn(_dual_total(kp), fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
            + fx.Float32(EPS)
        )

        kn = [bf16_round((xk[j] * nk) * weights[j]) for j in range(2)]
        kr = [
            bf16_round(kn[0] * cosine - kn[1] * sine),
            bf16_round(kn[0] * sine + kn[1] * cosine),
        ]
        amax = fx.max(abs(kr[0]), abs(kr[1]))
        for offset in (1, 2, 4, 8):
            amax = fx.max(amax, amax.shuffle_xor(offset, 32))
        code = code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX))
        step, mul = pow2(code), rcp_pow2(code)
        scaled = [clamp_fp8(v * mul) for v in kr]
        qat = [fp8_value(v) * step for v in scaled]
        for j in range_constexpr(2):
            bo.buffer_store(qat[j].to(fx.BFloat16), ko, row * KV_DIM + tid * 2 + j)

    fx.gpu.barrier()
