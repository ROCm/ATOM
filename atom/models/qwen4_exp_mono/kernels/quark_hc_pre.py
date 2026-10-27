# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Quark HC down projection using packed BF16 dot2 and one K reduction.

Combined streams and normalization retain the native BF16 endpoints. The
five chunk dot products accumulate before the final K reduction. An explicit
wait after inline dot2 protects the dependent DPP reduction on gfx950.
"""

import triton
import triton.language as tl

from atom.model_ops.qwen4_exp.ops.hc_fused import _hc_row_chunk


@triton.jit
def _hc_w_chunk(w_ptr, ns, nmask, k0, KTOT: tl.constexpr, BK: tl.constexpr):
    ks = k0 + tl.arange(0, BK)
    values = tl.load(
        w_ptr + ns[:, None].to(tl.int64) * KTOT + ks[None, :],
        mask=nmask[:, None],
        other=0,
    )
    lo, hi = tl.split(
        values.to(tl.int16, bitcast=True)
        .to(tl.int32)
        .reshape(values.shape[0], BK // 2, 2)
    )
    return (lo & 65535) | (hi << 16)


@triton.jit
def _hc_norm_dot(
    x,
    w,
    rstd,
    norm_w_ptr,
    h_out_ptr,
    normed_ptr,
    k0,
    pid_n,
    COMBINE: tl.constexpr,
    BK: tl.constexpr,
):
    ks = k0 + tl.arange(0, BK)
    g = tl.load(norm_w_ptr + ks).to(tl.float32) + 1.0
    xn = (x * rstd * g).to(normed_ptr.dtype.element_ty)
    if pid_n == 0:
        tl.store(normed_ptr + ks, xn)
        if COMBINE:
            tl.store(h_out_ptr + ks, x.to(h_out_ptr.dtype.element_ty))
    lo, hi = tl.split(xn.to(tl.int16, bitcast=True).to(tl.int32).reshape(BK // 2, 2))
    packed = (lo & 65535) | (hi << 16)
    products = tl.inline_asm_elementwise(
        "v_dot2_f32_bf16 $0, $1, $2, $3\n s_nop 2",
        constraints="=v,v,v,v",
        args=[w, packed[None, :], tl.full((1,), 0, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return products


@triton.jit
def _hc_pre_dot2(
    h_ptr,  # [HC*H] bf16, previous residual streams
    block_ptr,  # [H] bf16, previous sub-layer output (COMBINE only)
    raw_ptr,  # [HC] bf16, previous injection logits (COMBINE only)
    norm_w_ptr,  # [HC*H] norm weight (Gemma offset applied here)
    w_ptr,  # [NOUT, HC*H] bf16, concatenated [down | inject]
    h_out_ptr,  # [HC*H] bf16, combined streams (COMBINE only)
    normed_ptr,  # [HC*H] bf16
    part_ptr,  # [M, HC, NOUT] fp32
    eps,
    stride_h,
    stride_block,
    stride_raw,
    H: tl.constexpr,
    HC: tl.constexpr,
    NOUT: tl.constexpr,
    COMBINE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Program (N tile, stream, row). The stream's five K chunks are unrolled
    so every weight and activation load is issued before any arithmetic."""
    tl.static_assert(H == 5 * BK)
    pid_n = tl.program_id(0)
    s = tl.program_id(1)
    row = tl.program_id(2).to(tl.int64)
    h_ptr += row * stride_h
    h_out_ptr += row * stride_h
    normed_ptr += row * stride_h
    block_ptr += row * stride_block
    raw_ptr += row * stride_raw
    part_ptr += row * (HC * NOUT)
    base = s * H
    ns = pid_n * BN + tl.arange(0, BN)
    nmask = ns < NOUT
    KT: tl.constexpr = HC * H
    w0 = _hc_w_chunk(w_ptr, ns, nmask, base + 0 * BK, KT, BK)
    w1 = _hc_w_chunk(w_ptr, ns, nmask, base + 1 * BK, KT, BK)
    w2 = _hc_w_chunk(w_ptr, ns, nmask, base + 2 * BK, KT, BK)
    w3 = _hc_w_chunk(w_ptr, ns, nmask, base + 3 * BK, KT, BK)
    w4 = _hc_w_chunk(w_ptr, ns, nmask, base + 4 * BK, KT, BK)
    inj = 0.0
    if COMBINE:
        raw = tl.load(raw_ptr + s).to(tl.float32)
        raw = (raw / HC).to(raw_ptr.dtype.element_ty).to(tl.float32)
        inj = 2.0 * tl.sigmoid(raw)
    x0 = _hc_row_chunk(h_ptr, block_ptr, 0, base, 0 * BK, inj, COMBINE, BK)
    x1 = _hc_row_chunk(h_ptr, block_ptr, 0, base, 1 * BK, inj, COMBINE, BK)
    x2 = _hc_row_chunk(h_ptr, block_ptr, 0, base, 2 * BK, inj, COMBINE, BK)
    x3 = _hc_row_chunk(h_ptr, block_ptr, 0, base, 3 * BK, inj, COMBINE, BK)
    x4 = _hc_row_chunk(h_ptr, block_ptr, 0, base, 4 * BK, inj, COMBINE, BK)
    sumsq = tl.sum(x0 * x0, 0) + tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)
    sumsq += tl.sum(x3 * x3, 0) + tl.sum(x4 * x4, 0)
    rstd = tl.math.rsqrt(sumsq / H + eps)
    acc = _hc_norm_dot(
        x0,
        w0,
        rstd,
        norm_w_ptr,
        h_out_ptr,
        normed_ptr,
        base + 0 * BK,
        pid_n,
        COMBINE,
        BK,
    )
    acc += _hc_norm_dot(
        x1,
        w1,
        rstd,
        norm_w_ptr,
        h_out_ptr,
        normed_ptr,
        base + 1 * BK,
        pid_n,
        COMBINE,
        BK,
    )
    acc += _hc_norm_dot(
        x2,
        w2,
        rstd,
        norm_w_ptr,
        h_out_ptr,
        normed_ptr,
        base + 2 * BK,
        pid_n,
        COMBINE,
        BK,
    )
    acc += _hc_norm_dot(
        x3,
        w3,
        rstd,
        norm_w_ptr,
        h_out_ptr,
        normed_ptr,
        base + 3 * BK,
        pid_n,
        COMBINE,
        BK,
    )
    acc += _hc_norm_dot(
        x4,
        w4,
        rstd,
        norm_w_ptr,
        h_out_ptr,
        normed_ptr,
        base + 4 * BK,
        pid_n,
        COMBINE,
        BK,
    )
    tl.store(part_ptr + s * NOUT + ns, tl.sum(acc, 1), mask=nmask)
