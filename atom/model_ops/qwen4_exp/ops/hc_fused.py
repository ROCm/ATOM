# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused Qwen3.8-Flash-Next hyper-connection operators.

The unfused hyper-connection runs, per sub-layer, a grouped RMSNorm, the
low-rank down GEMM, a scaled SiLU, the up GEMM, the gated stream mean, the
injection GEMM and the injection combine: seven launches around two small
replicated GEMMs. Here the combine of the previous sub-layer is deferred into
the next hyper-connection and the work is regrouped as:

* single token (decode at batch 1), two launches:
  `_hc_pre_kernel` combines, normalizes and multiplies by the concatenated
  `[down | inject]` weight, split-K over the four streams (each program
  normalizes only its own stream); `_hc_mix_kernel` sums the four partials,
  applies the scaled SiLU, the up projection and the gated stream mean.
  Both use CUDA-core dot products: at one row the MFMA operand shuffles
  through LDS cost more than they save.
* larger batches: `_hc_combine_norm_kernel`, then the two GEMMs (one of them
  concatenated) around the existing SiLU and gated-mean kernels.

Rounding follows the unfused path: combined streams, normalized values, GEMM
outputs and the scaled SiLU input/output are rounded to BF16 before use.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_row_chunk(
    h_ptr, block_ptr, stride_block, base, k0, inj,
    COMBINE: tl.constexpr, BK: tl.constexpr,
):
    ks = k0 + tl.arange(0, BK)
    x = tl.load(h_ptr + base + ks)
    if COMBINE:
        b = tl.load(block_ptr + ks).to(tl.float32)
        x = (x.to(tl.float32) + b * inj).to(h_ptr.dtype.element_ty)
    return x.to(tl.float32)


@triton.jit
def _hc_w_chunk(w_ptr, ns, nmask, k0, KTOT: tl.constexpr, BK: tl.constexpr):
    ks = k0 + tl.arange(0, BK)
    return tl.load(
        w_ptr + ns[:, None].to(tl.int64) * KTOT + ks[None, :],
        mask=nmask[:, None],
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _hc_norm_dot(
    x, w, rstd, norm_w_ptr, h_out_ptr, normed_ptr, k0, pid_n,
    COMBINE: tl.constexpr, BK: tl.constexpr,
):
    ks = k0 + tl.arange(0, BK)
    g = tl.load(norm_w_ptr + ks).to(tl.float32) + 1.0
    xn = (x * rstd * g).to(normed_ptr.dtype.element_ty)
    if pid_n == 0:
        tl.store(normed_ptr + ks, xn)
        if COMBINE:
            tl.store(h_out_ptr + ks, x.to(h_out_ptr.dtype.element_ty))
    return tl.sum(w * xn.to(tl.float32)[None, :], axis=1)


@triton.jit
def _hc_pre_kernel(
    h_ptr,  # [HC*H] bf16, previous residual streams
    block_ptr,  # [H] bf16, previous sub-layer output (COMBINE only)
    raw_ptr,  # [HC] bf16, previous injection logits (COMBINE only)
    norm_w_ptr,  # [HC*H] norm weight (Gemma offset applied here)
    w_ptr,  # [NOUT, HC*H] bf16, concatenated [down | inject]
    h_out_ptr,  # [HC*H] bf16, combined streams (COMBINE only)
    normed_ptr,  # [HC*H] bf16
    part_ptr,  # [HC, NOUT] fp32
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    NOUT: tl.constexpr,
    COMBINE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Program (N tile, stream). The stream's five K chunks are unrolled so
    every weight and activation load is issued before any arithmetic."""
    tl.static_assert(H == 5 * BK)
    pid_n = tl.program_id(0)
    s = tl.program_id(1)
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
    acc = _hc_norm_dot(x0, w0, rstd, norm_w_ptr, h_out_ptr, normed_ptr, base + 0 * BK, pid_n, COMBINE, BK)
    acc += _hc_norm_dot(x1, w1, rstd, norm_w_ptr, h_out_ptr, normed_ptr, base + 1 * BK, pid_n, COMBINE, BK)
    acc += _hc_norm_dot(x2, w2, rstd, norm_w_ptr, h_out_ptr, normed_ptr, base + 2 * BK, pid_n, COMBINE, BK)
    acc += _hc_norm_dot(x3, w3, rstd, norm_w_ptr, h_out_ptr, normed_ptr, base + 3 * BK, pid_n, COMBINE, BK)
    acc += _hc_norm_dot(x4, w4, rstd, norm_w_ptr, h_out_ptr, normed_ptr, base + 4 * BK, pid_n, COMBINE, BK)
    tl.store(part_ptr + s * NOUT + ns, acc, mask=nmask)


@triton.jit
def _load_up_rows(wu_ptr, rows, kr, kmask, R: tl.constexpr):
    return tl.load(
        wu_ptr + rows[:, None].to(tl.int64) * R + kr[None, :],
        mask=kmask[None, :],
        other=0.0,
    ).to(tl.float32)


@triton.jit
def _gated_stream(w, gate, x_ptrs, dtype):
    u = tl.sum(w * gate[None, :], axis=1).to(dtype).to(tl.float32)
    return tl.sigmoid(u) * tl.load(x_ptrs).to(tl.float32)


@triton.jit
def _hc_mix_kernel(
    part_ptr,  # [HC, NOUT] fp32
    normed_ptr,  # [HC*H] bf16
    wu_ptr,  # [HC*H, R] bf16
    mixed_ptr,  # [H] bf16
    raw_out_ptr,  # [HC] bf16
    H: tl.constexpr,
    HC: tl.constexpr,
    R: tl.constexpr,
    RP: tl.constexpr,  # next power of two >= R
    NOUT: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    BH: tl.constexpr,
):
    tl.static_assert(HC == 4)
    pid_h = tl.program_id(0)
    hs = pid_h * BH + tl.arange(0, BH)
    kr = tl.arange(0, RP)
    kmask = kr < R
    w0 = _load_up_rows(wu_ptr, 0 * H + hs, kr, kmask, R)
    w1 = _load_up_rows(wu_ptr, 1 * H + hs, kr, kmask, R)
    w2 = _load_up_rows(wu_ptr, 2 * H + hs, kr, kmask, R)
    w3 = _load_up_rows(wu_ptr, 3 * H + hs, kr, kmask, R)
    d = tl.zeros((RP,), dtype=tl.float32)
    for p in tl.static_range(HC):
        d += tl.load(part_ptr + p * NOUT + kr, mask=kmask, other=0.0)
    dt = mixed_ptr.dtype.element_ty
    d = d.to(dt).to(tl.float32)
    d = (d / HC).to(dt).to(tl.float32)
    gate = (d * tl.sigmoid(d)).to(dt).to(tl.float32)
    total = _gated_stream(w0, gate, normed_ptr + 0 * H + hs, dt)
    total += _gated_stream(w1, gate, normed_ptr + 1 * H + hs, dt)
    total += _gated_stream(w2, gate, normed_ptr + 2 * H + hs, dt)
    total += _gated_stream(w3, gate, normed_ptr + 3 * H + hs, dt)
    tl.store(mixed_ptr + hs, (total / HC).to(dt))
    if HAS_INJECT:
        if pid_h == 0:
            cs = tl.arange(0, HC)
            acc = tl.zeros((HC,), dtype=tl.float32)
            for p in tl.static_range(HC):
                acc += tl.load(part_ptr + p * NOUT + R + cs)
            tl.store(raw_out_ptr + cs, acc.to(raw_out_ptr.dtype.element_ty))


@triton.jit
def _hc_combine_norm_kernel(
    h_ptr,
    block_ptr,
    raw_ptr,
    norm_w_ptr,
    h_out_ptr,
    normed_ptr,
    stride_h,
    stride_block,
    stride_raw,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    COMBINE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(h_ptr + row * stride_h + s * H + cols, mask=mask, other=0.0)
    if COMBINE:
        raw = tl.load(raw_ptr + row * stride_raw + s).to(tl.float32)
        raw = (raw / HC).to(raw_ptr.dtype.element_ty).to(tl.float32)
        b = tl.load(block_ptr + row * stride_block + cols, mask=mask, other=0.0)
        x = (x.to(tl.float32) + b.to(tl.float32) * (2.0 * tl.sigmoid(raw))).to(
            h_ptr.dtype.element_ty
        )
        tl.store(h_out_ptr + row * stride_h + s * H + cols, x, mask=mask)
    x = x.to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    g = tl.load(norm_w_ptr + s * H + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    tl.store(
        normed_ptr + row * stride_h + s * H + cols,
        (x * rstd * g).to(normed_ptr.dtype.element_ty),
        mask=mask,
    )


def hc_single_token(
    h: torch.Tensor,
    norm_weight: torch.Tensor,
    w_cat: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
    hc: int,
    has_inject: bool,
    block_out: torch.Tensor | None = None,
    raw: torch.Tensor | None = None,
):
    """Deferred combine + full mix for one token.

    Returns (streams, mixed, inject logits or None).
    """
    if h.shape[0] != 1:
        raise ValueError("hc_single_token expects exactly one token")
    width = h.shape[1]
    hidden = width // hc
    nout = w_cat.shape[0]
    rank = w_up.shape[1]
    combine = block_out is not None
    normed = torch.empty_like(h)
    h_new = torch.empty_like(h) if combine else h
    part = torch.empty((hc, nout), dtype=torch.float32, device=h.device)
    bn = 8
    _hc_pre_kernel[(triton.cdiv(nout, bn), hc)](
        h,
        block_out if combine else h,
        raw if combine else h,
        norm_weight,
        w_cat,
        h_new,
        normed,
        part,
        eps,
        H=hidden,
        HC=hc,
        NOUT=nout,
        COMBINE=combine,
        BN=bn,
        BK=hidden // 5,
        num_warps=4,
    )
    mixed = torch.empty((1, hidden), dtype=h.dtype, device=h.device)
    raw_out = torch.empty((1, hc), dtype=h.dtype, device=h.device) if has_inject else None
    bh = 16
    _hc_mix_kernel[(triton.cdiv(hidden, bh),)](
        part,
        normed,
        w_up,
        mixed,
        raw_out if has_inject else mixed,
        H=hidden,
        HC=hc,
        R=rank,
        RP=triton.next_power_of_2(rank),
        NOUT=nout,
        HAS_INJECT=has_inject,
        BH=bh,
        num_warps=4,
    )
    return h_new, mixed, raw_out


def hc_combine_norm(
    h: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc: int,
    block_out: torch.Tensor | None = None,
    raw: torch.Tensor | None = None,
):
    """Deferred combine (optional) + grouped norm; returns (streams, normed)."""
    m, width = h.shape
    hidden = width // hc
    combine = block_out is not None
    normed = torch.empty_like(h)
    h_new = torch.empty_like(h) if combine else h
    if m:
        _hc_combine_norm_kernel[(m, hc)](
            h,
            block_out if combine else h,
            raw if combine else h,
            norm_weight,
            h_new,
            normed,
            h.stride(0),
            block_out.stride(0) if combine else 0,
            raw.stride(0) if combine else 0,
            eps,
            H=hidden,
            HC=hc,
            COMBINE=combine,
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=4,
        )
    return h_new, normed
