# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused Qwen3.8-Flash-Next hyper-connection operators.

The unfused hyper-connection runs, per sub-layer, a grouped RMSNorm, the
low-rank down GEMM, a scaled SiLU, the up GEMM, the gated stream mean, the
injection GEMM and the injection combine: seven launches around two small
replicated GEMMs. Here the combine of the previous sub-layer is deferred into
the next hyper-connection and the work is regrouped as:

* one or two tokens, two launches:
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

# Batches from which the multi-row combine+norm kernel is used (prefill).
COMBINE_NORM_ROWS_MIN_M = 256


@triton.jit
def _hc_row_chunk(
    h_ptr,
    block_ptr,
    stride_block,
    base,
    k0,
    inj,
    COMBINE: tl.constexpr,
    BK: tl.constexpr,
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
    stride_normed,
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
    row = tl.program_id(1).to(tl.int64)
    part_ptr += row * (HC * NOUT)
    normed_ptr += row * stride_normed
    mixed_ptr += row * H
    raw_out_ptr += row * HC
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
    if HAS_INJECT:  # noqa: SIM102 -- compile-time guard for the inject branch
        if pid_h == 0:
            cs = tl.arange(0, HC)
            acc = tl.zeros((HC,), dtype=tl.float32)
            for p in tl.static_range(HC):
                acc += tl.load(part_ptr + p * NOUT + R + cs)
            tl.store(raw_out_ptr + cs, acc.to(raw_out_ptr.dtype.element_ty))


@triton.jit
def _hc_gated_mean_tiled_kernel(
    normed_ptr,
    gate_ptr,
    out_ptr,
    M,
    stride_row,
    stride_out,
    H: tl.constexpr,
    HC: tl.constexpr,
    RB: tl.constexpr,
    BH: tl.constexpr,
):
    """`mean_s(sigmoid(gate) * normed)` on an [RB rows, BH columns] tile."""
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    rmask = (rows < M)[:, None]
    base = rows[:, None].to(tl.int64) * stride_row + cols[None, :]
    total = tl.zeros((RB, BH), dtype=tl.float32)
    for s in tl.static_range(HC):
        x = tl.load(normed_ptr + base + s * H, mask=rmask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + base + s * H, mask=rmask, other=0.0).to(tl.float32)
        total += tl.sigmoid(g) * x
    tl.store(
        out_ptr + rows[:, None].to(tl.int64) * stride_out + cols[None, :],
        (total / HC).to(out_ptr.dtype.element_ty),
        mask=rmask,
    )


def hc_gated_mean(normed: torch.Tensor, gate: torch.Tensor, hc: int) -> torch.Tensor:
    """Tiled `mix_gated_mean` for large batches (same rounding)."""
    m, width = normed.shape
    hidden = width // hc
    out = torch.empty((m, hidden), dtype=normed.dtype, device=normed.device)
    rb, bh = 4, 512
    if m and hidden % bh == 0 and gate.stride(0) == normed.stride(0):
        _hc_gated_mean_tiled_kernel[(triton.cdiv(m, rb), hidden // bh)](
            normed,
            gate,
            out,
            m,
            normed.stride(0),
            out.stride(0),
            H=hidden,
            HC=hc,
            RB=rb,
            BH=bh,
            num_warps=4,
        )
        return out
    from atom.model_ops.qwen4_exp.ops.gated import mix_gated_mean

    return mix_gated_mean(normed, gate, hc)


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


@triton.jit
def _hc_cn_part(h_ptr, block_ptr, off_h, off_b, rmask, inj, COMBINE: tl.constexpr):
    x = tl.load(h_ptr + off_h, mask=rmask, other=0.0)
    if COMBINE:
        b = tl.load(block_ptr + off_b, mask=rmask, other=0.0).to(tl.float32)
        x = (x.to(tl.float32) + b * inj[:, None]).to(h_ptr.dtype.element_ty)
    return x


@triton.jit
def _hc_combine_norm_rows_kernel(
    h_ptr,
    block_ptr,
    raw_ptr,
    norm_w_ptr,
    h_out_ptr,
    normed_ptr,
    M,
    stride_h,
    stride_block,
    stride_raw,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    COMBINE: tl.constexpr,
    RB: tl.constexpr,
    BA: tl.constexpr,  # H == BA + BB, both powers of two
    BB: tl.constexpr,
):
    """`_hc_combine_norm_kernel` over RB rows of one stream, no masked lanes."""
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    s = tl.program_id(1)
    rmask = (rows < M)[:, None]
    r64 = rows[:, None].to(tl.int64)
    ca = tl.arange(0, BA)[None, :]
    cb = BA + tl.arange(0, BB)[None, :]
    inj = tl.zeros((RB,), dtype=tl.float32)
    if COMBINE:
        raw = tl.load(
            raw_ptr + rows.to(tl.int64) * stride_raw + s, mask=rows < M, other=0.0
        )
        raw = (raw.to(tl.float32) / HC).to(raw_ptr.dtype.element_ty).to(tl.float32)
        inj = 2.0 * tl.sigmoid(raw)
    oh = r64 * stride_h + s * H
    ob = r64 * stride_block
    xa = _hc_cn_part(h_ptr, block_ptr, oh + ca, ob + ca, rmask, inj, COMBINE)
    xb = _hc_cn_part(h_ptr, block_ptr, oh + cb, ob + cb, rmask, inj, COMBINE)
    if COMBINE:
        tl.store(h_out_ptr + oh + ca, xa, mask=rmask)
        tl.store(h_out_ptr + oh + cb, xb, mask=rmask)
    fa = xa.to(tl.float32)
    fb = xb.to(tl.float32)
    rstd = tl.math.rsqrt((tl.sum(fa * fa, 1) + tl.sum(fb * fb, 1)) / H + eps)
    ga = tl.load(norm_w_ptr + s * H + ca).to(tl.float32) + 1.0
    gb = tl.load(norm_w_ptr + s * H + cb).to(tl.float32) + 1.0
    dt = normed_ptr.dtype.element_ty
    tl.store(normed_ptr + oh + ca, (fa * rstd[:, None] * ga).to(dt), mask=rmask)
    tl.store(normed_ptr + oh + cb, (fb * rstd[:, None] * gb).to(dt), mask=rmask)


def hc_rows(
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
    """Deferred combine + full mix, one program set per token.

    Returns (streams, mixed, inject logits or None). Rows re-read the weights,
    which stay cache resident, so this is for small batches.
    """
    m, width = h.shape
    hidden = width // hc
    nout = w_cat.shape[0]
    rank = w_up.shape[1]
    combine = block_out is not None
    normed = torch.empty_like(h)
    h_new = torch.empty_like(h) if combine else h
    part = torch.empty((m, hc, nout), dtype=torch.float32, device=h.device)
    bn = 8
    _hc_pre_kernel[(triton.cdiv(nout, bn), hc, m)](
        h,
        block_out if combine else h,
        raw if combine else h,
        norm_weight,
        w_cat,
        h_new,
        normed,
        part,
        eps,
        h.stride(0),
        block_out.stride(0) if combine else 0,
        raw.stride(0) if combine else 0,
        H=hidden,
        HC=hc,
        NOUT=nout,
        COMBINE=combine,
        BN=bn,
        BK=hidden // 5,
        num_warps=4,
    )
    mixed = torch.empty((m, hidden), dtype=h.dtype, device=h.device)
    raw_out = (
        torch.empty((m, hc), dtype=h.dtype, device=h.device) if has_inject else None
    )
    bh = 16
    _hc_mix_kernel[(triton.cdiv(hidden, bh), m)](
        part,
        normed,
        w_up,
        mixed,
        raw_out if has_inject else mixed,
        normed.stride(0),
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
    ba = 1 << (hidden.bit_length() - 1)
    bb = hidden - ba
    if m >= COMBINE_NORM_ROWS_MIN_M and bb > 0 and bb & (bb - 1) == 0:
        rb = 2
        _hc_combine_norm_rows_kernel[(triton.cdiv(m, rb), hc)](
            h,
            block_out if combine else h,
            raw if combine else h,
            norm_weight,
            h_new,
            normed,
            m,
            h.stride(0),
            block_out.stride(0) if combine else 0,
            raw.stride(0) if combine else 0,
            eps,
            H=hidden,
            HC=hc,
            COMBINE=combine,
            RB=rb,
            BA=ba,
            BB=bb,
            num_warps=4,
        )
        return h_new, normed
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
