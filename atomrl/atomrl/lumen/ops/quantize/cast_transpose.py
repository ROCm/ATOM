# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused FP8 cast + transpose kernel.

Combines per-tensor FP8 quantization with matrix transposition into a single
Triton kernel launch, replacing the separate quantize → transpose → contiguous
chain that currently requires multiple kernel launches.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atomrl.lumen.ops.quantize.quant_amax_fused import _get_amax_scratch


def _prepare_scale_1d(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert scale to 1-D float32 contiguous, avoiding copies when already correct."""
    if scale.dtype == torch.float32 and scale.ndim == 1 and scale.is_contiguous():
        s = scale
    else:
        s = scale.float().reshape(-1).contiguous()
    if s.device != device:
        s = s.to(device=device)
    return s


__all__ = [
    "cast_amax_fp8",
    "cast_transpose_amax_fp8",
    "cast_transpose_fp8",
    "rmsnorm_quant_amax_fp8",
]

_TORCH_TO_TL_FP8: dict[torch.dtype, tl.dtype] = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e5m2: tl.float8e5,
    torch.float8_e5m2fnuz: tl.float8e5b16,
}


@triton.jit
def _cast_transpose_fp8_kernel(
    X_ptr,
    OUT_ptr,
    OUT_T_ptr,
    scale_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    stride_ot0,
    stride_ot1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """Each program handles a BLOCK_M x BLOCK_N tile.

    Reads X[bm:bm+BLOCK_M, bn:bn+BLOCK_N], scales to FP8, writes both:
    - OUT[bm:bm+BLOCK_M, bn:bn+BLOCK_N] (same layout as X)
    - OUT_T[bn:bn+BLOCK_N, bm:bm+BLOCK_M] i.e. OUT_T[n,m] = quant(X[m,n])
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    scale = tl.load(scale_ptr).to(tl.float32)
    scale = tl.where(scale > 1.0e-30, scale, 1.0)
    inv_scale = 1.0 / scale

    x_scaled = x * inv_scale
    x_clamped = tl.where(
        x_scaled > CLAMP_MAX,
        CLAMP_MAX,
        tl.where(x_scaled < -CLAMP_MAX, -CLAMP_MAX, x_scaled),
    )
    x_fp8 = x_clamped.to(FP8_TY)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, x_fp8, mask=mask)

    # OUT_T has shape (N, M); element (n, m) gets X[m, n].
    out_t_ptrs = OUT_T_ptr + offs_n[None, :] * stride_ot0 + offs_m[:, None] * stride_ot1
    tl.store(out_t_ptrs, x_fp8, mask=mask)


@triton.jit
def _cast_transpose_amax_fp8_kernel(
    X_ptr,
    OUT_ptr,
    OUT_T_ptr,
    scale_ptr,
    amax_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    stride_ot0,
    stride_ot1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """Tile-based fused cast + transpose + amax.

    Same tiling as _cast_transpose_fp8_kernel but also computes per-tile
    max(abs(x)) and atomically reduces it into amax_ptr[0].
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    tile_amax = tl.max(tl.where(mask, tl.abs(x), 0.0))
    tl.atomic_max(amax_ptr, tile_amax, sem="relaxed")

    scale = tl.load(scale_ptr).to(tl.float32)
    scale = tl.where(scale > 1.0e-30, scale, 1.0)
    inv_scale = 1.0 / scale

    x_scaled = x * inv_scale
    x_clamped = tl.where(
        x_scaled > CLAMP_MAX,
        CLAMP_MAX,
        tl.where(x_scaled < -CLAMP_MAX, -CLAMP_MAX, x_scaled),
    )
    x_fp8 = x_clamped.to(FP8_TY)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, x_fp8, mask=mask)

    out_t_ptrs = OUT_T_ptr + offs_n[None, :] * stride_ot0 + offs_m[:, None] * stride_ot1
    tl.store(out_t_ptrs, x_fp8, mask=mask)


def cast_transpose_amax_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    clamp_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused cast to FP8 + transpose + amax in a single kernel.

    Combines :func:`cast_transpose_fp8` with amax computation so the
    BF16 input is read exactly once.

    Args:
        x: Input tensor (M, N) in BF16/FP16/FP32 on CUDA.
        scale: Per-tensor scale (1,) float32 — divisor style.
        fp8_dtype: Target FP8 dtype.
        clamp_max: Absolute clamp bound before FP8 cast.

    Returns:
        ``(out, out_t, amax)``: row-major FP8 ``(M, N)``, transposed FP8
        ``(N, M)``, and scalar float32 amax of the *unscaled* input.
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_cuda, "cast_transpose_amax_fp8 requires a CUDA tensor"
    if fp8_dtype not in _TORCH_TO_TL_FP8:
        raise TypeError(f"Unsupported fp8_dtype: {fp8_dtype}")

    M, N = x.shape
    out = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    out_t = torch.empty((N, M), dtype=fp8_dtype, device=x.device)
    amax_out = _get_amax_scratch(x.device)
    amax_out.zero_()

    if clamp_max is None:
        clamp_max = float(torch.finfo(fp8_dtype).max)

    scale_1 = _prepare_scale_1d(scale, x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fp8_tl = _TORCH_TO_TL_FP8[fp8_dtype]
    _cast_transpose_amax_fp8_kernel[grid](
        x,
        out,
        out_t,
        scale_1,
        amax_out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        out_t.stride(0),
        out_t.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CLAMP_MAX=clamp_max,
        FP8_TY=fp8_tl,
    )

    return out, out_t, amax_out.clone()


def cast_transpose_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    clamp_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cast to FP8 + transpose in a single kernel.

    Args:
        x: Input tensor (M, N) in BF16/FP16/FP32 on CUDA.
        scale: Per-tensor scale (1,) float32 — divisor style (same as
            ``tensor / scale`` in :meth:`ScalingManager._quantize_core`).
        fp8_dtype: Target FP8 dtype.
        clamp_max: Absolute clamp bound before FP8 cast. Defaults to
            ``torch.finfo(fp8_dtype).max``.

    Returns:
        ``(out, out_t)``: row-major FP8 ``(M, N)`` and transposed FP8 ``(N, M)``.
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_cuda, "cast_transpose_fp8 requires a CUDA tensor"
    if fp8_dtype not in _TORCH_TO_TL_FP8:
        raise TypeError(f"Unsupported fp8_dtype for cast_transpose_fp8: {fp8_dtype}")

    M, N = x.shape
    out = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    out_t = torch.empty((N, M), dtype=fp8_dtype, device=x.device)

    if clamp_max is None:
        clamp_max = float(torch.finfo(fp8_dtype).max)

    scale_1 = _prepare_scale_1d(scale, x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fp8_tl = _TORCH_TO_TL_FP8[fp8_dtype]
    _cast_transpose_fp8_kernel[grid](
        x,
        out,
        out_t,
        scale_1,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        out_t.stride(0),
        out_t.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CLAMP_MAX=clamp_max,
        FP8_TY=fp8_tl,
    )

    return out, out_t


@triton.jit
def _cast_amax_fp8_kernel(
    X_ptr,
    OUT_ptr,
    scale_ptr,
    amax_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """Tile-based fused cast + amax (no transpose output).

    Same as _cast_transpose_amax_fp8_kernel but skips the transposed write,
    halving write bandwidth. Use when the caller only needs row-major FP8
    output (e.g. hipBLASLt handles .t() views natively).
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    tile_amax = tl.max(tl.where(mask, tl.abs(x), 0.0))
    tl.atomic_max(amax_ptr, tile_amax, sem="relaxed")

    scale = tl.load(scale_ptr).to(tl.float32)
    scale = tl.where(scale > 1.0e-30, scale, 1.0)
    inv_scale = 1.0 / scale

    x_scaled = x * inv_scale
    x_clamped = tl.where(
        x_scaled > CLAMP_MAX,
        CLAMP_MAX,
        tl.where(x_scaled < -CLAMP_MAX, -CLAMP_MAX, x_scaled),
    )
    x_fp8 = x_clamped.to(FP8_TY)

    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, x_fp8, mask=mask)


def cast_amax_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    clamp_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cast to FP8 + amax in a single kernel (no transpose output).

    Identical to :func:`cast_transpose_amax_fp8` but skips the transposed
    output write, saving one ``(N, M)`` FP8 allocation and halving write
    bandwidth.  Use when the GEMM backend handles ``.t()`` views natively
    (hipBLASLt with ``LUMEN_PREFER_HIPBLASLT=1``).

    Args:
        x: Input tensor (M, N) in BF16/FP16/FP32 on CUDA.
        scale: Per-tensor scale (1,) float32 — divisor style.
        fp8_dtype: Target FP8 dtype.
        clamp_max: Absolute clamp bound before FP8 cast.

    Returns:
        ``(out, amax)``: row-major FP8 ``(M, N)`` and scalar float32 amax
        of the *unscaled* input.
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_cuda, "cast_amax_fp8 requires a CUDA tensor"
    if fp8_dtype not in _TORCH_TO_TL_FP8:
        raise TypeError(f"Unsupported fp8_dtype: {fp8_dtype}")

    M, N = x.shape
    out = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    amax_out = _get_amax_scratch(x.device)
    amax_out.zero_()

    if clamp_max is None:
        clamp_max = float(torch.finfo(fp8_dtype).max)

    scale_1 = _prepare_scale_1d(scale, x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fp8_tl = _TORCH_TO_TL_FP8[fp8_dtype]
    _cast_amax_fp8_kernel[grid](
        x,
        out,
        scale_1,
        amax_out,
        M,
        N,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CLAMP_MAX=clamp_max,
        FP8_TY=fp8_tl,
    )

    return out, amax_out.clone()


# ---------------------------------------------------------------------------
# Fused RMSNorm + FP8 quant + amax  (single kernel, one read of x)
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_quant_amax_fp8_kernel(
    X_ptr,
    W_ptr,
    OUT_ptr,
    scale_ptr,
    amax_ptr,
    M,
    N,
    stride_xm,
    stride_om,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """Per-row fused RMSNorm + FP8 cast + amax.

    Each program handles one row of the (M, N) input:
      1. Load row, compute variance = mean(x^2)
      2. Normalize: x_norm = x * rsqrt(variance + eps) * weight
      3. Track amax(abs(x_norm)) via atomic max
      4. Scale → clamp → cast to FP8, store to OUT
    """
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_ptrs = X_ptr + row * stride_xm + offs_n
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + EPS)

    w = tl.load(W_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
    x_norm = x * rrms * w

    row_amax = tl.max(tl.where(mask, tl.abs(x_norm), 0.0))
    tl.atomic_max(amax_ptr, row_amax, sem="relaxed")

    scale = tl.load(scale_ptr).to(tl.float32)
    scale = tl.where(scale > 1.0e-30, scale, 1.0)
    inv_scale = 1.0 / scale
    x_scaled = x_norm * inv_scale
    x_clamped = tl.where(
        x_scaled > CLAMP_MAX,
        CLAMP_MAX,
        tl.where(x_scaled < -CLAMP_MAX, -CLAMP_MAX, x_scaled),
    )
    x_fp8 = x_clamped.to(FP8_TY)

    out_ptrs = OUT_ptr + row * stride_om + offs_n
    tl.store(out_ptrs, x_fp8, mask=mask)


@triton.jit
def _rmsnorm_quant_amax_fp8_bf16_kernel(
    X_ptr,
    W_ptr,
    OUT_FP8_ptr,
    OUT_BF16_ptr,
    scale_ptr,
    amax_ptr,
    M,
    N,
    stride_xm,
    stride_o8m,
    stride_obm,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    FP8_TY: tl.constexpr,
):
    """Per-row fused RMSNorm + dual output (FP8 + BF16) + amax.

    Produces both:
      - FP8 output (for the forward GEMM, skipping separate quantization)
      - BF16 output (for autograd backward — needed to compute norm grad)

    One HBM read of x, two HBM writes.  Replaces:
      separate RMSNorm kernel + separate cast_amax_fp8 kernel.
    """
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    x_ptrs = X_ptr + row * stride_xm + offs_n
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + EPS)

    w = tl.load(W_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
    x_norm = x * rrms * w

    # BF16 output for autograd backward
    bf16_ptrs = OUT_BF16_ptr + row * stride_obm + offs_n
    tl.store(bf16_ptrs, x_norm.to(tl.bfloat16), mask=mask)

    # amax of normalized output
    row_amax = tl.max(tl.where(mask, tl.abs(x_norm), 0.0))
    tl.atomic_max(amax_ptr, row_amax, sem="relaxed")

    # FP8 output
    scale = tl.load(scale_ptr).to(tl.float32)
    scale = tl.where(scale > 1.0e-30, scale, 1.0)
    inv_scale = 1.0 / scale
    x_scaled = x_norm * inv_scale
    x_clamped = tl.where(
        x_scaled > CLAMP_MAX,
        CLAMP_MAX,
        tl.where(x_scaled < -CLAMP_MAX, -CLAMP_MAX, x_scaled),
    )
    x_fp8 = x_clamped.to(FP8_TY)

    fp8_ptrs = OUT_FP8_ptr + row * stride_o8m + offs_n
    tl.store(fp8_ptrs, x_fp8, mask=mask)


def _row_rsigma(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row RMSNorm rsigma, matching the fused kernel's fp32 statistic."""
    xf = x.detach().float()
    variance = torch.sum(xf * xf, dim=-1) / x.shape[-1]
    return torch.rsqrt(variance + eps)


def rmsnorm_quant_amax_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    *,
    clamp_max: float | None = None,
    output_bf16: bool = True,
    output_rsigma: bool = False,
):
    """Fused RMSNorm + FP8 quant + amax in a single kernel launch.

    Reads x once, computes RMSNorm(x, weight, eps), quantizes to FP8 with
    the provided scale, and computes amax of the normalized output.

    Args:
        x: Input tensor (M, N) in BF16/FP16/FP32 on CUDA.
        weight: RMSNorm weight (N,).
        eps: RMSNorm epsilon.
        scale: Per-tensor scale (1,) float32 — divisor style.
        fp8_dtype: Target FP8 dtype.
        clamp_max: Absolute clamp bound before FP8 cast.
        output_bf16: If True, also produce BF16 norm output (for backward).
        output_rsigma: If True, also return per-row ``rsigma`` for the backward.

    Returns:
        ``(out_fp8, out_bf16_or_None, amax)``, or that tuple plus ``rsigma``
        when ``output_rsigma`` is set.
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    assert x.is_cuda, "rmsnorm_quant_amax_fp8 requires a CUDA tensor"
    if fp8_dtype not in _TORCH_TO_TL_FP8:
        raise TypeError(f"Unsupported fp8_dtype: {fp8_dtype}")

    M, N = x.shape
    out_fp8 = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    amax_out = _get_amax_scratch(x.device)
    amax_out.zero_()

    if clamp_max is None:
        clamp_max = float(torch.finfo(fp8_dtype).max)

    scale_1 = _prepare_scale_1d(scale, x.device)

    BLOCK_N = triton.next_power_of_2(N)
    fp8_tl = _TORCH_TO_TL_FP8[fp8_dtype]

    if output_bf16:
        out_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        _rmsnorm_quant_amax_fp8_bf16_kernel[(M,)](
            x,
            weight,
            out_fp8,
            out_bf16,
            scale_1,
            amax_out,
            M,
            N,
            x.stride(0),
            out_fp8.stride(0),
            out_bf16.stride(0),
            EPS=eps,
            BLOCK_N=BLOCK_N,
            CLAMP_MAX=clamp_max,
            FP8_TY=fp8_tl,
        )
        result = (out_fp8, out_bf16, amax_out.clone())
    else:
        _rmsnorm_quant_amax_fp8_kernel[(M,)](
            x,
            weight,
            out_fp8,
            scale_1,
            amax_out,
            M,
            N,
            x.stride(0),
            out_fp8.stride(0),
            EPS=eps,
            BLOCK_N=BLOCK_N,
            CLAMP_MAX=clamp_max,
            FP8_TY=fp8_tl,
        )
        result = (out_fp8, None, amax_out.clone())
    if output_rsigma:
        return (*result, _row_rsigma(x, eps))
    return result
