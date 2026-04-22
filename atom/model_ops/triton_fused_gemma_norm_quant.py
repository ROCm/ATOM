"""Fused GemmaRMSNorm + FP8 per-1x128 block-scale quantization.

Replaces: GemmaRMSNorm (BF16 output) → LinearBase internal FP8 quant
With:      Single kernel that reads BF16 input, computes norm, writes FP8 + scales.

The output ``(x_fp8, x_scale)`` can be passed directly to
``LinearBase.forward(x_fp8, x_scale=x_scale)`` to skip the internal quant step.

Optionally writes a BF16 copy of the normed output (for consumers that need
non-quantized input, e.g. ``in_proj_ba`` in linear attention).
"""

import torch
from torch import Tensor
import triton
import triton.language as tl

import aiter

fp8_dtype = aiter.dtypes.fp8

DTYPE_MAX = torch.finfo(fp8_dtype).max
DTYPE_MIN = -DTYPE_MAX


@triton.jit
def _fused_gemma_norm_fp8_group_quant_kernel(
    # Input
    x_ptr,
    x_stride_m,
    # Residual (optional)
    residual_ptr,
    residual_stride_m,
    residual_out_ptr,
    residual_out_stride_m,
    # Norm weight
    weight_ptr,
    # Outputs
    out_bf16_ptr,
    out_bf16_stride_m,
    out_fp8_ptr,
    out_fp8_stride_m,
    out_scale_ptr,
    # Scale strides (transpose_scale layout: [num_scale_cols, M])
    out_scale_stride_m,
    out_scale_stride_n,
    # Dimensions
    eps: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_BF16: tl.constexpr,
):
    row = tl.program_id(0)

    # Cast strides to int64 to avoid overflow
    x_stride_m = tl.cast(x_stride_m, tl.int64)
    out_fp8_stride_m = tl.cast(out_fp8_stride_m, tl.int64)

    col_offs = tl.arange(0, HIDDEN_SIZE)

    # Load input row
    x = tl.load(x_ptr + row * x_stride_m + col_offs).to(tl.float32)

    # Residual add
    if HAS_RESIDUAL:
        residual_stride_m = tl.cast(residual_stride_m, tl.int64)
        residual_out_stride_m = tl.cast(residual_out_stride_m, tl.int64)
        res = tl.load(residual_ptr + row * residual_stride_m + col_offs).to(tl.float32)
        x = x + res
        # Store updated residual (pre-norm sum)
        tl.store(
            residual_out_ptr + row * residual_out_stride_m + col_offs,
            x.to(residual_out_ptr.dtype.element_ty),
        )

    # GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight)
    variance = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    x_normed = x * tl.math.rsqrt(variance + eps)
    w = tl.load(weight_ptr + col_offs).to(tl.float32)
    x_normed = x_normed * (1.0 + w)

    # Optionally store BF16 output
    if WRITE_BF16:
        out_bf16_stride_m = tl.cast(out_bf16_stride_m, tl.int64)
        tl.store(
            out_bf16_ptr + row * out_bf16_stride_m + col_offs,
            x_normed.to(out_bf16_ptr.dtype.element_ty),
        )

    # Per-group FP8 quantization (group_size = GROUP_SIZE = 128)
    NUM_GROUPS: tl.constexpr = HIDDEN_SIZE // GROUP_SIZE

    x_grouped = x_normed.reshape(NUM_GROUPS, GROUP_SIZE)
    group_max = tl.max(tl.abs(x_grouped), axis=1)  # [NUM_GROUPS]
    group_max = tl.maximum(group_max, 1e-10)
    scale = group_max / FP8_MAX  # [NUM_GROUPS]
    scale_recip = 1.0 / scale.reshape(NUM_GROUPS, 1)  # [NUM_GROUPS, 1]
    x_fp8 = tl.clamp(x_grouped * scale_recip, FP8_MIN, FP8_MAX)
    x_fp8_flat = x_fp8.reshape(HIDDEN_SIZE)

    # Store FP8 output
    tl.store(
        out_fp8_ptr + row * out_fp8_stride_m + col_offs,
        x_fp8_flat.to(out_fp8_ptr.dtype.element_ty),
    )

    # Store scales in transpose_scale layout
    out_scale_stride_m = tl.cast(out_scale_stride_m, tl.int64)
    out_scale_stride_n = tl.cast(out_scale_stride_n, tl.int64)
    group_offs = tl.arange(0, NUM_GROUPS)
    scale_ptrs = (
        out_scale_ptr + row * out_scale_stride_m + group_offs * out_scale_stride_n
    )
    tl.store(scale_ptrs, scale.to(out_scale_ptr.dtype.element_ty))


def fused_gemma_norm_fp8_quant(
    x: Tensor,
    weight: Tensor,
    eps: float,
    residual: Tensor | None = None,
    write_bf16: bool = False,
    group_size: int = 128,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    """Fused GemmaRMSNorm + FP8 per-group quantization.

    Args:
        x: [M, N] input tensor (BF16).
        weight: [N] GemmaRMSNorm weight.
        eps: Variance epsilon.
        residual: [M, N] optional residual to add before norm.
        write_bf16: If True, also write normed BF16 output.
        group_size: Quantization group size (default 128 for per_1x128).

    Returns:
        (out_bf16, out_fp8, out_scale, residual_out):
            out_bf16:     [M, N] normed BF16 (None if write_bf16=False)
            out_fp8:      [M, N] FP8 quantized normed output
            out_scale:    [M, N // group_size] float32 scales (transpose_scale layout)
            residual_out: [M, N] = x + residual if residual given, else x
    """
    M, N = x.shape
    assert (
        N % group_size == 0
    ), f"N ({N}) must be divisible by group_size ({group_size})"

    has_residual = residual is not None

    # Allocate outputs
    out_fp8 = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    num_scale_cols = N // group_size
    # transpose_scale layout: [num_scale_cols, M] contiguous
    out_scale = torch.empty((num_scale_cols, M), dtype=torch.float32, device=x.device)

    if write_bf16:
        out_bf16 = torch.empty((M, N), dtype=x.dtype, device=x.device)
    else:
        out_bf16 = None

    if has_residual:
        residual_out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    else:
        residual_out = x  # first layer: residual = hidden_states (no add)

    grid = (M,)
    _fused_gemma_norm_fp8_group_quant_kernel[grid](
        x,
        x.stride(0),
        residual if has_residual else x,  # dummy ptr when no residual
        residual.stride(0) if has_residual else 0,
        residual_out if has_residual else x,  # dummy ptr when no residual
        residual_out.stride(0) if has_residual else 0,
        weight,
        out_bf16 if out_bf16 is not None else x,  # dummy ptr when no bf16
        out_bf16.stride(0) if out_bf16 is not None else 0,
        out_fp8,
        out_fp8.stride(0),
        out_scale,
        # transpose_scale: swap strides so kernel writes column-major
        out_scale.stride(1),  # stride_m = stride along dim 1 (M dim)
        out_scale.stride(0),  # stride_n = stride along dim 0 (scale_col dim)
        eps=eps,
        HIDDEN_SIZE=N,
        GROUP_SIZE=group_size,
        FP8_MAX=DTYPE_MAX,
        FP8_MIN=DTYPE_MIN,
        HAS_RESIDUAL=has_residual,
        WRITE_BF16=write_bf16,
    )

    # Transpose to [M, num_scale_cols] shape with stride=(1, M).
    # Data is column-major (same memory layout as HIP shuffle_scale=True).
    out_scale = out_scale.t()

    return out_bf16, out_fp8, out_scale, residual_out
