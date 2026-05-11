"""Fused sigmoid(gate) * attn_output + FP8 per-1x128 block-scale quantization.

Replaces three separate ops (sigmoid, elementwise mul, FP8 quant) with a
single Triton kernel. The output can be passed directly to
``LinearBase.forward(x_fp8, x_scale=scale)`` to skip the internal quant step.
"""

import torch
from torch import Tensor
import triton
import triton.language as tl

import aiter

fp8_dtype = aiter.dtypes.fp8

DTYPE_MAX = torch.finfo(fp8_dtype).max
DTYPE_MIN = -DTYPE_MAX


@triton.heuristics(
    {
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    }
)
@triton.jit
def _fused_sigmoid_mul_fp8_group_quant_kernel(
    # Input pointers
    attn_ptr,
    gate_ptr,
    # Output pointers
    out_fp8_ptr,
    out_scale_ptr,
    # Strides
    stride_attn_m,
    stride_attn_n,
    stride_gate_m,
    stride_gate_n,
    stride_fp8_m,
    stride_fp8_n,
    stride_scale_m,
    stride_scale_n,
    # Dimensions
    N,
    # Constexprs
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_MIN: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Cast strides to int64 to avoid overflow for large tensors
    stride_attn_m = tl.cast(stride_attn_m, tl.int64)
    stride_attn_n = tl.cast(stride_attn_n, tl.int64)
    stride_gate_m = tl.cast(stride_gate_m, tl.int64)
    stride_gate_n = tl.cast(stride_gate_n, tl.int64)
    stride_fp8_m = tl.cast(stride_fp8_m, tl.int64)
    stride_fp8_n = tl.cast(stride_fp8_n, tl.int64)
    stride_scale_m = tl.cast(stride_scale_m, tl.int64)
    stride_scale_n = tl.cast(stride_scale_n, tl.int64)

    NUM_QUANT_BLOCKS: tl.constexpr = BLOCK_SIZE_N // QUANT_BLOCK_SIZE

    col_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load attn_output and gate
    attn_offs = pid_m * stride_attn_m + col_offs * stride_attn_n
    gate_offs = pid_m * stride_gate_m + col_offs * stride_gate_n

    if EVEN_N:
        attn = tl.load(attn_ptr + attn_offs).to(tl.float32)
        gate = tl.load(gate_ptr + gate_offs).to(tl.float32)
    else:
        mask = col_offs < N
        attn = tl.load(attn_ptr + attn_offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + gate_offs, mask=mask, other=0.0).to(tl.float32)

    # sigmoid(gate) * attn_output
    x = tl.sigmoid(gate) * attn

    # FP8 per-group quantization (group_size = QUANT_BLOCK_SIZE = 128)
    x = x.reshape(1, NUM_QUANT_BLOCKS, QUANT_BLOCK_SIZE)
    m = tl.maximum(tl.max(tl.abs(x), axis=-1), 1e-10)
    scale_out = m.to(tl.float32) / FP8_MAX
    scale_recip = 1.0 / scale_out.reshape(1, NUM_QUANT_BLOCKS, 1)
    x = tl.clamp(x * scale_recip, FP8_MIN, FP8_MAX)
    x_fp8 = tl.ravel(x)
    x_scale = tl.ravel(scale_out)

    # Store FP8 output
    fp8_offs = pid_m * stride_fp8_m + col_offs * stride_fp8_n
    if EVEN_N:
        tl.store(out_fp8_ptr + fp8_offs, x_fp8.to(out_fp8_ptr.dtype.element_ty))
    else:
        tl.store(
            out_fp8_ptr + fp8_offs,
            x_fp8.to(out_fp8_ptr.dtype.element_ty),
            mask=mask,
        )

    # Store scale (transpose_scale layout: scale is [num_scale_cols, M] contiguous)
    scale_col_offs = pid_n * NUM_QUANT_BLOCKS + tl.arange(0, NUM_QUANT_BLOCKS)
    scale_offs = pid_m * stride_scale_m + scale_col_offs * stride_scale_n
    num_scale_cols = (N + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
    if EVEN_N:
        tl.store(out_scale_ptr + scale_offs, x_scale.to(out_scale_ptr.dtype.element_ty))
    else:
        scale_mask = scale_col_offs < num_scale_cols
        tl.store(
            out_scale_ptr + scale_offs,
            x_scale.to(out_scale_ptr.dtype.element_ty),
            mask=scale_mask,
        )


def fused_sigmoid_mul_fp8_quant(
    attn_output: Tensor,
    gate: Tensor,
    group_size: int = 128,
    transpose_scale: bool | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused sigmoid(gate) * attn_output + FP8 per-group quantization.

    Args:
        attn_output: [M, N] bf16/fp16 — attention output tensor.
        gate: [M, N] bf16/fp16 — gating tensor (pre-sigmoid).
        group_size: Quantization group size (default 128, matching per_1x128).
        transpose_scale: If True, produce column-major x_scale (for preshuffle GEMM).
                         If False, produce row-major x_scale (for non-preshuffle GEMM).
                         If None (default), follows ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE env var.

    Returns:
        (x_fp8, x_scale):
            x_fp8: [M, N] FP8 — quantized sigmoid(gate) * attn_output.
            x_scale: [M, N // group_size] float32 — per-group scales.
    """
    if transpose_scale is None:
        from atom.utils import envs

        transpose_scale = envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE

    M, N = attn_output.shape
    assert (
        N % group_size == 0
    ), f"N ({N}) must be divisible by group_size ({group_size})"

    BLOCK_SIZE_N = group_size

    out_fp8 = torch.empty((M, N), dtype=fp8_dtype, device=attn_output.device)
    num_scale_cols = N // group_size

    if transpose_scale:
        # column-major: allocate as (num_scale_cols, M) contiguous,
        # then view as (M, num_scale_cols) after kernel
        out_scale = torch.empty(
            (num_scale_cols, M), dtype=torch.float32, device=attn_output.device
        )
        stride_scale_m = out_scale.stride(1)
        stride_scale_n = out_scale.stride(0)
    else:
        # row-major: allocate as (M, num_scale_cols) contiguous
        out_scale = torch.empty(
            (M, num_scale_cols), dtype=torch.float32, device=attn_output.device
        )
        stride_scale_m = out_scale.stride(0)
        stride_scale_n = out_scale.stride(1)

    grid = (M, N // BLOCK_SIZE_N)
    _fused_sigmoid_mul_fp8_group_quant_kernel[grid](
        attn_output,
        gate,
        out_fp8,
        out_scale,
        attn_output.stride(0),
        attn_output.stride(1),
        gate.stride(0),
        gate.stride(1),
        out_fp8.stride(0),
        out_fp8.stride(1),
        stride_scale_m,
        stride_scale_n,
        N=N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=group_size,
        FP8_MAX=DTYPE_MAX,
        FP8_MIN=DTYPE_MIN,
    )

    if transpose_scale:
        # View transposed buffer back to (M, num_scale_cols) shape
        out_scale = out_scale.view(M, num_scale_cols)

    return out_fp8, out_scale
