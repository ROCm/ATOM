# SPDX-License-Identifier: MIT
"""Group32 FP8/W4A8 GEMMs and microscaling quantization primitives."""

import torch
import triton

from .blockscale_kernels.blockscale_gemm import blockscale_gemm_kernel
from .blockscale_kernels.quantization import quantize_fp4_kernel, quantize_fp8_kernel


def _check_quant_input(x, group):
    if x.ndim < 1 or x.shape[-1] % group:
        raise ValueError(f"Last dimension must be divisible by {group}")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Quantization requires a floating-point CUDA/ROCm tensor")
    return x.contiguous()


def quantize_fp8(x: torch.Tensor, *, dequantize: bool = False):
    """E4M3, group32 E8M0 ceil scales; optionally return QAT values in x.dtype."""
    x = _check_quant_input(x, 32)
    output = torch.empty_like(x, dtype=x.dtype if dequantize else torch.float8_e4m3fn)
    scales = (
        None
        if dequantize
        else torch.empty(
            (*x.shape[:-1], x.shape[-1] // 32),
            device=x.device,
            dtype=torch.float8_e8m0fnu,
        )
    )
    if x.numel():
        quantize_fp8_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            output,
            None if scales is None else scales.view(torch.uint8),
            x.numel(),
            dequantize,
        )
    return output if dequantize else (output, scales)


def quantize_fp4(
    x: torch.Tensor,
    *,
    group_size: int = 32,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
    dequantize: bool = False,
):
    """Packed E2M1: group16/E4M3 for main KV, group32/E8M0 for index Q/K."""
    if (group_size, scale_dtype) not in (
        (16, torch.float8_e4m3fn),
        (32, torch.float8_e8m0fnu),
    ):
        raise ValueError("FP4 requires group16/E4M3 or group32/E8M0 scales")
    x = _check_quant_input(x, group_size)
    shape = x.shape if dequantize else (*x.shape[:-1], x.shape[-1] // 2)
    output = torch.empty(
        shape, device=x.device, dtype=x.dtype if dequantize else torch.uint8
    )
    scales = (
        None
        if dequantize
        else torch.empty(
            (*x.shape[:-1], x.shape[-1] // group_size),
            device=x.device,
            dtype=scale_dtype,
        )
    )
    scale_ptr = (
        scales.view(torch.uint8)
        if scales is not None and scale_dtype == torch.float8_e8m0fnu
        else scales
    )
    if x.numel():
        quantize_fp4_kernel[(triton.cdiv(x.numel(), 32 * group_size),)](
            x,
            output,
            scale_ptr,
            x.numel(),
            group_size,
            scale_dtype == torch.float8_e4m3fn,
            dequantize,
            enable_fp_fusion=False,
        )
    return output if dequantize else (output.view(torch.float4_e2m1fn_x2), scales)


def native_quant_linear(
    x,
    weight,
    weight_scale,
    *,
    x_scale=None,
    weight_group_rows=32,
    dtype=torch.bfloat16,
    split_k=None,
):
    """FP8 32x32/1x32 or W4A8 1x32 GEMM; inputs and weights stay native.

    Weight scales remain compact and apply to group32 partials in FP32. The
    correctness path uses BF16 MFMA on register tiles; it retains A8 QAT and
    native weight storage. Decode splits K with FP32 partial sums.
    """
    if x.ndim < 2 or weight.ndim != 2 or dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            "Expected batched activations, a matrix weight and BF16/FP32 output"
        )
    fp4 = weight.dtype == torch.float4_e2m1fn_x2
    if weight.dtype not in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2):
        raise ValueError("Weight must be E4M3 or packed E2M1")
    if weight_group_rows not in (1, 32) or (fp4 and weight_group_rows != 1):
        raise ValueError("FP4 requires row/group32 scales; FP8 accepts 1x32 or 32x32")
    n, stored_k = weight.shape
    k = stored_k * (2 if fp4 else 1)
    if k <= 0 or k % 32 or x.shape[-1] != k or n <= 0:
        raise ValueError("Invalid matrix dimensions for group32 GEMM")
    if x_scale is None:
        x, x_scale = quantize_fp8(x)
    m = x.numel() // k
    if (
        x.dtype != torch.float8_e4m3fn
        or x_scale.dtype != torch.float8_e8m0fnu
        or weight_scale.dtype != torch.float8_e8m0fnu
    ):
        raise ValueError(
            "Activations must be E4M3 with E8M0 activation and weight scales"
        )
    if x_scale.shape != (*x.shape[:-1], k // 32) or weight_scale.shape != (
        triton.cdiv(n, weight_group_rows),
        k // 32,
    ):
        raise ValueError("Scale shape does not match the declared source blocks")
    tensors = (x, weight, x_scale, weight_scale)
    if any(
        t.device != x.device or not t.is_cuda or not t.is_contiguous() for t in tensors
    ):
        raise ValueError(
            "GEMM operands must be contiguous on the same CUDA/ROCm device"
        )
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=dtype)
    if m == 0:
        return output
    bk = 32
    splits = (
        split_k
        if split_k is not None
        else (min(16, triton.cdiv(k, 256)) if m <= 16 else 1)
    )
    if not isinstance(splits, int) or splits < 1:
        raise ValueError("split_k must be a positive integer")
    part_k = triton.cdiv(triton.cdiv(k, splits), bk) * bk
    splits = triton.cdiv(k, part_k)
    partial = (
        output
        if splits == 1
        else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    )
    bm, bn = (16 if m <= 16 else 32), 64
    blockscale_gemm_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        x.view(torch.uint8),
        weight.view(torch.uint8),
        x_scale.view(torch.uint8),
        weight_scale.view(torch.uint8),
        partial,
        m,
        n,
        k,
        weight_group_rows,
        fp4,
        splits,
        part_k,
        bm,
        bn,
        bk,
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    if splits > 1:
        from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
            _gemm_splitk_reduce_kernel,
        )

        _gemm_splitk_reduce_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            partial,
            output,
            None,
            m,
            n,
            m * n,
            n,
            1,
            n,
            1,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            ACTUAL_KSPLIT=splits,
            MAX_KSPLIT=triton.next_power_of_2(splits),
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="native_blockscale_reduce",
        )
    return output


def dequantize_fp8_weight(weight, scale, *, group_rows=32, dtype=torch.bfloat16):
    """Load-time conversion for grouped wo_a or bounded host embedding gathers."""
    if (
        weight.ndim != 2
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float8_e8m0fnu
    ):
        raise ValueError("Expected an E4M3 matrix and E8M0 scale grid")
    n, k = weight.shape
    if k % 32 or n % group_rows or scale.shape != (n // group_rows, k // 32):
        raise ValueError("FP8 source block dimensions do not match")
    return (
        (
            weight.float().reshape(n // group_rows, group_rows, k // 32, 32)
            * scale.float()[:, None, :, None]
        )
        .reshape(n, k)
        .to(dtype)
    )
