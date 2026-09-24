# SPDX-License-Identifier: MIT
"""Group32 FP8/W4A8 GEMMs and microscaling quantization primitives.

Split-K writes FP32 partials that a second kernel reduces. Folding that in --
each split announcing itself on an atomic counter, the last one reducing --
measured 1.7-2.9x slower: a kernel boundary gives the same ordering for free
and spreads the reduction over every workgroup.
"""

import functools

import torch
import triton

from .blockscale_kernels.blockscale_gemm import (
    blockscale_gemm_fp4_kernel,
)
from .blockscale_kernels.quantization import (
    FP8_DTYPE,
    quantize_fp4_kernel,
    quantize_fp8_kernel,
)


def _get_aiter_fp8_gemm():
    try:
        from aiter import gemm_a8w8_blockscale
    except ImportError:
        return None
    # The older same-named interface only supports FP32 128x128 scales.
    # The registered schema retains the signature hidden by torch_compile_guard.
    try:
        schema = torch.ops.aiter.gemm_a8w8_blockscale.default._schema
    except AttributeError:
        return None
    return (
        gemm_a8w8_blockscale
        if any(arg.name == "split_k" for arg in schema.arguments)
        else None
    )


_aiter_fp8_gemm = _get_aiter_fp8_gemm()


def _check_quant_input(x, group):
    if x.ndim < 1 or x.shape[-1] % group:
        raise ValueError(f"Last dimension must be divisible by {group}")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Quantization requires a floating-point CUDA/ROCm tensor")
    return x.contiguous()


def quantize_fp8(x: torch.Tensor, *, dequantize: bool = False):
    """E4M3, group32 E8M0 ceil scales; optionally return QAT values in x.dtype."""
    x = _check_quant_input(x, 32)
    output = torch.empty_like(x, dtype=x.dtype if dequantize else FP8_DTYPE)
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


@functools.lru_cache(maxsize=8)
def _units(index: int | None = None) -> int:
    """Compute units used to size local split-K grids."""
    return torch.cuda.get_device_properties(
        torch.cuda.current_device() if index is None else index
    ).multi_processor_count


def _fp4_split_k(m: int, n: int, k: int, device) -> int:
    """Splits for the W4A8 path, as tuned before the FP8 rewrite.

    Each split writes a partial that a second pass reduces, so a short K pays
    more for the partials than it wins in occupancy: on MI355X K=2048 already
    costs ~15% while K>=4096 wins ~2x.
    """
    if m <= 16:
        return min(16, triton.cdiv(k, 256))
    if k < 4096:
        return 1
    blocks = triton.cdiv(m, 32) * triton.cdiv(n, 64)
    units = _units(device.index)
    if blocks >= units:
        return 1
    return max(1, min(16, k // 1024, -(-units // blocks)))


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

    Weight scales remain compact. FP8 prefers AITER's configured GEMM interface
    and falls back to local kernels on older AITER builds. W4A8 unpacks to
    BF16 for a plain MFMA; both accumulate in FP32 before the
    requested output conversion. A8 QAT and native
    weight storage are retained without materializing a dequantized weight.
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
    if split_k is not None and (not isinstance(split_k, int) or split_k < 1):
        raise ValueError("split_k must be a positive integer")
    if weight_scale.shape != (-(-n // weight_group_rows), k // 32):
        raise ValueError("Weight scale shape does not match the declared source blocks")
    if not fp4 and _aiter_fp8_gemm is not None:
        batched = x.ndim > 2
        output = _aiter_fp8_gemm(
            x.view(m, k) if batched else x,
            weight,
            x_scale.view(m, k // 32) if batched else x_scale,
            weight_scale,
            dtype=dtype,
            split_k=split_k,
        )
        return output.view(*x.shape[:-1], n) if batched else output

    if (
        x.dtype != torch.float8_e4m3fn
        or x_scale.dtype != torch.float8_e8m0fnu
        or weight_scale.dtype != torch.float8_e8m0fnu
    ):
        raise ValueError(
            "Activations must be E4M3 with E8M0 activation and weight scales"
        )
    if x_scale.shape != (*x.shape[:-1], k // 32):
        raise ValueError("Scale shape does not match the declared source blocks")
    tensors = (x, weight, x_scale, weight_scale)
    if any(
        t.device != x.device or not t.is_cuda or not t.is_contiguous() for t in tensors
    ):
        raise ValueError(
            "GEMM operands must be contiguous on the same CUDA/ROCm device"
        )
    if not fp4:
        from .blockscale_kernels.fp8 import gemm_fp8_local

        return gemm_fp8_local(
            x,
            weight,
            x_scale,
            weight_scale,
            dtype=dtype,
            weight_group_rows=weight_group_rows,
            split_k=split_k,
            cu_num=_units(x.device.index),
        )
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=dtype)
    if m == 0:
        return output
    bm, bn, bk = (16 if m <= 16 else 32), 64, 32
    splits = split_k if split_k is not None else _fp4_split_k(m, n, k, x.device)
    slice_k = -(-k // splits)
    part_k = -(-slice_k // bk) * bk
    splits = -(-k // part_k)
    partial = (
        output
        if splits == 1
        else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    )
    blockscale_gemm_fp4_kernel[(-(-m // bm), -(-n // bn), splits)](
        x,
        weight.view(torch.uint8),
        x_scale.view(torch.uint8),
        weight_scale.view(torch.uint8),
        partial,
        m,
        n,
        k,
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
