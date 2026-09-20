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
    blockscale_gemm_fp8_kernel,
)
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


_TARGET_WAVES = 2


@functools.lru_cache(maxsize=8)
def _units(index: int | None = None) -> int:
    """Compute units on the current device; both tile and split gates use it."""
    return torch.cuda.get_device_properties(
        torch.cuda.current_device() if index is None else index
    ).multi_processor_count


def _auto_split_k(m: int, n: int, k: int, device, fp4: bool = True) -> int:
    """How many ways to split K, so the launch grid fills the device."""
    return _fp4_split_k(m, n, k, device) if fp4 else _fp8_split_k(m, n, k, device)


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


def _fp8_split_k(m: int, n: int, k: int, device) -> int:
    """Splits for the FP8 path, from the tile the same shape will run.

    The grid against the CU count is the whole gate: a 16-workgroup kv_a at
    m=1 more than halves, a 288-workgroup shared w1 at m=128 loses 1.3x. Two
    waves rather than one, because shorter splits also even out the tail.
    Asking for more than K/BK is harmless -- the caller re-derives the count
    from ``part_k``.
    """
    bm, bn, bk, _, _ = _select_fp8_tile(m, n, k)
    blocks = -(-m // bm) * -(-n // bn)
    units = _units(device.index)
    if blocks >= units:
        return 1
    return max(1, min(16, k // bk, -(-(_TARGET_WAVES * units) // blocks)))


def _select_fp8_tile(m: int, n: int, k: int):
    """(BM, BN, BK, num_warps, num_stages) for the FP8 kernel.

    Swept through ``native_quant_linear`` so each candidate ran with its real
    split count, and with the inputs rotated -- a sweep on pinned tensors reads
    a third of the decode time and picks differently. The M bands are also the
    JIT variant count per projection; N and K are constexpr there, so branching
    on them is free.

    Small M is weight-bandwidth bound on a grid too small to fill the CUs, so
    BM drops to the MFMA minimum and BK widens. The tile squares up on the grid
    it would leave rather than on N: a narrow projection at m=1024 still wants
    the small tile, because a 128-wide one leaves 80 workgroups of 256.
    """
    if m <= 16:
        return 16, 32, 512, 2, 2
    if m <= 64:
        # A wide, deep weight has enough work per output tile to pay for a
        # square one this early; kv_a and wq_a at m=32 do not.
        return (64, 64, 256, 4, 2) if n * k >= 2**25 else (32, 32, 512, 2, 2)
    if m <= 256:
        # A narrow projection cannot make enough 64-wide tiles to fill the CUs
        # here -- kv_a at m=128 leaves 16 workgroups -- and split-K only lifts
        # that to 160. Halving the tile is what fills it. Picked on the sum
        # over N in {512, 768, 1024} x m in {128, 256}, not on kv_a alone: the
        # 16x16x1024 tile that wins kv_a at m=128 is 30% slower at N=768.
        return (32, 32, 512, 2, 2) if n <= 1024 else (64, 64, 256, 4, 2)
    if n <= 1024 or -(-m // 128) * -(-n // 128) < _units() // 2:
        return 64, 64, 256, 4, 2
    return 128, 128, 256, 4, 1


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

    Weight scales remain compact. FP8 goes through the microscaling MFMA, W4A8
    unpacks to BF16 for a plain one; both accumulate in FP32 through the
    split-K reduction to the requested output conversion. A8 QAT and native
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
    if (
        x.dtype != torch.float8_e4m3fn
        or x_scale.dtype != torch.float8_e8m0fnu
        or weight_scale.dtype != torch.float8_e8m0fnu
    ):
        raise ValueError(
            "Activations must be E4M3 with E8M0 activation and weight scales"
        )
    if x_scale.shape != (*x.shape[:-1], k // 32) or weight_scale.shape != (
        -(-n // weight_group_rows),
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
    if fp4:
        bm, bn, bk, warps, stages = (16 if m <= 16 else 32), 64, 32, 4, 2
    else:
        bm, bn, bk, warps, stages = _select_fp8_tile(m, n, k)
    splits = split_k if split_k is not None else _auto_split_k(m, n, k, x.device, fp4)
    if not isinstance(splits, int) or splits < 1:
        raise ValueError("split_k must be a positive integer")
    # Round each slice up to the tile, then re-derive how many slices that is.
    slice_k = -(-k // splits)
    part_k = -(-slice_k // bk) * bk
    splits = -(-k // part_k)
    # Folding the reduction into the kernel costs more than the extra launch:
    # see the split-K note in the module docstring.
    partial = (
        output
        if splits == 1
        else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    )
    # Integer ceildivs rather than triton.cdiv: at decode this launch path
    # costs more than the kernel it launches. The E8M0 grids go in as uint8
    # because Triton has no dtype for them; E4M3 goes in as itself.
    grid = (-(-m // bm), -(-n // bn), splits)
    scales = (x_scale.view(torch.uint8), weight_scale.view(torch.uint8))
    if fp4:
        blockscale_gemm_fp4_kernel[grid](
            x,
            weight.view(torch.uint8),
            *scales,
            partial,
            m,
            n,
            k,
            part_k,
            bm,
            bn,
            bk,
            num_warps=warps,
            num_stages=stages,
            # Pins the BF16 MFMA shape this path was tuned against.
            matrix_instr_nonkdim=16,
        )
    else:
        blockscale_gemm_fp8_kernel[grid](
            x,
            weight,
            *scales,
            partial,
            m,
            n,
            k,
            weight_group_rows,
            part_k,
            bm,
            bn,
            bk,
            num_warps=warps,
            num_stages=stages,
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
