# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SwiGLU-OAI activation used by MiniMax-M3.

MiniMax-M3 stores dense and expert gate/up activations in split layout:
``[gate | up]``.  The activation is:

    gate * sigmoid(alpha * gate) * (up + beta)

with optional clamping.  ATOM only supports this path on AMD GPU, so the
implementation is Triton-only.

The optional shared-expert path writes native group32 E4M3 values and
row-major E8M0 scales for ``blockscale.native_quant_linear``. It is a distinct
kernel, leaving the existing BF16 activation path unchanged.
"""

import functools

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx_runtime

from .blockscale_kernels.quantization import FP8_DTYPE, fp8_groups


@triton.jit
def _swiglu_oai_kernel(
    gate_up_ptr,
    out_ptr,
    n_inter: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    alpha,
    beta,
    limit,
    has_limit: tl.constexpr,
    block_i: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * block_i + tl.arange(0, block_i)
    mask = cols < n_inter

    gate = tl.load(
        gate_up_ptr + row * stride_gm + cols * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        gate_up_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if has_limit:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)

    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _swiglu_oai_mxfp8_kernel(
    gate_up_ptr,
    out_ptr,
    scale_ptr,
    n_groups: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_om: tl.constexpr,
    stride_sm: tl.constexpr,
    alpha,
    beta,
    limit,
    has_limit: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    """SwiGLU-OAI, its BF16 store boundary, then native group32 MXFP8."""
    alpha = tl.cast(alpha, tl.float32)
    beta = tl.cast(beta, tl.float32)
    limit = tl.cast(limit, tl.float32)

    row = tl.program_id(0)
    groups = tl.program_id(1) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    cols = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    group_mask = groups < n_groups
    mask = group_mask[:, None]
    n_inter = n_groups * 32
    gate = tl.load(gate_up_ptr + row * stride_gm + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        gate_up_ptr + row * stride_gm + n_inter + cols, mask=mask, other=0.0
    ).to(tl.float32)
    if has_limit:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)

    # The split path stores BF16 here and quantize_fp8 reloads it as FP32.
    # Preserve that rounding point before using the same group32 primitive.
    activated = (gate * tl.sigmoid(alpha * gate) * (up + beta)).to(tl.bfloat16)
    quantized, scale_code, _ = fp8_groups(activated.to(tl.float32))
    tl.store(out_ptr + row * stride_om + cols, quantized, mask=mask)
    tl.store(
        scale_ptr + row * stride_sm + groups,
        scale_code.to(tl.uint8),
        mask=group_mask,
    )


@functools.lru_cache(maxsize=1)
def swiglu_oai_mxfp8_arch_supported() -> bool:
    return get_gfx_runtime() == "gfx1250"


def swiglu_oai_mxfp8_supported(gate_up: torch.Tensor) -> bool:
    """Return whether ``gate_up`` is eligible for native group32 fusion."""
    layout_supported = (
        gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.is_contiguous()
        and gate_up.shape[0] > 0
        and gate_up.shape[1] > 0
        and gate_up.shape[1] % 64 == 0
    )
    # The model validates the arch once at initialization. Keep native probing
    # out of a Dynamo full graph; it expands into os.stat/rocminfo calls.
    return layout_supported and (
        torch.compiler.is_compiling() or swiglu_oai_mxfp8_arch_supported()
    )


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply MiniMax-M3 SwiGLU-OAI to a split-layout ``[..., 2I]`` tensor."""
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            f"SwiGLU-OAI expects an even last dimension, got {gate_up.shape[-1]}."
        )
    if not gate_up.is_cuda:
        raise RuntimeError("SwiGLU-OAI is only supported on AMD GPU tensors.")

    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n_inter = two_i // 2
    x2 = gate_up.reshape(-1, two_i)
    out = torch.empty(
        (x2.shape[0], n_inter),
        dtype=out_dtype or gate_up.dtype,
        device=gate_up.device,
    )

    block_i = 512 if n_inter >= 2048 else 256
    grid = (x2.shape[0], triton.cdiv(n_inter, block_i))
    _swiglu_oai_kernel[grid](
        x2,
        out,
        n_inter,
        x2.stride(0),
        x2.stride(1),
        out.stride(0),
        out.stride(1),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        has_limit=limit is not None,
        block_i=block_i,
        num_warps=4,
    )
    return out.reshape(*orig_shape[:-1], n_inter)


def swiglu_oai_split_mxfp8(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SwiGLU-OAI and native group32 MXFP8 quantization in one kernel.

    Return contiguous E4M3 payload ``[M, I]`` and row-major E8M0 scales
    ``[M, I / 32]``. The bytes match ``blockscale.quantize_fp8`` applied to
    the BF16 output of :func:`swiglu_oai_split`.
    """
    if not swiglu_oai_mxfp8_supported(gate_up):
        raise ValueError(
            "SwiGLU-OAI MXFP8 requires a nonempty contiguous gfx1250 BF16 "
            "matrix [M, 2I] with I divisible by 32"
        )

    rows, twice_intermediate = gate_up.shape
    intermediate = twice_intermediate // 2
    groups = intermediate // 32
    out = torch.empty((rows, intermediate), dtype=FP8_DTYPE, device=gate_up.device)
    scales = torch.empty(
        (rows, groups), dtype=torch.float8_e8m0fnu, device=gate_up.device
    )

    # Dense I=3072 benefits from wider 1024-value tiles; the shared I=768 path
    # retains its tuned 512-value tiles.
    block_groups = 32 if groups >= 96 else 16
    _swiglu_oai_mxfp8_kernel[(rows, triton.cdiv(groups, block_groups))](
        gate_up,
        out,
        scales.view(torch.uint8),
        groups,
        gate_up.stride(0),
        out.stride(0),
        scales.stride(0),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        has_limit=limit is not None,
        BLOCK_GROUPS=block_groups,
        num_warps=4,
    )
    return out, scales
