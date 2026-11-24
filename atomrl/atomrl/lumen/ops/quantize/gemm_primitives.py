# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Standalone dgrad / wgrad primitives for BF16 and FP8 GEMM.

Extracted from ``QuantizedLinearFunction.backward`` to allow reuse in fused
comm-GEMM overlap autograd functions.

These are additive — ``QuantizedLinearFunction`` is not modified.
"""

from collections.abc import Callable

import torch

from atomrl.lumen.ops.quantize.linear import dispatch_gemm, quantize_input


def compute_dgrad_bf16(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute input gradient: dgrad = grad_output @ weight.

    Args:
        grad_output: ``[*, N]`` where N = out_features.
        weight: ``[N, K]`` where K = in_features.

    Returns:
        Tensor with shape ``[*, K]``.
    """
    return dispatch_gemm(
        grad_output,
        weight.t().contiguous(),
        None,
        None,
        "none",
    )


def compute_wgrad_bf16(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute weight gradient: wgrad = grad_output^T @ input.

    Args:
        grad_output: ``[M, N]``.
        input_tensor: ``[M, K]``.

    Returns:
        Tensor with shape ``[N, K]``.
    """
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    input_flat = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()
    return dispatch_gemm(
        grad_flat.t().contiguous(),
        input_flat.t().contiguous(),
        None,
        None,
        "none",
    )


def make_wgrad_closure_bf16(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    weight_ref: torch.nn.Parameter,
    gradient_accumulation_fusion: bool,
) -> Callable[[], None]:
    """Build a deferred wgrad closure compatible with ``_DeferredWgrad``.

    The closure computes wgrad and accumulates into ``weight_ref.main_grad``
    (if *gradient_accumulation_fusion*) or ``weight_ref.grad``.
    """
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    input_flat = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()

    def _wgrad_fn():
        gw = dispatch_gemm(
            grad_flat.t().contiguous(),
            input_flat.t().contiguous(),
            None,
            None,
            "none",
        )
        if gradient_accumulation_fusion and hasattr(weight_ref, "main_grad"):
            weight_ref.main_grad.add_(gw)
        elif weight_ref.grad is not None:
            weight_ref.grad.add_(gw)
        else:
            weight_ref.grad = gw

    return _wgrad_fn


def _bwd_scaling_for(scaling_type: str) -> str:
    """Return the backward scaling type corresponding to a forward scaling type.

    Mirrors the logic in ``QuantizedLinearFunction.backward``:
    per_token, blockwise, and blockwise2d use dynamic scaling in
    the backward pass; delayed and dynamic pass through unchanged.
    """
    if scaling_type in ("per_token", "blockwise", "blockwise2d"):
        return "dynamic"
    return scaling_type


def _dequant_fp8_weight(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize FP8 weight ``(N, K)`` to float32, handling any scale shape.

    Scale shapes by quantization mode:
      - per-tensor / dynamic / delayed: ``(1,)``
      - per_token: ``(N, 1)``
      - blockwise: ``(N, ceil(K / block_size))``
      - blockwise2d (Jet-RL / DeepSeek-V3 weight scheme):
        ``(ceil(N / block_size), ceil(K / block_size))``
    """
    if weight_scale.numel() == 1:
        return weight_fp8.bfloat16() * weight_scale.bfloat16()

    N, K = weight_fp8.shape[-2], weight_fp8.shape[-1]

    # 2D block scale: (N/block, K/block) — expand along both axes.
    if (
        weight_scale.dim() == 2
        and weight_scale.shape[0] != N
        and weight_scale.shape[0] * block_size >= N
    ):
        scales_expanded = (
            weight_scale.bfloat16()
            .repeat_interleave(block_size, dim=0)[:N]
            .repeat_interleave(block_size, dim=1)[:, :K]
        )
        return weight_fp8.bfloat16() * scales_expanded

    scale_cols = weight_scale.shape[-1] if weight_scale.dim() > 1 else 1

    if scale_cols == 1 or scale_cols >= K:
        return weight_fp8.bfloat16() * weight_scale.bfloat16()

    scales_expanded = weight_scale.bfloat16().repeat_interleave(block_size, dim=-1)[
        :, :K
    ]
    return weight_fp8.bfloat16() * scales_expanded


def compute_dgrad_fp8(
    grad_chunk: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    scaling_type: str,
    scaling_manager: object | None,
    fp8_dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    """Compute FP8 dgrad for a single grad chunk.

    Quantizes *grad_chunk* (BF16) to FP8 and dispatches the dgrad GEMM
    against the pre-quantized *weight_fp8*.  When the gradient is quantized
    to a different FP8 dtype (e.g. E5M2 in hybrid mode) and the weight is
    E4M3, uses mixed-dtype FP8 GEMM via hipBLASLt.

    Falls back to BF16 GEMM when the forward weight scale is multi-element
    (per_token / blockwise) and therefore incompatible with per-tensor dgrad.

    Args:
        grad_chunk: ``[S_chunk, N]`` gradient chunk (BF16).
        weight_fp8: ``[N, K]`` pre-quantized FP8 weight.
        weight_scale: Weight quantization scale.
        scaling_type: Forward scaling type.
        scaling_manager: Optional :class:`ScalingManager`.
        fp8_dtype: FP8 dtype (e.g. ``torch.float8_e4m3fn``).
        block_size: Block size for block-wise quantization.
    """
    from atomrl.lumen.ops.quantize.linear import gemm_per_tensor_mixed

    bwd_scaling = _bwd_scaling_for(scaling_type)

    bwd_dtype = (
        getattr(scaling_manager, "fp8_dtype_bwd", fp8_dtype)
        if scaling_manager
        else fp8_dtype
    )

    grad_flat = grad_chunk.reshape(-1, grad_chunk.shape[-1]).contiguous()
    grad_desc = quantize_input(grad_flat, bwd_scaling, bwd_dtype, block_size)
    grad_fp8, grad_scale = grad_desc.data, grad_desc.scale

    _needs_dequant = scaling_type in ("per_token", "blockwise", "blockwise2d")
    if _needs_dequant:
        grad_bf16 = (grad_fp8.bfloat16() * grad_scale.bfloat16()).contiguous()
        weight_bf16 = _dequant_fp8_weight(
            weight_fp8, weight_scale, block_size
        ).bfloat16()
        result = dispatch_gemm(
            grad_bf16, weight_bf16.t().contiguous(), None, None, "none"
        )
    elif grad_fp8.dtype != weight_fp8.dtype:
        result = gemm_per_tensor_mixed(
            grad_fp8,
            weight_fp8,
            grad_scale,
            weight_scale,
        )
    else:
        result = dispatch_gemm(
            grad_desc,
            weight_fp8.t().contiguous(),
            scale_w=weight_scale,
            scaling_type=bwd_scaling,
        )
    return result.view(*grad_chunk.shape[:-1], weight_fp8.shape[-1])
