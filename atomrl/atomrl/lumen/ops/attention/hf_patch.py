# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Process-wide Lumen routes for Hugging Face/PyTorch SDPA callers."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_original_sdpa = None
_active_mode = None
_min_seq = 1024
_stats = {"routed": 0, "stock": 0}


def _claim_mode(mode: str) -> bool:
    """Claim the one process-wide SDPA replacement."""
    global _active_mode, _original_sdpa
    if _active_mode == mode:
        return False
    if _active_mode is not None:
        raise RuntimeError(
            f"SDPA is already patched in {_active_mode!r} mode; "
            f"cannot also install {mode!r}"
        )
    _original_sdpa = F.scaled_dot_product_attention
    _active_mode = mode
    return True


def patch_sdpa() -> None:
    """Route every PyTorch SDPA call through Lumen attention."""
    if not _claim_mode("lumen"):
        return
    from atomrl.lumen.ops.attention import attention as lumen_attention

    def _lumen_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        output = lumen_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=is_causal,
            bias=attn_mask,
        )
        return output.transpose(1, 2)

    F.scaled_dot_product_attention = _lumen_sdpa
    logger.info("Patched SDPA with the general Lumen/AITER attention route")


def patch_hf_sdpa() -> None:
    """Compatibility alias for the general Lumen route."""
    patch_sdpa()


def prefer_efficient_backend() -> None:
    """Prefer PyTorch's ROCm mem-efficient backward over flash backward."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    logger.info("SDPA flash backend disabled; mem-efficient backend preferred")


def _triton_forward(q, k, v, scale):
    """Run AITER's public Triton forward and return SDPA-layout output/LSE."""
    from aiter.ops.triton.attention.mha import flash_attn_func

    qs, ks, vs = (tensor.transpose(1, 2) for tensor in (q, k, v))
    output, lse = flash_attn_func(
        qs,
        ks,
        vs,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
        return_lse=True,
        backend="triton",
    )
    return output.transpose(1, 2), lse


class TritonFwdEfficientBwd(torch.autograd.Function):
    """AITER Triton forward joined to PyTorch's efficient SDPA backward."""

    @staticmethod
    def forward(ctx, q, k, v, scale):
        output, lse = _triton_forward(q, k, v, scale)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        zero = torch.zeros((), dtype=torch.int64)
        dq, dk, dv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            grad_output.contiguous() if grad_output.stride(-1) != 1 else grad_output,
            q,
            k,
            v,
            None,
            output,
            lse,
            zero,
            zero,
            0.0,
            [True, True, True, False],
            False,
            scale=ctx.scale,
        )
        return dq, dk, dv, None


def _dit_eligible(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa):
    return (
        attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and not enable_gqa
        and q.dim() == 4
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
        and q.shape[1] == k.shape[1] == v.shape[1]
        and q.shape[2] >= _min_seq
        and q.is_cuda
    )


def dit_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Route eligible DiT attention through AITER forward/efficient backward."""
    if _dit_eligible(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
        _stats["routed"] += 1
        softmax_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
        if torch.is_grad_enabled() and (
            query.requires_grad or key.requires_grad or value.requires_grad
        ):
            return TritonFwdEfficientBwd.apply(query, key, value, softmax_scale)
        return _triton_forward(query, key, value, softmax_scale)[0]
    _stats["stock"] += 1
    return _original_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def install_triton_forward(min_seq: int = 1024) -> None:
    """Install the selective DiT SDPA route."""
    global _min_seq
    _min_seq = int(min_seq)
    if not _claim_mode("dit"):
        return
    from aiter.ops.triton.attention.mha import flash_attn_func  # noqa: F401

    F.scaled_dot_product_attention = dit_scaled_dot_product_attention
    logger.info("Installed Lumen DiT SDPA route for sequence lengths >= %d", _min_seq)


def uninstall_sdpa_patch() -> None:
    """Restore PyTorch's original SDPA function."""
    global _active_mode, _original_sdpa
    if _original_sdpa is not None:
        F.scaled_dot_product_attention = _original_sdpa
    _original_sdpa = None
    _active_mode = None


def uninstall_triton_forward() -> None:
    """Compatibility name for removing the DiT route."""
    if _active_mode == "dit":
        uninstall_sdpa_patch()


def stats() -> dict[str, int]:
    """Return DiT calls routed to AITER and left on stock SDPA."""
    return dict(_stats)


__all__ = [
    "TritonFwdEfficientBwd",
    "dit_scaled_dot_product_attention",
    "install_triton_forward",
    "patch_hf_sdpa",
    "patch_sdpa",
    "prefer_efficient_backend",
    "stats",
    "uninstall_sdpa_patch",
    "uninstall_triton_forward",
]
