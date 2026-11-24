# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Deferred BDA (bias-dropout-add) + RMSNorm for TransformerLayer.

When ``hidden_dropout=0`` and cross-attention is a no-op (decoder-only),
the self-attention BDA reduces to ``x + residual``.  This module defers
that add from ``_forward_attention`` to ``_forward_mlp``, where it is
merged with the RMSNorm computation.  This eliminates one kernel launch
(the standalone BDA add) by computing ``residual_out = x + residual``
and ``normed = RMSNorm(residual_out)`` back-to-back.

IMPORTANT: The fused CK/Triton ``fused_add_rms_norm`` kernels do NOT
participate in PyTorch autograd and silently break backward when inputs
require grad.  This module intentionally uses ``torch.add`` + Lumen's
autograd-aware ``rmsnorm()`` to preserve correct training gradients.

Gated by ``LUMEN_FUSED_RESIDUAL_NORM=1`` environment variable.

Public API
----------
- :func:`can_fuse_bda_norm` — check if deferred BDA+norm is safe for a layer
- :func:`deferred_bda_add` — consume deferred BDA operands, do the add
- :func:`rmsnorm_from_module` — extract weight/eps from a norm module, call rmsnorm
- :func:`deferred_bda_rmsnorm` — full pipeline: add + norm from deferred operands
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_FUSED_RESIDUAL_NORM = os.environ.get("LUMEN_FUSED_RESIDUAL_NORM", "0") == "1"


def is_enabled() -> bool:
    """Whether deferred BDA + RMSNorm fusion is enabled."""
    return _FUSED_RESIDUAL_NORM


def get_norm_params(norm_module: torch.nn.Module) -> tuple[torch.Tensor, float]:
    """Extract weight and eps from a norm module.

    Handles Lumen/Megatron wrapper types where the actual norm may live
    under ``norm_module._norm``.

    Args:
        norm_module: An ``nn.Module`` with a ``.weight`` parameter and
            ``.eps`` / ``.epsilon`` attribute.

    Returns:
        ``(weight, eps)``
    """
    w = norm_module.weight
    if hasattr(norm_module, "_norm"):
        eps = getattr(
            norm_module._norm, "eps", getattr(norm_module._norm, "epsilon", 1e-5)
        )
    else:
        eps = getattr(norm_module, "eps", getattr(norm_module, "epsilon", 1e-5))
    return w, eps


def rmsnorm_from_module(
    x: torch.Tensor,
    norm_module: torch.nn.Module,
) -> torch.Tensor:
    """Apply RMSNorm by extracting weight/eps from *norm_module*.

    Uses Lumen's autograd-aware ``rmsnorm()`` functional API, which
    dispatches to AITER Triton (preferred when grad is needed) or CK.

    Args:
        x: Input tensor ``(*, hidden_size)``.
        norm_module: The layernorm module (e.g. ``self.pre_mlp_layernorm``).

    Returns:
        Normalised tensor with same shape as *x*.
    """
    from atomrl.lumen.ops.normalization.rmsnorm import rmsnorm

    w, eps = get_norm_params(norm_module)
    return rmsnorm(x, w, eps)


def deferred_bda_add(
    attention_output_with_bias: tuple[torch.Tensor, torch.Tensor | None],
    residual: torch.Tensor,
) -> torch.Tensor:
    """Compute the deferred BDA residual add: ``x + bias + residual``.

    When ``hidden_dropout=0``, the Megatron BDA reduces to a simple add.
    This function performs that add explicitly with standard PyTorch ops
    (autograd-safe).

    Args:
        attention_output_with_bias: ``(x, bias)`` tuple from self-attention.
            ``bias`` may be ``None``.
        residual: The residual stream tensor.

    Returns:
        ``hidden_states = x [+ bias] + residual``
    """
    x, bias = attention_output_with_bias
    if bias is not None:
        x = x + bias
    return x + residual


def deferred_bda_rmsnorm(
    attention_output_with_bias: tuple[torch.Tensor, torch.Tensor | None],
    residual: torch.Tensor,
    norm_module: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deferred BDA add + RMSNorm in one call.

    Combines :func:`deferred_bda_add` and :func:`rmsnorm_from_module` into
    a single logical operation for cleaner call sites.

    Args:
        attention_output_with_bias: ``(x, bias)`` from self-attention.
        residual: The residual stream.
        norm_module: The ``pre_mlp_layernorm`` module.

    Returns:
        ``(normed_output, updated_residual)`` where
        ``updated_residual = x [+ bias] + residual`` and
        ``normed_output = RMSNorm(updated_residual)``.
    """
    hidden_states = deferred_bda_add(attention_output_with_bias, residual)
    normed = rmsnorm_from_module(hidden_states, norm_module)
    return normed, hidden_states
