# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark's pure arithmetic, apart from ``muse_spark`` so that it imports
without AITER and the CPU tests can run it."""

import torch

# Reference constants that are architecture, not config.
_ROUTE_EPS = 1e-15
_RESIDUAL_GATE_EPS = 1e-3


def muse_rms(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim in fp32; returns fp32."""
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out


def muse_route(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    multiplier: float,
    route_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled-sigmoid routing; the bias picks experts but does not weight them."""
    scores = torch.sigmoid(router_logits.float() * multiplier)
    _, ids = torch.topk(scores + bias.float(), top_k, dim=-1)
    picked = scores.gather(1, ids)
    picked = picked / (picked.sum(-1, keepdim=True) + _ROUTE_EPS) * route_scale
    return picked, ids.to(torch.int32)


def residual_gate_coeffs(
    gate: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(alpha, beta)`` of the norm-preserving gated residual, in fp32."""
    w = gate.float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(
        torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=_RESIDUAL_GATE_EPS)
    )
    return alpha, beta


def post_expert_combine(
    y: torch.Tensor, scores: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-slot input-scaled RMSNorm, then the routing-weighted sum over slots.

    ``y`` is ``[tokens, top_k, dim]`` complete expert outputs (zeros for a slot
    another rank owns), ``weight`` the effective norm weight. The scale is
    applied and rounded in ``y``'s dtype before the norm (the reference uses fp32).
    """
    scaled = (weight.to(y.dtype) * y).float()
    inv = torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)
    out = (scores.float().unsqueeze(-1) * scaled * inv).sum(1)
    return out.to(torch.bfloat16)


def muse_logits(
    logits: torch.Tensor, multiplier: float, cap: float | None
) -> torch.Tensor:
    """``tanh(logits * multiplier / cap) * cap`` in fp32; no softcap when
    ``cap`` is None or not positive."""
    if cap is None or cap <= 0:
        return logits.float() * multiplier
    return torch.tanh(logits.float() * (multiplier / cap)) * cap
