# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen-native fused MoE router ops replacing TE equivalents.

Three torch.autograd.Function subclasses providing:
1. fused_topk_with_score_function -- softmax + topk + scatter (Component 1)
2. fused_compute_score_for_moe_aux_loss -- softmax + topk for aux loss (Component 2)
3. fused_moe_aux_loss -- Switch load-balancing loss (Component 3)
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _aiter_softmax_topk(logits_fp32: Tensor, k: int, need_renorm: bool):
    """Call AITER's public top-k router and return full softmax scores."""
    num_tokens = logits_fp32.shape[0]
    device = logits_fp32.device
    topk_weights = torch.empty(num_tokens, k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(num_tokens, k, dtype=torch.int32, device=device)
    token_expert_indices = torch.empty(num_tokens, k, dtype=torch.int32, device=device)

    from aiter.ops.moe_op import topk_softmax

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        logits_fp32.contiguous(),
        need_renorm,
    )
    scores = torch.softmax(logits_fp32, dim=-1)
    return scores, topk_weights, topk_indices


def _pytorch_softmax_topk(logits_fp32: Tensor, k: int, need_renorm: bool):
    """Pure PyTorch fallback for softmax + topk."""
    scores = torch.softmax(logits_fp32, dim=-1)
    topk_weights, topk_indices = scores.topk(k, dim=-1, largest=True, sorted=True)
    if need_renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return scores, topk_weights, topk_indices


class LumenFusedComputeScoreForMoEAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, topk: int) -> tuple[Tensor, Tensor]:
        logits_fp32 = logits.float()

        scores, _, topk_indices = _aiter_softmax_topk(logits_fp32, topk, False)

        N, E = scores.shape
        routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
        routing_map.scatter_(1, topk_indices.long(), True)

        ctx.save_for_backward(scores)

        return routing_map, scores

    @staticmethod
    def backward(ctx, grad_routing_map, grad_scores):
        (scores,) = ctx.saved_tensors
        dot = (grad_scores * scores).sum(dim=-1, keepdim=True)
        grad_logits = scores * (grad_scores - dot)
        return grad_logits, None


class LumenFusedTopkWithScoreFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: Tensor,
        topk: int,
        use_pre_softmax: bool,
        scaling_factor: float | None,
    ) -> tuple[Tensor, Tensor]:
        N, E = logits.shape

        if use_pre_softmax:
            logits_fp32 = logits.float()
            s, w, topk_indices = _aiter_softmax_topk(logits_fp32, topk, True)

            s_k = s.gather(1, topk_indices.long())
            V = s_k.sum(dim=-1, keepdim=True)

            if scaling_factor is not None:
                w = w * scaling_factor

            routing_probs = torch.zeros(N, E, dtype=w.dtype, device=logits.device)
            routing_probs.scatter_(1, topk_indices.long(), w)

            routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
            routing_map.scatter_(1, topk_indices.long(), True)

            ctx.save_for_backward(s, s_k, topk_indices.long(), V)
            ctx.scaling_factor = scaling_factor
            ctx.use_pre_softmax = True

            return routing_map, routing_probs

        else:
            ctx.use_pre_softmax = False
            logits_fp32 = logits.float().detach().requires_grad_(True)
            with torch.enable_grad():
                _, topk_indices = logits_fp32.topk(topk, dim=-1)
                gathered = logits_fp32.gather(1, topk_indices)
                w = torch.softmax(gathered, dim=-1)

                if scaling_factor is not None:
                    w = w * scaling_factor

                routing_probs = torch.zeros(N, E, dtype=w.dtype, device=logits.device)
                routing_probs.scatter_(1, topk_indices, w)

            routing_map = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
            routing_map.scatter_(1, topk_indices, True)

            ctx.save_for_backward(logits_fp32, routing_probs)
            ctx._pre_softmax_false_graph = routing_probs

            return routing_map, routing_probs.detach()

    @staticmethod
    def backward(ctx, grad_routing_map, grad_routing_probs):
        if ctx.use_pre_softmax:
            s, s_k, topk_indices, V = ctx.saved_tensors
            N, E = s.shape

            grad_w = grad_routing_probs.gather(1, topk_indices)

            if ctx.scaling_factor is not None:
                grad_w = grad_w * ctx.scaling_factor

            w = s_k / V
            grad_s_k = (grad_w - (grad_w * w).sum(dim=-1, keepdim=True)) / V

            grad_s = torch.zeros(N, E, dtype=s.dtype, device=s.device)
            grad_s.scatter_add_(1, topk_indices, grad_s_k)

            dot = (grad_s * s).sum(dim=-1, keepdim=True)
            grad_logits = s * (grad_s - dot)

            return grad_logits, None, None, None

        else:
            logits_fp32, routing_probs = ctx.saved_tensors
            routing_probs.backward(grad_routing_probs)
            return logits_fp32.grad, None, None, None


class LumenFusedMoEAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        probs: Tensor,
        tokens_per_expert: Tensor,
        total_num_tokens: int,
        num_experts: int,
        topk: int,
        coeff: float,
    ) -> Tensor:
        C = num_experts * coeff / (topk * total_num_tokens * total_num_tokens)
        ctx.save_for_backward(tokens_per_expert)
        ctx.C = C
        ctx.N = probs.shape[0]
        ctx.E = num_experts

        return (probs.float().sum(dim=0) * tokens_per_expert.float()).sum() * C

    @staticmethod
    def backward(ctx, grad_aux_loss: Tensor):
        (tokens_per_expert,) = ctx.saved_tensors

        grad_probs = (
            tokens_per_expert.float().unsqueeze(0).expand(ctx.N, ctx.E)
            * ctx.C
            * grad_aux_loss
        )

        return grad_probs, None, None, None, None, None


def fused_topk_with_score_function(
    logits: Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int | None,
    group_topk: int | None,
    scaling_factor: float | None,
    score_function: str,
    expert_bias: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if score_function != "softmax":
        raise NotImplementedError(
            f"score_function='{score_function}' not supported, only 'softmax'"
        )
    if num_groups and num_groups > 0:
        raise NotImplementedError("Group routing (num_groups > 0) not supported")
    if group_topk and group_topk > 0:
        raise NotImplementedError("Group top-k not supported")
    if expert_bias is not None:
        raise NotImplementedError("expert_bias not supported")

    return LumenFusedTopkWithScoreFunction.apply(
        logits, topk, use_pre_softmax, scaling_factor
    )


def fused_compute_score_for_moe_aux_loss(
    logits: Tensor,
    topk: int,
    score_function: str,
) -> tuple[Tensor, Tensor]:
    if score_function != "softmax":
        raise NotImplementedError(
            f"score_function='{score_function}' not supported, only 'softmax'"
        )
    return LumenFusedComputeScoreForMoEAuxLoss.apply(logits, topk)


def fused_moe_aux_loss(
    probs: Tensor,
    tokens_per_expert: Tensor,
    total_num_tokens: int,
    num_experts: int,
    topk: int,
    coeff: float,
) -> Tensor:
    return LumenFusedMoEAuxLoss.apply(
        probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff
    )
