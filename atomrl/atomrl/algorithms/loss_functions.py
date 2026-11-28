# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/trainer/ppo/core_algos.py

"""Shared PyTorch loss helpers for policy optimization algorithms."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from atomrl.utils.torch_functional import masked_sum

logger = logging.getLogger(__name__)

__all__ = [
    "agg_loss",
    "asymmetric_clip_loss",
    "entropy_bonus",
    "gmpo_loss",
    "hidden_state_loss",
    "kl_penalty",
    "opd_kl_divergence",
    "policy_gradient_loss",
    "reduce_reported_loss",
    "sft_loss",
    "value_loss",
]


def reduce_reported_loss(unit_losses) -> float:
    """Reduce per-accumulation-unit losses into the scalar reported as ``loss``.

    An "accumulation unit" is whatever the engine backwards one at a time: a row
    on the Megatron path, a packed micro-batch on the FSDP path. ``agg_loss`` has
    already divided each unit by the GLOBAL ``batch_num_tokens`` and scaled it by
    ``dp_size``, so the SUM over units is the globally-normalized objective -- the
    quantity that was actually differentiated.

    It must therefore be a sum and never a mean. Both engines used to average
    instead, each over its own unit count, and those counts are properties of the
    configuration rather than of the model: rows-per-rank is ``global_batch / DP``
    and micro-batch count follows ``max_token_len_per_gpu``. Two runs optimizing
    the same objective to within 2% reported losses 17x apart (Qwen3-8B DAPO,
    DP=2 vs DP=8). Keeping the reduction here means the two paths cannot drift
    apart again.
    """
    if isinstance(unit_losses, (list, tuple)):
        return float(sum(unit_losses))
    return float(unit_losses)


def agg_loss(
    loss_mat: Tensor,
    loss_mask: Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: int | None = None,
    global_batch_size: int | None = None,
) -> Tensor:
    """Aggregate token losses using verl-compatible reduction modes."""
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = int(loss_mask.sum().item())
        return masked_sum(loss_mat, loss_mask) / max(batch_num_tokens, 1) * dp_size

    if loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = int(seq_mask.sum().item())
        loss = masked_sum(seq_losses, seq_mask) / max(global_batch_size, 1) * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            loss = loss / loss_mask.shape[-1]
        return loss

    if loss_agg_mode == "seq-mean-token-mean":
        seq_count = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_count + 1e-8)
        seq_mask = (seq_count > 0).float()
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = int(seq_mask.sum().item())
        return masked_sum(seq_losses, seq_mask) / max(global_batch_size, 1) * dp_size

    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode!r}")


def sft_loss(
    log_probs: Tensor,
    loss_mask: Tensor,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    dp_group=None,
) -> tuple[Tensor, dict[str, float]]:
    """Supervised fine-tuning NLL loss over masked tokens.

    Args:
        log_probs: Per-token log-probabilities ``[B, T]``.
        loss_mask: Binary mask ``[B, T]`` (1 = supervised token).
        loss_agg_mode: Aggregation mode forwarded to :func:`agg_loss`.
        dp_size: Data-parallel world size for global normalization.
        dp_group: Optional process group for cross-rank token count all-reduce.

    Returns:
        ``(loss, metrics_dict)`` matching the worker loss_fn contract.
    """
    import torch.distributed as dist

    batch_num_tokens: int | None = None
    global_batch_size: int | None = None

    if loss_agg_mode == "token-mean":
        num_tokens = loss_mask.sum()
        if dp_group is not None and dist.is_initialized():
            dist.all_reduce(num_tokens, group=dp_group)
        batch_num_tokens = max(int(num_tokens.item()), 1)
    else:
        seq_mask = (loss_mask.sum(dim=-1) > 0).float()
        num_seqs = seq_mask.sum()
        if dp_group is not None and dist.is_initialized():
            dist.all_reduce(num_seqs, group=dp_group)
        global_batch_size = max(int(num_seqs.item()), 1)

    loss = agg_loss(
        -log_probs,
        loss_mask,
        loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
    )
    return loss, {
        "sft_loss": float(loss.detach()),
        "num_tokens": float(loss_mask.sum().detach()),
    }


def policy_gradient_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    clip_ratio: float,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Clipped surrogate policy-gradient (PPO-style) objective as a loss to minimize.

    Computes ``-mean(min(r_t * A, clip(r_t) * A))`` where ``r_t = exp(logp - logp_old)``.

    Args:
        logprobs: Current log-probabilities, shape broadcastable with ``old_logprobs``.
        old_logprobs: Behavior-policy log-probabilities (same shape as ``logprobs``).
        advantages: Advantage estimates (same shape as ``logprobs``).
        clip_ratio: Symmetric clipping radius ``epsilon``; ratio is clamped to
            ``[1 - clip_ratio, 1 + clip_ratio]``.
        mask: Optional boolean/float mask of the same shape as ``logprobs`` for
            token-level masking (1 = include).

    Returns:
        Scalar tensor: clipped surrogate loss (negative of the PPO objective).
    """
    ratio = torch.exp(logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    pg = -torch.minimum(surr1, surr2)
    if mask is not None:
        w = mask.to(dtype=pg.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (pg * w).sum() / denom
    return pg.mean()


def asymmetric_clip_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    clip_low: float,
    clip_high: float,
    *,
    mask: Tensor | None = None,
    clip_ratio_c: float = 0.0,
    batch_num_tokens: int | None = None,
    dp_size: int = 1,
    rollout_is_weights: Tensor | None = None,
    loss_agg_mode: str = "token-mean",
    global_batch_size: int | None = None,
) -> Tensor:
    """Policy-gradient surrogate with asymmetric ratio clipping (DAPO-style).

    ``clip_low`` and ``clip_high`` are **epsilon** values: the importance ratio
    ``exp(logp - logp_old)`` is clamped to ``[1 - clip_low, 1 + clip_high]``.

    When ``clip_ratio_c > 0``, applies DAPO dual-clip: for negative advantages,
    the loss is capped at ``-advantages * clip_ratio_c``, preventing domination
    by highly negative advantages and preserving gradient signal for positive
    advantage tokens. Matches verl's implementation.

    Args:
        logprobs: Current log-probabilities.
        old_logprobs: Reference log-probabilities.
        advantages: Advantage tensor (broadcastable).
        clip_low: Lower epsilon; ratio lower bound is ``1 - clip_low`` (e.g. 0.2 → 0.8).
        clip_high: Upper epsilon; ratio upper bound is ``1 + clip_high`` (e.g. 0.28 → 1.28).
        mask: Optional token mask for masked mean reduction.
        clip_ratio_c: Dual-clip bound C (DAPO). 0 disables. Typical value: 3.0.
        batch_num_tokens: Global token count across all GPUs. When provided,
            uses Verl-aligned normalization: ``pg.sum() / batch_num_tokens * dp_size``.
        dp_size: Data-parallel world size (for FSDP all-reduce compensation).
        rollout_is_weights: Optional per-token truncated importance-sampling
            weights (TIS/MIS) ``exp(old_logp - rollout_logp)`` between the
            training policy and the rollout (e.g. FP8 / vLLM) policy. When
            provided, the per-token PG loss is multiplied by these weights
            *before* aggregation (verl ``pg_losses * rollout_is_weights``).
        loss_agg_mode: Aggregation mode forwarded to :func:`agg_loss`.
        global_batch_size: Global sequence count used by sequence-mean modes.

    Returns:
        Scalar loss tensor to minimize.
    """
    neg_approx_kl = logprobs - old_logprobs
    neg_approx_kl = torch.clamp(neg_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(neg_approx_kl)
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * clipped
    pg = torch.maximum(pg_losses1, pg_losses2)

    if clip_ratio_c > 0.0:
        pg_losses3 = -advantages * clip_ratio_c
        pg_clipped = torch.minimum(pg_losses3, pg)
        pg = torch.where(advantages < 0, pg_clipped, pg)

    if rollout_is_weights is not None:
        pg = pg * rollout_is_weights.to(dtype=pg.dtype)

    if mask is None and loss_agg_mode == "token-mean" and batch_num_tokens is None:
        return pg.mean()

    loss_mask = (
        mask.to(dtype=pg.dtype)
        if mask is not None
        else torch.ones_like(pg, dtype=pg.dtype)
    )
    return agg_loss(
        pg,
        loss_mask,
        loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
    )


def gmpo_loss(
    logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    clip_low: float,
    clip_high: float,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Geometric-Mean Policy Optimization (GMPO) loss.

    Instead of clipping the importance ratio in probability space, GMPO clips
    the log-ratio at the token level, computes a geometric mean ratio per
    sequence, and multiplies by the sequence-level advantage.

    Reference: https://arxiv.org/abs/2507.20673
    Matches verl's ``compute_policy_loss_geo_mean`` implementation.

    Args:
        logprobs: Current log-probabilities ``[B, T]``.
        old_logprobs: Reference log-probabilities ``[B, T]``.
        advantages: Advantage tensor ``[B, T]`` (token-level, but typically
            the same scalar broadcast across the sequence).
        clip_low: Lower clip bound on log-ratio (e.g. 0.4).
        clip_high: Upper clip bound on log-ratio (e.g. 0.4).
        mask: Response mask ``[B, T]`` (1 for response tokens, 0 for prompt/pad).

    Returns:
        Scalar loss tensor to minimize.
    """
    neg_approx_kl = logprobs - old_logprobs

    # Token-level clipping in log-space with sign-aware min (GMPO trick):
    # For positive advantage tokens: keep the smaller of (kl, clipped_kl)
    # For negative advantage tokens: keep the larger of (kl, clipped_kl)
    # This is equivalent to: clip toward zero when the update is too large.
    sgn_adv = torch.sign(advantages)
    neg_approx_kl_clamped = torch.clamp(neg_approx_kl, -clip_low, clip_high)
    neg_approx_kl_min = torch.min(
        sgn_adv * neg_approx_kl, sgn_adv * neg_approx_kl_clamped
    )
    neg_approx_kl_min = sgn_adv * neg_approx_kl_min

    if mask is not None:
        w = mask.to(dtype=neg_approx_kl_min.dtype)
        mask_sum = w.sum(dim=-1)  # [B]
        # Geometric mean of token-level ratios → sequence-level ratio
        ratio = torch.exp((neg_approx_kl_min * w).sum(dim=-1) / (mask_sum + 1e-8))
        # Sequence-level advantage: mean of token advantages
        seq_adv = (advantages * w).sum(dim=-1) / (mask_sum + 1e-8)
    else:
        ratio = torch.exp(neg_approx_kl_min.mean(dim=-1))
        seq_adv = advantages.mean(dim=-1)

    pg_losses = -seq_adv * ratio
    return pg_losses.mean()


def value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_ratio: float,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Clipped value-function regression loss (PPO-style).

    Args:
        values: Current value predictions ``V(s)``.
        old_values: Old value predictions ``V_old(s)`` from behavior critic.
        returns: TD(lambda) or Monte-Carlo return targets.
        clip_ratio: Epsilon for clipping ``values`` around ``old_values``.
        mask: Optional mask for token-level values.

    Returns:
        Scalar MSE-style clipped value loss.
    """
    values_clipped = old_values + torch.clamp(
        values - old_values, -clip_ratio, clip_ratio
    )
    loss_unclipped = F.mse_loss(values, returns, reduction="none")
    loss_clipped = F.mse_loss(values_clipped, returns, reduction="none")
    vf = torch.maximum(loss_unclipped, loss_clipped)
    if mask is not None:
        w = mask.to(dtype=vf.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (vf * w).sum() / denom
    return vf.mean()


def kl_penalty(
    logprobs: Tensor, ref_logprobs: Tensor, *, mask: Tensor | None = None
) -> Tensor:
    """Token-level KL penalty ``mean(ref_logp - logp)`` (non-negative when ref is teacher).

    This matches the common RLHF surrogate ``KL(pi || pi_ref)`` Monte Carlo estimate
    along the sampled trajectory.

    Args:
        logprobs: Log-probs under the trainable policy.
        ref_logprobs: Log-probs under the reference policy.
        mask: Optional mask for valid tokens.

    Returns:
        Scalar KL penalty tensor.
    """
    kl = ref_logprobs - logprobs
    if mask is not None:
        w = mask.to(dtype=kl.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (kl * w).sum() / denom
    return kl.mean()


def opd_kl_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    mask: Tensor | None = None,
    kl_direction: str = "reverse",
    temperature: float = 1.0,
    position_weights: Tensor | None = None,
) -> Tensor:
    """Full-vocabulary KL divergence for On-Policy Distillation.

    Unlike :func:`kl_penalty` which operates on sampled-token log-probs, this
    computes the exact KL over the entire vocabulary at each token position.

    Args:
        student_logits: Raw logits from the student model, ``[B, T, V]``.
        teacher_logits: Raw logits from the teacher model, ``[B, T, V]`` (detached).
        mask: Optional ``[B, T]`` mask (1 = include token in loss).
        kl_direction: ``"reverse"`` for ``D_KL(student || teacher)`` (DeepSeek-V4),
            ``"forward"`` for ``D_KL(teacher || student)`` (Eagle3).
        temperature: Softmax temperature applied to both distributions.
        position_weights: Optional ``[T]`` weights (e.g. ``0.8 ** i``) for
            position-dependent weighting.

    Returns:
        Scalar KL divergence loss tensor.
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)

    if kl_direction == "reverse":
        # D_KL(student || teacher) = sum_v p_s(v) * [log p_s(v) - log p_t(v)]
        student_probs = student_log_probs.exp()
        kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(
            dim=-1
        )
    elif kl_direction == "forward":
        # D_KL(teacher || student) = sum_v p_t(v) * [log p_t(v) - log p_s(v)]
        teacher_probs = teacher_log_probs.exp()
        kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(
            dim=-1
        )
    else:
        raise ValueError(
            f"Unknown kl_direction: {kl_direction!r}. Use 'reverse' or 'forward'."
        )

    if position_weights is not None:
        T_len = kl_per_token.shape[1]
        pw = position_weights[:T_len].to(
            dtype=kl_per_token.dtype, device=kl_per_token.device
        )
        kl_per_token = kl_per_token * pw.unsqueeze(0)

    if mask is not None:
        w = mask.to(dtype=kl_per_token.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (kl_per_token * w).sum() / denom
    return kl_per_token.mean()


def hidden_state_loss(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    *,
    mask: Tensor | None = None,
    loss_type: str = "mse",
    projection: torch.nn.Module | None = None,
) -> Tensor:
    """Hidden-state matching loss for speculative decoding draft model training.

    Args:
        student_hidden: Student hidden states, ``[B, T, D_s]``.
        teacher_hidden: Teacher hidden states, ``[B, T, D_t]``.
        mask: Optional ``[B, T]`` mask.
        loss_type: ``"mse"`` for mean squared error, ``"cosine"`` for cosine distance.
        projection: Optional linear layer to project ``D_s`` to ``D_t`` if they differ.

    Returns:
        Scalar loss tensor.
    """
    if projection is not None:
        student_hidden = projection(student_hidden)

    if loss_type == "mse":
        per_token = F.mse_loss(
            student_hidden.float(), teacher_hidden.float(), reduction="none"
        ).mean(dim=-1)
    elif loss_type == "cosine":
        cos_sim = F.cosine_similarity(
            student_hidden.float(), teacher_hidden.float(), dim=-1
        )
        per_token = 1.0 - cos_sim
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Use 'mse' or 'cosine'.")

    if mask is not None:
        w = mask.to(dtype=per_token.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (per_token * w).sum() / denom
    return per_token.mean()


def entropy_bonus(logprobs: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """Surrogate entropy term from sampled-action log-probabilities.

    For a sampled action ``a`` with ``logp = log pi(a|s)``, the quantity
    ``-exp(logp) * logp`` equals ``-p(a) log p(a)`` for that draw and is a
    nonnegative scalar that can be used as a per-token entropy surrogate when
    full logits are unavailable.

    Args:
        logprobs: Log-probabilities of actions that were sampled from the policy.
        mask: Optional mask for valid tokens.

    Returns:
        Scalar tensor; **subtract** ``coeff * entropy_bonus(...)`` from the loss
        to encourage larger per-token entropy mass (typical PPO sign convention).
    """
    ent = -(torch.exp(logprobs) * logprobs)
    if mask is not None:
        w = mask.to(dtype=ent.dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        return (ent * w).sum() / denom
    return ent.mean()
