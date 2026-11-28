# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/trainer/ppo/core_algos.py
# Source: https://github.com/verl-project/verl/blob/ca439aa5afcfe913c075570a9cb3f0f83730904d/verl/trainer/ppo/ray_trainer.py

# Derived from verl (verl-project/verl):
#   verl/trainer/ppo/core_algos.py L153-212 (AdaptiveKLController, FixedKLController, get_kl_controller)
#   verl/trainer/ppo/core_algos.py L2126-2189 (kl_penalty, kl_penalty_forward)
#   verl/trainer/ppo/ray_trainer.py L76-115 (apply_kl_penalty)
#
# Licensed under the Apache License, Version 2.0.
"""Adaptive / fixed KL controllers and trainer-level KL penalty application."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KL Controllers  (verl/trainer/ppo/core_algos.py L153-212)
# ---------------------------------------------------------------------------


class AdaptiveKLController:
    """Adaptive KL controller from https://arxiv.org/pdf/1909.08593.pdf"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int) -> None:
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller (no-op update)."""

    def __init__(self, kl_coef: float) -> None:
        self.value = kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(
    kl_ctrl_type: str, kl_coef: float, target_kl: float = 0.01, horizon: int = 10000
) -> AdaptiveKLController | FixedKLController:
    """Factory for KL controllers.  (verl/trainer/ppo/core_algos.py L193-212)"""
    if kl_ctrl_type == "fixed":
        return FixedKLController(kl_coef=kl_coef)
    elif kl_ctrl_type == "adaptive":
        assert horizon > 0, f"horizon must be > 0, got {horizon}"
        return AdaptiveKLController(
            init_kl_coef=kl_coef, target_kl=target_kl, horizon=horizon
        )
    else:
        raise ValueError(
            f"Unknown kl_ctrl_type: {kl_ctrl_type!r}. Use 'fixed' or 'adaptive'."
        )


# ---------------------------------------------------------------------------
# KL penalty functions  (verl/trainer/ppo/core_algos.py L2126-2189)
# ---------------------------------------------------------------------------


def kl_penalty_forward(
    logprob: Tensor, ref_logprob: Tensor, kl_penalty_type: str
) -> Tensor:
    """Compute KL divergence estimate.  See http://joschu.net/blog/kl-approx.html"""
    if kl_penalty_type in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty_type in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty_type in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    raise NotImplementedError(f"Unknown kl_penalty type: {kl_penalty_type!r}")


def kl_penalty_token(
    logprob: Tensor, ref_logprob: Tensor, kl_penalty_type: str
) -> Tensor:
    """Token-level KL penalty with optional straight-through gradient trick.

    If ``kl_penalty_type`` ends with ``'+'`` (e.g. ``'k3+'``), uses the k2 (MSE)
    gradient estimator via straight-through while keeping the forward value.
    (verl/trainer/ppo/core_algos.py L2126-2151)
    """
    base_type = kl_penalty_type.rstrip("+")
    forward_score = kl_penalty_forward(logprob, ref_logprob, base_type)
    if not kl_penalty_type.endswith("+") or kl_penalty_type in ("mse", "k2"):
        return forward_score
    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward_score.detach()


# ---------------------------------------------------------------------------
# Trainer-level apply_kl_penalty  (verl/trainer/ppo/ray_trainer.py L76-115)
# ---------------------------------------------------------------------------


def masked_mean(values: Tensor, mask: Tensor, axis: int = -1) -> Tensor:
    masked = values * mask
    return masked.sum(dim=axis) / mask.sum(dim=axis).clamp(min=1.0)


def apply_kl_penalty(
    batch: DataProto,
    kl_ctrl: AdaptiveKLController | FixedKLController,
    kl_penalty_type: str = "kl",
) -> tuple[DataProto, dict[str, Any]]:
    """Apply KL penalty to token-level rewards and update the KL controller.

    Modifies ``batch.tensors["rewards"]`` in-place:
        ``token_level_rewards = rewards - beta * KL(pi || pi_ref)``

    Returns ``(batch, kl_metrics)``.
    (verl/trainer/ppo/ray_trainer.py L76-115)
    """
    response_mask = batch.tensors.get("response_mask")
    if response_mask is None:
        response_mask = batch.tensors.get("attention_mask")
    if response_mask is None:
        raise KeyError("apply_kl_penalty requires 'response_mask' or 'attention_mask'.")
    response_mask_f = response_mask.to(dtype=torch.float32)

    old_log_probs = batch.tensors["old_log_probs"]
    ref_log_probs = batch.tensors["ref_log_probs"]
    rewards = batch.tensors["rewards"]

    kld = kl_penalty_token(old_log_probs, ref_log_probs, kl_penalty_type)
    kld = kld * response_mask_f

    beta = kl_ctrl.value

    if rewards.dim() == 1 or (rewards.dim() == 2 and rewards.shape[-1] == 1):
        # Scalar rewards — scatter to last token before subtracting KL
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        from atomrl.algorithms.advantage_estimators import (
            _response_mask_from_batch,
            _scalar_rewards_to_token_rewards,
        )

        mask_for_scatter = _response_mask_from_batch(batch)
        if mask_for_scatter is None:
            mask_for_scatter = response_mask
        token_rewards = _scalar_rewards_to_token_rewards(rewards, mask_for_scatter)
    else:
        token_rewards = rewards.to(dtype=torch.float32)

    token_level_rewards = token_rewards - beta * kld

    current_kl = masked_mean(kld, mask=response_mask_f, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    batch_size = old_log_probs.shape[0]
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    batch.tensors["rewards"] = token_level_rewards
    batch.tensors["token_level_scores"] = token_rewards
    batch.meta["kl_penalty_applied"] = True

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }
    return batch, metrics
