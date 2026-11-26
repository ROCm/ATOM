# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Proximal Policy Optimization (PPO) with GAE and value loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from atomrl.algorithms.base_algorithm import BaseAlgorithm
from atomrl.algorithms.loss_functions import (
    kl_penalty,
    policy_gradient_loss,
    value_loss,
)
from atomrl.core.protocol import DataProto
from atomrl.core.registry import ALGORITHM_REGISTRY
from atomrl.core.types import AlgorithmName


class PPOAlgorithm(BaseAlgorithm):
    """Standard PPO with clipped surrogate, optional KL to reference, and value loss."""

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        required = ("log_probs", "old_log_probs", "advantages", "values", "returns")
        for k in required:
            if k not in batch.tensors:
                raise KeyError(f"PPO loss missing tensor '{k}'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        values = batch.tensors["values"]
        rets = batch.tensors["returns"]
        mask = batch.tensors.get("attention_mask")
        if mask is None:
            mask_t = torch.ones_like(adv)
        else:
            mask_t = mask.to(dtype=adv.dtype)

        clip = self._config.algorithm.ppo.clip_ratio
        pg = policy_gradient_loss(logp, old_logp, adv, clip, mask=mask_t)

        old_values = batch.tensors.get("old_values", values.detach())
        vf = value_loss(values, old_values, rets, clip, mask=mask_t)

        loss = pg + 0.5 * vf
        metrics: dict[str, Any] = {
            "loss_pg": float(pg.detach().cpu()),
            "loss_vf": float(vf.detach().cpu()),
        }

        kl_c = self._config.algorithm.ppo.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=mask_t.bool())
            loss = loss + kl_c * kl
            metrics["kl"] = float(kl.detach().cpu())

        # Optional value clip diagnostics
        with torch.no_grad():
            metrics["explained_variance"] = float(
                1.0
                - F.mse_loss(values[mask_t.bool()], rets[mask_t.bool()])
                / torch.var(rets[mask_t.bool()].float()).clamp_min(1e-8)
                if bool(mask_t.sum() > 0)
                else 0.0
            )

        metrics["loss_total"] = float(loss.detach().cpu())
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.PPO.value, PPOAlgorithm)
