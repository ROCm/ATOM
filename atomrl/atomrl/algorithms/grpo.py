# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Group Relative Policy Optimization (GRPO)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from atomrl.algorithms.base_algorithm import BaseAlgorithm
from atomrl.algorithms.loss_functions import asymmetric_clip_loss, kl_penalty
from atomrl.core.protocol import DataProto
from atomrl.core.registry import ALGORITHM_REGISTRY
from atomrl.core.types import AlgorithmName


def _response_mask(batch: DataProto) -> Tensor | None:
    if "response_mask" in batch.tensors:
        return batch.tensors["response_mask"].to(dtype=torch.bool)
    if "attention_mask" in batch.tensors:
        return batch.tensors["attention_mask"].to(dtype=torch.bool)
    return None


def _expand_adv_to_tokens(advantages: Tensor, mask: Tensor | None) -> Tensor:
    """Broadcast [B] or [B, 1] advantages to [B, T] using ``mask``."""
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    if mask is None:
        return advantages
    return advantages.expand_as(mask.to(dtype=advantages.dtype)) * mask.to(
        dtype=advantages.dtype
    )


class GRPOAlgorithm(BaseAlgorithm):
    """GRPO: group-normalized scalar rewards as advantages (no critic).

    Trajectories must be laid out so that each group of ``num_generations``
    consecutive rows corresponds to the same prompt. Configure
    ``config.algorithm.grpo.num_generations``.
    """

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        if "log_probs" not in batch.tensors or "old_log_probs" not in batch.tensors:
            raise KeyError("GRPO loss requires 'log_probs' and 'old_log_probs'.")
        if "advantages" not in batch.tensors:
            raise KeyError("GRPO loss requires precomputed 'advantages'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        mask = _response_mask(batch)
        rollout_is_weights = batch.tensors.get("rollout_is_weights")

        adv_tok = _expand_adv_to_tokens(adv, mask)
        clip_low = self._config.algorithm.grpo.clip_ratio
        clip_high = self._config.algorithm.clip_ratio_high
        if clip_high is None:
            clip_high = 0.28
        loss_agg_mode = self._config.algorithm.loss_agg_mode
        dp_size = int(batch.meta.get("dp_size", 1) or 1)
        global_batch_size = int(
            batch.meta.get("global_batch_size") or batch.batch_size * dp_size
        )
        batch_num_tokens = batch.meta.get("batch_num_tokens")
        if loss_agg_mode == "token-mean" and batch_num_tokens is None and dp_size > 1:
            token_mask = mask if mask is not None else torch.ones_like(logp)
            batch_num_tokens = int(token_mask.sum().item()) * dp_size
        pg = asymmetric_clip_loss(
            logp,
            old_logp,
            adv_tok,
            float(clip_low),
            float(clip_high),
            mask=mask,
            batch_num_tokens=batch_num_tokens,
            dp_size=dp_size,
            loss_agg_mode=loss_agg_mode,
            global_batch_size=global_batch_size,
            rollout_is_weights=rollout_is_weights,
        )

        loss = pg
        metrics: dict[str, Any] = {"loss_pg": float(pg.detach().cpu())}

        kl_c = self._config.algorithm.grpo.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=mask)
            loss = loss + kl_c * kl
            metrics["kl"] = float(kl.detach().cpu())

        metrics["loss_total"] = float(loss.detach().cpu())
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.GRPO.value, GRPOAlgorithm)
