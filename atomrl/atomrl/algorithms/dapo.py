# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Decoupled Advantage Policy Optimization (DAPO-style) algorithm."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from atomrl.algorithms.advantage_estimators import compute_dapo_advantage
from atomrl.algorithms.base_algorithm import BaseAlgorithm
from atomrl.algorithms.loss_functions import asymmetric_clip_loss, gmpo_loss, kl_penalty
from atomrl.core.protocol import DataProto
from atomrl.core.registry import ALGORITHM_REGISTRY
from atomrl.core.types import AlgorithmName


def _response_mask(batch: DataProto) -> Tensor | None:
    if "response_mask" in batch.tensors:
        return batch.tensors["response_mask"].to(dtype=torch.bool)
    if "attention_mask" in batch.tensors:
        return batch.tensors["attention_mask"].to(dtype=torch.bool)
    return None


class DAPOAlgorithm(BaseAlgorithm):
    """DAPO-style training: asymmetric clipping, dynamic sampling, token-level PG.

    Advantages are group-relative (like GRPO) with optional variance-based
    dynamic sampling. Loss uses asymmetric PPO-style clipping bounds.
    """

    def compute_advantages(self, batch: DataProto) -> DataProto:
        """Use the single verl-derived DAPO estimator implementation."""
        return compute_dapo_advantage(batch, self._config)

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        if "log_probs" not in batch.tensors or "old_log_probs" not in batch.tensors:
            raise KeyError("DAPO loss requires 'log_probs' and 'old_log_probs'.")
        if "advantages" not in batch.tensors:
            raise KeyError("DAPO loss requires 'advantages'.")

        logp = batch.tensors["log_probs"]
        old_logp = batch.tensors["old_log_probs"]
        adv = batch.tensors["advantages"]
        mask = _response_mask(batch)
        sample_mask = batch.tensors.get("dapo_sample_mask")

        cfg = self._config.algorithm.dapo
        batch_num_tokens = batch.meta.get("batch_num_tokens")
        dp_size = batch.meta.get("dp_size", 1)
        # Truncated importance sampling weights (rollout policy vs training
        # policy), aligned per response token. See verl rollout_correction.
        rollout_is_weights = batch.tensors.get("rollout_is_weights")

        # Expand sequence-level advantages [B] -> [B, T] for token-level loss.
        # Multiply by response_mask so prompt tokens get zero advantage
        # (matching verl: scores = scores.unsqueeze(-1) * response_mask).
        if adv.dim() == 1:
            adv = adv.unsqueeze(-1)
        if mask is not None:
            adv = adv.expand_as(logp) * mask.to(dtype=adv.dtype)
        else:
            adv = adv.expand_as(logp)

        # Build the combined mask for loss aggregation (response_mask * sample_mask).
        if sample_mask is not None:
            sm = sample_mask
            if sm.dim() == 1:
                sm = sm.unsqueeze(-1)
            sm = sm.expand_as(logp).to(dtype=logp.dtype)
            if mask is not None:
                sm = sm * mask.to(dtype=logp.dtype)
        else:
            sm = (
                mask.to(dtype=logp.dtype) if mask is not None else torch.ones_like(logp)
            )

        low = float(cfg.clip_ratio_low)
        high = float(cfg.clip_ratio_high)
        clip_c = float(getattr(cfg, "clip_ratio_c", 0.0))
        loss_mode = getattr(cfg, "loss_mode", "token_level")

        # Align rollout IS weights to the response mask shape if present.
        ris = None
        if rollout_is_weights is not None:
            ris = rollout_is_weights
            if ris.dim() == 1:
                ris = ris.unsqueeze(-1)
            if ris.shape != logp.shape:
                try:
                    ris = ris.expand_as(logp)
                except RuntimeError:
                    ris = None

        if loss_mode == "gmpo":
            # GMPO: token-level log-ratio clip → geometric mean ratio → seq-level advantage
            pg = gmpo_loss(logp, old_logp, adv, low, high, mask=sm)
        elif cfg.token_level_pg:
            pg = asymmetric_clip_loss(
                logp,
                old_logp,
                adv,
                low,
                high,
                mask=sm,
                clip_ratio_c=clip_c,
                batch_num_tokens=batch_num_tokens,
                dp_size=dp_size,
                rollout_is_weights=ris,
            )
        else:
            # Sequence-level: mean logp per row, scalar adv per row
            denom = torch.clamp(sm.sum(dim=-1, keepdim=True), min=1.0)
            seq_logp = (logp * sm).sum(dim=-1, keepdim=True) / denom
            seq_old = (old_logp * sm).sum(dim=-1, keepdim=True) / denom
            seq_adv = adv.sum(dim=-1, keepdim=True) / torch.clamp(
                sm.sum(dim=-1, keepdim=True), min=1.0
            )
            pg = asymmetric_clip_loss(
                seq_logp,
                seq_old,
                seq_adv,
                low,
                high,
                mask=None,
                clip_ratio_c=clip_c,
            )

        loss = pg
        metrics: dict[str, Any] = {"loss_pg": float(pg.detach().cpu())}

        kl_c = cfg.kl_coeff
        if kl_c > 0.0 and "ref_log_probs" in batch.tensors:
            kl = kl_penalty(logp, batch.tensors["ref_log_probs"], mask=mask)
            loss = loss + kl_c * kl
            metrics["kl"] = float(kl.detach().cpu())

        metrics["loss_total"] = float(loss.detach().cpu())
        return loss, metrics


ALGORITHM_REGISTRY.register(AlgorithmName.DAPO.value, DAPOAlgorithm)
