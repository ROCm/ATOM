# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""On-Policy Distillation (OPD) algorithm.

Implements DeepSeek-V4 style on-policy distillation where the student model
generates sequences (rollout), then minimises KL divergence against teacher
logits at every token position.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from atomrl.algorithms.base_algorithm import BaseAlgorithm
from atomrl.algorithms.loss_functions import opd_kl_divergence
from atomrl.core.protocol import DataProto
from atomrl.core.registry import ALGORITHM_REGISTRY
from atomrl.core.types import AlgorithmName

logger = logging.getLogger(__name__)


def _response_mask(batch: DataProto) -> Tensor | None:
    if "response_mask" in batch.tensors:
        return batch.tensors["response_mask"].to(dtype=torch.bool)
    if "attention_mask" in batch.tensors:
        return batch.tensors["attention_mask"].to(dtype=torch.bool)
    return None


class OPDAlgorithm(BaseAlgorithm):
    """On-Policy Distillation: minimise KL(student || teacher) with no reward signal.

    Requires the batch to contain ``student_logits`` and ``teacher_logits``
    (full-vocabulary, ``[B, T, V]``).
    """

    def compute_advantages(self, batch: DataProto) -> DataProto:
        """No-op: OPD does not use reward-based advantages."""
        return batch

    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        if "student_logits" not in batch.tensors:
            raise KeyError("OPD requires 'student_logits' on the batch.")
        if "teacher_logits" not in batch.tensors:
            raise KeyError("OPD requires 'teacher_logits' on the batch.")

        cfg = self._config.algorithm.opd
        mask = _response_mask(batch)

        position_weights = None
        if cfg.position_weighting:
            T_len = batch["student_logits"].shape[1]
            position_weights = torch.tensor(
                [cfg.position_decay**i for i in range(T_len)],
                dtype=torch.float32,
            )

        kl = opd_kl_divergence(
            batch["student_logits"],
            batch["teacher_logits"].detach(),
            mask=mask,
            kl_direction=cfg.kl_direction,
            temperature=cfg.temperature,
            position_weights=position_weights,
        )

        loss = cfg.opd_coeff * kl
        metrics: dict[str, Any] = {
            "opd_kl": float(kl.detach().cpu()),
            "loss_total": float(loss.detach().cpu()),
        }
        return loss, metrics

    def get_config(self) -> dict[str, Any]:
        algo = self._config.algorithm
        return {
            "name": algo.name,
            "opd": vars(algo.opd),
        }


ALGORITHM_REGISTRY.register(AlgorithmName.OPD.value, OPDAlgorithm)
