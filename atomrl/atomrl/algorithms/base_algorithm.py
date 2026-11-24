# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Abstract base class for RL optimization algorithms."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from atomrl.core.config import AtomRLConfig
from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """Algorithm interface: advantages, loss, and serialized hyperparameters."""

    def __init__(self, config: AtomRLConfig) -> None:
        self._config = config

    def compute_advantages(self, batch: DataProto) -> DataProto:
        """Populate ``advantages`` (and optionally related tensors) on ``batch``.

        Default implementation delegates to the pluggable advantage estimator
        registry.  Subclasses may override for custom behaviour.
        """
        from atomrl.algorithms.advantage_estimators import ADV_ESTIMATOR_REGISTRY

        estimator_name = self._resolve_estimator()
        if estimator_name not in ADV_ESTIMATOR_REGISTRY:
            raise KeyError(
                f"Unknown advantage estimator '{estimator_name}'. "
                f"Available: {list(ADV_ESTIMATOR_REGISTRY.keys())}"
            )
        estimator_fn = ADV_ESTIMATOR_REGISTRY[estimator_name]
        return estimator_fn(batch, self._config)

    def _resolve_estimator(self) -> str:
        """Choose an estimator name from explicit config or algorithm name."""
        explicit = self._config.algorithm.adv_estimator
        if explicit:
            return explicit
        name = self._config.algorithm.name.lower()
        mapping = {"grpo": "grpo", "dapo": "dapo", "ppo": "gae", "opd": "grpo"}
        return mapping.get(name, "grpo")

    @abstractmethod
    def compute_loss(self, batch: DataProto) -> tuple[Tensor, dict[str, Any]]:
        """Return scalar loss and a dict of detached metrics for logging."""

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of algorithm-relevant settings."""
        algo = self._config.algorithm
        return {
            "name": algo.name,
            "grpo": vars(algo.grpo),
            "dapo": vars(algo.dapo),
            "ppo": vars(algo.ppo),
            "opd": vars(algo.opd),
            "spec_distill": vars(algo.spec_distill),
        }
