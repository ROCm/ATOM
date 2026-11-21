"""RL optimization algorithms and registration."""

from __future__ import annotations

from atomrl.algorithms.advantage_estimators import ADV_ESTIMATOR_REGISTRY
from atomrl.algorithms.base_algorithm import BaseAlgorithm
from atomrl.algorithms.dapo import DAPOAlgorithm
from atomrl.algorithms.grpo import GRPOAlgorithm
from atomrl.algorithms.opd import OPDAlgorithm
from atomrl.algorithms.ppo import PPOAlgorithm
from atomrl.core.registry import ALGORITHM_REGISTRY

__all__ = [
    "ADV_ESTIMATOR_REGISTRY",
    "ALGORITHM_REGISTRY",
    "BaseAlgorithm",
    "DAPOAlgorithm",
    "GRPOAlgorithm",
    "OPDAlgorithm",
    "PPOAlgorithm",
]
