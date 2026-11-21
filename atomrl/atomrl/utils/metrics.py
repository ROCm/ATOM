"""Lightweight metric aggregation for training loops."""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Accumulate scalar metrics and report simple running means."""

    def __init__(self) -> None:
        self._sums: defaultdict[str, float] = defaultdict(float)
        self._counts: defaultdict[str, int] = defaultdict(int)

    def update(self, key: str, value: float) -> None:
        """Record one scalar observation for ``key``."""
        self._sums[key] += float(value)
        self._counts[key] += 1

    def get_mean(self, key: str) -> float:
        """Return the mean for ``key`` or ``0.0`` if no samples were stored."""
        c = self._counts[key]
        if c == 0:
            logger.debug("MetricsTracker.get_mean: no samples for key=%s", key)
            return 0.0
        return self._sums[key] / c

    def reset(self) -> None:
        """Clear all stored statistics."""
        self._sums.clear()
        self._counts.clear()


def compute_kl_divergence(logprobs: Tensor, ref_logprobs: Tensor) -> float:
    """Monte Carlo estimate ``mean(ref_logp - logp)`` as a Python float."""
    with torch.no_grad():
        kl = (ref_logprobs - logprobs).mean()
        return float(kl.detach().cpu())


def compute_entropy(logprobs: Tensor) -> float:
    """Surrogate entropy ``mean(-exp(logp) * logp)`` as a Python float."""
    with torch.no_grad():
        ent = -(torch.exp(logprobs) * logprobs).mean()
        return float(ent.detach().cpu())
