# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
KV Output Aggregator for Multi-Worker Transfer Coordination.

In tensor-parallel (TP) setups, each TP worker independently tracks its own
KV cache transfer progress.  The scheduler, however, needs a single unified
view of which requests have completed across *all* workers.

This module provides:

- :class:`KVOutputAggregator`: Combines per-worker outputs into a single
  scheduler-level view using a countdown-based approach.
"""

from __future__ import annotations

import logging

from atom.kv_transfer.disaggregation.types import KVConnectorOutput

logger = logging.getLogger("atom")

__all__ = ["KVOutputAggregator"]


class KVOutputAggregator:
    """Aggregates :class:`KVConnectorOutput` from all TP workers.

    Uses a countdown approach: when a request ID first appears, a counter
    is initialized to ``world_size``.  Each worker that reports it as
    finished decrements the counter.  When the counter reaches zero,
    the request is considered globally complete and is emitted.

    Args:
        world_size: Number of TP workers to aggregate over.

    Example::

        aggregator = KVOutputAggregator(world_size=8)
        per_worker_outputs = [worker.get_kv_output() for worker in workers]
        result = aggregator.aggregate(per_worker_outputs)
        # result.finished_recving contains only IDs done on ALL 8 workers
    """

    def __init__(self, world_size: int = 8) -> None:
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        self._world_size = world_size
        self._remaining_sending: dict[str, int] = {}
        self._remaining_recving: dict[str, int] = {}

    @property
    def world_size(self) -> int:
        return self._world_size

    def aggregate(self, worker_outputs: list[KVConnectorOutput]) -> KVConnectorOutput:
        """Aggregate per-worker outputs and return globally completed request IDs.

        Args:
            worker_outputs: One :class:`KVConnectorOutput` per worker.

        Returns:
            A new :class:`KVConnectorOutput` containing only request IDs
            that have been reported as finished by **all** workers.
        """
        if not worker_outputs:
            return KVConnectorOutput()

        all_sending_ids: set[str] = set()
        all_recving_ids: set[str] = set()
        for wo in worker_outputs:
            if wo.finished_sending:
                all_sending_ids.update(wo.finished_sending)
            if wo.finished_recving:
                all_recving_ids.update(wo.finished_recving)

        for rid in all_sending_ids:
            if rid not in self._remaining_sending:
                self._remaining_sending[rid] = self._world_size

        for rid in all_recving_ids:
            if rid not in self._remaining_recving:
                self._remaining_recving[rid] = self._world_size

        for wo in worker_outputs:
            if wo.finished_sending:
                for rid in wo.finished_sending:
                    if rid in self._remaining_sending:
                        self._remaining_sending[rid] -= 1

            if wo.finished_recving:
                for rid in wo.finished_recving:
                    if rid in self._remaining_recving:
                        self._remaining_recving[rid] -= 1

        done_sending = {rid for rid, cnt in self._remaining_sending.items() if cnt <= 0}
        done_recving = {rid for rid, cnt in self._remaining_recving.items() if cnt <= 0}

        for rid in done_sending:
            del self._remaining_sending[rid]
        for rid in done_recving:
            del self._remaining_recving[rid]

        return KVConnectorOutput(
            finished_sending=done_sending,
            finished_recving=done_recving,
        )

    def reset(self) -> None:
        """Clear all internal tracking state."""
        self._remaining_sending.clear()
        self._remaining_recving.clear()

    @property
    def pending_count(self) -> tuple[int, int]:
        """Return ``(num_pending_sending, num_pending_recving)``."""
        return len(self._remaining_sending), len(self._remaining_recving)
