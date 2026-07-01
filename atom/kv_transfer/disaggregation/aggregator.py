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

from atom.kv_transfer.disaggregation.types import KVConnectorOutput, ReqId

logger = logging.getLogger("atom")

__all__ = ["KVOutputAggregator"]


class KVOutputAggregator:
    """Aggregates :class:`KVConnectorOutput` from all TP workers.

    Tracks which unique worker indices have reported each request as
    finished.  A request is globally complete only when all
    ``world_size`` workers have reported it — duplicate reports from
    the same worker (e.g. from retried notifications) are ignored.

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
        self._seen_sending: dict[ReqId, set[int]] = {}
        self._seen_recving: dict[ReqId, set[int]] = {}
        self._seen_recv_failed: dict[ReqId, set[int]] = {}
        self._seen_saving: dict[ReqId, set[int]] = {}

    @property
    def world_size(self) -> int:
        return self._world_size

    def aggregate(self, worker_outputs: list[KVConnectorOutput]) -> KVConnectorOutput:
        """Aggregate per-worker outputs and return globally completed request IDs.

        Args:
            worker_outputs: One :class:`KVConnectorOutput` per worker.
                The list index is the worker index.

        Returns:
            A new :class:`KVConnectorOutput` containing only request IDs
            that have been reported as finished by **all** workers.
        """
        if not worker_outputs:
            return KVConnectorOutput()

        for worker_idx, wo in enumerate(worker_outputs):
            if wo.finished_sending:
                for rid in wo.finished_sending:
                    self._seen_sending.setdefault(rid, set()).add(worker_idx)
            if wo.finished_recving:
                for rid in wo.finished_recving:
                    self._seen_recving.setdefault(rid, set()).add(worker_idx)
            if wo.failed_recving:
                for rid in wo.failed_recving:
                    self._seen_recv_failed.setdefault(rid, set()).add(worker_idx)
            if wo.finished_saving:
                for rid in wo.finished_saving:
                    self._seen_saving.setdefault(rid, set()).add(worker_idx)

        done_sending = {
            rid
            for rid, workers in self._seen_sending.items()
            if len(workers) >= self._world_size
        }
        failed_recving = set()
        recv_ids = set(self._seen_recving) | set(self._seen_recv_failed)
        for rid in recv_ids:
            done_workers = self._seen_recving.get(rid, set())
            failed_workers = self._seen_recv_failed.get(rid, set())
            if (
                failed_workers
                and len(done_workers | failed_workers) >= self._world_size
            ):
                failed_recving.add(rid)
        done_recving = {
            rid
            for rid, workers in self._seen_recving.items()
            if len(workers) >= self._world_size and rid not in failed_recving
        }
        done_saving = {
            rid
            for rid, workers in self._seen_saving.items()
            if len(workers) >= self._world_size
        }

        for rid in done_sending:
            del self._seen_sending[rid]
        for rid in done_recving:
            del self._seen_recving[rid]
            self._seen_recv_failed.pop(rid, None)
        for rid in failed_recving:
            self._seen_recving.pop(rid, None)
            self._seen_recv_failed.pop(rid, None)
        for rid in done_saving:
            del self._seen_saving[rid]

        return KVConnectorOutput(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
            finished_saving=done_saving,
        )

    def reset(self) -> None:
        """Clear all internal tracking state."""
        self._seen_sending.clear()
        self._seen_recving.clear()
        self._seen_recv_failed.clear()
        self._seen_saving.clear()

    @property
    def pending_count(self) -> tuple[int, int]:
        """Return ``(num_pending_sending, num_pending_recving)``."""
        return (
            len(self._seen_sending),
            len(set(self._seen_recving) | set(self._seen_recv_failed))
            + len(self._seen_saving),
        )
