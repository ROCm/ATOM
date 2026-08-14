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
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from typing import Generic, TypeVar

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    ConnectorCompletionKey,
    KVConnectorOutput,
    LoadCompletionId,
    LoadOperationId,
    ReqId,
    SaveCompletionId,
    SaveOperationId,
    SendCompletionId,
    SendOperationId,
)

logger = logging.getLogger("atom")

__all__ = ["KVOutputAggregator"]

_KeyT = TypeVar("_KeyT", bound=Hashable)


class _TPCompletionGroup(Generic[_KeyT]):
    """Collect one completion channel with optional bounded duplicate memory."""

    def __init__(
        self,
        world_size: int,
        tombstone_limit: int,
        should_tombstone: Callable[[_KeyT], bool] | None = None,
    ) -> None:
        self._world_size = world_size
        self._tombstone_limit = tombstone_limit
        self._should_tombstone = should_tombstone
        # key -> worker index -> success. A failure remains failure if one
        # worker contradicts or replays its report.
        self._reports: dict[_KeyT, dict[int, bool]] = {}
        self._tombstone_order: deque[_KeyT] = deque()
        self._tombstones: set[_KeyT] = set()

    def report(self, worker_idx: int, key: _KeyT, *, succeeded: bool) -> None:
        if key in self._tombstones:
            return
        reports = self._reports.setdefault(key, {})
        reports[worker_idx] = reports.get(worker_idx, True) and succeeded

    def report_many(
        self,
        worker_idx: int,
        keys: Iterable[_KeyT],
        *,
        succeeded: bool,
    ) -> None:
        for key in keys:
            self.report(worker_idx, key, succeeded=succeeded)

    def drain(self) -> tuple[set[_KeyT], set[_KeyT]]:
        succeeded: set[_KeyT] = set()
        failed: set[_KeyT] = set()
        for key, reports in list(self._reports.items()):
            if len(reports) < self._world_size:
                continue
            (succeeded if all(reports.values()) else failed).add(key)
            del self._reports[key]
            if self._should_tombstone is not None and self._should_tombstone(key):
                self._remember_tombstone(key)
        return succeeded, failed

    def _remember_tombstone(self, key: _KeyT) -> None:
        if key in self._tombstones:
            return
        self._tombstones.add(key)
        self._tombstone_order.append(key)
        while len(self._tombstone_order) > self._tombstone_limit:
            self._tombstones.discard(self._tombstone_order.popleft())

    def reset(self) -> None:
        self._reports.clear()
        self._tombstone_order.clear()
        self._tombstones.clear()

    @property
    def pending_count(self) -> int:
        return len(self._reports)

    @property
    def tombstone_count(self) -> int:
        return len(self._tombstones)


class KVOutputAggregator:
    """Aggregates :class:`KVConnectorOutput` from all TP workers.

    Tracks which unique worker indices have reported each request or exact save
    generation as finished. A transfer is globally complete only when all
    ``world_size`` workers report the same identity. Duplicate reports from
    one worker do not increase quorum, and different operation generations
    cannot complete one another.

    Args:
        world_size: Number of TP workers to aggregate over.

    Example::

        aggregator = KVOutputAggregator(world_size=8)
        per_worker_outputs = [worker.get_kv_output() for worker in workers]
        result = aggregator.aggregate(per_worker_outputs)
        # result.finished_recving contains only IDs done on ALL 8 workers
    """

    def __init__(
        self,
        world_size: int = 8,
        terminal_tombstone_limit: int = 4096,
    ) -> None:
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if terminal_tombstone_limit <= 0:
            raise ValueError("terminal_tombstone_limit must be positive")
        self._world_size = world_size
        self._sending = _TPCompletionGroup[SendCompletionId](
            world_size,
            terminal_tombstone_limit,
            lambda key: isinstance(key, SendOperationId),
        )
        self._receiving = _TPCompletionGroup[ReqId](
            world_size,
            terminal_tombstone_limit,
        )
        self._saving = _TPCompletionGroup[SaveCompletionId](
            world_size,
            terminal_tombstone_limit,
            lambda key: isinstance(key, SaveOperationId),
        )
        self._loading = _TPCompletionGroup[LoadCompletionId](
            world_size,
            terminal_tombstone_limit,
            lambda key: isinstance(key, LoadOperationId),
        )
        self._connector_completions = _TPCompletionGroup[ConnectorCompletionKey](
            world_size,
            terminal_tombstone_limit,
            lambda _key: True,
        )

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

        for worker_idx, output in enumerate(worker_outputs):
            self._sending.report_many(
                worker_idx,
                output.finished_sending,
                succeeded=True,
            )
            self._receiving.report_many(
                worker_idx,
                output.finished_recving,
                succeeded=True,
            )
            self._receiving.report_many(
                worker_idx,
                output.failed_recving,
                succeeded=False,
            )
            self._saving.report_many(
                worker_idx,
                output.finished_saving,
                succeeded=True,
            )
            self._loading.report_many(
                worker_idx,
                output.finished_loading,
                succeeded=True,
            )
            self._loading.report_many(
                worker_idx,
                output.failed_loading,
                succeeded=False,
            )
            for completion in output.connector_completions:
                self._connector_completions.report(
                    worker_idx,
                    completion.key,
                    succeeded=completion.succeeded,
                )

        done_sending, _ = self._sending.drain()
        done_recving, failed_recving = self._receiving.drain()
        done_saving, _ = self._saving.drain()
        done_loading, failed_loading = self._loading.drain()
        done_connector_keys, failed_connector_keys = self._connector_completions.drain()
        connector_completions = {
            ConnectorCompletion(
                channel=key[0],
                operation_id=key[1],
                succeeded=succeeded,
            )
            for keys, succeeded in (
                (done_connector_keys, True),
                (failed_connector_keys, False),
            )
            for key in keys
        }

        return KVConnectorOutput(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
            finished_saving=done_saving,
            finished_loading=done_loading,
            failed_loading=failed_loading,
            connector_completions=connector_completions,
            # A single rank with a live fence is enough to require another
            # poll.  Ranks that quarantined an unobservable operation report
            # False, so a permanently pinned scheduler lease cannot spin the
            # engine forever.
            pending_work=any(output.pending_work for output in worker_outputs),
        )

    def reset(self) -> None:
        """Clear all internal tracking state."""
        self._sending.reset()
        self._receiving.reset()
        self._saving.reset()
        self._loading.reset()
        self._connector_completions.reset()

    @property
    def terminal_tombstone_count(self) -> tuple[int, int]:
        return self._saving.tombstone_count, self._connector_completions.tombstone_count

    @property
    def terminal_load_tombstone_count(self) -> int:
        return self._loading.tombstone_count

    @property
    def terminal_send_tombstone_count(self) -> int:
        return self._sending.tombstone_count

    @property
    def pending_count(self) -> tuple[int, int]:
        """Return ``(num_pending_sending, num_pending_other_transfers)``."""
        return (
            self._sending.pending_count,
            self._receiving.pending_count
            + self._saving.pending_count
            + self._loading.pending_count
            + self._connector_completions.pending_count,
        )
