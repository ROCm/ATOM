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


class KVOutputAggregator:
    """Aggregates :class:`KVConnectorOutput` from all TP workers.

    Tracks which unique worker indices have reported each request or exact save
    generation as finished. A transfer is globally complete only when all
    ``world_size`` workers report the same identity — duplicate reports from
    one worker and cross-generation PAGE/SLOT reports are ignored.

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
        self._terminal_tombstone_limit = terminal_tombstone_limit
        self._seen_sending: dict[SendCompletionId, set[int]] = {}
        self._seen_recving: dict[ReqId, set[int]] = {}
        self._seen_recv_failed: dict[ReqId, set[int]] = {}
        self._seen_saving: dict[SaveCompletionId, set[int]] = {}
        self._seen_loading: dict[LoadCompletionId, set[int]] = {}
        self._seen_load_failed: dict[LoadCompletionId, set[int]] = {}
        self._seen_connector_completion: dict[ConnectorCompletionKey, set[int]] = {}
        self._seen_connector_completion_failed: dict[
            ConnectorCompletionKey, set[int]
        ] = {}
        self._terminal_saving_order: deque[SaveOperationId] = deque()
        self._terminal_saving: set[SaveOperationId] = set()
        self._terminal_load_order: deque[LoadOperationId] = deque()
        self._terminal_load: set[LoadOperationId] = set()
        self._terminal_sending_order: deque[SendOperationId] = deque()
        self._terminal_sending: set[SendOperationId] = set()
        self._terminal_connector_completion_order: deque[ConnectorCompletionKey] = (
            deque()
        )
        self._terminal_connector_completion: set[ConnectorCompletionKey] = set()

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
                    if (
                        isinstance(rid, SendOperationId)
                        and rid in self._terminal_sending
                    ):
                        continue
                    self._seen_sending.setdefault(rid, set()).add(worker_idx)
            if wo.finished_recving:
                for rid in wo.finished_recving:
                    self._seen_recving.setdefault(rid, set()).add(worker_idx)
            if wo.failed_recving:
                for rid in wo.failed_recving:
                    self._seen_recv_failed.setdefault(rid, set()).add(worker_idx)
            if wo.finished_saving:
                for rid in wo.finished_saving:
                    if (
                        isinstance(rid, SaveOperationId)
                        and rid in self._terminal_saving
                    ):
                        continue
                    self._seen_saving.setdefault(rid, set()).add(worker_idx)
            if wo.finished_loading:
                for rid in wo.finished_loading:
                    if isinstance(rid, LoadOperationId) and rid in self._terminal_load:
                        continue
                    self._seen_loading.setdefault(rid, set()).add(worker_idx)
            if wo.failed_loading:
                for rid in wo.failed_loading:
                    if isinstance(rid, LoadOperationId) and rid in self._terminal_load:
                        continue
                    self._seen_load_failed.setdefault(rid, set()).add(worker_idx)
            for completion in wo.connector_completions:
                key = completion.key
                if key in self._terminal_connector_completion:
                    continue
                seen = (
                    self._seen_connector_completion
                    if completion.succeeded
                    else self._seen_connector_completion_failed
                )
                seen.setdefault(key, set()).add(worker_idx)

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
        failed_loading = set()
        load_ids = set(self._seen_loading) | set(self._seen_load_failed)
        for rid in load_ids:
            done_workers = self._seen_loading.get(rid, set())
            failed_workers = self._seen_load_failed.get(rid, set())
            if (
                failed_workers
                and len(done_workers | failed_workers) >= self._world_size
            ):
                failed_loading.add(rid)
        done_loading = {
            rid
            for rid, workers in self._seen_loading.items()
            if len(workers) >= self._world_size and rid not in failed_loading
        }
        connector_completions: set[ConnectorCompletion] = set()
        connector_completion_keys = set(self._seen_connector_completion) | set(
            self._seen_connector_completion_failed
        )
        terminal_connector_completion_keys: set[ConnectorCompletionKey] = set()
        for key in connector_completion_keys:
            done_workers = self._seen_connector_completion.get(key, set())
            failed_workers = self._seen_connector_completion_failed.get(key, set())
            if len(done_workers | failed_workers) < self._world_size:
                continue
            terminal_connector_completion_keys.add(key)
            connector_completions.add(
                ConnectorCompletion(
                    channel=key[0],
                    operation_id=key[1],
                    succeeded=not bool(failed_workers),
                )
            )

        for rid in done_sending:
            del self._seen_sending[rid]
            if isinstance(rid, SendOperationId):
                self._remember_terminal(
                    rid,
                    self._terminal_sending_order,
                    self._terminal_sending,
                )
        for rid in done_recving:
            del self._seen_recving[rid]
            self._seen_recv_failed.pop(rid, None)
        for rid in failed_recving:
            self._seen_recving.pop(rid, None)
            self._seen_recv_failed.pop(rid, None)
        for rid in done_saving:
            del self._seen_saving[rid]
            if isinstance(rid, SaveOperationId):
                self._remember_terminal(
                    rid,
                    self._terminal_saving_order,
                    self._terminal_saving,
                )
        for rid in done_loading:
            del self._seen_loading[rid]
            self._seen_load_failed.pop(rid, None)
            if isinstance(rid, LoadOperationId):
                self._remember_terminal(
                    rid,
                    self._terminal_load_order,
                    self._terminal_load,
                )
        for rid in failed_loading:
            self._seen_loading.pop(rid, None)
            self._seen_load_failed.pop(rid, None)
            if isinstance(rid, LoadOperationId):
                self._remember_terminal(
                    rid,
                    self._terminal_load_order,
                    self._terminal_load,
                )
        for key in terminal_connector_completion_keys:
            self._seen_connector_completion.pop(key, None)
            self._seen_connector_completion_failed.pop(key, None)
            self._remember_terminal(
                key,
                self._terminal_connector_completion_order,
                self._terminal_connector_completion,
            )

        return KVConnectorOutput(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
            finished_saving=done_saving,
            finished_loading=done_loading,
            failed_loading=failed_loading,
            connector_completions=connector_completions,
        )

    def _remember_terminal(
        self,
        operation: (
            SaveOperationId | LoadOperationId | SendOperationId | ConnectorCompletionKey
        ),
        order: deque,
        tombstones: set,
    ) -> None:
        """Bound late-duplicate suppression by exact operation identity."""
        if operation in tombstones:
            return
        order.append(operation)
        tombstones.add(operation)
        while len(order) > self._terminal_tombstone_limit:
            tombstones.discard(order.popleft())

    def reset(self) -> None:
        """Clear all internal tracking state."""
        self._seen_sending.clear()
        self._seen_recving.clear()
        self._seen_recv_failed.clear()
        self._seen_saving.clear()
        self._seen_loading.clear()
        self._seen_load_failed.clear()
        self._seen_connector_completion.clear()
        self._seen_connector_completion_failed.clear()
        self._terminal_saving_order.clear()
        self._terminal_saving.clear()
        self._terminal_load_order.clear()
        self._terminal_load.clear()
        self._terminal_sending_order.clear()
        self._terminal_sending.clear()
        self._terminal_connector_completion_order.clear()
        self._terminal_connector_completion.clear()

    @property
    def terminal_tombstone_count(self) -> tuple[int, int]:
        return len(self._terminal_saving), len(self._terminal_connector_completion)

    @property
    def terminal_load_tombstone_count(self) -> int:
        return len(self._terminal_load)

    @property
    def terminal_send_tombstone_count(self) -> int:
        return len(self._terminal_sending)

    @property
    def pending_count(self) -> tuple[int, int]:
        """Return ``(num_pending_sending, num_pending_other_transfers)``."""
        return (
            len(self._seen_sending),
            len(set(self._seen_recving) | set(self._seen_recv_failed))
            + len(self._seen_saving)
            + len(set(self._seen_loading) | set(self._seen_load_failed))
            + len(
                set(self._seen_connector_completion)
                | set(self._seen_connector_completion_failed)
            ),
        )
