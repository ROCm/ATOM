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

from collections import deque
import logging

from atom.kv_transfer.disaggregation.types import (
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
        self._seen_sidecar_saving: dict[SaveCompletionId, set[int]] = {}
        self._seen_sidecar_save_failed: dict[SaveCompletionId, set[int]] = {}
        self._seen_checkpoint_staging: dict[int, set[int]] = {}
        self._seen_checkpoint_staging_aborted: dict[int, set[int]] = {}
        self._terminal_saving_order: deque[SaveOperationId] = deque()
        self._terminal_saving: set[SaveOperationId] = set()
        self._terminal_sidecar_order: deque[SaveOperationId] = deque()
        self._terminal_sidecar: set[SaveOperationId] = set()
        self._terminal_load_order: deque[LoadOperationId] = deque()
        self._terminal_load: set[LoadOperationId] = set()
        self._terminal_sending_order: deque[SendOperationId] = deque()
        self._terminal_sending: set[SendOperationId] = set()
        self._terminal_checkpoint_staging_order: deque[int] = deque()
        self._terminal_checkpoint_staging: set[int] = set()

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
            if wo.finished_sidecar_saving:
                for rid in wo.finished_sidecar_saving:
                    if (
                        isinstance(rid, SaveOperationId)
                        and rid in self._terminal_sidecar
                    ):
                        continue
                    self._seen_sidecar_saving.setdefault(rid, set()).add(worker_idx)
            if wo.failed_sidecar_saving:
                for rid in wo.failed_sidecar_saving:
                    if (
                        isinstance(rid, SaveOperationId)
                        and rid in self._terminal_sidecar
                    ):
                        continue
                    self._seen_sidecar_save_failed.setdefault(rid, set()).add(
                        worker_idx
                    )
            for copy_id in wo.finished_checkpoint_staging:
                if copy_id not in self._terminal_checkpoint_staging:
                    self._seen_checkpoint_staging.setdefault(copy_id, set()).add(
                        worker_idx
                    )
            for copy_id in wo.aborted_checkpoint_staging:
                if copy_id not in self._terminal_checkpoint_staging:
                    self._seen_checkpoint_staging_aborted.setdefault(
                        copy_id, set()
                    ).add(worker_idx)

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
        failed_sidecar_saving = set()
        sidecar_ids = set(self._seen_sidecar_saving) | set(
            self._seen_sidecar_save_failed
        )
        for rid in sidecar_ids:
            done_workers = self._seen_sidecar_saving.get(rid, set())
            failed_workers = self._seen_sidecar_save_failed.get(rid, set())
            if (
                failed_workers
                and len(done_workers | failed_workers) >= self._world_size
            ):
                failed_sidecar_saving.add(rid)
        done_sidecar_saving = {
            rid
            for rid, workers in self._seen_sidecar_saving.items()
            if len(workers) >= self._world_size and rid not in failed_sidecar_saving
        }
        aborted_checkpoint_staging = set()
        checkpoint_copy_ids = set(self._seen_checkpoint_staging) | set(
            self._seen_checkpoint_staging_aborted
        )
        for copy_id in checkpoint_copy_ids:
            done_workers = self._seen_checkpoint_staging.get(copy_id, set())
            aborted_workers = self._seen_checkpoint_staging_aborted.get(copy_id, set())
            if (
                aborted_workers
                and len(done_workers | aborted_workers) >= self._world_size
            ):
                aborted_checkpoint_staging.add(copy_id)
        done_checkpoint_staging = {
            copy_id
            for copy_id, workers in self._seen_checkpoint_staging.items()
            if len(workers) >= self._world_size
            and copy_id not in aborted_checkpoint_staging
        }

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
        for rid in done_sidecar_saving:
            self._seen_sidecar_saving.pop(rid, None)
            self._seen_sidecar_save_failed.pop(rid, None)
            if isinstance(rid, SaveOperationId):
                self._remember_terminal(
                    rid,
                    self._terminal_sidecar_order,
                    self._terminal_sidecar,
                )
        for rid in failed_sidecar_saving:
            self._seen_sidecar_saving.pop(rid, None)
            self._seen_sidecar_save_failed.pop(rid, None)
            if isinstance(rid, SaveOperationId):
                self._remember_terminal(
                    rid,
                    self._terminal_sidecar_order,
                    self._terminal_sidecar,
                )
        for copy_id in done_checkpoint_staging | aborted_checkpoint_staging:
            self._seen_checkpoint_staging.pop(copy_id, None)
            self._seen_checkpoint_staging_aborted.pop(copy_id, None)
            self._remember_terminal(
                copy_id,
                self._terminal_checkpoint_staging_order,
                self._terminal_checkpoint_staging,
            )

        return KVConnectorOutput(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
            finished_saving=done_saving,
            finished_loading=done_loading,
            failed_loading=failed_loading,
            finished_sidecar_saving=done_sidecar_saving,
            failed_sidecar_saving=failed_sidecar_saving,
            finished_checkpoint_staging=done_checkpoint_staging,
            aborted_checkpoint_staging=aborted_checkpoint_staging,
        )

    def _remember_terminal(
        self,
        operation: SaveOperationId | LoadOperationId | SendOperationId | int,
        order: deque,
        tombstones: set,
    ) -> None:
        """Bound late-duplicate suppression by exact save generation."""
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
        self._seen_sidecar_saving.clear()
        self._seen_sidecar_save_failed.clear()
        self._seen_checkpoint_staging.clear()
        self._seen_checkpoint_staging_aborted.clear()
        self._terminal_saving_order.clear()
        self._terminal_saving.clear()
        self._terminal_sidecar_order.clear()
        self._terminal_sidecar.clear()
        self._terminal_load_order.clear()
        self._terminal_load.clear()
        self._terminal_sending_order.clear()
        self._terminal_sending.clear()
        self._terminal_checkpoint_staging_order.clear()
        self._terminal_checkpoint_staging.clear()

    @property
    def terminal_tombstone_count(self) -> tuple[int, int]:
        return len(self._terminal_saving), len(self._terminal_sidecar)

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
            + len(set(self._seen_sidecar_saving) | set(self._seen_sidecar_save_failed))
            + len(
                set(self._seen_checkpoint_staging)
                | set(self._seen_checkpoint_staging_aborted)
            ),
        )
