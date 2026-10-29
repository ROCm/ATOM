# SPDX-License-Identifier: MIT
"""Gate migrated requests on the scheduler's actual, recoverable prefix.

The scheduler and connector retain ownership of queues, PAGE/STATE reservations
and DMA lifetimes. This component owns preparation tracking and work reports.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from atom.model_engine.dp_cache.state import CacheAdmissionEvent, CacheWorkEvent

if TYPE_CHECKING:
    from atom.model_engine.block_manager import BlockManager
    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import Sequence


class AdmissionHost(Protocol):
    """Scheduler resources and cleanup hooks required by cache admission.

    Queues are read from the host because the scheduler may replace them.
    The rejection hook is the scheduler's existing DMA-safe abort cleanup.
    """

    waiting: deque[Sequence]
    running: deque[Sequence]
    block_manager: BlockManager
    kv_connector: object

    def abort_request(self, req_id: int) -> bool: ...

    def _reject_aborted_waiting(self, seq: Sequence) -> None: ...


class DPCacheAdmission:
    def __init__(self, host: AdmissionHost):
        self._host = host
        self.on_admitted: Callable[[CacheAdmissionEvent], None] | None = None
        self._work: dict[object, CacheWorkEvent] = {}
        self._preparing: dict[object, Sequence] = {}

    def _queue_work(self, seq: Sequence, remaining: int, prefix: int) -> None:
        state = seq.dp_cache
        if state is None or not state.admission_sent:
            return
        remaining = max(0, int(remaining))
        prefix = max(0, int(prefix))
        if state.last_work == (remaining, prefix):
            return
        state.last_work = (remaining, prefix)
        state.work_revision += 1
        # Coalesce once per engine step, including preemption and re-admission.
        # No synchronous lookup or per-token decode reporting.
        self._work[seq.id] = (
            seq.id,
            state.migration_id,
            state.work_revision,
            remaining,
            prefix,
        )

    def take_work(self) -> list[CacheWorkEvent]:
        pending = list(self._work.values())
        self._work.clear()
        return pending

    def preempt(self, seq: Sequence) -> None:
        if seq.dp_cache is not None:
            seq.dp_cache.work_epoch += 1
            self._queue_work(seq, seq.num_tokens, 0)

    def complete_work(
        self, batch: ScheduledBatch | None, seq_by_id: dict[object, Sequence]
    ) -> None:
        if batch is None:
            return
        epochs = batch.dp_cache_work_epochs
        for i, req_id in enumerate(batch.req_ids[: batch.total_seqs_num_prefill]):
            seq = seq_by_id.get(req_id)
            if seq is None or req_id not in epochs:
                continue
            state = seq.dp_cache
            # A pre-preemption forward cannot retire the new execution's work.
            # Use completed-batch offsets, not schedule-time advancement.
            if state is None or epochs[req_id] != state.work_epoch:
                continue
            end = int(batch.num_cached_tokens[i]) + int(batch.num_scheduled_tokens[i])
            remaining = 0 if batch.is_final_chunk[i] else seq.num_tokens - end
            self._queue_work(seq, remaining, end)

    def track(self, seq: Sequence) -> None:
        state = seq.dp_cache
        if (
            state is None
            or not state.migration_id
            or state.preparation_role == "source"
        ):
            return
        state.prepare_deadline = time.monotonic() + state.prepare_timeout
        self._preparing[seq.id] = seq

    def forget(self, seq: Sequence) -> None:
        self._preparing.pop(seq.id, None)

    def reject_waiting(self, seq: Sequence, reason: str) -> None:
        self.forget(seq)
        assert seq.dp_cache is not None
        seq.dp_cache.retry_reason = reason
        # Live transfers remain pinned until the connector proves them terminal.
        self._host._reject_aborted_waiting(seq)

    def reject_running(self, seq: Sequence, reason: str) -> None:
        self.forget(seq)
        assert seq.dp_cache is not None
        seq.dp_cache.retry_reason = reason
        # Running buffers may belong to the previous forward; postprocess frees.
        self._host.abort_request(seq.id)

    def expire(self) -> None:
        if not self._preparing:
            return
        queued = {id(seq) for seq in self._host.waiting}
        queued.update(id(seq) for seq in self._host.running)
        now = time.monotonic()
        for seq in list(self._preparing.values()):
            if id(seq) not in queued:
                # Terminal scheduler paths own resource reclamation. Drop only
                # our reference; an in-flight transfer may still own its lease.
                self.forget(seq)
                continue
            state = seq.dp_cache
            assert state is not None
            expired = now >= state.prepare_deadline
            if expired and seq in self._host.waiting:
                self._host.waiting.remove(seq)
                self.reject_waiting(seq, "preparation_timeout")
            elif seq in self._host.running:
                # A completed alignment must have parked before this check.
                # Losing its tier lease must not turn migration into recompute.
                handoff = self._prefill_handoff(seq)
                if (
                    expired
                    or not handoff
                    or seq.num_cached_tokens >= state.alignment_boundary
                ):
                    self.reject_running(
                        seq,
                        (
                            "preparation_timeout"
                            if expired
                            else "alignment_load_unavailable"
                        ),
                    )

    def _prefill_handoff(self, seq: Sequence):
        callback = getattr(self._host.kv_connector, "pending_prefill_handoff", None)
        return callback(seq) if callback else None

    def alignment_fits(self, seq: Sequence, start: int, chunk: int) -> bool:
        state = seq.dp_cache
        return (
            state is None
            or not state.alignment_boundary
            or state.admission_sent
            or start < start + chunk <= state.alignment_boundary
        )

    def reject_short_lookup(self, seq: Sequence, upper_bound: int) -> bool:
        """Reject a known-short migration before allocating or loading its KV.

        The bound includes HBM, the tier's STATE/PAGE reach and any alignment
        handoff. It can rule out admission, but cannot establish residency:
        feasible requests still pass the existing post-load admission gate.
        """
        state = seq.dp_cache
        if state is None or not state.migration_id or state.admission_sent:
            return False
        callback = getattr(self._host.kv_connector, "lookup_recovery_upper_bound", None)
        tier_bound = callback(seq) if callable(callback) else None
        if tier_bound is None:
            return False
        upper_bound = max(upper_bound, tier_bound)
        handoff = self._prefill_handoff(seq)
        if handoff is not None:
            upper_bound = max(upper_bound, handoff[1])
        if upper_bound >= state.minimum_hit:
            return False
        state.recoverable_hit = max(0, int(upper_bound))
        self.reject_waiting(seq, "lookup_prefix_too_short")
        return True

    def admit(self, seq: Sequence) -> bool:
        state = seq.dp_cache
        if state is None:
            return True
        # Bound by the actual forward start, including model-required replay.
        # PAGE-only coverage must not masquerade as restored STATE.
        cached = min(
            seq.num_cached_tokens,
            self._host.block_manager.replay_start(seq, seq.num_cached_tokens, hit=True),
        )
        if state.admission_sent:
            self._queue_work(seq, seq.num_tokens - cached, cached)
            return True
        if cached < state.minimum_hit:
            handoff = self._prefill_handoff(seq)
            if (
                state.migration_id
                and handoff is not None
                and seq.num_cached_tokens < handoff[0] < state.minimum_hit
                and handoff[1] >= state.minimum_hit
                and self.on_admitted is not None
            ):
                # Compute only to the next transport boundary; ownership stays
                # unchanged until joint PAGE/STATE restore passes this gate.
                state.alignment_boundary = handoff[0]
                return True
            # Negative routing evidence, captured before cleanup; not a lease.
            state.recoverable_hit = max(0, int(cached))
            callback = getattr(
                self._host.kv_connector, "lookup_recovery_upper_bound", None
            )
            reason = (
                "lookup_unavailable"
                if callable(callback) and callback(seq) is None
                else "recoverable_prefix_too_short"
            )
            self.reject_waiting(seq, reason)
            return False
        if self.on_admitted is None:
            if state.migration_id:
                self.reject_waiting(seq, "admission_callback_unavailable")
                return False
            return True
        self.on_admitted((seq.id, state.migration_id, cached))
        state.admission_sent = True
        self.forget(seq)
        return True
