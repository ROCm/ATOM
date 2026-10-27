# SPDX-License-Identifier: MIT
"""Gate migrated requests on the scheduler's actual, recoverable prefix.

Existing offload admission reserves destination PAGE and STATE units, parks
until joint completion, and owns DMA lifetimes. A rejected migration follows
the existing abort cleanup; its target buffers are never freed on a timer.
"""

import time


class DPCacheAdmission:
    def _queue_cache_work(self, seq, remaining):
        if not getattr(seq, "dp_cache_admission_sent", False):
            return
        remaining = max(0, int(remaining))
        if getattr(seq, "dp_cache_last_work", None) == remaining:
            return
        seq.dp_cache_last_work = remaining
        seq.dp_cache_work_revision = getattr(seq, "dp_cache_work_revision", 0) + 1
        # Coalesce once per engine step, including preemption and re-admission
        # in that step. No synchronous lookup or per-token decode reporting.
        self.__dict__.setdefault("_dp_cache_work", {})[seq.id] = (
            seq.id,
            getattr(seq, "dp_cache_migration", 0),
            seq.dp_cache_work_revision,
            remaining,
        )

    def take_cache_work(self):
        pending = self.__dict__.pop("_dp_cache_work", None)
        return list(pending.values()) if pending else []

    def _preempt_cache_work(self, seq):
        if getattr(seq, "dp_cache_balance", False):
            seq.dp_cache_work_epoch = getattr(seq, "dp_cache_work_epoch", 0) + 1
            self._queue_cache_work(seq, seq.num_tokens)

    def _complete_cache_work(self, batch, seq_by_id):
        if batch is None:
            return
        epochs = getattr(batch, "dp_cache_work_epochs", {})
        for i, req_id in enumerate(batch.req_ids[: batch.total_seqs_num_prefill]):
            seq = seq_by_id.get(req_id)
            if seq is None or req_id not in epochs:
                continue
            # A forward issued before preemption cannot retire recompute work
            # reserved by the new execution. Offsets come from the completed
            # batch, never from speculative schedule-time advancement.
            if epochs[req_id] != getattr(seq, "dp_cache_work_epoch", 0):
                continue
            end = int(batch.num_cached_tokens[i]) + int(batch.num_scheduled_tokens[i])
            remaining = 0 if batch.is_final_chunk[i] else seq.num_tokens - end
            self._queue_cache_work(seq, remaining)

    def _track_cache_migration(self, seq):
        if not getattr(seq, "dp_cache_migration", 0):
            return
        pending = self.__dict__.setdefault("_dp_cache_preparing", {})
        seq.dp_cache_prepare_deadline = time.monotonic() + seq.dp_cache_prepare_timeout
        pending[seq.id] = seq

    def _reject_cache_migration(self, seq, reason):
        self.__dict__.get("_dp_cache_preparing", {}).pop(seq.id, None)
        seq.dp_cache_retry_reason = reason
        # A live target transfer remains pinned in deferred_free_blocks until
        # the connector proves it terminal. Retrying on a different rank does
        # not reuse those destination addresses.
        self._reject_aborted_waiting(seq)

    def _expire_cache_migrations(self):
        pending = self.__dict__.get("_dp_cache_preparing")
        if not pending:
            return
        now = time.monotonic()
        for seq in list(pending.values()):
            expired = now >= seq.dp_cache_prepare_deadline
            if expired and seq in self.waiting:
                self.waiting.remove(seq)
                self._reject_cache_migration(seq, "preparation_timeout")
            elif seq in self.running:
                boundary = getattr(seq, "dp_cache_alignment_boundary", 0)
                # A completed alignment must have parked before this check.
                # If its tier lease disappeared, no further local prefill is
                # allowed to turn a failed migration into a full recompute.
                handoff = self._cache_prefill_handoff(seq)
                if expired or not handoff or seq.num_cached_tokens >= boundary:
                    pending.pop(seq.id, None)
                    seq.dp_cache_retry_reason = (
                        "preparation_timeout"
                        if expired
                        else "alignment_load_unavailable"
                    )
                    # Running buffers may still belong to the previous forward.
                    # Normal abort postprocess owns their eventual release.
                    self.abort_request(seq.id)

    def _cache_prefill_handoff(self, seq):
        callback = getattr(
            getattr(self, "kv_connector", None), "pending_prefill_handoff", None
        )
        return callback(seq) if callback else None

    def _cache_alignment_fits(self, seq, start, chunk):
        boundary = getattr(seq, "dp_cache_alignment_boundary", 0)
        return (
            not boundary
            or getattr(seq, "dp_cache_admission_sent", False)
            or start < start + chunk <= boundary
        )

    def _admit_cache_placement(self, seq):
        if not getattr(seq, "dp_cache_balance", False):
            return True
        if getattr(seq, "dp_cache_admission_sent", False):
            cached = min(
                seq.num_cached_tokens,
                self.block_manager.replay_start(seq, seq.num_cached_tokens, hit=True),
            )
            self._queue_cache_work(seq, seq.num_tokens - cached)
            return True
        cached = seq.num_cached_tokens
        # Bound by the actual forward start, including any model-required
        # replay. PAGE-only coverage must not masquerade as a restored STATE.
        cached = min(cached, self.block_manager.replay_start(seq, cached, hit=True))
        if cached < getattr(seq, "dp_cache_minimum_hit", 0):
            handoff = self._cache_prefill_handoff(seq)
            if (
                getattr(seq, "dp_cache_migration", 0)
                and handoff is not None
                and seq.num_cached_tokens < handoff[0] < seq.dp_cache_minimum_hit
                and handoff[1] >= seq.dp_cache_minimum_hit
                and getattr(self, "dp_cache_admission_callback", None) is not None
            ):
                # Only compute to the next transport boundary. Owner remains
                # unchanged until joint PAGE/STATE restore passes this gate.
                seq.dp_cache_alignment_boundary = handoff[0]
                return True
            # Preserve numeric coverage before abort cleanup resets the local
            # buffers. This is negative routing evidence, never a live lease.
            seq.dp_cache_recoverable_hit = max(0, int(cached))
            self._reject_cache_migration(seq, "recoverable_prefix_too_short")
            return False
        callback = getattr(self, "dp_cache_admission_callback", None)
        if callback is None:
            if getattr(seq, "dp_cache_migration", 0):
                self._reject_cache_migration(seq, "admission_callback_unavailable")
                return False
            return True
        callback((seq.id, getattr(seq, "dp_cache_migration", 0), cached))
        seq.dp_cache_admission_sent = True
        self.__dict__.get("_dp_cache_preparing", {}).pop(seq.id, None)
        return True
