# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offload prefetch: Phase 2's "admit and park" half, run on every tick.

Phase 2 of the scheduler admits only on the prefill delayer's FIRE ticks and
only at the head of `waiting`: a request that needs a tier load spends one FIRE
parking (that is when its load is issued) and prefills at a later one, a whole
delayer cycle on, although the load takes milliseconds. `OffloadPrefetch.run`
runs the same admission (`Scheduler._admit_after_probe`) on any tick (HOLD or
FIRE), wherever the request waits, for a request whose tier lookup has answered
and that loads from the tier. Nothing else moves earlier:

- the request keeps its place in `waiting`, and Phase 2 reaches it in arrival
  order: a load in flight does not push it to the tail, and a finished
  prefetch moves up only when Phase 2 is stuck on KV blocks or parked-load
  slots (`Scheduler._blocked_head`), which prefetches hold and
  release only by running;
- it is not prefillable while its load is in flight, and once loaded it
  prefills only when a FIRE admits it;
- no collective runs and no step type changes, so DP lockstep holds.

An admission the connector turns into a local prefill after `allocate` keeps
its blocks and waits in place as a resume (`Scheduler._hold_for_local_prefill`).
Bounded by KV pressure: no prefetch once the pool would pass
`Config.offload_prefetch_max_kv_usage`. Anything else the park path cannot
take (an unaligned HBM hit, a multimodal prompt) is left to Phase 2.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING, Any

from atom.model_engine.scheduler import Admit
from atom.model_engine.sequence import SequenceStatus

if TYPE_CHECKING:
    from atom.model_engine.scheduler import Scheduler

logger = logging.getLogger("atom")


class OffloadPrefetch:
    """The scheduler's offload prefetch: its state and the per-tick pass."""

    # How far into `waiting` a pass looks, and how many requests it may park
    # in one tick (each runs an HBM probe and an allocation).
    SCAN = 64
    PER_TICK = 8
    # Ticks before a request skipped for a transient reason is looked at again.
    RETRY_TICKS = 8
    # How often (in ticks) to forget requests that left `waiting`.
    PRUNE_TICKS = 256

    def __init__(
        self, sched: Scheduler, max_kv_usage: float, min_load_tokens: int
    ) -> None:
        self.sched = sched
        self.max_kv_usage = max_kv_usage
        # The shortest tier hit past the HBM prefix the connector loads
        # (`OffloadTierProfile.min_load_tokens`): a shorter one would be
        # admitted only to prefill locally.
        self.min_load_tokens = min_load_tokens
        # seq id -> seq holding prefetched blocks, until it is scheduled for
        # prefill, aborted or otherwise gone.
        self.entries: dict[Any, Any] = {}
        # seq id -> the tick before which it is not looked at again (inf: a
        # request the pass has nothing to do for).
        self.skip_until: dict[Any, float] = {}

    # -- what the scheduler asks ----------------------------------------------

    def holds(self, seq) -> bool:
        return self.entries.get(seq.id) is seq

    def note_gone(self, seq_id: Any) -> None:
        """`seq_id` was scheduled for prefill, aborted, or holds nothing now."""
        self.entries.pop(seq_id, None)

    # -- the per-tick pass ----------------------------------------------------

    def run(self) -> int:
        """Admit-and-park waiting requests whose tier lookup answered and that
        load a prefix from the tier. Returns how many it admitted (parked, or
        held for a local prefill), bounded by `PER_TICK`."""
        s = self.sched
        tick = s._schedule_tick
        if tick % self.PRUNE_TICKS == 0:
            self._prune({seq.id for seq in s.waiting})
        if not s.waiting:
            return 0
        bm = s.block_manager
        chunk = int(getattr(s.kv_connector, "chunk_size", 0) or 0)
        hbs = bm.hash_block_size
        replays = bool(getattr(bm, "state_replay_tokens", 0))
        admitted = 0
        for seq in list(itertools.islice(s.waiting, self.SCAN)):
            if admitted >= self.PER_TICK or s._parked_slots_full():
                break
            if not self._candidate(seq, tick):
                continue
            probe = s._probe_hbm_prefix(seq)
            if probe.num_cached_blocks < 0:
                # The pool cannot take it now; neither could Phase 2.
                self.skip_until[seq.id] = tick + self.RETRY_TICKS
                break
            # Under a bounded replay the hit `can_allocate` offers moves with
            # the tier answer (`_tier_hit`): checked again once it is taken.
            rewalks = replays and seq.has_per_req_cache
            cached = probe.num_cached_blocks * hbs
            if chunk and cached % chunk and not rewalks:
                self._unaligned(seq)
                continue
            if s._should_wait_for_producer(seq, cached, True, record=False):
                # Phase 2 waits for the in-flight producer instead.
                self.skip_until[seq.id] = tick + self.RETRY_TICKS
                continue
            if s._lookup_detach_enabled(seq) and s._can_detach_tier_lookup(seq, probe):
                self.skip_until[seq.id] = math.inf  # no load possible
                continue
            # The tier answer last: once taken it holds the lookup's read
            # locks until admission consumes it, so every outcome below that
            # does not admit gives them back (`release_taken_lookup`; the hit
            # stays remembered for Phase 2).
            probed_tier = s._tier_hit(seq)
            if not s.kv_connector.take_ready_lookup(seq):
                continue
            needs_remote_load = s._query_connector_prefill_match(seq, skip=False)
            # `can_allocate` reads the answer under a bounded replay: probe
            # again if it moved (as Phase 2).
            if rewalks and s._tier_hit(seq) != probed_tier:
                probe = s._probe_hbm_prefix(seq)
                if probe.num_cached_blocks < 0:
                    self._give_back(seq, tick + self.RETRY_TICKS)
                    break
            cached = probe.num_cached_blocks * hbs
            if chunk and cached % chunk:
                self._give_back(seq, math.inf)  # `_unaligned`
                continue
            # The connector loads only a rest of at least `min_load_tokens`
            # (`_decide_load_after_alloc`); a shorter one would only hold
            # blocks for a local prefill.
            loads = needs_remote_load and (
                seq.offload_joint.kv_prefix_tokens - cached >= self.min_load_tokens
            )
            if not (loads or seq.replay_kv_load):
                # Nothing to load: it prefills when Phase 2 admits it, from
                # the remembered answer.
                self._give_back(seq, math.inf)
                continue
            outcome = s._admit_after_probe(seq, probe, needs_remote_load, None)
            if outcome is Admit.REFUSED:
                self._give_back(seq, tick + self.RETRY_TICKS)
                break
            if outcome is Admit.REQUEUE:
                self.skip_until[seq.id] = math.inf
                continue
            if outcome is Admit.LOCAL:
                s._hold_for_local_prefill(seq)
            admitted += 1
            self.entries[seq.id] = seq
        if admitted:
            logger.debug("[OFFLOAD-PREFETCH] admitted %d request(s)", admitted)
        return admitted

    def _give_back(self, seq, until: float) -> None:
        """Not admitted after taking its tier answer: release the answer's read
        locks, and look again at `until`."""
        self.sched.kv_connector.release_taken_lookup(seq)
        self.skip_until[seq.id] = until

    def _unaligned(self, seq) -> None:
        """An HBM hit off the tier's chunk grid runs its hand-off prefill
        first (`_maybe_start_unaligned_handoff`): left to Phase 2."""
        self.skip_until[seq.id] = math.inf

    def _candidate(self, seq, tick: int) -> bool:
        """The checks that need only the request, before the pass probes the
        pool or touches the connector."""
        s = self.sched
        if seq.status != SequenceStatus.WAITING:
            return False
        if tick < self.skip_until.get(seq.id, 0):
            return False
        if seq.multimodal_data is not None or s._is_offload_prefill_resume(seq):
            self.skip_until[seq.id] = math.inf
            return False
        if s._unschedulable_reason(seq) is not None:
            return False
        admission_ready = getattr(s.kv_connector, "admission_ready", None)
        if callable(admission_ready) and not admission_ready(seq):
            return False
        bm = s.block_manager
        total = max(1, int(bm.kv.num_blocks))
        # What `_admit_after_probe` allocates: the prompt and the draft
        # lookahead past it.
        need = bm.num_pool_blocks(len(seq)) + bm.draft_lookahead_blocks(len(seq))
        if s._kv_usage() + need / total > self.max_kv_usage:
            self.skip_until[seq.id] = tick + self.RETRY_TICKS
            return False
        # A peek; `run` takes the answer (`take_ready_lookup`) only after its
        # other checks.
        return bool(s.kv_connector.lookup_ready(seq))

    def _prune(self, waiting_ids: set) -> None:
        self.skip_until = {k: v for k, v in self.skip_until.items() if k in waiting_ids}
        self.entries = {k: v for k, v in self.entries.items() if k in waiting_ids}


__all__ = ["OffloadPrefetch"]
