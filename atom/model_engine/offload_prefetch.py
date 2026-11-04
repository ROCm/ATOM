# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offload prefetch: start a waiting request's tier load before its turn.

Without it the only admission is the scheduler's Phase 2, which runs on the
prefill delayer's FIRE ticks and only at the head of `waiting`: a request that
needs a tier load spends one FIRE parking (that is when its load is issued) and
prefills at a later one, a whole delayer cycle on, although the load takes
milliseconds. `OffloadPrefetch.run` allocates the blocks and submits the load on
any tick (HOLD or FIRE), wherever the request waits. Nothing else moves earlier:

- the request keeps its place in `waiting`, and Phase 2 reaches it in arrival
  order (a load in flight does not push it to the tail, and a finished prefetch
  is promoted only past a head blocked on KV blocks or parked-load slots --
  `Scheduler._head_blocked_on_resources`);
- it is not prefillable while its load is in flight, and once loaded it
  prefills only when a FIRE admits it;
- no collective runs and no step type changes, so DP lockstep holds.

Bounded by KV pressure: no prefetch allocation once the pool would pass
`KV_WATERMARK`. Anything the park path cannot take (an unaligned HBM hit, a
multimodal prompt, a load the connector refuses after `allocate`) is left to
Phase 2, as before.
"""

from __future__ import annotations

import enum
import itertools
import logging
from typing import TYPE_CHECKING, Any

from atom.model_engine.sequence import SequenceStatus
from atom.utils import envs

if TYPE_CHECKING:
    from atom.model_engine.scheduler import Scheduler

logger = logging.getLogger("atom")


class Admit(enum.Enum):
    """Outcome of `Scheduler._admit_after_probe`."""

    REFUSED = "refused"
    REQUEUE = "requeue"
    PARKED = "parked"
    LOCAL = "local"


class OffloadPrefetch:
    """The scheduler's offload prefetch: its state and the per-tick pass."""

    # How deep into `waiting` a pass looks, and how many requests it may admit
    # in one tick (each runs an HBM probe and an allocation).
    SCAN = 64
    PER_TICK = 8
    # Ticks before a request skipped for KV pressure is looked at again.
    RETRY_TICKS = 8
    # No prefetch allocation once the KV pool would be fuller than this.
    KV_WATERMARK = 0.8
    # Forget skip memory of requests that left `waiting` this often (in ticks).
    PRUNE_TICKS = 256

    def __init__(self, sched: Scheduler) -> None:
        self.sched = sched
        # seq id -> seq holding prefetched blocks, until it is scheduled for
        # prefill, aborted or otherwise gone.
        self.entries: dict[Any, Any] = {}
        # Requests never to try again (no load, unaligned, taken back, ...).
        self.settled: set[Any] = set()
        # Requests to retry only at this tick (a KV-pressure skip).
        self.retry_at: dict[Any, int] = {}
        self.tick = 0

    # -- what the scheduler asks ----------------------------------------------

    def holds(self, seq) -> bool:
        return seq.id in self.entries

    def note_gone(self, seq_id: Any) -> None:
        """`seq_id` was scheduled for prefill, aborted or otherwise left."""
        self.entries.pop(seq_id, None)

    # -- the per-tick pass ----------------------------------------------------

    def run(self) -> int:
        """Admit-and-park waiting requests whose tier lookup answered and that
        must load a prefix from the tier. Returns how many parked."""
        s = self.sched
        if s.kv_connector is None or not s._connector_flag("is_offload"):
            return 0
        bm = s.block_manager
        total_blocks = int(bm.kv.num_blocks)
        self.tick += 1
        if self.entries:
            self._sweep()
        if self.tick % self.PRUNE_TICKS == 0:
            self._prune({seq.id for seq in s.waiting})
        if not s.waiting:
            return 0

        chunk = int(getattr(s.kv_connector, "chunk_size", 0) or 0)
        hbs = bm.hash_block_size
        parked = 0
        for seq in list(itertools.islice(s.waiting, 0, self.SCAN)):
            if parked >= self.PER_TICK:
                break
            if not self._candidate(seq):
                continue
            if len(s.running) + s._num_parked_remote_kv >= s.max_num_seqs:
                break
            hbm_probe = None
            if s._lookup_detach_enabled(seq):
                hbm_probe = s._probe_hbm_prefix(seq)
                if hbm_probe.num_cached_blocks >= 0 and s._can_detach_tier_lookup(
                    seq, hbm_probe
                ):
                    self.settled.add(seq.id)
                    continue  # no load possible: it prefills when a FIRE admits it
            # Never block the scheduler thread -- and every DP rank with it -- on
            # a synchronous lookup: only an answered async lookup qualifies,
            # checked and consumed under the lookup client's lock so a driver
            # thread cannot expire the answer in between. Asked without
            # `lookup_pending`, which would start the request's lookup deferral
            # clock long before Phase 2 reaches it.
            with s.kv_connector.lookup_guard():
                if not s.kv_connector.lookup_ready(seq):
                    continue
                needs_remote_load = s._query_connector_prefill_match(seq, skip=False)
            # One HBM walk, after the tier answer: `can_allocate` reads that
            # answer under a bounded replay, so only then re-probe (as Phase 2).
            if hbm_probe is None or (
                getattr(bm, "state_replay_tokens", 0) and seq.has_per_req_cache
            ):
                hbm_probe = s._probe_hbm_prefix(seq)
            if hbm_probe.num_cached_blocks < 0:
                # The pool cannot take it now; neither could Phase 2.
                self.retry_at[seq.id] = self.tick + self.RETRY_TICKS
                break
            cached = hbm_probe.num_cached_blocks * hbs
            if s._wait_for_inflight_prefix(seq, cached, record=False):
                continue  # Phase 2 waits for the in-flight producer instead
            if not (needs_remote_load or seq.replay_kv_load) or (
                chunk and cached % chunk
            ):
                # No tier load, or an unaligned HBM hit whose hand-off prefill
                # runs first (`_maybe_start_unaligned_handoff`).
                self.settled.add(seq.id)
                continue
            # KV pressure. Counted only for a request that would load.
            need = bm.num_pool_blocks(len(seq))
            if (bm.kv.num_used + need) / max(1, total_blocks) > self.KV_WATERMARK:
                self.retry_at[seq.id] = self.tick + self.RETRY_TICKS
                continue
            if self._admit(seq, hbm_probe, needs_remote_load):
                parked += 1
            else:
                break  # the pool refused the allocation
        if parked:
            logger.debug("[OFFLOAD-PREFETCH] parked %d tier load(s)", parked)
        return parked

    def _candidate(self, seq) -> bool:
        """Cheap filters before the pass touches the connector. A request that
        fails one for good is remembered (`settled`)."""
        s = self.sched
        if seq.status != SequenceStatus.WAITING or seq.id in self.settled:
            return False
        retry = self.retry_at.get(seq.id)
        if retry is not None and self.tick < retry:
            return False
        if (
            seq.multimodal_data is not None
            # No tier hit can be worth loading (`_load_ruled_out`).
            or seq.num_tokens < envs.OFFLOAD_MIN_LOAD_TOKENS
            or s._is_offload_prefill_resume(seq)
        ):
            self.settled.add(seq.id)
            return False
        if s._unschedulable_reason(seq) is not None:
            return False
        admission_ready = getattr(s.kv_connector, "admission_ready", None)
        if callable(admission_ready) and not admission_ready(seq):
            return False
        # Cheap pre-filter; `run` checks again under the lookup guard.
        return bool(s.kv_connector.lookup_ready(seq))

    def _admit(self, seq, hbm_probe, needs_remote_load: bool) -> bool:
        """Phase 2's admission (`Scheduler._admit_after_probe`), kept only if
        the request parks for a load, in place in `waiting`.

        Returns False when the pool refused the request (nothing is held);
        True otherwise, whether it parked or was taken back. A request that
        would prefill locally -- a refused load, a joint boundary (Phase 2
        retreats or disowns it at its turn) -- is handed back unadmitted.
        """
        s = self.sched
        prev_hit = seq.prefix_cache_hit_tokens
        outcome = s._admit_after_probe(seq, hbm_probe, needs_remote_load, requeue=None)
        if outcome is Admit.REFUSED:
            return False
        if outcome is Admit.REQUEUE:
            self._take_back(seq, prev_hit, release=False)
            return True
        oj = seq.offload_joint
        if outcome is Admit.LOCAL:
            if oj.boundary_tokens or oj.load_hash == -1:
                self._take_back(seq, prev_hit)
                return True
            # A state-only load parks like a KV load (as in Phase 2).
            s._park_for_remote_load(seq, None)
            s._inflight_prefix_wait.pop(seq.id, None)
        self.entries[seq.id] = seq
        self.retry_at.pop(seq.id, None)
        return True

    def _take_back(self, seq, prev_hit: int, *, release: bool = True) -> None:
        """Undo the connector notification (and, unless already released,
        `allocate`) of an admission that did not park; the request is back in
        `waiting` exactly as unadmitted, and is not tried again."""
        s = self.sched
        s.kv_connector.cancel_preadmission(seq)
        if release:
            if seq.offload_joint.load_hash != -1:
                s.block_manager.cancel_state_load(seq)
            s.block_manager.deallocate(seq)
        seq.prefix_cache_hit_tokens = prev_hit
        self.settled.add(seq.id)

    def _sweep(self) -> None:
        """Drop entries whose request is gone from `waiting`, and reclaim an
        aborted one at once: Phase 2 only pops it on a FIRE tick, and until
        then it would hold its blocks."""
        s = self.sched
        for seq_id, seq in list(self.entries.items()):
            if seq.status == SequenceStatus.ABORTED:
                try:
                    s.waiting.remove(seq)
                except ValueError:
                    self.note_gone(seq_id)
                else:
                    s._reject_aborted_waiting(seq)
            elif seq.status not in (
                SequenceStatus.WAITING,
                SequenceStatus.WAITING_FOR_REMOTE_KVS,
            ):
                self.note_gone(seq_id)

    def _prune(self, waiting_ids: set) -> None:
        self.settled &= waiting_ids
        self.retry_at = {k: v for k, v in self.retry_at.items() if k in waiting_ids}


__all__ = ["Admit", "OffloadPrefetch"]
