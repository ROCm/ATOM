# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Offload load wait: tier-load completions seen before `schedule()`.

Without it a waiting request with a tier hit parks inside `schedule()`, its
load leaves with that step's batch and lands within milliseconds, but the
engine reads worker completions only through the query answered after a
forward: the request prefills one or two forwards later.

`OffloadLoadWait` is the engine step hook (`EngineCore.step_hooks`) that fixes
this; nothing of it lives on the engine or the scheduler. Before each
`schedule()` it

1. runs that tick's early tier load ahead of `schedule()` (`prestart_tier_loads`,
   which `schedule()` then skips: `EarlyLoadState.take_tick_pass`) and ships
   the loads it started at once;
2. when a load ended since the last answer -- a worker rang the KV progress
   doorbell (`progress_doorbell`), which sets the engine's KV progress event --
   fetches one completion answer requested after that ring, so a landed load
   is admitted by this `schedule()` (`ATOM_OFFLOAD_LOAD_EVENT_WAIT`, on);
3. only with `ATOM_OFFLOAD_LOAD_WAIT_HOLD` (off by default): holds the step for
   the loads it started while holding pays (`LoadWaitState.budget_s` over
   `load_wait_tradeoff`), woken by the doorbell, ending as soon as every
   started load landed. The bound is rank-local; its cost lands on every DP
   peer at the forward collective, so it stays off until it is group-aware or
   measured with the prefill delayer on and in 4P8D.

Rank-local throughout: no collective runs here and the step count is
unchanged, so DP lockstep holds.
"""

from __future__ import annotations

import logging
import time

from atom.kv_transfer import progress_doorbell
from atom.kv_transfer.disaggregation.types import connector_metadata_has_work
from atom.model_engine.sequence import SequenceStatus
from atom.utils import envs

logger = logging.getLogger("atom")


class LoadWaitState:
    """The load wait's inputs and counters.

    `tick_s` is the wall time of the last forward this engine ran
    (`OffloadLoadWait.on_forward_done`): the step cadence the waits are
    measured against. 0 until a forward ran, so a cold engine never waits.
    """

    LOG_INTERVAL_S = 60.0

    def __init__(self, dp_size: int) -> None:
        self.dp_size = max(1, int(dp_size))
        self.tick_s = 0.0
        self._last_log = time.monotonic()
        self._reset()

    def budget_s(self, gain: int, cost: int) -> float:
        """How long holding this step for `gain` loading requests is worth.

        Each request that lands in time prefills one forward (`tick_s`) earlier;
        each of the `cost` requests the step carries anyway is delayed by the
        hold, and under DP so are the peers' (the forward is a collective, and
        the router keeps the ranks alike, so the group carries about `dp_size`
        times this rank's count). The hold pays while it stays under
        `tick_s * gain / (cost * dp_size)`, and it is never longer than one
        tick: past that the request is no later than without the hold. With a
        running decode, `cost` is the whole decode batch and the bound is ~0.
        """
        if gain <= 0 or self.tick_s <= 0.0:
            return 0.0
        return min(self.tick_s, self.tick_s * gain / (max(1, cost) * self.dp_size))

    def _reset(self) -> None:
        self.steps = 0  # steps that started at least one load ahead of schedule()
        self.loads = 0  # loads started ahead of schedule()
        self.fresh = 0  # answers fetched before schedule() for a doorbell
        self.skipped = 0  # hold on: steps with nothing worth holding for
        self.waits = 0  # hold on: steps that held
        self.landed = 0  # held-for loads that landed (done or failed) in time
        self.timed_out = 0  # held-for loads still in flight at the bound
        self.wakes = 0  # doorbells taken while holding
        self.wait_s = 0.0
        self.max_wait_s = 0.0
        self.budget_total_s = 0.0

    def note_wait(
        self, landed: int, timed_out: int, waited_s: float, budget_s: float
    ) -> None:
        self.waits += 1
        self.landed += landed
        self.timed_out += timed_out
        self.wait_s += waited_s
        self.max_wait_s = max(self.max_wait_s, waited_s)
        self.budget_total_s += budget_s

    def maybe_log(self, label: str) -> None:
        now = time.monotonic()
        if now - self._last_log < self.LOG_INTERVAL_S:
            return
        if self.steps or self.fresh:
            logger.info(
                "%s: [LOAD-WAIT] last %.0fs: steps=%d loads=%d fresh=%d "
                "skipped=%d waits=%d landed=%d timed_out=%d wakes=%d "
                "avg_wait=%.1fms max_wait=%.1fms avg_bound=%.1fms tick=%.1fms",
                label,
                now - self._last_log,
                self.steps,
                self.loads,
                self.fresh,
                self.skipped,
                self.waits,
                self.landed,
                self.timed_out,
                self.wakes,
                1e3 * self.wait_s / self.waits if self.waits else 0.0,
                1e3 * self.max_wait_s,
                1e3 * self.budget_total_s / self.waits if self.waits else 0.0,
                1e3 * self.tick_s,
            )
        self._last_log = now
        self._reset()


# -- scheduler-side passes (read the scheduler; no state of their own on it) ----


def has_running_decode(scheduler) -> bool:
    """Whether `running` holds anything besides partial prefills."""
    return len(scheduler.running) > scheduler._partial_prefill_count


def prestart_tier_loads(scheduler) -> list:
    """Run this tick's early tier load ahead of `schedule()` and return the
    requests it parked.

    Exactly the early load `Scheduler._schedule` runs (same lookup prefetch,
    same `_early_load_tier_prefixes`, same KV-pressure bounds), only earlier, so
    the loads can be shipped before the engine schedules. It *is* that tick's
    pass: the `schedule()` that follows skips its own
    (`EarlyLoadState.take_tick_pass`), so the per-tick bounds
    (`_EARLY_LOAD_PER_TICK`, retry and prune cadence) stay per step. State loads
    it queued are published now so a joint load leaves with the first metadata
    build.

    Not while a decode is running: this pass allocates, and a running decode's
    page reservation (`reserved_decode_blocks`) is installed only inside
    `_schedule()`. That is an ordering requirement, not a tuning choice; the
    pass inside `_schedule()`, behind the reservation, runs instead.
    """
    if not scheduler.waiting or has_running_decode(scheduler):
        return []
    scheduler._prefetch_waiting_lookups()
    early_load = scheduler._early_load
    entries = early_load.entries
    before = set(entries)
    early_load.mark_tick_pass_ran()
    if not scheduler._early_load_tier_prefixes():
        return []
    started = [
        seq
        for seq_id, (seq, _) in entries.items()
        if seq_id not in before and seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    ]
    if started:
        scheduler._publish_state_loads()
    return started


def load_wait_tradeoff(scheduler, loading: list) -> tuple[int, int]:
    """(gain, cost) of holding this tick's forward for the tier loads of
    `loading`.

    gain: requests of `loading` still in flight that this tick could admit if
    their load landed now -- a free request slot each, and the prefill ahead of
    them in `waiting` order (partial prefills, then admittable waiters with the
    router's cached-prefix estimate) leaves token budget. Each one that lands
    in time prefills one forward earlier.

    cost: requests this tick's forward carries anyway -- everything in
    `running` (partial prefills and decodes) plus the admittable waiters that
    fit ahead of the last admittable target. Each one is delayed by whatever
    the engine waits. Requests further back are delayed too, but the next
    forward no longer has to carry the loaded prefill, which hands them that
    capacity back; to first order the two cancel, so they are not counted.

    Pure: no allocation, no collective, no state change. Best effort like the
    delayer's fill signal; a wrong count costs at most the bounded wait or one
    step of latency, never correctness.
    """
    targets = {id(seq) for seq in loading if scheduler._still_loading(seq)}
    cost = len(scheduler.running)
    slots = scheduler.max_num_seqs - len(scheduler.running)
    if not targets or slots <= 0:
        return 0, cost
    budget = scheduler.max_num_batched_tokens
    ahead = scheduler._partial_prefill_remaining_tokens()
    gain = 0
    carried = 0  # admittable non-target waiters seen so far
    for seq in scheduler.waiting:
        if ahead >= budget or slots <= 0:
            break
        if id(seq) in targets:
            gain += 1
            slots -= 1
            cost += carried
            carried = 0
            ahead += scheduler.pending_prefill_tokens(seq)
            targets.discard(id(seq))
            if not targets:
                break
        elif scheduler._is_admittable_prefill(seq):
            carried += 1
            slots -= 1
            ahead += scheduler.pending_prefill_tokens(seq, estimate=True)
    return gain, cost


# -- the engine step hook ----------------------------------------------------------


class OffloadLoadWait:
    """Engine step hook (`EngineCore.step_hooks`): `before_schedule` ahead of
    every `schedule()`, `on_forward_done` after every real forward.

    Reads the engine it serves (`scheduler`, `runner_mgr`, `label`, the KV
    progress event and `_collect_kv_transfer_progress`) and keeps everything
    else here.
    """

    def __init__(self, engine, dp_size: int, *, hold: bool) -> None:
        self._engine = engine
        self.hold = hold
        self.state = LoadWaitState(dp_size)
        # In this process a doorbell is rung by the KV output threads when a
        # worker's marker arrives (`AsyncIOProcManager`) and by a Store lease
        # renewal that finished or timed out (`AsyncStoreLookup.renew_async`).
        # Either sets the engine's KV progress event; the ring count tells a
        # ring from the event's other wakes (an answer's arrival, a request).
        progress_doorbell.install(engine._kv_progress_event.set, coalesce=False)
        self._seen_rings = progress_doorbell.rings()
        # When the completion query now outstanding was requested; -inf when
        # it was requested outside this hook (taken as before any doorbell).
        self._asked_at = float("-inf")

    # -- hooks ---------------------------------------------------------------

    def on_forward_done(self, seconds: float) -> None:
        self.state.tick_s = seconds

    def before_schedule(self) -> None:
        engine = self._engine
        scheduler = engine.scheduler
        state = self.state
        self._asked_at = float("-inf")
        started = prestart_tier_loads(scheduler)
        held = False
        if started:
            self._ship_connector_meta()
            state.steps += 1
            state.loads += len(started)
            if self.hold:
                budget = state.budget_s(*load_wait_tradeoff(scheduler, started))
                if budget > 0.0:
                    t0 = time.monotonic()
                    pending = self._wait_for_loads(started, t0 + budget)
                    state.note_wait(
                        len(started) - pending, pending, time.monotonic() - t0, budget
                    )
                    held = True
                else:
                    state.skipped += 1
        if not held:
            self._fetch_fresh_answer()
        state.maybe_log(engine.label)

    # -- internals -------------------------------------------------------------

    def _ship_connector_meta(self) -> None:
        """Dispatch whatever loads the connector releases now, outside a batch
        (the loads the pre-phase started; with `ATOM_STORE_ASYNC_RENEW`, a load
        whose lease renewal just finished). No RPC when there is nothing."""
        engine = self._engine
        meta = engine.scheduler.kv_connector.build_connector_meta()
        if connector_metadata_has_work(meta):
            engine.runner_mgr.call_func("process_kvconnector_output", meta)

    def _unread_doorbell(self) -> bool:
        return progress_doorbell.rings() != self._seen_rings

    def _take_doorbell(self) -> float:
        """Mark every ring so far read; returns when (engine clock)."""
        self._seen_rings = progress_doorbell.rings()
        return time.monotonic()

    def _collect_answer_after(self, rung_at: float) -> bool:
        """Apply the outstanding completion answer if it is in (which requests
        the next) and ship what it released; True when the answer applied was
        requested after `rung_at`, so it holds the completion that rang."""
        engine = self._engine
        asked_at = self._asked_at
        now = time.monotonic()
        answered = engine._collect_kv_transfer_progress() is not None
        if answered:
            # The next query went out inside that call, so not before `now`.
            self._asked_at = now
        # A load parked on its lease renewal leaves as soon as it is free.
        self._ship_connector_meta()
        return answered and asked_at >= rung_at

    def _fetch_fresh_answer(self) -> None:
        """A load ended since the last answer (an unread doorbell): fetch one
        answer requested after that ring, so this `schedule()` sees it. Waits
        for the query's round trip only, bounded by one tick; a cold engine
        (no forward yet) leaves the ring for the next step."""
        if not self._unread_doorbell() or self.state.tick_s <= 0.0:
            return
        event = self._engine._kv_progress_event
        rung_at = self._take_doorbell()
        deadline = rung_at + self.state.tick_s
        self.state.fresh += 1
        while True:
            # Cleared before the poll: an answer arriving after it wakes the wait.
            event.clear()
            if self._collect_answer_after(rung_at):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            event.wait(remaining)

    def _wait_for_loads(self, loading, deadline: float) -> int:
        """Block until every request of `loading` stopped loading or `deadline`;
        returns how many still load. Woken by events only: the KV doorbell (a
        load job ended on a worker, or a lease renewal here finished or reached
        its timeout unfinished) and the arrival of a completion answer."""
        engine = self._engine
        scheduler = engine.scheduler
        event = engine._kv_progress_event
        # When the last unread doorbell was taken: an answer requested after it
        # holds the completion that rang it. None: no unread doorbell.
        rung_at = None
        while True:
            pending = sum(1 for seq in loading if scheduler._still_loading(seq))
            now = time.monotonic()
            if not pending or now >= deadline:
                return pending
            if rung_at is None:
                if not self._unread_doorbell():
                    # Any other wake (an answer, a request) is not a ring.
                    event.clear()
                    if not self._unread_doorbell():
                        event.wait(deadline - now)
                    continue
                rung_at = self._take_doorbell()
                self.state.wakes += 1
            # Cleared before the poll: an answer arriving after it wakes the wait.
            event.clear()
            if self._collect_answer_after(rung_at):
                rung_at = None
                continue
            # The outstanding answer (or the one just requested) is still out.
            event.wait(max(0.0, deadline - time.monotonic()))


def load_wait_hook(engine, config) -> OffloadLoadWait | None:
    """The offload load wait for `engine`, or None when it is off
    (`ATOM_OFFLOAD_LOAD_EVENT_WAIT=0`)."""
    if not envs.ATOM_OFFLOAD_LOAD_EVENT_WAIT:
        return None
    return OffloadLoadWait(
        engine,
        config.parallel_config.data_parallel_size,
        hold=envs.ATOM_OFFLOAD_LOAD_WAIT_HOLD,
    )
