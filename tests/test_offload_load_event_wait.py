# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Offload load wait (`atom/kv_transfer/offload/load_wait.py`).

The early tier load runs ahead of `schedule()` and its loads are shipped at
once; after a KV progress doorbell (a worker's load job ended) one completion
answer requested after the ring is fetched before `schedule()`
(`ATOM_OFFLOAD_LOAD_EVENT_WAIT`, on). With `ATOM_OFFLOAD_LOAD_WAIT_HOLD` (on by
default) the step is also held for the loads it started while holding pays:
the hold blocks on the doorbell, not on a sleep, and ends as soon as every
started load landed; its bound comes from the measured forward time and the
requests the hold would delay, not from a constant. All of it is an engine
step hook: the engine and the scheduler carry none of its state.
"""

import pickle
import queue
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from test_mooncake_store_offload import (  # noqa: F401 (fixtures)
    aiter_arch,
    cluster,
    scheduler_clock,
)
from test_scheduler_early_load import _free, _make

from atom.kv_transfer import progress_doorbell
from atom.kv_transfer.offload import load_wait
from atom.model_engine.sequence import SequenceStatus

WAITING_LOAD = SequenceStatus.WAITING_FOR_REMOTE_KVS


@pytest.fixture(autouse=True)
def _no_doorbell():
    progress_doorbell.install(None)
    yield
    progress_doorbell.install(None)


@pytest.fixture
def engine_core(monkeypatch):
    # Scheduling only, not the aiter-backed worker RPC.
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine import engine_core as module

    monkeypatch.setattr(module.tracing, "record_forward", Mock())
    return module


def _fire(monkeypatch, seq_factory, n=1):
    """A scheduler that admits every tick (no prefill delayer)."""
    sched, delayer, seqs = _make(monkeypatch, seq_factory, n=n)
    sched.set_prefill_delayer(None)
    # Metadata the engine treats as carrying work, so it is shipped.
    sched.kv_connector.build_connector_meta = lambda: SimpleNamespace(
        reqs_to_recv={"load": object()}
    )
    return sched, delayer, seqs


class _Mgr:
    """Runner-manager double.

    A shipped load lands on a timer thread after its delay (None: only with a
    forward) and rings the doorbell, as `LoadExecutor` does (the marker's
    arrival rings the engine's doorbell, which sets its KV progress event). A
    completion answer is a snapshot taken when it is requested, as a worker
    answers the query when it reaches it, and arrives at once (setting the
    engine's KV progress event); `collect` closes the outstanding one and
    requests the next (`AsyncIOProcManager.call_func_with_aggregation`'s
    pipelining). A forward lands whatever is still in flight by its end.
    """

    def __init__(self, sched, progress_event, delay=0.0, fail=False, forward_s=0.0):
        self.sched = sched
        self.progress_event = progress_event
        self.delay = delay  # seconds, None, or {seq id: seconds | None}
        self.fail = fail
        self.forward_s = forward_s
        self.lock = threading.Lock()
        self.shipped, self.inflight, self.landed = set(), set(), set()
        self.pending = None  # (requested at, snapshot)
        self.calls, self.forwards, self.timers = [], [], []

    def _delay(self, seq_id):
        return self.delay.get(seq_id) if isinstance(self.delay, dict) else self.delay

    def land(self, seq_id):
        with self.lock:
            if seq_id not in self.inflight:
                return
            self.inflight.discard(seq_id)
            self.landed.add(seq_id)
        progress_doorbell.ring()

    def call_func(self, name, *args, **kwargs):
        self.calls.append(name)
        if name == "process_kvconnector_output":
            for seq in list(self.sched.waiting):
                if seq.status != WAITING_LOAD or seq.id in self.shipped:
                    continue
                self.shipped.add(seq.id)
                with self.lock:
                    self.inflight.add(seq.id)
                delay = self._delay(seq.id)
                if delay == 0:
                    self.land(seq.id)
                elif delay is not None:
                    timer = threading.Timer(delay, self.land, (seq.id,))
                    timer.start()
                    self.timers.append(timer)
            return None
        if name == "forward":
            if self.forward_s:
                time.sleep(self.forward_s)
            self.forwards.append(list(args[0].req_ids))
            for seq_id in list(self.inflight):
                self.land(seq_id)
            return "fwd"
        raise AssertionError(name)

    def _request(self):
        with self.lock:
            snapshot = set(self.landed)
            self.landed -= snapshot
        self.pending = (time.monotonic(), snapshot)
        self.progress_event.set()  # the answer arrives

    def collect(self):
        self.calls.append("poll")
        if self.pending is None:
            self._request()
            return None
        _, snapshot = self.pending
        for seq_id in snapshot:
            # As `_update_from_kv_xfer_finished`: an aborted load only cleans up.
            if self.sched._finish_aborted_load_cleanup(seq_id):
                continue
            target = (
                self.sched.failed_recving_kv_req_ids
                if self.fail
                else self.sched.finished_recving_kv_req_ids
            )
            target.append(seq_id)
        self._request()
        return snapshot

    def stop(self):
        for timer in self.timers:
            timer.cancel()


def _engine(module, sched, *, on=True, hold=True, tick_s=1.0, dp=1, **mgr_kwargs):
    """An engine double running the real step (`_process_engine_step_inner`)
    with the real hook (`OffloadLoadWait`) as its only step hook; `on=False`:
    no hook, as with `ATOM_OFFLOAD_LOAD_EVENT_WAIT=0`."""
    event = threading.Event()
    mgr = _Mgr(sched, event, **mgr_kwargs)
    engine = SimpleNamespace(
        scheduler=sched,
        kv_transfer_enabled=True,
        runner_mgr=mgr,
        _collect_kv_transfer_progress=mgr.collect,
        _poll_kv_transfer_progress=lambda: bool(mgr.collect()),
        _wake_event=event,
        input_queue=queue.Queue(),
        utility_queue=queue.Queue(),
        output_queue=queue.Queue(),
        _postprocess=Mock(),
        _release_request_resources=Mock(),
        _wait_for_kv_progress=lambda: None,
        _advance_idle_kv_transfer=mgr.collect,
        label="test",
        step_hooks=(),
    )
    if on:
        hook = load_wait.OffloadLoadWait(engine, dp, hold=hold)
        hook.state.tick_s = tick_s
        engine.step_hooks = (hook,)
    return engine, mgr


def _stats(engine):
    (hook,) = engine.step_hooks
    return hook.state


def _step(module, engine):
    return module.EngineCore._process_engine_step_inner(engine)


def _steps_until_prefill(module, engine, mgr, seq, limit=6):
    for step in range(1, limit + 1):
        _step(module, engine)
        if mgr.forwards and seq.id in mgr.forwards[-1]:
            return step
    return None


# -- load lands before / while / after the hold ------------------------------


def test_load_that_already_landed_prefills_in_this_step_without_waiting(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, delay=0)  # lands as it ships

    t0 = time.monotonic()
    assert _steps_until_prefill(engine_core, engine, mgr, seq) == 1
    assert time.monotonic() - t0 < 0.5
    stats = _stats(engine)
    assert (stats.steps, stats.loads, stats.waits, stats.landed) == (1, 1, 1, 1)
    assert stats.timed_out == 0 and stats.wakes == 1
    assert sched.kv_connector.allocs == [seq.id]  # one admission only
    assert sched._num_parked_remote_kv == 0


def test_hold_ends_on_the_doorbell_not_at_its_bound(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    # Bound 2 s (one tick, nothing else to delay); the load lands at 50 ms.
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, delay=0.05)

    t0 = time.monotonic()
    _step(engine_core, engine)
    elapsed = time.monotonic() - t0

    assert mgr.forwards == [[seq.id]]  # this step's forward
    assert 0.05 <= elapsed < 1.0
    # Shipped first; no poll before the doorbell rang (no sleep-polling).
    assert mgr.calls[0] == "process_kvconnector_output"
    held = mgr.calls[: mgr.calls.index("forward")]
    assert held.count("poll") <= 3
    assert _stats(engine).wakes == 1


def test_late_load_rides_the_forward_after_the_bound(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=0.05, delay=None)

    t0 = time.monotonic()
    _step(engine_core, engine)
    elapsed = time.monotonic() - t0

    assert 0.05 <= elapsed < 1.0  # held for the bound, one tick
    assert seq.status == WAITING_LOAD
    stats = _stats(engine)
    assert (stats.waits, stats.landed, stats.timed_out, stats.wakes) == (1, 0, 1, 0)
    # It lands later and is admitted when its completion is read, as without
    # the hold; it is not held for again (not a new load).
    mgr.land(seq.id)
    assert _steps_until_prefill(engine_core, engine, mgr, seq) is not None
    assert _stats(engine).waits == 1


def test_failed_load_wakes_the_hold_and_falls_back_to_prefill(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, delay=0.02, fail=True)

    t0 = time.monotonic()
    assert _steps_until_prefill(engine_core, engine, mgr, seq) == 1  # recomputed
    assert time.monotonic() - t0 < 1.0
    assert _stats(engine).landed == 1
    assert sched._num_parked_remote_kv == 0
    assert seq.id not in sched._early_load.entries


def test_without_the_hook_the_same_load_prefills_later(
    engine_core, monkeypatch, seq_factory
):
    # Off: the load lands during step 1's forward and is read only after it.
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, on=False, delay=0)
    assert _steps_until_prefill(engine_core, engine, mgr, seq) >= 2
    assert engine.step_hooks == ()


# -- hold off: doorbell + a fresh answer before schedule() only -----------------


def test_hold_off_does_not_hold_for_a_load_in_flight(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, hold=False, tick_s=2.0, delay=0.3)
    try:
        t0 = time.monotonic()
        _step(engine_core, engine)
        assert time.monotonic() - t0 < 0.2  # no hold for the 300 ms load
        assert seq.status == WAITING_LOAD
        stats = _stats(engine)
        assert (stats.steps, stats.loads, stats.waits, stats.skipped) == (1, 1, 0, 0)
        assert mgr.calls[0] == "process_kvconnector_output"  # still shipped early
    finally:
        mgr.stop()


def _steps_to_prefill_with_pauses(module, engine, mgr, seq, pause_s=0.1):
    """Steps until `seq` prefills, idling `pause_s` between steps (the load
    lands in a pause, not during a forward)."""
    for step in range(1, 7):
        _step(module, engine)
        if mgr.forwards and seq.id in mgr.forwards[-1]:
            return step
        time.sleep(pause_s)
    return None


def test_hold_off_fresh_answer_admits_a_landed_load_before_schedule(
    engine_core, monkeypatch, seq_factory
):
    """The load lands between steps. The doorbell makes the next step fetch an
    answer requested after the ring before it schedules, so it prefills on
    that step; without the hook it waits for the pipelined answers."""
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, hold=False, tick_s=2.0, delay=0.03)
    assert _steps_to_prefill_with_pauses(engine_core, engine, mgr, seq) == 2
    stats = _stats(engine)
    assert stats.fresh == 1 and stats.waits == 0

    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, on=False, delay=0.03)
    assert _steps_to_prefill_with_pauses(engine_core, engine, mgr, seq) >= 3


def test_hold_off_fresh_answer_runs_while_a_decode_is_running(
    engine_core, monkeypatch, seq_factory
):
    # No pre-phase load while a decode runs (page reservation), but a load that
    # landed is still read before schedule().
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, hold=False, tick_s=2.0, delay=None)
    _step(engine_core, engine)  # parks and ships the load
    assert seq.status == WAITING_LOAD
    allocs = list(sched.kv_connector.allocs)
    sched.running = deque([_decode()])
    mgr.land(seq.id)  # rings

    (hook,) = engine.step_hooks
    hook.before_schedule()

    assert seq.id in sched.finished_recving_kv_req_ids
    assert hook.state.fresh == 1
    assert sched.kv_connector.allocs == allocs  # nothing allocated ahead


def _bare_hook(tick_s, collect):
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(
            waiting=[], kv_connector=SimpleNamespace(build_connector_meta=lambda: None)
        ),
        runner_mgr=SimpleNamespace(call_func=Mock()),
        _collect_kv_transfer_progress=collect,
        _wake_event=threading.Event(),
        label="test",
    )
    hook = load_wait.OffloadLoadWait(engine, 1, hold=False)
    hook.state.tick_s = tick_s
    return hook


def test_fresh_answer_is_bounded_by_one_tick():
    # A query that is never answered: the step goes on after one tick.
    collect = Mock(return_value=None)
    hook = _bare_hook(0.05, collect)
    progress_doorbell.ring()
    t0 = time.monotonic()
    hook.before_schedule()
    assert 0.05 <= time.monotonic() - t0 < 0.5
    assert hook.state.fresh == 1 and collect.called


def test_no_fresh_answer_without_a_ring_or_before_the_first_forward():
    collect = Mock(return_value=None)
    hook = _bare_hook(0.05, collect)
    hook.before_schedule()  # nothing rang
    assert not collect.called
    cold = _bare_hook(0.0, collect)
    progress_doorbell.ring()
    cold.before_schedule()  # no forward yet: left for the next step
    assert not collect.called and cold.state.fresh == 0


def test_load_wait_hook_follows_its_envs(monkeypatch):
    engine = SimpleNamespace(
        _wake_event=threading.Event(),
        scheduler=SimpleNamespace(
            kv_connector=SimpleNamespace(supports_early_load=True)
        ),
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=4))
    monkeypatch.delenv("ATOM_OFFLOAD_LOAD_EVENT_WAIT", raising=False)
    monkeypatch.delenv("ATOM_OFFLOAD_LOAD_WAIT_HOLD", raising=False)
    hook = load_wait.load_wait_hook(engine, config)
    assert hook is not None and hook.hold is True  # default: hold on
    assert hook.state.dp_size == 4
    monkeypatch.setenv("ATOM_OFFLOAD_LOAD_WAIT_HOLD", "0")
    hook = load_wait.load_wait_hook(engine, config)
    assert hook is not None and hook.hold is False  # off: doorbell only
    monkeypatch.setenv("ATOM_OFFLOAD_LOAD_EVENT_WAIT", "0")
    assert load_wait.load_wait_hook(engine, config) is None


@pytest.mark.parametrize(
    "connector",
    [
        SimpleNamespace(supports_early_load=False),  # dense / DSV4 offload, P/D only
        None,  # no connector (or a disaggregated decode engine before its scheduler)
    ],
    ids=["dense", "none"],
)
def test_no_hook_without_an_offload_tier_that_loads_early(monkeypatch, connector):
    # `LoadExecutor` rings the doorbell on every offload tier: without the gate
    # a dense tier's ring would make each step wait for a fresh answer.
    from atom.kv_transfer.engine_step_hooks import kv_transfer_step_hooks

    monkeypatch.delenv("ATOM_OFFLOAD_LOAD_EVENT_WAIT", raising=False)
    monkeypatch.delenv("ATOM_OFFLOAD_LOAD_WAIT_HOLD", raising=False)
    engine = SimpleNamespace(
        _wake_event=threading.Event(),
        scheduler=SimpleNamespace(kv_connector=connector),
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=4))
    assert load_wait.load_wait_hook(engine, config) is None
    assert kv_transfer_step_hooks(engine, config) == ()
    # The doorbell stays uninstalled: a ring sets no engine event.
    progress_doorbell.ring()
    assert not engine._wake_event.is_set()


def test_dense_offload_connector_gets_no_hook(monkeypatch, seq_factory):
    # A real scheduler whose connector cannot load early: the factory leaves
    # the engine without a step hook, so its steps are main's.
    sched, _, _ = _make(monkeypatch, seq_factory)
    sched.kv_connector.supports_early_load = False
    monkeypatch.delenv("ATOM_OFFLOAD_LOAD_EVENT_WAIT", raising=False)
    engine = SimpleNamespace(_wake_event=threading.Event(), scheduler=sched)
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))
    assert load_wait.load_wait_hook(engine, config) is None
    sched.kv_connector.supports_early_load = True
    assert load_wait.load_wait_hook(engine, config) is not None


def test_cold_engine_never_holds(engine_core, monkeypatch, seq_factory):
    sched, _, _ = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=0.0, delay=None)

    t0 = time.monotonic()
    _step(engine_core, engine)

    assert time.monotonic() - t0 < 0.5
    stats = _stats(engine)
    assert (stats.steps, stats.skipped, stats.waits) == (1, 1, 0)
    assert mgr.calls[0] == "process_kvconnector_output"  # still shipped early


def test_abort_while_held_releases_blocks(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    free = _free(sched)
    engine, mgr = _engine(engine_core, sched, tick_s=0.02, delay=None)
    _step(engine_core, engine)
    assert seq.status == WAITING_LOAD

    assert sched.abort_request(seq.id)
    assert _free(sched) < free  # the aborted-load protocol: freed on its report
    mgr.land(seq.id)
    _step(engine_core, engine)
    _step(engine_core, engine)
    assert _free(sched) == free
    assert sched._num_parked_remote_kv == 0


# -- several requests ----------------------------------------------------------


def test_hold_waits_for_every_started_load(engine_core, monkeypatch, seq_factory):
    sched, _, (a, b) = _fire(monkeypatch, seq_factory, n=2)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, delay=None)
    mgr.delay = {a.id: 0.03, b.id: 0.08}

    t0 = time.monotonic()
    _step(engine_core, engine)
    elapsed = time.monotonic() - t0

    assert mgr.forwards == [[a.id, b.id]]  # both, in arrival order
    assert 0.08 <= elapsed < 1.0
    assert _stats(engine).landed == 2
    assert _stats(engine).wakes >= 1


def test_a_load_that_misses_the_bound_does_not_hold_back_one_that_landed(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (a, b) = _fire(monkeypatch, seq_factory, n=2)
    # Both admittable, nothing else to delay: bound = one tick.
    engine, mgr = _engine(engine_core, sched, tick_s=0.1, delay=None)
    mgr.delay = {a.id: None, b.id: 0.02}

    _step(engine_core, engine)

    assert mgr.forwards == [[b.id]]
    assert a.status == WAITING_LOAD
    stats = _stats(engine)
    assert (stats.landed, stats.timed_out) == (1, 1)


# -- the bound: gain against cost, no constant ----------------------------------


def _partial_prefill(remaining=8):
    return SimpleNamespace(
        is_partial_prefill=True, num_tokens=remaining, num_cached_tokens=0
    )


def _decode():
    return SimpleNamespace(is_partial_prefill=False)


def test_budget_formula(engine_core):
    state = load_wait.LoadWaitState(dp_size=1)
    assert state.budget_s(1, 0) == 0.0  # no forward measured yet
    state.tick_s = 0.2
    assert state.budget_s(0, 0) == 0.0  # nothing to gain
    assert state.budget_s(1, 0) == pytest.approx(0.2)  # nothing to delay: 1 tick
    assert state.budget_s(3, 1) == pytest.approx(0.2)  # never past one tick
    assert state.budget_s(1, 4) == pytest.approx(0.05)
    dp = load_wait.LoadWaitState(dp_size=4)
    dp.tick_s = 0.2
    assert dp.budget_s(1, 4) == pytest.approx(0.0125)  # peers pay too
    assert dp.budget_s(1, 0) == pytest.approx(0.05)


def test_tradeoff_on_a_prefill_only_step(monkeypatch, seq_factory):
    sched, _, (a, b, c) = _fire(monkeypatch, seq_factory, n=3)
    sched.kv_connector.not_ready_ids = {a.id}  # a stays an ordinary waiter
    started = load_wait.prestart_tier_loads(sched)
    assert started == [b, c]
    # a (16 tokens) ahead of b and c: both still fit the 64-token budget, and
    # the step carries a anyway.
    assert load_wait.load_wait_tradeoff(sched, started) == (2, 1)
    sched.running = deque([_partial_prefill()])
    sched._partial_prefill_count = 1
    assert load_wait.load_wait_tradeoff(sched, started) == (2, 2)
    # A load that landed or was aborted is not a gain.
    c.status = SequenceStatus.ABORTED
    assert load_wait.load_wait_tradeoff(sched, started) == (1, 2)
    assert load_wait.load_wait_tradeoff(sched, []) == (0, 1)


def test_tradeoff_stops_at_the_token_budget_and_slots(monkeypatch, seq_factory):
    sched, _, seqs = _fire(monkeypatch, seq_factory, n=6)
    *ahead, last = seqs
    sched.kv_connector.not_ready_ids = {s.id for s in ahead}
    started = load_wait.prestart_tier_loads(sched)
    assert started == [last]
    # 5 x 16 tokens ahead fill the 64-token budget: nothing to gain.
    assert load_wait.load_wait_tradeoff(sched, started)[0] == 0
    sched2, _, _ = _fire(monkeypatch, seq_factory)
    started = load_wait.prestart_tier_loads(sched2)
    sched2.running = deque(_partial_prefill(1) for _ in range(sched2.max_num_seqs))
    sched2._partial_prefill_count = sched2.max_num_seqs
    assert load_wait.load_wait_tradeoff(sched2, started)[0] == 0  # no free slot


def test_running_decodes_shrink_the_bound_to_almost_nothing(
    engine_core, monkeypatch, seq_factory
):
    # The same rule, no special case: every running decode is delayed by a
    # hold, so the bound falls with the decode batch.
    sched, _, _ = _fire(monkeypatch, seq_factory)
    started = load_wait.prestart_tier_loads(sched)
    sched.running = deque(_decode() for _ in range(6))
    gain, cost = load_wait.load_wait_tradeoff(sched, started)
    assert (gain, cost) == (1, 6)
    state = load_wait.LoadWaitState(dp_size=4)
    state.tick_s = 0.012  # a decode step
    assert state.budget_s(gain, cost) == pytest.approx(0.0005)


def test_no_pre_phase_load_while_a_decode_is_running(monkeypatch, seq_factory):
    # Ordering, not tuning: the pre-phase allocates, and a running decode's
    # page reservation is installed only inside `_schedule()`.
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    sched.running = deque([_decode()])
    free = _free(sched)
    assert load_wait.prestart_tier_loads(sched) == []
    assert sched.kv_connector.allocs == [] and _free(sched) == free
    assert seq.status == SequenceStatus.WAITING
    assert sched._early_load.take_tick_pass()  # `_schedule()` runs its own pass
    sched.running = deque([_partial_prefill(), _decode()])
    sched._partial_prefill_count = 1
    assert load_wait.prestart_tier_loads(sched) == []
    # Partial prefills only (the P/D prefill case): the pre-phase runs.
    sched.running = deque([_partial_prefill()])
    assert load_wait.prestart_tier_loads(sched) == [seq]


def test_no_decode_prefill_only_step_holds(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, delay=0.03)
    assert not sched.has_running_decode()
    _step(engine_core, engine)
    assert mgr.forwards == [[seq.id]]
    assert _stats(engine).waits == 1


def test_forward_time_is_the_tick(engine_core, monkeypatch, seq_factory):
    sched, _, _ = _fire(monkeypatch, seq_factory)
    engine, _ = _engine(engine_core, sched, tick_s=2.0, delay=0, forward_s=0.03)
    _step(engine_core, engine)
    assert 0.03 <= _stats(engine).tick_s < 0.5


def test_dp_dummy_forward_updates_the_tick(engine_core, monkeypatch, seq_factory):
    # An idle DP rank runs dummy forwards paired with its peers' real ones:
    # their wall time is the group's step, so the tick follows it instead of
    # keeping this rank's last real (long prefill) forward.
    sched, _, _ = _fire(monkeypatch, seq_factory)
    engine, _ = _engine(engine_core, sched, tick_s=0.5, dp=8)  # a 500 ms prefill

    def dummy_execution(name, wait_out=False):
        assert name == "dummy_execution" and wait_out
        time.sleep(0.03)
        return "fwd"

    engine.runner_mgr = SimpleNamespace(call_func=dummy_execution)
    engine.scheduler = SimpleNamespace(running=())
    engine_core.DPEngineCoreProc._execute_dummy_batch(engine)
    state = _stats(engine)
    assert 0.03 <= state.tick_s < 0.3
    assert state.budget_s(1, 0) == pytest.approx(state.tick_s / 8)


def test_a_peer_decode_makes_the_hold_budget_zero(engine_core):
    state = load_wait.LoadWaitState(dp_size=8)
    state.tick_s = 0.03
    # No decode on this rank (cost 0) and a fresh Store hit: tick / dp_size ...
    assert state.budget_s(1, 0) == pytest.approx(0.03 / 8)
    # ... unless a peer decodes: the hold would add to its ITL.
    assert state.budget_s(1, 0, group_decoding=True) == 0.0
    assert state.budget_s(3, 1, group_decoding=True) == 0.0


def test_a_peer_decode_keeps_the_step_from_holding(
    engine_core, monkeypatch, seq_factory
):
    # This rank has no decode, so its pre-phase still starts and ships the
    # load, but a peer's running decode (reduced on `_sync_dp_state`) leaves
    # nothing to hold for: the step goes on and the load rides the forward.
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, dp=4, delay=None)
    engine.dp_group_decoding = True

    t0 = time.monotonic()
    _step(engine_core, engine)

    assert time.monotonic() - t0 < 0.5
    stats = _stats(engine)
    assert (stats.steps, stats.loads, stats.skipped, stats.waits) == (1, 1, 1, 0)
    assert mgr.calls[0] == "process_kvconnector_output"  # still shipped early
    assert seq.status == WAITING_LOAD


def test_prefill_only_group_keeps_the_hold_and_its_bound(
    engine_core, monkeypatch, seq_factory
):
    # The measured case (a P/D prefill instance: no decode on any rank): the
    # group flag is clear, the bound is the same formula as before, and the
    # step holds for the load and prefills it in this step.
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, dp=4, delay=0.03)
    engine.dp_group_decoding = False

    _step(engine_core, engine)

    state = _stats(engine)
    assert mgr.forwards == [[seq.id]]
    assert (state.waits, state.landed, state.timed_out) == (1, 1, 0)
    # tick * gain / (max(1, cost) * dp_size) = 2.0 * 1 / (1 * 4): unchanged.
    assert state.budget_total_s == pytest.approx(2.0 / 4)


def test_sync_dp_state_carries_any_rank_decoding(engine_core, monkeypatch):
    # One more flag on the same all_reduce (no new collective): a peer's
    # running decode reaches every rank as `dp_group_decoding`.
    import torch

    p = engine_core.DPEngineCoreProc.__new__(engine_core.DPEngineCoreProc)
    p.label = "dp0"
    p._shutting_down = False
    p.dp_group = object()
    p.scheduler = SimpleNamespace(has_running_decode=lambda: False)
    calls = []
    peer = {"decoding": 1}

    def all_reduce(tensor, op=None, group=None):
        calls.append(tensor.tolist())
        tensor[4] = max(int(tensor[4]), peer["decoding"])

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert p._sync_dp_state(True, False, False) == (True, False, False)
    assert p.dp_group_decoding is True
    peer["decoding"] = 0
    p._sync_dp_state(True, False, False)
    assert p.dp_group_decoding is False
    p.scheduler = SimpleNamespace(has_running_decode=lambda: True)
    p._sync_dp_state(True, False, False)
    assert calls[-1] == [1, 0, 0, 0, 1]  # this rank's own decode is sent
    assert p.dp_group_decoding is True
    assert len(calls) == 3  # one collective per sync


# -- off is main, one early-load pass per step ------------------------------------


def test_off_step_is_unchanged(engine_core, monkeypatch):
    monkeypatch.setattr(
        load_wait,
        "prestart_tier_loads",
        Mock(side_effect=AssertionError("pre-phase ran")),
    )
    batch = SimpleNamespace(
        req_ids=["r"], connector_meta_output=None, total_seqs_num_prefill=1
    )
    order = Mock()
    scheduler = SimpleNamespace(
        schedule=Mock(return_value=(batch, {})),
        take_rejected=Mock(return_value=[]),
        compute_detailed_aggregates=Mock(),
        metrics=SimpleNamespace(record_forward=Mock()),
        prefill_delayer=None,
    )
    order.attach_mock(scheduler.schedule, "schedule")
    runner = SimpleNamespace(call_func=Mock(return_value="fwd"))
    order.attach_mock(runner.call_func, "call_func")
    poll = Mock(return_value=False)
    order.attach_mock(poll, "poll")
    engine = SimpleNamespace(
        step_hooks=(),
        scheduler=scheduler,
        kv_transfer_enabled=True,
        runner_mgr=runner,
        _poll_kv_transfer_progress=poll,
        _wake_event=threading.Event(),
        _postprocess=Mock(),
    )
    assert engine_core.EngineCore._process_engine_step_inner(engine) is True
    assert [c[0] for c in order.mock_calls] == ["schedule", "call_func", "poll"]


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_one_lookup_prefetch_and_one_early_load_pass_per_step(
    engine_core, monkeypatch, seq_factory, on
):
    sched, _, seqs = _fire(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {s.id for s in seqs}  # stay waiting
    events = []
    conn = sched.kv_connector
    prefetch = conn.prefetch_lookups
    early_load = sched._early_load_tier_prefixes

    def counting_prefetch(waiting):
        events.append("prefetch")
        return prefetch(waiting)

    def tracing_early_load():
        events.append("early_load")
        return early_load()

    conn.prefetch_lookups = counting_prefetch
    monkeypatch.setattr(sched, "_early_load_tier_prefixes", tracing_early_load)
    engine, _ = _engine(engine_core, sched, on=on, delay=None)
    for _ in range(3):
        events.clear()
        _step(engine_core, engine)
        assert events == ["prefetch", "early_load"]
    assert sched._early_load.tick == 3


def test_state_load_queued_by_the_pre_phase_ships_with_its_first_build(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    conn = sched.kv_connector
    queued, enqueued, shipped = [], [], []
    record_alloc = conn.update_state_after_alloc

    def update_state_after_alloc(s):
        record_alloc(s)
        queued.append((s.id, "state-hash", 0))

    def take_state_loads():
        out = list(queued)
        queued.clear()
        return out

    def enqueue_state_loads(loads):
        enqueued.extend(loads)
        return True

    def build_connector_meta():
        meta = SimpleNamespace(reqs_to_recv={"load": object()}, state=list(enqueued))
        enqueued.clear()
        return meta

    conn.update_state_after_alloc = update_state_after_alloc
    conn.enqueue_state_loads = enqueue_state_loads
    conn.build_connector_meta = build_connector_meta
    monkeypatch.setattr(
        sched.block_manager, "take_state_loads", take_state_loads, raising=False
    )
    engine, mgr = _engine(engine_core, sched, tick_s=0.01, delay=None)
    inner = mgr.call_func

    def call_func(name, *args, **kwargs):
        if name == "process_kvconnector_output":
            shipped.append(args[0])
        return inner(name, *args, **kwargs)

    mgr.call_func = call_func
    _step(engine_core, engine)
    assert shipped and shipped[0].state == [(seq.id, "state-hash", 0)]


# -- the doorbell -------------------------------------------------------------------


def test_doorbell_coalesces_until_rearmed():
    sent = []
    progress_doorbell.ring()  # not installed: a no-op
    progress_doorbell.install(lambda: sent.append(1))
    progress_doorbell.ring()
    progress_doorbell.ring()
    assert sent == [1]
    progress_doorbell.rearm()
    progress_doorbell.ring()
    assert sent == [1, 1]
    progress_doorbell.install(lambda: sent.append(2), coalesce=False)
    progress_doorbell.ring()
    progress_doorbell.ring()
    assert sent == [1, 1, 2, 2]


def test_a_failed_send_does_not_silence_the_doorbell():
    calls = []

    def send():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("queue closed")

    progress_doorbell.install(send)
    progress_doorbell.ring()
    progress_doorbell.ring()
    assert len(calls) == 2


def test_load_executor_rings_after_the_job_recorded_its_result():
    from atom.kv_transfer.offload._offload_common import LoadExecutor

    record, seen = [], []
    progress_doorbell.install(lambda: seen.append(list(record)))
    pool = LoadExecutor(max_workers=1)
    pool.submit(record.append, "done")
    pool.shutdown(wait=True)
    assert seen == [["done"]]


def test_manager_rings_the_engine_doorbell_and_keeps_the_marker_out_of_answers():
    import zmq
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.kv_transfer.disaggregation import KVConnectorOutput
        from atom.kv_transfer.progress_doorbell import KV_PROGRESS_DOORBELL
        from atom.model_engine.async_proc import AsyncIOProcManager

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    addr = f"tcp://127.0.0.1:{port}"
    mgr = AsyncIOProcManager.__new__(AsyncIOProcManager)
    mgr.label = "test"
    mgr.zmq_ctx = zmq.Context()
    mgr.still_running = True
    mgr.kv_outputs_queues = [queue.Queue()]
    mgr.kv_progress_event = threading.Event()
    # The engine's doorbell, as `OffloadLoadWait` installs it.
    rung = threading.Event()
    progress_doorbell.install(rung.set, coalesce=False)
    rings = progress_doorbell.rings()
    thread = threading.Thread(
        target=mgr.process_kv_output_sockets, args=(addr, 0), daemon=True
    )
    thread.start()
    push = mgr.zmq_ctx.socket(zmq.PUSH)
    push.connect(addr)
    try:
        push.send(pickle.dumps(KV_PROGRESS_DOORBELL))
        assert rung.wait(5)
        assert progress_doorbell.rings() == rings + 1
        assert mgr.kv_progress_event.wait(5)
        assert mgr.kv_outputs_queues[0].empty()  # never joins an aggregation
        push.send(pickle.dumps(KVConnectorOutput()))
        assert isinstance(mgr.kv_outputs_queues[0].get(timeout=5), KVConnectorOutput)
        assert progress_doorbell.rings() == rings + 1  # an answer is not a ring
    finally:
        mgr.still_running = False
        push.close(linger=0)
        thread.join(timeout=3)
        mgr.zmq_ctx.term()


def test_offload_worker_rearms_the_doorbell_in_get_finished():
    from atom.kv_transfer.offload._offload_common import OffloadWorkerMixin

    sent = []
    progress_doorbell.install(lambda: sent.append(1))
    worker = OffloadWorkerMixin.__new__(OffloadWorkerMixin)
    worker._lock = threading.Lock()
    worker._done_load, worker._failed_load, worker._done_save = {"a"}, set(), set()
    worker._connector_completions = set()
    progress_doorbell.ring()
    progress_doorbell.ring()  # coalesced
    assert worker.get_finished().finished_loading == {"a"}
    progress_doorbell.ring()  # rearmed by the answer
    assert sent == [1, 1]


def test_shared_engine_classes_carry_no_load_wait_state():
    """The load wait lives in its offload module, reached through the
    engine's generic step hooks: the shared engine classes name none of it."""
    root = Path(__file__).resolve().parents[1] / "atom" / "model_engine"
    feature_names = (
        "load_wait",
        "LoadWait",
        "prestart",
        "_load_event_wait",
        "kv_doorbell_event",
        "kv_aggregation_started_at",
        "ATOM_OFFLOAD_LOAD_EVENT_WAIT",
        "ATOM_OFFLOAD_LOAD_WAIT_HOLD",
        "rearm",
    )
    for name in ("engine_core.py", "scheduler.py", "async_proc.py", "model_runner.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert [n for n in feature_names if n in source] == [], name


# -- a load parked on its Store lease renewal -----------------------------------------


def _hold_for_a_renewal(engine_core, clock, monkeypatch, *, stall):
    """Hold a step for one load parked on its async Store lease renewal
    (`ATOM_STORE_ASYNC_RENEW`), bound 2 s. The renewal finishes at 30 ms, or
    with `stall` never (its timeout is 0.1 s). Each shipped GET lands 20 ms
    later. Returns (seconds held, GETs shipped, loads reported, wait stats)."""
    from concurrent.futures import Future

    from test_mooncake_store_offload import (
        _arm_load,
        _config,
        _install_future,
        _seq,
    )

    from atom.kv_transfer.offload.mooncake_store import async_lookup
    from atom.kv_transfer.offload.mooncake_store.scheduler import (
        MooncakeStoreOffloadScheduler,
    )

    monkeypatch.setenv("ATOM_STORE_ASYNC_RENEW", "1")
    connector = MooncakeStoreOffloadScheduler(_config("kv_consumer"))
    seq = _seq(59, 50)
    _arm_load(connector, seq, hbm=16, lmc=40)
    clock.now += 2.0  # old enough to need a renewal
    future = Future()
    if stall:
        # Real clock: the renewal's timeout passes while the step is held.
        monkeypatch.setattr(async_lookup, "RENEW_TIMEOUT_S", 0.1)
        client = connector._lookup_client
        client.renew_executor.shutdown()
        client.renew_executor = SimpleNamespace(
            submit=lambda fn, keys: future, shutdown=lambda: None
        )
    else:
        _install_future(connector, clock, monkeypatch, future, [])

    event = threading.Event()
    shipped, worker_landed, reported = [], set(), set()
    pending = []

    def land():
        worker_landed.add(seq.id)
        progress_doorbell.ring()  # the worker's marker, rung in the engine

    def call_func(name, meta):
        assert name == "process_kvconnector_output"
        shipped.extend(r.load_operation for r in meta.requests)
        threading.Timer(0.02, land).start()  # the GET lands 20 ms later

    def collect():
        answer = None
        if pending:
            answer = pending.pop()
            reported.update(answer)
        pending.append(set(worker_landed))
        event.set()
        return answer

    monkeypatch.setattr(load_wait, "prestart_tier_loads", lambda sched: [seq])
    monkeypatch.setattr(load_wait, "load_wait_tradeoff", lambda sched, s: (1, 0))
    scheduler = SimpleNamespace(
        kv_connector=connector,
        _still_loading=lambda s: s.id not in reported,
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        runner_mgr=SimpleNamespace(call_func=call_func),
        _collect_kv_transfer_progress=collect,
        _wake_event=event,
        label="test",
    )
    hook = load_wait.OffloadLoadWait(engine, 1, hold=True)
    hook.state.tick_s = 2.0
    if not stall:
        threading.Timer(0.03, future.set_result, (5,)).start()  # renewal finishes
    try:
        t0 = time.monotonic()
        hook.before_schedule()
        elapsed = time.monotonic() - t0
        assert not connector._reqs_need_recv
        return elapsed, shipped, reported, seq, hook.state
    finally:
        connector.close()


def test_renewal_finishing_mid_hold_wakes_it_and_ships_the_load(
    engine_core,
    aiter_arch,  # noqa: F811 (fixture)
    cluster,  # noqa: F811 (fixture)
    scheduler_clock,  # noqa: F811 (fixture)
    monkeypatch,
):
    """`ATOM_STORE_ASYNC_RENEW`: the renewal's completion rings the engine's
    doorbell, the hold ships the GET it released, and the GET's own landing
    ends the hold -- long before its bound."""
    elapsed, shipped, reported, seq, stats = _hold_for_a_renewal(
        engine_core, scheduler_clock, monkeypatch, stall=False
    )
    assert len(shipped) == 1  # the GET went out once, after the renewal
    assert reported == {seq.id}
    assert elapsed < 1.0  # ended by events, not the 2 s bound
    assert (stats.waits, stats.landed, stats.timed_out) == (1, 1, 0)
    assert stats.wakes >= 2  # the renewal, then the GET


def test_renewal_stalled_past_its_timeout_wakes_the_hold_and_ships_the_load(
    engine_core,
    aiter_arch,  # noqa: F811 (fixture)
    cluster,  # noqa: F811 (fixture)
    scheduler_clock,  # noqa: F811 (fixture)
    monkeypatch,
):
    """A renewal that never finishes rings nothing itself, but at its timeout
    the load may go out ("timeout"): the expiry rings the doorbell, the hold
    ships the GET and ends on its landing, not at the 2 s bound."""
    elapsed, shipped, reported, seq, stats = _hold_for_a_renewal(
        engine_core, scheduler_clock, monkeypatch, stall=True
    )
    assert len(shipped) == 1  # the GET went out at the renewal's timeout
    assert reported == {seq.id}
    assert 0.1 <= elapsed < 1.0
    assert (stats.waits, stats.landed, stats.timed_out) == (1, 1, 0)
    assert stats.wakes >= 2  # the renewal's expiry, then the GET
