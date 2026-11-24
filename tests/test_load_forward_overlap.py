# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load/forward overlap (`ATOM_OFFLOAD_PRESTART_LOAD_WAIT_MS`).

The early tier load of a waiting request runs ahead of `schedule()`; its load
is shipped at once and the engine waits a bounded time for it, so a request
whose Store restore lands in that window prefills in this step's forward
instead of one forward later. Off (the default) the step is unchanged.
"""

import queue
import sys
import threading
import time
from collections import deque
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from test_scheduler_early_load import META, _free, _make  # noqa: F401

from atom.model_engine.sequence import SequenceStatus


@pytest.fixture
def engine_core(monkeypatch):
    # Scheduling only, not the aiter-backed worker RPC.
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine import engine_core as module

    monkeypatch.setattr(module.tracing, "record_forward", Mock())
    return module


def _fire(monkeypatch, seq_factory, n=1, **env):
    """A scheduler whose delayer always fires (the P node runs without one)."""
    sched, delayer, seqs = _make(monkeypatch, seq_factory, n=n, **env)
    delayer.allow = True
    delayer.notify_prefill_executed = lambda *args: None
    # Metadata the engine treats as carrying work, so it is shipped.
    sched.kv_connector.build_connector_meta = lambda: SimpleNamespace(
        reqs_to_recv={"load": object()}
    )
    return sched, delayer, seqs


class _Workers:
    """Runner double. A load shipped to it lands once `land_after` polls have
    seen it (None: never); `fail` lands it as a failure."""

    def __init__(self, sched, land_after=1, fail=False):
        self.sched = sched
        self.land_after = land_after
        self.fail = fail
        self.calls = []
        self.forwards = []
        self.inflight = {}  # seq id -> polls seen

    def call_func(self, name, *args, **kwargs):
        self.calls.append(name)
        if name == "process_kvconnector_output":
            for seq in self.sched.waiting:
                if seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS:
                    self.inflight.setdefault(seq.id, 0)
            return None
        if name == "forward":
            batch = args[0]
            self.forwards.append(list(batch.req_ids))
            # A load in flight during the forward lands by its end.
            for seq_id in list(self.inflight):
                self._land(seq_id)
            return "fwd"
        raise AssertionError(name)

    def _land(self, seq_id):
        self.inflight.pop(seq_id)
        # As `_update_from_kv_xfer_finished`: an aborted load only cleans up.
        if self.sched._finish_aborted_load_cleanup(seq_id):
            return
        target = (
            self.sched.failed_recving_kv_req_ids
            if self.fail
            else self.sched.finished_recving_kv_req_ids
        )
        target.append(seq_id)

    def poll(self):
        self.calls.append("poll")
        for seq_id in list(self.inflight):
            self.inflight[seq_id] += 1
            if self.land_after is not None and self.inflight[seq_id] >= self.land_after:
                self._land(seq_id)
        return False


def _engine(module, sched, workers, wait_ms):
    engine = SimpleNamespace(
        scheduler=sched,
        kv_transfer_enabled=True,
        runner_mgr=SimpleNamespace(call_func=workers.call_func),
        _poll_kv_transfer_progress=workers.poll,
        _kv_progress_event=threading.Event(),
        input_queue=queue.Queue(),
        utility_queue=queue.Queue(),
        output_queue=queue.Queue(),
        _postprocess=Mock(),
        _release_request_resources=Mock(),
        _wait_for_kv_progress=lambda: None,
        label="test",
        _prestart_load_wait_s=wait_ms / 1000.0,
        _prestart_stats=module.PrestartLoadStats(),
    )
    engine._prestart_tier_loads = lambda: module.EngineCore._prestart_tier_loads(
        engine
    )
    return engine


def _steps_until_prefill(module, engine, workers, seq, limit=5):
    for step in range(1, limit + 1):
        module.EngineCore._process_engine_step_inner(engine)
        if workers.forwards and seq.id in workers.forwards[-1]:
            return step
    return None


# -- end to end through the engine step ------------------------------------


def test_restore_that_lands_in_the_window_prefills_in_this_steps_forward(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    workers = _Workers(sched, land_after=2)
    engine = _engine(engine_core, sched, workers, wait_ms=500)

    assert _steps_until_prefill(engine_core, engine, workers, seq) == 1

    # Shipped before the first poll, polled until it landed, then the forward.
    assert workers.calls[:3] == ["process_kvconnector_output", "poll", "poll"]
    assert workers.calls.index("forward") > 2
    assert seq.status == SequenceStatus.RUNNING
    assert sched.kv_connector.allocs == [seq.id]  # one admission only
    assert sched._num_parked_remote_kv == 0
    stats = engine._prestart_stats
    assert (stats.steps, stats.loads, stats.waits, stats.landed) == (1, 1, 1, 1)
    assert stats.timed_out == 0 and stats.skipped == 0


def test_off_the_same_restore_costs_one_extra_step(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    workers = _Workers(sched, land_after=1)
    engine = _engine(engine_core, sched, workers, wait_ms=0)

    # The load leaves with step 1's batch and is read only after it: prefill
    # waits for step 2 (on a busy rank, a whole forward later).
    assert _steps_until_prefill(engine_core, engine, workers, seq) == 2
    assert engine._prestart_stats.steps == 0


def test_off_step_is_unchanged(engine_core):
    # Wait 0 never reaches the pre-phase: a scheduler without it still runs.
    batch = SimpleNamespace(req_ids=["r"], connector_meta_output=None,
                            total_seqs_num_prefill=1)
    order = Mock()
    scheduler = SimpleNamespace(
        schedule=Mock(return_value=(batch, {})),
        take_rejected=Mock(return_value=[]),
        compute_detailed_aggregates=Mock(),
        metrics=SimpleNamespace(record_forward=Mock()),
        prefill_delayer=None,
        prestart_tier_loads=Mock(side_effect=AssertionError("pre-phase ran")),
    )
    order.attach_mock(scheduler.schedule, "schedule")
    runner = SimpleNamespace(call_func=Mock(return_value="fwd"))
    order.attach_mock(runner.call_func, "call_func")
    poll = Mock(return_value=False)
    order.attach_mock(poll, "poll")
    for engine in (
        SimpleNamespace(),  # an engine built before the knob existed
        SimpleNamespace(_prestart_load_wait_s=0.0),
    ):
        engine.__dict__.update(
            scheduler=scheduler,
            kv_transfer_enabled=True,
            runner_mgr=runner,
            _poll_kv_transfer_progress=poll,
            _kv_progress_event=threading.Event(),
            _postprocess=Mock(),
        )
        order.reset_mock()
        assert engine_core.EngineCore._process_engine_step_inner(engine) is True
        assert [c[0] for c in order.mock_calls] == ["schedule", "call_func", "poll"]
    scheduler.prestart_tier_loads.assert_not_called()


def test_default_is_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_OFFLOAD_PRESTART_LOAD_WAIT_MS", raising=False)
    assert envs.ATOM_OFFLOAD_PRESTART_LOAD_WAIT_MS == 0.0
    monkeypatch.setenv("ATOM_OFFLOAD_PRESTART_LOAD_WAIT_MS", "30")
    assert envs.ATOM_OFFLOAD_PRESTART_LOAD_WAIT_MS == 30.0
    monkeypatch.setenv("ATOM_OFFLOAD_PRESTART_LOAD_WAIT_MS", "-1")
    with pytest.raises(ValueError):
        envs.ATOM_OFFLOAD_PRESTART_LOAD_WAIT_MS


def test_wait_is_bounded_and_a_late_load_rides_the_forward_as_before(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    workers = _Workers(sched, land_after=None)  # lands only with a forward
    engine = _engine(engine_core, sched, workers, wait_ms=20)

    t0 = time.monotonic()
    engine_core.EngineCore._process_engine_step_inner(engine)
    elapsed = time.monotonic() - t0

    assert 0.02 <= elapsed < 1.0
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    stats = engine._prestart_stats
    assert (stats.waits, stats.landed, stats.timed_out) == (1, 0, 1)
    # Still in flight: no second wait for it (it is not new); it is read after
    # the step, as without the knob, and admitted by the step after that.
    workers.land_after = 1
    assert _steps_until_prefill(engine_core, engine, workers, seq) == 2
    assert engine._prestart_stats.waits == 1


def test_failed_restore_falls_back_to_prefill_in_the_same_step(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    workers = _Workers(sched, land_after=1, fail=True)
    engine = _engine(engine_core, sched, workers, wait_ms=500)

    step = _steps_until_prefill(engine_core, engine, workers, seq)

    assert step == 1  # recomputed, not left parked
    assert engine._prestart_stats.landed == 1
    assert sched._num_parked_remote_kv == 0
    assert seq.id not in sched._early_load.entries


def test_abort_while_waiting_releases_blocks(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    free = _free(sched)
    workers = _Workers(sched, land_after=None)
    engine = _engine(engine_core, sched, workers, wait_ms=5)
    engine_core.EngineCore._process_engine_step_inner(engine)
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS

    assert sched.abort_request(seq.id)
    # The existing aborted-load protocol: blocks go back once the load reports.
    assert _free(sched) < free
    workers.land_after = 1
    engine_core.EngineCore._process_engine_step_inner(engine)
    assert _free(sched) == free
    assert sched._num_parked_remote_kv == 0
    assert not workers.forwards or seq.id not in workers.forwards[-1]


def test_no_wait_when_the_budget_ahead_is_full(engine_core, monkeypatch, seq_factory):
    sched, _, seqs = _fire(monkeypatch, seq_factory, n=6)
    *ahead, last = seqs
    # 5 x 16 tokens ahead of it fill the 64-token budget.
    sched.kv_connector.not_ready_ids = {s.id for s in ahead}
    workers = _Workers(sched, land_after=None)
    engine = _engine(engine_core, sched, workers, wait_ms=500)

    t0 = time.monotonic()
    engine_core.EngineCore._process_engine_step_inner(engine)

    assert time.monotonic() - t0 < 0.4
    assert last.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    stats = engine._prestart_stats
    assert (stats.steps, stats.skipped, stats.waits) == (1, 1, 0)
    assert workers.calls[0] == "process_kvconnector_output"  # still shipped early
    assert workers.calls.count("poll") <= 1  # only the step's own poll


def test_nothing_to_start_ships_and_polls_nothing(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    sched.kv_connector.not_ready_ids = {seq.id}  # lookup not answered
    workers = _Workers(sched)
    engine = _engine(engine_core, sched, workers, wait_ms=500)

    engine_core.EngineCore._process_engine_step_inner(engine)

    assert workers.calls.count("poll") <= 1  # only the step's own poll
    assert engine._prestart_stats.steps == 0  # (Phase 2 may still park it, as before)


# -- scheduler half -----------------------------------------------------------


def test_prestart_parks_like_the_tick_and_runs_no_collective(monkeypatch, seq_factory):
    sched, delayer, (seq,) = _make(monkeypatch, seq_factory)

    started = sched.prestart_tier_loads()

    assert started == [seq]
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert list(sched.waiting) == [seq]  # keeps its place
    assert delayer.calls == 0  # no cross-DP all_reduce off the lockstep path
    # The tick's own early load finds nothing new to do.
    batch, scheduled = sched.schedule()
    assert scheduled == {}
    assert sched.kv_connector.allocs == [seq.id]
    assert sched._early_load.submitted == 1


def test_prestart_then_landed_load_is_admitted_by_this_schedule(
    monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    sched.prestart_tier_loads()
    sched.finished_recving_kv_req_ids.append(seq.id)

    batch, scheduled = sched.schedule()

    assert seq.id in scheduled and batch.total_seqs_num_prefill == 1
    assert seq.id not in sched._early_load.entries


def test_prestart_keeps_fifo_among_ready_requests(monkeypatch, seq_factory):
    sched, _, (a, b) = _fire(monkeypatch, seq_factory, n=2)
    assert sched.prestart_tier_loads() == [a, b]
    sched.finished_recving_kv_req_ids.extend([b.id, a.id])  # b landed first

    _batch, scheduled = sched.schedule()

    assert list(scheduled) == [a.id, b.id]  # arrival order, not landing order


def test_a_load_still_in_flight_does_not_hold_back_one_that_landed(
    monkeypatch, seq_factory
):
    sched, _, (a, b) = _fire(monkeypatch, seq_factory, n=2)
    assert sched.prestart_tier_loads() == [a, b]
    sched.finished_recving_kv_req_ids.append(b.id)

    _batch, scheduled = sched.schedule()

    assert list(scheduled) == [b.id]
    assert a.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert list(sched.waiting) == [a]


def test_prestart_without_an_offload_tier_does_nothing(monkeypatch, seq_factory):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    sched.kv_connector.supports_early_load = False
    assert sched.prestart_tier_loads() == []
    sched.kv_connector = None
    assert sched.prestart_tier_loads() == []
    assert seq.status == SequenceStatus.WAITING


def test_wait_gate(monkeypatch, seq_factory):
    sched, _, (a, b) = _fire(monkeypatch, seq_factory, n=2)
    sched.kv_connector.not_ready_ids = {a.id}
    started = sched.prestart_tier_loads()
    assert started == [b]
    assert sched.prestart_wait_worthwhile(started)  # 16 tokens ahead of 64
    assert not sched.prestart_wait_worthwhile([])
    sched.running = deque([object()] * sched.max_num_seqs)
    assert not sched.prestart_wait_worthwhile(started)
