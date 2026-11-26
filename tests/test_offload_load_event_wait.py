# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Event-driven load wait (`ATOM_OFFLOAD_LOAD_EVENT_WAIT`).

The early tier load runs ahead of `schedule()`, its loads are shipped at once,
and the step is held for them only while holding pays. The hold blocks on the
KV progress doorbell a worker rings when a load job ends, not on a sleep, and
ends as soon as every started load landed; its bound comes from the measured
forward time and the requests the hold would delay, not from a constant.
"""

import pickle
import queue
import socket
import sys
import threading
import time
from collections import deque
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from test_mooncake_store_offload import (  # noqa: F401 (fixtures)
    aiter_arch,
    cluster,
    scheduler_clock,
)
from test_scheduler_early_load import _free, _make

from atom.kv_transfer import progress_doorbell
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
    forward) and rings the doorbell, as `LoadExecutor` does. A completion
    answer is a snapshot taken when it is requested, as a worker answers the
    query when it reaches it, and arrives at once (setting the engine's KV
    progress event); `collect` closes the outstanding one and requests the next
    (`AsyncIOProcManager.call_func_with_aggregation`'s pipelining). A forward
    lands whatever is still in flight by its end.
    """

    def __init__(self, sched, progress_event, delay=0.0, fail=False, forward_s=0.0):
        self.sched = sched
        self.progress_event = progress_event
        self.delay = delay  # seconds, None, or {seq id: seconds | None}
        self.fail = fail
        self.forward_s = forward_s
        self.kv_doorbell_event = threading.Event()
        self.lock = threading.Lock()
        self.shipped, self.inflight, self.landed = set(), set(), set()
        self.pending = None  # (requested at, snapshot)
        self.calls, self.forwards, self.timers = [], [], []

    @property
    def kv_aggregation_started_at(self):
        return None if self.pending is None else self.pending[0]

    def _delay(self, seq_id):
        return self.delay.get(seq_id) if isinstance(self.delay, dict) else self.delay

    def land(self, seq_id):
        with self.lock:
            if seq_id not in self.inflight:
                return
            self.inflight.discard(seq_id)
            self.landed.add(seq_id)
        self.kv_doorbell_event.set()

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


def _engine(module, sched, *, on=True, tick_s=1.0, dp=1, **mgr_kwargs):
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
        _load_event_wait=on,
        _load_wait=module.LoadWaitState(dp),
    )
    engine._load_wait.tick_s = tick_s
    for name in (
        "_prestart_tier_loads_and_wait",
        "_ship_connector_meta",
        "_wait_for_loads",
    ):
        setattr(engine, name, MethodType(getattr(module.EngineCore, name), engine))
    return engine, mgr


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
    stats = engine._load_wait
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
    assert engine._load_wait.wakes == 1


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
    stats = engine._load_wait
    assert (stats.waits, stats.landed, stats.timed_out, stats.wakes) == (1, 0, 1, 0)
    # It lands with the forward and is admitted later, as without the hold;
    # it is not held for again (not a new load).
    assert _steps_until_prefill(engine_core, engine, mgr, seq) is not None
    assert engine._load_wait.waits == 1


def test_failed_load_wakes_the_hold_and_falls_back_to_prefill(
    engine_core, monkeypatch, seq_factory
):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, delay=0.02, fail=True)

    t0 = time.monotonic()
    assert _steps_until_prefill(engine_core, engine, mgr, seq) == 1  # recomputed
    assert time.monotonic() - t0 < 1.0
    assert engine._load_wait.landed == 1
    assert sched._num_parked_remote_kv == 0
    assert seq.id not in sched._early_load.entries


def test_without_the_hold_the_same_load_prefills_later(
    engine_core, monkeypatch, seq_factory
):
    # Off: the load lands during step 1's forward and is read only after it.
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, on=False, delay=0)
    assert _steps_until_prefill(engine_core, engine, mgr, seq) >= 2
    assert engine._load_wait.steps == 0


def test_cold_engine_never_holds(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=0.0, delay=None)

    t0 = time.monotonic()
    _step(engine_core, engine)

    assert time.monotonic() - t0 < 0.5
    stats = engine._load_wait
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
    assert engine._load_wait.landed == 2
    assert engine._load_wait.wakes >= 1


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
    stats = engine._load_wait
    assert (stats.landed, stats.timed_out) == (1, 1)


# -- the bound: gain against cost, no constant ----------------------------------


def _partial_prefill(remaining=8):
    return SimpleNamespace(
        is_partial_prefill=True, num_tokens=remaining, num_cached_tokens=0
    )


def _decode():
    return SimpleNamespace(is_partial_prefill=False)


def test_budget_formula(engine_core):
    state = engine_core.LoadWaitState(dp_size=1)
    assert state.budget_s(1, 0) == 0.0  # no forward measured yet
    state.tick_s = 0.2
    assert state.budget_s(0, 0) == 0.0  # nothing to gain
    assert state.budget_s(1, 0) == pytest.approx(0.2)  # nothing to delay: 1 tick
    assert state.budget_s(3, 1) == pytest.approx(0.2)  # never past one tick
    assert state.budget_s(1, 4) == pytest.approx(0.05)
    dp = engine_core.LoadWaitState(dp_size=4)
    dp.tick_s = 0.2
    assert dp.budget_s(1, 4) == pytest.approx(0.0125)  # peers pay too
    assert dp.budget_s(1, 0) == pytest.approx(0.05)


def test_tradeoff_on_a_prefill_only_step(monkeypatch, seq_factory):
    sched, _, (a, b, c) = _fire(monkeypatch, seq_factory, n=3)
    sched.kv_connector.not_ready_ids = {a.id}  # a stays an ordinary waiter
    started = sched.prestart_tier_loads()
    assert started == [b, c]
    # a (16 tokens) ahead of b and c: both still fit the 64-token budget, and
    # the step carries a anyway.
    assert sched.load_wait_tradeoff(started) == (2, 1)
    sched.running = deque([_partial_prefill()])
    sched._partial_prefill_count = 1
    assert sched.load_wait_tradeoff(started) == (2, 2)
    # A load that landed or was aborted is not a gain.
    c.status = SequenceStatus.ABORTED
    assert sched.load_wait_tradeoff(started) == (1, 2)
    assert sched.load_wait_tradeoff([]) == (0, 1)


def test_tradeoff_stops_at_the_token_budget_and_slots(monkeypatch, seq_factory):
    sched, _, seqs = _fire(monkeypatch, seq_factory, n=6)
    *ahead, last = seqs
    sched.kv_connector.not_ready_ids = {s.id for s in ahead}
    started = sched.prestart_tier_loads()
    assert started == [last]
    # 5 x 16 tokens ahead fill the 64-token budget: nothing to gain.
    assert sched.load_wait_tradeoff(started)[0] == 0
    sched2, _, (seq,) = _fire(monkeypatch, seq_factory)
    started = sched2.prestart_tier_loads()
    sched2.running = deque(_partial_prefill(1) for _ in range(sched2.max_num_seqs))
    sched2._partial_prefill_count = sched2.max_num_seqs
    assert sched2.load_wait_tradeoff(started)[0] == 0  # no free slot


def test_running_decodes_shrink_the_bound_to_almost_nothing(
    engine_core, monkeypatch, seq_factory
):
    # The same rule, no special case: every running decode is delayed by a
    # hold, so the bound falls with the decode batch.
    sched, _, _ = _fire(monkeypatch, seq_factory)
    started = sched.prestart_tier_loads()
    sched.running = deque(_decode() for _ in range(6))
    gain, cost = sched.load_wait_tradeoff(started)
    assert (gain, cost) == (1, 6)
    state = engine_core.LoadWaitState(dp_size=4)
    state.tick_s = 0.012  # a decode step
    assert state.budget_s(gain, cost) == pytest.approx(0.0005)


def test_no_pre_phase_load_while_a_decode_is_running(monkeypatch, seq_factory):
    # Ordering, not tuning: the pre-phase allocates, and a running decode's
    # page reservation is installed only inside `_schedule()`.
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    sched.running = deque([_decode()])
    free = _free(sched)
    assert sched.prestart_tier_loads() == []
    assert sched.kv_connector.allocs == [] and _free(sched) == free
    assert seq.status == SequenceStatus.WAITING
    assert not sched._early_load_prestarted  # `_schedule()` runs its own pass
    sched.running = deque([_partial_prefill(), _decode()])
    sched._partial_prefill_count = 1
    assert sched.prestart_tier_loads() == []
    # Partial prefills only (the P/D prefill case): the pre-phase runs.
    sched.running = deque([_partial_prefill()])
    assert sched.prestart_tier_loads() == [seq]


def test_no_decode_prefill_only_step_holds(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, delay=0.03)
    assert not sched._has_running_decode()
    _step(engine_core, engine)
    assert mgr.forwards == [[seq.id]]
    assert engine._load_wait.waits == 1


def test_forward_time_is_the_tick(engine_core, monkeypatch, seq_factory):
    sched, _, (seq,) = _fire(monkeypatch, seq_factory)
    engine, mgr = _engine(engine_core, sched, tick_s=2.0, delay=0, forward_s=0.03)
    _step(engine_core, engine)
    assert 0.03 <= engine._load_wait.tick_s < 0.5


# -- off is main, one early-load pass per step ------------------------------------


def test_off_step_is_unchanged(engine_core):
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
        prestart_tier_loads=Mock(side_effect=AssertionError("pre-phase ran")),
    )
    order.attach_mock(scheduler.schedule, "schedule")
    runner = SimpleNamespace(call_func=Mock(return_value="fwd"))
    order.attach_mock(runner.call_func, "call_func")
    poll = Mock(return_value=False)
    order.attach_mock(poll, "poll")
    engine = SimpleNamespace(
        _load_event_wait=False,
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


def test_manager_turns_the_doorbell_into_an_event_not_an_answer():
    import zmq
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.kv_transfer.disaggregation import KVConnectorOutput
        from atom.model_engine.async_proc import AsyncIOProcManager
        from atom.kv_transfer.progress_doorbell import KV_PROGRESS_DOORBELL

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    addr = f"tcp://127.0.0.1:{port}"
    mgr = AsyncIOProcManager.__new__(AsyncIOProcManager)
    mgr.label = "test"
    mgr.zmq_ctx = zmq.Context()
    mgr.still_running = True
    mgr.kv_outputs_queues = [queue.Queue()]
    mgr.kv_doorbell_event = threading.Event()
    mgr.kv_progress_event = threading.Event()
    thread = threading.Thread(
        target=mgr.process_kv_output_sockets, args=(addr, 0), daemon=True
    )
    thread.start()
    push = mgr.zmq_ctx.socket(zmq.PUSH)
    push.connect(addr)
    try:
        push.send(pickle.dumps(KV_PROGRESS_DOORBELL))
        assert mgr.kv_doorbell_event.wait(5)
        assert mgr.kv_progress_event.is_set()
        assert mgr.kv_outputs_queues[0].empty()  # never joins an aggregation
        push.send(pickle.dumps(KVConnectorOutput()))
        assert isinstance(mgr.kv_outputs_queues[0].get(timeout=5), KVConnectorOutput)
    finally:
        mgr.still_running = False
        push.close(linger=0)
        thread.join(timeout=3)
        mgr.zmq_ctx.term()


def test_aggregation_start_time_is_exposed():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.async_proc import AsyncIOProcManager

    mgr = AsyncIOProcManager.__new__(AsyncIOProcManager)
    mgr.label = "test"
    mgr.proc_num = 1
    mgr.rpc_broadcast_mq = Mock()
    mgr.kv_output_aggregator = None
    mgr.kv_outputs_queues = [queue.Queue()]
    mgr._pending_kv_aggregation = None
    assert mgr.kv_aggregation_started_at is None
    t0 = time.monotonic()
    assert mgr.call_func_with_aggregation("async_proc_aggregation", timeout=0) is None
    assert mgr.kv_aggregation_started_at >= t0


# -- a load parked on its Store lease renewal -----------------------------------------


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
    from concurrent.futures import Future

    from test_mooncake_store_offload import (
        _arm_load,
        _config,
        _install_future,
        _seq,
    )

    from atom.kv_transfer.offload.mooncake_store.scheduler import (
        MooncakeStoreOffloadScheduler,
    )

    monkeypatch.setenv("ATOM_STORE_ASYNC_RENEW", "1")
    connector = MooncakeStoreOffloadScheduler(_config("kv_consumer"))
    seq = _seq(59, 50)
    _arm_load(connector, seq, hbm=16, lmc=40)
    scheduler_clock.now += 2.0  # old enough to need a renewal
    future = Future()
    _install_future(connector, scheduler_clock, monkeypatch, future, [])

    event = threading.Event()
    doorbell = threading.Event()
    progress_doorbell.install(doorbell.set, coalesce=False)
    shipped, worker_landed, reported = [], set(), set()
    pending = []

    def land():
        worker_landed.add(seq.id)
        doorbell.set()

    def call_func(name, meta):
        assert name == "process_kvconnector_output"
        shipped.extend(r.load_operation for r in meta.requests)
        threading.Timer(0.02, land).start()  # the GET lands 20 ms later

    def collect():
        answer = None
        if pending:
            answer = pending.pop()[1]
            reported.update(answer)
        pending.append((time.monotonic(), set(worker_landed)))
        event.set()
        return answer

    class _RunnerMgr(SimpleNamespace):
        @property
        def kv_aggregation_started_at(self):
            return pending[0][0] if pending else None

    scheduler = SimpleNamespace(
        kv_connector=connector,
        prestart_tier_loads=lambda: [seq],
        load_wait_tradeoff=lambda started: (1, 0),
        _still_loading=lambda s: s.id not in reported,
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        runner_mgr=_RunnerMgr(call_func=call_func, kv_doorbell_event=doorbell),
        _collect_kv_transfer_progress=collect,
        _wake_event=event,
        _load_wait=engine_core.LoadWaitState(1),
        label="test",
    )
    engine._load_wait.tick_s = 2.0
    for name in ("_ship_connector_meta", "_wait_for_loads"):
        setattr(engine, name, MethodType(getattr(engine_core.EngineCore, name), engine))
    threading.Timer(0.03, future.set_result, (5,)).start()  # renewal finishes
    try:
        t0 = time.monotonic()
        engine_core.EngineCore._prestart_tier_loads_and_wait(engine)
        elapsed = time.monotonic() - t0

        assert len(shipped) == 1  # the GET went out once, after the renewal
        assert reported == {seq.id}
        assert elapsed < 1.0  # ended by events, not the 2 s bound
        stats = engine._load_wait
        assert (stats.waits, stats.landed, stats.timed_out) == (1, 1, 0)
        assert stats.wakes >= 2  # the renewal, then the GET
        assert not connector._reqs_need_recv
    finally:
        connector.close()
