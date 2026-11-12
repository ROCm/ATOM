# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A command that zeroes, frees or reallocates the KV pool, or commits new
weights, first preempts what is running and afterwards drops the prefix-cache
index. Refused, it runs nothing and leaves what was running as it was.
"""

import contextlib
import pickle
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

with stubbed_aiter():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc
from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.model_engine.collective_rpc import RpcResponseRouter, RpcResult
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.engine_utility import EngineUtilityHandler, kv_change
from atom.model_engine.scheduler import (
    DecodeScheduler,
    PrefillScheduler,
    ScheduledBatchOutput,
    Scheduler,
)
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams


def _running(n, scheduler_cls=Scheduler):
    """A scheduler with *n* sequences past their prefill, decoding."""
    sched = scheduler_cls(MockConfig(num_kvcache_blocks=64))
    seqs = [
        Sequence([10 + i, 11, 12, 13, 14], 4, SamplingParams(max_tokens=40))
        for i in range(n)
    ]
    for seq in seqs:
        sched.add(seq)
    if n:
        batch, active = sched.schedule()
        sched.postprocess(
            list(active.values()),
            ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True),
            batch=batch,
        )
    assert list(sched.running) == seqs
    return sched, seqs


def _engine(scheduler, cls=EngineCore):
    engine = cls.__new__(cls)
    engine.label = "test"
    engine.scheduler = scheduler
    engine.kv_transfer_enabled = False
    engine._next_idle_kv_drain = 0.0
    engine._has_pending_utility = True
    engine._is_rl_weights_offloaded = False
    engine._weight_sync_fenced = False
    engine._rl_weights_inconsistent = False
    engine._generation_paused = False
    return engine


def _run(engine, cmd, args=None):
    """Drain one command through process_queue; return (reply, commands run).

    Only the runner side is stood in for: what reaches
    ``_execute_utility_command`` is recorded and answered.
    """
    out = queue.Queue()
    handler = EngineUtilityHandler(None, out, label="test", scheduler=engine.scheduler)
    ran = []

    def execute(cmd, args):
        ran.append(cmd)
        reply = {"cmd": cmd}
        handler.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    handler._execute_utility_command = execute
    pending = queue.Queue()
    pending.put_nowait((cmd, {} if args is None else args))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)
    kind, reply = out.get_nowait()
    assert kind == "UTILITY_RESPONSE"
    return reply, ran


def _count_index_drops(monkeypatch, sched):
    drops = []
    monkeypatch.setattr(sched.block_manager, "clear_cache", lambda: drops.append(1))
    return drops


# ── what counts as a KV change ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "cmd, args, change",
    [
        ("update_weights", {"sync_id": None}, "weights"),
        ("update_weights_shm", {"sync_id": None, "is_last": True}, "weights"),
        ("update_weights_ipc", {"sync_id": None}, "weights"),
        ("update_weights_shm", {"sync_id": None, "is_last": False}, None),
        ("clear_kv_cache", {}, "kv"),
        ("release_memory", {"tags": ["weights", "kv_cache"]}, "kv"),
        ("release_memory", {}, "kv"),  # no tags: both, as the runner reads it
        ("resume_memory", {"tags": ["kv_cache"]}, "kv"),
        ("release_memory", {"tags": ["weights"]}, None),
        ("start_profile", {}, None),
    ],
)
def test_what_changes_the_kv_pool(cmd, args, change):
    assert kv_change(cmd, args) == change


# ── a KV change preempts first, then drops the index ───────────────────────


@pytest.mark.parametrize(
    "cmd, args",
    [("update_weights", {"sync_id": None}), ("clear_kv_cache", {})],
)
def test_a_kv_change_preempts_what_runs_and_drops_the_index(monkeypatch, cmd, args):
    sched, seqs = _running(3)
    drops = _count_index_drops(monkeypatch, sched)

    reply, ran = _run(_engine(sched), cmd, args)

    assert "error" not in reply and ran == [cmd]
    assert not sched.running
    assert list(sched.waiting)[:3] == seqs, "they resume in the order they ran"
    assert drops == [1]


def test_preempted_sequences_recompute_their_context_when_scheduled_again():
    sched, seqs = _running(2)
    _run(_engine(sched), "update_weights", {"sync_id": None})

    batch, active = sched.schedule()

    assert batch.total_seqs_num_prefill == 2
    assert sorted(active) == sorted(seq.id for seq in seqs)
    for i, seq in enumerate(seqs):
        assert list(seq.token_ids) == [10 + i, 11, 12, 13, 14]


def test_a_non_last_bucket_leaves_what_runs(monkeypatch):
    sched, seqs = _running(2)
    drops = _count_index_drops(monkeypatch, sched)

    _run(_engine(sched), "update_weights_shm", {"sync_id": None, "is_last": False})

    assert list(sched.running) == seqs
    assert drops == []


# ── refused, nothing runs and nothing moves ────────────────────────────────


def _assert_refused(monkeypatch, sched, seqs, engine, match, cmd="update_weights"):
    drops = _count_index_drops(monkeypatch, sched)

    reply, ran = _run(engine, cmd, {"sync_id": None, "timeout": 0.05})

    assert match in reply["error"] and reply["error"].startswith("refused")
    assert ran == [], "a refused change runs nothing"
    assert list(sched.running) == seqs, "and preempts none"
    assert drops == []


def test_refused_while_a_connector_holds_a_running_sequence(monkeypatch):
    sched, seqs = _running(3)
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: seq is not seqs[1])

    _assert_refused(monkeypatch, sched, seqs, _engine(sched), "a connector holds")


def test_a_refused_direct_update_leaves_the_engine_serving(monkeypatch):
    sched, _ = _running(2)
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: False)
    engine = _engine(sched)

    reply, ran = _run(engine, "update_weights", {"sync_id": None})

    assert reply["error"].startswith("refused") and ran == []
    assert not engine._weight_sync_fenced, "it wrote nothing, so nothing to fence"


def test_a_refused_last_bucket_fails_its_sync(monkeypatch):
    sched, seqs = _running(2)
    engine = _engine(sched)
    out = queue.Queue()
    handler = EngineUtilityHandler(None, out, label="test", scheduler=sched)
    handler._execute_utility_command = lambda cmd, args: {"cmd": cmd}
    pending = queue.Queue()
    pending.put_nowait(("update_weights_shm", {"sync_id": "s1", "is_last": False}))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: False)

    pending.put_nowait(("update_weights_shm", {"sync_id": "s1", "is_last": True}))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)

    assert "refused" in out.get_nowait()[1]["error"]
    assert handler._weight_sync_failure, "its first bucket was written"
    assert engine._weight_sync_fenced and list(sched.running) == seqs


def test_refused_while_a_token_is_in_the_pipeline(monkeypatch):
    sched, seqs = _running(2)
    sched._pp_inflight_token_block.add(seqs[0].id)

    _assert_refused(monkeypatch, sched, seqs, _engine(sched), "in the pipeline")


def test_refused_while_remote_kv_is_parked(monkeypatch):
    sched, seqs = _running(2)
    sched._num_parked_remote_kv = 1

    _assert_refused(monkeypatch, sched, seqs, _engine(sched), "wait for remote KV")


def test_refused_while_kv_transfers_stay_in_flight(monkeypatch):
    sched, seqs = _running(2)
    engine = _engine(sched)
    polled = []
    engine.has_pending_kv_work = lambda: True
    engine._advance_idle_kv_transfer = lambda dispatch_new=True: polled.append(
        dispatch_new
    )

    _assert_refused(monkeypatch, sched, seqs, engine, "still in flight")
    assert polled and not any(polled), "reported back, nothing new started"


def test_waits_for_kv_transfers_to_report_back(monkeypatch):
    sched, _ = _running(2)
    engine = _engine(sched)
    pending = [True, True, False]
    engine.has_pending_kv_work = lambda: pending.pop(0)
    engine._advance_idle_kv_transfer = lambda dispatch_new=True: None

    _, ran = _run(engine, "clear_kv_cache", {"timeout": 5})

    assert ran == ["clear_kv_cache"] and not sched.running


@pytest.mark.parametrize("scheduler_cls", [DecodeScheduler, PrefillScheduler])
def test_a_disaggregated_engine_refuses_with_requests_in_flight(
    monkeypatch, scheduler_cls
):
    # A decode engine's prompts were prefilled by the prefill engine: preempted,
    # a request would wait for a PrefillDone that is not coming again.
    sched = scheduler_cls(MockConfig(num_kvcache_blocks=64))
    sched.add(Sequence([1, 2, 3, 4], 4, SamplingParams(max_tokens=8)))

    reply, ran = _run(_engine(sched), "clear_kv_cache", {"timeout": 0.05})

    assert "cannot recompute" in reply["error"] and ran == []


@pytest.mark.parametrize("scheduler_cls", [DecodeScheduler, PrefillScheduler])
def test_a_disaggregated_engine_with_nothing_in_flight_changes(scheduler_cls):
    sched = scheduler_cls(MockConfig(num_kvcache_blocks=64))

    _, ran = _run(_engine(sched), "clear_kv_cache")

    assert ran == ["clear_kv_cache"]


# ── an offload tier must drop what it keeps ────────────────────────────────


def test_an_offload_tier_cannot_drop_its_store_until_it_says_so():
    assert not KVConnectorSchedulerBase.drop_cached_kv(SimpleNamespace(is_offload=True))
    assert KVConnectorSchedulerBase.drop_cached_kv(SimpleNamespace(is_offload=False))


def test_a_weight_change_is_refused_with_an_offload_tier_a_kv_change_is_not():
    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    sched.kv_connector = SimpleNamespace(drop_cached_kv=lambda: False)

    reply, ran = _run(_engine(sched), "update_weights", {"sync_id": None})
    assert "cannot drop KV computed with the old weights" in reply["error"]
    assert ran == []

    reply, ran = _run(_engine(sched), "clear_kv_cache")
    assert ran == ["clear_kv_cache"], "the weights stay, so its KV does too"


# ── through CoreManager: a refusal fences only what wrote ──────────────────


class _Runners:
    """The TP ranks of one engine, each applying whatever it is sent."""

    def utility_rpc(self, method, payload, timeout=300.0):
        return [RpcResult(payload.request_id, 0, value=1)]


def _manager(engine_count):
    mgr = object.__new__(CoreManager)
    mgr.label = "test-mgr"
    mgr.control_sockets = [object() for _ in range(engine_count)]
    mgr._rpc_router = RpcResponseRouter()
    mgr._weight_sync_lock = threading.RLock()
    mgr._weight_sync_context = threading.local()
    mgr._weight_sync_dispatch_token = object()
    mgr._rpc_ranks_per_engine = 1
    return mgr


def _wired(schedulers):
    """A CoreManager whose DP engines are real utility handlers over
    *schedulers*, each control message delivered and answered in order."""
    mgr = _manager(len(schedulers))
    engines = [_engine(sched) for sched in schedulers]
    outs = [queue.Queue() for _ in schedulers]
    handlers = [
        EngineUtilityHandler(_Runners(), out, label=f"dp{i}", scheduler=sched)
        for i, (out, sched) in enumerate(zip(outs, schedulers))
    ]

    def send(dp_rank, payload, copy=False):
        _, body = pickle.loads(payload)
        body = dict(body)
        pending = queue.Queue()
        pending.put_nowait((body.pop("cmd"), body))
        engines[dp_rank]._has_pending_utility = True
        handlers[dp_rank].process_queue(pending, engines[dp_rank])
        while not outs[dp_rank].empty():
            kind, reply = outs[dp_rank].get_nowait()
            if kind == "UTILITY_RESPONSE":
                mgr._route_utility_response(dp_rank, reply)

    mgr._send_control = send
    return mgr, engines


def _update(mgr):
    mgr.broadcast_utility_command_sync("update_weights", named_tensors=[], timeout=5)


def test_a_refused_update_fences_no_engine(monkeypatch):
    """CoreManager answers the refusal with DISCARD on every engine. None of
    them wrote, so all of them keep serving the weights they hold."""
    schedulers = [_running(2)[0], _running(2)[0]]
    for sched in schedulers:
        monkeypatch.setattr(sched, "_is_preemptable", lambda seq: False)
    mgr, engines = _wired(schedulers)

    with pytest.raises(RuntimeError, match="refused"):
        _update(mgr)

    assert not any(e._weight_sync_fenced or e._rl_weights_inconsistent for e in engines)


def test_a_peer_that_wrote_stays_fenced_until_the_next_sync(monkeypatch):
    refusing, writing = _running(2)[0], _running(2)[0]
    monkeypatch.setattr(refusing, "_is_preemptable", lambda seq: False)
    mgr, (refused, wrote) = _wired([refusing, writing])

    with pytest.raises(RuntimeError, match="refused"):
        _update(mgr)

    assert not refused._weight_sync_fenced
    assert wrote._weight_sync_fenced and wrote._rl_weights_inconsistent

    monkeypatch.setattr(refusing, "_is_preemptable", lambda seq: True)
    _update(mgr)

    assert not any(e._weight_sync_fenced for e in (refused, wrote))


# ── a pipeline: the head and a later stage running its own loop ────────────


class _Wire:
    """The PP transport between a head and its one later stage, in memory."""

    def __init__(self):
        self.metadata = queue.Queue()
        self.completions = queue.Queue()

    def send_metadata(self, batch):
        self.metadata.put(batch)

    def recv_metadata(self, timeout_ms):
        try:
            return self.metadata.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None

    def send_completion(self, req_ids, fwd_out=None):
        self.completions.put((list(req_ids), fwd_out))

    def recv_completion(self, timeout_ms):
        try:
            return self.completions.get(timeout=timeout_ms / 1000)
        except queue.Empty:
            return None


class _StageRunners(_Runners):
    """A later stage's TP ranks, which also run the forwards they are sent."""

    def call_func(self, name, *args, wait_out=False):
        return None


class _Replies:
    """A later stage's output queue: its replies go straight to *mgr*."""

    def __init__(self, mgr, engine_index):
        self.mgr, self.engine_index = mgr, engine_index

    def put_nowait(self, item):
        if item[0] == "UTILITY_RESPONSE":
            self.mgr._route_utility_response(self.engine_index, item[1])


def _middle_chunk(req_id):
    """A prefill chunk in flight whose acknowledgement the head waits for."""
    return SimpleNamespace(
        req_ids=[req_id],
        connector_meta_output=None,
        pp_completion_ack=True,
        total_seqs_num_decode=0,
        is_final_chunk=[False],
        produces_output=lambda: False,
    )


def _launch(head, wire, req_id):
    """What `_pp_head_step` leaves behind for one batch it sent down."""
    batch = _middle_chunk(req_id)
    wire.send_metadata(batch)
    head._in_flight.append((batch, {}, False))


def _bucket(mgr, *, is_last):
    mgr.broadcast_utility_command_sync(
        "update_weights_shm", shm_name="s", bucket_meta={}, is_last=is_last, timeout=5
    )


@contextlib.contextmanager
def _pipeline(sched, *, stage_runs=True):
    """A PP=2 engine under a real CoreManager: the head's commands are run in
    the caller's thread; stage 1 runs `_downstream_busy_loop` on its own,
    fenced or held as its real utility handler leaves it."""
    mgr = _manager(2)
    mgr.pp_size = 2
    wire = _Wire()
    head = _engine(sched, PPEngineCoreProc)
    stage = _engine(_running(0)[0], PPEngineCoreProc)
    for engine, is_head in ((head, True), (stage, False)):
        engine.is_head, engine.is_last = is_head, not is_head
        engine.pp_size = 2
        engine.pp_transport = wire
        engine._pp_kv_aggregator = None
        engine._has_pending_utility = False
    head._in_flight = deque()
    head._defer_prefix_hash = False
    head.runner_mgr = _StageRunners()
    head.output_queue = queue.Queue()
    head._poll_kv_transfer_progress = lambda: None
    head._poll_dynamic_chunking_calibration = lambda: None
    head._advance_idle_kv_transfer = lambda dispatch_new=True: None
    head_out = queue.Queue()
    head_handler = EngineUtilityHandler(_Runners(), head_out, scheduler=sched)
    stage.runner_mgr = _StageRunners()
    stage.utility_handler = EngineUtilityHandler(
        stage.runner_mgr, _Replies(mgr, 1), label="stage1", scheduler=stage.scheduler
    )
    stage.utility_handler.push_metrics = lambda **_: None
    stage.utility_queue = queue.Queue()
    stage.received = []
    stop = threading.Event()
    stage.pull_and_process_input_queue = stop.is_set

    def send(engine_index, payload, copy=False):
        _, body = pickle.loads(payload)
        body = dict(body)
        cmd = body.pop("cmd")
        if engine_index == 1:
            stage.received.append(cmd)
            stage.utility_queue.put_nowait((cmd, body))
            stage._has_pending_utility = True
            return
        pending = queue.Queue()
        pending.put_nowait((cmd, body))
        head._has_pending_utility = True
        head_handler.process_queue(pending, head)
        while not head_out.empty():
            kind, reply = head_out.get_nowait()
            if kind == "UTILITY_RESPONSE":
                mgr._route_utility_response(0, reply)

    mgr._send_control = send
    errors = []

    def run_stage():
        try:
            stage._downstream_busy_loop()
        except BaseException as exc:  # noqa: BLE001 - reported by the test
            errors.append(exc)

    thread = threading.Thread(target=run_stage, daemon=True)
    stage.run = thread.start
    if stage_runs:
        stage.run()
    try:
        yield mgr, head, stage, wire
    finally:
        stop.set()
        if thread.is_alive():
            thread.join(timeout=5)
    assert not errors


def _prefilling():
    """A PP head's scheduler with one prompt that takes several chunks."""
    sched = Scheduler(
        MockConfig(
            num_kvcache_blocks=64,
            max_model_len=256,
            max_num_batched_tokens=16,
            pipeline_parallel_size=2,
        )
    )
    sched.add(Sequence(list(range(10, 110)), 4, SamplingParams(max_tokens=4)))
    return sched


def test_the_pp_head_waits_for_middle_chunks_the_last_stage_has_not_run():
    """A middle chunk has no output, so only its ack shows the head that the
    last stage has run it. Retired on sending, the chunks left the head's
    pipeline looking empty while stage 1 had yet to run them."""
    with _pipeline(_prefilling(), stage_runs=False) as (_, head, stage, wire):
        head._pp_head_step()
        assert wire.metadata.qsize() == 2

        with pytest.raises(RuntimeError, match="2 batch"):
            head._drain_pipeline(time.monotonic() + 0.2)

        stage.run()
        head._drain_pipeline(time.monotonic() + 5)
        assert not head._in_flight and wire.metadata.empty()


def test_a_later_stage_fenced_by_a_sync_forwards_nothing():
    """Why the head must empty the pipeline before stage 1 hears of a sync:
    once fenced, stage 1 forwards nothing, and what the head sent stays."""
    with (
        _pipeline(_running(2)[0]) as (mgr, head, stage, wire),
        mgr.weight_sync_session(),
    ):
        _bucket(mgr, is_last=False)
        assert stage._weight_sync_fenced
        _launch(head, wire, req_id=7)

        with pytest.raises(RuntimeError, match="still in the pipeline"):
            head._drain_pipeline(time.monotonic() + 0.3)
        assert wire.metadata.qsize() == 1


def test_a_pp_sync_empties_the_pipeline_before_the_later_stage_fences():
    sched, _ = _running(2)
    with _pipeline(sched) as (mgr, head, stage, wire):
        _launch(head, wire, req_id=7)
        _launch(head, wire, req_id=8)
        with mgr.weight_sync_session():
            _bucket(mgr, is_last=False)
            assert not head._in_flight
            assert stage._weight_sync_fenced
            _bucket(mgr, is_last=True)

    assert not (head._weight_sync_fenced or stage._weight_sync_fenced)
    assert not sched.running
    assert stage.received == [
        "update_weights_shm",
        "update_weights_shm",
        "finish_weight_sync",
    ]


def test_the_pp_head_empties_the_pipeline_before_a_kv_change():
    sched, _ = _running(2)
    with _pipeline(sched) as (mgr, head, stage, wire):
        _launch(head, wire, req_id=7)

        mgr.broadcast_utility_command_sync("clear_kv_cache", timeout=5)

        assert not head._in_flight and not sched.running
        assert stage.received == ["clear_kv_cache"]


def test_a_command_the_pp_head_refuses_reaches_no_later_stage():
    """Stage 1 forwards nothing here, so the head cannot empty its pipeline
    and refuses. Stage 1 never hears of the update and writes nothing."""
    sched, seqs = _running(2)
    with _pipeline(sched, stage_runs=False) as (mgr, head, stage, wire):
        _launch(head, wire, req_id=7)

        with pytest.raises(RuntimeError, match="still in the pipeline"):
            mgr.broadcast_utility_command_sync(
                "update_weights", named_tensors=[], timeout=1
            )

        assert stage.received == ["discard_failed_weight_sync"]
        assert list(sched.running) == seqs
        assert not any(
            e._weight_sync_fenced or e._rl_weights_inconsistent for e in (head, stage)
        )


def test_a_pause_the_pp_head_refuses_reaches_no_later_stage():
    """A later stage that paused would hold, and forward nothing the head sent."""
    sched, _ = _running(2)
    with _pipeline(sched, stage_runs=False) as (mgr, head, stage, wire):
        _launch(head, wire, req_id=7)

        with pytest.raises(RuntimeError, match="still in the pipeline"):
            mgr.broadcast_utility_command_sync(
                "pause_generation", mode="keep", timeout=1
            )

        assert stage.received == []
        assert not (head._generation_paused or stage._generation_paused)


def test_a_command_that_leaves_weights_and_kv_alone_reaches_every_stage():
    """It went to the head first like a weight update, so one the head
    failed never reached the later stages, as a profile start would not."""
    sched, _ = _running(2)
    with _pipeline(sched) as (mgr, _head, stage, _wire):
        with pytest.raises(RuntimeError, match="unknown utility command"):
            mgr.broadcast_utility_command_sync("no_such_command", timeout=5)

        assert stage.received == ["no_such_command"]


def test_the_held_check_does_not_sleep(monkeypatch):
    """A predicate that slept blocked every extra caller for a tick."""
    slept = []
    monkeypatch.setattr(time, "sleep", slept.append)
    engine = _engine(_running(0)[0])
    engine._weight_sync_fenced = True

    assert engine._is_engine_held()
    assert slept == []
