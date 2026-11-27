# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pausing generation around a weight update, and resuming after it.

A pause only stops scheduling; what runs waits. The weight update that follows
preempts it (see test_kv_change.py), and after the resume it recomputes its
context on the new weights.
"""

import queue
from collections import deque
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

with stubbed_aiter():
    from atom.model_engine import llm_engine
    from atom.model_engine.engine_core import DPEngineCoreProc, EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc
from atom.model_engine.collective_rpc import RpcResult
from atom.model_engine.engine_utility import (
    PAUSE_GENERATION_CMD,
    RESUME_GENERATION_CMD,
    WEIGHTS_IN_USE_CMD,
    WEIGHTS_STOPPED_CMD,
    EngineUtilityHandler,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.rollout.async_engine import AsyncLLMEngine
from atom.sampling_params import SamplingParams


def _running(n):
    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    seqs = [
        Sequence([10 + i, 11, 12, 13, 14], 4, SamplingParams(max_tokens=40))
        for i in range(n)
    ]
    for seq in seqs:
        sched.add(seq)
    batch, active = sched.schedule()
    sched.postprocess(
        list(active.values()),
        ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True),
        batch=batch,
    )
    return sched, seqs


def _engine(scheduler, cls=EngineCore):
    engine = cls.__new__(cls)
    engine.scheduler = scheduler
    engine.kv_transfer_enabled = False
    engine._next_idle_kv_drain = 0.0
    engine._has_pending_utility = True
    engine._is_rl_weights_offloaded = False
    engine._weight_sync_fenced = False
    engine._rl_weights_inconsistent = False
    engine._generation_paused = False
    return engine


def _run(engine, cmd, args=None, runners=None):
    out = queue.Queue()
    handler = EngineUtilityHandler(
        runners, out, label="test", scheduler=engine.scheduler
    )
    pending = queue.Queue()
    pending.put_nowait((cmd, {} if args is None else args))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)
    kind, reply = out.get_nowait()
    assert kind == "UTILITY_RESPONSE"
    assert out.empty(), "one answer per command"
    return reply


# ── the commands ───────────────────────────────────────────────────────────


def test_pause_stops_scheduling_and_leaves_what_runs():
    sched, seqs = _running(2)
    engine = _engine(sched)

    reply = _run(engine, PAUSE_GENERATION_CMD, {"mode": "keep"})

    assert reply == {"cmd": PAUSE_GENERATION_CMD}
    assert engine._generation_paused
    assert list(sched.running) == seqs


def test_pause_defaults_to_keep_and_refuses_any_other_mode():
    engine = _engine(Scheduler(MockConfig(num_kvcache_blocks=64)))

    assert "error" not in _run(engine, PAUSE_GENERATION_CMD)
    engine._generation_paused = False

    reply = _run(engine, PAUSE_GENERATION_CMD, {"mode": "abort"})
    assert "unsupported pause mode" in reply["error"]
    assert not engine._generation_paused


def test_resume_lets_the_engine_schedule_again():
    engine = _engine(Scheduler(MockConfig(num_kvcache_blocks=64)))
    _run(engine, PAUSE_GENERATION_CMD)

    reply = _run(engine, RESUME_GENERATION_CMD)

    assert reply == {"cmd": RESUME_GENERATION_CMD}
    assert not engine._generation_paused


def test_paused_update_resumed_recomputes_on_the_new_weights():
    sched, seqs = _running(2)
    engine = _engine(sched)
    handler_runs = []

    _run(engine, PAUSE_GENERATION_CMD)
    out = queue.Queue()
    handler = EngineUtilityHandler(None, out, label="test", scheduler=sched)
    handler._execute_utility_command = lambda cmd, args: handler_runs.append(cmd)
    pending = queue.Queue()
    pending.put_nowait(("update_weights", {"sync_id": None}))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)
    _run(engine, RESUME_GENERATION_CMD)

    assert handler_runs == ["update_weights"]
    batch, active = sched.schedule()
    assert batch.total_seqs_num_prefill == 2
    assert sorted(active) == sorted(seq.id for seq in seqs)


def test_stopped_weights_hold_the_engine_until_they_are_in_use_again():
    # A weight cache daemon's `stop` and `load`, for the engine that maps its
    # weights.
    sched, seqs = _running(2)
    engine = _engine(sched)
    asked = []

    def utility_rpc(method, payload, timeout):
        asked.append((method, payload.args))
        return [RpcResult(payload.request_id, 0)]

    runners = SimpleNamespace(utility_rpc=utility_rpc)
    reply = _run(engine, WEIGHTS_STOPPED_CMD, {"rewrite": True}, runners)

    assert "error" not in reply and engine._generation_paused
    # Every rank let what it had sent to its device finish.
    assert asked == [("await_forwards", (True,))]
    # Unchanged, what was running goes on where it was...
    _run(engine, WEIGHTS_IN_USE_CMD, {"changed": False})
    assert not engine._generation_paused and list(sched.running) == seqs
    # ...and on other values every rank derives again what it had made of the
    # weights, and it recomputes its context.
    _run(engine, WEIGHTS_STOPPED_CMD, {}, runners)
    _run(engine, WEIGHTS_IN_USE_CMD, {"changed": True}, runners)
    assert asked[-1] == ("weights_changed", ()) and not engine._generation_paused
    batch, active = sched.schedule()
    assert batch.total_seqs_num_prefill == 2
    assert sorted(active) == sorted(seq.id for seq in seqs)


def _answering_runners():
    return SimpleNamespace(
        utility_rpc=lambda method, payload, timeout: [RpcResult(payload.request_id, 0)]
    )


def test_a_stop_ends_every_request_the_engine_has_and_later_ones_wait():
    sched, seqs = _running(2)
    queued = Sequence([20, 21, 22], 4, SamplingParams(max_tokens=40))
    sched.add(queued)
    engine = _engine(sched)
    engine.output_queue = queue.Queue()
    why = "server_abort: other weights are loaded"

    reply = _run(engine, WEIGHTS_STOPPED_CMD, {"abort": why}, _answering_runners())

    # Two ran and one waited; each is answered now, while the engine is held.
    assert reply["aborted"] == (2, 1) and engine._generation_paused
    ended = engine.output_queue.get_nowait()
    assert {seq.id for seq in ended} == {seq.id for seq in (*seqs, queued)}
    assert all(seq.leave_reason == why for seq in ended)
    assert not sched.running and not sched.waiting
    # One that comes during the stop waits, and runs on the weights after it.
    later = Sequence([30, 31, 32], 4, SamplingParams(max_tokens=40))
    sched.add(later)
    _run(engine, WEIGHTS_IN_USE_CMD, {"changed": True}, _answering_runners())
    _, active = sched.schedule()
    assert list(active) == [later.id]


def test_a_stop_that_cannot_end_what_runs_is_refused_and_ends_nothing():
    sched, seqs = _running(2)
    engine = _engine(sched)

    def held(**kw):
        raise RuntimeError("KV transfers are still in flight")

    engine.prepare_kv_change = held
    reply = _run(
        engine, WEIGHTS_STOPPED_CMD, {"abort": "server_abort: x"}, _answering_runners()
    )

    assert "still in flight" in reply["error"] and list(sched.running) == seqs
    # Without a reason no request ends: a launch asking whether it could stop.
    del engine.prepare_kv_change
    reply = _run(engine, WEIGHTS_STOPPED_CMD, {"rewrite": True}, _answering_runners())
    assert "aborted" not in reply and list(sched.running) == seqs


def test_a_launch_tells_its_engines_why_and_logs_how_many_requests_ended(monkeypatch):
    sent, logged = [], []

    def tell(cmd, **kw):
        sent.append((cmd, kw))
        return [{"aborted": (2, 1)}, {"aborted": (0, 3)}]  # two DP engines

    llm = SimpleNamespace(core_mgr=SimpleNamespace(broadcast_utility_command_sync=tell))
    monkeypatch.setattr(llm_engine.logger, "warning", logged.append)
    llm_engine.LLMEngine._weights_stopped(llm, False, "other weights are loaded")

    reason = "server_abort: other weights are loaded"
    assert sent == [(WEIGHTS_STOPPED_CMD, {"rewrite": False, "abort": reason})]
    assert "ended 6 request(s) (2 running, 4 waiting)" in logged[0]
    # Only asked whether it could stop: no request ends, nothing to say.
    llm_engine.LLMEngine._weights_stopped(llm, True)
    assert sent[-1][1] == {"rewrite": True, "abort": None} and len(logged) == 1


def test_a_rank_that_cannot_follow_other_weights_leaves_the_engine_stopped():
    engine = _engine(Scheduler(MockConfig(num_kvcache_blocks=64)))

    def utility_rpc(method, payload, timeout):
        error = "kept the old ones" if method == "weights_changed" else None
        return [RpcResult(payload.request_id, 0, error=error)]

    runners = SimpleNamespace(utility_rpc=utility_rpc)
    _run(engine, WEIGHTS_STOPPED_CMD, {"rewrite": True}, runners)
    reply = _run(engine, WEIGHTS_IN_USE_CMD, {"changed": True}, runners)

    assert "kept the old ones" in reply["error"] and engine._generation_paused


# ── the busy loops ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "flag",
    ["_is_rl_weights_offloaded", "_weight_sync_fenced", "_generation_paused"],
)
def test_each_flag_alone_holds_the_engine(flag):
    engine = _engine(Scheduler(MockConfig(num_kvcache_blocks=64)))
    assert not engine._is_engine_held()

    setattr(engine, flag, True)

    assert engine._is_engine_held()


@pytest.mark.parametrize("offloaded, reported", [(False, [False]), (True, [])])
def test_a_held_engine_reports_kv_transfers_back_but_starts_none(offloaded, reported):
    engine = _engine(Scheduler(MockConfig(num_kvcache_blocks=64)))
    engine._is_rl_weights_offloaded = offloaded
    engine.has_pending_kv_work = lambda: True
    calls = []
    engine._advance_idle_kv_transfer = lambda dispatch_new=True: calls.append(
        dispatch_new
    )

    engine._report_held_kv_work()

    assert calls == reported, "nothing runs while the KV pool may be released"


def test_one_held_dp_rank_holds_every_rank():
    reported = []
    closed = []

    def sync_dp_state(local_unfinished, shutdown, held, hidden=False):
        reported.append((local_unfinished, held))
        return False, True, held  # everyone agrees to stop

    proc = SimpleNamespace(
        label="test",
        utility_queue=None,
        _is_rl_weights_offloaded=False,
        _weight_sync_fenced=False,
        _generation_paused=True,
        _report_backlog=False,
        _idle_when_blocked=False,
        _dp_group_hidden_work=False,
        pull_and_process_input_queue=lambda: True,
        _sync_dp_state=sync_dp_state,
        _validate_dp_sync_interval=lambda interval: interval,
        _drain_kv_work_at_exit=lambda: None,
        scheduler=SimpleNamespace(
            heartbeat_throughput=lambda _: None,
            is_finished=lambda: False,  # requests are waiting
            publish_kv_events=lambda: None,
            shutdown_kv_events=lambda: None,
            close_kv_connector=lambda: closed.append(True),
        ),
        utility_handler=SimpleNamespace(
            process_queue=lambda *args: None, push_metrics=lambda **kw: None
        ),
    )

    DPEngineCoreProc.busy_loop(proc)

    # It offers no work and asks to be held; the MAX reduction holds the group.
    assert reported == [(False, True)]
    assert closed == [True]


# ── pipeline parallelism ───────────────────────────────────────────────────


def _pp_engine(is_head, in_flight, completes):
    engine = _engine(Scheduler(MockConfig(num_kvcache_blocks=64)), PPEngineCoreProc)
    engine.is_head = is_head
    engine._in_flight = deque(in_flight)

    def collect(poll_ms):
        if completes and engine._in_flight:
            engine._in_flight.popleft()

    engine._collect_pp_completions = collect
    return engine


def test_the_pp_head_pauses_once_its_batches_are_retired():
    engine = _pp_engine(True, ["batch 1", "middle chunk"], completes=True)

    reply = _run(engine, PAUSE_GENERATION_CMD)

    assert "error" not in reply and engine._generation_paused
    assert not engine._in_flight


def test_the_pp_head_refuses_a_pause_its_pipeline_does_not_empty_for():
    engine = _pp_engine(True, ["middle chunk"], completes=False)

    reply = _run(engine, PAUSE_GENERATION_CMD, {"timeout": 0.05})

    assert "still in the pipeline" in reply["error"]
    assert not engine._generation_paused, "paused or not, never half"


def test_a_later_pp_stage_keeps_forwarding_what_the_head_sent():
    engine = _pp_engine(False, ["batch 1"], completes=False)

    reply = _run(engine, PAUSE_GENERATION_CMD)

    assert "error" not in reply
    assert not engine._generation_paused and not engine._is_engine_held()


# ── the API ────────────────────────────────────────────────────────────────


def test_async_llm_engine_pauses_and_resumes_every_engine():
    sent = []
    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.core_mgr = SimpleNamespace(
        broadcast_utility_command_sync=lambda cmd, **kw: sent.append((cmd, kw))
    )

    engine.pause_generation()
    engine.resume_generation()

    assert [cmd for cmd, _ in sent] == [PAUSE_GENERATION_CMD, RESUME_GENERATION_CMD]
    assert sent[0][1]["mode"] == "keep"
