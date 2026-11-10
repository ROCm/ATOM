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
    from atom.model_engine.engine_core import DPEngineCoreProc, EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc
from atom.model_engine.engine_utility import (
    PAUSE_GENERATION_CMD,
    RESUME_GENERATION_CMD,
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


def _run(engine, cmd, args=None):
    out = queue.Queue()
    handler = EngineUtilityHandler(None, out, label="test", scheduler=engine.scheduler)
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

    def sync_dp_state(local_unfinished, shutdown, held):
        reported.append((local_unfinished, held))
        return False, True, held  # everyone agrees to stop

    proc = SimpleNamespace(
        label="test",
        utility_queue=None,
        _is_rl_weights_offloaded=False,
        _weight_sync_fenced=False,
        _generation_paused=True,
        pull_and_process_input_queue=lambda: True,
        _sync_dp_state=sync_dp_state,
        _drain_kv_work_at_exit=lambda: None,
        scheduler=SimpleNamespace(
            heartbeat_throughput=lambda _: None,
            is_finished=lambda: False,  # requests are waiting
            publish_kv_events=lambda: None,
            shutdown_kv_events=lambda: None,
        ),
        utility_handler=SimpleNamespace(
            process_queue=lambda *args: None, push_metrics=lambda **kw: None
        ),
    )

    DPEngineCoreProc.busy_loop(proc)

    # It offers no work and asks to be held; the MAX reduction holds the group.
    assert reported == [(False, True)]


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
