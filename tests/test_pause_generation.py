# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pausing generation around a weight update, and resuming after it.

An RL trainer that updates the rollout's weights while requests are in flight
(vLLM's ``pause_generation(mode="keep")``) needs the engine to stop scheduling
for the update, and the running sequences to recompute their context on the new
weights afterwards, since the update's commit clears the KV cache.
"""

import queue
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

with stubbed_aiter():
    from atom.model_engine.engine_core import DPEngineCoreProc, EngineCore
from atom.model_engine.engine_utility import (
    GENERATION_PAUSE_MODES,
    PAUSE_GENERATION_CMD,
    RESUME_GENERATION_CMD,
    EngineUtilityHandler,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

# ── the scheduler ──────────────────────────────────────────────────────────


def _running(n):
    """A scheduler with *n* sequences past their prefill, decoding."""
    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    free_before = sched.block_manager.kv.num_free
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
    assert list(sched.running) == seqs
    return sched, seqs, free_before


def test_preempt_running_preempts_every_running_sequence_in_order():
    sched, seqs, free_before = _running(3)

    assert sched.preempt_running() == 3

    assert not sched.running
    assert list(sched.waiting)[:3] == seqs, "they resume in the order they ran"
    assert sched.block_manager.kv.num_free == free_before
    for i, seq in enumerate(seqs):
        # The deferred-output placeholder goes; the context stays.
        assert list(seq.token_ids) == [10 + i, 11, 12, 13, 14]


def test_preempted_sequences_recompute_their_context_when_scheduled_again():
    sched, seqs, _ = _running(2)
    sched.preempt_running()

    batch, active = sched.schedule()

    assert batch.total_seqs_num_prefill == 2
    assert sorted(active) == sorted(seq.id for seq in seqs)


def test_preempt_running_refuses_when_a_connector_holds_a_sequence(monkeypatch):
    sched, seqs, _ = _running(3)
    held = seqs[1]
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: seq is not held)

    with pytest.raises(RuntimeError, match="a connector holds"):
        sched.preempt_running()

    assert list(sched.running) == seqs, "all or nothing"


def test_preempt_running_refuses_while_a_token_is_in_the_pipeline():
    sched, seqs, _ = _running(2)
    sched._pp_inflight_token_block.add(seqs[0].id)

    with pytest.raises(RuntimeError, match="in the pipeline"):
        sched.preempt_running()

    assert list(sched.running) == seqs


def test_preempt_running_with_nothing_running_is_a_no_op():
    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    assert sched.preempt_running() == 0


# ── the utility commands ───────────────────────────────────────────────────


class _BlockManager:
    def __init__(self):
        self.cleared = 0

    def clear_cache(self):
        self.cleared += 1


class _Scheduler:
    def __init__(self, preempted=3, raises=None):
        self.block_manager = _BlockManager()
        self.calls = 0
        self._preempted = preempted
        self._raises = raises

    def preempt_running(self):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._preempted


class _Engine:
    _has_pending_utility = True
    _is_rl_weights_offloaded = False
    _weight_sync_fenced = False
    _rl_weights_inconsistent = False
    _generation_paused = False


def _run(scheduler, engine, cmd, args=None):
    out = queue.Queue()
    handler = EngineUtilityHandler(None, out, label="test", scheduler=scheduler)
    engine._has_pending_utility = True
    pending = queue.Queue()
    pending.put_nowait((cmd, {} if args is None else args))
    handler.process_queue(pending, engine)
    kind, reply = out.get_nowait()
    assert kind == "UTILITY_RESPONSE"
    assert out.empty(), "one answer per command"
    return reply


def test_pause_keeps_requests_and_stops_the_engine():
    scheduler, engine = _Scheduler(preempted=3), _Engine()

    reply = _run(scheduler, engine, PAUSE_GENERATION_CMD, {"mode": "keep"})

    assert reply == {"cmd": PAUSE_GENERATION_CMD, "preempted": 3}
    assert engine._generation_paused
    assert scheduler.block_manager.cleared == 1, "no reuse of old-weight blocks"


def test_pause_defaults_to_keep():
    assert GENERATION_PAUSE_MODES == ("keep",)
    scheduler, engine = _Scheduler(), _Engine()

    reply = _run(scheduler, engine, PAUSE_GENERATION_CMD)

    assert "error" not in reply
    assert engine._generation_paused


def test_pause_can_leave_the_prefix_cache():
    scheduler, engine = _Scheduler(), _Engine()

    _run(scheduler, engine, PAUSE_GENERATION_CMD, {"clear_cache": False})

    assert scheduler.block_manager.cleared == 0


def test_resume_lets_the_engine_schedule_again():
    scheduler, engine = _Scheduler(), _Engine()
    _run(scheduler, engine, PAUSE_GENERATION_CMD)

    reply = _run(scheduler, engine, RESUME_GENERATION_CMD)

    assert reply == {"cmd": RESUME_GENERATION_CMD}
    assert not engine._generation_paused


@pytest.mark.parametrize(
    "scheduler, args, message",
    [
        (_Scheduler(), {"mode": "abort"}, "unsupported pause mode"),
        (_Scheduler(raises=RuntimeError("connector holds 1")), {}, "connector"),
        (object(), {}, "cannot preempt to pause"),
    ],
)
def test_a_refused_pause_answers_the_error_and_keeps_serving(scheduler, args, message):
    engine = _Engine()

    reply = _run(scheduler, engine, PAUSE_GENERATION_CMD, args)

    assert reply["cmd"] == PAUSE_GENERATION_CMD
    assert message in reply["error"]
    assert not engine._generation_paused


# ── the busy loop ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "flag",
    ["_is_rl_weights_offloaded", "_weight_sync_fenced", "_generation_paused"],
)
def test_the_busy_loop_skips_the_forward_while_any_flag_is_set(flag):
    engine = EngineCore.__new__(EngineCore)
    engine._is_rl_weights_offloaded = False
    engine._weight_sync_fenced = False
    assert not engine._generation_paused, "unpaused unless told"
    assert not engine._is_idle_rl_weights_offloaded()

    setattr(engine, flag, True)

    assert engine._is_idle_rl_weights_offloaded()


def test_a_paused_dp_rank_tells_its_peers_it_has_no_work():
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

    assert reported == [(False, True)]
