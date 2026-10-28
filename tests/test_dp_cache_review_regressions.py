# SPDX-License-Identifier: MIT
"""Regressions for placement math, composite recovery and engine event ordering."""

import pickle
import queue
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig
from test_dp_cache_balance import AdmissionHarness, chain
from test_dp_cache_integration import manager, request
from test_dp_load_balance import _route
from test_multi_connector import FakeSchedSub, _sched

from atom.model_engine.dp_cache.balance import (
    CacheBalanceConfig,
    CacheBalancer,
    Placement,
    PrefixHint,
    Session,
)
from atom.model_engine.dp_cache.state import DPCacheState
from atom.model_engine.engine_core_mgr import _add_frame
from atom.model_engine.engine_core_protocol import (
    EngineCoreAbortRequest,
    EngineCoreRequestType,
)
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus


def imbalance(counts, contexts, prompt, rank):
    counts, contexts = list(counts), list(contexts)
    counts[rank] += 1
    contexts[rank] += prompt
    mean_n, mean_c = sum(counts) / len(counts), sum(contexts) / len(contexts)
    return sum((n / mean_n) ** 2 + (c / mean_c) ** 2 for n, c in zip(counts, contexts))


@pytest.mark.parametrize(
    "lengths", [[[115200], [4224, 4224, 4352]], [[80000, 4000], [], [3000], [32, 64]]]
)
@pytest.mark.parametrize("prompt", [1, 17920, 262144])
def test_placement_score_orders_actual_post_placement_imbalance(lengths, prompt):
    balancer = CacheBalancer(len(lengths), CacheBalanceConfig())
    for rank, requests in enumerate(lengths):
        for i, tokens in enumerate(requests):
            balancer.requests[rank, i] = Placement(rank, tokens, None, None)
    counts, contexts = [len(xs) for xs in lengths], [sum(xs) for xs in lengths]
    scores = balancer._decode_placement_scores(prompt, counts)
    actual = [imbalance(counts, contexts, prompt, r) for r in range(len(lengths))]
    for source in range(len(lengths)):
        for target in range(len(lengths)):
            assert (scores[source] <= scores[target]) == (
                actual[source] <= actual[target]
            )


def test_review_migration_must_not_increase_post_placement_imbalance():
    balancer = CacheBalancer(2, CacheBalanceConfig(checkpoint_tokens=128))
    for i, (rank, tokens) in enumerate([(0, 115200), (1, 4224), (1, 4224), (1, 4352)]):
        balancer.requests[i] = Placement(rank, tokens, None, None)
    balancer.sessions["session"] = Session(
        owner=1, hints={1: PrefixHint(16384, 128, -1, 127, 1)}
    )
    assert imbalance([1, 3], [115200, 12800], 17920, 0) > imbalance(
        [1, 3], [115200, 12800], 17920, 1
    )
    placement = balancer.place(
        "move",
        "session",
        17920,
        chain(17920),
        [1, 3],
        [0, 1000000],
        2,
        can_migrate=True,
    )
    assert (placement.rank, placement.migration) == (1, 0)
    assert balancer.counters["migration_decode_balance_rejected"] == 1


@pytest.mark.parametrize("winner_has_handoff", [True, False])
def test_multi_alignment_uses_load_winner_and_never_a_losing_tier(winner_has_handoff):
    loser = FakeSchedSub(match=(0, False), offload_methods=True)
    loser.pending_prefill_handoff = lambda seq: pytest.fail("losing tier was consulted")
    winner = FakeSchedSub(match=(102400, True))
    if winner_has_handoff:
        winner.pending_prefill_handoff = lambda seq: (65536, 102400)
    multi = _sched([loser, winner])
    host = AdmissionHarness()
    host.kv_connector = multi
    seq = NS(
        id=7,
        num_tokens=103424,
        num_cached_tokens=64000,
        dp_cache=DPCacheState(migration_id=7, minimum_hit=102400),
    )
    assert multi.get_num_new_matched_tokens(seq) == (102400, True)
    assert host.cache_admission.admit(seq) is winner_has_handoff
    if winner_has_handoff:
        assert seq.dp_cache.alignment_boundary == 65536
        assert seq.dp_cache.retry_reason is None
        assert not seq.dp_cache.admission_sent
        assert not host.rejected and not host.events
    else:
        assert seq.dp_cache.retry_reason == "recoverable_prefix_too_short"
    multi.cancel_pending_load(seq)
    assert multi.pending_prefill_handoff(seq) is None


def engine(scheduler, kind="EngineCore"):
    with stubbed_aiter():
        from atom.model_engine import engine_core
    core = getattr(engine_core, kind).__new__(getattr(engine_core, kind))
    core.scheduler, core.label = scheduler, "review-regression"
    core.kv_transfer_enabled = False
    core._next_idle_kv_drain = 0.0
    core.input_queue, core.utility_queue = queue.Queue(), queue.Queue()
    core.output_queue, core.stream_output_queue = queue.Queue(), queue.Queue()
    core._has_pending_utility = False
    return core, engine_core


def test_fallback_add_then_abort_during_forward_cannot_resurrect_request(monkeypatch):
    mgr = manager(monkeypatch)
    seq = request()
    _route(mgr, [seq])
    returned = pickle.loads(_add_frame([seq]))[1][0]
    returned.dp_cache.retry_reason = "preparation_timeout"
    frames = []
    mgr.control_sockets = [None, None]
    mgr.label = "review-manager"
    mgr._send_control = lambda rank, payload, **kwargs: (
        frames.append(payload) if rank == 0 else None
    )
    assert mgr.cache_router.retry(1, returned)
    mgr.abort_request(seq.id)
    assert seq.id not in mgr._seq_load
    assert len(frames) == 2

    scheduler = Scheduler(MockConfig())
    core, module = engine(scheduler)
    core.utility_handler = EngineUtilityHandler(
        None, core.output_queue, scheduler=scheduler
    )
    feed = iter([*frames, pickle.dumps((EngineCoreRequestType.SHUTDOWN, None))])
    sock = NS(send=lambda _: None, recv=lambda **kwargs: next(feed))
    monkeypatch.setattr(
        module, "make_zmq_socket", lambda *args, **kwargs: nullcontext(sock)
    )
    monkeypatch.setattr(
        module.zmq,
        "Poller",
        lambda: NS(register=lambda *args: None, poll=lambda: [(sock, 1)]),
    )
    # IO receives both while the engine is busy; its next tick drains utilities first.
    core.process_input_sockets("input", "control")
    # SHUTDOWN only stops the fake IO feed. Remove its sentinel before the
    # engine tick so it cannot hide a resurrected ADD by dropping the batch.
    incoming = []
    while not core.input_queue.empty():
        incoming.append(core.input_queue.get_nowait())
    [sentinel] = incoming.pop()
    assert sentinel.status == SequenceStatus.EXIT_ENGINE
    for event in incoming:
        core.input_queue.put_nowait(event)
    core.utility_handler.process_queue(core.utility_queue, core)
    assert not core.pull_and_process_input_queue()
    assert not scheduler.waiting and not scheduler.running
    [rejected] = scheduler.take_rejected()
    assert rejected.id == seq.id and rejected.leave_reason == "aborted"
    assert not scheduler.cache_admission._preparing


def test_ordered_abort_supports_prefill_scheduler_without_abort_method():
    # RapidServe's PrefillScheduler relies on the utility handler's status
    # fallback. An ordered abort must keep supporting that scheduler surface.
    scheduler = NS(waiting=deque(), running=[])
    scheduler.extend = scheduler.waiting.extend
    core, _ = engine(scheduler, "PrefillEngineCore")
    core.utility_handler = EngineUtilityHandler(
        None, core.output_queue, scheduler=scheduler
    )
    seq = Sequence([1, 2, 3], 16)
    core.input_queue.put_nowait([seq])
    core.input_queue.put_nowait(EngineCoreAbortRequest(seq.id))
    assert not core.pull_and_process_input_queue()
    assert list(scheduler.waiting) == [seq]
    assert seq.status == SequenceStatus.ABORTED


@pytest.mark.parametrize("kind", ["EngineCore", "DPEngineCoreProc"])
def test_last_cancelled_partial_prefill_finishes_after_one_safe_dummy(kind):
    scheduler = Scheduler(
        MockConfig(
            kv_cache_block_size=4,
            num_kvcache_blocks=128,
            max_model_len=256,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
        )
    )
    seq = Sequence(list(range(65)), 4)
    scheduler.add(seq)
    core, _ = engine(scheduler, kind)
    calls = []

    def forward(method, *args, **kwargs):
        calls.append(method)
        if method == "dummy_execution":
            assert kwargs["wait_out"]
            assert seq.status == SequenceStatus.ABORTED and seq.block_table
        return ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True)

    core.runner_mgr = NS(call_func=forward)
    assert core._process_engine_step()
    assert seq.is_partial_prefill and seq.block_table
    assert scheduler.abort_request(seq.id)
    # True tells DP's busy loop that its collective tick already executed.
    assert core._process_engine_step()
    assert calls == ["forward", "dummy_execution"]
    assert not scheduler.running and not seq.block_table
    assert scheduler._partial_prefill_count == 0
    assert core.output_queue.get_nowait() == [seq]


@pytest.mark.parametrize("path", ["engine_step", "dummy"])
def test_preempted_work_is_reported_with_no_running_sequences(monkeypatch, path):
    scheduler = Scheduler(MockConfig())
    seq = Sequence(list(range(65)), scheduler.block_manager.block_size)
    seq.dp_cache = DPCacheState(admission_sent=True, last_work=0)
    assert scheduler.preempt(seq)
    assert not scheduler.running and list(scheduler.waiting) == [seq]
    # The next schedule is held by remote recovery; no forward/postprocess follows.
    monkeypatch.setattr(scheduler, "schedule", lambda: (None, {}))
    core, _ = engine(scheduler, "DPEngineCoreProc")
    core.runner_mgr = NS(call_func=lambda *args, **kwargs: None)
    if path == "engine_step":
        assert not core._process_engine_step()
    else:
        core._execute_dummy_batch()
    assert core.output_queue.get_nowait() == ("CACHE_WORK", [(seq.id, 0, 1, 65)])
    assert scheduler.cache_admission.take_work() == []
