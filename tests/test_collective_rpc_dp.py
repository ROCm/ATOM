# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DP half: ``CoreManager.collective_rpc`` and request-id routing.

``broadcast_utility_command_sync`` reads a fixed count of replies off one shared
queue, so it matches by position. Two overlapping callers take each other's
replies, and a late reply from an abandoned call becomes the next caller's --
upstream's own ``push_metrics`` docstring records that biting the old pull-based
metrics. These tests pin the correlated replacement, and pin that every other
utility command still uses the legacy queue.
"""

import queue
import threading
import time
from contextlib import ExitStack

import pytest

from atom.model_engine.collective_rpc import (
    COLLECTIVE_RPC_CMD,
    RpcResponseRouter,
    RpcResult,
)
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager

# ── harness ────────────────────────────────────────────────────────────────


class _Socket:
    closed = False


def _mgr(engine_count=2, tp=1, cls=CoreManager):
    """A ``CoreManager`` with only what the RPC path reads.

    ``__init__`` spawns engine processes and binds sockets, so it cannot run
    here; the fan-out and correlation logic is what is under test.
    """
    mgr = object.__new__(cls)
    mgr.label = "test-mgr"
    mgr.control_sockets = [_Socket() for _ in range(engine_count)]
    mgr.utility_response_queue = queue.Queue()
    mgr._rpc_router = RpcResponseRouter()
    mgr._rpc_ranks_per_engine = tp
    mgr.sent = []
    mgr.broadcast_utility_command = lambda cmd, **kw: mgr.sent.append((cmd, kw))
    return mgr


def _tp_body(request_id, method="m", results=None, error=None):
    body = {"cmd": COLLECTIVE_RPC_CMD, "request_id": request_id, "method": method}
    if error is not None:
        body["error"] = error
    else:
        body["results"] = results or []
    return body


def _tp(rank, value=None, error=None):
    return {"tp_rank": rank, "value": value, "error": error}


def _answer_after(mgr, delay, dp_rank, body):
    """Deliver a reply from a thread, the way the output thread really does."""

    def run():
        time.sleep(delay)
        mgr._route_utility_response(dp_rank, body)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# ── the router ─────────────────────────────────────────────────────────────


def test_the_router_isolates_concurrent_ids():
    r = RpcResponseRouter()
    with r.register("a") as qa, r.register("b") as qb:
        assert r.route("a", 1) is True
        assert r.route("b", 2) is True
        assert qa.get_nowait() == 1
        assert qb.get_nowait() == 2
    assert r.in_flight() == 0


def test_the_router_refuses_a_duplicate_id():
    r = RpcResponseRouter()
    with ExitStack() as stack:
        stack.enter_context(r.register("dup"))
        with pytest.raises(RuntimeError, match="already in flight"), r.register("dup"):
            pass


def test_the_router_drops_a_reply_nobody_awaits():
    """This is the whole point: on the shared queue it became someone else's."""
    r = RpcResponseRouter()
    with r.register("live"):
        pass  # unregistered on exit, as a timed-out caller would be
    assert r.route("live", "late") is False


def test_the_router_unregisters_even_when_the_body_raises():
    r = RpcResponseRouter()
    with pytest.raises(ValueError), r.register("x"):
        raise ValueError("caller blew up")
    assert r.in_flight() == 0
    assert r.route("x", "late") is False


# ── routing at the manager ─────────────────────────────────────────────────


def test_non_rpc_responses_still_use_the_legacy_queue():
    """Every pre-existing utility command must be unaffected."""
    mgr = _mgr()
    for body in (
        {"cmd": "clear_kv_cache", "result": True},
        {"cmd": "release_memory", "result": None},
        "not even a dict",
    ):
        mgr._route_utility_response(0, body)

    drained = []
    while not mgr.utility_response_queue.empty():
        drained.append(mgr.utility_response_queue.get_nowait())
    assert len(drained) == 3


def test_an_rpc_response_does_not_touch_the_legacy_queue():
    mgr = _mgr(1)
    with mgr._rpc_router.register("r1") as replies:
        mgr._route_utility_response(0, _tp_body("r1", results=[_tp(0, "v")]))
        assert replies.get_nowait()[0] == 0
    assert mgr.utility_response_queue.empty()


def test_a_late_rpc_response_is_dropped_not_queued():
    mgr = _mgr(1)
    mgr._route_utility_response(0, _tp_body("nobody-waiting", results=[_tp(0)]))
    assert mgr.utility_response_queue.empty(), (
        "a late correlated reply must not fall through to the shared queue, "
        "or the next caller inherits it"
    )


def test_a_reply_without_a_usable_id_is_dropped_not_routed_or_queued():
    """Routing runs on the output thread, which a raise would end, and an
    unhashable id made the router's dict lookup raise. An id-less reply on the
    shared queue would become the next synchronous caller's instead."""
    mgr = _mgr(1)
    for request_id in (["not", "hashable"], None, "", 7):
        mgr._route_utility_response(
            0, {"cmd": COLLECTIVE_RPC_CMD, "request_id": request_id, "error": "x"}
        )
    assert mgr.utility_response_queue.empty()


# ── collective_rpc ─────────────────────────────────────────────────────────


def test_it_flattens_dp_major_then_tp_rank():
    mgr = _mgr(2)

    # The request id is minted inside collective_rpc, so replies have to be sent
    # reactively from the broadcast hook. DP 1 answers first, to prove the order
    # comes from the rank and not from arrival.
    def broadcast_and_answer(cmd, **kw):
        mgr.sent.append((cmd, kw))
        rid = kw["request_id"]
        for dp in (1, 0):
            _answer_after(
                mgr,
                0.02,
                dp,
                _tp_body(rid, results=[_tp(0, f"dp{dp}tp0"), _tp(1, f"dp{dp}tp1")]),
            )

    mgr.broadcast_utility_command = broadcast_and_answer
    results = mgr.collective_rpc("m", timeout=10)

    assert [r.value for r in results] == ["dp0tp0", "dp0tp1", "dp1tp0", "dp1tp1"]
    assert [r.tp_rank for r in results] == [0, 1, 0, 1]
    assert all(r.ok for r in results)


def test_the_broadcast_carries_the_full_request():
    mgr = _mgr(1)

    def broadcast_and_answer(cmd, **kw):
        mgr.sent.append((cmd, kw))
        _answer_after(mgr, 0.01, 0, _tp_body(kw["request_id"], results=[_tp(0, 1)]))

    mgr.broadcast_utility_command = broadcast_and_answer
    mgr.collective_rpc("meth", args=(1, 2), kwargs={"k": "v"}, barrier=True, timeout=9)

    cmd, kw = mgr.sent[0]
    assert cmd == COLLECTIVE_RPC_CMD
    assert kw["method"] == "meth"
    assert kw["args"] == (1, 2)
    assert kw["kwargs"] == {"k": "v"}
    assert kw["barrier"] is True
    assert kw["timeout"] == 9
    assert kw["request_id"]


def test_a_silent_dp_engine_yields_a_placeholder_not_a_short_list():
    """A short list would make the caller attribute results to the wrong rank."""
    mgr = _mgr(2)

    def broadcast_and_answer(cmd, **kw):
        # Only DP 0 ever answers.
        _answer_after(mgr, 0.01, 0, _tp_body(kw["request_id"], results=[_tp(0, "ok")]))

    mgr.broadcast_utility_command = broadcast_and_answer
    results = mgr.collective_rpc("m", timeout=0.4)

    assert len(results) == 2
    assert results[0].ok and results[0].value == "ok"
    assert not results[1].ok
    assert "DP rank 1" in results[1].error


def test_a_silent_dp_engine_fails_every_one_of_its_tp_ranks():
    """One placeholder for a whole engine left the list DP x TP short, so every
    position after it named the wrong rank -- and under TP>1 nothing said which
    of that engine's ranks had not been reached."""
    mgr = _mgr(2, tp=2)

    def broadcast_and_answer(cmd, **kw):
        _answer_after(
            mgr,
            0.01,
            0,
            _tp_body(kw["request_id"], results=[_tp(0, "a"), _tp(1, "b")]),
        )

    mgr.broadcast_utility_command = broadcast_and_answer
    results = mgr.collective_rpc("m", timeout=0.4)

    assert len(results) == 4, "one result per DP x TP rank, answered or not"
    assert [r.tp_rank for r in results] == [0, 1, 0, 1]
    assert [r.ok for r in results] == [True, True, False, False]
    assert all("DP rank 1" in r.error for r in results[2:])


def test_an_engine_level_error_fails_every_tp_rank_of_that_engine():
    mgr = _mgr(1, tp=2)

    def broadcast_and_answer(cmd, **kw):
        _answer_after(
            mgr, 0.01, 0, _tp_body(kw["request_id"], error="RuntimeError: shm gone")
        )

    mgr.broadcast_utility_command = broadcast_and_answer
    results = mgr.collective_rpc("m", timeout=5)

    assert [r.tp_rank for r in results] == [0, 1]
    assert not any(r.ok for r in results)
    assert all(r.error == "RuntimeError: shm gone" for r in results)


def test_worker_failures_arrive_as_failures_not_exceptions():
    mgr = _mgr(1)

    def broadcast_and_answer(cmd, **kw):
        _answer_after(
            mgr,
            0.01,
            0,
            _tp_body(
                kw["request_id"],
                results=[_tp(0, "fine"), _tp(1, error="ValueError: boom")],
            ),
        )

    mgr.broadcast_utility_command = broadcast_and_answer
    results = mgr.collective_rpc("m", timeout=5)

    assert results[0].ok
    assert not results[1].ok
    assert results[1].error == "ValueError: boom"


def test_a_duplicate_dp_reply_is_ignored():
    mgr = _mgr(1)

    def broadcast_and_answer(cmd, **kw):
        rid = kw["request_id"]
        _answer_after(mgr, 0.01, 0, _tp_body(rid, results=[_tp(0, "first")]))
        _answer_after(mgr, 0.02, 0, _tp_body(rid, results=[_tp(0, "second")]))

    mgr.broadcast_utility_command = broadcast_and_answer
    results = mgr.collective_rpc("m", timeout=5)
    assert [r.value for r in results] == ["first"]


def test_two_overlapping_calls_keep_their_own_replies():
    """The bug the router exists to close."""
    mgr = _mgr(1)
    ids = []

    def capture(cmd, **kw):
        ids.append(kw["request_id"])

    mgr.broadcast_utility_command = capture

    done = {}

    def caller(tag):
        def run():
            done[tag] = mgr.collective_rpc("m", timeout=5)

        return run

    t1 = threading.Thread(target=caller("a"), daemon=True)
    t1.start()
    while not ids:
        time.sleep(0.01)
    first_id = ids[0]

    t2 = threading.Thread(target=caller("b"), daemon=True)
    t2.start()
    while len(ids) < 2:
        time.sleep(0.01)
    second_id = ids[1]

    # Answer in the opposite order to the calls.
    mgr._route_utility_response(0, _tp_body(second_id, results=[_tp(0, "for-b")]))
    mgr._route_utility_response(0, _tp_body(first_id, results=[_tp(0, "for-a")]))
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert done["a"][0].value == "for-a"
    assert done["b"][0].value == "for-b"


def test_the_id_is_released_so_it_can_be_reused_serially():
    mgr = _mgr(1)

    def broadcast_and_answer(cmd, **kw):
        _answer_after(mgr, 0.01, 0, _tp_body(kw["request_id"], results=[_tp(0, "v")]))

    mgr.broadcast_utility_command = broadcast_and_answer
    for _ in range(3):
        assert mgr.collective_rpc("m", timeout=5)[0].value == "v"
    assert mgr._rpc_router.in_flight() == 0, "a completed call must not leak its slot"


def test_results_are_rpcresult_instances():
    """The public API's return type, which LumenRL reads .ok/.value/.error off."""
    mgr = _mgr(1)

    def broadcast_and_answer(cmd, **kw):
        _answer_after(mgr, 0.01, 0, _tp_body(kw["request_id"], results=[_tp(0, "v")]))

    mgr.broadcast_utility_command = broadcast_and_answer
    (result,) = mgr.collective_rpc("m", timeout=5)
    assert isinstance(result, RpcResult)
    assert result.ok and result.value == "v"


# ── engines that are not DP ranks ──────────────────────────────────────────


def test_pipeline_stages_are_refused_rather_than_reported_as_dp_ranks():
    """Under PP there is one engine per stage, each holding a slice of the
    layers. Their replies came back labelled DP 0, DP 1, ..., with nothing to
    say which stage a result was from."""
    mgr = _mgr(2)
    mgr.pp_size = 2
    with pytest.raises(NotImplementedError, match="pipeline stages"):
        mgr.collective_rpc("m", timeout=1)
    assert mgr.sent == [], "nothing may be broadcast for a refused call"


def test_prefill_and_decode_engines_are_refused_too():
    mgr = _mgr(2, cls=DisaggCoreManager)
    with pytest.raises(NotImplementedError, match="prefill and decode"):
        mgr.collective_rpc("m", timeout=1)
    assert mgr.sent == []


# ── the legacy synchronous path ────────────────────────────────────────────


def _answering(mgr, replies):
    """A broadcast that delivers one reply per DP rank, as the output thread does."""

    def broadcast(cmd, **kw):
        mgr.sent.append((cmd, kw))
        for dp_rank, body in enumerate(replies):
            mgr._route_utility_response(dp_rank, body)

    return broadcast


def test_sync_refuses_a_fire_and_forget_command_up_front():
    """abort_request never answers. Waiting on it used to cost the full timeout;
    making it answer instead left replies nobody asked for on the shared queue,
    for the next synchronous caller to take as its own."""
    mgr = _mgr(1)
    with pytest.raises(ValueError, match="fire-and-forget"):
        mgr.broadcast_utility_command_sync("abort_request", req_id="r")
    assert mgr.sent == [], "nothing may be sent for a call that cannot complete"


def test_sync_raises_the_cause_when_an_engine_reports_an_error():
    mgr = _mgr(2)
    mgr.broadcast_utility_command = _answering(
        mgr,
        [
            {"cmd": "update_weights", "result": 3},
            {"cmd": "update_weights", "error": "RuntimeError: loader rejected"},
        ],
    )
    with pytest.raises(RuntimeError, match="1 of 2 engine.*loader rejected"):
        mgr.broadcast_utility_command_sync("update_weights", named_tensors=[])


def test_sync_still_returns_every_reply_when_all_succeed():
    mgr = _mgr(2)
    replies = [{"cmd": "clear_kv_cache", "result": True}] * 2
    mgr.broadcast_utility_command = _answering(mgr, replies)
    assert mgr.broadcast_utility_command_sync("clear_kv_cache") == replies
