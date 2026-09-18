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
from atom.model_engine.engine_core_mgr import CoreManager

# ── harness ────────────────────────────────────────────────────────────────


class _Socket:
    closed = False


def _mgr(engine_count=2):
    """A ``CoreManager`` with only what the RPC path reads.

    ``__init__`` spawns engine processes and binds sockets, so it cannot run
    here; the fan-out and correlation logic is what is under test.
    """
    mgr = object.__new__(CoreManager)
    mgr.label = "test-mgr"
    mgr.control_sockets = [_Socket() for _ in range(engine_count)]
    mgr.utility_response_queue = queue.Queue()
    mgr._rpc_router = RpcResponseRouter()
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


def test_an_engine_level_error_is_reported_without_per_tp_detail():
    mgr = _mgr(1)

    def broadcast_and_answer(cmd, **kw):
        _answer_after(
            mgr, 0.01, 0, _tp_body(kw["request_id"], error="RuntimeError: shm gone")
        )

    mgr.broadcast_utility_command = broadcast_and_answer
    results = mgr.collective_rpc("m", timeout=5)

    assert len(results) == 1
    assert not results[0].ok
    assert results[0].error == "RuntimeError: shm gone"


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
