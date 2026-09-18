# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``EngineUtilityHandler`` dispatch for the generic collective RPC.

Three pre-existing hangs are covered here, all with the same shape: a command
that produces no ``UTILITY_RESPONSE`` leaves ``broadcast_utility_command_sync``
blocked on its 300s queue get, so the caller learns "timeout" and never the
cause.

- an unrecognised command was logged and dropped
- ``update_weights`` never answered, so it could only ever time out
- ``abort_request`` never answered, including on its early returns

``engine_utility`` must stay importable without a real AITER build, which is why
the wire types live in ``atom.model_engine.collective_rpc`` rather than in
``async_proc``; ``test_dispatch_does_not_need_aiter`` pins that.
"""

import queue

import pytest

from atom.model_engine.collective_rpc import RpcPayload, RpcResult
from atom.model_engine.engine_utility import EngineUtilityHandler

# ── harness ────────────────────────────────────────────────────────────────


class _RunnerMgr:
    """Stands in for ``AsyncIOProcManager``."""

    def __init__(self, proc_num=2, replies=None, raises=None):
        self.proc_num = proc_num
        self.calls = []
        self._replies = replies
        self._raises = raises

    def collective_rpc(self, method, payload, timeout=300.0):
        self.calls.append((method, payload, timeout))
        if self._raises is not None:
            raise self._raises
        if self._replies is not None:
            return self._replies
        return [
            RpcResult(payload.request_id, r, value=f"rank{r}")
            for r in range(self.proc_num)
        ]

    def call_func(self, name, *args, wait_out=False):
        self.calls.append((name, args, wait_out))
        return 7


class _Engine:
    _has_pending_utility = True
    _is_rl_weights_offloaded = False


def _handler(**kw):
    mgr = _RunnerMgr(**kw)
    out = queue.Queue()
    return EngineUtilityHandler(mgr, out, label="test"), mgr, out


def _responses(out):
    drained = []
    while not out.empty():
        kind, body = out.get_nowait()
        assert kind == "UTILITY_RESPONSE"
        drained.append(body)
    return drained


# ── the command is registered and reachable ────────────────────────────────


def test_collective_rpc_is_registered():
    """Without the registry entry the handler is unreachable and every call
    falls into the unknown-command path."""
    assert EngineUtilityHandler._UTILITY_HANDLERS["collective_rpc"] == (
        "_handle_collective_rpc"
    )


def test_every_registered_command_resolves_to_a_real_method():
    """A registry naming a method that does not exist would raise inside the
    busy loop and take the EngineCore down."""
    h, _, _ = _handler()
    for cmd, name in EngineUtilityHandler._UTILITY_HANDLERS.items():
        assert callable(getattr(h, name, None)), f"{cmd} -> {name} is not callable"


def test_it_runs_through_the_real_dispatcher_and_answers():
    h, mgr, out = _handler(proc_num=3)
    h._execute_utility_command("collective_rpc", {"method": "ping", "request_id": "x1"})

    assert [c[0] for c in mgr.calls] == ["ping"]
    body = _responses(out)[0]
    assert body["cmd"] == "collective_rpc"
    assert body["request_id"] == "x1"
    assert body["tp_world_size"] == 3
    assert [r["tp_rank"] for r in body["results"]] == [0, 1, 2]
    assert [r["value"] for r in body["results"]] == ["rank0", "rank1", "rank2"]
    assert all(r["error"] is None for r in body["results"])


def test_args_kwargs_barrier_and_timeout_all_reach_the_manager():
    h, mgr, _ = _handler()
    h._handle_collective_rpc(
        {
            "method": "m",
            "request_id": "x2",
            "args": [1, 2],
            "kwargs": {"k": "v"},
            "barrier": True,
            "timeout": 12.5,
        }
    )
    method, payload, timeout = mgr.calls[0]
    assert method == "m"
    assert isinstance(payload, RpcPayload)
    assert payload.args == (1, 2)  # list in, tuple out: the wire type is frozen
    assert payload.call_kwargs() == {"k": "v"}
    assert payload.barrier is True
    assert timeout == 12.5


def test_omitted_optionals_get_safe_defaults():
    h, mgr, _ = _handler()
    h._handle_collective_rpc({"method": "m", "request_id": "x3"})
    _, payload, timeout = mgr.calls[0]
    assert payload.args == ()
    assert payload.call_kwargs() == {}
    assert payload.barrier is False
    assert timeout == 300.0


# ── failures answer instead of hanging or killing the loop ──────────────────


def test_a_missing_method_or_request_id_answers_with_an_error():
    for args in ({"request_id": "x4"}, {"method": "m"}, {}):
        h, mgr, out = _handler()
        h._handle_collective_rpc(args)
        body = _responses(out)[0]
        assert body.get("error")
        assert mgr.calls == [], "nothing should be broadcast for a malformed request"


def test_a_raising_manager_is_reported_not_propagated():
    """Raising out of a handler kills the EngineCore busy loop, which takes the
    whole engine with it."""
    h, _, out = _handler(raises=RuntimeError("shm is gone"))
    h._handle_collective_rpc({"method": "m", "request_id": "x5"})
    body = _responses(out)[0]
    assert body["error"] == "RuntimeError: shm is gone"
    assert body["request_id"] == "x5"


def test_per_rank_failures_are_reported_without_losing_the_successes():
    h, _, out = _handler(
        replies=[
            RpcResult("x6", 0, value="fine"),
            RpcResult("x6", 1, error="ValueError: boom"),
        ]
    )
    h._handle_collective_rpc({"method": "m", "request_id": "x6"})
    results = _responses(out)[0]["results"]
    assert results[0]["value"] == "fine" and results[0]["error"] is None
    assert results[1]["error"] == "ValueError: boom"


# ── the three pre-existing hangs ───────────────────────────────────────────


def test_an_unknown_command_answers_instead_of_being_dropped():
    h, _, out = _handler()
    h._execute_utility_command("no_such_command", {})
    body = _responses(out)[0]
    assert body["cmd"] == "no_such_command"
    assert "unknown utility command" in body["error"]


def test_update_weights_now_answers():
    """Previously response-less, so broadcast_utility_command_sync on it could
    only time out. The _shm and _ipc variants always answered."""
    h, _, out = _handler()
    h._execute_utility_command("update_weights", {"named_tensors": []})
    body = _responses(out)[0]
    assert body == {"cmd": "update_weights", "result": 7}


@pytest.mark.parametrize(
    "args",
    [
        {"req_id": "present"},
        {"req_id": None},  # early return in the original
        {},  # no req_id at all
    ],
)
def test_abort_request_answers_on_every_path(args):
    h, _, out = _handler()
    h.scheduler = None  # the other early return
    h._execute_utility_command("abort_request", args)
    body = _responses(out)[0]
    assert body["cmd"] == "abort_request"
    assert "result" in body


def test_abort_request_still_aborts_the_matching_sequence():
    """The response must not come at the cost of the actual behaviour."""
    from atom.model_engine.sequence import SequenceStatus

    class _Seq:
        def __init__(self, sid):
            self.id = sid
            self.status = None

    class _Sched:
        def __init__(self, seqs):
            self.running = seqs
            self.waiting = []

    target = _Seq("abc")
    other = _Seq("xyz")
    h, _, out = _handler()
    h.scheduler = _Sched([target, other])
    h._execute_utility_command("abort_request", {"req_id": "abc"})

    assert target.status == SequenceStatus.ABORTED
    assert other.status is None
    assert _responses(out)[0]["result"] is True


def test_every_handler_answers_or_is_a_documented_exception():
    """A response-less handler is a 300s hang waiting to happen. Two remain
    deliberately fire-and-forget, so they are named rather than assumed."""
    import ast
    import inspect

    fire_and_forget = {"_handle_get_mtp_stats"}  # logs only, has no sync caller

    src = inspect.getsource(EngineUtilityHandler)
    tree = ast.parse(
        "class C:\n" + "\n".join("    " + ln for ln in src.splitlines()[1:])
    )
    silent = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("_handle_"):
            continue
        if node.name in fire_and_forget:
            continue
        body = ast.dump(node)
        if "UTILITY_RESPONSE" not in body:
            silent.append(node.name)
    assert silent == [], f"handlers that never answer: {silent}"


# ── the import boundary ────────────────────────────────────────────────────


def test_dispatch_does_not_need_aiter():
    """This module imported ``engine_utility`` at the top without stubbing
    AITER, which only works while the wire types stay out of ``async_proc``.
    Putting ``RpcPayload`` back there would make the dispatch layer
    unimportable on any machine without a GPU build."""
    import sys

    assert "aiter" not in sys.modules or sys.modules["aiter"] is not None
    mod = sys.modules["atom.model_engine.collective_rpc"]
    assert not hasattr(mod, "MessageQueue")
    assert RpcPayload.__module__ == "atom.model_engine.collective_rpc"
    assert RpcResult.__module__ == "atom.model_engine.collective_rpc"
