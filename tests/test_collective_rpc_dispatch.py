# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``EngineUtilityHandler`` dispatch for the generic collective RPC.

Three pre-existing hangs are covered here, all with the same shape: a command
that produces no ``UTILITY_RESPONSE`` leaves ``broadcast_utility_command_sync``
blocked on its 300s queue get, so the caller learns "timeout" and never the
cause.

- an unrecognised command was logged and dropped
- ``update_weights`` never answered, so it could only ever time out
- a handler that raised never answered either, on any command

The opposite mistake is covered too. ``abort_request`` is fire-and-forget, and
answering it anyway leaves a reply nobody asked for on the shared queue, where
the next synchronous caller takes it as its own.

``engine_utility`` must stay importable without a real AITER build, which is why
the wire types live in ``atom.model_engine.collective_rpc`` rather than in
``async_proc``; ``test_dispatch_does_not_need_aiter`` pins that.
"""

import queue

import pytest

from atom.model_engine.collective_rpc import RpcPayload, RpcResult
from atom.model_engine.engine_utility import (
    FIRE_AND_FORGET_UTILITY_CMDS,
    EngineUtilityHandler,
)

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
        if method == "discard_failed_weight_sync":
            return [
                RpcResult(payload.request_id, r, value=True)
                for r in range(self.proc_num)
            ]
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
    _rl_weights_inconsistent = False


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


def _counts(*updated):
    """One direct-update reply per rank, each with its updated-parameter count."""
    return [RpcResult("u", rank, value=n) for rank, n in enumerate(updated)]


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


def test_a_non_string_method_is_refused_before_the_broadcast():
    """Truthy but not a name: it reached getattr on every TP worker, which
    raises TypeError outside the worker's own error handling."""
    h, mgr, out = _handler()
    h._handle_collective_rpc({"method": 123, "request_id": "x8"})
    (body,) = _responses(out)
    assert body["request_id"] == "x8", "the reply must still reach its caller"
    assert body["error"]
    assert mgr.calls == [], "nothing may reach the workers"


def test_a_non_string_request_id_is_refused_before_the_broadcast():
    """The manager routes replies with the id as a dict key, so an unhashable
    one ended its output thread once the replies came back."""
    h, mgr, out = _handler()
    h._handle_collective_rpc({"method": "m", "request_id": ["x9"]})
    (body,) = _responses(out)
    assert body["error"]
    assert mgr.calls == [], "nothing may reach the workers"


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
    h, _, out = _handler(replies=_counts(4, 4))
    h._execute_utility_command("update_weights", {"named_tensors": []})
    body = _responses(out)[0]
    assert body == {"cmd": "update_weights", "result": 4}


_DIRECT_UPDATES = [
    ("update_weights", {"named_tensors": ["t"], "flush_cache": False}),
    ("update_weights_shm", {"shm_name": "s", "bucket_meta": {"w": 1}}),
    (
        "update_weights_ipc",
        {"ipc_handle": "h", "bucket_meta": {"w": 1}, "is_last": False},
    ),
]
_DIRECT_UPDATE_CALLS = {
    "update_weights": ("update_weights", (["t"], False)),
    "update_weights_shm": ("update_weights_from_shm", ("s", {"w": 1}, True)),
    "update_weights_ipc": ("update_weights_from_ipc", ("h", {"w": 1}, False, None)),
}


@pytest.mark.parametrize(("cmd", "args"), _DIRECT_UPDATES)
def test_a_direct_update_runs_on_every_rank(cmd, args):
    """call_func waited on rank 0's answer alone, so every other rank's outcome
    went unheard."""
    h, mgr, out = _handler(replies=_counts(4, 4, 4))
    h._execute_utility_command(cmd, args)

    ((method, payload, _),) = mgr.calls
    assert (method, payload.args) == _DIRECT_UPDATE_CALLS[cmd]
    assert _responses(out) == [{"cmd": cmd, "result": 4}]


@pytest.mark.parametrize(
    ("cmd", "barrier"),
    [
        ("update_weights", False),
        ("update_weights_shm", True),
        ("update_weights_ipc", True),
    ],
)
def test_the_shared_buffer_updates_keep_their_barrier(cmd, barrier):
    """On the call_func path, _BARRIER_FUNCS held every rank until all had read
    the caller's buffer. Moving these updates to the generic path must not
    quietly drop that."""
    h, mgr, _ = _handler(replies=_counts(4, 4))
    h._execute_utility_command(cmd, dict(_DIRECT_UPDATES)[cmd])
    ((_, payload, _),) = mgr.calls
    assert payload.barrier is barrier


@pytest.mark.parametrize(("cmd", "args"), _DIRECT_UPDATES)
def test_ranks_that_updated_different_counts_are_an_error(cmd, args):
    """Every rank is sent the same tensors. One that updated fewer skipped a
    tensor its peers wrote -- a shape it could not shard, say -- and still
    returned normally, so no rank failed and rank 0's count read as success."""
    h, _, out = _handler(replies=_counts(4, 3))
    h._execute_utility_command(cmd, args)

    assert _responses(out) == [
        {
            "cmd": cmd,
            "error": "TP ranks updated different numbers of parameters: "
            "rank 0 updated 4, rank 1 updated 3",
        }
    ]
    assert h.runner_mgr.calls[-1][0] == "discard_failed_weight_sync"


@pytest.mark.parametrize(("cmd", "args"), _DIRECT_UPDATES)
def test_a_direct_update_that_fails_on_a_nonzero_rank_is_an_error(cmd, args):
    """Rank 0 succeeding said nothing about rank 1, and the caller went on with
    a partly updated model."""
    h, _, out = _handler(
        replies=[
            RpcResult("u1", 0, value=4),
            RpcResult("u1", 1, error="ValueError: rejected q_proj"),
        ]
    )
    h._execute_utility_command(cmd, args)

    assert _responses(out) == [
        {"cmd": cmd, "error": "TP rank 1: ValueError: rejected q_proj"}
    ]
    assert h.runner_mgr.calls[-1][0] == "discard_failed_weight_sync"


@pytest.mark.parametrize(("cmd", "args"), _DIRECT_UPDATES)
def test_a_direct_update_the_manager_cannot_run_is_answered(cmd, args):
    h, _, out = _handler(raises=RuntimeError("shm is gone"))
    h._execute_utility_command(cmd, args)
    assert _responses(out) == [{"cmd": cmd, "error": "RuntimeError: shm is gone"}]


# ── a bucketed sync and the engine's sleep state ───────────────────────────


def _asleep_after(cmd, is_last, **kw):
    """Run one bucket of a sync on an engine whose weights are offloaded."""
    h, _, out = _handler(**kw)
    engine = _Engine()
    engine._is_rl_weights_offloaded = True
    pending = queue.Queue()
    pending.put_nowait((cmd, {"bucket_meta": {}, "is_last": is_last}))
    h.process_queue(pending, engine)
    assert len(_responses(out)) == 1, "the bucket must still be answered"
    return engine._is_rl_weights_offloaded


@pytest.mark.parametrize("cmd", ["update_weights_shm", "update_weights_ipc"])
def test_the_last_bucket_landing_on_every_rank_wakes_the_engine(cmd):
    assert not _asleep_after(cmd, True, replies=_counts(4, 4))
    assert _asleep_after(cmd, False, replies=_counts(4, 4)), "not before the last"


@pytest.mark.parametrize("cmd", ["update_weights_shm", "update_weights_ipc"])
@pytest.mark.parametrize(
    "outcome",
    [
        {"replies": [RpcResult("u", 0, value=4), RpcResult("u", 1, error="boom")]},
        {"replies": _counts(4, 3)},
        {"raises": RuntimeError("shm is gone")},
    ],
    ids=["a rank failed", "the ranks disagree", "the manager failed"],
)
def test_a_failed_last_bucket_leaves_the_engine_asleep(cmd, outcome):
    """A failed update is answered rather than taking the engine down, so the
    loop carries on -- and waking would then schedule onto weights that are
    part old and part new."""
    assert _asleep_after(cmd, True, **outcome)


def _process_one(handler, out, engine, cmd, args):
    engine._has_pending_utility = True
    pending = queue.Queue()
    pending.put_nowait((cmd, args))
    handler.process_queue(pending, engine)
    return _responses(out)


def test_any_failed_bucket_fences_an_engine_that_was_awake():
    h, _, out = _handler(
        replies=[RpcResult("u", 0, value=4), RpcResult("u", 1, error="boom")]
    )
    engine = _Engine()
    engine._is_rl_weights_offloaded = False
    engine._rl_weights_inconsistent = False

    _process_one(h, out, engine, "update_weights_ipc", {"is_last": False})

    assert engine._rl_weights_inconsistent
    assert engine._is_rl_weights_offloaded


def test_wake_up_cannot_unfence_a_failed_update_but_a_complete_sync_can():
    h, mgr, out = _handler(
        replies=[RpcResult("u", 0, value=4), RpcResult("u", 1, error="boom")]
    )
    engine = _Engine()
    engine._is_rl_weights_offloaded = False
    engine._rl_weights_inconsistent = False
    _process_one(h, out, engine, "update_weights_shm", {"is_last": False})

    _process_one(h, out, engine, "resume_memory", {"tags": ["weights"]})
    assert engine._is_rl_weights_offloaded
    assert engine._rl_weights_inconsistent

    mgr._replies = _counts(4, 4)
    _process_one(h, out, engine, "update_weights_shm", {"is_last": True})
    assert not engine._is_rl_weights_offloaded
    assert not engine._rl_weights_inconsistent


def test_every_answered_utility_command_stamps_the_callers_request_id():
    h, _, out = _handler()
    h._execute_utility_command("clear_kv_cache", {"request_id": "utility-1"})
    assert _responses(out) == [
        {"cmd": "clear_kv_cache", "result": 7, "request_id": "utility-1"}
    ]


def _raise(exc):
    def fail(*args, **kwargs):
        raise exc

    return fail


def test_a_raising_handler_answers_before_the_engine_goes_down():
    """A handler that raised escaped the busy loop with no reply, and the
    caller waited out its timeout for an engine that was already gone."""
    h, mgr, out = _handler()
    mgr.call_func = _raise(RuntimeError("cannot release weights"))

    with pytest.raises(RuntimeError, match="cannot release"):
        h._execute_utility_command("release_memory", {"tags": ["weights"]})

    assert _responses(out) == [
        {"cmd": "release_memory", "error": "RuntimeError: cannot release weights"}
    ]


@pytest.mark.parametrize("field", ["args", "kwargs"])
def test_a_malformed_collective_rpc_payload_is_answered_without_ending_the_loop(field):
    """Payload conversion itself must sit inside the handler's guard: a scalar
    args or kwargs value used to raise before it and end the EngineCore loop."""
    h, mgr, out = _handler()
    h._execute_utility_command(
        "collective_rpc", {"method": "m", "request_id": "x7", field: 5}
    )
    (body,) = _responses(out)
    assert body["request_id"] == "x7"
    assert body["error"].startswith("TypeError")
    assert mgr.calls == []


@pytest.mark.parametrize(
    "args",
    [
        {"req_id": "present"},
        {"req_id": None},
        {},  # no req_id at all
    ],
)
def test_abort_request_never_answers(args):
    """Every caller is a client-disconnect path that sends and moves on, so an
    answer is only ever a stray reply on the shared queue."""
    h, _, out = _handler()
    h.scheduler = None
    h._execute_utility_command("abort_request", args)
    assert out.empty()


def test_a_fire_and_forget_handler_that_raises_stays_silent():
    class _TornDown:
        waiting = ()

        @property
        def running(self):
            raise RuntimeError("scheduler torn down")

    h, _, out = _handler()
    h.scheduler = _TornDown()
    with pytest.raises(RuntimeError, match="torn down"):
        h._execute_utility_command("abort_request", {"req_id": "abc"})
    assert out.empty()


def test_abort_request_still_aborts_the_matching_sequence():
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
    assert out.empty()


def test_every_handler_answers_unless_it_is_fire_and_forget():
    """Both directions are bugs. A handler that never answers is a 300s hang for
    a synchronous caller; a fire-and-forget one that answers leaves a reply for
    the next synchronous caller to take as its own. So each handler is pinned
    to exactly one side, by the same set the manager refuses to wait on."""
    import ast
    import inspect

    handlers = EngineUtilityHandler._UTILITY_HANDLERS
    assert FIRE_AND_FORGET_UTILITY_CMDS <= set(handlers)
    fire_and_forget = {handlers[cmd] for cmd in FIRE_AND_FORGET_UTILITY_CMDS}

    src = inspect.getsource(EngineUtilityHandler)
    tree = ast.parse(
        "class C:\n" + "\n".join("    " + ln for ln in src.splitlines()[1:])
    )
    answers = {
        node.name: "UTILITY_RESPONSE" in ast.dump(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_handle_")
    }
    silent = sorted(n for n, a in answers.items() if not a and n not in fire_and_forget)
    chatty = sorted(n for n, a in answers.items() if a and n in fire_and_forget)
    assert silent == [], f"handlers that never answer: {silent}"
    assert chatty == [], f"fire-and-forget handlers that answer anyway: {chatty}"


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
