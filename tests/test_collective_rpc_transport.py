# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The generic collective-RPC transport in ``AsyncIOProc``.

Every failure mode here is a *hang* in the pre-existing code, not an exception:

- ``busy_loop`` forwards a worker return only ``if out is not None``, and
  ``call_func(wait_out=True)`` blocks on an untimed ``outputs_queue.get()``, so
  a method returning ``None`` strands its caller.
- ``getattr(runner, name, None)`` skips an unknown method silently, so a typo
  produces the same strand rather than an error.
- an unpicklable return kills the socket sender thread, which strands every
  later caller too.

So these tests assert that the generic path *always answers*, and that the
pre-existing non-payload path is left bit-for-bit alone.
"""

import pickle
import queue
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine.async_proc import (
        AsyncIOProc,
        AsyncIOProcManager,
        RpcPayload,
        RpcResult,
    )


# ── helpers ────────────────────────────────────────────────────────────────


class _Runner:
    """Stand-in for a ModelRunner: one method per outcome under test."""

    def returns_none(self):
        return None

    def returns_value(self, a, b=0):
        return a + b

    def raises(self):
        raise ValueError("boom")

    def returns_unpicklable(self):
        return lambda x: x  # local functions cannot be pickled


class _Barrier:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


def _proc(
    rank: int = 0, *, runners=None, barrier=None, primary_out=True, rpc_channel=True
):
    """An ``AsyncIOProc`` with only the attributes the dispatch path reads.

    ``__init__`` spawns threads and then calls ``busy_loop`` forever, so it
    cannot be used here; the dispatch logic is what is under test.
    """
    proc = object.__new__(AsyncIOProc)
    proc.label = f"test-rank-{rank}"
    proc.rank = rank
    proc.runners = [_Runner()] if runners is None else runners
    proc.all_ranks_barrier = barrier
    proc.io_addrs = (None, "ipc:///tmp/fake" if primary_out else None)
    proc.io_queues = (queue.Queue(), queue.Queue())
    proc.kv_queue = None
    proc.rpc_queue = queue.Queue() if rpc_channel else None
    return proc


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _drive(proc, calls):
    """Run ``busy_loop`` over a fixed script of ``(func_name, args)`` calls.

    A trailing ``exit`` breaks the loop, which is how the real worker stops.
    Returns ``(primary_replies, rpc_replies)``.
    """
    script = list(calls) + [("exit", [])]
    it = iter(script)
    proc.get_func = lambda: next(it)
    proc.busy_loop()
    rpc = _drain(proc.rpc_queue) if proc.rpc_queue is not None else []
    return _drain(proc.io_queues[1]), rpc


# ── the three hangs ────────────────────────────────────────────────────────


def test_a_none_return_still_answers():
    """The original deadlock: `if out is not None` drops the only reply."""
    proc = _proc()
    payload = RpcPayload(request_id="r1")
    out = proc._run_generic_rpc(proc.runners[0], "returns_none", (), {}, payload)

    assert isinstance(out, RpcResult)
    assert out.ok, out.error
    assert out.value is None
    assert out.request_id == "r1"
    assert out.tp_rank == 0

    # And it must actually reach the reply channel, not be filtered on the way.
    primary, rpc = _drive(_proc(), [("returns_none", [payload])])
    assert rpc == [out]
    assert primary == [], "generic replies belong on the per-rank channel"


def test_an_unknown_method_is_an_error_not_a_silent_skip():
    proc = _proc()
    out = proc._run_generic_rpc(
        proc.runners[0], "no_such_method", (), {}, RpcPayload(request_id="r2")
    )
    assert not out.ok
    assert "no_such_method" in out.error
    assert "_Runner" in out.error

    # The strand this replaces: the non-payload path still skips silently, so
    # nothing is queued and a wait_out caller would block.
    assert _drive(_proc(), [("no_such_method", [])]) == ([], [])


def test_a_raising_target_is_reported_not_propagated():
    proc = _proc()
    out = proc._run_generic_rpc(
        proc.runners[0], "raises", (), {}, RpcPayload(request_id="r3")
    )
    assert not out.ok
    assert out.error == "ValueError: boom"
    assert out.value is None


def test_an_unpicklable_result_becomes_an_error():
    """Otherwise the reply kills the sender thread instead of failing the call."""
    proc = _proc()
    out = proc._run_generic_rpc(
        proc.runners[0], "returns_unpicklable", (), {}, RpcPayload(request_id="r4")
    )
    assert not out.ok
    assert "unpicklable" in out.error
    pickle.dumps(out)  # the error reply itself must survive the wire


# ── the payload contract ───────────────────────────────────────────────────


def test_args_and_kwargs_both_reach_the_target():
    proc = _proc()
    payload = RpcPayload(request_id="r5", args=(1,), kwargs={"b": 41})
    out = proc._run_generic_rpc(
        proc.runners[0], "returns_value", (1,), {"b": 41}, payload
    )
    assert out.ok, out.error
    assert out.value == 42

    # Through busy_loop, which is what unpacks the payload.
    _, rpc = _drive(_proc(), [("returns_value", [payload])])
    assert rpc[0].value == 42


def test_without_a_dedicated_channel_it_falls_back_rather_than_discards():
    """A manager that predates the per-rank channel still gets its reply."""
    proc = _proc(rpc_channel=False)
    primary, rpc = _drive(
        proc, [("returns_value", [RpcPayload("r5b", args=(2,), kwargs={"b": 3})])]
    )
    assert rpc == []
    assert [r.value for r in primary] == [5]


def test_kwargs_default_to_empty_when_omitted():
    payload = RpcPayload(request_id="r6")
    assert payload.call_kwargs() == {}
    assert payload.kwargs is None  # frozen dataclass: no mutable default


def test_the_barrier_is_driven_by_the_payload_not_the_name():
    """A generic RPC reusing the IPC buffers needs the barrier too, and its
    method name is not in ``_BARRIER_FUNCS``."""
    barrier = _Barrier()
    proc = _proc(barrier=barrier)
    _drive(proc, [("returns_value", [RpcPayload("r7", args=(1,), barrier=True)])])
    assert barrier.waits == 1

    barrier = _Barrier()
    proc = _proc(barrier=barrier)
    _drive(proc, [("returns_value", [RpcPayload("r8", args=(1,), barrier=False)])])
    assert barrier.waits == 0


def test_the_legacy_barrier_names_still_barrier():
    """The non-payload path must keep using ``_BARRIER_FUNCS``."""
    assert "update_weights_from_ipc" in AsyncIOProc._BARRIER_FUNCS
    assert "update_weights_from_shm" in AsyncIOProc._BARRIER_FUNCS

    class _W:
        def update_weights_from_ipc(self, *a):
            return "done"

    barrier = _Barrier()
    proc = _proc(runners=[_W()], barrier=barrier)
    primary, _ = _drive(proc, [("update_weights_from_ipc", [None, {}, True, None])])
    assert primary == ["done"]
    assert barrier.waits == 1


# ── the untouched path ─────────────────────────────────────────────────────


def test_a_lone_non_payload_arg_is_not_mistaken_for_a_payload():
    """Only an ``RpcPayload`` selects the generic path. A single ordinary
    argument -- which plenty of existing call sites send -- must not."""

    class _One:
        def takes_one(self, value):
            return {"got": value}

    proc = _proc(runners=[_One()])
    primary, rpc = _drive(proc, [("takes_one", ["plain-string"])])
    assert primary == [{"got": "plain-string"}]
    assert rpc == [], "an ordinary arg must not be routed as a generic reply"


def test_the_legacy_path_still_drops_none():
    """Deliberately unchanged: fixing it here would alter the semantics every
    existing ``call_func`` caller was written against."""
    proc = _proc()
    assert _drive(proc, [("returns_none", [])]) == ([], [])


@pytest.mark.parametrize("field", ["request_id", "tp_rank", "value", "error"])
def test_the_reply_carries_what_the_manager_needs_to_route(field):
    assert field in RpcResult.__dataclass_fields__


def test_both_protocol_types_survive_pickle():
    """They cross a ZMQ socket, so this is the wire format working at all."""
    payload = RpcPayload(request_id="r9", args=(1, "x"), kwargs={"k": 2}, barrier=True)
    assert pickle.loads(pickle.dumps(payload)) == payload

    result = RpcResult(request_id="r9", tp_rank=3, value={"a": 1})
    assert pickle.loads(pickle.dumps(result)) == result


# ── the manager side: one reply per rank ───────────────────────────────────


class _Alive:
    def is_alive(self):
        return True


class _Dead:
    def is_alive(self):
        return False


class _Mq:
    def __init__(self):
        self.sent = []

    def enqueue(self, msg):
        self.sent.append(msg)


def _mgr(proc_num=4, *, procs=None):
    """An ``AsyncIOProcManager`` with only what ``collective_rpc`` reads.

    ``__init__`` spawns worker processes and ZMQ threads, so it cannot be used
    here; the collection logic is what is under test.
    """
    mgr = object.__new__(AsyncIOProcManager)
    mgr.label = "test-mgr"
    mgr.proc_num = proc_num
    mgr.rpc_broadcast_mq = _Mq()
    mgr.rpc_outputs_queues = [queue.Queue() for _ in range(proc_num)]
    mgr.procs = [_Alive() for _ in range(proc_num)] if procs is None else procs
    return mgr


def _reply(mgr, rank, request_id, **kw):
    mgr.rpc_outputs_queues[rank].put_nowait(RpcResult(request_id, rank, **kw))


def test_every_rank_answers_and_order_is_rank_order():
    """The point of the whole channel: not just rank 0."""
    mgr = _mgr(4)
    payload = RpcPayload(request_id="m1")
    # Deliberately reply out of order; the result list must still be by rank.
    for rank in (2, 0, 3, 1):
        _reply(mgr, rank, "m1", value=f"rank{rank}")

    results = mgr.collective_rpc("some_method", payload, timeout=5)

    assert [r.tp_rank for r in results] == [0, 1, 2, 3]
    assert [r.value for r in results] == ["rank0", "rank1", "rank2", "rank3"]
    assert all(r.ok for r in results)
    assert mgr.rpc_broadcast_mq.sent == [("some_method", payload)]


def test_a_payload_is_required():
    mgr = _mgr(1)
    with pytest.raises(TypeError, match="RpcPayload"):
        mgr.collective_rpc("m", {"not": "a payload"})


def test_a_dead_rank_is_named_not_waited_out():
    """Without the liveness probe this would burn the whole timeout and then
    report a timeout, which does not say the worker died."""
    mgr = _mgr(2, procs=[_Alive(), _Dead()])
    _reply(mgr, 0, "m2", value="ok")

    started = time.monotonic()
    results = mgr.collective_rpc("m", RpcPayload("m2"), timeout=30)
    elapsed = time.monotonic() - started

    assert results[0].ok
    assert not results[1].ok
    assert "died" in results[1].error
    assert "rank 1" in results[1].error
    assert elapsed < 10, "a dead rank should be reported promptly, not at the deadline"


def test_a_silent_rank_times_out_naming_itself():
    mgr = _mgr(2)
    _reply(mgr, 0, "m3", value="ok")

    results = mgr.collective_rpc("slow_method", RpcPayload("m3"), timeout=0.2)

    assert results[0].ok
    assert not results[1].ok
    assert "timed out" in results[1].error
    assert "slow_method" in results[1].error
    assert "rank 1" in results[1].error


def test_the_result_list_is_always_full_length():
    """A short list would make the caller index the wrong rank."""
    mgr = _mgr(3, procs=[_Alive(), _Dead(), _Alive()])
    _reply(mgr, 0, "m4", value="a")
    _reply(mgr, 2, "m4", value="c")
    results = mgr.collective_rpc("m", RpcPayload("m4"), timeout=5)
    assert len(results) == 3
    assert [r.tp_rank for r in results] == [0, 1, 2]


def test_a_stale_reply_is_dropped_and_the_right_one_still_lands():
    """The bug this closes: the KV channel matches by count, so a late reply
    from an abandoned call is handed to the next caller as its own."""
    mgr = _mgr(1)
    _reply(mgr, 0, "OLD-abandoned", value="wrong")
    _reply(mgr, 0, "m5", value="right")

    results = mgr.collective_rpc("m", RpcPayload("m5"), timeout=5)

    assert results[0].ok
    assert results[0].value == "right"
    assert results[0].request_id == "m5"


def test_two_ids_do_not_take_each_others_replies():
    mgr = _mgr(1)
    _reply(mgr, 0, "first", value=1)
    first = mgr.collective_rpc("m", RpcPayload("first"), timeout=5)
    _reply(mgr, 0, "second", value=2)
    second = mgr.collective_rpc("m", RpcPayload("second"), timeout=5)

    assert first[0].value == 1
    assert second[0].value == 2


def test_a_rank_mismatch_is_reported():
    """A reply on rank 1's socket claiming to be rank 0 means the channels are
    crossed, which would silently attribute results to the wrong GPU."""
    mgr = _mgr(2)
    _reply(mgr, 0, "m6", value="ok")
    mgr.rpc_outputs_queues[1].put_nowait(RpcResult("m6", 0, value="mislabelled"))

    results = mgr.collective_rpc("m", RpcPayload("m6"), timeout=5)

    assert results[0].ok
    assert not results[1].ok
    assert "mismatch" in results[1].error


def test_a_non_result_reply_is_reported():
    mgr = _mgr(1)
    mgr.rpc_outputs_queues[0].put_nowait("just a string")
    results = mgr.collective_rpc("m", RpcPayload("m7"), timeout=5)
    assert not results[0].ok
    assert "unexpected reply type" in results[0].error


def test_worker_errors_survive_the_trip_as_failures():
    """An error on one rank must not look like success, and must not raise
    either -- the caller needs the ranks that did succeed."""
    mgr = _mgr(2)
    _reply(mgr, 0, "m8", value="fine")
    _reply(mgr, 1, "m8", error="ValueError: boom")

    results = mgr.collective_rpc("m", RpcPayload("m8"), timeout=5)

    assert results[0].ok
    assert not results[1].ok
    assert results[1].error == "ValueError: boom"


# ── wiring: the reply address must not reach the runner ────────────────────


def test_rpc_output_addr_is_keyword_only_on_the_worker():
    """It is passed via ``kwargs``, and ``AsyncIOProc`` forwards ``*args`` and
    ``**kwargs`` to the runner's constructor. Keyword-only is what stops it
    being handed to the runner as a stray argument."""
    import inspect

    sig = inspect.signature(AsyncIOProc.__init__)
    param = sig.parameters["rpc_output_addr"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None

    # And it must sit after *args, or it would shift the positional contract.
    names = list(sig.parameters)
    var_positional = next(
        n for n, p in sig.parameters.items() if p.kind is p.VAR_POSITIONAL
    )
    assert names.index("rpc_output_addr") > names.index(var_positional)
