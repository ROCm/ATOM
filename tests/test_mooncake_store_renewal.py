# SPDX-License-Identifier: MIT
"""Keys-based renewal on `AsyncStoreLookup`, with the native RPC controlled."""

import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.mooncake_store import async_lookup
from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup


class _Executor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))
        return future

    def shutdown(self):
        pass

    def finish(self, index, result=0):
        future = self.jobs[index][0]
        if future.set_running_or_notify_cancel():
            future.set_result(result)


def _renew(client, sid, ident):
    """`renew_async` whose keys are the ident itself."""
    return client.renew_async(sid, ident, lambda: ident)


@pytest.fixture
def lookup(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        async_lookup, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    backend = SimpleNamespace(renew_keys=lambda keys: 0, close=lambda: None)
    client = AsyncStoreLookup(backend)
    client.renew_executor.shutdown()
    client.renew_executor = executor = _Executor()
    yield client, executor, clock
    client.close()


def test_pending_renewal_is_submitted_once_and_zero_hit_still_dispatches(lookup):
    client, executor, _ = lookup
    for _ in range(3):
        assert _renew(client, "a", ("key",)) == "pending"
    assert len(executor.jobs) == 1
    future, fn, args = executor.jobs[0]
    assert fn == client._renew_keys and args == (("key",),)
    future.set_result(fn(*args))
    assert _renew(client, "a", ("key",)) == "ready"
    assert not client._renewals


@pytest.mark.parametrize("late", [0.0, 0.99, 1.0, 1.01, 30.0, 3600.0])
def test_completed_renewal_polled_late_is_ready_not_failed(lookup, late):
    """The Store answered in 5 ms; the scheduler only came back `late` s later."""
    client, executor, clock = lookup
    assert _renew(client, "a", ("key",)) == "pending"
    clock.now += 0.005
    executor.finish(0, 3)
    clock.now += late
    assert _renew(client, "a", ("key",)) == "ready"
    assert not client._renewals


@pytest.mark.parametrize("lateness", [0.05, 1.0, 5.0])
def test_burst_of_renewals_on_one_serial_thread_all_dispatch(lookup, lateness):
    """60 renewals submitted a step (50 ms) apart, then all answered at once."""
    client, executor, clock = lookup
    for i in range(60):
        clock.now = i * 0.05
        assert _renew(client, f"s{i}", (f"k{i}",)) == "pending"
    for i in range(60):  # the serial thread has answered every one
        executor.finish(i)
    clock.now += lateness
    assert {_renew(client, f"s{i}", (f"k{i}",)) for i in range(60)} == {"ready"}
    assert not client._renewals


@pytest.mark.parametrize("cause", ["error", "cancelled"])
def test_failed_renewal_falls_back_to_dispatch_with_its_reason(lookup, caplog, cause):
    client, executor, _ = lookup
    assert _renew(client, "a", ("key",)) == "pending"
    future = executor.jobs[0][0]
    if cause == "error":
        future.set_exception(RuntimeError("native RPC failed"))
    else:
        future.cancel()
    with caplog.at_level("WARNING", logger="atom"):
        assert _renew(client, "a", ("key",)) == cause
    assert ("native RPC failed" in caplog.text) == (cause == "error")
    assert not client._renewals


def test_timeout_dispatches_and_a_hung_rpc_does_not_fail_later_loads(lookup):
    client, executor, clock = lookup
    assert _renew(client, "a", ("a",)) == "pending"
    future = executor.jobs[0][0]
    assert future.set_running_or_notify_cancel()  # the RPC hangs
    clock.now = 0.99
    assert _renew(client, "a", ("a",)) == "pending"
    clock.now = 1.0
    assert _renew(client, "a", ("a",)) == "timeout"
    # Behind the hung RPC every later hit dispatches at once, like the sync
    # path that skips renewal while one is running; none waits or fails.
    assert [_renew(client, sid, (sid,)) for sid in "bc"] == ["renewal_busy"] * 2
    assert len(executor.jobs) == 1
    future.set_result(0)  # the Store recovers: renewals resume
    # ... for every load of a burst, not only the first.
    assert [_renew(client, sid, (sid,)) for sid in "bcd"] == ["pending"] * 3
    assert len(executor.jobs) == 4


def test_abort_of_running_renewal_then_new_load_does_not_wait(lookup):
    client, executor, _ = lookup
    assert _renew(client, "a", ("a",)) == "pending"
    assert executor.jobs[0][0].set_running_or_notify_cancel()  # RPC hangs
    client.cancel_renewal("a")  # abort: a running future cannot be stopped
    assert not client._renewals
    assert _renew(client, "b", ("b",)) == "renewal_busy"
    assert len(executor.jobs) == 1


def test_queued_renewal_that_times_out_is_cancelled_and_dispatches(lookup):
    client, executor, clock = lookup
    assert _renew(client, "a", ("a",)) == "pending"
    clock.now = 2.0
    assert _renew(client, "a", ("a",)) == "timeout"
    assert executor.jobs[0][0].cancelled()  # it never runs, nor blocks the thread
    assert not client._stuck_renewals


def test_other_keys_replace_the_ticket(lookup):
    client, executor, _ = lookup
    assert _renew(client, "a", ("key",)) == "pending"
    assert _renew(client, "a", ("other",)) == "pending"
    assert executor.jobs[0][0].cancelled()
    assert len(executor.jobs) == 2


def test_cancel_and_rejected_submission_are_bounded(lookup):
    client, executor, _ = lookup
    assert _renew(client, "a", ("a",)) == "pending"
    client.cancel_renewal("a")
    client.cancel_renewal("a")
    assert executor.jobs[0][0].cancelled()
    assert not client._renewals
    executor.submit = lambda fn, *args: None  # the executor's queue is full
    assert _renew(client, "b", ("b",)) == "rejected"
    assert not client._renewals


def test_real_executor_pending_while_the_rpc_runs_then_ready_after_close_rejects():
    release, entered = threading.Event(), threading.Event()
    calls = []

    def renew_keys(keys):
        calls.append(keys)
        entered.set()
        release.wait(5)
        return len(keys)

    client = AsyncStoreLookup(
        SimpleNamespace(renew_keys=renew_keys, close=lambda: None)
    )
    assert _renew(client, "a", ("k1", "k2")) == "pending"
    assert entered.wait(5)
    assert _renew(client, "a", ("k1", "k2")) == "pending"
    release.set()
    deadline = time.monotonic() + 5
    while (state := _renew(client, "a", ("k1", "k2"))) == "pending":
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert state == "ready" and calls == [("k1", "k2")]
    client.close()
    assert _renew(client, "b", ("k3",)) == "rejected"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("off", False),
        ("1", True),
        ("true", True),
        (" TRUE ", True),
    ],
)
def test_async_renew_flag_is_opt_in(monkeypatch, raw, expected):
    from atom.utils import envs

    if raw is None:
        monkeypatch.delenv("ATOM_STORE_ASYNC_RENEW", raising=False)
    else:
        monkeypatch.setenv("ATOM_STORE_ASYNC_RENEW", raw)
    assert envs.ATOM_STORE_ASYNC_RENEW is expected


def test_lookup_and_renewal_failures_do_not_mute_each_other(lookup, caplog):
    client, _, _ = lookup
    with caplog.at_level("WARNING", logger="atom"):
        client._warn_failure(
            "lookup", "answer unavailable", RuntimeError("lookup boom")
        )
        client._warn_failure(
            "lease renewal",
            "the load is dispatched without it",
            RuntimeError("renew boom"),
        )
        client._warn_failure("lease renewal", "unused", RuntimeError("renew again"))
    assert "lookup boom" in caplog.text and "renew boom" in caplog.text
    assert "renew again" not in caplog.text
    assert "lookup failed; answer unavailable: lookup boom" in caplog.text
    assert (
        "renewal failed; the load is dispatched without it: renew boom" in caplog.text
    )


def test_sync_renew_asks_pools_before_building_any_key(monkeypatch):
    from atom.kv_transfer.offload.mooncake_store import scheduler as store_sched

    lookup = object.__new__(store_sched._StoreLookup)
    lookup._chunk_tokens = 4
    monkeypatch.setattr(store_sched, "prompt_chunk_hashes", lambda seq, n: b"\0" * 64)
    monkeypatch.setattr(lookup, "_pools_to_ask", list, raising=False)
    monkeypatch.setattr(
        lookup,
        "_rank_keys",
        lambda *a: pytest.fail("keys built with no pool to ask"),
        raising=False,
    )
    assert lookup.renew(SimpleNamespace(), 0, 8) == 0


def test_a_full_renewal_queue_rejects_new_submissions_then_recovers():
    release = threading.Event()
    entered = threading.Event()

    def renew_keys(keys):
        entered.set()
        return release.wait(5) and 1

    backend = SimpleNamespace(renew_keys=renew_keys, close=lambda: None)
    client = AsyncStoreLookup(backend, capacity=2)
    try:
        # The bound excludes the running job, so wait until the worker has
        # taken s0 before filling the two queue slots: otherwise s0 may still
        # sit in the queue and only one of the later submissions is refused.
        assert _renew(client, "s0", ("k0",)) == "pending"
        assert entered.wait(5)
        states = [_renew(client, f"s{i}", (f"k{i}",)) for i in range(1, 5)]
        assert states == ["pending", "pending", "rejected", "rejected"]
        assert not {"s3", "s4"} & set(client._renewals)
        release.set()
        deadline = time.monotonic() + 5
        for i in range(3):
            # Drain each accepted renewal once it has finished.
            while (state := _renew(client, f"s{i}", (f"k{i}",))) == "pending":
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert state == "ready"
        assert _renew(client, "s4", ("k4",)) in ("pending", "ready")
    finally:
        release.set()
        client.close()
