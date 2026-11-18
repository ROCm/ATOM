# SPDX-License-Identifier: MIT
"""Scheduler renewal tickets use actual futures, with the native RPC controlled."""

from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.mooncake_store.renewal_gate import (
    RenewalGate,
    RenewalState,
)


class _Executor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, keys):
        future = Future()
        self.jobs.append((future, fn, keys))
        return future


@pytest.fixture
def gate():
    clock = SimpleNamespace(now=0.0)
    executor = _Executor()
    client = SimpleNamespace(
        capacity=2,
        renew_executor=executor,
        backend_slot=SimpleNamespace(take=lambda **kw: nullcontext()),
        backend=SimpleNamespace(renew_keys=lambda keys: 0),
    )
    return RenewalGate(client, clock=lambda: clock.now), executor, clock


def test_pending_renewal_is_submitted_once_and_zero_hit_still_dispatches(gate):
    ticket_gate, executor, _ = gate
    owner = object()
    for _ in range(3):
        assert (
            ticket_gate.request(owner, "a", ("key",), (0, 256)) is RenewalState.PENDING
        )
    assert len(executor.jobs) == 1
    future, fn, keys = executor.jobs[0]
    assert keys == ("key",)  # No Sequence reaches the executor.
    future.set_result(fn(keys))
    assert ticket_gate.request(owner, "a", keys, (0, 256)) is RenewalState.PINNED
    assert ticket_gate.tickets == {}


@pytest.mark.parametrize("cause", ["deadline", "rpc_error", "stale_completion"])
def test_failed_renewal_retires_ticket(gate, cause):
    ticket_gate, executor, clock = gate
    owner = object()
    assert ticket_gate.request(owner, "a", ("key",), (0, 256)) is RenewalState.PENDING
    future, _, _ = executor.jobs[0]
    if cause == "deadline":
        clock.now = 1.01
    elif cause == "rpc_error":
        future.set_exception(RuntimeError("native RPC failed"))
    else:
        future.set_result((1, -2.0, 0.0))
    assert ticket_gate.request(owner, "a", ("key",), (0, 256)) is RenewalState.FAILED
    assert not ticket_gate.tickets


@pytest.mark.parametrize("change", ["owner", "keys", "physical_blocks"])
def test_lifecycle_or_physical_identity_change_replaces_the_ticket(gate, change):
    ticket_gate, executor, _ = gate
    owner, keys, blocks = object(), ("key",), (0, 256, (4, 5))
    assert ticket_gate.request(owner, "a", keys, blocks) is RenewalState.PENDING
    old = executor.jobs[0][0]
    if change == "owner":
        owner = object()
    elif change == "keys":
        keys = ("other",)
    else:
        blocks = (0, 256, (6, 7))
    assert ticket_gate.request(owner, "a", keys, blocks) is RenewalState.PENDING
    assert old.cancelled()
    assert len(executor.jobs) == 2


def test_cancellation_close_and_capacity_are_bounded(gate):
    ticket_gate, executor, _ = gate
    a, b = object(), object()
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.PENDING
    assert ticket_gate.request(b, "b", ("b",), ()) is RenewalState.PENDING
    assert ticket_gate.request(object(), "c", ("c",), ()) is RenewalState.FAILED
    ticket_gate.cancel("a", object())  # An old owner cannot cancel a reused ID.
    assert "a" in ticket_gate.tickets
    ticket_gate.cancel("a", a)
    assert executor.jobs[0][0].cancelled()
    ticket_gate.close()
    assert executor.jobs[1][0].cancelled()
    assert not ticket_gate.tickets
    assert ticket_gate.request(a, "a", ("a",), ()) is RenewalState.FAILED
