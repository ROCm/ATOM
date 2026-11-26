# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU checks of real ATOM completion quarantine and Commit reporter threads."""

import array
import threading
import time
import uuid
from itertools import pairwise
from unittest.mock import Mock

import pytest

from atom.kv_transfer.pool.umbp import pool as umbp_pool
from atom.kv_transfer.pool.umbp.pool import (
    FabricFetchInDoubtError,
    UmbpL1Pool,
    UmbpPoolMemory,
)


@pytest.fixture
def pools():
    native = pytest.importorskip("mori_umbp_l1")
    address = f"unix:///tmp/atom_lifecycle_{uuid.uuid4().hex}.sock"
    server = native.HostTestServer(address)
    assert server.ok()
    client = native.L1Client(address)
    for name in ("source", "target"):
        client.attach(name, "group", 0, 8 * 32)
        client.bind_layout(name, 8, [(0, 32)])
    client.open("source")
    slots, _ = client.allocate("source", 2, 2)
    client.apply(
        "source",
        [native.OP_PUBLISH] * 2,
        slots,
        [h.to_bytes(8, "little") for h in (101, 102)],
        1,
    )
    client.commit("source", 1)

    class Proxy:
        fault = None

        def __getattr__(self, name):
            return getattr(client, name)

        def fetch(self, *args, **kwargs):
            if kwargs.get("probe_only"):
                return client.fetch(*args, **kwargs)
            if self.fault == "empty_prefix":
                return [], [], 0, 0, 0, 0
            if self.fault == "partial_prefix":
                return client.fetch(
                    args[0], args[1][:1], dst_slots=kwargs["dst_slots"][:1]
                )
            result = client.fetch(*args, **kwargs)
            if self.fault == "lost_reply":
                raise RuntimeError("response lost after submitted copy")
            if self.fault == "non_prefix":
                result = (list(reversed(result[0])), *result[1:])
            return result

    proxy = Proxy()
    pool = UmbpL1Pool(
        proxy,
        (native.OP_HOLD, native.OP_RELEASE, native.OP_PUBLISH),
        "target",
        8,
        grant_batch=1,
        transport="fabric",
    )
    yield pool, proxy, client
    pool.close()


@pytest.mark.parametrize("fault", ["lost_reply", "non_prefix"])
@pytest.mark.parametrize("clear", [False, True])
def test_ambiguous_fetch_retains_held_slots_and_stops_future_fetches(
    pools, fault, clear
):
    pool, proxy, client = pools
    proxy.fault = fault
    assert pool.fetch_async("request", [101, 102], lambda i: array.array("i", [i]))
    future, slots, *_ = pool._fetches["request"]
    try:
        future.result(timeout=3)
    except RuntimeError:
        pass  # The production poller must turn this into quarantine.
    if clear:
        pool.clear_index()
    with pytest.raises(FabricFetchInDoubtError, match="quarantining 2") as caught:
        pool.poll_fetches()
    assert pool.num_used == 2 and pool.num_indexed == 0
    assert all(pool.is_used(slot) for slot in slots)
    assert "request" in pool._fetches
    target = next(row for row in client.stats() if row["pool_id"] == "target")
    assert target["held"] == 2
    for call in (
        pool.flush,
        pool.poll_fetches,
        lambda: pool.fetch_pending("request"),
        lambda: pool.fetch_async("another", [101], lambda _: array.array("i", [1])),
    ):
        with pytest.raises(FabricFetchInDoubtError) as repeated:
            call()
        assert repeated.value is caught.value


@pytest.mark.parametrize(
    "fault,clear,landed",
    [
        ("empty_prefix", False, 0),
        ("partial_prefix", False, 1),
        (None, False, 2),
        (None, True, 0),
    ],
)
def test_confirmed_prefix_publishes_only_completed_blocks_and_releases_reservations(
    pools, fault, clear, landed
):
    pool, proxy, client = pools
    proxy.fault = fault
    assert pool.fetch_async("request", [101, 102], lambda i: array.array("i", [i]))
    pool._fetches["request"][0].result(timeout=3)
    if clear:
        pool.clear_index()
    pool.poll_fetches()
    assert pool.num_used == 0 and pool.num_indexed == landed
    assert [pool.lookup(h) != -1 for h in (101, 102)] == [i < landed for i in range(2)]
    pool.flush()
    target = next(row for row in client.stats() if row["pool_id"] == "target")
    assert target["held"] == 0
    assert not pool.fetch_pending("request")


class FakeEvent(threading.Event):
    def record(self):
        pass

    def query(self):
        return self.is_set()


@pytest.fixture
def owner(monkeypatch, request):
    import torch

    event = FakeEvent()
    event.set()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    native_memory = Mock()
    monkeypatch.setattr(
        umbp_pool,
        "_load_native_l1_helpers",
        lambda: (lambda *_: native_memory, None),
    )
    memory = UmbpPoolMemory(
        Mock(), "owner", "group", 1.0, transport=getattr(request, "param", "ipc")
    )
    memory._mapped_base = 0x1000
    yield memory, event
    # Cleanup must also work when checking an old close() that leaks its thread.
    if memory._watcher is not None:
        memory._stopping.set()
        memory._wake.set()
        memory._watcher.join(timeout=2)
        assert not memory._watcher.is_alive()


@pytest.mark.parametrize(
    "owner,detach_fails",
    [("ipc", False), ("ipc", True), ("fabric", False)],
    indirect=["owner"],
)
@pytest.mark.parametrize("state", ["idle", "retrying"])
def test_close_stops_reporter_and_is_idempotent(
    owner, monkeypatch, state, detach_fails
):
    memory, _ = owner
    ready = threading.Event()
    attempts = []

    def commit(*_):
        attempts.append(time.monotonic())
        if state == "retrying":
            if len(attempts) >= 3:
                ready.set()
            raise RuntimeError("UMBP unavailable")

    memory._client.commit.side_effect = commit
    wait_for_wake = memory._wake.wait

    def wait_for_work(timeout=None):
        ready.set()
        return wait_for_wake(timeout)

    monkeypatch.setattr(memory._wake, "wait", wait_for_work)

    def detach(*_):
        assert not memory._watcher.is_alive()
        if detach_fails:
            raise RuntimeError("UMBP unavailable")

    if memory._native_memory is not None:
        memory._native_memory.close.side_effect = detach
    else:
        memory._client.detach.side_effect = detach
    memory.writes_issued()
    assert ready.wait(1)
    memory.close()  # Exercise the real shutdown path, including join().
    assert not memory._watcher.is_alive()
    assert memory._closed and memory._mapped_base is None and not memory._issued
    attempts_at_close = len(attempts)
    memory.close()
    with pytest.raises(RuntimeError, match="memory is closing"):
        memory.writes_issued()
    assert len(attempts) == attempts_at_close
    if memory._native_memory is not None:
        memory._native_memory.close.assert_called_once()
        memory._client.detach.assert_not_called()
    else:
        memory._client.detach.assert_called_once_with("owner")
    if state == "retrying":
        assert len(attempts) >= 3
        assert all(b - a >= 0.008 for a, b in pairwise(attempts))


@pytest.mark.parametrize("owner", ["fabric"], indirect=True)
@pytest.mark.parametrize("commit_fails", [False, True])
def test_fabric_close_waits_for_inflight_commit(owner, monkeypatch, commit_fails):
    memory, _ = owner
    entered_commit = threading.Event()
    finish_commit = threading.Event()
    entered_join = threading.Event()
    errors = []

    def commit(*_):
        entered_commit.set()
        assert finish_commit.wait(5), "test did not release the blocked Commit"
        if commit_fails:
            # Simulate the native client's gRPC deadline expiring, without
            # making a CPU regression wait for the real 120-second deadline.
            raise RuntimeError("Commit deadline exceeded")

    def detach():
        assert finish_commit.is_set()
        assert not memory._watcher.is_alive()

    memory._client.commit.side_effect = commit
    memory._native_memory.close.side_effect = detach
    memory.writes_issued()
    assert entered_commit.wait(2)
    join_reporter = memory._watcher.join

    def join(*args, **kwargs):
        entered_join.set()
        return join_reporter(*args, **kwargs)

    monkeypatch.setattr(memory._watcher, "join", join)

    def close():
        try:
            memory.close()
        except BaseException as exc:
            errors.append(exc)

    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    try:
        assert entered_join.wait(2)
        assert closer.is_alive()
        memory._native_memory.close.assert_not_called()
        assert not memory._closed and memory._mapped_base == 0x1000
        with pytest.raises(RuntimeError, match="memory is closing"):
            memory.writes_issued()
    finally:
        finish_commit.set()
        closer.join(timeout=2)

    assert not closer.is_alive() and not errors
    assert not memory._watcher.is_alive()
    # Once stopping begins, a failed Commit is not retried before Detach.
    memory._client.commit.assert_called_once()
    memory._native_memory.close.assert_called_once()
    assert memory._closed and memory._mapped_base is None


@pytest.mark.parametrize("owner", ["fabric"], indirect=True)
def test_fabric_close_failure_preserves_state_for_retry(owner):
    memory, event = owner
    event.clear()
    native_memory = memory._native_memory
    native_memory.close.side_effect = [RuntimeError("Detach unconfirmed"), None]
    memory.writes_issued()
    with pytest.raises(RuntimeError, match="Detach unconfirmed"):
        memory.close()
    assert not memory._closed and memory._mapped_base == 0x1000
    assert memory._native_memory is native_memory
    assert len(memory._issued) == 1 and memory._issued[0][1] is event
    with pytest.raises(RuntimeError, match="memory is closing"):
        memory.writes_issued()
    memory.close()
    assert memory._closed and memory._mapped_base is None and not memory._issued
    memory.close()
    assert native_memory.close.call_count == 2


@pytest.mark.parametrize("later_pending", [False, True])
def test_commit_retries_only_completed_writes(owner, monkeypatch, later_pending):
    import torch

    memory, event = owner
    event.clear()
    first_attempt = threading.Event()
    allow_failure = threading.Event()
    acknowledged = threading.Event()
    later = FakeEvent()

    def commit(*_):
        if memory._client.commit.call_count == 1:
            first_attempt.set()
            assert allow_failure.wait(1)
            raise RuntimeError("temporary unavailable")
        acknowledged.set()

    memory._client.commit.side_effect = commit
    memory.writes_issued()
    assert not first_attempt.wait(0.03)
    memory._client.commit.assert_not_called()
    event.set()
    assert first_attempt.wait(1)
    if later_pending:
        monkeypatch.setattr(torch.cuda, "Event", lambda: later)
        memory.writes_issued()  # This forward must not advance the retried stamp.
    allow_failure.set()
    assert acknowledged.wait(1), "the idle owner's final Commit was not retried"
    attempts = memory._client.commit.call_args_list
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    if later_pending:
        acknowledged.clear()
        assert not acknowledged.wait(0.03)
        later.set()
        assert acknowledged.wait(1)
        assert len(attempts) == 3 and attempts[2].args[1] > attempts[1].args[1]
