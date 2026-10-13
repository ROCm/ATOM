# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The progress thread advances MP lookups between scheduling passes.

Without it an async lookup moves one protocol phase (lookup ack, then
status) per `pump`, i.e. per forward step, so a ready answer is seen about
two steps late. The thread only drives the message-queue futures; the
scheduler thread still does every piece of bookkeeping in `_advance`.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from test_lmcache_mp_async_lookup import CHUNK, _Adapter, _Client

from atom.kv_transfer.offload.mp import lookup as mp_lookup


def _client(monkeypatch, adapter, progress_ms):
    monkeypatch.setenv("OFFLOAD_MP_LOOKUP_PROGRESS_MS", str(progress_ms))
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector_extra_config": {}},
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    return mp_lookup._MPLookupClient(
        adapter, config=config, timeout=10.0, poll_interval=0.01
    )


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def test_without_the_thread_a_lookup_needs_one_pump_per_phase(monkeypatch):
    fake = _Client(chunks=2)
    client = _client(monkeypatch, _Adapter(fake), 0)
    assert client._progress is None
    client.submit(list(range(10)), "req")
    time.sleep(0.02)  # nothing moves between passes
    assert fake.status_queries == []


def test_the_thread_reaches_the_answer_before_the_next_pass(monkeypatch):
    fake = _Client(chunks=2)
    adapter = _Adapter(fake)
    client = _client(monkeypatch, adapter, 1)
    client.submit(list(range(10)), "req")
    progress = client._async["req"][4]
    assert _wait(progress.done.is_set)  # status asked and answered off-thread
    assert len(fake.status_queries) == 1
    # Bookkeeping is still the scheduler thread's: one pass consumes it.
    assert client.poll("req") is True
    rid = mp_lookup._mp_session_id(client._config, "req")
    assert rid in adapter._pending_lookups
    assert adapter._lookup_results[rid] == 2 * CHUNK
    assert client.lookup(list(range(10)), "req") == 2 * CHUNK


def test_a_pending_status_is_requeried_with_backoff(monkeypatch):
    fake = _Client(chunks=3, status=[None, None, 3])
    client = _client(monkeypatch, _Adapter(fake), 1)
    client.submit(list(range(12)), "req")
    assert _wait(client._async["req"][4].done.is_set)
    assert len(fake.status_queries) == 3
    assert client.poll("req") is True
    assert client.lookup(list(range(12)), "req") == 3 * CHUNK


def test_a_failed_lookup_is_raised_on_the_scheduler_thread(monkeypatch):
    fake = _Client(chunks=2)

    def boom(timeout=None):
        raise RuntimeError("send failed")

    fake.lookup_future.result = boom
    client = _client(monkeypatch, _Adapter(fake), 1)
    client.submit(list(range(10)), "req")
    assert _wait(client._async["req"][4].done.is_set)
    assert fake.status_queries == []  # never reached the status phase
    assert client.poll("req") is True  # dropped through the usual path
    assert "req" not in client._async


def test_a_dropped_lookup_stops_being_tracked(monkeypatch):
    fake = _Client(chunks=2, lookup_done=False)
    client = _client(monkeypatch, _Adapter(fake), 1)
    client.submit(list(range(10)), "req")
    progress = client._async["req"][4]
    client._drop_failed("req")
    assert progress.cancelled
    assert _wait(lambda: not client._progress._active)


def test_answers_arrive_while_the_scheduler_thread_is_busy(monkeypatch):
    """The point of the thread: progress during a forward step."""
    fake = _Client(chunks=1, lookup_done=False)
    client = _client(monkeypatch, _Adapter(fake), 1)
    client.submit(list(range(8)), "req")
    busy = threading.Event()

    def forward():  # the scheduler thread, blocked in a forward
        fake.lookup_future.done = True  # the server acks mid-step
        busy.wait(0.2)

    forward()
    assert client._async["req"][4].done.is_set()
    assert client.poll("req") is True


@pytest.mark.parametrize("raw", ["-1", "nan"])
def test_the_interval_must_be_a_nonnegative_number(monkeypatch, raw):
    from atom.utils import envs

    monkeypatch.setenv("OFFLOAD_MP_LOOKUP_PROGRESS_MS", raw)
    with pytest.raises(ValueError):
        _ = envs.OFFLOAD_MP_LOOKUP_PROGRESS_MS
