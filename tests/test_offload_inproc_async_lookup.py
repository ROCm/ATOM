# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`AsyncInprocLookup`: LMCache's synchronous lookup client behind the async contract."""

import threading
import time

from atom.kv_transfer.offload.inproc_async_lookup import AsyncInprocLookup


class FakeBackend:
    """Records the order of calls and the thread they ran on; lookups can be held."""

    token_database = object()

    def __init__(self, hits=None):
        self.hits = hits or {}
        self.calls = []
        self.threads = set()
        self.gate = threading.Event()
        self.gate.set()

    def lookup(self, token_ids, lookup_id, request_configs=None):
        self.threads.add(threading.get_ident())
        self.gate.wait()
        self.calls.append(("lookup", lookup_id, len(token_ids), request_configs))
        return self.hits.get(lookup_id, 0)

    def clear_lookup_status(self, lookup_id):
        self.threads.add(threading.get_ident())
        self.calls.append(("clear", lookup_id))

    def close(self):
        self.calls.append(("close",))


def wait_until(cond, timeout=5.0):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout
        time.sleep(0.005)


def test_submit_then_consume_returns_the_hit_without_waiting():
    b = FakeBackend({"r1": 1024})
    c = AsyncInprocLookup(b)
    assert c.supports_async()
    assert c.submit([1] * 300, "r1", request_configs={"salt": "x"})
    assert not c.submit([1] * 300, "r1")  # one in flight per id
    wait_until(lambda: c.answered("r1"))
    assert not c.is_pending("r1") and c.poll("r1")
    assert "r1" in c.pending_ids()
    assert c.lookup([1] * 300, lookup_id="r1") == 1024
    assert "r1" not in c.pending_ids()
    assert b.calls == [("lookup", "r1", 300, {"salt": "x"})]
    assert threading.get_ident() not in b.threads  # never on the caller's thread
    assert c.take_unpins() == []  # consumed: the scheduler's dispatch owns the unpin
    c.close()


def test_in_flight_lookup_is_pending_until_it_lands():
    b = FakeBackend({"r1": 256})
    b.gate.clear()
    c = AsyncInprocLookup(b)
    c.submit([2] * 10, "r1")
    time.sleep(0.05)
    assert c.is_pending("r1") and not c.poll("r1") and not c.answered("r1")
    b.gate.set()
    wait_until(lambda: c.answered("r1"))
    c.close()


def test_lookup_without_submit_runs_on_the_backend_thread():
    b = FakeBackend({"r2": 512})
    c = AsyncInprocLookup(b)
    assert c.lookup([3] * 5, lookup_id="r2") == 512
    assert threading.get_ident() not in b.threads
    assert c.stats["sync"] == 1
    c.close()


def test_answered_but_released_lookup_is_unpinned_once():
    b = FakeBackend({"r3": 2048})
    c = AsyncInprocLookup(b)
    c.submit([4] * 5, "r3")
    wait_until(lambda: c.answered("r3"))
    c.clear_lookup_status("r3")  # request finished before admission consumed it
    assert c.take_unpins() == ["r3"]
    assert c.take_unpins() == []
    wait_until(lambda: ("clear", "r3") in b.calls)
    assert b.calls.index(("lookup", "r3", 5, None)) < b.calls.index(("clear", "r3"))
    c.close()


def test_discarded_in_flight_lookup_reports_its_hit_and_unpin_after_landing():
    b = FakeBackend({"r4": 4096})
    b.gate.clear()
    c = AsyncInprocLookup(b)
    c.submit([5] * 5, "r4")
    c.discard("r4")
    assert c.pop_discarded_hits() == {} and c.take_unpins() == []
    b.gate.set()
    wait_until(lambda: c.pop_discarded_hits() == {"r4": 4096} or "r4" not in c.pending)
    assert "r4" not in c.pending_ids()
    assert c.take_unpins() == ["r4"]
    c.close()


def test_consumer_timeout_is_a_non_answer_and_unpins_later():
    b = FakeBackend({"r5": 128})
    b.gate.clear()
    c = AsyncInprocLookup(b, wait_s=0.05)
    c.submit([6] * 5, "r5")
    assert c.lookup([6] * 5, lookup_id="r5") is None
    assert c.stats["timeouts"] == 1
    b.gate.set()
    wait_until(lambda: "r5" not in c.pending_ids())
    assert c.take_unpins() == ["r5"]
    c.close()


def test_close_closes_the_backend():
    b = FakeBackend()
    c = AsyncInprocLookup(b)
    c.close()
    assert ("close",) in b.calls
    assert not c.submit([1], "x")
