# SPDX-License-Identifier: MIT
import time
from threading import Event
from types import SimpleNamespace

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup
from atom.kv_transfer.offload.mooncake_store.config import uses_replay_pages


def test_async_query_is_bounded_cancellable_and_consumed_once():
    gate = Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            assert gate.wait(2)
            return 8192

    client = AsyncStoreLookup(Backend(), capacity=2)
    try:
        assert client.submit(object(), "a")
        assert not client.submit(object(), "a")
        assert client.take_answer("a", lambda: None) is None
        assert client.submit(object(), "b")
        assert not client.submit(object(), "c")
        client.clear_lookup_status("b")
        gate.set()
        for _ in range(200):
            if client.has_answer("a"):
                break
            time.sleep(0.005)
        assert client.take_answer("a", lambda: None) == 8192
        assert client.take_answer("a", lambda: None) is None
        assert not client.pending_ids()
    finally:
        gate.set()
        client.close()
    assert not client.submit(object(), "after-close")


def test_v41_page_path_does_not_admit_other_hybrid_models():
    for name, allowed in [
        ("deepseek_v41_text", True),
        ("kimi_linear", False),
        ("deepseek_v4", False),
        ("qwen3_next", False),
    ]:
        assert (
            uses_replay_pages(
                SimpleNamespace(hf_config=SimpleNamespace(model_type=name))
            )
            == allowed
        )


def test_multi_prefetch_only_on_producer_not_decode_receiver():
    tier = SimpleNamespace(
        is_offload=True,
        chunk_size=256,
        lookup_ready=lambda s: True,
    )
    m = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    m._connectors = [SimpleNamespace(is_offload=False), tier]
    m.is_producer = True
    assert m.chunk_size == 256
    assert m.lookup_ready(object())
    m.is_producer = False
    assert m.chunk_size == 0
    assert not m.lookup_ready(object())


def test_answered_is_read_only_and_marks_lookup_ready():
    gate = Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            gate.wait(5)
            return 512

    from atom.kv_transfer.offload.chunked_scheduler import (
        ChunkedOffloadSchedulerBase,
    )

    lookup = AsyncStoreLookup(Backend())
    sched = ChunkedOffloadSchedulerBase.__new__(ChunkedOffloadSchedulerBase)
    sched._lookup_client = lookup
    sched._lookup_results = {}
    seq = SimpleNamespace(id=7)
    try:
        assert lookup.submit(seq, "7")
        assert not lookup.answered("7")
        assert not sched.lookup_ready(seq)  # in flight
        gate.set()
        deadline = time.time() + 5
        while not lookup.answered("7") and time.time() < deadline:
            time.sleep(0.01)
        # Answered but not taken: still listed as pending, and early-loadable.
        assert lookup.is_pending("7") and lookup.answered("7")
        assert sched.lookup_ready(seq)
        assert lookup.answered("7")  # reading it consumed nothing
        assert lookup.take_answer("7", None) == 512
        assert not sched.lookup_ready(seq)
    finally:
        lookup.close()


def test_a_detached_lookup_is_reconciled_and_dropped_each_step():
    """The Store detaches lookups too; its detached requests must not pile up."""
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase

    sched = ChunkedOffloadSchedulerBase.__new__(ChunkedOffloadSchedulerBase)
    seq = SimpleNamespace(id=1)
    lowered = []
    sched._lookup_client = SimpleNamespace(
        pump=lambda: None,
        pop_discarded_hits=lambda: {"1": 512},
        pending_ids=lambda: set(),
    )
    sched._detached_lookups["1"] = seq
    sched._chunk_floor = lambda tokens: tokens // 256 * 256
    sched._lower_save_floor = lambda sid, s, tokens: lowered.append(tokens) or True
    sched._perf_bump = lambda name: None
    sched._reconcile_lookups_each_step()
    assert lowered == [512]
    assert sched._detached_lookups == {}  # answered: forgotten, nothing piles up


def test_renew_does_not_queue_behind_lookups_and_is_bounded(monkeypatch):
    import threading

    from atom.kv_transfer.offload.mooncake_store import async_lookup
    from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup

    release = threading.Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            release.wait(5)
            return 1

        def renew(self, seq, start, end):
            return 7

    monkeypatch.setattr(async_lookup, "RENEW_TIMEOUT_S", 0.2)
    lookup = AsyncStoreLookup(Backend())
    try:
        for i in range(3):  # one query is being served, two are queued behind it
            lookup.submit(SimpleNamespace(id=i), str(i))
        started = time.monotonic()
        assert lookup.renew(SimpleNamespace(id=9), 0, 256) == 0  # bounded wait
        assert time.monotonic() - started < 2.0
        release.set()
        time.sleep(0.3)
        assert lookup.renew(SimpleNamespace(id=9), 0, 256) == 7  # nothing in its way
    finally:
        release.set()
        lookup.close()


def test_close_does_not_wait_for_a_hung_query():
    import threading

    from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup

    release = threading.Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            release.wait(10)

    lookup = AsyncStoreLookup(Backend())
    lookup.submit(SimpleNamespace(id=1), "1")
    time.sleep(0.1)
    started = time.monotonic()
    lookup.close()
    release.set()
    assert time.monotonic() - started < 2.0
