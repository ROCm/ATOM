# SPDX-License-Identifier: MIT
import time
from concurrent.futures import Future
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup
from atom.kv_transfer.offload.mooncake_store.layouts import (
    REPLAY_PAGE_CODEC_ID,
    mooncake_codec_id,
)


def test_arrival_offer_cannot_overtake_terminal_cleanup():
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
    from atom.kv_transfer.offload.mooncake_store.scheduler import (
        MooncakeStoreOffloadScheduler,
    )
    from atom.model_engine.sequence import Sequence, SequenceStatus

    lookup = AsyncStoreLookup(SimpleNamespace(lookup=lambda *a, **kw: 0), capacity=1)
    sched = MooncakeStoreOffloadScheduler.__new__(MooncakeStoreOffloadScheduler)
    ChunkedOffloadSchedulerBase.__init__(
        sched, MockConfig(kv_cache_block_size=16), chunk_size=256, lookup_client=lookup
    )
    entered, finishing, finished = Event(), Event(), Event()
    seq = Sequence(list(range(512)), 16)
    errors = []

    def lookup_len(value):
        entered.set()  # WAITING was observed; now let teardown run.
        assert finishing.wait(5)
        # Without the shared lock, deterministically finish cleanup before
        # the late submit. With it, teardown must wait for this offer.
        if not lookup.lock._is_owned():
            assert finished.wait(5)
        return value.num_prompt_tokens

    sched._lookup_len = lookup_len

    def finish():
        try:
            assert entered.wait(5)
            seq.status = SequenceStatus.ABORTED
            finishing.set()
            sched.request_finished(seq)
        except BaseException as exc:  # noqa: BLE001 - propagate thread failures below
            errors.append(exc)
        finally:
            finished.set()

    thread = Thread(target=finish)
    thread.start()
    try:
        sched.offer_lookups([seq])
        thread.join(5)
        assert not thread.is_alive()
        assert not errors
        assert lookup.pending_ids() == set()
        assert not lookup.discarded_hits
        # No permanent terminal tombstone: a new lifecycle can reuse the ID.
        assert lookup.submit(Sequence(list(range(512)), 16), str(seq.id))
    finally:
        finishing.set()
        thread.join(5)
        lookup.close()


def test_async_lookup_preserves_media_request_configs():
    received = []

    class Backend:
        def lookup(self, seq, lookup_id=None, request_configs=None):
            received.append((seq, lookup_id, request_configs))
            return 8192

    client = AsyncStoreLookup(Backend())
    seq = SimpleNamespace(id="media", cache_seed=77)
    configs = {"lmcache.tag.cache_salt": "77"}
    try:
        assert client.submit(seq, "media", request_configs=configs)
        for _ in range(200):
            if client.has_answer("media"):
                break
            time.sleep(0.005)
        assert client.lookup(seq, "media", request_configs=configs) == 8192
        assert received == [(seq, "media", configs)]
    finally:
        client.close()


def test_failed_lookup_warning_is_rate_limited_and_preserves_nonanswer(caplog):
    client = AsyncStoreLookup(SimpleNamespace(lookup=lambda *args, **kwargs: None))
    try:
        future = Future()
        future.set_exception(RuntimeError("Store unavailable"))
        client.pending["failed"] = future
        assert client.has_answer("failed") is False
        assert client.has_answer("failed") is False
        assert client.lookup(SimpleNamespace(id="failed")) is None
        warnings = [r for r in caplog.records if "Store lookup failed" in r.message]
        assert len(warnings) == 1
        assert "Store unavailable" in warnings[0].message
    finally:
        client.close()


def test_async_query_is_bounded_and_cancellable():
    gate = Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            assert gate.wait(2)
            return 8192

    client = AsyncStoreLookup(Backend(), capacity=2)
    try:
        assert client.submit(object(), "a")
        assert not client.submit(object(), "a")
        assert not client.has_answer("a")  # still in flight
        assert client.submit(object(), "b")
        assert not client.submit(object(), "c")
        client.clear_lookup_status("b")
        gate.set()
        for _ in range(200):
            if client.has_answer("a"):
                break
            time.sleep(0.005)
        assert client.has_answer("a")
        client.clear_lookup_status("a")
        assert not client.has_answer("a")
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
        config = SimpleNamespace(hf_config=SimpleNamespace(model_type=name))
        assert (offcfg.select_offload_layout(config) == "dsv41") == allowed
        if allowed:
            assert mooncake_codec_id(config) == REPLAY_PAGE_CODEC_ID


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
        lookup.clear_lookup_status("7")
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
        time.sleep(0.2)  # the worker is inside the first query, holding the client
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


def test_batch_renewal_skips_a_stalled_backend_without_per_request_wait(monkeypatch):
    from atom.kv_transfer.offload.mooncake_store import async_lookup

    entered, release = Event(), Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            entered.set()
            release.wait(10)
            return 1

        def renew(self, seq, start, end):
            return 7

    monkeypatch.setattr(async_lookup, "RENEW_TIMEOUT_S", 0.05)
    lookup = AsyncStoreLookup(Backend())
    try:
        lookup.submit(SimpleNamespace(id=1), "1")
        assert entered.wait(2)
        started = time.monotonic()
        for i in range(32):
            assert lookup.renew(SimpleNamespace(id=i), 0, 256) == 0
        assert time.monotonic() - started < 0.8
    finally:
        release.set()
        lookup.close()


def test_batch_renewal_skips_an_earlier_stalled_renewal(monkeypatch):
    from atom.kv_transfer.offload.mooncake_store import async_lookup

    release = Event()

    class Backend:
        def renew(self, seq, start, end):
            release.wait(10)
            return 7

    monkeypatch.setattr(async_lookup, "RENEW_TIMEOUT_S", 0.05)
    lookup = AsyncStoreLookup(Backend())
    try:
        assert lookup.renew(SimpleNamespace(id=1), 0, 256) == 0
        started = time.monotonic()
        for i in range(32):
            assert lookup.renew(SimpleNamespace(id=i), 0, 256) == 0
        assert time.monotonic() - started < 0.8
    finally:
        release.set()
        lookup.close()


def test_a_hung_store_call_does_not_keep_the_process_alive():
    """The workers are daemon threads: a process that closed its lookup while an
    RPC hangs still exits."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import threading, time
        from types import SimpleNamespace
        from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup

        class Backend:
            def lookup(self, seq, lookup_id=None):
                threading.Event().wait()  # never returns
            def renew(self, seq, start, end):
                threading.Event().wait()

        lookup = AsyncStoreLookup(Backend())
        lookup.submit(SimpleNamespace(id=1), "1")
        time.sleep(0.2)
        lookup.close()
        print("closed", flush=True)
        """)
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert "closed" in done.stdout, done.stderr
    assert time.monotonic() - started < 20


@pytest.mark.parametrize("pools", [1, 2])
def test_a_hung_store_master_does_not_keep_the_process_alive(pools):
    """The real lookup client, asking one shared pool or two per-NIC pools at
    once, with a master RPC that never returns: after close() the process
    exits. Two pools are asked on per-pool workers, which must be daemons."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(f"""
        import threading
        from atom.kv_transfer.offload.mooncake_store import scheduler as S
        from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup
        from atom.kv_transfer.offload.mooncake_store.config import (
            parse_mooncake_store_config,
        )
        from atom.model_engine.sequence import Sequence

        entered = threading.Event()

        class HungClient:
            def __init__(self, **kwargs):
                pass

            def exists(self, keys):
                entered.set()
                threading.Event().wait()  # the master never answers

        S.store_client.MooncakeStoreClient = HungClient
        if {pools} == 1:
            extra = {{"mooncake_store.master": "127.0.0.1:1",
                      "mooncake_store.metadata": "http://127.0.0.1:2/metadata"}}
        else:
            extra = {{"mooncake_store.pools": {{
                f"nic{{i}}": {{"master": f"127.0.0.{{i + 1}}:1",
                              "metadata": f"http://127.0.0.{{i + 1}}:2/metadata"}}
                for i in range({pools})}}}}
        cfg = parse_mooncake_store_config({{"kv_connector_extra_config": extra}})
        lookup = AsyncStoreLookup(
            S._StoreLookup(cfg, namespace="t", world=1, chunk_tokens=cfg.chunk_tokens)
        )
        assert lookup.submit(Sequence(list(range(4 * cfg.chunk_tokens)), 16), "1")
        assert entered.wait(10)
        lookup.close()
        print("closed", flush=True)
        """)
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert "closed" in done.stdout, done.stderr
    assert done.returncode == 0, done.stderr
    assert time.monotonic() - started < 20


def test_cancelled_lookups_do_not_pile_up_behind_a_stalled_rpc():
    import threading

    from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup

    gate = threading.Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            gate.wait(5)
            return 0

    lk = AsyncStoreLookup(Backend(), capacity=4)
    try:
        accepted = 0
        for i in range(50):
            if lk.submit(object(), f"s{i}"):
                accepted += 1
            lk.clear_lookup_status(f"s{i}")
        # One is being served; at most `capacity` more wait in the queue.
        assert accepted <= 5 + 1
    finally:
        gate.set()
        lk.close()


def test_multi_connector_fans_the_forward_stream_out_to_children():
    from atom.kv_transfer.disaggregation.base import KVConnectorBase
    from atom.kv_transfer.disaggregation.multi.multi_connector import MultiConnector

    class DefaultStreamConnector(KVConnectorBase):
        def register_kv_caches(self, *args, **kwargs): ...
        def start_load_kv(self, metadata): ...
        def get_finished(self):
            return set(), set()

    got = []
    m = MultiConnector.__new__(MultiConnector)
    m._connectors = [
        SimpleNamespace(set_forward_stream=got.append),
        DefaultStreamConnector(),
    ]
    m.set_forward_stream("stream-B")
    assert got == ["stream-B"]


def test_hashing_a_prompt_on_the_lookup_worker_never_blocks_a_token_append():
    """The async lookup hashes the prompt on its worker while the scheduler
    thread appends the request's tokens: the append must never meet an array
    that exports its buffer (BufferError fails the engine step)."""
    from atom.kv_transfer.offload.mooncake_store import scheduler as store_sched
    from atom.kv_transfer.offload.mooncake_store.config import (
        parse_mooncake_store_config,
    )
    from atom.model_engine.sequence import Sequence

    class Client:
        def __init__(self, **kwargs):
            pass

        def exists(self, keys):
            return [0] * len(keys)

    original = store_sched.store_client.MooncakeStoreClient
    store_sched.store_client.MooncakeStoreClient = Client
    try:
        cfg = parse_mooncake_store_config(
            {
                "kv_connector_extra_config": {
                    "mooncake_store.master": "127.0.0.1:1",
                    "mooncake_store.metadata": "http://127.0.0.1:2/metadata",
                }
            }
        )
        lookup = AsyncStoreLookup(
            store_sched._StoreLookup(
                cfg, namespace="t", world=1, chunk_tokens=cfg.chunk_tokens
            )
        )
        appends = 0
        for trial in range(20):
            seq = Sequence(list(range(1_000_000)), 16)
            sid = f"s{trial}"
            assert lookup.submit(seq, sid)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not lookup.answered(sid):
                seq.append_token(7)  # raised BufferError while hashing in place
                appends += 1
            assert lookup.answered(sid)
            lookup.clear_lookup_status(sid)
        assert appends > 0
        lookup.close()
    finally:
        store_sched.store_client.MooncakeStoreClient = original


def test_pending_store_fallback_never_blocks_the_admission_thread():
    gate = Event()
    entered = Event()

    class Backend:
        def lookup(self, seq, lookup_id=None):
            entered.set()
            assert gate.wait(5)
            return 8192

    client = AsyncStoreLookup(Backend(), capacity=4)
    seq = SimpleNamespace(id="pending")
    try:
        assert client.submit(seq, "pending")
        assert entered.wait(1)
        start = time.monotonic()
        for _ in range(4):
            assert client.lookup(seq, lookup_id="pending") is None
        assert time.monotonic() - start < 0.2
        assert client.is_pending("pending")
        gate.set()
        deadline = time.monotonic() + 1
        while not client.poll("pending") and time.monotonic() < deadline:
            time.sleep(0.005)
        assert client.lookup(seq, lookup_id="pending") == 8192
        assert not client.is_pending("pending")
    finally:
        gate.set()
        client.close()
