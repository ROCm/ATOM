# SPDX-License-Identifier: MIT
from threading import Event
import time
from types import SimpleNamespace
from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup
from atom.kv_transfer.offload.mooncake_store.config import uses_replay_pages
from atom.kv_transfer.disaggregation.multi.multi_connector import MultiConnectorScheduler


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
            time.sleep(.005)
        assert client.take_answer("a", lambda: None) == 8192
        assert client.take_answer("a", lambda: None) is None
        assert not client.pending_ids()
    finally:
        gate.set()
        client.close()
    assert not client.submit(object(), "after-close")


def test_v41_page_path_does_not_admit_other_hybrid_models():
    for name, allowed in [("deepseek_v41_text", True), ("kimi_linear", False),
                          ("deepseek_v4", False), ("qwen3_next", False)]:
        assert uses_replay_pages(SimpleNamespace(hf_config=SimpleNamespace(model_type=name))) == allowed


def test_multi_prefetch_only_on_producer_not_decode_receiver():
    tier = SimpleNamespace(is_offload=True, supports_async_lookup=True, chunk_size=256,
                           lookup_ready=lambda s: True, take_ready_lookup=lambda s: True)
    m = MultiConnectorScheduler.__new__(MultiConnectorScheduler)
    m._connectors = [SimpleNamespace(is_offload=False), tier]
    m.is_producer = True
    assert m.supports_async_lookup and m.chunk_size == 256
    assert m.lookup_ready(object()) and m.take_ready_lookup(object())
    m.is_producer = False
    assert not m.supports_async_lookup
    assert not m.lookup_ready(object())
