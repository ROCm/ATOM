# SPDX-License-Identifier: MIT
"""Arrival cancellation, lookup cost, and finite copy completion contracts."""

import threading
from types import MethodType, SimpleNamespace

import pytest

from atom.kv_transfer.offload import atom_lmcache_staging
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.lookup_client import LookupClient
from atom.kv_transfer.offload.mooncake_store.async_lookup import AsyncStoreLookup
from atom.kv_transfer.offload.mooncake_store.worker import MooncakeStoreOffloadConnector
from atom.kv_transfer.offload.mp.deployment import require_chunk_matches_server


def test_terminal_arrival_cancels_queued_lookup_but_live_discard_does_not():
    started, release = threading.Event(), threading.Event()

    class Backend(LookupClient):
        def lookup(self, _seq, lookup_id):
            assert lookup_id in ("1", "3")
            started.set()
            assert release.wait(5)
            return 0

        def close(self):
            return None

    client = AsyncStoreLookup(Backend(), capacity=2)
    first, terminal = SimpleNamespace(id=1), SimpleNamespace(id=2)
    try:
        assert client.submit(first, "1") and started.wait(1)
        assert client.submit(terminal, "2")
        future = client.pending["2"]
        client.discard("2")  # A live HBM-detached query is not cancelled.
        assert not future.cancelled() and "2" in client.pending
        state = SimpleNamespace(
            _load_lifecycles={},
            _lookup_client=client,
            _active_load_operations={},
            _lookup_defer_since={},
            _save_tracker={},
            _load_specs={},
            _reqs_need_recv={},
            _handoff_loads=set(),
            _load_save_floors={},
            _hit_save_floors={},
            _release_failed_load_attempt=lambda *_: None,
            _forget_tier_hit=lambda *_: None,
        )
        for name in ("_clear_pending_load", "_release_lookup"):
            setattr(
                state,
                name,
                MethodType(getattr(ChunkedOffloadSchedulerBase, name), state),
            )
        ChunkedOffloadSchedulerBase.request_finished(state, terminal)
        assert future.cancelled() and "2" not in client.pending
        assert client.submit(SimpleNamespace(id=3), "3")
    finally:
        release.set()
        client.close()


def test_answered_lookup_does_not_copy_pending_ids():
    copies = []
    client = SimpleNamespace(
        answered=lambda _sid: True,
        pending_ids=lambda: copies.append(set(map(str, range(4096)))),
        is_pending=lambda _sid: False,
    )
    state = SimpleNamespace(_lookup_results={}, _lookup_client=client)
    for _ in range(64):
        assert ChunkedOffloadSchedulerBase.lookup_ready(state, SimpleNamespace(id=1))
    assert not copies


def test_in_place_copy_waits_for_its_event_before_later_forward(monkeypatch):
    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("whole-stream"))

    class Completion:
        def record(self, target):
            assert target is stream
            calls.extend(("copy-frontier", "later-forward-enqueued"))

        def synchronize(self):
            calls.append("copy-frontier-complete")

    monkeypatch.setattr(
        atom_lmcache_staging.torch.cuda, "Event", lambda **_: Completion()
    )
    codec = SimpleNamespace(
        device="GPU",
        prepare_block_id_groups=lambda *_, **__: None,
        chunk_major_device_buffer_to_gpu_prepared=lambda *_, **__: calls.append(
            "scatter"
        ),
        gpu_to_chunk_major_device_buffer_prepared=lambda *_, **__: calls.append(
            "gather"
        ),
    )
    state = SimpleNamespace(
        _codec=codec,
        chunk_size=256,
        virtual_block_size=256,
        _in_place_stream=lambda: stream,
        _gpu_connector=SimpleNamespace(_ranges_to_block_ids=lambda *_, **__: [[1]]),
    )
    MooncakeStoreOffloadConnector._copy_in_place(
        state, object(), range(1), [1], pack=False
    )
    assert calls == [
        "scatter",
        "copy-frontier",
        "later-forward-enqueued",
        "copy-frontier-complete",
    ]


@pytest.mark.parametrize("policy", ["prefix_spread", "share_cache"])
def test_prefix_policy_rejects_mismatched_mp_chunk(policy):
    config = SimpleNamespace(
        dp_load_balance=policy,
        data_parallel_size=2,
        pipeline_parallel_size=1,
        enable_rapidserve=False,
        kv_transfer_config={
            "kv_connector": "lmcache_mp",
            "kv_connector_extra_config": {"lmcache.chunk_size": 512},
        },
    )
    with pytest.raises(ValueError, match="configured 512.*server 1024"):
        require_chunk_matches_server(
            SimpleNamespace(lmcache_tokens_per_chunk=1024), config
        )
    require_chunk_matches_server(SimpleNamespace(lmcache_tokens_per_chunk=512), config)
