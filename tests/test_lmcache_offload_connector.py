# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import threading
import sys
import types
from types import SimpleNamespace

import pytest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from atom.kv_transfer.disaggregation import KVConnectorOutput, KVOutputAggregator
from atom.kv_transfer.offload.connector import (
    LMCacheOffloadConnector,
    LMCacheOffloadConnectorScheduler,
)
from atom.kv_transfer.offload.gpu_connector import ATOMKVByteCodec
from atom.model_engine.scheduler import Scheduler


class _LookupClient:
    def __init__(self, hit: int) -> None:
        self.hit = hit

    def lookup(self, token_ids, lookup_id):
        return self.hit


def _scheduler() -> LMCacheOffloadConnectorScheduler:
    sched = LMCacheOffloadConnectorScheduler.__new__(LMCacheOffloadConnectorScheduler)
    sched._config = SimpleNamespace()
    sched.kv_role = "offload"
    sched.block_size = 4
    sched.chunk_size = 4
    sched._lookup_client = _LookupClient(hit=0)
    sched._load_specs = {}
    sched._reqs_need_recv = {}
    sched._save_tracker = {}
    sched._save_inflight = set()
    sched._lookup_in_step = []
    return sched


@pytest.mark.parametrize("layout", ["segment", "segment_indexed"])
def test_segment_major_codec_roundtrip_noncontiguous_blocks(monkeypatch, layout):
    import torch
    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", layout)

    original = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(8 * 2 * 3, dtype=torch.uint8).reshape(8, 2, 3),
            v_cache=(torch.arange(8 * 4, dtype=torch.uint8).reshape(8, 4) + 51),
            k_scale=torch.arange(8, dtype=torch.uint8).reshape(8, 1) + 101,
            v_scale=torch.arange(8, dtype=torch.uint8).reshape(8, 1) + 151,
        ),
        "l1": SimpleNamespace(
            k_cache=(torch.arange(8 * 3, dtype=torch.uint8).reshape(8, 3) + 201),
            v_cache=(torch.arange(8 * 2, dtype=torch.uint8).reshape(8, 2) + 31),
            k_scale=None,
            v_scale=None,
        ),
    }
    kv_caches = {
        name: SimpleNamespace(
            k_cache=layer.k_cache.clone(),
            v_cache=layer.v_cache.clone(),
            k_scale=layer.k_scale.clone() if layer.k_scale is not None else None,
            v_scale=layer.v_scale.clone() if layer.v_scale is not None else None,
        )
        for name, layer in original.items()
    }

    codec = ATOMKVByteCodec(kv_caches)
    block_ids = [1, 2, 4, 6, 7]
    host = torch.empty(len(block_ids) * codec.bytes_per_block, dtype=torch.uint8)

    codec.gpu_to_host(host, block_ids)
    expected_calls = codec.segments_per_block * (3 if layout == "segment" else 2)
    assert codec.copy_calls_for_block_ids(block_ids) == expected_calls

    for layer in kv_caches.values():
        layer.k_cache.zero_()
        layer.v_cache.zero_()
        if layer.k_scale is not None:
            layer.k_scale.zero_()
        if layer.v_scale is not None:
            layer.v_scale.zero_()

    codec.host_to_gpu(host, block_ids)

    for name, layer in kv_caches.items():
        src = original[name]
        for bid in block_ids:
            assert torch.equal(layer.k_cache[bid], src.k_cache[bid])
            assert torch.equal(layer.v_cache[bid], src.v_cache[bid])
            if layer.k_scale is not None:
                assert torch.equal(layer.k_scale[bid], src.k_scale[bid])
            if layer.v_scale is not None:
                assert torch.equal(layer.v_scale[bid], src.v_scale[bid])


def test_segment_indexed_stitches_chunk_buffers(monkeypatch):
    import torch
    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(8 * 2 * 3, dtype=torch.uint8).reshape(8, 2, 3),
            v_cache=(torch.arange(8 * 4, dtype=torch.uint8).reshape(8, 4) + 51),
            k_scale=torch.arange(8, dtype=torch.uint8).reshape(8, 1) + 101,
            v_scale=torch.arange(8, dtype=torch.uint8).reshape(8, 1) + 151,
        ),
        "l1": SimpleNamespace(
            k_cache=(torch.arange(8 * 3, dtype=torch.uint8).reshape(8, 3) + 201),
            v_cache=(torch.arange(8 * 2, dtype=torch.uint8).reshape(8, 2) + 31),
            k_scale=None,
            v_scale=None,
        ),
    }
    codec = ATOMKVByteCodec(kv_caches)
    chunks = [[1, 2], [4], [6, 7]]
    flat_ids = [bid for bids in chunks for bid in bids]
    direct = torch.empty(len(flat_ids) * codec.bytes_per_block, dtype=torch.uint8)
    codec.gpu_to_host(direct, flat_ids)

    chunk_buffers = []
    for bids in chunks:
        host = torch.empty(len(bids) * codec.bytes_per_block, dtype=torch.uint8)
        codec.gpu_to_host(host, bids)
        chunk_buffers.append(host)

    stitched = torch.empty_like(direct)
    codec.stitch_chunk_buffers(stitched, chunk_buffers, [len(b) for b in chunks])

    assert torch.equal(stitched, direct)


def test_full_prompt_hit_is_clamped_before_load_spec():
    sched = _scheduler()
    sched._lookup_client = _LookupClient(hit=8)
    seq = SimpleNamespace(
        id=123,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        num_cached_tokens=0,
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)

    assert need == 7
    assert should_park is True
    assert sched._load_specs[str(seq.id)].lmcache_cached_tokens == 7


def test_load_exception_is_reported_as_failed_recving():
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._done_save = set()
    conn._failed_load = set()
    req = SimpleNamespace(req_id=42)

    def boom(_req):
        raise RuntimeError("load failed")

    conn._guard("load", boom, req)

    assert conn._done_load == set()
    assert conn._failed_load == {42}


def test_aggregator_emits_failed_recving_if_any_worker_failed():
    agg = KVOutputAggregator(world_size=2)

    result = agg.aggregate(
        [
            KVConnectorOutput(finished_recving={77}),
            KVConnectorOutput(failed_recving={77}),
        ]
    )

    assert result.finished_recving == set()
    assert result.failed_recving == {77}


def test_aggregator_failure_overrides_late_success():
    agg = KVOutputAggregator(world_size=2)

    result = agg.aggregate(
        [
            KVConnectorOutput(finished_recving={77}, failed_recving={77}),
            KVConnectorOutput(finished_recving={77}),
        ]
    )

    assert result.finished_recving == set()
    assert result.failed_recving == {77}
    assert agg.pending_count == (0, 0)


def test_save_inflight_defers_free_until_save_finishes():
    sched = _scheduler()
    seq = SimpleNamespace(
        id=9,
        token_ids=list(range(8)),
        block_table=[3, 4],
        num_prompt_tokens=8,
        prefix_hashes_published=True,
    )
    sched._save_tracker[str(seq.id)] = [seq, 0]

    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    assert meta.requests[0].save_spec is not None
    assert sched.should_defer_free(seq) is True

    sched.save_finished(seq.id)

    assert sched.should_defer_free(seq) is False


def test_finished_saving_releases_deferred_free_with_string_req_id():
    class _BlockManager:
        def __init__(self) -> None:
            self.deallocated = []

        def deallocate(self, seq) -> None:
            self.deallocated.append(seq.id)

    class _Connector:
        is_producer = False

        def __init__(self) -> None:
            self.inflight = {"9"}

        def save_finished(self, req_id) -> None:
            self.inflight.discard(str(req_id))

        def should_defer_free(self, seq) -> bool:
            return str(seq.id) in self.inflight

    sched = Scheduler.__new__(Scheduler)
    sched.block_manager = _BlockManager()
    sched.kv_connector = _Connector()
    seq = SimpleNamespace(id=9)
    sched.deferred_free_blocks = {seq.id: seq}

    sched._update_from_kv_xfer_finished(KVConnectorOutput(finished_saving={"9"}))

    assert sched.block_manager.deallocated == [9]
    assert sched.deferred_free_blocks == {}


def test_finished_recv_matches_string_req_id():
    sched = Scheduler.__new__(Scheduler)
    sched.finished_recving_kv_req_ids = ["123"]

    assert sched._update_waiting_for_remote_kv(SimpleNamespace(id=123)) is True
    assert sched.finished_recving_kv_req_ids == []
