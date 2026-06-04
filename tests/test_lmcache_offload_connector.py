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
from atom.kv_transfer.offload import connector as offload_connector_mod
from atom.kv_transfer.offload.gpu_connector import ATOMKVByteCodec
from atom.kv_transfer.offload.lmcache_compat import (
    ATOMLMCacheGPUConnector,
    ATOMRawBytesLMCacheMetadata,
)
from atom.model_engine.scheduler import Scheduler


class _LookupClient:
    def __init__(self, hit: int) -> None:
        self.hit = hit
        self.cleared = []

    def lookup(self, token_ids, lookup_id):
        return self.hit

    def clear_lookup_status(self, lookup_id):
        self.cleared.append(lookup_id)


def _scheduler() -> LMCacheOffloadConnectorScheduler:
    sched = LMCacheOffloadConnectorScheduler.__new__(LMCacheOffloadConnectorScheduler)
    sched._config = SimpleNamespace()
    sched.kv_role = "offload"
    sched.block_size = 4
    sched.chunk_size = 4
    sched._lookup_client = _LookupClient(hit=0)
    sched._load_specs = {}
    sched._reqs_need_recv = {}
    sched._load_save_floors = {}
    sched._hit_save_floors = {}
    sched._save_tracker = {}
    sched._save_inflight = set()
    sched._lookup_in_step = []
    sched._handoff_loads = set()
    sched._allow_unaligned_handoff = False
    sched._min_load_tokens = 0
    sched._lock = threading.Lock()
    sched._done_load = set()
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

    split_buffers = [torch.empty_like(buf) for buf in chunk_buffers]
    codec.split_request_buffer(stitched, split_buffers, [len(b) for b in chunks])
    for actual, expected in zip(split_buffers, chunk_buffers):
        assert torch.equal(actual, expected)


def test_raw_bytes_metadata_shapes_are_block_rounded():
    import torch

    if not hasattr(torch, "Size"):
        pytest.skip("real torch is unavailable")

    base = SimpleNamespace(chunk_size=8)
    base.is_first_rank = lambda: True
    meta = ATOMRawBytesLMCacheMetadata(
        base,
        atom_block_size=4,
        bytes_per_block=32,
    )

    assert meta.get_dtypes() == [torch.uint8]
    assert meta.get_shapes(8) == [torch.Size((64,))]
    assert meta.get_shapes(6) == [torch.Size((64,))]
    assert meta.get_shapes(4) == [torch.Size((32,))]
    assert meta.get_shapes() == [torch.Size((64,))]


def test_raw_bytes_metadata_rejects_unaligned_chunk_size():
    import torch

    if not hasattr(torch, "Size"):
        pytest.skip("real torch is unavailable")

    base = SimpleNamespace(chunk_size=10)
    with pytest.raises(ValueError, match="chunk size must be divisible"):
        ATOMRawBytesLMCacheMetadata(
            base,
            atom_block_size=4,
            bytes_per_block=32,
        )


def test_codec_device_buffer_roundtrip_noncontiguous_blocks(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "block")
    original = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(8 * 2 * 3, dtype=torch.uint8).reshape(8, 2, 3),
            v_cache=(torch.arange(8 * 4, dtype=torch.uint8).reshape(8, 4) + 51),
            k_scale=None,
            v_scale=None,
        ),
        "l1": SimpleNamespace(
            k_cache=(torch.arange(8 * 3, dtype=torch.uint8).reshape(8, 3) + 101),
            v_cache=(torch.arange(8 * 2, dtype=torch.uint8).reshape(8, 2) + 151),
            k_scale=None,
            v_scale=None,
        ),
    }
    kv_caches = {
        name: SimpleNamespace(
            k_cache=layer.k_cache.clone(), v_cache=layer.v_cache.clone()
        )
        for name, layer in original.items()
    }
    for layer in kv_caches.values():
        layer.k_scale = None
        layer.v_scale = None

    codec = ATOMKVByteCodec(kv_caches)
    block_ids = [1, 3, 4, 7]
    device_buf = torch.empty(
        len(block_ids) * codec.bytes_per_block,
        dtype=torch.uint8,
        device=codec.device,
    )

    codec.gpu_to_device_buffer(device_buf, block_ids)
    for layer in kv_caches.values():
        layer.k_cache.zero_()
        layer.v_cache.zero_()
    codec.device_buffer_to_gpu(device_buf, block_ids)

    for name, layer in kv_caches.items():
        src = original[name]
        for bid in block_ids:
            assert torch.equal(layer.k_cache[bid], src.k_cache[bid])
            assert torch.equal(layer.v_cache[bid], src.v_cache[bid])


def test_lmcache_connector_maps_token_ranges_to_block_ids(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "block")
    original = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2),
            v_cache=(torch.arange(6 * 3, dtype=torch.uint8).reshape(6, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=original["l0"].k_cache.clone(),
            v_cache=original["l0"].v_cache.clone(),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=8)
    memory_obj = SimpleNamespace(
        tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
    )

    connector.batched_from_gpu(
        [memory_obj],
        [4],
        [12],
        block_ids=[0, 1, 2, 3, 4, 5],
    )

    kv_caches["l0"].k_cache.zero_()
    kv_caches["l0"].v_cache.zero_()
    connector.batched_to_gpu(
        [memory_obj],
        [4],
        [12],
        block_ids=[0, 1, 2, 3, 4, 5],
    )

    for bid in [1, 2]:
        assert torch.equal(kv_caches["l0"].k_cache[bid], original["l0"].k_cache[bid])
        assert torch.equal(kv_caches["l0"].v_cache[bid], original["l0"].v_cache[bid])
    assert torch.count_nonzero(kv_caches["l0"].k_cache[0]) == 0
    assert torch.count_nonzero(kv_caches["l0"].v_cache[0]) == 0


def test_lmcache_connector_fused_chunk_fastpath_uses_chunk_major(monkeypatch):
    from contextlib import nullcontext

    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "2")
    original = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2),
            v_cache=(torch.arange(6 * 3, dtype=torch.uint8).reshape(6, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=original["l0"].k_cache.clone(),
            v_cache=original["l0"].v_cache.clone(),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=8)
    monkeypatch.setattr(connector, "_can_use_fused_chunk_major", lambda: True)
    orig_pack = codec.gpu_to_chunk_major_device_buffer
    orig_unpack = codec.chunk_major_device_buffer_to_gpu

    pack_groups = []
    unpack_groups = []
    slot_requests = []

    monkeypatch.setattr(
        codec,
        "gpu_to_chunk_major_device_buffer",
        lambda device_buf, block_id_groups, stream=None: (
            pack_groups.append([list(group) for group in block_id_groups]),
            orig_pack(device_buf, block_id_groups, stream=None),
        )[-1],
    )
    monkeypatch.setattr(
        codec,
        "chunk_major_device_buffer_to_gpu",
        lambda device_buf, block_id_groups, stream=None: (
            unpack_groups.append([list(group) for group in block_id_groups]),
            orig_unpack(device_buf, block_id_groups, stream=None),
        )[-1],
    )
    orig_ensure_slot = connector._ensure_slot

    def _ensure_slot(slot, nbytes):
        device_buf = orig_ensure_slot(slot, nbytes)
        slot_requests.append((nbytes, int(slot.tensor.numel())))
        return device_buf

    monkeypatch.setattr(connector, "_ensure_slot", _ensure_slot)

    class _FakeEvent:
        def record(self, stream) -> None:
            pass

    class _FakeStream:
        def wait_event(self, event) -> None:
            pass

        def synchronize(self) -> None:
            pass

    class _FakeState:
        def __init__(self) -> None:
            self.use_cuda = True
            self.pack_stream = _FakeStream()
            self.copy_stream = _FakeStream()
            self.next_slot = 0
            self.slots = [
                SimpleNamespace(
                    tensor=None,
                    ready_event=_FakeEvent(),
                    free_event=_FakeEvent(),
                    free_event_valid=False,
                ),
                SimpleNamespace(
                    tensor=None,
                    ready_event=_FakeEvent(),
                    free_event=_FakeEvent(),
                    free_event_valid=False,
                ),
            ]

        def stream_ctx(self, stream):
            return nullcontext()

    fake_state = _FakeState()
    monkeypatch.setattr(connector, "_thread_state", lambda: fake_state)
    memory_objs = [
        SimpleNamespace(
            tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
        ),
        SimpleNamespace(
            tensor=torch.empty(1 * codec.bytes_per_block, dtype=torch.uint8)
        ),
    ]

    connector.batched_from_gpu(
        memory_objs,
        [4, 12],
        [12, 16],
        block_ids=[0, 1, 2, 3, 4, 5],
    )

    expected0 = torch.cat(
        [
            original["l0"].k_cache[[1, 2]].reshape(-1),
            original["l0"].v_cache[[1, 2]].reshape(-1),
        ]
    )
    expected1 = torch.cat(
        [
            original["l0"].k_cache[[3]].reshape(-1),
            original["l0"].v_cache[[3]].reshape(-1),
        ]
    )
    assert connector.last_fastpath() == "fused_chunk"
    transfer_stats = connector.last_transfer_stats()
    assert transfer_stats["chunks"] == 2
    assert transfer_stats["groups"] == 1
    assert transfer_stats["max_chunk_bytes"] == 2 * codec.bytes_per_block
    assert transfer_stats["max_group_bytes"] == 3 * codec.bytes_per_block
    assert transfer_stats["total_bytes"] == 3 * codec.bytes_per_block
    assert transfer_stats["gpu_staging_chunk_bytes"] == 2 * codec.bytes_per_block
    assert transfer_stats["gpu_staging_group_chunks"] == 2
    assert transfer_stats["gpu_staging_capacity_bytes"] == 4 * codec.bytes_per_block
    assert transfer_stats["gpu_staging_slots"] == 1
    assert transfer_stats["transfer_ms"] >= 0
    assert pack_groups == [[[1, 2], [3]]]
    assert all(nbytes <= 4 * codec.bytes_per_block for nbytes, _ in slot_requests)
    assert all(capacity == 4 * codec.bytes_per_block for _, capacity in slot_requests)
    assert torch.equal(memory_objs[0].tensor, expected0)
    assert torch.equal(memory_objs[1].tensor, expected1)

    kv_caches["l0"].k_cache.zero_()
    kv_caches["l0"].v_cache.zero_()
    connector.batched_to_gpu(
        memory_objs,
        [4, 12],
        [12, 16],
        block_ids=[0, 1, 2, 3, 4, 5],
    )

    assert connector.last_fastpath() == "fused_chunk"
    assert unpack_groups == [[[1, 2], [3]]]
    for bid in [1, 2, 3]:
        assert torch.equal(kv_caches["l0"].k_cache[bid], original["l0"].k_cache[bid])
        assert torch.equal(kv_caches["l0"].v_cache[bid], original["l0"].v_cache[bid])
    assert torch.count_nonzero(kv_caches["l0"].k_cache[0]) == 0
    assert torch.count_nonzero(kv_caches["l0"].v_cache[0]) == 0


def test_lmcache_connector_fallback_staging_is_chunk_bounded(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "1")
    original = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(8 * 2, dtype=torch.uint8).reshape(8, 2),
            v_cache=(torch.arange(8 * 3, dtype=torch.uint8).reshape(8, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=original["l0"].k_cache.clone(),
            v_cache=original["l0"].v_cache.clone(),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=8)
    cap = 2 * codec.bytes_per_block
    slot_requests = []
    host_requests = []
    orig_ensure_slot = connector._ensure_slot
    orig_ensure_host_tmp = connector._ensure_host_tmp

    def _ensure_slot(slot, nbytes):
        device_buf = orig_ensure_slot(slot, nbytes)
        slot_requests.append((nbytes, int(slot.tensor.numel())))
        return device_buf

    def _ensure_host_tmp(state, nbytes):
        host_buf = orig_ensure_host_tmp(state, nbytes)
        host_requests.append((nbytes, int(state.host_tmp.numel())))
        return host_buf

    monkeypatch.setattr(connector, "_ensure_slot", _ensure_slot)
    monkeypatch.setattr(connector, "_ensure_host_tmp", _ensure_host_tmp)
    memory_objs = [
        SimpleNamespace(
            tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
        ),
        SimpleNamespace(
            tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
        ),
        SimpleNamespace(
            tensor=torch.empty(1 * codec.bytes_per_block, dtype=torch.uint8)
        ),
    ]

    connector.batched_from_gpu(
        memory_objs,
        [0, 8, 16],
        [8, 16, 20],
        block_ids=list(range(8)),
    )

    assert connector.last_fastpath() == "chunk"
    assert connector.last_transfer_stats()["chunks"] == 3
    assert connector.last_transfer_stats()["max_chunk_bytes"] == cap
    assert all(nbytes <= cap for nbytes, _ in slot_requests)
    assert all(capacity == cap for _, capacity in slot_requests)
    assert all(nbytes <= cap for nbytes, _ in host_requests)
    assert all(capacity == cap for _, capacity in host_requests)

    kv_caches["l0"].k_cache.zero_()
    kv_caches["l0"].v_cache.zero_()
    connector.batched_to_gpu(
        memory_objs,
        [0, 8, 16],
        [8, 16, 20],
        block_ids=list(range(8)),
    )

    for bid in range(5):
        assert torch.equal(kv_caches["l0"].k_cache[bid], original["l0"].k_cache[bid])
        assert torch.equal(kv_caches["l0"].v_cache[bid], original["l0"].v_cache[bid])
    assert torch.count_nonzero(kv_caches["l0"].k_cache[5]) == 0
    assert torch.count_nonzero(kv_caches["l0"].v_cache[5]) == 0


def test_lmcache_connector_release_covers_fallback_chunks(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "1")
    monkeypatch.setenv("OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER", "1")
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=(torch.arange(4 * 3, dtype=torch.uint8).reshape(4, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=8)
    memory_objs = [
        SimpleNamespace(
            tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
        ),
        SimpleNamespace(
            tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
        ),
    ]

    connector.batched_from_gpu(
        memory_objs,
        [0, 8],
        [8, 16],
        block_ids=list(range(4)),
    )

    state = connector._thread_state()
    assert state.host_tmp is not None
    assert int(state.host_tmp.numel()) == 2 * codec.bytes_per_block
    assert all(slot.tensor is None for slot in state.slots)


def test_lmcache_connector_rejects_oversized_memory_obj(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=(torch.arange(4 * 3, dtype=torch.uint8).reshape(4, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=4)
    memory_obj = SimpleNamespace(
        tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
    )

    with pytest.raises(ValueError, match="single MemoryObj exceeds"):
        connector.batched_from_gpu(
            [memory_obj],
            [0],
            [8],
            block_ids=list(range(4)),
        )


def test_lmcache_connector_respects_staging_slot_env(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    monkeypatch.setenv("OFFLOAD_GPU_STAGING_SLOTS", "2")
    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "3")
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(2 * 2, dtype=torch.uint8).reshape(2, 2),
            v_cache=torch.arange(2 * 3, dtype=torch.uint8).reshape(2, 3),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=4)

    assert connector.staging_slots == 2
    assert connector.gpu_staging_group_chunks == 3
    assert connector.gpu_staging_capacity_bytes == 3 * connector.gpu_staging_chunk_bytes
    assert len(connector._thread_state().slots) == 2


def test_lmcache_connector_default_staging_group_chunks_is_two(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    monkeypatch.delenv("OFFLOAD_GPU_STAGING_CHUNKS", raising=False)
    monkeypatch.delenv("OFFLOAD_GPU_STAGING_MAX_BYTES", raising=False)
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(2 * 2, dtype=torch.uint8).reshape(2, 2),
            v_cache=torch.arange(2 * 3, dtype=torch.uint8).reshape(2, 3),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=4)

    assert connector.gpu_staging_group_chunks == 2
    assert connector.gpu_staging_capacity_bytes == 2 * connector.gpu_staging_chunk_bytes


def test_codec_chunk_major_device_buffer_layout(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    original = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=(torch.arange(4 * 3, dtype=torch.uint8).reshape(4, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=original["l0"].k_cache.clone(),
            v_cache=original["l0"].v_cache.clone(),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    block_id_groups = [[0, 1], [2, 3]]
    device_buf = torch.empty(
        4 * codec.bytes_per_block,
        dtype=torch.uint8,
        device=codec.device,
    )

    codec.gpu_to_chunk_major_device_buffer(device_buf, block_id_groups)

    expected = torch.cat(
        [
            original["l0"].k_cache[[0, 1]].reshape(-1),
            original["l0"].v_cache[[0, 1]].reshape(-1),
            original["l0"].k_cache[[2, 3]].reshape(-1),
            original["l0"].v_cache[[2, 3]].reshape(-1),
        ]
    )
    assert torch.equal(device_buf.cpu(), expected.cpu())

    kv_caches["l0"].k_cache.zero_()
    kv_caches["l0"].v_cache.zero_()
    codec.chunk_major_device_buffer_to_gpu(device_buf, block_id_groups)

    assert torch.equal(kv_caches["l0"].k_cache, original["l0"].k_cache)
    assert torch.equal(kv_caches["l0"].v_cache, original["l0"].v_cache)


def test_codec_chunk_major_handles_tail_and_sparse_blocks(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    original = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2),
            v_cache=(torch.arange(6 * 4, dtype=torch.uint8).reshape(6, 4) + 31),
            k_scale=(torch.arange(6, dtype=torch.uint8).reshape(6, 1) + 101),
            v_scale=None,
        ),
        "l1": SimpleNamespace(
            k_cache=(torch.arange(6 * 3, dtype=torch.uint8).reshape(6, 3) + 151),
            v_cache=(torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2) + 201),
            k_scale=None,
            v_scale=None,
        ),
    }
    kv_caches = {
        name: SimpleNamespace(
            k_cache=layer.k_cache.clone(),
            v_cache=layer.v_cache.clone(),
            k_scale=layer.k_scale.clone() if layer.k_scale is not None else None,
            v_scale=None,
        )
        for name, layer in original.items()
    }
    codec = ATOMKVByteCodec(kv_caches)
    block_id_groups = [[4, 1, 3], [0]]
    device_buf = torch.empty(
        4 * codec.bytes_per_block,
        dtype=torch.uint8,
        device=codec.device,
    )

    codec.gpu_to_chunk_major_device_buffer(device_buf, block_id_groups)
    for layer in kv_caches.values():
        layer.k_cache.zero_()
        layer.v_cache.zero_()
        if layer.k_scale is not None:
            layer.k_scale.zero_()
    codec.chunk_major_device_buffer_to_gpu(device_buf, block_id_groups)

    for name, layer in kv_caches.items():
        src = original[name]
        for bid in [4, 1, 3, 0]:
            assert torch.equal(layer.k_cache[bid], src.k_cache[bid])
            assert torch.equal(layer.v_cache[bid], src.v_cache[bid])
            if layer.k_scale is not None:
                assert torch.equal(layer.k_scale[bid], src.k_scale[bid])


def test_codec_chunk_major_rejects_duplicate_block_ids(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = ATOMKVByteCodec(kv_caches)
    device_buf = torch.empty(3 * codec.bytes_per_block, dtype=torch.uint8)

    with pytest.raises(ValueError, match="duplicate block ids"):
        codec.gpu_to_chunk_major_device_buffer(device_buf, [[0, 1], [1]])


@pytest.mark.parametrize("layout", ["block", "segment", "segment_indexed"])
@pytest.mark.parametrize("method_name", ["gpu_to_host", "host_to_gpu"])
def test_codec_rejects_invalid_block_ids_before_copy(monkeypatch, layout, method_name):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", layout)
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            k_scale=None,
            v_scale=None,
        ),
    }
    codec = ATOMKVByteCodec(kv_caches)
    host = torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
    method = getattr(codec, method_name)

    with pytest.raises(ValueError, match="block id out of range"):
        method(host, [0, 4])

    with pytest.raises(ValueError, match="block id out of range"):
        method(host, [-1])


def test_codec_rejects_short_host_buffer(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_CODEC_LAYOUT", "segment_indexed")
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            k_scale=None,
            v_scale=None,
        ),
    }
    codec = ATOMKVByteCodec(kv_caches)
    host = torch.empty(codec.bytes_per_block - 1, dtype=torch.uint8)

    with pytest.raises(ValueError, match="host_buf is too small"):
        codec.gpu_to_host(host, [0])


def test_copy_stream_is_cached_per_codec_device(monkeypatch):
    import torch

    if not hasattr(torch, "device") or not hasattr(torch, "cuda"):
        pytest.skip("torch cuda API is unavailable")

    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._tls = threading.local()
    conn._codec = SimpleNamespace(device=torch.device("cuda:1"))
    active_devices = []
    created_on = []

    class _FakeDeviceCtx:
        def __init__(self, device) -> None:
            self.device = str(device)

        def __enter__(self):
            active_devices.append(self.device)
            return None

        def __exit__(self, *args):
            active_devices.pop()
            return False

    class _FakeStream:
        def __init__(self) -> None:
            created_on.append(active_devices[-1] if active_devices else "default")

    monkeypatch.setattr(
        offload_connector_mod.torch.cuda,
        "device",
        lambda device: _FakeDeviceCtx(device),
    )
    monkeypatch.setattr(offload_connector_mod.torch.cuda, "Stream", _FakeStream)

    rank1_stream = conn._stream()
    assert conn._stream() is rank1_stream
    assert created_on == ["cuda:1"]

    conn._codec = SimpleNamespace(device=torch.device("cuda:0"))
    rank0_stream = conn._stream()

    assert rank0_stream is not rank1_stream
    assert created_on == ["cuda:1", "cuda:0"]


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


def test_load_is_skipped_if_hbm_satisfies_after_allocation():
    sched = _scheduler()
    lookup = _LookupClient(hit=8)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=321,
        num_prompt_tokens=12,
        token_ids=list(range(12)),
        num_cached_tokens=0,
        block_table=[1, 2, 3],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 8
    assert should_park is True

    # Prefix-cache allocation can discover a larger HBM hit than the lookup-time
    # snapshot. Scheme A should skip the CPU load before parking instead of
    # emitting a no-op load.
    seq.num_cached_tokens = 8
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is False
    meta = sched.build_connector_meta()

    assert meta.requests == []
    assert [req for req in meta.requests if req.load_spec is not None] == []
    assert seq.offload_loaded_tokens == 8
    assert sched._save_tracker[str(seq.id)][1] == 8
    assert lookup.cleared == ["321"]
    assert str(seq.id) not in sched._load_specs
    assert str(seq.id) not in sched._reqs_need_recv


def test_lookup_time_hbm_satisfies_does_not_resave_hit_prefix():
    sched = _scheduler()
    lookup = _LookupClient(hit=8)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=322,
        num_prompt_tokens=12,
        token_ids=list(range(12)),
        num_cached_tokens=8,
        block_table=[1, 2, 3],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 0
    assert should_park is False

    sched.update_state_after_alloc(seq)
    meta1 = sched.build_connector_meta()

    assert meta1.requests == []
    assert sched._save_tracker[str(seq.id)][1] == 8
    assert lookup.cleared == ["322"]

    seq.num_cached_tokens = 12
    meta2 = sched.build_connector_meta()
    save_reqs = [req for req in meta2.requests if req.save_spec is not None]

    assert len(save_reqs) == 1
    assert save_reqs[0].token_ids == list(range(12))
    assert save_reqs[0].save_spec.skip_leading_tokens == 8


def test_load_is_skipped_if_hbm_floor_is_not_chunk_aligned():
    sched = _scheduler()
    lookup = _LookupClient(hit=12)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=654,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 12
    assert should_park is True

    # HBM prefix cache can return block-size granularity, while LMCache chunks
    # are larger. Loading from a non-chunk boundary would either overlap shared
    # HBM blocks or leave a gap, so Scheme A skips CPU load and suffix-prefills
    # from the HBM floor.
    seq.num_cached_tokens = 6
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is False
    meta = sched.build_connector_meta()

    assert [req for req in meta.requests if req.load_spec is not None] == []
    assert seq.offload_loaded_tokens == 6
    assert lookup.cleared == ["654"]


def test_unaligned_hbm_handoff_prefills_boundary_then_emits_load():
    sched = _scheduler()
    sched._allow_unaligned_handoff = True
    sched._min_load_tokens = 8
    lookup = _LookupClient(hit=16)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=657,
        num_prompt_tokens=20,
        token_ids=list(range(20)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4, 5],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 16
    assert should_park is True

    seq.num_cached_tokens = 6
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is False
    assert str(seq.id) in sched._handoff_loads
    assert seq.offload_handoff_boundary_tokens == 8
    assert seq.offload_loaded_tokens == 6
    assert sched.adjust_prefill_chunk_after_alloc(seq, 10) == 2

    seq.num_cached_tokens = 8
    assert sched.should_park_partial_prefill_for_load(seq) is True
    meta = sched.build_connector_meta()
    load_reqs = [req for req in meta.requests if req.load_spec is not None]

    assert len(load_reqs) == 1
    req = load_reqs[0]
    assert req.req_id == 657
    assert req.token_ids == list(range(16))
    assert req.load_spec.hbm_cached_tokens == 8
    assert req.load_spec.lmcache_cached_tokens == 16
    assert seq.offload_loaded_tokens == 16
    assert str(seq.id) not in sched._handoff_loads
    assert lookup.cleared == []


def test_unaligned_handoff_skips_if_boundary_remainder_is_too_small():
    sched = _scheduler()
    sched._allow_unaligned_handoff = True
    sched._min_load_tokens = 8
    lookup = _LookupClient(hit=12)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=658,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 12
    assert should_park is True

    seq.num_cached_tokens = 6
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is False

    assert str(seq.id) not in sched._handoff_loads
    assert str(seq.id) not in sched._load_specs
    assert str(seq.id) not in sched._reqs_need_recv
    assert seq.offload_loaded_tokens == 6
    assert lookup.cleared == ["658"]


def test_load_is_skipped_if_aligned_hit_is_below_threshold():
    sched = _scheduler()
    sched._min_load_tokens = 8
    lookup = _LookupClient(hit=12)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=655,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 12
    assert should_park is True

    seq.num_cached_tokens = 8
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is False
    meta = sched.build_connector_meta()

    assert [req for req in meta.requests if req.load_spec is not None] == []
    assert seq.offload_loaded_tokens == 8
    assert lookup.cleared == ["655"]


def test_aligned_large_hit_parks_and_emits_load_metadata():
    sched = _scheduler()
    sched._min_load_tokens = 8
    lookup = _LookupClient(hit=12)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=656,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 12
    assert should_park is True

    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is True
    meta = sched.build_connector_meta()
    load_reqs = [req for req in meta.requests if req.load_spec is not None]

    assert len(load_reqs) == 1
    req = load_reqs[0]
    assert req.req_id == 656
    assert req.token_ids == list(range(12))
    assert req.block_ids == [1, 2, 3, 4]
    assert req.load_spec.hbm_cached_tokens == 4
    assert req.load_spec.lmcache_cached_tokens == 12
    assert seq.offload_loaded_tokens == 12
    assert lookup.cleared == []


def test_loaded_prefix_is_not_saved_again_after_success():
    sched = _scheduler()
    sched._min_load_tokens = 8
    sched._lookup_client = _LookupClient(hit=12)
    seq = SimpleNamespace(
        id=659,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 12
    assert should_park is True

    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is True

    load_meta = sched.build_connector_meta()
    assert len([req for req in load_meta.requests if req.load_spec is not None]) == 1
    assert [req for req in load_meta.requests if req.save_spec is not None] == []
    assert sched._save_tracker[str(seq.id)][1] == 12

    seq.num_cached_tokens = 16
    save_meta = sched.build_connector_meta()
    save_reqs = [req for req in save_meta.requests if req.save_spec is not None]

    assert len(save_reqs) == 1
    assert save_reqs[0].token_ids == list(range(16))
    assert save_reqs[0].save_spec.skip_leading_tokens == 12


def test_load_failure_allows_recomputed_hit_range_to_be_saved():
    sched = _scheduler()
    sched._min_load_tokens = 8
    sched._lookup_client = _LookupClient(hit=12)
    seq = SimpleNamespace(
        id=660,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
    )

    sched.get_num_new_matched_tokens(seq)
    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is True
    sched.build_connector_meta()
    assert sched._save_tracker[str(seq.id)][1] == 12

    sched.load_failed(seq.id)
    assert sched._save_tracker[str(seq.id)][1] == 4

    seq.num_cached_tokens = 12
    save_meta = sched.build_connector_meta()
    save_reqs = [req for req in save_meta.requests if req.save_spec is not None]

    assert len(save_reqs) == 1
    assert save_reqs[0].token_ids == list(range(12))
    assert save_reqs[0].save_spec.skip_leading_tokens == 4


def test_worker_completes_noop_load_when_hbm_satisfies():
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._failed_load = set()
    conn._done_save = set()
    conn._engine = SimpleNamespace(unpinned=[])
    conn._engine.lookup_unpin = lambda ids: conn._engine.unpinned.extend(ids)

    req = SimpleNamespace(
        req_id=321,
        token_ids=list(range(8)),
        block_ids=[1, 2, 3],
        load_spec=SimpleNamespace(hbm_cached_tokens=8, lmcache_cached_tokens=8),
    )

    conn._do_load_req(req)

    assert conn._done_load == {321}
    assert conn._failed_load == set()
    assert conn._engine.unpinned == ["321"]


def test_worker_reports_unaligned_hbm_load_as_failed_without_exception():
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._failed_load = set()
    conn._done_save = set()
    conn.chunk_size = 4
    conn._engine = SimpleNamespace(unpinned=[])
    conn._engine.lookup_unpin = lambda ids: conn._engine.unpinned.extend(ids)

    req = SimpleNamespace(
        req_id=654,
        token_ids=list(range(12)),
        block_ids=[1, 2, 3],
        load_spec=SimpleNamespace(hbm_cached_tokens=6, lmcache_cached_tokens=12),
    )

    conn._do_load_req(req)

    assert conn._done_load == set()
    assert conn._failed_load == {654}
    assert conn._engine.unpinned == ["654"]


def test_worker_save_uses_lmcache_engine_store():
    import torch

    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is unavailable")

    class _Engine:
        def __init__(self) -> None:
            self.calls = []

        def store(self, tokens, mask=None, **kwargs) -> None:
            self.calls.append((tokens.tolist(), mask.tolist(), kwargs))

    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_save = set()
    conn.chunk_size = 4
    conn._engine = _Engine()

    req = SimpleNamespace(
        req_id=987,
        token_ids=list(range(12)),
        block_ids=[3, 4, 5],
        is_last_prefill=True,
        save_spec=SimpleNamespace(skip_leading_tokens=4),
    )

    conn._do_save_req(req)

    assert conn._done_save == {987}
    assert len(conn._engine.calls) == 1
    tokens, mask, kwargs = conn._engine.calls[0]
    assert tokens == list(range(12))
    assert mask == [False, False, False, False] + [True] * 8
    assert kwargs["block_ids"] == [3, 4, 5]
    assert kwargs["req_id"] == "987"


def test_worker_load_uses_lmcache_engine_retrieve_and_marks_done():
    import torch

    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is unavailable")

    class _Engine:
        def __init__(self) -> None:
            self.calls = []
            self.unpinned = []

        def retrieve(self, tokens, mask=None, **kwargs):
            self.calls.append((tokens.tolist(), mask.tolist(), kwargs))
            return torch.tensor([False] * 4 + [True] * 8, dtype=torch.bool)

        def lookup_unpin(self, ids) -> None:
            self.unpinned.extend(ids)

    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._failed_load = set()
    conn._done_save = set()
    conn.chunk_size = 4
    conn._engine = _Engine()

    req = SimpleNamespace(
        req_id=988,
        token_ids=list(range(16)),
        block_ids=[3, 4, 5, 6],
        load_spec=SimpleNamespace(hbm_cached_tokens=4, lmcache_cached_tokens=12),
    )

    conn._do_load_req(req)

    assert conn._done_load == {988}
    assert conn._failed_load == set()
    assert conn._engine.unpinned == ["988"]
    tokens, mask, kwargs = conn._engine.calls[0]
    assert tokens == list(range(12))
    assert mask == [False, False, False, False] + [True] * 8
    assert kwargs["block_ids"] == [3, 4, 5, 6]
    assert kwargs["req_id"] == "988"


def test_worker_load_partial_retrieve_marks_failed():
    import torch

    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is unavailable")

    class _Engine:
        def __init__(self) -> None:
            self.unpinned = []

        def retrieve(self, tokens, mask=None, **kwargs):
            return torch.tensor([False] * 4 + [True] * 4 + [False] * 4)

        def lookup_unpin(self, ids) -> None:
            self.unpinned.extend(ids)

    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._failed_load = set()
    conn._done_save = set()
    conn.chunk_size = 4
    conn._engine = _Engine()

    req = SimpleNamespace(
        req_id=989,
        token_ids=list(range(16)),
        block_ids=[3, 4, 5, 6],
        load_spec=SimpleNamespace(hbm_cached_tokens=4, lmcache_cached_tokens=12),
    )

    conn._do_load_req(req)

    assert conn._done_load == set()
    assert conn._failed_load == {989}
    assert conn._engine.unpinned == ["989"]


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
        num_cached_tokens=8,
        prefix_hashes_published=True,
    )
    sched._save_tracker[str(seq.id)] = [seq, 0]

    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    assert meta.requests[0].save_spec is not None
    assert sched.should_defer_free(seq) is True

    sched.save_finished(seq.id)

    assert sched.should_defer_free(seq) is False


def test_chunked_prefill_save_uses_computed_frontier_and_serializes_inflight():
    sched = _scheduler()
    seq = SimpleNamespace(
        id=10,
        token_ids=list(range(12)),
        block_table=[3, 4, 5],
        num_prompt_tokens=12,
        num_cached_tokens=8,
        is_partial_prefill=True,
    )
    sched._save_tracker[str(seq.id)] = [seq, 0]

    meta1 = sched.build_connector_meta()

    assert len(meta1.requests) == 1
    assert len(meta1.requests[0].token_ids) == 8
    assert meta1.requests[0].save_spec.skip_leading_tokens == 0
    assert meta1.requests[0].is_last_prefill is False
    assert sched.should_defer_free(seq) is True

    seq.num_cached_tokens = 12
    seq.is_partial_prefill = False
    meta2 = sched.build_connector_meta()
    assert len(meta2.requests) == 0

    sched.save_finished(seq.id)
    meta3 = sched.build_connector_meta()

    assert len(meta3.requests) == 1
    assert len(meta3.requests[0].token_ids) == 12
    assert meta3.requests[0].save_spec.skip_leading_tokens == 8
    assert meta3.requests[0].is_last_prefill is True


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
