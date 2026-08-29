# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
import sys
import threading
import types
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

try:
    import torch
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from conftest import MockConfig

from atom.kv_transfer.disaggregation import (
    ConnectorCompletion,
    KVConnectorOutput,
    KVOutputAggregator,
)
from atom.kv_transfer.disaggregation.types import (
    KVTransferRegion,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import OffloadSchedulerMixin
from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector
from atom.kv_transfer.offload.dense.kv_byte_codec import (
    DenseKVByteCodec,
)
from atom.kv_transfer.offload.hybrid.dsv4 import policy as connector_module
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4_CHECKPOINT_SAVE_CHANNEL,
)
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4OffloadConnector as LMCacheOffloadConnector,
)
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4OffloadScheduler as LMCacheOffloadConnectorScheduler,
)
from atom.kv_transfer.offload.hybrid.dsv4.policy import (
    _chained_prefix_hashes,
)
from atom.kv_transfer.offload.metadata import (
    ATOMRawBytesLMCacheMetadata,
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    SlotLoadSpec,
    SlotSaveSpec,
)
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus


class _LookupClient:
    def __init__(self, hit: int) -> None:
        self.hit = hit
        self.cleared = []

    def lookup(self, token_ids, lookup_id):
        return self.hit

    def clear_lookup_status(self, lookup_id):
        self.cleared.append(lookup_id)


class _FailingLookupClient(_LookupClient):
    def lookup(self, token_ids, lookup_id):
        raise RuntimeError("lookup failed")


def _scheduler() -> LMCacheOffloadConnectorScheduler:
    sched = LMCacheOffloadConnectorScheduler.__new__(LMCacheOffloadConnectorScheduler)
    sched._config = SimpleNamespace()
    sched.kv_role = "offload"
    sched._do_save = True
    sched._do_load = True
    sched.block_size = 4
    sched.chunk_size = 4
    sched._lookup_client = _LookupClient(hit=0)
    sched._load_specs = {}
    sched._reqs_need_recv = {}
    sched._load_save_floors = {}
    sched._hit_save_floors = {}
    sched._save_tracker = {}
    sched._save_nonce = 0
    sched._load_nonce = 0
    sched._load_lifecycles = {}
    sched._active_load_operations = {}
    sched._save_inflight = {}
    sched._lookup_in_step = []
    sched._handoff_loads = set()
    sched.hash_block_size = 4
    sched.resume_alignment = 4
    sched.sidecar_interval = 0
    sched._committed_sidecar_hashes = set()
    sched._sidecar_save_inflight = {}
    sched._failed_sidecar_saves = {}
    sched._pending_slot_loads = {}
    sched._active_slot_loads = {}
    sched._sidecar_hash_cache = {}
    sched._load_inflight_tokens = {}
    sched._save_inflight_tokens = {}
    sched.total_load_requests = 0
    sched.total_loaded_tokens = 0
    sched.total_load_failures = 0
    sched.total_save_requests = 0
    sched.total_saved_tokens = 0
    sched._min_load_tokens = 0
    sched._lock = threading.Lock()
    sched._done_load = set()
    return sched


def _stateful_scheduler(hit: int) -> LMCacheOffloadConnectorScheduler:
    sched = _scheduler()
    sched.block_size = 256
    sched.hash_block_size = 256
    sched.chunk_size = 8192
    sched.resume_alignment = 8192
    sched.sidecar_interval = 8192
    sched._lookup_client = _LookupClient(hit=hit)
    return sched


def _stateful_seq(
    *,
    req_id: int,
    num_prompt_tokens: int,
    num_cached_tokens: int = 0,
    group: int = -1,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=req_id,
        num_prompt_tokens=num_prompt_tokens,
        token_ids=list(range(num_prompt_tokens)),
        num_cached_tokens=num_cached_tokens,
        block_table=list(range((num_prompt_tokens + 255) // 256)),
        has_per_req_cache=True,
        state_slot=group,
        prefix_hashes_published=True,
        _state_initialized_after_alloc=True,
    )


def _commit_sidecar(
    sched: LMCacheOffloadConnectorScheduler,
    seq: SimpleNamespace,
    boundary: int,
) -> int:
    boundary_hash = _chained_prefix_hashes(
        seq.token_ids,
        sched.hash_block_size,
    )[boundary]
    sched._committed_sidecar_hashes.add(boundary_hash)
    return boundary_hash


def test_bounded_commit_index_evicts_oldest_and_supports_set_operations():
    index = connector_module._BoundedLRUSet(2)

    index.add(1)
    index.add(2)
    index.add(3)

    assert set(index) == {2, 3}
    assert 1 not in index
    index.discard(2)
    assert set(index) == {3}
    index.clear()
    assert len(index) == 0


def test_bounded_commit_index_touching_duplicate_refreshes_recency():
    index = connector_module._BoundedLRUSet(2)
    index.add(1)
    index.add(2)

    index.add(1)
    index.add(3)

    assert set(index) == {1, 3}
    assert len(index) == 2


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "1.5", "not-an-int"],
)
def test_committed_sidecar_capacity_rejects_invalid_values(
    monkeypatch,
    value,
):
    monkeypatch.delenv("OFFLOAD_COMMITTED_SIDECAR_CAPACITY", raising=False)
    kvc = {
        "kv_connector_extra_config": {
            "committed_sidecar_index_capacity": value,
        }
    }

    with pytest.raises(ValueError, match="committed sidecar index capacity"):
        connector_module._committed_sidecar_capacity(kvc)


def test_committed_sidecar_capacity_precedence(monkeypatch):
    monkeypatch.setenv("OFFLOAD_COMMITTED_SIDECAR_CAPACITY", "7")

    assert connector_module._committed_sidecar_capacity({}) == 7
    assert (
        connector_module._committed_sidecar_capacity(
            {
                "kv_connector_extra_config": {
                    "committed_sidecar_index_capacity": 3,
                }
            }
        )
        == 3
    )

    monkeypatch.delenv("OFFLOAD_COMMITTED_SIDECAR_CAPACITY")
    assert connector_module._committed_sidecar_capacity({}) == 65536


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-an-int"])
def test_committed_sidecar_capacity_rejects_invalid_env(monkeypatch, value):
    monkeypatch.setenv("OFFLOAD_COMMITTED_SIDECAR_CAPACITY", value)

    with pytest.raises(ValueError, match="committed sidecar index capacity"):
        connector_module._committed_sidecar_capacity({})


def _install_fake_fused_chunk_major(codec: DenseKVByteCodec) -> None:
    def _pack(
        segments,
        seg_block_bytes,
        chunk_block_counts,
        flat_block_ids,
        device_buf,
    ) -> None:
        offset = 0
        cursor = 0
        for count in chunk_block_counts:
            block_ids = flat_block_ids[cursor : cursor + count]
            cursor += count
            idx = torch.tensor(block_ids, dtype=torch.long, device=codec.device)
            for seg, nbytes in zip(segments, seg_block_bytes):
                src = seg.index_select(0, idx).contiguous().view(torch.uint8)
                device_buf[offset : offset + count * nbytes].copy_(src.reshape(-1))
                offset += count * nbytes

    def _unpack(
        device_buf,
        segments,
        seg_block_bytes,
        chunk_block_counts,
        flat_block_ids,
    ) -> None:
        offset = 0
        cursor = 0
        for count in chunk_block_counts:
            block_ids = flat_block_ids[cursor : cursor + count]
            cursor += count
            idx = torch.tensor(block_ids, dtype=torch.long, device=codec.device)
            for seg, nbytes in zip(segments, seg_block_bytes):
                src = device_buf[offset : offset + count * nbytes]
                src = src.view(seg.dtype).reshape((count,) + tuple(seg.shape[1:]))
                seg.index_copy_(0, idx, src)
                offset += count * nbytes

    codec._fused_kv_staging = SimpleNamespace(
        fused_pack_chunk_major=_pack,
        fused_unpack_chunk_major=_unpack,
    )


def _registration_connector() -> LMCacheOffloadConnector:
    connector = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    connector._config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
    )
    connector.block_size = 4
    connector.chunk_size = None
    connector._do_save = True
    connector._do_load = True
    connector._engine = None
    connector._codec = None
    connector._lookup_server = None
    return connector


def _dense_registration_connector() -> DenseOffloadConnector:
    connector = DenseOffloadConnector.__new__(DenseOffloadConnector)
    connector._config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
        decode_context_parallel_size=1,
    )
    connector.block_size = 4
    connector.virtual_block_size = 4
    connector.chunk_size = None
    connector._do_save = True
    connector._do_load = True
    connector._engine = None
    connector._codec = None
    connector._lookup_server = None
    return connector


def _install_registration_dependencies(monkeypatch) -> dict:
    captured = {}

    parallel_state_module = types.ModuleType("aiter.dist.parallel_state")
    parallel_state_module.get_tp_group = lambda: SimpleNamespace(
        rank_in_group=0,
        world_size=1,
    )
    aiter_dist_module = types.ModuleType("aiter.dist")
    aiter_dist_module.__path__ = []
    aiter_dist_module.parallel_state = parallel_state_module
    aiter_module = types.ModuleType("aiter")
    aiter_module.__path__ = []
    aiter_module.dist = aiter_dist_module

    class _FakeEngine:
        def __init__(self, gpu_connector) -> None:
            self.gpu_connector = gpu_connector
            self.storage_manager = SimpleNamespace(list_backends=dict)
            self.post_initialized = False

        def post_init(self) -> None:
            self.post_initialized = True

    class _FakeEngineBuilder:
        @staticmethod
        def get_or_create(
            instance_id,
            cfg,
            metadata,
            gpu_connector,
            process_tokens,
            create_gpu_connector,
        ):
            captured.update(
                instance_id=instance_id,
                cfg=cfg,
                metadata=metadata,
                gpu_connector=gpu_connector,
                process_tokens=process_tokens,
                create_gpu_connector=create_gpu_connector,
            )
            engine = _FakeEngine(gpu_connector)
            captured["engine"] = engine
            return engine

    cache_engine_module = types.ModuleType("lmcache.v1.cache_engine")
    cache_engine_module.LMCacheEngineBuilder = _FakeEngineBuilder
    memory_management_module = types.ModuleType("lmcache.v1.memory_management")
    memory_management_module.MemoryFormat = SimpleNamespace(KV_2LTD=object())

    class _FakeLookupClientFactory:
        @staticmethod
        def create_lookup_server(engine, metadata):
            captured["lookup_server_args"] = (engine, metadata)
            return SimpleNamespace()

    lookup_factory_module = types.ModuleType("lmcache.v1.lookup_client.factory")
    lookup_factory_module.LookupClientFactory = _FakeLookupClientFactory
    lookup_client_module = types.ModuleType("lmcache.v1.lookup_client")
    lookup_client_module.__path__ = []
    lookup_client_module.factory = lookup_factory_module
    lmcache_v1_module = types.ModuleType("lmcache.v1")
    lmcache_v1_module.__path__ = []
    lmcache_v1_module.cache_engine = cache_engine_module
    lmcache_v1_module.memory_management = memory_management_module
    lmcache_v1_module.lookup_client = lookup_client_module
    lmcache_module = types.ModuleType("lmcache")
    lmcache_module.__path__ = []
    lmcache_module.v1 = lmcache_v1_module

    for name, module in (
        ("aiter", aiter_module),
        ("aiter.dist", aiter_dist_module),
        ("aiter.dist.parallel_state", parallel_state_module),
        ("lmcache", lmcache_module),
        ("lmcache.v1", lmcache_v1_module),
        ("lmcache.v1.cache_engine", cache_engine_module),
        ("lmcache.v1.memory_management", memory_management_module),
        ("lmcache.v1.lookup_client", lookup_client_module),
        ("lmcache.v1.lookup_client.factory", lookup_factory_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    cfg = SimpleNamespace(
        chunk_size=8,
        local_cpu=True,
        max_local_cpu_size=1,
        local_disk=None,
        max_local_disk_size=0,
        store_location=None,
        retrieve_locations=None,
    )
    base_metadata = SimpleNamespace(chunk_size=cfg.chunk_size)
    base_metadata.is_first_rank = lambda: True
    monkeypatch.setattr(offcfg, "build_lmcache_config", lambda _config: cfg)
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_metadata",
        lambda _config, _cfg, _world, _rank: base_metadata,
    )
    # Registration tests run without importing a real Triton runtime. Individual
    # startup-failure coverage overrides this property below.
    monkeypatch.setattr(
        DSV4PageSlotCodec,
        "has_fused_chunk_major_staging",
        property(lambda _codec: True),
    )
    return captured


def _page_transfer_tensors(
    *unit_bytes: int,
    num_blocks: int,
) -> SimpleNamespace:
    regions = [
        KVTransferRegion(
            base_addr=0x1000 + index * 0x1000,
            total_bytes=num_blocks * nbytes,
            unit_bytes=nbytes,
        )
        for index, nbytes in enumerate(unit_bytes)
    ]
    return SimpleNamespace(block_regions=regions, num_blocks=num_blocks)


def test_register_empty_kv_caches_with_page_regions_selects_page_codec(monkeypatch):
    import torch

    if not hasattr(torch, "device"):
        pytest.skip("real torch is unavailable")

    captured = _install_registration_dependencies(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    connector = _registration_connector()
    transfer_tensors = _page_transfer_tensors(24, 40, num_blocks=3)

    connector.register_kv_caches({}, transfer_tensors, num_blocks=3)

    assert isinstance(connector._codec, DSV4PageSlotCodec)
    assert connector._codec.num_blocks == 3
    assert connector._codec.device == torch.device("cuda:7")
    assert captured["gpu_connector"].codec is connector._codec


def test_register_page_regions_fails_before_engine_when_triton_is_unavailable(
    monkeypatch,
):
    captured = _install_registration_dependencies(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        DSV4PageSlotCodec,
        "has_fused_chunk_major_staging",
        property(lambda _codec: False),
    )
    connector = _registration_connector()

    with pytest.raises(RuntimeError, match="requires Triton fused staging"):
        connector.register_kv_caches(
            {},
            _page_transfer_tensors(32, num_blocks=2),
            num_blocks=2,
        )

    assert "engine" not in captured


def test_dense_backend_registers_dense_kv_byte_codec(monkeypatch):
    import torch

    if not hasattr(torch, "zeros"):
        pytest.skip("real torch is unavailable")

    _install_registration_dependencies(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    connector = _dense_registration_connector()
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.zeros((3, 8), dtype=torch.uint8),
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )
    }

    connector.register_kv_caches(
        kv_caches,
        _page_transfer_tensors(8, num_blocks=3),
        num_blocks=3,
    )

    assert type(connector._codec) is DenseKVByteCodec
    assert connector._codec.bytes_per_block == 8


def test_register_page_regions_rejects_invalid_region_geometry(monkeypatch):
    import torch

    if not hasattr(torch, "device"):
        pytest.skip("real torch is unavailable")

    _install_registration_dependencies(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    connector = _registration_connector()
    transfer_tensors = SimpleNamespace(
        block_regions=[
            KVTransferRegion(
                base_addr=0x1000,
                total_bytes=63,
                unit_bytes=32,
            )
        ],
        num_blocks=2,
    )

    with pytest.raises(
        ValueError,
        match=r"PAGE region 0 total_bytes is too small; got 63, need 64",
    ):
        connector.register_kv_caches({}, transfer_tensors, num_blocks=2)


def test_register_stateful_page_requires_full_slot_regions_before_engine_start(
    monkeypatch,
):
    import torch

    if not hasattr(torch, "device"):
        pytest.skip("real torch is unavailable")

    captured = _install_registration_dependencies(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    connector = _registration_connector()
    transfer_tensors = _page_transfer_tensors(32, num_blocks=2)
    transfer_tensors.num_slots = 4
    transfer_tensors.swa_block_regions = []
    transfer_tensors.slot_regions = []

    with pytest.raises(
        ValueError,
        match="full per-request SLOT regions",
    ):
        connector.register_kv_caches({}, transfer_tensors, num_blocks=2)

    assert "instance_id" not in captured
    assert "engine" not in captured
    assert connector._engine is None


@pytest.mark.parametrize(
    ("regions", "expected_count", "message"),
    [
        pytest.param(
            [KVTransferRegion(0x1000, 128, 32, reverse_indexed=True)],
            2,
            "expected 2 full per-request SLOT regions, got 1",
            id="missing-fp8-plane",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 128, 32, reverse_indexed=True)],
            None,
            "expected_full_slot_region_count",
            id="missing-expected-count",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 128, 32, reverse_indexed=True)],
            0,
            "expected_full_slot_region_count must be a positive integer",
            id="invalid-expected-count",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 128, 32, reverse_indexed=True)],
            True,
            "expected_full_slot_region_count must be a positive integer",
            id="boolean-expected-count",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 128, 32, reverse_indexed=False)],
            1,
            "reverse_indexed=True",
            id="forward-indexed",
        ),
        pytest.param(
            [KVTransferRegion(0, 128, 32, reverse_indexed=True)],
            1,
            "base_addr must be > 0",
            id="zero-base",
        ),
        pytest.param(
            [KVTransferRegion(0x1000 + 0.5, 128, 32, reverse_indexed=True)],
            1,
            "base_addr must be an integer",
            id="noninteger-base",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 128, 0, reverse_indexed=True)],
            1,
            "unit_bytes must be > 0",
            id="zero-unit",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 128, 32.5, reverse_indexed=True)],
            1,
            "unit_bytes must be an integer",
            id="noninteger-unit",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 0, 32, reverse_indexed=True)],
            1,
            "total_bytes must be > 0",
            id="zero-total",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 128.5, 32, reverse_indexed=True)],
            1,
            "total_bytes must be an integer",
            id="noninteger-total",
        ),
        pytest.param(
            [KVTransferRegion(0x1000, 127, 32, reverse_indexed=True)],
            1,
            "total_bytes is too small; got 127, need 128",
            id="insufficient-capacity",
        ),
    ],
)
def test_register_rejects_malformed_full_slot_geometry_before_engine_start(
    monkeypatch,
    regions,
    expected_count,
    message,
):
    import torch

    if not hasattr(torch, "device"):
        pytest.skip("real torch is unavailable")

    captured = _install_registration_dependencies(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    connector = _registration_connector()
    transfer_tensors = _page_transfer_tensors(32, num_blocks=2)
    transfer_tensors.num_slots = 4
    for index, region in enumerate(regions):
        if region.semantic_role is None:
            region.semantic_role = f"test.slot.{index}"
    transfer_tensors.swa_block_regions = regions
    transfer_tensors.slot_regions = []
    transfer_tensors.expected_full_slot_region_count = expected_count

    with pytest.raises(ValueError, match=message):
        connector.register_kv_caches({}, transfer_tensors, num_blocks=2)

    assert "instance_id" not in captured
    assert "engine" not in captured
    assert connector._engine is None


def test_lmcache_engine_uses_selected_page_codec_bytes_per_block(monkeypatch):
    import torch

    if not hasattr(torch, "device"):
        pytest.skip("real torch is unavailable")

    captured = _install_registration_dependencies(monkeypatch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    connector = _registration_connector()
    transfer_tensors = _page_transfer_tensors(24, 40, num_blocks=4)

    connector.register_kv_caches({}, transfer_tensors, num_blocks=4)

    assert connector._codec.bytes_per_block == 64
    assert captured["metadata"].atom_bytes_per_block == 64
    assert captured["gpu_connector"].gpu_staging_chunk_bytes == 128
    assert captured["engine"].post_initialized is True


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


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            SimpleNamespace(
                local_disk="/nvme/lmcache",
                max_local_disk_size=0,
                max_local_cpu_size=1,
            ),
            "LMCACHE_MAX_LOCAL_DISK_SIZE must be > 0",
        ),
        (
            SimpleNamespace(
                local_disk=None,
                max_local_disk_size=1,
                max_local_cpu_size=1,
            ),
            "LMCACHE_LOCAL_DISK is missing",
        ),
        (
            SimpleNamespace(
                local_disk="/nvme/lmcache",
                max_local_disk_size=1,
                max_local_cpu_size=0,
            ),
            "LMCACHE_MAX_LOCAL_CPU_SIZE > 0",
        ),
    ],
)
def test_lmcache_disk_config_requires_complete_host_staging(cfg, message):
    with pytest.raises(ValueError, match=message):
        offcfg.validate_lmcache_storage_config(cfg)


def test_build_lmcache_config_validates_extras_and_keeps_gds_disabled(monkeypatch):
    fake_config_module = types.ModuleType("lmcache.v1.config")

    class _FakeEngineConfig:
        @staticmethod
        def from_env():
            return SimpleNamespace(
                chunk_size=256,
                local_cpu=True,
                max_local_cpu_size=1,
                local_disk=None,
                max_local_disk_size=0,
                use_gds=True,
                lookup_server_worker_ids=None,
            )

    fake_config_module.LMCacheEngineConfig = _FakeEngineConfig
    monkeypatch.setitem(sys.modules, "lmcache", types.ModuleType("lmcache"))
    monkeypatch.setitem(sys.modules, "lmcache.v1", types.ModuleType("lmcache.v1"))
    monkeypatch.setitem(sys.modules, "lmcache.v1.config", fake_config_module)

    cfg = offcfg.build_lmcache_config(
        {
            "kv_connector_extra_config": {
                "lmcache.local_cpu": False,
                "lmcache.max_local_cpu_size": 2,
                "lmcache.local_disk": "/nvme/lmcache",
                "lmcache.max_local_disk_size": 10,
                "lmcache.use_gds": True,
            }
        }
    )

    assert cfg.local_cpu is False
    assert cfg.max_local_cpu_size == 2
    assert cfg.local_disk == "/nvme/lmcache"
    assert cfg.max_local_disk_size == 10
    assert cfg.use_gds is False
    assert cfg.lookup_server_worker_ids == [0]


def test_lmcache_disk_startup_fails_if_backend_was_not_created():
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._rank = 2
    conn._engine = SimpleNamespace(
        storage_manager=SimpleNamespace(
            list_backends=lambda: {"LocalCPUBackend": "LocalCPUBackend"}
        )
    )
    cfg = SimpleNamespace(
        local_cpu=False,
        max_local_cpu_size=1,
        local_disk="/nvme/lmcache",
        max_local_disk_size=10,
        store_location=None,
        retrieve_locations=None,
    )

    with pytest.raises(RuntimeError, match="LocalDiskBackend was not created"):
        conn._validate_and_log_storage_backends(cfg)


def test_lmcache_disk_startup_logs_realized_backend_topology(caplog):
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._rank = 0
    conn._engine = SimpleNamespace(
        storage_manager=SimpleNamespace(
            list_backends=lambda: {
                "LocalCPUBackend": "LocalCPUBackend",
                "LocalDiskBackend": "LocalDiskBackend",
            }
        )
    )
    cfg = SimpleNamespace(
        local_cpu=False,
        max_local_cpu_size=1,
        local_disk="/nvme/lmcache",
        max_local_disk_size=10,
        store_location="LocalDiskBackend",
        retrieve_locations=["LocalDiskBackend"],
    )

    with caplog.at_level(logging.INFO, logger="atom"):
        conn._validate_and_log_storage_backends(cfg)

    assert "LocalDiskBackend" in caplog.text
    assert "local_disk=/nvme/lmcache" in caplog.text


def test_lmcache_connector_maps_token_ranges_to_block_ids():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2),
            v_cache=(torch.arange(6 * 3, dtype=torch.uint8).reshape(6, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = DenseKVByteCodec(kv_caches)
    connector = BlockGPUConnector(codec, block_size=4, chunk_size=8)

    assert connector._ranges_to_block_ids(
        [4],
        [12],
        block_ids=[0, 1, 2, 3, 4, 5],
    ) == [[1, 2]]
    assert connector._ranges_to_block_ids(
        [0, 8],
        [8, 16],
        block_ids=[0, 1, 2, 3, 4, 5],
    ) == [[0, 1], [2, 3]]
    with pytest.raises(ValueError, match="block-aligned"):
        connector._ranges_to_block_ids(
            [2],
            [8],
            block_ids=[0, 1, 2, 3, 4, 5],
        )


def test_lmcache_connector_maps_dcp_ranges_on_virtual_block_grid():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2),
            v_cache=torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = DenseKVByteCodec(kv_caches)
    connector = BlockGPUConnector(
        codec,
        block_size=4,
        virtual_block_size=8,
        chunk_size=16,
    )

    assert connector._ranges_to_block_ids(
        [8],
        [24],
        block_ids=[10, 11, 12, 13],
    ) == [[11, 12]]
    assert connector.gpu_staging_chunk_bytes == 2 * codec.bytes_per_block


@pytest.mark.parametrize(
    ("first_stream_name", "second_stream_name"),
    [("pack", "copy"), ("copy", "pack")],
)
def test_staged_pipeline_allocates_on_first_consumer_stream(
    first_stream_name,
    second_stream_name,
):
    from atom.kv_transfer.offload.atom_lmcache_staging import (
        _PipelineStage,
        run_staged_pipeline,
    )

    trace = []

    class _FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            trace.append(("wait", self.name, event.name))

        def synchronize(self):
            trace.append(("synchronize", self.name))

    class _FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            trace.append(("record", self.name, stream.name))

    class _StreamContext:
        def __init__(self, state, stream):
            self.state = state
            self.stream = stream

        def __enter__(self):
            assert self.state.current_stream is None
            self.state.current_stream = self.stream
            trace.append(("enter", self.stream.name))

        def __exit__(self, *_args):
            trace.append(("exit", self.stream.name))
            self.state.current_stream = None

    class _FakeState:
        def __init__(self):
            self.current_stream = None
            self.staging_buffer = SimpleNamespace(
                tensor=None,
                ready_event=_FakeEvent("ready"),
                free_event=_FakeEvent("free"),
                free_event_valid=False,
            )

        def stream_ctx(self, stream):
            return _StreamContext(self, stream)

    state = _FakeState()
    first_stream = _FakeStream(first_stream_name)
    second_stream = _FakeStream(second_stream_name)
    device_buf = object()

    def ensure_buffer(_staging_buffer, nbytes):
        assert nbytes == 8
        assert state.current_stream is first_stream
        trace.append(("ensure", state.current_stream.name))
        return device_buf

    def run_stage(stage_name, expected_stream, _group, actual_buf):
        assert state.current_stream is expected_stream
        assert actual_buf is device_buf
        trace.append(("run", stage_name, state.current_stream.name))

    run_staged_pipeline(
        state,
        [SimpleNamespace(nbytes=8)],
        stage_a=_PipelineStage(
            first_stream,
            lambda group, buf: run_stage("a", first_stream, group, buf),
        ),
        stage_b=_PipelineStage(
            second_stream,
            lambda group, buf: run_stage("b", second_stream, group, buf),
        ),
        ensure_buffer=ensure_buffer,
        group_nbytes=lambda group: group.nbytes,
    )

    assert trace[:3] == [
        ("enter", first_stream_name),
        ("ensure", first_stream_name),
        ("run", "a", first_stream_name),
    ]
    assert ("run", "b", second_stream_name) in trace


def _exception_pipeline(monkeypatch, direction: str, *, failed_stream: str | None):
    import torch

    class _FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name
            self.fail_sync = name == failed_stream
            self.sync_calls = 0

        def wait_event(self, event) -> None:
            pass

        def synchronize(self) -> None:
            self.sync_calls += 1
            if self.fail_sync:
                raise RuntimeError(f"{self.name} fence failed")

    class _FakeEvent:
        def __init__(self) -> None:
            self.record_calls = 0

        def record(self, stream) -> None:
            self.record_calls += 1

    class _FakeState:
        def __init__(self) -> None:
            self.pack_stream = _FakeStream("pack")
            self.copy_stream = _FakeStream("copy")
            self.staging_buffer = SimpleNamespace(
                tensor=None,
                ready_event=_FakeEvent(),
                free_event=_FakeEvent(),
                free_event_valid=False,
            )

        def stream_ctx(self, stream):
            return nullcontext()

    callback_calls = 0
    raise_on_second_group = True

    def _maybe_raise(device_buf, block_id_groups, stream=None):
        nonlocal callback_calls
        callback_calls += 1
        if raise_on_second_group and callback_calls == 2:
            raise RuntimeError(f"second {direction} group failed")

    codec = SimpleNamespace(
        device=torch.device("cpu"),
        num_blocks=2,
        bytes_per_block=1,
        has_fused_chunk_major_staging=True,
        gpu_to_chunk_major_device_buffer=_maybe_raise,
        chunk_major_device_buffer_to_gpu=_maybe_raise,
    )
    connector = BlockGPUConnector(codec, block_size=1, chunk_size=1)
    connector._release_gpu_staging_after_transfer = False
    monkeypatch.setattr(connector, "_assert_fused_chunk_major_available", lambda: None)
    state = _FakeState()

    def _groups(count: int):
        return [
            SimpleNamespace(
                chunks=[
                    SimpleNamespace(
                        block_ids=[index],
                        nbytes=1,
                        tensor=torch.zeros(1, dtype=torch.uint8),
                    )
                ],
                nbytes=1,
            )
            for index in range(count)
        ]

    current_groups = _groups(2)
    monkeypatch.setattr(
        connector,
        "_prepare_transfer",
        lambda *args, **kwargs: (state, current_groups),
    )
    ensured_tensors = []
    original_ensure = connector._ensure_staging_buffer

    def _ensure(staging_buffer, nbytes):
        device_buf = original_ensure(staging_buffer, nbytes)
        ensured_tensors.append(staging_buffer.tensor)
        return device_buf

    monkeypatch.setattr(connector, "_ensure_staging_buffer", _ensure)

    def _invoke() -> None:
        if direction == "save":
            connector.batched_from_gpu([], [], [])
        else:
            connector.batched_to_gpu([], [], [])

    def _prepare_retry() -> None:
        nonlocal current_groups, raise_on_second_group
        current_groups = _groups(1)
        raise_on_second_group = False

    return SimpleNamespace(
        connector=connector,
        state=state,
        invoke=_invoke,
        prepare_retry=_prepare_retry,
        ensured_tensors=ensured_tensors,
    )


@pytest.mark.parametrize("direction", ["save", "load"])
def test_pipeline_exception_successful_fences_allow_safe_buffer_reuse(
    monkeypatch,
    direction,
):
    case = _exception_pipeline(monkeypatch, direction, failed_stream=None)

    with pytest.raises(RuntimeError, match=f"second {direction} group failed"):
        case.invoke()

    old_tensor = case.ensured_tensors[0]
    assert case.state.staging_buffer.free_event.record_calls == 1
    assert case.state.pack_stream.sync_calls == 1
    assert case.state.copy_stream.sync_calls == 1
    assert case.state.staging_buffer.tensor is old_tensor
    assert case.state.staging_buffer.free_event_valid is False
    assert case.connector._quarantined_staging_tensors == []

    case.prepare_retry()
    case.invoke()

    assert case.ensured_tensors[-1] is old_tensor


@pytest.mark.parametrize(
    ("direction", "failed_stream"),
    [("save", "pack"), ("load", "copy")],
)
def test_pipeline_exception_failed_fence_quarantines_even_in_release_mode(
    monkeypatch,
    direction,
    failed_stream,
):
    case = _exception_pipeline(monkeypatch, direction, failed_stream=failed_stream)
    case.connector._release_gpu_staging_after_transfer = True

    with pytest.raises(RuntimeError, match=f"second {direction} group failed"):
        case.invoke()

    old_tensor = case.ensured_tensors[0]
    assert case.state.staging_buffer.free_event.record_calls == 1
    assert case.state.pack_stream.sync_calls == 1
    assert case.state.copy_stream.sync_calls == 1
    assert case.state.staging_buffer.tensor is None
    assert case.state.staging_buffer.free_event_valid is False
    assert len(case.connector._quarantined_staging_tensors) == 1
    assert case.connector._quarantined_staging_tensors[0] is old_tensor

    case.state.pack_stream.fail_sync = False
    case.state.copy_stream.fail_sync = False
    case.prepare_retry()
    case.invoke()

    assert case.ensured_tensors[-1] is not old_tensor
    assert case.state.staging_buffer.tensor is None
    assert case.connector._quarantined_staging_tensors[0] is old_tensor


def test_lmcache_connector_fused_chunk_fastpath_uses_chunk_major(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

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
    codec = DenseKVByteCodec(kv_caches)
    connector = BlockGPUConnector(codec, block_size=4, chunk_size=8)
    _install_fake_fused_chunk_major(codec)
    monkeypatch.setattr(connector, "_assert_fused_chunk_major_available", lambda: None)

    pack_groups = []
    unpack_groups = []
    buffer_requests = []

    monkeypatch.setattr(
        codec,
        "gpu_to_chunk_major_device_buffer",
        lambda device_buf, block_id_groups, stream=None: (
            pack_groups.append([list(group) for group in block_id_groups]),
            DenseKVByteCodec.gpu_to_chunk_major_device_buffer(
                codec, device_buf, block_id_groups, stream=None
            ),
        )[-1],
    )
    monkeypatch.setattr(
        codec,
        "chunk_major_device_buffer_to_gpu",
        lambda device_buf, block_id_groups, stream=None: (
            unpack_groups.append([list(group) for group in block_id_groups]),
            DenseKVByteCodec.chunk_major_device_buffer_to_gpu(
                codec, device_buf, block_id_groups, stream=None
            ),
        )[-1],
    )
    orig_ensure_staging_buffer = connector._ensure_staging_buffer

    def _ensure_staging_buffer(staging_buffer, nbytes):
        device_buf = orig_ensure_staging_buffer(staging_buffer, nbytes)
        buffer_requests.append((nbytes, int(staging_buffer.tensor.numel())))
        return device_buf

    monkeypatch.setattr(connector, "_ensure_staging_buffer", _ensure_staging_buffer)

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
            self.pack_stream = _FakeStream()
            self.copy_stream = _FakeStream()
            self.staging_buffer = SimpleNamespace(
                tensor=None,
                ready_event=_FakeEvent(),
                free_event=_FakeEvent(),
                free_event_valid=False,
            )

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
    assert pack_groups == [[[1, 2], [3]]]
    assert all(nbytes <= 4 * codec.bytes_per_block for nbytes, _ in buffer_requests)
    assert all(capacity == 4 * codec.bytes_per_block for _, capacity in buffer_requests)
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

    assert unpack_groups == [[[1, 2], [3]]]
    for bid in [1, 2, 3]:
        assert torch.equal(kv_caches["l0"].k_cache[bid], original["l0"].k_cache[bid])
        assert torch.equal(kv_caches["l0"].v_cache[bid], original["l0"].v_cache[bid])
    assert torch.count_nonzero(kv_caches["l0"].k_cache[0]) == 0
    assert torch.count_nonzero(kv_caches["l0"].v_cache[0]) == 0


def test_lmcache_connector_requires_fused_chunk_major_staging():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=(torch.arange(4 * 3, dtype=torch.uint8).reshape(4, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = DenseKVByteCodec(kv_caches)
    connector = BlockGPUConnector(codec, block_size=4, chunk_size=8)
    memory_objs = [
        SimpleNamespace(
            tensor=torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)
        )
    ]

    with pytest.raises(RuntimeError, match="requires Triton fused"):
        connector.batched_from_gpu(
            memory_objs,
            [0],
            [8],
            block_ids=list(range(4)),
        )


def test_lmcache_connector_rejects_oversized_memory_obj():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=(torch.arange(4 * 3, dtype=torch.uint8).reshape(4, 3) + 51),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = DenseKVByteCodec(kv_caches)
    connector = BlockGPUConnector(codec, block_size=4, chunk_size=4)
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


def test_lmcache_connector_respects_staging_buffer_chunks_env(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "3")
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(2 * 2, dtype=torch.uint8).reshape(2, 2),
            v_cache=torch.arange(2 * 3, dtype=torch.uint8).reshape(2, 3),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = DenseKVByteCodec(kv_caches)
    connector = BlockGPUConnector(codec, block_size=4, chunk_size=4)

    assert connector.gpu_staging_buffer_chunks == 3
    assert connector.gpu_staging_buffer_bytes == 3 * connector.gpu_staging_chunk_bytes
    assert connector._thread_state().staging_buffer.tensor is None


def test_lmcache_connector_default_staging_buffer_chunks_is_two(monkeypatch):
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

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
    codec = DenseKVByteCodec(kv_caches)
    connector = BlockGPUConnector(codec, block_size=4, chunk_size=4)

    assert connector.gpu_staging_buffer_chunks == 2
    assert connector.gpu_staging_buffer_bytes == 2 * connector.gpu_staging_chunk_bytes


def test_codec_chunk_major_device_buffer_layout():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

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
    codec = DenseKVByteCodec(kv_caches)
    _install_fake_fused_chunk_major(codec)
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


def test_codec_chunk_major_handles_tail_and_sparse_blocks():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

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
    codec = DenseKVByteCodec(kv_caches)
    _install_fake_fused_chunk_major(codec)
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


def test_codec_chunk_major_rejects_duplicate_block_ids():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            v_cache=torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2),
            k_scale=None,
            v_scale=None,
        )
    }
    codec = DenseKVByteCodec(kv_caches)
    device_buf = torch.empty(3 * codec.bytes_per_block, dtype=torch.uint8)

    with pytest.raises(ValueError, match="duplicate block ids"):
        codec.gpu_to_chunk_major_device_buffer(device_buf, [[0, 1], [1]])


def test_scheduler_alignment_uses_dcp_hash_blocks_and_lmcache_chunks(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config: SimpleNamespace(chunk_size=24),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        state_checkpoint_interval_tokens=16,
        tensor_parallel_size=1,
    )

    sched = LMCacheOffloadConnectorScheduler(config)

    assert sched.hash_block_size == 8
    assert sched.resume_alignment == 24
    assert sched.sidecar_interval == 48


def test_scheduler_snaps_checkpoint_interval_before_sidecar_cadence(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config: SimpleNamespace(chunk_size=24),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        state_checkpoint_interval_tokens=25,
        tensor_parallel_size=1,
    )

    sched = LMCacheOffloadConnectorScheduler(config)

    assert sched.hash_block_size == 8
    assert sched.resume_alignment == 24
    assert sched.sidecar_interval == 24


def test_zero_checkpoint_interval_keeps_terminal_alignment(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config: SimpleNamespace(chunk_size=24),
    )
    monkeypatch.setattr(offcfg, "build_lmcache_metadata", lambda *_args: object())
    config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        state_checkpoint_interval_tokens=0,
        tensor_parallel_size=1,
    )

    sched = LMCacheOffloadConnectorScheduler(config)

    assert sched.resume_alignment == 24
    assert sched.sidecar_interval == 0


def test_dsv4_scheduler_invalid_lmcache_config_fails_fast(monkeypatch):
    def invalid_config(_config=None):
        raise ValueError("invalid LMCache storage")

    monkeypatch.setattr(offcfg, "build_lmcache_config", invalid_config)
    config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        tensor_parallel_size=1,
    )

    with pytest.raises(ValueError, match="invalid LMCache storage"):
        LMCacheOffloadConnectorScheduler(config)


def test_dsv4_scheduler_invalid_lmcache_metadata_fails_fast(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config: SimpleNamespace(chunk_size=24),
    )

    def invalid_metadata(*_args):
        raise ValueError("invalid LMCache TP geometry")

    monkeypatch.setattr(offcfg, "build_lmcache_metadata", invalid_metadata)
    config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        tensor_parallel_size=1,
    )

    with pytest.raises(ValueError, match="invalid LMCache TP geometry"):
        LMCacheOffloadConnectorScheduler(config)


def test_dsv4_scheduler_rejects_fractional_lmcache_chunk(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config: SimpleNamespace(chunk_size=24.5),
    )
    config = SimpleNamespace(
        kv_transfer_config={},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        tensor_parallel_size=1,
    )

    with pytest.raises(ValueError, match="LMCache chunk size must be an integer"):
        LMCacheOffloadConnectorScheduler(config)


@pytest.mark.parametrize(
    "connector_cls",
    [LMCacheOffloadConnector, LMCacheOffloadConnectorScheduler],
)
def test_unknown_kv_role_fails_fast(connector_cls):
    config = SimpleNamespace(
        kv_transfer_config={"kv_role": "unknown"},
        kv_cache_block_size=4,
    )

    with pytest.raises(ValueError, match="kv_role"):
        connector_cls(config)


def test_scheduler_producer_role_only_tracks_saves():
    sched = _scheduler()
    sched.kv_role = "kv_producer"
    sched._do_save = True
    sched._do_load = False
    sched._lookup_client = _LookupClient(hit=12)
    seq = SimpleNamespace(
        id=735,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
        has_per_req_cache=False,
    )

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()

    assert sched._load_specs == {}
    assert [req for req in meta.requests if req.load_spec is not None] == []
    assert "735" in sched._save_tracker


def test_scheduler_consumer_role_only_emits_loads_without_save_deferral():
    sched = _scheduler()
    sched.kv_role = "kv_consumer"
    sched._do_save = False
    sched._do_load = True
    sched._lookup_client = _LookupClient(hit=12)
    seq = SimpleNamespace(
        id=736,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
        has_per_req_cache=False,
    )

    assert sched.get_num_new_matched_tokens(seq) == (12, True)
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is True
    meta = sched.build_connector_meta()

    loads = [req for req in meta.requests if req.load_spec is not None]
    assert len(loads) == 1
    assert sched._save_tracker == {}
    assert sched.should_defer_free(seq) is True

    assert sched.load_finished(loads[0].load_operation) is True
    assert sched.should_defer_free(seq) is False


def test_lookup_unpin_ids_are_consumed_by_metadata_build():
    sched = _scheduler()
    sched._lookup_in_step = ["lookup-2"]

    meta = sched.build_connector_meta()

    assert meta.requests == []
    assert meta.lookup_requests_in_step == ["lookup-2"]
    assert sched._lookup_in_step == []


def test_engine_core_dispatches_idle_lookup_unpin_metadata(monkeypatch):
    fake_async_proc = types.ModuleType("atom.model_engine.async_proc")
    fake_async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(
        sys.modules,
        "atom.model_engine.async_proc",
        fake_async_proc,
    )
    from atom.model_engine.engine_core import EngineCore

    meta = LMCacheOffloadMetadata()
    meta.lookup_requests_in_step = ["lookup-3"]
    dispatched = []

    class _Connector:
        is_offload = True

        def build_connector_meta(self):
            return meta

    core = EngineCore.__new__(EngineCore)
    core.kv_transfer_enabled = True
    core.scheduler = SimpleNamespace(kv_connector=_Connector())
    core.runner_mgr = SimpleNamespace(
        call_func=lambda name, value: dispatched.append((name, value))
    )

    core._dispatch_idle_offload_work()

    assert dispatched == [("process_kvconnector_output", meta)]


def test_chained_prefix_hashes_match_block_manager_with_dcp_hash_size():
    config = MockConfig(
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        num_kvcache_blocks=20,
    )
    block_manager = BlockManager(config)
    tokens = list(range(32))

    actual = _chained_prefix_hashes(tokens, block_manager.hash_block_size)
    expected = {}
    parent = -1
    for boundary in range(
        block_manager.hash_block_size,
        len(tokens) + 1,
        block_manager.hash_block_size,
    ):
        block_tokens = tokens[boundary - block_manager.hash_block_size : boundary]
        parent = BlockManager.compute_hash(block_tokens, parent)
        expected[boundary] = parent

    assert block_manager.hash_block_size == 8
    assert actual == expected


def test_stateful_page_hit_shrinks_from_16k_to_committed_8k_sidecar():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(req_id=700, num_prompt_tokens=24_576)
    boundary_hash = _commit_sidecar(sched, seq, 8192)

    need, should_park = sched.get_num_new_matched_tokens(seq)

    assert (need, should_park) == (8192, True)
    assert sched._load_specs["700"].lmcache_cached_tokens == 8192
    assert sched._pending_slot_loads["700"] == (8192, boundary_hash)


def test_stateful_page_hit_selects_newest_committed_boundary_not_after_page():
    sched = _stateful_scheduler(hit=20_480)
    # PAGE chunks may advance more frequently than the 8K SLOT cadence.  The
    # composite hit must be the newest committed SLOT boundary covered by the
    # continuous PAGE prefix, not the PAGE frontier itself.
    sched.chunk_size = 256
    seq = _stateful_seq(req_id=743, num_prompt_tokens=32_768)
    _commit_sidecar(sched, seq, 8192)
    newest_hash = _commit_sidecar(sched, seq, 16_384)

    need, should_park = sched.get_num_new_matched_tokens(seq)

    assert (need, should_park) == (16_384, True)
    assert sched._load_specs["743"].lmcache_cached_tokens == 16_384
    assert sched._pending_slot_loads["743"] == (16_384, newest_hash)


def test_stateful_page_hit_ignores_committed_slot_after_page_frontier():
    sched = _stateful_scheduler(hit=12_288)
    sched.chunk_size = 256
    seq = _stateful_seq(req_id=744, num_prompt_tokens=24_576)
    previous_hash = _commit_sidecar(sched, seq, 8192)
    _commit_sidecar(sched, seq, 16_384)

    need, should_park = sched.get_num_new_matched_tokens(seq)

    assert (need, should_park) == (8192, True)
    assert sched._load_specs["744"].lmcache_cached_tokens == 8192
    assert sched._pending_slot_loads["744"] == (8192, previous_hash)


def test_stateful_page_hit_rejects_slot_from_different_prefix_hash():
    sched = _stateful_scheduler(hit=16_384)
    stored = _stateful_seq(req_id=745, num_prompt_tokens=24_576)
    _commit_sidecar(sched, stored, 8192)
    query = _stateful_seq(req_id=746, num_prompt_tokens=24_576)
    query.token_ids = [token + 100_000 for token in query.token_ids]

    result = sched.get_num_new_matched_tokens(query)

    assert result == (0, False)
    assert sched._load_specs == {}
    assert sched._pending_slot_loads == {}
    assert sched._lookup_client.cleared == ["746"]


def test_stateful_page_hit_without_sidecar_is_rejected_and_lookup_cleared():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(req_id=701, num_prompt_tokens=24_576)

    result = sched.get_num_new_matched_tokens(seq)

    assert result == (0, False)
    assert sched._load_specs == {}
    assert sched._pending_slot_loads == {}
    assert sched._lookup_client.cleared == ["701"]


def test_new_scheduler_session_fails_closed_without_rebuilt_slot_index():
    original = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(req_id=747, num_prompt_tokens=24_576)
    boundary_hash = _commit_sidecar(original, seq, 8192)
    assert boundary_hash in original._committed_sidecar_hashes

    # A new scheduler can rediscover PAGE through LMCache lookup, but the
    # worker-side opaque SLOT objects are not part of that token lookup.  Until
    # a persistent SLOT lookup adapter rebuilds the index, fail closed.
    restarted = _stateful_scheduler(hit=16_384)
    replacement = _stateful_seq(req_id=748, num_prompt_tokens=24_576)

    result = restarted.get_num_new_matched_tokens(replacement)

    assert result == (0, False)
    assert restarted._committed_sidecar_hashes == set()
    assert restarted._pending_slot_loads == {}
    assert restarted._lookup_client.cleared == ["748"]


def test_stateful_lookup_fails_closed_after_commit_index_eviction():
    sched = _stateful_scheduler(hit=16_384)
    sched._committed_sidecar_hashes = connector_module._BoundedLRUSet(1)
    stale = _stateful_seq(req_id=740, num_prompt_tokens=24_576)
    recent = _stateful_seq(req_id=741, num_prompt_tokens=24_576)
    recent.token_ids = [token + 100_000 for token in recent.token_ids]
    stale_hash = _commit_sidecar(sched, stale, 8192)
    recent_hash = _commit_sidecar(sched, recent, 8192)

    result = sched.get_num_new_matched_tokens(stale)

    assert stale_hash not in sched._committed_sidecar_hashes
    assert recent_hash in sched._committed_sidecar_hashes
    assert result == (0, False)
    assert sched._pending_slot_loads == {}
    assert sched._lookup_client.cleared == ["740"]


def test_request_cleanup_retains_recent_global_commit():
    sched = _stateful_scheduler(hit=16_384)
    sched._committed_sidecar_hashes = connector_module._BoundedLRUSet(2)
    seq = _stateful_seq(req_id=742, num_prompt_tokens=24_576)
    boundary_hash = _commit_sidecar(sched, seq, 8192)

    sched.request_finished(seq)

    assert boundary_hash in sched._committed_sidecar_hashes


@pytest.mark.parametrize(
    ("lookup", "num_cached_tokens"),
    [
        (_LookupClient(hit=0), 0),
        (_FailingLookupClient(hit=0), 0),
        (_LookupClient(hit=8192), 8192),
    ],
    ids=["no-hit", "exception", "need-not-positive"],
)
def test_page_lookup_retry_clears_stale_load_state(lookup, num_cached_tokens):
    sched = _stateful_scheduler(hit=0)
    sched._lookup_client = lookup
    seq = SimpleNamespace(
        id=720,
        num_prompt_tokens=16_384,
        token_ids=list(range(16_384)),
        num_cached_tokens=num_cached_tokens,
        block_table=list(range(64)),
        has_per_req_cache=False,
    )
    sched._load_specs["720"] = object()
    sched._pending_slot_loads["720"] = (8192, 123)
    sched._hit_save_floors["720"] = 8192

    result = sched.get_num_new_matched_tokens(seq)

    assert result == (0, False)
    assert "720" not in sched._load_specs
    assert "720" not in sched._pending_slot_loads
    assert "720" not in sched._hit_save_floors
    assert lookup.cleared == ["720"]

    seq.num_cached_tokens = 8192
    sched.update_state_after_alloc(seq)
    save_meta = sched.build_connector_meta()

    assert sched._save_tracker["720"][1] == 8192
    assert len(save_meta.requests) == 1
    assert save_meta.requests[0].save_spec.skip_leading_tokens == 0
    assert save_meta.requests[0].token_ids == seq.token_ids[:8192]


def test_stateless_page_lookup_remains_unchanged():
    sched = _stateful_scheduler(hit=16_384)
    seq = SimpleNamespace(
        id=702,
        num_prompt_tokens=24_576,
        token_ids=list(range(24_576)),
        num_cached_tokens=0,
        has_per_req_cache=False,
    )

    result = sched.get_num_new_matched_tokens(seq)

    assert result == (16_384, True)
    assert sched._load_specs["702"].lmcache_cached_tokens == 16_384
    assert sched._pending_slot_loads == {}


def test_stateful_full_prompt_hit_uses_prior_aligned_sidecar():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(req_id=703, num_prompt_tokens=16_384)
    previous_hash = _commit_sidecar(sched, seq, 8192)
    _commit_sidecar(sched, seq, 16_384)

    need, should_park = sched.get_num_new_matched_tokens(seq)

    assert (need, should_park) == (8192, True)
    assert sched._load_specs["703"].lmcache_cached_tokens == 8192
    assert sched._pending_slot_loads["703"] == (8192, previous_hash)


def test_stateful_load_metadata_includes_slot_destination_group():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(req_id=704, num_prompt_tokens=24_576)
    boundary_hash = _commit_sidecar(sched, seq, 8192)
    sched.get_num_new_matched_tokens(seq)
    seq.state_slot = 3

    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is True
    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    request = meta.requests[0]
    assert request.load_spec.lmcache_cached_tokens == 8192
    assert request.slot_load_spec == SlotLoadSpec(
        boundary_tokens=8192,
        boundary_block_hash=boundary_hash,
        destination_group=3,
    )
    assert sched._pending_slot_loads == {}


def test_load_metadata_uses_scheduler_lifetime_generation_on_reused_req_id():
    sched = _stateful_scheduler(hit=16_384)
    operations = []
    for group in (3, 4):
        seq = _stateful_seq(
            req_id=733,
            num_prompt_tokens=24_576,
            group=group,
        )
        seq.offload_loaded = True
        seq.offload_load_failed = True
        _commit_sidecar(sched, seq, 8192)
        sched.get_num_new_matched_tokens(seq)
        sched.update_state_after_alloc(seq)
        request = sched.build_connector_meta().requests[0]
        operations.append(request.load_operation)
        assert seq.offload_loaded is False
        assert seq.offload_load_failed is False
        assert seq._load_operation == request.load_operation
        sched.load_finished(request.load_operation)
        sched.request_finished(seq)

    assert operations == [
        LoadOperationId(733, 0),
        LoadOperationId(733, 1),
    ]


def test_late_load_terminal_cannot_mutate_reused_request_lifecycle():
    sched = _scheduler()
    sched._lookup_client = _LookupClient(hit=12)

    def emit(seq):
        sched.get_num_new_matched_tokens(seq)
        sched.update_state_after_alloc(seq)
        return sched.build_connector_meta().requests[0].load_operation

    old = SimpleNamespace(
        id=737,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
        has_per_req_cache=False,
    )
    new = SimpleNamespace(
        id=737,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[5, 6, 7, 8],
        has_per_req_cache=False,
    )
    old_operation = emit(old)
    new_operation = emit(new)

    sched.load_failed(old_operation)

    assert sched._active_load_operations["737"] == (new, new_operation)
    assert sched._load_save_floors["737"] == 0

    sched.load_finished(new_operation)
    assert sched._active_load_operations == {}


def test_save_tracker_resets_when_raw_req_id_gets_new_sequence_lifecycle():
    sched = _scheduler()
    old = _stateful_seq(
        req_id=734,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    new = _stateful_seq(req_id=734, num_prompt_tokens=16_384, group=3)
    sched._save_tracker["734"] = [old, 8192]
    sched._sidecar_hash_cache["734"] = (old, {8192: 1}, [(8192, 1)])
    sched._failed_sidecar_saves["734"] = {(8192, 1)}

    sched.update_state_after_alloc(new)

    assert sched._save_tracker["734"] == [new, 0]
    assert "734" not in sched._sidecar_hash_cache
    assert "734" not in sched._failed_sidecar_saves


def test_stateful_load_rejects_nonzero_post_alloc_hbm_floor():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(req_id=721, num_prompt_tokens=24_576)
    _commit_sidecar(sched, seq, 8192)
    sched.get_num_new_matched_tokens(seq)
    seq.state_slot = 3
    # Faithfully models BlockManager's post-allocation state-copy/prefix hit:
    # a group is assigned and the HBM frontier advances before offload decides.
    seq.num_cached_tokens = 256

    sched.update_state_after_alloc(seq)

    assert sched.should_park_for_load_after_alloc(seq) is False
    assert sched.build_connector_meta().requests == []
    assert sched._pending_slot_loads == {}
    assert sched._load_specs == {}
    assert seq.offload_loaded_tokens == 256
    assert sched._lookup_client.cleared == ["721"]


def test_stateful_load_tracks_identity_until_terminal_callback():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(
        req_id=722,
        num_prompt_tokens=24_576,
        group=3,
    )
    boundary_hash = _commit_sidecar(sched, seq, 8192)
    sched.get_num_new_matched_tokens(seq)
    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()
    load_operation = meta.requests[0].load_operation

    assert sched._active_slot_loads["722"] == (8192, boundary_hash)

    assert sched.load_finished(load_operation) is True
    assert sched.load_finished(load_operation) is False

    assert sched._active_slot_loads == {}
    assert boundary_hash in sched._committed_sidecar_hashes


def test_stateful_load_failure_evicts_committed_boundary_idempotently():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(
        req_id=723,
        num_prompt_tokens=24_576,
        group=3,
    )
    boundary_hash = _commit_sidecar(sched, seq, 8192)
    sched.get_num_new_matched_tokens(seq)
    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()
    load_operation = meta.requests[0].load_operation

    assert sched.load_failed(load_operation) is True
    assert sched.load_failed(load_operation) is False

    assert sched._active_slot_loads == {}
    assert boundary_hash not in sched._committed_sidecar_hashes


def test_stateful_load_failure_falls_back_to_older_committed_boundary():
    sched = _stateful_scheduler(hit=20_480)
    sched.chunk_size = 256
    first = _stateful_seq(
        req_id=749,
        num_prompt_tokens=32_768,
        group=3,
    )
    older_hash = _commit_sidecar(sched, first, 8192)
    failed_hash = _commit_sidecar(sched, first, 16_384)

    sched.get_num_new_matched_tokens(first)
    sched.update_state_after_alloc(first)
    request = sched.build_connector_meta().requests[0]
    assert request.slot_load_spec.boundary_tokens == 16_384

    assert sched.load_failed(request.load_operation) is True
    assert failed_hash not in sched._committed_sidecar_hashes
    assert older_hash in sched._committed_sidecar_hashes

    retry = _stateful_seq(
        req_id=750,
        num_prompt_tokens=32_768,
        group=4,
    )
    need, should_park = sched.get_num_new_matched_tokens(retry)

    assert (need, should_park) == (8192, True)
    assert sched._pending_slot_loads["750"] == (8192, older_hash)


def test_sidecar_hashes_are_computed_once_per_sequence_lifecycle(monkeypatch):
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(
        req_id=724,
        num_prompt_tokens=32_768,
        num_cached_tokens=16_384,
        group=3,
    )
    calls = []
    original = _chained_prefix_hashes

    def counted(token_ids, hash_block_size):
        calls.append(len(token_ids) // hash_block_size)
        return original(token_ids, hash_block_size)

    monkeypatch.setattr(
        "atom.kv_transfer.offload.hybrid.dsv4.connector._chained_prefix_hashes",
        counted,
    )

    records = sched._sidecar_boundary_records(seq)
    sched._sidecar_boundary_records(seq)
    sched._next_pending_sidecar_boundary(seq, 0, 32_768)
    sched._sidecar_save_candidate(seq, 16_384)

    assert records
    assert calls == [128]

    replacement = _stateful_seq(
        req_id=724,
        num_prompt_tokens=8192,
        num_cached_tokens=8192,
        group=3,
    )
    sched._sidecar_boundary_records(replacement)
    assert calls == [128, 32]
    sched.request_finished(replacement)
    assert sched._sidecar_hash_cache == {}


def test_stateful_load_without_destination_group_fails_closed():
    sched = _stateful_scheduler(hit=16_384)
    seq = _stateful_seq(req_id=705, num_prompt_tokens=24_576)
    _commit_sidecar(sched, seq, 8192)
    sched.get_num_new_matched_tokens(seq)

    sched.update_state_after_alloc(seq)

    assert sched.should_park_for_load_after_alloc(seq) is False
    assert sched.build_connector_meta().requests == []
    assert sched._pending_slot_loads == {}
    assert sched._lookup_client.cleared == ["705"]


def test_sidecar_save_emits_at_regular_checkpoint_boundary():
    sched = _stateful_scheduler(hit=0)
    sched.sidecar_interval = 16_384
    seq = _stateful_seq(
        req_id=706,
        num_prompt_tokens=24_576,
        num_cached_tokens=16_384,
        group=2,
    )
    sched._save_tracker["706"] = [seq, 0]
    boundary_hash = _chained_prefix_hashes(seq.token_ids, 256)[16_384]

    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    request = meta.requests[0]
    assert request.save_spec is not None
    assert request.slot_save_spec == SlotSaveSpec(
        boundary_tokens=16_384,
        boundary_block_hash=boundary_hash,
        source_group=2,
    )
    assert sched._sidecar_save_inflight["706"] == (
        request.save_operation,
        16_384,
        boundary_hash,
    )


def test_slot_save_waits_until_post_allocation_state_copy_forward_completes():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=732,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=5,
    )
    seq._state_initialized_after_alloc = False
    sched._save_tracker["732"] = [seq, 0]
    destination_slot = {"value": "garbage"}
    snapshots = []

    before_forward = sched.build_connector_meta()
    before_request = before_forward.requests[0]
    if before_request.slot_save_spec is not None:
        snapshots.append(destination_slot["value"])

    assert before_request.save_spec is not None
    assert before_request.slot_save_spec is None
    assert snapshots == []

    # StateTransfer.copy initializes the live destination inside forward;
    # postprocess publishes hashes only after that forward has completed.
    destination_slot["value"] = "initialized-from-checkpoint"
    seq._state_initialized_after_alloc = True

    after_forward = sched.build_connector_meta()
    after_request = after_forward.requests[0]
    if after_request.slot_save_spec is not None:
        snapshots.append(destination_slot["value"])

    assert after_request.save_spec is None
    assert after_request.slot_save_spec.boundary_tokens == 8192
    assert after_request.slot_save_spec.source_group == 5
    assert snapshots == ["initialized-from-checkpoint"]


def test_off_interval_terminal_saves_page_without_slot():
    sched = _stateful_scheduler(hit=0)
    sched.sidecar_interval = 16_384
    seq = _stateful_seq(
        req_id=707,
        num_prompt_tokens=24_576,
        num_cached_tokens=24_576,
        group=2,
    )
    sched._save_tracker["707"] = [seq, 16_384]
    regular_hash = _chained_prefix_hashes(seq.token_ids, 256)[16_384]
    sched._committed_sidecar_hashes.add(regular_hash)
    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    request = meta.requests[0]
    assert request.save_spec.skip_leading_tokens == 16_384
    assert request.slot_save_spec is None


def test_zero_sidecar_interval_disables_all_slot_saves():
    sched = _stateful_scheduler(hit=0)
    sched.sidecar_interval = 0
    seq = _stateful_seq(
        req_id=714,
        num_prompt_tokens=24_576,
        num_cached_tokens=16_384,
        group=2,
    )
    sched._save_tracker["714"] = [seq, 24_576]

    assert sched.build_connector_meta().requests == []

    seq.num_cached_tokens = 24_576
    assert sched.build_connector_meta().requests == []


def test_sidecar_only_save_emits_when_page_boundary_is_already_stored():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=708,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["708"] = [seq, 8192]
    boundary_hash = _chained_prefix_hashes(seq.token_ids, 256)[8192]

    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    request = meta.requests[0]
    assert request.save_spec is None
    assert request.token_ids == seq.token_ids[:8192]
    assert request.slot_save_spec == SlotSaveSpec(8192, boundary_hash, 2)


def test_page_save_inflight_still_cuts_and_emits_exact_sidecar_boundary():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=721,
        num_prompt_tokens=16_384,
        num_cached_tokens=4096,
        group=2,
    )
    sched._save_tracker["721"] = [seq, 4096]
    prior_operation = SaveOperationId(seq.id, 0)
    sched._save_nonce = 1
    sched._save_inflight["721"] = {prior_operation}

    assert sched.adjust_prefill_chunk_after_alloc(seq, 8192) == 4096

    seq.num_cached_tokens = 8192
    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    request = meta.requests[0]
    assert request.save_spec.skip_leading_tokens == 4096
    assert request.token_ids == seq.token_ids[:8192]
    assert request.slot_save_spec.boundary_tokens == 8192
    assert sched._save_tracker["721"][1] == 8192
    assert sched._save_inflight["721"] == {
        prior_operation,
        request.save_operation,
    }
    assert sched._sidecar_save_inflight["721"][:2] == (
        request.save_operation,
        8192,
    )


def test_save_callbacks_clear_only_matching_operation_generation():
    sched = _stateful_scheduler(hit=0)
    sched.chunk_size = 4096
    sched._save_inflight = {}
    sched._save_nonce = 0
    seq = _stateful_seq(
        req_id=725,
        num_prompt_tokens=16_384,
        num_cached_tokens=4096,
        group=2,
    )
    sched._save_tracker["725"] = [seq, 0]

    page = sched.build_connector_meta().requests[0]
    assert page.save_operation == SaveOperationId(seq.id, 0)

    seq.num_cached_tokens = 8192
    boundary = sched.build_connector_meta().requests[0]
    assert boundary.save_operation == SaveOperationId(seq.id, 1)
    assert sched._save_inflight["725"] == {
        SaveOperationId(seq.id, 0),
        SaveOperationId(seq.id, 1),
    }
    assert sched._sidecar_save_inflight["725"][0] == SaveOperationId(seq.id, 1)

    sched.sidecar_save_finished(page.save_operation)
    assert "725" in sched._sidecar_save_inflight
    assert sched._committed_sidecar_hashes == set()

    sched.save_finished(page.save_operation)
    assert sched._save_inflight["725"] == {SaveOperationId(seq.id, 1)}

    boundary_hash = boundary.slot_save_spec.boundary_block_hash
    sched.sidecar_save_finished(boundary.save_operation)
    assert boundary_hash in sched._committed_sidecar_hashes
    assert "725" not in sched._sidecar_save_inflight

    sched.save_finished(boundary.save_operation)
    assert sched._save_inflight == {}
    assert sched.should_defer_free(seq) is False


def test_raw_callbacks_cannot_retire_exact_active_operations():
    page_sched = _scheduler()
    page_seq = SimpleNamespace(
        id=728,
        token_ids=list(range(8)),
        block_table=[1, 2],
        num_prompt_tokens=8,
        num_cached_tokens=8,
        has_per_req_cache=False,
    )
    page_sched._save_tracker["728"] = [page_seq, 0]
    page_request = page_sched.build_connector_meta().requests[0]

    page_sched.save_finished(page_seq.id)

    assert page_sched._save_inflight["728"] == {page_request.save_operation}

    stateful_sched = _stateful_scheduler(hit=0)
    stateful_seq = _stateful_seq(
        req_id=729,
        num_prompt_tokens=8192,
        num_cached_tokens=8192,
        group=2,
    )
    stateful_sched._save_tracker["729"] = [stateful_seq, 8192]
    sidecar_request = stateful_sched.build_connector_meta().requests[0]
    boundary_hash = sidecar_request.slot_save_spec.boundary_block_hash

    stateful_sched.sidecar_save_finished(stateful_seq.id)
    stateful_sched.sidecar_save_failed(stateful_seq.id)

    assert stateful_sched._sidecar_save_inflight["729"][0] == (
        sidecar_request.save_operation
    )
    assert boundary_hash not in stateful_sched._committed_sidecar_hashes
    assert stateful_sched._failed_sidecar_saves == {}

    load_operation = LoadOperationId(stateful_seq.id, 99)
    stateful_sched._active_load_operations["729"] = (
        stateful_seq,
        load_operation,
    )
    stateful_sched._active_slot_loads["729"] = (8192, boundary_hash)
    stateful_sched._committed_sidecar_hashes.add(boundary_hash)

    assert stateful_sched.load_finished(stateful_seq.id) is False
    assert stateful_sched.load_failed(stateful_seq.id) is False
    assert stateful_sched._active_load_operations["729"] == (
        stateful_seq,
        load_operation,
    )
    assert stateful_sched._active_slot_loads["729"] == (8192, boundary_hash)
    assert boundary_hash in stateful_sched._committed_sidecar_hashes


def test_raw_callbacks_remain_compatible_with_legacy_operations():
    sched = _scheduler()
    seq = SimpleNamespace(
        id=730,
        num_prompt_tokens=8,
        num_cached_tokens=8,
    )
    sid = str(seq.id)

    sched._save_inflight[sid] = {seq.id}
    sched.save_finished(seq.id)
    assert sid not in sched._save_inflight

    sched._sidecar_save_inflight[sid] = (seq.id, 8, 123)
    sched.sidecar_save_finished(seq.id)
    assert sid not in sched._sidecar_save_inflight
    assert 123 in sched._committed_sidecar_hashes

    sched._sidecar_save_inflight[sid] = (seq.id, 8, 456)
    sched.sidecar_save_failed(seq.id)
    assert sid not in sched._sidecar_save_inflight
    assert sched._failed_sidecar_saves[sid] == {(8, 456)}

    sched._active_slot_loads[sid] = (8, 123)
    sched._load_save_floors[sid] = 4
    sched._save_tracker[sid] = [seq, 8]
    assert sched.load_failed(seq.id) is True
    assert sid not in sched._active_slot_loads
    assert 123 not in sched._committed_sidecar_hashes
    assert sched._save_tracker[sid][1] == 4

    sched._active_slot_loads[sid] = (8, 456)
    sched._load_save_floors[sid] = 4
    assert sched.load_finished(seq.id) is True
    assert sid not in sched._active_slot_loads
    assert sid not in sched._load_save_floors


def test_save_nonce_is_never_reused_after_request_id_cleanup():
    sched = _stateful_scheduler(hit=0)
    first_seq = _stateful_seq(req_id=727, num_prompt_tokens=8192, group=2)
    first = sched._next_save_operation(first_seq)

    sched.request_finished(first_seq)

    reused_seq = _stateful_seq(req_id=727, num_prompt_tokens=8192, group=2)
    second = sched._next_save_operation(reused_seq)
    sched._save_inflight["727"] = {second}
    sched.save_finished(first)

    assert first.req_id == second.req_id
    assert second.generation > first.generation
    assert sched._save_inflight["727"] == {second}


def test_tp_cross_generation_reports_do_not_clear_or_commit_until_matched():
    sched = _stateful_scheduler(hit=0)
    sched.chunk_size = 4096
    seq = _stateful_seq(
        req_id=726,
        num_prompt_tokens=16_384,
        num_cached_tokens=4096,
        group=2,
    )
    sched._save_tracker["726"] = [seq, 0]
    page = sched.build_connector_meta().requests[0]
    seq.num_cached_tokens = 8192
    boundary = sched.build_connector_meta().requests[0]
    boundary_hash = boundary.slot_save_spec.boundary_block_hash
    aggregator = KVOutputAggregator(world_size=2)
    host = Scheduler.__new__(Scheduler)
    host.kv_connector = sched
    host.deferred_free_blocks = {}
    host.finished_recving_kv_req_ids = []
    host.failed_recving_kv_req_ids = []

    mixed = aggregator.aggregate(
        [
            KVConnectorOutput(finished_saving={page.save_operation}),
            KVConnectorOutput(
                finished_saving={boundary.save_operation},
                connector_completions={
                    ConnectorCompletion(
                        DSV4_CHECKPOINT_SAVE_CHANNEL,
                        boundary.save_operation,
                        True,
                    )
                },
            ),
        ]
    )
    host._update_from_kv_xfer_finished(mixed)

    assert sched._save_inflight["726"] == {
        page.save_operation,
        boundary.save_operation,
    }
    assert boundary_hash not in sched._committed_sidecar_hashes

    matched = aggregator.aggregate(
        [
            KVConnectorOutput(
                finished_saving={boundary.save_operation},
                connector_completions={
                    ConnectorCompletion(
                        DSV4_CHECKPOINT_SAVE_CHANNEL,
                        boundary.save_operation,
                        True,
                    )
                },
            ),
            KVConnectorOutput(finished_saving={page.save_operation}),
        ]
    )
    host._update_from_kv_xfer_finished(matched)

    assert sched._save_inflight == {}
    assert sched._sidecar_save_inflight == {}
    assert boundary_hash in sched._committed_sidecar_hashes
    assert aggregator.pending_count == (0, 0)


def test_earlier_save_completion_clears_only_its_page_generation():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=722,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["722"] = [seq, 4096]
    prior_operation = SaveOperationId(seq.id, 0)
    sched._save_nonce = 1
    sched._save_inflight["722"] = {prior_operation}
    meta = sched.build_connector_meta()
    assert meta.requests[0].save_spec.skip_leading_tokens == 4096
    boundary_operation = meta.requests[0].save_operation

    sched.save_finished(prior_operation)

    assert sched._save_inflight["722"] == {boundary_operation}
    assert "722" in sched._sidecar_save_inflight

    sched.sidecar_save_finished(boundary_operation)
    assert sched._save_inflight["722"] == {boundary_operation}

    sched.save_finished(boundary_operation)
    assert "722" not in sched._save_inflight
    assert "722" not in sched._sidecar_save_inflight


def test_collapsed_sidecar_and_save_completion_clears_page_inflight():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=723,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["723"] = [seq, 4096]
    prior_operation = SaveOperationId(seq.id, 0)
    sched._save_nonce = 1
    sched._save_inflight["723"] = {prior_operation}
    boundary_operation = sched.build_connector_meta().requests[0].save_operation
    host = Scheduler.__new__(Scheduler)
    host.kv_connector = sched
    host.deferred_free_blocks = {}
    host.finished_recving_kv_req_ids = []
    host.failed_recving_kv_req_ids = []

    host._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_saving={prior_operation, boundary_operation},
            connector_completions={
                ConnectorCompletion(
                    DSV4_CHECKPOINT_SAVE_CHANNEL,
                    boundary_operation,
                    True,
                )
            },
        )
    )

    assert "723" not in sched._save_inflight
    assert "723" not in sched._sidecar_save_inflight


@pytest.mark.parametrize(
    ("field", "callback"),
    [
        ("finished_loading", "load_finished"),
        ("failed_loading", "load_failed"),
    ],
)
def test_offload_completion_processing_calls_load_terminal_once(field, callback):
    calls = []

    class _Connector(OffloadSchedulerMixin):
        is_producer = False
        is_offload = True

        def load_finished(self, req_id):
            calls.append(("load_finished", req_id))

        def load_failed(self, req_id):
            calls.append(("load_failed", req_id))

    output = _Connector().process_completions(KVConnectorOutput(**{field: {725}}))

    assert calls == [(callback, 725)]
    assert getattr(output, field) == {725}


@pytest.mark.parametrize(
    ("field", "terminal_callback"),
    [
        ("finished_loading", "load_finished"),
        ("failed_loading", "load_failed"),
    ],
)
def test_aborted_parked_load_defers_owned_resources_until_terminal(
    field,
    terminal_callback,
):
    events = []
    seq = SimpleNamespace(
        id=729,
        status=SequenceStatus.ABORTED,
        block_table=[10, 11],
        has_per_req_cache=True,
        state_slot=3,
        _counted_as_inflight_load=True,
    )

    class _Connector(OffloadSchedulerMixin):
        is_producer = False
        is_offload = True

        def load_finished(self, req_id):
            events.append(("load_finished", req_id))

        def load_failed(self, req_id):
            events.append(("load_failed", req_id))

        def request_finished(self, value):
            events.append(("request_finished", value.id))

    def deallocate(value):
        events.append(("deallocate", value.id))
        value.block_table.clear()
        value.state_slot = -1

    host = Scheduler.__new__(Scheduler)
    host.kv_connector = _Connector()
    host.block_manager = SimpleNamespace(deallocate=deallocate)
    host.deferred_free_blocks = {}
    host.finished_recving_kv_req_ids = []
    host.failed_recving_kv_req_ids = []
    host._num_parked_remote_kv = 1
    host._rejected = []

    host._reject_aborted_waiting(seq)

    assert host._rejected == [seq]
    assert host.deferred_free_blocks[seq.id] is seq
    assert seq._awaiting_aborted_load_cleanup is True
    assert seq.block_table == [10, 11]
    assert seq.state_slot == 3
    assert host._num_parked_remote_kv == 1

    host._update_from_kv_xfer_finished(KVConnectorOutput(**{field: {seq.id}}))

    assert events == [
        (terminal_callback, seq.id),
        ("request_finished", seq.id),
        ("deallocate", seq.id),
    ]
    assert host.deferred_free_blocks == {}
    assert host._num_parked_remote_kv == 0
    assert seq.block_table == []
    assert seq.state_slot == -1
    assert host.finished_recving_kv_req_ids == []
    assert host.failed_recving_kv_req_ids == []


@pytest.mark.parametrize(
    "queued_field",
    ["finished_recving_kv_req_ids", "failed_recving_kv_req_ids"],
)
def test_aborted_parked_load_consumes_already_queued_terminal(queued_field):
    events = []
    seq = SimpleNamespace(
        id=731,
        status=SequenceStatus.ABORTED,
        block_table=[12, 13],
        has_per_req_cache=True,
        state_slot=4,
        _counted_as_inflight_load=True,
    )

    class _Connector(OffloadSchedulerMixin):
        is_producer = False
        is_offload = True

        def load_finished(self, req_id):
            events.append(("unexpected_load_finished", req_id))

        def load_failed(self, req_id):
            events.append(("unexpected_load_failed", req_id))

        def request_finished(self, value):
            events.append(("request_finished", value.id))

    def deallocate(value):
        events.append(("deallocate", value.id))
        value.block_table.clear()
        value.state_slot = -1

    host = Scheduler.__new__(Scheduler)
    host.kv_connector = _Connector()
    host.block_manager = SimpleNamespace(deallocate=deallocate)
    host.deferred_free_blocks = {}
    host.finished_recving_kv_req_ids = []
    host.failed_recving_kv_req_ids = []
    getattr(host, queued_field).append(seq.id)
    host._num_parked_remote_kv = 1
    host._rejected = []

    host._reject_aborted_waiting(seq)

    assert events == [
        ("request_finished", seq.id),
        ("deallocate", seq.id),
    ]
    assert host._rejected == [seq]
    assert host.deferred_free_blocks == {}
    assert host.finished_recving_kv_req_ids == []
    assert host.failed_recving_kv_req_ids == []
    assert host._num_parked_remote_kv == 0
    assert seq.block_table == []
    assert seq.state_slot == -1
    assert not hasattr(seq, "_awaiting_aborted_load_cleanup")


@pytest.mark.parametrize(
    ("terminal_field", "terminal_callback", "consumed_flag"),
    [
        ("finished_loading", "load_finished", "offload_loaded"),
        ("failed_loading", "load_failed", "offload_load_failed"),
    ],
)
def test_abort_cleans_load_whose_terminal_was_already_consumed(
    terminal_field,
    terminal_callback,
    consumed_flag,
):
    events = []
    seq = SimpleNamespace(
        id=738,
        status=SequenceStatus.WAITING_FOR_REMOTE_KVS,
        block_table=[14, 15],
        has_per_req_cache=True,
        state_slot=5,
        _counted_as_inflight_load=True,
        num_cached_tokens=4,
        num_tokens=8,
        offload_loaded_tokens=4,
        offload_load_start_tokens=4,
    )

    class _Connector(OffloadSchedulerMixin):
        is_producer = False
        is_offload = True

        def load_finished(self, req_id):
            events.append(("load_finished", req_id))
            return True

        def load_failed(self, req_id):
            events.append(("load_failed", req_id))
            return True

        def request_finished(self, value):
            events.append(("request_finished", value.id))

    def deallocate(value):
        events.append(("deallocate", value.id))
        value.block_table.clear()
        value.state_slot = -1

    host = Scheduler.__new__(Scheduler)
    host.kv_connector = _Connector()
    host.block_manager = SimpleNamespace(
        kv_events_enabled=False,
        deallocate=deallocate,
    )
    host.deferred_free_blocks = {}
    host.finished_recving_kv_req_ids = []
    host.failed_recving_kv_req_ids = []
    host._num_parked_remote_kv = 1
    host._rejected = []

    host._update_from_kv_xfer_finished(KVConnectorOutput(**{terminal_field: {seq.id}}))
    assert host._resolve_waiting_remote_kv(seq, deque()) is False
    assert getattr(seq, consumed_flag) is True
    assert host.finished_recving_kv_req_ids == []
    assert host.failed_recving_kv_req_ids == []

    seq.status = SequenceStatus.ABORTED
    host._reject_aborted_waiting(seq)

    assert events == [
        (terminal_callback, seq.id),
        ("request_finished", seq.id),
        ("deallocate", seq.id),
    ]
    assert host.deferred_free_blocks == {}
    assert host._num_parked_remote_kv == 0
    assert seq.block_table == []
    assert seq.state_slot == -1
    assert not hasattr(seq, "_awaiting_aborted_load_cleanup")


def test_consume_failed_remote_kv_does_not_repeat_terminal_callback():
    calls = []
    seq = SimpleNamespace(
        id=730,
        status=SequenceStatus.WAITING_FOR_REMOTE_KVS,
        num_cached_tokens=4,
    )

    class _Connector:
        is_producer = False
        is_offload = True

        def load_failed(self, req_id):
            calls.append(req_id)

    host = Scheduler.__new__(Scheduler)
    host.kv_connector = _Connector()
    host.failed_recving_kv_req_ids = [seq.id]
    host._num_parked_remote_kv = 0

    assert host._consume_failed_remote_kv(seq) is True
    assert calls == []


def test_sidecar_save_waits_for_exact_completed_boundary():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=709,
        num_prompt_tokens=16_384,
        num_cached_tokens=8191,
        group=2,
    )
    sched._save_tracker["709"] = [seq, 8192]

    assert sched.build_connector_meta().requests == []

    seq.num_cached_tokens = 8192
    meta = sched.build_connector_meta()

    assert len(meta.requests) == 1
    assert meta.requests[0].slot_save_spec.boundary_tokens == 8192


def test_sidecar_save_is_not_duplicated_while_inflight_or_after_commit():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=710,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["710"] = [seq, 8192]

    first = sched.build_connector_meta()
    save_operation = first.requests[0].save_operation
    boundary_hash = first.requests[0].slot_save_spec.boundary_block_hash
    assert sched.build_connector_meta().requests == []

    sched.save_finished(save_operation)
    assert sched.build_connector_meta().requests == []

    sched.sidecar_save_finished(save_operation)
    assert boundary_hash in sched._committed_sidecar_hashes
    assert sched.build_connector_meta().requests == []


def test_terminal_sidecar_is_emitted_after_earlier_inflight_boundary_completes():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=720,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["720"] = [seq, 8192]

    first = sched.build_connector_meta()
    first_operation = first.requests[0].save_operation
    assert first.requests[0].slot_save_spec.boundary_tokens == 8192

    # The request reaches its terminal boundary while B1 is still publishing.
    seq.num_cached_tokens = 16_384
    sched.request_finished(seq)
    assert sched.should_defer_free(seq) is True
    page_only = sched.build_connector_meta()
    assert len(page_only.requests) == 1
    assert page_only.requests[0].slot_save_spec is None
    sched.save_finished(page_only.requests[0].save_operation)

    sched.sidecar_save_finished(first_operation)
    assert sched.should_defer_free(seq) is True

    terminal = sched.build_connector_meta()
    assert len(terminal.requests) == 1
    assert terminal.requests[0].slot_save_spec.boundary_tokens == 16_384


def test_sidecar_save_failure_clears_inflight_without_committing():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=715,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["715"] = [seq, 8192]
    meta = sched.build_connector_meta()
    save_operation = meta.requests[0].save_operation
    boundary_hash = meta.requests[0].slot_save_spec.boundary_block_hash

    sched.sidecar_save_failed(save_operation)

    assert "715" not in sched._sidecar_save_inflight
    assert boundary_hash not in sched._committed_sidecar_hashes


def test_failed_terminal_sidecar_does_not_pin_or_retry_finished_request():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=719,
        num_prompt_tokens=8192,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["719"] = [seq, 8192]
    meta = sched.build_connector_meta()
    save_operation = meta.requests[0].save_operation
    boundary_hash = meta.requests[0].slot_save_spec.boundary_block_hash

    sched.request_finished(seq)
    sched.sidecar_save_failed(save_operation)
    sched.save_finished(save_operation)

    assert boundary_hash not in sched._committed_sidecar_hashes
    assert sched.should_defer_free(seq) is False
    sched.request_finished(seq)
    assert "719" not in sched._save_tracker
    assert sched.build_connector_meta().requests == []


def test_tp_partial_sidecar_completion_never_commits():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=716,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["716"] = [seq, 8192]
    meta = sched.build_connector_meta()
    save_operation = meta.requests[0].save_operation
    boundary_hash = meta.requests[0].slot_save_spec.boundary_block_hash
    host = Scheduler.__new__(Scheduler)
    host.kv_connector = sched
    host.deferred_free_blocks = {}
    aggregator = KVOutputAggregator(world_size=2)

    partial = aggregator.aggregate(
        [
            KVConnectorOutput(
                connector_completions={
                    ConnectorCompletion(
                        DSV4_CHECKPOINT_SAVE_CHANNEL,
                        save_operation,
                        True,
                    )
                }
            ),
            KVConnectorOutput(),
        ]
    )
    host._update_from_kv_xfer_finished(partial)

    assert boundary_hash not in sched._committed_sidecar_hashes
    assert sched._sidecar_save_inflight["716"] == (
        save_operation,
        8192,
        boundary_hash,
    )

    terminal = aggregator.aggregate(
        [
            KVConnectorOutput(),
            KVConnectorOutput(
                connector_completions={
                    ConnectorCompletion(
                        DSV4_CHECKPOINT_SAVE_CHANNEL,
                        save_operation,
                        True,
                    )
                }
            ),
        ]
    )
    host._update_from_kv_xfer_finished(terminal)

    assert boundary_hash in sched._committed_sidecar_hashes
    assert "716" not in sched._sidecar_save_inflight


def test_tp_sidecar_failure_prevents_commit_after_all_workers_terminal():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=751,
        num_prompt_tokens=16_384,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["751"] = [seq, 8192]
    meta = sched.build_connector_meta()
    save_operation = meta.requests[0].save_operation
    boundary_hash = meta.requests[0].slot_save_spec.boundary_block_hash
    host = Scheduler.__new__(Scheduler)
    host.kv_connector = sched
    host.deferred_free_blocks = {}
    aggregator = KVOutputAggregator(world_size=2)

    partial = aggregator.aggregate(
        [
            KVConnectorOutput(
                connector_completions={
                    ConnectorCompletion(
                        DSV4_CHECKPOINT_SAVE_CHANNEL,
                        save_operation,
                        True,
                    )
                }
            ),
            KVConnectorOutput(),
        ]
    )
    host._update_from_kv_xfer_finished(partial)
    assert boundary_hash not in sched._committed_sidecar_hashes

    terminal = aggregator.aggregate(
        [
            KVConnectorOutput(),
            KVConnectorOutput(
                connector_completions={
                    ConnectorCompletion(
                        DSV4_CHECKPOINT_SAVE_CHANNEL,
                        save_operation,
                        False,
                    )
                }
            ),
        ]
    )
    host._update_from_kv_xfer_finished(terminal)

    assert boundary_hash not in sched._committed_sidecar_hashes
    assert "751" not in sched._sidecar_save_inflight
    assert sched._failed_sidecar_saves["751"] == {(8192, boundary_hash)}


def test_request_finish_retains_tracker_for_pending_and_inflight_sidecar():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=717,
        num_prompt_tokens=8192,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["717"] = [seq, 8192]

    sched.request_finished(seq)
    assert "717" in sched._save_tracker
    assert sched.should_defer_free(seq) is True

    meta = sched.build_connector_meta()
    save_operation = meta.requests[0].save_operation
    boundary_hash = meta.requests[0].slot_save_spec.boundary_block_hash
    assert "717" not in sched._save_inflight
    sched.request_finished(seq)
    assert "717" in sched._save_tracker
    assert sched.should_defer_free(seq) is True

    sched.sidecar_save_finished(save_operation)
    assert boundary_hash in sched._committed_sidecar_hashes
    assert sched.should_defer_free(seq) is False

    sched.save_finished(save_operation)
    sched.request_finished(seq)
    assert sched.should_defer_free(seq) is False
    assert "717" not in sched._save_tracker


def test_request_finish_retains_tracker_for_pending_page_save():
    sched = _scheduler()
    seq = SimpleNamespace(
        id=718,
        token_ids=list(range(8)),
        block_table=[3, 4],
        num_prompt_tokens=8,
        num_cached_tokens=8,
        has_per_req_cache=False,
    )
    sched._save_tracker["718"] = [seq, 0]

    sched.request_finished(seq)
    assert "718" in sched._save_tracker
    assert sched.should_defer_free(seq) is True

    meta = sched.build_connector_meta()
    assert meta.requests[0].save_spec is not None
    save_operation = meta.requests[0].save_operation
    sched.request_finished(seq)
    assert "718" in sched._save_tracker

    sched.save_finished(save_operation)
    sched.request_finished(seq)
    assert sched.should_defer_free(seq) is False
    assert "718" not in sched._save_tracker


def test_prefill_chunk_is_cut_at_nearest_pending_sidecar_boundary():
    sched = _stateful_scheduler(hit=0)
    sched.sidecar_interval = 16_384
    seq = _stateful_seq(
        req_id=711,
        num_prompt_tokens=24_576,
        num_cached_tokens=12_288,
        group=2,
    )

    assert sched.adjust_prefill_chunk_after_alloc(seq, 8192) == 4096


def test_prefill_chunk_does_not_recut_committed_or_inflight_boundary():
    sched = _stateful_scheduler(hit=0)
    sched.sidecar_interval = 16_384
    seq = _stateful_seq(
        req_id=712,
        num_prompt_tokens=24_576,
        num_cached_tokens=12_288,
        group=2,
    )
    boundary_hash = _chained_prefix_hashes(seq.token_ids, 256)[16_384]

    sched._committed_sidecar_hashes.add(boundary_hash)
    assert sched.adjust_prefill_chunk_after_alloc(seq, 8192) == 8192

    sched._committed_sidecar_hashes.clear()
    sched._sidecar_save_inflight["712"] = (
        SaveOperationId(seq.id, 0),
        16_384,
        boundary_hash,
    )
    assert sched.adjust_prefill_chunk_after_alloc(seq, 8192) == 8192


def test_partial_prefill_resumes_and_captures_next_boundary_after_inflight_commit():
    sched = _stateful_scheduler(hit=0)
    seq = _stateful_seq(
        req_id=731,
        num_prompt_tokens=24_576,
        num_cached_tokens=8192,
        group=2,
    )
    sched._save_tracker["731"] = [seq, 8192]

    first = sched.build_connector_meta().requests[0]
    assert first.slot_save_spec.boundary_tokens == 8192

    # The live SLOT was copied to connector-owned staging before the next
    # forward, so B1 publication does not pause progress. A second sidecar is
    # still suppressed until the exact B1 generation retires.
    seq.num_cached_tokens = 16_384
    assert sched._sidecar_save_candidate(seq, 16_384) is None

    sched.sidecar_save_finished(first.save_operation)

    second = sched.build_connector_meta().requests[0]
    assert second.slot_save_spec.boundary_tokens == 16_384


def test_sidecar_chunk_cut_preserves_earlier_load_handoff_cap():
    sched = _stateful_scheduler(hit=0)
    sched.sidecar_interval = 16_384
    seq = _stateful_seq(
        req_id=713,
        num_prompt_tokens=32_768,
        num_cached_tokens=12_288,
        group=2,
    )
    seq.offload_handoff_boundary_tokens = 20_000
    sched._handoff_loads.add("713")

    assert sched.adjust_prefill_chunk_after_alloc(seq, 16_000) == 4096


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


def test_lookup_miss_is_forwarded_for_worker_unpin():
    sched = _scheduler()
    seq = SimpleNamespace(
        id=124,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        num_cached_tokens=0,
    )

    need, should_park = sched.get_num_new_matched_tokens(seq)
    meta = sched.build_connector_meta()

    assert need == 0
    assert should_park is False
    assert meta.lookup_requests_in_step == ["124"]


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
    assert meta.lookup_requests_in_step == ["321"]
    assert seq.offload_loaded_tokens == 8
    assert sched._save_tracker[str(seq.id)][1] == 8
    assert lookup.cleared == ["321"]
    assert str(seq.id) not in sched._load_specs
    assert str(seq.id) not in sched._reqs_need_recv


def test_lookup_time_hbm_satisfies_can_resave_locally_computed_prefix():
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

    assert len(meta1.requests) == 1
    assert meta1.requests[0].token_ids == list(range(8))
    assert meta1.requests[0].save_spec.skip_leading_tokens == 0
    assert meta1.lookup_requests_in_step == ["322"]
    assert sched._save_tracker[str(seq.id)][1] == 8
    assert lookup.cleared == ["322"]

    sched.save_finished(meta1.requests[0].save_operation)
    seq.num_cached_tokens = 12
    meta2 = sched.build_connector_meta()
    save_reqs = [req for req in meta2.requests if req.save_spec is not None]

    assert len(save_reqs) == 1
    assert save_reqs[0].token_ids == list(range(12))
    assert save_reqs[0].save_spec.skip_leading_tokens == 8


def test_unaligned_hbm_handoff_prefills_boundary_then_emits_load():
    sched = _scheduler()
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

    handoff_meta = sched.build_connector_meta()
    assert handoff_meta.lookup_requests_in_step == []
    assert sched._lookup_in_step == ["657"]

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
    assert meta.lookup_requests_in_step == ["657"]
    assert seq.offload_loaded_tokens == 16
    assert str(seq.id) not in sched._handoff_loads
    assert lookup.cleared == []


def test_unaligned_handoff_skips_if_boundary_remainder_is_too_small():
    sched = _scheduler()
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
    assert sched.build_connector_meta().lookup_requests_in_step == ["658"]


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
    assert meta.lookup_requests_in_step == ["655"]
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
    assert meta.lookup_requests_in_step == ["656"]
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
    load_meta = sched.build_connector_meta()
    load_operation = load_meta.requests[0].load_operation
    assert sched._save_tracker[str(seq.id)][1] == 12

    sched.load_failed(load_operation)
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
    conn._engine.lookup_unpin = lambda lookup_id: conn._engine.unpinned.append(
        lookup_id
    )

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


def test_worker_load_terminal_paths_report_exact_operation_once():
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._failed_load = set()
    conn._done_save = set()
    conn._done_sidecar_save = set()
    conn._failed_sidecar_save = set()
    conn._pending_save_ops = {}
    conn._deferred_checkpoint_saves = []
    conn._checkpoint_staging_fences = {}
    conn._done_checkpoint_staging = set()
    conn._aborted_checkpoint_staging = set()
    conn._engine = SimpleNamespace(lookup_unpin=lambda _lookup_id: None)
    operation = LoadOperationId(322, 9)
    req = LMCacheReqMeta(
        req_id=322,
        token_ids=list(range(8)),
        block_ids=[1, 2],
        load_spec=SimpleNamespace(hbm_cached_tokens=8, lmcache_cached_tokens=8),
        load_operation=operation,
    )

    conn._do_load_req(req)
    conn._finish_rejected_load(req, None)

    output = conn.get_finished()
    assert output.finished_loading == set()
    assert output.failed_loading == {operation}
    assert conn.get_finished().is_empty()


def test_worker_unpins_only_lookups_without_an_emitted_load():
    class _Executor:
        def __init__(self) -> None:
            self.calls = []

        def submit(self, *args) -> None:
            self.calls.append(args)

    class _Engine:
        def __init__(self) -> None:
            self.unpinned = []

        def lookup_unpin(self, lookup_id) -> None:
            self.unpinned.append(lookup_id)

    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._do_load = True
    conn._do_save = False
    conn._engine = _Engine()
    conn._load_executor = _Executor()
    metadata = LMCacheOffloadMetadata()
    metadata.lookup_requests_in_step = ["skipped", "loading"]
    metadata.add_request(
        LMCacheReqMeta(
            req_id="loading",
            token_ids=list(range(8)),
            block_ids=[1, 2],
            load_spec=SimpleNamespace(
                hbm_cached_tokens=4,
                lmcache_cached_tokens=8,
            ),
        )
    )

    conn.start_load_kv(metadata)

    assert conn._engine.unpinned == ["skipped"]
    assert len(conn._load_executor.calls) == 1


def test_worker_reports_unaligned_hbm_load_as_failed_without_exception():
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._failed_load = set()
    conn._done_save = set()
    conn.chunk_size = 4
    conn._engine = SimpleNamespace(unpinned=[])
    conn._engine.lookup_unpin = lambda lookup_id: conn._engine.unpinned.append(
        lookup_id
    )

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
    conn._pending_save_ops = {}
    conn._pending_legacy_save_ops = {}
    conn._save_req_locks = {}
    conn.chunk_size = 4
    conn._engine = _Engine()

    req = SimpleNamespace(
        req_id=987,
        token_ids=list(range(12)),
        block_ids=[3, 4, 5],
        is_last_prefill=True,
        save_spec=SimpleNamespace(skip_leading_tokens=4),
    )

    req_lock = conn._begin_save_operation(req.req_id)
    conn._run_save_req(req, None, None, req_lock)

    assert conn._done_save == {987}
    assert len(conn._engine.calls) == 1
    tokens, mask, kwargs = conn._engine.calls[0]
    assert tokens == list(range(12))
    assert mask == [False, False, False, False] + [True] * 8
    assert kwargs["block_ids"] == [3, 4, 5]
    assert kwargs["req_id"] == "987"


def test_worker_save_waits_for_forward_event_before_store():
    import torch

    if not hasattr(torch, "tensor"):
        pytest.skip("real torch is unavailable")

    order = []

    class _Event:
        def synchronize(self) -> None:
            order.append("forward-ready")

    class _Engine:
        def store(self, *args, **kwargs) -> None:
            order.append("store")

    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._lock = threading.Lock()
    conn._done_save = set()
    conn._pending_save_ops = {}
    conn._pending_legacy_save_ops = {}
    conn._save_req_locks = {}
    conn.chunk_size = 4
    conn._engine = _Engine()

    req = SimpleNamespace(
        req_id=988,
        token_ids=list(range(8)),
        block_ids=[3, 4],
        is_last_prefill=True,
        save_spec=SimpleNamespace(skip_leading_tokens=0),
    )

    req_lock = conn._begin_save_operation(req.req_id)
    conn._run_save_req(req, _Event(), None, req_lock)

    assert order == ["forward-ready", "store"]
    assert conn._done_save == {988}


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

        def lookup_unpin(self, lookup_id) -> None:
            self.unpinned.append(lookup_id)

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

        def lookup_unpin(self, lookup_id) -> None:
            self.unpinned.append(lookup_id)

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

    sched.save_finished(meta.requests[0].save_operation)

    assert sched.should_defer_free(seq) is False


def test_pending_work_tracks_undispatched_loads_and_unreported_saves():
    # The engine keeps its idle drain alive on this, so it has to stay true
    # from the moment a load is queued until the matching save reports back.
    sched = _scheduler()
    assert sched.has_pending_work() is False

    sched._reqs_need_recv["9"] = object()
    assert sched.has_pending_work() is True

    sched._reqs_need_recv.clear()
    sched._save_inflight["9"] = {9}
    assert sched.has_pending_work() is True

    sched.save_finished(9)
    assert sched.has_pending_work() is False


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

    sched.save_finished(meta1.requests[0].save_operation)
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

    class _Connector(OffloadSchedulerMixin):
        is_producer = False
        is_offload = True

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
    # kv_events disabled: skip the remote-store recording path so this test
    # only exercises string/int req_id matching in _pop_req_id.
    sched.block_manager = SimpleNamespace(kv_events_enabled=False)

    assert sched._update_waiting_for_remote_kv(SimpleNamespace(id=123)) is True
    assert sched.finished_recving_kv_req_ids == []


# ── MLA (DeepSeek R1/V3, Kimi) offload support ──────────────────────────────
#
# MLA stores a single per-layer latent cache viewed token-major as
# ``(num_blocks * block_size, 1, latent)`` with no separate V/scale tensors,
# so a segment's dim 0 is the *token* count, not the block count. The codec
# must therefore take num_blocks explicitly and derive per-block byte strides
# from it (segment_bytes / num_blocks) rather than assuming dim 0 == blocks.


def _install_byte_addressing_fused(codec: DenseKVByteCodec) -> None:
    """Mock fused staging that addresses each physical block as a raw byte
    slice — block ``b`` maps to bytes ``[b*nbytes : (b+1)*nbytes]`` of the
    flattened segment, exactly like the Triton kernel. Unlike the block-major
    ``_install_fake_fused_chunk_major`` (which index_selects on dim 0), this is
    correct for MLA's token-major single-tensor layout."""

    def _pack(
        segments, seg_block_bytes, chunk_block_counts, flat_block_ids, device_buf
    ) -> None:
        offset = 0
        cursor = 0
        for count in chunk_block_counts:
            ids = flat_block_ids[cursor : cursor + count]
            cursor += count
            for seg, nbytes in zip(segments, seg_block_bytes):
                flat = seg.view(torch.uint8).reshape(-1)
                for b in ids:
                    device_buf[offset : offset + nbytes].copy_(
                        flat[b * nbytes : (b + 1) * nbytes]
                    )
                    offset += nbytes

    def _unpack(
        device_buf, segments, seg_block_bytes, chunk_block_counts, flat_block_ids
    ) -> None:
        offset = 0
        cursor = 0
        for count in chunk_block_counts:
            ids = flat_block_ids[cursor : cursor + count]
            cursor += count
            for seg, nbytes in zip(segments, seg_block_bytes):
                flat = seg.view(torch.uint8).reshape(-1)
                for b in ids:
                    flat[b * nbytes : (b + 1) * nbytes].copy_(
                        device_buf[offset : offset + nbytes]
                    )
                    offset += nbytes

    codec._fused_kv_staging = SimpleNamespace(
        fused_pack_chunk_major=_pack,
        fused_unpack_chunk_major=_unpack,
    )


def test_codec_mla_token_major_block_accounting():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    num_blocks, block_size, latent = 4, 2, 3
    # MLA: single latent k_cache, token-major (num_blocks*block_size, 1, latent),
    # no V / scale tensors.
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=torch.arange(
                num_blocks * block_size * latent, dtype=torch.uint8
            ).reshape(num_blocks * block_size, 1, latent),
            v_cache=None,
            k_scale=None,
            v_scale=None,
        )
    }
    codec = DenseKVByteCodec(kv_caches, num_blocks=num_blocks)

    # Block count comes from the explicit arg, not tensor.shape[0] (= tokens).
    assert codec.num_blocks == num_blocks
    # One scheduler block spans block_size tokens of `latent` bytes each.
    assert codec.bytes_per_block == block_size * latent

    # A segment whose element count is not divisible by num_blocks is rejected.
    with pytest.raises(ValueError):
        DenseKVByteCodec(
            {
                "l0": SimpleNamespace(
                    k_cache=torch.arange(7, dtype=torch.uint8),
                    v_cache=None,
                    k_scale=None,
                    v_scale=None,
                )
            },
            num_blocks=num_blocks,
        )


def test_codec_mla_round_trip_byte_identical():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    num_blocks, block_size, latent = 4, 2, 3
    n = num_blocks * block_size * latent
    original = torch.arange(n, dtype=torch.uint8).reshape(
        num_blocks * block_size, 1, latent
    )
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=original.clone(), v_cache=None, k_scale=None, v_scale=None
        )
    }
    codec = DenseKVByteCodec(kv_caches, num_blocks=num_blocks)
    _install_byte_addressing_fused(codec)

    block_id_groups = [[0, 1], [2, 3]]
    device_buf = torch.empty(
        num_blocks * codec.bytes_per_block, dtype=torch.uint8, device=codec.device
    )

    # Gather: each physical block is block_size*latent contiguous bytes.
    codec.gpu_to_chunk_major_device_buffer(device_buf, block_id_groups)
    flat = original.view(torch.uint8).reshape(num_blocks, -1)
    expected = torch.cat([flat[0], flat[1], flat[2], flat[3]])
    assert torch.equal(device_buf.cpu(), expected.cpu())

    # Scatter back into a zeroed cache reproduces the original byte-for-byte.
    kv_caches["l0"].k_cache.zero_()
    codec.chunk_major_device_buffer_to_gpu(device_buf, block_id_groups)
    assert torch.equal(kv_caches["l0"].k_cache, original)


def test_codec_dsa_includes_index_cache_segment():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    num_blocks, block_size, latent, index_dim = 4, 2, 3, 5
    k_cache = torch.arange(num_blocks * block_size * latent, dtype=torch.uint8).reshape(
        num_blocks * block_size, 1, latent
    )
    # Block-major indexer cache (num_blocks, block_size, index_dim).
    index_cache = torch.arange(
        num_blocks * block_size * index_dim, dtype=torch.uint8
    ).reshape(num_blocks, block_size, index_dim)
    kv_caches = {
        "l0": SimpleNamespace(
            k_cache=k_cache.clone(),
            v_cache=None,
            k_scale=None,
            v_scale=None,
            index_cache=index_cache.clone(),
        )
    }
    codec = DenseKVByteCodec(kv_caches, num_blocks=num_blocks)
    mla_only = block_size * latent
    index_only = block_size * index_dim
    assert codec.bytes_per_block == mla_only + index_only

    _install_byte_addressing_fused(codec)
    # Stage every block (two chunks) so the round trip below can assert the full
    # tensor is restored.
    block_id_groups = [[0, 1], [2, 3]]
    device_buf = torch.empty(
        num_blocks * codec.bytes_per_block, dtype=torch.uint8, device=codec.device
    )
    codec.gpu_to_chunk_major_device_buffer(device_buf, block_id_groups)

    k_flat = k_cache.view(torch.uint8).reshape(num_blocks, -1)
    idx_flat = index_cache.reshape(num_blocks, -1)
    # Staging is segment-major within a chunk (see DenseKVByteCodec docstring and
    # the Triton kernel's ``segment_prefix_bytes[seg] * nblocks`` base): within
    # each chunk it is all K blocks, then all index blocks.
    expected = torch.cat(
        [
            k_flat[0],
            k_flat[1],
            idx_flat[0],
            idx_flat[1],
            k_flat[2],
            k_flat[3],
            idx_flat[2],
            idx_flat[3],
        ],
    )
    assert torch.equal(device_buf.cpu(), expected.cpu())

    kv_caches["l0"].k_cache.zero_()
    kv_caches["l0"].index_cache.zero_()
    codec.chunk_major_device_buffer_to_gpu(device_buf, block_id_groups)
    assert torch.equal(kv_caches["l0"].k_cache, k_cache)
    assert torch.equal(kv_caches["l0"].index_cache, index_cache)


def test_codec_dsa_fp8_multilayer_including_mtp_round_trip():
    """GLM-5.2 realistic geometry: an ``fp8`` indexer cache
    (``aligned_index_dim=144``) alongside the token-major MLA latent (576),
    across main *and* MTP layers.

    For GLM-5.2 the MTP draft is MLA, so it shares the target's KV pool and is
    bound by the main attention builder exactly like a decoder layer (no
    ``eagle3_draft_builder``); its ``index_cache`` therefore reaches the codec
    as just another registered layer. This asserts the codec moves the fp8
    index segment byte-exact for every layer. Bytes are compared through a
    ``uint8`` view so fp8 NaN bit patterns (which are ``!=`` themselves) do not
    make a byte-identical round trip look unequal.
    """
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        pytest.skip("fp8 dtype unavailable")

    num_blocks, block_size = 4, 2
    latent, aligned_index_dim = 576, 144  # DeepSeek-V3.2 / GLM-5.2 real dims

    def _make_layer(seed: int):
        # MLA latent: token-major (num_blocks*block_size, 1, latent).
        k = (
            torch.arange(num_blocks * block_size * latent, dtype=torch.uint8) + seed
        ).reshape(num_blocks * block_size, 1, latent)
        # Indexer: block-major (num_blocks, block_size, aligned_index_dim), fp8.
        idx = (
            (
                torch.arange(
                    num_blocks * block_size * aligned_index_dim, dtype=torch.uint8
                )
                + seed * 7
            )
            .view(fp8)
            .reshape(num_blocks, block_size, aligned_index_dim)
        )
        return k, idx

    # layer_0/layer_1 are decoder layers; layer_2 stands in for the MTP layer,
    # which shares the pool and is registered identically.
    layers = {f"layer_{i}": _make_layer(i) for i in range(3)}
    kv_caches = {
        name: SimpleNamespace(
            k_cache=k.clone(),
            v_cache=None,
            k_scale=None,
            v_scale=None,
            index_cache=idx.clone(),
        )
        for name, (k, idx) in layers.items()
    }
    codec = DenseKVByteCodec(kv_caches, num_blocks=num_blocks)

    per_block = block_size * latent + block_size * aligned_index_dim
    assert codec.bytes_per_block == len(layers) * per_block

    _install_byte_addressing_fused(codec)
    block_id_groups = [[0, 1], [2, 3]]
    device_buf = torch.empty(
        num_blocks * codec.bytes_per_block, dtype=torch.uint8, device=codec.device
    )
    codec.gpu_to_chunk_major_device_buffer(device_buf, block_id_groups)

    # Wipe every segment (via the uint8 view for fp8) and scatter back.
    for cache in kv_caches.values():
        cache.k_cache.zero_()
        cache.index_cache.view(torch.uint8).zero_()
    codec.chunk_major_device_buffer_to_gpu(device_buf, block_id_groups)

    for name, (k, idx) in layers.items():
        assert torch.equal(kv_caches[name].k_cache, k)
        assert torch.equal(
            kv_caches[name].index_cache.view(torch.uint8),
            idx.view(torch.uint8),
        )


def test_scheduler_offload_statistics_are_cumulative():
    sched = _scheduler()
    sched._load_inflight_tokens["1"] = 8192
    sched._save_inflight_tokens["2"] = 4096

    sched.load_finished("1")
    sched.save_finished("2")

    assert sched.get_statistics() == {
        "load_requests": 1,
        "loaded_tokens": 8192,
        "load_failures": 0,
        "save_requests": 1,
        "saved_tokens": 4096,
        "loads_pending": 0,
        "saves_pending": 0,
    }


def test_sidecar_only_save_statistics_use_exact_generation_once():
    sched = _scheduler()
    operation = SaveOperationId(2, 4)
    stale = SaveOperationId(2, 3)
    sched._track_save_statistics(operation, 0)
    sched._sidecar_save_inflight["2"] = (operation, 8192, 0x1234)

    sched.save_finished(stale)
    assert sched.get_statistics()["saves_pending"] == 1
    assert sched.get_statistics()["save_requests"] == 0

    sched.save_finished(operation)
    sched.save_finished(operation)

    assert sched.get_statistics()["saves_pending"] == 0
    assert sched.get_statistics()["save_requests"] == 1
    assert sched.get_statistics()["saved_tokens"] == 0


def test_request_cleanup_drops_abandoned_load_statistics():
    sched = _scheduler()
    seq = SimpleNamespace(id=3)
    operation = LoadOperationId(seq.id, 7)
    sched._load_lifecycles["3"] = seq
    sched._active_load_operations["3"] = (seq, operation)
    sched._track_load_statistics(operation, 4096)

    sched.request_finished(seq)

    assert sched.get_statistics()["loads_pending"] == 0
    assert sched.get_statistics()["load_requests"] == 0
    assert sched.get_statistics()["load_failures"] == 0


# ---------------------------------------------------------------------------
# CPU budget split across PP stages
# ---------------------------------------------------------------------------


def _pp_config(pp_rank: int, pp_size: int, num_hidden: int, draft_layers: int | None):
    spec = None
    if draft_layers is not None:
        spec = SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(num_nextn_predict_layers=draft_layers)
        )
    return SimpleNamespace(
        pipeline_parallel_size=pp_size,
        hf_config=SimpleNamespace(num_hidden_layers=num_hidden),
        parallel_config=SimpleNamespace(pipeline_parallel_rank=pp_rank),
        speculative_config=spec,
    )


def _budgets(pp_size, num_hidden, draft_layers, configured=256.0, partition=None):
    from atom.kv_transfer.offload import config as offcfg

    out = []
    for rank in range(pp_size):
        cfg = SimpleNamespace(max_local_cpu_size=configured)
        offcfg.scale_cpu_size_for_pp(
            cfg, _pp_config(rank, pp_size, num_hidden, draft_layers)
        )
        out.append(cfg.max_local_cpu_size)
    return out


def test_cpu_budget_split_preserves_total_and_equalizes_horizon(monkeypatch):
    # Even split, no spec: every stage holds the same layers, so equal budgets.
    budgets = _budgets(pp_size=4, num_hidden=80, draft_layers=None)
    assert budgets == pytest.approx([256.0] * 4)
    assert sum(budgets) == pytest.approx(1024.0)


def test_cpu_budget_split_counts_the_draft_layer_on_the_last_stage(monkeypatch):
    # Last stage binds the draft KV layer, so its budget must cover 19
    # layers, not 18.
    import atom.models.utils as model_utils
    from atom.kv_transfer.offload import config as offcfg

    partition = [20, 20, 20, 18]
    starts = [0, 20, 40, 60]

    def fake_pp_indices(num_layers, rank, size):
        return (starts[rank], starts[rank] + partition[rank])

    monkeypatch.setattr(model_utils, "get_pp_indices", fake_pp_indices)
    monkeypatch.setattr(offcfg, "get_pp_indices", fake_pp_indices, raising=False)

    budgets = _budgets(pp_size=4, num_hidden=78, draft_layers=1)
    local = [20, 20, 20, 19]
    horizons = [b / n for b, n in zip(budgets, local)]

    assert sum(budgets) == pytest.approx(1024.0)
    assert horizons == pytest.approx([horizons[0]] * 4)
    assert budgets[3] < budgets[0]


# ---------------------------------------------------------------------------
# LMCache engine identity across DP replicas
# ---------------------------------------------------------------------------


def _dp_config(dp_rank: int, *, pp_size: int = 1, tp_size: int = 1, pp_rank: int = 0):
    return SimpleNamespace(
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size,
        parallel_config=SimpleNamespace(
            data_parallel_rank=dp_rank,
            pipeline_parallel_rank=pp_rank,
        ),
    )


def test_engine_id_is_distinct_per_dp_replica():
    # --enable-dp-attention folds TP into DP, so every GPU becomes a replica
    # owning a private CPU/NVMe pool. One shared id makes all replicas bind the
    # same lookup socket, leaving every scheduler reading replica 0's hits.
    ids = [offcfg.lmcache_engine_id(_dp_config(rank)) for rank in range(8)]
    assert len(set(ids)) == 8


def test_engine_id_pairs_scheduler_and_workers_of_one_replica():
    # The scheduler's LookupClient and its workers' LookupServer derive the
    # same ipc path from this id, so it must not vary with PP or TP position.
    ids = [
        offcfg.lmcache_engine_id(_dp_config(3, pp_size=2, tp_size=4, pp_rank=pp))
        for pp in range(2)
    ]
    assert ids == ["atom-offload-dp3", "atom-offload-dp3"]


def test_engine_id_defaults_to_replica_zero():
    assert offcfg.lmcache_engine_id(_dp_config(0)) == "atom-offload-dp0"
    assert offcfg.lmcache_engine_id(SimpleNamespace()) == "atom-offload-dp0"


def test_replica_world_size_counts_pp_and_tp_but_not_dp():
    # Worker ids index the replica-local PP x TP grid. Folding DP in would push
    # every replica but the first past lookup_server_worker_ids=[0], leaving
    # them with no lookup server at all.
    assert offcfg.lmcache_replica_world_size(_dp_config(5, pp_size=4, tp_size=8)) == 32
    assert offcfg.lmcache_replica_world_size(_dp_config(5)) == 1
    assert offcfg.lmcache_replica_world_size(SimpleNamespace()) == 1


class _RecordingRunnerMgr:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def call_func(self, name, *args, **kwargs):
        self.calls.append((name, args[0] if args else None))


class _RecordingPPTransport:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def send_metadata(self, batch):
        self.sent.append(batch)


def _idle_meta_with_only_unpins() -> LMCacheOffloadMetadata:
    meta = LMCacheOffloadMetadata()
    meta.lookup_requests_in_step = ["7"]
    return meta


def _idle_engine(meta: LMCacheOffloadMetadata):
    connector = SimpleNamespace(
        is_offload=True,
        build_connector_meta=lambda: meta,
    )
    return SimpleNamespace(
        kv_transfer_enabled=True,
        scheduler=SimpleNamespace(kv_connector=connector),
        runner_mgr=_RecordingRunnerMgr(),
        pp_transport=_RecordingPPTransport(),
    )


def test_idle_offload_dispatch_delivers_unpin_only_metadata():
    # build_connector_meta drains the pending lookup ids as a side effect, so a
    # metadata carrying nothing but unpins is the only chance those ids get
    # released. Dropping it pins their CPU chunks until LMCache's watchdog
    # times them out 300s later.
    from atom.model_engine.engine_core import EngineCore

    engine = _idle_engine(_idle_meta_with_only_unpins())
    EngineCore._dispatch_idle_offload_work(engine)

    assert [name for name, _ in engine.runner_mgr.calls] == [
        "process_kvconnector_output"
    ]
    assert engine.runner_mgr.calls[0][1].lookup_requests_in_step == ["7"]


def test_pp_connector_only_dispatch_delivers_unpin_only_metadata():
    from atom.model_engine.pp_engine_core import PPEngineCoreProc

    engine = _idle_engine(LMCacheOffloadMetadata())
    batch = SimpleNamespace(connector_meta_output=_idle_meta_with_only_unpins())
    PPEngineCoreProc._dispatch_connector_only_batch(engine, batch)

    assert [name for name, _ in engine.runner_mgr.calls] == [
        "process_kvconnector_output"
    ]
    # Every stage pinned on lookup, so every stage has to hear the unpin.
    assert engine.pp_transport.sent == [batch]


def test_dispatch_skips_metadata_with_no_work_at_all():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc

    engine = _idle_engine(LMCacheOffloadMetadata())
    EngineCore._dispatch_idle_offload_work(engine)
    PPEngineCoreProc._dispatch_connector_only_batch(
        engine, SimpleNamespace(connector_meta_output=LMCacheOffloadMetadata())
    )

    assert engine.runner_mgr.calls == []
    assert engine.pp_transport.sent == []


def test_multi_metadata_exposes_sub_meta_unpins():
    # The prefill node runs offload inside a `multi` connector, so the idle
    # dispatch only ever sees the wrapper. Without this aggregate it reads as
    # empty and the sub-meta's unpins are dropped.
    from atom.kv_transfer.disaggregation.multi.multi_connector import (
        MultiConnectorMetadata,
    )

    wrapper = MultiConnectorMetadata(
        metas=[SimpleNamespace(), _idle_meta_with_only_unpins()]
    )
    assert wrapper.requests == []
    assert wrapper.lookup_requests_in_step == ["7"]

    from atom.kv_transfer.disaggregation.types import connector_metadata_has_work

    assert connector_metadata_has_work(wrapper)
    assert not connector_metadata_has_work(
        MultiConnectorMetadata(metas=[LMCacheOffloadMetadata()])
    )
