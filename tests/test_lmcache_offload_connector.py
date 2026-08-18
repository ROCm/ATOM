# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
import sys
import threading
import types
from types import SimpleNamespace

import pytest

try:
    import torch
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

from atom.kv_transfer.disaggregation import KVConnectorOutput, KVOutputAggregator
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.atom_kv_byte_codec import ATOMKVByteCodec
from atom.kv_transfer.offload.atom_lmcache_gpu_connector import (
    ATOMLMCacheGPUConnector,
)
from atom.kv_transfer.offload.connector import (
    LMCacheOffloadConnector,
    LMCacheOffloadConnectorScheduler,
)
from atom.kv_transfer.offload.metadata import (
    ATOMRawBytesLMCacheMetadata,
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
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
    # In-flight byte counters this branch's connector keeps and the lmcache
    # branch's fixture predates, plus the cumulative totals it reports.
    sched._load_inflight_tokens = {}
    sched._save_inflight_tokens = {}
    sched.total_load_requests = 0
    sched.total_loaded_tokens = 0
    sched.total_load_failures = 0
    sched.total_save_requests = 0
    sched.total_saved_tokens = 0
    # Paged-KV saves for per-request-cache sequences: off in production because
    # `_decide_load_after_alloc` refuses their load leg, kept on here so the
    # save-path tests keep exercising the emission they are about.
    sched._save_per_req_cache = True
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
    sched._pending_state_loads = []
    sched._min_load_tokens = 0
    sched._lock = threading.Lock()
    sched._done_load = set()
    return sched


def _install_fake_fused_chunk_major(codec: ATOMKVByteCodec) -> None:
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
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=8)

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


def test_lmcache_connector_fused_chunk_fastpath_uses_chunk_major(monkeypatch):
    from contextlib import nullcontext

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
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=8)
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
            ATOMKVByteCodec.gpu_to_chunk_major_device_buffer(
                codec, device_buf, block_id_groups, stream=None
            ),
        )[-1],
    )
    monkeypatch.setattr(
        codec,
        "chunk_major_device_buffer_to_gpu",
        lambda device_buf, block_id_groups, stream=None: (
            unpack_groups.append([list(group) for group in block_id_groups]),
            ATOMKVByteCodec.chunk_major_device_buffer_to_gpu(
                codec, device_buf, block_id_groups, stream=None
            ),
        )[-1],
    )
    # Wrap `StagedTransfer.ensure_buffer`, not a connector-side delegate:
    # `run_pipeline` calls it on itself, so a patch on the connector is never
    # reached and `buffer_requests` stays empty -- which is what made the two
    # bound assertions below vacuous (`all()` over `[]` is True).
    orig_ensure_buffer = connector._staged.ensure_buffer

    def _ensure_buffer(staging_buffer, nbytes):
        device_buf = orig_ensure_buffer(staging_buffer, nbytes)
        buffer_requests.append((nbytes, int(staging_buffer.tensor.numel())))
        return device_buf

    monkeypatch.setattr(connector._staged, "ensure_buffer", _ensure_buffer)

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
    # Both bounds below are `all()` over this list, so an empty list passes
    # them for free. Pin it non-empty first.
    assert buffer_requests
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


def test_lmcache_connector_staged_pipeline_really_reaches_staged_transfer(monkeypatch):
    """`_run_staged_pipeline` must actually delegate to `StagedTransfer`.

    The fastpath test above monkeypatches `_thread_state` **on the
    connector**, replacing the very method that delegates — so it never
    exercises `StagedTransfer.run_pipeline` at all.
    This test leaves every delegating method intact and instead seeds the
    thread-local state *inside* `StagedTransfer`, so a delegation that is
    removed, inlined, or misrouted fails here. It also pins the intra-worker
    stage hand-off order (record ready -> stage_b waits on it -> record free).
    That is not commit 7427e05e's fence, which lives on the caller side
    (`save_ready_event`); these events order one worker thread's pack_stream
    against its own copy_stream.
    """
    from contextlib import nullcontext

    import torch

    from atom.kv_transfer.offload.staged_transfer import StagedTransfer

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "2")
    monkeypatch.delenv("OFFLOAD_GPU_STAGING_MAX_BYTES", raising=False)
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
    _install_fake_fused_chunk_major(codec)
    monkeypatch.setattr(connector, "_assert_fused_chunk_major_available", lambda: None)

    # The fake streams are not real CUDA streams, so keep them out of the codec.
    monkeypatch.setattr(
        codec,
        "gpu_to_chunk_major_device_buffer",
        lambda device_buf, block_id_groups, stream=None: (
            ATOMKVByteCodec.gpu_to_chunk_major_device_buffer(
                codec, device_buf, block_id_groups, stream=None
            )
        ),
    )

    trace: list[str] = []

    class _FakeEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream) -> None:
            trace.append(f"record:{self.name}")

    class _FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event) -> None:
            trace.append(f"wait:{self.name}:{event.name}")

        def synchronize(self) -> None:
            trace.append(f"sync:{self.name}")

    class _FakeState:
        def __init__(self) -> None:
            self.device = connector.device
            self.pack_stream = _FakeStream("pack")
            self.copy_stream = _FakeStream("copy")
            self.staging_buffer = SimpleNamespace(
                tensor=None,
                ready_event=_FakeEvent("ready"),
                free_event=_FakeEvent("free"),
                free_event_valid=False,
            )

        def stream_ctx(self, stream):
            return nullcontext()

    fake_state = _FakeState()
    # Seed StagedTransfer's own thread-local cache so the connector's
    # `_thread_state()` delegation is exercised for real.
    connector._staged._tls.states = {str(connector.device): fake_state}

    reached: list[tuple[bool, int]] = []
    real_run_pipeline = StagedTransfer.run_pipeline

    def _spy_run_pipeline(self, state, groups, stage_a, stage_b):
        reached.append((self is connector._staged, len(groups)))
        return real_run_pipeline(self, state, groups, stage_a, stage_b)

    monkeypatch.setattr(StagedTransfer, "run_pipeline", _spy_run_pipeline)

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

    # The delegation was genuinely taken, on this connector's StagedTransfer.
    assert reached == [(True, 1)]
    # The producer event is recorded before stage_b waits on it, and the free
    # event only after stage_b has been issued.
    assert trace == [
        "record:ready",
        "wait:copy:ready",
        "record:free",
        "sync:copy",
    ]
    # The real (non-monkeypatched) ensure_buffer allocated the bounded buffer.
    assert int(fake_state.staging_buffer.tensor.numel()) == (
        connector.gpu_staging_buffer_bytes
    )
    assert fake_state.staging_buffer.free_event_valid is True

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
    assert torch.equal(memory_objs[0].tensor, expected0)
    assert torch.equal(memory_objs[1].tensor, expected1)


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
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=8)
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
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=4)

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
    codec = ATOMKVByteCodec(kv_caches)
    connector = ATOMLMCacheGPUConnector(codec, block_size=4, chunk_size=4)

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
    codec = ATOMKVByteCodec(kv_caches)
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
    codec = ATOMKVByteCodec(kv_caches)
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
    codec = ATOMKVByteCodec(kv_caches)
    device_buf = torch.empty(3 * codec.bytes_per_block, dtype=torch.uint8)

    with pytest.raises(ValueError, match="duplicate block ids"):
        codec.gpu_to_chunk_major_device_buffer(device_buf, [[0, 1], [1]])


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
    assert meta1.lookup_requests_in_step == ["322"]
    assert sched._save_tracker[str(seq.id)][1] == 8
    assert lookup.cleared == ["322"]

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
    conn.chunk_size = 4
    conn._engine = _Engine()

    req = SimpleNamespace(
        req_id=988,
        token_ids=list(range(8)),
        block_ids=[3, 4],
        is_last_prefill=True,
        save_spec=SimpleNamespace(skip_leading_tokens=0),
    )

    conn._do_save_req(req, _Event())

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


def _install_byte_addressing_fused(codec: ATOMKVByteCodec) -> None:
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
    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)

    # Block count comes from the explicit arg, not tensor.shape[0] (= tokens).
    assert codec.num_blocks == num_blocks
    # One scheduler block spans block_size tokens of `latent` bytes each.
    assert codec.bytes_per_block == block_size * latent

    # Regression: passing the page-size-1 physical row count instead of the
    # scheduler block count shrinks each transfer block by block_size. The
    # connector compares this against its existing transfer-region metadata.
    wrong_codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks * block_size)
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._codec = wrong_codec
    with pytest.raises(ValueError, match="KV block geometry mismatch"):
        conn._validate_block_geometry(
            SimpleNamespace(
                block_regions=[SimpleNamespace(unit_bytes=block_size * latent)]
            )
        )

    # A segment whose element count is not divisible by num_blocks is rejected.
    with pytest.raises(ValueError):
        ATOMKVByteCodec(
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
    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)
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
    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)
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
    # Staging is segment-major within a chunk (see ATOMKVByteCodec docstring and
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
    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)

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


# ── the state offload tier is built only when it can actually work ────────


class _StateBackend:
    """Publishes one entry's worth of contiguous views, like a real backend."""

    def __init__(self, entry_bytes: int) -> None:
        self._entry_bytes = entry_bytes

    def state_entry_views(self, group: int):
        return [torch.empty(self._entry_bytes, dtype=torch.uint8)]


def _state_tier_conn(pipeline_parallel_size: int = 1):
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._state_tier = None
    conn._engine = SimpleNamespace(storage_manager=object())
    # `__init__` always sets `_config`, and the builder reads
    # `pipeline_parallel_size` off it, so the fake has to carry one too.
    conn._config = SimpleNamespace(pipeline_parallel_size=pipeline_parallel_size)
    return conn


def _gpu_conn(staging_bytes: int, chunk_bytes: int = 512):
    return SimpleNamespace(
        _staged=object(),
        gpu_staging_buffer_bytes=staging_bytes,
        gpu_staging_chunk_bytes=chunk_bytes,
        release_gpu_staging_after_transfer=False,
        device=torch.device("cpu"),
    )


def _state_meta():
    return SimpleNamespace(model_name="m")


def test_an_entry_larger_than_the_kv_buffer_gets_its_own_staging(
    monkeypatch, caplog
):
    """A state entry is one entry; the KV buffer is sized in LMCache chunks, and
    at the shipped default of 2 chunks it is ~8 MiB against a ~55 MiB entry.

    This used to refuse the tier, and that is a silent half-feature: the
    engine-side index still exists, keeps handing out staging slots and
    counting spills that never happen, and the only symptom is one line in a
    100k-line log -- which is exactly how a c8 and a c10 measurement were taken
    against a tier that was never running. Give the tier its own buffer of one
    entry instead, on the thread that packs into it."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    conn = _state_tier_conn()
    tt = SimpleNamespace(state_backend=_StateBackend(4096))

    with caplog.at_level(logging.INFO, logger="atom"):
        conn._maybe_build_state_tier(_gpu_conn(1024), tt, _state_meta(), 0, 1)

    assert conn._state_tier is not None
    assert conn._state_tier.codec._staged.staging_buffer_bytes == 4096
    # And it says how to make them share instead: 4096 / 512 = 8 chunks.
    assert "OFFLOAD_GPU_STAGING_CHUNKS >= 8" in caplog.text


def test_an_entry_that_fits_shares_the_kv_staging_buffer(monkeypatch):
    """The other half: when it fits there is nothing to allocate, and sharing is
    what keeps one bound configured in one place."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    conn = _state_tier_conn()
    gpu = _gpu_conn(4096)
    tt = SimpleNamespace(state_backend=_StateBackend(1024))

    conn._maybe_build_state_tier(gpu, tt, _state_meta(), 0, 1)

    assert conn._state_tier is not None
    assert conn._state_tier.codec._staged is gpu._staged


def test_state_tier_is_built_when_an_entry_fits(monkeypatch):
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    conn = _state_tier_conn()
    tt = SimpleNamespace(state_backend=_StateBackend(1024))

    conn._maybe_build_state_tier(_gpu_conn(4096), tt, _state_meta(), 0, 1)

    assert conn._state_tier is not None
    assert conn._state_tier.codec.entry_bytes == 1024


def test_state_tier_is_disabled_when_the_backend_has_no_state_views(
    monkeypatch, caplog
):
    """A builder that predates `state_entry_views` raises AttributeError, not
    NotImplementedError. Both mean the same thing here and neither is worth
    killing model load over."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    conn = _state_tier_conn()
    tt = SimpleNamespace(state_backend=SimpleNamespace())  # no state_entry_views

    with caplog.at_level(logging.WARNING, logger="atom"):
        conn._maybe_build_state_tier(_gpu_conn(1 << 20), tt, _state_meta(), 0, 1)

    assert conn._state_tier is None
    assert "no per-request state views" in caplog.text


def test_the_state_tier_is_refused_under_pipeline_parallelism(monkeypatch, caplog):
    """PP breaks the tier twice over, and neither failure raises.

    The key is `(model, world_size, worker_id, hash)` with `worker_id =
    tp.rank_in_group` -- no PP component -- so two stages holding different
    layer slices collide on one key and overwrite each other's bytes. And only
    the head stage polls `_poll_kv_transfer_progress`, so the other stages'
    `state_staging_released` reports are never drained and the ring starves.

    Non-vacuousness: drop the `pp_size > 1` guard and this fails, because the
    entry fits the staging buffer and the tier builds.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    conn = _state_tier_conn(pipeline_parallel_size=2)
    tt = SimpleNamespace(state_backend=_StateBackend(1024))

    with caplog.at_level(logging.WARNING, logger="atom"):
        conn._maybe_build_state_tier(_gpu_conn(4096), tt, _state_meta(), 0, 1)

    assert conn._state_tier is None
    assert "pipeline parallelism" in caplog.text


def test_the_state_tier_still_builds_without_pipeline_parallelism(monkeypatch):
    """The guard's control: pp=1 is the overwhelmingly common config, and a
    guard that also refused it would disable the feature outright while every
    refusal test above still passed."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    conn = _state_tier_conn(pipeline_parallel_size=1)
    tt = SimpleNamespace(state_backend=_StateBackend(1024))

    conn._maybe_build_state_tier(_gpu_conn(4096), tt, _state_meta(), 0, 1)

    assert conn._state_tier is not None


def test_a_zero_entry_state_pool_stays_loud(monkeypatch):
    """IndexError is deliberately not caught: group 0 missing with the tier on
    is a sizing bug, and degrading it to a server that silently never spills is
    exactly the failure this round is about."""

    class _Empty:
        def state_entry_views(self, group):
            raise IndexError("no such group")

    monkeypatch.setenv("OFFLOAD_STATE", "1")
    conn = _state_tier_conn()
    tt = SimpleNamespace(state_backend=_Empty())

    with pytest.raises(IndexError):
        conn._maybe_build_state_tier(_gpu_conn(1 << 20), tt, _state_meta(), 0, 1)


# ── a hybrid model's per-request state is not paged KV ────────────────────
#
# `build_kv_cache_tensor` on the two hybrid builders (`gdn_attn.py` for
# Qwen3-Next / Qwen3.5, `kimi_mla_gdn_attn.py` for Kimi-K3) returns the mamba /
# KDA recurrent-state rows as a `KVCacheTensor`, because the forward path reads
# its state out of the same `kv_cache_data` registry. Those rows are addressed
# by request slot, have no block stride, and must never reach the byte codec:
# the codec would either refuse them (`numel() % num_blocks`) or, worse, count
# their bytes into `bytes_per_block` and blow the geometry cross-check.


def _paged_entry(num_blocks, block_size, latent):
    import torch

    return SimpleNamespace(
        k_cache=torch.arange(
            num_blocks * block_size * latent, dtype=torch.uint8
        ).reshape(num_blocks * block_size, 1, latent),
        v_cache=None,
        k_scale=None,
        v_scale=None,
    )


def _state_entry(num_slots, state_elems, *, per_request_state=True):
    """Stand in for `KVCacheTensor(k_cache=runner.mamba_k_cache[i], ...)`."""
    import torch

    return SimpleNamespace(
        k_cache=torch.zeros(num_slots, state_elems, dtype=torch.uint8),
        v_cache=torch.zeros(num_slots, state_elems + 2, dtype=torch.uint8),
        k_scale=None,
        v_scale=None,
        per_request_state=per_request_state,
    )


def test_codec_skips_per_request_state_entries():
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    num_blocks, block_size, latent = 4, 2, 3
    kv_caches = {
        "layer_0": _paged_entry(num_blocks, block_size, latent),
        # 3 slots x 5 elements = 15, which does not divide num_blocks=4. This
        # is the shape that raises today.
        "layer_1": _state_entry(3, 5),
        "layer_2": _paged_entry(num_blocks, block_size, latent),
    }

    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)

    # Exactly the two paged K tensors; the state entry's k_cache and v_cache
    # are both gone.
    assert len(codec._segments) == 2
    assert codec.bytes_per_block == 2 * block_size * latent

    # And the geometry cross-check the connector runs at boot now agrees with
    # what the paged backend describes.
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._codec = codec
    conn._validate_block_geometry(
        SimpleNamespace(
            block_regions=[
                SimpleNamespace(unit_bytes=block_size * latent),
                SimpleNamespace(unit_bytes=block_size * latent),
            ]
        )
    )


def test_codec_state_only_registry_is_refused_loudly():
    """DeepSeek-V4 style: nothing paged reaches the codec at all. Better to
    refuse than to move a cache that is entirely per-request state."""
    import torch

    if not hasattr(torch, "zeros"):
        pytest.skip("real torch is unavailable")

    with pytest.raises(ValueError, match="no movable KV tensors registered"):
        ATOMKVByteCodec({"layer_0": _state_entry(3, 5)}, num_blocks=4)


def test_codec_dense_model_segment_list_is_unchanged():
    """The dense regression: no `per_request_state` marker anywhere, so the
    segment list must be byte-for-byte what it was before the filter existed."""
    import torch

    if not hasattr(torch, "arange"):
        pytest.skip("real torch is unavailable")

    num_blocks = 6
    kv_caches = {
        "layer_0": SimpleNamespace(
            k_cache=torch.arange(num_blocks * 2, dtype=torch.uint8).reshape(
                num_blocks, 2
            ),
            v_cache=torch.arange(num_blocks * 3, dtype=torch.uint8).reshape(
                num_blocks, 3
            ),
            k_scale=torch.zeros(num_blocks, dtype=torch.uint8),
            v_scale=torch.zeros(num_blocks, dtype=torch.uint8),
        ),
        "layer_1": SimpleNamespace(
            k_cache=torch.arange(num_blocks * 2, dtype=torch.uint8).reshape(
                num_blocks, 2
            ),
            v_cache=torch.arange(num_blocks * 3, dtype=torch.uint8).reshape(
                num_blocks, 3
            ),
            k_scale=None,
            v_scale=None,
        ),
    }

    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)

    # 4 segments for layer_0 (K, V, kS, vS) + 2 for layer_1.
    assert len(codec._segments) == 6
    assert codec.bytes_per_block == (2 + 3 + 1 + 1) + (2 + 3)


# ── a hybrid model declines the KV load, and says so once ─────────────────
#
# The state boundary P (`seq.num_cached_tokens` right after
# `BlockManager.allocate`) is the only history the recurrent state covers. An
# LMCache load lands on top with no second state gate and pushes the KV-loaded
# length L past P; the scheduler then forwards only `[L, num_prompt)`, so the
# GDN/KDA layers never see `[P, L)`. At P=0 the group was just recycled and
# `has_initial_state` is True, so the recurrence starts from another request's
# leftovers. No exception, wrong output. Refusing the load is exactly as
# useful as clamping it (a load clamped to P transfers nothing) and honest
# about why.


def _hybrid_seq(seq_id, num_prompt, block_table):
    return SimpleNamespace(
        id=seq_id,
        num_prompt_tokens=num_prompt,
        token_ids=list(range(num_prompt)),
        num_cached_tokens=0,
        block_table=list(block_table),
        has_per_req_cache=True,
    )


def test_per_req_cache_sequence_is_refused_the_offload_load(caplog):
    sched = _scheduler()
    sched._min_load_tokens = 8
    lookup = _LookupClient(hit=12)
    sched._lookup_client = lookup
    seq = _hybrid_seq(770, 16, [1, 2, 3, 4])

    need, should_park = sched.get_num_new_matched_tokens(seq)
    assert need == 12
    assert should_park is True

    # Exactly the configuration that parks a stateless sequence: hbm=4 is
    # chunk-aligned and the 8-token gap meets `min_load`.
    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)

    with caplog.at_level(logging.DEBUG, logger="atom"):
        assert sched.should_park_for_load_after_alloc(seq) is False

    assert "per_req_cache_state_boundary" in caplog.text
    meta = sched.build_connector_meta()
    assert [req for req in meta.requests if req.load_spec is not None] == []
    # The KV-loaded length never moves past the state boundary.
    assert seq.offload_loaded_tokens == 4
    assert str(seq.id) not in sched._load_specs
    assert str(seq.id) not in sched._reqs_need_recv
    # And the request is not left holding a lookup pin.
    assert lookup.cleared == ["770"]


def test_a_stateless_sequence_is_still_admitted():
    """Paired with the refusal above so a change that turns loads off
    wholesale fails here rather than looking like a pass."""
    sched = _scheduler()
    sched._min_load_tokens = 8
    sched._lookup_client = _LookupClient(hit=12)
    seq = SimpleNamespace(
        id=771,
        num_prompt_tokens=16,
        token_ids=list(range(16)),
        num_cached_tokens=0,
        block_table=[1, 2, 3, 4],
        has_per_req_cache=False,
    )

    assert sched.get_num_new_matched_tokens(seq) == (12, True)
    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is True

    meta = sched.build_connector_meta()
    load_reqs = [req for req in meta.requests if req.load_spec is not None]
    assert len(load_reqs) == 1
    assert load_reqs[0].load_spec.lmcache_cached_tokens == 12


def test_per_req_cache_refusal_also_covers_the_unaligned_handoff_resume():
    """The handoff path decides through the same `_decide_load_after_alloc`,
    and must not reach `_maybe_start_unaligned_handoff` either: that branch
    ends in a load too."""
    sched = _scheduler()
    sched._min_load_tokens = 8
    lookup = _LookupClient(hit=16)
    sched._lookup_client = lookup
    seq = _hybrid_seq(772, 20, [1, 2, 3, 4, 5])

    assert sched.get_num_new_matched_tokens(seq) == (16, True)

    # hbm=6 is unaligned; for a stateless seq this starts a handoff.
    seq.num_cached_tokens = 6
    sched.update_state_after_alloc(seq)
    assert sched.should_park_for_load_after_alloc(seq) is False
    assert str(seq.id) not in sched._handoff_loads
    assert seq.offload_loaded_tokens == 6

    # Nothing is left pending for the partial-prefill park to pick up. This
    # passes through `should_park_partial_prefill_for_load`'s
    # `sid not in _handoff_loads` early return, which is the point here -- the
    # handoff was never started. The guard *inside* that method is a separate
    # call site and needs its own test, below.
    assert sched.should_park_partial_prefill_for_load(seq) is False
    assert sched.build_connector_meta().requests == []


def test_per_req_cache_refusal_holds_at_the_partial_prefill_park():
    """The fourth `_decide_load_after_alloc` call site, reached on its own.

    `should_park_partial_prefill_for_load` refuses a hybrid three times over:
    at the `sid not in _handoff_loads` early return, at
    `_decide_load_after_alloc`'s `unaligned_hbm_prefill`, and at the guard
    itself. The test above trips the first, and an unaligned `hbm` would trip
    the second -- either leaves the guard's own revert green. So park the sid
    by hand AND keep `hbm` chunk-aligned, leaving the guard as the only thing
    that can say no. With it reverted this seq parks and emits a load.
    """
    sched = _scheduler()
    sched._min_load_tokens = 8
    sched._lookup_client = _LookupClient(hit=16)
    seq = _hybrid_seq(774, 20, [1, 2, 3, 4, 5])

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    # Aligned to `chunk_size` (4), so `unaligned_hbm_prefill` cannot be what
    # refuses this; 16 - 4 = 12 also clears `_min_load_tokens`.
    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)

    # What a stateless sequence's unaligned handoff would have left behind.
    sched._handoff_loads.add(str(seq.id))

    assert sched.should_park_partial_prefill_for_load(seq) is False
    # Refused, not merely deferred: the park is cleared and no load is emitted.
    assert str(seq.id) not in sched._handoff_loads
    assert sched.build_connector_meta().requests == []


def test_per_req_cache_refusal_holds_in_build_connector_meta():
    """The third `_decide_load_after_alloc` call site: a LoadSpec that reached
    `can_load` some other way must still be dropped before it is emitted."""
    sched = _scheduler()
    sched._min_load_tokens = 8
    sched._lookup_client = _LookupClient(hit=12)
    seq = _hybrid_seq(773, 16, [1, 2, 3, 4])

    sched.get_num_new_matched_tokens(seq)
    seq.num_cached_tokens = 4
    sched.update_state_after_alloc(seq)
    # Force the state `should_park_for_load_after_alloc` would have cleared.
    sched._load_specs[str(seq.id)].can_load = True
    sched._reqs_need_recv[str(seq.id)] = seq

    meta = sched.build_connector_meta()
    assert [req for req in meta.requests if req.load_spec is not None] == []


def _warn_config(model_type):
    """A config complete enough to run the real `__init__`. The lookup client
    is absent in the test env and its own warning is filtered out below."""
    return SimpleNamespace(
        kv_transfer_config={"kv_connector": "lmcache_offload"},
        kv_cache_block_size=16,
        tensor_parallel_size=1,
        hf_config=SimpleNamespace(model_type=model_type),
    )


def _hybrid_warnings(caplog):
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "per-request recurrent" in r.getMessage()
    ]


def test_hybrid_model_startup_warning_fires_once_and_names_the_model(caplog):
    # Constructed for real, so the `__init__` call site is covered too: a
    # warning nothing invokes would pass a direct-call test and warn nobody.
    with caplog.at_level(logging.WARNING, logger="atom"):
        LMCacheOffloadConnectorScheduler(_warn_config("qwen3_next"))

    warnings = _hybrid_warnings(caplog)
    assert len(warnings) == 1  # once per server, not once per request
    text = warnings[0].getMessage()
    assert "qwen3_next" in text
    assert "OFFLOAD_STATE" in text
    # Saves are unaffected; the user should not read this as "offload is off".
    assert "SAVES still run" in text


def test_dense_model_gets_no_hybrid_startup_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="atom"):
        LMCacheOffloadConnectorScheduler(_warn_config("deepseek_v3"))

    assert _hybrid_warnings(caplog) == []


# ---------------------------------------------------------------------------
# The state tier's load leg, worker side
# ---------------------------------------------------------------------------


class _FakeTier:
    """Enough of `StateOffloadTier` to see what the connector hands it."""

    def __init__(self, done=(), failed=()):
        self.submitted = []
        self._done, self._failed = set(done), set(failed)

    def submit_load(self, req_id, h, group):
        self.submitted.append((req_id, h, group))

    def get_finished(self):
        return set(self._done), set(self._failed)

    def take_spill_reports(self):
        return set(), set(), set()


def _load_worker(tier=None):
    conn = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
    conn._do_load = True
    conn._do_save = False
    conn._lock = threading.Lock()
    conn._done_load = set()
    conn._failed_load = set()
    conn._done_save = set()
    conn._engine = SimpleNamespace(lookup_unpin=lambda _lookup_id: None)
    conn._state_tier = tier
    return conn


def _state_load_meta(*loads):
    meta = LMCacheOffloadMetadata()
    meta.state_loads = list(loads)
    return meta


def test_state_loads_reach_the_tier_with_a_real_pool_group():
    tier = _FakeTier()
    conn = _load_worker(tier)

    conn.start_load_kv(_state_load_meta((7, 111, 3), (8, 222, 5)))

    assert tier.submitted == [(7, 111, 3), (8, 222, 5)]


def test_a_worker_with_no_tier_fails_the_load_rather_than_dropping_it(caplog):
    """The tier can legitimately refuse to build -- pipeline parallelism, an
    entry larger than the staging buffer, a backend with no state views --
    while the engine's index goes on offering loads. Each of those requests is
    already parked, and only a report unparks it, so silence here is a hang.
    """
    conn = _load_worker(tier=None)

    with caplog.at_level(logging.WARNING, logger="atom"):
        conn.start_load_kv(_state_load_meta((7, 111, 3)))

    assert conn._failed_load == {7}
    assert "no state tier" in caplog.text


def test_state_load_reports_merge_into_the_kv_loading_channels():
    """Sharing the channels is what gives the state leg the aggregator's
    per-request quorum for free. Nothing downstream distinguishes the two legs
    and nothing needs to: one request never has both."""
    conn = _load_worker(_FakeTier(done={7}, failed={8}))
    conn._done_load.add(100)
    conn._failed_load.add(200)

    out = conn.get_finished()

    assert out.finished_loading == {7, 100}
    assert out.failed_loading == {8, 200}
