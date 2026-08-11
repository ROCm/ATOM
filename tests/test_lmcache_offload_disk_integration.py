# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "Tensor") or not hasattr(torch, "arange"):
    pytest.skip("real torch is unavailable", allow_module_level=True)
pytest.importorskip("lmcache")

from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata

from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata


class _CPUBytesConnector:
    """Fill and capture LMCache MemoryObjs without requiring a GPU."""

    def __init__(self) -> None:
        self.expected: list[torch.Tensor] = []
        self.retrieved: list[torch.Tensor] = []

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        del starts, ends, kwargs
        self.expected = []
        for index, memory_obj in enumerate(memory_objs):
            assert memory_obj.tensor is not None
            payload = (
                torch.arange(memory_obj.tensor.numel(), dtype=torch.int64)
                .add(index * 73)
                .remainder(256)
                .to(torch.uint8)
            )
            memory_obj.tensor.copy_(payload)
            self.expected.append(payload.clone())

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        del starts, ends, kwargs
        self.retrieved = []
        for memory_obj in memory_objs:
            assert memory_obj.tensor is not None
            self.retrieved.append(memory_obj.tensor.detach().cpu().clone())


def _wait_for_disk_chunks(
    engine,
    tokens: list[int],
    disk_path: Path,
    *,
    timeout_s: float = 5.0,
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        files = list(disk_path.rglob("*.pt"))
        if len(files) == 2 and engine.lookup(
            tokens, search_range=["LocalDiskBackend"]
        ) == len(tokens):
            return files
        time.sleep(0.01)
    return list(disk_path.rglob("*.pt"))


def test_lmcache_local_disk_round_trip_without_cpu_hot_cache(tmp_path: Path):
    """Store raw ATOM bytes on disk, clear CPU cache, and retrieve them."""
    instance_id = f"atom-disk-test-{uuid.uuid4().hex}"
    connector = _CPUBytesConnector()
    tokens = list(range(16))
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=8,
        local_cpu=False,
        max_local_cpu_size=0.01,
        local_disk=str(tmp_path),
        max_local_disk_size=0.01,
        store_location="LocalDiskBackend",
        retrieve_locations=["LocalDiskBackend"],
        use_gds=False,
    )
    base_metadata = LMCacheMetadata(
        model_name="atom-disk-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(1, 2, 8, 1, 1),
        chunk_size=8,
        engine_id=instance_id,
    )
    metadata = ATOMRawBytesLMCacheMetadata(
        base_metadata,
        atom_block_size=4,
        bytes_per_block=32,
    )

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        config,
        metadata,
        connector,
        lambda tensor, source: None,
        lambda obj, source: obj,
    )
    try:
        engine.fmt = MemoryFormat.KV_2LTD
        engine.post_init()
        assert engine.storage_manager is not None
        assert "LocalDiskBackend" in engine.storage_manager.list_backends()

        engine.store(tokens)
        disk_files = _wait_for_disk_chunks(engine, tokens, tmp_path)

        assert len(disk_files) == 2
        assert all(path.stat().st_size > 0 for path in disk_files)
        assert engine.lookup(tokens, search_range=["LocalDiskBackend"]) == len(tokens)

        engine.set_hot_cache(False)
        assert engine.lookup(tokens, search_range=["LocalCPUBackend"]) == 0

        ret_mask = engine.retrieve(tokens)

        assert bool(ret_mask.all())
        assert len(connector.retrieved) == len(connector.expected) == 2
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                connector.retrieved, connector.expected, strict=True
            )
        )
    finally:
        LMCacheEngineBuilder.destroy(instance_id)
