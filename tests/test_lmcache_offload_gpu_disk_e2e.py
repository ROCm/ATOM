# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if (
    not hasattr(torch, "version")
    or getattr(torch.version, "hip", None) is None
    or not hasattr(torch, "cuda")
    or not torch.cuda.is_available()
):
    pytest.skip("a ROCm GPU is required", allow_module_level=True)
pytest.importorskip("lmcache")

from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata

from atom.kv_transfer.offload.atom_kv_byte_codec import ATOMKVByteCodec
from atom.kv_transfer.offload.atom_lmcache_gpu_connector import (
    ATOMLMCacheGPUConnector,
)
from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata


def _uint8_payload(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    count = math.prod(shape)
    return (
        torch.arange(count, dtype=torch.int64, device="cuda")
        .add(offset)
        .remainder(251)
        .to(torch.uint8)
        .reshape(shape)
    )


def _make_aiter_kv_caches(
    *,
    num_blocks: int,
    block_size: int,
) -> dict[str, SimpleNamespace]:
    num_heads = 2
    head_dim = 16
    pack = 16
    caches: dict[str, SimpleNamespace] = {}
    for layer_id in range(2):
        base = layer_id * 47
        caches[f"layer-{layer_id}"] = SimpleNamespace(
            # AITER x-packed K: (blocks, heads, head_dim/x, block_size, x).
            k_cache=_uint8_payload(
                (num_blocks, num_heads, head_dim // pack, block_size, pack),
                base,
            ),
            # AITER head-major V: (blocks, heads, head_dim, block_size).
            v_cache=_uint8_payload(
                (num_blocks, num_heads, head_dim, block_size),
                base + 13,
            ),
            k_scale=torch.linspace(
                0.25 + layer_id,
                1.75 + layer_id,
                steps=num_blocks * num_heads,
                dtype=torch.float32,
                device="cuda",
            ).reshape(num_blocks, num_heads),
            v_scale=torch.linspace(
                2.25 + layer_id,
                3.75 + layer_id,
                steps=num_blocks * num_heads,
                dtype=torch.float32,
                device="cuda",
            ).reshape(num_blocks, num_heads),
        )
    return caches


def _clone_segments(
    kv_caches: dict[str, SimpleNamespace],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        layer_name: {
            field: getattr(cache, field).clone()
            for field in ("k_cache", "v_cache", "k_scale", "v_scale")
        }
        for layer_name, cache in kv_caches.items()
    }


def _wait_for_disk_hit(
    engine,
    tokens: list[int],
    disk_path: Path,
    *,
    timeout_s: float = 20.0,
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


def test_gpu_kv_round_trip_through_cpu_staging_and_nvme(tmp_path: Path):
    """Round-trip AITER-layout GPU KV bytes through LMCache local disk."""
    num_blocks = 4
    block_size = 4
    chunk_size = 8
    tokens = list(range(num_blocks * block_size))
    block_ids = list(range(num_blocks))
    instance_id = f"atom-gpu-disk-e2e-{uuid.uuid4().hex}"

    kv_caches = _make_aiter_kv_caches(
        num_blocks=num_blocks,
        block_size=block_size,
    )
    expected = _clone_segments(kv_caches)
    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)
    assert codec.has_fused_chunk_major_staging
    gpu_connector = ATOMLMCacheGPUConnector(
        codec,
        hash_block_size=block_size,
        chunk_size=chunk_size,
    )

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=False,
        max_local_cpu_size=0.01,
        local_disk=str(tmp_path),
        max_local_disk_size=0.01,
        store_location="LocalDiskBackend",
        retrieve_locations=["LocalDiskBackend"],
        use_gds=False,
    )
    base_metadata = LMCacheMetadata(
        model_name="atom-gpu-disk-e2e",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(2, 2, chunk_size, 2, 16),
        chunk_size=chunk_size,
        engine_id=instance_id,
    )
    metadata = ATOMRawBytesLMCacheMetadata(
        base_metadata,
        atom_hash_block_size=block_size,
        bytes_per_block=codec.bytes_per_block,
    )

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        config,
        metadata,
        gpu_connector,
        lambda tensor, source: None,
        lambda obj, source: obj,
    )
    try:
        engine.fmt = MemoryFormat.KV_2LTD
        engine.post_init()
        assert engine.storage_manager is not None
        assert "LocalDiskBackend" in engine.storage_manager.list_backends()

        engine.store(tokens, block_ids=block_ids)
        disk_files = _wait_for_disk_hit(engine, tokens, tmp_path)

        assert len(disk_files) == 2
        assert all(path.stat().st_size > 0 for path in disk_files)
        engine.set_hot_cache(False)
        assert engine.lookup(tokens, search_range=["LocalCPUBackend"]) == 0

        for cache in kv_caches.values():
            cache.k_cache.zero_()
            cache.v_cache.zero_()
            cache.k_scale.zero_()
            cache.v_scale.zero_()
        torch.cuda.synchronize()

        ret_mask = engine.retrieve(tokens, block_ids=block_ids)
        torch.cuda.synchronize()

        assert bool(ret_mask.all())
        for layer_name, cache in kv_caches.items():
            for field, expected_tensor in expected[layer_name].items():
                assert torch.equal(
                    getattr(cache, field), expected_tensor
                ), f"GPU KV mismatch after NVMe round trip: {layer_name}.{field}"
    finally:
        LMCacheEngineBuilder.destroy(instance_id)
