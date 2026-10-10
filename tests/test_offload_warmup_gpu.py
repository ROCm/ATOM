# SPDX-License-Identifier: MIT
"""Exercise startup warmup through real LMCache, MLA byte codecs and GPU DMA."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload.connector import LMCacheOffloadConnector
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.kv_transfer.offload.hybrid.kimi_k3.connector import KimiK3OffloadConnector
from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm/CUDA GPU")
def test_k3_startup_warmup_round_trip_on_gpu(monkeypatch):
    pytest.importorskip("lmcache")
    pytest.importorskip("triton")
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.metadata import LMCacheMetadata

    monkeypatch.setenv("OFFLOAD_GPU_STAGING_CHUNKS", "32")
    monkeypatch.delenv("OFFLOAD_GPU_STAGING_MAX_BYTES", raising=False)
    monkeypatch.setenv("OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER", "0")
    # K3 TP8/DCP8: 24 MLA layers, 128 physical tokens per block,
    # 1024 global tokens per chunk, and 1,769,472 bytes/chunk on each rank.
    block_size, virtual_block_size, chunk_size = 128, 1024, 1024
    num_blocks = 67  # null block + 33 source blocks + 33 destination blocks
    shape = (num_blocks * block_size, 1, 576)
    caches = {
        str(layer): SimpleNamespace(
            k_cache=torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
        )
        for layer in range(24)
    }
    expected = {layer: cache.k_cache.clone() for layer, cache in caches.items()}
    state = torch.full((3, 19), 7, dtype=torch.uint8, device="cuda")
    caches["kda"] = SimpleNamespace(per_request_state=True, k_cache=state)
    codec = DenseKVByteCodec(
        caches, num_blocks=num_blocks, permit_per_request_state=True
    )
    assert codec.bytes_per_block == 1769472
    gpu = BlockGPUConnector(
        codec, block_size, chunk_size=chunk_size, virtual_block_size=virtual_block_size
    )
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
        max_local_cpu_size=0.25,
        local_disk=None,
        remote_url=None,
        enable_async_loading=False,
        store_location="LocalCPUBackend",
        retrieve_locations=["LocalCPUBackend"],
    )
    instance_id = f"atom-k3-warmup-test-{uuid.uuid4().hex}"
    metadata = ATOMRawBytesLMCacheMetadata(
        LMCacheMetadata(
            model_name="atom-k3-warmup-test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.uint8,
            kv_shape=(24, 1, chunk_size, 1, 576),
            chunk_size=chunk_size,
            use_mla=False,  # ATOM uses opaque per-rank bytes, including for MLA.
            engine_id=instance_id,
        ),
        atom_block_size=virtual_block_size,
        bytes_per_block=codec.bytes_per_block,
    )
    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        config,
        metadata,
        gpu,
        lambda tensor, source: None,
        lambda obj, source: obj,
    )
    try:
        engine.fmt = MemoryFormat.KV_2LTD
        engine.post_init()
        worker = KimiK3OffloadConnector.__new__(KimiK3OffloadConnector)
        worker._engine, worker._codec, worker._rank = engine, codec, 0
        worker._do_save = worker._do_load = True
        worker.save_workers = worker.load_workers = 1
        worker.block_size = block_size
        worker.virtual_block_size, worker.chunk_size = virtual_block_size, chunk_size
        shell = LMCacheOffloadConnector.__new__(LMCacheOffloadConnector)
        shell._impl = worker
        with ThreadPoolExecutor(1) as save_pool, ThreadPoolExecutor(1) as load_pool:
            worker._save_executor, worker._load_executor = save_pool, load_pool
            shell.warmup()
            stats = load_pool.submit(gpu.last_transfer_stats).result()
            assert stats["completed_bytes"] == 33 * codec.bytes_per_block
            assert stats["groups"] == 2
            assert stats["transfer_succeeded"] == 1
            # Idempotence must not repeat the warmup on live cache blocks.
            shell.warmup()
        torch.cuda.synchronize()
        for layer, before in expected.items():
            actual = caches[layer].k_cache.view(num_blocks, -1)
            before = before.view(num_blocks, -1)
            assert torch.equal(actual[:34], before[:34]), layer
            assert torch.equal(actual[34:67], before[1:34]), layer
        assert bool((state == 7).all()), "warmup changed KDA state"
        assert worker._offload_warmed
        assert engine.clear(locations=["LocalCPUBackend"]) == 0
    finally:
        gpu.close()
        LMCacheEngineBuilder.destroy(instance_id)
