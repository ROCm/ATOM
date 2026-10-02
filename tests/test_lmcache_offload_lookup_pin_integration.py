# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib.util
import uuid
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import LoadOperationId
from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector
from atom.kv_transfer.offload.metadata import (
    ATOMRawBytesLMCacheMetadata,
    LMCacheReqMeta,
    LoadSpec,
)

torch = pytest.importorskip(
    "torch",
    reason="real PyTorch is required for LMCache lookup pin integration",
)
if not hasattr(torch, "Tensor") or not hasattr(torch, "arange"):
    pytest.skip("real torch is unavailable", allow_module_level=True)
_LMCACHE_AVAILABLE = importlib.util.find_spec("lmcache") is not None
if _LMCACHE_AVAILABLE:
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.metadata import LMCacheMetadata

pytestmark = pytest.mark.skipif(
    not _LMCACHE_AVAILABLE,
    reason="the external lmcache package is required for lookup pin integration",
)

_CHUNK = 8


class _CPUChunkConnector:
    """Keep the MemoryObjs LMCache stores, so their pin counts can be read,
    without requiring a GPU."""

    def __init__(self) -> None:
        self.stored = []

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        del starts, ends, kwargs
        self.stored.extend(memory_objs)

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        del memory_objs, starts, ends, kwargs


def _load(req_id: int, tokens: list[int], *, hbm: int) -> LMCacheReqMeta:
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=list(tokens),
        block_ids=list(range(len(tokens) // 4)),
        load_spec=LoadSpec(
            hbm_cached_tokens=hbm,
            lmcache_cached_tokens=len(tokens),
            can_load=True,
        ),
        load_operation=LoadOperationId(req_id=req_id, generation=1),
    )


def test_dense_load_releases_each_real_lmcache_lookup_pin_once():
    """Two lookups pin the same three chunks. The dense worker's load releases
    exactly its own pins, whichever cleanup this LMCache's retrieve runs: 0.4.5
    and #5098 keep the pins of the chunks they return, #3884 drops them."""
    instance_id = f"atom-lookup-pin-test-{uuid.uuid4().hex}"
    connector = _CPUChunkConnector()
    tokens = list(range(3 * _CHUNK))
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=_CHUNK,
        local_cpu=True,
        max_local_cpu_size=0.01,
        enable_async_loading=False,
    )
    base_metadata = LMCacheMetadata(
        model_name="atom-lookup-pin-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(1, 2, _CHUNK, 1, 1),
        chunk_size=_CHUNK,
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
    worker = DenseOffloadConnector(
        SimpleNamespace(
            kv_transfer_config={"kv_role": "kv_consumer"}, kv_cache_block_size=4
        )
    )
    worker._engine = engine
    worker.chunk_size = _CHUNK
    try:
        engine.fmt = MemoryFormat.KV_2LTD
        engine.post_init()
        engine.store(tokens)
        chunks = connector.stored

        def pin_counts() -> list[int]:
            return [chunk.metadata.pin_count for chunk in chunks]

        assert len(chunks) == 3
        assert engine.lookup(tokens, lookup_id="71", pin=True) == len(tokens)
        assert engine.lookup(tokens, lookup_id="72", pin=True) == len(tokens)
        assert pin_counts() == [2, 2, 2]

        # Chunk 0 is already in HBM, so 71 retrieves only chunks 1 and 2.
        first = _load(71, tokens, hbm=_CHUNK)
        worker._do_load_req(first)

        assert pin_counts() == [1, 1, 1]
        assert list(engine.lookup_pins) == ["72"]

        second = _load(72, tokens, hbm=0)
        worker._do_load_req(second)

        assert pin_counts() == [0, 0, 0]
        assert not engine.lookup_pins
        assert worker.get_finished().finished_loading == {
            first.load_operation,
            second.load_operation,
        }
    finally:
        worker.close()
        LMCacheEngineBuilder.destroy(instance_id)
