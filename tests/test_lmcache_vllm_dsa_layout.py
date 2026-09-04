# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache must move a DSA KV layout without disturbing it.

The indexer pages are not token-major: aiter's
``indexer_k_quant_and_cache(preshuffle=True)`` scatters the fp8 keys into MFMA
16x16 tiles and parks the fp32 scales at the tail of the page, while LMCache's
transfer kernel addresses memory as ``slot * hidden_dim``. They still round-trip
because the scatter is a bijection within one page and LMCache moves whole
pages -- a property of its ``chunk_size`` / ``save_unfull_chunk`` defaults that
``enforce_lmcache_gpu_connector`` enforces rather than assumes.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("moves real paged KV cache; needs a GPU", allow_module_level=True)

lmcache_gpu_connectors = pytest.importorskip(
    "lmcache.v1.gpu_connector.gpu_connectors",
    reason="needs LMCache built for ROCm (BUILD_WITH_HIP=1)",
)
aiter = pytest.importorskip("aiter", reason="needs aiter for the indexer kernels")

from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata

VLLMPagedMemGPUConnectorV2 = lmcache_gpu_connectors.VLLMPagedMemGPUConnectorV2
VLLMPagedMemGPUConnectorV3 = lmcache_gpu_connectors.VLLMPagedMemGPUConnectorV3

BLOCK_SIZE = 64  # AiterSparseMlaBackendForVllm.get_preferred_block_size()
NUM_BLOCKS = 32
CHUNK_SIZE = 256  # LMCACHE_CHUNK_SIZE, four whole pages
NUM_MLA_LAYERS = 3
NUM_INDEXER_LAYERS = 2
MLA_HEAD_SIZE = 576  # kv_lora_rank 512 + qk_rope_head_dim 64
INDEXER_HEAD_DIM = 128
INDEXER_HEAD_SIZE = INDEXER_HEAD_DIM + 4  # fp8 keys + one fp32 scale per token

SRC_BLOCKS = [7, 2, 11, 5]
DST_BLOCKS = [1, 13, 4, 9]


class _MemoryObjStub:
    """The slice of LMCache's MemoryObj that a GPU connector touches."""

    def __init__(self, shapes, dtypes):
        # The transfer kernel copies straight into host memory, so it has to
        # be pinned -- an unpinned buffer fails with "Host tensor not
        # registered/pinned". LMCache's real allocator pins its CPU pool.
        self._tensors = [
            torch.zeros(shape, dtype=dtype, device="cpu").pin_memory()
            for shape, dtype in zip(shapes, dtypes)
        ]
        self.metadata = type("_Meta", (), {"fmt": MemoryFormat.KV_MLA_FMT})()

    @property
    def raw_tensor(self):
        return self._tensors[0]

    @property
    def tensor(self):
        return self._tensors[0]

    def get_tensor(self, index):
        return self._tensors[index]


def _slot_mapping(block_ids, device):
    return torch.tensor(
        [b * BLOCK_SIZE + offset for b in block_ids for offset in range(BLOCK_SIZE)],
        dtype=torch.long,
        device=device,
    )


def _metadata():
    """What lmcache.integration.vllm.utils.create_lmcache_metadata() builds.

    Note ``kv_shape`` counts only the model's transformer layers -- vLLM's model
    config has no idea the indexer caches exist. That mismatch is the bug.
    """
    return LMCacheMetadata(
        model_name="dsa-layout-test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(NUM_MLA_LAYERS, 1, CHUNK_SIZE, 1, MLA_HEAD_SIZE),
        use_mla=True,
        chunk_size=CHUNK_SIZE,
    )


@pytest.fixture
def dsa_kv_caches():
    """MLA + preshuffled indexer caches, as the ATOM plugin allocates them."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    mla = [
        torch.randint(
            0,
            255,
            (NUM_BLOCKS, BLOCK_SIZE, MLA_HEAD_SIZE),
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(NUM_MLA_LAYERS)
    ]
    indexer = [
        torch.zeros(
            (NUM_BLOCKS, BLOCK_SIZE, INDEXER_HEAD_SIZE),
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(NUM_INDEXER_LAYERS)
    ]

    slots = _slot_mapping(SRC_BLOCKS, device)
    for cache in indexer:
        keys = torch.randn(
            len(slots), INDEXER_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        aiter.indexer_k_quant_and_cache(
            keys,
            cache,
            slots,
            quant_block_size=INDEXER_HEAD_DIM,
            scale_fmt="ue8m0",
            preshuffle=True,
        )
    torch.cuda.synchronize()
    return device, mla, indexer


def _gather_indexer(cache, block_ids, device):
    """Read the indexer cache back the way the sparse attention layer does."""
    from aiter import cp_gather_indexer_k_quant_cache, dtypes

    keys = torch.empty([CHUNK_SIZE, INDEXER_HEAD_DIM], device=device, dtype=dtypes.fp8)
    scales = torch.empty([CHUNK_SIZE, 1], device=device, dtype=torch.float32)
    cp_gather_indexer_k_quant_cache(
        cache,
        keys,
        scales.view(dtypes.fp8),
        torch.tensor([block_ids], dtype=torch.int32, device=device),
        torch.tensor([0, CHUNK_SIZE], dtype=torch.int32, device=device),
        preshuffle=True,
    )
    return keys.view(torch.uint8).clone(), scales.clone()


def test_default_connector_cannot_express_a_dsa_layout(dsa_kv_caches):
    device, mla, indexer = dsa_kv_caches
    connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        _metadata(), use_gpu=False, device=device, layout_hints={"kv_layout": "NHD"}
    )
    kv_caches = mla + indexer
    memory_obj = _MemoryObjStub(
        [torch.Size([1, NUM_MLA_LAYERS, CHUNK_SIZE, MLA_HEAD_SIZE])], [torch.uint8]
    )

    with pytest.raises(ValueError, match="broadcast"):
        connector.from_gpu(
            memory_obj,
            0,
            CHUNK_SIZE,
            kvcaches=kv_caches,
            slot_mapping=_slot_mapping(SRC_BLOCKS, device),
        )


def test_v3_connector_round_trips_both_page_geometries(dsa_kv_caches):
    device, mla, indexer = dsa_kv_caches
    kv_caches = mla + indexer

    expected_mla = [cache[SRC_BLOCKS].clone() for cache in mla]
    expected_indexer = [cache[SRC_BLOCKS].clone() for cache in indexer]
    expected_gather = [_gather_indexer(cache, SRC_BLOCKS, device) for cache in indexer]

    metadata = _metadata()
    connector = VLLMPagedMemGPUConnectorV3.from_metadata(
        metadata, use_gpu=False, device=device, layout_hints={"kv_layout": "NHD"}
    )
    connector.initialize_kvcaches_ptr(kvcaches=kv_caches)
    connector._initialize_kv_cache_pointers()

    groups = metadata.kv_layer_groups_manager.kv_layer_groups
    assert [(g.num_layers, g.shape_desc.hs) for g in groups] == [
        (NUM_MLA_LAYERS, MLA_HEAD_SIZE),
        (NUM_INDEXER_LAYERS, INDEXER_HEAD_SIZE),
    ]
    assert {g.shape_desc.bs for g in groups} == {BLOCK_SIZE}, (
        "V3 forwards one scalar block_size to the transfer kernel; groups that "
        "disagree would silently corrupt transfers"
    )

    memory_obj = _MemoryObjStub(metadata.get_shapes(CHUNK_SIZE), metadata.get_dtypes())
    connector.from_gpu(
        memory_obj,
        0,
        CHUNK_SIZE,
        kvcaches=kv_caches,
        slot_mapping=_slot_mapping(SRC_BLOCKS, device),
    )
    torch.cuda.synchronize()

    for cache in kv_caches:
        for block_id in DST_BLOCKS:
            cache[block_id].zero_()
    torch.cuda.synchronize()

    connector.to_gpu(
        memory_obj,
        0,
        CHUNK_SIZE,
        kvcaches=kv_caches,
        slot_mapping=_slot_mapping(DST_BLOCKS, device),
    )
    torch.cuda.synchronize()

    for i, cache in enumerate(mla):
        assert torch.equal(cache[DST_BLOCKS], expected_mla[i]), f"MLA layer {i}"

    for i, cache in enumerate(indexer):
        assert torch.equal(cache[DST_BLOCKS], expected_indexer[i]), f"indexer {i} bytes"
        keys, scales = _gather_indexer(cache, DST_BLOCKS, device)
        assert torch.equal(keys, expected_gather[i][0]), f"indexer {i} keys"
        assert torch.equal(scales, expected_gather[i][1]), f"indexer {i} scales"
