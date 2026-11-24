# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1 NVFP4 KV + NVFP4 index pages through the M3 PAGE offload codec.

The pool is the one the attention backend declares under ``--kv-cache-dtype nvfp4
--index-cache-dtype nvfp4`` (one ``kv`` field of K and V with inline scales, the
``index`` pages riding the same entry) plus the DSpark draft's own bf16 pool, and
the regions are the ones ``get_kv_transfer_tensors`` publishes. A save gathers
blocks into the chunk-major staging buffer, a load scatters it into other blocks;
every byte of every region must come back, and nothing outside the target blocks
may change.
"""

import pytest
import torch

from atom.config import FP4_CACHE_DTYPES
from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.minimax_m3.nvfp4_pages import (
    INDEX_PAGE_BYTES,
    Nvfp4KvLayout,
    kv_page_bytes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the PAGE staging kernels need a GPU"
)

LAYERS = 6
DRAFT_LAYERS = 2
BLOCKS = 64


def _pools(device):
    target = MhaKvPool(
        layers=LAYERS,
        block_size=128,
        num_kv_heads=1,
        head_dim=128,
        kv_dtype=torch.uint8,
        extra_fields=(EntryField("index", LAYERS, (INDEX_PAGE_BYTES,), torch.uint8),),
        layout=Nvfp4KvLayout(),
    )
    draft = MhaKvPool(
        layers=DRAFT_LAYERS,
        block_size=128,
        num_kv_heads=1,
        head_dim=128,
        kv_dtype=torch.bfloat16,
        flash_layout=True,
    )
    target.allocate(BLOCKS, device)
    draft.allocate(BLOCKS, device)
    return target, draft


def _transfer(target, draft):
    layers = list(range(LAYERS))
    tensors = KVTransferTensors(
        pages=target.transfer_pages(
            "mha",
            "mha.nvfp4",
            layers,
            [16] * LAYERS,
            {"index": (layers, 128, torch.uint8)},
        )
    )
    tensors.merge_pages(
        KVTransferTensors(
            pages=draft.transfer_pages(
                "draft", "draft", [LAYERS + i for i in range(DRAFT_LAYERS)]
            )
        )
    )
    tensors.set_block_count(BLOCKS)
    return tensors


def _fill(pool, gen):
    for arena in (pool.cache, pool.scale, pool.extra):
        if arena is None:
            continue
        for field in arena.fields:
            view = arena.view(field.name)
            view.view(torch.uint8).copy_(
                torch.randint(
                    0, 256, view.view(torch.uint8).shape, dtype=torch.uint8, generator=gen
                ).to(view.device)
            )


def _block_bytes(pool, block):
    out = []
    for name in pool._views:
        view = pool._views[name]
        for layer in range(view.shape[0]):
            out.append(view[layer][block].reshape(-1).view(torch.uint8).clone())
    return torch.cat(out)


def test_nvfp4_page_bytes_match_the_pool():
    target, draft = _pools("cuda")
    assert "nvfp4" in FP4_CACHE_DTYPES
    # 60 layers of K/V + index at TP4 is the 1658880 B/block the server reports
    assert kv_page_bytes(1) + INDEX_PAGE_BYTES == 1658880 // 60
    assert target.entry_bytes == LAYERS * (kv_page_bytes(1) + INDEX_PAGE_BYTES)
    tensors = _transfer(target, draft)
    total = sum(r.unit_bytes for r in tensors.block_regions)
    assert total == target.entry_bytes + draft.entry_bytes


def test_nvfp4_pages_round_trip_through_the_m3_codec():
    from atom.kv_transfer.offload.page_codec import pure_page_codec

    gen = torch.Generator().manual_seed(0)
    target, draft = _pools("cuda")
    _fill(target, gen)
    _fill(draft, gen)
    codec = pure_page_codec(_transfer(target, draft), BLOCKS)
    assert codec.bytes_per_block == target.entry_bytes + draft.entry_bytes

    src = [3, 17, 4, 40, 41, 9, 63, 0]
    dst = [50, 22, 23, 1, 2, 33, 34, 35]
    before = {
        b: (_block_bytes(target, b), _block_bytes(draft, b)) for b in range(BLOCKS)
    }
    stage = torch.empty(
        len(src) * codec.bytes_per_block, dtype=torch.uint8, device="cuda"
    )
    codec.gpu_to_chunk_major_device_buffer(stage, src)
    # wipe the destination so a block the scatter misses cannot pass
    for pool in (target, draft):
        for view in pool._views.values():
            view[:, dst].view(torch.uint8).zero_()
    codec.chunk_major_device_buffer_to_gpu(stage, dst)
    torch.cuda.synchronize()

    for s, d in zip(src, dst):
        assert torch.equal(_block_bytes(target, d), before[s][0]), (s, d)
        assert torch.equal(_block_bytes(draft, d), before[s][1]), (s, d)
    for b in set(range(BLOCKS)) - set(dst):
        assert torch.equal(_block_bytes(target, b), before[b][0]), b
        assert torch.equal(_block_bytes(draft, b), before[b][1]), b


def test_m3_fp4_refusal_admits_only_the_in_process_offload():
    from atom.models.minimax_m3.config import _is_page_offload

    assert _is_page_offload({"kv_connector": "lmcache_offload", "kv_role": "offload"})
    assert not _is_page_offload({"kv_connector": "mooncake", "kv_role": "kv_producer"})
    assert not _is_page_offload({"kv_connector": "lmcache_mp", "kv_role": "offload"})
    assert not _is_page_offload("lmcache_offload")
