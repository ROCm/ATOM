# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What a sharded MiniMax-M3 index cache costs, and how many blocks that buys.

M3.1 at TP4 (NVFP4 sparse K/V, FP4 index, a 5-layer FP8 DSpark draft), per rank
and 128-token block: K/V 983,040 B + their scales 122,880 B + the index
552,960 B + the draft 168,960 B = 1,827,840 B. Sharded, a rank holds the index
of a quarter of the blocks, so a block costs 1,413,120 B, and the 0.95-
utilization budget that bought 176,606 blocks buys 228,436 (a multiple of 4,
so every lane of the allocator holds the same number).
"""

import pytest
import torch

from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    page_pool,
    plan_pools,
)

LAYERS, INDEX_LAYERS, DRAFT_LAYERS, BLOCK, P = 60, 60, 5, 128, 4
TODAY_BLOCKS = 176_606


def index_fields(rows=INDEX_LAYERS):
    """The FP4 index fields as `AiterAttentionMetadataBuilder` declares them:
    64 B of e2m1 payload and 8 B of E4M3 scales a token."""
    return (
        EntryField("index", rows, (BLOCK * 64,), torch.uint8),
        EntryField("index_scale", rows, (BLOCK * 8,), torch.uint8),
    )


def target_pool(divisor=1, layers=LAYERS, rows=INDEX_LAYERS, rank=0):
    return MhaKvPool(
        layers=layers,
        block_size=BLOCK,
        num_kv_heads=1,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
        extra_fields=index_fields(rows),
        kv_format="nvfp4",
        extra_shard=(rank, divisor),
    )


def draft_pool():
    return MhaKvPool(
        layers=DRAFT_LAYERS,
        block_size=BLOCK,
        num_kv_heads=1,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
    )


class TestPrice:
    def test_a_replicated_block_is_what_the_server_logs(self):
        assert target_pool().entry_bytes + draft_pool().entry_bytes == 1_827_840

    def test_a_sharded_block_holds_a_quarter_of_the_index(self):
        assert target_pool(P).entry_bytes == 983_040 + 122_880 + 552_960 // P
        assert target_pool(P).entry_bytes + draft_pool().entry_bytes == 1_413_120
        # 11,040 B a token, against 14,280.
        assert (target_pool(P).entry_bytes + draft_pool().entry_bytes) // BLOCK == (
            11_040
        )

    def test_the_budget_buys_29_percent_more_blocks(self):
        budget = TODAY_BLOCKS * 1_827_840  # what 0.95 left for KV: 300.6 GiB
        today = plan_pools(
            [page_pool(target_pool().entry_bytes), page_pool(draft_pool().entry_bytes)],
            budget,
            max_num_seqs=256,
        )
        sharded = plan_pools(
            [
                page_pool(target_pool(P).entry_bytes, P),
                page_pool(draft_pool().entry_bytes),
            ],
            budget,
            max_num_seqs=256,
        )
        assert today.paged_entries == TODAY_BLOCKS
        assert sharded.paged_entries == 228_436
        assert sharded.paged_entries % P == 0

    def test_the_block_count_is_a_multiple_of_every_contributor(self):
        plan = plan_pools([page_pool(100, 4), page_pool(10, 6)], 110 * 35, 1)
        assert plan.paged_entries == 24  # 35 rounded down to lcm(4, 6) = 12

    def test_a_split_that_leaves_a_remainder_is_refused(self):
        with pytest.raises(ValueError, match="do not split 7 ways"):
            target_pool(7)


class TestAllocation:
    def test_the_index_planes_are_a_quarter_of_the_blocks(self):
        pool = target_pool(P, layers=2, rows=2)
        pool.allocate(8, torch.device("cpu"))
        index = pool.field_view("index", 1, torch.uint8, (-1, BLOCK, 64))
        scale = pool.field_view("index_scale", 1, torch.uint8, (-1, BLOCK, 8))
        k, _ = pool.kv_views(1)
        assert (index.shape[0], scale.shape[0], k.shape[0]) == (2, 2, 8)
        arenas = pool.cache.total_bytes + pool.scale.total_bytes
        assert arenas + pool.extra.total_bytes == pool.pool_bytes(8)

    def test_a_count_the_divisor_does_not_divide_is_refused(self):
        pool = target_pool(P, layers=2, rows=2)
        with pytest.raises(ValueError, match="multiples of 4"):
            pool.allocate(6, torch.device("cpu"))
        with pytest.raises(ValueError, match="multiples of 4"):
            pool.pool_bytes(6)

    def test_published_index_regions_are_this_ranks_share(self):
        pool = target_pool(P, layers=2, rows=2, rank=3)
        pool.allocate(8, torch.device("cpu"))
        pages = [
            page_region(tensor, semantic_role=role, shard=pool.region_shard(role))
            for role, tensor in pool.region_tensors()
        ]
        sharded = {
            p.region.semantic_role
            for p in pages
            if (p.region.shard_rank, p.region.shard_world) == (3, P)
        }
        assert sharded == {
            f"{f}.layer_{i}" for f in ("index", "index_scale") for i in (0, 1)
        }
        tensors = KVTransferTensors(pages=pages)
        tensors.set_block_count(8)
        unsharded = KVTransferTensors(
            pages=[page_region(t, semantic_role=r) for r, t in pool.region_tensors()]
        )
        with pytest.raises(ValueError, match="holds 2 blocks"):
            unsharded.set_block_count(8)

    def test_a_regions_share_maps_block_ids_to_units(self):
        pool = target_pool(P, layers=1, rows=1)
        pool.allocate(8, torch.device("cpu"))
        role, tensor = pool.region_tensors()[-1]
        region = page_region(tensor, semantic_role=role, shard=(2, 4)).region
        assert region.shard_units([6, 3, 2, 10, 7]) == ([0, 2, 3], [1, 0, 2])
