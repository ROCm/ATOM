# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dsa_indexer.plan``: a rank's keys of a row, under DCP (interleave 1) and
replicated (128-key runs), against counting every position's owner."""

import pytest
import torch

from atom.model_ops.sparse_indexer_fp4 import (
    FP4_KV_BLOCK_SIZE,
    fp4_index_block_shapes,
    fp4_index_scale_rows,
)
from atom.mono.dsa_indexer.plan import (
    REPLICA_INTERLEAVE,
    IndexerShape,
    fp4_byte,
    fp4_scale_byte,
    key_pos,
    local_len,
    q_fp4_byte,
    q_fp4_scale,
    unsupported,
)

CTXS = [0, 1, 7, 127, 128, 129, 1023, 1024, 1025, 2049, 5000, 33333, 131073]


@pytest.mark.parametrize("interleave", [1, REPLICA_INTERLEAVE])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_a_ranks_keys_are_the_positions_it_owns_in_order(world, interleave):
    for ctx in CTXS:
        owned = [[] for _ in range(world)]
        for pos in range(ctx):
            owned[pos // interleave % world].append(pos)
        for rank in range(world):
            n = local_len(ctx, rank, world, interleave)
            assert n == len(owned[rank]), (ctx, rank)
            got = [key_pos(j, rank, world, interleave) for j in range(n)]
            assert got == owned[rank], (ctx, rank)


def _shape(kv_dtype="fp4", rope_neox=False):
    return IndexerShape(
        heads=32,
        head_dim=128,
        topk=2048,
        q_lora=2048,
        wq_dtype="bf16",
        kv_dtype=kv_dtype,
        block_rows=FP4_KV_BLOCK_SIZE,
        rope_dim=64,
        rope_neox=rope_neox,
    )


def test_an_fp4_cache_is_served_only_with_the_writers_rope():
    assert unsupported(_shape()) == ""
    assert "interleaved RoPE" in unsupported(_shape(rope_neox=True))


def test_fp4_cache_and_query_bytes_match_the_production_layout():
    """One independent byte oracle: fixed cache offsets from the writer's
    published shapes/row helper, plus the scorer's fixed query-row offsets."""
    sh = _shape()
    data_shape, scale_shape = fp4_index_block_shapes(sh.block_rows, sh.head_dim)
    assert torch.tensor(data_shape).prod().item() == sh.block_bytes
    assert torch.tensor(scale_shape).prod().item() == sh.scale_block_bytes

    # (logical row, even dim) -> byte in the writer's (1, 4, 64, 16) plane.
    assert {
        (0, 0): fp4_byte(sh, 0, 0),
        (1, 0): fp4_byte(sh, 1, 0),
        (63, 30): fp4_byte(sh, 63, 30),
        (0, 32): fp4_byte(sh, 0, 32),
        (17, 64): fp4_byte(sh, 17, 64),
        (63, 126): fp4_byte(sh, 63, 126),
    } == {
        (0, 0): 0,
        (1, 0): 16,
        (63, 30): 1023,
        (0, 32): 1024,
        (17, 64): 2320,
        (63, 126): 4095,
    }

    rows = torch.tensor([0, 16, 1, 63])
    physical_rows = fp4_index_scale_rows(rows, sh.block_rows).tolist()
    assert physical_rows == [0, 1, 4, 63]
    assert [
        fp4_scale_byte(sh, row, chunk)
        for row, chunk in ((0, 0), (16, 0), (1, 0), (63, 3))
    ] == [0, 1, 4, 255]

    # IQ8 reserves one qrows-byte row per token: packed heads first, then
    # their four E8M0s. These boundary bytes make overlap/padding visible.
    assert [
        q_fp4_byte(sh, 0, 0, 0),
        q_fp4_byte(sh, 0, 31, 126),
        q_fp4_scale(sh, 0, 0, 0),
        q_fp4_scale(sh, 0, 31, 3),
        q_fp4_byte(sh, 2, 0, 0),
    ] == [0, 2047, 2048, 2175, 8192]
