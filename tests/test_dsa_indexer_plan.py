# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dsa_indexer.plan``: a rank's keys of a row, under DCP (interleave 1) and
replicated (128-key runs), against counting every position's owner."""

import pytest

from atom.mono.dsa_indexer.plan import (
    REPLICA_INTERLEAVE,
    IndexerShape,
    fp4_byte,
    fp4_scale_byte,
    key_pos,
    local_len,
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


def _shape(kv_dtype="fp4", block_rows=64, rope_neox=False):
    return IndexerShape(
        heads=32,
        head_dim=128,
        topk=2048,
        q_lora=2048,
        wq_dtype="bf16",
        kv_dtype=kv_dtype,
        block_rows=block_rows,
        rope_dim=64,
        rope_neox=rope_neox,
    )


def test_an_fp4_cache_is_served_only_in_the_writers_layout():
    assert unsupported(_shape()) == ""
    assert "64 rows" in unsupported(_shape(block_rows=16))
    assert "interleaved RoPE" in unsupported(_shape(rope_neox=True))


def test_fp4_bytes_are_where_the_scorers_packer_puts_them():
    """Every key byte and E8M0 of a block, at the offset aiter's FP4 scorer
    packs it to (the cache writer's layout, ``fp4_index_block_shapes``)."""
    torch = pytest.importorskip("torch")
    rowgroup = pytest.importorskip(
        "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup"
    )
    sh = _shape()
    rows, half, chunks = sh.block_rows, sh.head_dim // 2, sh.head_dim // 32
    kv = torch.arange(rows * half, dtype=torch.int32).view(rows, half)
    ks = torch.arange(rows * chunks, dtype=torch.int32).view(rows, chunks)
    data, scale = (t.flatten() for t in rowgroup.pack_kv_cache(kv, ks, page_size=rows))
    for r in range(rows):
        for j in range(half):
            assert data[fp4_byte(sh, r, 2 * j)] == kv[r, j], (r, j)
        for ch in range(chunks):
            assert scale[fp4_scale_byte(sh, r, ch)] == ks[r, ch], (r, ch)
