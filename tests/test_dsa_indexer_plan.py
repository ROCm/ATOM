# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dsa_indexer.plan``: a rank's keys of a row, under DCP (interleave 1) and
replicated (128-key runs), against counting every position's owner."""

import pytest
import torch

from atom.model_ops.sparse_indexer_fp4 import (
    FP4_KV_BLOCK_SIZE,
    FP4_QUANT_BLOCK_SIZE,
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


def _pack_block(kv, ks, rows):
    """aiter's ``pack_kv_cache`` for a 64-row page (the cache writer's
    layout): a block's bytes behind (k tile, chunk), its E8M0s four tiles'
    keys 16 apart a dword."""
    k_tiles = kv.shape[1] * 2 // 128
    data = kv.reshape(rows, k_tiles, 4, 16).permute(1, 2, 0, 3)
    scale = (
        ks.reshape(rows, k_tiles, 4)
        .permute(1, 2, 0)
        .reshape(k_tiles, 4, rows // 16, 16)
        .transpose(-1, -2)
    )
    return data.flatten(), scale.flatten()


def test_fp4_bytes_are_where_the_cache_writer_puts_them():
    sh = _shape()
    rows, half = sh.block_rows, sh.head_dim // 2
    chunks = sh.head_dim // FP4_QUANT_BLOCK_SIZE
    kv = torch.arange(rows * half, dtype=torch.int32).view(rows, half)
    ks = torch.arange(rows * chunks, dtype=torch.int32).view(rows, chunks)
    data, scale = _pack_block(kv, ks, rows)
    assert (data.numel(), scale.numel()) == (sh.block_bytes, sh.scale_block_bytes)
    for r in range(rows):
        for j in range(half):
            assert data[fp4_byte(sh, r, 2 * j)] == kv[r, j], (r, j)
        for ch in range(chunks):
            assert scale[fp4_scale_byte(sh, r, ch)] == ks[r, ch], (r, ch)


def test_a_tokens_fp4_query_fills_its_row_in_the_scorers_loads():
    """IQ8: a token's E2M1 bytes and E8M0s take its row once each; a head's
    bytes and the token's scales start 16 B aligned (the 16 B loads that stage
    them), a head's four scales one dword."""
    sh = _shape()
    chunks = sh.head_dim // FP4_QUANT_BLOCK_SIZE
    for t in range(3):
        at = [
            q_fp4_byte(sh, t, h, col)
            for h in range(sh.heads)
            for col in range(0, sh.head_dim, 2)
        ]
        at += [
            q_fp4_scale(sh, t, h, ch) for h in range(sh.heads) for ch in range(chunks)
        ]
        assert sorted(at) == list(range(t * sh.qrows, t * sh.qrows + len(at)))
        assert len(at) <= sh.qrows
        assert q_fp4_scale(sh, t, 0, 0) % 16 == 0
        for h in range(sh.heads):
            assert q_fp4_byte(sh, t, h, 0) % 16 == 0
            assert q_fp4_scale(sh, t, h, 0) % 4 == 0
            assert [q_fp4_scale(sh, t, h, ch) for ch in range(chunks)] == [
                q_fp4_scale(sh, t, h, 0) + ch for ch in range(chunks)
            ]
