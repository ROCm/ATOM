# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What one DeepSeek-V4 KV row and one indexer block hold.

The sibling of `v4_pool_geometry`, which says where rows go; this says what a
row is. Declaration only -- which of these a build wants is policy and stays
with the backend (the gfx fallback off fp8, `--index_cache_dtype`, whether the
GPU has an FP4 indexer), so nothing here has a branch of its own.

`MAIN_KV_NOPE` and `MAIN_KV_ROPE` are the `semantic_role` a PD consumer matches
its producer's regions on: a wire format between two processes that may be
running different builds, not free to rename.
"""

from __future__ import annotations

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import EntryField

MAIN_KV_NOPE = "dsv4.main_kv.nope"
MAIN_KV_ROPE = "dsv4.main_kv.rope"

# Rows per block that `deepgemm_fp8_paged_mqa_logits` requires to stay in its
# preshuffled layout, which is the only layout it computes correctly -- with
# `Preshuffle=False` it disagrees with the flat `fp8_mqa_logits` kernel by ~100%
# at every block size, and aiter's assert guards only the preshuffle side. A
# cache that kernel pages over holds a multiple of this many rows per block, or
# exactly 8 -- the one shorter page it assembles a tile from.
#
# Here rather than beside one of its enforcers, because there are several: a
# block size (`atom.config`), a pooled row count (Kimi/GLM) and a CSA2 index
# plane all have to agree with it. Only CSA2 takes the short page.
MQA_LOGITS_PRESHUFFLE_ROWS = 16

# Internal: FP8 keeps both regions in one pool and so has fewer PD roles than
# it has regions, which is why those are spelled out where the pools are.
CSA_INDEXER_DATA = "csa_indexer_data"
CSA_INDEXER_SCALE = "csa_indexer_scale"

FP4_GFX950_PRESHUFFLE = "fp4-gfx950-preshuffle"
FP4_GFX1250_NATURAL = "fp4-gfx1250-natural"


def fp4_indexer_layout_for_arch(arch: str) -> str:
    """Return the on-wire FP4 indexer layout for a GPU architecture."""
    if arch == "gfx1250":
        return FP4_GFX1250_NATURAL
    return FP4_GFX950_PRESHUFFLE


def main_kv_plane_fields(
    head_dim: int,
    main_dtype: torch.dtype,
    rope: tuple[int, torch.dtype] | None = None,
) -> list[EntryField]:
    """One row of each plane the main KV pool has, in declared order.

    How many there are is the single answer the carve, the state split, the
    checkpoint copy and the PD registration all take from here. `rope` is the
    second plane's `(width, dtype)`, or None on a build that keeps RoPE inline
    in the NoPE row -- a pair so that half a plane cannot be described.
    """
    fields = [EntryField(MAIN_KV_NOPE, 1, (head_dim,), main_dtype)]
    if rope is not None:
        rope_head_dim, rope_dtype = rope
        fields.append(EntryField(MAIN_KV_ROPE, 1, (rope_head_dim,), rope_dtype))
    return fields


def fp8_indexer_block_fields(
    rows: int, index_head_dim: int, data_dtype: torch.dtype
) -> list[EntryField]:
    """`[rows*index_head_dim data][rows*4 fp32 scale]`, one layer's block.

    Two regions, NOT interleaved per row, which is how all three consumers
    address it: `fused_compress.py` (write), `cache_kernels.cu:1638/1651`, and
    `pa_mqa_logits.py:493-500`. Both live in one pool, so that pool's row width
    is the block over its rows.
    """
    return [
        EntryField(CSA_INDEXER_DATA, 1, (rows, index_head_dim), data_dtype),
        EntryField(CSA_INDEXER_SCALE, 1, (rows,), torch.float32),
    ]


def fp4_indexer_block_fields(
    rows: int,
    index_head_dim: int,
    layout: str = FP4_GFX950_PRESHUFFLE,
) -> list[EntryField]:
    """Packed E2M1 plus one e8m0 byte per group of 32, one layer's block.

    gfx950 keeps the legacy `pa_mqa_logits_fp4` preshuffle. gfx1250 OPUS reads
    natural rows: 64 packed E2M1 bytes and four E8M0 bytes for D=128. One pool
    per region here, so a region's shape is a pool's shape after the layer and
    block axes.
    """
    if index_head_dim % 128 != 0:
        raise ValueError(
            f"FP4 index_head_dim must be a multiple of 128, got {index_head_dim}"
        )
    k_tiles = index_head_dim // 128
    if layout == FP4_GFX1250_NATURAL:
        return [
            EntryField(CSA_INDEXER_DATA, 1, (rows, index_head_dim // 2), torch.uint8),
            EntryField(CSA_INDEXER_SCALE, 1, (rows, index_head_dim // 32), torch.uint8),
        ]
    if layout != FP4_GFX950_PRESHUFFLE:
        raise ValueError(f"unknown FP4 indexer layout {layout!r}")
    return [
        EntryField(CSA_INDEXER_DATA, 1, (k_tiles, 4, rows, 16), torch.uint8),
        EntryField(CSA_INDEXER_SCALE, 1, (k_tiles, 4, rows), torch.uint8),
    ]


def indexer_block_regions(fields: list[EntryField]) -> tuple[dict[str, int], int]:
    """Where each region starts inside one layer's block, and the block's size.

    Packed, with nothing between regions -- deliberately not the aligned walk
    `entry_arena.field_extents` does. Every consumer recomputes a region's
    start from `rows` and the head dim rather than reading one, so a gap would
    go unnoticed: a scale view would begin one gap early and read
    valid-looking numbers rather than crash.
    """
    offsets: dict[str, int] = {}
    total = 0
    for field in fields:
        offsets[field.name] = total
        total += field.bytes_per_entry
    return offsets, total
