# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stage 1 of the DSpark Markov ops: the bias GEMV over one (V tile, row
tile), the tile's (max, lowest id), and the default launch geometry.

The greedy op (`dspark_markov_argmax`), the sampled one
(`spec_sampling.dspark_markov_sample`) and the probabilistic stage-1 kernels
(`dspark_prob_sample`, `dspark_prob_markov_chain`) run the same GEMV and
differ only in what they keep of it, so the tile and its constants live here,
once: retuning the GEMV retunes all of them. `markov_launch_geometry` is the
default tiling; the fused probabilistic chain starts from it and overrides it
only for larger batches.
"""

from typing import NamedTuple

import triton
import triton.language as tl

from atom.model_ops.tile_argmax import argmax_lowest_id

# Chosen for headroom rather than peak: the stage-1 accumulator is
# [BLOCK_ROW, BLOCK_V] fp32 (64x128 -> 16 VGPR/lane at 8 warps) and one pipeline
# stage stages [BLOCK_ROW, BLOCK_K] + [BLOCK_K, BLOCK_V] bf16 = 24KB of the 64KB
# LDS.
BLOCK_V = 128
BLOCK_K = 64
# MFMA needs M >= 16, so a batch smaller than that runs padded-and-masked
# rather than falling off the kernel. Capped at 64 because the accumulator is
# [BLOCK_ROW, BLOCK_V] fp32 and lives in registers: batches past the cap take
# more row tiles rather than a wider accumulator, which would spill (native
# ATOM defaults max_num_seqs to 512, so this is reachable, not theoretical).
MIN_BLOCK_ROW = 16
MAX_BLOCK_ROW = 64


class MarkovGeometry(NamedTuple):
    """How a stage-1 launch tiles ``num_rows`` x ``vocab_size``: a
    ``(v_tiles, row_tiles)`` grid whose programs each cover ``block_row``
    rows and ``tiles_per_program`` consecutive ``block_v``-column vocab
    tiles. The defaults are the single-tile layout every Markov stage-1
    kernel uses; only the fused probabilistic chain overrides them."""

    block_row: int
    row_tiles: int
    v_tiles: int
    num_warps: int
    block_v: int = BLOCK_V
    tiles_per_program: int = 1
    num_stages: int = 2


def markov_launch_geometry(num_rows: int, vocab_size: int) -> MarkovGeometry:
    """The stage-1 grid ``(v_tiles, row_tiles)`` and its row tile and warps."""
    block_row = min(MAX_BLOCK_ROW, max(MIN_BLOCK_ROW, triton.next_power_of_2(num_rows)))
    return MarkovGeometry(
        block_row,
        triton.cdiv(num_rows, block_row),
        triton.cdiv(vocab_size, BLOCK_V),
        8 if block_row >= 32 else 4,
    )


@triton.jit
def markov_logits_tile(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    embed_out_ptr,
    tile,
    offs_row,
    offs_v,
    row_mask,
    v_mask,
    rank,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    vocab_offset,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STORE_EMBED: tl.constexpr,
    FROM_PACKED: tl.constexpr = False,
):
    """``base + W1[prev] @ W2^T`` on one (row tile, V tile), fp32, -inf on
    padded vocab lanes; with ``STORE_EMBED`` tile 0 also publishes
    ``W1[prev]`` to ``embed_out`` (the greedy op's confidence head reads it).
    With ``FROM_PACKED`` the previous ids are the probabilistic chain's
    packed uint64 keys (`dspark_prob_markov_chain`): the low word is the
    inverted id, and a zero key (empty row) reads id 0.

    The GEMV every Markov stage-1 kernel runs (`dspark_markov_argmax`'s,
    `dspark_markov_sample`'s, and the probabilistic ones through
    `_sampled_tile_argmax`); each keeps only its own tail. ``base``
    holds the tile's columns from 0, W2 is read from vocab id ``vocab_offset``
    on (a TP rank's shard of the LM head; 0 for the whole vocab). The vocab
    and row offsets are int64: `[max_bs, T, V]` logits and a whole W2 pass
    2**31 elements at serving sizes.
    """
    # Strided: the caller's ids are one column of its `[B, T + 1]` block, so
    # assuming unit stride here reads another position's tokens.
    prev = tl.load(prev_ids_ptr + offs_row * stride_prev, mask=row_mask, other=0)
    if FROM_PACKED:
        prev = tl.where(prev == 0, 0, ~prev.to(tl.uint32)).to(tl.int64)

    acc = tl.zeros((BLOCK_ROW, BLOCK_V), dtype=tl.float32)
    for k in range(tl.cdiv(rank, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < rank
        # W1[x_{k-1}] rows, [BLOCK_ROW, BLOCK_K], gathered here rather than
        # handed in: every tile program reads these same values anyway, so
        # the gather rides along in L2 and costs no launch of its own.
        embed = tl.load(
            w1_ptr + prev[:, None].to(tl.int64) * stride_w1_vocab + offs_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # One tile publishes the row for the confidence head; the rest hold
        # the same values and would only write them again.
        # Nested: a constexpr test, then a runtime one.
        if STORE_EMBED:  # noqa: SIM102
            if tile == 0:
                tl.store(
                    embed_out_ptr
                    + offs_row[:, None] * stride_embed_row
                    + offs_k[None, :],
                    embed,
                    mask=row_mask[:, None] & k_mask[None, :],
                )
        # W2^T tile, [BLOCK_K, BLOCK_V]. W2 is stored [V, r] so the rank axis
        # is the contiguous one: each vocab column is one 2*BLOCK_K-byte run.
        w2 = tl.load(
            w2_ptr
            + (vocab_offset + offs_v[None, :]).to(tl.int64) * stride_w2_vocab
            + offs_k[:, None],
            mask=v_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        # Zeros from the rank mask contribute exact 0.0 terms, so padding the
        # K tile does not perturb the sum.
        acc = tl.dot(embed, w2, acc)

    base = tl.load(
        base_ptr + offs_row[:, None].to(tl.int64) * stride_base_row + offs_v[None, :],
        mask=row_mask[:, None] & v_mask[None, :],
        other=0.0,
    )
    return tl.where(v_mask[None, :], base.to(tl.float32) + acc, float("-inf"))


@triton.jit
def publish_tile_argmax(
    vals,
    tile,
    offs_row,
    offs_v,
    row_mask,
    v_mask,
    num_rows,
    vocab_offset,
    part_val_ptr,
    part_idx_ptr,
):
    """Publish one (V tile, row tile)'s (max, lowest id) of ``vals``, the ids
    global (``vocab_offset`` on), for stage 2 to combine."""
    # `v_mask` also covers the all--inf row: without it a padded lane, which
    # is -inf too, could win the id.
    tile_max, tile_idx = argmax_lowest_id(
        vals, offs_v[None, :], v_mask[None, :], 2147483647
    )
    # Global id. A tile with no hit (all NaN) keeps `argmax_of_tiles`' one
    # sentinel, unoffset, which it answers as id 0 for an all-NaN row.
    tile_idx = tl.where(tile_idx != 2147483647, tile_idx + vocab_offset, 2147483647)
    tl.store(part_val_ptr + tile * num_rows + offs_row, tile_max, mask=row_mask)
    tl.store(part_idx_ptr + tile * num_rows + offs_row, tile_idx, mask=row_mask)
