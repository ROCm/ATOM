# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The lowest-id tile argmax every argmax and race kernel shares: the greedy
DSpark draft (`dspark_markov_argmax`) and the lossless sampling kernels
(`spec_sampling`), so their picks agree bit for bit."""

import triton
import triton.language as tl

# `argmax_lowest_id`'s answer for a tile with no hit: past every real id.
NO_HIT = tl.constexpr(2147483647)


@triton.jit
def argmax_lowest_id(key, ids, mask, sentinel):
    """``(max, lowest id holding it)`` of a ``[BLOCK]`` or ``[ROWS, BLOCK]``
    tile along its last axis, over the lanes ``mask`` keeps.

    The tie rule ``torch.argmax`` and ``topk_select(..., tie="low")`` make. A
    tile no lane of which equals its max (masked out throughout, or all NaN,
    since NaN equals nothing) answers ``sentinel``; callers pick one past
    every real id and map it back into range themselves.
    """
    best = tl.max(key, axis=-1)
    if len(key.shape) == 1:
        hit = key == best
    else:
        hit = key == best[:, None]
    return best, tl.min(tl.where(hit & mask, ids, sentinel), axis=-1)


@triton.jit
def argmax_of_tiles(part_val_ptr, part_idx_ptr, row, num_rows, num_tiles, BLOCK_TILE):
    """One row's ``(max, lowest id)`` from its tiles' pairs, stored
    ``[num_tiles, num_rows]``, picking among tiles that tie as
    `argmax_lowest_id` does within one. Tiles with no hit publish the
    `NO_HIT` sentinel, so a row whose every tile is all NaN answers id
    0, as ``torch.argmax`` would."""
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    vals = tl.load(
        part_val_ptr + offs_tile * num_rows + row, mask=tile_mask, other=float("-inf")
    )
    idxs = tl.load(
        part_idx_ptr + offs_tile * num_rows + row, mask=tile_mask, other=NO_HIT
    )
    best, winner = argmax_lowest_id(vals, idxs, tile_mask, NO_HIT)
    return best, tl.where(winner == NO_HIT, 0, winner)
