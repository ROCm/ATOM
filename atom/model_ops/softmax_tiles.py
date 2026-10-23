"""Tiled, max-subtracted softmax statistics shared by the sampling kernels.

A full-vocab softmax over ``[rows, V]`` is computed in two launches: every
(row, V tile) program reduces its tile to ``(max, sum exp(x - max))``, and one
program per row combines the tiles into the row's max and ``1/Z``. Consumers
then evaluate ``exp(x - max) * inv_z`` for exactly the entries they need.

Both the sampled DSpark draft (`dspark_markov_sample`) and the probability-ratio
verify (`rejection_sampler`) build distributions this way, and the accept rule
compares the two element by element, so they share the arithmetic rather than
keeping two copies in sync by comment.
"""

import triton
import triton.language as tl


@triton.jit
def tile_max_sumexp(x, mask):
    """``(max, sum exp(x - max))`` of one ``[BLOCK]`` or ``[ROWS, BLOCK]`` tile
    along its last axis. Masked lanes must already hold -inf.

    A fully masked lane is -inf, and ``exp(-inf - -inf)`` is NaN, so the sum is
    masked rather than relying on the exponential to zero it.
    """
    tile_max = tl.max(x, axis=-1)
    if len(x.shape) == 1:
        expd = tl.where(mask, tl.exp(x - tile_max), 0.0)
    else:
        expd = tl.where(mask, tl.exp(x - tile_max[:, None]), 0.0)
    return tile_max, tl.sum(expd, axis=-1)


@triton.jit
def softmax_combine_kernel(
    part_max_ptr,  # [num_tiles, num_rows] fp32
    part_sum_ptr,  # [num_tiles, num_rows] fp32
    row_max_ptr,  # [num_rows] fp32 out
    row_inv_z_ptr,  # [num_rows] fp32 out
    num_rows,
    num_tiles,
    BLOCK_TILE: tl.constexpr,
):
    """One row: the online-softmax combine, global max and 1/Z from the
    per-tile pairs, so V is never read a second time just to normalize."""
    row = tl.program_id(0)
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    maxes = tl.load(
        part_max_ptr + offs_tile * num_rows + row, mask=tile_mask, other=float("-inf")
    )
    sums = tl.load(part_sum_ptr + offs_tile * num_rows + row, mask=tile_mask, other=0.0)
    global_max = tl.max(maxes, axis=0)
    rescaled = tl.where(tile_mask, sums * tl.exp(maxes - global_max), 0.0)
    z = tl.sum(rescaled, axis=0)
    # An all--inf row (the fp8-KV warmup forward drafts before the pool is
    # written) leaves Z at 0, and 1/0 would make every probability inf. Answer
    # 0 instead, so every probability of that row is 0.
    tl.store(row_max_ptr + row, global_max)
    tl.store(row_inv_z_ptr + row, tl.where(z > 0.0, 1.0 / z, 0.0))
