"""Tiled, max-subtracted softmax statistics shared by the sampling kernels.

A full-vocab softmax over ``[rows, V]`` is computed in two launches: every
(row, V tile) program reduces its tile to ``(max, sum exp(x - max))``, and one
program per row combines the tiles into the row's max and ``1/Z``. Consumers
then evaluate ``exp(x - max) * inv_z`` for exactly the entries they need.

The probability-ratio verify (`rejection_sampler`) builds the target's p this
way. The sampled DSpark draft (`dspark_markov_sample`) takes only the row max
(and the raw argmax) from the combine: its q is normalized over the entries it
stores, not by a full-vocab 1/Z, so it skips the sum (``WITH_INV_Z=False``).
"""

import triton
import triton.language as tl


@triton.jit
def tile_max_sumexp(x, mask):
    """``(max, sum exp(x - max))`` of one ``[BLOCK]`` or ``[ROWS, BLOCK]`` tile
    along its last axis. Masked lanes must already hold -inf.

    A tile that is -inf throughout (masked lanes, or a vocab range the logits
    themselves mask) has max -inf, and ``exp(-inf - -inf)`` is NaN; subtracting
    0 instead makes its sum the 0 it is, and its max stays -inf.
    """
    tile_max = tl.max(x, axis=-1)
    shift = tl.where(tile_max == float("-inf"), 0.0, tile_max)
    if len(x.shape) == 1:
        expd = tl.where(mask, tl.exp(x - shift), 0.0)
    else:
        expd = tl.where(mask, tl.exp(x - shift[:, None]), 0.0)
    return tile_max, tl.sum(expd, axis=-1)


# The ratio verify launches this eagerly at whatever batch arrives, so the row
# count is not specialized on, as `rejection_sampler._BATCH_COUNTS` explains.
@triton.jit(do_not_specialize=["num_rows"])
def softmax_combine_kernel(
    part_max_ptr,  # [num_tiles, num_rows] fp32
    part_sum_ptr,  # [num_tiles, num_rows] fp32
    row_max_ptr,  # [num_rows] fp32 out
    row_inv_z_ptr,  # [num_rows] fp32 out
    part_amax_val_ptr,  # [num_tiles, num_rows] fp32, WITH_ARGMAX only
    part_amax_idx_ptr,  # [num_tiles, num_rows] int32, WITH_ARGMAX only
    row_argmax_ptr,  # [num_rows] int32 out, WITH_ARGMAX only
    num_rows,
    num_tiles,
    BLOCK_TILE: tl.constexpr,
    WITH_ARGMAX: tl.constexpr,
    WITH_INV_Z: tl.constexpr = True,
):
    """One row: the online-softmax combine, global max and 1/Z from the
    per-tile pairs, so V is never read a second time just to normalize.

    ``WITH_ARGMAX`` also reduces per-tile ``(value, lowest id)`` pairs to the
    row's lowest-id argmax, the pick ``torch.argmax`` and ``topk_select(...,
    tie="low")`` make. Without ``WITH_INV_Z`` only the max (and argmax) is
    produced; ``part_sum_ptr`` and ``row_inv_z_ptr`` are not touched.
    """
    row = tl.program_id(0)
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    maxes = tl.load(
        part_max_ptr + offs_tile * num_rows + row, mask=tile_mask, other=float("-inf")
    )
    global_max = tl.max(maxes, axis=0)
    tl.store(row_max_ptr + row, global_max)
    if WITH_INV_Z:
        sums = tl.load(
            part_sum_ptr + offs_tile * num_rows + row, mask=tile_mask, other=0.0
        )
        # An all--inf tile contributes nothing, and must not contribute the NaN
        # `exp(-inf - -inf)` would give when the whole row is -inf.
        rescaled = tl.where(
            tile_mask & (maxes > float("-inf")), sums * tl.exp(maxes - global_max), 0.0
        )
        z = tl.sum(rescaled, axis=0)
        # An all--inf row (the fp8-KV warmup forward drafts before the pool is
        # written) leaves Z at 0, and 1/0 would make every probability inf.
        # Answer 0 instead, so every probability of that row is 0.
        tl.store(row_inv_z_ptr + row, tl.where(z > 0.0, 1.0 / z, 0.0))
    if WITH_ARGMAX:
        vals = tl.load(
            part_amax_val_ptr + offs_tile * num_rows + row,
            mask=tile_mask,
            other=float("-inf"),
        )
        idxs = tl.load(
            part_amax_idx_ptr + offs_tile * num_rows + row,
            mask=tile_mask,
            other=2147483647,
        )
        best = tl.max(vals, axis=0)
        cand = tl.where((vals == best) & tile_mask, idxs, 2147483647)
        winner = tl.min(cand, axis=0)
        # An all-NaN row matches nothing; answer id 0, as `torch.argmax` would
        # not, but in range, so no consumer can index past the vocab.
        tl.store(row_argmax_ptr + row, tl.where(winner == 2147483647, 0, winner))
