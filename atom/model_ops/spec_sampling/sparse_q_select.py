# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stages 2-4 of the sampled DSpark draft (`dspark_markov_sample`): choose the
top of each row's q from the dense score scratch, store it (`SparseQStore`),
and draw the row's token from exactly what was stored.

A deterministic row (greedy, non-finite, degenerate) stores the delta on its
token; a negative row base writes nothing and drafts the logits' argmax.
The cut is by level, the gap below the row max in 1/8 nats, not by exact value:
the boundary level's lowest ids fill n = min(K, top-k, ids within 19 nats).
That only moves acceptance: the draft samples from what is stored.
Launches: histogram (which first reduces the row max from stage 1's tile
maxima), compact (which first makes the row's cut from the histogram tables),
finalize (normalize, the exponential race over the stored entries, and the
row's raw argmax from stage 1's tile pairs). Past 16 rows, whole vocab or
TP, the softmax combine reduces the row max and raw argmax and a cut launch
makes the cut, as their own launches. Each folded reduction reads the
same per-tile inputs in the same fixed order a separate launch did, so the
result does not change with the fold. Both histogram and compact level through
`_q_level`.
Fixed-order reductions, no atomics: TP ranks must store and draw alike.
Philox noise in-kernel, at a full-vocab race's counter per (position, row, id).
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.sampling_scratch import Region
from atom.model_ops.spec_sampling.softmax_tiles import (
    max_of_tiles,
    race_key,
    softmax_combine_kernel,
)
from atom.model_ops.tile_argmax import argmax_lowest_id, argmax_of_tiles

# Histogram and compact tile V identically (the cut hands each tile a slot range).
_Q_BLOCK_V = 4096
_Q_BINS = 256
# Level = 1 + floor(gap * 8) for a gap in (0, 19] nats: at most 153, so
# everything fits below the last bin, which collects the rest (and NaN/inf).
_Q_LEVELS_PER_NAT = 8.0
_Q_MAX_GAP = 19.0
# Stored entries one finalize iteration covers.
_Q_FINAL_BLOCK = 1024
# Up to this many rows (whole vocab or TP) the combine and the cut fold into
# the histogram, the compaction and finalize.
_FOLD_ROWS = 16


@triton.jit
def _q_level(
    s,
    row_max,
    valid,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
):
    """Each score's level: 0 at the row max, ``1 + floor(gap * LEVELS_PER_NAT)``
    for a gap in (0, MAX_GAP] nats, and the last bin for everything else --
    farther down, masked, NaN, or any score of a row whose max is not finite
    (every gap is then NaN or inf). The one definition the histogram and the
    compaction share; they must agree lane for lane.
    """
    gap = row_max - s
    # Truncation is floor here: the gap is >= 0 wherever the level is kept.
    level = 1 + (gap * LEVELS_PER_NAT).to(tl.int32)
    level = tl.where(gap == 0.0, 0, level)
    keep = valid & (gap >= 0.0) & (gap <= MAX_GAP)
    return tl.where(keep, level, NUM_BINS - 1)


@triton.jit
def _dspark_q_histogram(
    s_ptr,
    part_max_ptr,
    part_stride,
    num_parts,
    row_max_ptr,
    q_row_base_ptr,
    temperatures_ptr,
    greedy_temperature,
    hist_ptr,
    vocab_size,
    stride_s_row,
    hist_row_stride,
    hist_tile_stride,
    BLOCK_PARTS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
    REDUCE_ROW_MAX: tl.constexpr,
):
    """One (row, V tile): how many of the tile's ids sit at each level.

    ``REDUCE_ROW_MAX``: the row max is reduced here from stage 1's
    ``num_parts`` tile maxima (the whole row's, ``part_stride`` apart), and
    tile 0 publishes it for the compaction; otherwise it is read from
    ``row_max_ptr``. Its table goes to `hist_ptr + (row * hist_row_stride +
    tile * hist_tile_stride) * NUM_BINS`: row-major `[B, tiles, bins]` for the
    whole vocab, tile-major `[tiles, B, bins]` for a TP shard, whose tables
    the ranks then gather (`select_and_draw_q_tp`).
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    # Rows that store nothing stop here, and in the cut and the compaction too.
    if (tl.load(q_row_base_ptr + row) < 0) | (
        tl.load(temperatures_ptr + row) <= greedy_temperature
    ):
        return
    if REDUCE_ROW_MAX:
        _, row_max = max_of_tiles(
            part_max_ptr, row, part_stride, num_parts, BLOCK_PARTS
        )
        if tile == 0:
            tl.store(row_max_ptr + row, row_max)
    else:
        row_max = tl.load(row_max_ptr + row)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < vocab_size
    s = tl.load(
        s_ptr + row.to(tl.int64) * stride_s_row + offs_v,
        mask=v_mask,
        other=float("-inf"),
    )
    level = _q_level(s, row_max, v_mask, NUM_BINS, LEVELS_PER_NAT, MAX_GAP)
    counts = tl.histogram(level, NUM_BINS)
    tl.store(
        hist_ptr
        + (row * hist_row_stride + tile * hist_tile_stride) * NUM_BINS
        + tl.arange(0, NUM_BINS),
        counts,
    )


@triton.jit
def _row_cut(
    hist_ptr,
    row,
    top_k_ptr,
    top_p_ptr,
    num_tiles,
    hist_row_stride,
    hist_tile_stride,
    store_width,
    BLOCK_TILE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
):
    """One row's cut from its ``num_tiles`` histogram tables, read in id order:
    ``(n, boundary level, each tile's quota, each tile's first slot)``.

    n = min(store width, the request's top-k, ids within reach of the max),
    then the request's top-p at level granularity. The boundary level is the
    lowest one whose cumulative count reaches n; its ids are taken lowest id
    first, so each tile's share of them (its quota) is what the tiles before
    it leave over, and each tile's first output slot is the running total of
    the tiles before it. Exactly n ids are then stored.
    """
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    bins = tl.arange(0, NUM_BINS)
    table = tl.load(
        hist_ptr
        + (row * hist_row_stride + offs_tile[:, None] * hist_tile_stride) * NUM_BINS
        + bins[None, :],
        mask=tile_mask[:, None],
        other=0,
    )
    # The last bin holds what can never be stored.
    counts = tl.where(bins < NUM_BINS - 1, tl.sum(table, axis=0), 0)
    cum = tl.cumsum(counts, axis=0)
    top_k = tl.load(top_k_ptr + row)
    limit = tl.where(top_k > 0, tl.minimum(top_k, store_width), store_width)
    n = tl.minimum(limit, tl.sum(counts, axis=0))
    # Top-p over the unfiltered mass, then the smaller of the two cuts: the
    # rule `Sampler`'s aiter kernels apply (an intersection, not top-p over
    # the top-k survivors). Each level's mass is approximated by its count at
    # the level's mid gap; every level the threshold has not been reached
    # before is kept.
    mid_gap = tl.where(bins == 0, 0.0, (bins.to(tl.float32) - 0.5) / LEVELS_PER_NAT)
    mass = counts.to(tl.float32) * tl.exp(-mid_gap)
    mass_before = tl.cumsum(mass, axis=0) - mass
    top_p = tl.load(top_p_ptr + row)
    nucleus = tl.sum(
        tl.where(mass_before < top_p * tl.sum(mass, axis=0), counts, 0), axis=0
    )
    n = tl.where(top_p < 1.0, tl.minimum(n, nucleus), n)

    cut = tl.sum((cum < n).to(tl.int32), axis=0)
    quota = n - tl.sum(tl.where(bins < cut, counts, 0), axis=0)
    tile_below = tl.sum(tl.where(bins[None, :] < cut, table, 0), axis=1)
    # The boundary level's column, read back rather than reduced out of the
    # table.
    tile_edge = tl.load(
        hist_ptr
        + (row * hist_row_stride + offs_tile * hist_tile_stride) * NUM_BINS
        + cut,
        mask=tile_mask,
        other=0,
    )
    edge_before = tl.cumsum(tile_edge, axis=0) - tile_edge
    tile_quota = tl.minimum(tile_edge, tl.maximum(quota - edge_before, 0))
    tile_n = tile_below + tile_quota
    tile_off = tl.cumsum(tile_n, axis=0) - tile_n
    return n, cut, tile_quota, tile_off


@triton.jit
def _dspark_q_cut(
    hist_ptr,
    q_row_base_ptr,
    temperatures_ptr,
    greedy_temperature,
    top_k_ptr,
    top_p_ptr,
    tile_off_ptr,
    tile_quota_ptr,
    row_cut_ptr,
    row_n_ptr,
    num_tiles,
    hist_row_stride,
    hist_tile_stride,
    store_width,
    BLOCK_TILE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
):
    """One row: `_row_cut`, published for the compaction and finalize."""
    row = tl.program_id(0)
    if (tl.load(q_row_base_ptr + row) < 0) | (
        tl.load(temperatures_ptr + row) <= greedy_temperature
    ):
        return
    n, cut, tile_quota, tile_off = _row_cut(
        hist_ptr,
        row,
        top_k_ptr,
        top_p_ptr,
        num_tiles,
        hist_row_stride,
        hist_tile_stride,
        store_width,
        BLOCK_TILE,
        NUM_BINS,
        LEVELS_PER_NAT,
    )
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    tl.store(tile_off_ptr + row * num_tiles + offs_tile, tile_off, mask=tile_mask)
    tl.store(tile_quota_ptr + row * num_tiles + offs_tile, tile_quota, mask=tile_mask)
    tl.store(row_cut_ptr + row, cut)
    tl.store(row_n_ptr + row, n)


@triton.jit
def _dspark_q_compact(
    s_ptr,
    row_max_ptr,
    q_row_base_ptr,
    temperatures_ptr,
    greedy_temperature,
    top_k_ptr,
    top_p_ptr,
    hist_ptr,
    hist_row_stride,
    hist_tile_stride,
    tile_off_ptr,
    tile_quota_ptr,
    row_cut_ptr,
    row_n_ptr,
    q_ids_ptr,
    q_probs_ptr,
    vocab_size,
    stride_s_row,
    num_tiles,
    store_width,
    position,
    tile_base,
    id_base,
    BLOCK_TILE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
    MAKE_CUT: tl.constexpr,
    DEST_BY_ROW: tl.constexpr,
):
    """One (row, V tile): store the tile's chosen ids and their unnormalized
    weights ``exp(s - max)`` at its slot range, in id order.

    ``MAKE_CUT``: every tile of a row makes the row's `_row_cut` itself, and
    tile 0 publishes n, and under TP every tile's first slot for finalize
    (`_staged_slots`); otherwise the
    cut is `_dspark_q_cut`'s. ``s`` holds ``vocab_size`` ids from ``id_base``,
    which are the cut's tiles from ``tile_base`` (both 0 for the whole vocab;
    a TP shard's otherwise, `select_and_draw_q_tp`). ``DEST_BY_ROW`` writes
    batch row ``row``'s slots to row ``row`` of a ``[B, store_width]`` staging
    pair rather than to its store row.
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    q_base = tl.load(q_row_base_ptr + row)
    if (q_base < 0) | (tl.load(temperatures_ptr + row) <= greedy_temperature):
        return
    my_tile = tile_base + tile
    if MAKE_CUT:
        n, cut, tile_quota, tile_off = _row_cut(
            hist_ptr,
            row,
            top_k_ptr,
            top_p_ptr,
            num_tiles,
            hist_row_stride,
            hist_tile_stride,
            store_width,
            BLOCK_TILE,
            NUM_BINS,
            LEVELS_PER_NAT,
        )
        offs_tile = tl.arange(0, BLOCK_TILE)
        if tile == 0:
            tl.store(row_n_ptr + row, n)
            if DEST_BY_ROW:
                # Only a TP finalize reads the slot ranges (`_staged_slots`).
                tl.store(
                    tile_off_ptr + row * num_tiles + offs_tile,
                    tile_off,
                    mask=offs_tile < num_tiles,
                )
        quota = tl.sum(tl.where(offs_tile == my_tile, tile_quota, 0), axis=0)
        first = tl.sum(tl.where(offs_tile == my_tile, tile_off, 0), axis=0)
    else:
        cut = tl.load(row_cut_ptr + row)
        quota = tl.load(tile_quota_ptr + row * num_tiles + my_tile)
        first = tl.load(tile_off_ptr + row * num_tiles + my_tile)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < vocab_size
    s = tl.load(
        s_ptr + row.to(tl.int64) * stride_s_row + offs_v,
        mask=v_mask,
        other=float("-inf"),
    )
    row_max = tl.load(row_max_ptr + row)
    level = _q_level(s, row_max, v_mask, NUM_BINS, LEVELS_PER_NAT, MAX_GAP)
    edge = level == cut
    take = (level < cut) | (edge & (tl.cumsum(edge.to(tl.int32), axis=0) <= quota))
    slot = first + (tl.cumsum(take.to(tl.int32), axis=0) - 1)
    if DEST_BY_ROW:
        dest = row.to(tl.int64) * store_width + slot
    else:
        dest = (q_base + position).to(tl.int64) * store_width + slot
    tl.store(q_ids_ptr + dest, id_base + offs_v, mask=take)
    tl.store(q_probs_ptr + dest, tl.exp(s - row_max), mask=take)


@triton.jit
def _staged_slots(
    tile_off_ptr,
    row,
    slots,
    num_rows,
    num_tiles,
    rank_tiles,
    store_width,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Where in a gathered ``[world, 2, B, K]`` staging pair each stored slot
    of ``row`` sits: the rank whose tiles the cut gave it (the ranks' slot
    ranges follow one another in rank order)."""
    owner = tl.zeros((BLOCK,), dtype=tl.int32)
    for r in tl.static_range(1, WORLD):
        first = tl.load(tile_off_ptr + row * num_tiles + r * rank_tiles)
        owner += (slots >= first).to(tl.int32)
    return (owner.to(tl.int64) * 2 * num_rows + row) * store_width + slots


@triton.jit
def _dspark_q_finalize(
    q_ids_ptr,
    q_probs_ptr,
    q_meta_ptr,
    staged_ptr,
    tile_off_ptr,
    q_row_base_ptr,
    temperatures_ptr,
    greedy_temperature,
    part_val_ptr,
    part_idx_ptr,
    part_stride,
    num_parts,
    row_argmax_ptr,
    row_n_ptr,
    seed_ptr,
    out_ptr,
    num_rows,
    num_tiles,
    rank_tiles,
    vocab_size,
    store_width,
    position,
    noise_offset,
    stride_out,
    FROM_STAGING: tl.constexpr,
    ARGMAX_FROM_PARTS: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK_PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One row: normalize the stored weights into q, draw x from them, record
    x's slot.

    ``probs.div_(noise).argmax()`` over the stored entries, lowest id on a tie.
    A greedy row, or one with nothing stored or no positive race key, drafts
    its logits' lowest-id raw argmax (with ``ARGMAX_FROM_PARTS``, up to 16
    rows, reduced here from stage 1's tile pairs; past that, the combine's
    ``row_argmax``)
    and, if it has a destination, stores the exact delta on it: the id is not a
    draw from any softmax, and the accept rule must see the distribution it
    actually came from.

    ``FROM_STAGING`` (a TP shard): the entries are read from the gathered
    ``[world, 2, B, K]`` staging pair, each slot from the rank that owns it, and
    written to the store as they are normalized; the sum and the race see the
    same values, in the same slot order, as the whole-vocab op's.
    """
    row = tl.program_id(0)
    q_base = tl.load(q_row_base_ptr + row)
    has_q = q_base >= 0
    sampled = has_q & (tl.load(temperatures_ptr + row) > greedy_temperature)
    n = tl.where(sampled, tl.load(row_n_ptr + row), 0)
    base = (tl.where(has_q, q_base, 0) + position).to(tl.int64) * store_width
    offs = tl.arange(0, BLOCK)

    # One fixed-order sum, so every rank normalizes by the same number.
    total = 0.0
    for start in range(0, n, BLOCK):
        slots = start + offs
        mask = slots < n
        if FROM_STAGING:
            src = _staged_slots(
                tile_off_ptr,
                row,
                slots,
                num_rows,
                num_tiles,
                rank_tiles,
                store_width,
                WORLD,
                BLOCK,
            )
            bits = tl.load(
                staged_ptr + src + num_rows * store_width, mask=mask, other=0
            )
            weights = bits.to(tl.float32, bitcast=True)
        else:
            weights = tl.load(q_probs_ptr + base + slots, mask=mask, other=0.0)
        total += tl.sum(weights, axis=0)
    inv_total = 1.0 / total

    # Philox, keyed on a device scalar every rank fills with the same SPMD step
    # counter, counter = the vocab id's flat index: a given (step, position,
    # row, id) draws one uniform, the same one on every rank.
    seed = tl.load(seed_ptr)
    noise_base = noise_offset + row.to(tl.int64) * vocab_size
    best = 0.0
    best_id = 2147483647
    best_slot = 0
    for start in range(0, n, BLOCK):
        slots = start + offs
        mask = slots < n
        if FROM_STAGING:
            src = _staged_slots(
                tile_off_ptr,
                row,
                slots,
                num_rows,
                num_tiles,
                rank_tiles,
                store_width,
                WORLD,
                BLOCK,
            )
            ids = tl.load(staged_ptr + src, mask=mask, other=0)
            bits = tl.load(
                staged_ptr + src + num_rows * store_width, mask=mask, other=0
            )
            probs = bits.to(tl.float32, bitcast=True) * inv_total
            tl.store(q_ids_ptr + base + slots, ids, mask=mask)
        else:
            probs = (
                tl.load(q_probs_ptr + base + slots, mask=mask, other=0.0) * inv_total
            )
            ids = tl.load(q_ids_ptr + base + slots, mask=mask, other=0)
        tl.store(q_probs_ptr + base + slots, probs, mask=mask)
        # The verify's race and tie rule (`softmax_tiles`). Stored ids ascend
        # with slot (compaction order), so a later chunk wins only on a
        # strictly larger key, as the lowest-id tie rule requires.
        key = race_key(probs, seed, noise_base, ids, mask)
        chunk_best, chunk_id = argmax_lowest_id(key, ids, mask, 2147483647)
        chunk_slot = tl.min(
            tl.where(mask & (ids == chunk_id), slots, 2147483647), axis=0
        )
        better = (chunk_best > best) | ((chunk_best == best) & (chunk_id < best_id))
        best = tl.where(better, chunk_best, best)
        best_id = tl.where(better, chunk_id, best_id)
        best_slot = tl.where(better, chunk_slot, best_slot)

    drawn = best > 0.0
    if ARGMAX_FROM_PARTS:
        _, row_argmax = argmax_of_tiles(
            part_val_ptr, part_idx_ptr, row, part_stride, num_parts, BLOCK_PARTS
        )
    else:
        row_argmax = tl.load(row_argmax_ptr + row)
    x = tl.where(drawn, best_id, row_argmax)
    tl.store(q_ids_ptr + base, x, mask=has_q & ~drawn)
    tl.store(q_probs_ptr + base, 1.0, mask=has_q & ~drawn)
    tl.store(q_meta_ptr + (q_base + position).to(tl.int64) * 2, n, mask=has_q & drawn)
    tl.store(q_meta_ptr + (q_base + position).to(tl.int64) * 2, 1, mask=has_q & ~drawn)
    tl.store(
        q_meta_ptr + (q_base + position).to(tl.int64) * 2 + 1,
        tl.where(drawn, best_slot, 0),
        mask=has_q,
    )
    tl.store(out_ptr + row * stride_out, x.to(out_ptr.dtype.element_ty))


class QSelectScratch(NamedTuple):
    """`select_and_draw_q`'s work buffers, all int32."""

    hist: torch.Tensor  # [B, tiles, bins]
    tile_off: torch.Tensor  # [B, tiles]
    tile_quota: torch.Tensor  # [B, tiles]
    row_cut: torch.Tensor  # [B]
    row_n: torch.Tensor  # [B]
    row_argmax: torch.Tensor  # [B]


def q_select_scratch(num_rows: int, vocab_size: int) -> QSelectScratch:
    """The regions `select_and_draw_q` needs for ``num_rows`` rows."""
    q_tiles = triton.cdiv(vocab_size, _Q_BLOCK_V)
    i32 = torch.int32
    return QSelectScratch(
        Region((num_rows, q_tiles, _Q_BINS), i32),
        Region((num_rows, q_tiles), i32),
        Region((num_rows, q_tiles), i32),
        Region((num_rows,), i32),
        Region((num_rows,), i32),
        Region((num_rows,), i32),
    )


_LEVELS = {
    "NUM_BINS": _Q_BINS,
    "LEVELS_PER_NAT": _Q_LEVELS_PER_NAT,
    "MAX_GAP": _Q_MAX_GAP,
}


def _combine_row_max_and_argmax(
    part_max: torch.Tensor,
    part_amax_val: torch.Tensor,
    part_amax_idx: torch.Tensor,
    row_max: torch.Tensor,
    row_argmax: torch.Tensor,
) -> None:
    """Each row's max score and lowest-id raw argmax from stage 1's
    `[tiles, B]` partials, in one launch: the unfolded path."""
    num_rows = row_max.shape[0]
    softmax_combine_kernel[(num_rows,)](
        part_max,
        None,
        row_max,
        None,
        part_amax_val,
        part_amax_idx,
        row_argmax,
        part_max.stride(0),
        part_max.shape[0],
        BLOCK_TILE=triton.next_power_of_2(part_max.shape[0]),
        WITH_ARGMAX=True,
        WITH_INV_Z=False,
        num_warps=4,
        num_stages=1,
    )


def select_and_draw_q(
    s_scratch: torch.Tensor,
    part_max: torch.Tensor,
    part_amax_val: torch.Tensor,
    part_amax_idx: torch.Tensor,
    row_max: torch.Tensor,
    q_row_base: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    out: torch.Tensor,
    q_ids: torch.Tensor,
    q_probs: torch.Tensor,
    q_meta: torch.Tensor,
    scratch: QSelectScratch,
    position: int,
    noise_offset: int,
) -> None:
    """Stages 2-4 over the ``[B, V]`` scores in ``s_scratch``: store each
    row's q at ``q_row_base[b] + position`` and write its token to ``out``.

    ``part_max`` / ``part_amax_val`` / ``part_amax_idx`` are stage 1's
    ``[tiles, B]`` partials (each tile's max score, and its raw max and
    lowest id), which the histogram and finalize reduce up to `_FOLD_ROWS`
    rows and the softmax combine past that; ``row_max`` is the row max's
    scratch; ``scratch`` is carved from `q_select_scratch`.
    Arguments are as `dspark_markov_sample` documents them, which checks them.
    """
    num_rows = q_row_base.shape[0]
    vocab_size = s_scratch.shape[1]
    store_width = q_ids.shape[1]
    q_tiles = triton.cdiv(vocab_size, _Q_BLOCK_V)
    num_parts = part_max.shape[0]
    block_parts = triton.next_power_of_2(num_parts)
    hist, tile_off, tile_quota, row_cut, row_n, row_argmax = scratch
    # Folding the combine and the cut makes each of a row's q_tiles programs
    # redo per-row work; past _FOLD_ROWS rows that costs more than the two
    # launches it saves (gfx950, B=32), so larger batches keep them.
    fold = num_rows <= _FOLD_ROWS
    if not fold:
        _combine_row_max_and_argmax(
            part_max, part_amax_val, part_amax_idx, row_max, row_argmax
        )
    _dspark_q_histogram[(num_rows, q_tiles)](
        s_scratch,
        part_max,
        part_max.stride(0),
        num_parts,
        row_max,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        hist,
        vocab_size,
        s_scratch.stride(0),
        q_tiles,
        1,
        BLOCK_PARTS=block_parts,
        BLOCK_V=_Q_BLOCK_V,
        **_LEVELS,
        REDUCE_ROW_MAX=fold,
        num_warps=4,
    )
    if not fold:
        _dspark_q_cut[(num_rows,)](
            hist,
            q_row_base,
            temperatures,
            SAMPLER_EPS,
            top_k,
            top_p,
            tile_off,
            tile_quota,
            row_cut,
            row_n,
            q_tiles,
            q_tiles,
            1,
            store_width,
            BLOCK_TILE=triton.next_power_of_2(q_tiles),
            NUM_BINS=_Q_BINS,
            LEVELS_PER_NAT=_Q_LEVELS_PER_NAT,
            num_warps=8,
        )
    _dspark_q_compact[(num_rows, q_tiles)](
        s_scratch,
        row_max,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        top_k,
        top_p,
        hist,
        q_tiles,
        1,
        tile_off,
        tile_quota,
        row_cut,
        row_n,
        q_ids,
        q_probs,
        vocab_size,
        s_scratch.stride(0),
        q_tiles,
        store_width,
        position,
        0,
        0,
        BLOCK_TILE=triton.next_power_of_2(q_tiles),
        BLOCK_V=_Q_BLOCK_V,
        **_LEVELS,
        MAKE_CUT=fold,
        DEST_BY_ROW=False,
        num_warps=8 if fold else 4,
    )
    _dspark_q_finalize[(num_rows,)](
        q_ids,
        q_probs,
        q_meta,
        q_ids,
        tile_off,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        part_amax_val,
        part_amax_idx,
        part_amax_val.stride(0),
        num_parts,
        row_argmax,
        row_n,
        seed,
        out,
        num_rows,
        q_tiles,
        q_tiles,
        vocab_size,
        store_width,
        position,
        noise_offset,
        out.stride(0),
        FROM_STAGING=False,
        ARGMAX_FROM_PARTS=fold,
        WORLD=1,
        BLOCK_PARTS=block_parts,
        BLOCK=_Q_FINAL_BLOCK,
        num_warps=4,
    )


# TP-sharded vocab (`select_and_draw_q_tp`): each rank holds the scores of its
# own vocab shard only. Every stage above is per (row, V tile) but the cut,
# which reads one row's whole histogram table, and finalize, which reads the
# stored entries. So each rank histograms its own tiles, tile-major, and the
# gathered tables (rank-major, which is id order: a shard is a contiguous id
# range) are the whole row's; every rank makes the same cut over them. Each rank
# then compacts only its own tiles, into a `[2, B, K]` staging pair at the
# slots the cut gave them, and finalize reads the gathered pairs, each slot from
# the rank whose tiles own it, and stores them. The stored set, its order and
# every value are the whole-vocab op's: a level depends only on the score and
# the row max, and the cut takes the boundary level's lowest ids wherever the
# tile edges fall.


class QSelectTPScratch(NamedTuple):
    """`select_and_draw_q_tp`'s work buffers, all int32."""

    hist: torch.Tensor  # [local tiles, B, bins], this rank's tables
    tile_off: torch.Tensor  # [B, every rank's tiles]
    tile_quota: torch.Tensor  # [B, every rank's tiles], unfolded only
    row_cut: torch.Tensor  # [B], unfolded only
    row_n: torch.Tensor  # [B]
    stage: torch.Tensor  # [2, B, K]: this rank's stored ids and weight bits
    row_argmax: torch.Tensor  # [B], unfolded only


def q_select_tp_scratch(
    num_rows: int, local_vocab: int, world: int, store_width: int
) -> QSelectTPScratch:
    """The regions `select_and_draw_q_tp` needs for ``num_rows`` rows, a
    ``local_vocab`` shard on each of ``world`` ranks and a store of
    ``store_width``."""
    local_tiles = triton.cdiv(local_vocab, _Q_BLOCK_V)
    tiles = world * local_tiles
    i32 = torch.int32
    return QSelectTPScratch(
        Region((local_tiles, num_rows, _Q_BINS), i32),
        Region((num_rows, tiles), i32),
        Region((num_rows, tiles), i32),
        Region((num_rows,), i32),
        Region((num_rows,), i32),
        Region((2, num_rows, store_width), i32),
        Region((num_rows,), i32),
    )


def _gather_words(t: torch.Tensor, tp_group, use_custom: bool) -> torch.Tensor:
    """All-gather an int32 tensor along dim 0 as raw 32-bit words."""
    gathered = tp_group.all_gather(t.view(torch.float32), use_custom=use_custom, dim=0)
    return gathered.view(torch.int32)


def select_and_draw_q_tp(
    s_scratch: torch.Tensor,
    part_max: torch.Tensor,
    part_amax_val: torch.Tensor,
    part_amax_idx: torch.Tensor,
    row_max: torch.Tensor,
    q_row_base: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    out: torch.Tensor,
    q_ids: torch.Tensor,
    q_probs: torch.Tensor,
    q_meta: torch.Tensor,
    scratch: QSelectTPScratch,
    position: int,
    noise_offset: int,
    vocab_size: int,
    vocab_start: int,
    tp_group,
    use_custom: bool,
) -> None:
    """`select_and_draw_q` over a TP-sharded vocab (see the notes above).

    ``s_scratch`` is this rank's ``[B, V / tp]`` scores, ids ``vocab_start``
    on; ``part_*`` are the gathered stage-1 partials, so the whole row's tiles;
    ``vocab_size`` is the whole vocab's; ``scratch`` is carved from
    `q_select_tp_scratch`. Every rank stores and draws what the whole-vocab op
    would. What crosses ranks is each rank's histogram tables and its
    ``[2, B, K]`` staging pair.
    """
    num_rows, local_vocab = q_row_base.shape[0], s_scratch.shape[1]
    world = tp_group.world_size
    store_width = q_ids.shape[1]
    local_tiles = triton.cdiv(local_vocab, _Q_BLOCK_V)
    num_tiles = world * local_tiles
    num_parts = part_max.shape[0]
    block_parts = triton.next_power_of_2(num_parts)
    hist, tile_off, tile_quota, row_cut, row_n, stage, row_argmax = scratch
    # As on the whole vocab (`select_and_draw_q`): a folded cut is made by each
    # of a row's local_tiles compaction programs over all ranks' tables, so
    # past _FOLD_ROWS rows the combine and the cut run as their own launches.
    fold = num_rows <= _FOLD_ROWS
    if not fold:
        _combine_row_max_and_argmax(
            part_max, part_amax_val, part_amax_idx, row_max, row_argmax
        )
    # Tile-major, so the gathered tables are `[world * local_tiles, B, bins]`:
    # the whole row's tiles in id order. A row that stores nothing leaves its
    # tables unwritten, and the compaction never reads them.
    _dspark_q_histogram[(num_rows, local_tiles)](
        s_scratch,
        part_max,
        part_max.stride(0),
        num_parts,
        row_max,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        hist,
        local_vocab,
        s_scratch.stride(0),
        1,
        num_rows,
        BLOCK_PARTS=block_parts,
        BLOCK_V=_Q_BLOCK_V,
        **_LEVELS,
        REDUCE_ROW_MAX=fold,
        num_warps=4,
    )
    tables = _gather_words(hist, tp_group, use_custom)
    if not fold:
        _dspark_q_cut[(num_rows,)](
            tables,
            q_row_base,
            temperatures,
            SAMPLER_EPS,
            top_k,
            top_p,
            tile_off,
            tile_quota,
            row_cut,
            row_n,
            num_tiles,
            1,
            num_rows,
            store_width,
            BLOCK_TILE=triton.next_power_of_2(num_tiles),
            NUM_BINS=_Q_BINS,
            LEVELS_PER_NAT=_Q_LEVELS_PER_NAT,
            num_warps=8,
        )
    _dspark_q_compact[(num_rows, local_tiles)](
        s_scratch,
        row_max,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        top_k,
        top_p,
        tables,
        1,
        num_rows,
        tile_off,
        tile_quota,
        row_cut,
        row_n,
        stage[0],
        stage[1].view(torch.float32),
        local_vocab,
        s_scratch.stride(0),
        num_tiles,
        store_width,
        position,
        tp_group.rank_in_group * local_tiles,
        vocab_start,
        BLOCK_TILE=triton.next_power_of_2(num_tiles),
        BLOCK_V=_Q_BLOCK_V,
        **_LEVELS,
        MAKE_CUT=fold,
        DEST_BY_ROW=True,
        num_warps=8 if fold else 4,
    )
    _dspark_q_finalize[(num_rows,)](
        q_ids,
        q_probs,
        q_meta,
        _gather_words(stage, tp_group, use_custom),
        tile_off,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        part_amax_val,
        part_amax_idx,
        part_amax_val.stride(0),
        num_parts,
        row_argmax,
        row_n,
        seed,
        out,
        num_rows,
        num_tiles,
        local_tiles,
        vocab_size,
        store_width,
        position,
        noise_offset,
        out.stride(0),
        FROM_STAGING=True,
        ARGMAX_FROM_PARTS=fold,
        WORLD=world,
        BLOCK_PARTS=block_parts,
        BLOCK=_Q_FINAL_BLOCK,
        num_warps=4,
    )
