# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stages 3-6 of the sampled DSpark draft (`dspark_markov_sample`): choose the
top of each row's q from the dense score scratch, store it (`SparseQStore`),
and draw the row's token from exactly what was stored.

A deterministic row (greedy, non-finite, degenerate) stores the delta on its
token; a negative row base writes nothing and drafts the logits' argmax.
The cut is by level, the gap below the row max in 1/8 nats, not by exact value:
the boundary level's lowest ids fill n = min(K, top-k, ids within 19 nats).
That only moves acceptance: the draft samples from what is stored.
Launches: histogram, cut, compact, finalize (normalize, then the exponential
race over the stored entries). Both 3 and 5 level through `_q_level`.
Fixed-order reductions, no atomics: TP ranks must store and draw alike.
Philox noise in-kernel, at a full-vocab race's counter per (position, row, id).
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.sampling_scratch import Region
from atom.model_ops.spec_sampling.softmax_tiles import race_key
from atom.model_ops.tile_argmax import argmax_lowest_id

# Stages 3 and 5 tile V identically (the cut hands each tile a slot range).
_Q_BLOCK_V = 4096
_Q_BINS = 256
# Level = 1 + floor(gap * 8) for a gap in (0, 19] nats: at most 153, so
# everything fits below the last bin, which collects the rest (and NaN/inf).
_Q_LEVELS_PER_NAT = 8.0
_Q_MAX_GAP = 19.0
# Stored entries one finalize iteration covers.
_Q_FINAL_BLOCK = 1024


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
    row_max_ptr,
    q_row_base_ptr,
    temperatures_ptr,
    greedy_temperature,
    hist_ptr,
    vocab_size,
    stride_s_row,
    hist_row_stride,
    hist_tile_stride,
    BLOCK_V: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
):
    """One (row, V tile): how many of the tile's ids sit at each level.

    Its table goes to `hist_ptr + (row * hist_row_stride + tile *
    hist_tile_stride) * NUM_BINS`: row-major `[B, tiles, bins]` for the whole
    vocab, tile-major `[tiles, B, bins]` for a TP shard, whose tables the
    ranks then gather (`select_and_draw_q_tp`).
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    # Rows that store nothing stop here, and in the cut and compaction too.
    if (tl.load(q_row_base_ptr + row) < 0) | (
        tl.load(temperatures_ptr + row) <= greedy_temperature
    ):
        return
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < vocab_size
    s = tl.load(
        s_ptr + row.to(tl.int64) * stride_s_row + offs_v,
        mask=v_mask,
        other=float("-inf"),
    )
    level = _q_level(
        s, tl.load(row_max_ptr + row), v_mask, NUM_BINS, LEVELS_PER_NAT, MAX_GAP
    )
    counts = tl.histogram(level, NUM_BINS)
    tl.store(
        hist_ptr
        + (row * hist_row_stride + tile * hist_tile_stride) * NUM_BINS
        + tl.arange(0, NUM_BINS),
        counts,
    )


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
    """One row: how many to store, the boundary level, and each tile's slots.

    n = min(store width, the request's top-k, ids within reach of the max),
    then the request's top-p at level granularity. The boundary level is the
    lowest one whose cumulative count reaches n; its ids are taken lowest id
    first, so each tile's share of them (its quota) is what the tiles before
    it leave over, and each tile's first output slot is the running total of
    the tiles before it. Exactly n ids are then stored.

    The tables are read where `_dspark_q_histogram` wrote them, the
    ``num_tiles`` tiles in id order.
    """
    row = tl.program_id(0)
    if (tl.load(q_row_base_ptr + row) < 0) | (
        tl.load(temperatures_ptr + row) <= greedy_temperature
    ):
        return
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
    # table: one program per row runs this, so every full-table pass shows.
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
    tile_off_ptr,
    tile_quota_ptr,
    row_cut_ptr,
    q_ids_ptr,
    q_probs_ptr,
    vocab_size,
    stride_s_row,
    num_tiles,
    store_width,
    position,
    tile_base,
    id_base,
    BLOCK_V: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
    DEST_BY_ROW: tl.constexpr,
):
    """One (row, V tile): store the tile's chosen ids and their unnormalized
    weights ``exp(s - max)`` at its slot range, in id order.

    ``s`` holds ``vocab_size`` ids from ``id_base``, which are the cut's tiles
    from ``tile_base`` (both 0 for the whole vocab; a TP shard's otherwise,
    `select_and_draw_q_tp`). ``DEST_BY_ROW`` writes batch row ``row``'s slots
    to row ``row`` of a ``[B, store_width]`` staging pair rather than to its
    store row.
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    q_base = tl.load(q_row_base_ptr + row)
    if (q_base < 0) | (tl.load(temperatures_ptr + row) <= greedy_temperature):
        return
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < vocab_size
    s = tl.load(
        s_ptr + row.to(tl.int64) * stride_s_row + offs_v,
        mask=v_mask,
        other=float("-inf"),
    )
    row_max = tl.load(row_max_ptr + row)
    level = _q_level(s, row_max, v_mask, NUM_BINS, LEVELS_PER_NAT, MAX_GAP)
    cut = tl.load(row_cut_ptr + row)
    quota = tl.load(tile_quota_ptr + row * num_tiles + tile_base + tile)
    edge = level == cut
    take = (level < cut) | (edge & (tl.cumsum(edge.to(tl.int32), axis=0) <= quota))
    slot = tl.load(tile_off_ptr + row * num_tiles + tile_base + tile) + (
        tl.cumsum(take.to(tl.int32), axis=0) - 1
    )
    if DEST_BY_ROW:
        dest = row.to(tl.int64) * store_width + slot
    else:
        dest = (q_base + position).to(tl.int64) * store_width + slot
    tl.store(q_ids_ptr + dest, id_base + offs_v, mask=take)
    tl.store(q_probs_ptr + dest, tl.exp(s - row_max), mask=take)


@triton.jit
def _dspark_q_finalize(
    q_ids_ptr,
    q_probs_ptr,
    q_meta_ptr,
    q_row_base_ptr,
    temperatures_ptr,
    greedy_temperature,
    row_argmax_ptr,
    row_n_ptr,
    seed_ptr,
    out_ptr,
    vocab_size,
    store_width,
    position,
    noise_offset,
    stride_out,
    BLOCK: tl.constexpr,
):
    """One row: normalize the stored weights into q, draw x from them, record
    x's slot.

    ``probs.div_(noise).argmax()`` over the stored entries, lowest id on a tie.
    A greedy row, or one with nothing stored or no positive race key, drafts
    its logits' lowest-id raw argmax and, if it has a destination, stores the
    exact delta on it: the id is not a draw from any softmax, and the accept
    rule must see the distribution it actually came from.
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
        mask = start + offs < n
        total += tl.sum(
            tl.load(q_probs_ptr + base + start + offs, mask=mask, other=0.0), axis=0
        )
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
        probs = tl.load(q_probs_ptr + base + slots, mask=mask, other=0.0) * inv_total
        tl.store(q_probs_ptr + base + slots, probs, mask=mask)
        ids = tl.load(q_ids_ptr + base + slots, mask=mask, other=0)
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
    x = tl.where(drawn, best_id, tl.load(row_argmax_ptr + row))
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
    )


def select_and_draw_q(
    s_scratch: torch.Tensor,
    row_max: torch.Tensor,
    row_argmax: torch.Tensor,
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
    """Stages 3-6 over the ``[B, V]`` scores in ``s_scratch``: store each
    row's q at ``q_row_base[b] + position`` and write its token to ``out``.

    ``row_max`` / ``row_argmax`` are the row's max score and lowest-id raw
    argmax; ``scratch`` is carved from `q_select_scratch`. Arguments
    are as `dspark_markov_sample` documents them, which checks them.
    """
    num_rows = q_row_base.shape[0]
    vocab_size = s_scratch.shape[1]
    store_width = q_ids.shape[1]
    q_tiles = triton.cdiv(vocab_size, _Q_BLOCK_V)
    hist, tile_off, tile_quota, row_cut, row_n = scratch
    levels = {
        "NUM_BINS": _Q_BINS,
        "LEVELS_PER_NAT": _Q_LEVELS_PER_NAT,
        "MAX_GAP": _Q_MAX_GAP,
    }
    _dspark_q_histogram[(num_rows, q_tiles)](
        s_scratch,
        row_max,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        hist,
        vocab_size,
        s_scratch.stride(0),
        q_tiles,
        1,
        BLOCK_V=_Q_BLOCK_V,
        **levels,
        num_warps=4,
    )
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
        tile_off,
        tile_quota,
        row_cut,
        q_ids,
        q_probs,
        vocab_size,
        s_scratch.stride(0),
        q_tiles,
        store_width,
        position,
        0,
        0,
        BLOCK_V=_Q_BLOCK_V,
        **levels,
        DEST_BY_ROW=False,
        num_warps=4,
    )
    _dspark_q_finalize[(num_rows,)](
        q_ids,
        q_probs,
        q_meta,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        row_argmax,
        row_n,
        seed,
        out,
        vocab_size,
        store_width,
        position,
        noise_offset,
        out.stride(0),
        BLOCK=_Q_FINAL_BLOCK,
        num_warps=4,
    )


# TP-sharded vocab (`select_and_draw_q_tp`): each rank holds the scores of its
# own vocab shard only. Every stage above is per (row, V tile) but the cut,
# which reads one row's whole histogram table, and finalize, which reads the
# stored entries. So each rank histograms its own tiles, tile-major, and the
# gathered tables (rank-major, which is id order: a shard is a contiguous id
# range) are the whole row's; every rank runs the same cut over them. Each rank
# then compacts only its own tiles, into a `[2, B, K]` staging pair at the
# slots the cut gave them, and the gathered pairs are published to the store,
# each slot from the rank whose tiles own it. The stored set, its order and
# every value are the whole-vocab op's: a level depends only on the score and
# the row max, and the cut takes the boundary level's lowest ids wherever the
# tile edges fall.


class QSelectTPScratch(NamedTuple):
    """`select_and_draw_q_tp`'s work buffers, all int32."""

    hist: torch.Tensor  # [local tiles, B, bins], this rank's tables
    tile_off: torch.Tensor  # [B, every rank's tiles]
    tile_quota: torch.Tensor  # [B, every rank's tiles]
    row_cut: torch.Tensor  # [B]
    row_n: torch.Tensor  # [B]
    stage: torch.Tensor  # [2, B, K]: this rank's stored ids and weight bits


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
    )


@triton.jit
def _dspark_q_publish(
    gathered_ptr,  # [world, 2, B, K] int32: every rank's staging pair
    tile_off_ptr,  # [B, num_tiles] int32, the cut's first slot of each tile
    q_ids_ptr,
    q_probs_ptr,
    q_row_base_ptr,
    temperatures_ptr,
    greedy_temperature,
    row_n_ptr,
    num_rows,
    num_tiles,
    rank_tiles,
    store_width,
    position,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One row: copy its ``n`` stored entries into its store row, each slot
    from the rank whose tiles the cut gave it (the ranks' slot ranges follow
    one another in rank order). The weights travel as their fp32 bits."""
    row = tl.program_id(0)
    q_base = tl.load(q_row_base_ptr + row)
    if (q_base < 0) | (tl.load(temperatures_ptr + row) <= greedy_temperature):
        return
    n = tl.load(row_n_ptr + row)
    dst = (q_base + position).to(tl.int64) * store_width
    offs = tl.arange(0, BLOCK)
    for start in range(0, n, BLOCK):
        slots = start + offs
        mask = slots < n
        owner = tl.zeros((BLOCK,), dtype=tl.int32)
        for r in tl.static_range(1, WORLD):
            first = tl.load(tile_off_ptr + row * num_tiles + r * rank_tiles)
            owner += (slots >= first).to(tl.int32)
        src = (owner.to(tl.int64) * 2 * num_rows + row) * store_width + slots
        ids = tl.load(gathered_ptr + src, mask=mask, other=0)
        bits = tl.load(gathered_ptr + src + num_rows * store_width, mask=mask, other=0)
        tl.store(q_ids_ptr + dst + slots, ids, mask=mask)
        tl.store(
            q_probs_ptr + dst + slots, bits.to(tl.float32, bitcast=True), mask=mask
        )


def _gather_words(t: torch.Tensor, tp_group, use_custom: bool) -> torch.Tensor:
    """All-gather an int32 tensor along dim 0 as raw 32-bit words."""
    gathered = tp_group.all_gather(t.view(torch.float32), use_custom=use_custom, dim=0)
    return gathered.view(torch.int32)


def select_and_draw_q_tp(
    s_scratch: torch.Tensor,
    row_max: torch.Tensor,
    row_argmax: torch.Tensor,
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
    on; ``row_max`` / ``row_argmax`` are already the whole row's;
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
    hist, tile_off, tile_quota, row_cut, row_n, stage = scratch
    levels = {
        "NUM_BINS": _Q_BINS,
        "LEVELS_PER_NAT": _Q_LEVELS_PER_NAT,
        "MAX_GAP": _Q_MAX_GAP,
    }
    # Tile-major, so the gathered tables are `[world * local_tiles, B, bins]`:
    # the whole row's tiles in id order. A row that stores nothing leaves its
    # tables unwritten, and the cut never reads them.
    _dspark_q_histogram[(num_rows, local_tiles)](
        s_scratch,
        row_max,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        hist,
        local_vocab,
        s_scratch.stride(0),
        1,
        num_rows,
        BLOCK_V=_Q_BLOCK_V,
        **levels,
        num_warps=4,
    )
    _dspark_q_cut[(num_rows,)](
        _gather_words(hist, tp_group, use_custom),
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
        tile_off,
        tile_quota,
        row_cut,
        stage[0],
        stage[1].view(torch.float32),
        local_vocab,
        s_scratch.stride(0),
        num_tiles,
        store_width,
        position,
        tp_group.rank_in_group * local_tiles,
        vocab_start,
        BLOCK_V=_Q_BLOCK_V,
        **levels,
        DEST_BY_ROW=True,
        num_warps=4,
    )
    _dspark_q_publish[(num_rows,)](
        _gather_words(stage, tp_group, use_custom),
        tile_off,
        q_ids,
        q_probs,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        row_n,
        num_rows,
        num_tiles,
        local_tiles,
        store_width,
        position,
        WORLD=world,
        BLOCK=_Q_FINAL_BLOCK,
        num_warps=4,
    )
    _dspark_q_finalize[(num_rows,)](
        q_ids,
        q_probs,
        q_meta,
        q_row_base,
        temperatures,
        SAMPLER_EPS,
        row_argmax,
        row_n,
        seed,
        out,
        vocab_size,
        store_width,
        position,
        noise_offset,
        out.stride(0),
        BLOCK=_Q_FINAL_BLOCK,
        num_warps=4,
    )
