# SPDX-License-Identifier: MIT
"""The ratio verify's floor by a radix descent over the logits' key bits.

Every filtered row but a counted one (`sampling_filter_count`): passes of
``_FILTER_DIGIT_BITS`` narrow the key prefix, then the last count and the cut
resolve the final ``_FILTER_LAST_BITS``, where each bin is one logit. An
unweighted bf16 key skips the passes: one 8-bit count picks its top byte.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.sampling_filter import (
    _BATCH_COUNTS,
    _FILTER_BLOCK_V,
    _FILTER_DIGIT_BITS,
    _FILTER_F32_WIDTH,
    _FILTER_FLOOR_PROGRAMS,
    _FILTER_I32_WIDTH,
    _FILTER_LAST_BITS,
    _FILTER_SPLIT,
    RowFilters,
    _counted_row,
    _is_filtered,
    _key_logit,
    _logit_key,
    _request_filter,
    _request_of_row,
)

# The vocab tile and warps of a pass that weighs, which bins every id into all
# of its bins at once.
_FILTER_WEIGH_BLOCK_V = 256
_FILTER_WEIGH_WARPS = 4

# Where the last count (`_ratio_filter_floor_kernel`) and the cut
# (`_ratio_filter_cut_kernel`) take the key prefix from: the last pass's step
# of the descent; none yet, so the count bins every id by the key's top byte
# (an unweighted bf16 key, whose passes this replaces); or the prefix
# `_ratio_filter_byte_pick_kernel` recorded from that count.
_PREFIX_DESCEND = tl.constexpr(0)
_PREFIX_ZERO = tl.constexpr(1)
_PREFIX_RECORDED = tl.constexpr(2)


@triton.jit
def _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, MAX_K):
    """Is this a filtered row the radix descent finds the floor of? Every one
    but a `_counted_row` (``MAX_K`` 0 when the batch counts none)."""
    return _is_filtered(temp, top_k, top_p, vocab_size, greedy_temperature) & ~(
        _counted_row(temp, top_k, top_p, vocab_size, greedy_temperature, MAX_K)
    )


@triton.jit
def _filter_pick(counts, masses, above_n, above_m, top_k, top_p, BINS: tl.constexpr):
    """The bin the floor lies in, among bins in logit order holding
    ``counts`` ids of mass ``masses``, below ``above_n`` ids of mass
    ``above_m``: ``(bin, ids above it, their mass, its own mass)``, the last
    three within these bins.

    The floor is the lowest logit whose ids survive, and survival only gets
    harder going down the logits, so its bin is the lowest nonempty one whose
    highest ids survive -- those have exactly the higher bins above them. The
    scans run in a fixed order, so every TP rank picks alike.
    """
    bins = tl.arange(0, BINS)
    n_higher = tl.sum(counts, axis=0) - tl.cumsum(counts, axis=0)
    m_higher = tl.cumsum(masses, axis=0, reverse=True) - masses
    alive = (counts > 0) & (above_n + n_higher < top_k) & (above_m + m_higher < top_p)
    pick = tl.min(tl.where(alive, bins, BINS), axis=0)
    # The top bin always survives but on a NaN mass; keep it then, so the
    # floor never passes the max.
    pick = tl.where(pick == BINS, tl.max(tl.where(counts > 0, bins, 0), axis=0), pick)
    at = bins == pick
    return (
        pick,
        tl.sum(tl.where(at, n_higher, 0), axis=0),
        tl.sum(tl.where(at, m_higher, 0.0), axis=0),
        tl.sum(tl.where(at, masses, 0.0), axis=0),
    )


@triton.jit
def _filter_state_load(
    filter_i32_ptr,
    filter_f32_ptr,
    state,
    mask,
    AT: tl.constexpr,
    I32_WIDTH: tl.constexpr,
    F32_WIDTH: tl.constexpr,
):
    """The descent state recorded in record ``state`` (parity * num_tokens +
    row), past its ``AT`` bins: ``(key prefix, ids above, their mass)``, all
    zeros where ``mask`` is false."""
    prefix = tl.load(filter_i32_ptr + state * I32_WIDTH + AT, mask=mask, other=0)
    above_n = tl.load(filter_i32_ptr + state * I32_WIDTH + AT + 1, mask=mask, other=0)
    above_m = tl.load(filter_f32_ptr + state * F32_WIDTH + AT, mask=mask, other=0.0)
    return prefix.to(tl.uint32, bitcast=True), above_n, above_m


@triton.jit
def _filter_state_store(
    filter_i32_ptr,
    filter_f32_ptr,
    state,
    prefix,
    above_n,
    above_m,
    mask,
    AT: tl.constexpr,
    I32_WIDTH: tl.constexpr,
    F32_WIDTH: tl.constexpr,
):
    """Record ``(prefix, above_n, above_m)`` where `_filter_state_load` reads
    it, where ``mask``."""
    tl.store(
        filter_i32_ptr + state * I32_WIDTH + AT,
        prefix.to(tl.uint32).to(tl.int32, bitcast=True),
        mask=mask,
    )
    tl.store(filter_i32_ptr + state * I32_WIDTH + AT + 1, above_n, mask=mask)
    tl.store(filter_f32_ptr + state * F32_WIDTH + AT, above_m, mask=mask)


@triton.jit
def _summed_floor_counts(floor_counts_ptr, row, splits, LAST_BINS: tl.constexpr):
    """A row's last-digit counts, its ``splits`` added in order."""
    bins = tl.arange(0, LAST_BINS)
    counts = tl.zeros((LAST_BINS,), dtype=tl.int32)
    for split in range(splits):
        cell = (row * splits + split).to(tl.int64) * LAST_BINS
        counts += tl.load(floor_counts_ptr + cell + bins)
    return counts


@triton.jit
def _filter_descend(
    filter_i32_ptr,
    filter_f32_ptr,
    row,
    num_tokens,
    done_pass,
    has_pass,
    top_k,
    top_p,
    SPLIT: tl.constexpr,
    DIGIT_BITS: tl.constexpr,
    I32_WIDTH: tl.constexpr,
    F32_WIDTH: tl.constexpr,
):
    """The row's ``(key prefix, ids above, their mass)`` after pass
    ``done_pass``: the state before it, recorded at the other parity, one
    digit further down by that pass's bins. All zeros when ``has_pass`` is
    false, which is the state before the first pass."""
    BINS: tl.constexpr = 1 << DIGIT_BITS
    prefix, above_n, above_m = _filter_state_load(
        filter_i32_ptr,
        filter_f32_ptr,
        ((done_pass + 1) % 2 * num_tokens + row).to(tl.int64),
        has_pass & (done_pass > 0),
        SPLIT * BINS,
        I32_WIDTH,
        F32_WIDTH,
    )
    part = (done_pass % 2 * num_tokens + row).to(tl.int64)
    cells = tl.arange(0, SPLIT)[:, None] * BINS + tl.arange(0, BINS)[None, :]
    counts = tl.sum(
        tl.load(filter_i32_ptr + part * I32_WIDTH + cells, mask=has_pass, other=0),
        axis=0,
    )
    masses = tl.sum(
        tl.load(filter_f32_ptr + part * F32_WIDTH + cells, mask=has_pass, other=0.0),
        axis=0,
    )
    pick, n_add, m_add, _ = _filter_pick(
        counts, masses, above_n, above_m, top_k, top_p, BINS
    )
    return (
        (prefix << DIGIT_BITS) | pick.to(tl.uint32),
        above_n + n_add,
        above_m + m_add,
    )


@triton.jit(
    do_not_specialize=_BATCH_COUNTS + ["top_k_val", "search_steps", "filter_pass"]
)
def _ratio_filter_pass_kernel(
    logits_ptr,  # [num_tokens, V] raw target logits
    cu_num_draft_tokens_ptr,  # [batch_size]
    temperatures_ptr,  # [batch_size] fp32, pre-clamped
    top_k_ptr,  # [batch_size] int32, or None for the batch's one value
    top_k_val,
    top_p_ptr,  # [batch_size] fp32, or None for the batch's one value
    top_p_val,
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32
    filter_i32_ptr,  # [2, num_tokens, I32_WIDTH]
    filter_f32_ptr,  # [2, num_tokens, F32_WIDTH]
    num_tokens,
    vocab_size,
    logits_row_stride,
    batch_size,
    search_steps,
    greedy_temperature,
    filter_pass,
    tiles_per_split,
    BLOCK_V: tl.constexpr,
    KEY_BITS: tl.constexpr,
    SPLIT: tl.constexpr,
    DIGIT_BITS: tl.constexpr,
    I32_WIDTH: tl.constexpr,
    F32_WIDTH: tl.constexpr,
    K_PER_ROW: tl.constexpr,
    P_PER_ROW: tl.constexpr,
    WEIGH: tl.constexpr,
    COUNTED: tl.constexpr,
):
    """One (draft row, vocab split) of filter pass ``filter_pass``: count, and
    weigh by p where the row has a top-p (only in a ``WEIGH`` launch, one
    whose batch has a top-p at all), the ids under the row's key prefix,
    binned by their next key digit. Takes the previous pass's step of the
    descent first (every split alike; split 0 records it), so one launch per
    pass is enough.

    The records alternate between two parities, so a pass never writes what
    it reads: it reads the previous pass's bins and the state before them,
    and writes its own bins beside that state and the new state beside those
    bins.
    """
    BINS: tl.constexpr = 1 << DIGIT_BITS
    row = tl.program_id(0)
    split = tl.program_id(1)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    top_k, top_p = _request_filter(
        top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
    )
    if _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, COUNTED):
        has_pass = filter_pass > 0
        prefix, above_n, above_m = _filter_descend(
            filter_i32_ptr,
            filter_f32_ptr,
            row,
            num_tokens,
            filter_pass - 1,
            has_pass,
            top_k,
            top_p,
            SPLIT,
            DIGIT_BITS,
            I32_WIDTH,
            F32_WIDTH,
        )
        _filter_state_store(
            filter_i32_ptr,
            filter_f32_ptr,
            ((filter_pass + 1) % 2 * num_tokens + row).to(tl.int64),
            prefix,
            above_n,
            above_m,
            has_pass & (split == 0),
            SPLIT * BINS,
            I32_WIDTH,
            F32_WIDTH,
        )

        shift = (KEY_BITS - DIGIT_BITS * (filter_pass + 1)).to(tl.uint32)
        row_max = tl.load(row_max_ptr + row)
        inv_z = tl.load(row_inv_z_ptr + row)
        bins = tl.arange(0, BINS)
        if WEIGH:
            # Each lane keeps its own bins, summed once after the loop: a
            # cross-lane sum per tile costs more than the binning.
            lane_counts = tl.zeros((BLOCK_V, BINS), dtype=tl.int32)
            lane_masses = tl.zeros((BLOCK_V, BINS), dtype=tl.float32)
        else:
            counts = tl.zeros((BINS,), dtype=tl.int32)
        n_tiles = tl.cdiv(vocab_size, BLOCK_V)
        tile_end = tl.minimum((split + 1) * tiles_per_split, n_tiles)
        for tile in range(split * tiles_per_split, tile_end):
            offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
            vmask = offs < vocab_size
            x = tl.load(
                logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
                mask=vmask,
                other=float("-inf"),
            ).to(tl.float32)
            key = _logit_key(x, KEY_BITS) >> shift
            under = vmask & ((key >> DIGIT_BITS) == prefix)
            digit = (key & (BINS - 1)).to(tl.int32)
            if WEIGH:
                # Every row of the launch weighs (a branch on its own top-p
                # costs more than it saves); a row without one ignores it.
                hit = tl.where(under, digit, BINS)[:, None] == bins[None, :]
                lane_counts += hit.to(tl.int32)
                x = tl.where(x >= float("-inf"), x, float("-inf"))
                p = tl.exp(x / temp - row_max) * inv_z
                lane_masses += tl.where(hit, p[:, None], 0.0)
            else:
                counts += tl.histogram(digit, BINS, mask=under)
        if WEIGH:
            counts = tl.sum(lane_counts, axis=0)
            masses = tl.sum(lane_masses, axis=0)
        else:
            masses = tl.zeros((BINS,), dtype=tl.float32)
        part = (filter_pass % 2 * num_tokens + row).to(tl.int64)
        tl.store(filter_i32_ptr + part * I32_WIDTH + split * BINS + bins, counts)
        tl.store(filter_f32_ptr + part * F32_WIDTH + split * BINS + bins, masses)


@triton.jit(do_not_specialize=_BATCH_COUNTS + ["top_k_val", "search_steps", "splits"])
def _ratio_filter_byte_pick_kernel(
    cu_num_draft_tokens_ptr,  # [batch_size]
    temperatures_ptr,  # [batch_size] fp32, pre-clamped
    top_k_ptr,
    top_k_val,
    top_p_ptr,
    top_p_val,
    filter_i32_ptr,
    filter_f32_ptr,
    floor_counts_ptr,  # [num_tokens * splits, LAST_BINS] int32, the top byte's
    vocab_size,
    batch_size,
    search_steps,
    greedy_temperature,
    splits,
    LAST_BITS: tl.constexpr,
    SPLIT: tl.constexpr,
    DIGIT_BITS: tl.constexpr,
    I32_WIDTH: tl.constexpr,
    F32_WIDTH: tl.constexpr,
    K_PER_ROW: tl.constexpr,
    P_PER_ROW: tl.constexpr,
    COUNTED: tl.constexpr,
):
    """One draft row of an unweighted bf16 batch: the bin of the key's top
    byte the floor lies in, from the top-byte count's splits added in order,
    recorded as the descent's state at parity 0 (`_PREFIX_RECORDED`). One
    8-bit step in place of the two 4-bit passes: the bins are in logit order
    either way and the pick takes the lowest bin whose ids above number fewer
    than top_k, so it lands on the same prefix."""
    BINS: tl.constexpr = 1 << DIGIT_BITS
    LAST_BINS: tl.constexpr = 1 << LAST_BITS
    row = tl.program_id(0)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    top_k, top_p = _request_filter(
        top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
    )
    if _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, COUNTED):
        pick, n_add, _, _ = _filter_pick(
            _summed_floor_counts(floor_counts_ptr, row, splits, LAST_BINS),
            tl.zeros((LAST_BINS,), dtype=tl.float32),
            0,
            0.0,
            top_k,
            top_p,
            LAST_BINS,
        )
        _filter_state_store(
            filter_i32_ptr,
            filter_f32_ptr,
            row.to(tl.int64),
            pick,
            n_add,
            tl.full((), 0.0, tl.float32),
            True,
            SPLIT * BINS,
            I32_WIDTH,
            F32_WIDTH,
        )


@triton.jit(
    do_not_specialize=_BATCH_COUNTS
    + ["top_k_val", "search_steps", "last_pass", "tiles_per_split"]
)
def _ratio_filter_floor_kernel(
    logits_ptr,  # [num_tokens, V] raw target logits
    cu_num_draft_tokens_ptr,  # [batch_size]
    temperatures_ptr,  # [batch_size] fp32, pre-clamped
    top_k_ptr,
    top_k_val,
    top_p_ptr,
    top_p_val,
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32
    filter_i32_ptr,
    filter_f32_ptr,
    floor_counts_ptr,  # [num_tokens * splits, LAST_BINS] int32 out
    floor_mass_ptr,  # [n_tiles, num_tokens] fp32 out (unless WEIGHED)
    num_tokens,
    vocab_size,
    logits_row_stride,
    batch_size,
    search_steps,
    greedy_temperature,
    last_pass,
    tiles_per_split,
    BLOCK_V: tl.constexpr,
    KEY_BITS: tl.constexpr,
    LAST_BITS: tl.constexpr,
    SPLIT: tl.constexpr,
    DIGIT_BITS: tl.constexpr,
    I32_WIDTH: tl.constexpr,
    F32_WIDTH: tl.constexpr,
    K_PER_ROW: tl.constexpr,
    P_PER_ROW: tl.constexpr,
    WEIGHED: tl.constexpr,
    COUNTED: tl.constexpr,
    PREFIX: tl.constexpr,
):
    """One (draft row, vocab split): the key prefix (``PREFIX``: the last
    pass's step of the descent, or the recorded one), then count the ids
    under it by the key's last ``LAST_BITS``. Unless the passes ``WEIGHED``
    (and so carry the mass above the prefix down the descent), also weigh
    each tile's ids above the prefix: p_f's normalizer needs that mass either
    way. A tile with none skips the exponentials.

    `_PREFIX_ZERO` (unweighted bf16) instead counts every id by the key's top
    ``LAST_BITS``, which `_ratio_filter_byte_pick_kernel` picks from."""
    BINS: tl.constexpr = 1 << DIGIT_BITS
    LAST_BINS: tl.constexpr = 1 << LAST_BITS
    row = tl.program_id(0)
    split = tl.program_id(1)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    top_k, top_p = _request_filter(
        top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
    )
    if _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, COUNTED):
        if PREFIX == _PREFIX_RECORDED:
            prefix, _, _ = _filter_state_load(
                filter_i32_ptr,
                filter_f32_ptr,
                row.to(tl.int64),
                True,
                SPLIT * BINS,
                I32_WIDTH,
                F32_WIDTH,
            )
        elif PREFIX == _PREFIX_ZERO:
            prefix = tl.full((), 0, tl.uint32)
        else:
            prefix, _, _ = _filter_descend(
                filter_i32_ptr,
                filter_f32_ptr,
                row,
                num_tokens,
                last_pass,
                True,
                top_k,
                top_p,
                SPLIT,
                DIGIT_BITS,
                I32_WIDTH,
                F32_WIDTH,
            )
        row_max = tl.load(row_max_ptr + row)
        inv_z = tl.load(row_inv_z_ptr + row)
        counts = tl.zeros((LAST_BINS,), dtype=tl.int32)
        n_tiles = tl.cdiv(vocab_size, BLOCK_V)
        tile_end = tl.minimum((split + 1) * tiles_per_split, n_tiles)
        for tile in range(split * tiles_per_split, tile_end):
            offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
            vmask = offs < vocab_size
            x = tl.load(
                logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
                mask=vmask,
                other=float("-inf"),
            ).to(tl.float32)
            key = _logit_key(x, KEY_BITS)
            if PREFIX == _PREFIX_ZERO:
                counts += tl.histogram(
                    (key >> (KEY_BITS - LAST_BITS)).to(tl.int32), LAST_BINS, mask=vmask
                )
            else:
                under = vmask & ((key >> LAST_BITS) == prefix)
                counts += tl.histogram(
                    (key & (LAST_BINS - 1)).to(tl.int32), LAST_BINS, mask=under
                )
            if (not WEIGHED) and (PREFIX != _PREFIX_ZERO):
                above = vmask & ((key >> LAST_BITS) > prefix)
                mass = 0.0
                if tl.max(above.to(tl.int32), axis=0) > 0:
                    x = tl.where(x >= float("-inf"), x, float("-inf"))
                    mass = tl.sum(
                        tl.where(above, tl.exp(x / temp - row_max) * inv_z, 0.0),
                        axis=0,
                    )
                tl.store(floor_mass_ptr + tile * num_tokens + row, mass)
        tl.store(
            floor_counts_ptr
            + (row * tl.num_programs(1) + split).to(tl.int64) * LAST_BINS
            + tl.arange(0, LAST_BINS),
            counts,
        )


@triton.jit(
    do_not_specialize=_BATCH_COUNTS
    + ["top_k_val", "search_steps", "last_pass", "splits"]
)
def _ratio_filter_cut_kernel(
    cu_num_draft_tokens_ptr,  # [batch_size]
    temperatures_ptr,  # [batch_size] fp32, pre-clamped
    top_k_ptr,
    top_k_val,
    top_p_ptr,
    top_p_val,
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32, renormalized over the survivors here
    row_floor_ptr,  # [num_tokens] fp32 out: the lowest surviving logit
    filter_i32_ptr,
    filter_f32_ptr,
    floor_counts_ptr,  # [num_tokens * splits, LAST_BINS] int32
    floor_mass_ptr,  # [n_tiles, num_tokens] fp32 (unless WEIGHED)
    num_tokens,
    vocab_size,
    batch_size,
    search_steps,
    greedy_temperature,
    last_pass,
    splits,
    BLOCK_V: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
    KEY_BITS: tl.constexpr,
    LAST_BITS: tl.constexpr,
    SPLIT: tl.constexpr,
    DIGIT_BITS: tl.constexpr,
    I32_WIDTH: tl.constexpr,
    F32_WIDTH: tl.constexpr,
    K_PER_ROW: tl.constexpr,
    P_PER_ROW: tl.constexpr,
    WEIGHED: tl.constexpr,
    COUNTED: tl.constexpr,
    PREFIX: tl.constexpr,
):
    """One draft row: the floor, from the last count's splits added in
    order, under the prefix ``PREFIX`` names (the descent's or the recorded
    one, as the last count took it). Each of its bins is a single logit, so
    counting weighs it too (count times that logit's p) and the floor is
    exact. p_f's normalizer is the mass at or above the floor. An unfiltered
    row's floor is -inf and its p is left as it was."""
    BINS: tl.constexpr = 1 << DIGIT_BITS
    LAST_BINS: tl.constexpr = 1 << LAST_BITS
    row = tl.program_id(0)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    top_k, top_p = _request_filter(
        top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
    )
    floor = tl.full((), float("-inf"), tl.float32)
    if _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, COUNTED):
        if PREFIX == _PREFIX_RECORDED:
            prefix, above_n, above_m = _filter_state_load(
                filter_i32_ptr,
                filter_f32_ptr,
                row.to(tl.int64),
                True,
                SPLIT * BINS,
                I32_WIDTH,
                F32_WIDTH,
            )
        else:
            prefix, above_n, above_m = _filter_descend(
                filter_i32_ptr,
                filter_f32_ptr,
                row,
                num_tokens,
                last_pass,
                True,
                top_k,
                top_p,
                SPLIT,
                DIGIT_BITS,
                I32_WIDTH,
                F32_WIDTH,
            )
        bins = tl.arange(0, LAST_BINS)
        counts = _summed_floor_counts(floor_counts_ptr, row, splits, LAST_BINS)
        if WEIGHED:
            mass_above = above_m
        else:
            tiles = tl.arange(0, BLOCK_TILE)
            mass_above = tl.sum(
                tl.load(
                    floor_mass_ptr + tiles * num_tokens + row,
                    mask=tiles < tl.cdiv(vocab_size, BLOCK_V),
                    other=0.0,
                ),
                axis=0,
            )
        row_max = tl.load(row_max_ptr + row)
        inv_z = tl.load(row_inv_z_ptr + row)
        keys = (prefix << LAST_BITS) | bins.to(tl.uint32)
        p = tl.exp(_key_logit(keys, KEY_BITS) / temp - row_max) * inv_z
        masses = tl.where(counts > 0, counts.to(tl.float32) * p, 0.0)
        pick, _, m_add, m_at = _filter_pick(
            counts, masses, above_n, above_m, top_k, top_p, LAST_BINS
        )
        floor = _key_logit((prefix << LAST_BITS) | pick.to(tl.uint32), KEY_BITS)
        tl.store(row_inv_z_ptr + row, inv_z / (mass_above + m_add + m_at))
    # A counted row's floor is `_ratio_filter_count_kernel`'s.
    counted = _counted_row(temp, top_k, top_p, vocab_size, greedy_temperature, COUNTED)
    tl.store(row_floor_ptr + row, floor, mask=~counted)


def descend_floor(
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    cu_num_draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped
    filters: RowFilters,
    rows: tuple[torch.Tensor, ...],
    parts: tuple[torch.Tensor, ...],
    counted: int,
) -> None:
    """Each descended row's floor and renormalized ``row_inv_z``, in ``rows =
    (row_max, row_inv_z, row_floor)``, through ``parts = (filter_f32,
    floor_mass, filter_i32, floor_counts)``. ``counted`` is the count's
    largest top-k, 0 when the batch counts none; those rows are skipped."""
    num_tokens, vocab_size = target_logits.shape
    batch_size = cu_num_draft_tokens.shape[0]
    stride = target_logits.stride(0)
    search_steps = batch_size.bit_length() + 1
    row_max, row_inv_z, row_floor = rows
    filter_f32, floor_mass, filter_i32, floor_counts = parts
    request_filter = filters.args
    key_bits = filters.key_bits
    layout = {
        "KEY_BITS": key_bits,
        "SPLIT": _FILTER_SPLIT,
        "DIGIT_BITS": _FILTER_DIGIT_BITS,
        "I32_WIDTH": _FILTER_I32_WIDTH,
        "F32_WIDTH": _FILTER_F32_WIDTH,
        "K_PER_ROW": filters.k_rows,
        "P_PER_ROW": filters.p_rows,
        "COUNTED": counted,
    }
    passes = (key_bits - _FILTER_LAST_BITS) // _FILTER_DIGIT_BITS
    weigh = filters.weigh
    # Unweighted bf16: the key's top byte is one 8-bit count (the floor
    # kernel's) and a pick. A weighing batch keeps the passes, which carry
    # the mass down.
    top_byte = key_bits == 2 * _FILTER_LAST_BITS and not weigh
    block_v = _FILTER_WEIGH_BLOCK_V if weigh else _FILTER_BLOCK_V
    for filter_pass in range(0 if top_byte else passes):
        _ratio_filter_pass_kernel[(num_tokens, _FILTER_SPLIT)](
            target_logits,
            cu_num_draft_tokens,
            temperatures,
            *request_filter,
            row_max,
            row_inv_z,
            filter_i32,
            filter_f32,
            num_tokens,
            vocab_size,
            stride,
            batch_size,
            search_steps,
            SAMPLER_EPS,
            filter_pass,
            triton.cdiv(triton.cdiv(vocab_size, block_v), _FILTER_SPLIT),
            BLOCK_V=block_v,
            **layout,
            WEIGH=weigh,
            num_warps=_FILTER_WEIGH_WARPS if weigh else 8,
        )
    # The last count: one program per tile for a small batch, so a row's
    # whole vocab is not one program's to walk.
    floor_n_tiles = triton.cdiv(vocab_size, _FILTER_BLOCK_V)
    tiles_per_split = triton.cdiv(
        floor_n_tiles,
        max(1, min(floor_n_tiles, _FILTER_FLOOR_PROGRAMS // num_tokens)),
    )
    splits = triton.cdiv(floor_n_tiles, tiles_per_split)
    floor = {
        "BLOCK_V": _FILTER_BLOCK_V,
        "LAST_BITS": _FILTER_LAST_BITS,
        **layout,
        "WEIGHED": weigh,
    }

    def launch_floor(last_pass, prefix):
        _ratio_filter_floor_kernel[(num_tokens, splits)](
            target_logits,
            cu_num_draft_tokens,
            temperatures,
            *request_filter,
            row_max,
            row_inv_z,
            filter_i32,
            filter_f32,
            floor_counts,
            floor_mass,
            num_tokens,
            vocab_size,
            stride,
            batch_size,
            search_steps,
            SAMPLER_EPS,
            last_pass,
            tiles_per_split,
            **floor,
            PREFIX=prefix.value,
            num_warps=8,
        )

    if top_byte:
        launch_floor(0, _PREFIX_ZERO)
        _ratio_filter_byte_pick_kernel[(num_tokens,)](
            cu_num_draft_tokens,
            temperatures,
            *request_filter,
            filter_i32,
            filter_f32,
            floor_counts,
            vocab_size,
            batch_size,
            search_steps,
            SAMPLER_EPS,
            splits,
            LAST_BITS=_FILTER_LAST_BITS,
            SPLIT=_FILTER_SPLIT,
            DIGIT_BITS=_FILTER_DIGIT_BITS,
            I32_WIDTH=_FILTER_I32_WIDTH,
            F32_WIDTH=_FILTER_F32_WIDTH,
            K_PER_ROW=filters.k_rows,
            P_PER_ROW=filters.p_rows,
            COUNTED=counted,
            num_warps=4,
        )
    prefix = _PREFIX_RECORDED if top_byte else _PREFIX_DESCEND
    launch_floor(passes - 1, prefix)
    _ratio_filter_cut_kernel[(num_tokens,)](
        cu_num_draft_tokens,
        temperatures,
        *request_filter,
        row_max,
        row_inv_z,
        row_floor,
        filter_i32,
        filter_f32,
        floor_counts,
        floor_mass,
        num_tokens,
        vocab_size,
        batch_size,
        search_steps,
        SAMPLER_EPS,
        passes - 1,
        splits,
        BLOCK_TILE=triton.next_power_of_2(floor_n_tiles),
        **floor,
        PREFIX=prefix.value,
        num_warps=4,
    )
