"""The ratio verify's target p, reduced to per-row statistics.

Every (draft row, V tile) program reduces its tile of the row to the softmax
statistics (`softmax_tiles`) plus the row's raw-logit argmax, and a batch
with a top-k or top-p then finds each row's floor (`sampling_filter`). The
stats pass also keeps the run maxima a counted top-k reads, so it fuses the
filter's run-maxima pass. `ratio_verify` reads p from what this leaves.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.sampling_filter import (
    _BATCH_COUNTS,
    _FILTER_COUNT_GROUPS,
    _FILTER_COUNT_MAX_K,
    FilterScratch,
    RowFilters,
    _counted_row,
    _logit_key,
    _request_filter,
    _request_of_row,
    filter_scratch,
    row_filters,
)
from atom.model_ops.spec_sampling.sampling_filter_count import count_floor
from atom.model_ops.spec_sampling.sampling_filter_descent import descend_floor
from atom.model_ops.spec_sampling.sampling_scratch import Region
from atom.model_ops.spec_sampling.softmax_tiles import (
    softmax_combine_kernel,
    tile_max_sumexp,
)
from atom.model_ops.tile_argmax import argmax_lowest_id

# Vocab tile every ratio kernel walks in.
_RATIO_BLOCK_V = 4096


@triton.jit(do_not_specialize=_BATCH_COUNTS + ["search_steps", "top_k_val"])
def _ratio_stats_kernel(
    logits_ptr,  # [num_tokens, V] raw target logits
    cu_num_draft_tokens_ptr,  # [batch_size]
    temperatures_ptr,  # [batch_size] fp32, pre-clamped
    part_max_ptr,  # [n_tiles, num_tokens] fp32: max of logits / T
    part_sum_ptr,  # [n_tiles, num_tokens] fp32: sum exp(logits / T - max)
    part_amax_val_ptr,  # [n_tiles, num_tokens] fp32: max raw logit
    part_amax_idx_ptr,  # [n_tiles, num_tokens] int32: its lowest id
    num_tokens,
    vocab_size,
    logits_row_stride,
    batch_size,
    search_steps,
    BLOCK_V: tl.constexpr,
    # A COUNT launch's: the filters as `_request_filter` reads them, and the
    # [num_tokens * n_tiles, GROUPS] int32 run maxima (keyed) and the count
    # of ids at each out.
    greedy_temperature=0.0,
    top_k_ptr=None,
    top_k_val=0,
    top_p_ptr=None,
    top_p_val=2.0,
    group_max_ptr=None,
    group_at_max_ptr=None,
    COUNT: tl.constexpr = False,
    KEY_BITS: tl.constexpr = 16,
    GROUPS: tl.constexpr = 64,
    MAX_K: tl.constexpr = 0,
    K_PER_ROW: tl.constexpr = False,
    P_PER_ROW: tl.constexpr = False,
):
    """One (draft row, V tile): softmax statistics plus the raw argmax.

    One row per program, so a small batch still fills the device; the row's
    temperature is its request's, found in-kernel. The raw argmax is taken
    before the division: dividing by a near-zero temperature can round two
    distinct logits onto one value and invent a tie the greedy rule would
    not see.

    In a ``COUNT`` launch, a row whose filter is a small top-k alone
    (`_counted_row`) also keeps the largest logit key of each run of
    ``BLOCK_V // GROUPS`` ids and how many ids are at it, for
    `_ratio_filter_count_kernel`: one more max, no search -- a per-tile
    search here would cost more than the pass.
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    vmask = offs < vocab_size
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
        mask=vmask,
        other=float("-inf"),
    ).to(tl.float32)
    raw_max, raw_idx = argmax_lowest_id(x, offs, vmask, 2147483647)
    scaled_max, scaled_sum = tile_max_sumexp(x / temp, vmask)
    part = tile * num_tokens + row
    tl.store(part_max_ptr + part, scaled_max)
    tl.store(part_sum_ptr + part, scaled_sum)
    tl.store(part_amax_val_ptr + part, raw_max)
    tl.store(part_amax_idx_ptr + part, raw_idx)
    if COUNT:
        top_k, top_p = _request_filter(
            top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
        )
        if _counted_row(temp, top_k, top_p, vocab_size, greedy_temperature, MAX_K):
            # Max the logits, then key the maxima: NaN goes to -inf first, as
            # its key has it (masked lanes are -inf already).
            run_x = tl.reshape(
                tl.where(x >= float("-inf"), x, float("-inf")),
                (GROUPS, BLOCK_V // GROUPS),
            )
            run_max = tl.max(run_x, axis=1)
            at_max = tl.sum((run_x == run_max[:, None]).to(tl.int32), axis=1)
            first = (row * tl.num_programs(1) + tile).to(tl.int64) * GROUPS
            runs = first + tl.arange(0, GROUPS)
            group_max = _logit_key(run_max, KEY_BITS)
            tl.store(group_max_ptr + runs, group_max.to(tl.int32, bitcast=True))
            tl.store(group_at_max_ptr + runs, at_max)


class TargetPScratch(NamedTuple):
    """`_ratio_target_p`'s outputs and work buffers, per draft row."""

    row_max: torch.Tensor  # [rows] fp32
    row_inv_z: torch.Tensor  # [rows] fp32
    row_argmax: torch.Tensor  # [rows] int32, lowest id
    row_floor: torch.Tensor  # [rows] fp32, the lowest logit p_f keeps
    part_max: torch.Tensor  # [tiles, rows] fp32
    part_sum: torch.Tensor  # [tiles, rows] fp32
    part_amax_val: torch.Tensor  # [tiles, rows] fp32
    part_amax_idx: torch.Tensor  # [tiles, rows] int32
    filter: FilterScratch


def target_p_scratch(num_tokens: int, vocab_size: int) -> TargetPScratch:
    """The regions `_ratio_target_p` needs for ``num_tokens`` draft rows."""
    tiles = triton.cdiv(vocab_size, _RATIO_BLOCK_V)
    f32, i32 = torch.float32, torch.int32
    return TargetPScratch(
        Region((num_tokens,), f32),
        Region((num_tokens,), f32),
        Region((num_tokens,), i32),
        Region((num_tokens,), f32),
        Region((tiles, num_tokens), f32),
        Region((tiles, num_tokens), f32),
        Region((tiles, num_tokens), f32),
        Region((tiles, num_tokens), i32),
        filter_scratch(num_tokens, vocab_size, _RATIO_BLOCK_V),
    )


def filter_floor(
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    cu_num_draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped
    filters: RowFilters,
    rows: tuple[torch.Tensor, ...],
    work: FilterScratch,
    runs: tuple[torch.Tensor, torch.Tensor] | None,
    stats_block_v: int,
) -> None:
    """Each filtered draft row's floor, and its p's normalizer renormalized
    over the survivors.

    Fills ``row_floor`` and rewrites ``row_inv_z`` of ``rows = (row_max,
    row_inv_z, row_floor)``, through ``work``. ``runs`` (`CountRuns`, given
    when the batch counts) are what the stats pass kept per ``stats_block_v``
    tile in a ``COUNT`` launch; ``row_max`` and ``row_inv_z`` are its combined
    statistics."""
    if filters.counts:
        count_floor(
            target_logits,
            cu_num_draft_tokens,
            temperatures,
            filters,
            rows,
            runs,
            stats_block_v,
            _FILTER_COUNT_MAX_K,
        )
    if filters.descends:
        descend_floor(
            target_logits,
            cu_num_draft_tokens,
            temperatures,
            filters,
            rows,
            (work.filter_f32, work.floor_mass, work.filter_i32, work.floor_counts),
            _FILTER_COUNT_MAX_K if filters.counts else 0,
        )


def _ratio_target_p(
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    cu_num_draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped
    top_ks: int | torch.Tensor | None,
    top_ps: float | torch.Tensor | None,
    work: TargetPScratch,
) -> bool:
    """Each draft row's p, and p_f where its request filters.

    Fills ``work``'s ``row_max``, ``row_inv_z``, ``row_argmax`` and
    ``row_floor`` through its partials and `filter_floor`'s buffers: p(v) is
    ``exp(l(v) / T - row_max) * row_inv_z``, and on a filtered row 0 below
    ``row_floor`` with ``row_inv_z`` renormalized over what is left.
    ``top_ks``/``top_ps`` are what `Sampler` is given: None (no row filters),
    one value for the batch, or one per request, a top-k of -1 meaning none.
    Returns whether the batch filters at all; ``row_floor`` is written only
    then.
    """
    num_tokens, vocab_size = target_logits.shape
    batch_size = cu_num_draft_tokens.shape[0]
    n_tiles = triton.cdiv(vocab_size, _RATIO_BLOCK_V)
    stride = target_logits.stride(0)
    search_steps = batch_size.bit_length() + 1
    filters = row_filters(top_ks, top_ps, batch_size, target_logits.dtype)
    if not num_tokens:
        return filters.filtered
    runs = work.filter.runs if filters.counts else (None, None)
    _ratio_stats_kernel[(num_tokens, n_tiles)](
        target_logits,
        cu_num_draft_tokens,
        temperatures,
        work.part_max,
        work.part_sum,
        work.part_amax_val,
        work.part_amax_idx,
        num_tokens,
        vocab_size,
        stride,
        batch_size,
        search_steps,
        _RATIO_BLOCK_V,
        SAMPLER_EPS,
        *filters.args,
        # The stats pass fills the filter's run maxima itself.
        *runs,
        COUNT=filters.counts,
        KEY_BITS=filters.key_bits,
        GROUPS=_FILTER_COUNT_GROUPS,
        MAX_K=_FILTER_COUNT_MAX_K,
        K_PER_ROW=filters.k_rows,
        P_PER_ROW=filters.p_rows,
        num_warps=8,
    )
    softmax_combine_kernel[(num_tokens,)](
        work.part_max,
        work.part_sum,
        work.row_max,
        work.row_inv_z,
        work.part_amax_val,
        work.part_amax_idx,
        work.row_argmax,
        num_tokens,
        n_tiles,
        BLOCK_TILE=triton.next_power_of_2(n_tiles),
        WITH_ARGMAX=True,
        num_warps=4,
    )
    if not filters.filtered:
        return False
    filter_floor(
        target_logits,
        cu_num_draft_tokens,
        temperatures,
        filters,
        (work.row_max, work.row_inv_z, work.row_floor),
        work.filter,
        runs,
        _RATIO_BLOCK_V,
    )
    return True
