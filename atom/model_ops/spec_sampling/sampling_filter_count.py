# SPDX-License-Identifier: MIT
"""The ratio verify's floor for a top-k of at most 256 alone, counted.

Such a row's floor is its k-th largest logit. The stats pass keeps the max of
every run of 64 ids; the k-th largest max bounds it below, so only the fewer
than k runs above that max are read (`_ratio_filter_count_kernel`).
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.sampling_filter import (
    _BATCH_COUNTS,
    _FILTER_COUNT_GROUPS,
    RowFilters,
    _counted_row,
    _key_logit,
    _logit_key,
    _request_filter,
    _request_of_row,
)

# Runs the count's smallest gather covers (then 4x that, then the largest
# top-k: at most k - 1 runs are read), and its warps.
_FILTER_COUNT_CHUNK = 32
_FILTER_COUNT_WARPS = 4


def count_widths(max_k: int) -> tuple[int, int, int]:
    """The count's gather widths, ascending: the last holds every run a
    top-k of at most ``max_k`` can leave above its bound."""
    return (
        _FILTER_COUNT_CHUNK,
        4 * _FILTER_COUNT_CHUNK,
        max(triton.next_power_of_2(max_k), 4 * _FILTER_COUNT_CHUNK),
    )


@triton.jit
def _key_search(keys, weight, top_k, KEY_BITS: tl.constexpr):
    """The largest key c with at least ``top_k`` of ``keys`` at or above it
    (``weight`` masks), one bit at a time; 0 when there are fewer."""
    cut = tl.full((), 0, tl.uint32)
    for b in tl.static_range(KEY_BITS):
        trial = cut | (tl.full((), 1, tl.uint32) << (KEY_BITS - 1 - b))
        n = tl.sum(tl.sum((weight & (keys >= trial)).to(tl.int32), axis=1), axis=0)
        cut = tl.where(n >= top_k, trial, cut)
    return cut


@triton.jit
def _count_floor(
    logits_ptr,
    row_base,
    runs_ptr,
    n_runs,
    vocab_size,
    low,
    at_low,
    top_k,
    temp,
    row_max,
    inv_z,
    RUN: tl.constexpr,
    N: tl.constexpr,
    KEY_BITS: tl.constexpr,
):
    """The floor key and the mass at or above it, from the ``n_runs`` (at
    most ``N``) runs whose max is above ``low`` and the ``at_low`` ids at
    ``low`` in the runs whose max is ``low``. Every id above ``low`` is in
    those runs, so if fewer than ``top_k`` are, the floor is ``low``;
    otherwise it is above ``low`` and a search over them finds it."""
    j = tl.arange(0, N)
    run = tl.load(runs_ptr + j, mask=j < n_runs, other=0)
    ids = run[:, None] * RUN + tl.arange(0, RUN)[None, :]
    real = (j < n_runs)[:, None] & (ids < vocab_size)
    x = tl.load(logits_ptr + row_base + ids, mask=real, other=float("-inf"))
    keys = _logit_key(x.to(tl.float32), KEY_BITS)
    above = real & (keys > low)
    n_above = tl.sum(tl.sum(above.to(tl.int32), axis=1), axis=0)
    if n_above < top_k:
        cut = low
    else:
        cut = _key_search(keys, above, top_k, KEY_BITS)
    p = tl.exp(_key_logit(keys, KEY_BITS) / temp - row_max) * inv_z
    mass = tl.sum(tl.sum(tl.where(real & (keys >= cut), p, 0.0), 1), 0)
    if cut == low:
        p_low = tl.exp(_key_logit(low, KEY_BITS) / temp - row_max) * inv_z
        mass += at_low.to(tl.float32) * p_low
    return cut, mass


@triton.jit(do_not_specialize=_BATCH_COUNTS + ["top_k_val", "search_steps"])
def _ratio_filter_count_kernel(
    logits_ptr,  # [num_tokens, V] raw target logits
    cu_num_draft_tokens_ptr,  # [batch_size]
    temperatures_ptr,  # [batch_size] fp32, pre-clamped
    top_k_ptr,
    top_k_val,
    top_p_ptr,
    top_p_val,
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32, renormalized over the survivors here
    row_floor_ptr,  # [num_tokens] fp32 out: the lowest surviving logit
    group_max_ptr,  # [num_tokens * n_tiles, GROUPS] int32, the runs' maxima
    group_at_max_ptr,  # [num_tokens * n_tiles, GROUPS] int32, ids at each max
    num_tokens,
    vocab_size,
    logits_row_stride,
    batch_size,
    search_steps,
    greedy_temperature,
    BLOCK_V: tl.constexpr,
    BLOCK_RUNS: tl.constexpr,
    GROUPS: tl.constexpr,
    MAX_K: tl.constexpr,
    WIDTHS: tl.constexpr,
    KEY_BITS: tl.constexpr,
    K_PER_ROW: tl.constexpr,
    P_PER_ROW: tl.constexpr,
):
    """One draft row: a `_counted_row`'s floor, its k-th largest logit key
    (by multiplicity), and p_f's normalizer, the mass at or above it. Every
    other row's floor is -inf here (the cut kernel, when it runs after,
    writes the descended ones).

    The k-th largest of the runs' maxima, L, is at most the row's k-th
    largest key: k runs each hold an id at or above it. Every id above L
    lies in one of the at most k - 1 runs whose max is above L, so those
    runs are all the count reads, whatever the row's shape (a -inf mask or
    a plateau ties thousands of runs at L): fewer than k ids above L there
    and the floor is L, else it is the k-th largest of them. The ids at L
    in the runs whose max is L come from the stats pass's counts. Integer
    counts, so the floor is the row's own, exact; the mass is summed in a
    fixed order, the same on every rank."""
    RUN: tl.constexpr = BLOCK_V // GROUPS
    row = tl.program_id(0)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    top_k, top_p = _request_filter(
        top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
    )
    floor = tl.full((), float("-inf"), tl.float32)
    if _counted_row(temp, top_k, top_p, vocab_size, greedy_temperature, MAX_K):
        n_groups = tl.cdiv(vocab_size, BLOCK_V) * GROUPS
        runs_ptr = group_max_ptr + row.to(tl.int64) * n_groups
        g = tl.arange(0, BLOCK_RUNS)
        group_max = tl.load(runs_ptr + g, mask=g < n_groups, other=0)
        group_max = group_max.to(tl.uint32, bitcast=True)[None, :]
        at_max = tl.load(
            group_at_max_ptr + row.to(tl.int64) * n_groups + g[None, :],
            mask=g[None, :] < n_groups,
            other=0,
        )
        low = _key_search(group_max, group_max > 0, top_k, KEY_BITS)
        at_low = tl.sum(tl.sum(tl.where(group_max == low, at_max, 0), 1), 0)
        up = group_max > low
        slot = tl.cumsum(up.to(tl.int32), axis=1) - 1
        n_runs = tl.sum(tl.sum(up.to(tl.int32), axis=1), axis=0)
        # The runs above L's indices, in order, over the maxima just read.
        tl.debug_barrier()
        tl.store(runs_ptr + slot, g[None, :], mask=up)
        tl.debug_barrier()
        row_base = row.to(tl.int64) * logits_row_stride
        row_max = tl.load(row_max_ptr + row)
        inv_z = tl.load(row_inv_z_ptr + row)
        # Exactly one window (previous width, width] holds n_runs: WIDTHS
        # ascends from 0, and its last entry holds the at most k - 1 runs
        # there can be. That window's width is the smallest gather reading
        # every run above L.
        cut = low
        mass = 0.0
        below = -1
        for w in tl.static_range(len(WIDTHS)):
            if (n_runs > below) & (n_runs <= WIDTHS[w]):
                cut, mass = _count_floor(
                    logits_ptr,
                    row_base,
                    runs_ptr,
                    n_runs,
                    vocab_size,
                    low,
                    at_low,
                    top_k,
                    temp,
                    row_max,
                    inv_z,
                    RUN,
                    WIDTHS[w],
                    KEY_BITS,
                )
            below = WIDTHS[w]
        floor = _key_logit(cut, KEY_BITS)
        # The mass holds at least the floor's own id: one window always holds
        # n_runs. A NaN mass divides through, as on the descent path.
        # Checked only under TRITON_DEBUG.
        tl.device_assert(
            (mass > 0.0) | (mass != mass), "count floor without mass"  # noqa: PLR0124
        )
        tl.store(row_inv_z_ptr + row, inv_z / mass)
    tl.store(row_floor_ptr + row, floor)


def count_floor(
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    cu_num_draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped
    filters: RowFilters,
    rows: tuple[torch.Tensor, ...],
    runs: tuple[torch.Tensor, torch.Tensor],
    stats_block_v: int,
    max_k: int,
) -> None:
    """Each counted row's floor and renormalized ``row_inv_z``, in ``rows =
    (row_max, row_inv_z, row_floor)``; every other row's floor is -inf.
    ``runs = (run_max, run_at_max)`` are what the stats pass kept per
    ``stats_block_v`` tile; ``max_k`` is the largest top-k it counted."""
    num_tokens, vocab_size = target_logits.shape
    batch_size = cu_num_draft_tokens.shape[0]
    n_tiles = triton.cdiv(vocab_size, stats_block_v)
    row_max, row_inv_z, row_floor = rows
    run_max, run_at_max = runs
    _ratio_filter_count_kernel[(num_tokens,)](
        target_logits,
        cu_num_draft_tokens,
        temperatures,
        *filters.args,
        row_max,
        row_inv_z,
        row_floor,
        run_max,
        run_at_max,
        num_tokens,
        vocab_size,
        target_logits.stride(0),
        batch_size,
        batch_size.bit_length() + 1,
        SAMPLER_EPS,
        BLOCK_V=stats_block_v,
        BLOCK_RUNS=triton.next_power_of_2(n_tiles * _FILTER_COUNT_GROUPS),
        GROUPS=_FILTER_COUNT_GROUPS,
        MAX_K=max_k,
        WIDTHS=count_widths(max_k),
        KEY_BITS=filters.key_bits,
        K_PER_ROW=filters.k_rows,
        P_PER_ROW=filters.p_rows,
        num_warps=_FILTER_COUNT_WARPS,
    )
