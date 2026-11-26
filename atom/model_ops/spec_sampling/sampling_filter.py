"""The ratio verify's top-k / top-p floor: shared row helpers and scratch.

The ratio verify serves a filtered request `Sampler`'s p_f. aiter's rule: id t
survives iff fewer than top_k ids score strictly above it and their p sums to
less than top_p, both over the unfiltered p; a tie group survives or falls
whole. So the survivors are every id at or above one logit, the row's floor,
found on the device by a radix descent (`sampling_filter_descent`) or, for a
top-k of at most 256 alone, counted from one read (`sampling_filter_count`).
Both agree with aiter to fp32 rounding, not bit for bit: they rank by logit,
aiter by fp32 p, which can tie logits under one fp32 ulp of ``max - logit / T``
apart (about ``exp(-max)`` of mass each), and top-p compares a sum to top_p.
`Sampler.keep_top_k_ties` keeps the bonus draw's tie groups whole too.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.model_ops.spec_sampling.sampling_scratch import Region

# Logit-key bits one filter pass resolves, so each pass bins by this many. The
# last 8 bits are resolved apart (`_ratio_filter_floor_kernel` counts them,
# `_ratio_filter_cut_kernel` picks the floor among them): each of their
# bins is one logit, so its mass is its count times that logit's p, and
# counting is all that pass does. An fp32 logit's 32 bits take six passes and
# the floor kernels, a bf16 one's 16 two, or, unweighted (top-k alone), one
# 8-bit count of the top byte and the floor kernels.
_FILTER_DIGIT_BITS = 4
_FILTER_BINS = 1 << _FILTER_DIGIT_BITS
_FILTER_LAST_BITS = 8
_FILTER_LAST_BINS = 1 << _FILTER_LAST_BITS
# Programs the last count aims to spread over the batch's rows: a small batch
# splits each row's vocab across up to one per tile, a batch past this takes
# one per row. What it splits is integer counts and per-tile masses, summed in
# tile order whatever the split, so the floor does not depend on it.
_FILTER_FLOOR_PROGRAMS = 1024
# Programs a filter pass splits each row's vocab across, and the vocab tile
# the filter kernels walk in.
_FILTER_SPLIT = 8
_FILTER_BLOCK_V = 4096
# A top-k of at most this many with no top-p is counted, not descended
# (`_counted_row`, `sampling_filter_count`).
_FILTER_COUNT_MAX_K = 256
# Runs per 4096-id tile whose max the stats pass keeps (runs of 64 ids).
_FILTER_COUNT_GROUPS = 64
# Per-row record of one filter pass: each program's bin counts (or masses),
# then the descent so far -- (key prefix, count above) in int32, the mass
# above in fp32.
_FILTER_I32_WIDTH = _FILTER_SPLIT * _FILTER_BINS + 2
_FILTER_F32_WIDTH = _FILTER_SPLIT * _FILTER_BINS + 1
# The verify runs eagerly at whatever batch arrives. Triton would specialize
# its kernels on the row and request counts (== 1, a multiple of 16) and so
# compile each once per class of batch, mid-serve, the first time that class
# arrives. The counts only bound loops and index, so none is specialized on;
# with every scratch view `SCRATCH_ALIGN`-aligned (`sampling_scratch`), one compile per kernel
# variant serves every batch.
_BATCH_COUNTS = ["num_tokens", "batch_size"]


@triton.jit
def _count_below(values_ptr, n, x, steps):
    """How many of the ``n`` ascending ``values`` are below ``x``: the index of
    the first at least ``x`` (``n`` if none). A binary search of ``steps``
    probes, enough for ``n < 2**steps``. The verify kernels' search: a draft
    row's request in ``cu_num_draft_tokens``, and a V tile's span of a stored
    q row (`SparseQStore`: its ids ascend). The accept's lookup of one id
    counts in one vector load instead (`ratio_verify._stored_q`)."""
    lo = n * 0
    hi = n
    for _ in range(steps):
        mid = (lo + hi) // 2
        open_ = lo < hi
        v = tl.load(values_ptr + mid, mask=open_ & (mid < n), other=0)
        right = open_ & (v < x)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(open_ & ~right, mid, hi)
    return lo


@triton.jit
def _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps):
    """The request a draft row belongs to: the first ``r`` with
    ``cu_num_draft_tokens[r] > row``, so requests with no drafts are skipped.
    ``search_steps`` >= ceil(log2(batch_size)) + 1."""
    return _count_below(cu_num_draft_tokens_ptr, batch_size, row + 1, search_steps)


@triton.jit
def _request_filter(
    top_k_ptr,
    top_k_val,
    top_p_ptr,
    top_p_val,
    req_idx,
    K_PER_ROW: tl.constexpr,
    P_PER_ROW: tl.constexpr,
):
    """A request's ``(top_k, top_p)``, from its own entry or the batch's one
    value. Off reads as no bound: a top-k of 0 or below as 2**31 - 1, a top-p
    of 1 or above as inf. A top-p of 1 is off as `SamplingParams` has it;
    aiter's ``mass above < 1`` may also drop a tail whose mass above rounds to
    1 (around 1e-6 of mass), which no sum in another order would reproduce."""
    if K_PER_ROW:
        top_k = tl.load(top_k_ptr + req_idx)
    else:
        top_k = top_k_val
    if P_PER_ROW:
        top_p = tl.load(top_p_ptr + req_idx)
    else:
        top_p = top_p_val
    top_p = top_p.to(tl.float32)
    return (
        tl.where(top_k > 0, top_k, 2147483647),
        tl.where(top_p < 1.0, top_p, float("inf")),
    )


@triton.jit
def _is_filtered(temp, top_k, top_p, vocab_size, greedy_temperature):
    """Does this row filter? A greedy row never does: it is exact match."""
    return (temp > greedy_temperature) & ((top_k < vocab_size) | (top_p < 1.0))


@triton.jit
def _counted_row(temp, top_k, top_p, vocab_size, greedy_temperature, MAX_K):
    """Is this a row whose filter is a top-k of at most ``MAX_K`` alone? Its
    floor is its k-th largest logit, which `_ratio_filter_count_kernel`
    counts out; the radix descent skips it."""
    return (
        _is_filtered(temp, top_k, top_p, vocab_size, greedy_temperature)
        & (top_k <= MAX_K)
        & (top_p == float("inf"))
    )


@triton.jit
def _logit_key(x, KEY_BITS: tl.constexpr):
    """The top ``KEY_BITS`` of an unsigned key that orders logits as their
    values do. -0 keys as +0, so the two tie as they compare; NaN keys as
    -inf, below every logit the floor can keep."""
    x = tl.where(x >= float("-inf"), x, float("-inf"))  # only NaN fails
    x = tl.where(x == 0.0, 0.0, x)
    u = x.to(tl.uint32, bitcast=True)
    key = tl.where((u >> 31) != 0, ~u, u | 0x80000000)
    return key >> (32 - KEY_BITS)


@triton.jit
def _key_logit(key, KEY_BITS: tl.constexpr):
    """The logit a full ``KEY_BITS`` key stands for (`_logit_key` undone)."""
    u = key << (32 - KEY_BITS)
    u = tl.where((u >> 31) != 0, u ^ 0x80000000, ~u)
    # A negative logit's dropped low bits come back set; a 16-bit key is a
    # bf16 logit, whose low bits are 0.
    LOW_CLEAR: tl.constexpr = (0xFFFFFFFF >> (32 - KEY_BITS)) << (32 - KEY_BITS)
    return (u & LOW_CLEAR).to(tl.float32, bitcast=True)


class RowFilters(NamedTuple):
    """A batch's ``(top_ks, top_ps)`` as the filter kernels read them."""

    #: `_request_filter`'s ``(top_k_ptr, top_k_val, top_p_ptr, top_p_val)``.
    args: tuple
    k_rows: bool
    p_rows: bool
    #: Bits of a logit's key: 16 for a bf16 logit, 32 otherwise.
    key_bits: int
    #: Can a row be counted (a top-k of at most _FILTER_COUNT_MAX_K alone)?
    #: Only then does the stats pass keep run maxima and the count launch.
    counts: bool
    #: Can one be descended?
    descends: bool
    #: Must the descent's passes weigh (the batch has a top-p)?
    weigh: bool
    #: Does the batch hand over any filter at all (the verify's ``FILTER``)?
    filtered: bool


def row_filters(
    top_ks: int | torch.Tensor | None,
    top_ps: float | torch.Tensor | None,
    batch_size: int,
    logits_dtype: torch.dtype,
) -> RowFilters:
    """How a batch filters. ``top_ks``/``top_ps`` are what `Sampler` is
    given: None (no row filters), one value for the batch, or one per
    request, a top-k of -1 meaning none."""
    k_rows = isinstance(top_ks, torch.Tensor)
    p_rows = isinstance(top_ps, torch.Tensor)
    for name, rows_of in (("top_ks", top_ks), ("top_ps", top_ps)):
        assert not isinstance(rows_of, torch.Tensor) or rows_of.shape == (
            batch_size,
        ), f"{name} is per request: {tuple(rows_of.shape)} for {batch_size}"
    request_filter = (
        top_ks.to(torch.int32).contiguous() if k_rows else None,
        0 if k_rows or top_ks is None else int(top_ks),
        top_ps.to(torch.float32).contiguous() if p_rows else None,
        # Above 1 is no top-p.
        2.0 if p_rows or top_ps is None else float(top_ps),
    )
    # A bf16 logit is the top half of its fp32 bits: 16 key bits, not 32.
    key_bits = 16 if logits_dtype == torch.bfloat16 else 32
    # Can a row be counted? And can one be descended? Neither question needs
    # the device: a per-row tensor may hold either, a batch's one value is one
    # or the other.
    p_scalar = None if p_rows or top_ps is None else float(top_ps)
    k_scalar = None if k_rows or top_ks is None else int(top_ks)
    if p_scalar is not None and p_scalar >= 1.0:
        p_scalar = None
    k_small = k_scalar is not None and 0 < k_scalar <= _FILTER_COUNT_MAX_K
    return RowFilters(
        request_filter,
        k_rows,
        p_rows,
        key_bits,
        counts=(k_rows or k_small) and p_scalar is None,
        descends=not (k_small and not p_rows and p_scalar is None),
        # Only a top-p needs the passes to weigh; counting alone is cheaper.
        weigh=top_ps is not None,
        filtered=top_ks is not None or top_ps is not None,
    )


def _count_run_cells(num_tokens: int, vocab_size: int, stats_block_v: int) -> int:
    """Cells of each of the counted top-k's two int32 arrays: every run's
    max, and the id at it."""
    return num_tokens * triton.cdiv(vocab_size, stats_block_v) * _FILTER_COUNT_GROUPS


class CountRuns(NamedTuple):
    """The counted top-k's run maxima and the ids at each, which the stats
    pass keeps per tile: ``num_tokens * V / 8`` bytes in all."""

    maxima: torch.Tensor  # [cells] int32, keyed
    ids: torch.Tensor  # [cells] int32


class FilterScratch(NamedTuple):
    """`ratio_target_p.filter_floor`'s work buffers."""

    filter_f32: torch.Tensor  # [2, rows, _FILTER_F32_WIDTH] fp32
    floor_mass: torch.Tensor  # [mass tiles, rows] fp32
    filter_i32: torch.Tensor  # [2, rows, _FILTER_I32_WIDTH] int32
    #: A row of last-digit counts per (row, split) of the last count.
    floor_counts: torch.Tensor
    runs: CountRuns


def filter_scratch(
    num_tokens: int, vocab_size: int, stats_block_v: int
) -> FilterScratch:
    """The regions `ratio_target_p.filter_floor` needs, the run maxima of a
    stats pass over ``stats_block_v`` tiles included."""
    f32, i32 = torch.float32, torch.int32
    cells = _count_run_cells(num_tokens, vocab_size, stats_block_v)
    return FilterScratch(
        Region((2, num_tokens, _FILTER_F32_WIDTH), f32),
        Region((triton.cdiv(vocab_size, _FILTER_BLOCK_V), num_tokens), f32),
        Region((2, num_tokens, _FILTER_I32_WIDTH), i32),
        Region((max(num_tokens, _FILTER_FLOOR_PROGRAMS) * _FILTER_LAST_BINS,), i32),
        CountRuns(Region((cells,), i32), Region((cells,), i32)),
    )
