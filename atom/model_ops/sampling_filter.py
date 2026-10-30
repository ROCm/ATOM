"""The ratio verify's top-k / top-p floor, found on the device.

The probability-ratio verify (`rejection_sampler`) serves a filtered request
`Sampler`'s filtered p, p_f, at every position -- its bonus token is
`Sampler`'s own draw. `Sampler` filters with aiter (`top_k_renorm_probs`,
`top_[k_]top_p_sampling_from_probs`), whose rule is: id t survives iff fewer than top_k ids score strictly above it and
their p sums to less than top_p, both over the unfiltered p (an
intersection, not top-p over the top-k survivors). A tie group survives or
falls whole, and p_f is p over the survivors, renormalized. So the survivors
are every id at or above one logit, the row's floor, found without leaving
the device by a radix descent over the logits' bits (`_ratio_filter_*`); the
accept, the race and its fallback all read p as 0 below it. Both rules agree
with aiter's to fp32 rounding, not bit for bit. The count: the descent ranks
ids by logit, aiter by `Sampler`'s fp32 p, and the subtraction in its
``softmax(logits / T)`` can round two distinct logits onto one p, which aiter
keeps or drops as a tie and the floor may split. That takes a gap under one
fp32 ulp of ``max - logit / T``, max the row's largest ``logit / T`` (for bf16
logits, both within about ``max * 2**-15`` of 0), so such ids carry about
``exp(-max)`` of mass each; ranking by that difference instead would cost a
bf16 row the 32-bit descent, and would still leave the exp's own rounding. The
mass: it compares a sum against top_p, so a tie group whose mass above lies
within rounding of top_p may survive in one and not the other. The bonus draw
must keep tie groups too: for a scalar top-k with a top-p at a large vocab,
aiter's joint op cuts exactly k ids first, so with the flag on every `Sampler`
draw keeps them (`Sampler.keep_top_k_ties`: the bonus, and the first token
after a prefill, so a request follows one law throughout). Only a batch with a
filter pays for any of it: three more reads of its rows (bf16 logits) -- but
a row whose filter is a top-k of at most 256 alone is not descended. Its
floor is just its k-th largest logit, so the stats pass also keeps the max
of every run of 64 ids and one program per row counts the k-th largest
among the fewer than k runs whose max is above the k-th largest max
(`_ratio_filter_count_kernel`):
the same floor, counted exactly, from one read of the logits.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.model_ops.sampler import SAMPLER_EPS

# Logit-key bits one filter pass resolves, so each pass bins by this many. The
# last 8 bits are resolved apart (`_ratio_filter_floor_kernel` counts them,
# `_ratio_filter_cut_kernel` picks the floor among them): each of their
# bins is one logit, so its mass is its count times that logit's p, and
# counting is all that pass does. A bf16 logit's 16 bits take two passes and
# the floor kernels, an fp32 one's 32 take six.
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
# the filter kernels walk in -- a small one where a pass weighs, which bins
# every id into all of its bins at once.
_FILTER_SPLIT = 8
_FILTER_BLOCK_V = 4096
_FILTER_WEIGH_BLOCK_V = 256
_FILTER_WEIGH_WARPS = 4
# A top-k of at most this many with no top-p is counted, not descended: its
# floor is the row's k-th largest logit, which `_ratio_filter_count_kernel`
# finds among the fewer than k runs of ids whose max the stats pass kept
# above the k-th largest such max, so such a batch reads its logits once
# (`_counted_row`).
_FILTER_COUNT_MAX_K = 256
# Runs per 4096-id tile whose max the stats pass keeps (runs of 64 ids), the
# runs the count kernel's smallest gather covers (then 4x that, then
# _FILTER_COUNT_MAX_K: at most k - 1 runs are read), and its warps.
_FILTER_COUNT_GROUPS = 64
_FILTER_COUNT_CHUNK = 32
_FILTER_COUNT_WARPS = 4
# Per-row record of one filter pass: each program's bin counts (or masses),
# then the descent so far -- (key prefix, count above) in int32, the mass
# above in fp32.
_FILTER_I32_WIDTH = _FILTER_SPLIT * _FILTER_BINS + 2
_FILTER_F32_WIDTH = _FILTER_SPLIT * _FILTER_BINS + 1
# The verify runs eagerly at whatever batch arrives. Triton would specialize
# its kernels on the row and request counts (== 1, a multiple of 16) and so
# compile each once per class of batch, mid-serve, the first time that class
# arrives. The counts only bound loops and index, so none is specialized on;
# with every scratch view 16-byte aligned (`take`), one compile per kernel
# variant serves every batch.
_BATCH_COUNTS = ["num_tokens", "batch_size"]


@triton.jit
def _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps):
    """The request a draft row belongs to: the first ``r`` with
    ``cu_num_draft_tokens[r] > row`` (a binary search, so requests with no
    drafts are skipped). ``search_steps`` >= ceil(log2(batch_size)) + 1."""
    lo = 0
    hi = batch_size - 1
    for _ in range(search_steps):
        mid = (lo + hi) // 2
        right = tl.load(cu_num_draft_tokens_ptr + mid) <= row
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    return lo


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
def _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, MAX_K):
    """Is this a filtered row the radix descent finds the floor of? Every one
    but a `_counted_row` (``MAX_K`` 0 when the batch counts none)."""
    return _is_filtered(temp, top_k, top_p, vocab_size, greedy_temperature) & ~(
        _counted_row(temp, top_k, top_p, vocab_size, greedy_temperature, MAX_K)
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
    state = ((done_pass + 1) % 2 * num_tokens + row).to(tl.int64)
    recorded = has_pass & (done_pass > 0)
    prefix = tl.load(
        filter_i32_ptr + state * I32_WIDTH + SPLIT * BINS, mask=recorded, other=0
    ).to(tl.uint32, bitcast=True)
    above_n = tl.load(
        filter_i32_ptr + state * I32_WIDTH + SPLIT * BINS + 1, mask=recorded, other=0
    )
    above_m = tl.load(
        filter_f32_ptr + state * F32_WIDTH + SPLIT * BINS, mask=recorded, other=0.0
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
        state = ((filter_pass + 1) % 2 * num_tokens + row).to(tl.int64)
        record = has_pass & (split == 0)
        tl.store(
            filter_i32_ptr + state * I32_WIDTH + SPLIT * BINS,
            prefix.to(tl.int32, bitcast=True),
            mask=record,
        )
        tl.store(
            filter_i32_ptr + state * I32_WIDTH + SPLIT * BINS + 1, above_n, mask=record
        )
        tl.store(
            filter_f32_ptr + state * F32_WIDTH + SPLIT * BINS, above_m, mask=record
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
):
    """One (draft row, vocab split): the last pass's step of the descent,
    then count the ids under the prefix by the key's last ``LAST_BITS``.
    Unless the passes ``WEIGHED`` (and so carry the mass above the prefix
    down the descent), also weigh each tile's ids above the prefix: p_f's
    normalizer needs that mass either way. A tile with none skips the
    exponentials."""
    LAST_BINS: tl.constexpr = 1 << LAST_BITS
    row = tl.program_id(0)
    split = tl.program_id(1)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    top_k, top_p = _request_filter(
        top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
    )
    if _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, COUNTED):
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
            under = vmask & ((key >> LAST_BITS) == prefix)
            counts += tl.histogram(
                (key & (LAST_BINS - 1)).to(tl.int32), LAST_BINS, mask=under
            )
            if not WEIGHED:
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
):
    """One draft row: the floor, from the last count's splits added in
    order. Each of its bins is a single logit, so counting weighs it too
    (count times that logit's p) and the floor is exact. p_f's normalizer is
    the mass at or above the floor. An unfiltered row's floor is -inf and its
    p is left as it was."""
    LAST_BINS: tl.constexpr = 1 << LAST_BITS
    row = tl.program_id(0)
    req_idx = _request_of_row(cu_num_draft_tokens_ptr, row, batch_size, search_steps)
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    top_k, top_p = _request_filter(
        top_k_ptr, top_k_val, top_p_ptr, top_p_val, req_idx, K_PER_ROW, P_PER_ROW
    )
    floor = tl.full((), float("-inf"), tl.float32)
    if _descended_row(temp, top_k, top_p, vocab_size, greedy_temperature, COUNTED):
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
        counts = tl.zeros((LAST_BINS,), dtype=tl.int32)
        for split in range(splits):
            cell = (row * splits + split).to(tl.int64) * LAST_BINS
            counts += tl.load(floor_counts_ptr + cell + bins)
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
    CHUNK: tl.constexpr,
    MAX_RUNS: tl.constexpr,
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
        if n_runs <= CHUNK:
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
                CHUNK,
                KEY_BITS,
            )
        elif n_runs <= 4 * CHUNK:
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
                4 * CHUNK,
                KEY_BITS,
            )
        else:
            # At most k - 1 < MAX_RUNS runs.
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
                MAX_RUNS,
                KEY_BITS,
            )
        floor = _key_logit(cut, KEY_BITS)
        tl.store(row_inv_z_ptr + row, inv_z / mass)
    tl.store(row_floor_ptr + row, floor)


class RowFilters(NamedTuple):
    """A batch's ``(top_ks, top_ps)`` as the filter kernels read them."""

    #: `_request_filter`'s ``(top_k_ptr, top_k_val, top_p_ptr, top_p_val)``.
    args: tuple
    k_rows: bool
    p_rows: bool
    #: Bits of a logit's key: 16 for a bf16 logit, 32 otherwise.
    key_bits: int
    #: Can a row be counted (a top-k of at most _FILTER_COUNT_MAX_K alone)?
    counts: bool
    #: Can one be descended?
    descends: bool
    #: Must the descent's passes weigh (the batch has a top-p)?
    weigh: bool


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
    )


def filter_floor(
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    cu_num_draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped
    filters: RowFilters,
    rows: tuple[torch.Tensor, ...],
    parts: tuple[torch.Tensor, ...],
    stats_block_v: int,
) -> None:
    """Each filtered draft row's floor, and its p's normalizer renormalized
    over the survivors.

    Fills ``row_floor`` and rewrites ``row_inv_z`` of ``rows = (row_max,
    row_inv_z, row_floor)``, through the work buffers ``parts = (filter_f32,
    filter_i32, floor_mass, floor_counts, run_max, run_at_max)``. The last two
    are the run maxima, and the ids at each, that the stats pass kept per
    ``stats_block_v`` tile in a ``COUNT`` launch; ``row_max`` and ``row_inv_z``
    are its combined statistics."""
    num_tokens, vocab_size = target_logits.shape
    batch_size = cu_num_draft_tokens.shape[0]
    n_tiles = triton.cdiv(vocab_size, stats_block_v)
    stride = target_logits.stride(0)
    search_steps = batch_size.bit_length() + 1
    row_max, row_inv_z, row_floor = rows
    filter_f32, filter_i32, floor_mass, floor_counts, run_max, run_at_max = parts
    request_filter = filters.args
    key_bits = filters.key_bits
    k_rows, p_rows = filters.k_rows, filters.p_rows
    counted = _FILTER_COUNT_MAX_K if filters.counts else 0
    if filters.counts:
        _ratio_filter_count_kernel[(num_tokens,)](
            target_logits,
            cu_num_draft_tokens,
            temperatures,
            *request_filter,
            row_max,
            row_inv_z,
            row_floor,
            run_max,
            run_at_max,
            num_tokens,
            vocab_size,
            stride,
            batch_size,
            search_steps,
            SAMPLER_EPS,
            BLOCK_V=stats_block_v,
            BLOCK_RUNS=triton.next_power_of_2(n_tiles * _FILTER_COUNT_GROUPS),
            GROUPS=_FILTER_COUNT_GROUPS,
            MAX_K=_FILTER_COUNT_MAX_K,
            CHUNK=_FILTER_COUNT_CHUNK,
            MAX_RUNS=max(
                triton.next_power_of_2(_FILTER_COUNT_MAX_K), 4 * _FILTER_COUNT_CHUNK
            ),
            KEY_BITS=key_bits,
            K_PER_ROW=k_rows,
            P_PER_ROW=p_rows,
            num_warps=_FILTER_COUNT_WARPS,
        )
    if not filters.descends:
        return
    layout = {
        "KEY_BITS": key_bits,
        "SPLIT": _FILTER_SPLIT,
        "DIGIT_BITS": _FILTER_DIGIT_BITS,
        "I32_WIDTH": _FILTER_I32_WIDTH,
        "F32_WIDTH": _FILTER_F32_WIDTH,
        "K_PER_ROW": k_rows,
        "P_PER_ROW": p_rows,
        "COUNTED": counted,
    }
    passes = (key_bits - _FILTER_LAST_BITS) // _FILTER_DIGIT_BITS
    weigh = filters.weigh
    block_v = _FILTER_WEIGH_BLOCK_V if weigh else _FILTER_BLOCK_V
    for filter_pass in range(passes):
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
        passes - 1,
        tiles_per_split,
        **floor,
        num_warps=8,
    )
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
        num_warps=4,
    )
