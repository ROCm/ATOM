from typing import NamedTuple

import torch
import triton
import triton.language as tl
from aiter import topk_select
from torch import nn

from atom.model_ops.dspark_markov_sample import SparseQStore
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.sampling_scratch import SamplingScratch, take
from atom.model_ops.softmax_tiles import softmax_combine_kernel, tile_max_sumexp
from atom.utils import envs
from atom.utils.forward_context import SpecDecodeMetadata

ATOM_ENABLE_RELAXED_MTP = envs.ATOM_ENABLE_RELAXED_MTP
if ATOM_ENABLE_RELAXED_MTP:
    RELAXED_TOP_N = 10
    RELAXED_DELTA = 0.6
else:
    RELAXED_TOP_N = 1
    RELAXED_DELTA = 0.0

# --- Synthetic (forced) acceptance for spec-decode benchmarking ---
# Enabled via --spec-decode-acceptance-length / --spec-decode-acceptance-rate
# (SpeculativeConfig.synthetic_acceptance_rates, already resolved to per-position
# rates by the config). When set, the rejection sampler ignores the real
# draft/target comparison and force-accepts draft tokens so the measured mean
# acceptance length converges to the configured value. Purely a benchmarking /
# bring-up knob (e.g. while an MTP/EAGLE head is still training, or to replay a
# published acceptance-length curve). Mirrors vLLM's "synthetic"
# rejection_sample_method. See ROCm/ATOM#555.
#
# Cache {(rates, device): conditional-rate tensor} — the kernel walks positions
# sequentially, so it needs P(accept i | accepted through i-1) rather than the
# unconditional rates the config carries.
_SYNTHETIC_COND_CACHE: dict[tuple[tuple[float, ...], torch.device], torch.Tensor] = {}
# Base seed for the per-step synthetic RNG. The forced accept/reject draw MUST be
# identical on every TP rank: sampled_tokens is broadcast from rank 0 while
# num_bonus_tokens stays local, so a per-rank torch.rand would desync the two and
# make the anchor gather read a rejected (-1) column -> the draft model then
# embeds an invalid id (HSA out-of-bounds). A dedicated device generator re-seeded
# from the step counter gives that: same Philox stream, so bit-identical uniforms
# on every rank, and isolated from whatever else consumes the global RNG.
_SYNTHETIC_RNG_BASE_SEED = 0x5EED
# One generator per device, kept alive so the draw costs a kernel and nothing else.
_SYNTHETIC_GENERATORS: dict[torch.device, torch.Generator] = {}


def _synthetic_generator(device: torch.device) -> torch.Generator:
    generator = _SYNTHETIC_GENERATORS.get(device)
    if generator is None:
        generator = torch.Generator(device=device)
        _SYNTHETIC_GENERATORS[device] = generator
    return generator


# ---- probability-ratio verification ----------------------------------------
#
# Accept draft x at a position iff `u * q(x) < p(x)`, and on the first rejection
# draw the replacement from the residual `relu(p - q)` (normalized). p is the
# target's `softmax(logits / T)`, q the distribution the draft drew x from.
#
# Nothing `[rows, V]` is materialized for p. Every (draft row, V tile) program
# reduces its tile of the row to the softmax statistics
# (`softmax_tiles`) plus the row's raw-logit argmax; the accept walk then needs
# p at one id per position, `exp(l(x) / T - max) / Z`, and only a rejecting row
# is read across V again, by the residual race.
#
# q arrives sparse (`SparseQStore`: each draft row's stored ids, their
# probabilities and the slot of the id it emitted), and is 0 off the stored
# set. The accept walk reads q(x) at the emitted slot. The residual needs q at
# every id, so each rejecting request's one row is first spread into a dense
# scratch row of its own (zeroed, then the stored entries written at their
# ids); the race and its fallback then read that row as they would a dense q.
#
# Residual: an exponential race rather than a CDF walk. `argmax_v r(v) / E_v`
# with i.i.d. `E_v ~ Exp(1)` selects v with probability `r(v) / sum r`, so the
# residual needs no normalizer, no prefix sum, and no agreement between two
# reductions' rounding; tiles race in parallel and a cross-tile max decides.
#
# Top-k / top-p: a filtered request is served `Sampler`'s filtered p, p_f, at
# every position -- its bonus token is `Sampler`'s own draw. `Sampler` filters
# with aiter (`top_k_renorm_probs`, `top_[k_]top_p_sampling_from_probs`), whose
# rule is: id t survives iff fewer than top_k ids score strictly above it and
# their p sums to less than top_p, both over the unfiltered p (an
# intersection, not top-p over the top-k survivors). A tie group survives or
# falls whole, and p_f is p over the survivors, renormalized. So the survivors
# are every id at or above one logit, the row's floor, found without leaving
# the device by a radix descent over the logits' bits (`_ratio_filter_*`); the
# accept, the race and its fallback all read p as 0 below it. Both rules agree
# with aiter's to fp32 rounding, not bit for bit. The count: the descent ranks
# ids by logit, aiter by `Sampler`'s fp32 p, and the subtraction in its
# ``softmax(logits / T)`` can round two distinct logits onto one p, which aiter
# keeps or drops as a tie and the floor may split. That takes a gap under one
# fp32 ulp of ``max - logit / T``, max the row's largest ``logit / T`` (for bf16
# logits, both within about ``max * 2**-15`` of 0), so such ids carry about
# ``exp(-max)`` of mass each; ranking by that difference instead would cost a
# bf16 row the 32-bit descent, and would still leave the exp's own rounding. The
# mass: it compares a sum against top_p, so a tie group whose mass above lies
# within rounding of top_p may survive in one and not the other. The bonus draw
# must keep tie groups too: for a scalar top-k with a top-p at a large vocab,
# aiter's joint op cuts exactly k ids first, so with the flag on every `Sampler`
# draw keeps them (`Sampler.keep_top_k_ties`: the bonus, and the first token
# after a prefill, so a request follows one law throughout). Only a batch with a
# filter pays for any of it: three more reads of its rows (bf16 logits) -- but
# a row whose filter is a top-k of at most 256 alone is not descended. Its
# floor is just its k-th largest logit, so the stats pass also keeps the max
# of every run of 64 ids and one program per row counts the k-th largest
# among the fewer than k runs whose max is above the k-th largest max
# (`_ratio_filter_count_kernel`):
# the same floor, counted exactly, from one read of the logits.
#
# Greedy rows (temperature at or below SAMPLER_EPS) take exact match inside the
# same launch: accept iff the draft is the target's lowest-id argmax, emit that
# argmax on rejection. That is `rejection_greedy_sample_kernel`'s rule, so a
# greedy request's tokens do not depend on whether sampled requests share its
# batch. The ratio rule would not give that: on a bf16 logit tie p is 0.5/0.5,
# not a delta.
#
# Randomness: Philox via `tl.rand`, keyed on a per-step SPMD counter, so every
# TP rank draws the same uniforms. The key's high word names the purpose --
# 0 is the draft's noise (`dspark_markov_sample`), and the two below are the
# accept coins and the residual race -- so no two consumers can ever share a
# stream, whatever their step counters do. That holds for 64-bit offsets too:
# `tl.rand` goes through `randint4x`, which always splits the offset into two
# 32-bit Philox counters, so `philox` keeps the seed's high word as the second
# key word (only a direct 64-bit-counter `philox` call would drop it). Pinned by
# `test_the_key_purpose_word_separates_streams_at_any_offset_width`.
_RATIO_ACCEPT_KEY = 1 << 32
_RATIO_RESIDUAL_KEY = 2 << 32

# Vocab tile every ratio kernel walks in.
_RATIO_BLOCK_V = 4096
# Stored q entries one lookup or densify iteration covers.
_RATIO_BLOCK_Q = 1024
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


class RatioDraftQ(NamedTuple):
    """The q a ratio verify reads: the draft's sparse store, plus a dense
    ``[>= batch_size, V]`` fp32 scratch a rejecting request's q row is spread
    into for the residual. Its contents are overwritten; nothing else may need
    them across the verify."""

    store: SparseQStore
    dense: torch.Tensor


def _count_run_cells(num_tokens: int, vocab_size: int) -> int:
    """Cells of each of the counted top-k's two int32 arrays: every run's
    max, and the id at it."""
    return num_tokens * triton.cdiv(vocab_size, _RATIO_BLOCK_V) * _FILTER_COUNT_GROUPS


def ratio_count_scratch_bytes(num_tokens: int, vocab_size: int) -> int:
    """Bytes of `ratio_verify_scratch` at ``num_tokens`` draft rows that the
    counted top-k's run maxima and their ids take, reserved whether or not a
    request filters: about ``num_tokens * V / 8``, 160 MiB at
    ``max_num_seqs=512``, T=16, V=163840."""
    return 2 * _count_run_cells(num_tokens, vocab_size) * 4


def _ratio_scratch_shapes(num_tokens: int, batch_size: int, vocab_size: int):
    """fp32 and int32 shapes one ratio verify needs, in `_ratio_sample` order."""
    tiles = triton.cdiv(vocab_size, _RATIO_BLOCK_V)
    f32 = [
        (num_tokens,),  # row_max
        (num_tokens,),  # row_inv_z
        (num_tokens,),  # row_floor
        (tiles, num_tokens),  # part_max
        (tiles, num_tokens),  # part_sum
        (tiles, num_tokens),  # part_amax_val
        (tiles, batch_size),  # part_key
        (2, num_tokens, _FILTER_F32_WIDTH),  # filter_f32
        (tiles, num_tokens),  # floor_mass
    ]
    i32 = [
        (num_tokens,),  # row_argmax
        (tiles, num_tokens),  # part_amax_idx
        (batch_size,),  # residual_row
        (batch_size,),  # residual_q_row
        (tiles, batch_size),  # part_idx
        (2, num_tokens, _FILTER_I32_WIDTH),  # filter_i32
        # One row of last-digit counts per (row, split) of the last count,
        # then the run maxima of the counted rows and the ids at each.
        (
            max(num_tokens, _FILTER_FLOOR_PROGRAMS) * _FILTER_LAST_BINS
            + 2 * _count_run_cells(num_tokens, vocab_size),
        ),  # floor_counts
    ]
    return f32, i32


def ratio_verify_scratch(
    max_bs: int, num_spec_steps: int, vocab_size: int, device
) -> SamplingScratch:
    """Scratch for any ratio verify of up to ``max_bs`` requests of
    ``num_spec_steps`` drafts each, allocated once."""
    f32, i32 = _ratio_scratch_shapes(max_bs * num_spec_steps, max_bs, vocab_size)
    return SamplingScratch.for_shapes(f32, i32, device)


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
    raw_max = tl.max(x, axis=0)
    raw_idx = tl.min(tl.where((x == raw_max) & vmask, offs, 2147483647), axis=0)
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


@triton.jit
def _stored_q_of(q_ids_ptr, q_probs_ptr, q_row, n, x, BLOCK_Q: tl.constexpr):
    """x's q in one stored row: its probability among the first ``n``
    entries, 0 when it is not one of them."""
    q = 0.0
    for start in range(0, n, BLOCK_Q):
        offs = start + tl.arange(0, BLOCK_Q)
        mask = offs < n
        ids = tl.load(q_ids_ptr + q_row + offs, mask=mask, other=-1)
        probs = tl.load(q_probs_ptr + q_row + offs, mask=mask, other=0.0)
        q += tl.sum(tl.where(mask & (ids == x), probs, 0.0), axis=0)
    return q


@triton.jit(do_not_specialize=["num_spec_steps", "q_width", "accept_seed"])
def _ratio_accept_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    residual_row_ptr,  # [batch_size] int32 out: the row to resample, or -1
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    logits_ptr,  # [num_tokens, V] raw target logits
    temperatures_ptr,  # [batch_size] fp32, pre-clamped
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32
    row_argmax_ptr,  # [num_tokens] int32, lowest id
    row_floor_ptr,  # [num_tokens] fp32, the lowest logit p_f keeps (FILTER)
    q_ids_ptr,  # [num_q_rows, q_width] int32, the stored ids of each q row
    q_probs_ptr,  # [num_q_rows, q_width] fp32, their q
    q_meta_ptr,  # [num_q_rows, 2] int32: (n, slot of the emitted id)
    q_row_base_ptr,  # [batch_size] int32, this request's first q row, or < 0
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
    num_spec_steps,
    q_width,
    logits_row_stride,
    greedy_temperature,
    accept_seed,
    BLOCK_T: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    FILTER: tl.constexpr,
):
    """One request: walk its drafts until the first rejection.

    Leaves the rejecting position's output to the residual kernels for a
    sampled request, and fills it with the target argmax for a greedy one.
    """
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    greedy = temp <= greedy_temperature
    # A negative base means this request has no live q (never drafted here, or
    # its slot expired). Its q reads as 0, so its first draft is rejected and
    # the residual is p itself: exact, at the cost of one step's speculation.
    # Not 0 for "none": row 0 is in range and holds a stranger's q > 0, against
    # which `u * q < p` would accept.
    q_base = tl.load(q_row_base_ptr + req_idx)
    has_q = q_base >= 0
    out_row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)

    # Every position's (n, xslot) and the entry at xslot, in vector loads up
    # front, so the walk below does not chain three dependent loads per
    # position. A slot outside the stored entries reads as no entry.
    offs_t = tl.arange(0, BLOCK_T)
    t_mask = (offs_t < num_draft_tokens) & has_q & (temp > greedy_temperature)
    q_rows = (q_base + offs_t).to(tl.int64)
    stored_n = tl.load(q_meta_ptr + q_rows * 2, mask=t_mask, other=0)
    stored_n = tl.minimum(tl.maximum(stored_n, 0), q_width)
    xslots = tl.load(q_meta_ptr + q_rows * 2 + 1, mask=t_mask, other=-1)
    slot_mask = t_mask & (xslots >= 0) & (xslots < stored_n)
    slot_ids = tl.load(q_ids_ptr + q_rows * q_width + xslots, mask=slot_mask, other=-1)
    slot_qs = tl.load(
        q_probs_ptr + q_rows * q_width + xslots, mask=slot_mask, other=0.0
    )

    INVALID_TOKEN: tl.constexpr = -1
    rejected = False
    reject_pos = -1
    last_checked = -1
    for pos in range(num_draft_tokens):
        if rejected:
            tl.store(out_row + pos, INVALID_TOKEN)
        else:
            row = start_idx + pos
            x = _draft_token(input_ids_ptr, target_logits_indices_ptr, row)
            target_argmax = tl.load(row_argmax_ptr + row)
            if greedy:
                accept = x == target_argmax
            else:
                logit = tl.load(
                    logits_ptr + row.to(tl.int64) * logits_row_stride + x
                ).to(tl.float32)
                p = tl.exp(logit / temp - tl.load(row_max_ptr + row)) * tl.load(
                    row_inv_z_ptr + row
                )
                if FILTER:
                    # Filtered out: p_f(x) is 0, so x is rejected whatever q.
                    p = tl.where(logit >= tl.load(row_floor_ptr + row), p, 0.0)
                at_pos = offs_t == pos
                q = tl.sum(tl.where(at_pos, slot_qs, 0.0), axis=0)
                if tl.sum(tl.where(at_pos, slot_ids, 0), axis=0) != x:
                    # The emitted slot does not hold x (or there is no q).
                    # Not a reject on that alone: the rule needs x's true q,
                    # so look x up among the stored entries (0 if absent).
                    q = _stored_q_of(
                        q_ids_ptr,
                        q_probs_ptr,
                        (q_base + pos).to(tl.int64) * q_width,
                        tl.sum(tl.where(at_pos, stored_n, 0), axis=0),
                        x,
                        BLOCK_Q,
                    )
                u = tl.rand(accept_seed, row)
                # x was drawn from q, so q(x) must be a positive probability;
                # anything else is not the row x came from. `u * q < p` rather
                # than `u < p / q`: no division, so a denormal q cannot become
                # an infinite ratio.
                accept = (q > 0.0) & (q <= 1.0) & (u * q < p)
            if accept:
                tl.store(out_row + pos, tl.cast(x, tl.int32))
            else:
                rejected = True
                reject_pos = pos
                if greedy:
                    tl.store(out_row + pos, tl.cast(target_argmax, tl.int32))
            last_checked += 1

    residual_row = -1
    if rejected and not greedy:
        residual_row = start_idx + reject_pos
    tl.store(residual_row_ptr + req_idx, residual_row)
    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        last_checked,
        rejected,
        num_spec_steps,
    )


@triton.jit
def _ratio_densify_clear_kernel(
    residual_row_ptr,  # [batch_size] int32, the rejecting row, or -1
    q_row_base_ptr,  # [batch_size] int32
    dense_ptr,  # [>= batch_size, V] fp32 out: request r's q row at row r
    vocab_size,
    BLOCK_V: tl.constexpr,
):
    """One (request, V tile): zero the dense row of a rejecting request that
    has a q; every other program returns at once."""
    req_idx = tl.program_id(0)
    tile = tl.program_id(1)
    row = tl.load(residual_row_ptr + req_idx)
    if (row >= 0) & (tl.load(q_row_base_ptr + req_idx) >= 0):
        offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        tl.store(
            dense_ptr + req_idx.to(tl.int64) * vocab_size + offs,
            tl.zeros((BLOCK_V,), dtype=tl.float32),
            mask=offs < vocab_size,
        )


@triton.jit(do_not_specialize=["q_width"])
def _ratio_densify_scatter_kernel(
    residual_row_ptr,  # [batch_size] int32, the rejecting row, or -1
    cu_num_draft_tokens_ptr,  # [batch_size]
    q_ids_ptr,  # [num_q_rows, q_width] int32
    q_probs_ptr,  # [num_q_rows, q_width] fp32
    q_meta_ptr,  # [num_q_rows, 2] int32
    q_row_base_ptr,  # [batch_size] int32
    dense_ptr,  # [>= batch_size, V] fp32, zeroed where it will be read
    residual_q_row_ptr,  # [batch_size] int32 out: the dense row the residual reads
    q_width,
    vocab_size,
    BLOCK_Q: tl.constexpr,
):
    """One request: write the rejecting row's stored entries into its zeroed
    dense row, and name that row for the residual race. -1 for every other
    request, so the race reads q as 0 where there is none (as the accept walk
    did)."""
    req_idx = tl.program_id(0)
    row = tl.load(residual_row_ptr + req_idx)
    q_base = tl.load(q_row_base_ptr + req_idx)
    residual_q_row = -1
    if (row >= 0) & (q_base >= 0):
        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        pos = row - start_idx
        q_row = (q_base + pos).to(tl.int64)
        n = tl.minimum(tl.maximum(tl.load(q_meta_ptr + q_row * 2), 0), q_width)
        dense_row = dense_ptr + req_idx.to(tl.int64) * vocab_size
        for start in range(0, n, BLOCK_Q):
            offs = start + tl.arange(0, BLOCK_Q)
            mask = offs < n
            ids = tl.load(q_ids_ptr + q_row * q_width + offs, mask=mask, other=-1)
            probs = tl.load(q_probs_ptr + q_row * q_width + offs, mask=mask)
            # An id outside the vocab is no q at all; never a write past the row.
            tl.store(
                dense_row + ids, probs, mask=mask & (ids >= 0) & (ids < vocab_size)
            )
        residual_q_row = req_idx
    tl.store(residual_q_row_ptr + req_idx, residual_q_row)


@triton.jit(do_not_specialize=["batch_size", "residual_seed"])
def _ratio_residual_race_kernel(
    residual_row_ptr,  # [batch_size] int32, the rejecting row, or -1
    logits_ptr,  # [num_tokens, V] raw target logits
    temperatures_ptr,  # [batch_size] fp32
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32
    row_floor_ptr,  # [num_tokens] fp32, the lowest logit p_f keeps (FILTER)
    q_ptr,  # [num_q_rows, V] fp32
    q_row_ptr,  # [batch_size] int32, the rejecting position's q row, or < 0
    part_key_ptr,  # [n_tiles, batch_size] fp32: best residual key
    part_idx_ptr,  # [n_tiles, batch_size] int32: its lowest id
    batch_size,
    vocab_size,
    logits_row_stride,
    residual_seed,
    BLOCK_V: tl.constexpr,
    FILTER: tl.constexpr,
):
    """One (request, V tile) of the residual race ``relu(p_f - q) / E``, for
    rejecting requests; every other program returns at once. Launched for
    every request because which ones rejected is only known on the device,
    and finding out on the host would stall the verify on a readback."""
    req_idx = tl.program_id(0)
    tile = tl.program_id(1)
    row = tl.load(residual_row_ptr + req_idx)
    if row >= 0:
        temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
        q_row = tl.load(q_row_ptr + req_idx)
        has_q = q_row >= 0
        offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        vmask = offs < vocab_size
        x = tl.load(
            logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
            mask=vmask,
            other=float("-inf"),
        ).to(tl.float32)
        p = tl.exp(x / temp - tl.load(row_max_ptr + row)) * tl.load(row_inv_z_ptr + row)
        p = tl.where(vmask, p, 0.0)
        if FILTER:
            p = tl.where(x >= tl.load(row_floor_ptr + row), p, 0.0)
        q = tl.load(
            q_ptr + q_row.to(tl.int64) * vocab_size + offs,
            mask=vmask & has_q,
            other=0.0,
        )
        # Anything that is not a probability (NaN, inf, negative) counts as 0,
        # so the residual falls back toward p rather than inheriting garbage.
        q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
        residual = tl.maximum(p - q, 0.0)
        u = tl.rand(residual_seed, row.to(tl.int64) * vocab_size + offs)
        # u in [0, 1): E = -log(u) is in (0, inf], so a key is finite and a
        # zero-mass id keys exactly 0.
        e = -tl.log(u)
        key = tl.where(vmask, residual / e, -1.0)
        best = tl.max(key, axis=0)
        part = tile * batch_size + req_idx
        tl.store(part_key_ptr + part, best)
        tl.store(
            part_idx_ptr + part,
            tl.min(tl.where((key == best) & vmask, offs, 2147483647), axis=0),
        )


@triton.jit(do_not_specialize=["batch_size", "num_spec_steps", "residual_seed"])
def _ratio_residual_pick_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    residual_row_ptr,  # [batch_size] int32
    cu_num_draft_tokens_ptr,  # [batch_size]
    part_key_ptr,
    part_idx_ptr,
    logits_ptr,  # [num_tokens, V] raw target logits, for the fallback only
    temperatures_ptr,  # [batch_size] fp32
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32
    row_argmax_ptr,  # [num_tokens] int32, lowest id
    row_floor_ptr,  # [num_tokens] fp32, the lowest logit p_f keeps (FILTER)
    batch_size,
    n_tiles,
    num_spec_steps,
    vocab_size,
    logits_row_stride,
    residual_seed,
    BLOCK_TILE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    FILTER: tl.constexpr,
):
    """One request: the race's winner across tiles, lowest id among ties.

    If the residual had no mass anywhere (p_f == q on every id, which the
    accept rule should make unreachable), every key is 0 and there is no
    winner; draw from p_f itself instead, racing ``p_f / E`` with the same
    noise in this one program. Only that case pays for a second vocab pass.
    Should that find nothing either (a NaN row), emit the row's lowest-id
    max, which every filter keeps.
    """
    req_idx = tl.program_id(0)
    row = tl.load(residual_row_ptr + req_idx)
    if row >= 0:
        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        offs = tl.arange(0, BLOCK_TILE)
        tmask = offs < n_tiles
        keys = tl.load(
            part_key_ptr + offs * batch_size + req_idx, mask=tmask, other=-1.0
        )
        idxs = tl.load(
            part_idx_ptr + offs * batch_size + req_idx, mask=tmask, other=2147483647
        )
        best = tl.max(keys, axis=0)
        token = tl.min(tl.where((keys == best) & tmask, idxs, 2147483647), axis=0)
        if best <= 0.0:
            temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
            row_max = tl.load(row_max_ptr + row)
            inv_z = tl.load(row_inv_z_ptr + row)
            # From 0, not -1: a zero key (no mass anywhere, or inv_z 0 after
            # a NaN Z) never wins, so the sentinel survives into the
            # row_argmax fallback below rather than emitting id 0.
            pbest = 0.0
            token = 2147483647
            for v_start in range(0, vocab_size, BLOCK_V):
                v = v_start + tl.arange(0, BLOCK_V)
                vmask = v < vocab_size
                x = tl.load(
                    logits_ptr + row.to(tl.int64) * logits_row_stride + v,
                    mask=vmask,
                    other=float("-inf"),
                ).to(tl.float32)
                p = tl.where(vmask, tl.exp(x / temp - row_max) * inv_z, 0.0)
                if FILTER:
                    p = tl.where(x >= tl.load(row_floor_ptr + row), p, 0.0)
                u = tl.rand(residual_seed, row.to(tl.int64) * vocab_size + v)
                key = tl.where(vmask, p / -tl.log(u), -1.0)
                tile_best = tl.max(key, axis=0)
                tile_idx = tl.min(
                    tl.where((key == tile_best) & vmask, v, 2147483647), axis=0
                )
                # Strictly better only: an equal key keeps the lower id.
                token = tl.where(tile_best > pbest, tile_idx, token)
                pbest = tl.maximum(pbest, tile_best)
        # Every comparison above is false on NaN. The max has no id either on
        # an all-NaN row; stay inside the vocab then.
        token = tl.where(token == 2147483647, tl.load(row_argmax_ptr + row), token)
        token = tl.where(token < vocab_size, token, 0)
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + (row - start_idx),
            token.to(tl.int32),
        )


def _ratio_target_p(
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    cu_num_draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped
    top_ks: int | torch.Tensor | None,
    top_ps: float | torch.Tensor | None,
    rows: tuple[torch.Tensor, ...],
    parts: tuple[torch.Tensor, ...],
) -> bool:
    """Each draft row's p, and p_f where its request filters.

    Fills ``rows = (row_max, row_inv_z, row_argmax, row_floor)`` through the
    work buffers ``parts = (part_max, part_sum, part_amax_val, part_amax_idx,
    filter_f32, filter_i32, floor_mass, floor_counts)``, the last flat (the
    last count's cells, then the counted rows' run maxima and ids at each):
    p(v) is
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
    row_max, row_inv_z, row_argmax, row_floor = rows
    (
        part_max,
        part_sum,
        part_amax_val,
        part_amax_idx,
        filter_f32,
        filter_i32,
        floor_mass,
        floor_counts,
    ) = parts
    # The last count's cells, then the counted rows' run maxima and the ids
    # at each.
    radix_cells = max(num_tokens, _FILTER_FLOOR_PROGRAMS) * _FILTER_LAST_BINS
    run_cells = _count_run_cells(num_tokens, vocab_size)
    floor_counts, run_max, run_at_max = (
        floor_counts[:radix_cells],
        floor_counts[radix_cells : radix_cells + run_cells],
        floor_counts[radix_cells + run_cells : radix_cells + 2 * run_cells],
    )
    filtered = top_ks is not None or top_ps is not None
    if not num_tokens:
        return filtered
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
    key_bits = 16 if target_logits.dtype == torch.bfloat16 else 32
    # Can a row be counted (a top-k of at most _FILTER_COUNT_MAX_K alone)? And
    # can one be descended? Neither question needs the device: a per-row
    # tensor may hold either, a batch's one value is one or the other.
    p_scalar = None if p_rows or top_ps is None else float(top_ps)
    k_scalar = None if k_rows or top_ks is None else int(top_ks)
    if p_scalar is not None and p_scalar >= 1.0:
        p_scalar = None
    k_small = k_scalar is not None and 0 < k_scalar <= _FILTER_COUNT_MAX_K
    counts = (k_rows or k_small) and p_scalar is None
    descends = not (k_small and not p_rows and p_scalar is None)
    counted = _FILTER_COUNT_MAX_K if counts else 0
    _ratio_stats_kernel[(num_tokens, n_tiles)](
        target_logits,
        cu_num_draft_tokens,
        temperatures,
        part_max,
        part_sum,
        part_amax_val,
        part_amax_idx,
        num_tokens,
        vocab_size,
        stride,
        batch_size,
        search_steps,
        _RATIO_BLOCK_V,
        SAMPLER_EPS,
        *request_filter,
        run_max,
        run_at_max,
        COUNT=counts,
        KEY_BITS=key_bits,
        GROUPS=_FILTER_COUNT_GROUPS,
        MAX_K=_FILTER_COUNT_MAX_K,
        K_PER_ROW=k_rows,
        P_PER_ROW=p_rows,
        num_warps=8,
    )
    softmax_combine_kernel[(num_tokens,)](
        part_max,
        part_sum,
        row_max,
        row_inv_z,
        part_amax_val,
        part_amax_idx,
        row_argmax,
        num_tokens,
        n_tiles,
        BLOCK_TILE=triton.next_power_of_2(n_tiles),
        WITH_ARGMAX=True,
        num_warps=4,
    )
    if not filtered:
        return False
    if counts:
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
            BLOCK_V=_RATIO_BLOCK_V,
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
    if not descends:
        return True
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
    # Only a top-p needs the passes to weigh; counting alone is cheaper.
    weigh = top_ps is not None
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
    return True


def _ratio_sample(
    output_token_ids: torch.Tensor,
    verdict: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    temperatures: torch.Tensor,  # [batch_size] fp32, pre-clamped
    top_ks: int | torch.Tensor | None,
    top_ps: float | torch.Tensor | None,
    draft_q: RatioDraftQ,
    q_row_base: torch.Tensor,  # [batch_size] int32
    bonus_token_ids: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    num_spec_steps: int,
    ratio_step: int,
    scratch: SamplingScratch | None = None,
) -> None:
    """Launch the probability-ratio verify: stats, combine, the filter where
    the batch has one, accept, densify, residual.

    Its scratch comes out of ``scratch`` when given (the serving path) and is
    allocated per call otherwise."""
    num_tokens, vocab_size = target_logits.shape
    batch_size = cu_num_draft_tokens.shape[0]
    device = target_logits.device
    n_tiles = triton.cdiv(vocab_size, _RATIO_BLOCK_V)
    stride = target_logits.stride(0)
    temperatures = temperatures.to(torch.float32).contiguous()
    q_row_base = q_row_base.contiguous()
    store, dense = draft_q
    q_width = store.ids.shape[1]

    f32_shapes, i32_shapes = _ratio_scratch_shapes(num_tokens, batch_size, vocab_size)
    (
        (
            row_max,
            row_inv_z,
            row_floor,
            part_max,
            part_sum,
            part_amax_val,
            part_key,
            filter_f32,
            floor_mass,
        ),
        (
            row_argmax,
            part_amax_idx,
            residual_row,
            residual_q_row,
            part_idx,
            filter_i32,
            floor_counts,
        ),
    ) = take(
        f32_shapes,
        i32_shapes,
        device,
        None if scratch is None else scratch.f32,
        None if scratch is None else scratch.i32,
    )
    filtered = _ratio_target_p(
        target_logits,
        cu_num_draft_tokens,
        temperatures,
        top_ks,
        top_ps,
        (row_max, row_inv_z, row_argmax, row_floor),
        (
            part_max,
            part_sum,
            part_amax_val,
            part_amax_idx,
            filter_f32,
            filter_i32,
            floor_mass,
            floor_counts,
        ),
    )
    _ratio_accept_kernel[(batch_size,)](
        output_token_ids,
        verdict,
        residual_row,
        cu_num_draft_tokens,
        input_ids,
        target_logits_indices,
        target_logits,
        temperatures,
        row_max,
        row_inv_z,
        row_argmax,
        row_floor,
        *store,
        q_row_base,
        bonus_token_ids,
        bonus_logits_indices,
        num_spec_steps,
        q_width,
        stride,
        SAMPLER_EPS,
        _RATIO_ACCEPT_KEY | ratio_step,
        BLOCK_T=triton.next_power_of_2(max(num_spec_steps, 1)),
        BLOCK_Q=min(triton.next_power_of_2(q_width), _RATIO_BLOCK_Q),
        FILTER=filtered,
        num_warps=1,
    )
    # The residual reads q across V: spread each rejecting request's stored
    # row into its own dense row first. Two launches, so every zero lands
    # before any stored entry does.
    _ratio_densify_clear_kernel[(batch_size, n_tiles)](
        residual_row,
        q_row_base,
        dense,
        vocab_size,
        BLOCK_V=_RATIO_BLOCK_V,
        num_warps=8,
    )
    _ratio_densify_scatter_kernel[(batch_size,)](
        residual_row,
        cu_num_draft_tokens,
        *store,
        q_row_base,
        dense,
        residual_q_row,
        q_width,
        vocab_size,
        BLOCK_Q=min(triton.next_power_of_2(q_width), _RATIO_BLOCK_Q),
        num_warps=4,
    )
    _ratio_residual_race_kernel[(batch_size, n_tiles)](
        residual_row,
        target_logits,
        temperatures,
        row_max,
        row_inv_z,
        row_floor,
        dense,
        residual_q_row,
        part_key,
        part_idx,
        batch_size,
        vocab_size,
        stride,
        _RATIO_RESIDUAL_KEY | ratio_step,
        BLOCK_V=_RATIO_BLOCK_V,
        FILTER=filtered,
        num_warps=8,
    )
    _ratio_residual_pick_kernel[(batch_size,)](
        output_token_ids,
        residual_row,
        cu_num_draft_tokens,
        part_key,
        part_idx,
        target_logits,
        temperatures,
        row_max,
        row_inv_z,
        row_argmax,
        row_floor,
        batch_size,
        n_tiles,
        num_spec_steps,
        vocab_size,
        stride,
        _RATIO_RESIDUAL_KEY | ratio_step,
        BLOCK_TILE=triton.next_power_of_2(n_tiles),
        BLOCK_V=_RATIO_BLOCK_V,
        FILTER=filtered,
        num_warps=1,
    )


def warmup_ratio_verify(
    num_spec_steps: int,
    vocab_size: int,
    q_width: int,
    logits_dtype: torch.dtype,
    scratch: SamplingScratch,
) -> None:
    """Compile every ratio verify variant serving can launch, before it does.

    The verify runs eagerly and only on a step that scores drafts, which no
    warmup or capture forward is (`ModelRunner._verifies_drafts`). On an empty
    Triton cache each variant would otherwise compile inside the first serving
    step that needs it, seconds per rank, with DP peers waiting in the
    collective. So this verifies a one-request dummy batch once per filter
    form. It matches a serving batch in everything Triton specializes on: the
    logits' dtype and width, the index dtypes, ``num_spec_steps``, the store's
    width and ``scratch``'s alignment (batch counts are not specialized on,
    `_BATCH_COUNTS`).

    It touches no state: the coins use step 0 of the keys rather than a
    `RejectionSampler`'s counter, the store and dense rows are its own and
    never read (the row base is -1), and ``scratch`` holds nothing across a
    verify. It has no collective and takes no input from the device, so every
    rank runs it alike.
    """
    device = scratch.f32.device
    num_tokens = num_spec_steps

    def ints(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=device)

    def per_row_p():
        return torch.full((1,), 0.9, dtype=torch.float32, device=device)

    k_counted = _FILTER_COUNT_MAX_K
    k_descended = _FILTER_COUNT_MAX_K + 1
    # The (top_ks, top_ps) forms `ModelRunner.prepare_sample` hands over: none,
    # a scalar top-k (counted or descended), a scalar top-p, and a per-row k,
    # p or both. Never a scalar k beside a top-p: `Sampler` keeps top-k ties
    # with the flag on, so that k arrives per row, beside either kind of p.
    forms = (
        (None, None),
        (k_counted, None),
        (k_descended, None),
        (None, 0.9),
        (ints([k_counted]), None),
        (None, per_row_p()),
        (ints([k_counted]), 0.9),
        (ints([k_counted]), per_row_p()),
    )
    store = SparseQStore.allocate(num_tokens, q_width, device)
    dense = torch.empty((1, vocab_size), dtype=torch.float32, device=device)
    for top_ks, top_ps in forms:
        rejection_sample(
            ints([0] * (num_tokens + 2)),
            torch.arange(num_tokens, dtype=torch.int64, device=device),
            num_spec_steps,
            ints([num_tokens]),
            RatioDraftQ(store, dense),
            torch.zeros((num_tokens, vocab_size), dtype=logits_dtype, device=device),
            ints([0]),
            ints([num_tokens], torch.int64),
            temperatures=torch.ones(1, dtype=torch.float32, device=device),
            top_ks=top_ks,
            top_ps=top_ps,
            draft_probs_row_base=ints([-1]),
            ratio_scratch=scratch,
        )


def acceptance_length_to_rates(length: float, n: int) -> list[float]:
    """Mean acceptance length -> per-position *unconditional* acceptance rates.

    Entry ``i`` is the marginal probability that the first ``i+1`` draft tokens
    are all accepted, so the mean number of accepted draft tokens is the sum of
    the list and the mean acceptance length is ``1 + sum`` (the ``1`` being the
    target's own guaranteed token).

    The schedule is the minimum-variance one: accept ``floor(length - 1)``
    positions outright and the next with the leftover fraction. Any schedule
    summing to ``length - 1`` hits the requested mean, but the spread differs
    wildly between them, and spread is not a free parameter here -- it drives
    ITL tails and how the batch drains. This is the schedule vLLM resolves
    ``synthetic_acceptance_length`` to and the one SGLang's ``match-expected``
    draws, so a forced-length run stays comparable across all three engines.
    """
    num_accepted_drafts = length - 1
    num_full = int(num_accepted_drafts)
    return (
        [1.0] * num_full + [num_accepted_drafts - num_full] + [0.0] * (n - num_full - 1)
    )[:n]


def _get_synthetic_cond_rates(
    rates: tuple[float, ...], device: torch.device
) -> torch.Tensor:
    """Unconditional per-position rates -> conditional, as a device tensor.

    The kernel stops at the first rejection, so at position ``i`` it needs
    ``P(accept i | accepted through i-1) = rates[i] / rates[i-1]``, not the
    unconditional ``rates[i]``.
    """
    key = (rates, device)
    cached = _SYNTHETIC_COND_CACHE.get(key)
    if cached is not None:
        return cached
    cond: list[float] = []
    prev = 1.0
    for rate in rates:
        cond.append(rate / prev if prev > 0.0 else 0.0)
        prev = rate
    tensor = torch.tensor(cond, dtype=torch.float32, device=device)
    _SYNTHETIC_COND_CACHE[key] = tensor
    return tensor


class RejectionSampler(nn.Module):
    def __init__(self, synthetic_acceptance_rates: list[float] | None = None):
        super().__init__()
        # Debug/benchmark override: force an acceptance-length curve (see module
        # docstring). None => normal draft/target rejection sampling. The config
        # has already validated and resolved whichever knob the user set into
        # per-position unconditional rates.
        self.synthetic_acceptance_rates = (
            tuple(synthetic_acceptance_rates)
            if synthetic_acceptance_rates is not None
            else None
        )
        # Per-step seed for the synthetic RNG, advanced once per forward. Stays in
        # lockstep across TP ranks (SPMD), so the forced accept/reject pattern is
        # identical on every rank while still varying step to step.
        self._synthetic_step = 0
        # Same, for the probability-ratio path's draws. Advanced on every forward
        # that takes that path, so no two steps replay a stream.
        self._ratio_step = 0

    @staticmethod
    def needs_target_draw(all_greedy: bool, draft_probs) -> bool:
        """Does this verify step have to draw the target's own tokens?

        `Sampler.sample_verification_tokens` (plus, under TP, a broadcast of
        its result) feeds the exact-match kernel and nothing else. The ratio
        path reads `draft_probs` and the logits instead, so the draw is skipped
        exactly when `draft_probs` is given -- not whenever the batch is
        sampled: a sampled batch with no q (flag off) still verifies by exact
        match, and without the draw that kernel would fall
        back to the target argmax, a different and lossy rule for a stochastic
        request. An all-greedy batch never needed it.
        """
        return not all_greedy and draft_probs is None

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        *,
        target_token_ids: torch.Tensor | None = None,
        # ---- probability-ratio path ----------------------------------------
        # The q each draft token was drawn from (sparse, plus the dense scratch
        # the residual reads it through). None selects exact-match acceptance.
        draft_probs: RatioDraftQ | None = None,
        # [batch_size] int32: first q row of each request. Negative means "no
        # live q"; see `_ratio_accept_kernel`.
        draft_probs_row_base: torch.Tensor | None = None,
        # [batch_size] fp32, pre-clamped. Required with draft_probs.
        temperatures: torch.Tensor | None = None,
        # The filters `Sampler` is given for this batch (None, one value, or
        # one per request), so the ratio rule targets the same p_f.
        top_ks: int | torch.Tensor | None = None,
        top_ps: float | torch.Tensor | None = None,
        # Preallocated device scratch for the ratio path; None allocates.
        ratio_scratch: SamplingScratch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure target_logits is contiguous. For greedy sampling, we can use
        # logits directly (argmax is the same for logits and probs), but we
        # need to ensure it's contiguous to satisfy the assertion in rejection_sample.
        target_logits = target_logits.contiguous()

        output_token_ids = rejection_sample(
            metadata.input_ids,
            metadata.target_logits_indices,
            # metadata.num_draft_tokens_np,
            metadata.num_spec_steps,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            metadata.bonus_logits_indices,
            synthetic_acceptance_rates=self.synthetic_acceptance_rates,
            synthetic_step=self._synthetic_step,
            target_token_ids=target_token_ids,
            temperatures=temperatures,
            top_ks=top_ks,
            top_ps=top_ps,
            draft_probs_row_base=draft_probs_row_base,
            ratio_step=self._ratio_step,
            ratio_scratch=ratio_scratch,
        )
        if self.synthetic_acceptance_rates is not None:
            self._synthetic_step += 1
        if draft_probs is not None and self.synthetic_acceptance_rates is None:
            # The low key word; the high word is the draw's purpose.
            self._ratio_step = (self._ratio_step + 1) & 0xFFFFFFFF
        return output_token_ids


def rejection_sample(
    # the step's input ids; draft r is input_ids[target_logits_indices[r] + 1]
    input_ids: torch.Tensor,
    # [num_tokens] -- the row of the step's logits each target_logits row is
    target_logits_indices: torch.Tensor,
    # # [batch_size]
    # num_draft_tokens: list[int],
    num_spec_steps: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # The q each draft token was drawn from (`RatioDraftQ`), or None, which
    # keeps the exact-match acceptance below. Store rows are reached through
    # `draft_probs_row_base`, not `start_idx`.
    draft_probs: RatioDraftQ | None,
    # [num_tokens, vocab_size] raw target logits
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    # [batch_size] -- each request's last row, the one its bonus is sampled at
    bonus_logits_indices: torch.Tensor,
    # Debug override: per-position unconditional acceptance rates, one entry per
    # speculative position (None => normal draft/target path).
    synthetic_acceptance_rates: tuple[float, ...] | None = None,
    # Per-step seed for the (rank-consistent) synthetic RNG; ignored otherwise.
    synthetic_step: int = 0,
    target_token_ids: torch.Tensor | None = None,
    # ---- probability-ratio path (keyword-only, so the existing positional
    # callers keep meaning what they meant) ----------------------------------
    *,
    # [batch_size] fp32, pre-clamped to SAMPLER_EPS. The ratio rule compares
    # against p = softmax(target_logits / T), filtered as `Sampler` filters it
    # with these (None, one value for the batch, or one per request; a top-k
    # of -1 is none). See `_ratio_target_p`.
    temperatures: torch.Tensor | None = None,
    top_ks: int | torch.Tensor | None = None,
    top_ps: float | torch.Tensor | None = None,
    # [batch_size] int32: this request's first row in `draft_probs.store`, or a
    # negative sentinel when it has no live q. Required with draft_probs.
    draft_probs_row_base: torch.Tensor | None = None,
    # Per-step key for the (rank-consistent) accept coins and residual race,
    # < 2**32.
    ratio_step: int = 0,
    # Preallocated device scratch for the ratio path; None allocates per call.
    ratio_scratch: SamplingScratch | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_ids.ndim == 1 and target_logits_indices.ndim == 1
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(cu_num_draft_tokens)
    num_tokens = target_logits_indices.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert input_ids.is_contiguous() and target_logits_indices.is_contiguous()
    assert target_logits.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)
    if draft_probs is not None:
        store, dense = draft_probs
        assert all(t.is_contiguous() for t in store) and dense.is_contiguous()
        assert store.ids.shape == store.probs.shape and store.meta.shape == (
            store.ids.shape[0],
            2,
        )
        assert dense.ndim == 2 and dense.dtype == torch.float32
        assert dense.shape[-1] == vocab_size, (
            f"draft q covers {dense.shape[-1]} ids but the target scored "
            f"{vocab_size}; the ratio rule indexes both rows with the same id"
        )
        assert dense.shape[0] >= batch_size, (
            f"{batch_size} requests but dense q scratch for {dense.shape[0]}: "
            "each request's rejecting row is spread into a row of its own"
        )
        assert temperatures is not None, (
            "the probability-ratio path needs the target temperatures: p is "
            "softmax(target_logits / T), not softmax(target_logits)"
        )
        # The ratio branch below sits above the relaxed one, so the relaxed
        # path would simply never run, and the two are not combinable: relaxed
        # acceptance is deliberately not distribution-preserving. The pair is
        # refused when any drafter is constructed (`Drafter.__init__`), so this
        # only fires if that gate was bypassed.
        assert RELAXED_TOP_N <= 1, (
            "ATOM_SPEC_LOSSLESS_SAMPLING and ATOM_ENABLE_RELAXED_MTP are "
            "mutually exclusive; this should have been refused at drafter "
            "construction"
        )
    if target_token_ids is not None:
        if target_token_ids.shape != target_logits_indices.shape:
            raise ValueError("Sampled target IDs must cover every verification row")
        target_token_ids = target_token_ids.contiguous()

    # Every `topk_select` below passes `tie="low"`, which is the whole reason
    # its answer is the same pick `torch.argmax`/`torch.topk` made. The default
    # promises no direction among equal scores, so a near-tie would resolve
    # differently from one TP rank to the next -- and the accept/reject pattern
    # these feed has to agree across ranks or `num_bonus_tokens` desyncs. It is
    # free at k=1: the reduction that serves k=1 can keep the promise anyway.
    # Indices come back int32, which is what the kernels here load.

    # Create output buffer. Each kernel program writes positions
    # [0 .. num_draft_tokens] for its request and fills the unwritten tail
    # [num_draft_tokens+1 .. num_spec_steps] with the -1 truncation sentinel
    # itself (variable-length verification / DSpark Phase 2), so modify triton kernel
    output_token_ids = torch.empty(
        (batch_size, num_spec_steps + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    # Rows: accepted drafts, rejected drafts (against `num_spec_steps`), and the
    # anchor -- the flat row of the last token emitted. One buffer, so a caller
    # that must align them across ranks broadcasts once.
    verdict = torch.empty((3, batch_size), dtype=torch.int32, device=device)

    if synthetic_acceptance_rates is not None:
        # Synthetic path: force a target acceptance length independent of the
        # real draft/target agreement. Draft tokens are accepted with the
        # configured per-position probability; on the first rejection we emit the
        # target argmax as the correction token and stop (same output layout /
        # num_bonus_tokens semantics as the greedy path).
        cond_rates = _get_synthetic_cond_rates(synthetic_acceptance_rates, device)
        _, target_argmax = topk_select(target_logits, 1, tie="low")
        target_argmax = target_argmax.view(-1)
        # Rank-consistent uniforms: a dedicated device generator re-seeded from the
        # step counter draws the same Philox stream on every TP rank / GPU, so the
        # accept/reject pattern — and hence num_bonus_tokens — matches the
        # broadcast sampled_tokens. A per-rank unseeded torch.rand would desync
        # them and make the anchor gather read a -1 column (draft embeds an
        # invalid id -> crash).
        #
        # These stay on the device on purpose. Drawing on the host and copying in
        # costs a pageable H2D, which PyTorch issues as memcpy + stream sync: the
        # caller then blocks until the whole target forward has drained, right
        # between verify and propose, so the draft model's kernels cannot be
        # dispatched while the target is still running.
        generator = _synthetic_generator(device)
        generator.manual_seed(_SYNTHETIC_RNG_BASE_SEED + int(synthetic_step))
        uniform = torch.rand(
            num_tokens, dtype=torch.float32, device=device, generator=generator
        )
        rejection_synthetic_sample_kernel[(batch_size,)](
            output_token_ids,
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            target_argmax,
            bonus_token_ids,
            bonus_logits_indices,
            uniform,
            cond_rates,
            num_spec_steps,
            num_warps=1,
        )
    elif draft_probs is not None:
        # Probability-ratio path: accept with `u * q < p` against the q the
        # draft actually drew from, instead of requiring equality with the
        # target's own draw. Same output distribution, higher acceptance.
        assert (
            draft_probs_row_base is not None
        ), "the probability-ratio path needs each request's first q row"
        _ratio_sample(
            output_token_ids,
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            target_logits,
            temperatures,
            top_ks,
            top_ps,
            draft_probs,
            draft_probs_row_base,
            bonus_token_ids,
            bonus_logits_indices,
            num_spec_steps,
            ratio_step,
            ratio_scratch,
        )
    elif target_token_ids is not None or RELAXED_TOP_N <= 1:
        # Match the actual target draw for stochastic requests; greedy requests
        # take the target argmax and must match it exactly. The prefix kernel
        # needs only token IDs. `topk_select`, not `.argmax`: the tie rule
        # above is what keeps the accept/reject pattern equal across TP ranks.
        if target_token_ids is None:
            _, target_argmax = topk_select(target_logits, 1, tie="low")
            target_argmax = target_argmax.view(-1)
        else:
            target_argmax = target_token_ids
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            target_argmax,
            bonus_token_ids,
            bonus_logits_indices,
            num_spec_steps,
            num_warps=1,
        )
    else:
        # Relaxed acceptance path: accept if draft is among top-N
        # candidates with prob >= (top1_prob - delta)
        probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        # `sorted` is what puts the top-1 in slot 0 for `top1_probs` below.
        #
        # Among EQUAL probabilities the slot order is unspecified, here as it
        # was under `torch.topk` -- neither promises one. Measured over
        # 64-256 rows of three vocabularies: the id set per row is identical
        # every time and the values are bitwise equal, so `valid_mask` and the
        # kernel's membership test are unaffected; slot 0, which the kernel
        # emits as the correction token, moved on 2 rows of 256 and in both the
        # two ids held the same probability. Rank-consistent either way, since
        # `tie="low"` leaves only backends whose answer is a function of the row.
        topn_probs, topn_ids = topk_select(
            probs, RELAXED_TOP_N, sorted=True, tie="low", return_value=True
        )

        top1_probs = topn_probs[:, 0:1]
        valid_mask = topn_probs >= (top1_probs - RELAXED_DELTA)
        topn_ids[~valid_mask] = -1
        topn_ids = topn_ids.contiguous()

        rejection_relaxed_sample_kernel[(batch_size,)](
            output_token_ids,
            verdict,
            cu_num_draft_tokens,
            input_ids,
            target_logits_indices,
            topn_ids,
            bonus_token_ids,
            bonus_logits_indices,
            num_spec_steps,
            RELAXED_TOP_N,
            num_warps=1,
        )

    return output_token_ids, verdict


@triton.jit
def _draft_token(input_ids_ptr, target_logits_indices_ptr, r):
    """Draft r: the input token after the logits row that verifies it."""
    return tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1)


@triton.jit
def _finish_request(
    output_token_ids_ptr,
    verdict_ptr,
    bonus_token_ids_ptr,
    bonus_logits_indices_ptr,
    req_idx,
    num_draft_tokens,
    last_checked,
    rejected,
    num_spec_steps,
):
    """The bonus token, the -1 tail, and the request's verdict.

    `last_checked` is the last draft position compared (-1 before any). The
    anchor -- the last emitted token's row -- sits one row per accepted draft
    into the segment that ends at the bonus row.
    """
    INVALID_TOKEN: tl.constexpr = -1
    row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)
    num_accepted = last_checked
    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_accepted += 1
    tl.store(row + num_draft_tokens, bonus_token_id)
    # The -1 tail is what downstream first-`-1` truncation reads under
    # variable-length verification (the output buffer is torch.empty).
    for pos in range(num_draft_tokens + 1, num_spec_steps + 1):
        tl.store(row + pos, INVALID_TOKEN)
    batch_size = tl.num_programs(0)
    anchor = tl.load(bonus_logits_indices_ptr + req_idx) - num_draft_tokens
    tl.store(verdict_ptr + req_idx, num_accepted)
    tl.store(verdict_ptr + batch_size + req_idx, num_spec_steps - num_accepted)
    tl.store(verdict_ptr + 2 * batch_size + req_idx, anchor + num_accepted)


@triton.jit(do_not_specialize=["num_spec_steps"])
# TODO use the same sampler as main model
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
    num_spec_steps,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1
    for pos in range(num_draft_tokens):
        if rejected:
            target_argmax_id = INVALID_TOKEN
        else:
            draft_token_id = _draft_token(
                input_ids_ptr, target_logits_indices_ptr, start_idx + pos
            )
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            target_argmax_id = tl.cast(target_argmax_id, tl.int32)
            if draft_token_id != target_argmax_id:
                # rejected = False
                rejected = True
            num_bonus_token += 1
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            target_argmax_id,
        )

    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        num_bonus_token,
        rejected,
        num_spec_steps,
    )


@triton.jit(do_not_specialize=["num_spec_steps"])
def rejection_synthetic_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
    uniform_ptr,  # [num_tokens] — per-position U(0, 1) samples
    cond_rates_ptr,  # [num_spec_steps] — P(accept pos | accepted through pos-1)
    num_spec_steps,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1
    for pos in range(num_draft_tokens):
        if rejected:
            output_id = INVALID_TOKEN
        else:
            u = tl.load(uniform_ptr + start_idx + pos)
            acceptance_rate = tl.load(cond_rates_ptr + pos)
            if u < acceptance_rate:
                # Force-accept: emit the draft's own proposed token.
                output_id = _draft_token(
                    input_ids_ptr, target_logits_indices_ptr, start_idx + pos
                )
                output_id = tl.cast(output_id, tl.int32)
            else:
                # Reject: emit the target correction token and stop accepting.
                output_id = tl.load(target_argmax_ptr + start_idx + pos)
                output_id = tl.cast(output_id, tl.int32)
                rejected = True
            num_bonus_token += 1
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            output_id,
        )

    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        num_bonus_token,
        rejected,
        num_spec_steps,
    )


@triton.jit(do_not_specialize=["num_spec_steps", "top_n"])
def rejection_relaxed_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    verdict_ptr,  # [3, batch_size]: accepted drafts, rejected, anchor row
    cu_num_draft_tokens_ptr,  # [batch_size]
    input_ids_ptr,  # the step's input ids
    target_logits_indices_ptr,  # [num_tokens]
    topn_ids_ptr,  # [num_tokens, top_n] — candidate token ids, -1 = invalid
    bonus_token_ids_ptr,  # [batch_size]
    bonus_logits_indices_ptr,  # [batch_size]
    num_spec_steps,
    top_n,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1

    for pos in range(num_draft_tokens):
        if rejected:
            output_id = INVALID_TOKEN
        else:
            draft_token_id = _draft_token(
                input_ids_ptr, target_logits_indices_ptr, start_idx + pos
            )

            base_offset = (start_idx + pos) * top_n
            top1_id = tl.load(topn_ids_ptr + base_offset)

            found = False
            for k in range(top_n):
                candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                if candidate_id == draft_token_id:
                    found = True

            if found:
                output_id = draft_token_id
            else:
                output_id = top1_id
                rejected = True

            num_bonus_token += 1

        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            output_id,
        )

    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        num_bonus_token,
        rejected,
        num_spec_steps,
    )
