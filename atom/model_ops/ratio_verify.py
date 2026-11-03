"""The probability-ratio verify: lossless acceptance of sampled drafts.

Accept draft x at a position iff `u * q(x) < p(x)`, and on the first rejection
draw the replacement from the residual `relu(p - q)` (normalized). p is the
target's `softmax(logits / T)`, q the distribution the draft drew x from.

Nothing `[rows, V]` is materialized for p. Every (draft row, V tile) program
reduces its tile of the row to the softmax statistics plus the row's
raw-logit argmax (`ratio_target_p`); the accept walk then needs p at one id
per position, `exp(l(x) / T - max) / Z`, and only a rejecting row is read
across V again, by the residual race.

q arrives sparse (`SparseQStore`: each draft row's stored ids, their
probabilities and the slot of the id it emitted), and is 0 off the stored
set. The accept walk reads q(x) at the emitted slot. The residual needs q at
every id, so each rejecting request's one row is first spread into a dense
scratch row of its own (zeroed, then the stored entries written at their
ids); the race and its fallback then read that row as they would a dense q.

Residual: an exponential race rather than a CDF walk. `argmax_v r(v) / E_v`
with i.i.d. `E_v ~ Exp(1)` selects v with probability `r(v) / sum r`, so the
residual needs no normalizer, no prefix sum, and no agreement between two
reductions' rounding; tiles race in parallel and a cross-tile max decides.

Top-k / top-p: a filtered request is served `Sampler`'s filtered p, p_f, at
every position -- its bonus token is `Sampler`'s own draw. p_f is p over
every id at or above one logit per row, the row's floor, renormalized;
`sampling_filter` finds the floor on the device, and the accept, the race
and its fallback all read p as 0 below it.

Greedy rows (temperature at or below SAMPLER_EPS) take exact match inside the
same launch: accept iff the draft is the target's lowest-id argmax, emit that
argmax on rejection. That is `rejection_greedy_sample_kernel`'s rule
(`rejection_sampler`), so a greedy request's tokens do not depend on whether
sampled requests share its batch. The ratio rule would not give that: on a
bf16 logit tie p is 0.5/0.5, not a delta.

`rejection_sampler.rejection_sample` dispatches here when the drafter hands
it a `RatioVerifyInputs`; the exact-match, synthetic and relaxed rules stay
there.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from atom.model_ops.ratio_target_p import _RATIO_BLOCK_V, _ratio_target_p
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.sampling_filter import _FILTER_COUNT_MAX_K, filter_scratch_shapes
from atom.model_ops.sampling_scratch import SamplingScratch, take
from atom.model_ops.softmax_tiles import argmax_lowest_id, race_key
from atom.model_ops.sparse_q_store import RatioDraftQ, SparseQStore
from atom.model_ops.verify_kernels import _draft_token, _finish_request

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

# Stored q entries one lookup or densify iteration covers.
_RATIO_BLOCK_Q = 1024


class RatioVerifyInputs(NamedTuple):
    """What a drafter hands one probability-ratio verify."""

    #: The q each draft token was drawn from, and the dense rows the residual
    #: reads it through.
    q: RatioDraftQ
    #: [batch_size] int32: each request's first q row, or < 0 for no live q.
    q_row_base: torch.Tensor
    #: [batch_size] fp32, pre-clamped: p is softmax(target_logits / T).
    temperatures: torch.Tensor
    #: `Sampler`'s filters for the batch, so the verify targets the same p_f.
    top_ks: int | torch.Tensor | None
    top_ps: float | torch.Tensor | None
    #: Preallocated device scratch (`ratio_verify_scratch`).
    scratch: SamplingScratch


def _ratio_scratch_shapes(num_tokens: int, batch_size: int, vocab_size: int):
    """fp32 and int32 shapes one ratio verify needs, in `_ratio_sample` order."""
    tiles = triton.cdiv(vocab_size, _RATIO_BLOCK_V)
    filter_f32, filter_i32 = filter_scratch_shapes(
        num_tokens, vocab_size, _RATIO_BLOCK_V
    )
    f32 = [
        (num_tokens,),  # row_max
        (num_tokens,),  # row_inv_z
        (num_tokens,),  # row_floor
        (tiles, num_tokens),  # part_max
        (tiles, num_tokens),  # part_sum
        (tiles, num_tokens),  # part_amax_val
        (tiles, batch_size),  # part_key
        *filter_f32,
    ]
    i32 = [
        (num_tokens,),  # row_argmax
        (tiles, num_tokens),  # part_amax_idx
        (batch_size,),  # residual_row
        (tiles, batch_size),  # part_idx
        *filter_i32,
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
def _has_q(q_base):
    """Whether a request's q row base names live q rows. A negative base means
    it has none (never drafted here, or its slot expired); every verify kernel
    asks this, so the sentinel is decided here only. Not 0 for "none": row 0
    is in range and holds a stranger's q."""
    return q_base >= 0


@triton.jit
def _rejecting_row_of(residual_row_ptr, q_row_base_ptr, req_idx):
    """``(row, q_base, has_q)`` of one request: the draft row it rejected at
    (-1 if none, or greedy), its q row base, and whether that rejection has a
    q, which the densify kernels then spread into dense row ``req_idx`` and
    the residual race reads back from there."""
    row = tl.load(residual_row_ptr + req_idx)
    q_base = tl.load(q_row_base_ptr + req_idx)
    return row, q_base, (row >= 0) & _has_q(q_base)


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
    # A request without a live q (`_has_q`) reads q as 0, so its first draft
    # is rejected and the residual is p itself: exact, at the cost of one
    # step's speculation. A stranger's row would hold a q > 0, against which
    # `u * q < p` would accept.
    q_base = tl.load(q_row_base_ptr + req_idx)
    has_q = _has_q(q_base)
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
    _, _, has_q = _rejecting_row_of(residual_row_ptr, q_row_base_ptr, req_idx)
    if has_q:
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
    q_width,
    vocab_size,
    BLOCK_Q: tl.constexpr,
):
    """One request: write the rejecting row's stored entries into its zeroed
    dense row ``req_idx``, where the residual race reads them. A request
    `_rejecting_row_of` finds no q for is left alone; the race reads its q as
    0, as the accept walk did."""
    req_idx = tl.program_id(0)
    row, q_base, has_q = _rejecting_row_of(residual_row_ptr, q_row_base_ptr, req_idx)
    if has_q:
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


@triton.jit
def _target_p_tile(
    logits_ptr,
    row,
    logits_row_stride,
    offs,
    vmask,
    temp,
    row_max,
    inv_z,
    row_floor_ptr,
    FILTER: tl.constexpr,
):
    """``p_f`` on one V tile of ``row``: the tempered softmax from the row's
    max and 1/Z, 0 on masked lanes and, under ``FILTER``, below the row's
    floor. The residual race and its p fallback both race this p."""
    x = tl.load(
        logits_ptr + row.to(tl.int64) * logits_row_stride + offs,
        mask=vmask,
        other=float("-inf"),
    ).to(tl.float32)
    p = tl.where(vmask, tl.exp(x / temp - row_max) * inv_z, 0.0)
    if FILTER:
        p = tl.where(x >= tl.load(row_floor_ptr + row), p, 0.0)
    return p


@triton.jit(do_not_specialize=["batch_size", "residual_seed"])
def _ratio_residual_race_kernel(
    residual_row_ptr,  # [batch_size] int32, the rejecting row, or -1
    logits_ptr,  # [num_tokens, V] raw target logits
    temperatures_ptr,  # [batch_size] fp32
    row_max_ptr,  # [num_tokens] fp32
    row_inv_z_ptr,  # [num_tokens] fp32
    row_floor_ptr,  # [num_tokens] fp32, the lowest logit p_f keeps (FILTER)
    dense_ptr,  # [>= batch_size, V] fp32: a rejecting request's q at its row
    q_row_base_ptr,  # [batch_size] int32, this request's first q row, or < 0
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
    row, _, has_q = _rejecting_row_of(residual_row_ptr, q_row_base_ptr, req_idx)
    if row >= 0:
        temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
        offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        vmask = offs < vocab_size
        p = _target_p_tile(
            logits_ptr,
            row,
            logits_row_stride,
            offs,
            vmask,
            temp,
            tl.load(row_max_ptr + row),
            tl.load(row_inv_z_ptr + row),
            row_floor_ptr,
            FILTER,
        )
        q = tl.load(
            dense_ptr + req_idx.to(tl.int64) * vocab_size + offs,
            mask=vmask & has_q,
            other=0.0,
        )
        # Anything that is not a probability (NaN, inf, negative) counts as 0,
        # so the residual falls back toward p rather than inheriting garbage.
        q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
        residual = tl.maximum(p - q, 0.0)
        key = race_key(
            residual, residual_seed, row.to(tl.int64) * vocab_size, offs, vmask
        )
        best, best_id = argmax_lowest_id(key, offs, vmask, 2147483647)
        part = tile * batch_size + req_idx
        tl.store(part_key_ptr + part, best)
        tl.store(part_idx_ptr + part, best_id)


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
        best, token = argmax_lowest_id(keys, idxs, tmask, 2147483647)
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
                p = _target_p_tile(
                    logits_ptr,
                    row,
                    logits_row_stride,
                    v,
                    vmask,
                    temp,
                    row_max,
                    inv_z,
                    row_floor_ptr,
                    FILTER,
                )
                # The race kernel's key and tile reduction at the same Philox
                # counters, with p_f in place of the residual.
                key = race_key(
                    p, residual_seed, row.to(tl.int64) * vocab_size, v, vmask
                )
                tile_best, tile_idx = argmax_lowest_id(key, v, vmask, 2147483647)
                # Strictly better only: an equal key keeps the lower id. (Not
                # finalize's tie merge: a 0 key must not displace the
                # sentinel, which the row_argmax fallback below relies on.)
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


def _ratio_sample(
    output_token_ids: torch.Tensor,
    verdict: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    target_logits: torch.Tensor,  # [num_tokens, V] raw
    ratio: RatioVerifyInputs,
    bonus_token_ids: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    num_spec_steps: int,
    ratio_step: int,
) -> None:
    """Launch the probability-ratio verify: stats, combine, the filter where
    the batch has one, accept, densify, residual, carving its partials out of
    ``ratio.scratch`` (`ratio_verify_scratch`)."""
    num_tokens, vocab_size = target_logits.shape
    batch_size = cu_num_draft_tokens.shape[0]
    store, dense = ratio.q
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
    n_tiles = triton.cdiv(vocab_size, _RATIO_BLOCK_V)
    stride = target_logits.stride(0)
    temperatures = ratio.temperatures.to(torch.float32).contiguous()
    q_row_base = ratio.q_row_base.contiguous()
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
            *filter_f32,
        ),
        (row_argmax, part_amax_idx, residual_row, part_idx, *filter_i32),
    ) = take(f32_shapes, i32_shapes, ratio.scratch.f32, ratio.scratch.i32)
    filtered = _ratio_target_p(
        target_logits,
        cu_num_draft_tokens,
        temperatures,
        ratio.top_ks,
        ratio.top_ps,
        (row_max, row_inv_z, row_argmax, row_floor),
        (part_max, part_sum, part_amax_val, part_amax_idx),
        (*filter_f32, *filter_i32),
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
        q_row_base,
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
        # The buffers `rejection_sample` hands the verify.
        output_token_ids = torch.empty(
            (1, num_spec_steps + 1), dtype=torch.int32, device=device
        )
        verdict = torch.empty((3, 1), dtype=torch.int32, device=device)
        _ratio_sample(
            output_token_ids,
            verdict,
            ints([num_tokens]),
            ints([0] * (num_tokens + 2)),
            torch.arange(num_tokens, dtype=torch.int64, device=device),
            torch.zeros((num_tokens, vocab_size), dtype=logits_dtype, device=device),
            RatioVerifyInputs(
                RatioDraftQ(store, dense),
                ints([-1]),
                torch.ones(1, dtype=torch.float32, device=device),
                top_ks,
                top_ps,
                scratch,
            ),
            ints([0]),
            ints([num_tokens], torch.int64),
            num_spec_steps,
            0,
        )
