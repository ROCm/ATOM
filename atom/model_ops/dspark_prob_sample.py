# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Probabilistic DSpark drafting + probability-ratio verification (opt-in).

Default OFF; every path in this file is reached only when
``ATOM_DSPARK_PROB_DRAFT=1``. With it off the drafter keeps its greedy
(argmax) Markov sampler and the runner keeps its existing verification.

Drafting (``prob_markov_sample``)
    Position k of the block samples ``x_k ~ q_k = softmax(z_k / T_d)`` where
    ``z_k = base_logits_k + W1[x_{k-1}] @ W2^T`` (the same biased logits the
    greedy head takes the argmax of) and ``T_d`` is the request's temperature
    (times ``ATOM_DSPARK_PROB_DRAFT_TSCALE``, default 1). Sampling is
    Gumbel-max with counter-based (Philox) noise keyed by (step seed,
    position, row, GLOBAL vocab id), so it is deterministic, needs no host
    sync and replays inside the draft CUDA graph. Under TP with the sharded
    LM head (the default V4.1 path) each rank scores its own vocab shard and
    the ranks exchange one (score, id) pair per row -- the same exchange the
    greedy sharded sampler does -- so every rank ends with the same ids.
    ``z_k`` (pre-temperature, fp32) of the local shard is kept for
    verification. A row at the greedy temperature clamp takes the plain
    argmax, which is the greedy head's pick. No top-p/top-k on ``q`` (same as
    vLLM's ``draft_sample_method=probabilistic``); any ``q`` is valid.

Verification (``prob_rejection_sample``)
    Target ``p`` = softmax(logits / T) restricted to the top-p nucleus
    (smallest set of highest-probability tokens with mass >= top_p, ties at
    the boundary kept) and renormalized -- the distribution the runner's
    sampler draws from (aiter's top-p sampler; the native fallback's
    ``cumsum - p <= top_p`` differs only on exact boundary ties).
    * standard (Leviathan et al. 2023): accept x_k iff u < p(x_k)/q(x_k);
      on the first rejection resample from norm(max(p - q, 0)).
    * block (Sun et al. 2024, ``ATOM_DSPARK_BLOCK_VERIFY=1``): the
      cumulative-ratio block test vLLM's ``rejection_sample_method=block``
      implements; accepted length = max{i : eta_i <= h_i}, residual
      norm(max(P_tau p - q, 0)).
    A draft row whose ``q`` is not available (new request, draft id does not
    match the stored one, greedy-drafted row, ...) is verified one-hot:
    accept with prob p(x), else resample from p with x removed -- exact for
    a draft drawn from ANY distribution, i.e. the same guarantee the existing
    "accept iff draft == target sample" path gives. Greedy requests
    (temperature at the clamp) keep the argmax-match rule.
    With a sharded q the vocab sums (q's normalizer, block residual mass,
    the residual draw) are split by vocab shard and combined through small
    all-gathers; every rank then takes identical decisions.
    Output layout / verdict are exactly the existing rejection sampler's
    (``_finish_request``); the bonus token is still the runner's sampler draw.

Noise is keyed by host step counters, not by the request's ``seed``: seeded
requests are not reproducible on this path.
"""

import logging
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from atom.model_ops import dspark_markov_stage1 as _markov_stage1
from atom.model_ops.dspark_markov_argmax import (
    _dspark_markov_argmax_stage2,
    argmax_gather,
)
from atom.model_ops.dspark_markov_stage1 import markov_logits_tile
from atom.model_ops.rejection_sampler import _draft_token, _finish_request
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.tile_argmax import NO_HIT, argmax_lowest_id
from atom.utils import envs

_BLOCK_K = _markov_stage1.BLOCK_K
_BLOCK_V = _markov_stage1.BLOCK_V
_MAX_BLOCK_ROW = _markov_stage1.MAX_BLOCK_ROW
_MIN_BLOCK_ROW = _markov_stage1.MIN_BLOCK_ROW

logger = logging.getLogger("atom")

PROB_DRAFT = envs.ATOM_DSPARK_PROB_DRAFT
FUSED_MARKOV_CHAIN = envs.ATOM_DSPARK_PROB_FUSED_CHAIN
BLOCK_VERIFY = envs.ATOM_DSPARK_BLOCK_VERIFY
DRAFT_TSCALE = envs.ATOM_DSPARK_PROB_DRAFT_TSCALE
# 1 (default) => draft off the sharded LM head whenever the greedy path would;
# 0 => always sample the all-gathered vocab.
SHARDED_DRAFT = envs.ATOM_DSPARK_PROB_SHARDED

# The runner clamps temperature 0 to SAMPLER_EPS (1e-10) and the sampler treats
# exactly the rows at or below it as greedy (``_apply_greedy_tokens``); use the
# same rule. fp32-rounded, since every temperature compared against it is an
# fp32 value (device buffers, and host floats read back from them).
GREEDY_T = float(torch.tensor(SAMPLER_EPS, dtype=torch.float32).item())
_GREEDY_T = tl.constexpr(GREEDY_T)
# Smallest positive value of Triton's fp32 tl.rand.
_RAND_MIN = tl.constexpr(4.6566127342e-10)
_NUM_WARPS = 8
# Vocab-split verification: the full-vocab scans (row statistics, block
# residual mass, the residual draw) run one program per CHUNK columns of a row
# instead of one program per row, which left most of the GPU idle at decode
# batch sizes (a few dozen rows against the full vocab).
# A chunk reduces its columns per lane, then across the lanes; the chunks'
# records are combined in column order, and a tie on a maximum goes to the
# lowest column. The residual draw is a Gumbel max over the chunks' maxima.
# The row normalizer rescales each chunk's sumexp from its own max to the row
# max (`exp(m_c - m)`, `_combine_stats`) before summing; only the residual
# mass and the top-p records are plain sums of the chunks' partial sums.
_SPLIT_BLOCK = 2048
_CHUNK = 8192
assert _CHUNK % _SPLIT_BLOCK == 0, "_CHUNK must be a multiple of _SPLIT_BLOCK"
# Top-p threshold: the largest tau with mass(>= tau) >= top_p z. One pass
# records, per chunk, the mass and count at or above each of _TOPP_K evenly
# spaced thresholds of the bracket [m - c, m] (c = ln(V / (1 - top_p)) + 1:
# tokens below it hold at most 1 - top_p of the mass); the interval holding
# the cutoff is split _TOPP_K ways again while it holds more than _TOPP_RB
# tokens (once more per chunk, then over the whole row, at most
# _TOPP_ROW_LEVELS times); its tokens are compacted (chunk order, then column
# order) and the bisection finishes on them in registers. An interval still
# over _TOPP_RB tokens after that is kept whole: they lie within c / K**8 of
# one another, i.e. they tie.
_TOPP_K = 32
_TOPP_SB = 256
_TOPP_RB = 4096
_TOPP_ITERS = tl.constexpr(30)
_TOPP_ROW_LEVELS = tl.constexpr(6)


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


@triton.jit
def _gumbel(seed, offs):
    """Standard Gumbel noise from counter-based uniforms.

    w = tl.rand in (0, 1) plays 1 - u, so -log(u) = -log1p(-w) keeps full
    resolution in the winning tail (w -> 0)."""
    w = tl.maximum(tl.rand(seed, offs), _RAND_MIN)
    poly = 1.0 / 8.0
    poly = 1.0 / 7.0 + w * poly
    poly = 1.0 / 6.0 + w * poly
    poly = 1.0 / 5.0 + w * poly
    poly = 1.0 / 4.0 + w * poly
    poly = 1.0 / 3.0 + w * poly
    poly = 1.0 / 2.0 + w * poly
    poly = 1.0 + w * poly
    series = w * poly  # -log1p(-w), |err| < 6e-7 abs on [0, .25]
    direct = -tl.log(tl.maximum(1.0 - w, 5.960464477539063e-08))
    e = tl.where(w < 0.25, series, direct)  # Exp(1) variate
    return -tl.log(e)


# ---------------------------------------------------------------------------
# drafting
# ---------------------------------------------------------------------------


@triton.jit
def _draft_row_temps(temp_ptr, seed_ptr, offs_row, row_mask):
    """(greedy, 1 / T, seed) of a row tile: the rows at the greedy clamp
    take the plain argmax, the others Gumbel-max at their temperature."""
    temp = tl.load(temp_ptr + offs_row, mask=row_mask, other=1.0)
    greedy = temp <= _GREEDY_T
    inv_t = 1.0 / tl.where(greedy, 1.0, temp)
    return greedy, inv_t, tl.load(seed_ptr)


@triton.jit
def _sampled_tile_argmax(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    embed_out_ptr,
    q_out_ptr,
    tile,
    offs_row,
    offs_v,
    row_mask,
    v_mask,
    greedy,
    inv_t,
    seed,
    num_rows,
    vocab_global,
    vocab_offset,
    rank,
    pos_k,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    stride_q_row,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FROM_PACKED: tl.constexpr,
):
    """One (row tile, V tile) of a sampled block position: the biased logits
    z (`markov_logits_tile`), stored to ``q_out`` (fp32, pre-temperature:
    verification rebuilds q from exactly these), then the Gumbel-perturbed
    scores' (max, lowest local column). A tile with no hit answers `NO_HIT`:
    one with no lane inside the vocabulary as well as an all-NaN one. Shared
    by `_prob_markov_stage1` and the fused chain."""
    z = markov_logits_tile(
        base_ptr,
        prev_ids_ptr,
        w1_ptr,
        w2_ptr,
        embed_out_ptr,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        rank,
        stride_base_row,
        stride_prev,
        stride_w1_vocab,
        stride_embed_row,
        stride_w2_vocab,
        vocab_offset,
        BLOCK_ROW,
        BLOCK_V,
        BLOCK_K,
        STORE_EMBED=True,
        FROM_PACKED=FROM_PACKED,
    )
    tl.store(
        q_out_ptr + offs_row[:, None] * stride_q_row + offs_v[None, :],
        z,
        mask=row_mask[:, None] & v_mask[None, :],
    )
    noise_offs = (pos_k * num_rows + offs_row)[:, None] * vocab_global + (
        vocab_offset + offs_v
    )[None, :]
    g = _gumbel(seed, noise_offs)
    # Padded lanes hold z = -inf, hence score -inf, and are masked anyway.
    score = tl.where(greedy[:, None], z, z * inv_t[:, None] + g)
    return argmax_lowest_id(score, offs_v[None, :], v_mask[None, :], NO_HIT)


@triton.jit
def _prob_markov_stage1(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    embed_out_ptr,
    q_out_ptr,
    temp_ptr,
    seed_ptr,
    part_val_ptr,
    part_idx_ptr,
    num_rows,
    vocab_size,  # columns of base_logits (this rank's shard)
    vocab_global,
    vocab_offset,  # global id of column 0
    rank,
    pos_k,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    stride_q_row,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """`dspark_markov_argmax`'s stage 1 + store z + Gumbel-max on the scores."""
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    greedy, inv_t, seed = _draft_row_temps(temp_ptr, seed_ptr, offs_row, row_mask)
    tile_max, tile_idx = _sampled_tile_argmax(
        base_ptr,
        prev_ids_ptr,
        w1_ptr,
        w2_ptr,
        embed_out_ptr,
        q_out_ptr,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        greedy,
        inv_t,
        seed,
        num_rows,
        vocab_global,
        vocab_offset,
        rank,
        pos_k,
        stride_base_row,
        stride_prev,
        stride_w1_vocab,
        stride_embed_row,
        stride_w2_vocab,
        stride_q_row,
        BLOCK_ROW,
        BLOCK_V,
        BLOCK_K,
        FROM_PACKED=False,
    )
    tile_idx = tl.where(tile_idx != NO_HIT, tile_idx + vocab_offset, vocab_size)
    tl.store(part_val_ptr + tile * num_rows + offs_row, tile_max, mask=row_mask)
    tl.store(part_idx_ptr + tile * num_rows + offs_row, tile_idx, mask=row_mask)


def prob_markov_sample(
    base_logits: torch.Tensor,  # [B, Vloc] (row-strided ok)
    prev_ids: torch.Tensor,  # [B]
    markov_w1: torch.Tensor,  # [V, r]
    markov_w2: torch.Tensor,  # [V, r]
    out: torch.Tensor,  # [B] int32/int64 (strided ok)
    q_out: torch.Tensor,  # [B, >=Vloc] fp32 (row-strided ok)
    temps: torch.Tensor,  # [B] fp32
    seed: torch.Tensor,  # [1] int64
    pos_k: int,
    embed_out: torch.Tensor,  # [B, r]
    vocab_start: int = 0,
    tp_group=None,
    use_custom: bool = True,
) -> torch.Tensor:
    """One block position. ``tp_group`` given => ``base_logits`` is this
    rank's vocab shard starting at ``vocab_start`` and the winner is agreed
    through a ``[B, 2]`` all-gather."""
    num_rows, vocab_size = base_logits.shape
    vocab_global = markov_w2.shape[0]
    rank = markov_w2.shape[1]
    assert base_logits.stride(1) == 1 and q_out.stride(1) == 1
    assert markov_w1.stride(1) == 1 and markov_w2.stride(1) == 1
    assert embed_out.stride(1) == 1
    num_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    block_row = min(
        _MAX_BLOCK_ROW, max(_MIN_BLOCK_ROW, triton.next_power_of_2(num_rows))
    )
    dev = base_logits.device
    part_val = torch.empty((num_tiles, num_rows), dtype=torch.float32, device=dev)
    part_idx = torch.empty((num_tiles, num_rows), dtype=torch.int32, device=dev)
    _prob_markov_stage1[(num_tiles, triton.cdiv(num_rows, block_row))](
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        embed_out,
        q_out,
        temps,
        seed,
        part_val,
        part_idx,
        num_rows,
        vocab_size,
        vocab_global,
        vocab_start,
        rank,
        pos_k,
        stride_base_row=base_logits.stride(0),
        stride_prev=prev_ids.stride(0),
        stride_w1_vocab=markov_w1.stride(0),
        stride_embed_row=embed_out.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        stride_q_row=q_out.stride(0),
        BLOCK_ROW=block_row,
        BLOCK_V=_BLOCK_V,
        BLOCK_K=_BLOCK_K,
        num_warps=8 if block_row >= 32 else 4,
        num_stages=2,
    )
    packed = None
    if tp_group is not None:
        packed = torch.empty((num_rows, 2), dtype=torch.float32, device=dev)
    _dspark_markov_argmax_stage2[(num_rows,)](
        part_val,
        part_idx,
        out if packed is None else part_idx,
        num_rows,
        num_tiles,
        out.stride(0) if packed is None else 0,
        part_val if packed is None else packed,
        BLOCK_TILE=triton.next_power_of_2(num_tiles),
        PACKED=packed is not None,
        num_warps=4,
        num_stages=1,
    )
    if packed is not None:
        gathered = tp_group.all_gather(packed, dim=0, use_custom=use_custom)
        argmax_gather(gathered.view(tp_group.world_size, num_rows, 2), out)
    return embed_out


@dataclass
class ProbDraftState:
    """Persistent (graph-address-stable) buffers shared by drafter + runner."""

    q: torch.Tensor  # [max_bs, W, V] fp32, pre-temperature draft logits
    ids: torch.Tensor  # [max_bs, W] int32, the (global) ids drawn
    temps: torch.Tensor  # [max_bs] fp32, temperature each row was drafted at
    seed: torch.Tensor  # [1] int64, per-draft-step seed
    gen: int = 0  # draft steps run (host)
    # (gen, {req_id: row}) of the last real propose; None => no usable q.
    owner: object = None
    verify_step: int = 0
    src_cpu: list = field(default_factory=list)  # pinned double buffer
    src_gpu: torch.Tensor | None = None
    src_hits: int = 0
    src_total: int = 0
    # Layout of q's columns, set by the drafter when it runs: column c of a
    # row is global vocab id q_vs + c, c < q_cols; q_tp > 1 => the other
    # shards live on the other TP ranks.
    q_vs: int = 0
    q_cols: int = 0
    q_tp: int = 1
    # A position-major packed maximum, reset by the chain's final kernel.
    # Allocate before graph capture; one ordered draft stream owns this state.
    markov_keys: torch.Tensor | None = None

    @classmethod
    def allocate(cls, max_bs: int, width: int, vocab: int, device) -> "ProbDraftState":
        st = cls(
            q=torch.zeros(max_bs, width, vocab, dtype=torch.float32, device=device),
            ids=torch.full((max_bs, width), -1, dtype=torch.int32, device=device),
            temps=torch.ones(max_bs, dtype=torch.float32, device=device),
            seed=torch.zeros(1, dtype=torch.int64, device=device),
            markov_keys=(
                torch.zeros(width, max_bs, dtype=torch.uint64, device=device)
                if FUSED_MARKOV_CHAIN
                else None
            ),
        )
        st.q_cols = vocab
        st.src_cpu = [
            torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        st.src_gpu = torch.empty(max_bs, dtype=torch.int32, device=device)
        return st

    def set_layout(self, vocab_start: int, cols: int, tp: int) -> None:
        self.q_vs, self.q_cols, self.q_tp = int(vocab_start), int(cols), int(tp)

    def begin_draft(self, temps: torch.Tensor | None, scheduled_bs: int) -> None:
        """Host side of one draft step; runs eagerly before the graph replay."""
        self.gen += 1
        self.seed.fill_(self.gen * 4)  # salt 0 (verify uses 1/2)
        if temps is not None and temps.shape[0] == scheduled_bs:
            if DRAFT_TSCALE != 1.0:
                # greedy rows stay at the clamp
                self.temps[:scheduled_bs].copy_(
                    torch.where(temps > GREEDY_T, temps * DRAFT_TSCALE, temps)
                )
            else:
                self.temps[:scheduled_bs].copy_(temps)
        else:
            # unknown temperature: draft greedily, verification then treats the
            # rows as one-hot (always exact)
            self.temps[:scheduled_bs].fill_(1e-10)

    def record_owner(self, req_ids) -> None:
        self.owner = (self.gen, {rid: i for i, rid in enumerate(req_ids)})

    def src_for(self, req_ids, bs: int) -> torch.Tensor:
        """[bs] int32 on device: row of each request's q in the buffers, or -1."""
        buf = self.src_cpu[self.verify_step & 1]
        arr = buf.numpy()
        owner = self.owner
        if owner is None or owner[0] != self.gen:
            arr[:bs] = -1
        else:
            m = owner[1]
            for i in range(bs):
                arr[i] = m.get(req_ids[i], -1)
        # host-side hit rate of the request -> q-row map (no GPU readback)
        self.src_total += bs
        self.src_hits += int((arr[:bs] >= 0).sum())
        if self.verify_step % 5000 == 0 and self.verify_step:
            logger.info(
                "[DSpark prob] verify steps=%d q-map hit=%.1f%% block=%s q_tp=%d",
                self.verify_step,
                100.0 * self.src_hits / max(self.src_total, 1),
                BLOCK_VERIFY,
                self.q_tp,
            )
        out = self.src_gpu[:bs]
        out.copy_(buf[:bs], non_blocking=True)
        return out


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


@triton.jit
def _span_max_argmax(row_ptr, lo, hi, BLOCK: tl.constexpr):
    """(max, lowest argmax) of columns [lo, hi) of a row; per-lane
    accumulation, one final reduce."""
    mxv = tl.zeros((BLOCK,), tl.float32) + float("-inf")
    idxv = tl.zeros((BLOCK,), tl.int32)
    for start in range(lo, hi, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(row_ptr + offs, mask=offs < hi, other=float("-inf")).to(tl.float32)
        take = x > mxv  # strict: a lane keeps its first (lowest) index
        idxv = tl.where(take, offs, idxv)
        mxv = tl.where(take, x, mxv)
    best = tl.max(mxv, axis=0)
    best_i = tl.min(tl.where(mxv == best, idxv, 2147483647), axis=0)
    return best, best_i


@triton.jit
def _span_online_stats(row_ptr, lo, hi, inv_t, BLOCK: tl.constexpr):
    """One pass over columns [lo, hi) of a row: (max of x*inv_t,
    sum exp(x*inv_t - max), lowest argmax); per-lane online softmax, one final
    reduce."""
    mxv = tl.zeros((BLOCK,), tl.float32) + float("-inf")
    sv = tl.zeros((BLOCK,), tl.float32)
    idxv = tl.zeros((BLOCK,), tl.int32)
    for start in range(lo, hi, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(row_ptr + offs, mask=offs < hi, other=float("-inf")).to(tl.float32)
        y = x * inv_t
        take = y > mxv
        idxv = tl.where(take, offs, idxv)
        new = tl.maximum(mxv, y)
        alpha = tl.where(mxv == new, 1.0, tl.exp(mxv - new))
        beta = tl.where(y == float("-inf"), 0.0, tl.exp(y - new))
        sv = sv * alpha + beta
        mxv = new
    mx = tl.max(mxv, axis=0)
    s = tl.sum(tl.where(mxv == float("-inf"), 0.0, sv * tl.exp(mxv - mx)), axis=0)
    best_i = tl.min(tl.where(mxv == mx, idxv, 2147483647), axis=0)
    return mx, s, best_i


@triton.jit
def _combine_stats(stat_ptr, nch):
    """Fold per-chunk (max, sumexp) records [nch, 2], chunks in column order:
    (max, sumexp at that max, first chunk holding the max or -1 for none).
    The strict compare keeps the earliest chunk on a tie, so the lowest
    column wins."""
    m = tl.zeros((), tl.float32) + float("-inf")
    best_c = tl.zeros((), tl.int32) - 1
    for c in range(nch):
        mc = tl.load(stat_ptr + c * 2)
        t = mc > m
        best_c = tl.where(t, c, best_c)
        m = tl.where(t, mc, m)
    s = tl.zeros((), tl.float32)
    for c in range(nch):
        mc = tl.load(stat_ptr + c * 2)
        sc = tl.load(stat_ptr + c * 2 + 1)
        s += tl.where(mc == float("-inf"), 0.0, sc * tl.exp(mc - m))
    return m, s, best_c


@triton.jit
def _combine_argmax(arg_ptr, best_c):
    """A row's lowest argmax from its per-chunk ones [nch]: that of the first
    chunk holding the row max (`_combine_stats`), 0 when no chunk does."""
    return tl.where(best_c >= 0, tl.load(arg_ptr + tl.maximum(best_c, 0)), 0)


@triton.jit
def _target_row(target_logits_indices_ptr, r, IN_PLACE: tl.constexpr):
    """Row of `target_ptr` holding verify row r: r itself, or (IN_PLACE, the
    full logits) the row the step's `target_logits_indices` names, so the
    target rows are read in place instead of through a compacted copy."""
    if IN_PLACE:
        return tl.load(target_logits_indices_ptr + r).to(tl.int64)
    return r.to(tl.int64)


@triton.jit
def _req_span(cu_ptr, req_idx):
    if req_idx == 0:
        start = 0
    else:
        start = tl.load(cu_ptr + req_idx - 1)
    end = tl.load(cu_ptr + req_idx)
    return start, end - start


# ---- top-p threshold (see _TOPP_K) ----------------------------------------


@triton.jit
def _pick(vec, js, j):
    """vec[j] of a small vector (0 when j is past its end)."""
    return tl.sum(tl.where(js == j, vec, 0), axis=0)


@triton.jit
def _topp_row(temps_ptr, top_p_ptr, top_p_scalar, req_idx, HAS_TOPP_PTR: tl.constexpr):
    """(the row searches a top-p cutoff, its temperature, its top_p)."""
    temp = tl.load(temps_ptr + req_idx).to(tl.float32)
    if HAS_TOPP_PTR:
        top_p = tl.load(top_p_ptr + req_idx).to(tl.float32)
    else:
        top_p = top_p_scalar
    return (temp > _GREEDY_T) & (top_p < 1.0), temp, top_p


@triton.jit
def _ge_records(row_ptr, lo, hi, inv_t, m, thr, K: tl.constexpr, SB: tl.constexpr):
    """Over columns [lo, hi) of a row: for each of the K thresholds, the mass
    sum exp(y - m) and the count of y >= thr[j]. The mass accumulates per
    lane with one final reduce; the integer count is reduced every block."""
    accm = tl.zeros((K, SB), tl.float32)
    accc = tl.zeros((K,), tl.int32)
    for start in range(lo, hi, SB):
        offs = start + tl.arange(0, SB)
        y = (
            tl.load(row_ptr + offs, mask=offs < hi, other=float("-inf")).to(tl.float32)
            * inv_t
        )
        e = tl.exp(y - m)
        ge = y[None, :] >= thr[:, None]
        accm += tl.where(ge, e[None, :], 0.0)
        accc += tl.sum(ge.to(tl.int32), axis=1)
    return tl.sum(accm, axis=1), accc


@triton.jit
def _sum_records(m_ptr, c_ptr, nch, K: tl.constexpr):
    """Per-chunk [nch, K] records summed over the chunks, in chunk order."""
    js = tl.arange(0, K)
    M = tl.zeros((K,), tl.float32)
    C = tl.zeros((K,), tl.int32)
    for ch in range(nch):
        M += tl.load(m_ptr + ch * K + js)
        C += tl.load(c_ptr + ch * K + js)
    return M, C


@triton.jit
def _topp_thresholds(a, w, K: tl.constexpr):
    """Split [a, a + w) K ways: the thresholds a + j * w / K, j < K, and
    their spacing w / K."""
    w = w / K
    return a + tl.arange(0, K).to(tl.float32) * w, w


@triton.jit
def _topp_narrow(thr, M, C, target, b, mhi, above_b, K: tl.constexpr):
    """One top-p step: over the K ascending thresholds ``thr`` of an interval
    ending at ``b`` (mass ``mhi`` and count ``above_b`` at or above b), with
    the mass M and count C at or above each threshold, the cutoff lies in
    [a, b') for a the highest threshold whose mass still reaches ``target``.
    Returns (its index, a, b', mass at or above b', tokens in [a, b'), mass
    at or above a); b' is ``b`` for the top threshold."""
    js = tl.arange(0, K)
    kk = tl.max(tl.where(M >= target, js, 0), axis=0)
    top = kk + 1 >= K
    a = _pick(thr, js, kk)
    b = tl.where(top, b, _pick(thr, js, kk + 1))
    mhi = tl.where(top, mhi, _pick(M, js, kk + 1))
    n = _pick(C, js, kk) - tl.where(top, above_b, _pick(C, js, kk + 1))
    return kk, a, b, mhi, n, _pick(M, js, kk)


@triton.jit
def _topp_level1_narrow(thr, M, C, target, K: tl.constexpr):
    """`_topp_narrow` over the level-1 thresholds, whose interval is open
    above (b = +inf, nothing at or above it)."""
    return _topp_narrow(
        thr,
        M,
        C,
        target,
        tl.zeros((), tl.float32) + float("inf"),
        tl.zeros((), tl.float32),
        tl.zeros((), tl.int32),
        K,
    )


@triton.jit(do_not_specialize=["top_p_scalar", "rank_id"])
def _topp_level1_kernel(
    target_ptr,  # [N, V], or the full logits (IN_PLACE)
    target_logits_indices_ptr,  # [N] (IN_PLACE): logits row of each verify row
    cu_num_draft_tokens_ptr,
    temps_ptr,
    top_p_ptr,
    top_p_scalar,
    tstat_ptr,  # [N, NCH, 2] `_verify_stats_partial_kernel`'s target records
    thr1_ptr,  # [N, K] out: level-1 thresholds
    l1m_ptr,  # [N, NCH, K] out: chunk mass at or above each threshold
    l1c_ptr,  # [N, NCH, K] out: chunk count at or above each threshold
    V,
    rank_id,
    QTP: tl.constexpr,
    HAS_TOPP_PTR: tl.constexpr,
    K: tl.constexpr,
    SB: tl.constexpr,
    CHUNK: tl.constexpr,
    IN_PLACE: tl.constexpr,
):
    """K evenly spaced thresholds over the cutoff's bracket [m - c, m]; this
    chunk's mass and count at or above each."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1)
    c = tl.program_id(2)
    nch = tl.num_programs(2)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    if k >= n:
        return
    r = start + k
    active, temp, top_p = _topp_row(
        temps_ptr, top_p_ptr, top_p_scalar, req_idx, HAS_TOPP_PTR
    )
    if active & ((r % QTP) == rank_id):
        m, _, _ = _combine_stats(tstat_ptr + r * nch * 2, nch)
        cw = tl.log(V * 1.0) - tl.log(tl.maximum(1.0 - top_p, 1e-30)) + 1.0
        js = tl.arange(0, K)
        thr, _ = _topp_thresholds(m - cw, cw, K)
        if c == 0:
            tl.store(thr1_ptr + r * K + js, thr)
        lo = c * CHUNK
        hi = tl.minimum(lo + CHUNK, V)
        mm, cc = _ge_records(
            target_ptr + _target_row(target_logits_indices_ptr, r, IN_PLACE) * V,
            lo,
            hi,
            1.0 / temp,
            m,
            thr,
            K,
            SB,
        )
        tl.store(l1m_ptr + (r * nch + c) * K + js, mm)
        tl.store(l1c_ptr + (r * nch + c) * K + js, cc)


@triton.jit(do_not_specialize=["top_p_scalar", "rank_id"])
def _topp_level2_kernel(
    target_ptr,  # [N, V], or the full logits (IN_PLACE)
    target_logits_indices_ptr,  # [N] (IN_PLACE): logits row of each verify row
    cu_num_draft_tokens_ptr,
    temps_ptr,
    top_p_ptr,
    top_p_scalar,
    tstat_ptr,
    thr1_ptr,
    l1m_ptr,
    l1c_ptr,
    thr2_ptr,  # [N, K] out: level-2 thresholds
    l2m_ptr,  # [N, NCH, K] out
    l2c_ptr,  # [N, NCH, K] out
    V,
    rank_id,
    QTP: tl.constexpr,
    HAS_TOPP_PTR: tl.constexpr,
    K: tl.constexpr,
    SB: tl.constexpr,
    CHUNK: tl.constexpr,
    RB: tl.constexpr,
    IN_PLACE: tl.constexpr,
):
    """The level-1 interval holding the cutoff (`_topp_narrow`), split K ways
    again (`_topp_thresholds`) when it holds more than RB tokens."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1)
    c = tl.program_id(2)
    nch = tl.num_programs(2)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    if k >= n:
        return
    r = start + k
    active, temp, top_p = _topp_row(
        temps_ptr, top_p_ptr, top_p_scalar, req_idx, HAS_TOPP_PTR
    )
    if active & ((r % QTP) == rank_id):
        m, z, _ = _combine_stats(tstat_ptr + r * nch * 2, nch)
        target = top_p * z
        js = tl.arange(0, K)
        M, C = _sum_records(l1m_ptr + r * nch * K, l1c_ptr + r * nch * K, nch, K)
        thr = tl.load(thr1_ptr + r * K + js)
        _, a1, _, _, n1, _ = _topp_level1_narrow(thr, M, C, target, K)
        if n1 > RB:
            thr2, _ = _topp_thresholds(a1, _pick(thr, js, 1) - _pick(thr, js, 0), K)
            if c == 0:
                tl.store(thr2_ptr + r * K + js, thr2)
            lo = c * CHUNK
            hi = tl.minimum(lo + CHUNK, V)
            mm, cc = _ge_records(
                target_ptr + _target_row(target_logits_indices_ptr, r, IN_PLACE) * V,
                lo,
                hi,
                1.0 / temp,
                m,
                thr2,
                K,
                SB,
            )
            tl.store(l2m_ptr + (r * nch + c) * K + js, mm)
            tl.store(l2c_ptr + (r * nch + c) * K + js, cc)


@triton.jit
def _final_interval(
    thr1_ptr,
    l1m_ptr,
    l1c_ptr,
    thr2_ptr,
    l2m_ptr,
    l2c_ptr,
    r,
    nch,
    target,
    K: tl.constexpr,
    RB: tl.constexpr,
):
    """(a, b, mass at or above b, threshold spacing, tokens in [a, b), level 2
    ran, index, k1): the interval [a, b) holding the cutoff, from level 2
    where it ran (the level-1 interval held more than RB tokens); b is +inf
    for the top interval."""
    js = tl.arange(0, K)
    M, C = _sum_records(l1m_ptr + r * nch * K, l1c_ptr + r * nch * K, nch, K)
    thr = tl.load(thr1_ptr + r * K + js)
    k1, a, b, mhi, total, _ = _topp_level1_narrow(thr, M, C, target, K)
    need2 = total > RB
    w = _pick(thr, js, 1) - _pick(thr, js, 0)
    kk = k1
    if need2:
        M2, C2 = _sum_records(l2m_ptr + r * nch * K, l2c_ptr + r * nch * K, nch, K)
        thr2 = tl.load(thr2_ptr + r * K + js)
        kk, a, b, mhi, total, _ = _topp_narrow(
            thr2, M2, C2, target, b, mhi, _pick(C, js, k1 + 1), K
        )
        w = w / K
    return a, b, mhi, w, total, need2, kk, k1


@triton.jit
def _chunk_interval_counts(
    l1c_ptr, l2c_ptr, r, nch, need2, kk, k1, K: tl.constexpr, NCH: tl.constexpr
):
    """Every chunk's count in the final interval, one lane per chunk (lanes
    past nch hold 0): its count at or above a minus its count at or above b
    (records of the level the interval came from)."""
    chs = tl.arange(0, NCH)
    live = chs < nch
    base1 = l1c_ptr + (r * nch + chs) * K
    # k1 is a scalar: broadcast it so the mask is a [NCH] vector, which the
    # Triton interpreter requires.
    has_b1 = live & (chs * 0 + k1 + 1 < K)
    above_b1 = tl.load(base1 + tl.minimum(k1 + 1, K - 1), has_b1, 0)
    cnt = tl.load(base1 + k1, live, 0) - above_b1
    if need2:
        base2 = l2c_ptr + (r * nch + chs) * K
        above = tl.where(
            kk + 1 < K, tl.load(base2 + tl.minimum(kk + 1, K - 1), live, 0), above_b1
        )
        cnt = tl.load(base2 + kk, live, 0) - above
    return cnt


@triton.jit
def _compact_interval(
    row, lo, hi, inv_t, m, a, b, ybuf, pos0, RB: tl.constexpr, BLOCK: tl.constexpr
):
    """Append the (y, exp(y - m)) of columns [lo, hi) with y in [a, b) to
    ``ybuf`` ([2, RB]) from position pos0 on, in column order."""
    pos0 = tl.zeros((), tl.int32) + pos0
    for st in range(lo, hi, BLOCK):
        offs = st + tl.arange(0, BLOCK)
        y = (
            tl.load(row + offs, mask=offs < hi, other=float("-inf")).to(tl.float32)
            * inv_t
        )
        inb = (y >= a) & (y < b)
        csum = tl.cumsum(inb.to(tl.int32), axis=0)
        pos = pos0 + csum - 1
        tl.store(ybuf + pos, y, mask=inb)
        tl.store(ybuf + RB + pos, tl.exp(y - m), mask=inb)
        pos0 += tl.sum(inb.to(tl.int32), axis=0)


@triton.jit(do_not_specialize=["top_p_scalar", "rank_id"])
def _topp_compact_kernel(
    target_ptr,  # [N, V], or the full logits (IN_PLACE)
    target_logits_indices_ptr,  # [N] (IN_PLACE): logits row of each verify row
    cu_num_draft_tokens_ptr,
    temps_ptr,
    top_p_ptr,
    top_p_scalar,
    tstat_ptr,
    thr1_ptr,
    l1m_ptr,
    l1c_ptr,
    thr2_ptr,
    l2m_ptr,
    l2c_ptr,
    fin_ptr,  # [N, 4] out: a, b, mass at or above b, threshold spacing
    fin_n_ptr,  # [N] int32 out: tokens in [a, b)
    cand_ptr,  # [N, 2, RB] out: the interval's (y, exp(y - m)), chunk then column order
    V,
    rank_id,
    QTP: tl.constexpr,
    HAS_TOPP_PTR: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    RB: tl.constexpr,
    NCH: tl.constexpr,
    IN_PLACE: tl.constexpr,
):
    """Compact the final interval's tokens: each chunk writes at the sum of
    the earlier chunks' counts (all chunks' counts read at once, NCH >= nch
    lanes), so the order is fixed. An interval of more than RB tokens is left
    to `_topp_cutoff`."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1)
    c = tl.program_id(2)
    nch = tl.num_programs(2)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    if k >= n:
        return
    r = start + k
    active, temp, top_p = _topp_row(
        temps_ptr, top_p_ptr, top_p_scalar, req_idx, HAS_TOPP_PTR
    )
    if active & ((r % QTP) == rank_id):
        m, z, _ = _combine_stats(tstat_ptr + r * nch * 2, nch)
        a, b, mhi, w, total, need2, kk, k1 = _final_interval(
            thr1_ptr,
            l1m_ptr,
            l1c_ptr,
            thr2_ptr,
            l2m_ptr,
            l2c_ptr,
            r,
            nch,
            top_p * z,
            K,
            RB,
        )
        cnt = _chunk_interval_counts(l1c_ptr, l2c_ptr, r, nch, need2, kk, k1, K, NCH)
        off = tl.sum(tl.where(tl.arange(0, NCH) < c, cnt, 0), axis=0)
        if c == 0:
            tl.store(fin_ptr + r * 4, a)
            tl.store(fin_ptr + r * 4 + 1, b)
            tl.store(fin_ptr + r * 4 + 2, mhi)
            tl.store(fin_ptr + r * 4 + 3, w)
            tl.store(fin_n_ptr + r, total)
        if total <= RB:
            lo = c * CHUNK
            _compact_interval(
                target_ptr + _target_row(target_logits_indices_ptr, r, IN_PLACE) * V,
                lo,
                tl.minimum(lo + CHUNK, V),
                1.0 / temp,
                m,
                a,
                b,
                cand_ptr + r.to(tl.int64) * (2 * RB),
                off,
                RB,
                BLOCK,
            )


@triton.jit
def _topp_finish(a, b, total, mhi, ybuf, m, target, RB: tl.constexpr):
    """(tau, zk): bisection on [a, b) over the interval's compacted tokens,
    held in registers; mass at or above b added."""
    idx = tl.arange(0, RB)
    cm = idx < total
    yv = tl.load(ybuf + idx, mask=cm, other=float("-inf"))
    ev = tl.load(ybuf + RB + idx, mask=cm, other=0.0)
    lo = a
    hi = tl.where(b == float("inf"), m, b)
    zk = mhi + tl.sum(ev, axis=0)
    for _ in range(_TOPP_ITERS):
        mid = 0.5 * (lo + hi)
        s = mhi + tl.sum(tl.where(yv >= mid, ev, 0.0), axis=0)
        ge = s >= target
        lo = tl.where(ge, mid, lo)
        zk = tl.where(ge, s, zk)
        hi = tl.where(ge, hi, mid)
    return lo, zk


@triton.jit
def _topp_cutoff(
    row,
    fin_ptr,
    fin_n_ptr,
    ybuf,
    V,
    inv_t,
    m,
    target,
    K: tl.constexpr,
    SB: tl.constexpr,
    RB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """(tau, zk) of a top-p row from `_topp_compact_kernel`'s final interval
    [a, b). An interval of more than RB tokens is split K ways again over the
    whole row (the level kernels' step: `_topp_thresholds`, `_ge_records`,
    `_topp_narrow`) until it fits, then
    compacted here; one still too large after _TOPP_ROW_LEVELS splits is kept
    whole (its tokens tie): tau = a, zk = mass at or above a."""
    a = tl.load(fin_ptr)
    b = tl.load(fin_ptr + 1)
    mhi = tl.load(fin_ptr + 2)
    w = tl.load(fin_ptr + 3)
    total = tl.load(fin_n_ptr)
    mass_a = tl.zeros((), tl.float32)
    if total > RB:
        js = tl.arange(0, K)
        level = tl.zeros((), tl.int32)
        while (total > RB) & (level < _TOPP_ROW_LEVELS):
            thr, w = _topp_thresholds(a, w, K)
            M, C = _ge_records(row, 0, V, inv_t, m, thr, K, SB)
            above_b = _pick(C, js, 0) - total  # thr[0] is a
            _, a, b, mhi, total, mass_a = _topp_narrow(
                thr, M, C, target, b, mhi, above_b, K
            )
            level += 1
        if total <= RB:
            _compact_interval(row, 0, V, inv_t, m, a, b, ybuf, 0, RB, BLOCK)
            # The bisection reads back what other lanes just wrote.
            tl.debug_barrier()
    if total <= RB:
        tau, zk = _topp_finish(a, b, total, mhi, ybuf, m, target, RB)
    else:
        tau = a
        zk = mass_a
    return tau, zk


@triton.jit(do_not_specialize=["q_width", "q_vs", "q_cols", "q_stride", "rank_id"])
def _verify_stats_partial_kernel(
    target_ptr,  # [N, V], or the full logits (IN_PLACE)
    input_ids_ptr,
    target_logits_indices_ptr,
    cu_num_draft_tokens_ptr,  # [bs]
    temps_ptr,  # [bs]
    src_ptr,  # [bs]
    q_ptr,  # [max_bs, W, q_stride]
    q_ids_ptr,  # [max_bs, W]
    q_temps_ptr,  # [max_bs]
    tstat_ptr,  # [N, NCH, 2] out: target chunk max, sumexp
    targ_ptr,  # [N, NCH] int32 out: target chunk lowest argmax
    qstat_ptr,  # [N, NCH, 2] out: q chunk max, sumexp
    V,
    q_width,
    q_vs,
    q_cols,
    q_stride,
    rank_id,
    QTP: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    IN_PLACE: tl.constexpr,
):
    """One CHUNK of a verified row's two full-vocab statistics passes -- the
    target's (on the rank owning the row) and the stored q's -- which
    `_verify_row_stats_kernel` combines. A q chunk past ``q_cols`` stores the
    empty q record (-inf, 0); a row past its request's end stores nothing."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1)
    c = tl.program_id(2)
    nch = tl.num_programs(2)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    if k >= n:
        return
    r = start + k
    lo = c * CHUNK
    row = target_ptr + _target_row(target_logits_indices_ptr, r, IN_PLACE) * V
    x = tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1).to(tl.int32)
    x_ok = (x >= 0) & (x < V)
    own_row = (r % QTP) == rank_id
    temp = tl.load(temps_ptr + req_idx).to(tl.float32)
    greedy = temp <= _GREEDY_T
    tout = tstat_ptr + (r * nch + c) * 2
    hi = tl.minimum(lo + CHUNK, V)
    if greedy:
        if own_row:
            mx, ai = _span_max_argmax(row, lo, hi, BLOCK)
            tl.store(tout, mx)
            tl.store(tout + 1, 0.0)
            tl.store(targ_ptr + r * nch + c, ai)
    else:
        if own_row:
            mx, s, ai = _span_online_stats(row, lo, hi, 1.0 / temp, BLOCK)
            tl.store(tout, mx)
            tl.store(tout + 1, s)
            tl.store(targ_ptr + r * nch + c, ai)
        s_row = tl.load(src_ptr + req_idx)
        if (s_row >= 0) & (k < q_width):
            qid = tl.load(q_ids_ptr + s_row * q_width + k)
            tq = tl.load(q_temps_ptr + s_row).to(tl.float32)
            if (qid == x) & x_ok & (tq > _GREEDY_T):
                qrow = q_ptr + (s_row.to(tl.int64) * q_width + k) * q_stride
                hq = tl.minimum(lo + CHUNK, q_cols)
                mq, zq, _ = _span_online_stats(qrow, lo, hq, 1.0 / tq, BLOCK)
                qout = qstat_ptr + (r * nch + c) * 2
                tl.store(qout, mq)
                tl.store(qout + 1, zq)


@triton.jit(
    do_not_specialize=[
        "num_spec_steps",
        "top_p_scalar",
        "q_width",
        "q_vs",
        "q_cols",
        "q_stride",
        "rank_id",
        "nch",
    ]
)
def _verify_row_stats_kernel(
    target_ptr,  # [N, V], or the full logits (IN_PLACE)
    input_ids_ptr,
    target_logits_indices_ptr,
    cu_num_draft_tokens_ptr,  # [bs]
    temps_ptr,  # [bs]
    top_p_ptr,  # [bs] or dummy
    top_p_scalar,
    src_ptr,  # [bs]
    q_ptr,  # [max_bs, W, q_stride]
    q_ids_ptr,  # [max_bs, W]
    q_temps_ptr,  # [max_bs]
    tstat_ptr,  # [N, nch, 2] `_verify_stats_partial_kernel`'s target records
    targ_ptr,  # [N, nch] and their argmax
    qstat_ptr,  # [N, nch, 2] and its q records
    fin_ptr,  # [N, 4] `_topp_compact_kernel`'s final interval (top-p rows)
    fin_n_ptr,  # [N] its token count
    cand_ptr,  # [N, 2, RB] and its tokens
    rs_ptr,  # [N, 8] out: amax, m, tau, zk, p(x) | local q max, sumexp, x-logit
    hasq_ptr,  # [N] out
    V,
    q_width,
    q_vs,
    q_cols,
    q_stride,
    num_spec_steps,
    rank_id,
    nch,
    QTP: tl.constexpr,
    HAS_TOPP_PTR: tl.constexpr,
    K: tl.constexpr,
    SB: tl.constexpr,
    RB: tl.constexpr,
    BLOCK: tl.constexpr,
    IN_PLACE: tl.constexpr,
):
    """Per verified row. The target half (fields 0-4) is computed only by the
    rank that owns the row (r % QTP), the q half (5-7) by every rank over its
    own vocab columns; one all-gather then hands every rank both. The rows'
    max / sumexp / argmax come from `_verify_stats_partial_kernel`'s chunks;
    a top-p cutoff finishes on `_topp_compact_kernel`'s interval
    (`_topp_cutoff`)."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    if k >= n:
        return
    r = start + k
    out = rs_ptr + r * 8
    row = target_ptr + _target_row(target_logits_indices_ptr, r, IN_PLACE) * V
    x = tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1).to(tl.int32)
    x_ok = (x >= 0) & (x < V)
    xs = tl.where(x_ok, x, 0)
    own_row = (r % QTP) == rank_id

    active, temp, top_p = _topp_row(
        temps_ptr, top_p_ptr, top_p_scalar, req_idx, HAS_TOPP_PTR
    )
    greedy = temp <= _GREEDY_T
    hasq = 0
    if greedy:
        if own_row:
            _, _, best_c = _combine_stats(tstat_ptr + r * nch * 2, nch)
            amax = _combine_argmax(targ_ptr + r * nch, best_c)
            tl.store(out, amax.to(tl.float32))
    else:
        if own_row:
            inv_t = 1.0 / temp
            m, z, best_c = _combine_stats(tstat_ptr + r * nch * 2, nch)
            amax = _combine_argmax(targ_ptr + r * nch, best_c)
            tau = tl.zeros((), tl.float32) + float("-inf")
            zk = z
            if active:
                tau, zk = _topp_cutoff(
                    row,
                    fin_ptr + r * 4,
                    fin_n_ptr + r,
                    cand_ptr + r.to(tl.int64) * (2 * RB),
                    V,
                    inv_t,
                    m,
                    top_p * z,
                    K,
                    SB,
                    RB,
                    BLOCK,
                )
            lx = tl.load(row + xs).to(tl.float32) * inv_t
            px = tl.where(x_ok & (lx >= tau), tl.exp(lx - m) / zk, 0.0)
            tl.store(out, amax.to(tl.float32))
            tl.store(out + 1, m)
            tl.store(out + 2, tau)
            tl.store(out + 3, zk)
            tl.store(out + 4, px)

        s_row = tl.load(src_ptr + req_idx)
        if (s_row >= 0) & (k < q_width):
            qid = tl.load(q_ids_ptr + s_row * q_width + k)
            tq = tl.load(q_temps_ptr + s_row).to(tl.float32)
            if (qid == x) & x_ok & (tq > _GREEDY_T):
                inv_tq = 1.0 / tq
                qrow = q_ptr + (s_row.to(tl.int64) * q_width + k) * q_stride
                mq, zq, _ = _combine_stats(qstat_ptr + r * nch * 2, nch)
                xl = x - q_vs
                own = (xl >= 0) & (xl < q_cols)
                dx = tl.load(qrow + tl.where(own, xl, 0)) * inv_tq
                dx = tl.where(own, dx, float("-inf"))
                tl.store(out + 5, mq)
                tl.store(out + 6, zq)
                tl.store(out + 7, dx)
                hasq = 1
    tl.store(hasq_ptr + r, hasq)


@triton.jit
def _ts(rs_ptr, r, N, QTP: tl.constexpr, F: tl.constexpr):
    """Target-half field F of row r, from the slice of the rank that owns it."""
    return tl.load(rs_ptr + ((r % QTP) * N + r) * 8 + F)


@triton.jit
def _q_comb(rs_ptr, r, N, QTP: tl.constexpr):
    """Combine the per-shard (max, sumexp, x-logit) of row r: (mq, zq, q(x))."""
    mq = tl.load(rs_ptr + r * 8 + 5)
    for j in tl.static_range(1, QTP):
        mq = tl.maximum(mq, tl.load(rs_ptr + (j * N + r) * 8 + 5))
    zq = tl.zeros((), tl.float32)
    dx = tl.zeros((), tl.float32) + float("-inf")
    for j in tl.static_range(QTP):
        base = rs_ptr + (j * N + r) * 8
        zq += tl.load(base + 6) * tl.exp(tl.load(base + 5) - mq)
        dx = tl.maximum(dx, tl.load(base + 7))
    qx = tl.exp(dx - mq) / zq
    return mq, zq, qx


@triton.jit
def _cum_log_ratio(rs_ptr, start, upto, N, QTP: tl.constexpr):
    """log P_upto = running min(0, log P + log p/q) over rows < upto; P_0 = 1."""
    logp = tl.zeros((), tl.float32)
    for j in range(upto):
        px = _ts(rs_ptr, start + j, N, QTP, 4)
        _, _, qx = _q_comb(rs_ptr, start + j, N, QTP)
        lr = tl.where(px > 0.0, tl.log(px) - tl.log(qx), float("-inf"))
        logp = tl.minimum(logp + lr, 0.0)
    return logp


@triton.jit(do_not_specialize=["q_width", "q_vs", "q_cols", "q_stride"])
def _block_residual_mass_kernel(
    target_ptr,
    target_logits_indices_ptr,  # [N] (IN_PLACE): logits row of each verify row
    cu_num_draft_tokens_ptr,
    temps_ptr,
    src_ptr,
    q_ptr,
    q_temps_ptr,
    rs_ptr,  # [QTP, N, 8]
    hasq_ptr,
    rm_ptr,  # [N, NCH] out (this shard's part, per vocab chunk)
    N,
    V,
    q_width,
    q_vs,
    q_cols,
    q_stride,
    QTP: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    IN_PLACE: tl.constexpr,
):
    """rm[row k, chunk c] = sum_v max(P_k p_k(v) - q_k(v), 0) for 1 <= k < n,
    over chunk c of this rank's vocab columns; `_prob_decide_kernel` sums the
    chunks in order."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1) + 1
    c = tl.program_id(2)
    nch = tl.num_programs(2)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    if k >= n:
        return
    r = start + k
    temp = tl.load(temps_ptr + req_idx).to(tl.float32)
    acc = tl.zeros((), tl.float32)
    if (temp > _GREEDY_T) & (tl.load(hasq_ptr + r) != 0):
        logp = _cum_log_ratio(rs_ptr, start, k, N, QTP)
        P = tl.exp(logp)
        s_row = tl.load(src_ptr + req_idx)
        inv_t = 1.0 / temp
        inv_tq = 1.0 / tl.load(q_temps_ptr + s_row).to(tl.float32)
        m = _ts(rs_ptr, r, N, QTP, 1)
        tau = _ts(rs_ptr, r, N, QTP, 2)
        zk = _ts(rs_ptr, r, N, QTP, 3)
        mq, zq, _ = _q_comb(rs_ptr, r, N, QTP)
        row = (
            target_ptr + _target_row(target_logits_indices_ptr, r, IN_PLACE) * V + q_vs
        )
        qrow = q_ptr + (s_row.to(tl.int64) * q_width + k) * q_stride
        if P > 0.0:
            lo = c * CHUNK
            hi = tl.minimum(lo + CHUNK, q_cols)
            accv = tl.zeros((BLOCK,), tl.float32)
            for st in range(lo, hi, BLOCK):
                offs = st + tl.arange(0, BLOCK)
                mask = (offs < hi) & (q_vs + offs < V)
                y = (
                    tl.load(row + offs, mask=mask, other=float("-inf")).to(tl.float32)
                    * inv_t
                )
                p = tl.where(mask & (y >= tau), tl.exp(y - m) / zk, 0.0)
                d = tl.load(qrow + offs, mask=mask, other=float("-inf")) * inv_tq
                q = tl.where(mask, tl.exp(d - mq) / zq, 0.0)
                accv += tl.where(mask, tl.maximum(P * p - q, 0.0), 0.0)
            acc = tl.sum(accv, axis=0)
    tl.store(rm_ptr + r * nch + c, acc)


@triton.jit(do_not_specialize=["num_spec_steps", "seed_acc", "rm_nch"])
def _prob_decide_kernel(
    output_token_ids_ptr,  # [bs, num_spec_steps + 1]
    dec_ptr,  # [3, bs] f32: num_acc, rejected, log coef
    input_ids_ptr,
    target_logits_indices_ptr,
    cu_num_draft_tokens_ptr,
    temps_ptr,
    rs_ptr,  # [QTP, N, 8]
    hasq_ptr,
    rmall_ptr,  # [QTP, N, rm_nch]
    N,
    num_spec_steps,
    seed_acc,
    rm_nch,
    QTP: tl.constexpr,
    USE_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    bs = tl.num_programs(0)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    temp = tl.load(temps_ptr + req_idx).to(tl.float32)
    greedy = temp <= _GREEDY_T
    out_row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)

    num_acc = n
    rejected = False
    coef_log = tl.zeros((), tl.float32)
    if greedy:
        for k in range(n):
            if not rejected:
                x = _draft_token(input_ids_ptr, target_logits_indices_ptr, start + k)
                a = _ts(rs_ptr, start + k, N, QTP, 0).to(tl.int32)
                if x != a:
                    rejected = True
                    num_acc = k
    else:
        all_q = n > 0
        for k in range(n):
            all_q = all_q & (tl.load(hasq_ptr + start + k) != 0)
        use_block = False
        if USE_BLOCK:
            use_block = all_q
        if use_block:
            tau_len = tl.zeros((), tl.int32)
            logp = tl.zeros((), tl.float32)
            for k in range(n):
                px = _ts(rs_ptr, start + k, N, QTP, 4)
                _, _, qx = _q_comb(rs_ptr, start + k, N, QTP)
                lr = tl.where(px > 0.0, tl.log(px) - tl.log(qx), float("-inf"))
                logp = tl.minimum(logp + lr, 0.0)  # log P_{k+1}
                P = tl.exp(logp)
                if k < n - 1:
                    rm = tl.zeros((), tl.float32)
                    for j in tl.static_range(QTP):
                        rbase = rmall_ptr + (j * N + start + k + 1) * rm_nch
                        for c in range(rm_nch):
                            rm += tl.load(rbase + c)
                    denom = rm + 1.0 - P
                    h = tl.where(denom > 0.0, rm / denom, 1.0)
                else:
                    h = P
                u = tl.maximum(tl.rand(seed_acc, start + k), _RAND_MIN)
                tau_len = tl.where(u <= h, k + 1, tau_len)
            if tau_len < n:
                rejected = True
                num_acc = tau_len
                coef_log = _cum_log_ratio(rs_ptr, start, tau_len, N, QTP)
        else:
            for k in range(n):
                if not rejected:
                    px = _ts(rs_ptr, start + k, N, QTP, 4)
                    u = tl.rand(seed_acc, start + k)
                    if tl.load(hasq_ptr + start + k) != 0:
                        _, _, qx = _q_comb(rs_ptr, start + k, N, QTP)
                        acc = u * qx < px
                    else:
                        acc = u < px
                    if not acc:
                        rejected = True
                        num_acc = k
    for k in range(n):
        if k < num_acc:
            tok = _draft_token(input_ids_ptr, target_logits_indices_ptr, start + k)
            tl.store(out_row + k, tl.cast(tok, tl.int32))
        elif k > num_acc:
            tl.store(out_row + k, tl.cast(-1, tl.int32))
    # rejected <=> num_acc < n
    tl.store(dec_ptr + req_idx, num_acc.to(tl.float32))
    tl.store(dec_ptr + bs + req_idx, tl.where(num_acc < n, 1.0, 0.0))
    tl.store(dec_ptr + 2 * bs + req_idx, coef_log)


@triton.jit(do_not_specialize=["q_width", "q_vs", "q_cols", "q_stride", "seed_res"])
def _prob_resample_kernel(
    res_ptr,  # [bs, NCH, 4] f32: per vocab chunk best residual score, id, best p score, id
    dec_ptr,
    target_ptr,
    input_ids_ptr,
    target_logits_indices_ptr,
    cu_num_draft_tokens_ptr,
    temps_ptr,
    src_ptr,
    q_ptr,
    q_temps_ptr,
    rs_ptr,  # [QTP, N, 8]
    hasq_ptr,
    N,
    V,
    q_width,
    q_vs,
    q_cols,
    q_stride,
    seed_res,
    QTP: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    IN_PLACE: tl.constexpr,
):
    """One vocab chunk of this rank's columns of the residual draw at the
    rejected row: norm(max(c p - q, 0)) (one-hot q: p without the draft), with
    p's own draw as the fallback when the residual is all zero. Gumbel-max
    splits exactly: `_prob_finish_kernel` takes the best chunk."""
    req_idx = tl.program_id(0)
    chunk = tl.program_id(1)
    bs = tl.num_programs(0)
    nch = tl.num_programs(1)
    start, _n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    temp = tl.load(temps_ptr + req_idx).to(tl.float32)
    num_acc = tl.load(dec_ptr + req_idx).to(tl.int32)
    rejected = tl.load(dec_ptr + bs + req_idx) > 0.5
    brv = tl.zeros((BLOCK,), tl.float32) + float("-inf")
    briv = tl.zeros((BLOCK,), tl.int32)
    bpv = tl.zeros((BLOCK,), tl.float32) + float("-inf")
    bpiv = tl.zeros((BLOCK,), tl.int32)
    if rejected & (temp > _GREEDY_T):
        r = start + num_acc
        coef = tl.exp(tl.load(dec_ptr + 2 * bs + req_idx))
        hq = tl.load(hasq_ptr + r) != 0
        x = _draft_token(input_ids_ptr, target_logits_indices_ptr, r)
        inv_t = 1.0 / temp
        m = _ts(rs_ptr, r, N, QTP, 1)
        tau = _ts(rs_ptr, r, N, QTP, 2)
        zk = _ts(rs_ptr, r, N, QTP, 3)
        s_row = tl.maximum(tl.load(src_ptr + req_idx), 0)
        inv_tq = 1.0 / tl.maximum(
            tl.load(q_temps_ptr + s_row).to(tl.float32), _GREEDY_T
        )
        mq, zq, _ = _q_comb(rs_ptr, r, N, QTP)
        row = (
            target_ptr + _target_row(target_logits_indices_ptr, r, IN_PLACE) * V + q_vs
        )
        qrow = q_ptr + (s_row.to(tl.int64) * q_width + num_acc) * q_stride
        lo = chunk * CHUNK
        hi = tl.minimum(lo + CHUNK, q_cols)
        for st in range(lo, hi, BLOCK):
            offs = st + tl.arange(0, BLOCK)
            gid = q_vs + offs
            mask = (offs < hi) & (gid < V)
            y = (
                tl.load(row + offs, mask=mask, other=float("-inf")).to(tl.float32)
                * inv_t
            )
            p = tl.where(mask & (y >= tau), tl.exp(y - m) / zk, 0.0)
            if hq:
                d = tl.load(qrow + offs, mask=mask, other=float("-inf")) * inv_tq
                q = tl.where(mask, tl.exp(d - mq) / zq, 0.0)
                w = tl.maximum(coef * p - q, 0.0)
            else:
                w = tl.where(gid == x, 0.0, p)
            g = _gumbel(seed_res, r * V + gid)
            sr = tl.where(mask & (w > 0.0), tl.log(w) + g, float("-inf"))
            sp = tl.where(mask & (p > 0.0), tl.log(p) + g, float("-inf"))
            t = sr > brv
            briv = tl.where(t, gid, briv)
            brv = tl.where(t, sr, brv)
            t = sp > bpv
            bpiv = tl.where(t, gid, bpiv)
            bpv = tl.where(t, sp, bpv)
    best_r = tl.max(brv, axis=0)
    best_ri = tl.min(tl.where(brv == best_r, briv, 2147483647), axis=0)
    best_p = tl.max(bpv, axis=0)
    best_pi = tl.min(tl.where(bpv == best_p, bpiv, 2147483647), axis=0)
    best_ri = tl.where(best_r > float("-inf"), best_ri, 0)
    best_pi = tl.where(best_p > float("-inf"), best_pi, 0)
    out = res_ptr + (req_idx * nch + chunk) * 4
    tl.store(out, best_r)
    tl.store(out + 1, best_ri.to(tl.float32))
    tl.store(out + 2, best_p)
    tl.store(out + 3, best_pi.to(tl.float32))


@triton.jit(do_not_specialize=["num_spec_steps", "res_nch"])
def _prob_finish_kernel(
    output_token_ids_ptr,
    verdict_ptr,
    dec_ptr,
    resall_ptr,  # [QTP, bs, res_nch, 4]
    cu_num_draft_tokens_ptr,
    temps_ptr,
    rs_ptr,  # [QTP, N, 8]
    bonus_token_ids_ptr,
    bonus_logits_indices_ptr,
    N,
    num_spec_steps,
    res_nch,
    QTP: tl.constexpr,
):
    req_idx = tl.program_id(0)
    bs = tl.num_programs(0)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    temp = tl.load(temps_ptr + req_idx).to(tl.float32)
    num_acc = tl.load(dec_ptr + req_idx).to(tl.int32)
    rejected = tl.load(dec_ptr + bs + req_idx) > 0.5
    out_row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)
    if rejected:
        if temp <= _GREEDY_T:
            tok = _ts(rs_ptr, start + num_acc, N, QTP, 0).to(tl.int32)
        else:
            # Shards then chunks: ascending vocab order, so the strict compare
            # keeps the lowest id among equal scores, as a single scan would.
            # Every record of an all -inf scan carries id 0, so starting from
            # (-inf, 0) reproduces the first record's fallback.
            best_r = tl.zeros((), tl.float32) + float("-inf")
            id_r = tl.zeros((), tl.float32)
            best_p = tl.zeros((), tl.float32) + float("-inf")
            id_p = tl.zeros((), tl.float32)
            for j in tl.static_range(QTP):
                for c in range(res_nch):
                    base = resall_ptr + ((j * bs + req_idx) * res_nch + c) * 4
                    v = tl.load(base)
                    t = v > best_r
                    id_r = tl.where(t, tl.load(base + 1), id_r)
                    best_r = tl.where(t, v, best_r)
                    v = tl.load(base + 2)
                    t = v > best_p
                    id_p = tl.where(t, tl.load(base + 3), id_p)
                    best_p = tl.where(t, v, best_p)
            tok = tl.where(best_r > float("-inf"), id_r, id_p).to(tl.int32)
        tl.store(out_row + num_acc, tl.cast(tok, tl.int32))
        last_checked = num_acc
    else:
        last_checked = n - 1
    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        n,
        last_checked,
        rejected,
        num_spec_steps,
    )


class _TopPWorkspace:
    """The top-p search's per-step scratch, carved from one allocation: the
    compacted final-interval tokens ``cand`` [N, 2, RB]; the two chunked
    levels' thresholds ``thr`` [2, N, K] and per-chunk mass / count records
    ``lm`` / ``lc`` [2, N, nch, K]; the final interval ``fin`` [N, 4] (a, b,
    mass at or above b, threshold spacing); and its int32 token count
    ``fin_n`` [N]. ``nch`` 0 (no top-p): only the per-row slots, never
    read."""

    def __init__(self, nrow: int, nch: int, device):
        f32, i32 = torch.float32, torch.int32
        k = _TOPP_K if nch else 0
        layout = (
            ("cand", (nrow, 2, _TOPP_RB if nch else 0), f32),
            ("thr", (2, nrow, k), f32),
            ("lm", (2, nrow, nch, k), f32),
            ("lc", (2, nrow, nch, k), i32),
            ("fin", (nrow, 4), f32),
            ("fin_n", (nrow,), i32),
        )
        sizes = [torch.Size(shape).numel() for _, shape, _ in layout]
        words = torch.empty(sum(sizes), dtype=i32, device=device)
        off = 0
        for (name, shape, dtype), n in zip(layout, sizes):
            part = words[off : off + n]
            setattr(self, name, (part if dtype is i32 else part.view(f32)).view(shape))
            off += n


def prob_rejection_sample(
    input_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    num_spec_steps: int,
    cu_num_draft_tokens: torch.Tensor,  # [bs] inclusive
    target_logits: torch.Tensor,  # [N, V], or the full logits (in_place)
    bonus_token_ids: torch.Tensor,  # [bs]
    bonus_logits_indices: torch.Tensor,  # [bs]
    temperatures: torch.Tensor,  # [bs] (clamped to eps)
    top_ps,  # None | float | Tensor[bs]
    src: torch.Tensor,  # [bs] int32
    state: ProbDraftState,
    use_block: bool = False,
    tp_group=None,  # required when state.q_tp > 1
    use_custom: bool = True,
    bonus_ready=None,  # Event: bonus_token_ids produced on another stream
    in_place: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`in_place`: `target_logits` is the step's full logits and verify row r
    is its row `target_logits_indices[r]` (the target rows, read in place
    rather than copied out first); otherwise `target_logits` is those rows."""
    if in_place:
        # The kernels step rows by the vocab width and index the row list
        # with bare pointer arithmetic.
        assert target_logits.is_contiguous() and target_logits_indices.is_contiguous()
        num_rows, vocab = target_logits_indices.shape[0], target_logits.shape[1]
    else:
        target_logits = target_logits.contiguous()
        num_rows, vocab = target_logits.shape
    bs = cu_num_draft_tokens.shape[0]
    device = target_logits.device
    q_width = state.q.shape[1]
    q_stride = state.q.shape[2]
    qtp = state.q_tp
    assert qtp == 1 or (tp_group is not None and tp_group.world_size == qtp)
    rank_id = tp_group.rank_in_group if qtp > 1 else 0
    output = torch.empty((bs, num_spec_steps + 1), dtype=torch.int32, device=device)
    verdict = torch.empty((3, bs), dtype=torch.int32, device=device)
    if bs == 0:
        return output, verdict
    f32 = {"dtype": torch.float32, "device": device}
    nrow = max(num_rows, 1)
    # Vocab chunks: of the target row for the row statistics (which also span
    # the q columns), of this rank's q columns for the residual passes.
    nch_q = max(1, triton.cdiv(state.q_cols, _CHUNK))
    nch_stats = max(nch_q, triton.cdiv(vocab, _CHUNK))
    rs = torch.empty((nrow, 8), **f32)
    rm = torch.empty((nrow, nch_q), **f32)
    hasq = torch.empty(nrow, dtype=torch.int32, device=device)
    tstat = torch.empty((nrow, nch_stats, 2), **f32)
    targ = torch.empty((nrow, nch_stats), dtype=torch.int32, device=device)
    qstat = torch.empty((nrow, nch_stats, 2), **f32)
    dec = torch.empty((3, bs), **f32)
    res = torch.empty((bs * nch_q, 4), **f32)
    if isinstance(top_ps, torch.Tensor) and top_ps.numel() > 1:
        top_p_ptr, top_p_scalar, has_ptr = top_ps.to(torch.float32), 1.0, True
    else:
        if isinstance(top_ps, torch.Tensor):
            top_ps = float(top_ps.item())
        top_p_ptr, has_ptr = temperatures, False
        top_p_scalar = 1.0 if top_ps is None else float(top_ps)
    temps = temperatures.to(torch.float32)
    topp = has_ptr or top_p_scalar < 1.0
    ws = _TopPWorkspace(nrow, nch_stats if topp else 0, device)
    gather = (
        (lambda t: tp_group.all_gather(t, dim=0, use_custom=use_custom))
        if qtp > 1
        else (lambda t: t)
    )
    step = state.verify_step
    state.verify_step += 1
    seed_acc = (step % (1 << 28)) * 4 + 1
    seed_res = (step % (1 << 28)) * 4 + 2
    if num_rows > 0 and num_spec_steps > 0:
        _verify_stats_partial_kernel[(bs, num_spec_steps, nch_stats)](
            target_logits,
            input_ids,
            target_logits_indices,
            cu_num_draft_tokens,
            temps,
            src,
            state.q,
            state.ids,
            state.temps,
            tstat,
            targ,
            qstat,
            vocab,
            q_width,
            state.q_vs,
            state.q_cols,
            q_stride,
            rank_id,
            QTP=qtp,
            IN_PLACE=in_place,
            BLOCK=_SPLIT_BLOCK,
            CHUNK=_CHUNK,
            num_warps=_NUM_WARPS,
        )
    if num_rows > 0 and num_spec_steps > 0 and topp:
        grid = (bs, num_spec_steps, nch_stats)
        row_args = (
            target_logits,
            target_logits_indices,
            cu_num_draft_tokens,
            temps,
            top_p_ptr,
            top_p_scalar,
            tstat,
        )
        level1 = (ws.thr[0], ws.lm[0], ws.lc[0])
        level2 = (ws.thr[1], ws.lm[1], ws.lc[1])
        _topp_level1_kernel[grid](
            *row_args,
            *level1,
            vocab,
            rank_id,
            QTP=qtp,
            HAS_TOPP_PTR=has_ptr,
            K=_TOPP_K,
            SB=_TOPP_SB,
            CHUNK=_CHUNK,
            IN_PLACE=in_place,
            num_warps=_NUM_WARPS,
        )
        _topp_level2_kernel[grid](
            *row_args,
            *level1,
            *level2,
            vocab,
            rank_id,
            QTP=qtp,
            HAS_TOPP_PTR=has_ptr,
            K=_TOPP_K,
            SB=_TOPP_SB,
            CHUNK=_CHUNK,
            RB=_TOPP_RB,
            IN_PLACE=in_place,
            num_warps=_NUM_WARPS,
        )
        _topp_compact_kernel[grid](
            *row_args,
            *level1,
            *level2,
            ws.fin,
            ws.fin_n,
            ws.cand,
            vocab,
            rank_id,
            QTP=qtp,
            HAS_TOPP_PTR=has_ptr,
            K=_TOPP_K,
            BLOCK=_SPLIT_BLOCK,
            CHUNK=_CHUNK,
            RB=_TOPP_RB,
            NCH=triton.next_power_of_2(nch_stats),
            IN_PLACE=in_place,
            num_warps=_NUM_WARPS,
        )
    if num_rows > 0 and num_spec_steps > 0:
        _verify_row_stats_kernel[(bs, num_spec_steps)](
            target_logits,
            input_ids,
            target_logits_indices,
            cu_num_draft_tokens,
            temps,
            top_p_ptr,
            top_p_scalar,
            src,
            state.q,
            state.ids,
            state.temps,
            tstat,
            targ,
            qstat,
            ws.fin,
            ws.fin_n,
            ws.cand,
            rs,
            hasq,
            vocab,
            q_width,
            state.q_vs,
            state.q_cols,
            q_stride,
            num_spec_steps,
            rank_id,
            nch_stats,
            QTP=qtp,
            HAS_TOPP_PTR=has_ptr,
            K=_TOPP_K,
            SB=_TOPP_SB,
            RB=_TOPP_RB,
            BLOCK=_SPLIT_BLOCK,
            IN_PLACE=in_place,
            num_warps=_NUM_WARPS,
        )
    rsall = gather(rs)
    rmall = rm
    if use_block and num_rows > 0 and num_spec_steps > 1:
        _block_residual_mass_kernel[(bs, num_spec_steps - 1, nch_q)](
            target_logits,
            target_logits_indices,
            cu_num_draft_tokens,
            temps,
            src,
            state.q,
            state.temps,
            rsall,
            hasq,
            rm,
            nrow,
            vocab,
            q_width,
            state.q_vs,
            state.q_cols,
            q_stride,
            QTP=qtp,
            IN_PLACE=in_place,
            BLOCK=_SPLIT_BLOCK,
            CHUNK=_CHUNK,
            num_warps=_NUM_WARPS,
        )
        rmall = gather(rm)
    _prob_decide_kernel[(bs,)](
        output,
        dec,
        input_ids,
        target_logits_indices,
        cu_num_draft_tokens,
        temps,
        rsall,
        hasq,
        rmall,
        nrow,
        num_spec_steps,
        seed_acc,
        nch_q,
        QTP=qtp,
        USE_BLOCK=use_block,
        num_warps=1,
    )
    _prob_resample_kernel[(bs, nch_q)](
        res,
        dec,
        target_logits,
        input_ids,
        target_logits_indices,
        cu_num_draft_tokens,
        temps,
        src,
        state.q,
        state.temps,
        rsall,
        hasq,
        nrow,
        vocab,
        q_width,
        state.q_vs,
        state.q_cols,
        q_stride,
        seed_res,
        QTP=qtp,
        IN_PLACE=in_place,
        BLOCK=_SPLIT_BLOCK,
        CHUNK=_CHUNK,
        num_warps=_NUM_WARPS,
    )
    resall = gather(res)
    if bonus_ready is not None:
        # The bonus draw was issued on a side stream; it is first read here.
        torch.cuda.current_stream().wait_event(bonus_ready)
    _prob_finish_kernel[(bs,)](
        output,
        verdict,
        dec,
        resall,
        cu_num_draft_tokens,
        temps,
        rsall,
        bonus_token_ids,
        bonus_logits_indices,
        nrow,
        num_spec_steps,
        nch_q,
        QTP=qtp,
        num_warps=1,
    )
    return output, verdict


# ---------------------------------------------------------------------------
# diagnostics (ATOM_DSPARK_PROB_DEBUG=N: every N verify steps, sync + torch)
# ---------------------------------------------------------------------------

PROB_DEBUG = envs.ATOM_DSPARK_PROB_DEBUG
_DBG_TDS = (1.0, 0.85, 0.7, 0.5)
_dbg = {"calls": 0, "rows": 0, "rows0": 0}


def _nucleus_probs(logits_f32, temps_col, top_p):
    p = torch.softmax(logits_f32 / temps_col, dim=-1)
    if top_p is None or top_p >= 1.0:
        return p
    sp, _ = torch.sort(p, dim=-1, descending=True)
    cs = torch.cumsum(sp, dim=-1)
    keep_sorted = (cs - sp) < top_p
    thr = (
        torch.where(keep_sorted, sp, torch.full_like(sp, 2.0))
        .min(dim=-1, keepdim=True)
        .values
    )
    p = torch.where(p >= thr, p, torch.zeros_like(p))
    return p / p.sum(dim=-1, keepdim=True)


@torch.no_grad()
def debug_acceptance(
    target_logits,
    cu_num_draft_tokens,
    input_ids,
    target_logits_indices,
    temperatures,
    top_ps,
    src,
    state,
):
    """Expected per-position acceptance of alternative drafting rules on the
    SAME (p, q) rows: greedy draft + sample match = p(argmax z); probabilistic
    draft + ratio test = sum min(p, q_Td) for draft temperatures Td (x T),
    and q truncated to the same top-p. Rows whose stored q matches the draft.
    Needs the whole q on this rank (unsharded drafting)."""
    if state.q_tp != 1 or state.q_cols != target_logits.shape[1]:
        if _dbg["calls"] == 0:
            logger.info("[DSpark prob dbg] skipped: q is vocab-sharded")
            _dbg["calls"] = 1
        return
    if isinstance(top_ps, torch.Tensor):
        top_p = float(top_ps.flatten()[0].item())
    else:
        top_p = 1.0 if top_ps is None else float(top_ps)
    cu = cu_num_draft_tokens.cpu().tolist()
    src_c = src.cpu().tolist()
    temps_c = temperatures.float().cpu().tolist()
    W = state.q.shape[1]
    rows, srows, ks, ts = [], [], [], []
    start = 0
    for i, end in enumerate(cu):
        n = end - start
        if src_c[i] >= 0 and temps_c[i] > GREEDY_T:
            for k in range(min(n, W)):
                rows.append(start + k)
                srows.append(src_c[i])
                ks.append(k)
                ts.append(temps_c[i])
        start = end
    if not rows:
        return
    dev = target_logits.device
    rows_t = torch.tensor(rows, device=dev)
    s_t = torch.tensor(srows, device=dev)
    k_t = torch.tensor(ks, device=dev)
    T = torch.tensor(ts, device=dev).unsqueeze(1)
    x = input_ids[target_logits_indices[rows_t].long() + 1].long()
    valid = state.ids[s_t, k_t].long() == x
    if not bool(valid.any()):
        return
    rows_t, s_t, k_t, T, x = rows_t[valid], s_t[valid], k_t[valid], T[valid], x[valid]
    p = _nucleus_probs(target_logits[rows_t].float(), T, top_p)
    z = state.q[s_t, k_t]
    res = {"greedy": p.gather(1, z.argmax(dim=1, keepdim=True)).squeeze(1)}
    for td in _DBG_TDS:
        q = torch.softmax(z / (T * td), dim=-1)
        res[f"prob_T*{td}"] = torch.minimum(p, q).sum(dim=1)
    qn = _nucleus_probs(z, T, top_p)
    res["prob_q_topp"] = torch.minimum(p, qn).sum(dim=1)
    q1 = torch.softmax(z / T, dim=-1)
    px = p.gather(1, x[:, None]).squeeze(1)
    qx = q1.gather(1, x[:, None]).squeeze(1)
    res["realized_ratio"] = torch.clamp(px / qx.clamp_min(1e-30), max=1.0)
    res["p_max"] = p.max(dim=1).values
    first = k_t == 0
    for name, v in res.items():
        _dbg[name] = _dbg.get(name, 0.0) + float(v.sum())
        _dbg[name + "@0"] = _dbg.get(name + "@0", 0.0) + float(v[first].sum())
    _dbg["rows"] += int(first.numel())
    _dbg["rows0"] += int(first.sum())
    _dbg["calls"] += 1
    if _dbg["calls"] % 20 == 0:
        n, n0 = max(_dbg["rows"], 1), max(_dbg["rows0"], 1)
        msg = " ".join(f"{k}={_dbg[k] / n:.4f}/{_dbg[k + '@0'] / n0:.4f}" for k in res)
        logger.info(
            "[DSpark prob dbg] rows=%d rows@k0=%d top_p=%.3f (all/k0) %s",
            _dbg["rows"],
            _dbg["rows0"],
            top_p,
            msg,
        )
