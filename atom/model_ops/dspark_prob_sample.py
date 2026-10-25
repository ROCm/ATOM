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
    sampler draws from.
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
"""

import logging
import os
from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from atom.model_ops.dspark_markov_sample import (
    _BLOCK_K,
    _BLOCK_V,
    _MAX_BLOCK_ROW,
    _MIN_BLOCK_ROW,
    _dspark_markov_argmax_stage2,
    argmax_gather,
)
from atom.model_ops.rejection_sampler import _draft_token, _finish_request
from atom.model_ops.sampler import SAMPLER_EPS

logger = logging.getLogger("atom")

PROB_DRAFT = os.environ.get("ATOM_DSPARK_PROB_DRAFT", "0") == "1"
BLOCK_VERIFY = os.environ.get("ATOM_DSPARK_BLOCK_VERIFY", "0") == "1"
DRAFT_TSCALE = float(os.environ.get("ATOM_DSPARK_PROB_DRAFT_TSCALE", "1.0"))
# 0 => let the drafter use the sharded LM head when the greedy path would
SHARDED_DRAFT = os.environ.get("ATOM_DSPARK_PROB_SHARDED", "1") == "1"

# The runner clamps temperature 0 to SAMPLER_EPS (1e-10) and the sampler treats
# exactly the rows at or below it as greedy (``_apply_greedy_tokens``); use the
# same rule. fp32-rounded, since every temperature compared against it is an
# fp32 value (device buffers, and host floats read back from them).
GREEDY_T = float(torch.tensor(SAMPLER_EPS, dtype=torch.float32).item())
_GREEDY_T = tl.constexpr(GREEDY_T)
# Smallest positive value of Triton's fp32 tl.rand.
_RAND_MIN = tl.constexpr(4.6566127342e-10)
# Top-p search: coarse bisection passes over the row, then the tokens left in
# the bracket are compacted (<= _CAND_CAP) and the search finishes on them.
_COARSE_ITERS = int(os.environ.get("ATOM_DSPARK_PROB_COARSE", "12"))
_FINE_ITERS = tl.constexpr(18)
_CAND_CAP = int(os.environ.get("ATOM_DSPARK_PROB_CAP", "8192"))
_VBLOCK = int(os.environ.get("ATOM_DSPARK_PROB_VBLOCK", "8192"))
_NUM_WARPS = int(os.environ.get("ATOM_DSPARK_PROB_WARPS", "8"))


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
    """`_dspark_markov_argmax_stage1` + store z + Gumbel-max on the scores."""
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    prev = tl.load(prev_ids_ptr + offs_row * stride_prev, mask=row_mask, other=0)

    acc = tl.zeros((BLOCK_ROW, BLOCK_V), dtype=tl.float32)
    for k in range(tl.cdiv(rank, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < rank
        embed = tl.load(
            w1_ptr + prev[:, None] * stride_w1_vocab + offs_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if tile == 0:
            tl.store(
                embed_out_ptr + offs_row[:, None] * stride_embed_row + offs_k[None, :],
                embed,
                mask=row_mask[:, None] & k_mask[None, :],
            )
        w2 = tl.load(
            w2_ptr
            + (vocab_offset + offs_v[None, :]) * stride_w2_vocab
            + offs_k[:, None],
            mask=v_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(embed, w2, acc)

    mask2 = row_mask[:, None] & v_mask[None, :]
    base = tl.load(
        base_ptr + offs_row[:, None] * stride_base_row + offs_v[None, :],
        mask=mask2,
        other=0.0,
    )
    z = base.to(tl.float32) + acc
    # q's logits, pre-temperature: verification rebuilds q from exactly these.
    tl.store(q_out_ptr + offs_row[:, None] * stride_q_row + offs_v[None, :], z, mask=mask2)

    temp = tl.load(temp_ptr + offs_row, mask=row_mask, other=1.0)
    greedy = temp <= _GREEDY_T
    inv_t = 1.0 / tl.where(greedy, 1.0, temp)
    seed = tl.load(seed_ptr)
    noise_offs = (pos_k * num_rows + offs_row)[:, None] * vocab_global + (
        vocab_offset + offs_v
    )[None, :]
    g = _gumbel(seed, noise_offs)
    score = tl.where(greedy[:, None], z, z * inv_t[:, None] + g)
    vals = tl.where(v_mask[None, :], score, float("-inf"))
    tile_max = tl.max(vals, axis=1)
    cand = tl.where(
        (vals == tile_max[:, None]) & v_mask[None, :], offs_v[None, :], vocab_size
    )
    tile_idx = tl.min(cand, axis=1)
    tile_idx = tl.where(tile_idx < vocab_size, tile_idx + vocab_offset, vocab_size)
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
        vocab_size,
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

    @classmethod
    def allocate(cls, max_bs: int, width: int, vocab: int, device) -> "ProbDraftState":
        st = cls(
            q=torch.zeros(max_bs, width, vocab, dtype=torch.float32, device=device),
            ids=torch.full((max_bs, width), -1, dtype=torch.int32, device=device),
            temps=torch.ones(max_bs, dtype=torch.float32, device=device),
            seed=torch.zeros(1, dtype=torch.int64, device=device),
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
def _row_max_argmax(row_ptr, V, BLOCK: tl.constexpr):
    """(max, lowest argmax) of a row; per-lane accumulation, one final reduce."""
    mxv = tl.zeros((BLOCK,), tl.float32) + float("-inf")
    idxv = tl.zeros((BLOCK,), tl.int32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(row_ptr + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        take = x > mxv  # strict: a lane keeps its first (lowest) index
        idxv = tl.where(take, offs, idxv)
        mxv = tl.where(take, x, mxv)
    best = tl.max(mxv, axis=0)
    best_i = tl.min(tl.where(mxv == best, idxv, 2147483647), axis=0)
    return best, best_i


@triton.jit
def _row_mass_ge(row_ptr, V, inv_t, m, thr, BLOCK: tl.constexpr):
    """sum over v with l_v*inv_t >= thr of exp(l_v*inv_t - m)."""
    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(row_ptr + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        y = x * inv_t
        acc += tl.where(y >= thr, tl.exp(y - m), 0.0)
    return tl.sum(acc, axis=0)


@triton.jit
def _row_online_stats(row_ptr, V, inv_t, BLOCK: tl.constexpr):
    """One pass: (max of x*inv_t, sum exp(x*inv_t - max), lowest argmax);
    per-lane online softmax, one final reduce."""
    mxv = tl.zeros((BLOCK,), tl.float32) + float("-inf")
    sv = tl.zeros((BLOCK,), tl.float32)
    idxv = tl.zeros((BLOCK,), tl.int32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(row_ptr + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
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
def _req_span(cu_ptr, req_idx):
    if req_idx == 0:
        start = 0
    else:
        start = tl.load(cu_ptr + req_idx - 1)
    end = tl.load(cu_ptr + req_idx)
    return start, end - start


@triton.jit
def _bisect_pass(row, V, inv_t, m, lo, mid, hi, BLOCK: tl.constexpr):
    """mass(>= mid), #tokens in [lo, mid), #tokens in [mid, hi)."""
    accv = tl.zeros((BLOCK,), tl.float32)
    c_lo = tl.zeros((BLOCK,), tl.int32)
    c_hi = tl.zeros((BLOCK,), tl.int32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        y = tl.load(row + offs, mask=offs < V, other=float("-inf")).to(tl.float32) * inv_t
        ge = y >= mid
        accv += tl.where(ge, tl.exp(y - m), 0.0)
        c_lo += ((y >= lo) & (y < mid)).to(tl.int32)
        c_hi += (ge & (y < hi)).to(tl.int32)
    return tl.sum(accv, axis=0), tl.sum(c_lo, axis=0), tl.sum(c_hi, axis=0)


@triton.jit
def _topp_threshold(
    row, cand_ptr, V, inv_t, m, z, top_p, BLOCK: tl.constexpr, CAP: tl.constexpr,
    MAX_COARSE: tl.constexpr, FBLOCK: tl.constexpr = 2048,
):
    """(tau, zk): the largest tau with mass(>= tau) >= top_p z, and that mass.

    Bracket [m - c, m] with c = ln(V / (1 - top_p)) + 1: tokens below it hold
    at most (1 - top_p) of the mass, so the cutoff is inside. Bisection passes
    over the row (each also counting the tokens on both sides of the midpoint)
    run until the bracket holds <= CAP tokens; those are compacted to
    ``cand_ptr`` with their exp-weights and the search finishes on them."""
    target = top_p * z
    c = tl.log(V * 1.0) - tl.log(tl.maximum(1.0 - top_p, 1e-30)) + 1.0
    lo = m - c
    hi = m
    cnt_br = tl.zeros((), tl.int32) + V
    it = tl.zeros((), tl.int32)
    while (cnt_br > CAP) & (it < MAX_COARSE):
        mid = 0.5 * (lo + hi)
        s, c_lo, c_hi = _bisect_pass(row, V, inv_t, m, lo, mid, hi, BLOCK)
        ge = s >= target
        lo = tl.where(ge, mid, lo)
        hi = tl.where(ge, hi, mid)
        cnt_br = tl.where(ge, c_hi, c_lo)
        it += 1
    # compact the tokens in [lo, hi) (y, e); mass(>= hi) alongside
    cnt = tl.zeros((), tl.int32)
    min_v = tl.zeros((BLOCK,), tl.float32)
    mhi_v = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        y = tl.load(row + offs, mask=offs < V, other=float("-inf")).to(tl.float32) * inv_t
        e = tl.exp(y - m)
        inb = (y >= lo) & (y < hi)
        csum = tl.cumsum(inb.to(tl.int32), axis=0)
        pos = cnt + csum - 1
        keep = inb & (pos < CAP)
        tl.store(cand_ptr + pos, y, mask=keep)
        tl.store(cand_ptr + CAP + pos, e, mask=keep)
        cnt += tl.max(csum, axis=0)
        min_v += tl.where(inb, e, 0.0)
        mhi_v += tl.where(y >= hi, e, 0.0)
    mass_hi = tl.sum(mhi_v, axis=0)  # mass(>= hi)
    zk = mass_hi + tl.sum(min_v, axis=0)  # mass(>= lo)
    if cnt <= CAP:
        for _ in range(_FINE_ITERS):
            mid = 0.5 * (lo + hi)
            accv = tl.zeros((FBLOCK,), tl.float32)
            for cs in range(0, cnt, FBLOCK):
                co = cs + tl.arange(0, FBLOCK)
                cm = co < cnt
                yc = tl.load(cand_ptr + co, mask=cm, other=float("-inf"))
                ec = tl.load(cand_ptr + CAP + co, mask=cm, other=0.0)
                accv += tl.where(yc >= mid, ec, 0.0)
            s = mass_hi + tl.sum(accv, axis=0)
            ge = s >= target
            lo = tl.where(ge, mid, lo)
            zk = tl.where(ge, s, zk)
            hi = tl.where(ge, hi, mid)
    else:
        for _ in range(_FINE_ITERS):
            mid = 0.5 * (lo + hi)
            s = _row_mass_ge(row, V, inv_t, m, mid, BLOCK)
            ge = s >= target
            lo = tl.where(ge, mid, lo)
            zk = tl.where(ge, s, zk)
            hi = tl.where(ge, hi, mid)
    return lo, zk


@triton.jit(do_not_specialize=["num_spec_steps", "top_p_scalar", "q_width", "q_vs", "q_cols", "q_stride", "rank_id"])
def _verify_row_stats_kernel(
    target_ptr,  # [N, V]
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
    cand_ptr,  # [N, 2 * CAP] scratch
    rs_ptr,  # [N, 8] out: amax, m, tau, zk, p(x) | local q max, sumexp, x-logit
    hasq_ptr,  # [N] out
    V,
    q_width,
    q_vs,
    q_cols,
    q_stride,
    num_spec_steps,
    rank_id,
    QTP: tl.constexpr,
    HAS_TOPP_PTR: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    COARSE: tl.constexpr,
):
    """Per verified row. The target half (fields 0-4) is computed only by the
    rank that owns the row (r % QTP), the q half (5-7) by every rank over its
    own vocab columns; one all-gather then hands every rank both."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
    if k >= n:
        return
    r = start + k
    out = rs_ptr + r * 8
    row = target_ptr + r.to(tl.int64) * V
    x = tl.load(input_ids_ptr + tl.load(target_logits_indices_ptr + r) + 1).to(tl.int32)
    x_ok = (x >= 0) & (x < V)
    xs = tl.where(x_ok, x, 0)
    own_row = (r % QTP) == rank_id

    temp = tl.load(temps_ptr + req_idx).to(tl.float32)
    greedy = temp <= _GREEDY_T
    hasq = 0
    if greedy:
        if own_row:
            lmax, amax = _row_max_argmax(row, V, BLOCK)
            tl.store(out, amax.to(tl.float32))
    else:
        if own_row:
            inv_t = 1.0 / temp
            m, z, amax = _row_online_stats(row, V, inv_t, BLOCK)
            if HAS_TOPP_PTR:
                top_p = tl.load(top_p_ptr + req_idx).to(tl.float32)
            else:
                top_p = top_p_scalar
            tau = tl.zeros((), tl.float32) + float("-inf")
            zk = z
            if top_p < 1.0:
                tau, zk = _topp_threshold(
                    row, cand_ptr + r.to(tl.int64) * (2 * CAP), V, inv_t, m, z, top_p,
                    BLOCK, CAP, COARSE,
                )
            lx = tl.load(row + xs).to(tl.float32) * inv_t
            px = tl.where(x_ok & (lx >= tau), tl.exp(lx - m) / zk, 0.0)
            tl.store(out, amax.to(tl.float32))
            tl.store(out + 1, m)
            tl.store(out + 2, tau)
            tl.store(out + 3, zk)
            tl.store(out + 4, px)

        s_row = tl.load(src_ptr + req_idx)
        if s_row >= 0:
            if k < q_width:
                qid = tl.load(q_ids_ptr + s_row * q_width + k)
                tq = tl.load(q_temps_ptr + s_row).to(tl.float32)
                if (qid == x) & x_ok & (tq > _GREEDY_T):
                    inv_tq = 1.0 / tq
                    qrow = q_ptr + (s_row.to(tl.int64) * q_width + k) * q_stride
                    mq, zq, _ = _row_online_stats(qrow, q_cols, inv_tq, BLOCK)
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
    cu_num_draft_tokens_ptr,
    temps_ptr,
    src_ptr,
    q_ptr,
    q_temps_ptr,
    rs_ptr,  # [QTP, N, 8]
    hasq_ptr,
    rm_ptr,  # [N] out (this shard's part)
    N,
    V,
    q_width,
    q_vs,
    q_cols,
    q_stride,
    QTP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """rm[row k] = sum_v max(P_k p_k(v) - q_k(v), 0) for 1 <= k < n, over
    this rank's vocab columns."""
    req_idx = tl.program_id(0)
    k = tl.program_id(1) + 1
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
        row = target_ptr + r.to(tl.int64) * V + q_vs
        qrow = q_ptr + (s_row.to(tl.int64) * q_width + k) * q_stride
        if P > 0.0:
            accv = tl.zeros((BLOCK,), tl.float32)
            for st in range(0, q_cols, BLOCK):
                offs = st + tl.arange(0, BLOCK)
                mask = (offs < q_cols) & (q_vs + offs < V)
                y = tl.load(row + offs, mask=mask, other=float("-inf")).to(tl.float32) * inv_t
                p = tl.where(mask & (y >= tau), tl.exp(y - m) / zk, 0.0)
                d = tl.load(qrow + offs, mask=mask, other=float("-inf")) * inv_tq
                q = tl.where(mask, tl.exp(d - mq) / zq, 0.0)
                accv += tl.where(mask, tl.maximum(P * p - q, 0.0), 0.0)
            acc = tl.sum(accv, axis=0)
    tl.store(rm_ptr + r, acc)


@triton.jit(do_not_specialize=["num_spec_steps", "seed_acc"])
def _prob_decide_kernel(
    output_token_ids_ptr,  # [bs, num_spec_steps + 1]
    dec_ptr,  # [3, bs] f32: num_acc, rejected, log coef
    input_ids_ptr,
    target_logits_indices_ptr,
    cu_num_draft_tokens_ptr,
    temps_ptr,
    rs_ptr,  # [QTP, N, 8]
    hasq_ptr,
    rmall_ptr,  # [QTP, N]
    N,
    num_spec_steps,
    seed_acc,
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
                        rm += tl.load(rmall_ptr + j * N + start + k + 1)
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
    res_ptr,  # [bs, 4] f32: best residual score, id, best p score, id
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
):
    """This rank's vocab columns of the residual draw at the rejected row:
    norm(max(c p - q, 0)) (one-hot q: p without the draft), with p's own draw
    as the fallback when the residual is all zero."""
    req_idx = tl.program_id(0)
    bs = tl.num_programs(0)
    start, n = _req_span(cu_num_draft_tokens_ptr, req_idx)
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
        inv_tq = 1.0 / tl.maximum(tl.load(q_temps_ptr + s_row).to(tl.float32), _GREEDY_T)
        mq, zq, _ = _q_comb(rs_ptr, r, N, QTP)
        row = target_ptr + r.to(tl.int64) * V + q_vs
        qrow = q_ptr + (s_row.to(tl.int64) * q_width + num_acc) * q_stride
        for st in range(0, q_cols, BLOCK):
            offs = st + tl.arange(0, BLOCK)
            gid = q_vs + offs
            mask = (offs < q_cols) & (gid < V)
            y = tl.load(row + offs, mask=mask, other=float("-inf")).to(tl.float32) * inv_t
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
    tl.store(res_ptr + req_idx * 4, best_r)
    tl.store(res_ptr + req_idx * 4 + 1, best_ri.to(tl.float32))
    tl.store(res_ptr + req_idx * 4 + 2, best_p)
    tl.store(res_ptr + req_idx * 4 + 3, best_pi.to(tl.float32))


@triton.jit(do_not_specialize=["num_spec_steps"])
def _prob_finish_kernel(
    output_token_ids_ptr,
    verdict_ptr,
    dec_ptr,
    resall_ptr,  # [QTP, bs, 4]
    cu_num_draft_tokens_ptr,
    temps_ptr,
    rs_ptr,  # [QTP, N, 8]
    bonus_token_ids_ptr,
    bonus_logits_indices_ptr,
    N,
    num_spec_steps,
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
            best_r = tl.load(resall_ptr + req_idx * 4)
            id_r = tl.load(resall_ptr + req_idx * 4 + 1)
            best_p = tl.load(resall_ptr + req_idx * 4 + 2)
            id_p = tl.load(resall_ptr + req_idx * 4 + 3)
            for j in tl.static_range(1, QTP):
                base = resall_ptr + (j * bs + req_idx) * 4
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


def prob_rejection_sample(
    input_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    num_spec_steps: int,
    cu_num_draft_tokens: torch.Tensor,  # [bs] inclusive
    target_logits: torch.Tensor,  # [N, V]
    bonus_token_ids: torch.Tensor,  # [bs]
    bonus_logits_indices: torch.Tensor,  # [bs]
    temperatures: torch.Tensor,  # [bs] (clamped to eps)
    top_ps,  # None | float | Tensor[bs]
    src: torch.Tensor,  # [bs] int32
    state: ProbDraftState,
    use_block: bool = False,
    tp_group=None,  # required when state.q_tp > 1
    use_custom: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    f32 = dict(dtype=torch.float32, device=device)
    nrow = max(num_rows, 1)
    rs = torch.empty((nrow, 8), **f32)
    rm = torch.empty(nrow, **f32)
    hasq = torch.empty(nrow, dtype=torch.int32, device=device)
    cand = torch.empty((nrow, 2 * _CAND_CAP), **f32)
    dec = torch.empty((3, bs), **f32)
    res = torch.empty((bs, 4), **f32)
    if isinstance(top_ps, torch.Tensor) and top_ps.numel() > 1:
        top_p_ptr, top_p_scalar, has_ptr = top_ps.to(torch.float32), 1.0, True
    else:
        if isinstance(top_ps, torch.Tensor):
            top_ps = float(top_ps.item())
        top_p_ptr, has_ptr = temperatures, False
        top_p_scalar = 1.0 if top_ps is None else float(top_ps)
    temps = temperatures.to(torch.float32)
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
            cand,
            rs,
            hasq,
            vocab,
            q_width,
            state.q_vs,
            state.q_cols,
            q_stride,
            num_spec_steps,
            rank_id,
            QTP=qtp,
            HAS_TOPP_PTR=has_ptr,
            BLOCK=_VBLOCK,
            CAP=_CAND_CAP,
            COARSE=_COARSE_ITERS,
            num_warps=_NUM_WARPS,
        )
    rsall = gather(rs)
    rmall = rm
    if use_block and num_rows > 0 and num_spec_steps > 1:
        _block_residual_mass_kernel[(bs, num_spec_steps - 1)](
            target_logits,
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
            BLOCK=_VBLOCK,
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
        QTP=qtp,
        USE_BLOCK=use_block,
        num_warps=1,
    )
    _prob_resample_kernel[(bs,)](
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
        BLOCK=_VBLOCK,
        num_warps=_NUM_WARPS,
    )
    resall = gather(res)
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
        QTP=qtp,
        num_warps=1,
    )
    return output, verdict


# ---------------------------------------------------------------------------
# diagnostics (ATOM_DSPARK_PROB_DEBUG=N: every N verify steps, sync + torch)
# ---------------------------------------------------------------------------

PROB_DEBUG = int(os.environ.get("ATOM_DSPARK_PROB_DEBUG", "0") or 0)
_DBG_TDS = (1.0, 0.85, 0.7, 0.5)
_dbg = {"calls": 0, "rows": 0, "rows0": 0}


def _nucleus_probs(logits_f32, temps_col, top_p):
    p = torch.softmax(logits_f32 / temps_col, dim=-1)
    if top_p is None or top_p >= 1.0:
        return p
    sp, idx = torch.sort(p, dim=-1, descending=True)
    cs = torch.cumsum(sp, dim=-1)
    keep_sorted = (cs - sp) < top_p
    thr = torch.where(keep_sorted, sp, torch.full_like(sp, 2.0)).min(dim=-1, keepdim=True).values
    p = torch.where(p >= thr, p, torch.zeros_like(p))
    return p / p.sum(dim=-1, keepdim=True)


@torch.no_grad()
def debug_acceptance(
    target_logits, cu_num_draft_tokens, input_ids, target_logits_indices,
    temperatures, top_ps, src, state,
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
        msg = " ".join(
            f"{k}={_dbg[k] / n:.4f}/{_dbg[k + '@0'] / n0:.4f}" for k in res
        )
        logger.info(
            "[DSpark prob dbg] rows=%d rows@k0=%d top_p=%.3f (all/k0) %s",
            _dbg["rows"], _dbg["rows0"], top_p, msg,
        )
