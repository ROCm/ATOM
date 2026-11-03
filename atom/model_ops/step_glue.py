# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The decode step's post-verify glue in two kernels (TP > 1 top-k candidate sampling).

ATOM_GLUE_SHARD_TOPK: ``shard_topk_packed`` replaces, on one vocab shard's logits, the
presence penalty kernel, the bf16 / f32 conversions and ``torch.topk`` of
``compute_topk_candidates``: one program a row takes the shard's top ``k`` with the
penalty applied on load, by an exact radix select on the bf16 bits (two 8-bit histogram
passes find the k-th largest value, a third pass compacts), ties to the lower id. It
writes the ``[rows, 2k]`` f32 pack (values, then ids) the candidate all-gather sends.

ATOM_GLUE_STEP_SAMPLE: ``StepSampler.sample`` replaces, on a speculative verify, the
candidate sampler (``sample_topk_candidates`` over the verify and bonus rows), the
rank-0 broadcast of its tokens, the greedy rejection kernel, the presence marks and the
next-anchor gather: one program a request merges its rows' candidates of every shard,
takes the top ``top_k``, applies temperature, softmax and top-p (the smallest prefix
reaching ``top_p``), draws by an exponential race and verifies the drafts in order.
Its noise comes from a counter-based generator (Philox) keyed by a seed and a step
counter that live on the device (the last program advances the counter), so every TP
rank, holding the same gathered candidates, draws the same tokens: no broadcast. The
draws are not torch's; the distribution is the same.
"""

import os

import torch
import triton
import triton.language as tl

from atom.model_ops.presence_penalty import PresencePenaltyStep

_BLOCK = int(os.environ.get("ATOM_GLUE_TOPK_BLOCK", "2048"))
_METHOD = int(os.environ.get("ATOM_GLUE_TOPK_METHOD", "0"))
_WARPS = int(os.environ.get("ATOM_GLUE_TOPK_WARPS", "4"))


@triton.jit
def _shard_keys(
    row_ptr,
    mrow_ptr,
    cols,
    valid,
    vocab_start,
    p,
    anchor,
    drafts,
    HAS_PEN: tl.constexpr,
):
    """The block's penalized bf16 logits and their order-preserving 16-bit keys."""
    x = tl.load(row_ptr + cols, mask=valid, other=0.0).to(tl.float32)
    if HAS_PEN:
        tok = cols + vocab_start
        present = tl.load(mrow_ptr + cols, mask=valid, other=0) != 0
        present = present | (tok == anchor)
        hit = tl.max((tok[:, None] == drafts[None, :]).to(tl.int32), axis=1)
        present = present | (hit != 0)
        x = x - tl.where(present, p, 0.0)
    xb = x.to(tl.bfloat16)
    bits = xb.to(tl.uint16, bitcast=True).to(tl.int32)
    key = tl.where(bits >= 32768, 65535 - bits, bits + 32768)
    key = tl.where(valid, key, 0)
    return key, xb


@triton.jit
def _suffix_pick(hist, need, NB: tl.constexpr):
    """The largest bin h with sum(hist[h:]) >= need, and sum(hist[h + 1:])."""
    idx = tl.arange(0, NB)
    total = tl.sum(hist, axis=0)
    ge = total - (tl.cumsum(hist, axis=0) - hist)
    h = tl.max(tl.where(ge >= need, idx, -1), axis=0)
    above = tl.sum(tl.where(idx == h, ge - hist, 0), axis=0)
    return h, above


@triton.jit
def _pack(key, cols, valid):
    """(16-bit order key, column) as one int64 that orders by value, then lower column
    first; -1 for padding."""
    return tl.where(valid, (key.to(tl.int64) << 32) | (2147483647 - cols).to(tl.int64), -1)


@triton.jit
def _shard_topk_chunk_kernel(
    logits_ptr,
    stride_n,
    V,
    vocab_start,
    tmp_ptr,
    input_ids_ptr,
    row_seq_ptr,
    row_pos_ptr,
    seq_start_ptr,
    slot_ptr,
    penalty_ptr,
    mask_ptr,
    mask_stride,
    KP2: tl.constexpr,
    HAS_PEN: tl.constexpr,
    EXTRA_P2: tl.constexpr,
    BLOCK: tl.constexpr,
    METHOD: tl.constexpr,
):
    """Stage 1: the top KP2 packed keys of one BLOCK-wide chunk of a row."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    nchunk = tl.num_programs(1)
    p = 0.0
    anchor = -1
    drafts = tl.full([EXTRA_P2], -1, tl.int32)
    mrow_ptr = mask_ptr
    if HAS_PEN:
        sq = tl.load(row_seq_ptr + row)
        p = tl.load(penalty_ptr + sq)
        slot = tl.load(slot_ptr + sq).to(tl.int64)
        pos = tl.load(row_pos_ptr + row)
        start = tl.load(seq_start_ptr + sq)
        mrow_ptr = mask_ptr + slot * mask_stride + vocab_start
        # the request's anchor (its last emitted token) and the drafts this row's
        # context already holds (input ids start + 1 .. start + pos)
        anchor = tl.load(input_ids_ptr + start).to(tl.int32)
        e = tl.arange(0, EXTRA_P2)
        drafts = tl.load(input_ids_ptr + start + 1 + e, mask=e < pos, other=-1).to(tl.int32)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = cols < V
    row_ptr = logits_ptr + row.to(tl.int64) * stride_n
    key, _ = _shard_keys(row_ptr, mrow_ptr, cols, valid, vocab_start, p, anchor, drafts, HAS_PEN)
    packed = _pack(key, cols, valid)
    i = tl.arange(0, KP2)
    if METHOD == 0:
        top = tl.topk(packed, KP2)
    else:
        # the packed keys are unique: KP2 successive maxima are the top KP2
        top = tl.full([KP2], -1, tl.int64)
        for t in range(KP2):
            m = tl.max(packed, axis=0)
            top = tl.where(i == t, m, top)
            packed = tl.where(packed == m, -1, packed)
    dst = tmp_ptr + (row.to(tl.int64) * nchunk + chunk) * KP2 + i
    tl.store(dst, top)


@triton.jit
def _topk_merge_kernel(tmp_ptr, nchunk, out_ptr, vocab_start, K: tl.constexpr, KP2: tl.constexpr, C_P2: tl.constexpr):
    """Stage 2: a row's top K of its chunks' packed keys -> K f32 values, K f32 ids."""
    row = tl.program_id(0)
    c = tl.arange(0, C_P2)
    keys = tl.load(tmp_ptr + row.to(tl.int64) * nchunk * KP2 + c, mask=c < nchunk * KP2, other=-1)
    top = tl.topk(keys, KP2)
    key16 = (top >> 32).to(tl.int32)
    col = 2147483647 - (top & 4294967295).to(tl.int32)
    bits = tl.where(key16 >= 32768, key16 - 32768, 65535 - key16)
    val = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    i = tl.arange(0, KP2)
    dst = out_ptr + row.to(tl.int64) * (2 * K) + i
    tl.store(dst, val, mask=i < K)
    tl.store(dst + K, (col + vocab_start).to(tl.float32), mask=i < K)


def shard_topk_packed(
    logits: torch.Tensor,
    k: int,
    vocab_start: int,
    step: PresencePenaltyStep | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[rows, 2k]`` f32: each row's top ``k`` values of this vocab shard (bf16 logits,
    less the presence penalty when ``step`` is given) and their global token ids. The
    set of ``torch.topk(penalized, k)``; among equal values the lower ids."""
    rows, V = logits.shape
    assert logits.stride(1) == 1 and k <= V and V < 2**31
    if out is None:
        out = torch.empty(rows, 2 * k, dtype=torch.float32, device=logits.device)
    if rows == 0:
        return out
    nchunk = triton.cdiv(V, _BLOCK)
    kp2 = triton.next_power_of_2(k)
    assert kp2 <= 64 and V - (nchunk - 1) * _BLOCK >= kp2
    tmp = torch.empty(rows, nchunk, kp2, dtype=torch.int64, device=logits.device)
    if step is not None:
        assert step.num_rows == rows, "every candidate row is a penalized decode row"
        extra = triton.next_power_of_2(max(step.state.max_extra, 1))
        args = (step.input_ids, step.row_seq, step.row_pos, step.seq_start, step.slots,
                step.penalties, step.state.mask, step.state.mask.stride(0))
    else:
        extra = 1
        args = (logits, logits, logits, logits, logits, logits, logits, 0)
    _shard_topk_chunk_kernel[(rows, nchunk)](
        logits, logits.stride(0), V, vocab_start, tmp, *args,
        KP2=kp2, HAS_PEN=step is not None, EXTRA_P2=extra, BLOCK=_BLOCK, METHOD=_METHOD,
        num_warps=_WARPS,
    )
    _topk_merge_kernel[(rows,)](tmp, nchunk, out, vocab_start, K=k, KP2=kp2,
                                C_P2=triton.next_power_of_2(nchunk * kp2), num_warps=4)
    return out


@triton.jit
def _sample_row(
    gathered_ptr,
    nrows,
    lrow,
    flat,
    temp,
    top_p,
    seed,
    eps,
    TP: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    C_P2: tl.constexpr,
    S_P2: tl.constexpr,
):
    """One logits row's token: the top TOPK of its TP x K candidates, temperature,
    softmax, top-p, an exponential race (greedy rows: the largest logit)."""
    c = tl.arange(0, C_P2)
    r = c // K
    kk = c - r * K
    valid = c < TP * K
    src = gathered_ptr + (r * nrows + lrow).to(tl.int64) * (2 * K) + kk
    v = tl.load(src, mask=valid, other=float("-inf"))
    ids = tl.load(src + K, mask=valid, other=0.0).to(tl.int32)
    # the top TOPK, descending; among equal values the earlier candidate (lower id)
    s = tl.arange(0, S_P2)
    tv = tl.full([S_P2], float("-inf"), tl.float32)
    tid = tl.zeros([S_P2], dtype=tl.int32)
    for i in range(TOPK):
        m = tl.max(v, axis=0)
        j = tl.argmax(v, axis=0, tie_break_left=True)
        tv = tl.where(s == i, m, tv)
        tid = tl.where(s == i, tl.sum(tl.where(c == j, ids, 0), axis=0), tid)
        v = tl.where(c == j, float("-inf"), v)
    first = tl.sum(tl.where(s == 0, tid, 0), axis=0)
    z = tv / tl.maximum(temp, eps)
    z = z - tl.max(z, axis=0)
    e = tl.where(s < TOPK, tl.exp(z), 0.0)
    probs = e / tl.sum(e, axis=0)
    excl = tl.cumsum(probs, axis=0) - probs
    probs = tl.where(excl >= top_p, 0.0, probs)
    u = tl.rand(seed, flat * S_P2 + s)
    race = probs / (-tl.log(1.0 - u))
    w = tl.argmax(race, axis=0, tie_break_left=True)
    drawn = tl.sum(tl.where(s == w, tid, 0), axis=0)
    return tl.where(temp <= eps, first, drawn)


@triton.jit
def _step_sample_kernel(
    gathered_ptr,
    nrows,
    target_idx_ptr,
    bonus_idx_ptr,
    cu_draft_ptr,
    input_ids_ptr,
    temp_ptr,
    top_p_ptr,
    top_p_scalar,
    out_ptr,
    verdict_ptr,
    next_ptr,
    seq_start_ptr,
    slot_ptr,
    mask_ptr,
    mask_stride,
    vocab,
    rng_ptr,
    done_ptr,
    num_spec_steps,
    eps,
    TP: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    C_P2: tl.constexpr,
    S_P2: tl.constexpr,
    TOP_P_TENSOR: tl.constexpr,
    HAS_PEN: tl.constexpr,
):
    req = tl.program_id(0)
    bs = tl.num_programs(0)
    start = tl.load(cu_draft_ptr + req - 1, mask=req > 0, other=0)
    nd = tl.load(cu_draft_ptr + req) - start
    total_targets = tl.load(cu_draft_ptr + bs - 1)
    seed0 = tl.load(rng_ptr)
    ctr = tl.load(rng_ptr + 1)
    seed = ((seed0 + ctr * 2654435761) & 0x7FFFFFFF).to(tl.int32)
    temp = tl.load(temp_ptr + req)
    if TOP_P_TENSOR:
        top_p = tl.load(top_p_ptr + req)
    else:
        top_p = top_p_scalar
    out_row = out_ptr + req.to(tl.int64) * (num_spec_steps + 1)
    if HAS_PEN:
        slot = tl.load(slot_ptr + req).to(tl.int64)
        mrow = mask_ptr + slot * mask_stride
        anchor = tl.load(input_ids_ptr + tl.load(seq_start_ptr + req))
        tl.store(mrow + anchor, 1, mask=(anchor >= 0) & (anchor < vocab))
    # verify the drafts in order (rejection: a draft stands while it equals the
    # sampled target token); the first mismatch is emitted as the correction
    rejected = tl.full([], 0, tl.int32)
    num_accepted = tl.full([], 0, tl.int32)
    corr = tl.full([], -1, tl.int32)
    for j in range(nd):
        tok = tl.full([], -1, tl.int32)
        if rejected == 0:
            lrow = tl.load(target_idx_ptr + start + j)
            tok = _sample_row(gathered_ptr, nrows, lrow, start + j, temp, top_p, seed, eps,
                              TP, K, TOPK, C_P2, S_P2)
            draft = tl.load(input_ids_ptr + lrow + 1).to(tl.int32)
            if draft != tok:
                rejected = tl.full([], 1, tl.int32)
                corr = tok
            else:
                num_accepted += 1
            if HAS_PEN:
                tl.store(mrow + tok, 1, mask=(tok >= 0) & (tok < vocab))
        tl.store(out_row + j, tok)
    bonus = tl.full([], -1, tl.int32)
    if rejected == 0:
        brow = tl.load(bonus_idx_ptr + req)
        bonus = _sample_row(gathered_ptr, nrows, brow, total_targets + req, temp, top_p, seed, eps,
                            TP, K, TOPK, C_P2, S_P2)
        if HAS_PEN:
            tl.store(mrow + bonus, 1, mask=(bonus >= 0) & (bonus < vocab))
    tl.store(out_row + nd, bonus)
    for t in range(nd + 1, num_spec_steps + 1):
        tl.store(out_row + t, -1)
    # verdict as rejection_greedy_sample_kernel's: the drafts accepted (all nd when
    # none was rejected), the rest of num_spec_steps, the last emitted token's row
    na = tl.where(rejected == 0, nd, num_accepted)
    tl.store(verdict_ptr + req, na)
    tl.store(verdict_ptr + bs + req, num_spec_steps - na)
    tl.store(verdict_ptr + 2 * bs + req, tl.load(bonus_idx_ptr + req) - nd + na)
    tl.store(next_ptr + req, tl.where(rejected == 0, bonus, corr))
    # the last program advances the step counter
    old = tl.atomic_add(done_ptr, 1)
    if old == bs - 1:
        tl.store(done_ptr, 0)
        tl.store(rng_ptr + 1, ctr + 1)


class StepSampler:
    """The step kernel's state: the device-side seed and step counter."""

    def __init__(self, device, seed: int | None = None):
        if seed is None:
            seed = int(os.environ.get("ATOM_GLUE_SEED", "0")) or 0x5EED
        self.rng = torch.tensor([seed, 0], dtype=torch.int64, device=device)
        self.done = torch.zeros(1, dtype=torch.int32, device=device)

    def sample(
        self,
        gathered: torch.Tensor,
        nrows: int,
        tp: int,
        k: int,
        top_k: int,
        metadata,
        input_ids: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps,
        step: PresencePenaltyStep | None,
        eps: float,
    ):
        """(output token ids ``[bs, S + 1]`` int32, verdict ``[3, bs]`` int32, next
        anchor ids ``[bs]`` int32), as the candidate sampler + rejection + marks."""
        cu = metadata.cu_num_draft_tokens
        bs = cu.numel()
        S = metadata.num_spec_steps
        dev = gathered.device
        out = torch.empty(bs, S + 1, dtype=torch.int32, device=dev)
        verdict = torch.empty(3, bs, dtype=torch.int32, device=dev)
        nxt = torch.empty(bs, dtype=torch.int32, device=dev)
        if bs == 0:
            return out, verdict, nxt
        tp_tensor = isinstance(top_ps, torch.Tensor) and top_ps.numel() > 1
        top_p_scalar = 2.0 if top_ps is None else (float(top_ps) if not tp_tensor else 0.0)
        if tp_tensor:
            top_p_arg = top_ps.float().contiguous()
        else:
            top_p_arg = temperatures
        if step is not None:
            pen = (step.seq_start, step.slots, step.state.mask, step.state.mask.stride(0),
                   step.state.mask.shape[1])
            step._anchors_marked = True
        else:
            pen = (cu, cu, out, 0, 0)
        _step_sample_kernel[(bs,)](
            gathered, nrows, metadata.target_logits_indices, metadata.bonus_logits_indices, cu,
            input_ids, temperatures, top_p_arg, top_p_scalar, out, verdict, nxt, *pen,
            self.rng, self.done, S, eps,
            TP=tp, K=k, TOPK=top_k, C_P2=triton.next_power_of_2(tp * k),
            S_P2=triton.next_power_of_2(max(top_k, 2)), TOP_P_TENSOR=tp_tensor,
            HAS_PEN=step is not None, num_warps=1,
        )
        return out, verdict, nxt
