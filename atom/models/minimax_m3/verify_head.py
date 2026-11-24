# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1's DSpark verify head inside the decode graph (``compute_head``).

ATOM's verify of a decode step runs after the graph, eagerly: the LM head
all-gathers ``[rows, V]`` bf16 logits to every rank, the sampler materializes a
``[rows, V]`` fp32 exponential for the verify rows and another for the bonus
rows, makes more passes over the logits (the draw, the greedy fallback, the
argmax the forced-acceptance kernel reads), and rank 0's ids are broadcast.

Here every rank keeps its own vocab shard of the logits and reduces each row to
two ``(value, id)`` pairs in one pass -- the raw argmax and the Gumbel-max draw
``argmax(l / T - log(-log u))`` with ``u`` Philox's uniform keyed on (step, rank,
row, column) -- and only those ``[rows, 4]`` fp32 cross ranks.
Every rank then holds the same pairs, so the verdict kernel picks the same
target ids everywhere: the raw argmax for a greedy row (bit-identical to ATOM's
``topk_select(..., tie="low")`` over the same bf16 logits), the draw for a
sampled one (the same law as ATOM's sample-and-match: an independent draw per
row from ``softmax(l / T)``). The verdict is ATOM's exact-match / forced
acceptance arithmetic (``rejection_sampler``), laid out as the rejection
sampler lays it out: output ids ``[B, q]`` and verdict ``[3, B]``, plus a fourth
verdict row, each request's next id (its output at the anchor row).

The runner takes these outputs in place of the logits exactly where they stand
for the step (``ModelRunner._head_ids_serve``: every request verifying the full
span, no logprobs / constraints, rows all greedy or all sampled at top-k -1 and
top-p 1); any other step falls back to the logits path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.verify_head import VerifyHead
from atom.model_ops.verify_ids import VerifyOutput
from atom.model_ops.tile_argmax import argmax_lowest_id, argmax_of_tiles
from atom.models.minimax_m3.packed_argmax import (
    EMPTY,
    pack_max_id,
    ranks_best,
    unpack_id,
)
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

# Philox key high words: the row draws and the forced-acceptance uniforms never
# share a stream.
_DRAW_KEY = 3 << 32
_ACCEPT_KEY = 4 << 32
_SENTINEL = 2147483647
_BLOCK = 1024


def _chunk_for(rows: int) -> int:
    """V columns one partials program reduces: enough programs to fill the
    machine at a few rows, few enough partials at many."""
    if rows <= 16:
        return 2048
    if rows <= 32:
        return 4096
    if rows <= 128:
        return 8192
    return 16384


def _atomic_chunk_for(rows: int) -> int:
    """`_chunk_for` when the tiles fold by atomic_max (no partials to pack): the
    measured fastest per row count (MI355X, 50016 columns). A row's draws do not
    depend on the chunk (each column's Philox counter is its own)."""
    if rows <= 16 or 32 < rows <= 64:
        return 2048
    return 4096


@triton.jit
def _shard_partials_kernel(
    logits_ptr,  # [N, V_local] bf16, this rank's vocab shard
    stride_row,
    temp_ptr,  # [>= N / RQ] fp32, a request's temperature (clamped to eps)
    step_ptr,  # [1] int64, the head's step counter
    ps_val_ptr,  # [C, N] fp32: the tile's best draw key
    ps_idx_ptr,  # [C, N] int32: its lowest global id
    pl_val_ptr,  # [C, N] fp32: the tile's best raw logit
    pl_idx_ptr,  # [C, N] int32
    packed_ptr,  # ATOMIC: [N, 2] int64 packed (draw, raw) bests, folded by atomic_max
    num_rows,
    v_local,
    vocab_start,
    vocab_total,
    DRAW_KEY: tl.constexpr,
    SENTINEL: tl.constexpr,
    RQ: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    """One (row, tile): the tile's best Gumbel key ``l / T - ln(-ln u)`` and
    best raw logit, each with its lowest global id.

    Keys are kept in log2 units (``l * log2(e) / T - log2(-log2 u)`` is the
    natural-log key over ln 2 plus a constant), so both logs are the native
    log2. One Philox call yields four uniforms (``rand4x``): the four quarters
    of a BLOCK share a counter and take one lane each. The counter is this
    rank's (row, column quarter); the rank is folded into the key, so every
    (rank, row, id) draws its own uniform. Lanes keep running maxima across
    blocks (strict ``>``: a later block holds higher ids) and reduce once."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    seed = tl.load(step_ptr) + DRAW_KEY + (vocab_start.to(tl.int64) << 40)
    scale = 1.4426950408889634 / tl.load(temp_ptr + row // RQ)
    QB: tl.constexpr = BLOCK // 4
    ctr_base = row.to(tl.int64) * ((v_local + 3) // 4 + QB)
    row_ptr = logits_ptr + row.to(tl.int64) * stride_row
    q = tl.arange(0, QB)
    best_s = tl.full([4, QB], float("-inf"), tl.float32)
    best_si = tl.full([4, QB], SENTINEL, tl.int32)
    best_l = tl.full([4, QB], float("-inf"), tl.float32)
    best_li = tl.full([4, QB], SENTINEL, tl.int32)
    quarter = tl.arange(0, 4)[:, None]
    start = tile * CHUNK
    for off in range(0, CHUNK, BLOCK):
        base = start + off
        cols = base + quarter * QB + q[None, :]
        mask = cols < v_local
        logit = tl.load(row_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        ids = vocab_start + cols
        r0, r1, r2, r3 = tl.rand4x(seed, ctr_base + base // 4 + q)
        u = tl.where(
            quarter == 0,
            r0[None, :],
            tl.where(quarter == 1, r1[None, :], tl.where(quarter == 2, r2[None, :], r3[None, :])),
        )
        key = tl.where(mask, logit * scale - tl.log2(-tl.log2(u)), float("-inf"))
        take_s = key > best_s
        best_si = tl.where(take_s, ids, best_si)
        best_s = tl.where(take_s, key, best_s)
        take_l = logit > best_l
        best_li = tl.where(take_l, ids, best_li)
        best_l = tl.where(take_l, logit, best_l)
    best_s = tl.reshape(best_s, [BLOCK])
    best_si = tl.reshape(best_si, [BLOCK])
    best_l = tl.reshape(best_l, [BLOCK])
    best_li = tl.reshape(best_li, [BLOCK])
    s, si = argmax_lowest_id(best_s, best_si, best_si != SENTINEL, SENTINEL)
    lv, li = argmax_lowest_id(best_l, best_li, best_li != SENTINEL, SENTINEL)
    if ATOMIC:
        tl.atomic_max(packed_ptr + row * 2, pack_max_id(s, si), mask=si != SENTINEL, sem="relaxed")
        tl.atomic_max(
            packed_ptr + row * 2 + 1, pack_max_id(lv, li), mask=li != SENTINEL, sem="relaxed"
        )
    else:
        out = tile * num_rows + row
        tl.store(ps_val_ptr + out, s)
        tl.store(ps_idx_ptr + out, si)
        tl.store(pl_val_ptr + out, lv)
        tl.store(pl_idx_ptr + out, li)


@triton.jit
def _shard_pack_kernel(
    ps_val_ptr,
    ps_idx_ptr,
    pl_val_ptr,
    pl_idx_ptr,
    packed_ptr,  # [N, 4] fp32: draw key, draw id, logit, argmax id
    num_rows,
    num_tiles,
    BLOCK_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    s, si = argmax_of_tiles(ps_val_ptr, ps_idx_ptr, row, num_rows, num_tiles, BLOCK_TILE)
    lv, li = argmax_of_tiles(pl_val_ptr, pl_idx_ptr, row, num_rows, num_tiles, BLOCK_TILE)
    # ids < 2**24 are exact in fp32 (M3's vocab is 200064)
    tl.store(packed_ptr + row * 4, s)
    tl.store(packed_ptr + row * 4 + 1, si.to(tl.float32))
    tl.store(packed_ptr + row * 4 + 2, lv)
    tl.store(packed_ptr + row * 4 + 3, li.to(tl.float32))


@triton.jit
def _verdict_kernel(
    gathered_ptr,  # [W, N, 4] fp32, every rank's packed pairs
    input_ids_ptr,  # the step's input ids: draft j of request b is row b * RQ + j + 1
    temp_ptr,
    step_ptr,
    cond_rates_ptr,  # [RQ - 1] fp32, forced acceptance only
    out_ptr,  # [B, RQ] int32
    verdict_ptr,  # [4, B] int32: accepted drafts, rejected, anchor row, next id
    local_ptr,  # ATOMIC: [N, 2] int64, this rank's packed bests, reset here
    num_rows,
    ACCEPT_KEY: tl.constexpr,
    W: tl.constexpr,
    RQ: tl.constexpr,
    SP: tl.constexpr,
    SYNTHETIC: tl.constexpr,
    EPS: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    b = tl.program_id(0)
    nreq = tl.num_programs(0)
    p = tl.arange(0, SP)
    live = p < RQ
    rows = b * RQ + p
    if ATOMIC:
        # gathered: [W, N, 2] int64 packed (draw, raw); the larger word is the
        # larger value, then the lower id -- the lower rank among equal values
        si = unpack_id(ranks_best(gathered_ptr, rows, live, num_rows, W, 2))
        li = unpack_id(ranks_best(gathered_ptr + 1, rows, live, num_rows, W, 2))
        empty = tl.full([SP], -9223372036854775808, tl.int64)  # packed_argmax.EMPTY
        tl.store(local_ptr + rows * 2, empty, mask=live)
        tl.store(local_ptr + rows * 2 + 1, empty, mask=live)
    else:
        s = tl.load(gathered_ptr + rows * 4, mask=live, other=float("-inf"))
        si = tl.load(gathered_ptr + rows * 4 + 1, mask=live, other=0.0)
        lv = tl.load(gathered_ptr + rows * 4 + 2, mask=live, other=float("-inf"))
        li = tl.load(gathered_ptr + rows * 4 + 3, mask=live, other=0.0)
        for j in tl.static_range(1, W):
            base = gathered_ptr + (j * num_rows + rows) * 4
            s_j = tl.load(base, mask=live, other=float("-inf"))
            si_j = tl.load(base + 1, mask=live, other=0.0)
            lv_j = tl.load(base + 2, mask=live, other=float("-inf"))
            li_j = tl.load(base + 3, mask=live, other=0.0)
            # strict `>`: an equal value keeps the lower rank, the lower ids
            take_s = s_j > s
            si = tl.where(take_s, si_j, si)
            s = tl.where(take_s, s_j, s)
            take_l = lv_j > lv
            li = tl.where(take_l, li_j, li)
            lv = tl.where(take_l, lv_j, lv)
    greedy = tl.load(temp_ptr + b) <= EPS
    target = tl.where(greedy, li, si).to(tl.int32)
    n = RQ - 1
    verify = p < n
    draft = tl.load(input_ids_ptr + rows + 1, mask=verify, other=0).to(tl.int32)
    if SYNTHETIC:
        seed = tl.load(step_ptr) + ACCEPT_KEY
        u = tl.rand(seed, rows)
        ok = u < tl.load(cond_rates_ptr + p, mask=verify, other=0.0)
        kept = draft
    else:
        ok = draft == target
        kept = target
    k = tl.min(tl.where(verify & (ok == 0), p, n), axis=0)
    rejected = k < n
    row_out = tl.where(
        verify,
        tl.where(p < k, kept, tl.where(p == k, target, -1)),
        tl.where((p == n) & (rejected == 0), target, -1),
    )
    tl.store(out_ptr + b * RQ + p, row_out, mask=live)
    accepted = tl.where(rejected, k, n)
    tl.store(verdict_ptr + b, accepted)
    tl.store(verdict_ptr + nreq + b, n - accepted)
    tl.store(verdict_ptr + 2 * nreq + b, b * RQ + accepted)
    tl.store(verdict_ptr + 3 * nreq + b, tl.sum(tl.where(p == accepted, row_out, 0), axis=0))


class M3VerifyHead:
    """The in-graph verify head (module docstring). Built at install with the
    runner; every buffer a launch touches is allocated once at the largest
    batch and only sliced, so a captured graph replays against fixed
    addresses."""

    def __init__(self, causal_lm, runner, drafter) -> None:
        self.lm = causal_lm
        self.runner = runner
        self.rq = int(drafter.mtp_k) + 1
        self.max_bs = int(runner.config.max_num_seqs)
        head = causal_lm.lm_head
        self.vocab_start = int(head.vocab_start_idx)
        self.v_local = int(head.weight.shape[0])
        self.vocab_total = int(head.num_embeddings)
        self.world = int(head.tp_size)
        device = head.weight.device
        self.device = device
        from atom.models.minimax_m3 import dspark as _dsp

        if _dsp._DG_LMHEAD_AK and getattr(head, "bias", None) is None:
            # p1/dense-gemm: a shuffled copy of the shard for amd-kernels' lm_head at 8..64 rows (+615 MB per
            # rank; the row-major weight stays for tgemm and every other reader), tiles compiled before capture
            from amd_kernels.gemm.bf16_n50016_k6144 import host as _akh

            if _akh.supports(head.weight) and getattr(head, "_dg_ak_wshuf", None) is None:
                head._dg_ak_wshuf = _akh.shuffle_weight(head.weight.data)
                _akh.warmup_bf16_gemm_n50016_k6144(device, True)
                if _dsp._P1_LMHEAD_RING:  # p1/L5-dense-gemm: the 65-128-row tiers, compiled before capture
                    from atom.model_ops import p1_lmhead_ring as _ring

                    _ring.warmup(device, head._dg_ak_wshuf)
        max_rows = self.max_bs * self.rq
        max_tiles = triton.cdiv(self.v_local, _chunk_for(1))
        f32 = {"dtype": torch.float32, "device": device}
        i32 = {"dtype": torch.int32, "device": device}
        self.step = torch.zeros(1, dtype=torch.int64, device=device)
        self.ps_val = torch.empty(max_tiles * max_rows, **f32)
        self.ps_idx = torch.empty(max_tiles * max_rows, **i32)
        self.pl_val = torch.empty(max_tiles * max_rows, **f32)
        self.pl_idx = torch.empty(max_tiles * max_rows, **i32)
        self.packed = torch.empty(max_rows, 4, **f32)
        # ATOMIC: each row's (draw, raw) packed bests, folded by the partials
        # kernel's atomic_max and reset by the verdict kernel after the gather
        self.atomic = envs.ATOM_M3_VERIFY_HEAD_ATOMIC
        self.packed64 = torch.full((max_rows, 2), EMPTY, dtype=torch.int64, device=device)
        self.out = torch.zeros(self.max_bs, self.rq, **i32)
        self.verdict = torch.zeros(4 * self.max_bs, **i32)
        self._cond = None
        self._dummy_cond = torch.zeros(self.rq, **f32)
        # The side stream the decode forward writes the draft's context rows on
        # (`MiniMaxM3SparseForCausalLM._write_draft_context`), joined at the end
        # of `__call__`; None runs that write inline.
        self.ctx_stream = (
            torch.cuda.Stream(device=device) if envs.ATOM_M3_DRAFT_CTX_OVERLAP else None
        )
        self.ctx_pending = False
        # the side stream's input, kept alive until the join
        self.ctx_held = None

    def join_ctx(self) -> None:
        """Make the current stream wait for the draft context write, if one is
        in flight on the side stream."""
        if self.ctx_pending:
            torch.cuda.current_stream().wait_stream(self.ctx_stream)
            self.ctx_pending, self.ctx_held = False, None

    def _cond_rates(self):
        """The forced-acceptance conditional rates, or None for a real verify."""
        if self._cond is None:
            from atom.model_ops.rejection_sampler import _get_synthetic_cond_rates

            rates = self.runner.rejection_sampler.synthetic_acceptance_rates
            self._cond = (
                (_get_synthetic_cond_rates(rates, self.device),)
                if rates is not None
                else ()
            )
        return self._cond[0] if self._cond else None

    def __call__(self, hidden_states: torch.Tensor):
        try:
            return self._head(hidden_states)
        finally:
            # whatever this step's head output is, the draft's context rows are
            # written before it is complete (and a capture joins its fork)
            self.join_ctx()

    def _head(self, hidden_states: torch.Tensor):
        fwd = get_forward_context()
        ctx = fwd.context
        md = fwd.attn_metadata
        if ctx is None or ctx.is_prefill or ctx.is_draft or md is None:
            return None
        rq = self.rq
        rows = hidden_states.shape[0]
        ids = self.lm._step_input_ids
        if (
            getattr(md, "max_seqlen_q", None) != rq
            or rows % rq
            or rows > self.max_bs * rq
            or ids is None
            or ids.numel() < rows
        ):
            return None
        from atom.models.minimax_m3.dspark import local_logits

        temps = self.runner.forward_vars["temperatures"].gpu
        logits = local_logits(self.lm.lm_head, hidden_states)  # [rows, V / tp]
        return self.verify(logits, ids, temps, self._cond_rates())

    def verify(self, logits, ids, temps, cond):
        """The head on this rank's ``[rows, V / tp]`` logits: output ids
        ``[B, q]`` and verdict ``[4, B]`` (``cond``: the forced-acceptance
        rates, None for a real verify)."""
        packed = self.shard_pack(logits, temps)
        if self.world > 1:
            gathered = get_tp_group().all_gather(
                packed, dim=0, use_custom=envs.ATOM_USE_CUSTOM_ALL_GATHER
            )
        else:
            gathered = packed
        result = self.verdict_from(gathered, ids, temps, cond, logits.shape[0])
        self.step.add_(1)
        return result

    def shard_pack(self, logits, temps):
        """This rank's ``[rows, 4]`` 32-bit words per row for the exchange: the
        packed (draw, raw) int64 bests (ATOMIC), else fp32 (draw key, draw id,
        logit, argmax id), each over its vocab shard."""
        rq = self.rq
        rows = logits.shape[0]
        chunk = (
            _atomic_chunk_for(rows)
            if self.atomic and envs.ATOM_M3_VERIFY_HEAD_CHUNK_TUNED
            else _chunk_for(rows)
        )
        tiles = triton.cdiv(self.v_local, chunk)
        n = rows * tiles
        ps_val, ps_idx = self.ps_val[:n], self.ps_idx[:n]
        pl_val, pl_idx = self.pl_val[:n], self.pl_idx[:n]
        packed64 = self.packed64[:rows]
        _shard_partials_kernel[(rows, tiles)](
            logits, logits.stride(0), temps, self.step,
            ps_val, ps_idx, pl_val, pl_idx, packed64,
            rows, self.v_local, self.vocab_start, self.vocab_total,
            DRAW_KEY=_DRAW_KEY, SENTINEL=_SENTINEL,
            RQ=rq, CHUNK=chunk, BLOCK=_BLOCK, ATOMIC=self.atomic, num_warps=4,
        )  # fmt: skip
        if self.atomic:
            return packed64.view(torch.float32)  # [rows, 4]
        packed = self.packed[:rows]
        _shard_pack_kernel[(rows,)](
            ps_val, ps_idx, pl_val, pl_idx, packed, rows, tiles,
            BLOCK_TILE=triton.next_power_of_2(tiles), num_warps=1,
        )  # fmt: skip
        return packed

    def verdict_from(self, gathered, ids, temps, cond, rows, local=None):
        """Output ids and verdict from every rank's ``[W * rows, 4]`` words.
        ATOMIC: ``local`` (default this head's) is the packed buffer the verdict
        resets for the next step."""
        rq = self.rq
        nb = rows // rq
        out = self.out[:nb]
        verdict = self.verdict[: 4 * nb].view(4, nb)
        if self.atomic:
            gathered = gathered.view(torch.int64)
            local = self.packed64[:rows] if local is None else local
        _verdict_kernel[(nb,)](
            gathered, ids, temps, self.step,
            self._dummy_cond if cond is None else cond,
            out, verdict, self.packed64 if local is None else local, rows,
            ACCEPT_KEY=_ACCEPT_KEY, W=self.world, RQ=rq,
            SP=triton.next_power_of_2(rq), SYNTHETIC=cond is not None,
            EPS=SAMPLER_EPS, ATOMIC=self.atomic, num_warps=1,
        )  # fmt: skip
        return out, verdict


class M3RunnerVerifyHead(VerifyHead):
    """``M3VerifyHead`` as the runner's verify head (``ModelRunner._bind_verify_head``):
    run after the decode forward, outside its graph, where ROCm/ATOM#3736 put every
    head. Draws from the temperature alone (no top-p), and every rank picks the same
    ids from the same gathered pairs."""

    samples = ids_agree = True

    def __init__(self, head: M3VerifyHead) -> None:
        self.head = head

    def sample_token_ids(self, hidden: torch.Tensor):
        # a replayed graph ran no Python forward: read the runner's staged ids
        self.head.lm._step_input_ids = self.head.runner.forward_vars["input_ids"].gpu
        out = self.head(hidden)
        if out is None:
            return None
        ids, verdict = out
        return VerifyOutput(ids, verdict[:3])
