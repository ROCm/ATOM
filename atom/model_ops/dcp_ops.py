# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DCP (Decode Context Parallel) communication ops for ATOM.

Implements the AG+RS backend for combining partial attention outputs
across DCP ranks using LSE (Log-Sum-Exp) correction.
Uses vllm-style algorithm: AllGather LSE -> correct local output -> ReduceScatter.
"""

import numpy as np
import torch
import triton
import triton.language as tl


class CPTritonContext:
    """Cache compiled Triton kernel to avoid recompilation on every call."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    """Correct local rank's attention output using all-gathered LSEs.

    For each (batch, head):
      1. global_lse = logsumexp(lse_0, ..., lse_{N-1})
      2. factor = exp(local_lse - global_lse)
      3. output *= factor

    After ReduceScatter(sum), the corrected outputs from all ranks
    combine into the final attention output.
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)

    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    lse_exp = tl.exp(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    global_lse = tl.log(lse_acc) + lse_max

    lse_out_offset = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_out_offset, global_lse)

    local_lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    local_lse = tl.load(lses_ptr + local_lse_offset)
    lse_diff = local_lse - global_lse
    lse_diff = tl.where(
        (lse_diff != lse_diff) | (lse_diff == float("inf")),
        -float("inf"),
        lse_diff,
    )
    factor = tl.exp(lse_diff)

    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + output_offsets, output)


def correct_attn_out(out, lses, cp_rank, ctx=None):
    """Correct local rank's attention output using all-gathered LSEs.

    Args:
        out: [B, H, D] local attention output
        lses: [N, B, H] all-gathered LSE values
        cp_rank: this rank's index in the CP group
        ctx: optional CPTritonContext to cache compiled kernel

    Returns:
        (out, lse): corrected output [B, H, D] and global LSE [B, H]
    """
    B, H, D = out.shape
    N = lses.shape[0]

    lse = torch.empty(B, H, device=lses.device, dtype=lses.dtype)

    grid = (B, H, 1)
    regular_args = (
        out,
        out,
        lses,
        lse,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lses.stride(0),
        lses.stride(1),
        lses.stride(2),
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N}

    if ctx is not None:
        ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    else:
        _correct_attn_cp_out_kernel[grid](*regular_args, **const_args)

    return out, lse


def cp_lse_ag_out_rs(cp_attn_out, cp_attn_lse, cp_group, ctx=None):
    """AG+RS backend: AllGather LSE -> Triton correct -> ReduceScatter output.

    Args:
        cp_attn_out: [B, H_full, D] local attention output (full heads after AG Q)
        cp_attn_lse: [B, H_full] local LSE values
        cp_group: DCP communication group (GroupCoordinator)
        ctx: optional CPTritonContext to cache compiled kernel

    Returns:
        output: [B, H_local, D] corrected output with local heads only
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = cp_group.all_gather(cp_attn_lse, dim=0)
    lses = lses.reshape((cp_group.world_size,) + cp_attn_lse.shape)

    out, _ = correct_attn_out(cp_attn_out, lses, cp_group.rank_in_group, ctx=ctx)

    out = out.movedim(1, 0).contiguous()  # [B, H_full, D] -> [H_full, B, D]
    out = cp_group.reduce_scatter(out, dim=0)
    out = out.movedim(0, 1).contiguous()  # [H_local, B, D] -> [B, H_local, D]
    return out


def dcp_gather_compressed_kv(
    kv_cache: torch.Tensor, slot_ids: torch.Tensor
) -> torch.Tensor:
    """Gather this rank's compressed KV entries from the paged cache.

    Local-gather step: the MLA cache stores compressed latent KV as
    ``[num_slots, 1, kv_lora_rank + qk_rope_head_dim]`` (or ``[num_slots, d]``),
    so gathering the local rank's interleaved tokens for a chunk is a plain
    index_select over the token-slot axis. This replaces vLLM's
    ``cp_gather_cache`` custom op (unavailable in the aiter server path).

    Args:
        kv_cache: paged compressed KV cache, token-slot major on dim 0.
        slot_ids: int tensor of absolute slot ids to gather (this rank's local
            tokens for the chunk, in per-seq order).

    Returns:
        [len(slot_ids), kv_lora_rank + qk_rope_head_dim] compressed KV.
    """
    gathered = kv_cache.index_select(0, slot_ids)
    # Collapse any singleton head dim -> [toks, kv_lora_rank + qk_rope_head_dim].
    return gathered.reshape(slot_ids.shape[0], -1)


def reorg_kvcache(
    allgatered_kv_c_normed: torch.Tensor,
    allgatered_k_pe: torch.Tensor,
    padded_local_chunk_seq_lens_lst: list,
    local_context_lens_allranks: list,
    sum_seq_len: int,
    max_seq_len: int,
    chunk_size: int,
    chunk_idx: int,
    toks: int,
):
    """Reorg + unpad AllGathered compressed KV into per-sequence contiguous
    layout for the attention kernel.

    The AllGather concatenates every rank's local (padded) chunk gather along
    dim 0, so tokens for one sequence are interleaved across the per-rank
    blocks. This walks each seq's per-rank contribution and concatenates them
    back into the original token order, dropping padding.

    e.g.
    allgatered = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,      # rank 0 block
                  T0_4, T0_5, pad, pad, T1_2, pad, ...]         # rank 1 block
    -> reorganized = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                      T1_0, T1_1, T1_2, ...]

    Args:
        padded_local_chunk_seq_lens_lst: per-seq local chunk lengths (padded)
            under the current CP rank.
        local_context_lens_allranks: per-seq local context lengths on each rank.
        sum_seq_len: sum of the per-seq (global) chunk lengths.
        max_seq_len: max per-seq (global) chunk length.
        chunk_size: local padded max context chunk from metadata building.
        chunk_idx: chunk index of the chunked prefill.
        toks: number of tokens per rank's local gather (one AllGather block).
    """
    kv_c_segments = []
    k_pe_segments = []
    src_token_idx = 0
    max_seq_len_check = 0
    for padded_local_chunk_seq_len, local_context_lens in zip(
        padded_local_chunk_seq_lens_lst, local_context_lens_allranks
    ):
        cur_seq_len = 0
        for rank, local_context_len in enumerate(local_context_lens):
            # We split the context into multiple chunks depending on the
            # workspace size, so the last chunk on a shorter rank may be
            # partial: clamp to what actually remains on that rank.
            local_chunk_len = min(
                max(0, local_context_len - chunk_idx * chunk_size),
                padded_local_chunk_seq_len,
            )
            if local_chunk_len != 0:
                kv_c_segment = allgatered_kv_c_normed[
                    rank * toks
                    + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                k_pe_segment = allgatered_k_pe[
                    rank * toks
                    + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                kv_c_segments.append(kv_c_segment)
                k_pe_segments.append(k_pe_segment)
                cur_seq_len += local_chunk_len
        max_seq_len_check = max(max_seq_len_check, cur_seq_len)
        src_token_idx += padded_local_chunk_seq_len
    reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
    reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
    assert reorganized_kv_c_normed.shape[0] == sum_seq_len
    assert reorganized_k_pe.shape[0] == sum_seq_len
    assert max_seq_len_check == max_seq_len
    return reorganized_kv_c_normed, reorganized_k_pe


def get_dcp_local_seq_lens(seq_lens, dcp_size, dcp_rank, interleave_size=1):
    """Compute per-DCP-rank local sequence lengths.

    With interleaved storage, token i is stored on rank
    (i // interleave_size) % dcp_size.

    Args:
        seq_lens: numpy array of sequence lengths
        dcp_size: DCP world size
        dcp_rank: this rank's DCP rank
        interleave_size: interleaving granularity (default 1 = token-level)

    Returns:
        local_seq_lens: numpy array of local sequence lengths
    """
    full_chunks = seq_lens // (interleave_size * dcp_size)
    base = full_chunks * interleave_size

    remainder_total = seq_lens - base * dcp_size
    remainder = np.clip(
        remainder_total - dcp_rank * interleave_size, 0, interleave_size
    )
    return base + remainder


# ---------------------------------------------------------------------------
# Deterministic global top-k over exchanged DCP candidates. Replaces "all-gather
# the full logit plane" with "all-gather W*topk (score, global_id) candidates" on
# the decode indexer path.
#
# Every rank runs the merge independently, so the selection must be a *function*
# of its input, not merely "usually the same": an ambiguous choice resolved
# differently on two ranks breaks the disjoint-partition property that
# `cp_lse_ag_out_rs` relies on (6.1.3). Ambiguity can only arise among
# candidates whose score exactly equals the selection threshold, so ties are
# broken by the globally unique token id:
#
#     keep  score > T
#     keep  score == T  and  gid <= G     where G is the `need`-th smallest gid
#
# Measured on real data (25,600 rows, 6.1.4): 0.06% of rows have a boundary
# tie and the tied set is never larger than 2.
# ---------------------------------------------------------------------------

# Ties measured <= 2 (6.1.4); 256 leaves a 128x margin. Overflow is reported in
# a per-row flag rather than silently mis-selecting. Must be a power of two
# (`tl.sort`).
DCP_TOPK_TIE_CAP = 256


@triton.jit
def _dcp_stable_topk_kernel(
    scores,  # fp32 [rows, n]
    gids,  # int32 [rows, n], < 0 marks a padding candidate
    thr,  # fp32 [rows] -- k-th largest score, from topk_plain
    out,  # int32 [rows, k]
    tie_buf,  # int32 [rows, CAP] scratch
    overflow,  # int32 [rows]
    k,
    n,
    s_s0: tl.int64,
    g_s0: tl.int64,
    o_s0: tl.int64,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
):
    """One program per row: find the tie boundary, then emit.

    Loop 1 reads the score plane; its gid loads are masked to the handful of
    tied lanes, so they touch almost no extra cache lines. Loop 2 reads both.
    """
    row = tl.program_id(0)
    sbase = row * s_s0
    gbase = row * g_s0
    obase = row * o_s0
    t = tl.load(thr + row)
    # A row whose candidate set is smaller than k emits fewer than k ids. `out`
    # is reused across steps, so pad first or stale ids from the previous step
    # survive in the tail.
    for start in range(0, k, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        tl.store(out + obase + cols, tl.full([BLOCK], -1, tl.int32), mask=cols < k)

    # t == -inf means the row has fewer than k real candidates. Everything real
    # is then strictly greater and the padding (-inf) must be dropped, so the
    # tie branch has to stay off -- otherwise every pad lane looks "tied" and
    # floods tie_buf.
    finite_t = t > -float("inf")
    tl.debug_barrier()

    # ---- loop 1: count strict winners, scatter tied gids to scratch ----
    # The tied gids go to GLOBAL scratch, not a register array: Triton cannot
    # index registers dynamically, so a register buffer needs an O(CAP) fold per
    # tile -- measured 2117 us at CAP=256 against a ~54 us two-pass floor
    # (6.1.6). A global store takes the computed per-lane slot directly.
    c_strict = 0
    n_tie = 0
    for start in range(0, n, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < n
        sc = tl.load(scores + sbase + cols, mask=m, other=-float("inf"))
        gt = m & (sc > t)
        eq = m & (sc == t) & finite_t
        c_strict += tl.sum(gt.to(tl.int32))
        ei = eq.to(tl.int32)
        dst = n_tie + tl.cumsum(ei, 0) - ei
        g = tl.load(gids + gbase + cols, mask=eq, other=0)
        tl.store(tie_buf + row * CAP + dst, g, mask=eq & (dst < CAP))
        n_tie += tl.sum(ei)

    tl.store(overflow + row, (n_tie > CAP).to(tl.int32))
    tl.debug_barrier()

    # ---- G = the `need`-th smallest tied gid ----
    slot = tl.arange(0, CAP)
    buf = tl.load(tie_buf + row * CAP + slot, mask=slot < n_tie, other=2147483647)
    need = k - c_strict
    srt = tl.sort(buf, 0)  # ascending; padding is INT32_MAX and sorts last
    g_thr = tl.sum(tl.where(slot == (need - 1), srt, 0))
    g_thr = tl.where(need > 0, g_thr, -1)

    # ---- loop 2: emit ----
    written = 0
    for start in range(0, n, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < n
        sc = tl.load(scores + sbase + cols, mask=m, other=-float("inf"))
        g = tl.load(gids + gbase + cols, mask=m, other=-1)
        take = m & (g >= 0) & ((sc > t) | ((sc == t) & finite_t & (g <= g_thr)))
        ti = take.to(tl.int32)
        dst = written + tl.cumsum(ti, 0) - ti
        tl.store(out + obase + dst, g, mask=take & (dst < k))
        written += tl.sum(ti)


def dcp_stable_topk(scores, gids, k, out=None):
    """Deterministic top-k over a candidate set. scores/gids: [rows, n].

    Returns (out int32 [rows, k] of global ids, overflow int32 [rows]).
    `out` is padded with -1 when a row holds fewer than k real candidates.

    CUDAGraph note: every shape here is static, and `torch.empty` inside a
    captured region lands in the graph's private pool at a stable address. The
    caller must NOT read `overflow` back to host per step -- that is a D2H sync
    and breaks capture. Read it only under a diagnostic switch.
    """
    from aiter import topk_plain

    rows, n = scores.shape
    dev = scores.device
    if out is None:
        out = torch.empty(rows, k, dtype=torch.int32, device=dev)
    val_buf = torch.empty(rows, k, dtype=torch.float32, device=dev)
    idx_buf = torch.empty(rows, k, dtype=torch.int32, device=dev)
    row_starts = torch.zeros(rows, dtype=torch.int32, device=dev)
    row_ends = torch.full((rows,), n, dtype=torch.int32, device=dev)
    tie_buf = torch.empty(rows, DCP_TOPK_TIE_CAP, dtype=torch.int32, device=dev)
    overflow = torch.zeros(rows, dtype=torch.int32, device=dev)

    # aiter's tuned topk_plain is fp32-only, so it cannot take a 64-bit
    # composite key; it supplies the score threshold and the kernel above adds
    # the deterministic tie-break. A from-scratch Triton radix-select measured
    # 524 us against a ~26.9 us/pass memory floor (6.1.5).
    topk_plain(
        scores,
        idx_buf,
        val_buf,
        k,
        True,
        row_starts,
        row_ends,
        scores.stride(0),
        1,
    )
    thr = val_buf.min(dim=-1).values

    _dcp_stable_topk_kernel[(rows,)](
        scores,
        gids,
        thr,
        out,
        tie_buf,
        overflow,
        k,
        n,
        scores.stride(0),
        gids.stride(0),
        out.stride(0),
        BLOCK=1024,
        CAP=DCP_TOPK_TIE_CAP,
        num_warps=8,
    )
    return out, overflow


def dcp_pack_topk_candidates(
    local_logits, local_idx, local_lens, dcp_rank, dcp_world_size, out_pair
):
    """Turn a rank-local top-k into exchangeable (score, global_id) pairs.

    out_pair: fp32 [2, rows, k] -- plane 0 holds scores, plane 1 holds int32
    global ids reinterpreted as fp32 so both travel in one collective. Slots the
    local top-k did not fill get (-inf, -1); `dcp_stable_topk` drops gid < 0.

    Under round-robin (interleave=1) sharding a local position j on rank r is
    global position j*W + r, so the id is globally unique -- which is what makes
    the tie-break a total order.
    """
    rows, k = local_idx.shape
    # Bound-check rather than assume a padding convention from the aiter kernel.
    valid = (local_idx >= 0) & (local_idx < local_lens.view(rows, 1))
    safe = torch.where(valid, local_idx, torch.zeros_like(local_idx))
    sc = torch.gather(local_logits, 1, safe.to(torch.int64))
    out_pair[0].copy_(torch.where(valid, sc, torch.full_like(sc, -float("inf"))))
    gid = torch.where(
        valid,
        local_idx * dcp_world_size + dcp_rank,
        torch.full_like(local_idx, -1),
    )
    out_pair.view(torch.int32)[1].copy_(gid)
