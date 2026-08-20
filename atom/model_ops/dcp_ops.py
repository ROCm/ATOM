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

_AG_CUSTOM_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _ag_custom_view_dtype(x: torch.Tensor) -> torch.dtype | None:
    """Float dtype to reinterpret `x` as for aiter's custom all-gather.

    The custom kernel dispatches on a float enum (fp32/fp16/bf16) and raises on
    anything else, even though an all-gather is a pure copy that only cares
    about element width. The fp8 query therefore has to be viewed first; 8-bit
    payloads have no 8-bit entry in the enum, so they pair up into fp16, which
    needs an even trailing dim. Returns None when no safe view exists.
    """
    if x.dtype in _AG_CUSTOM_DTYPES:
        return x.dtype
    itemsize = x.element_size()
    if itemsize == 4:
        return torch.float32
    if itemsize == 2:
        return torch.float16
    if itemsize == 1 and x.shape[-1] % 2 == 0:
        return torch.float16
    return None


def dcp_all_gather(cp_group, x: torch.Tensor, dim: int) -> torch.Tensor:
    """AllGather that prefers aiter's custom collective over pynccl.

    ``GroupCoordinator.all_gather`` takes ``use_custom=False`` by default, so
    every DCP gather lands on pynccl. At decode payloads that is the wrong
    trade: the nccl kernel carries a fixed ~5us launch bubble on each side, so
    gathering a few KB of LSE costs more than the custom kernel spends on a
    1.5MB reduce-scatter. The device communicator picks the custom kernel when
    the shape qualifies (dim 0 or last dim, 16B-aligned, within the registered
    pool) and falls back to pynccl otherwise, with identical concat-along-`dim`
    semantics either way.
    """
    device_comm = getattr(cp_group, "device_communicator", None)
    view_dtype = _ag_custom_view_dtype(x) if device_comm is not None else None
    if view_dtype is None:
        return cp_group.all_gather(x, dim=dim)
    if view_dtype is x.dtype:
        return device_comm.all_gather(x, dim)
    # Viewing narrows the trailing dim, so only a last-dim gather sees a
    # different split; both land on the same byte layout after the view back.
    return device_comm.all_gather(x.view(view_dtype), dim).view(x.dtype)


def dcp_all_gather_query_heads(cp_group, q: torch.Tensor) -> torch.Tensor:
    """AllGather decode Q ``[tokens, heads, head_dim]`` over the DCP group's heads.

    Flattened to 2-D so the gather lands on the last dim. aiter's custom
    collective only serves dim 0 and the last dim, and its last-dim variant
    concatenates rank-major -- which for a ``[tokens, heads*head_dim]`` view
    is exactly head-dim concat. Gathering the 3-D tensor on dim=1 instead
    both misses the custom kernel and makes the pynccl path materialise an
    extra reshape copy of the gathered result.
    """
    tokens, _, head_dim = q.shape
    if not q.is_contiguous():
        return cp_group.all_gather(q, dim=1)
    gathered = dcp_all_gather(cp_group, q.reshape(tokens, -1), -1)
    return gathered.view(tokens, -1, head_dim)


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
    # A rank that owns no candidate for this row has local_lse=-inf -> factor=0,
    # and its attention output is NaN (0/0 in the kernel's acc/e_sum). NaN*0=NaN
    # would poison the ReduceScatter sum, so force the empty-rank contribution to
    # 0. Matches vLLM's cp merge kernel; covers both decode and prefill so callers
    # need no separate NaN scrub.
    output = tl.where(factor == 0.0, 0.0, output)
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
    assert (N & (N - 1)) == 0, f"cp world size must be a power of two, got {N}"

    lse = torch.empty_strided(
        (B, H), (lses.stride(1), lses.stride(2)), device=lses.device, dtype=lses.dtype
    )

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
    lses = dcp_all_gather(cp_group, cp_attn_lse, 0)
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


def dcp_pack_topk_candidates(
    local_logits, local_idx, local_lens, dcp_rank, dcp_world_size, out_pair
):
    """Turn a rank-local top-k into exchangeable (score, global_id) pairs.

    out_pair: fp32 [2, rows, k] -- plane 0 holds scores, plane 1 holds int32
    global ids reinterpreted as fp32 so both travel in one collective. Slots the
    local top-k did not fill get (-inf, -1); the merge sinks them via -inf and
    never selects the -1 gids.

    Under round-robin (interleave=1) sharding a local position j on rank r is
    global position j*W + r, so the id is globally unique -- which is what makes
    the tie-break a total order.
    """
    rows, _k = local_idx.shape
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


# ---------------------------------------------------------------------------
# DCP sparse index filter + round-robin localize (decode & prefill).
# Two-pass compacting filter: count this rank's owned top-k, then pack the
# owned slots to the front of each region (no -1 holes -- see cp_lse_ag_out_rs).
# ---------------------------------------------------------------------------


@triton.jit
def _count_owned_dcp_kernel(
    qo_indptr,  # int32 [num_requests + 1]
    global_kv_indptr,  # int32 [num_requests + 1] -- GLOBAL context (column range)
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- GLOBAL top-k positions
    out_counts,  # int32 [num_requests] -- owned top-k count per request
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0,
    ti_stride1,
):
    """Pass 1 of the compacting DCP filter: how many of the global top-k
    positions does this rank own, per request? Its exclusive cumsum gives the
    compacted output offsets used by ``_compact_filter_dcp_kernel``.

    qlen==1 only (DCP + sparse + MTP is rejected upstream), so each request has
    exactly one query token.
    """
    batch_id = tl.program_id(0)
    token_id = tl.load(qo_indptr + batch_id)

    count = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        indice_id = tile_start + tl.arange(0, BLOCK_N)
        col_valid = indice_id < NUM_TOPK_TOKENS
        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr, mask=col_valid, other=-1)
        owned = col_valid & (tok >= 0) & ((tok % DCP_WORLD) == DCP_RANK)
        count += tl.sum(owned.to(tl.int32))

    tl.store(out_counts + batch_id, count)


@triton.jit
def _compact_filter_dcp_kernel(
    qo_indptr,  # int32 [num_requests + 1]
    global_kv_indptr,  # int32 [num_requests + 1] -- GLOBAL context (column range)
    out_kv_indptr,  # int32 [num_requests + 1] -- COMPACTED offsets (cumsum of pass 1)
    block_table,  # int32 [num_req, max_num_blocks_per_req] -- logical(global) blocks
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- GLOBAL top-k positions
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,  # runner (physical) block size
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0,
    ti_stride1,
    bt_stride0: tl.int64,
    bt_stride1: tl.constexpr,
):
    # DCP round-robin (interleave=1): a GLOBAL position g is owned by rank
    # ``g % W``; on the owner rank its physical slot follows the round-robin
    # (virtual-block) layout used by _dcp_round_robin_slot / ATOM PR #847:
    #     vbs  = PAGE_SIZE * W
    #     slot = block_table[req, g // vbs] * PAGE_SIZE + (g % vbs) // W
    # token_indices holds GLOBAL positions (the indexer scored the full sequence
    # via all-gathered logits). This rank keeps ONLY the positions it owns and
    # writes them COMPACTED to the front of its region -- no -1 holes. Holes are
    # exactly what breaks aiter's lse path (immediate fault on the persistent
    # kernel, silently unwritten lse on the split-KV one).
    #
    # Compaction is order-preserving (tl.cumsum within a tile plus a running
    # offset across tiles) rather than atomic-allocated like vLLM, so the KV
    # order -- and hence the floating-point accumulation order -- is
    # deterministic run to run, which the dcp=1 vs dcp=N comparison relies on.
    #
    # NOTE: the slot is computed from block_table directly (like vLLM) rather
    # than gathered from a precomputed kv_indices -- the DCP round-robin
    # per-token slot array does not exist on the sparse path (dense reads go
    # through block_tables in-kernel).
    batch_id = tl.program_id(0)

    out_kv_start = tl.load(out_kv_indptr + batch_id)
    token_id = tl.load(qo_indptr + batch_id)

    vbs = PAGE_SIZE * DCP_WORLD
    written = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        indice_id = tile_start + tl.arange(0, BLOCK_N)
        # Full top-k width; `tok >= 0` is the only valid-id guard. Must stay in
        # lock-step with `_count_owned_dcp_kernel` (see the note there on why the
        # old `indice_id < g_kv_len` mask is wrong) -- if the two disagree, the
        # counted offsets and the written entries diverge.
        col_valid = indice_id < NUM_TOPK_TOKENS

        ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
        tok = tl.load(ti_ptr, mask=col_valid, other=-1)  # GLOBAL position

        idx_valid = col_valid & (tok >= 0) & ((tok % DCP_WORLD) == DCP_RANK)

        block_id = tok // vbs
        inblock_offset = (tok % vbs) // DCP_WORLD
        physical_block = tl.load(
            block_table + batch_id * bt_stride0 + block_id * bt_stride1,
            mask=idx_valid,
            other=0,
        )
        slot = physical_block * PAGE_SIZE + inblock_offset

        # Exclusive prefix sum of the owned mask -> destination inside this tile.
        owned_i32 = idx_valid.to(tl.int32)
        dst = written + tl.cumsum(owned_i32, axis=0) - owned_i32
        tl.store(out_kv_indices + out_kv_start + dst, slot, mask=idx_valid)
        written += tl.sum(owned_i32)


def triton_filter_and_convert_dcp_index(
    qo_indptr: torch.Tensor,  # int32 [num_requests + 1]
    global_kv_indptr: torch.Tensor,  # int32 [num_requests + 1]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req] logical
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS] GLOBAL pos
    dcp_rank: int,
    dcp_world_size: int,
    block_size: int,  # runner (physical) block size == PAGE_SIZE
    out_kv_indptr: torch.Tensor,  # int32 [num_requests + 1] COMPACTED, written here
    owned_counts: torch.Tensor,  # int32 [>= num_requests] scratch for pass 1
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    out: torch.Tensor | None = None,
):
    """DCP (interleave=1) filter + round-robin localize of global top-k positions,
    **compacting** each rank's owned slots to the front of its region.

    ``token_indices[token_id, indice_id]`` is a GLOBAL token position selected by
    the indexer (scored over the full sequence via all-gathered logits). This
    rank keeps a position ``g`` only if ``g % W == dcp_rank`` and maps it to its
    physical slot via the round-robin (virtual-block) layout, computed directly
    from ``block_table`` (like vLLM):
        vbs  = block_size * W
        slot = block_table[req, g // vbs] * block_size + (g % vbs) // W

    Non-owned positions are **dropped**, not marked: the kept slots are packed
    contiguously (original top-k order preserved) and ``out_kv_indptr`` is
    rewritten to the resulting per-request lengths. This replaces the earlier
    "fixed length + -1 sentinel" layout, whose holes broke aiter's lse output.
    Because the kept count depends on the per-layer top-k selection,
    ``out_kv_indptr`` is layer-dependent and must be recomputed on every call --
    it cannot feed the once-per-step persistent metadata, which is why sparse+DCP
    runs non-persistent for now.

    The 8 ranks' kept sets are disjoint and their union is exactly the global
    top-k, which is what makes the downstream ``cp_lse_ag_out_rs`` merge valid.
    """
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )
    assert 0 <= dcp_rank < dcp_world_size
    assert out is not None, "sparse_kv_indices_buffer (out) is required"

    num_batch = global_kv_indptr.shape[0] - 1

    qo_indptr_c = qo_indptr.contiguous()
    global_kv_indptr_c = global_kv_indptr.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()

    ti_stride0, ti_stride1 = token_indices_c.stride()
    bt_stride0, bt_stride1 = block_table_c.stride()
    grid = (num_batch,)

    # Pass 1: per-request count of owned top-k positions.
    counts = owned_counts[:num_batch]
    _count_owned_dcp_kernel[grid](
        qo_indptr_c,
        global_kv_indptr_c,
        token_indices_c,
        counts,
        dcp_rank,
        dcp_world_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )

    # Exclusive cumsum -> compacted offsets. Written in place so the caller's
    # tensor (and anything already holding a view of it) sees the update.
    # dtype=int32 keeps the accumulation in int32 (torch would promote integral
    # cumsum to int64 by default, which the kernels' int32 pointers reject).
    # zero_() rather than `out_kv_indptr[0] = 0`: assigning a Python scalar goes
    # through a host->device copy, which HIP rejects while a graph is capturing
    # (hipErrorStreamCaptureUnsupported). Everything here must stay device-side.
    out_kv_indptr[:1].zero_()
    torch.cumsum(counts, dim=0, dtype=torch.int32, out=out_kv_indptr[1 : num_batch + 1])

    # Pass 2: write the owned slots packed to the front of each region.
    _compact_filter_dcp_kernel[grid](
        qo_indptr_c,
        global_kv_indptr_c,
        out_kv_indptr,
        block_table_c,
        token_indices_c,
        out,
        dcp_rank,
        dcp_world_size,
        block_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return out


@triton.jit
def _count_owned_dcp_prefill_kernel(
    dsa_kv_indptr,  # int32 [num_tokens + 1] -- GLOBAL per-token candidate counts
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- FLAT KV indices
    cu_seqlens_k,  # int32 [num_req + 1] -- per-seq base of the flat KV axis
    out_counts,  # int32 [num_tokens]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
):
    """Pass 1 of the compacting DCP filter for sparse PREFILL: how many of the
    global top-k candidates does this rank own, per QUERY TOKEN?

    Differs from the decode twin (`_count_owned_dcp_kernel`) in two ways:
    the row unit is a query token rather than a request (prefill has many query
    tokens per request), and `topk_indices` holds a FLAT index into the
    concatenated KV of all sequences, so the within-sequence position -- which
    is what the round-robin owner is derived from -- is `indice - cu_seqlens_k[req]`.
    """
    token_id = tl.program_id(0)
    req_id = tl.load(token_to_seq_idxs + token_id)
    base = tl.load(cu_seqlens_k + req_id)

    count = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        col_id = tile_start + tl.arange(0, BLOCK_N)
        col_valid = col_id < NUM_TOPK_TOKENS
        indice = tl.load(
            topk_indices + token_id * ti_stride0 + col_id * ti_stride1,
            mask=col_valid,
            other=-1,
        )
        pos = indice - base  # position within the sequence
        owned = col_valid & (indice >= 0) & ((pos % DCP_WORLD) == DCP_RANK)
        count += tl.sum(owned.to(tl.int32))

    tl.store(out_counts + token_id, count)


@triton.jit
def _compact_filter_dcp_prefill_kernel(
    dsa_kv_indptr,  # int32 [num_tokens + 1] -- GLOBAL per-token candidate counts
    out_kv_indptr,  # int32 [num_tokens + 1] -- COMPACTED offsets (cumsum of pass 1)
    token_to_seq_idxs,  # int32 [num_tokens]
    topk_indices,  # int32 [num_tokens, NUM_TOPK_TOKENS] -- FLAT KV indices
    cu_seqlens_k,  # int32 [num_req + 1]
    block_table,  # int32 [num_req, max_num_blocks_per_req] -- logical(global) blocks
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    DCP_RANK: tl.constexpr,
    DCP_WORLD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,  # runner (physical) block size
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ti_stride0: tl.int64,
    ti_stride1: tl.constexpr,
    bt_stride0: tl.int64,
    bt_stride1: tl.constexpr,
):
    """Pass 2 for sparse PREFILL: keep only this rank's owned candidates, map
    them through the round-robin (virtual-block) layout, and pack them to the
    front of the token's region.

        vbs  = PAGE_SIZE * W
        slot = block_table[req, pos // vbs] * PAGE_SIZE + (pos % vbs) // W

    Same rationale as the decode twin: no `-1` holes (they break aiter's lse
    path) and compaction is order-preserving so the fp accumulation order is
    deterministic.
    """
    token_id = tl.program_id(0)
    req_id = tl.load(token_to_seq_idxs + token_id)
    base = tl.load(cu_seqlens_k + req_id)
    out_kv_start = tl.load(out_kv_indptr + token_id)

    vbs = PAGE_SIZE * DCP_WORLD
    written = 0
    for tile_start in range(0, NUM_TOPK_TOKENS, BLOCK_N):
        col_id = tile_start + tl.arange(0, BLOCK_N)
        # Full-width scan; `indice >= 0` is the only validity guard. Must stay in
        # lockstep with pass 1 -- if the two disagree on which candidates are
        # valid, `out_kv_indptr` (built from pass 1's counts) and the writes here
        # go out of sync and rows overwrite each other.
        col_valid = col_id < NUM_TOPK_TOKENS
        indice = tl.load(
            topk_indices + token_id * ti_stride0 + col_id * ti_stride1,
            mask=col_valid,
            other=-1,
        )
        pos = indice - base
        idx_valid = col_valid & (indice >= 0) & ((pos % DCP_WORLD) == DCP_RANK)

        block_id = pos // vbs
        inblock_offset = (pos % vbs) // DCP_WORLD
        physical_block = tl.load(
            block_table + req_id * bt_stride0 + block_id * bt_stride1,
            mask=idx_valid,
            other=0,
        )
        slot = physical_block * PAGE_SIZE + inblock_offset

        owned_i32 = idx_valid.to(tl.int32)
        dst = written + tl.cumsum(owned_i32, axis=0) - owned_i32
        tl.store(out_kv_indices + out_kv_start + dst, slot, mask=idx_valid)
        written += tl.sum(owned_i32)

    # A row this rank owns nothing of is left EMPTY (zero-length region). That is
    # both legal and correct for mla_decode_fwd: it writes lse = -inf, which
    # cp_lse_ag_out_rs turns into a zero weight. Its `o` comes out NaN (0/0), so
    # the caller zeroes those rows -- see _forward_prefill_mla. No dummy candidate
    # is injected; the attention never sees fabricated KV.


def triton_filter_and_convert_dcp_index_prefill(
    dsa_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1] GLOBAL counts
    token_to_seq_idxs: torch.Tensor,  # int32 [num_tokens]
    topk_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS] FLAT indices
    cu_seqlens_k: torch.Tensor,  # int32 [num_req + 1]
    block_table: torch.Tensor,  # int32 [num_req, max_num_blocks_per_req] logical
    dcp_rank: int,
    dcp_world_size: int,
    block_size: int,  # runner (physical) block size == PAGE_SIZE
    out_kv_indptr: torch.Tensor,  # int32 [num_tokens + 1] COMPACTED, written here
    owned_counts: torch.Tensor,  # int32 [>= num_tokens] scratch for pass 1
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
    out: torch.Tensor | None = None,
):
    """Sparse-PREFILL twin of ``triton_filter_and_convert_dcp_index``.

    The decode version keys on requests (qlen==1); prefill has one row per query
    token and its ``topk_indices`` are flat KV indices rather than
    within-sequence positions. Everything else -- two passes, in-place int32
    cumsum, order-preserving compaction -- is identical, and the same
    layer-scoped buffers are reused (they are sized ``max_num_batched_tokens``,
    which bounds the prefill token count too).
    """
    assert topk_indices.dtype == torch.int32
    assert topk_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by"
        f"BLOCK_N ({BLOCK_N})"
    )
    assert 0 <= dcp_rank < dcp_world_size
    assert out is not None, "sparse_kv_indices_buffer (out) is required"

    num_tokens = dsa_kv_indptr.shape[0] - 1

    dsa_kv_indptr_c = dsa_kv_indptr.contiguous()
    token_to_seq_idxs_c = token_to_seq_idxs.contiguous()
    topk_indices_c = topk_indices.contiguous()
    cu_seqlens_k_c = cu_seqlens_k.contiguous()
    block_table_c = block_table.contiguous()

    ti_stride0, ti_stride1 = topk_indices_c.stride()
    bt_stride0, bt_stride1 = block_table_c.stride()
    grid = (num_tokens,)

    counts = owned_counts[:num_tokens]
    _count_owned_dcp_prefill_kernel[grid](
        dsa_kv_indptr_c,
        token_to_seq_idxs_c,
        topk_indices_c,
        cu_seqlens_k_c,
        counts,
        dcp_rank,
        dcp_world_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
    )

    # Same device-side-only cumsum as the decode path (see its comment for why
    # zero_() and dtype=torch.int32 are required).
    out_kv_indptr[:1].zero_()
    torch.cumsum(
        counts, dim=0, dtype=torch.int32, out=out_kv_indptr[1 : num_tokens + 1]
    )

    _compact_filter_dcp_prefill_kernel[grid](
        dsa_kv_indptr_c,
        out_kv_indptr,
        token_to_seq_idxs_c,
        topk_indices_c,
        cu_seqlens_k_c,
        block_table_c,
        out,
        dcp_rank,
        dcp_world_size,
        block_size,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        ti_stride0,
        ti_stride1,
        bt_stride0,
        bt_stride1,
    )
    return out
