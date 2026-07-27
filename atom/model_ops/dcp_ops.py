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
