# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Position-addressed k-pool history for speculative verification.

Keep one position-addressed ring of raw keys/gates. A rejected suffix may
overwrite future residues, but the next verification supplies those positions
as fresh rows before they are consumed. Pooled entries beyond the committed
position stay invisible; closing the same pool again overwrites its speculative
entry before scoring.
"""

import torch
import triton
import triton.language as tl

from . import kpool


@triton.jit
def _update_kpool_history_kernel(
    keys,
    gates,
    positions,
    cu_seqlens_q,
    source_slots,
    destination_slots,
    history,
    HISTORY_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    request_idx = tl.program_id(0)
    residue = tl.program_id(1)
    source_slot = tl.load(source_slots + request_idx)
    destination_slot = tl.load(destination_slots + request_idx)
    query_start = tl.load(cu_seqlens_q + request_idx)
    query_end = tl.load(cu_seqlens_q + request_idx + 1)
    if destination_slot < 0 or query_end <= query_start:
        return
    last_position = tl.load(positions + query_end - 1)
    position = last_position - (last_position - residue) % HISTORY_SIZE
    first_position = tl.load(positions + query_start)
    query_row = query_start + position - first_position
    is_fresh = (
        (position >= first_position)
        & (query_row >= query_start)
        & (query_row < query_end)
    )
    offsets = tl.arange(0, HEAD_DIM)
    for plane in tl.static_range(2):
        history_offset = (
            (tl.maximum(source_slot, 0) * 2 + plane) * HISTORY_SIZE + residue
        ) * HEAD_DIM
        previous = tl.load(history + history_offset + offsets)
        source = keys if plane == 0 else gates
        value = tl.load(
            source + query_row * HEAD_DIM + offsets,
            mask=is_fresh,
            other=0,
        )
        value = tl.where(is_fresh, value, previous)
        destination_offset = (
            (destination_slot * 2 + plane) * HISTORY_SIZE + residue
        ) * HEAD_DIM
        tl.store(history + destination_offset + offsets, value)


def get_query_request_indices(
    cu_seqlens_q: torch.Tensor,
    num_query_tokens: int,
) -> torch.Tensor:
    """Map each packed query row to its request index."""
    # Unlike repeat_interleave, this has no data-dependent output allocation.
    return torch.bucketize(
        torch.arange(num_query_tokens, device=cu_seqlens_q.device),
        cu_seqlens_q[1:],
        right=True,
    ).clamp_max(cu_seqlens_q.numel() - 2)


def build_speculative_pool_candidates(
    history: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    source_slots: torch.Tensor,
    pool_bias: torch.Tensor,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Form the pool ending at each query, before any history is overwritten."""
    num_query_tokens = keys.shape[0]
    request_indices = get_query_request_indices(cu_seqlens_q, num_query_tokens)
    request_start_rows = cu_seqlens_q[:-1].to(torch.int64)[request_indices]
    request_start_positions = positions[request_start_rows]
    pool_positions = (
        positions.to(torch.int64)[:, None]
        - pool_size
        + 1
        + torch.arange(pool_size, device=keys.device)
    )
    query_rows = (
        request_start_rows[:, None] + pool_positions - request_start_positions[:, None]
    )
    fresh_mask = pool_positions >= request_start_positions[:, None]
    history_slots = source_slots.to(torch.int64)[request_indices].clamp_min(0)
    history_positions = pool_positions % history.shape[2]
    previous_keys = history[history_slots[:, None], 0, history_positions]
    previous_gates = history[history_slots[:, None], 1, history_positions]
    query_rows = query_rows.clamp(0, num_query_tokens - 1)
    candidate_keys = torch.where(fresh_mask[..., None], keys[query_rows], previous_keys)
    candidate_gates = torch.where(
        fresh_mask[..., None], gates[query_rows], previous_gates
    )
    pooled_keys = kpool.pool_and_rotate(
        candidate_keys,
        candidate_gates,
        pool_bias,
    )
    return pooled_keys, request_indices


def update_speculative_kpool_history(
    history: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
) -> None:
    """Copy committed history and overlay fresh verification rows."""
    _update_kpool_history_kernel[(cu_seqlens_q.numel() - 1, history.shape[2])](
        keys,
        gates,
        positions,
        cu_seqlens_q,
        source_slots,
        destination_slots,
        history,
        HISTORY_SIZE=history.shape[2],
        HEAD_DIM=keys.shape[1],
    )


@triton.jit
def _map_token_indices_to_slots_kernel(
    token_indices,
    request_indices,
    block_tables,
    output_indptr,
    output,
    INPUT_WIDTH: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    row_index = tl.program_id(0)
    column_indices = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    output_start = tl.load(output_indptr + row_index)
    output_end = tl.load(output_indptr + row_index + 1)
    request_index = tl.load(request_indices + row_index)
    token_index = tl.load(
        token_indices + row_index * INPUT_WIDTH + column_indices,
        mask=column_indices < INPUT_WIDTH,
        other=-1,
    )
    is_valid = (token_index >= 0) & (token_index // BLOCK_SIZE < BLOCK_TABLE_STRIDE)
    block_index = tl.load(
        block_tables + request_index * BLOCK_TABLE_STRIDE + token_index // BLOCK_SIZE,
        mask=is_valid,
        other=0,
    )
    cache_slot = tl.where(
        is_valid,
        block_index * BLOCK_SIZE + token_index % BLOCK_SIZE,
        0,
    )
    tl.store(
        output + output_start + column_indices,
        cache_slot,
        mask=(column_indices < INPUT_WIDTH)
        & (output_start + column_indices < output_end)
        & (output_start + column_indices < OUTPUT_SIZE),
    )


def map_token_indices_to_slots(
    token_indices: torch.Tensor,
    request_indices: torch.Tensor,
    block_tables: torch.Tensor,
    output_indptr: torch.Tensor,
    output: torch.Tensor,
    block_size: int,
) -> None:
    """Convert request-local token indices to physical cache slots."""
    grid = (
        token_indices.shape[0],
        triton.cdiv(token_indices.shape[1], 128),
    )
    _map_token_indices_to_slots_kernel[grid](
        token_indices,
        request_indices,
        block_tables,
        output_indptr,
        output,
        INPUT_WIDTH=token_indices.shape[1],
        BLOCK_TABLE_STRIDE=block_tables.stride(0),
        BLOCK_SIZE=block_size,
        OUTPUT_SIZE=output.numel(),
        BLOCK_WIDTH=128,
    )


def run_speculative_kpool_indexer(
    metadata,
    kv_cache: torch.Tensor,
    queries: torch.Tensor,
    keys: torch.Tensor,
    gates: torch.Tensor,
    weights: torch.Tensor,
    pool_bias: torch.Tensor,
    history: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    positions: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    pool_size: int,
    topk_tokens: int,
    output_width: int,
    block_size: int,
    max_model_len: int,
    scale_fmt: str,
    stable_topk: bool,
) -> None:
    """Run the pooled indexer for a speculative verification batch."""
    from aiter.ops.cache import indexer_k_quant_and_cache
    from aiter.ops.topk import top_k_per_row_decode
    from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    cu_seqlens_q = metadata.cu_seqlens_q
    num_query_tokens, num_heads, head_dim = queries.shape
    pooled_keys, request_indices = build_speculative_pool_candidates(
        history,
        keys,
        gates,
        positions,
        cu_seqlens_q,
        source_slots,
        pool_bias,
        pool_size,
    )
    closes_pool = (positions % pool_size == pool_size - 1) & (
        destination_slots[request_indices] >= 0
    )
    pool_ids = torch.where(
        closes_pool,
        positions // pool_size,
        -1,
    ).to(torch.int64)
    pool_rows_per_block = block_size // pool_size
    cache_slots = kpool.pool_slot_mapping(
        metadata.block_tables,
        pool_ids,
        request_indices,
        pool_rows_per_block,
    )
    pooled_kv_cache = kv_cache.view(
        -1,
        pool_rows_per_block,
        kv_cache.shape[-1],
    )
    indexer_k_quant_and_cache(
        pooled_keys,
        pooled_kv_cache,
        cache_slots,
        head_dim,
        scale_fmt,
        preshuffle=True,
    )
    update_speculative_kpool_history(
        history,
        keys,
        gates,
        positions,
        cu_seqlens_q,
        source_slots,
        destination_slots,
    )
    if metadata.max_seqlen_k <= topk_tokens:
        return
    # A per-token paged scoring path also handles ragged verification. Chunking
    # bounds scratch memory during long prefills; optimize only after parity.
    selected = torch.full(
        (num_query_tokens, output_width),
        -1,
        device=keys.device,
        dtype=torch.int32,
    )
    max_pools = triton.cdiv(max_model_len, pool_size)
    for begin in range(0, num_query_tokens, 128):
        end = min(num_query_tokens, begin + 128)
        count = end - begin
        sequence_lengths = (positions[begin:end] + 1).to(torch.int32)
        pool_lengths = (sequence_lengths // pool_size).contiguous()
        logits = torch.empty(
            (count, max_pools),
            device=keys.device,
            dtype=torch.float32,
        )
        block_tables = metadata.block_tables[request_indices[begin:end]].contiguous()
        deepgemm_fp8_paged_mqa_logits(
            queries[begin:end].view(count, 1, num_heads, head_dim),
            pooled_kv_cache.unsqueeze(-2),
            weights[begin:end],
            logits,
            pool_lengths,
            block_tables,
            max_pools,
            KVBlockSize=pool_rows_per_block,
            Preshuffle=True,
        )
        selected_pools = torch.empty(
            (count, topk_tokens // pool_size),
            device=keys.device,
            dtype=torch.int32,
        )
        top_k_per_row_decode(
            logits,
            1,
            pool_lengths,
            selected_pools,
            count,
            logits.stride(0),
            logits.stride(1),
            k=topk_tokens // pool_size,
            stable=stable_topk,
        )
        kpool.expand_pools_and_append_tail(
            selected_pools,
            sequence_lengths,
            pool_size,
            out=selected[begin:end],
        )
    map_token_indices_to_slots(
        selected,
        request_indices,
        metadata.block_tables,
        metadata.sparse_kv_indptr,
        sparse_kv_indices,
        block_size,
    )
