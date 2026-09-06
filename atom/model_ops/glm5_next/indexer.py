# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged pooled sparse-indexer dispatch for GLM-5.3-Flash."""

from __future__ import annotations

import torch
from aiter import dtypes
from aiter.ops.cache import (
    cp_gather_indexer_k_quant_cache,
    indexer_k_quant_and_cache,
)
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from atom.config import get_current_atom_config
from atom.distributed.dcp_utils import get_dcp_world_size
from atom.distributed.pcp_utils import pcp_is_enabled
from atom.model_ops.attention_mla import (
    triton_convert_req_index_to_global_index,
    triton_convert_req_index_to_global_index_dsa_prefill,
)
from atom.utils import envs, forward_context
from atom.utils.custom_register import direct_register_custom_op

from . import kpool


def _kpool_request_index(cu_seqlens_q: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Return the request id for every token in a flat prefill batch."""
    counts = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(counts.shape[0], device=cu_seqlens_q.device, dtype=torch.int64),
        counts,
        output_size=n_tokens,
    )


def _kpool_write_completed_pools(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    positions: torch.Tensor,
    pool_bt: torch.Tensor,
    req_idx: torch.Tensor,
    compress_ape: torch.Tensor,
    index_kpool: int,
    head_dim: int,
    scale_fmt: str,
    pool_rows: int,
    chunk_start: torch.Tensor | None = None,
    tail_cache: torch.Tensor | None = None,
    state_slot_idx_in: torch.Tensor | None = None,
    state_slot_idx: torch.Tensor | None = None,
) -> None:
    """Compress and publish pools that close in this prefill batch."""
    n = k.shape[0]
    if n == 0:
        return

    rows = torch.arange(n, device=k.device)
    offsets = torch.arange(index_kpool, device=k.device)
    gather_idx = (
        (rows - (index_kpool - 1)).clamp_min(0)[:, None] + offsets[None, :]
    ).clamp_max(n - 1)
    pool_k, pool_gate = k[gather_idx], gate_score[gather_idx]

    if chunk_start is not None:
        abs_slot = (
            positions.to(torch.int64)[:, None] - (index_kpool - 1) + offsets[None, :]
        )
        from_tail = abs_slot < chunk_start[req_idx][:, None]
        read_slots = state_slot_idx if state_slot_idx_in is None else state_slot_idx_in
        safe_slots = read_slots[req_idx].clamp_min(0)
        stash = tail_cache[safe_slots]
        pool_k = torch.where(from_tail[..., None], stash[:, 0], pool_k)
        pool_gate = torch.where(from_tail[..., None], stash[:, 1], pool_gate)

    pooled = kpool.pool_and_rotate(pool_k, pool_gate, compress_ape)
    abs_pos = positions.to(torch.int64)
    closes = abs_pos % index_kpool == index_kpool - 1
    if state_slot_idx is not None:
        closes &= state_slot_idx[req_idx] >= 0
    slots = kpool.pool_slot_mapping(
        pool_bt,
        torch.where(
            closes,
            abs_pos // index_kpool,
            torch.full_like(abs_pos, -1),
        ),
        req_idx,
        pool_rows,
    )
    indexer_k_quant_and_cache(
        pooled,
        kv_cache,
        slots,
        head_dim,
        scale_fmt,
        preshuffle=True,
    )


def _kpool_pool_counts(seq_lens_k: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Return complete-pool counts; incomplete tails are not cached."""
    return seq_lens_k.to(torch.int64) // pool_size


def _sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    weights: torch.Tensor,
    compress_ape: torch.Tensor,
    tail_cache: torch.Tensor,
    state_slot_idx_in: torch.Tensor,
    state_slot_idx: torch.Tensor,
    positions: torch.Tensor,
    sparse_kv_indices_buffer: torch.Tensor,
    topk_tokens: int,
    index_kpool: int,
    head_dim: int,
    max_model_len: int,
    topk_out_width: int,
    scale_fmt: str,
    stable_topk: bool,
) -> torch.Tensor:
    """Run pooled sparse top-k and update its cache/index side buffers."""
    fwd = forward_context.get_forward_context()
    attn_metadata = fwd.attn_metadata
    context = fwd.context
    result = weights.to(dtype=torch.float32, copy=True)
    if context.is_dummy_run:
        return result

    if get_dcp_world_size() > 1 or pcp_is_enabled():
        raise NotImplementedError(
            "GLM-5.3 kpool does not support DCP/PCP; use dcp=pcp=1"
        )
    if not context.is_prefill and attn_metadata.max_seqlen_q > 1:
        raise NotImplementedError("GLM-5.3 kpool does not support speculative decode")

    device = hidden_states.device
    block_size = get_current_atom_config().kv_cache_block_size
    pool_rows = block_size // index_kpool
    kv_cache = kv_cache.view(-1, pool_rows, kv_cache.shape[-1])
    pool_bt = attn_metadata.block_tables
    select_k = topk_tokens // index_kpool
    n_tokens = hidden_states.shape[0]
    n_head = q_fp8.shape[1]
    topk_indices = torch.full(
        (n_tokens, topk_out_width),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if context.is_prefill:
        cu_q = attn_metadata.cu_seqlens_q
        req_idx = _kpool_request_index(cu_q, n_tokens)
        chunk_start = None
        if attn_metadata.has_cached:
            cu_k = attn_metadata.cu_seqlens_k
            nreq = cu_q.shape[0] - 1
            chunk_start = (cu_k[1 : nreq + 1] - cu_k[:nreq]).to(torch.int64) - (
                cu_q[1:] - cu_q[:-1]
            ).to(torch.int64)

        _kpool_write_completed_pools(
            kv_cache,
            k,
            gate_score,
            positions,
            pool_bt,
            req_idx,
            compress_ape,
            index_kpool,
            head_dim,
            scale_fmt,
            pool_rows,
            chunk_start=chunk_start,
            tail_cache=tail_cache,
            state_slot_idx_in=state_slot_idx_in,
            state_slot_idx=state_slot_idx,
        )
        kpool.kpool_seed_tail(
            tail_cache,
            k,
            gate_score,
            positions,
            cu_q,
            state_slot_idx,
            index_kpool,
            slot_idx_in=state_slot_idx_in,
        )
        if attn_metadata.max_seqlen_k <= topk_tokens:
            return result

        bs = cu_q.shape[0] - 1
        if attn_metadata.has_cached:
            cu_k = attn_metadata.cu_seqlens_k
            seq_lens_k = (cu_k[1 : bs + 1] - cu_k[:bs]).to(torch.int64)
        else:
            seq_lens_k = (cu_q[1:] - cu_q[:-1]).to(torch.int64)
        pool_counts = _kpool_pool_counts(seq_lens_k, index_kpool)
        pool_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        pool_cu[1:] = torch.cumsum(pool_counts, 0).to(torch.int32)

        max_pools = attn_metadata.kpool_total_pools
        if max_pools is None:
            raise RuntimeError("pooled prefill metadata is missing kpool_total_pools")
        if max_pools <= 0:
            return result
        k_fp8 = torch.empty((max_pools, head_dim), device=device, dtype=dtypes.fp8)
        k_scale = torch.empty((max_pools, 1), device=device, dtype=torch.float32)
        cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_fp8,
            k_scale.view(dtypes.fp8),
            pool_bt,
            pool_cu,
            preshuffle=True,
        )

        pool_ks = pool_cu.to(torch.int64)[req_idx]
        pool_ke = pool_ks + (positions.to(torch.int64) + 1) // index_kpool
        pool_ks = pool_ks.to(torch.int32)
        pool_ke = pool_ke.to(torch.int32)
        pool_topk = torch.empty((n_tokens, select_k), dtype=torch.int32, device=device)

        budget_bytes = envs.ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB * 1024 * 1024
        if budget_bytes > 0 and budget_bytes // (max_pools * 4) < n_tokens:
            budget_rows = budget_bytes // (max_pools * 4)
            chunk_rows = (
                (budget_rows // 128) * 128
                if budget_rows >= 128
                else 1 << (max(1, budget_rows).bit_length() - 1)
            )
        else:
            chunk_rows = n_tokens

        for start in range(0, n_tokens, chunk_rows):
            end = min(start + chunk_rows, n_tokens)
            row_starts = pool_ks[start:end]
            row_ends = pool_ke[start:end]
            logits = fp8_mqa_logits(
                Q=q_fp8[start:end],
                KV=k_fp8,
                kv_scales=k_scale.squeeze(-1).contiguous(),
                weights=weights[start:end],
                cu_starts=row_starts,
                cu_ends=row_ends,
            )
            top_k_per_row_prefill(
                logits=logits,
                rowStarts=row_starts,
                rowEnds=row_ends,
                indices=pool_topk[start:end],
                values=None,
                numRows=end - start,
                stride0=logits.stride(0),
                stride1=logits.stride(1),
                k=select_k,
                stable=stable_topk,
            )

        kpool.expand_pools_and_append_tail(
            pool_topk,
            positions.to(torch.int32) + 1,
            index_kpool,
            out=topk_indices,
            pool_base=pool_ks,
            tok_base=attn_metadata.cu_seqlens_k.to(torch.int32)[req_idx],
        )
        triton_convert_req_index_to_global_index_dsa_prefill(
            attn_metadata.sparse_cu_seqlens_q,
            attn_metadata.sparse_kv_indptr,
            attn_metadata.token_to_seq_idxs,
            topk_indices,
            attn_metadata.block_tables,
            attn_metadata.cu_seqlens_k,
            PAGE_SIZE=block_size,
            NUM_TOPK_TOKENS=topk_out_width,
            BLOCK_N=128,
            out=sparse_kv_indices_buffer,
        )
        return result

    bs = context.scheduled_bs
    pos = positions[:bs].to(torch.int64)
    pooled = kpool.kpool_decode_stash_and_pool(
        tail_cache,
        k[:bs],
        gate_score[:bs],
        positions[:bs],
        state_slot_idx,
        compress_ape,
        index_kpool,
        slot_idx_in=state_slot_idx_in,
    )
    closes = ((pos % index_kpool) == (index_kpool - 1)) & (state_slot_idx[:bs] >= 0)
    pool_ids = torch.where(
        closes,
        pos // index_kpool,
        torch.full_like(pos, -1),
    )
    slots = kpool.pool_slot_mapping(
        pool_bt,
        pool_ids,
        torch.arange(bs, device=device, dtype=torch.int64),
        pool_rows,
    )
    indexer_k_quant_and_cache(
        pooled,
        kv_cache,
        slots,
        head_dim,
        scale_fmt,
        preshuffle=True,
    )

    seq_lens = attn_metadata.context_lens[:bs]
    pool_ctx = (seq_lens.to(torch.int32) // index_kpool).contiguous()
    pool_max_len = -(-max_model_len // index_kpool)
    logits = torch.empty((bs, pool_max_len), dtype=torch.float32, device=device)
    deepgemm_fp8_paged_mqa_logits(
        q_fp8[:bs].view(bs, 1, n_head, head_dim),
        kv_cache.unsqueeze(-2),
        weights[:bs],
        logits,
        pool_ctx,
        pool_bt,
        pool_max_len,
        KVBlockSize=pool_rows,
        Preshuffle=True,
    )
    pool_topk = torch.empty((bs, select_k), dtype=torch.int32, device=device)
    top_k_per_row_decode(
        logits,
        1,
        pool_ctx,
        pool_topk,
        bs,
        logits.stride(0),
        logits.stride(1),
        k=select_k,
        stable=stable_topk,
    )
    kpool.expand_pools_and_append_tail(
        pool_topk,
        seq_lens.to(torch.int32),
        index_kpool,
        out=topk_indices[:bs],
    )
    triton_convert_req_index_to_global_index(
        attn_metadata.cu_seqlens_q,
        attn_metadata.kv_indptr,
        attn_metadata.sparse_kv_indptr,
        attn_metadata.kv_indices,
        topk_indices,
        NUM_TOPK_TOKENS=topk_out_width,
        out=sparse_kv_indices_buffer,
    )
    return result


def _sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    weights: torch.Tensor,
    compress_ape: torch.Tensor,
    tail_cache: torch.Tensor,
    state_slot_idx_in: torch.Tensor,
    state_slot_idx: torch.Tensor,
    positions: torch.Tensor,
    sparse_kv_indices_buffer: torch.Tensor,
    topk_tokens: int,
    index_kpool: int,
    head_dim: int,
    max_model_len: int,
    topk_out_width: int,
    scale_fmt: str,
    stable_topk: bool,
) -> torch.Tensor:
    return torch.empty_like(weights, dtype=torch.float32)


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool",
    op_func=_sparse_attn_indexer_kpool,
    mutates_args=["sparse_kv_indices_buffer", "tail_cache", "kv_cache"],
    fake_impl=_sparse_attn_indexer_kpool_fake,
)
