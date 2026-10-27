# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged pooled sparse-indexer dispatch for GLM-5.3-Flash."""

from __future__ import annotations

import torch
from aiter.ops.cache import (
    indexer_k_quant_and_cache,
)

from atom.config import get_current_atom_config
from atom.distributed.dcp_utils import get_dcp_world_size
from atom.distributed.pcp_utils import pcp_is_enabled
from atom.model_ops.attention_mla import (
    triton_convert_req_index_to_global_index,
    triton_convert_req_index_to_global_index_dsa_prefill,
)
from atom.model_ops.indexer import IndexCache, score_topk
from atom.utils import forward_context
from atom.utils.custom_register import direct_register_custom_op

from . import kpool
from .geometry import speculative_verify_enabled


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
        safe_slots = read_slots[req_idx].clamp_min(0)[:, None]
        # The tail holds position `p` at row `p % ROWS`; ROWS exceeds the pool
        # size when it doubles as the speculative history ring. Index the
        # [n, POOL] rows directly: `tail_cache[safe_slots]` would copy each
        # token's whole [2, ROWS, D] ring first.
        stash_rows = abs_slot.clamp_min(0) % tail_cache.shape[-2]
        pool_k = torch.where(
            from_tail[..., None], tail_cache[safe_slots, 0, stash_rows], pool_k
        )
        pool_gate = torch.where(
            from_tail[..., None], tail_cache[safe_slots, 1, stash_rows], pool_gate
        )

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
    mtp: bool,
    topk_out_width: int,
    scale_fmt: str,
    stable_topk: bool,
) -> torch.Tensor:
    """Run pooled sparse top-k and update its cache/index side buffers.

    `mtp`: an MTP layer, whose `positions` run one behind the step's, so its
    prefill rows close their pools a token later (`mtp_index_rows`)."""
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
    # The explicit scheduler field lives on the nested GDN metadata today. The
    # ragged-query condition remains a fallback for other metadata builders.
    gdn_metadata = getattr(attn_metadata, "gdn_metadata", None)
    is_speculative_verify = speculative_verify_enabled(
        is_prefill=context.is_prefill,
        num_spec_decodes=getattr(gdn_metadata, "num_spec_decodes", 0),
        max_seqlen_q=attn_metadata.max_seqlen_q,
    )
    if is_speculative_verify:
        from .speculative import run_speculative_kpool_indexer

        run_speculative_kpool_indexer(
            attn_metadata,
            attn_metadata.mtp_index_rows if mtp else attn_metadata.index_rows,
            kv_cache,
            q_fp8,
            k,
            gate_score,
            weights,
            compress_ape,
            tail_cache,
            state_slot_idx_in,
            state_slot_idx,
            positions,
            sparse_kv_indices_buffer,
            index_kpool,
            topk_tokens,
            topk_out_width,
            get_current_atom_config().kv_cache_block_size,
            scale_fmt,
            stable_topk,
        )
        return result

    device = hidden_states.device
    block_size = get_current_atom_config().kv_cache_block_size
    pool_rows = block_size // index_kpool
    kv_cache = kv_cache.view(-1, pool_rows, kv_cache.shape[-1])
    pool_bt = attn_metadata.block_tables
    select_k = topk_tokens // index_kpool
    n_tokens = hidden_states.shape[0]
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
        # The builder publishes no rows below index_topk (every pool is then
        # selected and prefill runs dense) or before any request closed a pool.
        rows = attn_metadata.mtp_index_rows if mtp else attn_metadata.index_rows
        if rows is None:
            return result
        pool_topk = score_topk(
            q_fp8,
            weights,
            IndexCache(kv_cache, None, pool_rows),
            rows,
            topk=select_k,
            stable=stable_topk,
        )
        kpool.expand_pools_and_append_tail(
            pool_topk,
            positions.to(torch.int32) + 1,
            index_kpool,
            out=topk_indices,
        )
        triton_convert_req_index_to_global_index_dsa_prefill(
            attn_metadata.sparse_cu_seqlens_q,
            attn_metadata.sparse_kv_indptr,
            attn_metadata.batch_id_per_q_token,
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

    pool_topk = score_topk(
        q_fp8[:bs],
        weights[:bs],
        IndexCache(kv_cache, None, pool_rows),
        attn_metadata.index_rows,
        topk=select_k,
        stable=stable_topk,
    )
    kpool.expand_pools_and_append_tail(
        pool_topk,
        attn_metadata.context_lens[:bs].to(torch.int32),
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
    mtp: bool,
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
