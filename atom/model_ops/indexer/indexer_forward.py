# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One DSA indexer layer's forward, as the `sparse_attn_indexer` custom op.

It writes the step's index keys, scores its rows (`score_topk`) and hands the
selection to sparse MLA as KV slots: through the DCP candidate exchange on a
DCP decode, the owned-slot filter on a DCP prefill, a request-to-global
conversion otherwise. The `Indexer` module (models/deepseek_v2.py) owns the
projections and calls this op.
"""

import torch
from aiter import cp_gather_indexer_k_quant_cache, dtypes, indexer_k_quant_and_cache

from atom.config import get_current_atom_config
from atom.distributed.dcp_utils import get_dcp_group, get_dcp_rank, get_dcp_world_size
from atom.distributed.pcp_utils import pcp_is_enabled
from atom.model_ops.attention_mla import (
    indexer_qk_rope_quant_and_cache,
    triton_convert_req_index_to_global_index,
    triton_convert_req_index_to_global_index_dsa_prefill,
    triton_gather_kv_indices_sparse,
)
from atom.model_ops.dcp_ops import (
    dcp_decode_candidate_exchange_fused,
    triton_filter_and_convert_dcp_index_prefill,
)
from atom.model_ops.sparse_indexer_fp4 import FP4_QUANT_BLOCK_SIZE, fp4_q_scale_shape
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import get_forward_context

from .nope import indexer_nope_quant_and_cache
from .rows import IndexCache
from .score import score_topk


def _dcp_gather_indexer_k_prefill(
    kv_cache: torch.Tensor,
    prefill_metadata,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the full-sequence indexer K under DCP for prefill top-k.

    The index cache holds only this rank's round-robin 1/W, so a plain gather
    would return a 1/W-long prefix of garbage. Gather the local shard with LOCAL
    cu_seqlens (the gather's own slot formula, block_table[j//bs]*bs + j%bs, is
    exactly the round-robin write layout), all-gather it, then de-interleave back
    to global order -- the same local-shard + all-gather the decode path uses, on
    k rather than logits. Returns (k_fp8, k_scale) in global order.
    """
    local_total = prefill_metadata.dcp_indexer_local_total
    local_cu_seqlens = prefill_metadata.dcp_indexer_local_cu_seqlens
    # Cached prefill pads block tables to running_bs for later consumers,
    # while these local cums contain only the scheduled requests. AITER takes
    # its request count from the table height: passing the padded rows would
    # read beyond local_cu_seqlens and can gather another request's keys.
    block_tables = prefill_metadata.block_tables[: local_cu_seqlens.numel() - 1]
    k_fp8_local = torch.empty([local_total, head_dim], device=device, dtype=dtypes.fp8)
    k_scale_local = torch.empty([local_total, 1], device=device, dtype=torch.float32)
    cp_gather_indexer_k_quant_cache(
        kv_cache,
        k_fp8_local,
        k_scale_local.view(dtypes.fp8),
        block_tables,
        local_cu_seqlens,
        preshuffle=True,
    )
    dcp_group = get_dcp_group()
    # fp8 has no collective support; move the bytes as uint8.
    gathered_k = dcp_group.all_gather(k_fp8_local.view(torch.uint8), dim=0).view(
        dtypes.fp8
    )
    gathered_scale = dcp_group.all_gather(k_scale_local, dim=0)
    gather_index = prefill_metadata.dcp_indexer_gather_index
    k_fp8 = gathered_k.index_select(0, gather_index)
    k_scale = gathered_scale.index_select(0, gather_index)
    return k_fp8, k_scale


def _dcp_stage_indexer_fp4_prefill(
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    prefill_metadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage the whole key set's FP4 index planes into pages of this rank's own.

    Same three steps as `_dcp_gather_indexer_k_prefill` -- read the local shard,
    all-gather it, de-interleave back to global order -- on the two E2M1/e8m0
    planes instead of the one FP8 one. It ends in pages rather than a flat
    buffer because every FP4 mqa-logits kernel is paged: each request's keys
    from a page of their own, which `dcp_indexer_fp4_block_tables` lists as
    the request's own block table does its cache pages.

    The two planes disagree on their row axis, so both the read and the write
    bend through `fp4_index_scale_rows`; see it for what goes wrong otherwise.
    """
    # Built once per forward by the metadata builder.
    page = prefill_metadata.dcp_indexer_fp4_read_page
    row = prefill_metadata.dcp_indexer_fp4_read_row
    data = kv_cache[page, :, :, row, :]
    scale = kv_cache_scale[page, :, :, prefill_metadata.dcp_indexer_fp4_read_scale_row]

    dcp_group = get_dcp_group()
    gather_index = prefill_metadata.dcp_indexer_gather_index
    data = dcp_group.all_gather(data, dim=0).index_select(0, gather_index)
    scale = dcp_group.all_gather(scale, dim=0).index_select(0, gather_index)

    page = prefill_metadata.dcp_indexer_fp4_stage_page
    row = prefill_metadata.dcp_indexer_fp4_stage_row
    scale_row = prefill_metadata.dcp_indexer_fp4_stage_scale_row
    pages = prefill_metadata.dcp_indexer_fp4_stage_pages
    staged = kv_cache.new_zeros(pages, *kv_cache.shape[1:])
    staged[page, :, :, row, :] = data
    staged_scale = kv_cache_scale.new_zeros(pages, *kv_cache_scale.shape[1:])
    staged_scale[page, :, :, scale_row] = scale
    return staged, staged_scale


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    sparse_kv_indices_buffer: torch.Tensor,
    dcp_sparse_kv_indptr_buffer: torch.Tensor,
    dcp_owned_counts_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
    stable_topk: bool,
) -> None:
    # Room for the DCP prefill all-gather to land in place. It produces
    # `world_size * ceil(N_q / world_size)` rows, up to `world_size - 1` more
    # than there are prefill rows; giving the allocation that slack is what
    # lets the gather write straight into this tensor instead of through a
    # staging buffer. Those trailing rows are never read: the owned-slot filter
    # sizes its grid from `dsa_kv_indptr` (`num_tokens = shape[0] - 1` in
    # `triton_filter_and_convert_dcp_index_prefill`), not from this tensor.
    _dcp_gather_slack = max(get_dcp_world_size() - 1, 0)
    topk_indices = torch.empty(
        hidden_states.shape[0] + _dcp_gather_slack,
        topk_tokens,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    context = forward_context.context
    slot_mapping = attn_metadata.slot_mapping
    # Skip for dummy runs to avoid corrupting KV cache
    if forward_context.context.is_dummy_run:
        # dummy runner
        return
    # For MTP verify decode, max_seqlen_q > 1 so total decode tokens = batch_size * max_seqlen_q
    num_decode_tokens = (
        context.scheduled_bs * attn_metadata.max_seqlen_q
        if not context.is_prefill
        else 0
    )
    runner_block_size = get_current_atom_config().kv_cache_block_size
    cp_kv_cache_interleave_size = get_current_atom_config().dcp_config.interleave_size
    # Which width this indexer's cache is in -- the one thing the arguments
    # cannot say, and a thing the traced graph piece already committed to.
    fwd_ctx = get_current_atom_config().compilation_config.static_forward_context
    # A custom op cannot take a module, so the Indexer is reached by undoing the
    # one name `Indexer.__init__` builds: `prefix` + ".k_cache". Both spellings
    # have to move together; a miss is a KeyError here, at the first forward.
    indexer_module = fwd_ctx[k_cache_prefix.rsplit(".k_cache", 1)[0]]
    indexer_fp4 = indexer_module._indexer_fp4
    q_fp4_scale = None
    if not indexer_fp4:
        kv_cache = kv_cache.view(-1, runner_block_size, kv_cache.shape[-1])
    # PCP prefill: `k` (and `positions`) arrive as the full PADDED key set
    # [S_pad] produced by an all-gather of the round-robin shards. The KV-cache
    # write (driven by slot_mapping) uses the real token count S_real ==
    # slot_mapping length, so trim the round-robin padding here. Gated on the
    # same condition as the caller's PCP path (a sparse prefill); no-op
    # otherwise.
    if pcp_is_enabled() and context.is_prefill and attn_metadata.run_sparse:
        n_real = slot_mapping.shape[0]
        k = k[:n_real]
        if positions is not None:
            positions = positions[:n_real]
    if indexer_fp4:
        # `weights_out` comes back unscaled in q's dtype: the FP4 mqa-logits
        # kernels apply the per-group q scale inside the MFMA and take
        # `weights_scale` as their own fp32 scalar, so either folded in here
        # would be applied twice. `preshuffle` has no FP4 spelling.
        q_bf16 = q_input
        q_quant = torch.empty(
            (*q_bf16.shape[:-1], head_dim // 2), device=q_bf16.device, dtype=torch.uint8
        )
        q_fp4_scale = torch.empty(
            fp4_q_scale_shape(q_bf16.shape[0], q_bf16.shape[1], head_dim),
            device=q_bf16.device,
            dtype=torch.uint8,
        )
        # AITER requires contiguous [rows, heads] output. Non-DCP skips
        # slot<0 rows, whose scorer bounds are empty; DCP computes every row.
        weights_mqa = torch.empty(
            weights.shape,
            dtype=weights.dtype,
            device=weights.device,
        )
        indexer_qk_rope_quant_and_cache(
            q_bf16,
            q_quant,
            weights,
            weights_mqa,
            k,
            kv_cache,
            slot_mapping,
            k_norm_weight,
            k_norm_bias,
            positions,
            cos_cache,
            sin_cache,
            k_norm_eps,
            FP4_QUANT_BLOCK_SIZE,
            scale_fmt,
            weights_scale,
            is_neox=is_neox_style,
            q_scale_out=q_fp4_scale,
            kv_cache_scale=indexer_module.k_cache.kv_cache_scale,
        )
    elif indexer_module.use_nope_cache_fusion:
        q_quant, weights = indexer_nope_quant_and_cache(
            q_input,
            k,
            weights,
            kv_cache.view(torch.uint8),
            slot_mapping,
            k_norm_weight,
            k_norm_bias,
            k_norm_eps,
            weights_scale,
        )
    elif use_qk_rope_cache_fusion:
        q_bf16 = q_input
        q_quant = torch.empty_like(q_bf16, dtype=dtypes.fp8)
        weights_out = torch.empty(
            weights.shape, device=weights.device, dtype=torch.float32
        )
        indexer_qk_rope_quant_and_cache(
            q_bf16,
            q_quant,
            weights,
            weights_out,
            k,
            kv_cache,
            slot_mapping,
            k_norm_weight,
            k_norm_bias,
            positions,
            cos_cache,
            sin_cache,
            k_norm_eps,
            quant_block_size,
            scale_fmt,
            weights_scale,
            preshuffle=True,
            is_neox=is_neox_style,
        )
        weights = weights_out
    else:
        q_quant = q_input
        indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
            preshuffle=True,
        )
    if not indexer_fp4:
        weights_mqa = weights
    if context.is_prefill and not attn_metadata.run_sparse:
        # A dense prefill reads every key: there is nothing to select.
        return
    if not context.is_prefill and get_dcp_world_size() > 1:
        # The fused exchange scores this rank's own shard and writes the KV
        # slots it owns -- ownership filter, slot localize and compaction all
        # inside the op -- straight into sparse_kv_indices_buffer and
        # dcp_sparse_kv_indptr_buffer, so nothing below is left to do.
        dcp_decode_candidate_exchange_fused(
            attn_metadata,
            q_quant[:num_decode_tokens].reshape(
                context.scheduled_bs, -1, *q_quant.shape[1:]
            ),
            kv_cache if indexer_fp4 else kv_cache.unsqueeze(-2),
            weights_mqa,
            get_dcp_rank(),
            num_decode_tokens,
            topk_tokens,
            max_model_len,
            runner_block_size,
            stable_topk,
            cp_kv_cache_interleave_size,
            out_kv_indices=sparse_kv_indices_buffer,
            out_kv_indptr=dcp_sparse_kv_indptr_buffer,
            owned_counts=dcp_owned_counts_buffer,
            q_scale=q_fp4_scale,
            kv_scale=indexer_module.k_cache.kv_cache_scale if indexer_fp4 else None,
            weights_scale=weights_scale,
        )
        return
    rows = attn_metadata.index_rows
    # A single-rank prefill's slot conversion subtracts concat rows' request
    # base itself; decode and the DCP filter read request-local ids.
    single_rank_prefill = context.is_prefill and get_dcp_world_size() == 1
    topk_indices = score_topk(
        ((q_quant.unsqueeze(1), q_fp4_scale.unsqueeze(1)) if indexer_fp4 else q_quant),
        weights_mqa,
        _index_cache(indexer_module, kv_cache, attn_metadata, context.is_prefill),
        rows,
        topk=topk_tokens,
        weight_scale=weights_scale,
        stable=stable_topk,
        out=topk_indices[: rows.out_rows],
        keys=(
            _dcp_gather_indexer_k_prefill(kv_cache, attn_metadata, head_dim, k.device)
            if rows.concat is not None and rows.split.world > 1
            else None
        ),
        request_local=not single_rank_prefill,
    )
    if not context.is_prefill:
        if attn_metadata.max_seqlen_q > 1:
            triton_gather_kv_indices_sparse(
                attn_metadata.sparse_kv_indptr,
                attn_metadata.batch_id_per_q_token,
                topk_indices,
                attn_metadata.kv_indices,
                attn_metadata.kv_indptr,
                NUM_TOPK_TOKENS=topk_tokens,
                out=sparse_kv_indices_buffer,
            )
        else:
            triton_convert_req_index_to_global_index(
                attn_metadata.cu_seqlens_q,
                attn_metadata.kv_indptr,
                attn_metadata.sparse_kv_indptr,
                attn_metadata.kv_indices,
                topk_indices,
                NUM_TOPK_TOKENS=topk_tokens,
                out=sparse_kv_indices_buffer,
            )
    elif get_dcp_world_size() > 1:
        # Keep only the positions this rank owns, mapped through the
        # round-robin virtual-block layout and compacted to the front -- the
        # decode exchange's treatment, a query token for a request. Handed the
        # WHOLE selection: the filter `.contiguous()`s what it gets, so a row
        # slice would copy what the in-place gather avoided moving, and its
        # grid follows `sparse_kv_indptr`, so the gather's tail is never read.
        triton_filter_and_convert_dcp_index_prefill(
            attn_metadata.sparse_kv_indptr,
            attn_metadata.batch_id_per_q_token,
            topk_indices,
            attn_metadata.block_tables,
            get_dcp_rank(),
            get_dcp_world_size(),
            runner_block_size,
            out_kv_indptr=dcp_sparse_kv_indptr_buffer,
            owned_counts=dcp_owned_counts_buffer,
            NUM_TOPK_TOKENS=topk_tokens,
            out=sparse_kv_indices_buffer,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
    else:
        triton_convert_req_index_to_global_index_dsa_prefill(
            attn_metadata.sparse_cu_seqlens_q,
            attn_metadata.sparse_kv_indptr,
            attn_metadata.batch_id_per_q_token,
            topk_indices,
            attn_metadata.block_tables,
            attn_metadata.cu_seqlens_k,
            NUM_TOPK_TOKENS=topk_tokens,
            PAGE_SIZE=runner_block_size,
            out=sparse_kv_indices_buffer,
            row_starts=None if rows.concat is None else rows.concat.row_starts,
        )


def _index_cache(indexer, kv_cache, attn_metadata, is_prefill) -> IndexCache:
    """The keys an indexer layer scores against: its own index cache -- but
    under DCP a prefill's FP4 keys from every rank's shard, staged."""
    block = get_current_atom_config().kv_cache_block_size
    if not indexer._indexer_fp4:
        return IndexCache(kv_cache, None, block)
    scales = indexer.k_cache.kv_cache_scale
    if is_prefill and get_dcp_world_size() > 1:
        kv_cache, scales = _dcp_stage_indexer_fp4_prefill(
            kv_cache, scales, attn_metadata
        )
    return IndexCache(kv_cache, scales, block)


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=[
        "kv_cache",
        "sparse_kv_indices_buffer",
        "dcp_sparse_kv_indptr_buffer",
        "dcp_owned_counts_buffer",
    ],
    fake_impl=lambda *_args, **_kwargs: None,
)
