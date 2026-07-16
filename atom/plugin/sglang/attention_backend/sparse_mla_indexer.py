# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang plugin sparse MLA indexer support for DeepSeek-V3.2."""

from __future__ import annotations

import re
from typing import Optional

import torch
from aiter import (
    cp_gather_indexer_k_quant_cache,
    dtypes,
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
    indexer_k_quant_and_cache,
    indexer_qk_rope_quant_and_cache,
    top_k_per_row_decode,
    top_k_per_row_prefill,
)
from aiter.mla import mla_decode_fwd
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
import triton
import triton.language as tl

from atom.utils.custom_register import direct_register_custom_op


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    cu_seqlens_ptr,
    out_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)
    seq_start = tl.load(cu_seqlens_ptr + token_id)
    seq_end = tl.load(cu_seqlens_ptr + token_id + 1)
    if tile_id * BLOCK_N + seq_start >= seq_end:
        return

    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)
    is_invalid_tok = tok < 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    out_val = tl.where(
        is_invalid_tok | (~valid_block), 0, base * BLOCK_SIZE + inblock_off
    )
    out_ptr_ij = out_ptr + seq_start + indice_id
    out_ptr_ij_mask = (seq_start + indice_id) < seq_end
    tl.store(out_ptr_ij, out_val, mask=out_ptr_ij_mask)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    num_tokens = req_id.shape[0]
    _, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()

    grid = (num_tokens, tiles_per_row)
    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        cu_seqlens,
        paged_kv_indices,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
    )


def _parse_layer_id_from_indexer_prefix(prefix: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", prefix)
    if match is None:
        raise RuntimeError(
            f"Cannot infer DeepSeek-V3.2 indexer layer id from prefix: {prefix!r}"
        )
    return int(match.group(1))


def _build_sglang_query_ranges(forward_batch) -> tuple[torch.Tensor, torch.Tensor]:
    device = forward_batch.seq_lens.device
    if forward_batch.forward_mode.is_decode_or_idle():
        bs = int(forward_batch.batch_size)
        starts = torch.zeros(bs, dtype=torch.int32, device=device)
        ends = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
        return starts, ends

    query_lens = getattr(forward_batch, "extend_seq_lens", None)
    if query_lens is None:
        query_lens = forward_batch.seq_lens
    query_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if query_lens_cpu is None:
        query_lens_cpu = query_lens.detach().cpu()
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.detach().cpu()

    starts = []
    ends = []
    kv_offset = 0
    for q_len_raw, seq_len_raw in zip(query_lens_cpu, seq_lens_cpu):
        q_len = int(q_len_raw)
        seq_len = int(seq_len_raw)
        prefix_len = seq_len - q_len
        starts.extend([kv_offset] * q_len)
        ends.extend(kv_offset + prefix_len + i + 1 for i in range(q_len))
        kv_offset += seq_len

    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(ends, dtype=torch.int32, device=device),
    )


def _build_sglang_block_table(forward_batch, page_size: int) -> torch.Tensor:
    req_pool_indices = forward_batch.req_pool_indices
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    token_table = req_to_token[req_pool_indices, :]
    if not forward_batch.forward_mode.is_decode_or_idle():
        token_table = token_table.clone()
        query_lens = getattr(forward_batch, "extend_seq_lens", None)
        if query_lens is None:
            query_lens = forward_batch.seq_lens
        prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix_lens is None:
            prefix_lens = forward_batch.seq_lens - query_lens
        query_lens_cpu = query_lens[: int(forward_batch.batch_size)].detach().cpu()
        prefix_lens_cpu = prefix_lens[: int(forward_batch.batch_size)].detach().cpu()
        offset = 0
        for req_idx, (prefix_len_raw, query_len_raw) in enumerate(
            zip(prefix_lens_cpu, query_lens_cpu)
        ):
            prefix_len = int(prefix_len_raw)
            query_len = int(query_len_raw)
            if query_len > 0:
                token_table[req_idx, prefix_len : prefix_len + query_len] = (
                    forward_batch.out_cache_loc[offset : offset + query_len]
                )
            offset += query_len
    if page_size == 1:
        return token_table
    return token_table[:, ::page_size] // page_size


def _build_sparse_req_id_per_token_for_sglang(
    forward_batch,
    device: torch.device,
) -> torch.Tensor:
    bs = int(forward_batch.batch_size)
    req_ids = torch.arange(bs, dtype=torch.int32, device=device)
    if forward_batch.forward_mode.is_decode_or_idle():
        return req_ids
    query_lens = getattr(forward_batch, "extend_seq_lens", None)
    if query_lens is None:
        query_lens = forward_batch.seq_lens
    return torch.repeat_interleave(req_ids, query_lens[:bs].to(torch.int32))


def _supports_sparse_mla_fast_metadata(
    nhead: int,
    *,
    max_seqlen_qo: int,
    uni_seqlen_qo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> bool:
    """Whether AITER get_mla_metadata_v1 supports this sparse MLA shape."""
    if nhead in (16, 64, 128):
        return True
    if uni_seqlen_qo == 1 and nhead % 16 == 0 and 2 <= nhead // 16 < 8:
        return True
    if nhead == 8 and max_seqlen_qo == 4:
        return (q_dtype == dtypes.fp8 and kv_dtype == dtypes.fp8) or (
            q_dtype == dtypes.bf16 and kv_dtype == dtypes.bf16
        )
    return False


def forward_sparse_mla_for_sglang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer,
    forward_batch,
    topk_indices: torch.Tensor,
    save_kv_cache: bool = True,
    input_dtype: Optional[torch.dtype] = None,
    q_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ATOM sparse MLA path for SGLang DeepSeek-V3.2."""
    if save_kv_cache and k is not None:
        assert v is not None
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    num_tokens = q.shape[0]
    topk_indices = topk_indices[:num_tokens]
    topk_tokens = topk_indices.shape[1]
    page_size = int(getattr(forward_batch.token_to_kv_pool, "page_size", 1))

    req_id_per_token = _build_sparse_req_id_per_token_for_sglang(
        forward_batch, q.device
    )
    block_table = _build_sglang_block_table(forward_batch, page_size).to(
        dtype=torch.int32
    )
    output_dtype = input_dtype or torch.bfloat16
    o = q.new_empty(
        (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
        dtype=output_dtype,
    )
    k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    q_descale = (
        (q_scale if q_scale is not None else getattr(layer, "q_scale", None))
        if q.dtype == dtypes.fp8
        else None
    )
    fp8_sparse_mla = q.dtype == dtypes.fp8 or k_buffer.dtype == dtypes.fp8

    seq_len = (topk_indices != -1).sum(dim=-1).to(dtype=torch.int32)
    paged_kv_indptr = torch.empty((num_tokens + 1,), dtype=torch.int32, device=q.device)
    paged_kv_indptr[0].zero_()
    torch.cumsum(seq_len, dim=0, out=paged_kv_indptr[1:])
    paged_kv_indices = torch.empty(
        (num_tokens * topk_tokens,), dtype=torch.int32, device=q.device
    )
    triton_convert_req_index_to_global_index(
        req_id_per_token,
        block_table,
        topk_indices.to(dtype=torch.int32),
        paged_kv_indptr,
        paged_kv_indices,
        BLOCK_SIZE=page_size,
        NUM_TOPK_TOKENS=topk_tokens,
    )

    qo_indptr = torch.arange(num_tokens + 1, dtype=torch.int32, device=q.device)
    last_page_len = torch.ones(num_tokens, dtype=torch.int32, device=q.device)

    work_metadata = None
    work_indptr = None
    work_info_set = None
    reduce_indptr = None
    reduce_final_map = None
    reduce_partial_map = None

    max_seqlen_qo = 1
    uni_seqlen_qo = 1
    use_fast_metadata = fp8_sparse_mla and _supports_sparse_mla_fast_metadata(
        layer.tp_q_head_num,
        max_seqlen_qo=max_seqlen_qo,
        uni_seqlen_qo=uni_seqlen_qo,
        q_dtype=q.dtype,
        kv_dtype=k_buffer.dtype,
    )

    if use_fast_metadata:
        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_set_size, work_info_set_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = get_mla_metadata_info_v1(
            num_tokens,
            1,
            layer.tp_q_head_num,
            q.dtype,
            k_buffer.dtype,
            is_sparse=True,
            fast_mode=True,
        )
        work_metadata = torch.empty(
            work_metadata_size, dtype=work_metadata_dtype, device=q.device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_dtype, device=q.device
        )
        work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_dtype, device=q.device
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_dtype, device=q.device
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_dtype, device=q.device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_dtype, device=q.device
        )
        get_mla_metadata_v1(
            qo_indptr,
            paged_kv_indptr,
            last_page_len,
            layer.tp_q_head_num,
            1,
            True,
            work_metadata,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=16,
            page_size=1,
            max_seqlen_qo=max_seqlen_qo,
            uni_seqlen_qo=uni_seqlen_qo,
            fast_mode=True,
            dtype_q=q.dtype,
            dtype_kv=k_buffer.dtype,
        )

    mla_decode_fwd(
        q,
        k_buffer.view(-1, 1, 1, layer.qk_head_dim),
        o,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        last_page_len,
        1,
        sm_scale=layer.scaling,
        logit_cap=layer.logit_cap,
        q_scale=q_descale,
        kv_scale=layer.k_scale,
        page_size=1,
        work_meta_data=work_metadata,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )
    return o.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)


def sparse_attn_indexer_sglang_plugin_mode(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    from atom.plugin.sglang.models.base_model_wrapper import get_current_forward_batch

    del kv_cache, total_seq_lens
    forward_batch = get_current_forward_batch()
    if forward_batch is None or forward_batch.forward_mode.is_idle():
        return torch.zeros_like(weights, dtype=torch.float32)

    token_to_kv_pool = forward_batch.token_to_kv_pool
    if not hasattr(token_to_kv_pool, "get_index_k_with_scale_buffer"):
        raise RuntimeError(
            "[SGL+ATOM] DeepSeek-V3.2 sparse MLA requires SGLang NSA KV pool "
            "with index_k_with_scale_buffer support."
        )

    layer_id = _parse_layer_id_from_indexer_prefix(k_cache_prefix)
    index_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
    page_size = int(getattr(token_to_kv_pool, "page_size", 1))
    kv_cache = index_cache.view(-1, page_size, head_dim + 4)
    preshuffle_cache = page_size != 1
    slot_mapping = forward_batch.out_cache_loc

    if use_qk_rope_cache_fusion:
        q_bf16 = q_input
        q_fp8 = torch.empty_like(q_bf16, dtype=dtypes.fp8)
        weights_out = torch.empty(
            weights.shape, device=weights.device, dtype=torch.float32
        )
        indexer_qk_rope_quant_and_cache(
            q_bf16,
            q_fp8,
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
            preshuffle=preshuffle_cache,
            is_neox=is_neox_style,
        )
        weights = weights_out
    else:
        q_fp8 = q_input
        indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
            preshuffle=preshuffle_cache,
        )

    num_tokens = hidden_states.shape[0]
    topk_indices_buffer[:num_tokens] = -1
    block_table = _build_sglang_block_table(forward_batch, page_size)

    if forward_batch.forward_mode.is_decode_or_idle():
        bs = int(forward_batch.batch_size)
        if q_fp8.shape[0] < bs or weights.shape[0] < bs:
            raise RuntimeError(
                "[SGL+ATOM] sparse indexer decode expected at least "
                f"{bs} token rows, got q={q_fp8.shape[0]}, weights={weights.shape[0]}. "
                "This usually means TP-scattered indexer inputs were not gathered."
            )
        seq_lens_i32 = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
        padded_q_fp8 = q_fp8[:bs].reshape(bs, 1, *q_fp8.shape[1:])
        logits = torch.empty([bs, max_model_len], dtype=torch.float32, device=k.device)
        deepgemm_fp8_paged_mqa_logits(
            padded_q_fp8,
            kv_cache.unsqueeze(-2),
            weights[:bs],
            logits,
            seq_lens_i32,
            block_table,
            max_model_len,
            ChunkK=256,
            Preshuffle=preshuffle_cache,
            KVBlockSize=page_size,
            WavePerEU=2,
        )
        top_k_per_row_decode(
            logits,
            1,
            seq_lens_i32,
            topk_indices_buffer[:bs, :topk_tokens],
            bs,
            logits.stride(0),
            logits.stride(1),
        )
        return weights

    cu_starts, cu_ends = _build_sglang_query_ranges(forward_batch)
    total_kv = int(forward_batch.seq_lens_sum)
    k_fp8 = torch.empty([total_kv, head_dim], device=k.device, dtype=dtypes.fp8)
    k_scale = torch.empty([total_kv, 1], device=k.device, dtype=torch.float32)
    cp_gather_indexer_k_quant_cache(
        kv_cache,
        k_fp8,
        k_scale.view(dtypes.fp8),
        block_table,
        torch.nn.functional.pad(
            torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
        ),
        preshuffle=preshuffle_cache,
    )
    logits = fp8_mqa_logits(
        Q=q_fp8[:num_tokens],
        KV=k_fp8,
        kv_scales=k_scale,
        weights=weights[:num_tokens],
        cu_starts=cu_starts,
        cu_ends=cu_ends,
    )
    assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
    topk_indices = topk_indices_buffer[:num_tokens, :topk_tokens]
    top_k_per_row_prefill(
        logits=logits,
        rowStarts=cu_starts,
        rowEnds=cu_ends,
        indices=topk_indices,
        values=None,
        numRows=logits.shape[0],
        stride0=logits.stride(0),
        stride1=logits.stride(1),
    )
    topk_indices.copy_(
        torch.where(topk_indices >= 0, topk_indices - cu_starts[:, None], topk_indices)
    )
    return weights


def sparse_attn_indexer_sglang_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    del (
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_input,
        k,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        k_norm_weight,
        k_norm_bias,
        k_norm_eps,
        positions,
        cos_cache,
        sin_cache,
        weights_scale,
        is_neox_style,
        use_qk_rope_cache_fusion,
    )
    return torch.empty(weights.shape, device=weights.device, dtype=torch.float32)


direct_register_custom_op(
    op_name="sparse_attn_indexer_sglang_plugin_mode",
    op_func=sparse_attn_indexer_sglang_plugin_mode,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_sglang_fake,
)
