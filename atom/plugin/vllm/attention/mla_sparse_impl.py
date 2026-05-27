# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Plugin mode extensions for MLAAttention with sparse MLA support.

In vLLM plugin mode, the execution path is:
  ATOM AttentionForVllm.forward() → custom op → AttentionForVllmMLA.forward_impl()
  → AttentionForVllmMLA.forward_impl_sparse()

forward_impl_sparse handles everything end-to-end: RoPE, KV cache
write, Q absorption, topk index conversion, sparse kernel, V up-projection.
"""

import torch

from aiter import (
    cp_gather_indexer_k_quant_cache,
    dtypes,
    indexer_k_quant_and_cache,
    indexer_qk_rope_quant_and_cache,
    top_k_per_row_decode,
)
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

from atom.plugin.prepare import is_vllm
from atom.utils.custom_register import direct_register_custom_op

import triton
import triton.language as tl

from typing import Optional
import logging

logger = logging.getLogger("atom")


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


@triton.jit
def generate_sparse_seqlen_kernel(
    seq_len_ptr,
    cu_query_lens_ptr,
    out_ptr,
    topk_token: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    query_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    query_start = tl.load(cu_query_lens_ptr + seq_id)
    query_end = tl.load(cu_query_lens_ptr + seq_id + 1)
    if query_start + tl.program_id(1) * BLOCK_SIZE > query_end:
        return
    query_len = query_end - query_start
    query_mask = query_offset + query_start < query_end
    seq_len = tl.load(seq_len_ptr + seq_id)
    if seq_len == 0:
        return
    context_start_point = seq_len - query_len
    sparse_seqlen = context_start_point + query_offset
    sparse_seqlen_masked = tl.where(
        sparse_seqlen + 1 < topk_token, sparse_seqlen + 1, topk_token
    )
    tl.store(
        out_ptr + query_start + query_offset,
        sparse_seqlen_masked,
        mask=query_mask,
    )


def generate_sparse_seqlen_triton(
    query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    topk_token: int,
    num_tokens: int,
    max_query_len: int,
):
    num_seqs = query_lens.size(0)
    out = torch.zeros([num_tokens], dtype=torch.int32, device=query_lens.device)
    block_size = 64
    num_block_per_row = triton.cdiv(max_query_len, block_size)
    grid = (num_seqs, num_block_per_row)
    generate_sparse_seqlen_kernel[grid](
        seq_lens,
        cu_query_lens,
        out,
        topk_token,
        block_size,
    )
    return out


@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,  # [num_seq, topk]
    cumsum_ptr,  # [num_seq + 1]
    out_tensor_ptr,  # [max_num_seq * topk]
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    token_start = tl.load(cumsum_ptr + seq_id)
    token_end = tl.load(cumsum_ptr + seq_id + 1)
    token_num = token_end - token_start
    row_offset = block_id * BLOCK_SIZE
    if row_offset >= token_num:
        return
    in_tensor_offset = seq_id * in_tensor_ptr_stride + row_offset + offset
    in_tensor_mask = (row_offset + offset) < TOPK
    in_tensor_val = tl.load(in_tensor_ptr + in_tensor_offset, mask=in_tensor_mask)
    out_tensor_offset = token_start + row_offset + offset
    out_tensor_mask = (out_tensor_offset < token_end) & in_tensor_mask
    tl.store(out_tensor_ptr + out_tensor_offset, in_tensor_val, mask=out_tensor_mask)


def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
):
    num_tokens = in_tensor.size(0)
    block_size = 64
    num_block_per_row = triton.cdiv(topk, block_size)
    grid = (
        num_tokens,
        num_block_per_row,
    )
    fetch_id_to_ragged_kernel[grid](
        in_tensor, cumsum, out_tensor, in_tensor.stride(0), topk, num_tokens, block_size
    )


def _get_sparse_mla_metadata(attn_metadata_dict, k_cache_prefix: str):
    if not k_cache_prefix.endswith(".indexer.k_cache"):
        return None

    attention_prefix = k_cache_prefix[: -len(".indexer.k_cache")]
    for sparse_attn_prefix in (f"{attention_prefix}.attn", attention_prefix):
        sparse_attn_meta = attn_metadata_dict.get(sparse_attn_prefix)
        if getattr(sparse_attn_meta, "paged_kv_indices", None) is not None:
            return sparse_attn_meta
    return None


def sparse_attn_indexer_plugin_mode(
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
    sparse_kv_indices_buffer: torch.Tensor,
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
    topk_indices = torch.empty(
        hidden_states.shape[0],
        topk_tokens,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    try:
        from vllm.forward_context import (
            get_forward_context as get_vllm_forward_context,
            is_forward_context_available as is_vllm_ctx_available,
        )

        if is_vllm_ctx_available():
            vllm_ctx = get_vllm_forward_context()
            attn_metadata_dict = vllm_ctx.attn_metadata
    except ImportError:
        raise ImportError("vLLM forward context not available")

    # During profile/dummy run the metadata dict may not contain
    # our layer or may be None.
    if attn_metadata_dict is None:
        return torch.zeros_like(weights, dtype=torch.float32)
    if k_cache_prefix not in attn_metadata_dict:
        return torch.zeros_like(weights, dtype=torch.float32)
    layer_meta = attn_metadata_dict[k_cache_prefix]
    if layer_meta is None:
        return torch.zeros_like(weights, dtype=torch.float32)

    # vLLM sparse indexer builders return AiterMlaSparseIndexerMetadataForVllm directly
    indexer_meta = layer_meta
    sparse_meta = _get_sparse_mla_metadata(attn_metadata_dict, k_cache_prefix)
    if sparse_meta is None:
        raise RuntimeError(
            "Sparse MLA metadata not found for indexer cache "
            f"{k_cache_prefix!r}. The indexer cannot populate paged_kv_indices."
        )
    slot_mapping = indexer_meta.slot_mapping
    has_decode = indexer_meta.num_decodes > 0
    has_prefill = indexer_meta.num_prefills > 0
    num_decode_tokens = indexer_meta.num_decode_tokens

    kv_block_size = kv_cache.shape[1]
    preshuffle_cache = kv_block_size != 1

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

    if has_prefill:
        prefill_metadata = indexer_meta.prefill
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=k.device,
                dtype=dtypes.fp8,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 1],
                device=k.device,
                dtype=torch.float32,
            )

            cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale.view(dtypes.fp8),
                chunk.block_table,
                chunk.cu_seq_lens,
                preshuffle=preshuffle_cache,
            )

            logits = fp8_mqa_logits(
                Q=q_fp8[chunk.token_start : chunk.token_end],
                KV=k_fp8,
                kv_scales=k_scale,
                weights=weights[chunk.token_start : chunk.token_end],
                cu_starts=chunk.cu_seqlen_ks,
                cu_ends=chunk.cu_seqlen_ke,
            )
            num_rows = logits.shape[0]
            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
            topk_indices_prefill = topk_indices[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            # Use top_k_per_row_prefill from vLLM to correctly handle row starts
            # and ends. It also produces 0-based local indices, eliminating the
            # need for conversion from global.
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices_prefill,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = indexer_meta.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            from vllm.v1.attention.ops.common import pack_seq_triton

            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        logits = torch.empty(
            [batch_size * next_n, max_model_len], dtype=torch.float32, device="cuda"
        )
        deepgemm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            logits,
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            max_model_len,
            ChunkK=256,
            KVBlockSize=kv_block_size,
            Preshuffle=preshuffle_cache,
            WavePerEU=2,
        )

        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices_decode = topk_indices[:num_decode_tokens, :topk_tokens]
        top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices_decode,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            from vllm.v1.attention.ops.common import unpack_seq_triton

            unpacked_topk_indices = unpack_seq_triton(
                topk_indices_decode.reshape(
                    batch_size, -1, topk_indices_decode.shape[-1]
                ),
                decode_lens,
            )
            topk_indices[:num_decode_tokens, : unpacked_topk_indices.shape[-1]] = (
                unpacked_topk_indices
            )

    triton_convert_req_index_to_global_index(
        sparse_meta.req_id_per_token.to(dtype=torch.int32),
        sparse_meta.block_table.to(dtype=torch.int32),
        topk_indices[: sparse_meta.num_actual_tokens].to(dtype=torch.int32),
        sparse_meta.paged_kv_indptr,
        sparse_kv_indices_buffer,
        BLOCK_SIZE=sparse_meta.block_size,
        NUM_TOPK_TOKENS=sparse_meta.topk_tokens,
    )

    return weights


def sparse_attn_indexer_fake(
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
    sparse_kv_indices_buffer: torch.Tensor,
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
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    _k_fp8 = _flattened_kv[..., :head_dim].view(torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return torch.empty(weights.shape, device=weights.device, dtype=torch.float32)


direct_register_custom_op(
    op_name="sparse_attn_indexer_plugin_mode",
    op_func=sparse_attn_indexer_plugin_mode,
    mutates_args=["sparse_kv_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
)


def IndexerDecoratorForPluginMode(cls):
    if getattr(cls, "_atom_vllm_indexer_decorated", False):
        return cls

    orig_init = cls.__init__

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if is_vllm():
            self.sparse_attn_indexer_impl = (
                torch.ops.aiter.sparse_attn_indexer_plugin_mode
            )

    cls.__init__ = new_init
    cls._atom_vllm_indexer_decorated = True
    return cls


def _deepseek_v32_indexer_get_kv_cache_spec(self, vllm_config):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return MLAAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=1,
        head_size=self.head_dim,
        dtype=self.dtype,
    )


def _deepseek_v32_indexer_get_attn_backend(self):
    from atom.plugin.vllm.attention.backend import (
        AiterSparseMlaIndexerBackendForVllm,
    )

    return AiterSparseMlaIndexerBackendForVllm


def DeepseekV32IndexerCacheDecoratorForPluginMode(cls):
    if getattr(cls, "_atom_vllm_indexer_cache_decorated", False):
        return cls
    if not is_vllm():
        return cls
    cls.get_kv_cache_spec = _deepseek_v32_indexer_get_kv_cache_spec
    cls.get_attn_backend = _deepseek_v32_indexer_get_attn_backend

    # In ATOM, kv cache is a list of tensors and accessed through indexing [0].
    # But in vLLM plugin mode, kv cache is a single tensor. So we wrap it in a
    # list so that the kv cache can be fully accessed.
    original_setattr = cls.__setattr__

    def _wrapped_setattr(self, name, value):
        if name == "kv_cache" and isinstance(value, torch.Tensor):
            original_setattr(self, name, [value])
        else:
            original_setattr(self, name, value)

    cls.__setattr__ = _wrapped_setattr

    cls._atom_vllm_indexer_cache_decorated = True
    return cls
