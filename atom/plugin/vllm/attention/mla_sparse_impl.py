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
from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as _aiter_triton_fp8_bmm,
)
from aiter.mla import mla_decode_fwd
from aiter import (
    fused_qk_rope_concat_and_cache_mla,
    cp_gather_indexer_k_quant_cache,
    dtypes,
    indexer_k_quant_and_cache,
    top_k_per_row_decode,
    top_k_per_row_prefill,
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
def sparse_attn_indexer_plugin_mode(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
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
        return weights
    if k_cache_prefix not in attn_metadata_dict:
        return weights
    layer_meta = attn_metadata_dict[k_cache_prefix]
    if layer_meta is None:
        return weights

    # In plugin mode, plugin_metadata is vllmDeepseekV32IndexerMetadata from
    # AiterMLASparseIndexerMetadataBuilder.
    plugin_meta = layer_meta.plugin_metadata
    indexer_meta = plugin_meta
    slot_mapping = indexer_meta.slot_mapping
    has_decode = indexer_meta.num_decodes > 0
    has_prefill = indexer_meta.num_prefills > 0
    num_decode_tokens = indexer_meta.num_decode_tokens

    indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    # topk_indices_buffer[: num_actual_tokens] = -1
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
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            # Use top_k_per_row_prefill from vLLM to correctly handle row starts
            # and ends. It also produces 0-based local indices, eliminating the
            # need for conversion from global.
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
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
        )

        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            from vllm.v1.attention.ops.common import unpack_seq_triton

            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return weights


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    _k_fp8 = _flattened_kv[..., :head_dim].view(torch.float8_e4m3fn).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return weights


direct_register_custom_op(
    op_name="sparse_attn_indexer_plugin_mode",
    op_func=sparse_attn_indexer_plugin_mode,
    mutates_args=["topk_indices_buffer"],
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
        block_size=1,  # block_size = 1 for indexer on ROCm
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
