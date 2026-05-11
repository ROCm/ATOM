# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Plugin mode extensions for MLAAttention with sparse MLA support.

In vLLM plugin mode, the execution path is:
  vLLM MLAAttention.forward() → custom op → forward_impl() [PATCHED]
  → ATOM's forward_impl_plugin_mode() → forward_impl_sparse_plugin_mode()

The patched forward_impl (see register.py) redirects to ATOM's impl, which
dispatches to sparse or non-sparse based on topk_indices_buffer presence.
forward_impl_sparse_plugin_mode handles everything end-to-end: RoPE, KV cache
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

from atom.config import get_current_atom_config
from atom.plugin.prepare import is_vllm
from atom.utils import envs
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


class MLASparseAttentionImplPluginModeMethods:
    """
    Container class for sparse MLA plugin mode methods.
    This class cannot be instantiated - it only serves as a namespace for methods
    that will be added to MLAAttentionImpl via decorator.
    """

    def __init__(self):
        raise TypeError(
            "MLASparseAttentionImplPluginModeMethods cannot be instantiated. "
            "It is only used as a method container for the decorator."
        )

    def _forward_sparse_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices_global: torch.Tensor,  # [sq, topk]
        attn_metadata,
        layer,
    ) -> torch.Tensor:
        sparse_meta = attn_metadata.plugin_metadata

        num_tokens = q.shape[0]
        output = torch.empty(
            [num_tokens, self.padded_num_heads, self.kv_lora_rank],
            # dtype=q.dtype,
            dtype=torch.bfloat16,
            device=q.device,
        )

        seq_len = (topk_indices_global != -1).sum(dim=-1)
        torch.cumsum(seq_len, dim=0, out=sparse_meta.paged_kv_indptr[1:])
        sparse_meta.paged_kv_indptr_rest.fill_(sparse_meta.paged_kv_indptr[-1])
        fetch_id_to_ragged_triton(
            topk_indices_global,
            sparse_meta.paged_kv_indptr,
            sparse_meta.paged_kv_indices,
            sparse_meta.topk_tokens,
        )

        kv_buffer = kv_cache.unsqueeze(2)

        # TODO: Currently, persistent mode has memory access issues when input context is long
        # in fp8 kv cache settings, so it is not used for now.
        # Re-enable persistent mode when the issue is fixed.
        mla_decode_fwd(
            q,
            kv_buffer.view(-1, 1, 1, q.shape[-1]),
            output,
            sparse_meta.qo_indptr,
            sparse_meta.paged_kv_indptr,
            sparse_meta.paged_kv_indices,
            sparse_meta.paged_kv_last_page_len,
            1,
            sm_scale=self.scale,
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
            page_size=1,
        )

        if self.head_repeat_factor > 1:
            output = output[:, :: self.head_repeat_factor, :].contiguous()

        return output[:, : self.num_heads, :]

    def forward_impl_sparse_plugin_mode(
        self,
        layer,
        q,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output,
    ):
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        from vllm.distributed.parallel_state import get_dcp_group
        from vllm.platforms import current_platform

        if self.dcp_world_size == -1:
            self.dcp_world_size = get_dcp_group().world_size

        sparse_meta = attn_metadata.plugin_metadata

        num_actual_toks = sparse_meta.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...].unsqueeze(1)

        positions = None
        if self._is_vllm_forward_context_available():
            positions = self._get_vllm_forward_context().additional_kwargs.get(
                "atom_positions"
            )

        if positions is None:
            atom_config = get_current_atom_config()
            positions = atom_config.compilation_config.static_forward_context[
                "positions"
            ]

        positions = positions[:num_actual_toks]
        fp8_attention = self.kv_cache_dtype.startswith("fp8")
        if fp8_attention:
            from vllm.platforms import current_platform

            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        # Q absorption: q_nope -> W_K BMM -> ql_nope, then concat with q_pe
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)

        if self.q_pad_num_heads is not None:
            B, N, L = q_pe.shape
            pe_padded = q_pe.new_empty((B, self.q_pad_num_heads, L))
            pe_padded.resize_((B, N, L))
            pe_padded.copy_(q_pe)
            q_pe = pe_padded

        if self.is_aiter_triton_fp4_bmm_enabled:
            ql_nope = batched_gemm_a16wfp4(
                q_nope,
                self.W_K,
                self.W_K_scale,
                transpose_bm=True,
                prequant=True,
                y_scale=layer._q_scale if fp8_attention else None,
            )
        else:
            # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
            ql_nope = _aiter_triton_fp8_bmm(
                q_nope,
                self.W_K,
                self.W_K_scale,
                group_size=128,
                transpose_bm=True,
            )

        q_out = torch.empty(
            (
                ql_nope.shape[0],
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=ql_nope.dtype,
            device=ql_nope.device,
        )
        if kv_cache.numel() > 0:
            fused_qk_rope_concat_and_cache_mla(
                ql_nope,
                q_pe,
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache.view(
                    kv_cache.shape[0], -1, self.kv_lora_rank + self.qk_rope_head_dim
                ),
                q_out,
                attn_metadata.slot_mapping,
                layer._k_scale,
                layer._q_scale,
                positions,
                self.rotary_emb.cos_cache,
                self.rotary_emb.sin_cache,
                is_neox=self.rotary_emb.is_neox_style,
                is_nope_first=True,
            )

        if self.head_repeat_factor > 1:
            q_out = q_out.repeat_interleave(self.head_repeat_factor, dim=1)

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        try:
            from vllm.v1.attention.backends.mla.sparse_utils import (
                triton_convert_req_index_to_global_index,
            )
        except ImportError:
            from vllm.v1.attention.backends.mla.flashmla_sparse import (
                triton_convert_req_index_to_global_index,
            )

        req_id_i32 = sparse_meta.req_id_per_token.to(dtype=torch.int32)
        block_table_i32 = sparse_meta.block_table.to(dtype=torch.int32)
        topk_indices_i32 = topk_indices.to(dtype=torch.int32)
        topk_indices_global = triton_convert_req_index_to_global_index(
            req_id_i32,  # sparse_meta.req_id_per_token,
            block_table_i32,  # sparse_meta.block_table,
            topk_indices_i32,  # topk_indices,
            BLOCK_SIZE=sparse_meta.block_size,
            NUM_TOPK_TOKENS=sparse_meta.topk_tokens,
        )
        if fp8_attention:
            from vllm import _custom_ops as ops

            # Reshape to 2D for scaled_fp8_quant, then restore
            q_flat, _ = ops.scaled_fp8_quant(
                q_out.reshape(q_out.shape[0], -1),
                layer._q_scale,
            )
            q_out = q_flat.reshape(q_out.shape)
        attn_out = self._forward_sparse_bf16_kv(
            q_out, kv_cache, topk_indices_global, attn_metadata, layer
        )

        # V up-projection
        self._v_up_proj(attn_out, out=output[:num_actual_toks])

        return output_padded


def _mla_sparse_plugin_mode_init(self, *args, **kwargs):
    """Extra initialization for MLAAttentionImpl in sparse plugin mode (vllm)."""
    if is_vllm():
        self.supports_quant_query_input = False
        self.dcp_world_size: int = -1
        self.is_aiter_triton_fp4_bmm_enabled = (
            envs.ATOM_USE_TRITON_MXFP4_BMM
            and self.kv_b_proj.weight.dtype == torch.bfloat16
        )
        self.q_pad_num_heads = kwargs.get("q_pad_num_heads", None)

        from atom.model_ops.attention_mla import _MLA_MIN_HEADS

        self.padded_num_heads = max(self.num_heads, _MLA_MIN_HEADS)
        self.head_repeat_factor = self.padded_num_heads // self.num_heads

        # Sparse MLA specific: verify topk_indices_buffer is present
        assert getattr(self, "topk_indices_buffer", None) is not None, (
            "topk_indices_buffer must be set for sparse MLA plugin mode. "
            "Ensure the model's Indexer is properly initialized."
        )
        self._is_sparse_mla = True


def MLASparseAttentionImplDecoratorForPluginMode(cls):
    """
    Decorator that injects sparse MLA methods into the MLAAttentionImpl class.
    Applied alongside the regular MLAAttentionImplDecoratorForPluginMode.

    Injects forward_impl_sparse_plugin_mode and _forward_sparse_bf16_kv.
    The patched forward_impl in register.py calls forward_impl_plugin_mode,
    which dispatches to forward_impl_sparse_plugin_mode when topk_indices_buffer
    is set.
    """
    sparse_method_names = [
        "_forward_sparse_bf16_kv",
        "forward_impl_sparse_plugin_mode",
    ]

    logger.info(
        "Use MLASparseAttentionImplDecoratorForPluginMode to decorate MLAAttention"
    )

    # Add all sparse methods to the target class
    for method_name in sparse_method_names:
        method = getattr(MLASparseAttentionImplPluginModeMethods, method_name)
        setattr(cls, method_name, method)

    # Wrap __init__ to inject sparse plugin-mode initialization
    orig_init = cls.__init__

    def new_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Only run sparse init if this instance has a topk_indices_buffer
        # (i.e., the model uses sparse MLA)
        if getattr(self, "topk_indices_buffer", None) is not None:
            _mla_sparse_plugin_mode_init(self, *args, **kwargs)

    cls.__init__ = new_init

    return cls


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
    from atom.plugin.vllm.attention_backend.mla_sparse import (
        AiterMLASparseIndexerBackend,
    )

    return AiterMLASparseIndexerBackend


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
