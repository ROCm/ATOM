# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Plugin mode extensions for PagedAttentionImpl.
This module provides additional methods for PagedAttentionImpl when running in plugin mode.
"""

import torch
import aiter
from aiter import dtypes, fused_qk_norm_rope_cache_quant_shuffle
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits
from typing import TYPE_CHECKING

import logging
logger = logging.getLogger("atom")

if TYPE_CHECKING:
    from atom.utils.forward_context import AttentionMetaData

from atom.utils import envs

ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION = envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION

_PARTITION_SIZE_ROCM = 256
_CP_TOKENS_PER_ITER_ROCM = 32 * 1024


class PagedAttentionImplPluginModeMethods:
    """
    Container class for plugin mode methods.
    This class cannot be instantiated - it only serves as a namespace for methods
    that will be added to PagedAttentionImpl via decorator.
    """

    def __init__(self):
        raise TypeError(
            "PagedAttentionImplPluginModeMethods cannot be instantiated. "
            "It is only used as a method container for the decorator."
        )

    def rope_cache_plugin_mode(self,
                               q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               qkv: torch.Tensor,
                               position: torch.Tensor,
                               attention_metadata: "AttentionMetaData",
                               k_cache: torch.Tensor,
                               v_cache: torch.Tensor,
                               k_scale: torch.Tensor,
                               v_scale: torch.Tensor,
                               flash_layout: bool = False):

        num_blocks, block_size, num_kv_heads, head_size = k_cache.shape

        if not flash_layout:
            x = 16 // k_cache.element_size()
            k_cache_template = torch.empty(
                [num_blocks, num_kv_heads, head_size // x, block_size, x],
                dtype=k_cache.dtype,
                device="meta",
            )
            # ATOM: [num_blocks, num_kv_heads, head_size, block_size],
            # vLLM: [num_blocks, num_kv_heads, block_size // x, head_size, x],
            v_cache_template = torch.empty(
                [num_blocks, num_kv_heads, block_size // x, head_size, x],
                dtype=v_cache.dtype,
                device="meta",
            )
            new_key_cache = k_cache.view_as(k_cache_template)
            new_value_cache = v_cache.view_as(v_cache_template)
        else:
            new_key_cache = k_cache
            new_value_cache = v_cache

        # if flash kv_cache layout, the shape of kv_cache is:
        #
        # key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        #
        # if not, the shape is:
        #
        # key_cache:   [num_blocks, num_kv_heads, head_size // x, block_size, x]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        # 
        # and the origin kv cache layout in fwd_args is not flash

        attn_metadata = attention_metadata

        use_triton_attn = self.sliding_window != -1 or self.head_dim != 128
        self.use_triton_attn = use_triton_attn

        if (
            self.rotary_emb is not None
            and self.q_norm is not None
            and self.k_norm is not None
        ):
            fused_qk_norm_rope_cache_quant_shuffle(
                qkv,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.q_norm.eps,
                qw=self.q_norm.weight,
                kw=self.k_norm.weight,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                is_neox_style=self.rotary_emb.is_neox_style,
                pos_ids=position,
                k_cache=new_key_cache,
                v_cache=new_value_cache,
                slot_mapping=attn_metadata.slot_mapping,
                kv_cache_dtype=(
                    "auto" if self.kv_cache_dtype == "bf16" else self.kv_cache_dtype
                ),
                k_scale=k_scale,
                v_scale=v_scale,
            )

            qkv = qkv.view(qkv.shape[0], -1, self.head_dim)
            q, k, v = qkv.split(
                [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=1)
        elif use_triton_attn and self.rotary_emb is not None:
            k_scale = v_scale = self.kv_scale

            q, k, k_cache, v_cache = fused_qk_rope_reshape_and_cache(
                q,
                k,
                v,
                new_key_cache,
                new_value_cache,
                attn_metadata.slot_mapping,
                position,
                self.rotary_emb.cos_cache,
                self.rotary_emb.sin_cache,
                k_scale,
                v_scale,
                self.rotary_emb.is_neox_style,
                flash_layout=flash_layout,
                apply_scale=self.kv_cache_dtype.startswith("fp8"),
                offs=None,
                q_out=q,
                k_out=k,
                output_zeros=False,
            )
        else:
            # for asm paged attention 
            if self.rotary_emb is not None:
                assert position is not None
                q, k = self.rotary_emb(position, q, k)
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.kv_cache_dtype == "fp8":
                aiter.reshape_and_cache_with_pertoken_quant(
                    k,
                    v,
                    new_key_cache,
                    new_value_cache,
                    k_scale,
                    v_scale,
                    attn_metadata.slot_mapping,
                    asm_layout=True,
                )
            else:
                aiter.reshape_and_cache(
                    k,
                    v,
                    new_key_cache,
                    new_value_cache,
                    attn_metadata.slot_mapping,
                    kv_cache_dtype="auto",
                    k_scale=None,
                    v_scale=None,
                    asm_layout=True,
                )

        return q, k, v, k_cache, v_cache, k_scale, v_scale

    def _get_cp_mha_gather_cache_views(
        self, key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # For SHUFFLE layout, the wrapper derives PAGE_SIZE/num_heads from
        # tensor shapes; provide a reshape-only view to keep storage unchanged.
        if key_cache.ndim == 5:
            num_blocks = key_cache.shape[0]
            num_heads = key_cache.shape[1]
            page_size = key_cache.shape[3]
            x = key_cache.shape[4]
            head_size = key_cache.shape[2] * x
            key_cache = key_cache.view(num_blocks, page_size, num_heads, head_size)
            value_cache = value_cache.view(num_blocks, page_size, num_heads, head_size)
            return key_cache, value_cache, page_size
        return key_cache, value_cache, key_cache.shape[1]

    def paged_attention_triton_plugin_mode(self,
                                           q: torch.Tensor,
                                           k_cache: torch.Tensor,
                                           v_cache: torch.Tensor,
                                           k_scale: torch.Tensor,
                                           v_scale: torch.Tensor,
                                           out: torch.Tensor,
                                           attn_metadata: "AttentionMetaData"):

        o = out
        num_seqs, num_q_heads_total, head_size = q.shape
        num_blocks, num_kv_heads, _, block_size, _ = k_cache.shape
        query_group_size = num_q_heads_total // num_kv_heads
        assert num_q_heads_total % num_kv_heads == 0

        max_context_partition_num = get_recommended_splits(num_seqs, num_kv_heads)

        context_partition_size = 256
        if self.sliding_window > 0:
            max_context_partition_num = 1
            context_partition_size = 128

        # Output buffers (same as Triton)
        intermediate_shape = (
            num_seqs,
            num_kv_heads,
            max_context_partition_num,
            query_group_size,
        )
        exp_sums = torch.empty(
            intermediate_shape, dtype=torch.float32, device=q.device
        )
        max_logits = torch.empty(
            intermediate_shape, dtype=torch.float32, device=q.device
        )
        temporary_output = torch.empty(
            *intermediate_shape,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )

        per_tensor = False
        if k_scale is not None:
            per_tensor = k_scale.numel() == 1
            if not per_tensor:
                k_scale = k_scale.unsqueeze(-1)
                v_scale = v_scale.unsqueeze(-1)
        compute_type = torch.bfloat16 if self.kv_cache_dtype == "bf16" or per_tensor else aiter.dtypes.fp8

        torch.ops.aiter.pa_decode_gluon(
            o,
            q,
            k_cache,
            v_cache,
            attn_metadata.plugin_metadata.seq_lens,
            attn_metadata.block_tables,
            self.scale,
            1, # query_lenth
            max_context_partition_num, 
            context_partition_size,
            compute_type,
            None,
            None if self.kv_cache_dtype == "bf16" else k_scale,
            None if self.kv_cache_dtype == "bf16" else v_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=self.sinks,
            sliding_window=self.sliding_window,
            ps=True,
        )

        return o

    def paged_attention_asm_plugin_mode(self,
                                        q: torch.Tensor,
                                        k_cache: torch.Tensor,
                                        v_cache: torch.Tensor,
                                        k_scale: torch.Tensor,
                                        v_scale: torch.Tensor,
                                        num_decodes: int,
                                        num_decode_tokens: int,
                                        attn_metadata: "AttentionMetaData",
                                        out: torch.Tensor):
        aiter.pa_fwd_asm(
            Q=q,
            K=k_cache,
            V=v_cache,
            block_tables=attn_metadata.plugin_metadata.block_table[:num_decodes],
            context_lens=attn_metadata.plugin_metadata.seq_lens[:num_decodes],
            block_tables_stride0=attn_metadata.plugin_metadata.block_table[
                :num_decodes
            ].stride(0),
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=out[:num_decode_tokens],
            high_precision=0,
        )

        return

    def extend_for_sliding_window(
        self,
        attn_metadata: "AttentionMetaData",
        query: torch.Tensor,
        key_cache,
        value_cache,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        block_table: torch.Tensor,
        k_scale: float,
        v_scale: float,
    ):
        assert attn_metadata.plugin_metadata.extend_metadata is not None
        assert attn_metadata.plugin_metadata.extend_metadata.chunk_context_metadata is not None
        chunked_metadata = attn_metadata.plugin_metadata.extend_metadata.chunk_context_metadata
        swa_metadata = chunked_metadata.swa_metadata
        assert swa_metadata is not None
        swa_cu_seqlens = swa_metadata.swa_cu_seqlens
        swa_seq_starts = swa_metadata.swa_seq_starts
        swa_token_to_batch = swa_metadata.swa_token_to_batch
        swa_max_seqlens = swa_metadata.swa_max_seqlens
        swa_total_tokens = swa_metadata.swa_total_tokens
        key_fetched, value_fetched = (
            swa_metadata.swa_workspace[0],
            swa_metadata.swa_workspace[1],
        )

        from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache
        # key_cache_for_gather, value_cache_for_gather, _ = (
        #     self._get_cp_mha_gather_cache_views(key_cache, value_cache)
        # )
        cp_mha_gather_cache(
            key_cache=key_cache,
            value_cache=value_cache,
            key=key_fetched,
            value=value_fetched,
            block_tables=block_table,
            k_scales=k_scale,
            v_scales=v_scale,
            cu_seqlens_kv=swa_cu_seqlens,
            token_to_batch=swa_token_to_batch,
            seq_starts=swa_seq_starts,
            dequant=self.kv_cache_dtype.startswith("fp8"),
            kv_cache_layout="NHD",
            total_tokens=swa_total_tokens,
        )

        sliding_window = (self.sliding_window, 0, 0) if self.sliding_window is not None else (-1, -1, 0)
        aiter.flash_attn_varlen_func(
            q=query,
            k=key_fetched,
            v=value_fetched,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=swa_cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=swa_max_seqlens,
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            window_size=sliding_window,
            alibi_slopes=self.alibi_slopes,
            return_lse=False,
            out=output,
        )

    def extend_forward(
        self,
        attn_metadata: "AttentionMetaData",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ):
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
        from vllm.v1.attention.backends.rocm_aiter_fa import cp_mha_gather_cache

        if self.sliding_window != -1:
            self.extend_for_sliding_window(
                attn_metadata,
                query,
                key_cache,
                value_cache,
                output,
                cu_seqlens_q,
                max_seqlen_q,
                block_table,
                k_scale,
                v_scale,
            )
            return
        out, lse = aiter.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            min_seqlen_q=min_seqlen_q,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            return_lse=True,
        )
        assert attn_metadata.plugin_metadata.extend_metadata is not None
        chunk_context_metadata = attn_metadata.plugin_metadata.extend_metadata.chunk_context_metadata
        num_chunks = chunk_context_metadata.num_chunks
        workspace = chunk_context_metadata.workspace
        cu_seqlens_kv = chunk_context_metadata.cu_seq_lens_chunk
        max_seqlens = chunk_context_metadata.max_seq_lens
        chunk_starts = chunk_context_metadata.chunk_starts
        token_to_batch = chunk_context_metadata.token_to_batch
        total_token_per_batch = chunk_context_metadata.total_token_per_batch
        key_fetched, value_fetched = workspace[0], workspace[1]
        chunked_output = None
        chunked_lse = None
        # key_cache_for_gather, value_cache_for_gather, _ = (
        #     self._get_cp_mha_gather_cache_views(key_cache, value_cache)
        # )
        for chunk_idx in range(num_chunks):
            cp_mha_gather_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                key=key_fetched,
                value=value_fetched,
                block_tables=block_table,
                k_scales=k_scale,
                v_scales=v_scale,
                cu_seqlens_kv=cu_seqlens_kv[chunk_idx],
                token_to_batch=token_to_batch[chunk_idx],
                seq_starts=chunk_starts[chunk_idx],
                dequant=self.kv_cache_dtype.startswith("fp8"),
                kv_cache_layout="SHUFFLE",
                total_tokens=total_token_per_batch[chunk_idx],
            )

            suf_out, suf_lse = aiter.flash_attn_varlen_func(
                q=query,
                k=key_fetched,
                v=value_fetched,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv[chunk_idx],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlens[chunk_idx],
                min_seqlen_q=min_seqlen_q,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
                window_size=(-1, -1, 0),
                alibi_slopes=self.alibi_slopes,
                return_lse=True,
            )

            if chunked_output is None:
                chunked_output = suf_out
                chunked_lse = suf_lse
            else:
                tmp_output = torch.empty_like(out)
                tmp_lse = torch.empty_like(lse)
                merge_attn_states(
                    output=tmp_output,
                    output_lse=tmp_lse,
                    prefix_output=chunked_output,
                    prefix_lse=chunked_lse,
                    suffix_output=suf_out,
                    suffix_lse=suf_lse,
                )
                chunked_output = tmp_output
                chunked_lse = tmp_lse

        merge_attn_states(
            output=output,
            prefix_output=chunked_output,
            prefix_lse=chunked_lse,
            suffix_output=out,
            suffix_lse=lse,
        )

    def forward_impl_plugin_mode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetaData" = None,
        position: torch.Tensor = None,
        q_scale: torch.Tensor = None,
        qkv: torch.Tensor = None,
        output: torch.Tensor = None,
    ):
        # create the output here, it use query shape
        num_tokens = query.shape[0]
        output_dtype = query.dtype
        output_shape = torch.Size(
            (num_tokens, self.num_heads * self.head_size)
        )
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)

        # dummy run will skip attention in cuda graph capture phase
        if attn_metadata is None:
            return output.fill_(0)

        # when using this optimization, the qkv tensor and 
        # position tensor are passed through q,k,v
        if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
            assert position is None, "position should be None because it is passed through k"

            position = key
            qkv = value

            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            query, key, value = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        else:
            # the position is computed by ATOM, and contained in attention metadata
            # when dummy run, the attn metadata is None
            if attn_metadata is not None:
                position = attn_metadata.plugin_metadata.context.positions

        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        output = output.view(-1, self.num_heads, self.head_dim)

        num_actual_tokens = attn_metadata.plugin_metadata.num_actual_tokens
        k_cache, v_cache = kv_cache.unbind(0)
        num_blocks, block_size, num_kv_heads, _ = k_cache.shape

        if self.kv_cache_dtype == "fp8":
            target_dtype = dtypes.d_dtypes[self.kv_cache_dtype]
            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)

        # create kv scale according to the num_blocks
        # usually it is created when cuda graph capture for decode phase
        if self.kv_cache_dtype == "fp8":
            if self.k_scale is None or self.v_scale is None:
                self.kv_scale = torch.zeros(
                    2,
                    num_blocks,
                    num_kv_heads,
                    block_size,
                    dtype=dtypes.fp32,
                    device=self.device,
                )
            # update the layer kv scale tensor
            self.k_scale = self.kv_scale[0]
            self.v_scale = self.kv_scale[1]
            layer.k_scale = self.k_scale
            layer.v_scale = self.v_scale

        # rope and cache flush fusion
        result = self.rope_cache_plugin_mode(q=query,
                                             k=key,
                                             v=value,
                                             qkv=qkv,
                                             position=position,
                                             attention_metadata=attn_metadata,
                                             k_cache=k_cache,
                                             v_cache=v_cache,
                                             k_scale=self.k_scale,
                                             v_scale=self.v_scale,
                                             flash_layout=False)
        (query, key, value, k_cache, v_cache, k_scale, v_scale) = result

        # The tokens are storaged as [decode:extend:prefill] order
        # which is decided by the vllm
        query = query[:num_actual_tokens]
        if key is not None:
            key = key[:num_actual_tokens]
        if value is not None:
            value = value[:num_actual_tokens]

        output_actual_tokens = output[:num_actual_tokens]

        num_decodes = attn_metadata.plugin_metadata.num_decodes
        num_prefills = attn_metadata.plugin_metadata.num_prefills
        num_extends = attn_metadata.plugin_metadata.num_extends

        num_decode_tokens = attn_metadata.plugin_metadata.num_decode_tokens
        num_extend_tokens = attn_metadata.plugin_metadata.num_extend_tokens

        # calculate for prefills
        if num_prefills > 0:
            assert attn_metadata.plugin_metadata.prefill_metadata is not None

            # prefill part is after decode and extend
            prefill_query = query[num_decode_tokens + num_extend_tokens :]
            prefill_key = key[num_decode_tokens + num_extend_tokens :]
            prefill_value = value[num_decode_tokens + num_extend_tokens :]

            sliding_window = (self.sliding_window, 0, 0) if self.sliding_window is not None else (-1, -1, 0)

            aiter.flash_attn_varlen_func(
                q=prefill_query,
                k=prefill_key,
                v=prefill_value,
                cu_seqlens_q=attn_metadata.plugin_metadata.prefill_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.plugin_metadata.prefill_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.plugin_metadata.prefill_metadata.max_query_len,
                max_seqlen_k=attn_metadata.plugin_metadata.prefill_metadata.max_seq_len,
                min_seqlen_q=1,
                dropout_p=attn_metadata.dropout_p,
                softmax_scale=self.scale,
                causal=True,
                window_size=sliding_window,
                alibi_slopes=None,
                sink_ptr=self.sinks,
                out=output_actual_tokens[num_decode_tokens + num_extend_tokens :],
            )

        # calculate for extends
        if num_extends > 0:
            assert attn_metadata.plugin_metadata.extend_metadata is not None
            extend_tokens_slice = slice(
                num_decode_tokens, num_decode_tokens + num_extend_tokens
            )
            extend_querys = query[extend_tokens_slice]
            extend_keys = key[extend_tokens_slice]
            extend_values = value[extend_tokens_slice]
            extend_outputs = output[extend_tokens_slice]
            self.extend_forward(
                attn_metadata=attn_metadata,
                query=extend_querys,
                key=extend_keys,
                value=extend_values,
                key_cache=k_cache,
                value_cache=v_cache,
                output=extend_outputs,
                cu_seqlens_q=attn_metadata.plugin_metadata.extend_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.plugin_metadata.extend_metadata.max_query_len,
                max_seqlen_k=attn_metadata.plugin_metadata.extend_metadata.max_seq_len,
                min_seqlen_q=1,
                block_table=attn_metadata.plugin_metadata.block_table[
                    num_decodes : num_decodes + num_extends
                ],
                slot_mapping=attn_metadata.plugin_metadata.slot_mapping[
                    num_decodes : num_decodes + num_extends
                ],
                k_scale=k_scale,
                v_scale=v_scale,
            )

        # calculate for decodes
        if num_decodes > 0:
            assert attn_metadata.plugin_metadata.decode_metadata is not None

            num_blocks, block_size, num_kv_heads, head_size = k_cache.shape
            x = 16 // k_cache.element_size()
            k_cache_template = torch.empty(
                [num_blocks, num_kv_heads, head_size // x, block_size, x],
                dtype=k_cache.dtype,
                device="meta",
            )
            v_cache_template = torch.empty(
                [num_blocks, num_kv_heads, block_size // x, head_size, x],
                dtype=v_cache.dtype,
                device="meta",
            )
            new_key_cache = k_cache.view_as(k_cache_template)
            new_value_cache = v_cache.view_as(v_cache_template)

            if self.use_triton_attn:
                self.paged_attention_triton_plugin_mode(
                    q=query[:num_decode_tokens],
                    k=new_key_cache,
                    v=new_value_cache,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    out=output_actual_tokens[:num_decode_tokens],
                    attn_metadata=attn_metadata,
                )   
            else:
                # Qwen only uses gluon pa decode when bs=64
                if num_decodes == 64:
                    self.paged_attention_triton_plugin_mode(
                        q=query[:num_decode_tokens],
                        k_cache=new_key_cache,
                        v_cache=new_value_cache,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        out=output_actual_tokens[:num_decode_tokens],
                        attn_metadata=attn_metadata,
                    )
                else:
                    self.paged_attention_asm_plugin_mode(
                        q=query[:num_decode_tokens],
                        k_cache=new_key_cache,
                        v_cache=new_value_cache,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        num_decodes=num_decodes,
                        num_decode_tokens=num_decode_tokens,
                        out=output_actual_tokens[:num_decode_tokens],
                        attn_metadata=attn_metadata,
                    )

        output = output.view(-1, self.num_heads * self.head_dim)

        return output


def PagedAttentionImplDecoratorForPluginMode(cls):

    method_names = [
        'rope_cache_plugin_mode',
        '_get_cp_mha_gather_cache_views',
        'paged_attention_triton_plugin_mode',
        'paged_attention_asm_plugin_mode',
        'extend_for_sliding_window',
        'extend_forward',
        'forward_impl_plugin_mode',
    ]

    logger.info('Use PagedAttentionImplDecoratorForPluginMode to decorate PagedAttentionImpl')

    # Add all methods to the target class
    for method_name in method_names:
        method = getattr(PagedAttentionImplPluginModeMethods, method_name)
        setattr(cls, method_name, method)

    return cls
