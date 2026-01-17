# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# from flash_attn import flash_attn_with_kvcache
from dataclasses import dataclass

import aiter
import torch
import triton
import triton.language as tl
from aiter.paged_attn import PagedAttention
from torch import nn
from typing import Optional

from atom.utils.forward_context import (
    ForwardContext,
    get_forward_context,
    AttentionMetaData,
)
from .attention_mla import MLAModules
from aiter.ops.triton.unified_attention import unified_attention
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter import dtypes, fused_qk_norm_rope_cache_quant_shuffle
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits
from atom.plugin.prepare import is_plugin_mode, is_vllm

from atom.utils import envs
ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION = envs.ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION
ATOM_ENABLE_QK_NORM_ROPE_FUSION = envs.ATOM_ENABLE_QK_NORM_ROPE_FUSION

class PagedAttentionImpl(nn.Module):
    """
    Attention paged implementation
    """
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        alibi_slopes: list[float] | None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype="bf16",
        logits_soft_cap: float | None = None,
        attn_type = None,
        kv_sharing_target_layer_name: int | None = None,
        layer_num=0,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        q_norm: Optional[torch.nn.Module] = None,
        k_norm: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # for upper framework, it uses head_size in built-in methods
        self.head_size = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.device = 'cuda:' + str(torch.cuda.current_device())
        self.layer_num = layer_num
        self.kv_scale_float = (
            torch.finfo(torch.float8_e4m3fn).max / torch.finfo(aiter.dtypes.fp8).max
            if self.kv_cache_dtype == "fp8"
            else 1.0
        )
        self.kv_scale = torch.tensor(self.kv_scale_float, dtype=torch.float32)
        self.sinks = sinks
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.rotary_emb = rotary_emb
        # kv cache layout
        self.flash_layout = False
        self.q_norm = q_norm
        self.k_norm = k_norm

        # for plugin mode(vllm), the query quant is disabled for now
        if is_vllm():
            self.supports_quant_query_input = False
        # if self.rotary_emb is not None:
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.rotary_emb = ', self.rotary_emb, flush=True)
        # else:
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.rotary_emb is None', flush=True)
        # if self.q_norm is not None:
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.q_norm shape = ', self.q_norm, flush=True)
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.k_norm shape = ', self.k_norm, flush=True)
        # else:
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.q_norm is None', flush=True)
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.k_norm is None', flush=True)
        # if self.sinks is not None:
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.sinks shape = ', self.sinks.shape, flush=True)
        # else:
        #     print('[zejun] ATOM PagedAttentionImpl, init, self.sinks is None', flush=True)

        if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
            assert self.rotary_emb is not None, "rotary_emb must be provided when enabling QK_NORM_ROPE_CACHE_QUANT_FUSION for Qwen models."
            assert self.q_norm is not None, "q_norm must be provided when enabling QK_NORM_ROPE_CACHE_QUANT_FUSION for Qwen models."
            assert self.k_norm is not None, "k_norm must be provided when enabling QK_NORM_ROPE_CACHE_QUANT_FUSION for Qwen models."

    def process_weights_after_loading(self, act_dtype: torch.dtype = torch.bfloat16):
        pass

    def forward_impl_server_mode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor = None,
        q_scale: torch.Tensor=None,
        qkv: torch.Tensor = None,
    ):

        fwd_args: ForwardContext = get_forward_context()

        # dummy run will skip attention in cuda graph capture phase
        if fwd_args.attn_metadata.slot_mapping.numel() == 0:
            o = torch.empty_like(q)
            return o

        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # rope cache
        q, k, v, k_cache, v_cache, k_scale, v_scale = self.rope_cache(q, k, v, qkv, position, fwd_args)
        
        attn_impl = self.dispatch_backend(is_prefill=fwd_args.context.is_prefill, batch_size=fwd_args.context.batch_size)

        o = attn_impl(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args)

        o = o.view(-1, self.num_heads * self.head_dim)

        return o
    
    
    def rope_cache(self, q, k, v, qkv, position, fwd_args: ForwardContext, flash_layout=False):
        
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

        attn_metadata = fwd_args.attn_metadata
        kv_cache_data = fwd_args.kv_cache_data

        k_cache = kv_cache_data[f"layer_{self.layer_num}"].k_cache
        v_cache = kv_cache_data[f"layer_{self.layer_num}"].v_cache
        k_scale = kv_cache_data[f"layer_{self.layer_num}"].k_scale
        v_scale = kv_cache_data[f"layer_{self.layer_num}"].v_scale

        use_triton_attn = (
            self.sliding_window != -1 or self.head_dim != 128
        )
        self.use_triton_attn = use_triton_attn

        if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
            # print('[zejun] ATOM PagedAttentionImpl, call fused_qk_norm_rope_cache_quant_shuffle', flush=True)
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
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=attn_metadata.slot_mapping,
                kv_cache_dtype="auto" if self.kv_cache_dtype == "bf16" else self.kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
            )

            qkv = qkv.view(qkv.shape[0], 
                           -1,
                           self.head_dim)
            q, k, v = qkv.split([self.num_heads,
                                self.num_kv_heads,
                                self.num_kv_heads], dim=1)
        elif use_triton_attn or not ATOM_ENABLE_QK_NORM_ROPE_FUSION:
            if flash_layout:
                k_cache = k_cache.view(
                    k_cache.shape[0], -1, self.num_kv_heads, self.head_dim
                )
                v_cache = v_cache.view(
                    v_cache.shape[0], -1, self.num_kv_heads, self.head_dim
                )

            # TODO: if kv_scale has value, do not use one scale here.
            k_scale = v_scale = self.kv_scale

            # print('[zejun] ATOM PagedAttentionImpl, call fused_qk_rope_reshape_and_cache', flush=True)
            q, k, k_cache, v_cache = fused_qk_rope_reshape_and_cache(
                q,
                k,
                v,
                k_cache,
                v_cache,
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
            assert position is not None
            q, k = self.rotary_emb(position, q, k)
            if self.kv_cache_dtype == "fp8":
                # print('[zejun] ATOM PagedAttentionImpl, call aiter.reshape_and_cache_with_pertoken_quant', flush=True)
                aiter.reshape_and_cache_with_pertoken_quant(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    k_scale,
                    v_scale,
                    attn_metadata.slot_mapping,
                    asm_layout=True,
                )
            else:
                # print('[zejun] ATOM PagedAttentionImpl, call aiter.reshape_and_cache', flush=True)
                aiter.reshape_and_cache(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    attn_metadata.slot_mapping,
                    kv_cache_dtype="auto",
                    k_scale=None,
                    v_scale=None,
                    asm_layout=True,
                )
        
        return q, k, v, k_cache, v_cache, k_scale, v_scale
        
        
    def paged_attention_triton(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args: ForwardContext):
        
        attn_metadata = fwd_args.attn_metadata
        
        o = torch.empty_like(q)
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
            attn_metadata.context_lens,
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


    def paged_attention_asm(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args: ForwardContext):
        
        attn_metadata = fwd_args.attn_metadata
        
        # import os
        # pid = os.getpid()
        # print(f'[zejun][pid={pid}] paged_attention_asm, q.shape = {q.shape}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, q.dtype = {q.dtype}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, q.mean  = {q.mean()}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, q.std   = {q.std()}', flush=True)
        
        # print(f'[zejun][pid={pid}] paged_attention_asm, k_cache.shape = {k_cache.shape}', flush=True)
        # # print(f'[zejun][pid={pid}] paged_attention_asm, k_cache.mean  = {k_cache.mean()}', flush=True)
        # # print(f'[zejun][pid={pid}] paged_attention_asm, k_cache.std   = {k_cache.std()}', flush=True)
        
        # print(f'[zejun][pid={pid}] paged_attention_asm, v_cache.shape = {v_cache.shape}', flush=True)
        # # print(f'[zejun][pid={pid}] paged_attention_asm, v_cache.mean  = {v_cache.mean()}', flush=True)
        # # print(f'[zejun][pid={pid}] paged_attention_asm, v_cache.std   = {v_cache.std()}', flush=True)
        
        # print(f'[zejun][pid={pid}] paged_attention_asm, attn_metadata.block_tables = {attn_metadata.block_tables}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, attn_metadata.context_lens = {attn_metadata.context_lens}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, attn_metadata.block_tables.stride(0) = {attn_metadata.block_tables.stride(0)}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, k_scale = {k_scale}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, v_scale = {v_scale}', flush=True)

        o = aiter.pa_fwd_asm(
            q,
            k_cache,
            v_cache,
            attn_metadata.block_tables,
            attn_metadata.context_lens,
            attn_metadata.block_tables.stride(0),
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=None,
            high_precision=0,
        )
        
        # print(f'[zejun][pid={pid}] paged_attention_asm, o.shape = {o.shape}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, o.dtype = {o.dtype}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, o.mean  = {o.mean()}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm, o.std   = {o.std()}', flush=True)

        # import sys
        # sys.exit(0)
        
        return o


    def prefill_attention_asm(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args: ForwardContext):
        
        # variable lenth attention use key value as input
        attn_metadata = fwd_args.attn_metadata
        sliding_window = (self.sliding_window, 0, 0) if self.sliding_window is not None else (-1, -1, 0)

        # import os
        # pid = os.getpid()
        # print('[zejun][pid=', pid, '] prefill_attention_asm, q shape = ', q.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, q mean  = ', q.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, q std   = ', q.std(), flush=True)

        # print('[zejun][pid=', pid, '] prefill_attention_asm, k shape = ', k.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, k mean  = ', k.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, k std   = ', k.std(), flush=True)

        # print('[zejun][pid=', pid, '] prefill_attention_asm, v shape = ', v.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, v mean  = ', v.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, v std   = ', v.std(), flush=True)

        # print('[zejun][pid=', pid, '] prefill_attention_asm, attn_metadata.cu_seqlens_q = ', attn_metadata.cu_seqlens_q, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, attn_metadata.cu_seqlens_k = ', attn_metadata.cu_seqlens_k, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, attn_metadata.max_seqlen_q = ', attn_metadata.max_seqlen_q, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, attn_metadata.max_seqlen_k = ', attn_metadata.max_seqlen_k, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, attn_metadata.min_seqlen_q = ', attn_metadata.min_seqlen_q, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, attn_metadata.dropout_p    = ', attn_metadata.dropout_p, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, self.scale                 = ', self.scale, flush=True)

        o = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            min_seqlen_q=attn_metadata.min_seqlen_q,
            dropout_p=attn_metadata.dropout_p,
            softmax_scale=self.scale,
            causal=True,
            window_size=sliding_window,
            sink_ptr=self.sinks,
        )

        # print('[zejun][pid=', pid, '] prefill_attention_asm, o shape = ', o.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, o mean  = ', o.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm, o std   = ', o.std(), flush=True)

        # import sys
        # sys.exit(0)

        return o
        

    def prefill_attention_triton(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args: ForwardContext):
        
        # the unified_attention supports both prefill attention and decode attention, but it only support
        # flash-layout kv_cache.
        #
        # key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        #
        # if the paged_attention supports only non-flash-layout kv_cache and kv_cache is also cached as 
        # non-flash-layout in rope_cache phase, the unified_attention should use key and value as kv_cache
        # with block_size 1 and fake block_table.
        #
        # key:    [num_blocks, 1, num_kv_heads, head_size]
        # value:  [num_blocks, 1, num_kv_heads, head_size]

        attn_metadata = fwd_args.attn_metadata
        ctx = fwd_args.context
        
        block_tables = attn_metadata.block_tables
        if ctx.is_prefill:
            k_cache = k.unsqueeze(1)
            v_cache = v.unsqueeze(1)
            block_tables = attn_metadata.fake_block_tables
        
        o = torch.empty_like(q)
        descale_shape = (attn_metadata.cu_seqlens_q.shape[0] - 1, k.shape[1])
        sliding_window = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
        unified_attention(
            q,
            k_cache,
            v_cache,
            o,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            seqused_k=attn_metadata.context_lens,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=None,
            window_size=sliding_window,
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=self.kv_scale.expand(descale_shape),
            v_descale=self.kv_scale.expand(descale_shape),
            sinks=self.sinks,
        )
        
        return o
    
    # for both server and plugin mode
    def dispatch_backend(self, is_prefill: bool, batch_size: int, is_plugin_mode: bool = False):
        # here maintain a suffix for calling ops for plugin mode, aligned with the dispatch
        # policy working under the server mode
        suffix = "_plugin_mode" if is_plugin_mode else ""
        # print('[zejun] ATOM PagedAttentionImpl, dispatch_backend, is_prefill = ', is_prefill, '. batch_size = ', batch_size, flush=True)
        # print('[zejun] ATOM PagedAttentionImpl, dispatch_backend, batch_size = ', batch_size, flush=True)
        # print('[zejun] ATOM PagedAttentionImpl, dispatch_backend, is_plugin_mode = ', is_plugin_mode, flush=True)
        # print('[zejun] ATOM PagedAttentionImpl, dispatch_backend, self.use_triton_attn = ', self.use_triton_attn, flush=True)
        if is_prefill:
            # if self.use_triton_attn:
            #     return getattr(self, f"prefill_attention_triton{suffix}")
            # else:
                # print('[zejun] ATOM PagedAttentionImpl, call prefill_attention_asm', flush=True)
                return getattr(self, f"prefill_attention_asm{suffix}")
        else:
            if self.use_triton_attn:
                # print('[zejun] ATOM PagedAttentionImpl, call paged_attention_triton', flush=True)
                return getattr(self, f"paged_attention_triton{suffix}")
            else:
                # Qwen only uses gluon pa decode when bs=64
                if batch_size == 64:
                    # print('[zejun] ATOM PagedAttentionImpl, call paged_attention_triton', flush=True)
                    return getattr(self, f"paged_attention_triton{suffix}")
                else:
                    # print('[zejun] ATOM PagedAttentionImpl, call paged_attention_asm', flush=True)
                    return getattr(self, f"paged_attention_asm{suffix}")

    # for plugin mode
    def rope_cache_plugin_mode(self,
                               q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               qkv: torch.Tensor,
                               position: torch.Tensor,
                               attention_metadata: AttentionMetaData,
                               k_cache: torch.Tensor,
                               v_cache: torch.Tensor,
                               k_scale: torch.Tensor,
                               v_scale: torch.Tensor,
                               flash_layout: bool = False):
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

        use_triton_attn = (
            self.sliding_window != -1 or self.head_dim != 128
        )
        self.use_triton_attn = use_triton_attn

        # print('[zejun] ATOM, use_triton_attn = ', use_triton_attn, flush=True)
        # print('[zejun] ATOM, self.sliding_window = ', self.sliding_window, flush=True)
        # print('[zejun] ATOM, self.head_dim = ', self.head_dim, flush=True)
        # print('[zejun] ATOM, ATOM_ENABLE_QK_NORM_ROPE_FUSION = ', ATOM_ENABLE_QK_NORM_ROPE_FUSION, flush=True)

        # FIXME: disable all the fusion here
        # TODO: pass the qkv through the q,k,v
        # if ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION:
        if 0:
            print('[zejun] ATOM PagedAttentionImpl, call fused_qk_norm_rope_cache_quant_shuffle', flush=True)
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
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=attn_metadata.slot_mapping,
                kv_cache_dtype="auto" if self.kv_cache_dtype == "bf16" else self.kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
            )

            qkv = qkv.view(qkv.shape[0], 
                        -1,
                        self.head_dim)
            q, k, v = qkv.split([self.num_heads,
                                self.num_kv_heads,
                                self.num_kv_heads], dim=1)
        # elif use_triton_attn or not ATOM_ENABLE_QK_NORM_ROPE_FUSION:
        elif 0:
            if flash_layout:
                k_cache = k_cache.view(
                    k_cache.shape[0], -1, self.num_kv_heads, self.head_dim
                )
                v_cache = v_cache.view(
                    v_cache.shape[0], -1, self.num_kv_heads, self.head_dim
                )

            # TODO: if kv_scale has value, do not use one scale here.
            k_scale = v_scale = self.kv_scale

            # TODO: run here
            q, k, k_cache, v_cache = fused_qk_rope_reshape_and_cache(
                q,
                k,
                v,
                k_cache,
                v_cache,
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
            assert position is not None
            q, k = self.rotary_emb(position, q, k)
            if self.kv_cache_dtype == "fp8":
                # print('[zejun] ATOM PagedAttentionImpl, call aiter.reshape_and_cache_with_pertoken_quant', flush=True)
                aiter.reshape_and_cache_with_pertoken_quant(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    k_scale,
                    v_scale,
                    attn_metadata.slot_mapping,
                    asm_layout=True,
                )
            else:
                # print('[zejun] ATOM PagedAttentionImpl, call aiter.reshape_and_cache', flush=True)
                aiter.reshape_and_cache(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    attn_metadata.slot_mapping,
                    kv_cache_dtype="auto",
                    k_scale=None,
                    v_scale=None,
                    asm_layout=True,
                )
        
        return q, k, v, k_cache, v_cache, k_scale, v_scale

    # for plugin mode
    def paged_attention_triton_plugin_mode(self, q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata: AttentionMetaData):
        
        o = torch.empty_like(q)
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
            attn_metadata.context_lens,
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

    # for plugin mode
    def paged_attention_asm_plugin_mode(self, q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata: AttentionMetaData):

        # import os
        # pid = os.getpid()
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, q.shape = {q.shape}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, q.dtype = {q.dtype}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, q.mean  = {q.mean()}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, q.std   = {q.std()}', flush=True)
        
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, k_cache.shape = {k_cache.shape}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, k_cache.mean  = {k_cache.mean()}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, k_cache.std   = {k_cache.std()}', flush=True)
        
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, v_cache.shape = {v_cache.shape}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, v_cache.mean  = {v_cache.mean()}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, v_cache.std   = {v_cache.std()}', flush=True)
        
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, attn_metadata.block_tables = {attn_metadata.block_tables}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, attn_metadata.context_lens = {attn_metadata.context_lens}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, attn_metadata.block_tables.stride(0) = {attn_metadata.block_tables.stride(0)}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, k_scale = {k_scale}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, v_scale = {v_scale}', flush=True)

        o = aiter.pa_fwd_asm(
            q,
            k_cache,
            v_cache,
            attn_metadata.block_tables,
            attn_metadata.context_lens,
            attn_metadata.block_tables.stride(0),
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=None,
            high_precision=0,
        )

        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, o.shape = {o.shape}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, o.dtype = {o.dtype}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, o.mean  = {o.mean()}', flush=True)
        # print(f'[zejun][pid={pid}] paged_attention_asm_plugin_mode, o.std   = {o.std()}', flush=True)

        # import sys
        # sys.exit(0)
        return o

    # for plugin mode
    def prefill_attention_asm_plugin_mode(self, q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata: AttentionMetaData):

        # variable lenth attention use key value as input
        # import os
        # pid = os.getpid()
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, q shape = ', q.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, q mean  = ', q.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, q std   = ', q.std(), flush=True)

        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, k shape = ', k.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, k mean  = ', k.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, k std   = ', k.std(), flush=True)

        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, v shape = ', v.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, v mean  = ', v.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, v std   = ', v.std(), flush=True)

        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, attn_metadata.cu_seqlens_q = ', attn_metadata.cu_seqlens_q, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, attn_metadata.cu_seqlens_k = ', attn_metadata.cu_seqlens_k, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, attn_metadata.max_seqlen_q = ', attn_metadata.max_seqlen_q, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, attn_metadata.max_seqlen_k = ', attn_metadata.max_seqlen_k, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, attn_metadata.min_seqlen_q = ', attn_metadata.min_seqlen_q, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, attn_metadata.dropout_p    = ', attn_metadata.dropout_p, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, self.scale                 = ', self.scale, flush=True)

        o = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            min_seqlen_q=attn_metadata.min_seqlen_q,
            dropout_p=attn_metadata.dropout_p,
            softmax_scale=self.scale,
            causal=True,
        )

        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, o shape = ', o.shape, flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, o mean  = ', o.mean(), flush=True)
        # print('[zejun][pid=', pid, '] prefill_attention_asm_plugin_mode, o std   = ', o.std(), flush=True)

        # import sys
        # sys.exit(0)
        return o

    # for plugin mode
    def prefill_attention_triton_plugin_mode(self, q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata: AttentionMetaData):
        # the unified_attention supports both prefill attention and decode attention, but it only support
        # flash-layout kv_cache.
        #
        # key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        #
        # if the paged_attention supports only non-flash-layout kv_cache and kv_cache is also cached as 
        # non-flash-layout in rope_cache phase, the unified_attention should use key and value as kv_cache
        # with block_size 1 and fake block_table.
        #
        # key:    [num_blocks, 1, num_kv_heads, head_size]
        # value:  [num_blocks, 1, num_kv_heads, head_size]

        ctx = attn_metadata.context

        block_tables = attn_metadata.block_tables
        if ctx.is_prefill:
            k_cache = k.unsqueeze(1)
            v_cache = v.unsqueeze(1)
            block_tables = attn_metadata.fake_block_tables

        o = torch.empty_like(q)
        descale_shape = (attn_metadata.cu_seqlens_q.shape[0] - 1, k.shape[1])
        sliding_window = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, -1)
        unified_attention(
            q,
            k_cache,
            v_cache,
            o,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            seqused_k=attn_metadata.context_lens,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=None,
            window_size=sliding_window,
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=self.kv_scale.expand(descale_shape),
            v_descale=self.kv_scale.expand(descale_shape),
            sinks=self.sinks,
        )

        return o

    # for plugin mode
    def forward_impl_plugin_mode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata = None,
        position: torch.Tensor = None,
        q_scale: torch.Tensor=None,
        qkv: torch.Tensor = None,
    ):
        # dummy run will skip attention in cuda graph capture phase
        if attn_metadata is None:
            o = torch.empty_like(query)
            return o

        o: torch.Tensor
        q = query.view(-1, self.num_heads, self.head_dim)
        k = key.view(-1, self.num_kv_heads, self.head_dim)
        v = value.view(-1, self.num_kv_heads, self.head_dim)

        original_k_cache, original_v_cache = kv_cache.unbind(0)
        num_blocks, block_size, num_kv_heads, head_size = original_k_cache.shape

        # FIXME: create kv scale according to the num_blocks
        if self.kv_cache_dtype == "fp8":
            if self.k_scale is None or self.v_scale is None:
                self.kv_scale = torch.zeros(
                    2,
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    dtype=dtypes.fp32,
                    device=self.device,
                )
            # update the layer kv scale tensor
            self.k_scale = self.kv_scale[0]
            self.v_scale = self.kv_scale[1]
            k_scale = layer.k_scale = self.k_scale
            v_scale = layer.v_scale = self.v_scale

            # print("[zejun] ATOM forward_impl_plugin_mode k_scale shape = ", k_scale.shape, flush=True)
            # print("[zejun] ATOM forward_impl_plugin_mode k_scale data ptr = ", k_scale.data_ptr(), flush=True)
            # print("[zejun] ATOM forward_impl_plugin_mode v_scale shape = ", v_scale.shape, flush=True)
            # print("[zejun] ATOM forward_impl_plugin_mode v_scale data ptr = ", v_scale.data_ptr(), flush=True)

        # here do 2 things:
        # 1. change the kv cache layout, which is required by the asm kernel 
        # and the cache flash kernel
        # 2. use meta tensor to make sure the kv cache tensor, 
        # passed into following kernel, has the same data ptr as 
        # the originial one. It means they share the same storage impl
        x = 16 // original_k_cache.element_size()
        k_cache_template = torch.empty(
            [num_blocks, num_kv_heads, head_size // x, block_size, x],
            dtype=original_k_cache.dtype,
            device="meta",
        )
        v_cache_template = torch.empty(
            [num_blocks, num_kv_heads, head_size, block_size],
            dtype=original_v_cache.dtype,
            device="meta",
        )
        k_cache = original_k_cache.view_as(k_cache_template)
        v_cache = original_v_cache.view_as(v_cache_template)

        if self.kv_cache_dtype == "fp8":
            target_dtype = dtypes.d_dtypes[self.kv_cache_dtype]
            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)


        result = self.rope_cache_plugin_mode(q=q,
                                             k=k,
                                             v=v,
                                             qkv=qkv,
                                             position=attn_metadata.context.positions,
                                             attention_metadata=attn_metadata,
                                             k_cache=k_cache,
                                             v_cache=v_cache,
                                             k_scale=self.k_scale,
                                             v_scale=self.v_scale,
                                             flash_layout=False)
        (q, k, v, k_cache, v_cache, k_scale, v_scale) = result

        attn_impl = self.dispatch_backend(is_prefill=attn_metadata.context.is_prefill,
                                          batch_size=attn_metadata.context.batch_size,
                                          is_plugin_mode=True)

        o = attn_impl(q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata)

        o = o.view(-1, self.num_heads * self.head_dim)

        return o

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor = None,
        attn_metadata = None,
        position: torch.Tensor = None,
        q_scale: Optional[torch.Tensor]=None,
        qkv: torch.Tensor = None,
        **kwargs,
    ):
        if is_plugin_mode():
            # TODO: for plugin mode, there is no qkv tensor passed
            # but we can use query to represent the qkv tensor
            o = self.forward_impl_plugin_mode(layer=layer,
                                              query=query,
                                              key=key,
                                              value=value,
                                              kv_cache=kv_cache,
                                              attn_metadata=attn_metadata,
                                              position=position,
                                              q_scale=q_scale,
                                              qkv=qkv)
        else:
            o = self.forward_impl_server_mode(q=query,
                                              k=key,
                                              v=value,
                                              position=position,
                                              q_scale=q_scale,
                                              qkv=qkv)

        return o
