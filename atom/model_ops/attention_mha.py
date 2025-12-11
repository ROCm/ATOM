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
)
from .attention_mla import MLAModules
from aiter.ops.triton.unified_attention import unified_attention
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        mla_modules: Optional[MLAModules] = None,
        sinks: Optional[nn.Parameter] = None,
        sliding_window: Optional[int] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.layer_num = layer_num
        self.one_scale = torch.tensor(1.0, dtype=torch.float32)
        self.sinks = sinks
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.rotary_emb = rotary_emb
        # kv cache layout
        self.flash_layout = False


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor = None,
    ):

        fwd_args: ForwardContext = get_forward_context()
        kv_cache_data = fwd_args.kv_cache_data

        # dummy run will skip attention in cuda graph capture phase
        if kv_cache_data is None:
            o = torch.empty_like(q)
            return o

        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # rope cache
        q, k, v, k_cache, v_cache, k_scale, v_scale = self.rope_cache(q, k, v, position, fwd_args)           
        
        attn_impl = self.dispatch_backend(fwd_args)
                
        o = attn_impl(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args)
        
        o = o.view(-1, self.num_heads * self.head_dim)

        return o
    
    
    def rope_cache(self, q, k, v, position, fwd_args: ForwardContext, flash_layout=False):
        
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
        
        # TODO: if kv_scale has value, do not use one scale here.
        # if k_scale is None and v_scale is None:
        #    k_scale = v_scale = self.one_scale
        k_scale = v_scale = self.one_scale

        if flash_layout:
            k_cache = k_cache.view(
                k_cache.shape[0], -1, self.num_kv_heads, self.head_dim
            )
            v_cache = v_cache.view(
                v_cache.shape[0], -1, self.num_kv_heads, self.head_dim
            )
        
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
        
        return q, k, v, k_cache, v_cache, k_scale, v_scale
        
        
    def paged_attention_triton(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args: ForwardContext):
        
        attn_metadata = fwd_args.attn_metadata
        
        o = torch.empty_like(q)
        num_seqs, num_q_heads_total, head_size = q.shape
        num_blocks, num_kv_heads, _, block_size, _ = k_cache.shape
        query_group_size = num_q_heads_total // num_kv_heads
        assert num_q_heads_total % num_kv_heads == 0

        max_context_length = (
            min(attn_metadata.max_seqlen_k, self.sliding_window)
            if self.sliding_window > 0
            else attn_metadata.max_seqlen_k
        )
        
        context_partition_size = 256
        if self.sliding_window> 0:
            max_context_length = min(max_context_length, self.sliding_window)
            if max_context_length <= 128:
                context_partition_size = 128
        
        # cdiv
        max_context_partition_num = (max_context_length + context_partition_size - 1) // context_partition_size

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
        
        pa_decode_gluon(
            o,
            q,
            k_cache,
            v_cache,
            attn_metadata.context_lens,
            attn_metadata.block_tables,
            self.scale,
            1, # query_lenth
            max_context_length, # max_context_len
            tl.bfloat16, #compute_type
            None,
            self.one_scale,
            self.one_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=None,
            sinks=self.sinks,
            sliding_window=self.sliding_window,
        )
        
        return o


    def paged_attention_asm(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args: ForwardContext):
        
        attn_metadata = fwd_args.attn_metadata        
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
        
        return o


    def prefill_attention_asm(self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_args: ForwardContext):
        
        # variable lenth attention use key value as input
        attn_metadata = fwd_args.attn_metadata
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
            k_descale=self.one_scale.expand(descale_shape),
            v_descale=self.one_scale.expand(descale_shape),
            sinks=self.sinks,
        )
        
        return o
    
    def dispatch_backend(self, fwd_args: ForwardContext):
        
        ctx = fwd_args.context
        
        if ctx.is_prefill:
            if self.sliding_window or self.sinks is not None:
                return self.prefill_attention_triton
            else:
                return self.prefill_attention_asm
        else:
            if self.sliding_window or self.sinks is not None:
                return self.paged_attention_triton
            else:
                return self.paged_attention_asm