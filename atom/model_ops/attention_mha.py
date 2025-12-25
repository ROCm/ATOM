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
)
from .attention_mla import MLAModules
from aiter.ops.triton.unified_attention import unified_attention
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
        self.kv_scale_float = (
            torch.finfo(torch.float8_e4m3fn).max / torch.finfo(aiter.dtypes.fp8).max
            if self.kv_cache_dtype == "fp8"
            else 1.0
        )
        self.kv_scale = torch.tensor(self.kv_scale_float, dtype=torch.float32)
        self.sinks = sinks
        self.sliding_window = (
            (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
        )
        self.rotary_emb = rotary_emb

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor = None,
        q_scale: torch.Tensor = None,
    ):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        use_triton_unified_attention = (
            self.sliding_window != (-1, -1) or self.head_dim != 128
        )

        # o = torch.ops.aiter.unified_attention_with_output(q, k, v,
        #             self.scale, self.kv_cache_dtype, self.layer_num)
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        context = forward_context.context

        kv_cache_data = forward_context.kv_cache_data
        if attn_metadata.slot_mapping.numel():
            # not dummy run
            k_cache = kv_cache_data[f"layer_{self.layer_num}"].k_cache
            v_cache = kv_cache_data[f"layer_{self.layer_num}"].v_cache
            k_scale = kv_cache_data[f"layer_{self.layer_num}"].k_scale
            v_scale = kv_cache_data[f"layer_{self.layer_num}"].v_scale
        else:
            # dummy run before allocate kv_cache, thus we create manually
            k_cache = v_cache = torch.tensor([])
            k_scale = v_scale = None

        assert self.rotary_emb is None or (
            self.rotary_emb is not None and position is not None
        )
        if k_cache.numel() and v_cache.numel():
            if use_triton_unified_attention:
                k_scale = v_scale = self.kv_scale
                k_cache = k_cache.view(
                    k_cache.shape[0], -1, self.num_kv_heads, self.head_dim
                )
                v_cache = v_cache.view(
                    v_cache.shape[0], -1, self.num_kv_heads, self.head_dim
                )
                if context.is_prefill or self.rotary_emb is None:
                    if self.rotary_emb is not None:
                        q, k = self.rotary_emb(position, q, k)
                    aiter.reshape_and_cache_flash(
                        k,
                        v,
                        k_cache,
                        v_cache,
                        attn_metadata.slot_mapping,
                        (
                            self.kv_cache_dtype
                            if self.kv_cache_dtype.startswith("fp8")
                            else "auto"
                        ),
                        k_scale,
                        v_scale,
                    )
                else:
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
                        flash_layout=True,
                        apply_scale=self.kv_cache_dtype.startswith("fp8"),
                        offs=None,
                        q_out=q,
                        k_out=k,
                        output_zeros=False,
                    )
            else:
                if self.rotary_emb is not None:
                    assert position is not None
                    q, k = self.rotary_emb(position, q, k)
                if self.kv_cache_dtype == "fp8":
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

        if use_triton_unified_attention:
            o = torch.empty_like(q)
            descale_shape = (attn_metadata.cu_seqlens_q.shape[0] - 1, k.shape[1])
            if k_cache.numel() and v_cache.numel():
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
                    window_size=self.sliding_window,
                    block_table=attn_metadata.block_tables,
                    softcap=0,
                    q_descale=None,
                    k_descale=k_scale.expand(descale_shape),
                    v_descale=v_scale.expand(descale_shape),
                    sinks=self.sinks,
                )
        elif context.is_prefill:
            # if context.block_tables is not None:  # prefix cache
            #     k, v = k_cache, v_cache
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
        else:  # decode
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

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
