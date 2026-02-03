# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import aiter
import torch
from aiter import fused_qk_norm_rope_cache_quant_shuffle
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits
from aiter.ops.triton.unified_attention import unified_attention
from atom.config import get_current_atom_config
from atom.utils.forward_context import ForwardContext, get_forward_context
from torch import nn
from aiter.dist.parallel_state import get_tp_group

from .attention_mla import MLAModules


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
        q_norm: Optional[torch.nn.Module] = None,
        k_norm: Optional[torch.nn.Module] = None,
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
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position: torch.Tensor = None,
        q_scale: torch.Tensor = None,
        qkv: torch.Tensor = None,
    ):

        fwd_ctx: ForwardContext = get_forward_context()

        # dummy run will skip attention in cuda graph capture phase
        if fwd_ctx.attn_metadata.slot_mapping.numel() == 0:
            o = torch.empty_like(q)
            return o

        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # rope cache
        q, k, v, k_cache, v_cache, k_scale, v_scale = self.rope_cache(
            q, k, v, qkv, position, fwd_ctx
        )

        attn_impl = self.dispatch_backend(fwd_ctx)

        o = attn_impl(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx)

        o = o.view(-1, self.num_heads * self.head_dim)

        return o

    def rope_cache(self, q, k, v, qkv, position, fwd_ctx: ForwardContext):
        attn_metadata = fwd_ctx.attn_metadata
        kv_cache_data = fwd_ctx.kv_cache_data

        k_cache = kv_cache_data[f"layer_{self.layer_num}"].k_cache
        v_cache = kv_cache_data[f"layer_{self.layer_num}"].v_cache
        k_scale = kv_cache_data[f"layer_{self.layer_num}"].k_scale
        v_scale = kv_cache_data[f"layer_{self.layer_num}"].v_scale

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
                k_cache=k_cache,
                v_cache=v_cache,
                slot_mapping=attn_metadata.slot_mapping,
                kv_cache_dtype=(
                    "auto" if self.kv_cache_dtype == "bf16" else self.kv_cache_dtype
                ),
                k_scale=k_scale,
                v_scale=v_scale,
            )

            qkv = qkv.view(qkv.shape[0], -1, self.head_dim)
            q, k, v = qkv.split(
                [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=1
            )
        elif use_triton_attn and self.rotary_emb is not None:
            k_scale = v_scale = self.kv_scale

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
                flash_layout=False,
                apply_scale=self.kv_cache_dtype.startswith("fp8"),
                offs=None,
                q_out=q,
                k_out=k,
                output_zeros=False,
            )
        else:
            # for asm paged attention
            asm_layout = True
            if use_triton_attn:
                asm_layout = False
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
                    k_cache,
                    v_cache,
                    k_scale,
                    v_scale,
                    attn_metadata.slot_mapping,
                    asm_layout=asm_layout,
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
                    asm_layout=asm_layout,
                )

        return q, k, v, k_cache, v_cache, k_scale, v_scale

    def paged_attention_triton(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        attn_metadata = fwd_ctx.attn_metadata

        o = torch.empty_like(q)
        num_seqs = attn_metadata.context_lens.shape[0]
        # print("num seqs: ", num_seqs, flush=True)
        _, num_q_heads_total, head_size = q.shape
        num_blocks, num_kv_heads, _, block_size, _ = k_cache.shape
        # assuem all query have same length
        query_group_size = attn_metadata.max_seqlen_q * (num_q_heads_total // num_kv_heads)
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
        exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
        max_logits = torch.empty(
            intermediate_shape, dtype=torch.float32, device=q.device
        )
        temporary_output = torch.empty(
            *intermediate_shape,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )

        if k_scale is not None and k_scale.numel() > 1:
            k_scale = k_scale.unsqueeze(-1)
            v_scale = v_scale.unsqueeze(-1)

        compute_type = (
            torch.bfloat16
            if self.kv_cache_dtype == "bf16"  # or per_tensor
            else aiter.dtypes.fp8
        )
        # if get_tp_group().rank_in_group == 0:
        #     torch.save(q, "query.pt")
        #     torch.save(k_cache, "key_cache.pt")
        #     torch.save(v_cache, "value_cache.pt")
        #     torch.save(attn_metadata.context_lens, "context_lens.pt")
        #     torch.save(attn_metadata.block_tables, "block_tables.pt")
        #     torch.save(exp_sums, "exp_sum.pt")
        #     torch.save(max_logits, "max_logits.pt")
        #     torch.save(temporary_output, "temporary_output.pt")
        #     print("scale: ", self.scale)
        #     print("max seqlen_q: ", attn_metadata.max_seqlen_q)
        #     print("max_context_partition_num: ", max_context_partition_num)
        #     print("context_partition_size: ", context_partition_size)
        #     print("self.sink: ", self.sinks)
        #     print("sekf.sliding_window: ", self.sliding_window, flush=True)
        torch.ops.aiter.pa_decode_gluon(
            o,
            q,
            k_cache,
            v_cache,
            attn_metadata.context_lens,
            attn_metadata.block_tables,
            self.scale,
            attn_metadata.max_seqlen_q,
            max_context_partition_num,
            context_partition_size,
            compute_type,
            None,  # q_scale
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
        # if get_tp_group().rank_in_group == 0:
        #     torch.cuda.synchronize()
        #     import os
        #     os.abort()
        return o

    def paged_attention_asm(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        attn_metadata = fwd_ctx.attn_metadata
        o = aiter.pa_fwd_asm(
            q,
            k_cache,
            v_cache,
            attn_metadata.block_tables,
            attn_metadata.context_lens,
            attn_metadata.block_tables.stride(0),
            max_qlen=attn_metadata.max_seqlen_q,
            K_QScale=k_scale,
            V_QScale=v_scale,
            out_=None,
            qo_indptr=attn_metadata.cu_seqlens_q,
            high_precision=0,
        )

        return o

    def paged_attention_persistent_asm(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):
        attn_metadata = fwd_ctx.attn_metadata
        output = torch.empty_like(q)

        aiter.pa_persistent_fwd(
            Q=q,
            K=k_cache,
            V=v_cache,
            output=output,
            max_qlen=attn_metadata.max_seqlen_q,
            qo_indptr=attn_metadata.cu_seqlens_q,
            kv_indptr=attn_metadata.kv_indptr,
            kv_indices=attn_metadata.kv_indices,
            context_lens=attn_metadata.context_lens,
            K_QScale=k_scale,
            V_QScale=v_scale,
            work_indptr=attn_metadata.work_indptr,
            work_info=attn_metadata.work_info_set,
            reduce_indptr=attn_metadata.reduce_indptr,
            reduce_final_map=attn_metadata.reduce_final_map,
            reduce_partial_map=attn_metadata.reduce_partial_map,
            softmax_scale=self.scale,
            mask=1,
        )

        return output

    def prefill_attention(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        # variable lenth attention use key value as input
        attn_metadata = fwd_ctx.attn_metadata
        sliding_window = (
            (self.sliding_window, 0, 0)
            if self.sliding_window is not None
            else (-1, -1, 0)
        )
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

        return o

    def prefill_attention_triton(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

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

        attn_metadata = fwd_ctx.attn_metadata
        ctx = fwd_ctx.context

        block_tables = attn_metadata.block_tables
        if ctx.is_prefill:
            k_cache = k.unsqueeze(1)
            v_cache = v.unsqueeze(1)
            block_tables = attn_metadata.fake_block_tables

        o = torch.empty_like(q)
        descale_shape = (attn_metadata.cu_seqlens_q.shape[0] - 1, k.shape[1])
        sliding_window = (
            (self.sliding_window - 1, 0)
            if self.sliding_window is not None
            else (-1, -1)
        )
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

    def dispatch_backend(self, fwd_ctx: ForwardContext):

        ctx = fwd_ctx.context

        if ctx.is_prefill:
            return self.prefill_attention
        else:
            if self.use_triton_attn:
                return self.paged_attention_triton
            else:
                # Only use pa persistent when block_size == 1024
                atom_config = get_current_atom_config()
                if atom_config.kv_cache_block_size == 1024:
                    return self.paged_attention_persistent_asm
                return self.paged_attention_asm
