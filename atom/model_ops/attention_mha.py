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

from .attention_mla import MLAModules

import logging

from atom.plugin.prepare import is_plugin_mode, is_vllm
from atom.plugin.attention_mha import PagedAttentionImplDecoratorForPluginMode
from atom.utils.decorators import mark_trace
from atom.model_ops.base_attention import cp_mha_gather_cache

logger = logging.getLogger("atom")


@PagedAttentionImplDecoratorForPluginMode
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
        attn_type=None,
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
        self.alibi_slopes = alibi_slopes
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.device = "cuda:" + str(torch.cuda.current_device())
        self.layer_num = layer_num
        self.kv_scale_float = (
            torch.finfo(torch.float8_e4m3fn).max / torch.finfo(aiter.dtypes.fp8).max
            if self.kv_cache_dtype == "fp8"
            else 1.0
        )
        self.kv_scale = torch.tensor(self.kv_scale_float, dtype=torch.float32)
        self.per_token_quant = True
        self.sinks = sinks
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm
        # Set by the attention backend's build_kv_cache_tensor when KV cache is
        # allocated in flash layout [num_blocks, block_size, num_kv_heads, head_dim]
        # for aiter triton unified_attention. AiterBackend keeps this False.
        self.use_flash_layout = False

        # for plugin mode(vllm), the query quant is disabled for now
        if is_vllm():
            self.supports_quant_query_input = False

    def forward_impl_server_mode(
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
        if fwd_ctx.context.is_dummy_run:
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

    @mark_trace(prefix="rope_cache", torch_compile=False)
    def rope_cache(self, q, k, v, qkv, position, fwd_ctx: ForwardContext):
        attn_metadata = fwd_ctx.attn_metadata
        kv_cache_data = fwd_ctx.kv_cache_data

        k_cache = kv_cache_data[f"layer_{self.layer_num}"].k_cache
        v_cache = kv_cache_data[f"layer_{self.layer_num}"].v_cache
        k_scale = kv_cache_data[f"layer_{self.layer_num}"].k_scale
        v_scale = kv_cache_data[f"layer_{self.layer_num}"].v_scale

        # MTP MHA must go through triton/gluon; aiter ASM non-persistent path may have some unexpected behavior.
        use_triton_attn = (
            self.sliding_window != -1
            or self.head_dim != 128
            or self.num_heads == self.num_kv_heads
        )
        self.use_triton_attn = use_triton_attn

        if (
            self.rotary_emb is not None
            and self.q_norm is not None
            and self.k_norm is not None
        ):
            from atom.model_ops.layernorm import GemmaRMSNorm

            if isinstance(self.q_norm, GemmaRMSNorm):
                # GemmaRMSNorm (1+w) path — use the Triton fused kernel
                from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
                    triton_fused_norm_rope_cache,
                )

                # qkv is a packed [q, k, v] tensor — split
                q_size = self.num_heads * self.head_dim
                kv_size = self.num_kv_heads * self.head_dim
                q_raw, k_raw, v_raw = torch.split(
                    qkv, [q_size, kv_size, kv_size], dim=-1
                )
                # Reshape V cache to SHUFFLE layout for the Triton kernel
                x = 16 // k_cache.element_size()
                if k_cache.dim() == 5 and v_cache.dim() == 4:
                    n, nh, hd, bs = v_cache.shape
                    v_cache_shuffle = v_cache.view(n, nh, bs // x, hd, x)
                else:
                    v_cache_shuffle = v_cache
                q, k = triton_fused_norm_rope_cache(
                    q_raw,
                    k_raw,
                    v_raw,
                    position,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    rotary_emb=self.rotary_emb,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    k_cache=k_cache,
                    v_cache=v_cache_shuffle,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    slot_mapping=attn_metadata.slot_mapping,
                    kv_cache_dtype=self.kv_cache_dtype,
                )
                q = q.view(-1, self.num_heads, self.head_dim)
                k = k.view(-1, self.num_kv_heads, self.head_dim)
                v = v_raw.view(-1, self.num_kv_heads, self.head_dim)
            else:
                # Standard RMSNorm — use existing aiter kernel
                # fused_qk_norm_rope_cache_quant_shuffle expects V cache layout
                # [num_blocks, num_kv_heads, block_size//x, head_size, x]
                x = 16 // k_cache.element_size()
                if k_cache.dim() == 5 and v_cache.dim() == 4:
                    n, nh, hd, bs = v_cache.shape
                    v_cache_shuffle = v_cache.view(n, nh, bs // x, hd, x)
                else:
                    v_cache_shuffle = v_cache
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
                    v_cache=v_cache_shuffle,
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
            self.per_token_quant = False
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
                flash_layout=self.use_flash_layout,
                apply_scale=self.kv_cache_dtype.startswith("fp8"),
                offs=None,
                q_out=q,
                k_out=k,
                output_zeros=False,
            )
        else:
            # for asm paged attention
            asm_layout = True
            if use_triton_attn and v_cache.dim() != 5:
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

        # Prefix cache hit: gather cached KV from paged cache and concat with new tokens
        if attn_metadata.has_cached:
            q, k, v, k_cache, v_cache, k_scale, v_scale = (
                self._gather_prefix_and_concat_kv(
                    q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata
                )
            )

        return q, k, v, k_cache, v_cache, k_scale, v_scale

    def _gather_prefix_and_concat_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        attn_metadata,
    ):
        """
        When prefix cache hits, gather full KV (cached + new) from paged cache in
        one pass. New tokens are already written by fused_qk_rope_reshape_and_cache.
        Same flow as gather_kv_b_proj: write new first, then read cached+new together.
        """
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        total_tokens = attn_metadata.total_kv
        bs = attn_metadata.context_lens.shape[0]
        token_to_batch = torch.repeat_interleave(
            torch.arange(
                bs, dtype=torch.int32, device=attn_metadata.context_lens.device
            ),
            attn_metadata.context_lens.long(),
        )

        num_kv_heads = k.shape[1]
        head_dim = k.shape[2]

        k_full = torch.empty(
            (total_tokens, num_kv_heads, head_dim), dtype=k.dtype, device=k.device
        )
        v_full = torch.empty(
            (total_tokens, num_kv_heads, head_dim), dtype=k.dtype, device=k.device
        )

        # Convert cache for cp_mha_gather_cache
        # fused_qk_norm_rope_cache_quant_shuffle: K [n, nh, hd//x, bs, x], V [n, nh, bs//x, hd, x] (SHUFFLE)
        # fused_qk_rope_reshape_and_cache: K [n, nh, hd//x, bs, x], V [n, nh, hd, bs] -> NHD
        if k_cache.dim() == 5:
            x = 16 // k_cache.element_size()
            n, nh, _, block_size, _ = k_cache.shape
            if v_cache.dim() == 4:
                # fused_qk_norm_rope_cache_quant_shuffle: V data in [n, nh, bs//x, hd, x] layout
                use_shuffle = True
                k_cache_gather = k_cache
                v_cache_gather = v_cache.view(n, nh, block_size // x, head_dim, x)
            else:
                # fused_qk_rope_reshape_and_cache: V [n, nh, hd, bs] -> NHD
                use_shuffle = False
                k_cache_gather = (
                    k_cache.permute(0, 3, 1, 2, 4)
                    .contiguous()
                    .view(n, block_size, nh, head_dim)
                )
                v_cache_gather = v_cache.permute(0, 3, 1, 2).contiguous()
        else:
            use_shuffle = False
            k_cache_gather = k_cache
            v_cache_gather = v_cache
            block_size = k_cache.shape[1]

        block_tables = attn_metadata.block_tables
        per_token_quant = (
            self.kv_cache_dtype.startswith("fp8")
            and k_scale is not None
            and v_scale is not None
            and k_scale.numel() > 1
            and v_scale.numel() > 1
        )
        cp_mha_gather_cache(
            key_cache=k_cache_gather,
            value_cache=v_cache_gather,
            key=k_full,
            value=v_full,
            block_tables=block_tables,
            k_scales=k_scale,
            v_scales=v_scale,
            cu_seqlens_kv=cu_seqlens_k,
            token_to_batch=token_to_batch,
            seq_starts=attn_metadata.seq_starts,
            dequant=self.kv_cache_dtype.startswith("fp8"),
            kv_cache_layout="SHUFFLE" if use_shuffle else "NHD",
            total_tokens=total_tokens,
            per_token_quant=per_token_quant,
        )

        return q, k_full, v_full, k_cache, v_cache, k_scale, v_scale

    @mark_trace(prefix="paged_attention_triton", torch_compile=False)
    def paged_attention_triton(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        attn_metadata = fwd_ctx.attn_metadata

        o = torch.empty_like(q)
        num_seqs = attn_metadata.context_lens.shape[0]

        if self.use_flash_layout:
            sliding_window = (
                (self.sliding_window - 1, 0) if self.sliding_window > 0 else (-1, -1)
            )

            # KV cache is already in flash layout (4D), allocated by
            # TritonMHAMetadataBuilder.build_kv_cache_tensor.
            nkv = k_cache.shape[2]
            descale_shape = (num_seqs, nkv)

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
                block_table=attn_metadata.block_tables,
                softcap=0,
                q_descale=None,
                k_descale=self.kv_scale.expand(descale_shape),
                v_descale=self.kv_scale.expand(descale_shape),
                sinks=self.sinks,
            )
        else:
            _, num_q_heads_total, head_size = q.shape
            num_blocks, num_kv_heads, _, block_size, _ = k_cache.shape
            query_group_size = attn_metadata.max_seqlen_q * (
                num_q_heads_total // num_kv_heads
            )
            assert num_q_heads_total % num_kv_heads == 0

            max_context_partition_num = get_recommended_splits(num_seqs, num_kv_heads)

            context_partition_size = 256
            if self.sliding_window > 0:
                max_context_partition_num = 1
                context_partition_size = 128

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

            if k_scale is not None and k_scale.numel() > 1:
                k_scale = k_scale.unsqueeze(-1)
                v_scale = v_scale.unsqueeze(-1)

            compute_type = (
                torch.bfloat16 if self.kv_cache_dtype == "bf16" else aiter.dtypes.fp8
            )
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

        return o

    @mark_trace(prefix="paged_attention_asm", torch_compile=False)
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

    @mark_trace(prefix="paged_attention_persistent_asm", torch_compile=False)
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

    @mark_trace(prefix="prefill_attention", torch_compile=False)
    def prefill_attention(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        # variable lenth attention use key value as input
        attn_metadata = fwd_ctx.attn_metadata
        sliding_window = (
            (self.sliding_window, 0, 0) if self.sliding_window > 0 else (-1, -1, 0)
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

        o = torch.empty_like(q)
        num_seqs = attn_metadata.cu_seqlens_q.shape[0] - 1
        descale_shape = (num_seqs, k.shape[1])
        sliding_window = (
            (self.sliding_window - 1, 0) if self.sliding_window > 0 else (-1, -1)
        )

        # `block_tables` is always populated by TritonMHAMetadataBuilder.
        # For pure prefill (no cached tokens) it is the fake table built in
        # prepare_prefill that maps seq i to token indices
        # [cu_seqlens_k[i], ..., cu_seqlens_k[i+1]-1], paired with raw K/V
        # treated as kv_cache with block_size=1.
        if attn_metadata.has_cached:
            k_for_attn = k_cache
            v_for_attn = v_cache
        else:
            #   k: [total_tokens, num_kv_heads, head_size]
            #     -> [total_tokens, 1, num_kv_heads, head_size]
            k_for_attn = k.unsqueeze(1)
            v_for_attn = v.unsqueeze(1)

        unified_attention(
            q,
            k_for_attn,
            v_for_attn,
            o,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            seqused_k=attn_metadata.context_lens,
            max_seqlen_q=attn_metadata.max_seqlen_q,
            max_seqlen_k=attn_metadata.max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=None,
            window_size=sliding_window,
            block_table=attn_metadata.block_tables,
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
            if self.use_flash_layout:
                return self.prefill_attention_triton
            return self.prefill_attention
        else:
            if self.use_triton_attn or self.use_flash_layout:
                return self.paged_attention_triton
            else:
                # Only use pa persistent when block_size == 1024
                atom_config = get_current_atom_config()
                if atom_config.kv_cache_block_size == 1024:
                    return self.paged_attention_persistent_asm
                return self.paged_attention_asm

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor = None,
        attn_metadata=None,
        position: torch.Tensor = None,
        q_scale: Optional[torch.Tensor] = None,
        qkv: torch.Tensor = None,
        output: torch.Tensor = None,
        **kwargs,
    ):
        if is_plugin_mode():
            # forward impl method are added by the decorator
            # PagedAttentionImplDecoratorForPluginMode
            return self.forward_impl_plugin_mode(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                position=position,
                q_scale=q_scale,
                qkv=qkv,
            )
        else:
            # only for server mode, keep the original method
            o = self.forward_impl_server_mode(
                q=query, k=key, v=value, position=position, q_scale=q_scale, qkv=qkv
            )

            return o
