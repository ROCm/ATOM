# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache
from typing import Optional

import aiter
import torch
from aiter import fused_qk_norm_rope_cache_quant_shuffle
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits
from aiter.ops.triton.unified_attention import unified_attention
from atom.config import get_current_atom_config
from atom.utils import envs
from atom.utils.forward_context import ForwardContext, get_forward_context
from torch import nn

from .attention_mla import MLAModules

from atom.utils.decorators import mark_trace
from atom.model_ops.base_attention import (
    cp_mha_gather_cache,
    run_pa_decode_gluon,
    run_pa_fwd_asm,
)


@cache
def use_pa_decode_bf16_asm() -> bool:
    return (
        envs.ATOM_USE_UNIFIED_ATTN
        and not envs.ATOM_FORCE_ATTN_TRITON
        and get_gfx() == "gfx1250"
    )


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
        # Pre-allocated fp8 dequant scale for the pa_decode_bf16_asm path. Built
        # here (outside CUDAGraph capture) and reused so the kernel wrapper never
        # allocates a tensor mid-capture.
        self._pa_decode_bf16_asm_scale = torch.full(
            (1,), self.kv_scale_float, dtype=torch.float32, device=self.device
        )
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

        self.supports_quant_query_input = False

    def process_weights_after_loading(self):
        if use_pa_decode_bf16_asm():
            if self.sinks is not None and self.sinks.dtype != torch.float32:
                self.sinks.data = self.sinks.data.to(torch.float32).contiguous()

    def _can_attempt_prefill_sink_asm(self, fwd_ctx: ForwardContext) -> bool:
        if not fwd_ctx.context.is_prefill:
            return False
        if envs.ATOM_FORCE_ATTN_TRITON:
            return False
        if not (self.use_flash_layout or envs.ATOM_USE_UNIFIED_ATTN):
            return False
        attn_metadata = fwd_ctx.attn_metadata
        if attn_metadata is None:
            return False
        if get_gfx() != "gfx1250":
            return False
        if self.head_dim != 64:
            return False
        if self.sinks is None:
            return False
        if self.sliding_window != -1 or self.alibi_slopes is not None:
            return False
        if getattr(attn_metadata, "dropout_p", 0.0) != 0.0:
            return False
        # Prefix-cache hit (has_cached) is supported: prefill_attention gathers
        # the cached+new KV into a dense packed [total_kv, ...] tensor and the
        # gfx1250 sink varlen ASM kernel handles bottom-right causal for
        # sq != sk (chunked-prefill). cu_seqlens_q / cu_seqlens_k carry the
        # per-request new-token vs cached+new lengths, so we no longer require
        # max_seqlen_q == max_seqlen_k.
        if attn_metadata.cu_seqlens_q is None or attn_metadata.cu_seqlens_k is None:
            return False
        return True

    def _can_use_prefill_sink_asm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        fwd_ctx: ForwardContext,
    ) -> bool:
        if not self._can_attempt_prefill_sink_asm(fwd_ctx):
            return False
        if (
            q.dtype != torch.bfloat16
            or k.dtype != torch.bfloat16
            or v.dtype != torch.bfloat16
        ):
            return False
        if (
            self.head_dim != 64
            or q.shape[-1] != 64
            or k.shape[-1] != 64
            or v.shape[-1] != 64
        ):
            return False
        if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
            return False
        if q.shape[1] % k.shape[1] != 0:
            return False
        return True

    def forward_impl(
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

        attn_impl = self.dispatch_backend(fwd_ctx, q, k, v)
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

        # Fall back to Triton/Gluon for layouts unsupported by AITer PA ASM.
        use_triton_attn = (
            envs.ATOM_FORCE_ATTN_TRITON
            or self.sliding_window != -1
            or self.head_dim != 128
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
                    q=q,
                    k=k,
                    v=v,
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

                q = q.view(-1, self.num_heads, self.head_dim)
                k = k.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)
            self._cache_format = "SHUFFLE"
        elif use_triton_attn and self.rotary_emb is not None:
            self.per_token_quant = False
            k_scale = v_scale = self.kv_scale
            if (
                envs.ATOM_USE_UNIFIED_ATTN
                and self.kv_cache_dtype.startswith("fp8")
                and not self._can_attempt_prefill_sink_asm(fwd_ctx)
            ):
                q_out = torch.empty(*q.shape, dtype=k_cache.dtype, device=q.device)
            else:
                q_out = q
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
                q_out=q_out,
                k_out=k,
                output_zeros=False,
            )
            self._cache_format = "NHD"
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
            self._cache_format = "SHUFFLE" if asm_layout else "NHD"

        # NOTE: on a prefix-cache hit the cached+new KV is gathered into a dense
        # packed tensor inside prefill_attention (the ASM varlen path that needs
        # it). The Triton path reads the paged KV cache directly, so it never
        # gathers. Keeping the gather out of here also means dispatch_backend
        # sees q/k with matching token counts (sq == sk).
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
        # The cache format depends on which rope_cache branch wrote the data:
        # - SHUFFLE: fused_qk_norm_rope_cache_quant_shuffle or reshape_and_cache(asm_layout=True)
        #   K [n, nh, hd//x, bs, x], V viewed as [n, nh, bs//x, hd, x]
        # - NHD: fused_qk_rope_reshape_and_cache or reshape_and_cache(asm_layout=False)
        #   K [n, nh, hd//x, bs, x] -> permute to [n, bs, nh, hd], V [n, nh, hd, bs] -> [n, bs, nh, hd]
        use_shuffle = getattr(self, "_cache_format", "SHUFFLE") == "SHUFFLE"
        if k_cache.dim() == 5:
            x = 16 // k_cache.element_size()
            n, nh, _, block_size, _ = k_cache.shape
            if use_shuffle:
                k_cache_gather = k_cache
                v_cache_gather = v_cache.view(n, nh, block_size // x, head_dim, x)
            elif v_cache.dim() == 5:
                # MiMo-V2-Flash per-layer allocator (aiter_attention.py:461) emits
                # v_cache natively as 5D SHUFFLE [n, nh, bs//x, hd, x]; pass through.
                use_shuffle = True
                k_cache_gather = k_cache
                v_cache_gather = v_cache
            else:
                # V is in ASM/NHD format [n, nh, hd, bs], convert to [n, bs, nh, hd]
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

    def _view_v_cache_for_pa_decode_bf16_asm(
        self, v_cache: torch.Tensor, k_cache: torch.Tensor
    ) -> torch.Tensor:
        if v_cache.dim() == 5:
            return v_cache
        n, nh, head_dim, block_size = v_cache.shape
        x = int(k_cache.shape[-1])
        return v_cache.view(n, nh, block_size // x, head_dim, x)

    @mark_trace(prefix="paged_attention_triton", torch_compile=False)
    def paged_attention_triton(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        attn_metadata = fwd_ctx.attn_metadata

        if envs.ATOM_USE_UNIFIED_ATTN and self.kv_cache_dtype.startswith("fp8"):
            o = torch.empty(*q.shape, dtype=torch.bfloat16, device=q.device)
        else:
            o = torch.empty_like(q)

        num_seqs = attn_metadata.context_lens.shape[0]

        if envs.ATOM_USE_UNIFIED_ATTN or self.use_flash_layout:
            # print(q.shape, k_cache.shape, v_cache.shape)
            sliding_window = (
                (self.sliding_window - 1, 0) if self.sliding_window > 0 else (-1, -1)
            )

            shuffled_kv_cache = not self.use_flash_layout

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
                k_descale=self.kv_scale,
                v_descale=self.kv_scale,
                sinks=self.sinks,
                shuffled_kv_cache=shuffled_kv_cache,
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
            run_pa_decode_gluon(
                output=o,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                context_lens=attn_metadata.context_lens,
                block_tables=attn_metadata.block_tables,
                softmax_scale=self.scale,
                max_seqlen_q=attn_metadata.max_seqlen_q,
                max_context_partition_num=max_context_partition_num,
                context_partition_size=context_partition_size,
                compute_type=compute_type,
                q_scale=None,
                k_scale=None if self.kv_cache_dtype == "bf16" else k_scale,
                v_scale=None if self.kv_cache_dtype == "bf16" else v_scale,
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
        # run_pa_fwd_asm has no sink support; route sink layers through the
        # Triton/bf16-ASM paths instead of silently dropping the sink.
        if self.sinks is not None:
            raise RuntimeError(
                "paged_attention_asm does not support attention sinks; "
                "use the Triton path (ATOM_FORCE_ATTN_TRITON=1) or the gfx1250 "
                "pa_decode_bf16_asm path for sink layers."
            )

        attn_metadata = fwd_ctx.attn_metadata
        o = run_pa_fwd_asm(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=attn_metadata.block_tables,
            context_lens=attn_metadata.context_lens,
            k_scale=k_scale,
            v_scale=v_scale,
            max_qlen=attn_metadata.max_seqlen_q,
            qo_indptr=attn_metadata.cu_seqlens_q,
        )

        return o

    @mark_trace(prefix="paged_attention_persistent_asm", torch_compile=False)
    def paged_attention_persistent_asm(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):
        attn_metadata = fwd_ctx.attn_metadata

        if self.sinks is None:
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
        else:
            batch_size = int(attn_metadata.context_lens.shape[0])
            max_seqlen_q = int(attn_metadata.max_seqlen_q)
            page_size = int(k_cache.shape[3])
            gqa = self.num_heads // self.num_kv_heads

            q_5d = q.view(
                batch_size, max_seqlen_q, self.num_kv_heads, gqa, self.head_dim
            )
            if q_5d.dtype == aiter.dtypes.fp8:
                q_fp8 = q_5d.contiguous()
            else:
                q_fp8 = (q_5d / self.kv_scale_float).to(aiter.dtypes.fp8).contiguous()
            v_cache_5d = self._view_v_cache_for_pa_decode_bf16_asm(v_cache, k_cache)

            output = torch.empty(q_5d.shape, dtype=torch.bfloat16, device=q.device)
            # CUDAGraph decode pads scheduled_bs up to graph_bs. PA ASM has no
            # work for padded rows (context_len == 0); zero output so padded rows
            # stay deterministic.
            split_rows = max(
                1,
                int(attn_metadata.reduce_partial_map.numel()) * max_seqlen_q,
            )
            split_o = torch.empty(
                (split_rows, 1, self.num_heads, self.head_dim),
                dtype=torch.float32,
                device=q.device,
            )
            split_lse = torch.empty(
                (split_rows, 1, self.num_heads, 1),
                dtype=torch.float32,
                device=q.device,
            )

            aiter.pa_decode_bf16_asm(
                Q=q_fp8,
                K=k_cache,
                V=v_cache_5d,
                kv_indices=attn_metadata.kv_indices,
                context_lens=attn_metadata.context_lens,
                softmax_scale=self.scale,
                kv_indptr=attn_metadata.kv_indptr,
                gqa=gqa,
                mtp=max_seqlen_q - 1,
                query_scale=self._pa_decode_bf16_asm_scale,
                key_scale=self._pa_decode_bf16_asm_scale,
                value_scale=self._pa_decode_bf16_asm_scale,
                qo_indptr=attn_metadata.cu_seqlens_q,
                work_indptr=attn_metadata.work_indptr,
                work_info=attn_metadata.work_info_set,
                split_o=split_o,
                split_lse=split_lse,
                sink=self.sinks,
                out=output,
            )

            if int(attn_metadata.max_seqlen_k) > page_size:
                final_lse = torch.empty(
                    (batch_size * max_seqlen_q, self.num_heads),
                    dtype=torch.float32,
                    device=q.device,
                )
                aiter.pa_reduce_v1(
                    split_o,
                    split_lse,
                    attn_metadata.reduce_indptr,
                    attn_metadata.reduce_final_map,
                    attn_metadata.reduce_partial_map,
                    max_seqlen_q,
                    output.view(
                        batch_size * max_seqlen_q, self.num_heads, self.head_dim
                    ),
                    final_lse,
                )

            return output.view(batch_size * max_seqlen_q, self.num_heads, self.head_dim)

    @mark_trace(prefix="prefill_attention", torch_compile=False)
    def prefill_attention(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

        # variable lenth attention use key value as input
        attn_metadata = fwd_ctx.attn_metadata
        # Prefix-cache hit: gather cached+new KV from the paged cache into a
        # dense packed [total_kv, ...] tensor (new tokens were already written
        # during rope_cache). flash_attn_varlen_func then attends over the full
        # sequence; cu_seqlens_q / cu_seqlens_k carry the new vs cached+new
        # lengths (sq < sk), which the varlen kernel handles via bottom-right
        # causal.
        if attn_metadata.has_cached:
            q, k, v, k_cache, v_cache, k_scale, v_scale = (
                self._gather_prefix_and_concat_kv(
                    q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata
                )
            )
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

        # unified_attention supports both prefill and decode, over either the 4D
        # flash layout (shuffled_kv_cache=False) or the 5D SHUFFLE layout
        # (shuffled_kv_cache=True):
        #
        # flash    K/V: [num_blocks, block_size, num_kv_heads, head_size]
        # shuffle  K:   [num_blocks, num_kv_heads, head_size // x, block_size, x]
        # shuffle  V:   [num_blocks, num_kv_heads, block_size // x, head_size, x]
        #
        # For pure prefill (no cached tokens), raw key/value are passed as a
        # block_size=1 flash-layout cache with a fake block_table:
        #
        # key:    [num_tokens, 1, num_kv_heads, head_size]
        # value:  [num_tokens, 1, num_kv_heads, head_size]

        attn_metadata = fwd_ctx.attn_metadata

        if envs.ATOM_USE_UNIFIED_ATTN and self.kv_cache_dtype.startswith("fp8"):
            o = torch.empty(*q.shape, dtype=torch.bfloat16, device=q.device)
        else:
            o = torch.empty_like(q)

        sliding_window = (
            (self.sliding_window - 1, 0) if self.sliding_window > 0 else (-1, -1)
        )

        # `block_tables` is always populated by TritonMHAMetadataBuilder.
        # For pure prefill (no cached tokens) it is, by default, the fake table
        # built in prepare_prefill that maps seq i to token indices
        # [cu_seqlens_k[i], ..., cu_seqlens_k[i+1]-1], paired with raw K/V
        # treated as kv_cache with block_size=1.
        #
        # Under ATOM_USE_UNIFIED_ATTN, prepare_prefill instead uploads the real
        # per-seq block_table and reads from KV cache, the new tokens
        # already written into the paged flash-layout cache during rope_cache
        # are read straight from `k_cache`/`v_cache`, identical to the
        # prefix-cache-hit path.
        if envs.ATOM_USE_UNIFIED_ATTN or attn_metadata.has_cached:
            k_for_attn = k_cache
            v_for_attn = v_cache
            # Reads the paged KV cache, which is 5D SHUFFLE unless the (default)
            # 4D flash layout is in use.
            shuffled_kv_cache = not self.use_flash_layout
        else:
            #   k: [total_tokens, num_kv_heads, head_size]
            #     -> [total_tokens, 1, num_kv_heads, head_size]
            k_for_attn = k.unsqueeze(1)
            v_for_attn = v.unsqueeze(1)
            # Raw K/V is fed as a block_size=1 flash-layout cache, never shuffled.
            shuffled_kv_cache = False

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
            k_descale=self.kv_scale,
            v_descale=self.kv_scale,
            sinks=self.sinks,
            shuffled_kv_cache=shuffled_kv_cache,
        )

        return o

    def _dispatch_decode(self):
        # Sliding-window layers must use triton (ASM paths don't support it)
        if self.sliding_window != -1:
            return self.paged_attention_triton

        atom_config = get_current_atom_config()

        if envs.ATOM_USE_UNIFIED_ATTN:
            if envs.ATOM_FORCE_ATTN_TRITON:
                return self.paged_attention_triton
            if atom_config.kv_cache_block_size == 256:
                return self.paged_attention_persistent_asm
            return self.paged_attention_triton

        if self.use_triton_attn or self.use_flash_layout:
            return self.paged_attention_triton

        if use_pa_decode_bf16_asm():
            return self.paged_attention_persistent_asm
        return self.paged_attention_asm

    def dispatch_backend(
        self,
        fwd_ctx: ForwardContext,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        if fwd_ctx.context.is_prefill:
            # q/k/v here still hold only the new tokens (the prefix gather happens
            # inside prefill_attention), so the q.shape[0] == k.shape[0] check in
            # _can_use_prefill_sink_asm is valid.
            if self._can_use_prefill_sink_asm(q, k, v, fwd_ctx):
                return self.prefill_attention
            if envs.ATOM_USE_UNIFIED_ATTN or self.use_flash_layout:
                return self.prefill_attention_triton
            return self.prefill_attention
        return self._dispatch_decode()

    def forward(
        self,
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
        return self.forward_impl(
            q=query,
            k=key,
            v=value,
            position=position,
            q_scale=q_scale,
            qkv=qkv,
        )


class SparseMHAPagedAttentionImpl(PagedAttentionImpl):
    """MiniMax-M3 sparse attention as a first-class ``PagedAttentionImpl``.

    Plugged into the standard ``Attention`` layer via ``impl_cls=`` so it reuses
    the generic per-layer custom op (``unified_attention_with_output_base``) for
    its torch.compile boundary, and the standard ``AiterAttentionMetadataBuilder``
    for KV-cache allocation/binding. Only two framework hooks are overridden:

    * :meth:`rope_cache` — MiniMax-M3 fused qk-norm + rope + page-16 SHUFFLE
      KV-insert + indexer-key insert (``aiter.fused_qknorm_idxrqknorm`` /
      ``minimax_m3_fused_qknorm_rope_kv_insert_shuffle``). Returns the rotated
      query in the parent's 7-tuple contract and stashes the rotated indexer
      query on ``self._index_q`` for :meth:`dispatch_backend` (the parent tuple
      has no slot for it; per-layer forward is single-threaded behind the op).
    * :meth:`dispatch_backend` — selects the M3 sparse prefill/decode runners
      (index top-k -> page-16 sparse block table -> gluon PA), with fp8 vs bf16
      chosen by the KV cache dtype, not an env gate.

    All indexer state (norms, rope, top-k params, index_cache handle) lives on
    this impl instance — the model holds no sparse-attention runtime state.
    """

    is_indexed_sparse_attention = True

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        alibi_slopes: list[float] | None = None,
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
        # --- MiniMax-M3 sparse-attention indexer kwargs (all impl-local) ---
        index_q_norm: Optional[torch.nn.Module] = None,
        index_k_norm: Optional[torch.nn.Module] = None,
        index_rotary_emb: Optional[torch.nn.Module] = None,
        index_q_size: int = 0,
        index_head_dim: int = 0,
        topk: int = 0,
        init_blocks: int = 0,
        local_blocks: int = 0,
        skip_index_topk: bool = False,
        sparse_layer_ordinal: int = -1,
        index_cache_dtype: str | None = None,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            layer_num=layer_num,
            mla_modules=mla_modules,
            sinks=sinks,
            rotary_emb=rotary_emb,
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        )
        # Indexer submodules + top-k parameters (impl-local state).
        self.index_q_norm = index_q_norm
        self.index_k_norm = index_k_norm
        # MiniMax-M3 shares the main rope with the indexer; default to it.
        self.index_rotary_emb = (
            index_rotary_emb if index_rotary_emb is not None else rotary_emb
        )
        self.index_q_size = index_q_size
        self.index_head_dim = index_head_dim
        # M3 has one index head per kv head (num_idx_heads == num_kv_heads).
        self.num_idx_heads = num_kv_heads
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.skip_index_topk = skip_index_topk
        self.sparse_layer_ordinal = sparse_layer_ordinal
        self.index_cache_dtype = (
            index_cache_dtype if index_cache_dtype is not None else kv_cache_dtype
        )
        # Bound by AiterAttentionMetadataBuilder.build_kv_cache_tensor (Task 6):
        # the page-128 indexer-key cache. None until the runner binds it.
        self.index_cache: Optional[torch.Tensor] = None
        # Optional shared dict bound by the metadata builder. It is scoped to the
        # current sparse metadata object and carries the last full layer top-k.
        self.index_topk_cache_state: Optional[dict] = None
        self._index_q_cache_key_info: Optional[tuple] = None
        # Rotated indexer query produced by rope_cache, consumed (and cleared) by
        # dispatch_backend within the same single-threaded layer forward.
        self._index_q: Optional[torch.Tensor] = None

    @staticmethod
    def _to_page16_shuffle(k_cache, v_cache, k_scale, v_scale):
        """Reinterpret the standard page-128 SHUFFLE KV/scale views as page-16
        SHUFFLE for the MiniMax-M3 ASM/gluon kernels. Zero-copy (128 == 8*16):

            K:     [N, nkv, hd//x, 128, x] -> [N*8, nkv, hd//x, 16, x]
            V:     [N, nkv, 128//x, hd, x] -> [N*8, nkv, 16//x, hd, x]
            scale: [N, nkv, 128]           -> [N*8, nkv, 16]   (fp8 only)

        Scales are re-viewed only when present (fp8); bf16 passes them through
        (None).
        """
        from atom.model_ops.minimax_m3.sparse_attn import (
            ASM_PAGE_SIZE,
            PAGES_PER_SPARSE_BLOCK,
        )

        n_blocks, nkv = k_cache.shape[0], k_cache.shape[1]
        x = k_cache.shape[-1]
        head_dim = k_cache.shape[2] * x
        num_phys16 = n_blocks * PAGES_PER_SPARSE_BLOCK

        k16 = k_cache.view(num_phys16, nkv, head_dim // x, ASM_PAGE_SIZE, x)
        v16 = v_cache.view(num_phys16, nkv, ASM_PAGE_SIZE // x, head_dim, x)
        if k_scale is not None and v_scale is not None:
            k_scale = k_scale.view(num_phys16, nkv, ASM_PAGE_SIZE)
            v_scale = v_scale.view(num_phys16, nkv, ASM_PAGE_SIZE)
        return k16, v16, k_scale, v_scale

    @mark_trace(prefix="rope_cache", torch_compile=False)
    def rope_cache(self, q, k, v, qkv, position, fwd_ctx: ForwardContext):
        """MiniMax-M3 fused qk-norm + partial-NeoX-RoPE + page-16 SHUFFLE KV insert
        + indexer-key insert, via ``aiter.fused_qknorm_idxrqknorm``.

        Consumes the packed ``qkv`` tensor laid out as
        ``[q | k | v | index_q | index_k]``. Writes:
          * normed+roped main K/V          -> SHUFFLE K/V cache (asm_layout=True)
          * normed+roped index_k           -> page-128 index_cache
          * fp8 per-token dequant scales   -> k_scale / v_scale (when fp8)
        and outputs the normed+roped main ``q`` (returned in the parent 7-tuple)
        and index ``q`` (stashed on ``self._index_q`` for dispatch_backend).

        Returns the parent contract tuple
        ``(q, k, v, k_cache, v_cache, k_scale, v_scale)``. ``k``/``v`` are returned
        unchanged (already inserted into the cache); the sparse backends read the
        cache, not these tensors.
        """
        attn_metadata = fwd_ctx.attn_metadata
        kv_cache_data = fwd_ctx.kv_cache_data

        # The KV cache is bound by the STANDARD MHA path (same allocation as every
        # other MHA model): page-128 SHUFFLE views
        #   K: [N, nkv, hd//x, 128, x]   V: [N, nkv, 128//x, hd, x]
        #   scale (fp8): [N, nkv, 128]
        # The M3 ASM/gluon kernels index this storage as page-16 SHUFFLE: each
        # logical 128-block is 8 contiguous physical 16-pages. 128 == 8*16, so the
        # page-16 view is a pure zero-copy reinterpretation of the page-128 view.
        # We re-view here (at attention time) instead of at bind time so the binder
        # has no M3-specific KV/scale code.
        layer = kv_cache_data[f"layer_{self.layer_num}"]
        k_cache, v_cache, k_scale, v_scale = self._to_page16_shuffle(
            layer.k_cache, layer.v_cache, layer.k_scale, layer.v_scale
        )

        # M3 sparse attention is fixed to head_dim == 128 (ASM/gluon requirement)
        # and the AITER fused path; no Triton fallback here.
        self.use_triton_attn = False
        self._cache_format = "SHUFFLE"

        sparse_metadata = getattr(attn_metadata, "sparse_attention_metadata", None)
        if sparse_metadata is None:
            sparse_metadata = attn_metadata
        slot_mapping = sparse_metadata.slot_mapping

        qkv = qkv.contiguous()
        num_tokens = qkv.shape[0]
        from atom.models.minimax_m3 import _minimax_m3_cos_sin_cache

        cos_sin_cache = _minimax_m3_cos_sin_cache(self.rotary_emb, qkv)

        is_fp8 = self.kv_cache_dtype == "fp8"
        kv_cache_dtype = "auto" if not is_fp8 else self.kv_cache_dtype
        # fp8: the fused op computes per-token dynamic quant and writes the
        # per-token dequant scales into k_scale / v_scale (outputs).
        fused_k_scale = k_scale if is_fp8 else None
        fused_v_scale = v_scale if is_fp8 else None
        fused_index_cache_dtype = (
            self.index_cache_dtype if self.index_cache_dtype == "fp8" else "auto"
        )

        if self.skip_index_topk:
            from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
                triton_fused_norm_rope_cache,
            )

            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            q_raw, k_raw, v_raw, _, _ = torch.split(
                qkv,
                [q_size, kv_size, kv_size, self.index_q_size, self.index_head_dim],
                dim=-1,
            )
            q_out, k_out = triton_fused_norm_rope_cache(
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
                v_cache=v_cache,
                k_scale=fused_k_scale,
                v_scale=fused_v_scale,
                slot_mapping=slot_mapping,
                kv_cache_dtype=self.kv_cache_dtype,
            )
            q = q_out.view(-1, self.num_heads, self.head_dim)
            k = k_out.view(-1, self.num_kv_heads, self.head_dim)
            v = v_raw.view(-1, self.num_kv_heads, self.head_dim)
            self._index_q = None
            self._index_q_cache_key_info = (
                (num_tokens, self.num_idx_heads, self.index_head_dim),
                qkv.dtype,
                qkv.device,
            )
            return q, k, v, k_cache, v_cache, k_scale, v_scale

        q_out = torch.empty(
            (num_tokens, self.num_heads * self.head_dim),
            dtype=qkv.dtype,
            device=qkv.device,
        )
        index_q = torch.empty(
            (num_tokens, self.index_q_size), dtype=qkv.dtype, device=qkv.device
        )
        aiter.fused_qknorm_idxrqknorm(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin_cache,
            position,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_emb.rotary_dim,
            self.q_norm.variance_epsilon,
            self.index_q_norm.weight,
            self.index_k_norm.weight,
            self.num_idx_heads,
            slot_mapping,
            k_cache,
            v_cache,
            self.index_cache,
            k_cache.shape[3],  # SHUFFLE page size (== ASM_PAGE_SIZE == 16)
            q_out,
            index_q,
            slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            index_cache_dtype=fused_index_cache_dtype,
            k_scale=fused_k_scale,
            v_scale=fused_v_scale,
            asm_layout=True,
        )

        q = q_out.view(-1, self.num_heads, self.head_dim)
        # Stash the rotated indexer query for dispatch_backend (same-forward,
        # single-threaded; cleared after the sparse backend consumes it).
        self._index_q = index_q.view(-1, self.num_idx_heads, self.index_head_dim)
        self._index_q_cache_key_info = (
            tuple(self._index_q.shape),
            self._index_q.dtype,
            self._index_q.device,
        )

        return q, k, v, k_cache, v_cache, k_scale, v_scale

    def dispatch_backend(
        self,
        fwd_ctx: ForwardContext,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Return the MiniMax-M3 sparse backend callable matching the parent
        contract ``fn(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx)``.

        Prefill and decode both: select per-(token/request) top-k index blocks
        (fusing the page-16 sparse block-table emit), then run the gluon split-KV
        paged-attention over the SHUFFLE cache. fp8 vs bf16 follows the cache
        dtype inside the runners. Consumes ``self._index_q`` from rope_cache.
        """
        if fwd_ctx.context.is_prefill:
            return self._sparse_prefill
        return self._sparse_decode

    def _sparse_metadata(self, fwd_ctx: ForwardContext):
        attn_metadata = fwd_ctx.attn_metadata
        sm = getattr(attn_metadata, "sparse_attention_metadata", None)
        return sm if sm is not None else attn_metadata

    def _topk_cache_state(self, sparse_metadata):
        if self.index_topk_cache_state is None:
            return None
        state = getattr(sparse_metadata, "_index_topk_cache_state", None)
        if state is None:
            state = {}
            setattr(sparse_metadata, "_index_topk_cache_state", state)
        return state

    def _topk_cache_key(
        self,
        mode: str,
        index_q: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
    ) -> tuple:
        if index_q is None:
            if self._index_q_cache_key_info is None:
                raise RuntimeError(
                    "MiniMax-M3 index cache key missing index_q metadata"
                )
            index_q_shape, index_q_dtype, index_q_device = self._index_q_cache_key_info
        else:
            index_q_shape = tuple(index_q.shape)
            index_q_dtype = index_q.dtype
            index_q_device = index_q.device
        return (
            mode,
            index_q_shape,
            index_q_dtype,
            index_q_device,
            tuple(block_table.shape),
            tuple(block_table.stride()),
            tuple(seq_lens.shape),
            self.topk,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            max_query_len,
            max_seq_len,
        )

    def _load_cached_topk(self, sparse_metadata, key: tuple):
        if not self.skip_index_topk:
            return None
        state = self._topk_cache_state(sparse_metadata)
        if state is None:
            return None
        entry = state.get("topk")
        if entry is None or entry.get("key") != key:
            return None
        return entry["value"]

    def _store_cached_topk(self, sparse_metadata, key: tuple, value: tuple):
        state = self._topk_cache_state(sparse_metadata)
        if state is not None:
            state["topk"] = {
                "key": key,
                "value": value,
                "layer_num": self.layer_num,
                "sparse_layer_ordinal": self.sparse_layer_ordinal,
            }

    @mark_trace(prefix="sparse_attention_prefill", torch_compile=False)
    def _sparse_prefill(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):
        from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk
        from atom.model_ops.minimax_m3.sparse_attn import (
            minimax_m3_sparse_attn_prefill_asm,
        )

        index_q = self._index_q
        sparse_metadata = self._sparse_metadata(fwd_ctx)
        prefill_md = sparse_metadata.prefill
        assert prefill_md is not None, "sparse prefill metadata missing"
        cu_seqlens_q = prefill_md.cu_seqlens_q
        seq_lens = prefill_md.seq_lens
        prefix_lens = prefill_md.context_lens
        block_tables = prefill_md.block_table

        topk_key = self._topk_cache_key(
            "prefill",
            index_q,
            block_tables,
            seq_lens,
            prefill_md.max_query_len,
            prefill_md.max_seq_len,
        )
        cached_topk = self._load_cached_topk(sparse_metadata, topk_key)
        if cached_topk is None:
            if index_q is None:
                raise RuntimeError("MiniMax-M3 index cache miss on a skip-index layer")
            topk_idx, sparse_bt, sparse_ctx = minimax_m3_index_topk(
                index_q,
                self.index_cache,
                block_tables,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                prefill_md.max_query_len,
                prefill_md.max_seq_len,
                self.topk,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                self.scale,
                emit_sparse_block_table=True,
            )
            self._store_cached_topk(
                sparse_metadata, topk_key, (topk_idx, sparse_bt, sparse_ctx)
            )
        else:
            topk_idx, sparse_bt, sparse_ctx = cached_topk
        output = torch.empty_like(q)
        minimax_m3_sparse_attn_prefill_asm(
            q,
            k_cache,
            v_cache,
            topk_idx,
            block_tables,
            None,  # query_req_id -> sync-free on-device fallback
            None,  # query_abs_pos -> sync-free on-device fallback
            prefill_md.qo_indptr,  # qo_indptr -> arange(total_q+1)
            self.num_kv_heads,
            self.scale,
            output,
            k_scale=k_scale,
            v_scale=v_scale,
            cu_seqlens_q=cu_seqlens_q,
            prefix_lens=prefix_lens,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
        )
        output = output.view(*q.shape)
        self._index_q = None
        self._index_q_cache_key_info = None
        from atom.utils.tbo.ubatching import tbo_active

        if tbo_active():
            from atom.utils.tbo.ubatching import tbo_yield

            tbo_yield()
        return output

    @mark_trace(prefix="sparse_attention_decode", torch_compile=False)
    def _sparse_decode(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):
        from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
        from atom.model_ops.minimax_m3.sparse_attn import (
            minimax_m3_sparse_attn_decode_asm,
        )

        index_q = self._index_q
        sparse_metadata = self._sparse_metadata(fwd_ctx)
        decode_md = sparse_metadata.decode
        assert decode_md is not None, "sparse decode metadata missing"
        max_query_len = getattr(decode_md, "max_query_len", 1)

        topk_key = self._topk_cache_key(
            "decode",
            index_q,
            decode_md.block_table,
            decode_md.seq_lens,
            max_query_len,
            sparse_metadata.max_seq_len,
        )
        cached_topk = self._load_cached_topk(sparse_metadata, topk_key)
        if cached_topk is None:
            if index_q is None:
                raise RuntimeError("MiniMax-M3 index cache miss on a skip-index layer")
            topk_idx, sparse_bt, sparse_ctx = minimax_m3_index_topk_decode(
                index_q,
                self.index_cache,
                decode_md.block_table,
                decode_md.seq_lens,
                sparse_metadata.max_seq_len,
                self.topk,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                self.scale,
                emit_sparse_block_table=True,
                max_query_len=max_query_len,
            )
            self._store_cached_topk(
                sparse_metadata, topk_key, (topk_idx, sparse_bt, sparse_ctx)
            )
        else:
            topk_idx, sparse_bt, sparse_ctx = cached_topk
        output = torch.empty_like(q)
        minimax_m3_sparse_attn_decode_asm(
            q,
            k_cache,
            v_cache,
            topk_idx,
            decode_md.block_table,
            decode_md.seq_lens,
            self.num_kv_heads,
            self.scale,
            output,
            k_scale=k_scale,
            v_scale=v_scale,
            sparse_bt=sparse_bt,
            sparse_ctx=sparse_ctx,
        )
        self._index_q = None
        self._index_q_cache_key_info = None
        output = output.view(*q.shape)
        return output
