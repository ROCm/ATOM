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

from atom.utils.decorators import mark_trace
from atom.model_ops.base_attention import (
    cp_mha_gather_cache,
    run_pa_decode_gluon,
    run_pa_fwd_asm,
)

logger = logging.getLogger("atom")


class PagedAttentionImpl(nn.Module):
    """
    Attention paged implementation
    """

    _gptoss_pa_decode_bf16_asm_log_keys: set[str] = set()

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
        self._gptoss_pa_decode_bf16_asm_scale_tensors = None
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

        if self._should_use_gptoss_pa_decode_bf16_asm(q, k_cache, v_cache, fwd_ctx):
            o = self.paged_attention_gptoss_pa_decode_bf16_asm(
                q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx
            )
        else:
            if self._requires_gptoss_pa_decode_bf16_asm(fwd_ctx):
                raise RuntimeError(
                    "ATOM_GPTOSS_USE_PA_DECODE_BF16_ASM is enabled, but this "
                    "full-attention GPT-OSS decode layer did not satisfy "
                    "pa_decode_bf16_asm requirements. Refusing to fall back to "
                    "Triton/Gluon on gfx1250."
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

        # Prefix cache hit: gather cached KV from paged cache and concat with new tokens
        if attn_metadata.has_cached:
            q, k, v, k_cache, v_cache, k_scale, v_scale = (
                self._gather_prefix_and_concat_kv(
                    q, k, v, k_cache, v_cache, k_scale, v_scale, attn_metadata
                )
            )

        return q, k, v, k_cache, v_cache, k_scale, v_scale

    def _log_gptoss_pa_decode_bf16_asm_once(self, key: str, msg: str, *args):
        if key in PagedAttentionImpl._gptoss_pa_decode_bf16_asm_log_keys:
            return
        PagedAttentionImpl._gptoss_pa_decode_bf16_asm_log_keys.add(key)
        logger.info(msg, *args)

    def _skip_gptoss_pa_decode_bf16_asm(self, reason: str) -> bool:
        if envs.ATOM_GPTOSS_USE_PA_DECODE_BF16_ASM:
            self._log_gptoss_pa_decode_bf16_asm_once(
                f"skip:{reason}",
                "ATOM_GPTOSS_USE_PA_DECODE_BF16_ASM fallback: %s",
                reason,
            )
        return False

    def _requires_gptoss_pa_decode_bf16_asm(self, fwd_ctx: ForwardContext) -> bool:
        return (
            envs.ATOM_GPTOSS_USE_PA_DECODE_BF16_ASM
            and not fwd_ctx.context.is_prefill
            and not self.use_flash_layout
            and self.sliding_window == -1
        )

    def _should_use_gptoss_pa_decode_bf16_asm(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        fwd_ctx: ForwardContext,
    ) -> bool:
        if not envs.ATOM_GPTOSS_USE_PA_DECODE_BF16_ASM:
            return False

        ctx = fwd_ctx.context
        attn_metadata = fwd_ctx.attn_metadata
        if ctx.is_prefill:
            return self._skip_gptoss_pa_decode_bf16_asm("prefill")
        if self.use_flash_layout:
            return self._skip_gptoss_pa_decode_bf16_asm("flash-layout cache")
        if self.sliding_window != -1:
            return self._skip_gptoss_pa_decode_bf16_asm("sliding-window layer")
        if self.kv_cache_dtype != "fp8":
            return self._skip_gptoss_pa_decode_bf16_asm("requires fp8 kv cache")
        if self.head_dim != 64:
            return self._skip_gptoss_pa_decode_bf16_asm("requires head_dim=64")
        if self.num_heads % self.num_kv_heads != 0:
            return self._skip_gptoss_pa_decode_bf16_asm(
                "q heads not divisible by kv heads"
            )

        gqa = self.num_heads // self.num_kv_heads
        if gqa != 8:
            return self._skip_gptoss_pa_decode_bf16_asm("requires gqa=8")

        max_seqlen_q = int(attn_metadata.max_seqlen_q)
        if max_seqlen_q < 1 or max_seqlen_q > 4:
            return self._skip_gptoss_pa_decode_bf16_asm(
                f"requires 1 <= max_seqlen_q <= 4, got {max_seqlen_q}"
            )

        required_metadata = (
            "cu_seqlens_q",
            "kv_indptr",
            "kv_indices",
            "context_lens",
        )
        for name in required_metadata:
            if getattr(attn_metadata, name, None) is None:
                return self._skip_gptoss_pa_decode_bf16_asm(f"missing {name}")

        num_seqs = int(attn_metadata.context_lens.shape[0])
        if q.shape[0] != num_seqs * max_seqlen_q:
            return self._skip_gptoss_pa_decode_bf16_asm(
                f"q tokens {q.shape[0]} != batch*max_q {num_seqs * max_seqlen_q}"
            )
        if k_cache.dim() != 5:
            return self._skip_gptoss_pa_decode_bf16_asm("requires 5D K cache")
        if k_cache.dtype != aiter.dtypes.fp8 or v_cache.dtype != aiter.dtypes.fp8:
            return self._skip_gptoss_pa_decode_bf16_asm("requires fp8 K/V tensors")

        page_size = int(k_cache.shape[3])
        if page_size != 256:
            return self._skip_gptoss_pa_decode_bf16_asm(
                f"requires page_size=256, got {page_size}"
            )
        if v_cache.dim() == 4:
            if int(v_cache.shape[-1]) != page_size:
                return self._skip_gptoss_pa_decode_bf16_asm("V cache block mismatch")
        elif v_cache.dim() == 5:
            if int(v_cache.shape[2] * v_cache.shape[4]) != page_size:
                return self._skip_gptoss_pa_decode_bf16_asm("V shuffle block mismatch")
        else:
            return self._skip_gptoss_pa_decode_bf16_asm("requires 4D/5D V cache")

        self._log_gptoss_pa_decode_bf16_asm_once(
            "enabled",
            "ATOM_GPTOSS_USE_PA_DECODE_BF16_ASM enabled: routing matching GPT-OSS "
            "decode layers through aiter.pa_decode_bf16_asm",
        )
        return True

    def _view_v_cache_for_pa_decode_bf16_asm(
        self, v_cache: torch.Tensor, k_cache: torch.Tensor
    ) -> torch.Tensor:
        if v_cache.dim() == 5:
            return v_cache
        n, nh, head_dim, block_size = v_cache.shape
        x = int(k_cache.shape[-1])
        return v_cache.view(n, nh, block_size // x, head_dim, x)

    def _quantize_gptoss_pa_decode_bf16_asm_query(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fp8_dtype = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_dtype).max
        q_amax = q.abs().max().clamp(min=1e-10)
        q_scale = (q_amax / fp8_max).float().reshape(1)
        q_fp8 = (q / q_scale).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
        return q_fp8.contiguous(), q_scale

    def _get_gptoss_pa_decode_bf16_asm_scale_tensors(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._gptoss_pa_decode_bf16_asm_scale_tensors
        if cached is not None and cached[0].device == device:
            return cached
        key_scale = torch.full(
            (1,), self.kv_scale_float * self.scale, dtype=torch.float32, device=device
        )
        value_scale = torch.full(
            (1,), self.kv_scale_float, dtype=torch.float32, device=device
        )
        cached = (key_scale, value_scale)
        self._gptoss_pa_decode_bf16_asm_scale_tensors = cached
        return cached

    def _get_gptoss_pa_decode_bf16_asm_metadata(
        self,
        attn_metadata,
        batch_size: int,
        max_seqlen_q: int,
        page_size: int,
    ) -> dict[str, torch.Tensor]:
        gqa = self.num_heads // self.num_kv_heads
        cache_key = (
            batch_size,
            max_seqlen_q,
            self.num_kv_heads,
            gqa,
            page_size,
            attn_metadata.cu_seqlens_q.data_ptr(),
            attn_metadata.kv_indptr.data_ptr(),
            attn_metadata.context_lens.data_ptr(),
        )
        cached = getattr(attn_metadata, "_gptoss_pa_decode_bf16_asm_metadata", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        device = attn_metadata.context_lens.device

        def _empty(spec):
            shape, dtype = spec
            return torch.empty(shape, dtype=dtype, device=device)

        (
            work_meta_data_spec,
            work_indptr_spec,
            work_info_spec,
            reduce_indptr_spec,
            reduce_final_map_spec,
            reduce_partial_map_spec,
        ) = aiter.get_ps_metadata_info_v1(
            batch_size,
            self.num_kv_heads,
            max_seqlen_q,
            qlen_granularity=max_seqlen_q,
        )

        work_meta_data = _empty(work_meta_data_spec)
        work_indptr = _empty(work_indptr_spec)
        work_info = _empty(work_info_spec)
        reduce_indptr = _empty(reduce_indptr_spec)
        reduce_final_map = _empty(reduce_final_map_spec)
        reduce_partial_map = _empty(reduce_partial_map_spec)

        aiter.get_ps_metadata_v1(
            attn_metadata.cu_seqlens_q,
            attn_metadata.kv_indptr,
            attn_metadata.context_lens,
            gqa,
            self.num_kv_heads,
            work_meta_data,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=gqa,
            qlen_granularity=max_seqlen_q,
            kvlen_granularity=page_size,
            block_size=page_size,
            is_causal=False,
        )

        metadata = {
            "work_meta_data": work_meta_data,
            "work_indptr": work_indptr,
            "work_info": work_info,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
        }
        setattr(
            attn_metadata,
            "_gptoss_pa_decode_bf16_asm_metadata",
            (cache_key, metadata),
        )
        return metadata

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

    @mark_trace(prefix="paged_attention_gptoss_pa_decode_bf16_asm", torch_compile=False)
    def paged_attention_gptoss_pa_decode_bf16_asm(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):
        del k, v, k_scale, v_scale

        attn_metadata = fwd_ctx.attn_metadata
        batch_size = int(attn_metadata.context_lens.shape[0])
        max_seqlen_q = int(attn_metadata.max_seqlen_q)
        page_size = int(k_cache.shape[3])
        gqa = self.num_heads // self.num_kv_heads

        q_5d = q.view(batch_size, max_seqlen_q, self.num_kv_heads, gqa, self.head_dim)
        q_fp8, query_scale = self._quantize_gptoss_pa_decode_bf16_asm_query(q_5d)
        key_scale, value_scale = self._get_gptoss_pa_decode_bf16_asm_scale_tensors(
            q.device
        )
        v_cache_5d = self._view_v_cache_for_pa_decode_bf16_asm(v_cache, k_cache)

        ps_metadata = self._get_gptoss_pa_decode_bf16_asm_metadata(
            attn_metadata,
            batch_size,
            max_seqlen_q,
            page_size,
        )

        output = torch.empty_like(q_5d)
        split_rows = max(
            1,
            int(ps_metadata["reduce_partial_map"].numel()) * max_seqlen_q,
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
        split_o.zero_()
        split_lse.fill_(float("-inf"))

        aiter.pa_decode_bf16_asm(
            Q=q_fp8,
            K=k_cache,
            V=v_cache_5d,
            kv_indices=attn_metadata.kv_indices,
            context_lens=attn_metadata.context_lens,
            softmax_scale=1.0,
            kv_indptr=attn_metadata.kv_indptr,
            gqa=gqa,
            mtp=max_seqlen_q - 1,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            qo_indptr=attn_metadata.cu_seqlens_q,
            work_indptr=ps_metadata["work_indptr"],
            work_info=ps_metadata["work_info"],
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
                ps_metadata["reduce_indptr"],
                ps_metadata["reduce_final_map"],
                ps_metadata["reduce_partial_map"],
                max_seqlen_q,
                output.view(batch_size * max_seqlen_q, self.num_heads, self.head_dim),
                final_lse,
            )

        return output.view(batch_size * max_seqlen_q, self.num_heads, self.head_dim)

    @mark_trace(prefix="paged_attention_asm", torch_compile=False)
    def paged_attention_asm(
        self, q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx: ForwardContext
    ):

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
            q=query, k=key, v=value, position=position, q_scale=q_scale, qkv=qkv
        )
