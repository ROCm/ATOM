# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch-native attention backend for ATOM.

Why this exists
---------------
The AITER package shipped in rocm/atom-dev:latest has prebuilt HIP .so files
only for gfx94x/95x. On gfx1201 (RDNA4) the first paged-attention HIP load
fails with 'No compatible code objects found for: gfx1201' and SIGSEGVs the
ModelRunner subprocess. This backend is a torch-only path that does not
load any of those prebuilt modules.

Selection: atom/utils/selector.py:get_attn_backend_cls routes here when
torch.cuda.get_device_properties(0).gcnArchName starts with 'gfx1201', or
when ATOM_TORCH_NATIVE_ATTN=1 is set on any device.

Dispatch: atom/model_ops/paged_attention.py:PagedAttention.forward checks
self.attn_backend.get_name() == 'TORCH_NATIVE_ATTENTION' and routes through
self.impl.forward() instead of torch.ops.aiter.unified_attention_with_output_base.

Status
------
- Prefill: implemented via torch.nn.functional.scaled_dot_product_attention
  with per-sequence slicing using cu_seqlens_q (variable-length attention).
  RoPE is applied if rotary_emb was passed in. Sliding window is honored.
- Decode: NOT implemented (raises). Requires a working KV cache write +
  block-table gather. Tracked as TODO-5.
- KV cache: NOT allocated. The metadata builder's allocate_kv_cache_tensors
  returns {} (default) so no paged KV pool exists. Prefill works without it
  because the full sequence's K/V is in the current call. Tracked as TODO-7.
- FP8 KV cache: NOT supported. Use --kv_cache_dtype bf16. (TODO-8)
- CUDAGraph capture: NOT supported. Use --enforce-eager and --level 0.

Goal of this iteration: get ModelRunner.warmup_model() to complete one
prefill forward pass without any aiter HIP module load.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Type

import torch
import torch.nn.functional as F
from torch import nn

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionImpl,
    CommonAttentionBuilder,
)
from atom.utils.forward_context import AttentionMetaData, get_forward_context

logger = logging.getLogger("atom")


def _is_gfx1201() -> bool:
    """Return True if the visible CUDA/HIP device is gfx1201 (RDNA4)."""
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_properties(0).gcnArchName or ""
    return name.startswith("gfx1201")


def use_torch_native_attn() -> bool:
    """True when ATOM should route attention through the torch-native backend."""
    if os.environ.get("ATOM_TORCH_NATIVE_ATTN", "").lower() in ("1", "true"):
        return True
    return _is_gfx1201()


class TorchNativeBackend(AttentionBackend):
    """AITER-free attention backend."""

    @staticmethod
    def get_name() -> str:
        return "TORCH_NATIVE_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["TorchNativeMetadataBuilder"]:
        return TorchNativeMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["TorchNativeAttentionImpl"]:
        return TorchNativeAttentionImpl


class TorchNativeMetadataBuilder(CommonAttentionBuilder):
    """Subclass CommonAttentionBuilder so we inherit prepare_prefill (which
    already uses only torch + a Triton helper for block-table conversion).
    The aiter-specific allocations done by AiterAttentionMetadataBuilder.__init__
    (get_pa_metadata_info_v1, work_meta_data, work_indptr, kv_indptr, ...) are
    deliberately omitted -- they target an aiter HIP kernel that does not
    have a gfx1201 build.

    KV cache allocation is also omitted for now (defaults from base class
    return empty dicts). Prefill works without it because the current
    forward() call has the full sequence's K/V in hand. Decode is TODO.
    """

    def __init__(
        self,
        kv_cache_spec=None,
        layer_names=None,
        config=None,
        device=None,
        model_runner=None,
    ):
        self.block_size = 16 if model_runner.block_size != 1024 else 1024
        CommonAttentionBuilder.__init__(self, model_runner)
        logger.info(
            "TorchNativeMetadataBuilder: initialized (no aiter HIP allocations)"
        )

    def compute_block_bytes(self) -> int:
        """Return a nonzero placeholder so engine_core.get_num_blocks does not
        ZeroDivisionError. We do not actually use this paged KV pool yet
        (decode is a TODO); a small constant per layer keeps the math sane.
        """
        runner = self.model_runner
        cfg = runner.config
        hf = cfg.hf_config
        from atom.config import _MULTIMODAL_MODEL_TYPES
        # Mistral3 etc: text fields live on text_config after flattening.
        num_kv_heads = max(1, runner._get_num_kv_heads())
        head_dim = getattr(hf, "head_dim", None) or (
            hf.hidden_size // hf.num_attention_heads
        )
        n_layers = runner._get_total_num_layers()
        # bytes per block for K and V together: 2 * layers * block * heads * d * 2
        return 2 * n_layers * self.block_size * num_kv_heads * head_dim * 2

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        # TODO: build slot_mapping/context_lens/block_tables for decode without
        # aiter's kv_indptr/kv_indices. Mirror aiter_attention.py:prepare_decode
        # stripped of all kv_indptr/kv_indices/persistent-worker buffers.
        raise NotImplementedError(
            "TorchNativeMetadataBuilder.prepare_decode is a TODO. The current "
            "impl only supports prefill (sufficient for ModelRunner.warmup_model)."
        )

    def build_for_cudagraph_capture(self, bs: int):
        raise NotImplementedError(
            "build_for_cudagraph_capture: run with --enforce-eager --level 0 "
            "(CUDAGraph capture not yet supported)."
        )


class TorchNativeAttentionImpl(AttentionImpl):
    """Torch-only paged-attention forward.

    Constructor mirrors PagedAttentionImpl
    (atom/model_ops/attention_mha.py:29-90); only the fields actually used by
    the prefill path are stored. The rest are accepted-and-ignored to stay
    signature-compatible with the existing PagedAttention dispatch site.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes=None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "bf16",
        logits_soft_cap=None,
        attn_type=None,
        kv_sharing_target_layer_name=None,
        layer_num: int = 0,
        mla_modules=None,
        sinks=None,
        rotary_emb=None,
        q_norm=None,
        k_norm=None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_size = head_dim  # ATOM convention
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.kv_cache_dtype = kv_cache_dtype
        self.layer_num = layer_num
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm
        # Sized by the q/kv split; accept-and-ignore the rest.
        self.q_size = num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        if kv_cache_dtype != "bf16":
            logger.warning(
                f"TorchNativeAttentionImpl: kv_cache_dtype={kv_cache_dtype} "
                "is a TODO; force --kv_cache_dtype bf16."
            )

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        use_mla: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Prefill-only torch-native attention.

        Layout:
          query : [total_tokens, num_heads * head_dim]
          key   : [total_tokens, num_kv_heads * head_dim]
          value : [total_tokens, num_kv_heads * head_dim]
        Output : [total_tokens, num_heads * head_dim]

        Steps:
          1. Reshape into (total_tokens, num_heads_or_kv, head_dim).
          2. Apply RoPE if rotary_emb is set.
          3. Repeat-interleave KV heads to match Q heads (GQA).
          4. For each sequence (per cu_seqlens_q), call SDPA with is_causal=True.
          5. Reassemble into the flat token-major output layout.
        """
        import sys
        if use_mla:
            raise NotImplementedError(
                "TorchNativeAttentionImpl: MLA path is not implemented; "
                "this backend is for plain MHA (Llama / Mistral)."
            )

        ctx = get_forward_context()
        attn_md: Optional[AttentionMetaData] = ctx.attn_metadata
        fc = ctx.context

        is_prefill = bool(getattr(fc, "is_prefill", True)) if fc is not None else True
        if not is_prefill:
            raise NotImplementedError(
                "TorchNativeAttentionImpl: decode path is a TODO. "
                "Only prefill works today (sufficient for warmup_model)."
            )

        if attn_md is None or getattr(attn_md, "cu_seqlens_q", None) is None:
            raise RuntimeError(
                "TorchNativeAttentionImpl: forward called without an "
                "AttentionMetaData with cu_seqlens_q."
            )

        total_tokens = query.shape[0]
        q = query.view(total_tokens, self.num_heads, self.head_dim)
        k = key.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = value.view(total_tokens, self.num_kv_heads, self.head_dim)

        # RoPE
        if self.rotary_emb is not None and positions is not None:
            # ATOM's rotary_emb expects (positions, q_flat, k_flat) in many
            # implementations; use the same shape the model passes in.
            q_flat = q.reshape(total_tokens, self.num_heads * self.head_dim)
            k_flat = k.reshape(total_tokens, self.num_kv_heads * self.head_dim)
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(total_tokens, self.num_heads, self.head_dim)
            k = k_flat.view(total_tokens, self.num_kv_heads, self.head_dim)

        # GQA: tile K/V heads so they match Q heads
        if self.num_kv_heads != self.num_heads:
            assert self.num_heads % self.num_kv_heads == 0
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        cu_q = attn_md.cu_seqlens_q
        if cu_q.dim() == 0:  # scalar slipped through
            raise RuntimeError("cu_seqlens_q is a 0-dim tensor, expected 1-D")
        cu_q_cpu = cu_q.detach().cpu().tolist()

        # Per-sequence SDPA prefill. SDPA with is_causal=True takes
        # [batch, heads, seq, head_dim] inputs.
        out = torch.empty_like(q)
        for i in range(len(cu_q_cpu) - 1):
            s, e = int(cu_q_cpu[i]), int(cu_q_cpu[i + 1])
            if s == e:
                continue
            q_i = q[s:e].transpose(0, 1).unsqueeze(0)  # [1, H, T, D]
            k_i = k[s:e].transpose(0, 1).unsqueeze(0)
            v_i = v[s:e].transpose(0, 1).unsqueeze(0)
            attn_mask = None
            if self.sliding_window is not None and self.sliding_window > 0:
                t = e - s
                idx = torch.arange(t, device=q.device)
                # allow positions j where i-j < sliding_window AND j <= i
                sw = self.sliding_window
                mask = (idx[:, None] >= idx[None, :]) & (
                    (idx[:, None] - idx[None, :]) < sw
                )
                attn_mask = mask  # [T, T] boolean
            o_i = F.scaled_dot_product_attention(
                q_i,
                k_i,
                v_i,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=(attn_mask is None),
                scale=self.scale,
            )
            out[s:e] = o_i.squeeze(0).transpose(0, 1)  # [T, H, D]

        return out.reshape(total_tokens, self.num_heads * self.head_dim)
