# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch-native (with triton-fast paths) attention backend for ATOM (gfx1201).

Why this exists
---------------
The AITER package shipped in rocm/atom-dev:latest has prebuilt HIP .so files
only for gfx94x/95x. On gfx1201 (RDNA4) the AITER paged-attention HIP modules
fail to load with "No compatible code objects found for: gfx1201" and SIGSEGV
the ModelRunner. This backend replaces that path with a mix of triton (fast)
and torch (correctness fallback) kernels that work on gfx1201.

Selection
---------
atom/utils/selector.py:get_attn_backend_cls routes here when
torch.cuda.get_device_properties(0).gcnArchName starts with 'gfx1201',
or when ATOM_TORCH_NATIVE_ATTN=1 is set.

KV cache layout (matches aiter's pa_decode triton kernel expectations)
----------------------------------------------------------------------
    runner.kv_cache : [2, num_layers, num_blocks, num_kv_heads, block_size, head_dim]
                     |--K-and-V--||--per-layer--||---paged storage in aiter format---|

Forward
-------
* Prefill: write current K/V at slot_mapping into the cache, then run
  per-sequence SDPA over the in-batch K/V (no history needed because
  prefill carries the full sequence).
* Decode: write the new K/V at slot_mapping (one slot per request),
  then call aiter's `paged_attention_decode` triton kernel
  (~1.8x faster than the torch gather + SDPA loop on gfx1201).
  Falls back to the torch path if the triton kernel raises (e.g. unusual
  shapes, sliding window, or a kernel-side AssertionError).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from atom.config import KVCacheTensor
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attentions.backends import (
    AttentionBackend,
    AttentionImpl,
    CommonAttentionBuilder,
)
from atom.utils.forward_context import AttentionMetaData, get_forward_context

logger = logging.getLogger("atom")


def _is_gfx1201() -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_properties(0).gcnArchName or ""
    return name.startswith("gfx1201")


def use_torch_native_attn() -> bool:
    if os.environ.get("ATOM_TORCH_NATIVE_ATTN", "").lower() in ("1", "true"):
        return True
    return _is_gfx1201()


# ---------------------------------------------------------------------------
# Cached triton paged-attention decode kernel
# ---------------------------------------------------------------------------
_TRITON_PA_DECODE = None
_TRITON_TL_BF16 = None
_TRITON_PREFILL = None


def _get_triton_prefill():
    global _TRITON_PREFILL
    if _TRITON_PREFILL is None:
        try:
            from aiter.ops.triton.attention.prefill_attention import context_attention_fwd
            _TRITON_PREFILL = context_attention_fwd
        except Exception as e:
            logger.warning("triton context_attention_fwd unavailable: %s", e)
            _TRITON_PREFILL = False
    return _TRITON_PREFILL if _TRITON_PREFILL is not False else None


def _get_triton_pa_decode():
    global _TRITON_PA_DECODE, _TRITON_TL_BF16
    if _TRITON_PA_DECODE is None:
        try:
            from aiter.ops.triton.attention.pa_decode import paged_attention_decode
            import triton.language as tl
            _TRITON_PA_DECODE = paged_attention_decode
            _TRITON_TL_BF16 = tl.bfloat16
        except Exception as e:
            logger.warning("triton paged_attention_decode unavailable: %s", e)
            _TRITON_PA_DECODE = False
    return (
        (_TRITON_PA_DECODE, _TRITON_TL_BF16)
        if _TRITON_PA_DECODE is not False
        else (None, None)
    )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TorchNativeBackend(AttentionBackend):
    """AITER-free attention backend (torch + selectively triton)."""

    @staticmethod
    def get_name() -> str:
        return "TORCH_NATIVE_ATTENTION"

    @staticmethod
    def get_builder_cls() -> Type["TorchNativeMetadataBuilder"]:
        return TorchNativeMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["TorchNativeAttentionImpl"]:
        return TorchNativeAttentionImpl


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


class TorchNativeMetadataBuilder(CommonAttentionBuilder):
    """Inherits prepare_prefill from CommonAttentionBuilder; provides decode
    metadata + KV cache allocation in aiter's [blocks, heads, block_size, d]
    layout."""

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

    # ------------------------------------------------------------------ #
    # KV pool sizing                                                     #
    # ------------------------------------------------------------------ #

    def _kv_layout_dims(self):
        runner = self.model_runner
        hf = runner.config.hf_config
        head_dim = getattr(hf, "head_dim", None) or (
            hf.hidden_size // hf.num_attention_heads
        )
        num_kv_heads = max(1, runner._get_num_kv_heads())
        n_layers = runner._get_total_num_layers()
        return n_layers, num_kv_heads, head_dim

    def _kv_dtype(self):
        return torch.bfloat16

    def compute_block_bytes(self) -> int:
        n_layers, num_kv_heads, head_dim = self._kv_layout_dims()
        elem = self._kv_dtype().itemsize
        return 2 * n_layers * self.block_size * num_kv_heads * head_dim * elem

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        runner = self.model_runner
        n_layers, _, head_dim = self._kv_layout_dims()
        # aiter pa_decode expects [num_blocks, num_kv_heads, block_size, head_dim].
        return {
            "kv_cache": torch.zeros(
                2,
                n_layers,
                runner.num_physical_kvcache_blocks,
                num_kv_heads,
                runner.physical_block_size,
                head_dim,
                dtype=self._kv_dtype(),
                device="cuda",
            ),
        }

    def build_kv_cache_tensor(self, layer_id: int, module):
        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None

        runner = self.model_runner
        # [num_blocks, num_kv_heads, block_size, head_dim]
        k_cache = runner.kv_cache[0, layer_id]
        v_cache = runner.kv_cache[1, layer_id]

        module.max_model_len = runner.config.max_model_len
        module.k_cache = k_cache
        module.v_cache = v_cache
        if not hasattr(module, "k_scale"):
            module.k_scale = None
            module.v_scale = None

        if hasattr(module, "impl") and module.impl is not None:
            module.impl.k_cache = k_cache
            module.impl.v_cache = v_cache

        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=module.k_scale,
            v_scale=module.v_scale,
        )

    # ------------------------------------------------------------------ #
    # Decode metadata                                                    #
    # ------------------------------------------------------------------ #

    def prepare_decode(self, batch: ScheduledBatch, bs: int):
        scheduled_bs = batch.total_seqs_num_decode
        max_seqlen_q = 1
        block_size = self.model_runner.block_size

        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables

        slot_mapping = [
            block_table[-1] * block_size + last_block_num - 1
            for block_table, last_block_num in zip(
                block_tables, batch.last_block_num_tokens
            )
        ]
        positions = np.array(
            [cl - 1 for cl in context_lens[:scheduled_bs]], dtype=np.int32
        )
        max_seqlen_k = int(context_lens[:scheduled_bs].max()) if scheduled_bs > 0 else 0

        self.prepare_block_tables(batch)

        var = self.model_runner.forward_vars
        sum_scheduled_tokens = batch.total_tokens_num_decode
        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping[
                :sum_scheduled_tokens
            ]
        var["positions"].np[:sum_scheduled_tokens] = positions[:sum_scheduled_tokens]
        var["context_lens"].np[:scheduled_bs] = context_lens[:scheduled_bs]
        var["context_lens"].np[scheduled_bs:bs] = 0

        vars_used = [
            ("slot_mapping", bs * max_seqlen_q),
            ("context_lens", bs),
            ("cu_seqlens_q", bs + 1),
            ("block_tables", bs),
        ]
        ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}

        attn_metadata = AttentionMetaData(
            max_seqlen_q=max_seqlen_q,
            min_seqlen_q=0,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            **ctx,
        )
        positions_gpu = var["positions"].copy_to_gpu(sum_scheduled_tokens)
        return attn_metadata, positions_gpu

    def build_for_cudagraph_capture(self, bs: int):
        raise NotImplementedError(
            "build_for_cudagraph_capture: run with --enforce-eager --level 0 "
            "(CUDAGraph capture not yet supported by torch-native backend)."
        )


# ---------------------------------------------------------------------------
# Attention impl
# ---------------------------------------------------------------------------


class TorchNativeAttentionImpl(AttentionImpl):
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
        self.head_size = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.kv_cache_dtype = kv_cache_dtype
        self.layer_num = layer_num
        self.rotary_emb = rotary_emb
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.q_size = num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        # Set by build_kv_cache_tensor after engine_core.allocate_kv_cache.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
        # Reusable scale tensors for the triton paged-attention kernel
        # (BF16 KV path -> identity scales).
        self._pa_k_scale = None
        self._pa_v_scale = None
        if kv_cache_dtype != "bf16":
            logger.warning(
                f"TorchNativeAttentionImpl: kv_cache_dtype={kv_cache_dtype} "
                "is a TODO; force --kv_cache_dtype bf16."
            )

    # ------------------------------------------------------------------ #
    # KV cache helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_kv_cache(
        k_cache: torch.Tensor,  # [B, H, S, D] (aiter layout)
        v_cache: torch.Tensor,  # [B, H, S, D]
        slot_mapping: torch.Tensor,  # [N] flat slot indices = block * S + within
        k_new: torch.Tensor,  # [N, H, D]
        v_new: torch.Tensor,  # [N, H, D]
    ) -> None:
        valid = slot_mapping >= 0
        if not bool(valid.all()):
            slot_mapping = slot_mapping[valid]
            k_new = k_new[valid]
            v_new = v_new[valid]
        if slot_mapping.numel() == 0:
            return
        S = k_cache.shape[2]
        slot_mapping = slot_mapping.long()
        block_idx = slot_mapping // S       # [N]
        within = slot_mapping % S           # [N]
        # Advanced indexing: cache[I, :, J, :] for parallel (I, J) of length N
        # gives a (N, H, D) view; assignment from (N, H, D) writes back.
        k_cache[block_idx, :, within, :] = k_new.to(k_cache.dtype)
        v_cache[block_idx, :, within, :] = v_new.to(v_cache.dtype)

    def _gather_kv_for_request(
        self,
        k_cache: torch.Tensor,  # [B, H, S, D]
        v_cache: torch.Tensor,  # [B, H, S, D]
        block_table: torch.Tensor,  # [num_blocks_assigned], int
        context_len: int,
    ):
        S = k_cache.shape[2]
        n_blocks_needed = (context_len + S - 1) // S
        bt = block_table[:n_blocks_needed].long()
        k_blocks = k_cache.index_select(0, bt)  # [n, H, S, D]
        v_blocks = v_cache.index_select(0, bt)
        # (n, H, S, D) -> (n*S, H, D) via permute + reshape (forces contiguous copy
        # — one-time per request, only used when the triton path falls back).
        flat_k = k_blocks.permute(0, 2, 1, 3).reshape(
            -1, k_cache.shape[1], k_cache.shape[3]
        )
        flat_v = v_blocks.permute(0, 2, 1, 3).reshape(
            -1, v_cache.shape[1], v_cache.shape[3]
        )
        return flat_k[:context_len], flat_v[:context_len]

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
        if use_mla:
            raise NotImplementedError(
                "TorchNativeAttentionImpl: MLA path is not implemented."
            )

        ctx = get_forward_context()
        attn_md: Optional[AttentionMetaData] = ctx.attn_metadata
        fc = ctx.context
        is_prefill = bool(getattr(fc, "is_prefill", True)) if fc is not None else True
        if attn_md is None:
            raise RuntimeError(
                "TorchNativeAttentionImpl: forward called without AttentionMetaData."
            )

        total_tokens = query.shape[0]
        q = query.view(total_tokens, self.num_heads, self.head_dim)
        k = key.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = value.view(total_tokens, self.num_kv_heads, self.head_dim)

        if self.rotary_emb is not None and positions is not None:
            q_flat = q.reshape(total_tokens, self.num_heads * self.head_dim)
            k_flat = k.reshape(total_tokens, self.num_kv_heads * self.head_dim)
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(total_tokens, self.num_heads, self.head_dim)
            k = k_flat.view(total_tokens, self.num_kv_heads, self.head_dim)

        slot_mapping = attn_md.slot_mapping
        if (
            slot_mapping is not None
            and getattr(self, "k_cache", torch.empty(0)).numel() > 0
            and getattr(self, "v_cache", torch.empty(0)).numel() > 0
        ):
            self._write_kv_cache(
                self.k_cache, self.v_cache, slot_mapping[:total_tokens], k, v
            )

        if is_prefill:
            return self._forward_prefill(q, k, v, attn_md, total_tokens)
        return self._forward_decode(q, attn_md)

    # ---------------- prefill ---------------- #

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_md: AttentionMetaData,
        total_tokens: int,
    ) -> torch.Tensor:
        # Prefer triton context_attention_fwd (handles GQA internally; ~2x
        # faster than the torch SDPA loop on gfx1201 at gsm8k context lengths).
        # Falls back to per-sequence torch SDPA when sliding window is active
        # (kernel doesn't support it) or on any kernel exception.
        sw_active = self.sliding_window is not None and self.sliding_window > 0
        prefill = _get_triton_prefill()
        if prefill is not None and not sw_active:
            try:
                out = torch.empty_like(q)
                cu_q_gpu = attn_md.cu_seqlens_q.to(torch.int32)
                # b_start_loc = cu_seqlens_q[:-1]; b_seq_len = diffs.
                b_start_loc = cu_q_gpu[:-1].contiguous()
                b_seq_len = (cu_q_gpu[1:] - cu_q_gpu[:-1]).contiguous()
                prefill(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    out,
                    b_start_loc,
                    b_seq_len,
                    int(attn_md.max_seqlen_q),
                    is_causal=True,
                )
                return out.reshape(total_tokens, self.num_heads * self.head_dim)
            except Exception as e:
                logger.warning(
                    "triton context_attention_fwd raised %s; falling back to torch SDPA", e
                )

        # Torch fallback: per-sequence SDPA loop.
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        cu_q = attn_md.cu_seqlens_q.detach().cpu().tolist()
        out = torch.empty_like(q)
        for i in range(len(cu_q) - 1):
            s, e = int(cu_q[i]), int(cu_q[i + 1])
            if s == e:
                continue
            q_i = q[s:e].transpose(0, 1).unsqueeze(0)
            k_i = k[s:e].transpose(0, 1).unsqueeze(0)
            v_i = v[s:e].transpose(0, 1).unsqueeze(0)
            attn_mask = None
            if self.sliding_window is not None and self.sliding_window > 0:
                t = e - s
                idx = torch.arange(t, device=q.device)
                attn_mask = (idx[:, None] >= idx[None, :]) & (
                    (idx[:, None] - idx[None, :]) < self.sliding_window
                )
            o_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=(attn_mask is None),
                scale=self.scale,
            )
            out[s:e] = o_i.squeeze(0).transpose(0, 1)
        return out.reshape(total_tokens, self.num_heads * self.head_dim)

    # ---------------- decode ---------------- #

    def _forward_decode(
        self,
        q: torch.Tensor,  # [bs, num_q_heads, head_dim]
        attn_md: AttentionMetaData,
    ) -> torch.Tensor:
        bs = q.shape[0]
        # Prefer triton paged-attention decode kernel; fall back to torch on any error.
        pa_decode, tl_bf16 = _get_triton_pa_decode()
        # Sliding window not supported by aiter pa_decode -> fall back if active.
        sw_active = self.sliding_window is not None and self.sliding_window > 0
        if pa_decode is not None and not sw_active and self.k_cache.numel() > 0:
            try:
                out = torch.empty_like(q)
                if self._pa_k_scale is None or self._pa_k_scale.device != q.device:
                    self._pa_k_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
                    self._pa_v_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
                # block_tables to int32 (kernel expects int32)
                block_tables = attn_md.block_tables[:bs].to(torch.int32)
                seq_lens = attn_md.context_lens[:bs].to(torch.int32)
                pa_decode(
                    out, q.contiguous(),
                    self.k_cache, self.v_cache,
                    seq_lens, block_tables,
                    float(self.scale), int(attn_md.max_seqlen_k),
                    tl_bf16, self._pa_k_scale, self._pa_v_scale,
                )
                return out.reshape(bs, self.num_heads * self.head_dim)
            except Exception as e:
                logger.warning(
                    "triton paged_attention_decode raised %s; falling back to torch", e
                )

        # Torch fallback: per-request gather + SDPA (correct, slower).
        ctx_lens = attn_md.context_lens.detach().cpu().tolist()
        block_tables = attn_md.block_tables
        outs = []
        for i in range(bs):
            ctx_len = int(ctx_lens[i])
            if ctx_len <= 0:
                outs.append(
                    torch.zeros(
                        self.num_heads, self.head_dim, dtype=q.dtype, device=q.device
                    )
                )
                continue
            k_past, v_past = self._gather_kv_for_request(
                self.k_cache, self.v_cache, block_tables[i], ctx_len
            )
            if self.num_kv_heads != self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_past = k_past.repeat_interleave(n_rep, dim=1)
                v_past = v_past.repeat_interleave(n_rep, dim=1)
            if self.sliding_window is not None and self.sliding_window > 0 and ctx_len > self.sliding_window:
                k_past = k_past[-self.sliding_window:]
                v_past = v_past[-self.sliding_window:]
            q_i = q[i : i + 1].unsqueeze(2)                         # (1, H, 1, D)
            k_i = k_past.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, T, D)
            v_i = v_past.transpose(0, 1).unsqueeze(0).contiguous()
            o_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )
            outs.append(o_i.view(self.num_heads, self.head_dim))
        out = torch.stack(outs, dim=0)
        return out.reshape(bs, self.num_heads * self.head_dim)
