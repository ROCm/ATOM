# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch-native attention backend for ATOM (gfx1201 / RDNA4).

Why this exists
---------------
The AITER package shipped in rocm/atom-dev:latest has prebuilt HIP .so files
only for gfx94x/95x. On gfx1201 the AITER paged-attention HIP modules fail
to load with 'No compatible code objects found for: gfx1201' and SIGSEGV
the ModelRunner subprocess. This backend is an in-tree torch-only path that
does not load any of those prebuilt modules.

Selection
---------
atom/utils/selector.py:get_attn_backend_cls routes here when
torch.cuda.get_device_properties(0).gcnArchName starts with 'gfx1201',
or when ATOM_TORCH_NATIVE_ATTN=1 is set on any device.

KV cache layout
---------------
We use a single contiguous tensor per backend:

    runner.kv_cache : [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                     |--K-and-V--||--per-layer--||----flat slot index space----|

`build_kv_cache_tensor` slices `runner.kv_cache[0, layer_id]` for K and
`[1, layer_id]` for V, exposing them on each `PagedAttention` module as
`module.k_cache` / `module.v_cache` with shape
`[num_blocks, block_size, num_kv_heads, head_dim]`. The engine's
`slot_mapping` is a flat token-index that views this as
`(num_blocks * block_size, num_kv_heads, head_dim)`.

Forward path
------------
* Prefill: apply RoPE -> write current K/V to cache at slot_mapping ->
  per-sequence SDPA with `is_causal=True` over the in-batch K/V (no
  history needed because prefill carries the full sequence).
* Decode: apply RoPE -> write the new K/V at slot_mapping (one slot per
  request) -> for each request, gather the historical K/V from
  block_tables up to context_len, then SDPA with no causal mask
  (length-1 query).

Sliding window is honored via an explicit boolean mask in both paths.
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
# Backend
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


class TorchNativeMetadataBuilder(CommonAttentionBuilder):
    """Inherits prepare_prefill from CommonAttentionBuilder; provides
    decode metadata + KV cache allocation."""

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
        # We only support BF16 KV today; FP8 KV is a TODO.
        return torch.bfloat16

    def compute_block_bytes(self) -> int:
        n_layers, num_kv_heads, head_dim = self._kv_layout_dims()
        elem = self._kv_dtype().itemsize
        # 2 (K and V) * layers * block_size * heads * d * elem
        return 2 * n_layers * self.block_size * num_kv_heads * head_dim * elem

    def allocate_kv_cache_tensors(
        self, num_kv_heads: int, num_draft_layers: int
    ) -> dict:
        runner = self.model_runner
        n_layers, _, head_dim = self._kv_layout_dims()
        return {
            "kv_cache": torch.zeros(
                2,
                n_layers,
                runner.num_physical_kvcache_blocks,
                runner.physical_block_size,
                num_kv_heads,
                head_dim,
                dtype=self._kv_dtype(),
                device="cuda",
            ),
        }

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind one MHA module to its KV cache slice."""
        # Same module-detection as aiter: must be a non-MLA paged attention.
        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None

        runner = self.model_runner
        # Mirror layout: [num_blocks, block_size, num_kv_heads, head_dim]
        k_cache = runner.kv_cache[0, layer_id]
        v_cache = runner.kv_cache[1, layer_id]

        module.max_model_len = runner.config.max_model_len
        module.k_cache = k_cache
        module.v_cache = v_cache
        # Also expose to the inner impl since PagedAttention.forward delegates
        # to self.impl.forward and our impl reads its own k_cache/v_cache.
        if hasattr(module, "impl") and module.impl is not None:
            module.impl.k_cache = k_cache
            module.impl.v_cache = v_cache
        # Scales unused for BF16 KV; keep attributes for compatibility.
        if not hasattr(module, "k_scale"):
            module.k_scale = None
            module.v_scale = None

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
        max_seqlen_q = 1  # no spec decode in this backend yet
        block_size = self.model_runner.block_size

        context_lens = np.asarray(batch.context_lens, dtype=np.int32)
        block_tables = batch.block_tables

        # One slot per request: the last position in the last assigned block.
        slot_mapping = [
            block_table[-1] * block_size + last_block_num - 1
            for block_table, last_block_num in zip(
                block_tables, batch.last_block_num_tokens
            )
        ]
        # Decode positions = current context_len - 1 (zero-indexed) per request.
        positions = np.array(
            [cl - 1 for cl in context_lens[:scheduled_bs]], dtype=np.int32
        )
        max_seqlen_k = int(context_lens[:scheduled_bs].max()) if scheduled_bs > 0 else 0

        # Pad block_tables into a fixed [bs, max_blocks_per_seq] grid.
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

        # cu_seqlens_q is already prefilled in CommonAttentionBuilder.__init__
        # to [0, 1, 2, ...], which is exactly what decode needs.

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
    """Torch-only paged attention forward (prefill + decode + KV cache)."""

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
        # Placeholders; populated by build_kv_cache_tensor after engine_core.allocate_kv_cache.
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])
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
        k_cache: torch.Tensor,  # [B, S, H, D]
        v_cache: torch.Tensor,  # [B, S, H, D]
        slot_mapping: torch.Tensor,  # [N]
        k_new: torch.Tensor,  # [N, H, D]
        v_new: torch.Tensor,  # [N, H, D]
    ) -> None:
        # Filter out -1 sentinels (dummy padding slots).
        valid = slot_mapping >= 0
        if not bool(valid.all()):
            slot_mapping = slot_mapping[valid]
            k_new = k_new[valid]
            v_new = v_new[valid]
        if slot_mapping.numel() == 0:
            return
        flat_k = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
        # index_copy_ requires a 1D index and same dtype/device.
        flat_k.index_copy_(0, slot_mapping.long(), k_new.to(flat_k.dtype))
        flat_v.index_copy_(0, slot_mapping.long(), v_new.to(flat_v.dtype))

    def _gather_kv_for_request(
        self,
        k_cache: torch.Tensor,  # [B, S, H, D]
        v_cache: torch.Tensor,  # [B, S, H, D]
        block_table: torch.Tensor,  # [num_blocks_assigned], int
        context_len: int,
    ):
        # Pick out the assigned blocks, flatten to (blocks*S, H, D), trim to ctx.
        n_blocks_needed = (context_len + k_cache.shape[1] - 1) // k_cache.shape[1]
        bt = block_table[:n_blocks_needed].long()
        k_blocks = k_cache.index_select(0, bt)  # [n, S, H, D]
        v_blocks = v_cache.index_select(0, bt)
        flat_k = k_blocks.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        flat_v = v_blocks.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
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
                "TorchNativeAttentionImpl: MLA path is not implemented; "
                "this backend is for plain MHA."
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

        # RoPE (model passes flat layouts to rotary_emb)
        if self.rotary_emb is not None and positions is not None:
            q_flat = q.reshape(total_tokens, self.num_heads * self.head_dim)
            k_flat = k.reshape(total_tokens, self.num_kv_heads * self.head_dim)
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(total_tokens, self.num_heads, self.head_dim)
            k = k_flat.view(total_tokens, self.num_kv_heads, self.head_dim)

        # Write current K/V into the paged cache at slot_mapping
        slot_mapping = attn_md.slot_mapping
        # KV caches may not be allocated yet during warmup_model (engine_core
        # calls allocate_kv_cache after ModelRunner construction). Skip the
        # write in that case; the prefill path does not need the cache because
        # it has the full sequence in (k, v).
        if (
            slot_mapping is not None
            and getattr(self, "k_cache", torch.empty(0)).numel() > 0
            and getattr(self, "v_cache", torch.empty(0)).numel() > 0
        ):
            self._write_kv_cache(self.k_cache, self.v_cache, slot_mapping[:total_tokens], k, v)

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
        # Optional GQA expansion
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
            q_i = q[s:e].transpose(0, 1).unsqueeze(0)  # [1, H, T, D]
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
        q: torch.Tensor,  # [bs, num_heads, head_dim]  (one token per request)
        attn_md: AttentionMetaData,
    ) -> torch.Tensor:
        bs = q.shape[0]
        ctx_lens = attn_md.context_lens.detach().cpu().tolist()
        block_tables = attn_md.block_tables  # [bs, max_blocks_per_seq]
        sw = self.sliding_window

        outs = []
        for i in range(bs):
            ctx_len = int(ctx_lens[i])
            if ctx_len <= 0:
                # padding row: produce zeros so the shape is consistent
                outs.append(torch.zeros(self.num_heads, self.head_dim, dtype=q.dtype, device=q.device))
                continue
            # Gather past K/V (which now includes the just-written current token)
            k_past, v_past = self._gather_kv_for_request(
                self.k_cache, self.v_cache, block_tables[i], ctx_len
            )
            # GQA expansion to num_heads
            if self.num_kv_heads != self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_past = k_past.repeat_interleave(n_rep, dim=1)
                v_past = v_past.repeat_interleave(n_rep, dim=1)
            # Sliding window: keep only the last `sw` keys
            if sw is not None and sw > 0 and ctx_len > sw:
                k_past = k_past[-sw:]
                v_past = v_past[-sw:]
            # SDPA wants (B, H, T, D); for one request: q -> (1, H, 1, D);
            # k/v -> (1, H, T_kv, D).
            q_i = q[i : i + 1].unsqueeze(2)                       # (1, H, 1, D)
            k_i = k_past.transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, T, D)
            v_i = v_past.transpose(0, 1).unsqueeze(0).contiguous()
            o_i = F.scaled_dot_product_attention(
                q_i, k_i, v_i,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )
            outs.append(o_i.view(self.num_heads, self.head_dim))   # (H, D)
        out = torch.stack(outs, dim=0)  # [bs, H, D]
        return out.reshape(bs, self.num_heads * self.head_dim)
