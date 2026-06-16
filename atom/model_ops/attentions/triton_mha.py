# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Type

import torch

from atom.config import KVCacheTensor
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.utils import envs

from .aiter_attention import AiterAttentionMetadataBuilder
from .backends import AttentionBackend

logger = logging.getLogger("atom")


class TritonMHABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_TRITON_MHA"

    @staticmethod
    def get_builder_cls() -> Type["TritonMHAMetadataBuilder"]:
        return TritonMHAMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return PagedAttentionImpl


class TritonMHAMetadataBuilder(AiterAttentionMetadataBuilder):
    """MHA metadata builder that allocates KV cache in 5D SHUFFLE layout.

    SHUFFLE layout (x = 16 // itemsize):
      K [num_blocks, num_kv_heads, head_dim // x, block_size, x]
      V [num_blocks, num_kv_heads, block_size // x, head_dim, x]
    Consumed by aiter triton `unified_attention` with `shuffled_kv_cache=True`
    for both prefill and decode.
    """

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = super().prepare_prefill(batch)

        # When there are no cached tokens, the base builder leaves
        # `block_tables=None` because AiterBackend's prefill consumes raw q/k/v
        # via flash_attn_varlen_func. The unified_attention path used by
        # TritonMHABackend instead requires a block_table even for pure prefill,
        # so build a fake one here that treats raw K/V as a kv_cache with
        # block_size=1: row i = [cu_seqlens_k[i], ..., cu_seqlens_k[i]+max-1].
        # TritonMHABackend instead requires a block_table even for pure prefill.
        if attn_metadata.block_tables is None:
            if envs.ATOM_USE_UNIFIED_ATTN and batch.block_tables:
                # Unified attention does better consuming paged KV: read the new
                # tokens straight from the paged flash-layout KV cache (already
                # written during rope_cache via slot_mapping) using the real
                # per-seq block_table, identical to the prefix-cache-hit path.
                # The base builder only uploads `block_tables` when `has_cached`,
                # so do it here for pure prefill and flag the consumer to read
                # from the cache.
                bs = batch.total_seqs_num_prefill
                self.prepare_block_tables(batch)
                attn_metadata.block_tables = self.model_runner.forward_vars[
                    "block_tables"
                ].copy_to_gpu(bs)
            else:
                # Fallback: build a fake block_size=1 block_table that treats
                # raw K/V as a kv_cache. row i = [cu_seqlens_k[i], ...,
                # cu_seqlens_k[i]+max-1].
                cu_k = attn_metadata.cu_seqlens_k
                num_seqs = cu_k.shape[0] - 1
                offsets = cu_k[:num_seqs]
                attn_metadata.block_tables = offsets.unsqueeze(1) + torch.arange(
                    attn_metadata.max_seqlen_k, dtype=torch.int32, device=cu_k.device
                )

        return attn_metadata, positions

    def build_kv_cache_tensor(self, layer_id: int, module):
        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None

        runner = self.model_runner
        config = runner.config
        hf_config = config.hf_config

        if runner.is_mimo_v2():
            raise NotImplementedError(
                "TritonMHABackend does not support MiMo-V2 (per-layer alloc path)"
            )

        impl = getattr(module, "impl", None)
        if impl is not None and (
            getattr(impl, "rotary_emb", None) is not None
            and getattr(impl, "q_norm", None) is not None
            and getattr(impl, "k_norm", None) is not None
        ):
            raise NotImplementedError(
                "TritonMHABackend is incompatible with the fused qk_norm+rope+shuffle "
                "cache path; use AiterBackend for this model."
            )

        if runner.is_qwen_next():
            mtp_start = runner.mtp_start_layer_idx
            if layer_id < mtp_start:
                attn_idx = layer_id // runner.full_attention_interval
            else:
                attn_idx = runner.num_full_attn + (layer_id - mtp_start)
        else:
            attn_idx = layer_id

        # 5D SHUFFLE (pre-shuffled) layout, consumed by
        # unified_attention(shuffled_kv_cache=True) for prefill+decode:
        #   K [num_blocks, num_kv_heads, head_dim // x, block_size, x]
        #   V [num_blocks, num_kv_heads, block_size // x, head_dim, x]
        x = 16 // runner.kv_cache.element_size()
        k_cache = runner.kv_cache[0, attn_idx].view(
            runner.num_physical_kvcache_blocks,
            runner.num_kv_heads,
            hf_config.head_dim // x,
            runner.physical_block_size,
            x,
        )
        v_cache = runner.kv_cache[1, attn_idx].view(
            runner.num_physical_kvcache_blocks,
            runner.num_kv_heads,
            runner.physical_block_size // x,
            hf_config.head_dim,
            x,
        )
        if config.kv_cache_dtype == "fp8":
            module.k_scale = runner.kv_scale[0, attn_idx]
            module.v_scale = runner.kv_scale[1, attn_idx]

        module.max_model_len = config.max_model_len
        module.k_cache = k_cache
        module.v_cache = v_cache
        if impl is not None:
            # KV cache is no longer in flash (4D) layout; unified_attention is
            # selected via ATOM_USE_UNIFIED_ATTN, and reads the SHUFFLE layout.
            impl.use_flash_layout = False

        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=module.k_scale,
            v_scale=module.v_scale,
        )
