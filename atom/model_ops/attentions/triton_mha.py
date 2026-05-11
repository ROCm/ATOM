# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Type

import torch

import atom.model_ops as ops
from atom.config import KVCacheTensor
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.paged_attention import PagedAttention

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
        attn_cls = ops.Attention
        if attn_cls == PagedAttention:
            return PagedAttentionImpl
        raise NotImplementedError(
            f"TritonMHABackend does not support attention class {attn_cls!r}"
        )


class TritonMHAMetadataBuilder(AiterAttentionMetadataBuilder):
    """MHA metadata builder that allocates KV cache in flash layout.

    Flash layout: K/V both [num_blocks, block_size, num_kv_heads, head_dim].
    Consumed directly by aiter triton `unified_attention` for prefill+decode.
    """

    def prepare_prefill(self, batch: ScheduledBatch):
        attn_metadata, positions = super().prepare_prefill(batch)

        # When there are no cached tokens, the base builder leaves
        # `block_tables=None` because AiterBackend's prefill consumes raw q/k/v
        # via flash_attn_varlen_func. The unified_attention path used by
        # TritonMHABackend instead requires a block_table even for pure prefill,
        # so build a fake one here that treats raw K/V as a kv_cache with
        # block_size=1: row i = [cu_seqlens_k[i], ..., cu_seqlens_k[i]+max-1].
        if attn_metadata.block_tables is None:
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

        k_cache = runner.kv_cache[0, attn_idx].view(
            runner.num_physical_kvcache_blocks,
            runner.physical_block_size,
            runner.num_kv_heads,
            hf_config.head_dim,
        )
        v_cache = runner.kv_cache[1, attn_idx].view(
            runner.num_physical_kvcache_blocks,
            runner.physical_block_size,
            runner.num_kv_heads,
            hf_config.head_dim,
        )
        if config.kv_cache_dtype == "fp8":
            module.k_scale = runner.kv_scale[0, attn_idx]
            module.v_scale = runner.kv_scale[1, attn_idx]

        module.max_model_len = config.max_model_len
        module.k_cache = k_cache
        module.v_cache = v_cache
        if impl is not None:
            impl.use_flash_layout = True

        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=module.k_scale,
            v_scale=module.v_scale,
        )
