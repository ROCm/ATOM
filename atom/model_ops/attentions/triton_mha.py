# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import ClassVar

import torch

from atom.model_engine.scheduler import ScheduledBatch
from atom.utils import envs

from .aiter_attention import AiterAttentionMetadataBuilder, AiterBackend

logger = logging.getLogger("atom")


class TritonMHABackend(AiterBackend):
    """`AiterBackend`'s cache read by a different kernel.

    A subclass and not a peer because that is the whole difference: one pool,
    one layout, one block rule, and the two answer `get_impl_cls` alike. Stated
    by inheritance so a rule added to the base cannot be forgotten here -- the
    pool and the block size used to be forwarded by hand.
    """

    @staticmethod
    def get_name() -> str:
        return "ROCM_TRITON_MHA"

    @staticmethod
    def get_builder_cls() -> type["TritonMHAMetadataBuilder"]:
        return TritonMHAMetadataBuilder


class TritonMHAMetadataBuilder(AiterAttentionMetadataBuilder):
    """MHA metadata builder that allocates KV cache in 5D SHUFFLE layout.

    SHUFFLE layout (x = 16 // itemsize):
      K [num_blocks, num_kv_heads, head_dim // x, block_size, x]
      V [num_blocks, num_kv_heads, block_size // x, head_dim, x]
    Consumed by aiter triton `unified_attention` with `shuffled_kv_cache=True`
    for both prefill and decode.
    """

    BACKEND: ClassVar[type[AiterBackend]] = TritonMHABackend

    def prepare_prefill(self, batch: ScheduledBatch, running_bs: int):
        attn_metadata, positions = super().prepare_prefill(batch, running_bs)

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
                # The base builder marshals `block_tables` every step but only
                # uploads it when `has_cached`, so upload it here for pure
                # prefill and flag the consumer to read from the cache.
                bs = batch.total_seqs_num_prefill
                attn_metadata.block_tables = self.model_runner.forward_vars[
                    "block_tables"
                ].copy_to_gpu(bs)
            else:
                # Fallback: build a fake block_size=1 block_table that treats
                # raw K/V as a kv_cache. row i = [cu_seqlens_k[i], ...,
                # cu_seqlens_k[i]+max-1].
                cu_k = attn_metadata.cu_seqlens_k
                # `cu_k` is padded past them; this table is one row per request.
                num_seqs = batch.total_seqs_num_prefill
                offsets = cu_k[:num_seqs]
                attn_metadata.block_tables = offsets.unsqueeze(1) + torch.arange(
                    attn_metadata.max_seqlen_k, dtype=torch.int32, device=cu_k.device
                )

        return attn_metadata, positions

    def build_kv_cache_tensor(self, module):
        """The parent's bind, plus the one thing that differs: the layout flag.

        Both backends read one pool at one layout, so there is nothing to say
        about shapes here — this used to say it again, and was one of the four
        copies that let a V view drift.
        """
        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None

        # Ahead of the parent call, and so duplicating its guard: the refusal
        # below is about *this* module, and a model's non-attention modules
        # must not trip it. (The MiMo-V2 refusal that used to sit here went
        # with the per-layer allocation path it named -- a second KV-head
        # geometry is a second pool now, which this backend reads like any
        # other.)
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

        bound = super().build_kv_cache_tensor(module)
        if bound is not None and impl is not None:
            # KV cache is not in flash (4D) layout; unified_attention is
            # selected via ATOM_USE_UNIFIED_ATTN, and reads the SHUFFLE layout.
            impl.use_flash_layout = False
        return bound
