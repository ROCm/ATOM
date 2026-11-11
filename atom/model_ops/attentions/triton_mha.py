# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import ClassVar

import torch

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

    def _finish_prefill_batch(self, batch, attn_metadata, prefix=""):
        super()._finish_prefill_batch(batch, attn_metadata, prefix)
        # `unified_attention` needs a block table even for pure prefill. With
        # no paged one uploaded (`_prefill_block_table_rows`), a fake one with
        # block_size=1 treats the raw K/V as the kv_cache: row i =
        # [cu_seqlens_k[i], ..., cu_seqlens_k[i] + max - 1].
        if attn_metadata.block_tables is None:
            cu_k = attn_metadata.cu_seqlens_k
            # `cu_k` is padded past them; this table is one row per request.
            offsets = cu_k[: batch.scheduled_bs]
            attn_metadata.block_tables = offsets.unsqueeze(1) + torch.arange(
                attn_metadata.max_seqlen_k, dtype=torch.int32, device=cu_k.device
            )

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
