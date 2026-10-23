# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from atom.model_ops.attention_mla import MLAAttention

from .aiter_mla import AiterMLAMetadataBuilder
from .backends import AttentionBackend


class TritonMLABackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_TRITON_MLA"

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MLAAttention"]:
        return MLAAttention


class TritonMLAMetadataBuilder(AiterMLAMetadataBuilder):

    # The paged Triton MLA kernel consumes the base builder's block tables,
    # context lengths and query offsets directly, including during MTP/capture.
    def set_mla_persistent_worker_buffers(
        self, bs, max_q_len, only_update=False, num_reject_tokens=None, **kwargs
    ):
        # Triton MLA does not use aiter persistent worker buffers.
        # `**kwargs` absorbs arguments the aiter caller grows over time (e.g.
        # `is_cp_round_robin`); this override returns nothing, so ignoring them
        # is correct, and pinning the exact signature only breaks the Triton
        # path whenever the aiter side adds a keyword.
        return {}
