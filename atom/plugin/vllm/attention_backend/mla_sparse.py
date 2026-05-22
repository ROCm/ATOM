# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type

from atom.plugin.vllm.attention.backend import AiterMlaBackendForVllm


class AiterMLASparseBackend(AiterMlaBackendForVllm):
    """
    Sparse MLA attention backend for main attention layers to provide sparse
    metadata builder for top-k index conversion and ragged kernel call.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1, 64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Prefer block_size == 64 so the indexer's preshuffled path is taken.
        return 64

    @staticmethod
    def get_builder_cls() -> Type:
        from atom.plugin.vllm.attention.metadata import AiterMLASparseMetadataBuilder

        return AiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def is_sparse(cls) -> bool:
        return True


class AiterMLASparseIndexerBackend(AiterMlaBackendForVllm):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1, 64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Prefer block_size == 64 so the indexer's preshuffled path is taken.
        return 64

    @staticmethod
    def get_builder_cls() -> Type:
        from atom.plugin.vllm.attention.metadata import (
            AiterMLASparseIndexerMetadataBuilder,
        )

        return AiterMLASparseIndexerMetadataBuilder

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def is_sparse(cls) -> bool:
        return True
