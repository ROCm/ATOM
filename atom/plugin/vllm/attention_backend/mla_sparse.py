# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Type

from atom.model_ops.attention_mla import MLAAttention
from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
from atom.model_ops.attentions.backends import AttentionBackend
from atom.plugin.attention import (
    AiterBackendDecoratorForPluginMode,
    AiterMLASparseAttentionMetadataBuilderDecoratorForPluginMode,
    AiterMLASparseIndexerAttentionMetadataBuilderDecoratorForPluginMode,
)
from atom.plugin.prepare import is_plugin_mode


@AiterBackendDecoratorForPluginMode
class AiterMLASparseBackend(AttentionBackend):
    """
    Sparse MLA attention backend for main attention layers to provide sparse
    metadata builder for top-k index conversion and ragged kernel call.
    """

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_SPARSE" if not is_plugin_mode() else "CUSTOM_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> Type["AiterMLASparseMetadataBuilder"]:
        return AiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def is_mla(cls) -> bool:
        return True


@AiterMLASparseAttentionMetadataBuilderDecoratorForPluginMode(
    default_base_class=AiterMLAMetadataBuilder
)
class AiterMLASparseMetadataBuilder(AiterMLAMetadataBuilder):
    """Metadata builder for sparse MLA.
    In standalone mode, delegates to CommonAttentionBuilder.
    In plugin mode, the decorator replaces __init__ and build() methods.
    """

    pass


@AiterBackendDecoratorForPluginMode
class AiterMLASparseIndexerBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return (
            "ROCM_AITER_MLA_SPARSE_INDEXER"
            if not is_plugin_mode()
            else "CUSTOM_MLA_SPARSE_INDEXER"
        )

    @staticmethod
    def get_builder_cls() -> Type["AiterMLASparseIndexerMetadataBuilder"]:
        return AiterMLASparseIndexerMetadataBuilder

    @staticmethod
    def get_impl_cls() -> Type["MLAAttention"]:
        return MLAAttention

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def is_mla(cls) -> bool:
        return True


@AiterMLASparseIndexerAttentionMetadataBuilderDecoratorForPluginMode(
    default_base_class=AiterMLAMetadataBuilder
)
class AiterMLASparseIndexerMetadataBuilder(AiterMLAMetadataBuilder):
    pass
