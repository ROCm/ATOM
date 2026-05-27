from typing import Type

import torch


class AiterMhaBackendForVllm:
    """vLLM-facing MHA backend surface for ATOM attention layers."""

    accept_output_buffer: bool = False
    supported_dtypes: list = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        sentinel = 1234567
        shape = cls.get_kv_cache_shape(
            sentinel,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return shape.index(sentinel)

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if cls.supports_block_size(default_block_size):
            return default_block_size
        return 16

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> Type:
        from atom.plugin.vllm.attention.metadata import AiterMhaMetadataBuilderForVllm

        return AiterMhaMetadataBuilderForVllm

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMHA

        return AttentionForVllmMHA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterMlaBackendForVllm:
    """vLLM-facing dense MLA backend surface for ATOM attention layers."""

    accept_output_buffer: bool = True
    supported_dtypes: list = [torch.float16, torch.bfloat16]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 1

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3) if include_num_layers_dimension else (0, 1, 2)

    @staticmethod
    def get_builder_cls() -> Type:
        from atom.plugin.vllm.attention.metadata import AiterMlaMetadataBuilderForVllm

        return AiterMlaMetadataBuilderForVllm

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterSparseMlaBackendForVllm(AiterMlaBackendForVllm):
    """vLLM-facing sparse MLA backend surface for ATOM attention layers."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [1, 64]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # Prefer block_size == 64 so the indexer's preshuffled path is taken.
        return 64

    @staticmethod
    def get_builder_cls() -> Type:
        from atom.plugin.vllm.attention.metadata import AiterMlaSparseMetadataBuilder

        return AiterMlaSparseMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterSparseMlaIndexerBackendForVllm(AiterMlaBackendForVllm):
    """vLLM-facing sparse MLA indexer backend surface."""

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
            AiterMlaSparseIndexerMetadataBuilder,
        )

        return AiterMlaSparseIndexerMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class GDNAttentionBackend:
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_impl_cls() -> Type:
        from atom.plugin.vllm.attention.layer_gdn import GatedDeltaNet

        return GatedDeltaNet
