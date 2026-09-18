from dataclasses import replace
from typing import ClassVar

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from atom.model_ops.minimax_m3.sparse_attn import SPARSE_BLOCK_SIZE


class _VllmAttentionBackendCompat:
    """Compatibility surface for duck-typed ATOM attention backends."""

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Publish the logical ``[B, H, N, C]`` page shape ATOM kernels use."""
        return spec

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        """ATOM kernels address a block as one dense, contiguous page.

        Every ATOM backend reinterprets its per-layer view with ``.view()`` into
        a page-16 kernel layout, which needs each block's ``[H, N, C]`` bytes to
        form a single contiguous run (``is_block_compact``). LBHNC is also the
        only such layout that keeps a layer contiguous, so the whole plugin
        pins it rather than negotiating per backend: mixed-HNC models (any
        hybrid attention/GDN or main/indexer pair) narrow the candidate set to
        block-compact layouts anyway.
        """
        return (KVCacheLayout.LBHNC,)

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        """ATOM metadata builders plan from exact CPU query boundaries."""
        return False


class AiterMhaBackendForVllm(_VllmAttentionBackendCompat):
    """vLLM-facing MHA backend surface for ATOM attention layers."""

    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes():
        # Keep the physical kernel page at 16 even when vLLM's hybrid KV manager
        # uses a larger logical page. Advertising arbitrary multiples makes
        # fp8 hybrid models execute cache kernels against the unsplit logical
        # page and corrupts TP output.
        return [16]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if cls.supports_block_size(default_block_size):
            return default_block_size
        return 16

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """Give K and V a head slot each so a page is ``[K page | V page]``.

        This is vLLM 0.29's spelling of the ``(2, num_blocks, block_size,
        num_kv_heads, head_size)`` cache ATOM used through 0.28: the K/V axis
        moves inside the block, and ``AttentionForVllmMHA._split_kv_cache``
        re-derives the K and V page planes from it.
        """
        if spec.state_content_bytes is not None:
            return spec
        assert (
            spec.head_size == spec.head_size_v
        ), "ATOM MHA stores K and V as two equally sized head groups."
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=(
                spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            ),
        )

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def is_ssm(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type:
        from atom.plugin.vllm.attention.metadata import AiterMhaMetadataBuilderForVllm

        return AiterMhaMetadataBuilderForVllm

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMHA

        return AttentionForVllmMHA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AiterMhaFlexibleBlockBackendForVllm(AiterMhaBackendForVllm):
    """Draft-only backend whose Triton path accepts the logical KV page size."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


class AiterMlaBackendForVllm(_VllmAttentionBackendCompat):
    """vLLM-facing dense MLA backend surface for ATOM attention layers."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
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

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_ssm(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type:
        from atom.plugin.vllm.attention.metadata import AiterMlaMetadataBuilderForVllm

        return AiterMlaMetadataBuilderForVllm

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmMLA

        return AttentionForVllmMLA

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AtomAiterMLAPrefillBackend(MLAPrefillBackend):
    """vLLM MLA prefill interface backed by ATOM's aiter path."""

    @staticmethod
    def get_name() -> str:
        return "ATOM_AITER_MLA_PREFILL"

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config,
        layer=None,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        self._layer = layer

    def clone(self):
        return self.__class__(
            num_heads=self.num_heads,
            scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            vllm_config=self.vllm_config,
            layer=self._layer,
        )

    def run_prefill_new_tokens(
        self,
        q,
        k,
        v,
        return_softmax_lse,
        out=None,
        output_scale=None,
    ):
        if self._layer is None:
            raise RuntimeError("ATOM MLA prefill backend is not bound to a layer.")
        if out is not None or output_scale is not None:
            raise NotImplementedError(
                "ATOM MLA prefill does not support fused quantized output."
            )
        return self._layer._run_prefill_new_tokens(
            self._prefill_metadata,
            q,
            k,
            v,
            return_softmax_lse,
        )

    def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
        if self._layer is None:
            raise RuntimeError("ATOM MLA prefill backend is not bound to a layer.")
        if out is not None:
            raise NotImplementedError(
                "ATOM MLA context prefill does not support an output buffer."
            )
        return self._layer._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,
            softmax_scale=self.scale,
            causal=False,
            return_softmax_lse=True,
        )


def build_vllm_mla_prefill_backend(layer, vllm_config):
    """Create the vLLM MLA prefill backend for an ATOM MLA layer."""
    return AtomAiterMLAPrefillBackend(
        layer=layer,
        num_heads=layer.num_heads,
        scale=layer.scale,
        kv_lora_rank=layer.kv_lora_rank,
        qk_nope_head_dim=layer.qk_nope_head_dim,
        qk_rope_head_dim=layer.qk_rope_head_dim,
        v_head_dim=layer.v_head_dim,
        vllm_config=vllm_config,
    )


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
    def get_builder_cls() -> type:
        from atom.plugin.vllm.attention.metadata import AiterMlaSparseMetadataBuilder

        return AiterMlaSparseMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.layer import AttentionForVllmSparseMLA

        return AttentionForVllmSparseMLA

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
    def get_builder_cls() -> type:
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


class MiniMaxM3SparseAttentionBackend(_VllmAttentionBackendCompat):
    """vLLM-facing sparse MHA backend surface for MiniMax-M3."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [SPARSE_BLOCK_SIZE]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == SPARSE_BLOCK_SIZE

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return SPARSE_BLOCK_SIZE

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """K and V as two head slots, matching 0.28's (num_blocks, 2, ...) cache."""
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=(
                spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            ),
        )

    @staticmethod
    def get_builder_cls() -> type:
        from atom.plugin.vllm.attention.metadata import (
            MinimaxM3SparseAttentionMetadataBuilder,
        )

        return MinimaxM3SparseAttentionMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def is_ssm(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls():
        from atom.plugin.vllm.attention.minimax_m3_attnetion import (
            MiniMaxM3SparseAttentionForVllm,
        )

        return MiniMaxM3SparseAttentionForVllm

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class SparseMHAIndexerBackend(AiterMlaBackendForVllm):
    """vLLM-facing key-only indexer backend surface for MiniMax-M3."""

    @staticmethod
    def get_name() -> str:
        return "MINIMAX_M3_SPARSE_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [SPARSE_BLOCK_SIZE]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return SPARSE_BLOCK_SIZE

    @staticmethod
    def get_builder_cls() -> type:
        from atom.plugin.vllm.attention.metadata import (
            MinimaxM3SparseAttentionMetadataBuilder,
        )

        return MinimaxM3SparseAttentionMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]


class GDNAttentionBackend(_VllmAttentionBackendCompat):
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_impl_cls() -> type:
        from atom.plugin.vllm.attention.layer_gdn import GatedDeltaNet

        return GatedDeltaNet
