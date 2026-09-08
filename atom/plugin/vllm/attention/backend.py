from typing import ClassVar

import torch
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.kv_cache_layout import KVCacheLayout

from atom.model_ops.minimax_m3.sparse_attn import SPARSE_BLOCK_SIZE


def _indexes_kv_by_block_stride_for_backend(backend_cls) -> bool:
    try:
        kv_cache_stride_order = backend_cls.get_kv_cache_stride_order(
            include_num_layers_dimension=False
        )
        layered_kv_cache_stride_order = backend_cls.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
    except (AttributeError, NotImplementedError):
        return False

    if len(layered_kv_cache_stride_order) != len(kv_cache_stride_order) + 1:
        return False

    return layered_kv_cache_stride_order[0] != 0


class _VllmAttentionBackendCompat:
    """Compatibility surface for duck-typed ATOM attention backends."""

    @classmethod
    def customize_spec(cls, spec):
        """Keep vLLM 0.28's post-hoc KV spec unchanged."""
        return spec

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        """ATOM metadata builders plan from exact CPU query boundaries."""
        return False

    @classmethod
    def supported_kv_cache_layouts(cls):
        """Layouts this backend's kernels accept, most preferred first.

        vLLM 0.28 asks every backend for this and ATOM's backends are
        duck-typed, so they never inherit ``AttentionBackend``'s default.
        They already state layout through the older
        ``get_required_kv_cache_layout`` hook that 0.28 stopped calling, so
        honour a subclass that pins one and otherwise express no preference,
        exactly as ``AttentionBackend.supported_kv_cache_layouts`` does.
        """
        required = getattr(cls, "get_required_kv_cache_layout", None)
        layout = required() if required is not None else None
        if layout is None:
            return None
        if isinstance(layout, str):
            layout = KVCacheLayout[layout]
        return (layout,)


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

    @classmethod
    def is_ssm(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_required_kv_cache_layout():
        return None

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return _indexes_kv_by_block_stride_for_backend(cls)

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
    """MHA surface whose Triton path accepts the logical KV page size.

    The strict parent pins the kernel page at 16 to keep fp8 hybrid models off
    the page-16 asm kernels when vLLM's manager page is larger. That guard is
    about the asm path: at any block != 16 ``AttentionForVllmMHA`` sets
    ``use_triton_attn`` and runs the block-size-agnostic Triton insert/decode
    instead, which is safe at the logical page. Used by the Eagle3 draft (which
    shares the target's page so it can join one uniform-type group) and by M3's
    dense layers (whose group is pinned to the sparse backend's page 128).
    """

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


class AiterMhaM3DenseBackendForVllm(AiterMhaFlexibleBlockBackendForVllm):
    """MiniMax-M3 dense layers under ATOM_M3_DENSE_ATTN_BACKEND=gluon.

    Separate from the shared MHA surfaces so the layout request below reaches
    only M3's 3 dense layers: ``resolve_kv_cache_layout`` intersects what every
    backend publishes, so a preference declared on a shared class would follow
    every MHA model into the negotiation.
    """

    @classmethod
    def supported_kv_cache_layouts(cls):
        """Publish the layouts whose K/V slot axis stays outside N.

        ``num_head_slots=2`` only separates K and V if the layout does not put
        N between H and C -- ``LBNHC``/``BLNHC`` would re-interleave the sides
        per token, which no later reinterpretation can undo. Of the layouts
        that keep them apart, only ``LBHNC`` is block-compact, which vLLM
        requires once specs disagree on HNC (M3's key-only indexer spec makes
        them disagree). ``LHBNC`` trails it for a vLLM carrying the
        single-uniform-group exemption; a single-element tuple would leave the
        candidate list empty and abort startup.
        """
        from atom.utils import envs

        if envs.ATOM_M3_DENSE_ATTN_BACKEND != "gluon":
            return None
        return (KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)


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

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return _indexes_kv_by_block_stride_for_backend(cls)

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

    @classmethod
    def supported_kv_cache_layouts(cls):
        """Publish the K/V-separated layouts the AITER sparse-PA path accepts.

        The base shim wraps ``get_required_kv_cache_layout`` in a SINGLE-element
        tuple, which is fatal for M3: vLLM's mixed-HNC narrowing
        (utils.py resolve_kv_cache_layout) keeps only ``is_block_compact``
        candidates once the sparse layer (num_head_slots=2, page 65536 B) and
        the key-only indexer (MLAAttentionSpec, page 32768 B) disagree on HNC.
        ``LHBNC = (0,2,1,3,4)`` is NOT block-compact (set((0,2)) != {0,1}) so a
        single-LHBNC tuple leaves an empty candidate list -> raise. ``LBHNC =
        (0,1,2,3,4)`` IS block-compact and still keeps the K/V slot axis outside
        N, so it survives both the resolve narrowing and validate_kv_cache_layout
        while giving the page-16 shuffle two separable K/V regions. Publish LBHNC
        first, LHBNC as a fallback for a vLLM carrying the single-uniform-group
        exemption. Mirrors
        vllm/models/minimax_m3/common/sparse_attention.py supported_kv_cache_layouts.
        """
        from vllm import envs

        if not bool(getattr(envs, "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", False)):
            return None
        return (KVCacheLayout.LBHNC, KVCacheLayout.LHBNC)

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [SPARSE_BLOCK_SIZE]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == SPARSE_BLOCK_SIZE

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
        return SPARSE_BLOCK_SIZE

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

    @staticmethod
    def get_required_kv_cache_layout():
        # Superseded for layout PUBLICATION by supported_kv_cache_layouts above
        # (vLLM 0.28 stopped calling this hook); kept for ATOM-internal readers.
        # AITER sparse-PA (fp8 gluon paged-attention) needs the K/V slot axis
        # (num_head_slots=2) separable from the content dim. LBHNC =
        # [L, B, H, N, C] is block-compact (survives mixed-HNC narrowing that
        # M3's indexer triggers) AND keeps K/V outside N; the page-16 shuffle
        # reinterprets each contiguous sparse block as 2*pages_in_side page-16s.
        # Gated on VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT so the working plain-4-D
        # Triton path stays the default when the env is off.
        from vllm import envs

        if bool(getattr(envs, "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", False)):
            return KVCacheLayout.LBHNC
        return None

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size != SPARSE_BLOCK_SIZE:
            raise ValueError(
                f"MiniMax-M3 sparse block size must be {SPARSE_BLOCK_SIZE}."
            )
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        # Keep the logical block dimension first so vLLM does not normalize this
        # cache together with the block-first index cache. Physically place K/V
        # first so each cache remains contiguous for the page-16 ASM kernels.
        return (1, 0, 2, 3, 4)

    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return _indexes_kv_by_block_stride_for_backend(cls)

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

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            raise NotImplementedError
        return (0, 1, 2)


class GDNAttentionBackend(_VllmAttentionBackendCompat):
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_impl_cls() -> type:
        from atom.plugin.vllm.attention.layer_gdn import GatedDeltaNet

        return GatedDeltaNet
