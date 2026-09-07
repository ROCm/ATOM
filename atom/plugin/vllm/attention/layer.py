from typing import Optional

from atom.config import get_current_atom_config
from atom.model_ops.attention_mla import MLAModules
from atom.plugin.vllm.attention.layer_mha import AttentionForVllmMHA
from atom.plugin.vllm.attention.layer_mla import (
    AttentionForVllmMLA,
    AttentionForVllmSparseMLA,
)
from atom.plugin.vllm.attention import ops as _atom_vllm_attention_ops  # noqa: F401

_MINIMAX_M3_MODEL_TYPES = {"minimax_m3", "minimax_m3_text", "minimax_m3_vl"}


def _is_minimax_m3_model(atom_config) -> bool:
    hf_config = getattr(atom_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", "")
    text_config = getattr(hf_config, "text_config", None)
    text_model_type = getattr(text_config, "model_type", "")
    return (
        model_type in _MINIMAX_M3_MODEL_TYPES
        or text_model_type in _MINIMAX_M3_MODEL_TYPES
    )


def _minimax_m3_attention_cls_for_vllm(atom_config, kwargs):
    if not _is_minimax_m3_model(atom_config):
        return None
    impl_cls = kwargs.get("impl_cls")
    if impl_cls is not None:
        from atom.model_ops.attention_mha import (
            SparseMHAPagedAttentionImpl as AtomSparseMHAPagedAttentionImpl,
        )

        if impl_cls is AtomSparseMHAPagedAttentionImpl:
            from atom.plugin.vllm.attention.minimax_m3_attnetion import (
                MiniMaxM3SparseAttentionForVllm,
            )

            return MiniMaxM3SparseAttentionForVllm
        return None

    if (
        kwargs.get("rotary_emb") is not None
        and kwargs.get("q_norm") is not None
        and kwargs.get("k_norm") is not None
    ):
        # M3's first 3 layers run FULL (non-sparse) attention. By default they go
        # through `MiniMaxM3DenseAttentionForVllm`, whose read is vLLM's Triton
        # `unified_attention` custom op (`kernel_unified_attention`) -- a per-kernel
        # breakdown shows it is the dense-layer hotspot (~93% of prefill / ~95% of
        # decode dense-layer time; the generic kernel upcasts fp8 KV to bf16).
        #
        # ATOM_M3_DENSE_ATTN_BACKEND=aiter instead routes the dense layers through
        # ATOM's own MHA path (`AttentionForVllmMHA`): prefill via
        # `aiter.flash_attn_varlen_func`, decode via `run_pa_decode_gluon` (native
        # fp8 MFMA paged-attention), on a 5-D shuffle KV cache. For M3 dense that
        # path selects the *flexible* backend (`AiterMhaFlexibleBlockBackendForVllm`,
        # see `_mha_backend_for_layer`), which advertises MultipleOf(16) so the
        # dense group coexists with the sparse group's mandatory block_size=128
        # (negotiation settles on the logical 128 page) instead of crashing
        # block-size negotiation the way the strict [16] backend -- or vLLM's own
        # AiterFA backend (kernel pages [16, 32] only) -- does. M3 dense uses
        # GemmaRMSNorm q/k-norm, which `AttentionForVllmMHA.rope_cache` already
        # handles via `triton_fused_norm_rope_cache` (norm + RoPE + fp8 shuffle
        # cache write), and at block != 16 always takes the block-agnostic Triton
        # read, so running at the 128 page is correct.
        import os

        if os.environ.get("ATOM_M3_DENSE_ATTN_BACKEND", "triton").lower() == "aiter":
            return AttentionForVllmMHA

        from atom.plugin.vllm.attention.minimax_m3_attnetion import (
            MiniMaxM3DenseAttentionForVllm,
        )

        return MiniMaxM3DenseAttentionForVllm
    return None


class AttentionForVllm:
    """Factory for ATOM-owned attention layers running under vLLM."""

    def __new__(
        cls,
        *args,
        use_mla: bool = False,
        mla_modules: Optional[MLAModules] = None,
        **kwargs,
    ):
        atom_config = get_current_atom_config()
        if atom_config is None:
            raise RuntimeError("atom_config is required for vLLM plugin attention")

        if use_mla:
            is_sparse_mla = mla_modules is not None and (
                mla_modules.is_sparse or mla_modules.indexer is not None
            )
            if is_sparse_mla:
                return AttentionForVllmSparseMLA(
                    *args, mla_modules=mla_modules, **kwargs
                )
            return AttentionForVllmMLA(*args, mla_modules=mla_modules, **kwargs)
        minimax_m3_attention_cls = _minimax_m3_attention_cls_for_vllm(
            atom_config, kwargs
        )
        if minimax_m3_attention_cls is not None:
            return minimax_m3_attention_cls(*args, **kwargs)
        kwargs.pop("impl_cls", None)
        return AttentionForVllmMHA(*args, **kwargs)
