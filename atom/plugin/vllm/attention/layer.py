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


def _use_minimax_m3_dense_attention_for_vllm(atom_config, kwargs) -> bool:
    if not _is_minimax_m3_model(atom_config):
        return False
    if kwargs.get("impl_cls") is not None:
        return False
    return (
        kwargs.get("rotary_emb") is not None
        and kwargs.get("q_norm") is not None
        and kwargs.get("k_norm") is not None
    )


def _use_minimax_m3_sparse_attention_for_vllm(atom_config, kwargs) -> bool:
    if not _is_minimax_m3_model(atom_config):
        return False
    impl_cls = kwargs.get("impl_cls")
    if impl_cls is None:
        return False
    from atom.model_ops.attention_mha import (
        SparseMHAPagedAttentionImpl as AtomSparseMHAPagedAttentionImpl,
    )

    return impl_cls is AtomSparseMHAPagedAttentionImpl


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
            if mla_modules is not None and mla_modules.indexer is not None:
                return AttentionForVllmSparseMLA(
                    *args, mla_modules=mla_modules, **kwargs
                )
            return AttentionForVllmMLA(*args, mla_modules=mla_modules, **kwargs)
        if _use_minimax_m3_sparse_attention_for_vllm(atom_config, kwargs):
            from atom.plugin.vllm.attention.minimax_m3_attnetion import (
                MiniMaxM3SparseAttentionForVllm,
            )

            return MiniMaxM3SparseAttentionForVllm(*args, **kwargs)
        if _use_minimax_m3_dense_attention_for_vllm(atom_config, kwargs):
            from atom.plugin.vllm.attention.minimax_m3_attnetion import (
                MiniMaxM3DenseAttentionForVllm,
            )

            return MiniMaxM3DenseAttentionForVllm(*args, **kwargs)
        return AttentionForVllmMHA.create(*args, **kwargs)
