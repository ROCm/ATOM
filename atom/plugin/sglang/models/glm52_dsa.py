"""Model-level GLM-5.2 DSA adaptation for SGLang plugin mode."""

from __future__ import annotations

from typing import Any

from atom.plugin.sglang.models.deepseek_mla import (
    _align_qknorm_fusion_for_sglang,
    _patch_indexer_layernorm_for_sglang,
)
from atom.plugin.sglang.models.deepseek_mla_forward import (
    _patch_attention_projs_for_sglang_mxfp4,
)
from atom.plugin.sglang.models.glm52_dsa_attention import (
    SGLangATOMGLM52MLAAttention,
)


def short_verify_skips_indexer(
    metadata: Any,
    *,
    is_prefill: bool,
    dcp_world_size: int,
    topk_tokens: int,
) -> bool:
    """Skip top-k only for a multi-token verify whose context is inside top-k.

    GLM-FP8 decode is one query token. Skipping the indexer there leaves sparse
    MLA with an empty index buffer, and GSM8K falls to chance.
    """

    if is_prefill or int(dcp_world_size) != 1 or int(topk_tokens) <= 0:
        return False
    if metadata is None:
        return False
    max_seqlen_q = int(getattr(metadata, "max_seqlen_q", 1) or 1)
    max_seqlen_k = int(getattr(metadata, "max_seqlen_k", 0) or 0)
    return max_seqlen_q > 1 and max_seqlen_k <= int(topk_tokens)


def _install_short_verify_indexer_skip(module: Any) -> None:
    """Skip paged MQA top-k when the verify context is already inside top-k.

    The native op writes that score on the decode path. Below ``index_topk``
    every token is selected, so attention reads the dense KV table instead.
    Installed on the plugin indexer object; ``deepseek_v2.py`` stays unchanged.
    """

    indexer = getattr(module, "indexer", None)
    if indexer is None or getattr(indexer, "_atom_glm52_short_verify_skip", False):
        return
    if getattr(module, "skip_topk", False):
        return

    original = indexer.sparse_attn_indexer_impl

    def _sparse_attn_indexer(*args: Any, **kwargs: Any):
        from atom.distributed.dcp_utils import get_dcp_world_size
        from atom.utils.forward_context import get_forward_context

        context = get_forward_context()
        metadata = getattr(context, "attn_metadata", None)
        is_prefill = bool(
            getattr(getattr(context, "context", None), "is_prefill", False)
        )
        if short_verify_skips_indexer(
            metadata,
            is_prefill=is_prefill,
            dcp_world_size=get_dcp_world_size(),
            topk_tokens=int(getattr(indexer, "topk_tokens", 0) or 0),
        ):
            return None
        return original(*args, **kwargs)

    indexer.sparse_attn_indexer_impl = _sparse_attn_indexer
    indexer._atom_glm52_short_verify_skip = True


def setup_glm52_dsa_for_sglang(model: Any) -> None:
    """Patch GLM-5.2 for native ATOM sparse MLA under SGLang.

    This deliberately does not install ``SGLangDeepseekMLAAttention``.  GLM-5.2
    should keep ATOM's native ``MLAAttention`` frontend so full-index layers run
    the ATOM indexer into a shared physical-index buffer and shared-index layers
    reuse that buffer.
    """

    if not hasattr(model, "atom_config"):
        from atom.config import get_current_atom_config

        model.atom_config = get_current_atom_config()

    from atom.models.deepseek_v2 import DeepseekV2MLAAttention

    try:
        from sglang.srt.configs.model_config import is_deepseek_dsa
    except ImportError:
        from sglang.srt.configs.model_config import is_deepseek_nsa as is_deepseek_dsa

    from sglang.srt.layers.communicator import get_attn_tp_context

    config = model.config
    get_attn_tp_context().init_context(config.q_lora_rank, is_deepseek_dsa(config))

    last_full_index_seen = False
    for module in model.modules():
        if not isinstance(module, DeepseekV2MLAAttention):
            continue

        _align_qknorm_fusion_for_sglang(module)
        _patch_attention_projs_for_sglang_mxfp4(module)
        _patch_indexer_layernorm_for_sglang(module)

        if not isinstance(module.mla_attn, SGLangATOMGLM52MLAAttention):
            raise TypeError(
                "GLM-5.2 SGLang native DSA setup expected "
                "SGLangATOMGLM52MLAAttention. Ensure the GLM construction "
                "context is installed before model initialization."
            )

        if getattr(module, "is_v32", False):
            _install_short_verify_indexer_skip(module)
            owns_active_indexer = getattr(
                module, "indexer", None
            ) is not None and not getattr(module, "skip_topk", False)
            if owns_active_indexer:
                last_full_index_seen = True
            elif not last_full_index_seen:
                raise RuntimeError(
                    "GLM-5.2 IndexShare cannot start with a shared-index layer; "
                    f"layer={getattr(module, 'prefix', '<unknown>')!r}"
                )

    model._atom_sglang_uses_glm52_native_dsa = True
