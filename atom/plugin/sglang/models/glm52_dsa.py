"""Model-level GLM-5.2 DSA adaptation for SGLang plugin mode."""

from __future__ import annotations

from typing import Any

from atom.plugin.sglang.models.deepseek_mla import _align_qknorm_fusion_for_sglang
from atom.plugin.sglang.models.deepseek_mla_forward import (
    _patch_attention_projs_for_sglang_mxfp4,
)
from atom.plugin.sglang.models.glm52_dsa_attention import (
    SGLangATOMGLM52MLAAttention,
)


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

        if not isinstance(module.mla_attn, SGLangATOMGLM52MLAAttention):
            raise RuntimeError(
                "GLM-5.2 SGLang native DSA setup expected "
                "SGLangATOMGLM52MLAAttention. Ensure the GLM construction "
                "context is installed before model initialization."
            )

        if getattr(module, "is_v32", False):
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
