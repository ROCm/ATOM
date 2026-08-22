"""Model-level GLM-5.2 DSA adaptation for SGLang plugin mode."""

from __future__ import annotations

import logging
import os
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

logger = logging.getLogger(__name__)


_MI308_GFX = "gfx942"


def _patch_aiter_flat_fmoe_for_glm52_mi308() -> None:
    """Keep GLM-5.2 from selecting gfx950-only flat FMOE kernels on MI308."""
    try:
        from aiter import fused_moe
    except Exception:
        logger.debug(
            "Failed to import aiter.fused_moe for GLM-5.2 patch", exc_info=True
        )
        return

    if getattr(fused_moe, "_atom_glm52_mi308_flat_fmoe_guarded", False):
        return

    original_get_2stage_cfgs = getattr(fused_moe, "get_2stage_cfgs", None)
    get_gfx = getattr(fused_moe, "get_gfx", None)
    if original_get_2stage_cfgs is None or get_gfx is None:
        logger.warning(
            "AITER fused_moe lacks get_2stage_cfgs/get_gfx; skip GLM-5.2 patch"
        )
        return

    try:
        gfx = get_gfx()
    except Exception:
        logger.debug("Failed to query gfx target for GLM-5.2 patch", exc_info=True)
        return

    if gfx != _MI308_GFX:
        return

    def _get_2stage_cfgs_without_unsupported_flat(*args, **kwargs):
        if get_gfx() != _MI308_GFX:
            return original_get_2stage_cfgs(*args, **kwargs)

        metadata = original_get_2stage_cfgs(*args, **kwargs)
        if not getattr(metadata, "flat", False):
            return metadata

        # MI308/gfx942 cannot run the flat FMOE tuned path selected by AITER for
        # GLM-5.2.  Re-query with tune-config bypass only for this GLM setup.
        previous_bypass = os.environ.get("AITER_BYPASS_TUNE_CONFIG")
        os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
        try:
            cache_clear = getattr(original_get_2stage_cfgs, "cache_clear", None)
            if cache_clear is not None:
                cache_clear()
            fallback_metadata = original_get_2stage_cfgs(*args, **kwargs)
        finally:
            if previous_bypass is None:
                os.environ.pop("AITER_BYPASS_TUNE_CONFIG", None)
            else:
                os.environ["AITER_BYPASS_TUNE_CONFIG"] = previous_bypass

        if getattr(fallback_metadata, "flat", False):
            logger.warning(
                "AITER fallback metadata is still flat on %s; forcing non-flat routing",
                gfx,
            )
            fallback_metadata.flat = False
        else:
            logger.warning(
                "Bypassed AITER flat FMOE tuned config for GLM-5.2 on %s",
                gfx,
            )
        return fallback_metadata

    fused_moe.get_2stage_cfgs = _get_2stage_cfgs_without_unsupported_flat
    fused_moe._atom_glm52_mi308_flat_fmoe_guarded = True


def setup_glm52_dsa_for_sglang(model: Any) -> None:
    """Patch GLM-5.2 for native ATOM sparse MLA under SGLang.

    This deliberately does not install ``SGLangDeepseekMLAAttention``.  GLM-5.2
    should keep ATOM's native ``MLAAttention`` frontend so full-index layers run
    the ATOM indexer into a shared physical-index buffer and shared-index layers
    reuse that buffer.
    """
    _patch_aiter_flat_fmoe_for_glm52_mi308()

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
