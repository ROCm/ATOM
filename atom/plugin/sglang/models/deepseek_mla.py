# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Model-level DeepSeek MLA patching for SGLang plugin mode.

This module owns the install-time hooks that adapt DeepSeek MLA models to
SGLang plugin mode. The heavy DeepSeek-specific runtime helpers live in
`atom.plugin.sglang.models.deepseek_mla_forward`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atom.plugin.sglang.models.deepseek_mla_attention import (
    SGLangDeepseekMLAAttention,
)
from atom.plugin.sglang.models.deepseek_mla_forward import (
    _patch_attention_projs_for_sglang_mxfp4,
    init_sgl_attrs,
    process_mla_kv_b_proj_after_loading,
)

if TYPE_CHECKING:
    from atom.models.deepseek_v2 import DeepseekV2MLAAttention


def setup_deepseek_for_sglang(model) -> None:
    """Patch a DeepSeek V2/V3 model for SGLang plugin mode."""
    config = model.config

    # Store atom_config for the OOT wrapper before install-time hooks run.
    if not hasattr(model, "atom_config"):
        from atom.config import get_current_atom_config

        model.atom_config = get_current_atom_config()

    kv_cache_dtype = model.atom_config.kv_cache_dtype

    # Initialise SGLang's MLA TP context before patching per-layer forwards.
    from sglang.srt.configs.model_config import is_deepseek_nsa
    from sglang.srt.layers.communicator import get_attn_tp_context

    get_attn_tp_context().init_context(config.q_lora_rank, is_deepseek_nsa(config))

    from atom.models.deepseek_v2 import DeepseekV2MLAAttention

    for module in model.modules():
        if isinstance(module, DeepseekV2MLAAttention):
            _patch_mla_attention_for_sglang(module, config, kv_cache_dtype)


def _patch_mla_attention_for_sglang(
    attn: "DeepseekV2MLAAttention",
    config: Any,
    kv_cache_dtype: str = "bf16",
) -> None:
    """Patch one DeepSeek MLA layer for SGLang plugin mode."""
    _align_qknorm_fusion_for_sglang(attn)
    init_sgl_attrs(attn, config, kv_cache_dtype)
    _patch_attention_projs_for_sglang_mxfp4(attn)
    if not isinstance(attn.mla_attn, SGLangDeepseekMLAAttention):
        attn.mla_attn = SGLangDeepseekMLAAttention(attn, attn.mla_attn)
    attn.process_weights_after_loading = lambda: process_mla_kv_b_proj_after_loading(
        attn
    )


def _align_qknorm_fusion_for_sglang(attn: "DeepseekV2MLAAttention") -> None:
    """Keep non-quant q/k norm fusion on the BF16 path in SGLang plugin mode."""
    if getattr(attn, "fuse_qknorm", False) and not getattr(
        attn, "fuse_qknorm_quant", False
    ):
        import torch

        attn.quant_dtype = torch.bfloat16
        attn.qknorm_quant_type = None
