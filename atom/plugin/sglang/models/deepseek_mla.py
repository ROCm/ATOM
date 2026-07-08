# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Model-level DeepSeek MLA patching for SGLang plugin mode.

This module owns the install-time hooks that adapt DeepSeek MLA models to
SGLang plugin mode. The heavy DeepSeek-specific runtime helpers live in
`atom.plugin.sglang.models.deepseek_mla_forward`.
"""

from __future__ import annotations

from types import MethodType
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

    last_sparse_indexer = None
    for module in model.modules():
        if isinstance(module, DeepseekV2MLAAttention):
            _patch_mla_attention_for_sglang(module, config, kv_cache_dtype)
            if getattr(module, "use_nsa", False):
                indexer = getattr(module, "indexer", None)
                if indexer is not None and not getattr(module, "skip_topk", False):
                    last_sparse_indexer = indexer
                    module._atom_sglang_topk_source_indexer = indexer
                else:
                    module._atom_sglang_topk_source_indexer = last_sparse_indexer


def _patch_mla_attention_for_sglang(
    attn: "DeepseekV2MLAAttention",
    config: Any,
    kv_cache_dtype: str = "bf16",
) -> None:
    """Patch one DeepSeek MLA layer for SGLang plugin mode."""
    _align_qknorm_fusion_for_sglang(attn)
    init_sgl_attrs(attn, config, kv_cache_dtype)
    _patch_attention_projs_for_sglang_mxfp4(attn)
    _patch_indexer_for_sglang_sparse_mla(attn)
    if not isinstance(attn.mla_attn, SGLangDeepseekMLAAttention):
        attn.mla_attn = SGLangDeepseekMLAAttention(attn, attn.mla_attn)
    attn.process_weights_after_loading = lambda: process_mla_kv_b_proj_after_loading(
        attn
    )


def _patch_indexer_for_sglang_sparse_mla(attn: "DeepseekV2MLAAttention") -> None:
    """Adapt DeepSeek-V3.2 sparse indexer buffers for SGLang plugin mode."""
    indexer = getattr(attn, "indexer", None)
    if indexer is None or getattr(indexer, "_atom_sglang_topk_buffer_patched", False):
        return

    import torch
    import atom.plugin.sglang.attention_backend.sparse_mla_indexer  # noqa: F401

    original_forward = indexer.forward
    indexer.use_qk_rope_cache_fusion = False
    indexer.sparse_attn_indexer_impl = (
        torch.ops.aiter.sparse_attn_indexer_sglang_plugin_mode
    )

    def _forward_with_topk_buffer(self, hidden_states, *args, **kwargs):
        num_tokens = int(hidden_states.shape[0])
        topk_tokens = int(self.topk_tokens)
        buffer = getattr(self, "topk_indices_buffer", None)
        needs_new_buffer = (
            buffer is None
            or buffer.dim() != 2
            or buffer.device != hidden_states.device
            or buffer.shape[0] < num_tokens
            or buffer.shape[1] < topk_tokens
        )
        if needs_new_buffer:
            buffer = torch.empty(
                num_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            self.topk_indices_buffer = buffer

        self.sparse_kv_indices_buffer = self.topk_indices_buffer
        return original_forward(hidden_states, *args, **kwargs)

    indexer.forward = MethodType(_forward_with_topk_buffer, indexer)
    indexer._atom_sglang_topk_buffer_patched = True


def _align_qknorm_fusion_for_sglang(attn: "DeepseekV2MLAAttention") -> None:
    """Keep non-quant q/k norm fusion on the BF16 path in SGLang plugin mode."""
    if getattr(attn, "fuse_qknorm", False) and not getattr(
        attn, "fuse_qknorm_quant", False
    ):
        import torch

        attn.quant_dtype = torch.bfloat16
        attn.qknorm_quant_type = None
