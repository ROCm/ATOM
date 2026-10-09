# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GLM-5.3-Flash NextN support for ATOM speculative decoding."""

import copy
from typing import ClassVar

import torch
from torch import nn

from atom.config import Config
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding
from atom.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictorLayer,
)
from atom.models.deepseek_v2 import ENABLE_ALLREDUCE_RMSNORM_FUSION
from atom.models.glm5_next import (
    _ROPE_PAD,
    Glm5NextMLAAttention,
    Glm5NextMoE,
    _normalize_glm5_next_config,
)


def _add_mtp_quant_excludes(atom_config: Config) -> None:
    """Mirror layer-45 BF16 exclusions onto the runtime ``mtp_block`` path."""
    quant_config = atom_config.quant_config
    if quant_config is None:
        return
    layer_prefix = f"model.layers.{atom_config.hf_config.num_hidden_layers}."
    block_prefixes = (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "mlp",
    )
    for attr in ("exclude_layers", "online_exclude_layers"):
        excludes = getattr(quant_config, attr, None)
        if not excludes:
            continue
        additions = []
        for name in excludes:
            if not name.startswith(layer_prefix):
                continue
            suffix = name[len(layer_prefix) :]
            if suffix.startswith(block_prefixes):
                additions.append(f"{layer_prefix}mtp_block.{suffix}")
        excludes.extend(name for name in additions if name not in excludes)


def _prepare_mtp_config(atom_config: Config) -> Config:
    """Return an isolated config for the checkpoint's NextN layer."""
    mtp_config = copy.copy(atom_config)
    draft_config = getattr(
        getattr(atom_config, "speculative_config", None),
        "draft_model_hf_config",
        None,
    )
    if draft_config is not None:
        mtp_config.hf_config = copy.copy(draft_config)

    if atom_config.quant_config is not None:
        mtp_config.quant_config = copy.copy(atom_config.quant_config)
        for attr in ("exclude_layers", "online_exclude_layers"):
            excludes = getattr(mtp_config.quant_config, attr, None)
            if excludes is not None:
                setattr(mtp_config.quant_config, attr, list(excludes))

    _normalize_glm5_next_config(mtp_config.hf_config)
    _add_mtp_quant_excludes(mtp_config)
    return mtp_config


class Glm5NextMTPDecoderLayer(nn.Module):
    """Checkpoint layer 45, which is a regular residual MLA+MoE block.

    Unlike backbone layers, the NextN layer has no mHC parameters.  It still
    needs GLM's k-pool indexer and clamped SwiGLU experts, so using the generic
    DeepSeek decoder would silently change both attention selection and MoE
    numerics.
    """

    def __init__(
        self,
        atom_config: Config,
        prefix: str,
        layer_num: int,
    ) -> None:
        super().__init__()
        config = atom_config.hf_config
        rotary_emb = NoPositionalRotaryEmbedding(
            head_size=_ROPE_PAD,
            rotary_dim=_ROPE_PAD,
            max_position_embeddings=int(config.max_position_embeddings),
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        self.self_attn = Glm5NextMLAAttention(
            atom_config,
            layer_num,
            rotary_emb,
            prefix=f"{prefix}.self_attn",
            is_mtp=True,
        )
        self.mlp = Glm5NextMoE(
            config,
            atom_config.quant_config,
            prefix=f"{prefix}.mlp",
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Glm5NextMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    """DeepSeek-compatible predictor prologue with a GLM NextN block."""

    @staticmethod
    def build_mtp_block(
        atom_config: Config,
        prefix: str,
        layer_idx: int,
        alt_stream: torch.cuda.Stream | None,
    ) -> nn.Module:
        del alt_stream
        return Glm5NextMTPDecoderLayer(
            atom_config=atom_config,
            prefix=prefix,
            layer_num=layer_idx,
        )


class Glm5NextMTP(DeepSeekMTP):
    """Load checkpoint layer 45 through ATOM's existing NextN runtime."""

    predictor_layer_cls: ClassVar[type[DeepSeekMultiTokenPredictorLayer]] = (
        Glm5NextMultiTokenPredictorLayer
    )
    packed_modules_mapping_override: ClassVar[dict[str, tuple[str, int]]] = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    supports_indexer_projection_fusion: ClassVar[bool] = False
    weights_mapping: ClassVar[dict[str, str]] = {
        "index_kpool_compress_gate": "index_kpool_compress_gate.weight",
    }

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__(atom_config=_prepare_mtp_config(atom_config), prefix=prefix)

    def remap_mtp_weight_name(self, name: str) -> str | None:
        name = name.replace("model.language_model.", "model.")
        return super().remap_mtp_weight_name(name)
