# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Gemma 4 (text-only) model for ATOM.
#
# Architecture highlights
# -----------------------
# * Hybrid attention: alternating ``sliding_attention`` (local, window=1024)
#   and ``full_attention`` (global) layers.
# * Sliding layers  : 32 Q-heads x 256 d_head, 16 KV-heads, RoPE theta=10000
# * Full layers     : 32 Q-heads x 512 d_head,  4 KV-heads, partial RoPE
#                     (rotary_dim = 128, θ=1 000 000)
# * Dense MLP with gelu_pytorch_tanh gate (GLU-style)
# * GemmaRMSNorm (weight = 1 + w, unlike standard RMSNorm)
# * Per-head QK-normalisation (GemmaRMSNorm applied to Q and K heads)
# * Logit soft-capping: tanh(x / cap) * cap, cap=30

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from aiter.dist.parallel_state import get_tp_group
from aiter.rotary_embedding import get_rope
from atom.config import Config
from atom.model_config.gemma4 import Gemma4TextConfig
from atom.model_loader.loader import load_model_in_plugin_mode
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile

# ---------------------------------------------------------------------------
# Activation: GELU gate (gelu_pytorch_tanh variant)
# ---------------------------------------------------------------------------


class GeluAndMul(nn.Module):
    """GLU-style activation: GELU(gate) * up."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        return F.gelu(gate, approximate="tanh") * up


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Gemma4Attention(nn.Module):
    """Unified attention module for both ``sliding_attention`` and
    ``full_attention`` layer types.

    The two types differ in KV-head count, head dimension, RoPE config, and
    whether a sliding window is applied.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_type: str,
        layer_num: int,
        atom_config: Config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        tp_size = get_tp_group().world_size

        if layer_type == "sliding_attention":
            total_kv_heads = config.num_key_value_heads  # 16
            self.head_dim = config.head_dim  # 256
            rope_params = config.rope_parameters.get("sliding_attention", {})
            per_layer_sliding_window = config.sliding_window  # 1024
        else:
            total_kv_heads = config.num_global_key_value_heads  # 4
            self.head_dim = config.global_head_dim  # 512
            rope_params = config.rope_parameters.get("full_attention", {})
            per_layer_sliding_window = None  # full context

        total_num_heads = config.num_attention_heads  # 32
        self.num_heads = total_num_heads // tp_size
        # Replicate KV heads when tp_size > total_kv_heads (GQA replication)
        self.num_kv_heads = max(1, total_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            total_num_heads,
            total_kv_heads,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rope_theta = rope_params.get("rope_theta", 10000.0)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            # Pass a simple "default" scaling dict so the backend does not try
            # to parse the nested Gemma4 rope_parameters structure.
            rope_scaling={"rope_type": "default", "rope_theta": rope_theta},
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=per_layer_sliding_window,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head QK normalisation (Gemma-style)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        o = self.attn(q, k, v, positions, **model_kwargs)
        output = self.o_proj(o)
        return output


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Gemma4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = GeluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        atom_config: Config,
        layer_num: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_type = config.layer_types[layer_num]
        self.self_attn = Gemma4Attention(
            config=config,
            layer_type=layer_type,
            layer_num=layer_num,
            atom_config=atom_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        # GemmaRMSNorm multiplies by (1 + weight) instead of weight
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **model_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, **model_kwargs
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
    }
)
class Gemma4Model(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        hf_config: Gemma4TextConfig = atom_config.hf_config

        self.embed_tokens = VocabParallelEmbedding(
            hf_config.vocab_size, hf_config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config=hf_config,
                    atom_config=atom_config,
                    layer_num=layer_num,
                    prefix=f"{prefix}.layers.{layer_num}",
                )
                for layer_num in range(hf_config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], hf_config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **model_kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------


class Gemma4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config: Gemma4TextConfig = config.hf_config
        self.model = Gemma4Model(
            atom_config=self.atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=self.hf_config.vocab_size,
            embedding_dim=self.hf_config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.hf_config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids, positions=positions, **model_kwargs
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        # Gemma 4 applies logit soft-capping
        cap = self.hf_config.final_logit_softcapping
        if cap is not None and cap > 0:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights_record = load_model_in_plugin_mode(
            model=self, config=self.atom_config, prefix="model."
        )
        return loaded_weights_record


# ---------------------------------------------------------------------------
# Multimodal text-only wrapper  (skips vision encoder weights)
# ---------------------------------------------------------------------------


class Gemma4ForConditionalGenerationTextOnly(nn.Module):
    """Text-only entry point for Gemma4ForConditionalGeneration checkpoints.

    The HF checkpoint contains both vision and language weights.  ATOM only
    needs the language model for text inference; vision weights are skipped via
    ``skip_weight_prefixes`` and ``weights_mapping`` remaps the
    ``language_model.*`` prefix used in HF checkpoints to the flat ``model.*``
    / ``lm_head.*`` layout expected by :class:`Gemma4ForCausalLM`.
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Map HF checkpoint prefixes → ATOM model prefixes.
    # HF Gemma4: language_model.model.* and language_model.lm_head.*
    weights_mapping = {
        "language_model.model.": "model.",
        "language_model.lm_head.": "lm_head.",
    }

    # Drop vision encoder and multimodal projector weights entirely.
    skip_weight_prefixes = ["vision_tower.", "multi_modal_projector."]

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = atom_config
        self.language_model = Gemma4ForCausalLM(
            config=atom_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights_record = load_model_in_plugin_mode(
            model=self, config=self.atom_config, prefix="model."
        )
        return loaded_weights_record
