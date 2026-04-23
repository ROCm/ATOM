# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.5 MTP model."""

import re

import torch
import torch.nn as nn
from aiter.dist.parallel_state import get_tp_group
from atom.config import Config
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import ColumnParallelLinear
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
from atom.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    get_qwen3_5_text_config,
)
from atom.models.utils import IntermediateTensors
from .utils import maybe_prefix


# @support_torch_compile
class Qwen3_5MultiTokenPredictor(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()

        quant_config = atom_config.quant_config

        config = get_qwen3_5_text_config(atom_config)

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                atom_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
                layer_num=idx,
            )
            for idx in range(
                self.mtp_start_layer_idx,
                self.mtp_start_layer_idx + self.num_mtp_layers,
            )
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        hidden_states = get_tp_group().all_gather(hidden_states)
        residual = None

        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# @support_torch_compile
class Qwen3_5MTP(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }
    weights_mapping = {"mtp.": "model."}

    def remap_mtp_weight_name(self, name: str) -> str | None:
        """Filter MTP weights; remap (mtp.* → model.*) is via weights_mapping."""
        shared_weight_names = ["embed_tokens", "lm_head"]

        # MTP-specific weights
        if name.startswith("mtp."):
            return name

        # Shared weights loaded into both target and draft
        if any(key in name for key in shared_weight_names):
            return name

        # Skip target model weights
        return None

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = get_qwen3_5_text_config(atom_config)
        if atom_config.enable_prefix_caching:
            raise ValueError("Qwen3_5MTP currently does not support prefix caching")
        self.config = config

        # Remap exclude entries: checkpoint uses 0-based MTP layer indices
        # (e.g. "mtp.layers.0.*") but the model constructs layers with absolute
        # indices starting at num_hidden_layers (e.g. "mtp.layers.60.*").
        # Reindex so _is_excluded matches the construction prefix.
        mtp_start = config.num_hidden_layers
        num_mtp = getattr(config, "mtp_num_hidden_layers", 1)
        if atom_config.quant_config is not None and mtp_start > 0:
            pat = re.compile(r"^mtp\.layers\.(\d+)\.")
            new_excludes = []
            for entry in atom_config.quant_config.exclude_layers:
                m = pat.match(entry)
                if m:
                    old_idx = int(m.group(1))
                    entry = pat.sub(f"mtp.layers.{mtp_start + old_idx}.", entry)
                new_excludes.append(entry)
            # Add layer-level prefixes so that _is_excluded matches module
            # prefixes like "mtp.layers.60.mlp.experts" (not just leaf names).
            for i in range(num_mtp):
                new_excludes.append(f"mtp.layers.{mtp_start + i}")
            atom_config.quant_config.exclude_layers = list(dict.fromkeys(new_excludes))

        self.model = Qwen3_5MultiTokenPredictor(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "mtp")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:  # noqa: D401
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if is_rocm_aiter_fusion_shared_expert_enabled()
                else 0
            ),
        )
