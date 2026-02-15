# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next MTP model."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from atom.config import Config, QuantizationConfig
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.moe import FusedMoE
from aiter.dist.parallel_state import get_tp_group
from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
from atom.models.utils import IntermediateTensors
from atom.models.qwen3_next import Qwen3NextDecoderLayer, Qwen3NextRMSNorm
from atom.model_ops.linear import ColumnParallelLinear
from atom.model_config.qwen3_next import Qwen3NextConfig
from .utils import maybe_prefix

KVCache = tuple[torch.Tensor, torch.Tensor]


# @support_torch_compile
class Qwen3NextMultiTokenPredictor(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()

        quant_config = atom_config.quant_config

        config: Qwen3NextConfig = atom_config.hf_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                atom_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
                layer_num=idx,
            )
            for idx in range(
                self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers
            )
        )

        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(
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
class Qwen3NextMTP(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, atom_config: Config, prefix: str = ""):
        config = atom_config.hf_config
        self.vllm_config = atom_config
        assert (
            not atom_config.enable_prefix_caching
        ), "Qwen3NextMTP currently does not support prefix caching"

        self.quant_config = atom_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3NextMultiTokenPredictor(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "mtp")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # self.logits_processor = LogitsProcessor(config.vocab_size)
        # self.make_empty_intermediate_tensors = (
        #     self.model.make_empty_intermediate_tensors
        # )

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
        # return self.logits_processor(self.lm_head, hidden_states)
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    # def remap_weight_names(weights):
    #     shared_weight_names = ["embed_tokens", "lm_head"]

    #     for name, weight in weights:
    #         if name.startswith("mtp."):
    #             name = name.replace("mtp.", "model.")
    #         elif not any(key in name for key in shared_weight_names):
    #             continue
    #         yield name, weight


def remap_mtp_weight_name(name: str) -> str | None:
    """
    Remap MTP weight names to match the model structure.
    Returns None if the weight should be skipped.

    MTP weights are stored with 'mtp.' prefix in checkpoints but loaded
    into 'model.' in the actual model structure.
    Shared weights (embed_tokens, lm_head) are loaded into both base model and MTP.
    """
    shared_weight_names = ["embed_tokens", "lm_head"]

    # Remap mtp.* -> model.*
    # print("remapping weight name: ", name, flush=True)
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.")

    # Allow shared weights to be loaded
    if any(key in name for key in shared_weight_names):
        return name

    # Skip all other weights (they belong to base model, not MTP)
    return None
