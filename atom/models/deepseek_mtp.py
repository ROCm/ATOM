# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers import DeepseekV2Config, DeepseekV3Config
from atom.config import QuantizationConfig, Config

from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from atom.model_ops.moe import FusedMoE

from atom.models.utils import IntermediateTensors

from .deepseek_v2 import DeepseekV2DecoderLayer
from .utils import maybe_prefix
from atom.model_ops.topK import (
    is_rocm_aiter_fusion_shared_expert_enabled,
)
from atom.utils.decorators import support_torch_compile
from aiter.dist.communication_op import tensor_model_parallel_all_reduce


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, atom_config: Config, prefix: str, layer_idx: int) -> None:
        super().__init__()

        config = atom_config.hf_config
        self.config = config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=atom_config.quant_config
        )
        self.mtp_block = DeepseekV2DecoderLayer(
            prefix=prefix,
            config=self.config,
            cache_config=atom_config.kv_cache_dtype,
            quant_config=atom_config.quant_config,
            layer_num=layer_idx,
            is_mtp_block=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        masked_inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(masked_inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        # mtp always has input_layernorm fused_allreduce off
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = atom_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    atom_config, f"{prefix}.layers.{idx}", layer_idx=idx
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = mtp_layer.shared_head.head(mtp_layer.shared_head(hidden_states))
        return logits


class DeepSeekMTP(nn.Module):

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.config = atom_config.hf_config

        if hasattr(self.config, "q_lora_rank") and self.config.q_lora_rank is not None:
            self.packed_modules_mapping = {
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }
        else:
            self.packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }

        self.model = DeepSeekMultiTokenPredictor(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "model")
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
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


def get_spec_layer_idx_from_weight_name(
    config: Union[DeepseekV2Config, DeepseekV3Config], weight_name: str
) -> Optional[int]:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None


def rewrite_spec_layer_name(spec_layer: int, name: str) -> str:
    """
    Rewrite the weight name to match the format of the original model.
    Add .mtp_block for modules in transformer layer block for spec layer
    and rename shared layer weights to be top level.
    """
    spec_layer_weight_names = [
        "embed_tokens",
        "enorm",
        "hnorm",
        "eh_proj",
        "shared_head",
    ]
    shared_weight_names = ["embed_tokens"]
    spec_layer_weight = False
    shared_weight = False
    for weight_name in spec_layer_weight_names:
        if weight_name in name:
            spec_layer_weight = True
            if weight_name in shared_weight_names:
                shared_weight = True
            break
    if not spec_layer_weight:
        # treat rest weights as weights for transformer layer block
        name = name.replace(
            f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
        )
    elif shared_weight:
        # treat shared weights as top level weights
        name = name.replace(f"model.layers.{spec_layer}.", "model.")
    return name
