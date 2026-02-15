# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gpt_oss.py
# Copyright 2023 The vLLM team.
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from aiter import ActivationType
from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size

# from vllm.model_executor.layers.logits_processor import LogitsProcessor
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding

# from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from atom.model_ops.moe import FusedMoE

# from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from atom.models.utils import (
    IntermediateTensors,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils import envs
from atom.utils.decorators import support_torch_compile
from torch import nn
from transformers import GptOssConfig

ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION


def cdiv(x, y):
    return (x + y - 1) // y


class OAIAttention(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_num
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        rope_params = config.rope_parameters
        rope_theta = rope_params["rope_theta"]

        if rope_params is None:
            raise ValueError(
                "GPT-OSS config is missing RoPE scaling parameters. Expected either "
                "`rope_scaling` (transformers < 5) or `rope_parameters` (transformers 5+)."
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            dtype=torch.bfloat16,
            rope_scaling=rope_params,
            is_neox_style=True,
        )

        tp_size = get_tensor_model_parallel_world_size()

        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads // tp_size, requires_grad=False)
        )

        self.q_size = self.num_attention_heads * self.head_dim // tp_size
        self.kv_size = self.num_key_value_heads * self.head_dim // tp_size
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_attention_heads,
            total_num_kv_heads=self.num_key_value_heads,
            quant_config=None,
            prefix=f"{prefix}.qkv_proj",
            bias=True,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.num_attention_heads * self.head_dim,
            output_size=self.hidden_size,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
            bias=True,
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )

        self.num_local_attention_heads = config.num_attention_heads // tp_size
        self.num_local_key_value_heads = config.num_key_value_heads // tp_size

        # Only apply sliding window to every other layer
        sliding_window = config.sliding_window if self.layer_idx % 2 == 0 else None
        self.attn = Attention(
            self.num_local_attention_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_local_key_value_heads,
            kv_cache_dtype=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
            layer_num=self.layer_idx,
            rotary_emb=self.rotary_emb,
        )

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, positions)
        output = self.o_proj(attn_output)
        return output


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        atom_config: Config,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()

        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.is_sequence_parallel = False
        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tp_size = get_tensor_model_parallel_world_size()
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        assert config.intermediate_size % self.world_size == 0
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            apply_router_weight_on_input=False,
            has_bias=True,
            activation=ActivationType.Swiglu,
            config=config,
        )
        # Detect MXFP4 MoE GEMM padding requirement from the quant method.
        # When hidden_size is not aligned to 256, MXFP4 weights are padded
        # and the kernel expects padded input. We handle padding here instead
        # of in the layernorm, so the layernorm can use fused AllReduce.
        if hasattr(self.experts.quant_method, "hidden_pad"):
            self.moe_hidden_pad = self.experts.quant_method.hidden_pad
        else:
            self.moe_hidden_pad = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.shape[0]

        g = self.router(x)

        # Pad input for MXFP4 MoE GEMM alignment if needed
        if self.moe_hidden_pad > 0:
            x = F.pad(x, (0, self.moe_hidden_pad))

        x = self.experts(hidden_states=x, router_logits=g)

        # Remove padding from output
        if self.moe_hidden_pad > 0:
            x = x[:, : self.hidden_size]

        if self.tp_size > 1 and not ENABLE_ALLREDUCE_RMSNORM_FUSION:
            x = tensor_model_parallel_all_reduce(x)

        if self.is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x.contiguous(), 0)
            x = x[:num_tokens]
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        atom_config: Config,
        quant_config: QuantizationConfig,
        prefix: str = "",
        layer_num: int = 0,
    ):
        super().__init__()

        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype

        self.layer_idx = layer_num
        self.hidden_size = atom_config.hf_config.hidden_size
        self.self_attn = OAIAttention(
            config,
            prefix=f"{prefix}.self_attn",
            quant_config=quant_config,
            cache_config=cache_config,
            layer_num=layer_num,
        )
        self.mlp = MLPBlock(atom_config, self.layer_idx, prefix=f"{prefix}.mlp")
        # Fuse MoE AllReduce into input_layernorm for layers > 0.
        # Layer 0 receives already-reduced embedding output, so no fusion needed.
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=1e-5,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION and layer_num > 0,
        )
        # Fuse o_proj AllReduce into post_attention_layernorm.
        # Padding for MXFP4 MoE GEMM alignment is now handled inside MLPBlock,
        # so this layernorm no longer needs x_pad_to_multiple.
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=1e-5,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, positions)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        output = self.mlp(hidden_states)
        return output, residual


@support_torch_compile
class GptOssModel(nn.Module):
    def __init__(
        self,
        *,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        self.config = atom_config.hf_config
        self.quant_config = atom_config.quant_config
        self.config.hidden_size = self.config.hidden_size
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.embed_tokens = self.embedding
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix, layer_num=None: TransformerBlock(
                atom_config,
                prefix=prefix,
                quant_config=self.quant_config,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=1e-5,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )
        self.aux_hidden_state_layers = tuple[int, ...]()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                x = self.get_input_embeddings(input_ids)

            residual = None
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if i in self.aux_hidden_state_layers:
                aux_hidden_states.append(x if residual is None else x + residual)
            x, residual = layer(x, positions, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": x, "residual": residual})
        x, _ = self.norm(x, residual)

        if len(aux_hidden_states) > 0:
            return x, aux_hidden_states
        return x


class GptOssForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    weights_mapping = {
        # MoE MXFP4 weights
        "gate_up_proj_blocks": "w13_weight",
        "down_proj_blocks": "w2_weight",
        "gate_up_proj_scales": "w13_weight_scale",
        "down_proj_scales": "w2_weight_scale",
        # MoE other weights
        "gate_up_proj": "w13_weight",
        "down_proj": "w2_weight",
        # MoE Bias
        "gate_up_proj_bias": "w13_bias",
        "down_proj_bias": "w2_bias",
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        self.atom_config = atom_config
        self.config = atom_config.hf_config

        self.model = GptOssModel(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, weight scales, activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
        )
