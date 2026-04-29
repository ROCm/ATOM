# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mimo_v2_flash.py

from typing import Optional, Union

import torch
import torch.nn.functional as F
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.utils import atom_parameter
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from torch import nn
from transformers import PretrainedConfig
from atom.utils.decorators import support_torch_compile


class MiMoV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
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
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class MiMoV2MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts = config.n_routed_experts

        if self.tp_size > self.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        # Gate weights in fp32 for routing precision
        old_wlp = self.gate.weight.weight_loader_process
        self.gate.weight = atom_parameter(self.gate.weight.data.to(torch.float32))
        self.gate.weight.weight_loader_process = old_wlp

        # Attach to self.gate so the parameter path matches the checkpoint:
        # model.layers.N.mlp.gate.e_score_correction_bias
        self.gate.e_score_correction_bias = atom_parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            prefix=f"{prefix}.experts",
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, "MiMoV2MoE only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = F.linear(hidden_states.float(), self.gate.weight.float())
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class MiMoV2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int | None = None,
        v_scale: float | None = None,
        sliding_window_size: int = -1,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        partial_rotary_factor: float = 1.0,
        kv_cache_dtype: str = "bf16",
        layer_num: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim

        self.v_scale = v_scale
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            v_head_size=self.v_head_dim,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
        )

        self.attention_sink_bias = (
            atom_parameter(torch.empty(self.num_heads))
            if add_swa_attention_sink_bias
            else None
        )

        sliding_window = sliding_window_size if sliding_window_size > 0 else None
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            per_layer_sliding_window=sliding_window,
            sinks=self.attention_sink_bias,
            prefix=f"{prefix}.attn",
            # rotary_emb=self.rotary_emb,
            rotary_emb=None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.k_size, self.v_size], dim=-1)

        # RoPE and reshape cache fusion is not enabled, since the kernel
        # fused_qk_rope_reshape_and_cache currently does not support head_dim=192.
        q, k = self.rotary_emb(positions, q, k)
        # Apply v_scale before attention
        if self.v_scale is not None:
            v = v * self.v_scale

        # Zero-pad V from v_head_dim to head_dim for the attention kernel
        if self.v_head_dim != self.head_dim:
            v = v.view(-1, self.num_kv_heads, self.v_head_dim)
            v = F.pad(v, [0, self.head_dim - self.v_head_dim], value=0)
            v = v.view(-1, self.num_kv_heads * self.head_dim)

        attn_output = self.attn(q, k, v, positions)

        # Truncate output back to v_head_dim
        if self.v_head_dim != self.head_dim:
            attn_output = attn_output.view(-1, self.num_heads, self.head_dim)[
                ..., : self.v_head_dim
            ].reshape(-1, self.num_heads * self.v_head_dim)

        output = self.o_proj(attn_output)
        return output


class MiMoV2FlashDecoderLayer(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        layer_num: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        kv_cache_dtype = atom_config.kv_cache_dtype

        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_num

        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        v_scale = getattr(config, "attention_value_scale", None)

        if self.is_compressed_softmax_layer():
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=config.sliding_window_size,
                attention_bias=getattr(config, "attention_bias", False),
                add_swa_attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                max_position_embeddings=max_position_embeddings,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                kv_cache_dtype=kv_cache_dtype,
                layer_num=layer_num,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=-1,
                attention_bias=getattr(config, "attention_bias", False),
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                kv_cache_dtype=kv_cache_dtype,
                layer_num=layer_num,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        if self.is_moe_layer(layer_num):
            self.mlp = MiMoV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MiMoV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=True,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            fused_allreduce=False,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            fused_allreduce=False,
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

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            hasattr(self.config, "moe_layer_freq")
            and layer_idx >= 0
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

    def is_compressed_softmax_layer(self) -> bool:
        return self.config.hybrid_layer_pattern[self.layer_idx] == 1


@support_torch_compile
class MiMoV2Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        self.config = atom_config.hf_config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix, layer_num=None: MiMoV2FlashDecoderLayer(
                atom_config=atom_config,
                layer_num=layer_num,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                fused_allreduce=False,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )


class MiMoV2FlashForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        self.atom_config = atom_config
        self.config = atom_config.hf_config

        self.model = MiMoV2Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                num_embeddings=self.config.vocab_size,
                embedding_dim=self.config.hidden_size,
                bias=False,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
