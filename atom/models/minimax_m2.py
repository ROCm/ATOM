from typing import Optional, Union

import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from atom.model_ops.moe import FusedMoE
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils import envs
from atom.utils.decorators import support_torch_compile
from torch import nn
from transformers import PretrainedConfig

ENABLE_ALLREDUCE_RMSNORM_FUSION = envs.ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.num_experts = config.num_local_experts
        if self.tp_size > self.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_experts}."
            )

        if getattr(config, "use_routing_bias", False):
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.num_experts, dtype=torch.float32)
            )
        else:
            self.register_parameter("e_score_correction_bias", None)

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=False,
            renormalize=True,
            quant_config=quant_config,
            use_grouped_topk=False,
            scoring_func=getattr(config, "scoring_func", "softmax"),
            e_score_correction_bias=self.e_score_correction_bias,
            prefix=f"{prefix}.experts",
            has_bias=False,
            config=config,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert (
            hidden_states.dim() <= 2
        ), "MiniMaxM2SparseMoeBlock only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.tp_size > 1 and not ENABLE_ALLREDUCE_RMSNORM_FUSION:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class MiniMaxM2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        rotary_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        rope_scaling: tuple | None,
        qkv_bias: bool,
        kv_cache_dtype: str,
        layer_num: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=not ENABLE_ALLREDUCE_RMSNORM_FUSION,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.q_size, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.kv_size, eps=rms_norm_eps)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_output = self.attn(q, k, v, positions)
        output = self.o_proj(attn_output)
        return output


class MiniMaxM2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        layer_num: int = 0,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_num
        self.hidden_size = config.hidden_size

        self.self_attn = MiniMaxM2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=getattr(config, "max_position_embeddings", 8192),
            head_dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            rotary_dim=getattr(
                config,
                "rotary_dim",
                getattr(
                    config, "head_dim", config.hidden_size // config.num_attention_heads
                ),
            ),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            qkv_bias=bool(getattr(config, "attention_bias", False) or False),
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            use_qk_norm=getattr(config, "use_qk_norm", True),
        )

        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe",
        )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION and self.layer_idx > 0,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
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

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)

        return hidden_states, residual


@support_torch_compile
class MiniMaxM2Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = MiniMaxM2DecoderLayer,
    ):
        super().__init__()

        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config
        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: layer_type(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
            )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )


class MiniMaxM2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = MiniMaxM2DecoderLayer,
    ):
        super().__init__()
        config = atom_config.hf_config
        self.config = config

        self.model = MiniMaxM2Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
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
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.lm_head(hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
