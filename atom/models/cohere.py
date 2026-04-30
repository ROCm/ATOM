# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from vLLM's Cohere implementation and ATOM's llama.py template.
# CohereForCausalLM differs from Llama in:
#   - LayerNorm (with bias) instead of RMSNorm
#   - rope_theta directly on CohereConfig (no rope_parameters wrapper)
#   - tie_word_embeddings=True by default
#   - Optional Q/K layer normalization (use_qk_norm)
#   - layer_types / sliding_window on some variants

"""Inference-only Cohere model compatible with HuggingFace weights."""

from typing import Any, Optional, Union

import torch
from aiter import QuantType
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import LayerNorm, RMSNorm
from atom.model_ops.layernorm import layernorm2d_fwd_, layernorm2d_fwd_with_add_
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.decorators import support_torch_compile
from torch import nn
from transformers import CohereConfig


class CohereLayerNorm(nn.Module):
    """LayerNorm using AITER kernels with a zero bias that is not a parameter.

    Cohere2 checkpoints do not store input_layernorm.bias (it is always zero).
    Using a non-parameter buffer avoids the vLLM weight-completeness check
    while still letting the AITER fused kernel accept a bias argument.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("bias", torch.zeros(dim))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return layernorm2d_fwd_(x, self.weight, self.bias, self.eps, self.dim)
        return layernorm2d_fwd_with_add_(
            x, self.weight, residual, self.bias, self.eps, self.dim
        )


class CohereMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul(fused_quant=False, quant_config=quant_config)

    def forward(self, x):
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class CohereAttention(nn.Module):
    def __init__(
        self,
        config: CohereConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
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
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_num = layer_num

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Optional Q/K layer normalization (used in some Cohere variants)
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.layer_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=None,
            is_neox_style=False,  # Cohere uses interleaved RoPE (repeat_interleave), not neox
        )

        # Sliding window support (some Cohere variants).
        # Use -1 (ATOM's "no sliding window" sentinel) as the explicit default
        # so that vLLM's Attention layer does not inherit the global
        # cache_config.sliding_window (from config.sliding_window = 4096)
        # for layers that are NOT sliding_attention layers.
        sliding_window = -1
        if layer_types := getattr(config, "layer_types", None):
            if layer_types[layer_idx] == "sliding_attention":
                sliding_window = config.sliding_window

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
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
            # Reshape to apply per-head norm, then flatten back
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q).view(-1, self.q_size)
            k = self.k_norm(k).view(-1, self.kv_size)

        attn_output = self.attn(q, k, v, positions)
        output = self.o_proj(attn_output)
        return output


class CohereDecoderLayer(nn.Module):
    def __init__(
        self,
        config: CohereConfig,
        quant_config: QuantizationConfig,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CohereAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            quant_config=quant_config,
            bias=getattr(config, "attention_bias", False),
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
        )
        self.mlp = CohereMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        # Cohere uses a single LayerNorm (with bias) shared by both attn and MLP
        self.input_layernorm = CohereLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Cohere parallel residual: one pre-norm feeds both attn and MLP.
        # residual accumulates the sum; hidden_states carries the new contribution.
        if residual is None:
            residual = hidden_states
            normed = self.input_layernorm(hidden_states)
        else:
            normed, residual = self.input_layernorm(hidden_states, residual)

        attn_output = self.self_attn(positions=positions, hidden_states=normed)
        mlp_output = self.mlp(normed)

        # Both attn and mlp outputs are summed; this becomes the new hidden_states
        # contribution that will be added to residual in the next layer.
        hidden_states = attn_output + mlp_output
        return hidden_states, residual


@support_torch_compile
class CohereModel(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
        layer_type: type[nn.Module] = CohereDecoderLayer,
    ):
        super().__init__()
        self.atom_config = atom_config
        config = atom_config.hf_config
        self.config = config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        if get_pp_group().is_last_rank:
            # Cohere uses LayerNorm for the final norm as well
            self.norm = CohereLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[
        torch.Tensor, IntermediateTensors, tuple[torch.Tensor, list[torch.Tensor]]
    ]:
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


class CohereForCausalLM(nn.Module):
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
        layer_type: type[nn.Module] = CohereDecoderLayer,
    ):
        super().__init__()
        config = atom_config.hf_config
        self.model = CohereModel(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits

