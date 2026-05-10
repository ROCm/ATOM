# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Eagle3 draft model (Llama full-attention) for speculative decoding.

Implements the Eagle3 draft model matching the lightseekorg/kimi-k2.5-eagle3
checkpoint layout:

    embed_tokens.weight   — independent embedding
    fc.weight             — aux fusion projection (hidden*3 -> hidden)
    midlayer.*            — single decoder layer (dual-norm, wide QKV)
    norm.weight           — final RMSNorm
    lm_head.weight        — independent lm_head

Weight keys map directly to model attribute paths; no key rewriting needed.
"""

import torch
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from atom.config import Config
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
from atom.utils.decorators import support_torch_compile
from torch import nn


class Eagle3LlamaAttention(nn.Module):
    """Llama full-attention with input_size = hidden_size * 2.

    The QKV projection accepts the concatenation of normalized embeddings
    and fc output, hence input_size is doubled compared to standard Llama.
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
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
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # QKV input_size = hidden_size * 2 (concat of embed + fc_output)
        attn_input_size = hidden_size * 2
        self.qkv_proj = QKVParallelLinear(
            hidden_size=attn_input_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
        )

        sliding_window = -1
        if getattr(config, "use_sliding_window", False) and getattr(
            config, "sliding_window", None
        ):
            sliding_window = config.sliding_window
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            prefix=f"{prefix}.attn",
            rotary_emb=self.rotary_emb,
            per_layer_sliding_window=sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, positions)
        output = self.o_proj(attn_output)
        return output


class Eagle3LlamaDecoderLayer(nn.Module):
    """Single decoder layer for Eagle3 with dual-norm input.

    Unlike standard LlamaDecoderLayer, this layer has:
    - input_layernorm: normalizes the embedding input
    - hidden_norm: normalizes the fc output (projected aux hidden states)
    - Attention input is concat(normed_embed, normed_hidden) -> [N, hidden*2]
    """

    def __init__(
        self,
        config,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Eagle3LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
        )

        self.mlp = Eagle3LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            prefix=f"{prefix}.mlp",
        )

        # Dual norms matching checkpoint keys: midlayer.input_layernorm, midlayer.hidden_norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        normed_embeds = self.input_layernorm(embeds)
        normed_hidden = self.hidden_norm(hidden_states)
        # Concat for attention input: [N, hidden*2]
        attn_input = torch.cat([normed_embeds, normed_hidden], dim=-1)
        attn_output = self.self_attn(positions, attn_input)
        # Residual connection on hidden_states
        hidden_states = hidden_states + attn_output
        # MLP with pre-norm + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Eagle3LlamaMLP(nn.Module):
    """Simple Llama MLP (gate+up fused, silu activation, down projection)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


@support_torch_compile
class Eagle3LlamaModel(nn.Module):
    """Eagle3 draft model (Llama full-attention, single decoder layer).

    Matches the lightseekorg/kimi-k2.5-eagle3 checkpoint layout:
        embed_tokens.weight   [163840, 7168]  independent embedding
        fc.weight             [7168, 21504]   aux fusion (hidden*3 -> hidden)
        midlayer.*            single decoder layer
        norm.weight           final RMSNorm
        lm_head.weight        [163840, 7168]  independent lm_head
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, atom_config: Config, prefix: str = "", layer_offset: int = 0):
        super().__init__()
        config = atom_config.hf_config
        cache_config = atom_config.kv_cache_dtype
        self.config = config

        # Independent embedding (vocab matches target model)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        # Aux fusion: concatenated aux hidden states [N, hidden*3] -> [N, hidden]
        self.fc = ReplicatedLinear(
            config.hidden_size * 3, config.hidden_size, bias=False
        )

        # Draft attention layer_num must start from the target model's layer
        # count so kv_cache_data["layer_N"] maps to the correct cache entry.
        self.midlayer = Eagle3LlamaDecoderLayer(
            config=config,
            cache_config=cache_config,
            prefix="midlayer",
            layer_num=layer_offset,
        )

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Independent lm_head (not shared with target model)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated aux hidden states through fc.

        Args:
            hidden_states: [N, hidden_size * 3] (3 aux layers concatenated)

        Returns:
            [N, hidden_size] projected hidden states
        """
        return self.fc(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.embed_tokens(input_ids)
        hidden_states = self.midlayer(positions, embeds, hidden_states)
        hidden_states_prenorm = hidden_states
        hidden_states = self.norm(hidden_states)
        return hidden_states, hidden_states_prenorm

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
