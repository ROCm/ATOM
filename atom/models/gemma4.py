# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Gemma 4 model (text-only backbone)."""

from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn

from aiter import gelu_tanh_and_mul
from aiter.dist.parallel_state import get_tp_group
from aiter.rotary_embedding import get_rope

from atom.config import Config, QuantizationConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from atom.model_loader.loader import load_model_in_plugin_mode
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.models.utils import maybe_prefix
from atom.utils.decorators import support_torch_compile



def fused_logit_softcap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """In-place fused logit softcapping: tanh(x / cap) * cap."""
    return logits.div_(cap).tanh_().mul_(cap)


# ---------------------------------------------------------------------------
# AITER-accelerated GeluAndMul activation
# Uses aiter.gelu_tanh_and_mul CUDA JIT kernel instead of PyTorch F.gelu
# ---------------------------------------------------------------------------

class GeluAndMul(nn.Module):
    """AITER-accelerated GELU-gated activation for Gemma 4.

    Uses aiter.gelu_tanh_and_mul CUDA JIT kernel instead of PyTorch F.gelu.
    """

    def __init__(
        self,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(
            [*x.shape[:-1], x.shape[-1] // 2], device=x.device, dtype=x.dtype
        )
        gelu_tanh_and_mul(out, x)
        return out


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class _Gemma4RMSNorm(nn.Module):
    """RMSNorm for Gemma 4 (standard x*weight formula, NOT the Gemma1/2 x*(1+weight) convention).

    Supports with_scale=False for v_norm (pure normalization, no learnable weights).
    Uses AITER rmsnorm2d_fwd kernel when available (requires aiter >= 0.1.0 with bf16 dtype support).
    """
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.variance_epsilon = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        try:
            from aiter import rmsnorm2d_fwd
            from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add
            self._aiter_rmsnorm = rmsnorm2d_fwd
            self._aiter_rmsnorm_add = rmsnorm2d_fwd_with_add
        except ImportError:
            self._aiter_rmsnorm = None
            self._aiter_rmsnorm_add = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None and self.with_scale and self._aiter_rmsnorm_add is not None:
            out = torch.empty_like(x)
            residual_out = torch.empty_like(x)
            self._aiter_rmsnorm_add(out, x, residual, residual_out, self.weight, self.eps)
            return out, residual_out

        if residual is not None:
            x = x + residual
            residual = x

        if self.with_scale and self._aiter_rmsnorm is not None:
            x = self._aiter_rmsnorm(x, self.weight, self.eps)
        else:
            orig_dtype = x.dtype
            x = self._norm(x.float())
            if self.with_scale:
                x = x * self.weight.float()
            x = x.to(orig_dtype)

        return x if residual is None else (x, residual)

class Gemma4Attention(nn.Module):
    """Multi-head attention for Gemma 4 with sliding/global window support.

    Gemma 4 uses different head_dim and num_kv_heads for global vs sliding
    attention layers, and applies per-type RoPE configurations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        rope_scaling: dict | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        layer_num: int,
        atom_config: Config,
        is_global: bool = False,
        attention_k_eq_v: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tp_group().world_size
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = 1
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Gemma4 uses scaling=1.0 (no 1/sqrt(head_dim)) because q_norm/k_norm
        # already control the magnitude of Q and K.
        self.scaling = 1.0
        self.is_global = is_global
        self.attention_k_eq_v = attention_k_eq_v
        self._layer_num = layer_num

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        rotary_dim = head_dim
        partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0) if rope_scaling else 1.0
        if partial_rotary_factor < 1.0:
            rotary_dim = int(head_dim * partial_rotary_factor)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        sw = sliding_window if not is_global else None
        self.q_norm = _Gemma4RMSNorm(self.head_dim, eps=rms_norm_eps, with_scale=True)
        self.k_norm = _Gemma4RMSNorm(self.head_dim, eps=rms_norm_eps, with_scale=True)
        self.v_norm = _Gemma4RMSNorm(self.head_dim, eps=rms_norm_eps, with_scale=False)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            config=atom_config,
            per_layer_sliding_window=sw,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **model_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(
            qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        num_tokens = q.shape[0]
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(num_tokens, -1)
        if self.attention_k_eq_v:
            v = self.v_norm(k.reshape(-1, self.head_dim)).reshape(num_tokens, -1)
        else:
            v = self.v_norm(v.reshape(-1, self.head_dim)).reshape(num_tokens, -1)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(num_tokens, -1)
        o = self.attn(q, k, v, positions, **model_kwargs)

        output = self.o_proj(o)
        return output


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
        self.act_fn = GeluAndMul(
            quant_config=quant_config,
            prefix=f"{prefix}.act_fn",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Gemma4SparseMoeBlock(nn.Module):
    """Sparse MoE block for Gemma 4 26B-A4B variant.

    Uses AITER FusedMoE (ASM + CK backend) for high-throughput expert dispatch.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.moe_intermediate_size = config.moe_intermediate_size

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        self.router = nn.Linear(
            config.hidden_size, self.num_experts, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(hidden_states)
        return self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )


class Gemma4DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma4TextConfig,
        atom_config: Config,
        layer_num: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.layer_type = config.layer_types[layer_num]
        is_global = self.layer_type == "full_attention"

        if is_global:
            num_kv_heads = config.num_global_key_value_heads
            head_dim = config.global_head_dim
            rope_params = config.rope_parameters.get("full_attention", {}) if config.rope_parameters else {}
        else:
            num_kv_heads = config.num_key_value_heads
            head_dim = config.head_dim
            rope_params = config.rope_parameters.get("sliding_attention", {}) if config.rope_parameters else {}

        rope_theta = rope_params.get("rope_theta", 10000.0)

        self.self_attn = Gemma4Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_params,
            sliding_window=config.sliding_window,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            layer_num=layer_num,
            atom_config=atom_config,
            is_global=is_global,
            attention_k_eq_v=is_global and getattr(config, "attention_k_eq_v", True),
            prefix=f"{prefix}.self_attn",
        )

        if config.enable_moe_block:
            self.mlp = Gemma4SparseMoeBlock(
                config=config,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Gemma4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=atom_config.quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = _Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = _Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = _Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = _Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **model_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **model_kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar

        # layer_scalar is folded into the full layer output, which is
        # incompatible with the cross-layer residual carry chain: the next
        # layer would need to recover (delta, residual) from the scaled
        # output to fuse input_layernorm with our residual, but that split
        # cannot be reconstructed once the scalar is applied. Return None
        # to deliberately break the carry chain and force the next layer's
        # input_layernorm to start a fresh residual from hidden_states.
        return hidden_states, None


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Gemma4Model(nn.Module):

    def __init__(self, *, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = atom_config.hf_config
        if hasattr(config, "text_config"):
            config = config.text_config

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config=config,
                    atom_config=atom_config,
                    layer_num=layer_num,
                    prefix=f"{prefix}.layers.{layer_num}",
                )
                for layer_num in range(config.num_hidden_layers)
            ]
        )
        self.norm = _Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hidden_size = config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        hidden_states = hidden_states * (self.hidden_size**0.5)

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                **model_kwargs,
            )

        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    weights_mapping = {
        "model.language_model.": "model.",
    }
    quant_exclude_name_mapping = {
        "model.language_model.": "model.",
    }
    skip_weight_prefixes = [
        "model.vision_tower.",
        "model.embed_vision.",
    ]

    def __init__(self, config: Any, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = self.atom_config.hf_config
        text_config = self.hf_config
        if hasattr(self.hf_config, "text_config"):
            text_config = self.hf_config.text_config

        self.model = Gemma4Model(
            atom_config=self.atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logit_softcapping = getattr(
            text_config, "final_logit_softcapping", None
        )

        if text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            **model_kwargs,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)

        if self.logit_softcapping is not None and self.logit_softcapping > 0:
            logits = fused_logit_softcap(logits, self.logit_softcapping)

        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loaded_weights_record = load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
        )
        for module in self.modules():
            if hasattr(module, '_invalidate_weight_cache'):
                module._invalidate_weight_cache()
        return loaded_weights_record
