# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Gemma 4 model (text-only backbone)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn

from aiter import ActivationType, gelu_tanh_and_mul
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import get_tp_group
from aiter.rotary_embedding import get_rope

from atom.config import Config, QuantizationConfig

# `transformers.models.gemma4` was introduced in transformers 5.5.0, but this
# repo pins `transformers==5.2.0` in pyproject.toml (see #322, which pinned
# down to avoid a 5.3.0 tokenizer regression). To avoid forcing every other
# ATOM user to upgrade for Gemma 4 support, we only need Gemma4TextConfig at
# type-check time — the runtime path receives an already-instantiated config
# via `atom_config.hf_config`, never the class itself. Combined with
# `from __future__ import annotations` above this turns the Gemma4TextConfig
# annotations into strings, so importing this module on transformers 5.2.0
# no longer fails at import time. Anyone actually instantiating Gemma 4 still
# needs transformers >= 5.5.0 — that's a runtime requirement enforced by the
# config loader, not by this module.
if TYPE_CHECKING:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from atom.model_loader.loader import load_model_in_plugin_mode
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
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


# ---------------------------------------------------------------------------
# Fused-expert checkpoint loading (Gemma 4 26B-A4B MoE)
#
# The MoE checkpoint stores all routed experts of a layer in two 3D tensors,
# `experts.gate_up_proj` [E, 2*I, H] and `experts.down_proj` [E, H, I], which
# map directly onto FusedMoE's `w13_weight` / `w2_weight`. These helpers split
# the fused gate_up tensor into gate (w1) / up (w3) halves and drive FusedMoE's
# per-expert `weight_loader`. Mirrors the Qwen3.5 BF16 fused-expert path.
# ---------------------------------------------------------------------------

def _gemma4_detect_fused_expert_format(weight_name: str) -> bool:
    return "experts.gate_up_proj" in weight_name or (
        "experts.down_proj" in weight_name and ".experts." in weight_name
    )


def _gemma4_get_fused_expert_mapping() -> list[tuple[str, str, str]]:
    # (param_name, weight_name, shard_id); gate_up_proj is chunked in the loader.
    return [
        ("experts.w13_weight", "experts.gate_up_proj", "w1"),
        ("experts.w2_weight", "experts.down_proj", "w2"),
    ]


# aiter's CK 2-stage MoE GEMM requires the per-partition intermediate dimension
# to be a multiple of this tile (the stage-2 contraction K = intermediate/TP).
# 32 is enough for the weight *shuffle*, but the GEMM instance itself needs 128;
# Gemma 4's 704 gives 88 at TP=8, which fails, so we round up to 128.
_GEMMA4_MOE_INTERMEDIATE_ALIGN = 128


def _gemma4_padded_moe_intermediate(moe_intermediate: int, tp_size: int) -> int:
    """Pad the MoE intermediate so every TP partition is a multiple of the tile.

    Gemma 4's moe_intermediate (704) gives 88 per partition at TP=8, which is not
    a valid tile for the aiter CK MoE GEMM. We round each partition up to the
    next multiple of ``_GEMMA4_MOE_INTERMEDIATE_ALIGN`` (128 -> per-partition 128,
    total 1024) and zero-pad the extra expert units. This is numerically exact: a
    zero gate/up column yields ``gelu(0) * 0 == 0``, and the matching zero
    down-projection row contributes nothing to the output.
    """
    align = _GEMMA4_MOE_INTERMEDIATE_ALIGN
    inter_pp = (moe_intermediate + tp_size - 1) // tp_size
    inter_pp = ((inter_pp + align - 1) // align) * align
    return inter_pp * tp_size


def _gemma4_pad_intermediate(
    t: torch.Tensor, dim: int, target: int
) -> torch.Tensor:
    pad_len = target - t.shape[dim]
    if pad_len <= 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_len
    zeros = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, zeros], dim=dim)


def _gemma4_load_fused_expert_weights(
    original_name: str,
    name: str,
    params_dict: dict,
    loaded_weight: torch.Tensor,
    shard_id: str,
    num_experts: int,
    intermediate_padded: Optional[int] = None,
) -> bool:
    param = params_dict[name]
    weight_loader = param.weight_loader
    loaded = False
    if "gate_up_proj" in original_name:
        # loaded_weight: [E, 2*I, H] -> gate/up each [E, I, H]
        gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
        if intermediate_padded is not None:
            gate_weight = _gemma4_pad_intermediate(gate_weight, 1, intermediate_padded)
            up_weight = _gemma4_pad_intermediate(up_weight, 1, intermediate_padded)
        for expert_id in range(num_experts):
            weight_loader(param, gate_weight[expert_id], name, "w1", expert_id)
            weight_loader(param, up_weight[expert_id], name, "w3", expert_id)
            loaded = True
    else:
        # down_proj: [E, H, I]
        if intermediate_padded is not None:
            loaded_weight = _gemma4_pad_intermediate(
                loaded_weight, loaded_weight.dim() - 1, intermediate_padded
            )
        for expert_id in range(num_experts):
            weight_loader(param, loaded_weight[expert_id], name, shard_id, expert_id)
            loaded = True
    return loaded


class Gemma4Router(nn.Module):
    """Router for the Gemma 4 MoE block.

    Produces expert logits via ``proj(rmsnorm(x) * scale * hidden**-0.5)``. The
    softmax / top-k / renormalization are delegated to FusedMoE. The learned
    ``per_expert_scale`` (a post-top-k multiplicative weight per selected expert)
    is folded into the routed experts' down-projection at load time — see
    ``Gemma4DecoderLayer.process_weights_after_loading`` — which is exact because
    it commutes with the renormalized weighted sum FusedMoE computes.
    """

    def __init__(self, config: Gemma4TextConfig, prefix: str = "") -> None:
        super().__init__()
        hidden = config.hidden_size
        self.scalar_root_size = hidden**-0.5
        self.norm = _Gemma4RMSNorm(hidden, eps=config.rms_norm_eps, with_scale=False)
        self.proj = ReplicatedLinear(
            hidden, config.num_experts, bias=False, prefix=f"{prefix}.proj"
        )
        self.scale = nn.Parameter(torch.ones(hidden))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        return self.proj(hidden_states)


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

        self.enable_moe_block = config.enable_moe_block
        self.tp_size = get_tp_group().world_size

        # Dense feed-forward. For dense Gemma 4 variants this is the whole FFN;
        # for the MoE variant it is the always-on shared expert (`mlp.*`), run in
        # parallel with the routed experts below.
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.mlp",
        )

        if self.enable_moe_block:
            self.router = Gemma4Router(config, prefix=f"{prefix}.router")
            # aiter's MoE weight shuffle needs each TP partition's intermediate
            # dimension to be 32-aligned; pad it (zero-filled) when it is not.
            moe_intermediate = _gemma4_padded_moe_intermediate(
                config.moe_intermediate_size, self.tp_size
            )
            self.experts = FusedMoE(
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                hidden_size=config.hidden_size,
                intermediate_size=moe_intermediate,
                reduce_results=False,
                renormalize=True,
                quant_config=atom_config.quant_config,
                scoring_func="softmax",
                activation=ActivationType.Gelu,
                has_bias=False,
                prefix=f"{prefix}.experts",
                config=config,
            )
            self.post_feedforward_layernorm_1 = _Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = _Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = _Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
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
        # The fused norm-add leaves `hidden_states` = pre_feedforward_layernorm of
        # the post-attention residual, and `residual` = the post-attention residual
        # itself (the input to both FFN branches below).
        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)

        if self.enable_moe_block:
            # Dense shared-expert branch (on the pre-FFN-normed hidden states).
            shared = self.mlp(hidden_states)
            shared = self.post_feedforward_layernorm_1(shared)
            # Routed-expert branch operates on the post-attention residual; the
            # router applies its own internal (scale-free) RMSNorm.
            router_logits = self.router(residual)
            routed = self.pre_feedforward_layernorm_2(residual)
            routed = self.experts(hidden_states=routed, router_logits=router_logits)
            if self.tp_size > 1:
                routed = tensor_model_parallel_all_reduce(routed)
            routed = self.post_feedforward_layernorm_2(routed)
            hidden_states = shared + routed
        else:
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

    def process_weights_after_loading(self) -> None:
        # Fold the router's learned per-expert scale into the routed experts'
        # down-projection (w2). `per_expert_scale[e]` multiplies expert e's
        # renormalized routing weight, which is exactly equivalent to scaling
        # expert e's linear output — so baking it into w2 is numerically exact.
        # named_modules() yields this layer before its `experts` child, so this
        # runs before FusedMoE's own weight shuffle; a per-expert scalar commutes
        # with the intra-expert shuffle, so ordering is safe.
        if not self.enable_moe_block:
            return
        w2 = self.experts.w2_weight
        per_expert_scale = self.router.per_expert_scale.data.to(w2.dtype)
        w2.data.mul_(per_expert_scale.view(-1, 1, 1))


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
        for layer in self.layers:
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
        # `mlp.`-anchored so the routed experts' fused `experts.gate_up_proj`
        # tensor (whose name contains the substring `up_proj`) is NOT misrouted
        # into the dense-MLP packing path — it is loaded via the fused-expert
        # path (`load_fused_expert_weights`) instead.
        "mlp.gate_proj": ("mlp.gate_up_proj", 0),
        "mlp.up_proj": ("mlp.gate_up_proj", 1),
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
        self._text_config = text_config

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
        # Pipeline parallelism: non-first PP ranks receive hidden_states via
        # `intermediate_tensors` instead of input_ids. Gemma4Model.forward does
        # not yet implement the PP-rank split (no make_layers, no
        # `if get_pp_group().is_first_rank` branch), so PP > 1 is unsupported
        # for this model today. Fail loudly here instead of crashing in
        # embed_tokens(None) inside Gemma4Model.
        if intermediate_tensors is not None:
            raise NotImplementedError(
                "Gemma4ForCausalLM does not support pipeline parallelism yet "
                "(received intermediate_tensors from a previous PP rank, but "
                "Gemma4Model.forward does not implement the rank-split / "
                "IntermediateTensors path). Use tensor parallelism only."
            )
        # Forward inputs_embeds through so the plugin's get_input_embeddings()
        # path (used e.g. by vLLM's spec-decode and multimodal preprocessing
        # even on a single PP rank) reaches Gemma4Model's existing
        # `if inputs_embeds is not None` branch instead of getting silently
        # dropped and re-embedding None.
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
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
            load_fused_expert_weights_fn=self.load_fused_expert_weights,
        )
        for module in self.modules():
            if hasattr(module, '_invalidate_weight_cache'):
                module._invalidate_weight_cache()
        return loaded_weights_record

    # ------------------------------------------------------------------
    # MoE expert-weight loading hooks (consumed by ATOM's weight loader).
    # These are only exercised by the 26B-A4B MoE checkpoint; for the dense
    # 31B checkpoint `get_expert_mapping` returns [] and the fused-format
    # detector never fires, so the dense load path is unchanged.
    # ------------------------------------------------------------------
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        if not getattr(self._text_config, "enable_moe_block", False):
            return []
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self._text_config.num_experts,
        )

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        return _gemma4_detect_fused_expert_format(weight_name)

    def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
        return _gemma4_get_fused_expert_mapping()

    def load_fused_expert_weights(
        self,
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        if not num_experts:
            num_experts = getattr(self._text_config, "num_experts", 0)
        intermediate_padded = _gemma4_padded_moe_intermediate(
            self._text_config.moe_intermediate_size, get_tp_group().world_size
        )
        return _gemma4_load_fused_expert_weights(
            original_name,
            name,
            params_dict,
            loaded_weight,
            shard_id,
            num_experts,
            intermediate_padded=intermediate_padded,
        )
