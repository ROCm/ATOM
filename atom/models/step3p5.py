# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Step-3.5 (Flash) model.

Step-3.5 is a sparse MoE transformer with:
  - 45 decoder layers, hidden_size=4096, head_dim=128
  - GQA with two attention configs: full_attention (64 q heads, 8 kv groups)
    and sliding_attention (96 q heads, 8 kv groups, window=512)
  - 3:1 sliding window pattern (1 full + 3 sliding)
  - Per-layer rope_theta and partial_rotary_factor
  - QK RMSNorm (zero-centered, i.e. weight * (1 + param))
  - Head-wise attention gating via g_proj (sigmoid)
  - MoE on layers 3-44: 288 routed experts + 1 shared expert, top-8,
    sigmoid routing with learnable router bias
  - Dense MLP on layers 0-2
  - Per-layer SwiGLU clamp limits
  - Multi-token prediction (MTP) with num_nextn_predict_layers=3
"""

import os
from typing import Optional, Union

import torch
from aiter import ActivationType
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from atom.config import Config, QuantizationConfig
from atom.model_ops.activation import SiluAndMul
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import GemmaRMSNorm as Step3p5RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
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
from transformers import PretrainedConfig


def _uses_swiglustep_at_layer(
    config: PretrainedConfig, layer_idx: Optional[int]
) -> bool:
    """Return True iff the routed FusedMoE at this layer needs the SwigluStep
    activation (i.e. ``swiglu_limits[layer_idx] > 0``).

    The CK kernel hard-codes the clamp at 7.0; Step-3.5-Flash uses 7.0 at
    layers 43 and 44, which is why the kernel is only valid at those layers.
    Other layers must keep the plain Silu path.
    """
    if layer_idx is None:
        return False
    # Toggle-off bit: ATOM_DISABLE_SWIGLUSTEP=1 forces plain Silu at every
    # layer (verification helper only).
    if os.environ.get("ATOM_DISABLE_SWIGLUSTEP"):
        return False
    swiglu_limits = getattr(config, "swiglu_limits", None)
    if not swiglu_limits or layer_idx >= len(swiglu_limits):
        return False
    return bool(swiglu_limits[layer_idx])


def _fuse_shared_at_layer(config: PretrainedConfig, layer_idx: Optional[int]) -> bool:
    """Whether to fuse the shared expert into the routed FusedMoE at this layer.

    R5 mitigation: at SwigluStep layers the kernel clamps every expert at 7.0,
    but the shared expert may use a different clamp (e.g. 16 at layer 44 or 0
    at layer 43). Therefore the shared expert MUST stay on the dense path at
    every SwigluStep layer, even when the global aiter fusion is enabled.
    """
    # ATOM_FORCE_FUSE_SHARED=1 always fuses the shared expert into the
    # routed kernel (verification helper: bypass R5 mitigation).
    if os.environ.get("ATOM_FORCE_FUSE_SHARED"):
        return is_rocm_aiter_fusion_shared_expert_enabled()
    return (
        is_rocm_aiter_fusion_shared_expert_enabled()
        and not _uses_swiglustep_at_layer(config, layer_idx)
    )


# ---------------------------------------------------------------------------
# MLP (dense, used for first few layers and shared expert)
# ---------------------------------------------------------------------------


class Step3p5MLP(nn.Module):
    """Dense SwiGLU MLP with optional activation clamping."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        clamp_limit: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()
        # 0.0 means no clamping (disabled), only apply if > 0
        self.clamp_limit = (
            clamp_limit if (clamp_limit is not None and clamp_limit > 0) else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        if self.clamp_limit is not None:
            # Match HF: clamp AFTER silu activation on gate, symmetric on up
            # gate_proj output is first half, up_proj output is second half
            half = x.shape[-1] // 2
            gate, up = x[..., :half], x[..., half:]
            gate = torch.nn.functional.silu(gate).clamp(max=self.clamp_limit)
            up = up.clamp(min=-self.clamp_limit, max=self.clamp_limit)
            x = self.down_proj(gate * up)
        else:
            x = self.act_fn(x)
            x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE block (routed experts + shared expert)
# ---------------------------------------------------------------------------


class Step3p5MoE(nn.Module):
    """Sparse MoE block for Step-3.5.

    Checkpoint weight layout under ``layers.{i}.moe.*``:
      - gate.weight            (router linear)
      - router_bias            (learnable additive bias for sigmoid routing)
      - gate_proj.weight       (per-expert, shape [num_experts, intermediate, hidden])
      - up_proj.weight         (per-expert)
      - down_proj.weight       (per-expert)

    The FusedMoE kernel maps these via ``get_expert_mapping``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        num_experts: int = config.moe_num_experts
        top_k: int = config.moe_top_k
        moe_intermediate_size: int = config.moe_intermediate_size

        # Per-layer SwiGLU clamp limit for routed experts.
        # Step-3.5 applies clamp(x, -limit, limit) after gate_up_proj and
        # before the SwiGLU activation inside each expert.  The CK kernel
        # implements this as ``ActivationType.SwigluStep`` with a hard-coded
        # ±7 clamp; Step-3.5-Flash uses 7 at layers 43-44 only.
        layer_idx = extract_layer_index(prefix) if prefix else None
        self._layer_idx = layer_idx
        swiglu_limits = getattr(config, "swiglu_limits", None)
        self.clamp_limit = (
            swiglu_limits[layer_idx]
            if (
                swiglu_limits and layer_idx is not None and swiglu_limits[layer_idx] > 0
            )
            else None
        )
        self._uses_swiglustep = self.clamp_limit is not None
        self._activation = (
            ActivationType.SwigluStep if self._uses_swiglustep else ActivationType.Silu
        )

        # Router ---------------------------------------------------------
        self.gate = ReplicatedLinear(
            self.hidden_size,
            num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # Learnable router bias (added to sigmoid probs before top-k)
        self.router_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)
        self._need_fp32_gate = getattr(config, "need_fp32_gate", False)

        # Routed experts (fused MoE kernel) --------------------------------
        # R5 mitigation: at SwigluStep layers we MUST NOT fuse the shared
        # expert into the routed FusedMoE (the kernel hard-codes ±7 clamp,
        # but the shared expert uses a different limit, e.g. 16 at layer 44
        # or 0 at layer 43).  Fall back to the dense Step3p5MLP path there.
        self._fuse_shared = _fuse_shared_at_layer(config, layer_idx)
        n_shared = 1 if self._fuse_shared else 0
        self._n_shared_fused = (
            n_shared  # 1 when shared expert is fused as expert num_experts
        )
        self.experts = FusedMoE(
            num_experts=num_experts + n_shared,
            top_k=top_k + n_shared,  # +1 so kernel selects top_k routed + 1 shared
            hidden_size=self.hidden_size,
            intermediate_size=moe_intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=self._routing_function,
            config=config,
            activation=self._activation,
        )

    def _routing_function(
        self,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        hidden_states: Optional[torch.Tensor] = None,
    ):
        """Sigmoid routing with additive bias and scaling.

        ``hidden_states`` is accepted for compatibility with the framework's
        custom-routing calling convention (used by per-token hash routing in
        other models); Step-3.5 routing is computed from ``gating_output`` only
        and ignores it.

        When the shared expert is fused (self._n_shared_fused == 1), topk is
        top_k_routed + 1.  We select top_k_routed routed experts and append
        the shared expert (index num_routed_experts) with weight 1.0.
        """
        n_shared = self._n_shared_fused
        top_k_routed = topk - n_shared  # number of routed experts to pick

        gate_prob = torch.sigmoid(gating_output.float())
        gate_prob_biased = gate_prob + self.router_bias.unsqueeze(0)
        _, indices = torch.topk(gate_prob_biased, k=top_k_routed, dim=1)
        topk_prob = torch.gather(gate_prob, 1, indices)
        if renormalize:
            topk_prob = topk_prob / (topk_prob.sum(dim=-1, keepdim=True) + 1e-20)
        topk_prob = topk_prob * self.routed_scaling_factor

        if n_shared > 0:
            # Append shared expert (always selected, weight=1.0)
            T = gating_output.shape[0]
            num_routed = gating_output.shape[1]  # 288
            shared_ids = torch.full(
                (T, n_shared),
                num_routed,
                dtype=torch.int32,
                device=gating_output.device,
            )
            shared_weights = torch.ones(
                (T, n_shared), dtype=torch.float32, device=gating_output.device
            )
            topk_prob = torch.cat([topk_prob, shared_weights], dim=1)
            indices = torch.cat([indices.to(torch.int32), shared_ids], dim=1)
            return topk_prob, indices

        return topk_prob, indices.to(torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Router logits must be computed in fp32 (need_fp32_gate=True in config)
        if getattr(self, "_need_fp32_gate", True):
            router_logits = torch.nn.functional.linear(
                hidden_states.float(), self.gate.weight.float()
            )
        else:
            router_logits = self.gate(hidden_states)

        # Routed experts.  At SwigluStep layers (43-44) the FusedMoE was
        # constructed with ``activation=ActivationType.SwigluStep`` so the CK
        # kernel applies ``silu(g).clamp(max=7) * up.clamp(±7)`` per expert.
        routed_out = self.experts(hidden_states, router_logits)

        return routed_out.view(orig_shape)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Step3p5Attention(nn.Module):
    """GQA attention for Step-3.5.

    Key differences from vanilla LLaMA attention:
      - Per-layer rope_theta and partial_rotary_factor (from config lists).
      - Two attention head configurations depending on full vs sliding.
      - QK RMSNorm (zero-centered / GemmaRMSNorm style).
      - Head-wise attention gating via g_proj (sigmoid).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: str = "bf16",
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size

        # Determine layer type and head counts ----------------------------
        layer_types = getattr(config, "layer_types", [])
        is_sliding = (
            layer_types[layer_idx] == "sliding_attention" if layer_types else False
        )
        attn_other = getattr(config, "attention_other_setting", None)

        if is_sliding and attn_other is not None:
            self.total_num_heads = attn_other["num_attention_heads"]
            self.total_num_kv_heads = attn_other["num_attention_groups"]
        else:
            self.total_num_heads = config.num_attention_heads
            self.total_num_kv_heads = config.num_attention_groups

        self.head_dim = getattr(config, "head_dim", 128)

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # RoPE configuration -----------------------------------------------
        rope_theta_cfg = getattr(config, "rope_theta", 10000.0)
        if isinstance(rope_theta_cfg, list):
            rope_theta = rope_theta_cfg[layer_idx]
        else:
            rope_theta = rope_theta_cfg

        partial_rotary_factors = getattr(config, "partial_rotary_factors", None)
        if partial_rotary_factors is not None:
            partial_rotary_factor = partial_rotary_factors[layer_idx]
        else:
            partial_rotary_factor = 1.0

        rotary_dim = int(self.head_dim * partial_rotary_factor)

        max_position_embeddings = getattr(config, "max_position_embeddings", 262144)

        # Determine rope_scaling for this layer
        rope_scaling = getattr(config, "rope_scaling", None)
        yarn_only_types = getattr(config, "yarn_only_types", None)
        if yarn_only_types and layer_types:
            layer_type = layer_types[layer_idx]
            if layer_type not in yarn_only_types:
                rope_scaling = None

        # Projections -------------------------------------------------------
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # QK Norm (zero-centered RMSNorm) -----------------------------------
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
        self.q_norm = Step3p5RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = Step3p5RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Head-wise attention gate -------------------------------------------
        self.use_head_wise_attn_gate = getattr(config, "use_head_wise_attn_gate", False)
        if self.use_head_wise_attn_gate:
            self.g_proj = ColumnParallelLinear(
                input_size=self.hidden_size,
                output_size=self.total_num_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        # Rotary embedding ---------------------------------------------------
        # Note: rotary_dim is already computed as head_dim * partial_rotary_factor,
        # so we do NOT pass partial_rotary_factor to get_rope (which would apply it twice).
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        # Sliding window and sink tokens per layer ---------------------------
        sliding_window = None
        sinks = None
        if is_sliding:
            sliding_window = getattr(config, "sliding_window", None)
            sink_size = getattr(config, "sink", 0)
            if sink_size > 0:
                sinks = nn.Parameter(torch.empty(self.num_heads, requires_grad=False))

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=cache_config,
            layer_num=layer_num,
            per_layer_sliding_window=sliding_window,
            sinks=sinks,
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

        # QK Norm – apply per-head RMSNorm
        # Reshape to (..., num_heads, head_dim), apply norm, reshape back
        q = self.q_norm(q.reshape(*q.shape[:-1], -1, self.head_dim)).flatten(-2)
        k = self.k_norm(k.reshape(*k.shape[:-1], -1, self.head_dim)).flatten(-2)

        attn_output = self.attn(q, k, v, positions)

        # Head-wise gating
        if self.use_head_wise_attn_gate:
            gate = self.g_proj(hidden_states)  # (tokens, num_heads_tp)
            # gate: (tokens, num_heads_tp) -> (tokens, num_heads_tp, 1)
            gate = torch.sigmoid(gate).unsqueeze(-1)
            reshaped = attn_output.reshape(*attn_output.shape[:-1], -1, self.head_dim)
            attn_output = (reshaped * gate).flatten(-2)

        output = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class Step3p5DecoderLayer(nn.Module):
    """Single decoder layer for Step-3.5.

    - Layers 0-2: dense MLP
    - Layers 3-44: MoE (288 routed + 1 shared)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: str = "bf16",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_idx = extract_layer_index(prefix)

        # Attention
        self.self_attn = Step3p5Attention(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_num=layer_num,
        )

        # FFN: dense MLP or MoE depending on layer index
        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            if isinstance(moe_layers_enum, str):
                moe_layers_idx = [int(i) for i in moe_layers_enum.strip().split(",")]
            else:
                moe_layers_idx = list(moe_layers_enum)
        else:
            moe_layers_idx = list(range(3, config.num_hidden_layers))

        self.is_moe_layer = layer_idx in moe_layers_idx

        # Per-layer SwiGLU clamp limits
        swiglu_limits_shared = getattr(config, "swiglu_limits_shared", None)
        clamp_limit_shared = (
            swiglu_limits_shared[layer_idx] if swiglu_limits_shared else None
        )

        if self.is_moe_layer:
            self.moe = Step3p5MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.moe",
            )
            # Shared expert (always active, sibling of moe in checkpoint).
            # Per-layer fuse decision: SwigluStep layers must keep the shared
            # expert on the dense path because the routed CK kernel hard-codes
            # the clamp at 7 (see _fuse_shared_at_layer).
            if not _fuse_shared_at_layer(config, layer_idx):
                self.share_expert = Step3p5MLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.share_expert_dim,
                    quant_config=quant_config,
                    prefix=f"{prefix}.share_expert",
                    clamp_limit=clamp_limit_shared,
                )
            else:
                self.share_expert = None
        else:
            self.mlp = Step3p5MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                clamp_limit=clamp_limit_shared,  # HF uses swiglu_limits_shared for dense MLP
            )

        # Layer norms (zero-centered RMSNorm)
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
        self.input_layernorm = Step3p5RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Step3p5RMSNorm(
            config.hidden_size, eps=rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # FFN
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.is_moe_layer:
            moe_output = self.moe(hidden_states)
            if self.share_expert is not None:
                shared_output = self.share_expert(hidden_states)
                hidden_states = moe_output + shared_output
            else:
                hidden_states = moe_output
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


@support_torch_compile
class Step3p5Model(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        self.config = config
        cache_config = atom_config.kv_cache_dtype
        quant_config = atom_config.quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            getattr(config, "tie_word_embeddings", False)
            and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Step3p5DecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
        if get_pp_group().is_last_rank:
            self.norm = Step3p5RMSNorm(config.hidden_size, eps=rms_norm_eps)
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
    ) -> Union[torch.Tensor, IntermediateTensors]:
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


# ---------------------------------------------------------------------------
# CausalLM wrapper
# ---------------------------------------------------------------------------


class Step3p5ForCausalLM(nn.Module):
    """Step-3.5 model with language modelling head."""

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
        config = atom_config.hf_config
        self.config = config

        self.model = Step3p5Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", False):
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

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        """Step-3.5-Flash expert weights are flat: moe.gate_proj.weight [E, I, H].
        When shared expert fusion is enabled, share_expert weights are also loaded
        as expert N in FusedMoE; otherwise they are loaded as a regular MLP.

        Per-layer override: at SwigluStep layers (43-44) the shared expert is
        not fused, so its weights must take the dense path even when the
        global aiter fusion flag is on.
        """
        is_routed_expert = (
            ".moe.gate_proj" in weight_name
            or ".moe.up_proj" in weight_name
            or ".moe.down_proj" in weight_name
        )
        if is_routed_expert:
            return True
        is_share_expert = (
            ".share_expert.gate_proj" in weight_name
            or ".share_expert.up_proj" in weight_name
            or ".share_expert.down_proj" in weight_name
        )
        if is_share_expert:
            layer_idx = extract_layer_index(weight_name)
            return _fuse_shared_at_layer(self.config, layer_idx)
        return False

    def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
        """Mapping from flat checkpoint names to FusedMoE parameter names.

        Weight names include the '.weight' suffix from the checkpoint so that
        the replace() in loader.py produces the correct param name without the
        extra '.weight' tail (e.g. 'moe.gate_proj.weight' -> 'moe.experts.w13_weight').
        """
        mapping = [
            ("moe.experts.w13_weight", "moe.gate_proj.weight", "w1"),
            ("moe.experts.w13_weight", "moe.up_proj.weight", "w3"),
            ("moe.experts.w2_weight", "moe.down_proj.weight", "w2"),
        ]
        if is_rocm_aiter_fusion_shared_expert_enabled():
            mapping += [
                ("moe.experts.w13_weight", "share_expert.gate_proj.weight", "w1"),
                ("moe.experts.w13_weight", "share_expert.up_proj.weight", "w3"),
                ("moe.experts.w2_weight", "share_expert.down_proj.weight", "w2"),
            ]
        return mapping

    def load_fused_expert_weights(
        self,
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        """Load flat expert weights [E, I, H] into FusedMoE per-expert params.

        For routed experts: loaded_weight is [num_experts, ...], loaded per-expert.
        For shared expert: loaded_weight is [I, H] or [H, I], loaded as expert num_experts.
        """
        # num_experts from loader may be 0 if hf_config uses non-standard attr name
        if num_experts == 0:
            num_experts = self.config.moe_num_experts

        if name not in params_dict:
            return False
        param = params_dict[name]
        weight_loader = param.weight_loader
        loaded_local_expert = False

        is_share_expert = "share_expert" in original_name

        if is_share_expert:
            # Defensive: if this layer keeps the shared expert dense (e.g.
            # SwigluStep layers 43-44), do not route it through FusedMoE.
            layer_idx = extract_layer_index(original_name)
            if not _fuse_shared_at_layer(self.config, layer_idx):
                return False
            # Shared expert is loaded as expert index num_experts (288)
            expert_id = num_experts
            try:
                success = weight_loader(
                    param,
                    loaded_weight,
                    name,
                    shard_id,
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_local_expert = True
            except TypeError:
                weight_loader(param, loaded_weight, name, shard_id, expert_id)
                loaded_local_expert = True
        else:
            for expert_id in range(num_experts):
                try:
                    success = weight_loader(
                        param,
                        loaded_weight[expert_id],
                        name,
                        shard_id,
                        expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_local_expert = True
                except TypeError:
                    weight_loader(
                        param, loaded_weight[expert_id], name, shard_id, expert_id
                    )
                    loaded_local_expert = True

        return loaded_local_expert

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Return the expert parameter mapping for weight loading.

        Note: Step-3.5-Flash uses flat expert weights in the checkpoint
        (moe.gate_proj.weight etc.), so get_expert_mapping is used only
        as a sentinel to enable the expert loading path in loader.py.
        The actual loading is handled by load_fused_expert_weights.
        """
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.moe_num_experts
            + (1 if is_rocm_aiter_fusion_shared_expert_enabled() else 0),
        )
