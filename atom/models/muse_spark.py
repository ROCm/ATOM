# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark Open, served text-only (``MuseSparkForConditionalGeneration``).

The text stack differs from a plain MoE decoder in five places, all written
out in eager torch here (accuracy first, no fusions yet):

* The residual stream is fp32, and each sub-block joins it through a
  norm-preserving gated residual (``alpha * stream + beta * branch``) after a
  post-norm of its output.
* Attention projects an extra query-sized output gate; q and k get a
  weightless per-head RMSNorm (q then times ``qk_scale_factor``), the output a
  weightless per-head RMSNorm times ``sigmoid(gate)``. Layers whose
  ``layer_rope_theta`` is 0 are NoPE; ``layer_types`` marks the
  sliding-window layers.
* The MoE runs in a 4096-wide space between a replicated ``pre_expert_proj``
  and ``post_expert_proj``, routes with a scaled sigmoid plus a selection-only
  bias, and RMS-normalizes every expert's output *before* the weighted top-k
  sum. The experts therefore run as a top-1 FusedMoE over tokens repeated
  ``top_k`` times, which returns one row per (token, slot).
* Every norm weight is zero-centred: the effective weight is ``w + 1`` (the
  final norm: ``w + 0``), rounded to the weight dtype (bf16).
* Logits are an fp32 lm_head times ``output_multiplier``, then a tanh softcap.
"""

from typing import ClassVar

import torch
import torch.nn.functional as F
from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from aiter.dist.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.config import Config
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.utils import atom_parameter
from atom.models.muse_spark_math import (
    muse_logits,
    muse_rms,
    muse_route,
    post_expert_combine,
    residual_gate_coeffs,
)
from atom.models.utils import IntermediateTensors, make_layers, maybe_prefix
from atom.utils import envs
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context


def _slot_routing(hidden_states, gating_output, topk, renormalize):
    """FusedMoE routing for pre-expanded slots: the ``[rows, 1]`` "logits" are
    the expert ids themselves, each slot taken with weight 1."""
    del hidden_states, topk, renormalize
    ids = gating_output.to(torch.int32)
    return torch.ones_like(gating_output, dtype=torch.float32), ids


def _slot_moe_unsupported(experts, atom_config) -> str | None:
    """Why ``experts`` cannot run the pre-expanded top-1 slots, or None.

    These FusedMoE paths cannot take ``[tokens * top_k]`` slot rows
    with preselected expert ids:
    - the Triton TP path routes from ``router_logits`` and ignores
      ``custom_routing_function`` (moe.py ``Mxfp4MoEMethod.apply``), so it would
      read the expert ids as logits and pick expert 0 for every slot; the
      Triton EP path clamps the SiLU gate at ``swiglu_limit`` (default 10);
    - with DP attention the MoE gathers rows across DP ranks against the
      forward context's token count, which the slots exceed ``top_k`` times
      (the gather/scatter path asserts; the all2all path is not validated with
      expanded slots);
    - DP > 1 with EP selects the all2all dispatch/combine, which is sized for
      one row per token;
    - ``--moe-backend mega`` is sized for one row per token and does its own
      weighted combine;
    - ``--fake-eplb`` replaces the router logits (here the expert ids) with a
      balance table;
    - prefill context parallel shards the experts over the TP x PCP ranks but
      the slots are reduced over the TP group only, and decode context parallel
      reaches MHA metadata that indexes the DCP-local block tables by global
      positions.
    """
    if getattr(experts.quant_method, "use_triton", False) or getattr(
        experts.quant_method, "use_triton_ep", False
    ):
        return (
            "Muse Spark routes preselected experts over top_k-expanded slots, "
            "which the Triton MoE paths do not support (TP routes from logits, "
            "EP clamps the SiLU gate). Set ATOM_USE_TRITON_MOE=0."
        )
    if atom_config.enable_dp_attention and experts.dp_size > 1:
        return (
            "Muse Spark does not support DP attention yet: its MoE runs "
            "top_k-expanded slots, and the DP MoE communication expects one row "
            "per token. Run without DP attention (TP or TP + EP)."
        )
    if experts.dp_size > 1 and experts.use_ep:
        return (
            "Muse Spark does not support DP with expert parallel yet: the "
            "all2all MoE dispatch expects one row per token. Run TP + EP "
            "without DP, or DP without expert parallel."
        )
    if atom_config.moe_backend == "mega":
        return (
            "Muse Spark does not support --moe-backend mega: it expects one row "
            "per token and combines the experts itself. Use --moe-backend "
            "standard."
        )
    if atom_config.fake_eplb:
        return (
            "Muse Spark does not support --fake-eplb: it replaces the "
            "preselected expert ids with a balance table. Run without it."
        )
    if (
        atom_config.prefill_context_parallel_size > 1
        or atom_config.decode_context_parallel_size > 1
    ):
        return (
            "Muse Spark does not support prefill or decode context parallel "
            "yet: its expert slots are reduced over the TP group only, and its "
            "MHA path does not index DCP-local block tables. Set -pcp 1 and "
            "-dcp 1."
        )
    return None


# ── modules ────────────────────────────────────────────────────────────────


class MuseRMSNorm(nn.Module):
    """Zero-centred RMSNorm: effective weight ``bf16(w + gain_center)``.

    ``input_scale`` selects the post-norm variant, which scales the input by the
    weight and then normalizes without one.
    """

    def __init__(
        self,
        dim: int,
        eps: float,
        gain_center: float = 1.0,
        input_scale: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.gain_center = gain_center
        self.input_scale = input_scale
        self.weight = atom_parameter(torch.zeros(dim, dtype=torch.bfloat16))

    def process_weights_after_loading(self) -> None:
        # The weight is constant after loading: fold the gain centre in once.
        effective = (self.weight.float() + self.gain_center).to(self.weight.dtype)
        self.register_buffer("effective_weight", effective, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fp32; callers cast to the dtype the next op wants."""
        if self.input_scale:
            return muse_rms(
                self.effective_weight * x.to(self.weight.dtype), None, self.eps
            )
        return muse_rms(x, self.effective_weight, self.eps)


class MuseResidualGate(nn.Module):
    def __init__(self, dim: int, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.gate = atom_parameter(torch.zeros(dim, dtype=torch.bfloat16))

    def process_weights_after_loading(self) -> None:
        # The gate is constant after loading: compute alpha and beta once.
        alpha, beta = residual_gate_coeffs(self.gate, self.temperature)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("beta", beta, persistent=False)

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return self.alpha * stream + self.beta * branch.float()


class MuseRouter(nn.Module):
    """fp32 router weight ``[experts, hidden]`` plus the selection bias."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = atom_parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32)
        )
        self.e_score_correction_bias = atom_parameter(
            torch.zeros(num_experts, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight)


class MuseSparkAttention(nn.Module):
    def __init__(self, atom_config: Config, layer_num: int, prefix: str):
        super().__init__()
        config = atom_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.norm_eps = config.rms_norm_eps
        self.qk_scale = config.qk_scale_factor

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        theta = config.layer_rope_theta[layer_num]
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=atom_config.max_model_len,
                base=theta,
                is_neox_style=True,
            )
            if theta
            else None
        )
        sliding = config.layer_types[layer_num] == "sliding_attention"
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            layer_num=layer_num,
            per_layer_sliding_window=config.sliding_window if sliding else None,
            rotary_emb=None,
            prefix=f"{prefix}.attn",
        )

    def _head_rms(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return muse_rms(x.reshape(-1, self.head_dim), None, self.norm_eps).reshape(
            shape
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate = self.gate_proj(x)
        dtype = x.dtype
        q = (self._head_rms(q).to(dtype) * self.qk_scale).contiguous()
        k = self._head_rms(k).to(dtype).contiguous()
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v.contiguous(), positions)
        o = self._head_rms(o).to(dtype) * torch.sigmoid(gate)
        return self.o_proj(o)


class MuseSparkMoE(nn.Module):
    def __init__(self, atom_config: Config, prefix: str):
        super().__init__()
        config = atom_config.hf_config
        if config.num_shared_experts:
            raise ValueError("Muse Spark has routed experts only")
        self.top_k = config.num_experts_per_tok
        self.multiplier = config.output_multiplier
        self.route_scale = config.routed_scaling_factor
        self.expert_dim = config.moe_hidden_size
        self.post_norm_eps = config.post_norm_eps

        self.gate = MuseRouter(config.hidden_size, config.num_local_experts)
        self.pre_expert_proj = ReplicatedLinear(
            config.hidden_size,
            self.expert_dim,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.pre_expert_proj",
        )
        self.pre_expert_norm = MuseRMSNorm(self.expert_dim, config.rms_norm_eps)
        # The checkpoint keeps this under experts.; weights_mapping moves it here.
        self.post_expert_norm = MuseRMSNorm(
            self.expert_dim, config.post_norm_eps, input_scale=True
        )
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=1,
            hidden_size=self.expert_dim,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.experts",
            custom_routing_function=_slot_routing,
            config=config,
        )
        unsupported = _slot_moe_unsupported(self.experts, atom_config)
        if unsupported:
            raise NotImplementedError(unsupported)
        self.post_expert_proj = ReplicatedLinear(
            self.expert_dim,
            config.hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=f"{prefix}.post_expert_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[0]
        scores, ids = muse_route(
            self.gate(x),
            self.gate.e_score_correction_bias,
            self.top_k,
            self.multiplier,
            self.route_scale,
        )
        h = self.pre_expert_norm(self.pre_expert_proj(x)).to(x.dtype)
        slots = h.repeat_interleave(self.top_k, dim=0)
        y = self.experts(slots, ids.reshape(-1, 1).float())
        # A TP-sharded expert returns a partial sum: complete it before the
        # per-slot norm. Under EP each slot is complete on its owning rank and
        # zero elsewhere, so the sum is reduced after the combine instead.
        if self.experts.tp_size > 1:
            y = tensor_model_parallel_all_reduce(y)
        out = post_expert_combine(
            y.view(tokens, self.top_k, self.expert_dim),
            scores,
            self.post_expert_norm.effective_weight,
            self.post_norm_eps,
        )
        if self.experts.tp_size == 1 and self.experts.ep_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return self.post_expert_proj(out)


class MuseSparkDecoderLayer(nn.Module):
    def __init__(self, atom_config: Config, prefix: str, layer_num: int):
        super().__init__()
        config = atom_config.hf_config
        eps, post_eps = config.rms_norm_eps, config.post_norm_eps
        temperature = config.residual_gate_temperature
        if config.mlp_layer_types[layer_num] != "sparse":
            raise ValueError("Muse Spark Open is MoE in every layer")
        self.input_layernorm = MuseRMSNorm(config.hidden_size, eps)
        self.self_attn = MuseSparkAttention(
            atom_config, layer_num, prefix=f"{prefix}.self_attn"
        )
        self.post_attn_eps = post_eps
        self.post_attention_residual_gate = MuseResidualGate(
            config.hidden_size, temperature
        )
        self.pre_feedforward_layernorm = MuseRMSNorm(config.hidden_size, eps)
        self.mlp = MuseSparkMoE(atom_config, prefix=f"{prefix}.mlp")
        self.post_feedforward_layernorm = MuseRMSNorm(
            config.hidden_size, post_eps, input_scale=True
        )
        self.post_feedforward_residual_gate = MuseResidualGate(
            config.hidden_size, temperature
        )

    def forward(self, positions: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        attn_in = self.input_layernorm(stream).to(torch.bfloat16)
        attn_out = self.self_attn(positions, attn_in)
        # post_attn_norm is weightless and leaves the branch in its dtype.
        attn_out = muse_rms(attn_out, None, self.post_attn_eps).to(attn_out.dtype)
        stream = self.post_attention_residual_gate(stream, attn_out)
        ffn_in = self.pre_feedforward_layernorm(stream).to(torch.bfloat16)
        ffn_out = self.mlp(ffn_in)
        ffn_out = self.post_feedforward_layernorm(ffn_out).to(ffn_out.dtype)
        return self.post_feedforward_residual_gate(stream, ffn_out)


@support_torch_compile
class MuseSparkTextModel(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = atom_config.hf_config
        if get_pp_group().world_size > 1:
            raise NotImplementedError("Muse Spark does not support pipeline parallel")
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: MuseSparkDecoderLayer(
                atom_config, prefix=prefix, layer_num=layer_num
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = MuseRMSNorm(
            config.hidden_size, config.rms_norm_eps, gain_center=0.0
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            h = self.get_input_embeddings(input_ids)
            # embed_norm: weightless, in the embedding dtype.
            h = muse_rms(h, None, self.config.rms_norm_eps).to(h.dtype)
        else:
            h = inputs_embeds
        stream = h.float()
        for layer in self.layers[self.start_layer : self.end_layer]:
            stream = layer(positions, stream)
        return self.norm(stream.to(torch.bfloat16)).to(torch.bfloat16)


class MuseSparkLMHead(ParallelLMHead):
    """fp32 logits (the reference accumulates and softcaps in fp32)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        context = forward_context.context
        if context.is_prefill and not context.is_draft:
            last_indices = forward_context.attn_metadata.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = torch.mm(x, self.weight.t(), out_dtype=torch.float32)
        if self.tp_size > 1:
            logits = tensor_model_parallel_all_gather(
                logits, use_custom=envs.ATOM_USE_CUSTOM_ALL_GATHER
            )
        return logits


class _LanguageModelHolder(nn.Module):
    """Gives the text stack the checkpoint's ``model.language_model.*`` names."""

    def __init__(self, atom_config: Config, prefix: str):
        super().__init__()
        self.language_model = MuseSparkTextModel(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "language_model")
        )


class MuseSparkForConditionalGeneration(nn.Module):
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str]]] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    weights_mapping: ClassVar[dict[str, str]] = {
        ".mlp.experts.post_expert_norm.": ".mlp.post_expert_norm."
    }
    skip_weight_prefixes: ClassVar[list[str]] = [
        "model.vision_tower.",
        "model.vision_adapter.",
        "model.vision_projection.",
        "model.audio_tower.",
    ]

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.atom_config = atom_config
        self.config = atom_config.hf_config
        if self.config.tie_word_embeddings:
            raise NotImplementedError("Muse Spark Open ships an untied lm_head")
        self.model = _LanguageModelHolder(atom_config, maybe_prefix(prefix, "model"))
        self.lm_head = MuseSparkLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    @property
    def language_model(self) -> MuseSparkTextModel:
        return self.model.language_model

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return muse_logits(
            self.lm_head(hidden_states),
            self.config.output_multiplier,
            self.config.final_logit_softcapping,
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_local_experts,
        )
