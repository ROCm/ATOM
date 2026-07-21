# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Kimi-K3 text model.

The checkpoint is multimodal, but ATOM serves the text path here.  The language
weights live under ``language_model.*`` in the checkpoint, so this module keeps
the same object hierarchy and skips the vision tower/projector tensors.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch
import torch.nn.functional as F
from aiter import ActivationType, QuantType
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from einops import rearrange
from torch import nn

from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_ops.base_attention import Attention

# Side-effect import: registers `torch.ops.aiter.maybe_dual_stream_forward`,
# the Dynamo-opaque dispatcher used to overlap shared experts with routed
# experts on a separate CUDA stream (shared with DeepSeek V2/V4).
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.mamba_ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from atom.model_ops.attention_mha import _torch_reshape_and_cache
from atom.model_ops.moe import FusedMoE
from atom.model_ops.utils import atom_parameter
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils import envs
from atom.utils.forward_context import get_forward_context


def _text_config(config):
    return getattr(config, "text_config", config)


def _normalize_kimi_config(config) -> None:
    """Fill the aliases expected by shared ATOM MoE/GDN infrastructure."""

    config.n_routed_experts = getattr(config, "n_routed_experts", config.num_experts)
    config.num_experts_per_tok = getattr(
        config, "num_experts_per_tok", config.num_experts_per_token
    )
    config.n_shared_experts = getattr(
        config, "n_shared_experts", getattr(config, "num_shared_experts", 0)
    )
    config.norm_topk_prob = getattr(
        config, "norm_topk_prob", getattr(config, "moe_renormalize", True)
    )
    config.scoring_func = getattr(
        config, "scoring_func", getattr(config, "moe_router_activation_func", "sigmoid")
    )
    config.n_group = getattr(config, "n_group", getattr(config, "num_expert_group", 1))

    lin = getattr(config, "linear_attn_config", {}) or {}
    config.linear_num_key_heads = getattr(
        config, "linear_num_key_heads", lin.get("num_heads", config.num_attention_heads)
    )
    config.linear_num_value_heads = getattr(
        config, "linear_num_value_heads", lin.get("num_heads", config.num_attention_heads)
    )
    config.linear_key_head_dim = getattr(
        config, "linear_key_head_dim", lin.get("head_dim", config.qk_nope_head_dim)
    )
    config.linear_value_head_dim = getattr(
        config, "linear_value_head_dim", lin.get("head_dim", config.v_head_dim)
    )
    config.linear_conv_kernel_dim = getattr(
        config, "linear_conv_kernel_dim", lin.get("short_conv_kernel_size", 4)
    )
    config.kimi_full_attn_layers = [
        int(i) - 1 for i in lin.get("full_attn_layers", [])
    ]
    config.kimi_kda_layers = [int(i) - 1 for i in lin.get("kda_layers", [])]
    config.num_gdn_attn_state = len(config.kimi_kda_layers)
    config.num_full_attn = len(config.kimi_full_attn_layers)

    # Kimi full-attention layers run MLA math but are stored in the standard
    # paged-MHA cache by padding V to q_head_dim.
    config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {
            "rope_theta": getattr(config, "rope_theta", 10000.0),
            "rope_type": "default",
        }


def _extract_layer_idx(prefix: str) -> int:
    for part in reversed(prefix.split(".")):
        if part.isdigit():
            return int(part)
    return 0


class SituAndMul(nn.Module):
    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if os.environ.get("ATOM_K3_FUSED", "1") == "1":
            from atom.models.kimi_k3_fused import situ_and_mul

            return situ_and_mul(x, self.beta, self.linear_beta)
        gate, up = x.chunk(2, dim=-1)
        gate_f = gate.float()
        up_f = up.float()
        out = self.beta * torch.tanh(gate_f / self.beta) * torch.sigmoid(gate_f)
        if self.linear_beta is not None:
            up_f = self.linear_beta * torch.tanh(up_f / self.linear_beta)
        return (out * up_f).to(x.dtype)


class KimiRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = atom_parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if os.environ.get("ATOM_K3_FUSED", "1") == "1":
            from atom.models.kimi_k3_fused import rmsnorm_gated

            return rmsnorm_gated(x, self.weight, gate, self.variance_epsilon)
        dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x = x_f * torch.rsqrt(var + self.variance_epsilon)
        return (x.to(dtype) * self.weight.to(dtype)) * torch.sigmoid(gate)


def _sharded_vector_loader(tp_rank: int, tp_size: int):
    def loader(param: nn.Parameter, loaded_weight: torch.Tensor):
        shard = loaded_weight.narrow(0, tp_rank * param.numel(), param.numel())
        param.data.copy_(shard.to(param.dtype).view_as(param))

    return loader


class KimiMLP(nn.Module):
    def __init__(
        self,
        config,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        hidden_size = hidden_size or config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
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
        if config.hidden_act != "situ":
            raise ValueError(f"Unsupported Kimi-K3 activation: {config.hidden_act}")
        self.act_fn = SituAndMul(
            beta=getattr(config, "activation_situ_beta", None) or 1.0,
            linear_beta=getattr(config, "activation_situ_linear_beta", None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class KimiSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: Optional["torch.cuda.Stream"] = None,
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.alt_stream = alt_stream
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size if self.use_latent_moe else config.hidden_size
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = atom_parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.moe_renormalize,
            quant_config=quant_config,
            use_grouped_topk=getattr(config, "use_grouped_topk", True),
            num_expert_group=getattr(config, "num_expert_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            scoring_func=config.moe_router_activation_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation=ActivationType.Situv2,
            config=config,
            prefix=f"{prefix}.experts",
        )
        if getattr(config, "num_shared_experts", 0):
            self.shared_experts = KimiMLP(
                config,
                intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        if self.use_latent_moe:
            def _routed_source_quant_dtype(layer_prefix: str) -> torch.dtype | None:
                if quant_config is None:
                    return None
                layer_quant_config = quant_config.get_layer_quant_config(layer_prefix)
                if (
                    layer_quant_config.quant_type == QuantType.per_1x32
                    and layer_quant_config.quant_dtype
                    == getattr(torch, "float4_e2m1fn_x2", None)
                ):
                    return torch.bfloat16
                return None

            down_proj_prefix = f"{prefix}.routed_expert_down_proj"
            up_proj_prefix = f"{prefix}.routed_expert_up_proj"
            self.routed_expert_down_proj = ReplicatedLinear(
                config.hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=quant_config,
                source_quant_dtype=_routed_source_quant_dtype(down_proj_prefix),
                prefix=down_proj_prefix,
            )
            self.routed_expert_up_proj = ReplicatedLinear(
                self.moe_hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                source_quant_dtype=_routed_source_quant_dtype(up_proj_prefix),
                prefix=up_proj_prefix,
            )
            self.routed_expert_norm = (
                RMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)
                if getattr(config, "latent_moe_use_norm", False)
                else None
            )

        # Shared-expert overlap (dual-stream): run the shared-expert MLP on
        # `alt_stream` concurrently with the routed experts. Only meaningful
        # when there ARE shared experts and the model handed us an alt_stream
        # (created iff ATOM_K3_SHARED_EXPERT_OVERLAP is set). Per-call
        # decode/prefill gating happens in the maybe_dual_stream_forward
        # dispatcher (ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD).
        self._use_dual_stream = (
            self.shared_experts is not None and self.alt_stream is not None
        )
        if self._use_dual_stream:
            # Register so the maybe_dual_stream_forward custom op can look us
            # up by layer_name (== self.prefix) without putting sub-modules /
            # per-fwd state in the op signature.
            get_current_atom_config().compilation_config.static_forward_context[
                prefix
            ] = self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_dual_stream:
            # Custom op = Dynamo barrier, so the alt_stream context inside
            # dual_stream_moe_forward stays opaque to torch.compile. The
            # dispatcher picks dual vs single per call (num_tokens vs
            # ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD; forces single under
            # piecewise cudagraph / TBO).
            return torch.ops.aiter.maybe_dual_stream_forward(hidden_states, self.prefix)
        return self.single_stream_moe_forward(hidden_states)

    def single_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # gfx1250 work-around (REQUIRED for correctness, not just stability): the
        # grouped MoE + MXFP4 latent projections are only numerically correct at
        # small M on gfx1250 — at large prefill M the contiguous-M path OOB-faults
        # AND the non-contiguous path returns coherent-but-wrong values (gsm8k
        # 0.0). The MoE block is per-token independent, so split a large prefill
        # into <=chunk sub-batches (numerically identical) so every kernel sees a
        # small, correct M. Keeps whole-prompt prefill (correct MLA). Gated by
        # ATOM_K3_MOE_CHUNK; remove once the gfx1250 MoE kernel is fixed at large M.
        chunk = int(os.environ.get("ATOM_K3_MOE_CHUNK", "0") or "0")
        n = hidden_states.shape[0]
        if chunk > 0 and n > chunk:
            outs = [
                self._forward_impl(hidden_states[i : i + chunk])
                for i in range(0, n, chunk)
            ]
            return torch.cat(outs, dim=0)
        return self._forward_impl(hidden_states)

    def dual_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Overlap the shared-expert MLP with the routed experts.

        The shared branch (``shared_experts``, a plain MLP producing a TP-partial
        output — no collective inside) runs on ``alt_stream`` while the routed
        experts (+ latent all-reduce / norm / up_proj) run on the main stream.
        All collectives stay on the main stream and keep their original order, so
        this is numerically identical to ``_forward_impl`` — only the shared
        GEMMs move off the critical path. Used for small (decode) batches only,
        so the single-stream chunking work-around is unnecessary here.
        """
        identity = hidden_states
        current_stream = get_forward_context().main_stream

        # Fork shared-expert compute onto alt_stream (waits for the main stream
        # to have produced `hidden_states` first).
        self.alt_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.alt_stream):
            shared_partial = self.shared_experts(identity)

        # Routed experts on the main stream.
        router_logits = F.linear(hidden_states.float(), self.gate.weight.float(), None)
        routed_input = (
            self.routed_expert_down_proj(hidden_states)
            if self.use_latent_moe
            else hidden_states
        )
        routed_output = self.experts(routed_input, router_logits)

        if self.use_latent_moe:
            if self.tp_size > 1:
                routed_output = tensor_model_parallel_all_reduce(routed_output)
            if self.routed_expert_norm is not None:
                routed_output = self.routed_expert_norm(routed_output)
            routed_output = self.routed_expert_up_proj(routed_output)
            # Join alt_stream, then all-reduce the shared partial on the main
            # stream (ordered after the routed all-reduce above).
            current_stream.wait_stream(self.alt_stream)
            if self.tp_size > 1:
                shared_partial = tensor_model_parallel_all_reduce(shared_partial)
            return routed_output + shared_partial

        # Non-latent: routed + shared are both TP-partial; a single deferred
        # all-reduce over their sum is correct.
        current_stream.wait_stream(self.alt_stream)
        routed_output = routed_output + shared_partial
        if self.tp_size > 1:
            routed_output = tensor_model_parallel_all_reduce(routed_output)
        return routed_output

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        # Match the reference Kimi router: route in fp32 before sigmoid/top-k.
        router_logits = F.linear(
            hidden_states.float(),
            self.gate.weight.float(),
            None,
        )
        routed_input = (
            self.routed_expert_down_proj(hidden_states)
            if self.use_latent_moe
            else hidden_states
        )
        routed_output = self.experts(routed_input, router_logits)
        if self.use_latent_moe:
            # self.experts runs with reduce_results=False, so routed_output is a
            # TP-partial sum over the sharded expert intermediate. routed_expert_norm
            # is a (nonlinear) RMSNorm, so it must operate on the FULL sum:
            # sum_r norm(partial_r) != norm(sum_r partial_r). All-reduce here first;
            # routed_expert_norm/up_proj are replicated, so the result stays full.
            if self.tp_size > 1:
                routed_output = tensor_model_parallel_all_reduce(routed_output)
            if self.routed_expert_norm is not None:
                routed_output = self.routed_expert_norm(routed_output)
            routed_output = self.routed_expert_up_proj(routed_output)
            if self.shared_experts is not None:
                # Shared branch is TP-partial (down_proj is row-parallel); reduce
                # it separately and add to the already-full routed output.
                shared_output = self.shared_experts(identity)
                if self.tp_size > 1:
                    shared_output = tensor_model_parallel_all_reduce(shared_output)
                routed_output = routed_output + shared_output
            return routed_output
        # Non-latent path: routed experts and shared experts are both TP-partial
        # and everything after them is linear, so a single deferred all-reduce
        # over their sum is correct.
        if self.shared_experts is not None:
            routed_output = routed_output + self.shared_experts(identity)
        if self.tp_size > 1:
            routed_output = tensor_model_parallel_all_reduce(routed_output)
        return routed_output


class KimiFullAttention(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.scaling = self.q_head_dim**-0.5
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // self.tp_size
        self.local_q_size = self.num_local_heads * self.q_head_dim
        self.local_v_size = self.num_local_heads * self.v_head_dim

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=1e-6)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.layer_num = _extract_layer_idx(prefix)
        self.attn = Attention(
            self.num_local_heads,
            self.q_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            quant_config=quant_config,
            use_mla=False,
            layer_num=self.layer_num,
            config=atom_config,
            prefix=prefix,
        )

    def _prefill_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sequence SDPA over the in-batch Q/K/V (unpadded V).

        The paged/unified attention prefill kernel faults on multi-sequence
        varlen batches at head_dim=192 (Kimi MLA-as-MHA), so run SDPA per
        sequence instead. The paged KV cache is populated separately via
        ``_write_kv_cache`` so decode can read the prompt K/V.
        """
        fwd_ctx = get_forward_context()
        cu_seqlens = getattr(fwd_ctx.attn_metadata, "cu_seqlens_q", None)
        out = torch.empty_like(v)
        if cu_seqlens is None:
            spans = [(0, q.shape[0])]
        else:
            starts = cu_seqlens[:-1].tolist()
            ends = cu_seqlens[1:].tolist()
            spans = list(zip(starts, ends))
        for start, end in spans:
            if end <= start:
                continue
            out_seq = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=True,
                scale=self.scaling,
            )
            out[start:end].copy_(out_seq.squeeze(0).transpose(0, 1))
        return out.reshape(-1, self.local_v_size)

    def _write_kv_cache(self, k: torch.Tensor, v_padded: torch.Tensor) -> None:
        """Write prefill K/V into the paged KV cache in the same layout the
        decode path (self.attn -> rope_cache -> reshape_and_cache) reads."""
        import aiter

        fwd_ctx = get_forward_context()
        # Dummy warmup runs have no populated KV cache / slot mapping.
        if fwd_ctx.context.is_dummy_run or fwd_ctx.kv_cache_data is None:
            return
        attn_metadata = fwd_ctx.attn_metadata
        cache = fwd_ctx.kv_cache_data[f"layer_{self.layer_num}"]
        k_cache, v_cache = cache.k_cache, cache.v_cache
        k_r = k.view(-1, self.num_local_heads, self.q_head_dim)
        v_r = v_padded.view(-1, self.num_local_heads, self.q_head_dim)
        # The full-attn layers are bound in the 4D FLASH layout (see
        # aiter_attention.build_kv_cache_tensor): [num_blocks, block_size,
        # num_kv_heads, head_dim]. Write prefill K/V with reshape_and_cache_flash
        # so it matches what decode's unified_attention(shuffled_kv_cache=False)
        # reads. head_dim=192 mis-indexes under the 5D SHUFFLE read, hence flash.
        if v_cache.dim() == 4 and envs.ATOM_USE_TORCH_CACHE:
            _torch_reshape_and_cache(
                k_r, v_r, k_cache, v_cache, attn_metadata.slot_mapping
            )
        elif v_cache.dim() == 4:
            flash_scale = self.attn.impl._pa_decode_bf16_asm_scale
            aiter.reshape_and_cache_flash(
                k_r,
                v_r,
                k_cache,
                v_cache,
                attn_metadata.slot_mapping,
                "auto",
                flash_scale,
                flash_scale,
            )
        else:
            # Legacy SHUFFLE fallback (asm layout when V cache is 5-D).
            asm_layout = not (v_cache.dim() != 5)
            aiter.reshape_and_cache(
                k_r,
                v_r,
                k_cache,
                v_cache,
                attn_metadata.slot_mapping,
                kv_cache_dtype="auto",
                k_scale=None,
                v_scale=None,
                asm_layout=asm_layout,
            )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(-1, self.num_local_heads, self.q_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_latent, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(k_latent))
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_local_heads, -1)

        q = torch.cat((q_nope, q_rope), dim=-1)
        k = torch.cat((k_nope, k_rope), dim=-1)
        # Kimi MLA is stored in the standard paged-MHA cache: pad V to the query
        # head dim so K and V share a cache entry, then slice V back afterwards.
        v_padded = F.pad(v, (0, self.q_head_dim - self.v_head_dim)).reshape(
            -1, self.local_q_size
        )

        fwd_ctx = get_forward_context()
        if fwd_ctx.context.is_prefill:
            # SDPA compute (handles multi-sequence varlen at head_dim=192) plus an
            # explicit cache write so decode can read the prompt K/V. Routing
            # prefill through self.attn instead faults on batched varlen prefill.
            attn_out = self._prefill_sdpa(q, k, v)
            self._write_kv_cache(
                k.reshape(-1, self.local_q_size), v_padded
            )
        else:
            attn_out = self.attn(
                q.reshape(-1, self.local_q_size),
                k.reshape(-1, self.local_q_size),
                v_padded,
            )
            attn_out = attn_out.view(-1, self.num_local_heads, self.q_head_dim)[
                :, :, : self.v_head_dim
            ].reshape(-1, self.local_v_size)
        attn_out = attn_out * torch.sigmoid(self.g_proj(hidden_states))
        return self.o_proj(attn_out)


class KimiKDAAttention(nn.Module):
    @property
    def mamba_type(self) -> str:
        return "kimi_kda"

    def __init__(
        self,
        atom_config: Config,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_key_heads
        self.head_dim = config.linear_key_head_dim
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_local_heads = self.num_heads // self.tp_size
        self.proj_size = self.num_heads * self.head_dim
        self.local_proj_size = self.num_local_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.prefix = prefix
        self.layer_num = _extract_layer_idx(prefix)
        self.activation = "silu"
        self.base_linear_attention = True

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.q_conv1d = ColumnParallelLinear(
            self.conv_kernel_size,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            self.conv_kernel_size,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            self.conv_kernel_size,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_conv1d",
        )
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            conv.weight.data = conv.weight.data.unsqueeze(1)

        self.A_log = atom_parameter(torch.empty(self.num_local_heads))
        self.dt_bias = atom_parameter(torch.empty(self.local_proj_size))
        loader = _sharded_vector_loader(self.tp_rank, self.tp_size)
        self.A_log.weight_loader = loader
        self.dt_bias.weight_loader = loader

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.proj_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_norm = KimiRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.proj_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def _conv_weights(self) -> torch.Tensor:
        return torch.cat(
            [
                self.q_conv1d.weight.view(self.local_proj_size, self.conv_kernel_size),
                self.k_conv1d.weight.view(self.local_proj_size, self.conv_kernel_size),
                self.v_conv1d.weight.view(self.local_proj_size, self.conv_kernel_size),
            ],
            dim=0,
        )

    def _run_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        output_final_state: bool,
        recurrent: bool,
    ):
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        kwargs = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta.float(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=getattr(self.config, "linear_attn_config", {}).get(
                "gate_lower_bound", None
            )
            is not None,
            lower_bound=getattr(self.config, "linear_attn_config", {}).get(
                "gate_lower_bound", None
            ),
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )
        if recurrent:
            kwargs.pop("safe_gate", None)
            return fused_recurrent_kda(**kwargs)
        return chunk_kda(**kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        fwd_ctx = get_forward_context()
        gdn_metadata = getattr(fwd_ctx.attn_metadata, "gdn_metadata", None)
        if gdn_metadata is None:
            return hidden_states.new_zeros(hidden_states.shape)

        cache = fwd_ctx.kv_cache_data[f"layer_{self.layer_num}"]
        conv_state = cache.k_cache
        ssm_state = cache.v_cache
        if conv_state.size(1) != self.local_proj_size * 3:
            conv_state = conv_state.transpose(-1, -2)

        num_actual_tokens = gdn_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_actual_tokens]
        mixed_qkv = torch.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)],
            dim=-1,
        )
        gate = self.f_b_proj(self.f_a_proj(hidden_states))
        gate = rearrange(gate, "t (h d) -> 1 t h d", d=self.head_dim)
        beta = self.b_proj(hidden_states).unsqueeze(0)
        out = hidden_states.new_empty((num_actual_tokens, self.num_local_heads, self.head_dim))

        conv_weights = self._conv_weights()
        state_indices = gdn_metadata.non_spec_state_indices_tensor
        query_start_loc = gdn_metadata.non_spec_query_start_loc

        if gdn_metadata.num_prefills > 0:
            q, k, v = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                None,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=gdn_metadata.has_initial_state,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                k_dim_size=self.local_proj_size,
                v_dim_size=self.local_proj_size,
                metadata=gdn_metadata,
            )
            q = rearrange(q, "t (h d) -> 1 t h d", d=self.head_dim)
            k = rearrange(k, "t (h d) -> 1 t h d", d=self.head_dim)
            v = rearrange(v, "t (h d) -> 1 t h d", d=self.head_dim)
            initial = ssm_state[state_indices].contiguous()
            initial[~gdn_metadata.has_initial_state, ...] = 0
            # gfx1250 workaround: chunk_kda NaNs on short prompts (seq < chunk
            # size) and its `transpose_state_layout` output can mismatch the
            # decode-time fused_recurrent_kda reader, producing NaN on the first
            # decode step. Forcing the recurrent path for prefill keeps the KDA
            # state layout consistent across prefill/decode. Env-gated so the
            # fast chunk path stays default on archs where it is correct.
            _kda_force_recurrent = (
                os.getenv("ATOM_KDA_FORCE_RECURRENT", "0") == "1"
            )
            kda_out, last_state = self._run_kda(
                q,
                k,
                v,
                gate,
                beta,
                initial,
                query_start_loc,
                True,
                recurrent=_kda_force_recurrent,
            )
            ssm_state[state_indices] = last_state.to(ssm_state.dtype)
            out.copy_(kda_out.squeeze(0))
        elif gdn_metadata.num_decodes > 0:
            q, k, v = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.local_proj_size,
                self.local_proj_size,
                None,
                self.activation,
                conv_state_indices=state_indices[:num_actual_tokens],
                validate_data=True,
            )
            q = rearrange(q, "t (h d) -> 1 t h d", d=self.head_dim)
            k = rearrange(k, "t (h d) -> 1 t h d", d=self.head_dim)
            v = rearrange(v, "t (h d) -> 1 t h d", d=self.head_dim)
            initial = ssm_state[state_indices[:num_actual_tokens]].contiguous()
            kda_out, last_state = self._run_kda(
                q,
                k,
                v,
                gate,
                beta,
                initial,
                query_start_loc[: gdn_metadata.num_decodes + 1],
                True,
                recurrent=True,
            )
            ssm_state[state_indices[:num_actual_tokens]] = last_state.to(ssm_state.dtype)
            out.copy_(kda_out.squeeze(0))
        else:
            out.zero_()

        out = self.o_norm(out, rearrange(self.g_proj(hidden_states), "t (h d) -> t h d", d=self.head_dim))
        return self.o_proj(rearrange(out, "t h d -> t (h d)"))


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str,
        layer_num: int = 0,
        alt_stream: Optional["torch.cuda.Stream"] = None,
    ):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        quant_config = atom_config.quant_config
        self.config = config
        self.layer_idx = layer_num
        self.hidden_size = config.hidden_size
        if layer_num in config.kimi_kda_layers:
            self.self_attn = KimiKDAAttention(atom_config, quant_config, prefix=f"{prefix}.self_attn")
            self.is_linear_attn = True
        else:
            self.self_attn = KimiFullAttention(atom_config, quant_config, prefix=f"{prefix}.self_attn")
            self.is_linear_attn = False

        if (
            config.num_experts is not None
            and layer_num >= config.first_k_dense_replace
            and layer_num % getattr(config, "moe_layer_freq", 1) == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                alt_stream=alt_stream,
            )
        else:
            self.mlp = KimiMLP(
                config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )

    def _ffn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(hidden_states)
        return self.mlp(hidden_states)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor | None = None,
    ):
        if not self.use_attn_residuals:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            if self.is_linear_attn:
                hidden_states = self.self_attn(hidden_states)
            else:
                hidden_states = self.self_attn(positions, hidden_states)
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self._ffn(hidden_states)
            return residual + hidden_states, block_residual

        prefix_sum = hidden_states
        if block_residual is not None and block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                prefix_sum, block_residual, self.self_attention_res_proj, self.self_attention_res_norm
            )
        if self.layer_idx % self.attn_res_block_size == 0:
            assert block_residual is not None
            block_residual = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        if self.is_linear_attn:
            hidden_states = self.self_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states = _apply_attn_res(
            prefix_sum, block_residual, self.mlp_res_proj, self.mlp_res_norm
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._ffn(hidden_states)
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        return prefix_sum, block_residual


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: ReplicatedLinear,
    norm: RMSNorm,
) -> torch.Tensor:
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    if os.environ.get("ATOM_K3_FUSED", "1") == "1":
        from atom.models.kimi_k3_fused import apply_attn_res

        return apply_attn_res(
            prefix_sum, block_residual, proj.weight.squeeze(0), norm.weight, eps
        )
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_f = values.float()
    var = values_f.pow(2).mean(-1, keepdim=True)
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    normed = values_f * torch.rsqrt(var + eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (normed * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, values_f).squeeze(1).to(prefix_sum.dtype)


class KimiLinearModel(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        _normalize_kimi_config(config)
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = PPMissingLayer()

        # Single alt_stream shared across layers for shared-expert overlap.
        # Created only when the feature is enabled (default off); when None,
        # every MoE block falls back to single-stream. Attention/MoE never
        # contend since each block runs its MoE after its attention.
        self.alt_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream()
            if envs.ATOM_K3_SHARED_EXPERT_OVERLAP and torch.cuda.is_available()
            else None
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: KimiDecoderLayer(
                atom_config,
                prefix=prefix,
                layer_num=layer_num or 0,
                alt_stream=self.alt_stream,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, "attn_res_block_size", None) is not None:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "block_residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            block_residual = (
                hidden_states.new_zeros(hidden_states.shape[0], 0, hidden_states.shape[1])
                if getattr(self.config, "attn_res_block_size", None) is not None
                else None
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            block_residual = intermediate_tensors["block_residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, block_residual = layer(positions, hidden_states, block_residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "block_residual": block_residual}
            )
        if getattr(self.config, "attn_res_block_size", None) is not None:
            hidden_states = _apply_attn_res(
                hidden_states,
                block_residual,
                self.output_attn_res_proj,
                self.output_attn_res_norm,
            )
        return self.norm(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts + (self.config.num_shared_experts or 0),
        )


class KimiLinearForCausalLM(nn.Module):
    packed_modules_mapping = {
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }
    weights_mapping = {
        "weight_packed": "weight",
    }

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        self.config = config
        self.quant_config = atom_config.quant_config
        self.model = KimiLinearModel(atom_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class KimiK3ForCausalLM(nn.Module):
    skip_weight_prefixes = ["vision_tower.", "mm_projector."]
    quant_exclude_name_mapping = {
        "language_model.model.": "language_model.model.",
        "language_model.lm_head": "language_model.lm_head",
    }
    packed_modules_mapping = KimiLinearForCausalLM.packed_modules_mapping
    weights_mapping = KimiLinearForCausalLM.weights_mapping

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        root_config = atom_config.hf_config
        if hasattr(root_config, "text_config") and root_config.text_config is not root_config:
            _normalize_kimi_config(root_config.text_config)
            if (
                getattr(root_config, "quantization_config", None) is None
                and getattr(root_config.text_config, "quantization_config", None)
                is not None
            ):
                atom_config.quant_config = QuantizationConfig(
                    root_config.text_config,
                    atom_config.online_quant_config,
                )
        else:
            _normalize_kimi_config(root_config)
        self.config = _text_config(root_config)
        self.quant_config = atom_config.quant_config
        self.language_model = KimiLinearForCausalLM(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # The loader matches expert entries as substrings of full checkpoint
        # names, so keep these generic enough to match each layer's
        # `block_sparse_moe.experts.{id}.w*.weight` entries.
        return self.language_model.get_expert_mapping()
