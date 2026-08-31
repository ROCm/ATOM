# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only Kimi-K3 text model.

The checkpoint is multimodal, but ATOM serves the text path here.  The language
weights live under ``language_model.*`` in the checkpoint, so this module keeps
the same object hierarchy and skips the vision tower/projector tensors.
"""

from typing import ClassVar

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from einops import rearrange
from torch import nn

from atom.config import Config, QuantizationConfig, get_current_atom_config

# Side-effect import: registers `torch.ops.aiter.maybe_dual_stream_forward`, the
# Dynamo-opaque custom op that dispatches the MoE between single- and dual-stream
# forwards (shared with deepseek_v2/v4). Imported for the registration only.
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.attention_mla import MLAModules
from atom.model_ops.attention_residual import AttnRes
from atom.model_ops.base_attention import Attention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from atom.model_ops.fla_ops.replayssm import (
    replayssm_sigmoid_gating_delta_rule,
)
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    ReplicatedLinear,
    RowParallelLinear,
    use_fp4_non_shuffle_triton_gemm,
    use_triton_gemm,
)
from atom.model_ops.mamba_ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.rotary_embedding import RotaryEmbedding
from atom.model_ops.utils import atom_parameter
from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.quant_spec import should_skip_online_quant
from atom.utils import envs, mark_spliting_op
from atom.utils.decorators import mark_trace, support_torch_compile
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
        config,
        "linear_num_value_heads",
        lin.get("num_heads", config.num_attention_heads),
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
    config.kimi_full_attn_layers = [int(i) - 1 for i in lin.get("full_attn_layers", [])]
    config.kimi_kda_layers = [int(i) - 1 for i in lin.get("kda_layers", [])]
    config.num_gdn_attn_state = len(config.kimi_kda_layers)
    config.num_full_attn = len(config.kimi_full_attn_layers)

    # Keep the logical Q/K head width available to shared model infrastructure.
    config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {
            "rope_theta": getattr(config, "rope_theta", 10000.0),
            "rope_type": "default",
        }


def _kda_packed_modules_mapping(
    kda_layer_indices: list[int],
) -> dict[str, tuple[str, int]]:
    mapping = {
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
        ".q_a_proj": (".fused_qkv_a_proj", 0),
        ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
    }
    projection_names = ("q_proj", "k_proj", "v_proj", "g_proj")
    for layer_idx in kda_layer_indices:
        prefix = f".layers.{layer_idx}.self_attn."
        for shard_id, projection_name in enumerate(projection_names):
            mapping[f"{prefix}{projection_name}"] = (f"{prefix}in_proj", shard_id)
    return mapping


def _extract_layer_idx(prefix: str) -> int:
    for part in reversed(prefix.split(".")):
        if part.isdigit():
            return int(part)
    return 0


# RMSNorm+quant fusion is scheme-agnostic: the aiter fused RMSNorm kernels
# (RMSNorm._aiter_rms_quant and deepseek's _fuse_rmsnorm_quant) emit any of these
# dynamic activation quant layouts, so a preceding norm can fold the quant for a
# Linear that runs one of them.
_RMS_FUSABLE_QUANT_TYPES = (
    QuantType.per_1x32,
    QuantType.per_1x128,
    QuantType.per_Token,
)


def _effective_layer_quant(
    quant_config: QuantizationConfig | None, prefix: str
) -> tuple[QuantType, torch.dtype | None]:
    """Resolve the ``(quant_type, quant_dtype)`` a Linear runs with at runtime.

    Same resolution the Linear itself performs (mirrors
    ``LinearBase.online_quantize_weight`` and deepseek_v2's MLA setup): the static
    checkpoint scheme, overridden by the online-quant target when that override
    actually applies (``should_skip_online_quant``). A preceding RMSNorm uses this
    to decide whether -- and in which scheme (fp8 / fp4x2, per-token / block) -- to
    fuse its activation quant, rather than hard-coding one layout.
    """
    if quant_config is None:
        return QuantType.No, None
    cfg = quant_config.get_layer_quant_config(prefix)
    if quant_config.online_quant:
        online_cfg = quant_config.get_layer_quant_config(prefix, use_online_quant=True)
        if not should_skip_online_quant(cfg.quant_type, cfg.quant_dtype, online_cfg):
            cfg = online_cfg
    return cfg.quant_type, cfg.quant_dtype


class _NoPositionalRotaryEmbedding(RotaryEmbedding):
    def _compute_cos_sin_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache_shape = (
            self.max_position_embeddings,
            1,
            1,
            self.rotary_dim // 2,
        )
        return (
            torch.ones(cache_shape, dtype=torch.float32),
            torch.zeros(cache_shape, dtype=torch.float32),
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return query, key


class SituAndMul(nn.Module):
    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from atom.model_ops.kimi_k3 import situ_and_mul

        return situ_and_mul(x, self.beta, self.linear_beta)


class KimiRMSNormGated(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        quant_type: QuantType | None = None,
        quant_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = atom_parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # When ``quant_type`` names a fusable per-token scheme, the per-head
        # sigmoid-gated norm also emits (quantized, scale) so the consuming
        # o_proj skips its standalone quant; otherwise it returns a bf16 tensor.
        self.quant_type = quant_type
        self.quant_dtype = quant_dtype

    def forward(self, x: torch.Tensor, gate: torch.Tensor):
        from atom.model_ops.kimi_k3 import rmsnorm_gated

        return rmsnorm_gated(
            x,
            self.weight,
            gate,
            self.variance_epsilon,
            quant_type=self.quant_type,
            quant_dtype=self.quant_dtype,
        )


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
        # `x` arrives as a (fp8, scale) tuple when the preceding RMSNorm fused its
        # activation quant (dense-MLP layers)
        x_scale = None
        if isinstance(x, tuple):
            x, x_scale = x
        return self.down_proj(self.act_fn(self.gate_up_proj(x, x_scale)))


class KimiSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.alt_stream = alt_stream
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_latent_moe = (
            getattr(config, "routed_expert_hidden_size", None) is not None
        )
        self.moe_hidden_size = (
            config.routed_expert_hidden_size
            if self.use_latent_moe
            else config.hidden_size
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = atom_parameter(
            torch.empty(config.num_experts, dtype=torch.bfloat16)
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
            # inter=3072/TP8=384 is a 128-multiple; pad to 128 (not the 256
            # default) to avoid padding the MXFP4 MoE intermediate up to 512.
            pad_align=128,
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
            up_proj_quant_type, up_proj_quant_dtype = _effective_layer_quant(
                quant_config, up_proj_prefix
            )
            latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)
            # AITER RMSNorm+quant emits the activation layout consumed directly by
            # the routed up-projection. FP4 Triton paths choose an M-dependent
            # shuffled/non-shuffled scale layout, so keep those on their existing
            # standalone quant path until the fused kernel supports both layouts.
            fp4_triton_active = up_proj_quant_type == QuantType.per_1x32 and (
                use_triton_gemm() or use_fp4_non_shuffle_triton_gemm()
            )
            self.fuse_routed_norm_quant = latent_moe_use_norm and (
                (
                    up_proj_quant_type == QuantType.per_1x32
                    and up_proj_quant_dtype == dtypes.fp4x2
                    and not fp4_triton_active
                )
                or (
                    up_proj_quant_type in (QuantType.per_1x128, QuantType.per_Token)
                    and up_proj_quant_dtype == dtypes.fp8
                )
            )
            self.routed_expert_norm = (
                RMSNorm(
                    self.moe_hidden_size,
                    eps=config.rms_norm_eps,
                    fused_quant=self.fuse_routed_norm_quant,
                    quant_config=quant_config,
                    prefix=up_proj_prefix,
                )
                if latent_moe_use_norm
                else None
            )

        # Dual-stream gate: overlap the shared-expert GEMMs (on alt_stream) with
        # the routed-expert path (on the main stream). Only meaningful when a
        # shared branch exists and an alt_stream was threaded in. TBO already
        # provides its own overlap, so the two are mutually exclusive.
        self._use_dual_stream = False
        if self.shared_experts is not None and self.alt_stream is not None:
            tbo_active = get_current_atom_config().enable_tbo
            if envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0 and not tbo_active:
                self._use_dual_stream = True
        if self._use_dual_stream:
            # Register self so `maybe_dual_stream_forward` can look this module up
            # by prefix from static_forward_context (the op is Dynamo-opaque).
            cc = get_current_atom_config().compilation_config
            cc.static_forward_context[self.prefix] = self

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns ``(routed, shared)``, left unsummed.

        The caller's next apply_attn_res folds both into its prefix on-load, so
        deferring the add here removes a whole [T, H] elementwise kernel and its
        HBM round-trip per MoE layer. ``shared`` is None when there are no shared
        experts, when the two branches had to be summed before a collective (see
        `split_moe_forward`), or when the dual-stream path ran.
        """
        if self._use_dual_stream:
            # maybe_dual_stream_forward is a registered custom op, and torch's
            # schema inference has no representation for a tuple return, so the
            # deferred add can't cross that boundary. Both dispatch targets sum
            # internally and the pair collapses to (summed, None). Cheap to give
            # up: dual-stream only engages at <= ATOM_DUAL_STREAM_MOE_TOKEN_
            # THRESHOLD tokens, where the [T, H] add it costs us is small.
            summed = torch.ops.aiter.maybe_dual_stream_forward(
                hidden_states, self.prefix
            )
            return summed, None
        return self.split_moe_forward(hidden_states)

    def routed_expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Routed-expert path only. For the latent MoE this includes the routed
        all-reduce (required before the nonlinear routed_expert_norm); the shared
        branch is handled by the caller."""
        router_logits = self.gate(hidden_states)
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
            if isinstance(routed_output, tuple):
                routed_output, routed_output_scale = routed_output
                routed_output = self.routed_expert_up_proj(
                    routed_output, x_scale=routed_output_scale
                )
            else:
                routed_output = self.routed_expert_up_proj(routed_output)
        return routed_output

    def single_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Single-tensor dispatch target for `maybe_dual_stream_forward`.

        The op's schema can't carry a tuple, so this sums what
        `split_moe_forward` hands back. Only reached via the dual-stream
        dispatcher; `forward` calls `split_moe_forward` directly otherwise.
        """
        routed, shared = self.split_moe_forward(hidden_states)
        return routed if shared is None else routed + shared

    def split_moe_forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Routed and shared branches, returned unsummed where deferring is
        legal. See `forward` for why the add is worth deferring."""
        identity = hidden_states
        routed_output = self.routed_expert_forward(hidden_states)
        if self.use_latent_moe:
            if self.shared_experts is not None:
                # Shared branch is TP-partial (down_proj is row-parallel); reduce
                # it separately. Both branches are full after their own
                # all-reduce, so the add between them is deferrable.
                shared_output = self.shared_experts(identity)
                if self.tp_size > 1:
                    shared_output = tensor_model_parallel_all_reduce(shared_output)
                return routed_output, shared_output
            return routed_output, None
        # Non-latent path: routed experts and shared experts are both TP-partial
        # and everything after them is linear, so a single deferred all-reduce
        # over their sum is correct.
        if self.shared_experts is not None:
            shared_output = self.shared_experts(identity)
            if self.tp_size == 1:
                # No collective to batch, so the add is free to defer.
                return routed_output, shared_output
            # With TP the branches must be summed BEFORE the all-reduce: both are
            # partial here, and while all_reduce is linear (so reducing them
            # separately would also be correct), that would cost two collectives
            # to save one elementwise add. Sum first, hand back a single tensor.
            routed_output = routed_output + shared_output
        if self.tp_size > 1:
            routed_output = tensor_model_parallel_all_reduce(routed_output)
        return routed_output, None

    def dual_stream_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Queue routed pre-AR work first on the current stream, then run the
        # shared-expert path on alt_stream. The latent path keeps both all-reduces
        # on their respective streams while preserving shared AR -> routed AR
        # order on the single TP communicator.
        current = torch.cuda.current_stream()
        alt = self.alt_stream
        alt.wait_stream(current)

        if self.use_latent_moe:
            router_logits = self.gate(hidden_states)
            routed_input = self.routed_expert_down_proj(hidden_states)
            routed_output = self.experts(routed_input, router_logits)
        else:
            routed_output = self.routed_expert_forward(hidden_states)

        with torch.cuda.stream(alt):
            shared_output = self.shared_experts(hidden_states)
            if self.use_latent_moe and self.tp_size > 1:
                shared_output = tensor_model_parallel_all_reduce(shared_output)

        if self.use_latent_moe:
            if self.tp_size > 1:
                current.wait_stream(alt)
                routed_output = tensor_model_parallel_all_reduce(routed_output)

            if self.routed_expert_norm is not None:
                routed_output = self.routed_expert_norm(routed_output)
            if isinstance(routed_output, tuple):
                routed_output, routed_output_scale = routed_output
                routed_output = self.routed_expert_up_proj(
                    routed_output,
                    x_scale=routed_output_scale,
                )
            else:
                routed_output = self.routed_expert_up_proj(routed_output)

            if self.tp_size == 1:
                current.wait_stream(alt)
            shared_output.record_stream(current)
            return routed_output + shared_output

        # Non-latent: shared has no AR yet; single deferred AR over the sum.
        current.wait_stream(alt)
        routed_output = routed_output + shared_output
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

        self.fused_qkv_a_proj = MergedReplicatedLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank, eps=1e-6, prefix=f"{prefix}.q_a_layernorm"
        )
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank, eps=1e-6, prefix=f"{prefix}.kv_a_layernorm"
        )
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

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta") or 10000.0
        # max_position_embeddings field only exists in the text config
        _text_max_pos = getattr(config, "max_position_embeddings", None)
        rope_max_position = int(
            _text_max_pos or getattr(atom_config, "max_model_len", None) or 16384
        )
        self.rotary_emb = _NoPositionalRotaryEmbedding(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=rope_max_position,
            base=rope_theta,
        )
        mla_modules = MLAModules(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.q_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=nn.Identity(),
            indexer=None,
            is_sparse=False,
            topk_tokens=None,
        )
        self.layer_num = _extract_layer_idx(prefix)
        self.attn = Attention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            kv_cache_dtype=atom_config.kv_cache_dtype,
            layer_num=self.layer_num,
            use_mla=True,
            mla_modules=mla_modules,
            prefix=prefix,
        )

        qknorm_type, qknorm_dtype = _effective_layer_quant(
            quant_config, f"{prefix}.q_b_proj"
        )
        self.fuse_qknorm_quant = qknorm_dtype in (dtypes.fp8, dtypes.fp4x2)
        self.qknorm_dtype = qknorm_dtype if self.fuse_qknorm_quant else torch.bfloat16
        self.qknorm_quant_type_value = (
            qknorm_type.value if self.fuse_qknorm_quant else QuantType.No.value
        )
        # input_layernorm fuses its activation quant only when BOTH consumers of
        # the normed hidden state -- fused_qkv_a_proj and g_proj -- run with the
        # same fusable RMSNorm quant scheme (else a mismatched consumer mis-GEMMs).
        a_scheme = _effective_layer_quant(quant_config, f"{prefix}.fused_qkv_a_proj")
        g_scheme = _effective_layer_quant(quant_config, f"{prefix}.g_proj")
        self.fuse_input_norm_quant = (
            a_scheme[0] in _RMS_FUSABLE_QUANT_TYPES and a_scheme == g_scheme
        )
        self.input_quant_prefix = f"{prefix}.fused_qkv_a_proj"

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # deepseek_v2 pattern: one _fuse_rmsnorm_quant kernel does q_a norm +
        # kv_a norm (+ q-activation quant), then q's scale is forwarded into the
        # MLA module (q_proj consumes it).
        from atom.models.deepseek_v2 import _fuse_rmsnorm_quant

        # hidden_states is a (fp8, scale) tuple when input_layernorm fused the
        # quant; both fused_qkv_a_proj and g_proj consume it directly.
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states

        q_c, kv_c, k_rope = torch.split(
            self.fused_qkv_a_proj(hidden_states, hidden_states_scale),
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        q_shuffle = False
        q_scale_shuffle_padding = False
        if self.qknorm_dtype == dtypes.fp4x2:
            from atom.model_ops.linear import use_triton_gemm
            from atom.models.deepseek_v2 import _mxfp4_activation_quant_layout

            if not use_triton_gemm():
                q_shuffle, q_scale_shuffle_padding = _mxfp4_activation_quant_layout(
                    q_c.shape[0]
                )
        (q, q_scale), _, kv, _ = _fuse_rmsnorm_quant(
            q_c,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.eps,
            kv_c,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.eps,
            None,
            dtype_quant=self.qknorm_dtype,
            shuffle=q_shuffle,
            scale_shuffle_padding=q_scale_shuffle_padding,
            group_size=128,
            quant_type=self.qknorm_quant_type_value,
            output_unquantized_inp1=False,
            transpose_scale=True,
        )
        attn_out = self.attn(q, kv, k_rope, positions, q_scale=q_scale)
        attn_out = attn_out * torch.sigmoid(
            self.g_proj(hidden_states, hidden_states_scale)
        )
        return self.o_proj(attn_out)


def _kda_attention_with_output_fake(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    # The mixer output (o_proj) is always bf16 even when the input activation is
    # fp8 (fused input_layernorm+quant), so pin the dtype rather than empty_like.
    return torch.empty(
        hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
    )


@mark_spliting_op(
    is_custom=True,
    gen_fake=_kda_attention_with_output_fake,
    mutates_args=[],
)
def kda_attention_with_output(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    """Opaque splitting-op boundary for the KDA mixer.

    The KDA recurrence reads the forward context, calls fla causal-conv/kda
    kernels and mutates the per-request conv/ssm cache in place. torch.compile
    (level 3) mis-compiles that stateful path into garbage if it is allowed to
    trace through it, so the whole mixer is wrapped in a custom op — inductor
    treats it as opaque and the piecewise backend splits the graph here,
    exactly as the GDN path does via aiter.linear_attention_with_output_base.
    """
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return self._forward_impl(hidden_states, hidden_states_scale)


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

        # Register under a stable name so the kda_attention_with_output custom op
        # can recover this module from the forward context. The op is the
        # graph-split boundary that keeps torch.compile from tracing (and
        # mis-compiling) the stateful KDA recurrence.
        self.layer_name = prefix
        compilation_config = atom_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # The top-level model maps four separate checkpoint projections
        # directly into this fused [q | k | v | g] parameter. Mapping keys
        # include the KDA layer index so KimiFullAttention.g_proj is untouched.
        self.in_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [
                self.proj_size,
                self.proj_size,
                self.proj_size,
                self.proj_size,
            ],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        # Keep beta separate so the fused in-proj output width remains the
        # tile-aligned 4 * local_proj_size. Beta is widened to fp32 in _run_kda.
        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.b_proj",
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
        # Lower bound of the KDA forget gate (Kimi uses -5.0). Consumed by both
        # the fla prefill path (_run_kda) and the fused decode kernel.
        self._kda_gate_lower_bound = (
            getattr(config, "linear_attn_config", {}) or {}
        ).get("gate_lower_bound", None)
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
        o_type, o_dtype = _effective_layer_quant(quant_config, f"{prefix}.o_proj")
        self.o_norm = KimiRMSNormGated(
            self.head_dim,
            eps=config.rms_norm_eps,
            quant_type=o_type,
            quant_dtype=o_dtype,
        )
        self.o_proj = RowParallelLinear(
            self.proj_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # The decoder's input_layernorm can fuse its activation quant into the
        # single fused in_proj GEMM (q|k|v|g|b|f_a) that consumes the normed hidden
        # state; the (fp8, scale) rides the splitting custom op into _forward_impl.
        # Enabled for any fusable RMSNorm quant scheme (fp8 / fp4x2).
        in_proj_type, _ = _effective_layer_quant(quant_config, f"{prefix}.in_proj")
        self.fuse_input_norm_quant = in_proj_type in _RMS_FUSABLE_QUANT_TYPES
        self.input_quant_prefix = f"{prefix}.in_proj"

    def get_streaming_deferred_modules(self) -> tuple[nn.Module, ...]:
        """Children that must remain unquantized until KDA fuses their weights."""
        return self.in_proj, self.f_a_proj

    def process_weights_after_loading(self) -> None:
        """Fuse all hidden-input projections into the single in-proj (one GEMM).

        Upstream already loads q/k/v/g into a single ``in_proj``
        (``MergedColumnParallelLinear``) via ``packed_modules_mapping``. Here we
        extend that fused weight in place with the two remaining projections that
        also consume ``hidden_states`` -- ``b_proj`` (beta) and ``f_a_proj`` --
        so ``forward`` runs one ``self.in_proj(...)`` producing
        ``[q | k | v | g | b | f_a]`` instead of three separate launches. The
        tail storage is then released (the modules stay as empty shells; their
        bf16 post-load hooks are no-ops and never re-run). Runs once; idempotent.

        f_b_proj is NOT fused: it consumes ``f_a_proj``'s output, not
        ``hidden_states``, so it is a data-dependent second GEMM and cannot ride
        the same launch.

        Fused output width is ``4*local_proj + num_local_heads + head_dim``. The
        two small tails (``b`` = num_local_heads, ``f_a`` = head_dim) make N a
        non-multiple of the GEMM tile, so the fused shape may fall back to an
        untuned tgemm config until one is tuned for it; the saved launches
        dominate on the launch-bound decode path.

        Assumes bf16 (unquantized) attention weights, which the Kimi-K3
        checkpoint guarantees (``re:.*self_attn.*`` is in the quant ignore
        list). A quantized-attention checkpoint would need per-shard scale
        handling and is rejected loudly rather than silently mis-fused.
        """
        if getattr(self, "_in_proj_fused", False):
            return
        # Order defines the forward-time slice boundaries below; keep in sync.
        # in_proj already holds the fused [q | k | v | g] (4 * local_proj_size);
        # b_proj and f_a_proj are appended as the two tails.
        tails = (self.b_proj, self.f_a_proj)
        assert all(m.quant_type == QuantType.No for m in (self.in_proj, *tails)), (
            "KDA in-proj fusion assumes unquantized (bf16) attention weights; "
            "this checkpoint quantizes self_attn projections."
        )
        fused = torch.cat(
            [self.in_proj.weight.data, *[m.weight.data for m in tails]], dim=0
        ).contiguous()
        # Grow in_proj's weight in place so the existing module (and its
        # unquantized tgemm.mm path) produces the wide fused output directly.
        self.in_proj.weight = nn.Parameter(fused, requires_grad=False)
        # Release the tail weight storage. The modules stay as empty shells;
        # their bf16 post-load hooks are no-ops and are never re-run.
        for m in tails:
            m.weight.data = m.weight.data.new_empty(0)

        # Pre-concatenate the static q/k/v causal-conv weights once here instead
        # of rebuilding the [3*local_proj_size, K] tensor on every forward.
        self.conv_weight = torch.cat(
            [
                self.q_conv1d.weight.view(self.local_proj_size, self.conv_kernel_size),
                self.k_conv1d.weight.view(self.local_proj_size, self.conv_kernel_size),
                self.v_conv1d.weight.view(self.local_proj_size, self.conv_kernel_size),
            ],
            dim=0,
        ).contiguous()
        self._in_proj_fused = True

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
    ):
        from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

        return chunk_kimi_delta_attn(
            q=q,
            k=k,
            v=v,
            g=g,
            # Keep beta in fp32: the sigmoid b = sigmoid(beta) is applied
            # in-kernel with use_beta_sigmoid_in_kernel, and triton's sigmoid
            # follows the input dtype -- a bf16 beta yields a bf16 write
            # strength, which erodes the delta-rule state update across the 71
            # KDA layers (measured gsm8k regression). b_proj stays bf16; only
            # this reduction is widened.
            beta=beta.float(),
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self._kda_gate_lower_bound is not None,
            lower_bound=self._kda_gate_lower_bound,
            cu_seqlens=cu_seqlens,
            # V-first state, matching the layout mamba_v_cache holds and the
            # fused decode kernel writes. Without it the state comes back
            # K-first and decode reads it transposed.
            state_v_first=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states is a (fp8, scale) tuple when input_layernorm fused the
        # per-token quant; carry the scale through the opaque splitting custom op
        # so in_proj consumes it in _forward_impl.
        hidden_states_scale = None
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        # Route through the opaque custom op so torch.compile splits the graph
        # here instead of tracing the stateful recurrence in _forward_impl.
        return torch.ops.aiter.kda_attention_with_output(
            hidden_states, hidden_states_scale, self.layer_name
        )

    @mark_trace
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fwd_ctx = get_forward_context()
        kda_metadata = getattr(fwd_ctx.attn_metadata, "kda_metadata", None)
        if kda_metadata is None:
            # Native ATOM/SGLang integrations still expose the shared legacy
            # field. vLLM 0.26+ uses the dedicated KDA metadata adapter.
            kda_metadata = getattr(fwd_ctx.attn_metadata, "gdn_metadata", None)
        if kda_metadata is None:
            # Output is bf16 even when the input activation is fp8 (fused quant).
            return torch.zeros(
                hidden_states.shape, dtype=torch.bfloat16, device=hidden_states.device
            )

        cache = fwd_ctx.kv_cache_data[f"layer_{self.layer_num}"]
        conv_state = cache.k_cache
        ssm_state = cache.v_cache
        if conv_state.size(1) != self.local_proj_size * 3:
            conv_state = conv_state.transpose(-1, -2)

        num_actual_tokens = kda_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_actual_tokens]
        if hidden_states_scale is not None:
            hidden_states_scale = hidden_states_scale[:num_actual_tokens]
        # Single fused in-proj GEMM producing [q | k | v | g]; slice out each
        # part. `out_gate` is the KDA output gate consumed at o_norm below
        # (computed here so it rides the same GEMM instead of a separate one
        # after the recurrence). in_proj's weight was grown in
        # process_weights_after_loading to the fused [q | k | v | g | b | f_a],
        # so this single unquantized call emits all six; f_b_proj stays a
        # separate GEMM because it consumes f_a's output, not hidden_states.
        lp = self.local_proj_size
        nlh = self.num_local_heads
        hd = self.head_dim
        fused_in = self.in_proj(hidden_states, x_scale=hidden_states_scale)
        # No .contiguous() needed: mixed_qkv is a column slice (feature stride 1,
        # row stride N_fused). Both causal-conv consumers read the token stride
        # from the tensor itself — causal_conv1d_fn uses x.stride(1) after
        # transpose (channel-last: stride(0)==1), and causal_conv1d_update only
        # requires x.stride(1)==1 (feature-contiguous, which the slice preserves).
        mixed_qkv = fused_in[..., : 3 * lp]
        out_gate = fused_in[..., 3 * lp : 4 * lp]
        # beta is widened to fp32 inside _run_kda (see the note there): the KDA
        # delta-rule write strength must stay fp32 for accuracy.
        beta = fused_in[..., 4 * lp : 4 * lp + nlh].unsqueeze(0)
        # f_a feeds a second GEMM (f_b_proj); make it contiguous so tgemm sees a
        # unit row stride rather than the fused output's N_fused stride.
        f_a = fused_in[..., 4 * lp + nlh : 4 * lp + nlh + hd].contiguous()
        gate = self.f_b_proj(f_a)
        gate = rearrange(gate, "t (h d) -> 1 t h d", d=self.head_dim)
        # Allocate from fused_in (bf16), not hidden_states, which may be fp8.
        out = fused_in.new_empty(
            (num_actual_tokens, self.num_local_heads, self.head_dim)
        )

        conv_weights = self.conv_weight
        state_indices = kda_metadata.non_spec_state_indices_tensor
        # Slot the incoming state is READ from. It differs from the write slot
        # for exactly one forward: the prefix-cache hit that forks off a
        # checkpoint (BlockManager._attach_state_group reads `state_fork_src`
        # and writes a freshly popped group). Reading the write slot there
        # resumes from whatever the recycled group still held. Only prefill can
        # carry a fork -- prepare_state_indices asserts it, and min_fork_tokens
        # keeps the chunk long enough -- so the decode branches below stay on
        # `state_indices`. Falling back to the write slot leaves every non-fork
        # forward bit-identical. Mirrors attention_gdn.py.
        state_indices_in = kda_metadata.non_spec_state_indices_in_tensor
        if state_indices_in is None:
            state_indices_in = state_indices
        query_start_loc = kda_metadata.non_spec_query_start_loc

        if kda_metadata.num_prefills > 0:
            q, k, v = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                None,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=kda_metadata.has_initial_state,
                cache_indices=state_indices,
                cache_indices_in=state_indices_in,
                query_start_loc=query_start_loc,
                k_dim_size=self.local_proj_size,
                v_dim_size=self.local_proj_size,
                metadata=kda_metadata,
            )
            q = rearrange(q, "t (h d) -> 1 t h d", d=self.head_dim)
            k = rearrange(k, "t (h d) -> 1 t h d", d=self.head_dim)
            v = rearrange(v, "t (h d) -> 1 t h d", d=self.head_dim)
            # Fused masked gather: ssm_state[state_indices] with fresh
            # sequences (~has_initial_state) written as zeros in one pass,
            # replacing the gather + separate zero-write.
            from atom.model_ops.kimi_k3 import gather_kda_initial_state

            initial = gather_kda_initial_state(
                ssm_state, state_indices_in, kda_metadata.has_initial_state
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
            )
            # last_state already has ssm_state's dtype (fla preserves the
            # initial_state dtype; the gathered initial is allocated as such),
            # so no .to() cast is needed.
            ssm_state[state_indices] = last_state
            out.copy_(kda_out.squeeze(0))
        elif kda_metadata.num_decodes > 0:
            # Slice the per-token cache-slot indices once (used for both the
            # conv update and the fused recurrence below).
            decode_state_indices = state_indices[:num_actual_tokens]
            q, k, v = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.local_proj_size,
                self.local_proj_size,
                None,
                self.activation,
                conv_state_indices=decode_state_indices,
                validate_data=False,
            )
            q = rearrange(q, "t (h d) -> 1 t h d", d=self.head_dim)
            k = rearrange(k, "t (h d) -> 1 t h d", d=self.head_dim)
            v = rearrange(v, "t (h d) -> 1 t h d", d=self.head_dim)
            # Fused KDA decode: the kernel gathers the initial state from
            # ssm_state[decode_state_indices], writes the final state back to
            # the same slots inplace (inplace_final_state), and writes the
            # recurrence output straight into `out`. This folds the manual
            # gather / scatter-back / out.copy_ that the fla path required into
            # one kernel. is_kda + lower_bound select the per-K-channel,
            # lower-bounded sigmoid gate that Kimi-KDA uses (beta stays raw
            # logits; the kernel applies sigmoid in fp32 internally).
            if getattr(kda_metadata, "replayssm", False):
                # ReplaySSM: one checkpoint per request; the per-token state
                # snapshots this pool used to hold are rebuilt from the
                # (k, u, g) records on demand.
                nd = kda_metadata.num_decodes
                replayssm_sigmoid_gating_delta_rule(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    self.A_log,
                    self.dt_bias,
                    ckpt=ssm_state,
                    buf_k=cache.replay_buf_k,
                    buf_u=cache.replay_buf_u,
                    buf_g=cache.replay_buf_g,
                    write_pos=kda_metadata.write_pos,
                    slot_idx=kda_metadata.slot_idx[:nd],
                    cu_seqlens=query_start_loc[: nd + 1],
                    max_query_len=kda_metadata.replayssm_max_query_len,
                    o=out,
                    use_qk_l2norm_in_kernel=True,
                    lower_bound=self._kda_gate_lower_bound,
                )
            else:
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=gate,
                    b=beta,
                    dt_bias=self.dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    o=out,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=query_start_loc[: kda_metadata.num_decodes + 1],
                    ssm_state_indices=decode_state_indices,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=True,
                    lower_bound=self._kda_gate_lower_bound,
                )
        elif kda_metadata.num_spec_decodes > 0:
            # Speculative-decode pass
            spec_state_indices = kda_metadata.spec_state_indices_tensor
            spec_query_start_loc = kda_metadata.spec_query_start_loc
            num_accepted_tokens = kda_metadata.num_accepted_tokens
            q, k, v = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.local_proj_size,
                self.local_proj_size,
                None,
                self.activation,
                # First reserved slot per seq holds the resume state; the kernel
                # walks forward via num_accepted_tokens + query_start_loc.
                conv_state_indices=spec_state_indices[:, 0][
                    : kda_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                # Verify window: sizes the conv rollback window and hence the
                # kernel's NP2_STATELEN tile. Under ReplaySSM the slot table
                # keeps its [bs, mtp_k+1] shape but only column 0 is live, so
                # read the window off the metadata instead of the table width.
                max_query_len=(
                    kda_metadata.replayssm_max_query_len
                    if getattr(kda_metadata, "replayssm", False)
                    else spec_state_indices.size(-1)
                ),
                validate_data=False,
            )
            q = rearrange(q, "t (h d) -> 1 t h d", d=self.head_dim)
            k = rearrange(k, "t (h d) -> 1 t h d", d=self.head_dim)
            v = rearrange(v, "t (h d) -> 1 t h d", d=self.head_dim)
            if getattr(kda_metadata, "replayssm", False):
                nsd = kda_metadata.num_spec_decodes
                replayssm_sigmoid_gating_delta_rule(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    self.A_log,
                    self.dt_bias,
                    ckpt=ssm_state,
                    buf_k=cache.replay_buf_k,
                    buf_u=cache.replay_buf_u,
                    buf_g=cache.replay_buf_g,
                    write_pos=kda_metadata.write_pos,
                    slot_idx=kda_metadata.slot_idx[:nsd],
                    cu_seqlens=spec_query_start_loc[: nsd + 1],
                    max_query_len=kda_metadata.replayssm_max_query_len,
                    o=out,
                    use_qk_l2norm_in_kernel=True,
                    lower_bound=self._kda_gate_lower_bound,
                )
            else:
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=gate,
                    b=beta,
                    dt_bias=self.dt_bias,
                    q=q,
                    k=k,
                    v=v,
                    o=out,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : kda_metadata.num_spec_decodes + 1
                    ],
                    # 2D [bs, 1+num_spec]: per-token snapshot slots. Paired with
                    # num_accepted_tokens the kernel reads the resume state from
                    # slot[num_accepted-1] and writes a snapshot after each token.
                    ssm_state_indices=spec_state_indices,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=True,
                    lower_bound=self._kda_gate_lower_bound,
                )
        else:
            out.zero_()

        normed = self.o_norm(
            out, rearrange(out_gate, "t (h d) -> t h d", d=self.head_dim)
        )
        # A fused per-token quant makes o_norm return (quantized, scale); feed it
        # straight to o_proj's x_scale path. Otherwise it is a bf16 tensor.
        if isinstance(normed, tuple):
            o_fp8, o_scale = normed
            return self.o_proj(o_fp8, x_scale=o_scale)
        return self.o_proj(rearrange(normed, "t h d -> t (h d)"))


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str,
        layer_num: int = 0,
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        quant_config = atom_config.quant_config
        self.config = config
        self.layer_idx = layer_num
        self.hidden_size = config.hidden_size
        if layer_num in config.kimi_kda_layers:
            self.self_attn = KimiKDAAttention(
                atom_config, quant_config, prefix=f"{prefix}.self_attn"
            )
            self.is_linear_attn = True
        else:
            self.self_attn = KimiFullAttention(
                atom_config, quant_config, prefix=f"{prefix}.self_attn"
            )
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
        # Fuse the activation quant into input_layernorm when the attention input
        # projection(s) run with a fusable quant scheme (self_attn decides and
        # exposes the flag + representative prefix). The normed output then flows
        # to the attention as a (fp8, scale) tuple instead of a bf16 tensor + a
        # standalone quant op.
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_quant=self.self_attn.fuse_input_norm_quant,
            quant_config=(
                quant_config if self.self_attn.fuse_input_norm_quant else None
            ),
            prefix=self.self_attn.input_quant_prefix,
        )
        # Fuse post_attention_layernorm's quant into the dense-MLP gate_up_proj.
        # MoE layers are skipped: their router gate is unquantized and the routed
        # experts are excluded, so the normed output has mixed-precision consumers.
        if hasattr(self, "mlp"):
            ffn_type, _ = _effective_layer_quant(
                quant_config, f"{prefix}.mlp.gate_up_proj"
            )
            self.fuse_ffn_norm_quant = ffn_type in _RMS_FUSABLE_QUANT_TYPES
        else:
            self.fuse_ffn_norm_quant = False
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_quant=self.fuse_ffn_norm_quant,
            quant_config=quant_config if self.fuse_ffn_norm_quant else None,
            prefix=f"{prefix}.mlp.gate_up_proj",
        )

        self.use_attn_residuals = (
            getattr(config, "attn_res_block_size", None) is not None
        )
        if self.use_attn_residuals:
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                prefix=f"{prefix}.self_attention_res_norm",
            )
            self.mlp_res_norm = RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                prefix=f"{prefix}.mlp_res_norm",
            )
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
        # Built in both modes: a disabled AttnRes is the ordinary pre-norm
        # residual step, which is exactly what this layer used to open-code.
        # Both sites feed a rmsnorm, so both fold it into the kernel's store.
        # The projs/norms above stay the parameter owners; these only alias
        # them (see AttnRes).
        self.self_attention_attn_res = AttnRes(
            getattr(self, "self_attention_res_proj", None),
            getattr(self, "self_attention_res_norm", None),
            out_norm=self.input_layernorm,
            enabled=self.use_attn_residuals,
            # Only this site banks the running prefix as a candidate.
            block_size=getattr(config, "attn_res_block_size", None),
            layer_idx=layer_num,
        )
        self.mlp_attn_res = AttnRes(
            getattr(self, "mlp_res_proj", None),
            getattr(self, "mlp_res_norm", None),
            out_norm=self.post_attention_layernorm,
            enabled=self.use_attn_residuals,
        )

    def _ffn(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the FFN, leaving the MoE's routed/shared add to the caller.

        The second tensor (when not None) is folded into the next
        apply_attn_res's prefix as ``add_hidden2``, skipping a [T, H]
        elementwise kernel. A dense mlp has nothing to defer.
        """
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(hidden_states)
        return self.mlp(hidden_states), None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor | None = None,
        pending_add: torch.Tensor | None = None,
        pending_add2: torch.Tensor | None = None,
    ):
        # Both sites go through AttnRes in either mode: with residuals enabled
        # it mixes the block candidates, and without it degenerates to the
        # ordinary pre-norm residual step. input_layernorm and
        # post_attention_layernorm are its out_norms, so hidden_states comes
        # back already normed at both.
        hidden_states, prefix_sum = self.self_attention_attn_res(
            hidden_states, block_residual, pending_add, pending_add2
        )
        block_residual, prefix_sum = self.self_attention_attn_res.maybe_close_block(
            prefix_sum, block_residual
        )

        if self.is_linear_attn:
            hidden_states = self.self_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states, prefix_sum = self.mlp_attn_res(
            prefix_sum, block_residual, hidden_states
        )
        # Routed and shared expert outputs come back unsummed: the next layer's
        # attn_res kernel folds both into its prefix on-load, so the [T, H]
        # elementwise add that would combine them here never runs.
        hidden_states, shared = self._ffn(hidden_states)
        return prefix_sum, hidden_states, shared, block_residual

    @staticmethod
    def aux_hidden_state(output: tuple) -> torch.Tensor | None:
        """Reconstruct this layer's post-layer hidden state from ``forward``'s
        return, for a drafter tapping it as an aux hidden state.

        Every DSpark draft is trained on the HF reference's
        ``output.hidden_states[layer_idx + 1]`` -- the plain residual stream
        after this layer, which the reference forms as
        ``prefix_sum = prefix_sum + hidden_states`` before returning. forward()
        deliberately does not: it hands the FFN output back unapplied so the
        NEXT layer's attn_res kernel can fold it into its on-load, and an MoE
        layer defers its routed and shared outputs separately so that same fold
        absorbs their sum too. So add the pendings back here. Each is None on
        the layers that already folded it in.

        This lives on the layer rather than in the drafter because it is a
        property of THIS layer's return protocol -- it has to change in lockstep
        with forward(), and any drafter trained against a Kimi-K3 target needs
        the same reconstruction.
        """
        prefix_sum, *pendings, _block_residual = output
        if prefix_sum is None:
            return None
        for pending in pendings:
            if pending is not None:
                prefix_sum = prefix_sum + pending
        return prefix_sum


@support_torch_compile
class KimiLinearModel(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        _normalize_kimi_config(config)
        self.config = config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Shared second stream for dual-stream MoE (shared-expert GEMMs overlap the
        # routed path). Created once and threaded into every decoder layer; only
        # used when the model has shared experts.
        self.alt_stream = None
        if getattr(config, "num_shared_experts", 0):
            self.alt_stream = torch.cuda.Stream()
        _alt_stream = self.alt_stream

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: KimiDecoderLayer(
                atom_config,
                prefix=prefix,
                layer_num=layer_num or 0,
                alt_stream=_alt_stream,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        use_attn_residuals = getattr(config, "attn_res_block_size", None) is not None
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, prefix=f"{prefix}.norm"
            )
            if use_attn_residuals:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                    prefix=f"{prefix}.output_attn_res_norm",
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
            # self.norm folds into the kernel's store, so the mix it returns is
            # the model's final hidden state. Disabled, this is just self.norm
            # applied to the pending adds -- the old non-residual tail.
            self.output_attn_res = AttnRes(
                getattr(self, "output_attn_res_proj", None),
                getattr(self, "output_attn_res_norm", None),
                out_norm=self.norm,
                enabled=use_attn_residuals,
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
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_tokens(input_ids)
            )
            block_residual = (
                hidden_states.new_zeros(
                    hidden_states.shape[0], 0, hidden_states.shape[1]
                )
                if getattr(self.config, "attn_res_block_size", None) is not None
                else None
            )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            block_residual = intermediate_tensors["block_residual"]

        # Each layer hands its FFN output back unapplied (pending_add), and an MoE
        # layer hands its shared-expert output back separately (pending_add2); the
        # next layer's attn_res kernel folds both into its prefix on-load.
        pending_add = pending_add2 = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, pending_add, pending_add2, block_residual = layer(
                positions,
                hidden_states,
                block_residual,
                pending_add=pending_add,
                pending_add2=pending_add2,
            )

        if not get_pp_group().is_last_rank:
            if pending_add is not None:
                hidden_states = hidden_states + pending_add
            if pending_add2 is not None:
                hidden_states = hidden_states + pending_add2
            return IntermediateTensors(
                {"hidden_states": hidden_states, "block_residual": block_residual}
            )
        hidden_states, _ = self.output_attn_res(
            hidden_states, block_residual, pending_add, pending_add2
        )
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_experts + (self.config.num_shared_experts or 0),
        )


class KimiLinearForCausalLM(nn.Module):
    packed_modules_mapping = _kda_packed_modules_mapping([])
    weights_mapping: ClassVar[dict[str, str]] = {
        "weight_packed": "weight",
    }

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = _text_config(atom_config.hf_config)
        _normalize_kimi_config(config)
        self.config = config
        self.quant_config = atom_config.quant_config
        self.packed_modules_mapping = _kda_packed_modules_mapping(
            config.kimi_kda_layers
        )
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
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class KimiK3ForCausalLM(nn.Module):
    skip_weight_prefixes: ClassVar[list[str]] = ["vision_tower.", "mm_projector."]
    quant_exclude_name_mapping: ClassVar[dict[str, str]] = {
        "language_model.model.": "language_model.model.",
        "language_model.lm_head": "language_model.lm_head",
    }
    packed_modules_mapping = KimiLinearForCausalLM.packed_modules_mapping
    weights_mapping = KimiLinearForCausalLM.weights_mapping

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        root_config = atom_config.hf_config
        rebuilt_quant_config = False
        if (
            hasattr(root_config, "text_config")
            and root_config.text_config is not root_config
        ):
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
                rebuilt_quant_config = True
        else:
            _normalize_kimi_config(root_config)
        self.config = _text_config(root_config)
        self.quant_config = atom_config.quant_config
        self.packed_modules_mapping = _kda_packed_modules_mapping(
            self.config.kimi_kda_layers
        )
        if rebuilt_quant_config:
            self.quant_config.remap_layer_name(
                self.config,
                packed_modules_mapping=self.packed_modules_mapping,
                quant_exclude_name_mapping=self.quant_exclude_name_mapping,
            )
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
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # The loader matches expert entries as substrings of full checkpoint
        # names, so keep these generic enough to match each layer's
        # `block_sparse_moe.experts.{id}.w*.weight` entries.
        return self.language_model.get_expert_mapping()


class KimiK3ForConditionalGeneration(KimiK3ForCausalLM):
    """Kimi-K3 with the MoonViT3d vision tower attached.

    Adds `vision_tower` / `mm_projector` next to the language stack, matching
    the checkpoint layout so no weight renaming is needed. Image embeddings are
    produced once per prefill by `ModelRunner.run_model` and scattered over the
    `<|media_pad|>` positions that the input processor expanded.
    """

    # Vision weights belong to this model, so nothing is skipped by default.
    # `__init__` re-adds the skips on pipeline ranks that hold no tower.
    skip_weight_prefixes: ClassVar[list[str]] = []
    vision_weight_prefixes: ClassVar[tuple[str, ...]] = (
        "vision_tower.",
        "mm_projector.",
    )

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__(atom_config, prefix=prefix)

        vision_config = getattr(
            getattr(atom_config, "multimodal_config", None), "vision_config", None
        )
        if vision_config is None:
            raise ValueError(
                "Kimi-K3 needs the full HF config (with `vision_config`) to "
                "build its vision tower. Start the server with "
                "--trust-remote-code."
            )

        # The tower only runs where the token embeddings are produced.
        self.has_vision_tower = get_pp_group().is_first_rank
        if not self.has_vision_tower:
            self.vision_tower = PPMissingLayer()
            self.mm_projector = PPMissingLayer()
            self.skip_weight_prefixes = list(self.vision_weight_prefixes)
            self.media_placeholder_token_id = None
            return

        from atom.models.kimi_k3_vl import build_vision_modules

        self.vision_tower, self.mm_projector = build_vision_modules(vision_config)
        self.media_placeholder_token_id = getattr(
            atom_config.multimodal_config, "media_placeholder_token_id", 163605
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings(input_ids)

    def get_vision_embeddings(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if not self.has_vision_tower:
            raise RuntimeError(
                "Kimi-K3 image embeddings were requested on a pipeline rank "
                "that holds no vision tower; they belong on the first rank."
            )
        return self.mm_projector(self.vision_tower(pixel_values, grid_thw))

    def merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> torch.Tensor:
        mask = input_ids == self.media_placeholder_token_id
        num_placeholders = int(mask.sum())
        if num_placeholders != vision_embeds.shape[0]:
            raise ValueError(
                f"Kimi-K3 got {vision_embeds.shape[0]} image embeddings for "
                f"{num_placeholders} placeholder tokens. The prompt's "
                "`<|media_pad|>` runs must be expanded to (h//2)*(w//2) tokens "
                "per image, and multimodal prefills must not be chunked."
            )
        inputs_embeds[mask] = vision_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # Stay on the inputs_embeds path once vision embeddings exist; the
        # language model would otherwise re-embed input_ids and drop them.
        if inputs_embeds is None and get_pp_group().is_first_rank:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
