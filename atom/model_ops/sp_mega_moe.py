# SPDX-License-Identifier: MIT
"""Opt-in M3 SP4 integration for AITER's MegaMoeTP ag_rs kernel.

The decoder boundary also fuses residual/norm/router and the next input norm.
One runtime is shared by the sequential MoE layers;
its collective buffers and launch epochs must survive layer/graph boundaries.
Large token shards are tiled locally, with the same tiles on every SP rank.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from itertools import pairwise

import torch

from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

logger = logging.getLogger("atom")


def copy_tail(source, source_scale, out, out_scale):
    # Keep dispatch and settings importable without the GPU kernel dependencies.
    from atom.model_ops.sp_mega_buffers import copy_tail as copy

    copy(source, source_scale, out, out_scale)


@dataclass(frozen=True)
class MegaMoESettings:
    max_local_tokens: int = 1024
    act_dtype: str = "fp4"
    comm_dtype: str = "fp8"
    allow_activation_change: bool = False

    def __post_init__(self):
        if self.act_dtype not in ("fp4", "fp8"):
            raise ValueError("ATOM_SP_MOE_MEGA_ACT_DTYPE must be fp4 or fp8")
        if self.comm_dtype not in ("fp8", "bf16"):
            raise ValueError("ATOM_SP_MOE_MEGA_COMM_DTYPE must be fp8 or bf16")
        if not 1 <= self.max_local_tokens <= 8192:
            raise ValueError("ATOM_SP_MOE_MEGA_MAX_LOCAL_TOKENS must be in [1, 8192]")
        if self.act_dtype == "fp8" and self.max_local_tokens > 64:
            raise ValueError(
                "AITER #5937 MXFP8 activation supports at most 256 global tokens; "
                "set ATOM_SP_MOE_MEGA_MAX_LOCAL_TOKENS<=64 for SP4"
            )


def _weights(layer):
    return {
        "w1": layer.w13_weight,
        "w1_scale": layer.w13_weight_scale,
        "w2": layer.w2_weight,
        "w2_scale": layer.w2_weight_scale,
    }


def _validate_layer(layer):
    from aiter import ActivationType, QuantType, dtypes

    from atom.model_ops.moe import Mxfp4MoEMethod

    method = layer.quant_method
    if (
        type(method) is not Mxfp4MoEMethod
        or method.quant_type != QuantType.per_1x32
        or method.quant_dtype != dtypes.fp4x2
        or method.is_guinterleave
        or method.use_triton
        or method.use_triton_decode
        or method.use_triton_ep
        or method.fused_experts is not None
        or method.hidden_pad != 0
        or method.intermediate_pad != 0
        or layer.use_ep
        or layer.reduce_results
        or layer.moe_parallel_config.use_all2all_kernels
        or layer.moe_parallel_config.dp_logical_ratio != 1
        or layer.expert_layout.uses_dispatch_remap
        or layer.expert_layout.shared_is_routed
        or layer.use_grouped_topk
        or layer.custom_routing_function is not None
        or layer.apply_router_weight_on_input
        or layer.activation != ActivationType.Swiglu
        or layer.scoring_func != "sigmoid"
        or not layer.renormalize
        or layer.hidden_size != 6144
        or layer.intermediate_size_per_partition != 768
        or layer.global_num_experts != 128
        or layer.local_num_experts != 129
        or layer.top_k != 4
        or layer.num_fused_shared_experts != 1
        or layer.w13_bias is not None
        or layer.w2_bias is not None
        or layer.w13_input_scale is not None
        or layer.w2_input_scale is not None
    ):
        raise ValueError(
            f"{layer.prefix}: SP MegaMoE requires M3 SP4 MXFP4, separated gate/up "
            "FlyDSL weights, sigmoid top-4 plus one legacy fused shared expert; "
            "disable Triton MoE, GU interleave, EP and dispatch remapping"
        )
    for name, shape in (
        ("w13_weight", (129, 1536, 3072)),
        ("w2_weight", (129, 6144, 384)),
        ("w13_weight_scale", (129 * 1536, 192)),
        ("w2_weight_scale", (129 * 6144, 24)),
    ):
        value = getattr(layer, name)
        if (
            tuple(value.shape) != shape
            or value.element_size() != 1
            or not value.is_contiguous()
            or (name.endswith("weight") and not getattr(value, "is_shuffled", False))
        ):
            raise ValueError(f"{layer.prefix}.{name}: incompatible MegaMoE layout")


class SPMegaMoE:
    """Shared collective runtime; outputs own their storage across layer calls."""

    def __init__(self, runtime, *, allow_activation_change=False):
        self.runtime = runtime
        self.allow_activation_change = allow_activation_change
        self.prefix = None

    def supports(self, layer, hidden_states):
        """Keep the native path whenever MegaMoE would change activation precision.

        Use the padded global row count, exactly as the native SP gather does.
        In particular, M3 decode normally uses BF16 activations, which this
        MegaMoE kernel cannot produce. All ranks make the same shape decision.
        """
        if self.allow_activation_change:
            return True
        from aiter import dtypes
        from aiter.fused_moe import resolve_activation_dtype
        from aiter.ops.flydsl.moe_common import GateMode

        native_dtype = resolve_activation_dtype(
            layer.quant_method.quant_type,
            layer.w13_weight.dtype,
            activation=layer.activation,
            gate_mode=GateMode.SEPARATED,
            M=hidden_states.shape[0] * self.runtime.cfg.world_size,
            hidden_dtype=hidden_states.dtype,
            gfx="gfx950",
        )
        candidate_dtype = (
            dtypes.fp4x2 if self.runtime.cfg.act_dtype == "fp4" else dtypes.fp8
        )
        return native_dtype == candidate_dtype

    @staticmethod
    def _route(layer, hidden_states, router_logits):
        return layer.quant_method.select_experts_with_record(
            layer=layer,
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=layer.use_grouped_topk,
            top_k=layer.top_k,
            renormalize=layer.renormalize,
            topk_group=layer.topk_group,
            num_expert_group=layer.num_expert_group,
            global_num_experts=layer.global_num_experts,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
            e_score_correction_bias=layer.e_score_correction_bias,
            fused_shared_experts_scoring_func=layer.shared_expert_scoring_func,
        )

    def forward(self, layer, hidden_states, router_logits, *, sp_topk=None):
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise ValueError("SP MegaMoE expects BF16 [local_tokens, hidden_size]")
        weights, ids = (
            self._route(layer, hidden_states, router_logits)
            if sp_topk is None
            else sp_topk
        )
        expected = (
            hidden_states.shape[0],
            layer.top_k + layer.num_fused_shared_experts,
        )
        if tuple(weights.shape) != expected or tuple(ids.shape) != expected:
            raise ValueError("SP MegaMoE routing must include the fused shared expert")
        self.runtime.set_weights(**_weights(layer))
        output = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
        tile = self.runtime.max_local_tokens
        for start in range(0, hidden_states.shape[0], tile):
            end = min(start + tile, hidden_states.shape[0])
            self.runtime.forward(
                hidden_states[start:end],
                weights[start:end],
                ids[start:end],
                out=output[start:end],
            )
        return output

    def forward_block(
        self,
        layer,
        attention,
        residual,
        post_weight,
        gate_weight,
        bias,
        next_weight,
        next_eps,
        fp8_output,
    ):
        """Complete SP-local FFN region, returning the NEXT normalized input.

        Attention is already reduced and token-sharded. The prefix uses TP1,
        while the MoE's intermediate shards communicate over SP4. Tail rows
        are replicated scratch in AITER: copy only this rank's rows before
        another tile/layer reuses them. No returned tensor aliases an input.
        """
        from aiter import dtypes

        prefix = self.prefix
        assert prefix is not None
        rows = attention.shape[0]
        hidden = torch.empty_like(attention)
        post_res = torch.empty_like(residual)
        ids = torch.empty((rows, 5), dtype=torch.int32, device=attention.device)
        weights = torch.empty((rows, 5), dtype=torch.float32, device=attention.device)
        aligned = rows - rows % 16
        cuts = list(range(0, aligned, prefix.mmax))
        if aligned < rows:
            cuts.append(aligned)
        cuts.append(rows)
        for start, end in pairwise(cuts):
            sl = slice(start, end)
            if prefix.routes(end - start):
                prefix.forward(
                    attention[sl],
                    residual[sl],
                    post_weight,
                    out=hidden[sl],
                    res_out=post_res[sl],
                    router=(gate_weight, bias, ids[sl], weights[sl]),
                )
            else:
                # The router requires a multiple of 16 local rows. Keep the
                # exact token count/activation decision on remainder tiles.
                x, _, r = _norm_input(
                    attention[sl], residual[sl], post_weight, prefix.eps, False
                )
                logits = torch.nn.functional.linear(x, gate_weight)
                tw, ti = self._route(layer, x, logits)
                hidden[sl].copy_(x)
                post_res[sl].copy_(r)
                weights[sl].copy_(tw)
                ids[sl].copy_(ti)
        if not self.supports(layer, hidden):
            # Native A16W4 decode still consumes the fused routing. Never force
            # A4W4 just to use MegaMoE's tail on a small batch.
            output = layer.forward_impl(hidden, None, sp_topk=(weights, ids))
            return _norm_input(output, post_res, next_weight, next_eps, fp8_output)

        self.runtime.set_weights(**_weights(layer))
        normalized = torch.empty(
            hidden.shape,
            dtype=dtypes.fp8 if fp8_output else hidden.dtype,
            device=hidden.device,
        )
        scale = torch.empty((rows, 1), dtype=torch.float32, device=hidden.device)
        next_res = torch.empty_like(post_res)
        tile = self.runtime.max_local_tokens
        for start in range(0, rows, tile):
            end = min(start + tile, rows)
            sl = slice(start, end)
            m = end - start
            if self.runtime.tail_ok(m) and next_eps == self.runtime.cfg.tail_eps:
                result = self.runtime.forward(
                    hidden[sl],
                    weights[sl],
                    ids[sl],
                    tail=(post_res[sl], next_res[sl], next_weight),
                    bf16=not fp8_output,
                )
                own = slice(self.runtime.cfg.rank * m, (self.runtime.cfg.rank + 1) * m)
                copy_tail(
                    result[1 if fp8_output else 3][own],
                    result[2][own] if fp8_output else None,
                    normalized[sl],
                    scale[sl],
                )
            else:
                # Dynamic schedule / unsupported tail eps: same output
                # contract, with the ordinary fused residual+norm kernel.
                y = self.runtime.forward(hidden[sl], weights[sl], ids[sl])
                x, s, r = _norm_input(
                    y, post_res[sl], next_weight, next_eps, fp8_output
                )
                normalized[sl].copy_(x)
                next_res[sl].copy_(r)
                if fp8_output:
                    scale[sl].copy_(s)
        return normalized, scale, next_res


def _norm_input(x, residual, weight, eps, fp8):
    if fp8:
        from atom.model_ops.minimax_m3.input_norm_fp8 import fused_m3_gemma_norm_fp8

        return fused_m3_gemma_norm_fp8(x, weight, eps, residual)
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

    out, res_out = torch.empty_like(x), torch.empty_like(residual)
    if x.shape[0]:
        fused_qk_rmsnorm_group_quant(
            q=x,
            q_weight=weight,
            q_epsilon=eps,
            q_out_unquantized=out,
            q_res_out=res_out,
            q_residual=residual,
            gemma_norm=True,
        )
    return (
        out,
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
        res_out,
    )


def _block_fake(
    attention: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    next_weight: torch.Tensor,
    layer_name: str,
    next_eps: float,
    fp8_output: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter import dtypes

    return (
        torch.empty_like(
            attention, dtype=dtypes.fp8 if fp8_output else attention.dtype
        ),
        torch.empty(
            (attention.shape[0], 1), device=attention.device, dtype=torch.float32
        ),
        torch.empty_like(residual),
    )


def minimax_m3_sp_mega_block(
    attention: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    next_weight: torch.Tensor,
    layer_name: str,
    next_eps: float,
    fp8_output: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from atom.config import get_current_atom_config

    layer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return layer._sp_mega_moe.forward_block(
        layer,
        attention,
        residual,
        post_weight,
        gate_weight,
        bias,
        next_weight,
        next_eps,
        fp8_output,
    )


# Keep the whole FFN region in one capturable opaque op. Dynamic row decisions
# and shared runtime state must not escape into torch.compile's traced graph.
direct_register_custom_op(
    op_name="minimax_m3_sp_mega_block",
    op_func=minimax_m3_sp_mega_block,
    mutates_args=[],
    fake_impl=_block_fake,
)


def initialize_sp_mega_moe(model, config):
    """After weight processing, before warmup/capture; disabled by default."""
    if config is None or not config.sp_mega_moe_enabled:
        return

    from aiter.dist.parallel_state import get_dp_group, get_tp_group
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.distributed.ulysses_sp import get_sp_group, get_sp_world_size
    from atom.model_ops.moe import FusedMoE
    from atom.plugin.prepare import is_plugin_mode

    if (
        get_sp_world_size() != 4
        or get_tp_group().world_size != 1
        or get_dp_group().world_size != 1
        or config.pipeline_parallel_size != 1
        or config.moe_backend != "standard"
        or config.enable_tbo
        or config.enable_tbo_decode
        or config.enable_rapidserve
        or config.enable_expert_parallel
        or config.eplb_enable
        or config.fake_eplb
        or config.speculative_config is not None
        or config.torch_dtype != torch.bfloat16
        or is_plugin_mode()
        or get_gfx_runtime() != "gfx950"
    ):
        raise ValueError(
            "ATOM_SP_MOE_MEGA=1 requires native gfx950 SP4/TP1/DP1/PP1 BF16 "
            "without TBO, RapidServe, speculative decoding or EPLB"
        )
    settings = MegaMoESettings(
        max_local_tokens=envs.ATOM_SP_MOE_MEGA_MAX_LOCAL_TOKENS,
        act_dtype=envs.ATOM_SP_MOE_MEGA_ACT_DTYPE,
        comm_dtype=envs.ATOM_SP_MOE_MEGA_COMM_DTYPE,
        allow_activation_change=envs.ATOM_SP_MOE_MEGA_ALLOW_ACTIVATION_CHANGE,
    )
    # The plugin decorator constructs the original class, not its lazy wrapper.
    moe_type = getattr(FusedMoE, "__wrapped__", FusedMoE)
    layers = [layer for layer in model.modules() if isinstance(layer, moe_type)]
    if not layers:
        raise ValueError("ATOM_SP_MOE_MEGA=1 found no M3 MoE layers")
    for layer in layers:
        _validate_layer(layer)
    from atom.models.minimax_m3.model import MiniMaxM3DecoderLayer

    decoders = [
        layer
        for layer in model.modules()
        if isinstance(layer, MiniMaxM3DecoderLayer) and layer.is_moe_layer
    ]
    tail_eps = decoders[0].input_layernorm.variance_epsilon if decoders else 1e-6
    if decoders:
        for decoder in decoders:
            moe = decoder.block_sparse_moe
            if (
                decoder.post_attention_layernorm.variance_epsilon != tail_eps
                or moe.gate.weight.dtype != torch.bfloat16
                or not moe.gate.weight.is_contiguous()
                or moe.e_score_correction_bias is None
                or moe.e_score_correction_bias.dtype != torch.float32
                or moe.experts.shared_expert_scoring_func is not None
                or moe.experts.shared_expert_weight != 1.0
                or moe.experts.routed_scaling_factor != layers[0].routed_scaling_factor
            ):
                raise ValueError(
                    "SP MegaMoE fused norm/router requires the M3 BF16 gate, FP32 bias and common norm/router settings"
                )
    limits = {float(layer.swiglu_limit) for layer in layers}
    if len(limits) != 1:
        raise ValueError("SP MegaMoE shared runtime requires one SwiGLU limit")
    try:
        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig
    except ImportError as exc:
        raise RuntimeError(
            "SP MegaMoE requires AITER main with MegaMoeTP (#5937)"
        ) from exc

    group = get_sp_group()
    # M3 prefill needs larger row tiles than AITER's generic MT3 default.
    # Preserve AITER's explicit overrides, including token-dependent tables.
    lb_mt = os.getenv("AITER_MEGAMOE_TP_LB_MT", "6")
    runtime = MegaMoeTP(
        MegaMoeTPConfig(
            rank=group.rank_in_group,
            world_size=group.world_size,
            model_dim=6144,
            inter_dim=768,
            experts=129,
            topk=5,
            max_local_tokens=settings.max_local_tokens,
            activation=layers[0].activation,
            swiglu_limit=limits.pop(),
            comm_mode="ag_rs",
            comm_dtype=settings.comm_dtype,
            act_dtype=settings.act_dtype,
            lb_mt=lb_mt,
            tail_eps=tail_eps,
        ),
        **_weights(layers[0]),
        group=group.cpu_group,
        device=layers[0].w13_weight.device,
    )
    # Prepare each distinct launch variant once. Cover every possible remainder
    # of a local tile so an unseen graph bucket cannot compile during capture.
    representatives = {}
    for rows in range(1, settings.max_local_tokens + 1):
        # Tail eligibility can change inside a launch bucket (TN_MAX).
        key = (
            (runtime.launch_config(rows), runtime.tail_ok(rows))
            if decoders
            else runtime.launch_config(rows)
        )
        representatives.setdefault(key, rows)
    if decoders:
        runtime.prepare(sorted(representatives.values()), tail=True, tail_bf16=True)
    else:
        runtime.prepare(sorted(representatives.values()))
    backend = SPMegaMoE(
        runtime, allow_activation_change=settings.allow_activation_change
    )
    for layer in layers:
        layer._sp_mega_moe = backend
    if decoders:
        from aiter.ops.flydsl.kernels.mega_moe_tp.sp_rs_norm import SpRsNorm

        backend.prefix = SpRsNorm(
            hidden=6144,
            max_tokens=2048,
            eps=tail_eps,
            group=get_tp_group().cpu_group,
            device=layers[0].w13_weight.device,
            router=(128, 4, layers[0].routed_scaling_factor, 1.0),
            gemma=True,
            logit_bf16=True,
        )
        backend.prefix.warm()
        for decoder in decoders:
            decoder._sp_mega_layer_name = decoder.block_sparse_moe.experts.layer_name
        logger.info(
            "SP MegaMoE full fusion: TP1 residual+GemmaNorm+gate+topk prefix; "
            "SP4 MegaMoE tail supplies next GemmaNorm/FP8 (final layer BF16). "
            "Unaligned prefix rows, unsupported tail tiles and native activation "
            "fallbacks retain the ordinary kernels."
        )
    logger.info(
        "SP MegaMoE enabled: %d layers, one shared runtime, tile=%d local tokens, "
        "activation=MX%s, weights=MXFP4, comm=%s, lb_mt=%s, "
        "allow_activation_change=%s; larger shards use multiple launches",
        len(layers),
        settings.max_local_tokens,
        settings.act_dtype.upper(),
        settings.comm_dtype,
        lb_mt,
        settings.allow_activation_change,
    )
