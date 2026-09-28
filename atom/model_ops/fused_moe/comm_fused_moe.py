# SPDX-License-Identifier: MIT
"""Optional communication-fused backend for :class:`FusedMoE`."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from atom.config import get_current_atom_config
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

if TYPE_CHECKING:
    from atom.model_ops.moe import FusedMoE


def _load_backend() -> tuple[ModuleType, ModuleType] | None:
    try:
        host = importlib.import_module("aiter.ops.flydsl.comm_fused_moe_host")
        runtime = importlib.import_module("aiter.ops.comm_fused_moe_runtime")
    except (ImportError, OSError):
        return None

    if not all(
        hasattr(host, name)
        for name in ("ShapeKey", "winners_for", "create_flydsl_comm_fused_runners")
    ) or not hasattr(runtime, "CommFusedMoeRuntime"):
        return None
    return host, runtime


def create_comm_fused_moe_backend(
    *,
    layer_quant_config: Any,
    online_quant: bool,
    parallel_config: Any,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    activation: ActivationType,
    apply_router_weight_on_input: bool,
) -> CommFusedMoeBackend | None:
    """Return a configured backend when this FusedMoE layout is supported."""
    config = get_current_atom_config()
    if (
        config.moe_backend != "standard"
        or online_quant
        or layer_quant_config is None
        or layer_quant_config.quant_dtype != dtypes.fp4x2
        or layer_quant_config.quant_type != QuantType.per_1x32
        or config.torch_dtype != torch.bfloat16
        or activation != ActivationType.Silu
        or not envs.ATOM_MOE_GU_ITLV
        or apply_router_weight_on_input
        or os.getenv("AITER_DISABLE_COMM_FUSED_MOE") == "1"
        or config.enable_tbo
        or config.enable_rapidserve
        or config.fake_eplb
        or config.enable_expert_parallel
        or parallel_config.dp_size != 1
        or parallel_config.use_ep
        or config.prefill_context_parallel_size != 1
        or parallel_config.tp_size == 1
    ):
        return None

    modules = _load_backend()
    if modules is None:
        return None
    host, runtime = modules
    try:
        winners = host.winners_for(
            host.ShapeKey(
                get_gfx_runtime(),
                model_dim,
                inter_dim,
                experts,
                topk,
                parallel_config.tp_size,
            )
        )
    except KeyError:
        return None
    return CommFusedMoeBackend(host, runtime) if winners else None


class CommFusedMoeBackend:
    """Communication-fused execution plugged into an ordinary FusedMoE."""

    def __init__(self, host: ModuleType, runtime: ModuleType) -> None:
        self.host = host
        self.runtime_module = runtime
        self.runtime = None

    def initialize(self, layer: FusedMoE) -> None:
        # Importing here avoids a cycle while atom.model_ops.moe defines FusedMoE.
        from atom.model_ops.moe import Mxfp4MoEMethod

        method = layer.quant_method
        if not isinstance(method, Mxfp4MoEMethod):
            raise TypeError("Communication-fused MoE requires MXFP4 weights")
        tp_size = int(get_tp_group().world_size)
        if (
            tp_size == 1
            or layer.dp_size != 1
            or layer.use_ep
            or layer.tp_size != tp_size
        ):
            raise ValueError(
                "Communication-fused MoE requires an unflattened TP-only layout: "
                f"tp_group={tp_size}, moe_tp={layer.tp_size}, "
                f"dp={layer.dp_size}, use_ep={layer.use_ep}"
            )

        method.use_triton = False
        method.use_triton_decode = False
        self.runtime = self.runtime_module.CommFusedMoeRuntime(
            runners=self.host.create_flydsl_comm_fused_runners(
                tp_group=get_tp_group(),
                model_dim=layer.hidden_size,
                inter_dim=layer.intermediate_size_per_partition,
                experts=layer.global_num_experts,
                topk=layer.top_k,
            )
        )

    def supports(self, tokens: int) -> bool:
        return self.runtime is not None and self.runtime.supports(tokens)

    def forward(
        self,
        layer: FusedMoE,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor | None,
        before_stage2: Callable[[], torch.Tensor] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if before_stage2 is not None:
            return self.forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_partial,
                before_stage2=before_stage2,
                stage2_stream=stage2_stream,
            )
        if shared_partial is None:
            raise ValueError("Communication-fused MoE requires a shared partial")
        return torch.ops.aiter.comm_fused_moe_forward(
            hidden_states,
            router_logits,
            shared_partial,
            layer.layer_name,
        )

    def forward_impl(
        self,
        layer: FusedMoE,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor | None,
        before_stage2: Callable[[], torch.Tensor] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        method = layer.quant_method
        topk_weights, topk_ids = method.select_experts_with_record(
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
        return self.runtime.run(
            hidden_states=hidden_states,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            expert_mask=layer.expert_mask,
            activation=layer.activation,
            quant_type=method.quant_type,
            doweight_stage1=layer.apply_router_weight_on_input,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            hidden_pad=method.hidden_pad,
            intermediate_pad=method.intermediate_pad,
            bias1=layer.w13_bias,
            bias2=layer.w2_bias,
            swiglu_limit=float(layer.swiglu_limit),
            gate_mode=(
                GateMode.INTERLEAVE.value
                if method.is_guinterleave
                else GateMode.SEPARATED.value
            ),
            shared_partial=shared_partial,
            before_stage2=before_stage2,
            stage2_stream=stage2_stream,
        )


def comm_fused_moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_partial: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return layer._comm_fused_moe.forward_impl(
        layer, hidden_states, router_logits, shared_partial
    )


def _comm_fused_moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_partial: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="comm_fused_moe_forward",
    op_func=comm_fused_moe_forward,
    mutates_args=[],
    fake_impl=_comm_fused_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


__all__ = ["CommFusedMoeBackend", "create_comm_fused_moe_backend"]
