# SPDX-License-Identifier: MIT
"""ATOM model adapter for communication-compute fused MoE."""

from __future__ import annotations

from collections.abc import Callable
import os

import torch
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from atom.config import get_current_atom_config
from atom.model_ops.moe import FusedMoE, Mxfp4MoEMethod
from atom.utils.custom_register import direct_register_custom_op


class CommFusedMoe(FusedMoE):
    """FusedMoE weights/routing with Stage2 + TP communication owned by AITer."""

    @staticmethod
    def supports_model(
        *, model_dim: int, inter_dim: int, experts: int, topk: int
    ) -> bool:
        config = get_current_atom_config()
        if (
            os.getenv("AITER_DISABLE_COMM_FUSED_MOE") == "1"
            or config.enable_expert_parallel
            or config.parallel_config.data_parallel_size != 1
            or config.prefill_context_parallel_size != 1
        ):
            return False

        from aiter.ops.flydsl.comm_fused_moe_host import ShapeKey, winners_for

        try:
            return bool(
                winners_for(
                    ShapeKey(
                        get_gfx_runtime(),
                        model_dim,
                        inter_dim,
                        experts,
                        topk,
                        int(get_tp_group().world_size),
                    )
                )
            )
        except KeyError:
            return False

    def _validate_parallel_layout(self) -> None:
        tp_size = int(get_tp_group().world_size)
        if (
            tp_size == 1
            or self.dp_size != 1
            or self.use_ep
            or self.tp_size != tp_size
        ):
            raise ValueError(
                "CommFusedMoe requires an unflattened TP-only MoE layout: "
                f"tp_group={tp_size}, moe_tp={self.tp_size}, "
                f"dp={self.dp_size}, use_ep={self.use_ep}"
            )

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        method = self.quant_method
        if not isinstance(method, Mxfp4MoEMethod):
            raise TypeError("CommFusedMoe currently requires MXFP4 MoE weights")
        self._validate_parallel_layout()

        method.use_triton = False
        method.use_triton_decode = False

        from aiter.ops.comm_fused_moe_runtime import CommFusedMoeRuntime
        from aiter.ops.flydsl.comm_fused_moe_host import (
            create_flydsl_comm_fused_runners,
        )

        self._comm_fused_runtime = CommFusedMoeRuntime(
            runners=create_flydsl_comm_fused_runners(
                tp_group=get_tp_group(),
                model_dim=self.hidden_size,
                inter_dim=self.intermediate_size_per_partition,
                experts=self.global_num_experts,
                topk=self.top_k,
            )
        )

    def supports_comm_fused(self, tokens: int) -> bool:
        return self._comm_fused_runtime.supports(tokens)

    def forward_comm_fused(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.aiter.comm_fused_moe_forward(
            hidden_states,
            router_logits,
            shared_partial,
            self.layer_name,
        )

    def forward_comm_fused_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor | None,
        before_stage2: Callable[[], torch.Tensor] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        method = self.quant_method
        topk_weights, topk_ids = method.select_experts_with_record(
            layer=self,
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.top_k,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            global_num_experts=self.global_num_experts,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            fused_shared_experts_scoring_func=self.shared_expert_scoring_func,
        )
        return self._comm_fused_runtime.run(
            hidden_states=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            expert_mask=self.expert_mask,
            activation=self.activation,
            quant_type=method.quant_type,
            doweight_stage1=self.apply_router_weight_on_input,
            w1_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
            a1_scale=self.w13_input_scale,
            a2_scale=self.w2_input_scale,
            hidden_pad=method.hidden_pad,
            intermediate_pad=method.intermediate_pad,
            bias1=self.w13_bias,
            bias2=self.w2_bias,
            swiglu_limit=float(self.swiglu_limit),
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
    return layer.forward_comm_fused_impl(
        hidden_states, router_logits, shared_partial
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
    mutates_args=["shared_partial"],
    fake_impl=_comm_fused_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


__all__ = ["CommFusedMoe"]
