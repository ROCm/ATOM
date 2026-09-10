# SPDX-License-Identifier: Apache-2.0
"""Minimal adapter for AITER's EP16 inter-node MoE operator."""

from __future__ import annotations

import torch

_WIDEEP_CACHE: dict[tuple, object] = {}


def build_wideep_weights(layer) -> None:
    """Convert ATOM's raw MXFP4 weights to TestWideEpMoe's layout."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    local_experts = int(layer.w13_weight.shape[0])
    if local_experts != layer.local_num_experts:
        raise RuntimeError(
            "WideEP weight count does not match the local expert layout: "
            f"weights={local_experts}, local_experts={layer.local_num_experts}"
        )

    layer._wideep_w1 = shuffle_weight_a16w4(
        layer.w13_weight.data, 16, True
    ).contiguous()
    layer._wideep_w1_scale = shuffle_scale_a16w4(
        layer.w13_weight_scale.data.flatten(0, 1), local_experts, True
    ).contiguous()
    layer._wideep_w2 = shuffle_weight_a16w4(
        layer.w2_weight.data, 16, False
    ).contiguous()
    layer._wideep_w2_scale = shuffle_scale_a16w4(
        layer.w2_weight_scale.data.flatten(0, 1), local_experts, False
    ).contiguous()
    layer._wideep_w1.is_shuffled = True
    layer._wideep_w2.is_shuffled = True


def _get_wideep_op(
    *,
    rank: int,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    mtpr: int,
    swiglu_limit: float,
    layer,
):
    """Share the MORI transport and bind the current layer's weights."""
    key = (
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        mtpr,
        swiglu_limit,
    )
    op = _WIDEEP_CACHE.get(key)
    if op is None:
        from aiter.ops.flydsl.test_wide_ep_moe import TestWideEpMoe

        op = TestWideEpMoe(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant="a8w4",
            w1=layer._wideep_w1,
            w1_scale=layer._wideep_w1_scale,
            w2=layer._wideep_w2,
            w2_scale=layer._wideep_w2_scale,
            max_tok_per_rank=mtpr,
            gpu_per_node=8,
            swiglu_limit=swiglu_limit,
            activation="silu",
            gate_mode="interleave",
        )
        _WIDEEP_CACHE[key] = op

    op.w1 = layer._wideep_w1
    op.w1_scale = layer._wideep_w1_scale
    op.w2 = layer._wideep_w2
    op.w2_scale = layer._wideep_w2_scale
    return op


def run_wideep_moe(
    layer,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    mtpr: int,
) -> torch.Tensor:
    """Run AITER dispatch, expert GEMMs, and combine as one eager call."""
    if torch.compiler.is_compiling():
        raise RuntimeError(
            "moe_backend='wideep' currently supports eager execution only; "
            "start ATOM with --enforce-eager --level 0"
        )

    from aiter.dist.parallel_state import get_ep_group

    # Accessing all2all_manager initializes MORI's symmetric heap.
    manager = get_ep_group().device_communicator.all2all_manager
    rank = int(manager.rank)
    world_size = int(manager.world_size)
    if world_size != 16:
        raise RuntimeError(f"WideEP requires EP16, got EP{world_size}")
    if int(hidden_states.shape[0]) > mtpr:
        raise ValueError(
            f"WideEP tokens={hidden_states.shape[0]} exceed max_num_tokens={mtpr}"
        )

    op = _get_wideep_op(
        rank=rank,
        world_size=world_size,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=int(topk_ids.shape[1]),
        mtpr=mtpr,
        swiglu_limit=float(getattr(layer, "swiglu_limit", 0.0)),
        layer=layer,
    )
    with torch.inference_mode(False), torch.no_grad():
        output = op.forward(
            hidden_states.contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            topk_ids.to(torch.int32).contiguous(),
        )
    # MORI's result aliases its reusable combine arena.
    return output.clone()


class WideEpFusedExperts:
    """Adapter for Mxfp4MoEMethod's whole-pipeline experts hook."""

    def __init__(
        self, layer, *, model_dim: int, inter_dim: int, experts: int, mtpr: int
    ):
        self.layer = layer
        self.model_dim = model_dim
        self.inter_dim = inter_dim
        self.experts = experts
        self.mtpr = mtpr

    def __call__(
        self,
        *,
        hidden_states,
        topk_weights,
        topk_ids,
        activation=None,
        apply_router_weight_on_input=False,
        **_ignored,
    ):
        from aiter import ActivationType

        if apply_router_weight_on_input:
            raise NotImplementedError(
                "WideEP does not support apply_router_weight_on_input=True"
            )
        if activation is not None and activation != ActivationType.Silu:
            raise NotImplementedError("WideEP A8W4 supports SiLU only")
        return run_wideep_moe(
            self.layer,
            hidden_states,
            topk_weights,
            topk_ids,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            experts=self.experts,
            mtpr=self.mtpr,
        )
