# SPDX-License-Identifier: Apache-2.0
"""AITER TestWideEpMoe integration for the EP16 inter-node path.

The AITER operator owns the complete routed-MoE step: FP8 dispatch through
MORI InterNodeV1LL, both MXFP4 expert GEMMs, and combine.  It therefore plugs
into ATOM as a whole-pipeline ``fused_experts`` backend, beside MegaMoEV2,
rather than through the standard prepare/GEMM/finalize seam.
"""

from __future__ import annotations

import torch

from atom.utils.custom_register import direct_register_custom_op

_WIDEEP_CACHE: dict = {}


def build_wideep_weights(layer) -> None:
    """Build TestWideEpMoe's expert-major A16W4 shuffled weight layout."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    w1 = layer.w13_weight.data
    local_experts = int(w1.shape[0])
    if local_experts != layer.local_num_experts:
        raise RuntimeError(
            "WideEP local weight width disagrees with the dispatch layout: "
            f"weights={local_experts}, dispatch={layer.local_num_experts}"
        )
    layer._wideep_w1 = shuffle_weight_a16w4(w1, 16, True).contiguous()

    w1_scale = layer.w13_weight_scale.data
    w1_scale = w1_scale.reshape(local_experts * w1_scale.shape[1], w1_scale.shape[2])
    layer._wideep_w1_scale = shuffle_scale_a16w4(
        w1_scale, local_experts, True
    ).contiguous()

    w2 = layer.w2_weight.data
    layer._wideep_w2 = shuffle_weight_a16w4(w2, 16, False).contiguous()
    w2_scale = layer.w2_weight_scale.data
    w2_scale = w2_scale.reshape(local_experts * w2_scale.shape[1], w2_scale.shape[2])
    layer._wideep_w2_scale = shuffle_scale_a16w4(
        w2_scale, local_experts, False
    ).contiguous()
    # TestWideEpMoe enters the generic fused_moe frontend, which uses this
    # marker to recognize that both weights already have kernel-ready layout.
    layer._wideep_w1.is_shuffled = True
    layer._wideep_w2.is_shuffled = True


def get_or_build_wideep_moe(
    *,
    rank: int,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str,
    mtpr: int,
    swiglu_limit: float,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
):
    """Return the process-wide operator for this shape and bind layer weights."""
    key = (
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        mtpr,
        swiglu_limit,
    )
    op = _WIDEEP_CACHE.get(key)
    if op is None:
        try:
            from aiter.ops.flydsl.test_wide_ep_moe import TestWideEpMoe
        except ImportError as exc:
            raise RuntimeError(
                "The EP16 WideEP path requires AITER dev/test_wide_ep_moe "
                "with aiter.ops.flydsl.test_wide_ep_moe.TestWideEpMoe"
            ) from exc

        with torch.inference_mode(False), torch.no_grad():
            op = TestWideEpMoe(
                rank=rank,
                world_size=world_size,
                model_dim=model_dim,
                inter_dim=inter_dim,
                experts=experts,
                topk=topk,
                quant=quant,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=mtpr,
                gpu_per_node=8,
                swiglu_limit=swiglu_limit,
                activation="silu",
                gate_mode="interleave",
            )
        _WIDEEP_CACHE[key] = op

    # One operator is shared by shape-identical MoE layers.  TestWideEpMoe
    # reads these tensors at execution time, so bind the current layer on every
    # call.  The shuffled tensors keep stable storage across graph replay.
    op.w1 = w1
    op.w1_scale = w1_scale
    op.w2 = w2
    op.w2_scale = w2_scale
    return op


def atom_wideep_forward_impl(
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    rank: int,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    mtpr: int,
    swiglu_limit: float,
) -> torch.Tensor:
    """Dynamo boundary with layer weights explicit in the graph."""
    op = get_or_build_wideep_moe(
        rank=rank,
        world_size=world_size,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant="a8w4",
        mtpr=mtpr,
        swiglu_limit=swiglu_limit,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
    )
    # The MORI combine result aliases its symmetric arena.  Materialize a
    # graph-owned output before another MoE layer reuses the shared transport.
    return op.forward_prequant(x_quant, x_scale, weights, ids).clone()


def atom_wideep_forward_fake(
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    rank: int,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    mtpr: int,
    swiglu_limit: float,
) -> torch.Tensor:
    del (
        x_scale,
        weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
        rank,
        world_size,
        inter_dim,
        experts,
        topk,
        mtpr,
        swiglu_limit,
    )
    return x_quant.new_empty((ids.shape[0], model_dim), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="atom_wideep_forward",
    op_func=atom_wideep_forward_impl,
    mutates_args=[],
    fake_impl=atom_wideep_forward_fake,
)


def run_wideep_moe(
    layer,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    mtpr: int,
    swiglu_limit: float,
    quant: str = "a8w4",
) -> torch.Tensor:
    """Run the complete EP16 InterNodeV1LL routed-MoE pipeline."""
    from aiter.dist.parallel_state import get_ep_group

    # This property initializes MORI's symmetric heap as a required side effect.
    all2all_manager = get_ep_group().device_communicator.all2all_manager
    rank = int(all2all_manager.rank)
    world_size = int(all2all_manager.world_size)
    if world_size != 16:
        raise RuntimeError(
            f"WideEP was selected for EP16, but the live EP group has {world_size} ranks"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "WideEP requires BF16 hidden states before its internal FP8 quantization, "
            f"got {hidden_states.dtype}"
        )
    if int(hidden_states.shape[0]) > mtpr:
        raise ValueError(
            f"WideEP run_tokens={hidden_states.shape[0]} exceeds max_num_tokens={mtpr}"
        )

    op = get_or_build_wideep_moe(
        rank=rank,
        world_size=world_size,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=quant,
        mtpr=mtpr,
        swiglu_limit=float(swiglu_limit),
        w1=layer._wideep_w1,
        w1_scale=layer._wideep_w1_scale,
        w2=layer._wideep_w2,
        w2_scale=layer._wideep_w2_scale,
    )

    hidden_states = hidden_states.contiguous()
    weights = topk_weights.to(torch.float32).contiguous()
    ids = topk_ids.to(torch.int32).contiguous()
    with torch.inference_mode(False), torch.no_grad():
        x_quant, x_scale = op.quantize(hidden_states)
        if torch.compiler.is_compiling():
            return torch.ops.aiter.atom_wideep_forward(
                x_quant,
                x_scale,
                weights,
                ids,
                layer._wideep_w1,
                layer._wideep_w1_scale,
                layer._wideep_w2,
                layer._wideep_w2_scale,
                rank,
                world_size,
                model_dim,
                inter_dim,
                experts,
                topk,
                mtpr,
                float(swiglu_limit),
            )
        return op.forward_prequant(x_quant, x_scale, weights, ids)


class WideEpFusedExperts:
    """Whole-pipeline ``fused_experts`` adapter for AITER TestWideEpMoe."""

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        model_dim: int,
        inter_dim: int,
        experts: int,
        mtpr: int,
        quant: str = "a8w4",
    ) -> None:
        self._layer = layer
        self._model_dim = model_dim
        self._inter_dim = inter_dim
        self._experts = experts
        self._mtpr = mtpr
        self._quant = quant

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor | None = None,
        w2: torch.Tensor | None = None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int = -1,
        activation=None,
        apply_router_weight_on_input: bool = False,
        expert_map: torch.Tensor | None = None,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        **_ignored,
    ) -> torch.Tensor:
        from aiter import ActivationType

        if apply_router_weight_on_input:
            raise NotImplementedError(
                "WideEP does not support apply_router_weight_on_input=True"
            )
        if activation is not None and activation != ActivationType.Silu:
            raise NotImplementedError(
                f"WideEP A8W4 requires SiLU activation, got {activation}"
            )
        if bias1 is not None or bias2 is not None:
            raise NotImplementedError("WideEP does not support expert bias")
        if hidden_pad or intermediate_pad:
            raise NotImplementedError(
                "WideEP requires model and intermediate dimensions with no padding"
            )
        if global_num_experts not in (-1, self._experts):
            raise RuntimeError(
                "WideEP routing width disagrees with its configured experts: "
                f"routing={global_num_experts}, configured={self._experts}"
            )

        # w1/w2 are emptied staging parameters.  The dedicated shuffled weights
        # live on the layer; expert_map is already reflected in global topk ids.
        del w1, w2, expert_map
        local_weight_experts = int(self._layer._wideep_w1.shape[0])
        if local_weight_experts != self._layer.local_num_experts:
            raise RuntimeError(
                "WideEP weight/dispatch layout changed after preparation: "
                f"weights={local_weight_experts}, "
                f"expected={self._layer.local_num_experts}"
            )

        return run_wideep_moe(
            self._layer,
            hidden_states,
            topk_weights,
            topk_ids,
            model_dim=self._model_dim,
            inter_dim=self._inter_dim,
            experts=self._experts,
            topk=int(topk_ids.shape[1]),
            mtpr=self._mtpr,
            swiglu_limit=getattr(self._layer, "swiglu_limit", 0.0),
            quant=self._quant,
        )
