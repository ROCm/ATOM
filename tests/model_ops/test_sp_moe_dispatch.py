# SPDX-License-Identifier: MIT
"""CPU coverage for SP MoE routing and transport selection."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_comm_fused_moe import atom_modules  # noqa: F401


@pytest.fixture
def sp_modules(request):
    # Preserve the external types used by an already imported MoE module.
    # Replacing AITER then would create a second, incompatible QuantType enum.
    moe = sys.modules.get("atom.model_ops.moe")
    if moe is None:
        moe = request.getfixturevalue("atom_modules").moe
    return SimpleNamespace(moe=moe)


def _layer(moe):
    method = object.__new__(moe.Mxfp4MoEMethod)
    method.quant_type = moe.QuantType.per_1x32
    method.fp8_activations = False
    return SimpleNamespace(
        quant_method=method,
        moe_parallel_config=SimpleNamespace(
            dp_logical_ratio=1, use_all2all_kernels=False
        ),
        expert_layout=SimpleNamespace(
            shared_is_routed=False, uses_dispatch_remap=False
        ),
        use_ep=False,
        use_grouped_topk=False,
        scoring_func="sigmoid",
        hidden_size=6144,
        intermediate_size_per_partition=768,
        global_num_experts=128,
        local_num_experts=129,
        top_k=4,
        num_fused_shared_experts=1,
        renormalize=True,
        expert_map=None,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        activation=moe.ActivationType.Swiglu,
        apply_router_weight_on_input=False,
        reduce_results=False,
        prefix="experts",
    )


@pytest.mark.parametrize("prequantize", [False, True])
def test_sp_routes_owned_rows_then_gathers_them_once(
    monkeypatch, sp_modules, prequantize
):
    moe = sp_modules.moe
    layer = _layer(moe)
    method = layer.quant_method
    rows, world, hidden_size = 8, 4, 64
    hidden = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
    logits = torch.empty(rows, 4)
    weights = torch.rand(rows, 5)
    ids = torch.randint(0, 129, (rows, 5), dtype=torch.int32)
    # MXFP4 rows and their e8m0 scales, as bytes.
    fp4 = torch.randint(0, 255, (rows, hidden_size // 2), dtype=torch.uint8)
    scale = torch.randint(0, 255, (rows, hidden_size // 32), dtype=torch.uint8)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.routed_rows_wire = None
    method._fused_moe_prequantizes = Mock(return_value=prequantize)
    method._fp4_rows_wire = SimpleNamespace(quantize=Mock(return_value=(fp4, scale)))
    method.select_experts = Mock(return_value=(weights, ids))
    layer._run_experts = Mock(return_value=torch.empty(rows * world, hidden_size))
    group = SimpleNamespace(world_size=world)
    gatherv = Mock(side_effect=lambda ts, sizes, g: [t.repeat(world, 1) for t in ts])
    scatter = Mock(side_effect=lambda value: value[:rows])
    reassign = Mock()
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: True)
    monkeypatch.setattr(moe, "get_sp_group", lambda: group)
    monkeypatch.setattr(moe, "all_gatherv", gatherv)
    monkeypatch.setattr(moe, "reassign_gathered_shared_ids", reassign)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    assert moe.FusedMoE.forward_impl(layer, hidden, logits).shape == hidden.shape

    # Routing runs on the owner's rows; one collective moves everything.
    method.select_experts.assert_called_once_with(layer, hidden, logits)
    method._fused_moe_prequantizes.assert_called_once_with(
        layer, rows * world, torch.bfloat16
    )
    gatherv.assert_called_once()
    assert gatherv.call_args.args[1:] == ([rows] * world, group)
    x, topk_weights, topk_ids, x_scale = layer._run_experts.call_args.args
    assert layer._run_experts.call_args.kwargs == {"gathered": True}
    assert torch.equal(topk_weights, weights.repeat(world, 1))
    assert topk_ids.dtype == torch.int32
    assert torch.equal(topk_ids, ids.repeat(world, 1))
    reassign.assert_called_once_with(layer, topk_ids)
    if prequantize:
        assert torch.equal(x, fp4.repeat(world, 1))
        assert torch.equal(x_scale, scale.repeat(world, 1))
    else:
        method._fp4_rows_wire.quantize.assert_not_called()
        assert torch.equal(x, hidden.repeat(world, 1))
        assert x_scale is None
    scatter.assert_called_once_with(layer._run_experts.return_value)


def test_non_sp_skips_sp_gather(monkeypatch, sp_modules):
    moe = sp_modules.moe
    layer = _layer(moe)
    hidden, logits = torch.empty(4, 2), torch.empty(4, 4)
    weights, ids = object(), object()
    layer.quant_method.select_experts = Mock(return_value=(weights, ids))
    layer._run_experts = Mock(return_value=hidden)
    gather, scatter = Mock(), Mock()
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(moe, "all_gatherv", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    assert moe.FusedMoE.forward_impl(layer, hidden, logits) is hidden

    gather.assert_not_called()
    scatter.assert_not_called()
    layer._run_experts.assert_called_once_with(
        hidden, weights, ids, None, gathered=False
    )
