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
def test_sp_gathers_every_pass_once_and_scatters_it_back(
    monkeypatch, sp_modules, prequantize
):
    moe = sp_modules.moe
    layer = _layer(moe)
    layer.moe_parallel_config.pcp_merge = True
    method = layer.quant_method
    rows, world, hidden_size = 8, 4, 64
    hidden = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(rows, 5)
    ids = torch.randint(0, 129, (rows, 5), dtype=torch.int32)
    # MXFP4 rows and their e8m0 scales, as bytes.
    fp4 = torch.randint(0, 255, (rows, hidden_size // 2), dtype=torch.uint8)
    scale = torch.randint(0, 255, (rows, hidden_size // 32), dtype=torch.uint8)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.routed_rows_wire = None
    method._fused_moe_prequantizes = Mock(return_value=prequantize)
    method._fp4_rows_wire = SimpleNamespace(quantize=Mock(return_value=(fp4, scale)))
    method.place_experts = Mock(side_effect=lambda layer, w, i, gathered: (w, i))
    method.apply = Mock(return_value=torch.empty(rows * world, hidden_size))
    group = SimpleNamespace(world_size=world)
    gatherv = Mock(side_effect=lambda ts, sizes, g: [t.repeat(world, 1) for t in ts])
    scatter = Mock(side_effect=lambda value: value[:rows])
    reassign = Mock()
    # A dummy pass that ordinary PCP would neither split nor reduce.
    context = SimpleNamespace(is_dummy_run=True, pcp_split=False)
    monkeypatch.setattr(
        moe, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: True)
    monkeypatch.setattr(moe, "get_pcp_group", lambda: group)
    monkeypatch.setattr(moe, "all_gatherv", gatherv)
    monkeypatch.setattr(moe, "reassign_gathered_shared_ids", reassign)
    monkeypatch.setattr(moe, "pcp_reduce_scatter", scatter)
    monkeypatch.setattr(moe, "tbo_active", lambda: False)

    out = moe.FusedMoE._run_experts(layer, hidden, weights, ids)

    assert out.shape == hidden.shape
    # The wire is chosen for the gathered rows; one collective moves them.
    method._fused_moe_prequantizes.assert_called_once_with(
        layer, rows * world, torch.bfloat16
    )
    gatherv.assert_called_once()
    assert gatherv.call_args.args[1:] == ([rows] * world, group)
    applied = method.apply.call_args.kwargs
    assert method.place_experts.call_args.args[3] is True
    assert applied["scheduled_tokens_gpu"] is None
    assert torch.equal(applied["topk_weights"], weights.repeat(world, 1))
    assert applied["topk_ids"].dtype == torch.int32
    assert torch.equal(applied["topk_ids"], ids.repeat(world, 1))
    reassign.assert_called_once_with(layer, applied["topk_ids"])
    if prequantize:
        assert torch.equal(applied["x"], fp4.repeat(world, 1))
        assert torch.equal(applied["x_scale"], scale.repeat(world, 1))
    else:
        method._fp4_rows_wire.quantize.assert_not_called()
        assert torch.equal(applied["x"], hidden.repeat(world, 1))
        assert applied["x_scale"] is None
    scatter.assert_called_once_with(method.apply.return_value)
