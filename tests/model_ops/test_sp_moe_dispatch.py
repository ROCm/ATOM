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


@pytest.mark.parametrize(
    "rows,override,prequantize,local_routing",
    [
        (8192, {}, True, True),
        (8191, {}, True, False),
        (8192, {"use_ep": True}, True, False),
        (8192, {}, False, False),
    ],
)
def test_forward_selects_local_routing_by_capability(
    monkeypatch, sp_modules, rows, override, prequantize, local_routing
):
    moe = sp_modules.moe
    layer = _layer(moe)
    vars(layer).update(override)
    method = layer.quant_method
    hidden = torch.empty(rows, 2, dtype=torch.bfloat16)
    logits = torch.empty(rows, 4)
    gathered = torch.empty(rows * 4, 2, dtype=torch.bfloat16)
    scale, weights, ids = (object() for _ in range(3))
    method._sp_input_can_prequantize = Mock(return_value=prequantize)
    method.gather_sp_input = Mock(return_value=(gathered, scale))
    method.gather_sp_routed_input = Mock(return_value=(gathered, scale, weights, ids))
    method.select_experts = Mock(return_value=(weights, ids))
    layer._run_experts = Mock(return_value=gathered)
    gather = Mock(side_effect=lambda value, group=None: value.repeat(4, 1))
    scatter = Mock(side_effect=lambda value: value[:rows])
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: True)
    monkeypatch.setattr(moe, "_all_gather_tokens", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    result = moe.FusedMoE.forward_impl(layer, hidden, logits)

    assert result.shape == hidden.shape
    # Gathered rows are the SP group's: no per-rank row count applies to them.
    layer._run_experts.assert_called_once_with(
        gathered, weights, ids, scale, gathered=True
    )
    scatter.assert_called_once_with(gathered)
    if local_routing:
        method.gather_sp_routed_input.assert_called_once_with(
            layer, hidden, logits, token_group=None
        )
        method.gather_sp_input.assert_not_called()
        method.select_experts.assert_not_called()
        gather.assert_not_called()
    else:
        method.gather_sp_routed_input.assert_not_called()
        method.gather_sp_input.assert_called_once_with(layer, hidden, token_group=None)
        gather.assert_called_once_with(logits, group=None)
        assert method.select_experts.call_args.args[1] is gathered
        assert method.select_experts.call_args.args[2].shape[0] == rows * 4


@pytest.mark.parametrize("sp_enabled,all2all", [(False, False), (True, True)])
def test_non_sp_and_routed_all2all_skip_sp_gather(
    monkeypatch, sp_modules, sp_enabled, all2all
):
    moe = sp_modules.moe
    layer = _layer(moe)
    layer.moe_parallel_config.use_all2all_kernels = all2all
    hidden, logits = torch.empty(4, 2), torch.empty(4, 4)
    weights, ids = object(), object()
    layer.quant_method.select_experts = Mock(return_value=(weights, ids))
    layer._run_experts = Mock(return_value=hidden)
    gather, scatter = Mock(), Mock()
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "sp_is_enabled", lambda: sp_enabled)
    monkeypatch.setattr(moe, "_all_gather_tokens", gather)
    monkeypatch.setattr(moe, "sp_moe_reduce_scatter", scatter)

    assert moe.FusedMoE.forward_impl(layer, hidden, logits) is hidden

    gather.assert_not_called()
    scatter.assert_not_called()
    layer._run_experts.assert_called_once_with(
        hidden, weights, ids, None, gathered=False
    )
