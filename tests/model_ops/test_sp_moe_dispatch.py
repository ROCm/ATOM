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


def test_sp_gathers_every_pass_once_and_scatters_it_back(monkeypatch, sp_modules):
    moe = sp_modules.moe
    rows, world, hidden_size = 8, 4, 64
    method = SimpleNamespace(
        routed_rows_wire=None,
        place_experts=Mock(side_effect=lambda layer, w, i, gathered: (w, i)),
        apply=Mock(return_value=torch.empty(rows * world, hidden_size)),
    )
    layer = SimpleNamespace(
        quant_method=method,
        moe_parallel_config=SimpleNamespace(pcp_merge=True),
        use_ep=False,
        global_num_experts=128,
        expert_map=None,
        activation=None,
        apply_router_weight_on_input=False,
        prefix="experts",
    )
    hidden = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
    weights = torch.rand(rows, 5)
    ids = torch.randint(0, 129, (rows, 5), dtype=torch.int32)
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
    # One collective moves the rows, as they are.
    gatherv.assert_called_once()
    assert gatherv.call_args.args[1:] == ([rows] * world, group)
    applied = method.apply.call_args.kwargs
    assert method.place_experts.call_args.args[3] is True
    assert applied["scheduled_tokens_gpu"] is None
    assert torch.equal(applied["x"], hidden.repeat(world, 1))
    assert applied["x_scale"] is None
    assert torch.equal(applied["topk_weights"], weights.repeat(world, 1))
    assert torch.equal(applied["topk_ids"], ids.repeat(world, 1))
    reassign.assert_called_once_with(layer, applied["topk_ids"])
    scatter.assert_called_once_with(method.apply.return_value)
