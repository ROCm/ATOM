# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU checks of where `FusedRoute` takes a layer's routing and where
`FusedMoE` uses it. The kernel itself: test_fused_route_gpu.py."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def fr(atom_modules, monkeypatch):
    module = sys.modules["atom.model_ops.fused_moe.fused_route"]
    monkeypatch.setattr(module, "_is_gfx9", lambda: True)
    return SimpleNamespace(module=module, moe=atom_modules.moe)


def _layer(moe, **overrides):
    """A V4.1 main-model routed-expert layer as `FusedRoute` sees it."""
    layout = SimpleNamespace(
        uses_dispatch_remap=True,
        num_redundant=0,
        num_routed=384,
        num_routed_physical=384,
        routed_physical_per_rank=48,
    )
    layer = SimpleNamespace(
        quant_method=object.__new__(moe.Mxfp4MoEMethod),
        expert_layout=layout,
        scoring_func="sqrtsoftplus",
        renormalize=True,
        use_grouped_topk=False,
        e_score_correction_bias=torch.zeros(384, dtype=torch.float32),
        balance_router_logits=None,
        apply_router_weight_on_input=False,
        top_k=6,
        num_fused_shared_experts=1,
        routed_scaling_factor=1.5,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def test_takes_the_v41_main_model_layer(fr, monkeypatch):
    monkeypatch.setattr(fr.module, "_HAS_TRITON", True)
    layer = _layer(fr.moe)
    assert fr.module.unsupported_reason(layer) is None
    route = fr.module.FusedRoute.for_layer(layer, routed_scaling_factor=2.5)
    assert route is not None and route.routed_scale == 2.5
    assert fr.module.FusedRoute.for_layer(layer).routed_scale == 1.5


def test_takes_the_layer_without_a_shared_expert(fr, monkeypatch):
    """48 routed experts a rank, the shared expert run on its own: no
    dispatch remap, the routes stay physical."""
    monkeypatch.setattr(fr.module, "_HAS_TRITON", True)
    layer = _layer(
        fr.moe,
        num_fused_shared_experts=0,
        expert_layout=SimpleNamespace(
            uses_dispatch_remap=False,
            num_redundant=0,
            num_routed=384,
            num_routed_physical=384,
            routed_physical_per_rank=48,
        ),
    )
    assert fr.module.unsupported_reason(layer) is None


def test_eplb_routed_shared_expert_keeps_the_usual_path(fr, monkeypatch):
    monkeypatch.setattr(fr.module, "_HAS_TRITON", True)
    layer = _layer(fr.moe)
    layer.expert_layout.shared_is_routed = True
    assert fr.module.unsupported_reason(layer) == "local replica or no shared"


def test_other_gpus_keep_the_usual_path(fr, monkeypatch):
    """aiter's tie order the kernel reproduces is its wave64 kernel's."""
    monkeypatch.setattr(fr.module, "_HAS_TRITON", True)
    monkeypatch.setattr(fr.module, "_is_gfx9", lambda: False)
    assert fr.module.unsupported_reason(_layer(fr.moe)) == "gfx9"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"quant_method": object()}, "mxfp4 method"),
        (
            {"expert_layout": SimpleNamespace(uses_dispatch_remap=False)},
            "local replica or no shared",
        ),
        ({"scoring_func": "sigmoid"}, "sqrtsoftplus"),
        ({"renormalize": False}, "renormalize"),
        ({"use_grouped_topk": True}, "no grouped top-k"),
        ({"e_score_correction_bias": None}, "routing bias"),
        ({"balance_router_logits": torch.zeros(1)}, "no balance logits"),
        ({"apply_router_weight_on_input": True}, "no router weight on input"),
        ({"top_k": 8}, "width"),
    ],
)
def test_other_routings_keep_the_usual_path(fr, monkeypatch, overrides, reason):
    monkeypatch.setattr(fr.module, "_HAS_TRITON", True)
    layer = _layer(fr.moe, **overrides)
    assert fr.module.unsupported_reason(layer) == reason
    assert fr.module.FusedRoute.for_layer(layer) is None


def test_redundant_experts_and_widths_keep_the_usual_path(fr, monkeypatch):
    monkeypatch.setattr(fr.module, "_HAS_TRITON", True)
    redundant = _layer(fr.moe)
    redundant.expert_layout.num_redundant = 8
    assert fr.module.unsupported_reason(redundant) == "no redundant experts"
    for experts in (32, 256, 512):
        layer = _layer(fr.moe)
        layer.expert_layout.num_routed = experts
        assert fr.module.unsupported_reason(layer) == "experts"


def test_no_triton_keeps_the_usual_path(fr, monkeypatch):
    monkeypatch.setattr(fr.module, "_HAS_TRITON", False)
    assert fr.module.unsupported_reason(_layer(fr.moe)) == "triton"


def test_steps_it_does_not_take_return_none(fr):
    """CPU logits (any host step), image rows, other widths: the usual path."""
    layer = _layer(fr.moe)
    route = fr.module.FusedRoute(layer)
    assert route(torch.zeros(4, 384, dtype=torch.bfloat16)) is None
    assert not fr.module.FusedRoute(layer, applies=lambda: False).runs(
        torch.zeros(4, 384, dtype=torch.bfloat16)
    )
    assert not route.runs(torch.zeros(fr.module.MAX_ROWS + 1, 384))
    assert not route.runs(torch.zeros(4, 256))


def _moe_layer(fused_route, select=None):
    weights, ids = torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int32)
    method = SimpleNamespace(
        select_experts=MagicMock(
            return_value=(weights, ids) if select is None else select
        )
    )
    return SimpleNamespace(quant_method=method, fused_route=fused_route), (weights, ids)


def test_route_places_where_fused_route_takes_the_step(fr):
    placed = (torch.ones(2, 7), torch.zeros(2, 7, dtype=torch.int32))
    fused = MagicMock(return_value=placed)
    layer, _ = _moe_layer(fused)
    logits = torch.zeros(2, 384)
    w, ids, is_placed = fr.moe.FusedMoE._route(layer, None, logits, True)
    assert is_placed and w is placed[0] and ids is placed[1]
    fused.assert_called_once_with(logits)
    layer.quant_method.select_experts.assert_not_called()


@pytest.mark.parametrize("may_place, fused", [(False, True), (True, False)])
def test_route_selects_where_rows_move_or_no_fused_route(fr, may_place, fused):
    """A gather before the experts (`may_place` False) never takes the fused
    routes; a step `FusedRoute` refuses (None) routes the usual way."""
    fused_route = MagicMock(return_value=None) if fused else None
    layer, routes = _moe_layer(fused_route)
    if not may_place:
        layer.fused_route = MagicMock(side_effect=AssertionError("fused"))
    x, logits = torch.zeros(2, 8), torch.zeros(2, 384)
    w, ids, is_placed = fr.moe.FusedMoE._route(layer, x, logits, may_place)
    assert not is_placed and w is routes[0] and ids is routes[1]
    layer.quant_method.select_experts.assert_called_once_with(layer, x, logits)


def test_fused_moe_has_no_fused_route_by_default(fr):
    assert fr.moe.FusedMoE.fused_route is None
