"""ATOM_MOE_FUSED_TOPK_ROUTE_QUANT defers only where nothing reads the routing
before aiter's grouped MoE, with the pass's device row count."""

from types import SimpleNamespace

import pytest
import torch


def _layer():
    return SimpleNamespace(
        moe_parallel_config=SimpleNamespace(pcp_merge=False),
        dp_size=1,
        expert_mask=None,
        balance_router_logits=None,
        _comm_fused_moe=None,
        top_k=8,
        renormalize=True,
        scoring_func="softmax",
        use_grouped_topk=False,
        custom_routing_function=None,
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        num_fused_shared_experts=0,
        expert_layout=SimpleNamespace(
            num_routed=256, uses_dispatch_remap=False, shared_is_routed=False
        ),
    )


@pytest.fixture
def env(monkeypatch):
    import sys
    import types

    import atom.utils.forward_context as fc
    from atom import config
    from atom.utils import envs
    from atom.utils.tbo import ubatching

    calls = []
    fake = types.ModuleType("aiter.ops.flydsl.grouped_moe_gfx1250")
    fake.defer_topk_softmax = lambda w, i, g, rn, nvr: calls.append((w, i, g, rn, nvr))
    fake.fused_topk_route_quant_supported = lambda g, x, k, rn: True
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.grouped_moe_gfx1250", fake)
    valid = torch.tensor([3], dtype=torch.int32)
    context = SimpleNamespace(
        running_tokens=4, scheduled_tokens_gpu=valid, is_dummy_run=False
    )
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(
        config, "get_current_atom_config", lambda: SimpleNamespace(eplb_enable=False)
    )
    monkeypatch.setattr(ubatching, "tbo_active", lambda: False)
    monkeypatch.setattr(envs, "ATOM_MOE_FUSED_TOPK_ROUTE_QUANT", True, raising=False)
    return calls, valid


def _method():
    return SimpleNamespace(fused_experts=None, use_triton=False, use_triton_ep=False)


def test_defers_with_pass_row_count(env):
    from atom.model_ops.fused_moe.fused_topk_route_quant import (
        try_defer_topk_route_quant,
    )

    calls, valid = env
    logits = torch.zeros((4, 257), dtype=torch.bfloat16)[:, :256]
    x = torch.zeros((4, 2048), dtype=torch.bfloat16)
    weights, ids = try_defer_topk_route_quant(_method(), _layer(), x, logits)
    assert len(calls) == 1
    w, i, g, rn, nvr = calls[0]
    assert w is weights and i is ids and g is logits and rn and nvr is valid
    assert weights.shape == ids.shape == (4, 8)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32


@pytest.mark.parametrize(
    "change",
    [
        lambda m, layer: setattr(m, "fused_experts", object()),
        lambda m, layer: setattr(m, "use_triton", True),
        lambda m, layer: setattr(layer, "expert_mask", torch.ones(256)),
        lambda m, layer: setattr(layer, "dp_size", 2),
        lambda m, layer: setattr(layer, "renormalize", False),
        lambda m, layer: setattr(layer, "num_fused_shared_experts", 1),
        lambda m, layer: setattr(layer.expert_layout, "uses_dispatch_remap", True),
        lambda m, layer: setattr(layer.moe_parallel_config, "pcp_merge", True),
    ],
)
def test_ineligible_keeps_normal_route(env, change):
    from atom.model_ops.fused_moe.fused_topk_route_quant import (
        try_defer_topk_route_quant,
    )

    calls, _ = env
    method, layer = _method(), _layer()
    change(method, layer)
    logits = torch.zeros((4, 256), dtype=torch.bfloat16)
    x = torch.zeros((4, 2048), dtype=torch.bfloat16)
    assert try_defer_topk_route_quant(method, layer, x, logits) is None
    assert not calls


def test_too_many_rows_keeps_normal_route(env):
    from atom.model_ops.fused_moe.fused_topk_route_quant import (
        try_defer_topk_route_quant,
    )

    calls, _ = env
    logits = torch.zeros((513, 256), dtype=torch.bfloat16)
    x = torch.zeros((513, 2048), dtype=torch.bfloat16)
    assert try_defer_topk_route_quant(_method(), _layer(), x, logits) is None
    assert not calls
