# SPDX-License-Identifier: MIT

import inspect
from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")

# scoring, grouped top-k (groups, selected groups), top_k and the layer's
# routed scale, which both routings fold in and HERD therefore applies.
HERD_PROFILES = {
    "deepseek_v4": ("sqrtsoftplus", False, None, None, 6, 1.5),
    "kimi_k3": ("sigmoid", True, 1, 1, 16, 2.5),
}
# Routings HERD declines.
OTHER_PROFILES = {
    "gpt_oss": ("softmax", False, None, None, 4, 1.0),
    "minimax_m3": ("sigmoid", False, None, None, 4, 2.0),
    "sqrtsoftplus_grouped": ("sqrtsoftplus", True, 1, 1, 6, 1.5),
}


def _layer(scoring_func, grouped, groups, topk_group, top_k, scale, **extra):
    return SimpleNamespace(
        scoring_func=scoring_func,
        use_grouped_topk=grouped,
        num_expert_group=groups,
        topk_group=topk_group,
        top_k=top_k,
        routed_scaling_factor=scale,
        renormalize=True,
        custom_routing_function=None,
        apply_router_weight_on_input=False,
        num_fused_shared_experts=0,
        expert_mask=None,
        e_score_correction_bias="bias",
        use_ep=False,
        **extra,
    )


def _herd_method():
    method = object.__new__(moe_mod.Mxfp4MoEMethod)
    method.use_flydsl_herd = True
    method.use_triton = False
    method.fused_experts = None
    return method


@pytest.fixture
def decode_step(monkeypatch):
    context = SimpleNamespace(is_prefill=False, running_tokens_are_unified=True)
    monkeypatch.setattr(
        moe_mod, "get_forward_context", lambda: SimpleNamespace(context=context)
    )


@pytest.mark.parametrize("profile", list(HERD_PROFILES))
def test_herd_candidates_take_the_layers_own_routing(monkeypatch, decode_step, profile):
    layer = _layer(*HERD_PROFILES[profile])
    signature = inspect.signature(moe_mod.FusedMoE.select_experts)
    calls = []

    def fake_select_experts(**kwargs):
        # A keyword FusedMoE.select_experts no longer takes fails here.
        calls.append(signature.bind(**kwargs).arguments)
        return "weights", "ids"

    monkeypatch.setattr(
        moe_mod.FusedMoE, "select_experts", staticmethod(fake_select_experts)
    )

    assert _herd_method().select_experts(layer, "x", torch.empty(64, 8)) == (
        "weights",
        "ids",
    )
    kwargs = calls.pop()
    # One extra candidate, raw scores: fused_moe drops, renormalises and scales.
    assert kwargs["top_k"] == layer.top_k + 1
    assert kwargs["renormalize"] is False
    assert kwargs.get("routed_scaling_factor", 1.0) == 1.0
    assert kwargs.get("custom_routing_function") is None
    assert kwargs.get("num_fused_shared_experts", 0) == 0
    assert kwargs["scoring_func"] == layer.scoring_func
    assert kwargs["e_score_correction_bias"] == "bias"
    assert kwargs["use_grouped_topk"] is layer.use_grouped_topk
    assert kwargs["num_expert_group"] == layer.num_expert_group
    assert kwargs["topk_group"] == layer.topk_group


@pytest.mark.parametrize("groups,topk_group", [(8, 4), (8, 1), (1, 1)])
def test_herd_takes_grouped_topk_with_one_group_only(decode_step, groups, topk_group):
    layer = _layer("sigmoid", True, groups, topk_group, 8, 2.5)

    assert _herd_method()._flydsl_herd_step(layer, torch.empty(64, 8)) is (
        groups == 1 and topk_group == 1
    )


@pytest.mark.parametrize(
    "profile,herd",
    [(p, True) for p in HERD_PROFILES] + [(p, False) for p in OTHER_PROFILES],
)
def test_herd_serves_deepseek_v4_and_kimi_k3_only(decode_step, profile, herd):
    layer = _layer(*{**HERD_PROFILES, **OTHER_PROFILES}[profile])

    assert _herd_method()._flydsl_herd_step(layer, torch.empty(64, 8)) is herd


@pytest.mark.parametrize(
    "dp_size,dp_attention,dp_logical_ratio,pcp_merge,own_rows",
    [
        (1, False, 1, False, True),
        # DP without DP attention runs each rank's own rows.
        (2, False, 1, False, True),
        (2, True, 1, False, False),
        (1, True, 2, False, False),
        (1, False, 1, True, False),
    ],
)
def test_herd_declines_gathered_rows(
    monkeypatch, dp_size, dp_attention, dp_logical_ratio, pcp_merge, own_rows
):
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(enable_dp_attention=dp_attention),
    )
    parallel_config = SimpleNamespace(
        dp_size=dp_size, dp_logical_ratio=dp_logical_ratio, pcp_merge=pcp_merge
    )

    assert moe_mod._herd_sees_own_rows(parallel_config) is own_rows


def test_herd_stays_off_without_aiters_herd_route(monkeypatch):
    from aiter.ops.flydsl import herd_topk

    moe_mod._aiter_serves_flydsl_herd.cache_clear()
    # An aiter older than v0.1.29 has the module but not herd_route_psum.
    monkeypatch.delattr(herd_topk, "herd_route_psum", raising=False)
    try:
        assert moe_mod._aiter_serves_flydsl_herd() is False
    finally:
        moe_mod._aiter_serves_flydsl_herd.cache_clear()


@pytest.mark.parametrize("herd", [True, False])
@pytest.mark.parametrize("profile", list(HERD_PROFILES))
def test_herd_applies_the_scale_and_the_pass_row_count(
    monkeypatch, decode_step, profile, herd
):
    calls = []

    def fake_fused_moe(*args, **kwargs):
        calls.append(kwargs)
        return "output"

    monkeypatch.setattr(moe_mod, "fused_moe", fake_fused_moe)
    method = _herd_method()
    method.quant_type = "mxfp4"
    method.is_guinterleave = True
    method.fp8_activations = False
    method.hidden_pad = 0
    method.intermediate_pad = 0
    method.use_triton_decode = False
    method.use_triton_ep = False
    layer = _layer(
        *HERD_PROFILES[profile],
        w13_weight="w1",
        w2_weight="w2",
        w13_weight_scale="w1_scale",
        w2_weight_scale="w2_scale",
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
        swiglu_limit=0.0,
    )

    # 8 rows sit below aiter's HERD window: a plain step.
    rows = 64 if herd else 8
    method.apply(
        layer=layer,
        x=torch.empty(rows, 8),
        topk_weights="topk_weights",
        topk_ids=torch.empty(rows, layer.top_k + 1, dtype=torch.int32),
        global_num_experts=128,
        expert_map=None,
        activation="silu",
        apply_router_weight_on_input=False,
        scheduled_tokens_gpu="count",
    )

    kwargs = calls.pop()
    # The drop leaves the pad rows out by the count the candidates were cut at.
    assert kwargs["num_local_tokens"] == "count"
    if not herd:
        # K+1 ids per row on a plain step: apply reads no HERD from the width.
        assert "herd_topk" not in kwargs
        return
    assert kwargs["herd_topk"] == layer.top_k
    assert kwargs["herd_routed_scaling_factor"] == layer.routed_scaling_factor
