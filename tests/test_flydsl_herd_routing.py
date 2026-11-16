# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")

# scoring, grouped top-k (groups, selected groups), top_k, the layer's routed
# scale and the scale HERD must apply: the one the layer's own routing folds in.
PROFILES = {
    "deepseek_v4": ("sqrtsoftplus", False, None, None, 6, 1.5, 1.5),
    # The softmax top-k never applies the routed scale.
    "gpt_oss": ("softmax", False, None, None, 4, 2.0, 1.0),
    "kimi_k3": ("sigmoid", True, 1, 1, 16, 2.5, 2.5),
    # Applied outside the MoE when the shared experts are not fused.
    "minimax_m3": ("sigmoid", False, None, None, 4, 2.0, 1.0),
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


@pytest.mark.parametrize("profile", list(PROFILES))
def test_herd_candidates_take_the_layers_own_routing(monkeypatch, decode_step, profile):
    scoring_func, grouped, groups, topk_group, top_k, scale, _ = PROFILES[profile]
    layer = _layer(scoring_func, grouped, groups, topk_group, top_k, scale)
    calls = []

    def fake_select_experts(**kwargs):
        calls.append(kwargs)
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
    assert kwargs["top_k"] == top_k + 1
    assert kwargs["renormalize"] is False
    assert kwargs.get("routed_scaling_factor", 1.0) == 1.0
    assert kwargs.get("custom_routing_function") is None
    assert kwargs.get("num_fused_shared_experts", 0) == 0
    # Pure TP routes pad rows like real ones, as _select_logical does, so the
    # candidates carry no -1 rows for the count fused_moe is not given.
    assert kwargs["mask_pad_rows"] is False
    assert kwargs["scoring_func"] == scoring_func
    assert kwargs["e_score_correction_bias"] == "bias"
    assert kwargs["use_grouped_topk"] is grouped
    assert kwargs["num_expert_group"] == groups
    assert kwargs["topk_group"] == topk_group


@pytest.mark.parametrize("groups,topk_group", [(8, 4), (8, 1), (1, 1)])
def test_herd_takes_grouped_topk_with_one_group_only(decode_step, groups, topk_group):
    layer = _layer("sigmoid", True, groups, topk_group, 8, 2.5)

    assert _herd_method()._flydsl_herd_step(layer, torch.empty(64, 8)) is (
        groups == 1 and topk_group == 1
    )


@pytest.mark.parametrize("hash_layers", [True, False])
def test_herd_routes_hash_layers_from_logits(monkeypatch, decode_step, hash_layers):
    monkeypatch.setattr(moe_mod.envs, "ATOM_FLYDSL_HERD_HASH_LAYERS", hash_layers)
    layer = _layer("sqrtsoftplus", False, None, None, 6, 1.5)
    layer.custom_routing_function = lambda **_: None

    assert _herd_method()._flydsl_herd_step(layer, torch.empty(64, 8)) is hash_layers


@pytest.mark.parametrize("profile", list(PROFILES))
def test_herd_applies_the_routed_scale_the_layer_folds_in(monkeypatch, profile):
    scoring_func, grouped, groups, topk_group, top_k, scale, herd_scale = PROFILES[
        profile
    ]
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
        scoring_func,
        grouped,
        groups,
        topk_group,
        top_k,
        scale,
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

    method.apply(
        layer=layer,
        x="x",
        topk_weights="topk_weights",
        topk_ids=torch.empty(4, top_k + 1, dtype=torch.int32),
        global_num_experts=128,
        expert_map=None,
        activation="silu",
        apply_router_weight_on_input=False,
    )

    kwargs = calls.pop()
    assert kwargs["herd_topk"] == top_k
    assert kwargs["herd_routed_scaling_factor"] == herd_scale
