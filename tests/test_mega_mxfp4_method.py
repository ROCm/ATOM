# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
except Exception as exc:  # noqa: BLE001
    skip_if_dependency_missing(exc, "requires full atom import env")


def test_mega_method_is_mxfp4_specialization():
    assert issubclass(moe_mod.MegaMxfp4MoEMethod, moe_mod.Mxfp4MoEMethod)


@pytest.mark.parametrize("backend_name", ["standard", "mega"])
def test_mxfp4_method_selection(monkeypatch, backend_name):
    standard_method = object()
    mega_method = object()
    config = SimpleNamespace(moe_backend=backend_name)
    monkeypatch.setattr(moe_mod, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe_mod, "Mxfp4MoEMethod", lambda *_args: standard_method)
    monkeypatch.setattr(moe_mod, "MegaMxfp4MoEMethod", lambda *_args: mega_method)

    selected = moe_mod._make_mxfp4_moe_method(object(), object())

    assert selected is (mega_method if backend_name == "mega" else standard_method)


@pytest.mark.parametrize(
    ("eplb_enabled", "expected_triton"),
    [(False, True), (True, False)],
)
def test_eplb_controls_effective_triton_backend(
    monkeypatch, eplb_enabled, expected_triton
):
    config = SimpleNamespace(eplb_enable=eplb_enabled)
    monkeypatch.setattr(moe_mod, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: "gfx942")
    monkeypatch.setattr(moe_mod.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE", True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE_DECODE", True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_MOE_GU_ITLV", False)

    quant_config = SimpleNamespace(
        quant_type=object(),
        quant_dtype=object(),
        quant_method=None,
        is_dynamic=True,
    )
    moe_config = SimpleNamespace(a_quant_dtype=None)

    method = moe_mod.Mxfp4MoEMethod(quant_config, moe_config)

    assert method.use_triton is expected_triton
    assert method.use_triton_decode is expected_triton


def test_standard_post_routing_arguments_are_preserved(monkeypatch):
    calls = []

    def fake_fused_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    monkeypatch.setattr(moe_mod, "fused_moe", fake_fused_moe)
    method = object.__new__(moe_mod.Mxfp4MoEMethod)
    method.fused_experts = None
    method.quant_type = "mxfp4"
    method.is_guinterleave = True
    method.hidden_pad = 0
    method.intermediate_pad = 0
    # Skip both pre-routing early returns so apply() reaches the post-routing
    # block. use_triton_decode=False also short-circuits get_forward_context().
    method.use_triton = False
    method.use_triton_decode = False
    method.select_experts_with_record = lambda **_kwargs: (
        "topk_weights",
        "topk_ids",
    )
    layer = SimpleNamespace(
        w13_weight="w1",
        w2_weight="w2",
        w13_weight_scale="w1_scale",
        w2_weight_scale="w2_scale",
        w13_input_scale="a1_scale",
        w2_input_scale="a2_scale",
        w13_bias="b1",
        w2_bias="b2",
        expert_mask="mask",
        swiglu_limit=7.0,
    )

    result = method.apply(
        layer=layer,
        x="x",
        router_logits="router_logits",
        top_k=4,
        renormalize=False,
        global_num_experts=16,
        expert_map="expert_map",
        activation="silu",
        apply_router_weight_on_input=False,
    )

    assert result == "output"
    args, kwargs = calls.pop()
    assert args == ("x", "w1", "w2", "topk_weights", "topk_ids")
    assert kwargs == {
        "expert_mask": "mask",
        "activation": "silu",
        "quant_type": "mxfp4",
        "w1_scale": "w1_scale",
        "w2_scale": "w2_scale",
        "a1_scale": "a1_scale",
        "a2_scale": "a2_scale",
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "bias1": "b1",
        "bias2": "b2",
        "gate_mode": moe_mod.GateMode.INTERLEAVE.value,
        "swiglu_limit": 7.0,
    }


def test_mega_eplb_views_are_expert_major_aliases():
    layer = SimpleNamespace(
        _mega_w1=torch.arange(24),
        _mega_w1_scale=torch.arange(12),
        _mega_w2=torch.arange(36),
        _mega_w2_scale=torch.arange(8),
        local_num_experts=4,
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    views = method.get_eplb_weight_views(layer)

    assert [tuple(view.shape) for view in views] == [(4, 6), (4, 3), (4, 9), (4, 2)]
    views[0][2, 1] = -1
    assert layer._mega_w1.view(4, -1)[2, 1].item() == -1


def test_mega_eplb_views_reject_non_expert_major_storage():
    layer = SimpleNamespace(
        _mega_w1=torch.arange(10),
        _mega_w1_scale=torch.arange(12),
        _mega_w2=torch.arange(36),
        _mega_w2_scale=torch.arange(8),
        local_num_experts=4,
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    with pytest.raises(RuntimeError, match="evenly divisible"):
        method.get_eplb_weight_views(layer)


def test_mega_eplb_views_include_shared_expert():
    layer = SimpleNamespace(
        _mega_w1=torch.arange(30),
        _mega_w1_scale=torch.arange(15),
        _mega_w2=torch.arange(45),
        _mega_w2_scale=torch.arange(10),
        local_num_experts=5,
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    views = method.get_eplb_weight_views(layer)

    assert [tuple(view.shape) for view in views] == [(5, 6), (5, 3), (5, 9), (5, 2)]
    assert torch.equal(views[0], layer._mega_w1.view(5, -1))


def test_mega_rejects_the_legacy_shared_expert_fusion(monkeypatch):
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(moe_backend="mega")
    )
    layer = SimpleNamespace(
        quant_method=object.__new__(moe_mod.MegaMxfp4MoEMethod),
        expert_layout=SimpleNamespace(mode=moe_mod.SharedExpertMode.LEGACY_AITER),
    )

    with pytest.raises(NotImplementedError, match="dispatch space"):
        moe_mod.FusedMoE._validate_moe_backend(layer)

    layer.expert_layout.mode = moe_mod.SharedExpertMode.LOCAL_REPLICA
    moe_mod.FusedMoE._validate_moe_backend(layer)
    layer.expert_layout.mode = moe_mod.SharedExpertMode.EPLB_ROUTED
    moe_mod.FusedMoE._validate_moe_backend(layer)


def test_mega_backend_uses_global_dispatch_width(monkeypatch):
    from atom.model_ops.fused_moe import flydsl_mega_experts as mega_module

    captured = {}

    def fake_run(_layer, _hidden, _weights, _ids, **kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(mega_module, "run_mega_moe", fake_run)
    layer = SimpleNamespace(
        _mega_w1=torch.empty(33, 1),
        local_num_experts=33,
        swiglu_limit=0.0,
    )
    backend = mega_module.MegaFusedExperts(
        layer,
        model_dim=16,
        inter_dim=8,
        mtpr=256,
    )

    output = backend(
        hidden_states="hidden",
        topk_weights="weights",
        topk_ids=SimpleNamespace(shape=(1, 9)),
        global_num_experts=264,
    )

    assert output == "output"
    assert captured["experts"] == 264
    assert captured["topk"] == 9
