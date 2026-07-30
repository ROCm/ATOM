# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch

try:
    import atom.model_ops.moe as moe_mod
except Exception as exc:  # aiter/triton absent under bare non-GPU pytest
    pytest.skip(f"requires full atom import env: {exc}", allow_module_level=True)


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

    result = method._apply_after_routing(
        layer=layer,
        x="x",
        topk_weights="topk_weights",
        topk_ids="topk_ids",
        top_k=4,
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
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    views = method.get_eplb_weight_views(layer, 4)

    assert [tuple(view.shape) for view in views] == [(4, 6), (4, 3), (4, 9), (4, 2)]
    views[0][2, 1] = -1
    assert layer._mega_w1.view(4, -1)[2, 1].item() == -1


def test_mega_eplb_views_reject_non_expert_major_storage():
    layer = SimpleNamespace(
        _mega_w1=torch.arange(10),
        _mega_w1_scale=torch.arange(12),
        _mega_w2=torch.arange(36),
        _mega_w2_scale=torch.arange(8),
    )
    method = object.__new__(moe_mod.MegaMxfp4MoEMethod)

    with pytest.raises(RuntimeError, match="evenly divisible"):
        method.get_eplb_weight_views(layer, 4)
