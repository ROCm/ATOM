# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


def _parallel(**overrides):
    values = {
        "use_ep": True,
        "ep_size": 16,
        "local_ep_size": 8,
        "dp_logical_ratio": 1,
        "use_all2all_kernels": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _moe(**overrides):
    values = {
        "moe_parallel_config": _parallel(),
        "a_quant_dtype": "fp8_e4m3",
        "expert_layout": SimpleNamespace(mode=moe_mod.SharedExpertMode.NONE),
        "has_bias": False,
        "in_dtype": torch.bfloat16,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quant(**overrides):
    values = {"is_dynamic": True}
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_supported_runtime(monkeypatch):
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(moe_mod.envs, "ATOM_MOE_GU_ITLV", True)
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(moe_backend="standard", eplb_enable=False),
    )


def test_ep16_two_node_selects_wideep(monkeypatch):
    _patch_supported_runtime(monkeypatch)
    standard_method = object()
    wideep_method = object()
    monkeypatch.setattr(moe_mod, "Mxfp4MoEMethod", lambda *_args: standard_method)
    monkeypatch.setattr(moe_mod, "WideEpMxfp4MoEMethod", lambda *_args: wideep_method)

    # TestWideEpMoe accepts BF16 hidden states and performs its own FP8
    # per-1x32 quantization, so the model need not declare FP8 activations.
    selected = moe_mod._make_mxfp4_moe_method(_quant(), _moe(a_quant_dtype=None))

    assert selected is wideep_method


@pytest.mark.parametrize(
    "parallel",
    [
        _parallel(ep_size=8, local_ep_size=8),
        _parallel(ep_size=16, local_ep_size=16),
        _parallel(use_ep=False, ep_size=1, local_ep_size=1),
    ],
)
def test_non_wideep_topology_keeps_standard_backend(monkeypatch, parallel):
    _patch_supported_runtime(monkeypatch)
    standard_method = object()
    wideep_method = object()
    monkeypatch.setattr(moe_mod, "Mxfp4MoEMethod", lambda *_args: standard_method)
    monkeypatch.setattr(moe_mod, "WideEpMxfp4MoEMethod", lambda *_args: wideep_method)

    selected = moe_mod._make_mxfp4_moe_method(
        _quant(), _moe(moe_parallel_config=parallel)
    )

    assert selected is standard_method


@pytest.mark.parametrize(
    ("quant", "moe", "config", "env_interleave", "gfx", "message"),
    [
        (_quant(is_dynamic=False), _moe(), {}, True, "gfx950", "dynamic"),
        (_quant(), _moe(), {"eplb_enable": True}, True, "gfx950", "EPLB"),
        (
            _quant(),
            _moe(
                expert_layout=SimpleNamespace(
                    mode=moe_mod.SharedExpertMode.LOCAL_REPLICA
                )
            ),
            {},
            True,
            "gfx950",
            "shared",
        ),
        (_quant(), _moe(has_bias=True), {}, True, "gfx950", "bias"),
        (_quant(), _moe(), {}, False, "gfx950", "ATOM_MOE_GU_ITLV"),
        (_quant(), _moe(), {}, True, "gfx942", "gfx950"),
    ],
)
def test_unsupported_ep16_configuration_is_rejected(
    monkeypatch, quant, moe, config, env_interleave, gfx, message
):
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: gfx)
    monkeypatch.setattr(moe_mod.envs, "ATOM_MOE_GU_ITLV", env_interleave)
    config_values = {"moe_backend": "standard", "eplb_enable": False}
    config_values.update(config)
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(**config_values),
    )

    with pytest.raises(ValueError, match=message):
        moe_mod._validate_wideep_mxfp4_config(quant, moe)


def test_wideep_wrapper_quantizes_and_uses_prequant_entry(monkeypatch):
    from atom.model_ops.fused_moe import wideep_experts

    calls = []

    class FakeOp:
        def quantize(self, hidden):
            calls.append(("quantize", hidden))
            return "x_quant", "x_scale"

        def forward_prequant(self, x_quant, x_scale, weights, ids):
            calls.append(("forward_prequant", x_quant, x_scale, weights, ids))
            return "output"

    monkeypatch.setattr(
        wideep_experts, "get_or_build_wideep_moe", lambda **_kw: FakeOp()
    )

    manager = SimpleNamespace(rank=0, world_size=16)
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(all2all_manager=manager)
    )
    from aiter.dist import parallel_state

    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: group)
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    weights = torch.ones(2, 2, dtype=torch.bfloat16).t()
    ids = torch.ones(2, 2, dtype=torch.int64).t()
    layer = SimpleNamespace(
        _wideep_w1="w1",
        _wideep_w1_scale="w1_scale",
        _wideep_w2="w2",
        _wideep_w2_scale="w2_scale",
    )

    output = wideep_experts.run_wideep_moe(
        layer,
        hidden,
        weights,
        ids,
        model_dim=4,
        inter_dim=8,
        experts=16,
        topk=2,
        mtpr=8,
        swiglu_limit=10.0,
    )

    assert output == "output"
    assert calls[0][0] == "quantize"
    assert calls[0][1] is hidden
    assert calls[1][0:3] == ("forward_prequant", "x_quant", "x_scale")
    assert calls[1][3].dtype == torch.float32
    assert calls[1][3].is_contiguous()
    assert calls[1][4].dtype == torch.int32
    assert calls[1][4].is_contiguous()


def test_wideep_cache_key_includes_limit_and_rebinds_weights(monkeypatch):
    from atom.model_ops.fused_moe import wideep_experts

    built = []

    class FakeTestWideEpMoe:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.w1 = kwargs["w1"]
            self.w1_scale = kwargs["w1_scale"]
            self.w2 = kwargs["w2"]
            self.w2_scale = kwargs["w2_scale"]

    fake_module = SimpleNamespace(TestWideEpMoe=FakeTestWideEpMoe)
    monkeypatch.setitem(
        __import__("sys").modules,
        "aiter.ops.flydsl.test_wide_ep_moe",
        fake_module,
    )
    wideep_experts._WIDEEP_CACHE.clear()
    common = {
        "rank": 0,
        "world_size": 16,
        "model_dim": 16,
        "inter_dim": 8,
        "experts": 32,
        "topk": 2,
        "quant": "a8w4",
        "mtpr": 256,
    }

    first = wideep_experts.get_or_build_wideep_moe(
        **common,
        swiglu_limit=10.0,
        w1="layer0_w1",
        w1_scale="layer0_s1",
        w2="layer0_w2",
        w2_scale="layer0_s2",
    )
    rebound = wideep_experts.get_or_build_wideep_moe(
        **common,
        swiglu_limit=10.0,
        w1="layer1_w1",
        w1_scale="layer1_s1",
        w2="layer1_w2",
        w2_scale="layer1_s2",
    )
    different_limit = wideep_experts.get_or_build_wideep_moe(
        **common,
        swiglu_limit=0.0,
        w1="layer2_w1",
        w1_scale="layer2_s1",
        w2="layer2_w2",
        w2_scale="layer2_s2",
    )

    assert first is rebound
    assert rebound.w1 == "layer1_w1"
    assert rebound.w1_scale == "layer1_s1"
    assert rebound.w2 == "layer1_w2"
    assert rebound.w2_scale == "layer1_s2"
    assert different_limit is not first
    assert [call["swiglu_limit"] for call in built] == [10.0, 0.0]


def test_compiled_boundary_receives_and_rebinds_layer_weights(monkeypatch):
    from atom.model_ops.fused_moe import wideep_experts

    captured = {}

    class FakeOp:
        def forward_prequant(self, x_quant, x_scale, weights, ids):
            captured["inputs"] = (x_quant, x_scale, weights, ids)
            return torch.ones((ids.shape[0], 4), dtype=torch.bfloat16)

    def fake_get_or_build(**kwargs):
        captured["builder"] = kwargs
        return FakeOp()

    monkeypatch.setattr(wideep_experts, "get_or_build_wideep_moe", fake_get_or_build)
    x_quant = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
    x_scale = torch.ones((2, 1), dtype=torch.uint8)
    weights = torch.ones((2, 2), dtype=torch.float32)
    ids = torch.ones((2, 2), dtype=torch.int32)
    w1 = torch.ones((2, 2))
    w1_scale = torch.ones((2, 2))
    w2 = torch.ones((2, 2))
    w2_scale = torch.ones((2, 2))

    output = wideep_experts.atom_wideep_forward_impl(
        x_quant,
        x_scale,
        weights,
        ids,
        w1,
        w1_scale,
        w2,
        w2_scale,
        0,
        16,
        4,
        8,
        32,
        2,
        256,
        10.0,
    )

    assert output.shape == (2, 4)
    assert captured["builder"]["w1"] is w1
    assert captured["builder"]["w1_scale"] is w1_scale
    assert captured["builder"]["w2"] is w2
    assert captured["builder"]["w2_scale"] is w2_scale
    assert captured["builder"]["swiglu_limit"] == 10.0
