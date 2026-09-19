# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch

try:
    import aiter.iq2r_moe as aiter_iq2r_moe
    import atom.model_ops.moe as moe_mod
    from atom.quant_spec import get_quant_parser
except Exception as exc:  # noqa: BLE001
    pytest.skip(
        f"requires full ATOM/AITER import environment: {exc}", allow_module_level=True
    )


def test_iq2r_quant_parser_is_explicit_and_preserves_exclusions():
    parsed = get_quant_parser("iq2r").parse(
        {
            "quant_method": "iq2r",
            "schema": "aiter-gpt-oss-iq2r-overlay",
            "schema_version": 1,
            "modules_to_not_convert": ["model.layers.*.self_attn", "lm_head"],
        }
    )
    assert parsed.global_spec.quant_method == "iq2r"
    assert parsed.global_spec.quant_dtype == torch.uint8
    assert parsed.global_spec.quant_type == moe_mod.QuantType.iq2r_2bit
    assert parsed.global_spec.quant_type != moe_mod.QuantType.per_1x32
    assert parsed.exclude_layers == ["model.layers.*.self_attn", "lm_head"]


@pytest.mark.parametrize(
    "config",
    [
        {"quant_method": "iq2r"},
        {
            "quant_method": "iq2r",
            "schema": "aiter-gpt-oss-iq2r-overlay",
            "schema_version": 2,
        },
        {
            "quant_method": "iq2r",
            "schema": "unknown",
            "schema_version": 1,
        },
    ],
)
def test_iq2r_quant_parser_rejects_unknown_overlay_contract(config):
    with pytest.raises(ValueError, match="unsupported IQ2R overlay contract"):
        get_quant_parser("iq2r").parse(config)


def test_iq2r_weight_loader_requires_exact_self_contained_shapes():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    parameter = torch.nn.Parameter(
        torch.empty((2, 8), dtype=torch.uint8), requires_grad=False
    )
    parameter.iq2r_name = "w13_weight"
    loaded = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    method.load_weight(parameter, loaded)
    assert torch.equal(parameter, loaded)

    with pytest.raises(ValueError, match="shape mismatch"):
        method.load_weight(parameter, loaded[:, :-1])


def test_iq2r_create_weights_exposes_exact_overlay_contract():
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=1, ep_size=1),
        experts_per_token=4,
    )
    layer = torch.nn.Module()
    with torch.device("meta"):
        method.create_weights(
            layer,
            num_experts=128,
            hidden_size=2880,
            intermediate_size_per_partition=2880,
            params_dtype=torch.bfloat16,
        )

    expected = {
        "w13_weight": (torch.uint8, (128, 4_945_920)),
        "w13_weight_scale": (torch.uint8, (128, 4_459)),
        "w2_weight": (torch.uint8, (128, 2_472_960)),
        "w2_weight_scale": (torch.uint8, (128, 4_279)),
        "w13_bias": (torch.bfloat16, (128, 5_760)),
        "w2_bias": (torch.bfloat16, (128, 2_880)),
    }
    for name, (dtype, shape) in expected.items():
        parameter = getattr(layer, name)
        assert parameter.device.type == "meta"
        assert parameter.dtype == dtype
        assert tuple(parameter.shape) == shape
        assert parameter.iq2r_name == name


def test_iq2r_moe_fake_propagates_logical_hidden_width(monkeypatch):
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(output_hidden_size=2),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.0.mlp.experts": layer}
        )
    )
    monkeypatch.setattr(moe_mod, "get_current_atom_config", lambda: config)

    output = moe_mod.moe_forward_fake(
        torch.empty((3, 4)),
        torch.empty((3, 128)),
        "model.layers.0.mlp.experts",
    )
    assert output.shape == (3, 2)


def test_iq2r_apply_only_routes_and_delegates_to_aiter(monkeypatch):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        max_num_tokens=8,
    )
    method._workspaces = {}
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        w13_weight=torch.empty((128, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((128, 5), dtype=torch.uint8),
        w2_weight=torch.empty((128, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((128, 7), dtype=torch.uint8),
        w13_bias=torch.empty((128, 4), dtype=torch.bfloat16),
        w2_bias=torch.empty((128, 2), dtype=torch.bfloat16),
        iq2r_gate_up_metadata=object(),
        iq2r_down_metadata=object(),
        iq2r_gate_up_tile_n=128,
        iq2r_down_tile_n=64,
    )
    # Model GPT-OSS TP1's logical width 2880 inside a physical 3072-wide row.
    # At this small scale the MoE receives two columns with a four-element row
    # stride, and must pass that exact storage/view to AITER without copying it.
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    logits = torch.randn(3, 128)
    topk_weights = torch.full((3, 4), 0.25, dtype=torch.float32)
    topk_ids = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    monkeypatch.setattr(
        moe_mod.FusedMoE,
        "select_experts",
        staticmethod(lambda **_kwargs: (topk_weights, topk_ids)),
    )

    allocated = []

    class FakeWorkspace:
        @classmethod
        def allocate(cls, tokens, topk, *, device):
            value = (tokens, topk, device)
            allocated.append(value)
            return value

    calls = []

    def fake_fused_moe(**kwargs):
        calls.append(kwargs)
        return torch.full_like(kwargs["hidden_states"], 3)

    monkeypatch.setattr(aiter_iq2r_moe, "IQ2RMoeWorkspace", FakeWorkspace)
    monkeypatch.setattr(moe_mod, "fused_moe", fake_fused_moe)
    monkeypatch.setattr(moe_mod, "_IQ2R_ROUTE_CAPTURE_DIR", "/tmp/capture")

    arguments = {
        "layer": layer,
        "x": hidden,
        "router_logits": logits,
        "top_k": 4,
        "renormalize": True,
        "global_num_experts": 128,
        "activation": moe_mod.ActivationType.Swiglu,
    }
    first = method.apply(**arguments)
    second = method.apply(**{**arguments, "x": hidden[:2], "router_logits": logits[:2]})
    assert torch.equal(first, torch.full_like(hidden[:, :2], 3))
    assert torch.equal(second, torch.full_like(hidden[:2, :2], 3))
    assert allocated == [(8, 4, hidden.device)]
    assert len(calls) == 2

    kwargs = calls[0]
    assert torch.equal(kwargs["hidden_states"], hidden[:, :2])
    assert kwargs["hidden_states"].untyped_storage().data_ptr() == (
        hidden.untyped_storage().data_ptr()
    )
    assert kwargs["hidden_states"].stride() == (4, 1)
    assert not kwargs["hidden_states"].is_contiguous()
    assert kwargs["w1"] is layer.w13_weight
    assert kwargs["w2"] is layer.w2_weight
    assert kwargs["topk_weight"].dtype == torch.float32
    assert kwargs["topk_ids"].dtype == torch.int32
    assert kwargs["quant_type"] == moe_mod.QuantType.iq2r_2bit
    assert kwargs["iq2r_w1_auxiliary"] is layer.w13_weight_scale
    assert kwargs["iq2r_w2_auxiliary"] is layer.w2_weight_scale
    assert kwargs["iq2r_w1_metadata"] is layer.iq2r_gate_up_metadata
    assert kwargs["iq2r_w2_metadata"] is layer.iq2r_down_metadata
    assert kwargs["iq2r_w1_tile_n"] == 128
    assert kwargs["iq2r_w2_tile_n"] == 64
    assert kwargs["iq2r_workspace"] == (8, 4, hidden.device)
    assert kwargs["iq2r_router_logits"] is None
    assert kwargs["iq2r_router_bias"] is None
    assert not any(name.startswith("qtip2_") for name in kwargs)


def test_iq2r_apply_folds_router_bias_only_in_fused_frontend(monkeypatch):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        max_num_tokens=8,
    )
    method._workspaces = {}
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        w13_weight=torch.empty((128, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((128, 5), dtype=torch.uint8),
        w2_weight=torch.empty((128, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((128, 7), dtype=torch.uint8),
        w13_bias=torch.empty((128, 4), dtype=torch.bfloat16),
        w2_bias=torch.empty((128, 2), dtype=torch.bfloat16),
        iq2r_gate_up_metadata=object(),
        iq2r_down_metadata=object(),
        iq2r_gate_up_tile_n=128,
        iq2r_down_tile_n=64,
    )
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    logits = torch.randn(3, 128, dtype=torch.bfloat16)
    router_bias = torch.randn(128, dtype=torch.bfloat16)

    class FakeWorkspace:
        def __init__(self, tokens, topk):
            self.topk_weights = torch.empty(tokens, topk, dtype=torch.float32)
            self.topk_ids = torch.empty(tokens, topk, dtype=torch.int32)

        @classmethod
        def allocate(cls, tokens, topk, *, device):
            del device
            return cls(tokens, topk)

    calls = []

    def fake_fused_moe(**kwargs):
        calls.append(kwargs)
        return torch.empty_like(kwargs["hidden_states"])

    monkeypatch.setattr(aiter_iq2r_moe, "IQ2RMoeWorkspace", FakeWorkspace)
    monkeypatch.setattr(moe_mod, "fused_moe", fake_fused_moe)
    monkeypatch.setattr(
        moe_mod.FusedMoE,
        "select_experts",
        staticmethod(lambda **_kwargs: pytest.fail("fused routing must own top-k")),
    )

    method.apply(
        layer=layer,
        x=hidden,
        router_logits=logits,
        router_bias=router_bias,
        top_k=4,
        renormalize=True,
        global_num_experts=128,
        activation=moe_mod.ActivationType.Swiglu,
    )

    assert len(calls) == 1
    assert calls[0]["iq2r_router_logits"] is logits
    assert calls[0]["iq2r_router_bias"] is router_bias


def test_iq2r_router_bias_fallback_restores_bf16_logits(monkeypatch):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        max_num_tokens=8,
    )
    method._workspaces = {}
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        w13_weight=torch.empty((128, 3), dtype=torch.uint8),
        w13_weight_scale=torch.empty((128, 5), dtype=torch.uint8),
        w2_weight=torch.empty((128, 2), dtype=torch.uint8),
        w2_weight_scale=torch.empty((128, 7), dtype=torch.uint8),
        w13_bias=torch.empty((128, 4), dtype=torch.bfloat16),
        w2_bias=torch.empty((128, 2), dtype=torch.bfloat16),
        iq2r_gate_up_metadata=object(),
        iq2r_down_metadata=object(),
        iq2r_gate_up_tile_n=128,
        iq2r_down_tile_n=64,
    )
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    logits = torch.randn(3, 128, dtype=torch.bfloat16)
    router_bias = torch.randn(128, dtype=torch.bfloat16)
    selected_logits = []

    class FakeWorkspace:
        @classmethod
        def allocate(cls, tokens, topk, *, device):
            return (tokens, topk, device)

    def fake_select(**kwargs):
        selected_logits.append(kwargs["router_logits"])
        return (
            torch.full((3, 4), 0.25, dtype=torch.float32),
            torch.arange(12, dtype=torch.int32).reshape(3, 4),
        )

    monkeypatch.setattr(aiter_iq2r_moe, "IQ2RMoeWorkspace", FakeWorkspace)
    monkeypatch.setattr(moe_mod.FusedMoE, "select_experts", staticmethod(fake_select))
    monkeypatch.setattr(moe_mod, "_IQ2R_ROUTE_CAPTURE_DIR", "/tmp/capture")
    monkeypatch.setattr(
        moe_mod,
        "fused_moe",
        lambda **kwargs: torch.empty_like(kwargs["hidden_states"]),
    )

    method.apply(
        layer=layer,
        x=hidden,
        router_logits=logits,
        router_bias=router_bias,
        top_k=4,
        renormalize=True,
        global_num_experts=128,
        activation=moe_mod.ActivationType.Swiglu,
    )

    assert len(selected_logits) == 1
    assert selected_logits[0].dtype == torch.bfloat16
    assert torch.equal(selected_logits[0], logits + router_bias)


@pytest.mark.parametrize(
    ("tokens", "defer_bias"),
    [(2, True), (2, False), (9, True)],
)
def test_iq2r_router_dispatch_uses_concrete_runtime_m(monkeypatch, tokens, defer_bias):
    quant_method = object.__new__(moe_mod.Iq2rMoEMethod)
    quant_method.output_hidden_size = 2
    calls = []

    def fake_mm(x, weight, bias, *, otype):
        calls.append((x, weight, bias, otype))
        return torch.empty((x.shape[0], 128), dtype=torch.bfloat16)

    monkeypatch.setattr(moe_mod.tgemm, "mm", fake_mm)
    monkeypatch.setattr(moe_mod, "_IQ2R_DEFER_ROUTER_BIAS", defer_bias)

    deferred_output = torch.full((tokens, 2), 1, dtype=torch.bfloat16)
    ordinary_output = torch.full((tokens, 2), 2, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        quant_method=quant_method,
        balance_router_logits=None,
        forward_iq2r_impl=lambda hidden, logits, bias: deferred_output,
        forward_impl=lambda hidden, logits: ordinary_output,
    )
    hidden = torch.randn(tokens, 4, dtype=torch.bfloat16)
    weight = torch.randn(128, 2, dtype=torch.bfloat16)
    bias = torch.randn(128, dtype=torch.bfloat16)

    output = moe_mod.FusedMoE.forward_iq2r_with_router_impl(
        layer,
        hidden,
        weight,
        bias,
    )

    assert len(calls) == 1
    assert calls[0][0].shape == (tokens, 2)
    assert calls[0][1] is weight
    assert calls[0][3] == torch.bfloat16
    if defer_bias and tokens <= 8:
        assert calls[0][2] is None
        assert output is deferred_output
    else:
        assert calls[0][2] is bias
        assert output is ordinary_output


def test_iq2r_expert_ablation_retains_topk_and_skips_experts(monkeypatch):
    method = object.__new__(moe_mod.Iq2rMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=False),
        max_num_tokens=8,
    )
    method._workspaces = {}
    layer = SimpleNamespace(
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        w2_bias=torch.empty((128, 2), dtype=torch.bfloat16),
    )
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    logits = torch.randn(3, 128)
    topk_weights = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.5, 0.2, 0.2, 0.1], [0.6, 0.2, 0.1, 0.1]],
        dtype=torch.float32,
    )
    topk_ids = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    selections = []

    def fake_select(**kwargs):
        selections.append(kwargs)
        return topk_weights, topk_ids

    monkeypatch.setattr(
        moe_mod.FusedMoE,
        "select_experts",
        staticmethod(fake_select),
    )
    monkeypatch.setattr(moe_mod, "_PROFILE_MOE_ABLATION", "experts")
    monkeypatch.setattr(
        moe_mod,
        "fused_moe",
        lambda **_kwargs: pytest.fail("expert kernel must not run in ablation"),
    )

    output = method.apply(
        layer=layer,
        x=hidden,
        router_logits=logits,
        top_k=4,
        renormalize=True,
        global_num_experts=128,
        activation=moe_mod.ActivationType.Swiglu,
    )

    assert len(selections) == 1
    assert output.shape == (3, 2)
    assert torch.equal(
        output,
        topk_weights[:, :1].to(torch.bfloat16).expand(3, 2),
    )
