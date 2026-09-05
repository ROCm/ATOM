# SPDX-License-Identifier: MIT
"""ATOM_USE_TRITON_MOE_DECODE: FlyDSL on prefill, Triton/gluon on decode.

The flag keeps ONE copy of the MXFP4 expert weights, in the FlyDSL layout, and
hands the Triton kernel a zero-copy view of it at decode time. That is only
sound because on gfx1250 + GUGU the two preshuffles write the same bytes, so
``test_triton_views_reproduce_the_triton_prep`` is the load-bearing test here --
it runs both real preps and compares.
"""

from types import SimpleNamespace

import pytest
import torch

try:
    # aiter/triton absent under bare non-GPU pytest
    from aiter.jit.utils.chip_info import get_gfx

    import atom.model_ops.moe as moe_mod
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"requires full atom import env: {exc}", allow_module_level=True)

gfx1250_only = pytest.mark.skipif(
    get_gfx() != "gfx1250", reason="the FlyDSL/Triton layouts only coincide on gfx1250"
)


def _make_method(monkeypatch, *, decode_flag, use_triton=True, use_ep=False):
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(eplb_enable=False)
    )
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(moe_mod.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE", use_triton)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE_DECODE", decode_flag)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE_A4W4", False)
    monkeypatch.setattr(moe_mod.envs, "ATOM_MOE_GU_ITLV", True)

    quant_config = SimpleNamespace(
        quant_type=object(),
        quant_dtype=object(),
        quant_method=None,
        is_dynamic=True,
    )
    moe_config = SimpleNamespace(a_quant_dtype=None, use_ep=use_ep)
    return moe_mod.Mxfp4MoEMethod(quant_config, moe_config)


@pytest.mark.parametrize(
    ("decode_flag", "use_triton", "use_ep", "expected"),
    [
        (True, True, False, True),
        (False, True, False, False),
        # The flag narrows ATOM_USE_TRITON_MOE; it cannot turn Triton on, and
        # it is inert under EP (which has its own weight prep).
        (True, False, False, False),
        (True, True, True, False),
    ],
)
def test_decode_flag_only_narrows_the_triton_path(
    monkeypatch, decode_flag, use_triton, use_ep, expected
):
    method = _make_method(
        monkeypatch, decode_flag=decode_flag, use_triton=use_triton, use_ep=use_ep
    )
    assert method.use_triton_decode is expected


def test_mega_backend_keeps_decode_triton_off(monkeypatch):
    monkeypatch.setattr(
        moe_mod, "get_current_atom_config", lambda: SimpleNamespace(eplb_enable=False)
    )
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(moe_mod.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE", True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE_DECODE", True)
    monkeypatch.setattr(moe_mod.envs, "ATOM_USE_TRITON_MOE_A4W4", False)
    monkeypatch.setattr(moe_mod.envs, "ATOM_MOE_GU_ITLV", True)

    method = moe_mod.MegaMxfp4MoEMethod(
        SimpleNamespace(
            quant_type=object(),
            quant_dtype=object(),
            quant_method=None,
            is_dynamic=True,
        ),
        SimpleNamespace(a_quant_dtype=None, use_ep=True),
    )

    assert method.use_triton_decode is False


# ── weight prep ────────────────────────────────────────────────────────────


def _prep_method(*, use_triton, use_triton_decode, num_experts, intermediate):
    """A Mxfp4MoEMethod carrying only what _process_weight_layout_after_loading
    reads, so the prep can run without create_weights()/a real FusedMoEConfig."""
    method = object.__new__(moe_mod.Mxfp4MoEMethod)
    method.use_triton = use_triton
    method.use_triton_ep = False
    method.use_triton_decode = use_triton_decode
    method.is_gfx1250 = True
    method.is_guinterleave = True
    method.num_experts = num_experts
    method.intermediate_size = intermediate
    method.hidden_pad = 0
    method.intermediate_pad = 0
    return method


def _make_layer(num_experts, hidden, intermediate, seed=0):
    torch.manual_seed(seed)
    return SimpleNamespace(
        activation=moe_mod.ActivationType.Silu,
        num_fused_shared_experts=0,
        w13_bias=None,
        w2_bias=None,
        w13_swizzle_layout=None,
        w2_swizzle_layout=None,
        w13_weight=torch.nn.Parameter(
            torch.randint(
                0, 255, (num_experts, 2 * intermediate, hidden // 2), dtype=torch.uint8
            ),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.randint(
                0, 255, (num_experts, hidden, intermediate // 2), dtype=torch.uint8
            ),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.randint(
                0, 255, (num_experts, 2 * intermediate, hidden // 32), dtype=torch.uint8
            ),
            requires_grad=False,
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.randint(
                0, 255, (num_experts, hidden, intermediate // 32), dtype=torch.uint8
            ),
            requires_grad=False,
        ),
    )


@gfx1250_only
def test_decode_prep_takes_the_flydsl_branch(monkeypatch):
    """With the flag on, weight prep must land in branch C: same shapes as a
    plain FlyDSL run, and no swizzle layout published on the layer."""
    E, HID, INTER = 3, 256, 128

    flydsl = _make_layer(E, HID, INTER)
    _prep_method(
        use_triton=False, use_triton_decode=False, num_experts=E, intermediate=INTER
    )._process_weight_layout_after_loading(flydsl)

    decode = _make_layer(E, HID, INTER)
    _prep_method(
        use_triton=True, use_triton_decode=True, num_experts=E, intermediate=INTER
    )._process_weight_layout_after_loading(decode)

    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        lhs, rhs = getattr(flydsl, name), getattr(decode, name)
        assert lhs.shape == rhs.shape, name
        assert torch.equal(lhs.data.view(torch.uint8), rhs.data.view(torch.uint8)), name
    assert decode.w13_swizzle_layout is None
    assert decode.w2_swizzle_layout is None


@gfx1250_only
def test_triton_views_reproduce_the_triton_prep():
    """The whole feature rests on this: the decode-time view of the FlyDSL
    buffer must equal, element for element, what the Triton prep would have
    produced from the same checkpoint."""
    E, HID, INTER = 3, 256, 128

    triton_layer = _make_layer(E, HID, INTER)
    _prep_method(
        use_triton=True, use_triton_decode=False, num_experts=E, intermediate=INTER
    )._process_weight_layout_after_loading(triton_layer)

    flydsl_layer = _make_layer(E, HID, INTER)
    method = _prep_method(
        use_triton=True, use_triton_decode=True, num_experts=E, intermediate=INTER
    )
    method._process_weight_layout_after_loading(flydsl_layer)

    w13, w2, w13_scale, w2_scale, w13_layout, w2_layout = (
        method._triton_views_of_flydsl_weights(flydsl_layer)
    )

    expected = (
        triton_layer.w13_weight.view(torch.uint8),
        triton_layer.w2_weight.view(torch.uint8),
        triton_layer.w13_weight_scale,
        triton_layer.w2_weight_scale,
    )
    for got, want in zip((w13, w2, w13_scale, w2_scale), expected):
        assert torch.equal(got, want)
        # Values alone are not enough: moe_gemm_a8w4 derives N from shape[-1]
        # and indexes the buffers by stride, so the view has to match those too.
        assert got.shape == want.shape
        assert got.stride() == want.stride()
        assert got.dtype == want.dtype
    assert w13_layout == triton_layer.w13_swizzle_layout == "GFX1250_SCALE"
    assert w2_layout == triton_layer.w2_swizzle_layout == "GFX1250_SCALE"

    # Zero-copy: the views must alias the stored FlyDSL weights, not a second
    # copy of them -- that is the point of doing this at runtime.
    assert w13.data_ptr() == flydsl_layer.w13_weight.data_ptr()
    assert w2.data_ptr() == flydsl_layer.w2_weight.data_ptr()
    assert w13_scale.data_ptr() == flydsl_layer.w13_weight_scale.data_ptr()
    assert w2_scale.data_ptr() == flydsl_layer.w2_weight_scale.data_ptr()
    assert method._triton_views_of_flydsl_weights(flydsl_layer) is not None


@gfx1250_only
def test_decode_prep_rejects_a_layout_the_two_kernels_do_not_share():
    """GGUU (ATOM_MOE_GU_ITLV=0) gives FlyDSL and Triton genuinely different
    layouts, so sharing one copy must be refused rather than read as garbage."""
    method = _prep_method(
        use_triton=True, use_triton_decode=True, num_experts=3, intermediate=128
    )
    method.is_guinterleave = False

    with pytest.raises(AssertionError, match="ATOM_MOE_GU_ITLV"):
        method._process_weight_layout_after_loading(_make_layer(3, 256, 128))


@gfx1250_only
def test_decode_prep_rejects_a_padded_expert_shape():
    """FlyDSL trims create_weights' padding on prefill and the Triton decode
    kernel cannot, so a padded layer would make the two phases disagree."""
    method = _prep_method(
        use_triton=True, use_triton_decode=True, num_experts=3, intermediate=128
    )
    method.intermediate_pad = 64

    with pytest.raises(AssertionError, match="intermediate_pad=64"):
        method._process_weight_layout_after_loading(_make_layer(3, 256, 128))


# ── dispatch ───────────────────────────────────────────────────────────────


def _apply_dispatch_probe(monkeypatch, *, use_triton_decode, is_prefill):
    """Run apply() far enough to see which kernel family it picked."""
    monkeypatch.setattr(
        moe_mod,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(is_prefill=is_prefill)),
    )
    flydsl_calls = []

    def fake_fused_moe(*args, **kwargs):
        flydsl_calls.append((args, kwargs))
        return "flydsl"

    monkeypatch.setattr(moe_mod, "fused_moe", fake_fused_moe)

    method = object.__new__(moe_mod.Mxfp4MoEMethod)
    method.use_triton = True
    method.use_triton_ep = False
    method.use_triton_decode = use_triton_decode
    method.is_gfx1250 = True
    method.is_guinterleave = True
    method.act_quant = None
    method.quant_type = "mxfp4"
    method.hidden_pad = 0
    method.intermediate_pad = 0
    method.fused_experts = None
    method.select_experts_with_record = lambda **_k: ("topk_weights", "topk_ids")
    method._triton_views_of_flydsl_weights = lambda _layer: (
        "w13_view",
        "w2_view",
        "w13_scale_view",
        "w2_scale_view",
        "GFX1250_SCALE",
        "GFX1250_SCALE",
    )

    captured = {}

    def fake_triton_kernel_fused_experts(_out, _x, w13, w2, *_a, **kwargs):
        captured["w13"] = w13
        captured["w2"] = w2
        captured["w13_scale"] = kwargs["w13_scale"]
        captured["w13_swizzle_layout"] = kwargs["w13_swizzle_layout"]
        return "triton"

    import atom.model_ops.fused_moe_triton as triton_mod

    monkeypatch.setattr(
        triton_mod, "triton_kernel_fused_experts", fake_triton_kernel_fused_experts
    )
    monkeypatch.setattr(
        moe_mod, "routing_stub_unused", lambda *a, **k: None, raising=False
    )

    layer = SimpleNamespace(
        w13_weight="flydsl_w13",
        w2_weight="flydsl_w2",
        w13_weight_scale="flydsl_w13_scale",
        w2_weight_scale="flydsl_w2_scale",
        w13_swizzle_layout=None,
        w2_swizzle_layout=None,
        w13_input_scale=None,
        w2_input_scale=None,
        w13_bias=None,
        w2_bias=None,
        expert_mask="mask",
        swiglu_limit=0.0,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    captured["flydsl_calls"] = flydsl_calls
    return method, layer, captured


def test_prefill_falls_through_to_flydsl(monkeypatch):
    method, layer, captured = _apply_dispatch_probe(
        monkeypatch, use_triton_decode=True, is_prefill=True
    )

    result = method.apply(
        layer=layer,
        x="x",
        router_logits="router_logits",
        top_k=4,
        renormalize=False,
        global_num_experts=16,
        expert_map=None,
        activation="silu",
        apply_router_weight_on_input=False,
    )

    assert result == "flydsl"
    # Prefill must see the stored FlyDSL tensors, never the Triton view, and
    # the GUGU gate_mode the FlyDSL prep shuffled them for.
    args, kwargs = captured["flydsl_calls"].pop()
    assert args[1] == "flydsl_w13" and args[2] == "flydsl_w2"
    assert kwargs["w1_scale"] == "flydsl_w13_scale"
    assert kwargs["w2_scale"] == "flydsl_w2_scale"
    assert kwargs["gate_mode"] == moe_mod.GateMode.INTERLEAVE.value


def test_decode_takes_triton_over_a_view_of_the_flydsl_weights(monkeypatch):
    method, layer, captured = _apply_dispatch_probe(
        monkeypatch, use_triton_decode=True, is_prefill=False
    )
    # Reach the fused-experts call without standing up aiter's real routing().
    import aiter.ops.triton.moe.moe_routing.routing as routing_mod

    monkeypatch.setattr(
        routing_mod,
        "routing",
        lambda *a, **k: (SimpleNamespace(n_expts_act=4), "gather", "scatter"),
    )
    monkeypatch.setattr(torch, "empty_like", lambda _x: "out", raising=False)

    result = method.apply(
        layer=layer,
        x="x",
        router_logits=SimpleNamespace(shape=(2, 16)),
        top_k=4,
        renormalize=False,
        global_num_experts=16,
        expert_map=None,
        activation=moe_mod.ActivationType.Silu,
        apply_router_weight_on_input=False,
    )

    assert result == "triton"
    # The stored FlyDSL tensors must never reach the Triton kernel directly.
    assert captured["w13"] == "w13_view"
    assert captured["w2"] == "w2_view"
    assert captured["w13_scale"] == "w13_scale_view"
    assert captured["w13_swizzle_layout"] == "GFX1250_SCALE"
