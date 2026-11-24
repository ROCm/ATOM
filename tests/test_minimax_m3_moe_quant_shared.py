# SPDX-License-Identifier: MIT
"""ATOM_MINIMAX_M3_MOE_QUANT_SHARED: the routed MoE's A1 quant writes the
shared expert's MXFP8 input, so the shared gate_up skips `quantize_fp8`.

CPU checks of the wiring: which rows the copy is asked for, where it falls back
to ATOM's own quant, and that the shared expert takes the copy. The bytes
themselves are aiter's to match (op_tests/test_moe_quant_shared_mxfp8.py).
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from conftest import gpu_stack_import

T, H = 3, 64


@pytest.fixture(scope="module")
def mods():
    # The MoE wiring runs on the CPU job against aiter stand-ins; the model
    # module needs the real stack (gpu_stack_import skips it there).
    with atom_without_aiter() as load:
        moe, blockscale = load("atom.model_ops.moe"), load("atom.model_ops.blockscale")
        yield SimpleNamespace(
            moe=moe, blockscale=blockscale, m3=lambda: gpu_stack_import(_M3)
        )


_M3 = "atom.models.minimax_m3.model"


def _copy():
    return (
        torch.empty(T, H, dtype=torch.uint8),
        torch.empty(T, H // 32, dtype=torch.uint8),
    )


class _Experts:
    def __init__(self):
        self.calls = []

    def __call__(self, hidden_states, router_logits):
        self.calls.append(("plain", hidden_states))
        return hidden_states * 2

    def forward_with_mxfp8_copy(self, hidden_states, router_logits):
        self.calls.append(("copy", hidden_states))
        return hidden_states * 2, "q", "s"


class _Shared:
    def __init__(self):
        self.calls = []

    def __call__(self, x, x_scale=None):
        self.calls.append((x, x_scale))
        return torch.ones(T, H)


def _m3_moe(m3, enabled):
    moe = object.__new__(m3.MiniMaxM3MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = lambda x: torch.zeros(x.shape[0], 4)
    moe.experts = _Experts()
    moe.shared_experts = _Shared()
    moe.fuse_shared_experts = False
    moe.routed_scaling_factor = 1.0
    moe.shared_input_from_moe_quant = enabled
    return moe


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_expert_takes_the_moe_quant_copy(mods, enabled):
    moe = _m3_moe(mods.m3(), enabled)
    x = torch.randn(T, H)
    out = moe(x)
    assert torch.equal(out, x * 2 + 1)
    kind, rows = moe.experts.calls[0]
    assert kind == ("copy" if enabled else "plain") and torch.equal(rows, x)
    shared_x, shared_scale = moe.shared_experts.calls[0]
    if enabled:
        assert (shared_x, shared_scale) == ("q", "s")
    else:
        assert torch.equal(shared_x, x) and shared_scale is None


@pytest.mark.parametrize(
    "flag,group_rows,runs_bf16,mxfp4,want",
    [
        ("1", 32, False, True, True),
        ("0", 32, False, True, False),
        ("1", None, False, True, False),  # not a native group-32 linear
        ("1", 32, True, True, False),  # BF16 fallback: takes no x_scale
        ("1", 32, False, False, False),  # experts not on the MXFP4 method
    ],
)
def test_gate(mods, monkeypatch, flag, group_rows, runs_bf16, mxfp4, want):
    monkeypatch.setenv("ATOM_MINIMAX_M3_MOE_QUANT_SHARED", flag)
    gate_up = SimpleNamespace(native_a8_group_rows=group_rows, runs_bf16=runs_bf16)
    method = object.__new__(mods.moe.Mxfp4MoEMethod) if mxfp4 else object()
    shared = SimpleNamespace(gate_up_proj=gate_up)
    experts = SimpleNamespace(quant_method=method)
    m3 = mods.m3()
    assert m3.shared_input_from_moe_quant(shared, experts) is want
    assert m3.shared_input_from_moe_quant(None, experts) is False


def _fused_moe(moe_mod, monkeypatch, dp=1, dp_repeat=1, mega=None):
    layer = object.__new__(moe_mod.FusedMoE)
    torch.nn.Module.__init__(layer)
    seen = {}

    def apply(**kw):
        seen["apply"] = kw
        return kw["x"].clone()

    layer.quant_method = SimpleNamespace(
        select_experts=lambda self, x, logits: (
            torch.ones(x.shape[0], 2),
            torch.zeros(x.shape[0], 2, dtype=torch.int32),
        ),
        place_experts=lambda self, w, ids, gathered: (w, ids),
        apply=apply,
    )
    layer.sp_mega_moe = mega
    layer.moe_parallel_config = SimpleNamespace(
        pcp_merge=False, dp_logical_ratio=dp_repeat
    )
    layer.global_num_experts = 4
    layer.expert_map = None
    layer.activation = None
    layer.apply_router_weight_on_input = False
    layer.prefix = "m"
    layer.reduce_results = False
    layer.forward_impl_graph = lambda h, logits, **kw: seen.setdefault("graph", h)
    monkeypatch.setattr(moe_mod, "get_dp_group", lambda: SimpleNamespace(world_size=dp))
    monkeypatch.setattr(
        moe_mod,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False, pcp_split=False)
        ),
    )
    monkeypatch.setattr(moe_mod, "pass_scheduled_tokens_gpu", lambda n: None)
    filled = []
    monkeypatch.setattr(
        moe_mod, "fill_mxfp8_copy", lambda h, out: filled.append((h, out))
    )
    return layer, seen, filled


def test_local_experts_get_the_copy(mods, monkeypatch):
    layer, seen, filled = _fused_moe(mods.moe, monkeypatch)
    x, copy = torch.randn(T, H), _copy()
    layer.forward_impl(x, torch.zeros(T, 4), mxfp8_copy_out=copy)
    assert seen["apply"]["mxfp8_copy_out"] is copy and seen["apply"]["x"] is x
    assert not filled


def test_no_copy_keeps_apply_call(mods, monkeypatch):
    layer, seen, filled = _fused_moe(mods.moe, monkeypatch)
    layer.forward_impl(torch.randn(T, H), torch.zeros(T, 4))
    assert "mxfp8_copy_out" not in seen["apply"] and not filled


@pytest.mark.parametrize("case", ["dp", "dp_repeat", "mega"])
def test_rows_the_experts_do_not_see_are_quantized_here(mods, monkeypatch, case):
    mega = None
    if case == "mega":
        mega = SimpleNamespace(
            runs=lambda n: True, forward=lambda layer, h, logits: h.clone()
        )
    layer, seen, filled = _fused_moe(
        mods.moe,
        monkeypatch,
        dp=2 if case == "dp" else 1,
        dp_repeat=2 if case == "dp_repeat" else 1,
        mega=mega,
    )
    monkeypatch.setattr(
        mods.moe, "repeat_rows", lambda t, n: torch.cat([t] * n), raising=False
    )
    x, copy = torch.randn(T, H), _copy()
    layer.forward_impl(x, torch.zeros(T, 4), mxfp8_copy_out=copy)
    assert len(filled) == 1 and filled[0][0] is x and filled[0][1] is copy
    assert "mxfp8_copy_out" not in seen.get("apply", {})


def test_fill_mxfp8_copy_quantizes_into_the_copy(mods, monkeypatch):
    """The fallback is the shared expert's one quant launch, written straight
    into the copy's buffers (no second launch to copy it there)."""
    calls = []
    monkeypatch.setattr(
        mods.blockscale, "quantize_fp8", lambda x, **kw: calls.append((x, kw))
    )
    x, copy = torch.randn(T, H), _copy()
    mods.moe.fill_mxfp8_copy(x, copy)
    assert len(calls) == 1 and calls[0][0] is x and calls[0][1] == {"out": copy}
    mods.moe.fill_mxfp8_copy(x, None)  # nothing asked, nothing done
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton quant on a GPU")
def test_quantize_fp8_out_matches_its_own_outputs(mods):
    x = torch.randn(T, 256, device="cuda", dtype=torch.bfloat16)
    want = mods.blockscale.quantize_fp8(x)
    got = tuple(torch.full_like(t.view(torch.uint8), 0x5A).view(t.dtype) for t in want)
    mods.blockscale.quantize_fp8(x, out=got)
    for g, w in zip(got, want):
        assert torch.equal(g.view(torch.uint8), w.view(torch.uint8))
    with pytest.raises(ValueError):
        mods.blockscale.quantize_fp8(x, out=(got[0][:1], got[1]))


def test_op_returns_the_copy_its_fake_describes(mods, monkeypatch):
    """`moe_forward_mxfp8_copy` hands the layer fresh copy buffers and returns
    them (scale as bytes) with the dtypes and shapes its fake promises;
    `forward_with_mxfp8_copy` views the scale back to E8M0."""
    moe = mods.moe

    def forward_impl(h, logits, mxfp8_copy_out):
        mxfp8_copy_out[0].view(torch.uint8).fill_(7)
        mxfp8_copy_out[1].view(torch.uint8).fill_(9)
        return h * 2

    ctx = {"L": SimpleNamespace(forward_impl=forward_impl)}
    monkeypatch.setattr(
        moe,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=ctx)
        ),
    )
    x, logits = torch.randn(T, H, dtype=torch.bfloat16), torch.zeros(T, 4)
    got = moe.moe_forward_mxfp8_copy(x, logits, "L")
    fake = moe.moe_forward_mxfp8_copy_fake(x, logits, "L")
    for g, f in zip(got, fake):
        assert (g.shape, g.dtype) == (f.shape, f.dtype)
    assert torch.equal(got[0], x * 2)
    assert bool((got[1].view(torch.uint8) == 7).all()) and bool((got[2] == 9).all())

    layer = object.__new__(moe.FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.layer_name, layer.balance_router_logits = "L", None
    monkeypatch.setattr(moe, "moe_forward_mxfp8_copy_op", moe.moe_forward_mxfp8_copy)
    *_, scale = layer.forward_with_mxfp8_copy(x, logits)
    assert scale.dtype == torch.float8_e8m0fnu and scale.shape == (T, H // 32)


@pytest.mark.parametrize("modular", [False, True])
def test_mxfp4_apply_hands_the_copy_to_fused_moe(mods, monkeypatch, modular):
    """FlyDSL experts: `fused_moe` gets `mxfp8_copy_out` and the local rows; a
    modular kernel takes no copy, so `apply` quantizes the rows itself."""
    moe = mods.moe
    method = object.__new__(moe.Mxfp4MoEMethod)
    seen = {}
    method.__dict__.update(
        use_triton=False,
        use_triton_decode=False,
        use_triton_ep=False,
        using_modular_kernel=False,
        use_flydsl_herd=False,
        is_guinterleave=True,
        fp8_activations=False,
        quant_type=None,
        hidden_pad=0,
        intermediate_pad=0,
        fused_experts=(
            (lambda **kw: seen.setdefault("modular", kw)) if modular else None
        ),
    )
    monkeypatch.setattr(
        moe, "fused_moe", lambda x, *a, **kw: seen.setdefault("fused_moe", (x, kw))
    )
    filled = []
    monkeypatch.setattr(moe, "fill_mxfp8_copy", lambda h, out: filled.append((h, out)))
    layer = SimpleNamespace(
        top_k=2,
        **dict.fromkeys(
            ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"), None
        ),
        **dict.fromkeys(("expert_mask", "w13_bias", "w2_bias"), None),
    )
    x, copy = torch.randn(T, H), _copy()
    ids = torch.zeros(T, 2, dtype=torch.int32)
    method.apply(layer, x, torch.ones(T, 2), ids, mxfp8_copy_out=copy)
    if modular:
        assert filled == [(x, copy)] and "mxfp8_copy_out" not in str(seen["modular"])
    else:
        got_x, kw = seen["fused_moe"]
        assert got_x is x and kw["mxfp8_copy_out"] is copy and not filled
