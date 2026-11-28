# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 decode MoE wiring (``decode_moe``): weight re-layouts, the layer gate and the row / fallback
dispatch, on CPU with fake modules and fake aiter ops. The kernels themselves are tested in aiter.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.models.minimax_m3 import decode_moe as dm


def test_routed_w13_layout_interleaves_gate_up_blocks():
    e, i, rest = 3, 32, 5
    w = torch.arange(e * 2 * (i // 16) * rest, dtype=torch.int32).to(torch.uint8)
    w = w.view(e, 2 * (i // 16) * rest)
    out = dm.routed_w13_layout(w, e, i)
    src = w.view(e, 2, i // 16, rest)
    assert out.shape == (e, i // 16, 2, rest) and out.is_contiguous()
    for blk in range(i // 16):
        assert torch.equal(out[:, blk, 0], src[:, 0, blk])  # gate block
        assert torch.equal(out[:, blk, 1], src[:, 1, blk])  # up block


def test_shared_scale_layouts_are_permutations():
    s13 = torch.randint(0, 255, (1536, 192), dtype=torch.uint8)
    s2 = torch.randint(0, 255, (6144, 24), dtype=torch.uint8)
    b13, b2 = dm.shared_scale_layouts(s13, s2)
    assert b13.numel() == s13.numel() and b2.numel() == s2.numel()
    assert b13.is_contiguous() and b2.is_contiguous()
    assert torch.equal(b13.flatten().sort().values, s13.flatten().sort().values)
    assert torch.equal(b2.flatten().sort().values, s2.flatten().sort().values)
    assert not torch.equal(b13.flatten(), s13.flatten())


def _qm(**kw):
    return SimpleNamespace(
        **{"is_guinterleave": False, "intermediate_pad": 0, "hidden_pad": 0, **kw}
    )


class _Sized:
    def __init__(self, numel, shuffled=True, dtype=torch.uint8):
        self._n, self.is_shuffled, self.dtype = numel, shuffled, dtype
        self.data = torch.empty(0, dtype=dtype)

    def numel(self):
        return self._n


def _shuffled(t, flag=True):
    t.is_shuffled = flag
    return t


def _fake_moe(**over):
    gate_w = nn.Parameter(torch.zeros(1, 1))
    gate_w.weight_loader = lambda param, w, *a, **k: None
    ex = SimpleNamespace(
        quant_method=_qm(),
        num_fused_shared_experts=0,
        swiglu_limit=7.0,
        top_k=dm.TOPK,
        scoring_func="sigmoid",
        use_ep=False,
        w13_weight=torch.empty(dm.E * 2 * dm.I * dm.H // 2, dtype=torch.uint8),
        w2_weight=torch.empty(dm.E * dm.H * dm.I // 2, dtype=torch.uint8),
        w13_weight_scale=torch.empty(dm.E * 2 * dm.I * dm.H // 32, dtype=torch.uint8),
        w2_weight_scale=torch.empty(dm.E * dm.H * dm.I // 32, dtype=torch.uint8),
    )
    for n in ("w13_weight", "w2_weight"):
        getattr(ex, n).is_shuffled = True
    sh = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=_shuffled(torch.empty(2 * dm.I, dm.H, dtype=torch.float8_e4m3fn)),
            weight_scale=torch.empty(2 * dm.I, dm.H // 32, dtype=torch.uint8),
        ),
        down_proj=SimpleNamespace(
            weight=_shuffled(torch.empty(dm.H, dm.I, dtype=torch.float8_e4m3fn)),
            weight_scale=torch.empty(dm.H, dm.I // 32, dtype=torch.uint8),
        ),
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
    )
    moe = SimpleNamespace(
        gate=SimpleNamespace(weight=gate_w),
        experts=ex,
        shared_experts=sh,
        e_score_correction_bias=torch.zeros(dm.E),
        routed_scaling_factor=2.0,
    )
    for k, v in over.items():
        setattr(moe, k, v)
    return moe


def _layer(moe, name):
    layer = dm.DecodeMoe(moe, name)
    layer._gate_fp32 = torch.zeros(dm.E, dm.H)
    return layer


def test_supported_layer_passes_the_gate():
    assert _layer(_fake_moe(), "ok")._unsupported(_fake_moe()) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: setattr(m, "routed_scaling_factor", 1.0),
        lambda m: setattr(m, "shared_experts", None),
        lambda m: setattr(m, "e_score_correction_bias", None),
        lambda m: setattr(m.experts, "num_fused_shared_experts", 1),
        lambda m: setattr(m.experts.quant_method, "is_guinterleave", True),
        lambda m: setattr(m.experts.quant_method, "intermediate_pad", 1),
        lambda m: setattr(m.experts, "swiglu_limit", 6.0),
        lambda m: setattr(m.shared_experts, "swiglu_alpha", 1.0),
        lambda m: setattr(m.experts.w13_weight, "is_shuffled", False),
        lambda m: setattr(m.shared_experts.gate_up_proj.weight, "is_shuffled", False),
        lambda m: setattr(m.shared_experts.down_proj.weight, "is_shuffled", False),
        lambda m: setattr(m.experts, "top_k", 8),
        lambda m: setattr(m.experts, "scoring_func", "softmax"),
        lambda m: setattr(m.experts, "use_ep", True),
        lambda m: setattr(m.shared_experts, "swiglu_limit", None),
        lambda m: setattr(m.shared_experts, "swiglu_alpha", None),
        lambda m: setattr(m.experts, "swiglu_limit", None),
        lambda m: setattr(m.experts, "w2_weight", torch.empty(4, dtype=torch.uint8)),
        lambda m: setattr(
            m.shared_experts.down_proj,
            "weight",
            torch.empty(dm.H, dm.I, dtype=torch.bfloat16),
        ),
    ],
)
def test_unsupported_layers_keep_the_default_path(mutate):
    moe = _fake_moe()
    layer = _layer(moe, "bad")
    mutate(moe)
    assert layer._unsupported(moe) is not None


def test_missing_fp32_gate_is_refused():
    moe = _fake_moe()
    layer = dm.DecodeMoe(moe, "nogate")
    assert "FP32" in layer._unsupported(moe)


def test_rows_gate_and_ops_missing(monkeypatch):
    monkeypatch.setattr(dm, "load_ops", lambda: None)
    layer = dm.DecodeMoe(_fake_moe(), "noops")
    layer.process_weights_after_loading()  # no ops: stays on the default path
    assert not layer.ready and not layer.runs(16)
    layer.ready = True
    assert (
        not layer.runs(7) and layer.runs(8) and layer.runs(128) and not layer.runs(129)
    )


class _Ops:
    def __init__(self, routed):
        self.calls = []
        self.router = SimpleNamespace(route_quant_small=self._route)
        self.gemms = SimpleNamespace(
            get_plan=lambda n, **kw: n, decode_expert_gemms=self._gemms
        )
        self._routed = routed

    def _route(self, x, gate, lines, bias, scale):
        self.calls.append("route")
        return self._routed

    def _gemms(self, plan, *a):
        self.calls.append("gemms")
        return torch.zeros(plan, 4, dm.H, dtype=torch.bfloat16), torch.zeros(
            plan, dm.H, dtype=torch.bfloat16
        )

    def reduce(self, slots, shared):
        self.calls.append("reduce")
        return shared

    def tuple(self):
        return self.router, self.gemms, None, self.reduce


def _dispatch_layer(name, routed):
    moe = SimpleNamespace(default_forward=lambda x: torch.full_like(x, 7))
    layer = dm.DecodeMoe(_fake_moe(), name)
    layer._moe = [moe]
    ops = _Ops(routed)
    layer.ops = ops.tuple()
    layer.ready = True
    layer.plan_kw = {}
    for n in (
        "gate",
        "gate_lines",
        "bias",
        "w13",
        "s13",
        "w2",
        "s2",
        "sw13",
        "ss13",
        "sw2",
        "ss2",
    ):
        setattr(layer, n, None)
    return ops


def _q(n):
    z = torch.zeros(n, 4, dtype=torch.uint8)
    return (z, z), (z, z), torch.zeros(n, 4), torch.zeros(n, 4, dtype=torch.int32)


def test_rows_outside_range_use_the_default_path():
    ops = _dispatch_layer("d1", _q(4))
    out = dm.decode_moe_forward(torch.zeros(4, dm.H, dtype=torch.bfloat16), "d1")
    assert ops.calls == [] and bool((out == 7).all())


def test_rows_in_range_run_the_chain():
    ops = _dispatch_layer("d2", _q(16))
    out = dm.decode_moe_forward(torch.zeros(16, dm.H, dtype=torch.bfloat16), "d2")
    assert ops.calls == ["route", "gemms", "reduce"] and out.shape == (16, dm.H)


def test_uncovered_router_input_falls_back():
    ops = _dispatch_layer("d3", None)
    out = dm.decode_moe_forward(torch.zeros(16, dm.H, dtype=torch.bfloat16), "d3")
    assert ops.calls == ["route"] and bool((out == 7).all())


def test_split_layout_probe():
    assert dm.split_layout_supported(
        SimpleNamespace(get_plan=lambda n, asm=True, w13_split=False: n)
    )
    assert not dm.split_layout_supported(
        SimpleNamespace(get_plan=lambda n, asm=True: n)
    )


def test_weight_discard_sleep_drops_the_views_and_keeps_the_default_path(caplog):
    layer = _dispatch_layer_obj()
    assert layer.ready
    layer.release_weight_memory()
    assert not layer.ready and layer.w13 is None and layer.ss2 is None
    layer.resume_weight_memory()
    assert not layer.runs(16)
    # the memory manager finds the hooks
    assert hasattr(layer, "resume_weight_memory")


def _dispatch_layer_obj():
    layer = dm.DecodeMoe(_fake_moe(), "sleep")
    layer.ready = True
    for n in ("gate", "gate_lines", "bias", "w13", "s13", "w2", "s2", "sw13", "sw2"):
        setattr(layer, n, torch.zeros(1))
    layer.ss13 = layer.ss2 = torch.zeros(1)
    return layer


@pytest.mark.parametrize(
    "rows,chain", [(1, False), (7, False), (8, True), (128, True), (129, False)]
)
def test_model_forward_decides_the_fallback_outside_the_op(monkeypatch, rows, chain):
    # `atom.models.minimax_m3.model` pulls in the mono decode kernel chain
    # (`mono.dispatch` -> ... -> `mono.kernels.index_score`), which bakes a
    # real aiter-sourced integer (`LUT_ENTRIES`) into an `fx.Array[...]`
    # class-body annotation evaluated at import time; `atom_without_aiter()`'s
    # MagicMock stand-ins fail that annotation's int check, so this needs a
    # real aiter and a live CUDA device, neither of which the CPU CI runner
    # has (same reasoning as tests/test_minimax_m3_sparse_pa_floor.py).
    pytest.importorskip("aiter")
    from atom.models.minimax_m3 import model as m

    layer = _dispatch_layer_obj()
    ops = []
    monkeypatch.setattr(
        m.decode_moe, "decode_moe_op", lambda x, name: ops.append(rows) or x
    )
    moe = SimpleNamespace(
        decode_moe=layer, default_forward=lambda x: torch.full_like(x, 7)
    )
    x = torch.zeros(rows, dm.H, dtype=torch.bfloat16)
    out = m.MiniMaxM3MoE.forward(moe, x)
    assert bool(ops) == chain and (chain or bool((out == 7).all()))
