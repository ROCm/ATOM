# SPDX-License-Identifier: MIT
"""ATOM_MINIMAX_M3_MOE_QUANT_SHARED: the routed MoE's A1 quant writes the
shared expert's MXFP8 input, so the shared gate_up skips `quantize_fp8`.

CPU checks of the wiring: which rows the copy is asked for, where it falls back
to ATOM's own quant, and that the shared expert takes the copy. The bytes
themselves are aiter's to match (op_tests/test_moe_quant_shared_mxfp8.py).
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
moe_mod = importlib.import_module("atom.model_ops.moe")
m3 = importlib.import_module("atom.models.minimax_m3.model")

T, H = 3, 64


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


def _m3_moe(enabled):
    moe = object.__new__(m3.MiniMaxM3MoE)
    torch.nn.Module.__init__(moe)
    moe.gate = lambda x: torch.zeros(x.shape[0], 4)
    moe.experts = _Experts()
    moe.shared_experts = _Shared()
    moe.fuse_shared_experts = False
    moe.routed_scaling_factor = 1.0
    moe.shared_input_from_moe_quant = enabled
    moe._use_dual_stream = False
    return moe


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_expert_takes_the_moe_quant_copy(enabled):
    moe = _m3_moe(enabled)
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


def _fused_moe(monkeypatch, dp=1, dp_repeat=1, mega=None):
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


def test_local_experts_get_the_copy(monkeypatch):
    layer, seen, filled = _fused_moe(monkeypatch)
    x, copy = torch.randn(T, H), _copy()
    layer.forward_impl(x, torch.zeros(T, 4), mxfp8_copy_out=copy)
    assert seen["apply"]["mxfp8_copy_out"] is copy and seen["apply"]["x"] is x
    assert not filled


def test_no_copy_keeps_apply_call(monkeypatch):
    layer, seen, filled = _fused_moe(monkeypatch)
    layer.forward_impl(torch.randn(T, H), torch.zeros(T, 4))
    assert "mxfp8_copy_out" not in seen["apply"] and not filled


@pytest.mark.parametrize("case", ["dp", "dp_repeat", "mega"])
def test_rows_the_experts_do_not_see_are_quantized_here(monkeypatch, case):
    mega = None
    if case == "mega":
        mega = SimpleNamespace(
            runs=lambda n: True, forward=lambda layer, h, logits: h.clone()
        )
    layer, seen, filled = _fused_moe(
        monkeypatch,
        dp=2 if case == "dp" else 1,
        dp_repeat=2 if case == "dp_repeat" else 1,
        mega=mega,
    )
    monkeypatch.setattr(
        moe_mod, "repeat_rows", lambda t, n: torch.cat([t] * n), raising=False
    )
    x, copy = torch.randn(T, H), _copy()
    layer.forward_impl(x, torch.zeros(T, 4), mxfp8_copy_out=copy)
    assert len(filled) == 1 and filled[0][0] is x and filled[0][1] is copy
    assert "mxfp8_copy_out" not in seen.get("apply", {})


def test_fill_mxfp8_copy_writes_quantize_fp8(monkeypatch):
    blockscale = importlib.import_module("atom.model_ops.blockscale")
    q = torch.arange(T * H, dtype=torch.uint8).reshape(T, H)
    s = torch.arange(T * H // 32, dtype=torch.uint8).reshape(T, H // 32)
    monkeypatch.setattr(blockscale, "quantize_fp8", lambda x: (q, s))
    copy = _copy()
    moe_mod.fill_mxfp8_copy(torch.randn(T, H), copy)
    assert torch.equal(copy[0], q) and torch.equal(copy[1], s)
    moe_mod.fill_mxfp8_copy(torch.randn(T, H), None)  # nothing asked, nothing done
