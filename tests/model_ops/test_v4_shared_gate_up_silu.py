# SPDX-License-Identifier: MIT
"""The V4 shared expert's gate_up GEMM with the SwiGLU and the MXFP8 quant
for w2 in its epilogue (`Expert._fused_gate_up_silu`): when it is taken, what
it hands AITER and w2, and that everything else keeps the unfused path."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _expert(atom_modules, *, limit=10.0, shuffled=True, group=32, bias=None):
    v4 = atom_modules.deepseek_v4
    expert = v4.Expert.__new__(v4.Expert)
    torch.nn.Module.__init__(expert)
    weight = torch.zeros(2 * 64, 256, dtype=torch.float8_e4m3fn)
    weight.is_shuffled = shuffled
    expert.gate_up_proj = SimpleNamespace(
        weight=weight,
        weight_scale=torch.zeros(4, 8, dtype=torch.float8_e8m0fnu),
        native_a8_group_rows=group,
        bias=bias,
    )
    expert.w2 = MagicMock(name="w2", native_a8_group_rows=32)
    expert.swiglu_limit = limit
    expert.use_fused_clamp_act_mul = False
    expert._gate_up_silu = None
    return expert


@pytest.fixture
def flydsl(monkeypatch, atom_modules):
    calls = {}
    module = ModuleType("aiter.ops.flydsl.batched_gemm_a8w8_gfx1250")

    def kernel_for(m, n2, k):
        calls["shape"] = (m, n2, k)
        return None if m > 4096 else "kernel"

    def run(xq, wq, xs, ws, out, out_scale, limit, kernel_name=None):
        calls["run"] = (xq, wq, xs, ws, out, out_scale, limit, kernel_name)
        return out

    module.gate_up_silu_kernel_for = kernel_for
    module.run_gate_up_silu_mxfp8_32_gfx1250 = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    v4 = atom_modules.deepseek_v4
    monkeypatch.setattr(v4, "get_gfx_runtime", lambda: "gfx1250")
    blockscale = importlib.import_module("atom.model_ops.blockscale")
    quantized = (
        torch.zeros(3, 256, dtype=torch.float8_e4m3fn),
        torch.zeros(3, 8, dtype=torch.float8_e8m0fnu),
    )
    monkeypatch.setattr(blockscale, "quantize_fp8", lambda x: quantized)
    calls["quantized"] = quantized
    return calls


def test_fused_gate_up_feeds_w2_the_quantized_activation(atom_modules, flydsl):
    expert = _expert(atom_modules)
    x = torch.zeros(3, 256, dtype=torch.bfloat16)
    expert.w2.return_value = "out"
    assert expert.forward(x) == "out"
    xq, wq, xs, ws, act, act_scale, limit, name = flydsl["run"]
    assert (xq, xs) == flydsl["quantized"]
    assert wq is expert.gate_up_proj.weight
    assert ws is expert.gate_up_proj.weight_scale
    assert act.shape == (3, 64) and act.dtype == torch.float8_e4m3fn
    assert act_scale.shape == (3, 2) and act_scale.dtype == torch.float8_e8m0fnu
    assert limit == 10.0 and name == "kernel"
    assert flydsl["shape"] == (3, 128, 256)
    args, kwargs = expert.w2.call_args
    assert args[0] is act and kwargs["x_scale"] is act_scale


@pytest.mark.parametrize(
    "change",
    [
        {"limit": 0.0},
        {"shuffled": False},
        {"group": None},
        {"bias": torch.zeros(128)},
    ],
)
def test_ineligible_experts_keep_the_unfused_path(atom_modules, flydsl, change):
    expert = _expert(atom_modules, **change)
    assert expert._fused_gate_up_silu(torch.zeros(3, 256, dtype=torch.bfloat16)) is None
    assert "run" not in flydsl


def test_other_arch_and_inputs_keep_the_unfused_path(monkeypatch, atom_modules, flydsl):
    expert = _expert(atom_modules)
    assert expert._fused_gate_up_silu(torch.zeros(3, 256)) is None  # fp32 rows
    assert (
        expert._fused_gate_up_silu(torch.zeros(5000, 256, dtype=torch.bfloat16)) is None
    )
    assert flydsl["shape"] == (5000, 128, 256)  # the kernel table said no
    other = _expert(atom_modules)
    monkeypatch.setattr(atom_modules.deepseek_v4, "get_gfx_runtime", lambda: "gfx950")
    assert other._fused_gate_up_silu(torch.zeros(3, 256, dtype=torch.bfloat16)) is None
    assert "run" not in flydsl
