# SPDX-License-Identifier: MIT
"""The V4 shared expert's gate_up GEMM with the SwiGLU and the MXFP8 quant
for w2 in its epilogue (`Expert._fused_gate_up_silu`): when it is taken, what
it hands AITER and w2, and that everything else keeps the unfused path. The
decision is made once at construction, never in forward."""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _expert(
    monkeypatch,
    atom_modules,
    *,
    limit=10.0,
    stored_preshuffled=True,
    group=32,
    fused_act=False,
    w2_group=32,
):
    v4 = atom_modules.deepseek_v4
    expert = v4.Expert.__new__(v4.Expert)
    torch.nn.Module.__init__(expert)
    weight = torch.zeros(2 * 64, 256, dtype=torch.float8_e4m3fn)
    expert.gate_up_proj = SimpleNamespace(
        weight=weight,
        weight_scale=torch.zeros(4, 8, dtype=torch.float8_e8m0fnu),
        native_a8_group_rows=group,
        quant_type="per_1x32",
    )
    expert.w2 = MagicMock(name="w2", native_a8_group_rows=w2_group)
    expert.swiglu_limit = limit
    expert.use_fused_clamp_act_mul = fused_act
    monkeypatch.setattr(
        v4, "weight_is_stored_preshuffled", lambda *a, **k: stored_preshuffled
    )
    expert._gate_up_silu = expert._gate_up_silu_applies()
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


def _no_arch_query(monkeypatch, atom_modules):
    def refuse():
        raise AssertionError("forward queried the GPU architecture")

    monkeypatch.setattr(atom_modules.deepseek_v4, "get_gfx_runtime", refuse)


def test_fused_gate_up_feeds_w2_the_quantized_activation(
    monkeypatch, atom_modules, flydsl
):
    expert = _expert(monkeypatch, atom_modules)
    assert expert._gate_up_silu
    _no_arch_query(monkeypatch, atom_modules)
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


def test_gated_forward_keeps_the_unfused_path(monkeypatch, atom_modules, flydsl):
    """`forward(x, weights=...)` (a gated expert) never takes the fused path:
    the epilogue has no per-row gate."""
    expert = _expert(monkeypatch, atom_modules)
    assert expert._gate_up_silu
    expert.gate_up_proj = MagicMock(return_value=torch.zeros(3, 128))
    monkeypatch.setattr(
        atom_modules.deepseek_v4, "aiter_silu_and_mul", lambda out, x, limit: None
    )
    expert.w2.return_value = "out"
    x = torch.zeros(3, 256, dtype=torch.bfloat16)
    weights = torch.ones(3, 1, dtype=torch.bfloat16)
    assert expert.forward(x, weights=weights) == "out"
    assert "run" not in flydsl and "shape" not in flydsl
    expert.gate_up_proj.assert_called_once_with(x)


@pytest.mark.parametrize(
    "change",
    [
        {"limit": 0.0},
        {"stored_preshuffled": False},
        {"group": None},
        {"fused_act": True},
        {"w2_group": None},
    ],
)
def test_ineligible_experts_keep_the_unfused_path(
    monkeypatch, atom_modules, flydsl, change
):
    expert = _expert(monkeypatch, atom_modules, **change)
    assert not expert._gate_up_silu
    assert expert._fused_gate_up_silu(torch.zeros(3, 256, dtype=torch.bfloat16)) is None
    assert "run" not in flydsl


def test_other_arch_and_inputs_keep_the_unfused_path(monkeypatch, atom_modules, flydsl):
    expert = _expert(monkeypatch, atom_modules)
    assert expert._fused_gate_up_silu(torch.zeros(3, 256)) is None  # fp32 rows
    assert (
        expert._fused_gate_up_silu(torch.zeros(5000, 256, dtype=torch.bfloat16)) is None
    )
    assert flydsl["shape"] == (5000, 128, 256)  # the kernel table said no
    monkeypatch.setattr(atom_modules.deepseek_v4, "get_gfx_runtime", lambda: "gfx950")
    other = _expert(monkeypatch, atom_modules)
    assert not other._gate_up_silu
    assert other._fused_gate_up_silu(torch.zeros(3, 256, dtype=torch.bfloat16)) is None
    assert "run" not in flydsl


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_init_decides_and_forward_never_queries_the_arch(
    monkeypatch, atom_modules, arch
):
    """The decision is made in `Expert.__init__`; forward, which runs inside
    torch.compile regions, never calls `get_gfx_runtime` (its file lookup
    cannot be traced)."""
    v4 = atom_modules.deepseek_v4
    queries = []

    def gfx():
        queries.append(arch)
        return arch

    def linear(*args, **kwargs):
        return SimpleNamespace(
            weight=torch.zeros(128, 256, dtype=torch.float8_e4m3fn),
            weight_scale=torch.zeros(4, 8, dtype=torch.float8_e8m0fnu),
            native_a8_group_rows=32,
            quant_type="per_1x32",
            input_size=64,
        )

    monkeypatch.setattr(v4, "get_gfx_runtime", gfx)
    monkeypatch.setattr(v4, "MergedColumnParallelLinear", linear)
    monkeypatch.setattr(v4, "RowParallelLinear", linear)
    monkeypatch.setattr(v4, "shared_expert_fused_act_enabled", lambda width: False)
    monkeypatch.setattr(v4, "weight_is_stored_preshuffled", lambda *a, **k: True)
    expert = v4.Expert(256, 64, swiglu_limit=10.0)
    assert queries == [arch]
    assert expert._gate_up_silu == (arch == "gfx1250")

    _no_arch_query(monkeypatch, atom_modules)
    expert._gate_up_silu = False  # the unfused path below is stubbed out
    expert.gate_up_proj = MagicMock(return_value=torch.zeros(3, 128))
    expert.w2 = MagicMock(return_value="out", native_a8_group_rows=32)
    monkeypatch.setattr(v4, "aiter_silu_and_mul", lambda out, x, limit: None)
    assert expert.forward(torch.zeros(3, 256, dtype=torch.bfloat16)) == "out"
