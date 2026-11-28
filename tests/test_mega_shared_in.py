# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The shared expert a Mega target layer hands to its routed call (AITER's
``shared_in``, ROCm/aiter#8491).

CPU only: AITER's ``SharedExpertInput`` is stood in by the part of its contract
ATOM relies on (it reads ``x_fp8`` / ``x_scale`` only for a call it fuses and
calls ``dense()`` otherwise); the GPU kernels are not exercised here.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    # aiter/triton absent under bare non-GPU pytest
    import atom.model_ops.moe as moe_mod
    from atom.model_ops.fused_moe import flydsl_mega_experts as mega
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


class _SharedExpertInput:
    """ROCm/aiter#8491's constructor and dense fallback."""

    def __init__(self, x_fp8, x_scale, gate_up, gate_up_scale, down, down_scale, dense):
        self.x_fp8, self.x_scale = x_fp8, x_scale
        self.gate_up, self.gate_up_scale = gate_up, gate_up_scale
        self.down, self.down_scale = down, down_scale
        self._dense = dense
        self.used_dense = False

    def dense(self):
        self.used_dense = True
        return self._dense()


@pytest.fixture
def aiter_weights(monkeypatch):
    """AITER's mega_moe.weights module, with or without shared_in, and a counted
    stand-in of ATOM's MXFP8 quantization."""
    state = SimpleNamespace(quantized=0)

    def install(with_shared_in=True):
        module = ModuleType("aiter.ops.flydsl.kernels.mega_moe.weights")
        if with_shared_in:
            module.SharedExpertInput = _SharedExpertInput
        package = ModuleType("aiter.ops.flydsl.kernels.mega_moe")
        package.weights = module
        monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.kernels.mega_moe", package)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        mega._shared_expert_input_cls.cache_clear()

    def quantize_fp8(x):
        state.quantized += 1
        return x.to(torch.float8_e4m3fn), torch.zeros(x.shape[0], x.shape[1] // 32)

    blockscale = ModuleType("atom.model_ops.blockscale")
    blockscale.quantize_fp8 = quantize_fp8
    monkeypatch.setitem(sys.modules, "atom.model_ops.blockscale", blockscale)
    state.sp = False
    monkeypatch.setattr("atom.distributed.ulysses_sp.sp_is_enabled", lambda: state.sp)
    state.install = install
    yield state
    mega._shared_expert_input_cls.cache_clear()


class _Expert:
    def __init__(self):
        self.calls = []
        weight = SimpleNamespace(weight=torch.zeros(4), weight_scale=torch.zeros(1))
        self.gate_up_proj = self.w2 = weight

    def __call__(self, x):
        self.calls.append(x.shape[0])
        return x * 2


def _layer(**overrides):
    values = {
        "quant_method": object.__new__(moe_mod.MegaMxfp4MoEMethod),
        "_comm_fused_moe": None,
        "reduce_results": False,
        "moe_parallel_config": SimpleNamespace(pcp_merge=False, dp_logical_ratio=1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_x_is_quantized_only_when_the_kernel_reads_it(aiter_weights):
    aiter_weights.install()
    expert, x = _Expert(), torch.randn(6, 64)

    fused = mega._shared_expert_input(expert, x)
    assert aiter_weights.quantized == 0
    x_fp8, x_scale = fused.x_fp8, fused.x_scale
    assert fused.x_fp8 is x_fp8 and fused.x_scale is x_scale
    assert aiter_weights.quantized == 1 and expert.calls == []

    unfused = mega._shared_expert_input(expert, x)
    assert torch.equal(unfused.dense(), x * 2)
    assert unfused.used_dense and expert.calls == [6]
    assert aiter_weights.quantized == 1


def test_attached_only_on_a_mega_layer_every_routed_call_reaches(aiter_weights):
    aiter_weights.install()
    expert = _Expert()
    layer = _layer()
    assert mega.attach_mega_shared_expert(layer, expert)
    assert layer._mega_shared_expert is expert

    refused = {
        "standard method": _layer(quant_method=object.__new__(moe_mod.Mxfp4MoEMethod)),
        "comm-fused": _layer(_comm_fused_moe=object()),
        "reduce_results": _layer(reduce_results=True),
        "pcp merge": _layer(
            moe_parallel_config=SimpleNamespace(pcp_merge=True, dp_logical_ratio=1)
        ),
        "dp logical split": _layer(
            moe_parallel_config=SimpleNamespace(pcp_merge=False, dp_logical_ratio=2)
        ),
    }
    for name, layer in refused.items():
        assert not mega.attach_mega_shared_expert(layer, expert), name
        assert not hasattr(layer, "_mega_shared_expert"), name

    aiter_weights.sp = True
    layer = _layer()
    assert not mega.attach_mega_shared_expert(layer, expert)
    assert not hasattr(layer, "_mega_shared_expert")


def test_aiter_without_shared_in_keeps_the_shared_expert_apart(aiter_weights):
    aiter_weights.install(with_shared_in=False)
    layer = _layer()
    assert mega._shared_expert_input_cls() is None
    assert not mega.attach_mega_shared_expert(layer, _Expert())
    assert not hasattr(layer, "_mega_shared_expert")
