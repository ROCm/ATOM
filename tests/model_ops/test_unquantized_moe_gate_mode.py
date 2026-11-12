# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The unquantized MoE's recorded gate/up layout reaches both expert paths.

The modular (all2all) kernel only sees `moe_extra_args`; it cannot tell
interleaved weights from separated ones, so a layout recorded at weight
processing must be forwarded to it as well as to `fused_moe`.
"""

from types import SimpleNamespace

import pytest
import torch


class _Recorder:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["hidden_states"]


def _apply(moe, layer, fused_experts=None):
    method = moe.UnquantizedFusedMoEMethod.__new__(moe.UnquantizedFusedMoEMethod)
    method.fused_experts = fused_experts
    x = torch.zeros(2, 8)
    ids = torch.zeros(2, 1, dtype=torch.int32)
    return method.apply(layer, x, torch.ones(2, 1), ids)


def _layer(**attrs):
    return SimpleNamespace(
        w13_weight=torch.zeros(1), w2_weight=torch.zeros(1), expert_mask=None, **attrs
    )


@pytest.mark.parametrize("mode", ["interleave", "separated"])
def test_modular_path_receives_the_recorded_gate_mode(atom_modules, mode):
    recorder = _Recorder()
    _apply(atom_modules.moe, _layer(moe_gate_mode=mode), fused_experts=recorder)
    assert recorder.kwargs["moe_extra_args"] == {"gate_mode": mode}


@pytest.mark.parametrize("mode", ["interleave", "separated"])
def test_fused_moe_path_receives_the_recorded_gate_mode(
    monkeypatch, atom_modules, mode
):
    recorder = _Recorder()
    monkeypatch.setattr(atom_modules.moe, "fused_moe", recorder)
    _apply(atom_modules.moe, _layer(moe_gate_mode=mode))
    assert recorder.kwargs["gate_mode"] == mode


def test_unprocessed_weights_default_to_separated(monkeypatch, atom_modules):
    recorder = _Recorder()
    monkeypatch.setattr(atom_modules.moe, "fused_moe", recorder)
    _apply(atom_modules.moe, _layer())
    assert recorder.kwargs["gate_mode"] == atom_modules.moe.GateMode.SEPARATED.value
