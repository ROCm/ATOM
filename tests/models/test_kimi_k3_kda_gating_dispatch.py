# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which KDA gating kernel the Kimi-K3 decode and spec calls reach.

aiter's gfx1250 entry replaces ATOM's kernel only where it is verified: the
lower-bound KDA gate, bf16 activations over an fp32 state, on gfx1250. Every
other call must still reach ATOM's own kernel, so these tests pin the routing
with the two kernels replaced by recorders.
"""

import pytest
import torch

model = pytest.importorskip("atom.models.kimi_k3.model")


def _kwargs(**overrides):
    base = {
        "A_log": None,
        "a": torch.empty(1, 2, 12, 128, dtype=torch.bfloat16),
        "b": torch.empty(1, 2, 12, dtype=torch.bfloat16),
        "dt_bias": None,
        "q": torch.empty(1, 2, 12, 128, dtype=torch.bfloat16),
        "k": torch.empty(1, 2, 12, 128, dtype=torch.bfloat16),
        "v": torch.empty(1, 2, 12, 128, dtype=torch.bfloat16),
        "o": None,
        "initial_state": torch.empty(4, 12, 128, 128, dtype=torch.float32),
        "is_kda": True,
        "lower_bound": -5.0,
        "use_qk_l2norm_in_kernel": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
def recorders(monkeypatch):
    calls = {"aiter": 0, "atom": 0}

    def aiter_kernel(**kwargs):
        calls["aiter"] += 1

    def atom_kernel(**kwargs):
        calls["atom"] += 1

    monkeypatch.setattr(model, "_aiter_kda_gating_update", aiter_kernel)
    monkeypatch.setattr(model, "_atom_kda_gating_update", atom_kernel)
    monkeypatch.setattr(model, "_current_gfx", lambda: "gfx1250")
    return calls


def test_gfx1250_kda_lower_bound_routes_to_aiter(recorders):
    model.fused_sigmoid_gating_delta_rule_update(**_kwargs())
    assert recorders == {"aiter": 1, "atom": 0}


def test_other_arch_stays_on_atom_kernel(recorders, monkeypatch):
    monkeypatch.setattr(model, "_current_gfx", lambda: "gfx950")
    model.fused_sigmoid_gating_delta_rule_update(**_kwargs())
    assert recorders == {"aiter": 0, "atom": 1}


@pytest.mark.parametrize(
    "overrides",
    [
        {"lower_bound": None},
        {"is_kda": False},
        {"a": torch.empty(1, 2, 12, 128, dtype=torch.float32)},
        {"q": torch.empty(1, 2, 12, 128, dtype=torch.float16)},
        {"initial_state": torch.empty(4, 12, 128, 128, dtype=torch.bfloat16)},
        {"initial_state": None},
    ],
)
def test_unverified_inputs_stay_on_atom_kernel(recorders, overrides):
    model.fused_sigmoid_gating_delta_rule_update(**_kwargs(**overrides))
    assert recorders == {"aiter": 0, "atom": 1}


def test_missing_aiter_entry_stays_on_atom_kernel(recorders, monkeypatch):
    monkeypatch.setattr(model, "_aiter_kda_gating_update", None)
    model.fused_sigmoid_gating_delta_rule_update(**_kwargs())
    assert recorders == {"aiter": 0, "atom": 1}
