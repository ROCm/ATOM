# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU contracts for Stage O: which layers bind the seam, and which calls take
aiter's one launch vs production's o_proj + all-reduce + norm."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.models.minimax_m3 import stage_o

HIDDEN = stage_o.HIDDEN


def _layer(**attn):
    attn = {"sparse_pa_mxfp8_out": True, "fast_oproj": True, **attn}
    return SimpleNamespace(
        is_moe_layer=attn.pop("moe", True),
        self_attn=SimpleNamespace(attend_mxfp8=lambda *a: None, **attn),
    )


@pytest.mark.parametrize(
    "layer",
    [
        _layer(moe=False),  # dense layer
        _layer(sparse_pa_mxfp8_out=False),  # bf16 attention output
        _layer(fast_oproj=False),  # o_proj not on the decode GEMM
    ],
)
def test_bind_refuses_layers_without_the_mxfp8_attention_moe_path(layer):
    assert stage_o.bind(layer, "model.layers.0") is None


@pytest.fixture
def seam(monkeypatch):
    calls = []
    o_proj = SimpleNamespace(weight="w", weight_scale="ws")
    norm = SimpleNamespace(weight=torch.ones(HIDDEN), variance_epsilon=1e-6)
    monkeypatch.setitem(
        stage_o._SEAMS, "l0.stage_o", SimpleNamespace(o_proj=o_proj, norm=norm)
    )
    # The model and AITER's ops need a GPU; Stage O reaches them by name.
    m3_model = ModuleType("atom.models.minimax_m3.model")
    layernorm = ModuleType("atom.model_ops.layernorm")
    monkeypatch.setitem(sys.modules, m3_model.__name__, m3_model)
    monkeypatch.setitem(sys.modules, layernorm.__name__, layernorm)
    m3_model._fast_oproj_resolved = lambda o: True
    monkeypatch.setattr(stage_o, "_k5_gluon_route", lambda r, w: True)
    k5 = stage_o.K5(inbox=[1, 2, 3, 4], ctr=5, slot=6, rank=0, max_rows=256)
    monkeypatch.setattr(stage_o, "_k5", lambda: k5)
    monkeypatch.setattr(stage_o, "_k5_maxit", lambda: 7)
    m3_model._o_proj_mxfp8 = lambda o, x, s: calls.append("o_proj") or x
    layernorm.fused_allreduce_gemma_rms_norm = lambda h, r, n: (
        calls.append("ar_norm") or (h, r)
    )
    fake = ModuleType("aiter.ops.flydsl.oproj_ar")
    fake.flydsl_oproj_ar = lambda *a, **k: calls.append(("fused", a, k)) or ("o", "r")
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.oproj_ar", fake)
    return calls


def _call(rows, dtype=torch.float8_e4m3fn):
    x = torch.zeros(rows, 2048, dtype=dtype)
    scale = torch.zeros(rows, 64, dtype=torch.uint8)
    residual = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    norm_w = torch.ones(HIDDEN)
    return stage_o.minimax_m3_attn_tail(x, scale, residual, norm_w, 1e-6, "l0.stage_o")


def test_a_decode_call_takes_the_one_launch_on_the_k5_context(seam):
    assert _call(8) == ("o", "r")
    (tag, args, kwargs), *rest = seam
    assert tag == "fused" and not rest
    assert args[2:4] == ("w", "ws") and args[7:] == ([1, 2, 3, 4], 5, 6, 0)
    assert kwargs == {"max_rows": 256, "maxit": 7}


@pytest.mark.parametrize("rows", [stage_o.MAX_ROWS + 1, 512])
def test_larger_calls_take_production(seam, rows):
    _call(rows)
    assert seam == ["o_proj", "ar_norm"]


def test_another_ar_route_takes_production(seam, monkeypatch):
    monkeypatch.setattr(stage_o, "_k5_gluon_route", lambda r, w: False)
    _call(8)
    assert seam == ["o_proj", "ar_norm"]
