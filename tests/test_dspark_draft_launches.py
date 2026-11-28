# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The DSpark draft layer's two launches (`dspark_draft_mono`) serve only their
shape: gfx1250, TP4, 1..8 rows, the draft's MXFP8 linears and fp8 page-16 KV
cache, and the k5 all-reduce context. Anything else keeps the production layer."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from atom.model_ops import dspark_draft_mono as mono


def _lin(n, k):
    w = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    w.is_shuffled = True
    return SimpleNamespace(
        native_a8_group_rows=1,
        weight=w,
        weight_scale=torch.zeros(n, k // 32),
        bias=None,
    )


def _layer():
    attn = SimpleNamespace(
        num_heads=16, num_kv_heads=1, head_dim=128,
        rotary_emb=SimpleNamespace(rotary_dim=64),
        qkv_proj=_lin(mono.QSIZE + 256, mono.HIDDEN), o_proj=_lin(mono.HIDDEN, mono.QSIZE),
    )  # fmt: skip
    mlp = SimpleNamespace(
        gate_up_proj=_lin(2 * mono.INTER, mono.HIDDEN),
        down_proj=_lin(mono.HIDDEN, mono.INTER),
        swiglu_limit=7.0,
    )
    return SimpleNamespace(self_attn=attn, mlp=mlp)


def _cache():
    k = torch.zeros(4, 1, 8, 16, 16, dtype=torch.float8_e4m3fn)
    v = torch.zeros(4, 1, 1, 128, 16, dtype=torch.float8_e4m3fn)
    s = torch.zeros(4, 1, 16)
    return k, v, s, s


@pytest.fixture
def served(monkeypatch):
    """gfx1250, TP4, a k5 context: the conditions a CPU job has none of."""
    state = {"gfx": "gfx1250", "tp": 4, "k5": ([1, 2, 3, 4], 5, 6, 0, 256)}
    ps = ModuleType("aiter.dist.parallel_state")
    ps.get_tensor_model_parallel_world_size = lambda: state["tp"]
    chip = ModuleType("aiter.jit.utils.chip_info")
    chip.get_gfx_runtime = lambda: state["gfx"]
    monkeypatch.setitem(sys.modules, ps.__name__, ps)
    monkeypatch.setitem(sys.modules, chip.__name__, chip)
    monkeypatch.setattr(mono, "_k5_info", lambda: state["k5"])
    return state


def test_the_served_shape(served):
    assert mono.usable(_layer(), 7, _cache())


@pytest.mark.parametrize(
    "change",
    [
        lambda st, lay, c: (st, lay, None, 7),  # no bound cache (warmup)
        lambda st, lay, c: (st, lay, c, 9),  # more rows than the launch holds
        lambda st, lay, c: ({**st, "gfx": "gfx950"}, lay, c, 7),
        lambda st, lay, c: ({**st, "tp": 8}, lay, c, 7),
        lambda st, lay, c: ({**st, "k5": None}, lay, c, 7),  # no k5 context
        lambda st, lay, c: (st, lay, (c[0].float(), *c[1:]), 7),  # bf16 KV
    ],
)
def test_anything_else_keeps_the_production_layer(served, change):
    st, lay, cache, rows = change(dict(served), _layer(), _cache())
    served.update(st)
    assert not mono.usable(lay, rows, cache)
