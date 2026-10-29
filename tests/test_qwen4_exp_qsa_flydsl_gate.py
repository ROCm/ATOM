# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FlyDSL QSA prefill path must hand anything it cannot compute to the native kernel."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from atom.model_ops.qwen4_exp import qsa_attention
from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpAttention


def _layer(**overrides):
    layer = SimpleNamespace(
        indexer=SimpleNamespace(compress_ratio=4, token_topk=2048),
        k_scale_cache=None,
        num_kv_heads=1,
        num_heads=12,
        head_dim=256,
        scaling=0.0625,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _call(layer, selected_width=2051, tokens=8):
    query = torch.zeros(tokens, layer.num_heads, layer.head_dim, dtype=torch.bfloat16)
    key = torch.zeros(tokens, 1, layer.head_dim, dtype=torch.bfloat16)
    selected = torch.full((tokens, selected_width), -1, dtype=torch.int32)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=None)
    return Qwen4ExpAttention._flydsl_attention(layer, query, key, key, selected, qsa)


@pytest.fixture
def flydsl(monkeypatch):
    calls = []
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: True)
    monkeypatch.setattr(
        qsa_attention,
        "_flydsl_attention_op",
        lambda: lambda q, k, v, indices, **kw: calls.append(indices.shape) or q,
    )
    return calls


def test_supported_layer_takes_flydsl(flydsl):
    assert _call(_layer()) is not None
    assert flydsl == [(8, 2051)]


@pytest.mark.parametrize("budget", [1024, 2048, 3072, 4096])
def test_short_selection_pads_within_supported_budget(flydsl, budget):
    layer = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=budget))
    assert _call(layer, 1027) is not None
    assert flydsl == [(8, 1027)]


@pytest.mark.parametrize(
    "overrides",
    [
        {"indexer": SimpleNamespace(compress_ratio=4, token_topk=1536)},
        {"indexer": SimpleNamespace(compress_ratio=8, token_topk=2048)},
        {"k_scale_cache": torch.ones(1)},
    ],
    ids=["unsupported-budget", "ratio-8", "fp8-kv"],
)
def test_unsupported_layer_falls_back(flydsl, overrides):
    assert _call(_layer(**overrides), 1539) is None
    assert flydsl == []


def test_unsupported_arch_falls_back(flydsl, monkeypatch):
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: False)
    assert _call(_layer()) is None
    assert flydsl == []


def test_flag_off_falls_back(flydsl, monkeypatch):
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
    assert _call(_layer()) is None
    assert flydsl == []
