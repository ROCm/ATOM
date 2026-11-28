# SPDX-License-Identifier: MIT
"""Routing of the aiter Gemma q/k norm + RoPE of a long prefill, on CPU: a layer
asks aiter's fused_qknorm_idxrqknorm_supports once, at its first long prefill,
and keeps the answer; it never asks off the gfx1250 family, for a non-NeoX
rotation or for q/k norms with different eps. The numerics are in
test_gemma_qknorm_rope_aiter_gpu.py."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn


@pytest.fixture
def mha(monkeypatch):
    with atom_without_aiter() as load:
        mha = load("atom.model_ops.attention_mha")
    # the layers' rotary is a stub: the cos/sin table is not read here
    monkeypatch.setattr(mha, "get_cos_sin_cache", lambda rope, qkv: None)
    monkeypatch.setattr(mha, "get_gfx", lambda: "gfx1250")
    return mha


def make_layer(mha, *, neox=True, k_eps=1e-6):
    layer = mha.PagedAttentionImpl.__new__(mha.PagedAttentionImpl)
    nn.Module.__init__(layer)
    layer.num_heads, layer.num_kv_heads, layer.head_dim = 2, 1, 128
    layer.q_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
    layer.k_norm = SimpleNamespace(weight=None, variance_epsilon=k_eps)
    layer.rotary_emb = SimpleNamespace(is_neox_style=neox, rotary_dim=64)
    return layer


@pytest.mark.parametrize("taken", [True, False])
def test_asks_aiter_once_per_layer(mha, monkeypatch, taken):
    asked, ran = [], []
    monkeypatch.setattr(mha, "is_gfx1250_family", lambda gfx: True)
    monkeypatch.setattr(
        mha.aiter,
        "fused_qknorm_idxrqknorm_supports",
        lambda *a, **kw: asked.append(1) or taken,
        raising=False,
    )
    monkeypatch.setattr(
        mha.aiter, "fused_qknorm_idxrqknorm", lambda *a, **kw: ran.append(1)
    )
    layer, other = make_layer(mha), make_layer(mha)
    qkv, position = torch.zeros(4, 4 * 128), torch.zeros(4, dtype=torch.long)
    for _ in range(3):
        q = layer._aiter_gemma_norm_rope(qkv, position)
        assert (q is not None) == taken
    assert (len(asked), len(ran)) == (1, 3 if taken else 0)
    other._aiter_gemma_norm_rope(qkv, position)  # its own answer
    assert len(asked) == 2


@pytest.mark.parametrize(
    ("gfx1250", "neox", "k_eps"),
    [(False, True, 1e-6), (True, False, 1e-6), (True, True, 1e-5)],
)
def test_declines_without_asking(mha, monkeypatch, gfx1250, neox, k_eps):
    monkeypatch.setattr(mha, "is_gfx1250_family", lambda gfx: gfx1250)
    monkeypatch.setattr(
        mha.aiter,
        "fused_qknorm_idxrqknorm_supports",
        lambda *a, **kw: pytest.fail("asked aiter"),
        raising=False,
    )
    layer = make_layer(mha, neox=neox, k_eps=k_eps)
    qkv, position = torch.zeros(4, 4 * 128), torch.zeros(4, dtype=torch.long)
    for _ in range(2):
        assert layer._aiter_gemma_norm_rope(qkv, position) is None
