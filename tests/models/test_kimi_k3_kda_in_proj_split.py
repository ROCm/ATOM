# SPDX-License-Identifier: MIT
"""Kimi-K3 KDA in_proj: a long prefill runs the fused [q | k | v | g | b | f_a]
weight as two GEMMs over its row slices, and what the layer returns does not
change."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from aiter import QuantType

from atom.model_ops import kimi_k3 as kimi_ops
from atom.model_ops import linear
from atom.models.kimi_k3 import model as kimi

HIDDEN, HEADS, HEAD_DIM, TOKENS = 16, 2, 4, 6
LP = HEADS * HEAD_DIM


def _attention(monkeypatch):
    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    torch.manual_seed(0)
    attn = kimi.KimiKDAAttention.__new__(kimi.KimiKDAAttention)
    nn.Module.__init__(attn)
    attn.local_proj_size, attn.num_local_heads, attn.head_dim = LP, HEADS, HEAD_DIM
    attn.layer_num, attn.fuse_f_a_quant = 0, False
    attn.activation, attn.conv_weight = "silu", None
    attn.in_proj = linear.MergedColumnParallelLinear(
        HIDDEN, [LP] * 4, bias=False, quant_config=None
    )
    fused = torch.randn(4 * LP + HEADS + HEAD_DIM, HIDDEN)
    attn.in_proj.weight = nn.Parameter(fused, requires_grad=False)
    w_fb = torch.randn(LP, HEAD_DIM)
    attn.f_b_proj = lambda x: F.linear(x, w_fb)
    # Identity conv and a KDA stand-in that reads q, k, v, the gate and beta.
    monkeypatch.setattr(
        kimi, "causal_conv1d_fn", lambda x, *a, **k: x.transpose(0, 1).split(LP, -1)
    )
    monkeypatch.setattr(
        kimi_ops, "gather_kda_initial_state", lambda s, idx, _: s[idx].clone()
    )
    attn._run_kda = lambda q, k, v, g, b, s, *_: (
        (q + k + v + g * b.unsqueeze(-1)).contiguous(),
        s,
    )
    attn.o_norm = lambda out, gate, is_prefill: out * gate
    attn.o_proj = lambda x: x
    meta = SimpleNamespace(
        num_actual_tokens=TOKENS,
        num_prefills=1,
        has_initial_state=torch.tensor([False]),
        non_spec_state_indices_tensor=torch.tensor([0]),
        non_spec_state_indices_in_tensor=None,
        non_spec_query_start_loc=torch.tensor([0, TOKENS]),
    )
    cache = SimpleNamespace(
        k_cache=torch.zeros(1, 3 * LP, 3),
        v_cache=torch.zeros(1, HEADS, HEAD_DIM, HEAD_DIM),
    )
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(kda_metadata=meta),
        kv_cache_data={"layer_0": cache},
    )
    monkeypatch.setattr(kimi, "get_forward_context", lambda: ctx)
    gemms = []

    def mm(self, inp, weights, bias=None, otype=None, **_):
        gemms.append(weights)
        return F.linear(inp, weights, bias)

    monkeypatch.setattr(type(kimi.tgemm), "mm", mm)
    return attn, gemms


def _forward(attn, monkeypatch, split_from):
    monkeypatch.setattr(kimi, "_KDA_IN_PROJ_SPLIT_MIN_TOKENS", split_from)
    torch.manual_seed(1)
    return attn._forward_impl(torch.randn(TOKENS, HIDDEN))


def test_long_prefill_splits_in_proj_and_matches_the_fused_gemm(monkeypatch):
    attn, gemms = _attention(monkeypatch)
    weight = attn.in_proj.weight

    fused_out = _forward(attn, monkeypatch, split_from=TOKENS + 1)
    assert [w.shape for w in gemms] == [weight.shape]

    gemms.clear()
    split_out = _forward(attn, monkeypatch, split_from=TOKENS)
    assert [w.shape for w in gemms] == [(4 * LP, HIDDEN), (HEADS + HEAD_DIM, HIDDEN)]
    assert all(w.untyped_storage().data_ptr() == weight.data_ptr() for w in gemms)
    torch.testing.assert_close(split_out, fused_out)


def test_quantized_in_proj_keeps_the_fused_gemm(monkeypatch):
    attn, gemms = _attention(monkeypatch)
    attn.in_proj.quant_type = QuantType.per_Token
    calls = []

    def quantized_forward(x, x_scale=None):
        calls.append(x.shape[0])
        return F.linear(x, attn.in_proj.weight)

    attn.in_proj.forward = quantized_forward
    _forward(attn, monkeypatch, split_from=TOKENS)
    assert calls == [TOKENS] and gemms == []
