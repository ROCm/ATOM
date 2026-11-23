# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side wiring of the split-K reduce left to the consumer
(`LinearBase.splitk_consumer_max_m`), against stubbed aiter/Triton so it runs on
the CPU runner: which layers defer, which attention impls claim the partials,
the reduce-vs-pass choice in rope_cache, and no deferral while a token graph
records. The kernels themselves are checked in test_splitk_consumer_reduce.py."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs
from torch import nn


@pytest.fixture(scope="module")
def m():
    with atom_under_stubs() as load:
        yield SimpleNamespace(
            mha=load("atom.model_ops.attention_mha"),
            linear=load("atom.model_ops.linear"),
            layernorm=load("atom.model_ops.layernorm"),
            paged=load("atom.model_ops.paged_attention"),
            region_graph=load("atom.utils.region_graph"),
            fused=load("atom.model_ops.triton_fused_qkv_norm_rope_cache"),
            m3=load("atom.models.minimax_m3.splitk"),
        )


def _module(cls, **attrs):
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def test_claims_splitk_partials_is_opt_in(m):
    class OwnRopeCache(m.mha.PagedAttentionImpl):
        pass

    assert m.mha.PagedAttentionImpl.claims_splitk_partials
    assert m.mha.SparseMHAPagedAttentionImpl.claims_splitk_partials
    assert not OwnRopeCache.claims_splitk_partials
    # Claiming is configured per layer; no impl claims by default.
    assert not m.mha.PagedAttentionImpl.qkv_splitk_deferred


def test_only_a_claiming_atom_attention_takes_the_deferral(m, monkeypatch):
    class OwnRopeCache(m.mha.PagedAttentionImpl):
        pass

    def attn(impl_cls=m.mha.PagedAttentionImpl, cls=m.paged.Attention, sp=None):
        return _module(cls, sp_columns=sp, impl=_module(impl_cls))

    def deferred(a, arch="gfx1250", max_m="64"):
        monkeypatch.setattr(m.m3, "get_gfx_runtime", lambda: arch)
        monkeypatch.setenv("ATOM_SPLITK_CONSUMER_REDUCE_MAX_M", max_m)
        qkv_proj = SimpleNamespace(splitk_consumer_max_m=0)
        m.m3.defer_qkv_splitk_reduce(qkv_proj, a)
        flag = a.impl.qkv_splitk_deferred
        assert (qkv_proj.splitk_consumer_max_m > 0) == flag
        return flag

    assert deferred(attn())
    assert deferred(attn(m.mha.SparseMHAPagedAttentionImpl))
    assert not deferred(attn(), arch="gfx950")  # M3 off gfx1250: main's path
    assert not deferred(attn(), max_m="0")
    assert not deferred(attn(OwnRopeCache))
    assert not deferred(attn(sp=(0, 1)))  # Ulysses reads qkv before the impl
    assert not deferred(attn(cls=nn.Module))  # a plugin's attention layer


@pytest.fixture
def claims(m, monkeypatch):
    log = SimpleNamespace(claimed=[], reduced=[], partials=torch.ones(2, 1, 4))
    monkeypatch.setattr(
        m.mha,
        "claim_splitk_partials",
        lambda y: log.claimed.append(y) or log.partials,
    )
    monkeypatch.setattr(
        m.mha, "gemm_a8w8_bpreshuffle_splitk_reduce", lambda y, p: log.reduced.append(p)
    )
    return log


def _impl(m, q_norm, deferred=True):
    return _module(
        m.mha.PagedAttentionImpl,
        qkv_splitk_deferred=deferred,
        rotary_emb=SimpleNamespace(cos_sin_cache=None, is_neox_style=True),
        q_norm=q_norm,
        k_norm=q_norm,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        layer_num=0,
        use_triton_attn=False,
        kv_cache_dtype="bf16",
    )


def _rope_cache(impl, qkv):
    layer = SimpleNamespace(
        k_cache=torch.zeros(1), v_cache=torch.zeros(1), k_scale=None, v_scale=None
    )
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(slot_mapping=None),
        kv_cache_data={"layer_0": layer},
    )
    q, k, v = torch.split(qkv, [8, 4, 4], dim=-1)
    return impl.rope_cache(q, k, v, qkv, None, ctx)


def test_rope_cache_passes_partials_only_to_the_fused_gemma_kernel(
    m, claims, monkeypatch
):
    passed = []

    def fused(q, k, v, position, **kw):
        passed.append(kw["qkv_partials"])
        return q, k

    monkeypatch.setattr(m.fused, "triton_fused_norm_rope_cache", fused)
    # The suite may have imported attention_mha against a real aiter already.
    monkeypatch.setattr(
        m.mha, "fused_qk_norm_rope_cache_quant_shuffle", lambda **kw: None
    )
    gemma = _module(m.layernorm.GemmaRMSNorm, weight=None, variance_epsilon=1e-6)
    qkv = torch.zeros(3, 16, dtype=torch.bfloat16)

    _rope_cache(_impl(m, gemma), qkv)
    assert passed[0] is claims.partials and not claims.reduced

    # aiter's fused RMSNorm kernel reads a reduced qkv.
    rms = SimpleNamespace(eps=1e-6, weight=None)
    _rope_cache(_impl(m, rms), qkv)
    assert len(claims.reduced) == 1 and claims.reduced[0] is claims.partials

    # A layer whose QKV does not defer never looks for partials.
    claims.claimed.clear()
    _rope_cache(_impl(m, gemma, deferred=False), qkv)
    _rope_cache(_impl(m, rms, deferred=False), qkv)
    assert not claims.claimed and passed[-1] is None


def test_dummy_run_claims_only_a_deferring_layer(m, claims, monkeypatch):
    ctx = SimpleNamespace(context=SimpleNamespace(is_dummy_run=True))
    monkeypatch.setattr(m.mha, "get_forward_context", lambda: ctx)
    q = torch.zeros(3, 8)
    for deferred in (False, True):
        _impl(m, None, deferred).forward_impl(q, q, q, qkv=q)
    assert len(claims.claimed) == 1 and claims.claimed[0] is q


def test_no_deferral_while_a_token_graph_records(m, monkeypatch):
    seen = []
    monkeypatch.setattr(
        m.linear,
        "gemm_a8w8_bpreshuffle_splitk_deferred",
        lambda *a, dtype, max_m: seen.append(max_m),
    )
    x = torch.zeros(1)
    for recording in (False, True):
        monkeypatch.setattr(
            m.region_graph._local,
            "recorder",
            object() if recording else None,
            raising=False,
        )
        assert m.region_graph.recording_segments() is recording
        m.linear.gemm_a8w8_bpreshuffle_splitk_deferred_impl(x, x, x, x, max_m=64)
    assert seen == [64, 0]
