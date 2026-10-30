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


def _selection(tokens, width):
    return torch.arange(tokens * width, dtype=torch.int32).view(tokens, width)


def _call(layer, selected=None, tokens=8):
    if selected is None:
        selected = _selection(tokens, 2051)
    query = torch.zeros(tokens, layer.num_heads, layer.head_dim, dtype=torch.bfloat16)
    key = torch.zeros(tokens, 1, layer.head_dim, dtype=torch.bfloat16)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=None)
    return Qwen4ExpAttention._flydsl_attention(layer, query, key, key, selected, qsa)


@pytest.fixture
def flydsl(monkeypatch):
    calls = []
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        qsa_attention,
        "_flydsl_attention_op",
        lambda: lambda q, k, v, indices, **kw: calls.append(indices) or q,
    )
    return calls


def test_supported_layer_takes_flydsl(flydsl):
    assert _call(_layer()) is not None
    assert [indices.shape for indices in flydsl] == [(8, 2051)]


@pytest.mark.parametrize("budget", [1024, 2048, 3072, 4096])
def test_short_selection_is_padded_with_minus_one(flydsl, budget):
    selected = _selection(8, 11)
    layer = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=budget))
    assert _call(layer, selected) is not None
    (indices,) = flydsl
    assert indices.shape == (8, 1027)
    assert torch.equal(indices[:, :11], selected)
    assert bool((indices[:, 11:] == -1).all())


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
    assert _call(_layer(**overrides), _selection(8, 1539)) is None
    assert flydsl == []


def test_selection_over_aiter_byte_limit_falls_back(flydsl):
    # 131072 rows x 4099 int32 columns span 2,149,056,512 bytes (>= 2**31).
    tokens, width = 131072, 4099
    layer = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=4096))
    query = torch.zeros(1, 12, 256, dtype=torch.bfloat16).expand(tokens, -1, -1)
    selected = torch.zeros(1, width, dtype=torch.int32).expand(tokens, -1)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=None)
    assert (
        Qwen4ExpAttention._flydsl_attention(layer, query, query, query, selected, qsa)
        is None
    )
    assert flydsl == []


def test_unsupported_arch_falls_back(flydsl, monkeypatch):
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: False)
    assert _call(_layer()) is None
    assert flydsl == []


def test_flag_off_falls_back(flydsl, monkeypatch):
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
    assert _call(_layer()) is None
    assert flydsl == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_mixed_prefix_batch_gathers_full_contexts(monkeypatch):
    """Prefixed and fresh requests with unequal lengths on non-consecutive
    pages, crossing page boundaries: the bridge's context rows must be every
    request's full context in request-major order, and the kernel must receive
    exactly those K/V rows plus the per-request query / prefix lengths."""
    from atom.plugin.sglang.qwen4_exp_bridge import _flydsl_prefill_layout

    device, block, pages = "cuda", 4, 12
    prefixes, extends = (5, 0, 2), (3, 6, 1)
    seqs = tuple(p + e for p, e in zip(prefixes, extends))
    tables = torch.tensor(
        [[7, 2, 11], [5, 9, 11], [0, 11, 11]], dtype=torch.int32, device=device
    )
    batch = SimpleNamespace(extend_seq_lens_cpu=list(extends), seq_lens_cpu=list(seqs))
    host_lens, context_slots = _flydsl_prefill_layout(
        batch, len(seqs), sum(extends), tables, block
    )
    assert host_lens == (extends, seqs)
    expected_slots = [
        int(tables[r, t // block]) * block + t % block
        for r, seq in enumerate(seqs)
        for t in range(seq)
    ]
    assert context_slots.tolist() == expected_slots

    # Each cache row holds its own flat index, so gathered rows name themselves.
    rows = torch.arange(pages * block, device=device, dtype=torch.bfloat16)
    k_cache = rows.view(pages, block, 1, 1).expand(-1, -1, 1, 256).contiguous()
    calls = []
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")
    monkeypatch.setattr(qsa_attention, "_flydsl_arch_supported", lambda: True)
    monkeypatch.setattr(
        qsa_attention,
        "_flydsl_attention_op",
        lambda: lambda q, k, v, indices, **kw: calls.append((k, v, kw)) or q,
    )
    layer = _layer(k_cache=k_cache, v_cache=-k_cache)
    tokens = sum(extends)
    query = torch.zeros(tokens, 12, 256, dtype=torch.bfloat16, device=device)
    qsa = SimpleNamespace(host_lens=host_lens, context_slots=context_slots)
    selected = _selection(tokens, 2051).to(device)
    assert (
        Qwen4ExpAttention._flydsl_attention(layer, query, query, query, selected, qsa)
        is not None
    )
    ((k, v, kwargs),) = calls
    assert k[:, 0, 0].tolist() == expected_slots
    assert torch.equal(v, -k)
    assert kwargs["query_lens"] == extends
    assert kwargs["prefix_lens"] == prefixes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "prefixes, extends",
    [
        ((0,), (7,)),
        ((0,), (300,)),
        ((0,), (512,)),
        ((200,), (300,)),
        ((0, 0, 100), (100, 37, 250)),
        ((0,), (1000,)),
    ],
    ids=["7", "300", "512", "prefix-200+300", "mixed-3-requests", "1000-unpadded"],
)
def test_short_context_matches_native_kernel(monkeypatch, prefixes, extends):
    """Short contexts shrink the selection (its width follows max_seq_len, not
    the budget), so it is padded with -1 to the next FlyDSL width, past the
    budget's own layout. Every row still lists all its complete blocks, then
    its tail, left-aligned, so the real kernel must match the native
    `qsa_sparse_paged_gqa` on the same selection."""
    from atom.model_ops.qwen4_exp.ops.qsa import (
        qsa_select_paged_tokens,
        qsa_sparse_paged_gqa,
    )
    from atom.plugin.sglang.qwen4_exp_bridge import _flydsl_prefill_layout

    if not qsa_attention._flydsl_arch_supported():
        pytest.skip("FlyDSL QSA attention needs gfx942")
    try:
        qsa_attention._flydsl_attention_op()
    except ImportError:
        pytest.skip("aiter without FlyDSL QSA attention (ROCm/aiter#6576)")
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")

    device, block, ratio, budget, heads, dim = "cuda", 64, 4, 2048, 6, 256
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device, generator=gen)

    seqs = [p + e for p, e in zip(prefixes, extends)]
    width = max(-(-s // block) for s in seqs)
    pages = len(seqs) * width
    tables = (
        torch.randperm(pages, device=device, generator=gen)
        .view(len(seqs), width)
        .to(torch.int32)
    )
    positions = torch.cat(
        [torch.arange(p, p + e, device=device) for p, e in zip(prefixes, extends)]
    )
    token_to_req = torch.cat(
        [
            torch.full((e,), r, dtype=torch.int32, device=device)
            for r, e in enumerate(extends)
        ]
    )
    selected = qsa_select_paged_tokens(
        randn(positions.numel(), 4, 128),
        randn(pages, block // ratio, 1, 128),
        tables,
        token_to_req,
        positions,
        torch.tensor(seqs, dtype=torch.int32, device=device),
        budget,
        ratio,
        max_seq_len=max(seqs),
        allow_flydsl=True,
    )
    if max(seqs) <= 512:
        assert selected.shape[1] < 1027  # the padded case under review

    k_cache, v_cache = randn(pages, block, 1, dim), randn(pages, block, 1, dim)
    query = randn(positions.numel(), heads, dim)
    slots = (
        tables[token_to_req.long(), positions // block].long() * block
        + positions % block
    )
    batch = SimpleNamespace(extend_seq_lens_cpu=list(extends), seq_lens_cpu=seqs)
    host_lens, context_slots = _flydsl_prefill_layout(
        batch, len(seqs), positions.numel(), tables, block
    )
    layer = _layer(
        indexer=SimpleNamespace(compress_ratio=ratio, token_topk=budget),
        num_heads=heads,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    qsa = SimpleNamespace(host_lens=host_lens, context_slots=context_slots)
    flydsl = Qwen4ExpAttention._flydsl_attention(
        layer,
        query,
        k_cache.flatten(0, 1)[slots],
        v_cache.flatten(0, 1)[slots],
        selected,
        qsa,
    )
    assert flydsl is not None
    native = qsa_sparse_paged_gqa(
        query,
        k_cache,
        v_cache,
        selected,
        tables,
        token_to_req,
        softmax_scale=layer.scaling,
        num_decode_requests=None,
    )
    torch.testing.assert_close(flydsl, native, atol=1.6e-2, rtol=1.6e-2)
