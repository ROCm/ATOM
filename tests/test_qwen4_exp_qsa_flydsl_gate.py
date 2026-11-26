# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The QSA FlyDSL prefill on the real kernels (GPU). The host-side decisions
are covered on CPU in `tests/model_ops/test_qsa_flydsl_prefill_host.py`."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpQSAMetadata
from atom.model_ops.qwen4_exp import qsa_attention
from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpAttention


def _layer(compressed_key_cache, **overrides):
    layer = SimpleNamespace(
        indexer=SimpleNamespace(compress_ratio=4, token_topk=2048),
        k_scale_cache=None,
        num_kv_heads=1,
        num_heads=12,
        head_dim=256,
        scaling=0.0625,
        compressed_key_cache=compressed_key_cache,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _qsa(tables, seqs, host_lens, context_slots):
    tokens = sum(host_lens[0])
    device = tables.device
    return Qwen4ExpQSAMetadata(
        block_tables=tables,
        slot_mapping=torch.zeros(tokens, dtype=torch.int64, device=device),
        compressed_slot_mapping=torch.full((tokens,), -1, device=device),
        token_to_req=torch.zeros(tokens, dtype=torch.int32, device=device),
        logical_positions=torch.zeros(tokens, dtype=torch.int64, device=device),
        seq_lens=torch.tensor(seqs, dtype=torch.int32, device=device),
        max_seq_len=max(seqs),
        host_lens=host_lens,
        context_slots=context_slots,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_mixed_prefix_batch_reads_full_contexts_through_the_slot_table(monkeypatch):
    """Prefixed and fresh requests with unequal lengths on non-consecutive
    pages, crossing page boundaries: the bridge's context rows must be every
    request's full context in request-major order, and the kernel must receive
    the pool with exactly those rows as its slot table, plus the per-request
    query / prefix lengths."""
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
    # Any GPU: the kernel is replaced, only the layout is checked.
    monkeypatch.setattr(
        qsa_attention, "qsa_sparse_prefill_widths", lambda: (1027, 2051, 3075, 4099)
    )
    monkeypatch.setattr(
        qsa_attention, "qsa_sparse_prefill_supports_selection", lambda r, w: True
    )
    monkeypatch.setattr(
        qsa_attention,
        "qsa_sparse_prefill",
        lambda q, k, v, indices, **kw: calls.append((k, v, kw)) or q,
    )
    compressed = torch.zeros(pages, block // 4, 1, 128, device=device)
    layer = _layer(compressed, k_cache=k_cache, v_cache=-k_cache)
    tokens = sum(extends)
    query = torch.zeros(tokens, 12, 256, dtype=torch.bfloat16, device=device)
    qsa = _qsa(tables, seqs, host_lens, context_slots)
    plan = qsa_attention._FlyDSLPrefill.of(layer, query, qsa)
    assert plan.ready
    plan.selection_out(device)
    assert (
        Qwen4ExpAttention._flydsl_attention(layer, query, query, query, plan, qsa)
        is not None
    )
    ((k, v, kwargs),) = calls
    assert context_slots.dtype == torch.int32
    assert kwargs["kv_slots"] is context_slots
    assert (
        k.data_ptr() == k_cache.data_ptr() and v.data_ptr() == layer.v_cache.data_ptr()
    )
    assert k[context_slots.long(), 0, 0].tolist() == expected_slots
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

    if not qsa_attention.qsa_sparse_prefill_widths():
        pytest.skip("aiter's FlyDSL QSA prefill does not run on this GPU")
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
    index_q = randn(positions.numel(), 4, 128)
    compressed = randn(pages, block // ratio, 1, 128)

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
        compressed,
        indexer=SimpleNamespace(compress_ratio=ratio, token_topk=budget),
        num_heads=heads,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    qsa = _qsa(tables, seqs, host_lens, context_slots)
    plan = qsa_attention._FlyDSLPrefill.of(layer, query, qsa)
    assert plan.ready
    selected = qsa_select_paged_tokens(
        index_q,
        compressed,
        tables,
        token_to_req,
        positions,
        torch.tensor(seqs, dtype=torch.int32, device=device),
        budget,
        ratio,
        out=plan.selection_out(device),
        max_seq_len=max(seqs),
        allow_flydsl=True,
    )
    if max(seqs) <= 512:
        assert selected.shape[1] < 1027  # narrower than any kernel width
    flydsl = Qwen4ExpAttention._flydsl_attention(
        layer,
        query,
        k_cache.flatten(0, 1)[slots],
        v_cache.flatten(0, 1)[slots],
        plan,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "prefixes, extends",
    [
        ((6080,), (2112,)),
        ((8192,), (2048,)),
        ((6001, 0, 1234), (2111, 300, 517)),
        ((200,), (300,)),
        ((0,), (2112,)),
    ],
    ids=["8192-hit-6080", "second-chunk", "mixed-unaligned", "short", "no-prefix"],
)
def test_in_place_reads_match_gathered_rows_bit_for_bit(monkeypatch, prefixes, extends):
    """The real kernel on the slot table (pool read in place, selection written
    into the kernel-width buffer, two layers sharing one layout) gives the same
    bits as the kernel on every context row gathered and a padded copy of the
    selection. Stale pool rows hold random finite values."""
    from atom.model_ops.qwen4_exp.ops.qsa import qsa_select_paged_tokens
    from atom.plugin.sglang.qwen4_exp_bridge import _flydsl_prefill_layout

    if not qsa_attention.qsa_sparse_prefill_widths():
        pytest.skip("aiter's FlyDSL QSA prefill does not run on this GPU")
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")

    device, block, ratio, budget, heads, dim, pages = "cuda", 64, 4, 2048, 12, 256, 512
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=device, generator=gen)

    seqs = [p + e for p, e in zip(prefixes, extends)]
    width = max(-(-s // block) for s in seqs)
    tables = (
        torch.randperm(pages, device=device, generator=gen)[: len(seqs) * width]
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
    compressed = randn(pages, block // ratio, 1, 128)
    index_q = randn(positions.numel(), 4, 128)
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device=device)

    def select(out=None):
        return qsa_select_paged_tokens(
            index_q,
            compressed,
            tables,
            token_to_req,
            positions,
            seq_lens,
            budget,
            ratio,
            out=out,
            max_seq_len=max(seqs),
            allow_flydsl=True,
        )

    query = randn(positions.numel(), heads, dim)
    slots = (
        tables[token_to_req.long(), positions // block].long() * block
        + positions % block
    )
    batch = SimpleNamespace(extend_seq_lens_cpu=list(extends), seq_lens_cpu=seqs)
    host_lens, context_slots = _flydsl_prefill_layout(
        batch, len(seqs), positions.numel(), tables, block
    )
    qsa = _qsa(tables, seqs, host_lens, context_slots)
    query_lens = host_lens[0]
    prefix_lens = tuple(s - q for q, s in zip(*host_lens))
    for _ in range(2):
        k_cache, v_cache = randn(pages, block, 1, dim), randn(pages, block, 1, dim)
        layer = _layer(
            compressed,
            indexer=SimpleNamespace(compress_ratio=ratio, token_topk=budget),
            num_heads=heads,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        if context_slots is None:
            key, value = k_cache.flatten(0, 1)[slots], v_cache.flatten(0, 1)[slots]
        else:
            key = k_cache.flatten(0, 1)[context_slots.long()]
            value = v_cache.flatten(0, 1)[context_slots.long()]
        plan = qsa_attention._FlyDSLPrefill.of(layer, query, qsa)
        assert plan.ready
        fresh = select()
        reference = qsa_attention.qsa_sparse_prefill(
            query,
            key,
            value,
            torch.nn.functional.pad(fresh, (0, plan.width - fresh.shape[1]), value=-1),
            query_lens=query_lens,
            prefix_lens=prefix_lens,
            softmax_scale=layer.scaling,
        )
        selected = select(plan.selection_out(device))
        assert selected.data_ptr() == plan.indices.data_ptr()
        in_place = Qwen4ExpAttention._flydsl_attention(
            layer, query, key, value, plan, qsa
        )
        assert in_place is not None
        assert torch.equal(in_place, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize(
    "prefixes, extends",
    [((6080,), (2112,)), ((6001, 0, 1234), (2111, 300, 517)), ((200,), (300,))],
    ids=["8192-hit-6080", "mixed-unaligned", "short"],
)
def test_sharded_selection_written_in_place_matches_a_fresh_one(prefixes, extends):
    """The sequence-parallel path expands the all-gathered groups itself; into
    the kernel-width buffer it writes the same columns as into a fresh
    tensor and leaves -1 past them."""
    from atom.model_ops.qwen4_exp.ops.qsa import (
        qsa_expand_block_indices,
        qsa_select_paged_groups,
    )

    device, block, ratio, budget, pages = "cuda", 64, 4, 2048, 512
    gen = torch.Generator(device=device).manual_seed(0)
    seqs = [p + e for p, e in zip(prefixes, extends)]
    width = max(-(-s // block) for s in seqs)
    tables = (
        torch.randperm(pages, device=device, generator=gen)[: len(seqs) * width]
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
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device=device)
    groups, columns = qsa_select_paged_groups(
        torch.randn(positions.numel(), 4, 128, device=device, generator=gen).to(
            torch.bfloat16
        ),
        torch.randn(pages, block // ratio, 1, 128, device=device, generator=gen).to(
            torch.bfloat16
        ),
        tables,
        token_to_req,
        positions,
        seq_lens,
        budget,
        ratio,
        max_seq_len=max(seqs),
        allow_flydsl=True,
    )
    args = (groups, positions, seq_lens, token_to_req, ratio, budget)
    fresh = qsa_expand_block_indices(*args, output_width=columns)
    kernel_width = next(
        w for w in qsa_attention.qsa_sparse_prefill_widths() if w >= columns
    )
    buffer = torch.full(
        (positions.numel(), kernel_width), -1, dtype=torch.int32, device=device
    )
    in_place = qsa_expand_block_indices(
        *args, buffer[:, :columns], output_width=columns
    )
    assert in_place.data_ptr() == buffer.data_ptr()
    assert torch.equal(in_place, fresh)
    assert bool((buffer[:, columns:] == -1).all())
