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
    qsa_attention.log_flydsl_fallback.cache_clear()
    qsa_attention._log_flydsl_active.cache_clear()
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


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if "native kernel" in r.message]


def test_each_fallback_reason_is_logged_once(flydsl, monkeypatch, caplog):
    budget = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=1536))
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert _call(budget, _selection(8, 1539)) is None
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        for _ in range(3):
            assert _call(_layer()) is None
    reasons = _warnings(caplog)
    assert len(reasons) == 2
    assert "budget 1536" in reasons[0] and "graph capture" in reasons[1]


def test_unlaid_forward_and_flag_off_log_nothing(flydsl, monkeypatch, caplog):
    """A forward the bridge did not lay out (target verify, warmup) is not a
    fallback, and with the flag off nothing is."""
    no_layout = SimpleNamespace(host_lens=None, context_slots=None)
    query = torch.zeros(8, 12, 256, dtype=torch.bfloat16)
    with caplog.at_level("WARNING"):
        assert (
            Qwen4ExpAttention._flydsl_attention(
                _layer(), query, query, query, _selection(8, 2051), no_layout
            )
            is None
        )
        monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
        assert (
            _call(_layer(indexer=SimpleNamespace(compress_ratio=8, token_topk=2048)))
            is None
        )
    assert _warnings(caplog) == [] and flydsl == []


def test_first_flydsl_call_is_logged_once(flydsl, caplog):
    with caplog.at_level("INFO"):
        for _ in range(3):
            assert _call(_layer()) is not None
    active = [r for r in caplog.records if "FlyDSL kernel" in r.message]
    assert len(active) == 1 and len(flydsl) == 3


def test_layout_is_made_once_per_forward(flydsl, monkeypatch):
    """The QSA layers of one forward share the first layer's layout: the
    checks and the prefix lengths are not redone per layer. Another row
    count is another layout."""
    checks = []
    real = qsa_attention.flydsl_fallback_reason
    monkeypatch.setattr(
        qsa_attention,
        "flydsl_fallback_reason",
        lambda *a, **kw: checks.append(a) or real(*a, **kw),
    )
    layer, tokens = _layer(), 8
    query = torch.zeros(tokens, 12, 256, dtype=torch.bfloat16)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=None)
    for _ in range(12):
        out = Qwen4ExpAttention._flydsl_attention(
            layer, query, query[:, :1], query[:, :1], _selection(tokens, 2051), qsa
        )
        assert out is not None
    assert len(checks) == 1 and len(flydsl) == 12
    plan = qsa.flydsl_prefill
    assert plan.query_lens == (tokens,) and plan.prefix_lens == (0,)
    shorter = query[:4]
    qsa.host_lens = ((4,), (4,))
    Qwen4ExpAttention._flydsl_attention(
        layer, shorter, shorter[:, :1], shorter[:, :1], _selection(4, 2051), qsa
    )
    assert len(checks) == 2 and qsa.flydsl_prefill is not plan


def test_selection_written_in_place_reaches_the_kernel_unpadded(flydsl):
    """`forward` selects into the layout's kernel-width buffer; that buffer,
    -1 past the selection, is what the kernel gets, with no padded copy."""
    layer, tokens = _layer(), 8
    query = torch.zeros(tokens, 12, 256, dtype=torch.bfloat16)
    qsa = SimpleNamespace(host_lens=((tokens,), (tokens,)), context_slots=None)
    plan = qsa_attention._FlyDSLPrefill.of(layer, query, qsa, 11, 4)
    buffer = plan.selection_out(query.device)
    assert buffer.shape == (tokens, 1027) and bool((buffer == -1).all())
    assert plan.selection_out(query.device) is buffer
    selected = buffer[:, :11]
    selected.copy_(_selection(tokens, 11))
    for _ in range(2):  # a second layer reuses the same buffer
        out = Qwen4ExpAttention._flydsl_attention(
            layer, query, query, query, selected, qsa
        )
        assert out is not None
    assert all(indices is buffer for indices in flydsl)
    assert torch.equal(buffer[:, :11], _selection(tokens, 11))
    assert bool((buffer[:, 11:] == -1).all())


@pytest.mark.parametrize(
    "slot_dtype, contiguous, in_place",
    [
        (torch.int32, True, True),
        (torch.int64, True, False),
        (torch.int32, False, False),
    ],
    ids=["int32-slot-table", "int64-rows-gathered", "strided-pool-gathered"],
)
def test_prefixed_layout_reads_the_pool_through_the_slot_table(
    flydsl, monkeypatch, slot_dtype, contiguous, in_place
):
    """With a prefix the kernel reads the paged pool in place through the
    bridge's int32 slot table; rows it cannot address that way (another slot
    dtype, a strided pool) are gathered as before."""
    calls = []
    monkeypatch.setattr(
        qsa_attention,
        "_flydsl_attention_op",
        lambda: lambda q, k, v, indices, **kw: calls.append((k, v, kw)) or q,
    )
    pages, block, prefix, tokens = 4, 8, 5, 3
    rows = torch.arange(pages * block, dtype=torch.bfloat16)
    k_cache = rows.view(pages, block, 1, 1).expand(-1, -1, 1, 256).contiguous()
    v_cache = -k_cache
    if not contiguous:
        k_cache = torch.cat([k_cache, k_cache], dim=-2)[:, :, :1]
        v_cache = -k_cache
    slots = torch.tensor([16, 17, 18, 19, 20, 21, 22, 23], dtype=slot_dtype)
    layer = _layer(k_cache=k_cache, v_cache=v_cache)
    query = torch.zeros(tokens, 12, 256, dtype=torch.bfloat16)
    qsa = SimpleNamespace(
        host_lens=((tokens,), (prefix + tokens,)), context_slots=slots
    )
    out = Qwen4ExpAttention._flydsl_attention(
        layer, query, query[:, :1], query[:, :1], _selection(tokens, 2051), qsa
    )
    assert out is not None
    ((k, v, kwargs),) = calls
    assert kwargs["prefix_lens"] == (prefix,)
    if in_place:
        assert kwargs["kv_slots"] is slots
        assert k.data_ptr() == k_cache.data_ptr() and k.shape == (pages * block, 1, 256)
        assert v.data_ptr() == v_cache.data_ptr()
    else:
        assert kwargs["kv_slots"] is None
        assert k[:, 0, 0].tolist() == slots.tolist()
        assert torch.equal(v, -k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_mixed_prefix_batch_gathers_full_contexts(monkeypatch):
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
    bits as gathering every context row and padding a copy of the selection.
    Stale pool rows hold random finite values."""
    from atom.model_ops.qwen4_exp.ops.qsa import qsa_select_paged_tokens
    from atom.plugin.sglang.qwen4_exp_bridge import _flydsl_prefill_layout

    if not qsa_attention._flydsl_arch_supported():
        pytest.skip("FlyDSL QSA attention needs gfx942")
    try:
        qsa_attention._flydsl_attention_op()
    except ImportError:
        pytest.skip("aiter without FlyDSL QSA attention (ROCm/aiter#6576)")
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
    qsa = SimpleNamespace(host_lens=host_lens, context_slots=context_slots)
    for _ in range(2):
        k_cache, v_cache = randn(pages, block, 1, dim), randn(pages, block, 1, dim)
        layer = _layer(
            indexer=SimpleNamespace(compress_ratio=ratio, token_topk=budget),
            num_heads=heads,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        key = k_cache.flatten(0, 1)[slots]
        value = v_cache.flatten(0, 1)[slots]
        gathered = SimpleNamespace(
            host_lens=host_lens,
            context_slots=None if context_slots is None else context_slots.long(),
        )
        reference = Qwen4ExpAttention._flydsl_attention(
            layer, query, key, value, select(), gathered
        )
        plan = qsa_attention._FlyDSLPrefill.of(layer, query, qsa, select().shape[1], 4)
        selected = select(plan.selection_out(device))
        assert selected.data_ptr() == plan.indices.data_ptr()
        in_place = Qwen4ExpAttention._flydsl_attention(
            layer, query, key, value, selected, qsa
        )
        assert reference is not None and in_place is not None
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
    kernel_width = next(w for w in qsa_attention._FLYDSL_WIDTHS if w >= columns)
    buffer = torch.full(
        (positions.numel(), kernel_width), -1, dtype=torch.int32, device=device
    )
    in_place = qsa_expand_block_indices(
        *args, buffer[:, :columns], output_width=columns
    )
    assert in_place.data_ptr() == buffer.data_ptr()
    assert torch.equal(in_place, fresh)
    assert bool((buffer[:, columns:] == -1).all())
