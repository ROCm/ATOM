# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host side of the QSA FlyDSL prefill: one layout per forward, the selection
written at the kernel width, the paged pool read through the slot table, and
every case the native kernel has to take instead."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

TOKENS = 8
# aiter's answers on gfx942 (ROCm/aiter#8083), where the CPU job has no aiter.
WIDTHS = (1027, 2051, 3075, 4099)


def _supports_selection(rows, width):
    return width in WIDTHS and rows * width * 4 < 2**31


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            qsa_attention=load("atom.model_ops.qwen4_exp.qsa_attention"),
            attn=load("atom.model_ops.attentions.qwen4_exp_attn"),
        )


def _layer(**overrides):
    pages, block = 4, 8
    rows = torch.arange(pages * block, dtype=torch.bfloat16)
    k_cache = rows.view(pages, block, 1, 1).expand(-1, -1, 1, 256).contiguous()
    layer = SimpleNamespace(
        indexer=SimpleNamespace(compress_ratio=4, token_topk=2048),
        k_scale_cache=None,
        num_kv_heads=1,
        num_heads=12,
        head_dim=256,
        scaling=0.0625,
        # 16 compressed groups per page: with 32 table pages a row can reach
        # 512 groups, the 2048 budget.
        compressed_key_cache=torch.zeros(pages, 16, 1, 128, dtype=torch.bfloat16),
        k_cache=k_cache,
        v_cache=-k_cache,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _qsa(
    modules, tokens=TOKENS, prefix=0, max_seq_len=8192, context_slots=None, pages=32
):
    """A real per-forward `Qwen4ExpQSAMetadata`, laid out for FlyDSL."""
    return modules.attn.Qwen4ExpQSAMetadata(
        block_tables=torch.zeros(1, pages, dtype=torch.int32),
        slot_mapping=torch.arange(tokens),
        compressed_slot_mapping=torch.full((tokens,), -1),
        token_to_req=torch.zeros(tokens, dtype=torch.int32),
        logical_positions=torch.arange(prefix, prefix + tokens),
        seq_lens=torch.tensor([prefix + tokens], dtype=torch.int32),
        max_seq_len=max_seq_len,
        host_lens=((tokens,), (prefix + tokens,)),
        context_slots=context_slots,
    )


def _query(tokens=TOKENS):
    return torch.zeros(tokens, 12, 256, dtype=torch.bfloat16)


def _attend(modules, layer, qsa, query=None, columns=None):
    """One QSA layer's FlyDSL part, as `forward` runs it: the layout, the
    selection written into its buffer, then the attention."""
    qa = modules.qsa_attention
    query = _query() if query is None else query
    plan = qa._FlyDSLPrefill.of(layer, query, qsa)
    if plan.ready:
        out = plan.selection_out(query.device)
        columns = plan.width if columns is None else columns
        out[:, :columns] = torch.arange(columns, dtype=torch.int32)
    kv = query[:, :1]
    return qa.Qwen4ExpAttention._flydsl_attention(layer, query, kv, kv, plan, qsa)


@pytest.fixture
def flydsl(modules, monkeypatch):
    qa = modules.qsa_attention
    calls = []
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "1")
    qa.log_flydsl_fallback.cache_clear()
    qa._log_flydsl_active.cache_clear()
    monkeypatch.setattr(qa, "qsa_sparse_prefill_widths", lambda: WIDTHS)
    monkeypatch.setattr(
        qa, "qsa_sparse_prefill_supports_selection", _supports_selection
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        qa,
        "qsa_sparse_prefill",
        lambda q, k, v, indices, **kw: calls.append((k, v, indices, kw)) or q,
    )
    return calls


def test_supported_layer_takes_flydsl(modules, flydsl):
    qsa = _qsa(modules)
    assert _attend(modules, _layer(), qsa) is not None
    ((_, _, indices, kwargs),) = flydsl
    assert indices is qsa.flydsl_prefill.indices and indices.shape == (TOKENS, 2051)
    assert kwargs["kv_slots"] is None
    assert kwargs["query_lens"] == (TOKENS,) and kwargs["prefix_lens"] == (0,)


@pytest.mark.parametrize("budget", [1024, 2048, 3072, 4096])
def test_short_context_selects_into_the_narrowest_width(modules, flydsl, budget):
    """A 40-token context reaches 10 groups, rounded up to a power of two: 16,
    so the selection is 67 columns and the kernel gets the narrowest width,
    -1 past the selection."""
    layer = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=budget))
    qsa = _qsa(modules, max_seq_len=40)
    assert _attend(modules, layer, qsa, columns=67) is not None
    ((_, _, indices, _),) = flydsl
    assert indices.shape == (TOKENS, 1027)
    assert torch.equal(
        indices[:, :67], torch.arange(67, dtype=torch.int32).expand(8, -1)
    )
    assert bool((indices[:, 67:] == -1).all())


@pytest.mark.parametrize(
    "overrides",
    [
        {"indexer": SimpleNamespace(compress_ratio=4, token_topk=1536)},
        {"indexer": SimpleNamespace(compress_ratio=8, token_topk=2048)},
        {"k_scale_cache": torch.ones(1)},
    ],
    ids=["unsupported-budget", "ratio-8", "fp8-kv"],
)
def test_unsupported_layer_falls_back(modules, flydsl, overrides):
    qsa = _qsa(modules)
    assert _attend(modules, _layer(**overrides), qsa) is None
    assert flydsl == [] and qsa.flydsl_prefill.indices is None


def test_selection_over_aiter_byte_limit_falls_back(modules, flydsl):
    # 131072 rows x 4099 int32 columns span 2,149,056,512 bytes (>= 2**31).
    tokens = 131072
    layer = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=4096))
    qsa = _qsa(modules, tokens=tokens, max_seq_len=tokens, pages=64)
    query = torch.zeros(1, 12, 256, dtype=torch.bfloat16).expand(tokens, -1, -1)
    assert _attend(modules, layer, qsa, query) is None
    assert flydsl == []


def test_aiter_unavailable_falls_back(modules, flydsl, monkeypatch, caplog):
    monkeypatch.setattr(modules.qsa_attention, "qsa_sparse_prefill_widths", tuple)
    with caplog.at_level("WARNING"):
        assert _attend(modules, _layer(), _qsa(modules)) is None
    assert flydsl == []
    assert "does not run on this GPU" in _warnings(caplog)[0]


def test_flag_off_falls_back(modules, flydsl, monkeypatch):
    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
    qsa = _qsa(modules)
    assert _attend(modules, _layer(), qsa) is None
    assert flydsl == [] and not qsa.flydsl_prefill.laid_out


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if "native kernel" in r.message]


def test_each_fallback_reason_is_logged_once(modules, flydsl, monkeypatch, caplog):
    budget = _layer(indexer=SimpleNamespace(compress_ratio=4, token_topk=1536))
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert _attend(modules, budget, _qsa(modules)) is None
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        for _ in range(3):
            assert _attend(modules, _layer(), _qsa(modules)) is None
    reasons = _warnings(caplog)
    assert len(reasons) == 2
    assert "budget 1536" in reasons[0] and "graph capture" in reasons[1]


def test_unlaid_forward_and_flag_off_log_nothing(modules, flydsl, monkeypatch, caplog):
    """A forward the bridge did not lay out (target verify, warmup, ATOM's own
    runner) is not a fallback, and with the flag off nothing is."""
    unlaid = _qsa(modules)
    unlaid.host_lens = None
    with caplog.at_level("WARNING"):
        assert _attend(modules, _layer(), unlaid) is None
        monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
        ratio_8 = _layer(indexer=SimpleNamespace(compress_ratio=8, token_topk=2048))
        assert _attend(modules, ratio_8, _qsa(modules)) is None
    assert _warnings(caplog) == [] and flydsl == []


def test_first_flydsl_call_is_logged_once(modules, flydsl, caplog):
    with caplog.at_level("INFO"):
        for _ in range(3):
            assert _attend(modules, _layer(), _qsa(modules)) is not None
    active = [r for r in caplog.records if "FlyDSL kernel" in r.message]
    assert len(active) == 1 and len(flydsl) == 3


def test_layout_is_made_once_per_forward(modules, flydsl, monkeypatch):
    """The QSA layers of one forward share the first layer's layout: the
    checks, the width and the prefix lengths are not redone per layer, and
    they all get the same selection buffer. Another row count is another
    layout."""
    qa = modules.qsa_attention
    checks, widths = [], []
    real_check, real_widths = qa.flydsl_fallback_reason, qa.qsa_selection_widths
    monkeypatch.setattr(
        qa,
        "flydsl_fallback_reason",
        lambda *a, **kw: checks.append(a) or real_check(*a, **kw),
    )
    monkeypatch.setattr(
        qa, "qsa_selection_widths", lambda *a: widths.append(a) or real_widths(*a)
    )
    layer, qsa = _layer(), _qsa(modules, prefix=6080)
    for _ in range(12):
        assert _attend(modules, layer, qsa) is not None
    assert len(checks) == 1 and len(widths) == 1 and len(flydsl) == 12
    plan = qsa.flydsl_prefill
    assert all(indices is plan.indices for _, _, indices, _ in flydsl)
    assert plan.query_lens == (TOKENS,) and plan.prefix_lens == (6080,)
    qsa.host_lens = ((4,), (6084,))
    assert _attend(modules, layer, qsa, _query(4)) is not None
    assert len(checks) == 2 and qsa.flydsl_prefill is not plan


def test_prefixed_layout_reads_the_pool_through_the_slot_table(modules, flydsl):
    """With a prefix the kernel gets the paged pool as is and the bridge's
    context rows as its slot table; nothing is gathered."""
    slots = torch.arange(16, 24, dtype=torch.int32)
    layer = _layer()
    qsa = _qsa(modules, tokens=3, prefix=5, context_slots=slots)
    assert _attend(modules, layer, qsa, _query(3)) is not None
    ((k, v, _, kwargs),) = flydsl
    assert kwargs["kv_slots"] is slots and kwargs["prefix_lens"] == (5,)
    assert k.data_ptr() == layer.k_cache.data_ptr() and k.shape == (32, 1, 256)
    assert v.data_ptr() == layer.v_cache.data_ptr() and v.shape == (32, 1, 256)


def test_draft_rows_trim_drops_the_layout(modules, flydsl):
    """`for_tokens` keeps the layout only when it returns the same rows."""
    qsa = _qsa(modules)
    assert _attend(modules, _layer(), qsa) is not None
    plan = qsa.flydsl_prefill
    assert (
        qsa.for_tokens(TOKENS) is qsa and qsa.for_tokens(TOKENS).flydsl_prefill is plan
    )
    trimmed = qsa.for_tokens(TOKENS - 2)
    assert trimmed.flydsl_prefill is None


def _mark(rows, width):
    """What the selection spies write: row r, column c holds 1000 * r + c."""
    return (
        1000 * torch.arange(rows, dtype=torch.int32)[:, None]
        + torch.arange(width, dtype=torch.int32)
    ).contiguous()


@pytest.mark.parametrize("sharded", [False, True], ids=["whole", "sequence-parallel"])
def test_forward_selects_into_the_kernel_buffer(modules, flydsl, monkeypatch, sharded):
    """`forward` on a prefill: the layout is made, the selection lands in its
    kernel-width buffer (`out=`, and `out[:, :width]` when the rows are
    sharded), and that buffer is what the kernel reads. Without `out` the
    kernel would read an all -1 buffer; a wrong slice moves the selection."""
    qa = modules.qsa_attention
    heads, dim = 12, 256
    q_size, kv_size = heads * dim, dim
    layer = _layer(
        indexer=SimpleNamespace(
            compress_ratio=4,
            token_topk=2048,
            # The spies read the query rows back from the index query.
            project_qk=lambda h, p: (
                torch.arange(h.shape[0], dtype=torch.float32)[:, None],
                None,
            ),
            normalize_compressed_keys=lambda pooled, first: None,
        ),
        q_size=q_size,
        kv_size=kv_size,
        raw_key_cache=None,
        rope_position_cache=None,
        compressed_key_scale=None,
        rotary_emb=None,
        qkv_proj=lambda h: torch.zeros(
            h.shape[0], 2 * q_size + 2 * kv_size, dtype=torch.bfloat16
        ),
        qk_norm=lambda q, k: (q, k),
        o_proj=lambda x: x,
        _store_kv=lambda key, value, slots: None,
    )
    for name in ("_select_tokens", "_flydsl_attention"):
        setattr(layer, name, getattr(qa.Qwen4ExpAttention, name).__get__(layer))

    def native(*args, **kwargs):
        raise AssertionError("a ready layout must not reach the native kernel")

    layer._attend = native
    qsa = _qsa(modules, max_seq_len=40)
    width = qa.qsa_selection_widths(
        layer.compressed_key_cache, qsa.block_tables, 2048, 4, qsa.max_seq_len
    )[2]
    context = SimpleNamespace(is_prefill=True, is_draft=False)
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(qsa_metadata=qsa), context=context
    )

    def select_tokens(index_q, *args, out=None, **kwargs):
        if out is None:
            out = torch.empty(index_q.shape[0], width, dtype=torch.int32)
        assert out.shape[0] == index_q.shape[0] and out.shape[1] >= width
        out = out[:, :width]
        out.copy_(_mark(out.shape[0], width))
        return out

    def select_groups(index_q, *args, **kwargs):
        return index_q.to(torch.int32), width

    def expand(
        groups,
        positions,
        seq_lens,
        token_to_req,
        ratio,
        topk,
        out=None,
        *,
        output_width,
    ):
        selection = groups * 1000 + torch.arange(output_width, dtype=torch.int32)
        if out is None:
            return selection
        assert out.shape == selection.shape, "out must be exactly [rows, width]"
        return out.copy_(selection)

    for name, value in {
        "get_forward_context": lambda: forward_context,
        "qsa_apply_mrope": lambda rope, positions, q, k: (q, k),
        "qsa_store_rows": lambda *args: None,
        "qsa_compress_groups": lambda *args, **kwargs: (None, None),
        "qsa_select_paged_tokens": select_tokens,
        "qsa_select_paged_groups": select_groups,
        "qsa_expand_block_indices": expand,
        "is_per_token_fp8_linear": lambda proj: False,
        "sigmoid_mul": lambda x, gate: x,
    }.items():
        monkeypatch.setattr(qa, name, value)

    rows = None
    if sharded:
        # This rank holds rows 4..7 of two ranks' 8; the gather returns every
        # rank's groups, row r's as r.
        def gather(local):
            groups = torch.arange(TOKENS, dtype=torch.int32)[:, None]
            assert torch.equal(local, groups[4:])
            return groups

        rows = SimpleNamespace(start=4, chunk=4, num_tokens=TOKENS, gather=gather)
    hidden = torch.zeros(TOKENS, 64, dtype=torch.bfloat16)
    positions = torch.arange(TOKENS)
    out = qa.Qwen4ExpAttention.forward(layer, positions, hidden, rows=rows)

    assert out.shape == (TOKENS, q_size)
    ((_, _, indices, _),) = flydsl
    plan = qsa.flydsl_prefill
    assert plan.ready and indices is plan.indices
    assert indices.shape == (TOKENS, plan.width) and plan.width > width
    assert torch.equal(indices[:, :width], _mark(TOKENS, width))
    assert bool((indices[:, width:] == -1).all())
