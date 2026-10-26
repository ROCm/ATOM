# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whole-request workspace limits preserve paged causal attention semantics."""

from types import SimpleNamespace

import pytest
import torch

from atom.config import Config
from atom.model_ops.prefill_workspace import PrefillRequestLengths, run_grouped_prefill
from atom.utils.forward_context import AttentionMetaData


def test_original_stalled_batch_fits_eight_gib_per_group():
    cached = [
        440192,
        68352,
        75904,
        100992,
        168448,
        227072,
        128,
        128,
        92672,
        70656,
        210816,
        695296,
        653312,
        682496,
        511104,
    ]
    new = [
        1674,
        730,
        11224,
        2951,
        303,
        271,
        505,
        374,
        3604,
        598,
        2587,
        2692,
        2217,
        2223,
        768,
    ]
    cu_q, cu_k = [0], [0]
    for old, added in zip(cached, new):
        cu_q.append(cu_q[-1] + added)
        cu_k.append(cu_k[-1] + old + added)
    lengths = PrefillRequestLengths(tuple(cu_q), tuple(cu_k))
    groups = lengths.groups((8 << 30) // (16 * 128 * 2 * 2))
    assert cu_k[-1] == 4030289
    assert len(groups) > 1
    assert [i for start, end in groups for i in range(start, end)] == list(range(15))
    assert max(cu_k[end] - cu_k[start] for start, end in groups) <= 1048576


def test_exact_fit_keeps_requests_together():
    lengths = PrefillRequestLengths((0, 2, 5, 6), (0, 7, 10, 19))
    assert lengths.groups(10) == [(0, 2), (2, 3)]


@pytest.mark.parametrize(
    "cu_q,cu_k",
    [((0, 4), (0, 3)), ((1, 3), (0, 3)), ((0, 2, 2), (0, 3, 4)), ((0, 1), (0, 1, 2))],
)
def test_invalid_host_geometry_is_rejected(cu_q, cu_k):
    with pytest.raises(ValueError):
        PrefillRequestLengths(cu_q, cu_k)


def test_single_request_is_never_truncated():
    with pytest.raises(ValueError, match="request 0 needs 1048577 KV tokens"):
        PrefillRequestLengths((0, 2), (0, 1048577)).groups(1048576)


@pytest.mark.parametrize("max_tokens", [9, 10, 27])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("is_draft", [False, True])
def test_grouped_paged_attention_matches_unsplit_reference(
    max_tokens, kv_heads, is_draft
):
    torch.manual_seed(23)
    q_lengths = torch.tensor([2, 3, 1, 4, 2], dtype=torch.int32)
    k_lengths = torch.tensor([7, 3, 9, 6, 2], dtype=torch.int32)
    cu_q = torch.cat([torch.zeros(1, dtype=torch.int32), q_lengths.cumsum(0).int()])
    cu_k = torch.cat([torch.zeros(1, dtype=torch.int32), k_lengths.cumsum(0).int()])
    blocks = torch.randperm(25).reshape(5, 5)
    starts = torch.tensor([0, 1, 3, 0, 2], dtype=torch.int32)
    k_cache = torch.randn(25, 4, kv_heads, 4, dtype=torch.float64)
    v_cache = torch.randn_like(k_cache)
    k_scale = torch.rand(25, 4, kv_heads, 1, dtype=torch.float64) + 0.2
    v_scale = torch.rand_like(k_scale) + 0.2
    # Four query heads exercise MQA, GQA and MHA (Eagle can have Q heads == KV heads).
    q = torch.randn(int(cu_q[-1]), 4, 4, dtype=torch.float64)
    k = torch.empty(len(q), kv_heads, 4, dtype=q.dtype)
    meta = AttentionMetaData(
        has_cached=True,
        total_kv=int(cu_k[-1]),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        block_tables=blocks,
        seq_starts=starts,
        context_lens=k_lengths,
        batch_id_per_k_token=torch.repeat_interleave(
            torch.arange(5, dtype=torch.int32), k_lengths
        ),
        max_seqlen_q=4,
        max_seqlen_k=9,
        prefill_request_lengths=PrefillRequestLengths(
            tuple(cu_q.tolist()), tuple(cu_k.tolist())
        ),
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_draft=is_draft), attn_metadata=meta
    )
    original = {
        name: value.clone()
        for name, value in vars(meta).items()
        if isinstance(value, torch.Tensor)
    }
    seen = []

    def reference(queries, md):
        outputs = []
        for i, (a, b) in enumerate(zip(md.cu_seqlens_q[:-1], md.cu_seqlens_q[1:])):
            count = int(md.context_lens[i])
            pos = torch.arange(count) + md.seq_starts[i]
            pages, offsets = md.block_tables[i, pos // 4], pos % 4
            keys = (
                k_cache[pages, offsets] * k_scale[pages, offsets]
            ).repeat_interleave(4 // kv_heads, dim=1)
            values = (
                v_cache[pages, offsets] * v_scale[pages, offsets]
            ).repeat_interleave(4 // kv_heads, dim=1)
            sub_q = queries[int(a) : int(b)]
            logits = torch.einsum("qhd,khd->hqk", sub_q, keys) / 2.0
            mask = torch.arange(count)[None, :] > torch.arange(len(sub_q))[
                :, None
            ] + count - len(sub_q)
            logits.masked_fill_(mask, float("-inf"))
            outputs.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), values))
        return torch.cat(outputs)

    expected = reference(q, meta)

    def attention(q, k, v, kc, vc, ks, vs, context):
        assert kc is k_cache and vc is v_cache and ks is k_scale and vs is v_scale
        md = context.attn_metadata
        seen.append(md.total_kv)
        assert md.total_kv <= max_tokens
        torch.testing.assert_close(
            md.batch_id_per_k_token,
            torch.repeat_interleave(
                torch.arange(len(md.context_lens), dtype=torch.int32), md.context_lens
            ),
        )
        assert md.max_seqlen_q == int(
            (md.cu_seqlens_q[1:] - md.cu_seqlens_q[:-1]).max()
        )
        assert md.max_seqlen_k == int(md.context_lens.max())
        return reference(q, md)

    actual = run_grouped_prefill(
        attention,
        q,
        k,
        k,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        ctx,
        max_tokens * 2 * kv_heads * 4 * 8,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(seen) == {9: 4, 10: 3, 27: 1}[max_tokens]
    assert ctx.attn_metadata is meta
    for name, value in original.items():
        torch.testing.assert_close(getattr(meta, name), value, rtol=0, atol=0)


def test_missing_or_stale_lengths_fail_before_workspace_allocation():
    q = torch.empty(2, 1, 4)
    meta = AttentionMetaData(has_cached=True, total_kv=10)
    ctx = SimpleNamespace(attn_metadata=meta)

    def unexpected(*args):
        pytest.fail("attention must not run with invalid lengths")

    with pytest.raises(ValueError, match="requires CPU request lengths"):
        run_grouped_prefill(unexpected, q, q, q, q, q, None, None, ctx, 64)
    meta.prefill_request_lengths = PrefillRequestLengths((0, 1), (0, 10))
    with pytest.raises(ValueError, match="disagree"):
        run_grouped_prefill(unexpected, q, q, q, q, q, None, None, ctx, 64)


def test_under_budget_does_not_need_host_metadata_or_copy_output():
    q = torch.empty(2, 1, 4)
    ctx = SimpleNamespace(attn_metadata=AttentionMetaData(has_cached=True, total_kv=2))

    def attention(*args):
        assert args[-1] is ctx
        return q

    assert run_grouped_prefill(attention, q, q, q, q, q, None, None, ctx, 64) is q


@pytest.mark.parametrize("budget", [-1, 0.5, True, "8192", None])
def test_workspace_config_rejects_invalid_budgets_before_model_loading(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        Config(model="unused", gqa_prefill_kv_workspace_mb=budget)


def test_workspace_rejects_sharded_prefill_before_model_loading():
    with pytest.raises(ValueError, match="requires PCP=1"):
        Config(
            model="unused",
            gqa_prefill_kv_workspace_mb=8192,
            prefill_context_parallel_size=2,
        )


def test_layer_geometry_changes_groups_without_changing_shared_metadata():
    # One shared request layout is used first by target MQA, then by Eagle MHA.
    # No target-derived group plan may be reused for the larger draft heads.
    cu = torch.tensor([0, 8, 16], dtype=torch.int32)
    md = AttentionMetaData(
        has_cached=True,
        total_kv=16,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        context_lens=torch.tensor([8, 8], dtype=torch.int32),
        seq_starts=torch.zeros(2, dtype=torch.int32),
        block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        batch_id_per_k_token=torch.tensor([0] * 8 + [1] * 8, dtype=torch.int32),
        prefill_request_lengths=PrefillRequestLengths((0, 8, 16), (0, 8, 16)),
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_draft=False), attn_metadata=md
    )
    q = torch.randn(16, 4, 4)
    seen = []

    def attention(q, k, v, kc, vc, ks, vs, fc):
        seen.append(fc.attn_metadata.total_kv)
        return q

    budget = 1024
    for is_draft, heads, expected in [(False, 1, [16]), (True, 4, [8, 8])]:
        ctx.context.is_draft = is_draft
        k = torch.empty(16, heads, 4)
        seen.clear()
        out = run_grouped_prefill(attention, q, k, k, k, k, None, None, ctx, budget)
        torch.testing.assert_close(out, q)
        assert seen == expected
        assert ctx.attn_metadata is md
        assert md.prefill_request_lengths.cu_k_lens == (0, 8, 16)


@pytest.mark.parametrize("has_cached", [False, True])
@pytest.mark.parametrize("attach_legacy_lens", [False, True])
def test_tbo_request_and_token_views_keep_owned_host_geometry(
    monkeypatch, has_cached, attach_legacy_lens
):
    import numpy as np

    from atom.utils.tbo.ubatch_splitting import (
        UBatchSlice,
        attach_tbo_cpu_lens,
        split_attn_metadata,
    )

    cu_q = np.array([0, 2, 5, 9], dtype=np.int32)
    cu_k = np.array([0, 7, 16, 24], dtype=np.int32) if has_cached else cu_q.copy()
    md = AttentionMetaData(
        has_cached=has_cached,
        total_kv=int(cu_k[-1]),
        cu_seqlens_q=torch.from_numpy(cu_q.copy()),
        cu_seqlens_k=torch.from_numpy(cu_k.copy()),
        context_lens=torch.from_numpy(np.diff(cu_k).copy()),
        block_tables=torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.int32),
        seq_starts=torch.zeros(3, dtype=torch.int32),
        batch_id_per_k_token=torch.repeat_interleave(
            torch.arange(3, dtype=torch.int32), torch.from_numpy(np.diff(cu_k))
        ),
        prefill_request_lengths=PrefillRequestLengths(
            tuple(map(int, cu_q)), tuple(map(int, cu_k))
        ),
    )
    if attach_legacy_lens:
        attach_tbo_cpu_lens(md, "cu_seqlens_q", cu_q.copy())
        attach_tbo_cpu_lens(md, "cu_seqlens_k", cu_k.copy())
        attach_tbo_cpu_lens(md, "context_lens", np.diff(cu_k))

    def no_readback(*args, **kwargs):
        pytest.fail("owned host lengths must not require device readback")

    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, "item", no_readback)
        m.setattr(torch.Tensor, "cpu", no_readback)
        # The first selected request contributes only its last two query tokens.
        ub = split_attn_metadata(
            md, UBatchSlice(slice(1, 3), slice(3, 9)), running_bs=4
        )
    expected_k = (0, 9, 17) if has_cached else (0, 2, 6)
    assert ub.prefill_request_lengths == PrefillRequestLengths((0, 2, 6), expected_k)
    assert ub.cu_seqlens_q.tolist() == [0, 2, 6, 6, 6]
    assert ub.cu_seqlens_k.tolist() == list(expected_k) + [expected_k[-1]] * 2
    assert ub.total_kv == (17 if has_cached else None)
    assert md.total_kv == int(cu_k[-1])
    assert md.prefill_request_lengths.cu_q_lens == (0, 2, 5, 9)
    cu_q[:] = -1
    cu_k[:] = -1
    assert ub.prefill_request_lengths.cu_q_lens == (0, 2, 6)


def test_tp1_eagle_1m_requests_fit_32_gib_but_not_8_gib():
    """Exercise real grouping at the production length without allocating KV."""
    length = 1 << 20
    q = torch.empty((2, 64, 128), dtype=torch.bfloat16)
    meta = AttentionMetaData(
        has_cached=True,
        total_kv=2 * length,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, length, 2 * length], dtype=torch.int32),
        block_tables=torch.zeros((2, 1), dtype=torch.int32),
        seq_starts=torch.zeros(2, dtype=torch.int32),
        context_lens=torch.full((2,), length, dtype=torch.int32),
        batch_id_per_k_token=torch.repeat_interleave(torch.arange(2), length),
        prefill_request_lengths=PrefillRequestLengths(
            (0, 1, 2), (0, length, 2 * length)
        ),
    )
    ctx = SimpleNamespace(attn_metadata=meta)
    seen = []

    def attention(q, k, v, kc, vc, ks, vs, context):
        seen.append(context.attn_metadata.total_kv)
        return torch.ones_like(q)

    with pytest.raises(ValueError, match="workspace fits 262144"):
        run_grouped_prefill(attention, q, q, q, None, None, None, None, ctx, 8 << 30)
    result = run_grouped_prefill(
        attention, q, q, q, None, None, None, None, ctx, 32 << 30
    )
    assert seen == [length, length]
    assert torch.equal(result, torch.ones_like(q))


@pytest.mark.parametrize("overlap", [1, 2])
def test_workspace_reservation_includes_local_draft_not_sparse_layers(overlap):
    from atom.model_ops.prefill_workspace import prefill_workspace_reserve_bytes

    def model(*layers):
        return SimpleNamespace(modules=lambda: iter(layers))

    sparse = SimpleNamespace(
        gqa_prefill_kv_workspace_bytes=64 << 30, is_indexed_sparse_attention=True
    )
    dense = SimpleNamespace(gqa_prefill_kv_workspace_bytes=8 << 30)
    draft = SimpleNamespace(gqa_prefill_kv_workspace_bytes=32 << 30)
    other = SimpleNamespace()
    assert prefill_workspace_reserve_bytes([model(sparse, other)]) == 0
    assert (
        prefill_workspace_reserve_bytes(
            [model(SimpleNamespace(gqa_prefill_kv_workspace_bytes=0))]
        )
        == 0
    )
    assert (
        prefill_workspace_reserve_bytes(
            [model(dense, dense, sparse), model(draft)], overlapping_batches=overlap
        )
        == (32 << 30) * overlap
    )
