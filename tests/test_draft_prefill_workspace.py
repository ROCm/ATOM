# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whole-request workspace limits preserve paged causal attention semantics."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from atom.config import Config, SpeculativeConfig
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
def test_grouped_paged_attention_matches_unsplit_reference(max_tokens):
    torch.manual_seed(23)
    q_lengths = torch.tensor([2, 3, 1, 4, 2], dtype=torch.int32)
    k_lengths = torch.tensor([7, 3, 9, 6, 2], dtype=torch.int32)
    cu_q = torch.cat([torch.zeros(1, dtype=torch.int32), q_lengths.cumsum(0).int()])
    cu_k = torch.cat([torch.zeros(1, dtype=torch.int32), k_lengths.cumsum(0).int()])
    blocks = torch.randperm(25).reshape(5, 5)
    starts = torch.tensor([0, 1, 3, 0, 2], dtype=torch.int32)
    k_cache = torch.randn(25, 4, 2, 4, dtype=torch.float64)
    v_cache = torch.randn_like(k_cache)
    k_scale = torch.rand(25, 4, 2, 1, dtype=torch.float64) + 0.2
    v_scale = torch.rand_like(k_scale) + 0.2
    # GQA: four query heads for two KV heads.
    q = torch.randn(int(cu_q[-1]), 4, 4, dtype=torch.float64)
    k = torch.empty(len(q), 2, 4, dtype=q.dtype)
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
    ctx = SimpleNamespace(context=SimpleNamespace(is_draft=True), attn_metadata=meta)
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
            ).repeat_interleave(2, dim=1)
            values = (
                v_cache[pages, offsets] * v_scale[pages, offsets]
            ).repeat_interleave(2, dim=1)
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
        max_tokens * 2 * 2 * 4 * 8,
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


@pytest.fixture
def draft_config(monkeypatch):
    # Synthetic acceptance is unrelated to workspace validation and imports
    # AITER kernels. Keep configuration tests runnable on CPU workers.
    monkeypatch.setattr(
        SpeculativeConfig, "_resolve_synthetic_acceptance", lambda _: None
    )

    def build(**kwargs):
        values = {
            "method": "eagle3",
            "num_speculative_tokens": 3,
            "draft_model_hf_config": PretrainedConfig(
                architectures=["LlamaForCausalLMEagle3"]
            ),
        }
        values.update(kwargs)
        return SpeculativeConfig(**values)

    return build


def test_workspace_config_is_opt_in(draft_config):
    assert draft_config().draft_prefill_kv_workspace_mb == 0
    spec = draft_config(draft_prefill_kv_workspace_mb=8192)
    assert spec.draft_prefill_kv_workspace_mb == 8192
    assert spec.draft_model_hf_config.architectures == ["Eagle3LlamaModel"]


@pytest.mark.parametrize("budget", [-1, 0.5, True, "8192", None])
def test_workspace_config_rejects_invalid_budgets(draft_config, budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        draft_config(draft_prefill_kv_workspace_mb=budget)


@pytest.mark.parametrize("method", [None, "mtp", "dspark"])
def test_workspace_requires_eagle3(draft_config, method):
    with pytest.raises(ValueError, match="requires method='eagle3'"):
        draft_config(method=method, draft_prefill_kv_workspace_mb=8192)


def test_workspace_rejects_mla_draft(draft_config):
    with pytest.raises(ValueError, match="requires an MHA draft"):
        draft_config(
            draft_prefill_kv_workspace_mb=8192,
            draft_model_hf_config=PretrainedConfig(
                architectures=["Eagle3DeepseekV2ForCausalLM"], kv_lora_rank=512
            ),
        )


def test_workspace_rejects_dspark_routed_checkpoint(draft_config):
    with pytest.raises(ValueError, match="does not support DSpark"):
        draft_config(
            draft_prefill_kv_workspace_mb=8192,
            draft_model_hf_config=PretrainedConfig(
                architectures=["Eagle3LlamaModel"], dspark_with_draft=True
            ),
        )


def test_workspace_rejects_sharded_prefill_before_loading_model(draft_config):
    spec = draft_config(draft_prefill_kv_workspace_mb=8192)
    with pytest.raises(ValueError, match="requires PCP=1"):
        Config(model="unused", speculative_config=spec, prefill_context_parallel_size=2)
