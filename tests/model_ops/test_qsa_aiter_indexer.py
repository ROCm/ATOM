# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER selection preserves ATOM cache ownership, SP groups and YaRN axes."""

import math
from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.qwen4_exp.ops import aiter_indexer, qsa


def _inputs(device, queries=(8, 1), prefixes=(0, 2999), padding=2):
    lengths = tuple(q + p for q, p in zip(queries, prefixes))
    pages = -(-max(lengths) // 64)
    table = torch.arange(
        1, 1 + len(lengths) * pages, dtype=torch.int32, device=device
    ).view(len(lengths), pages)
    table = table.flip(1).contiguous()
    cache = torch.randn(
        (1 + len(lengths) * pages, 16, 1, 128), dtype=torch.bfloat16, device=device
    )
    requests = torch.cat(
        [
            torch.full((n,), r, dtype=torch.int32, device=device)
            for r, n in enumerate(queries)
        ]
        + [torch.full((padding,), -1, dtype=torch.int32, device=device)]
    )
    positions = torch.cat(
        [
            torch.arange(p, p + n, dtype=torch.int64, device=device)
            for n, p in zip(queries, prefixes)
        ]
        + [torch.full((padding,), -1, dtype=torch.int64, device=device)]
    )
    return {
        "q": torch.randn(
            (positions.numel(), 4, 128), dtype=torch.bfloat16, device=device
        ),
        "compressed_k_cache": cache,
        "page_table": table,
        "token_to_request": requests,
        "query_positions": positions,
        "context_lens": torch.tensor(lengths, dtype=torch.int32, device=device),
        "token_topk": 2048,
        "max_seq_len": max(lengths),
        "use_aiter_indexer": True,
    }


def _check(inputs, indices):
    assert indices.shape == (inputs["q"].shape[0], 2051)
    assert indices.dtype == torch.int32
    for row, request in enumerate(inputs["token_to_request"].cpu().tolist()):
        if request < 0:
            assert bool((indices[row] == -1).all())
            continue
        visible = max(
            0,
            min(
                int(inputs["query_positions"][row]) + 1,
                int(inputs["context_lens"][request]),
            ),
        )
        count, chosen = visible // 4, min(512, visible // 4)
        groups = indices[row, : chosen * 4 : 4].long() // 4
        assert groups.unique().numel() == chosen
        assert bool(((groups >= 0) & (groups < count)).all())
        expected_tokens = (
            groups[:, None] * 4 + torch.arange(4, device=indices.device)
        ).flatten()
        assert torch.equal(indices[row, : chosen * 4], expected_tokens)
        tail = torch.arange(count * 4, visible, device=indices.device)
        assert torch.equal(indices[row, chosen * 4 : chosen * 4 + tail.numel()], tail)
        assert bool((indices[row, chosen * 4 + tail.numel() :] == -1).all())
        if count > 512:
            keys = (
                inputs["compressed_k_cache"][inputs["page_table"][request].long()]
                .view(-1, 128)[:count]
                .double()
            )
            query = inputs["q"][row].double()
            scores = (query @ keys.T).relu().sum(0) / math.sqrt(128)
            selected = torch.zeros(count, dtype=torch.bool, device=indices.device)
            selected[groups] = True
            gap = scores[~selected].max() - scores[selected].min()
            assert float(gap / scores.abs().max().clamp_min(1e-30)) <= 1e-5


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    if not aiter_indexer._available():
        pytest.skip("requires gfx942 and AITER with the migrated QSA selector")
    return torch.device("cuda", torch.cuda.current_device())


def _forbid_fallback(monkeypatch):
    from aiter.ops import qsa as aiter_qsa

    def fail(*args, **kwargs):
        raise AssertionError("Supported case unexpectedly used the old QSA scorer")

    monkeypatch.setattr(qsa, "qsa_paged_mqa_logits", fail)
    monkeypatch.setattr(aiter_qsa, "qsa_paged_mqa_logits", fail)


def test_decode_selection_and_graph(device, monkeypatch):
    inputs = _inputs(device)
    _forbid_fallback(monkeypatch)
    before = inputs["compressed_k_cache"].clone()
    warm = qsa.qsa_select_paged_tokens(**inputs)
    _check(inputs, warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = qsa.qsa_select_paged_tokens(**inputs)
    inputs["q"].neg_()
    inputs["query_positions"].sub_(1)
    graph.replay()
    _check(inputs, out)
    assert torch.equal(inputs["compressed_k_cache"], before)


def test_groups_exclude_causal_tail(device, monkeypatch):
    inputs = _inputs(device, padding=0)
    inputs.update(query_lens=(8, 1), prefix_lens=(0, 2999))
    _forbid_fallback(monkeypatch)
    groups, width = qsa.qsa_select_paged_groups(**inputs)
    visible = (inputs["query_positions"] + 1).clamp_min(0)
    counts = (visible // 4).clamp_max(512)
    assert width == 2051
    assert torch.equal((groups >= 0).sum(1), counts)
    selected = qsa.qsa_expand_block_indices(
        groups,
        inputs["query_positions"],
        inputs["context_lens"],
        inputs["token_to_request"],
        4,
        2048,
        output_width=width,
    )
    _check(inputs, selected)


@pytest.mark.parametrize(
    "start,expected",
    [
        (5, ((2, 0, 4), (2053, 16, 4))),
        (13, ((0, 0, 3), (2055, 16, 10))),
        (18, ((0, 0, 0), (2055, 16, 13))),
    ],
)
def test_packed_lengths_clip_sp_rows(start, expected):
    metadata = SimpleNamespace(
        indexer_query_lens=(7, 0, 9), indexer_prefix_lens=(2048, 16, 4)
    )
    rows = SimpleNamespace(start=start, chunk=6, num_tokens=16)
    assert aiter_indexer.packed_lengths(metadata, rows) == expected


def test_prefill_shards_match_full(device, monkeypatch):
    queries, prefixes = (129, 0, 301), (2048, 16, 7001)
    inputs = _inputs(device, queries, prefixes, padding=0)
    inputs.update(query_lens=queries, prefix_lens=prefixes)
    _forbid_fallback(monkeypatch)
    full_groups, width = qsa.qsa_select_paged_groups(**inputs)
    metadata = SimpleNamespace(indexer_query_lens=queries, indexer_prefix_lens=prefixes)
    share = -(-inputs["q"].shape[0] // 3)
    outputs = []
    for rank in range(3):
        frame = SimpleNamespace(
            start=rank * share, chunk=share, num_tokens=inputs["q"].shape[0]
        )
        local = slice(frame.start, min(frame.start + share, frame.num_tokens))
        args = dict(inputs)
        for name in ("q", "token_to_request", "query_positions"):
            args[name] = inputs[name][local]
        args["query_lens"], args["prefix_lens"] = aiter_indexer.packed_lengths(
            metadata, frame
        )
        groups, local_width = qsa.qsa_select_paged_groups(**args)
        assert local_width == width
        outputs.append(groups)
    assert torch.equal(torch.cat(outputs), full_groups)


def test_compile_keeps_aiter_operator(device, monkeypatch):
    inputs = _inputs(device, queries=(1, 1), prefixes=(2999, 2199), padding=0)
    _forbid_fallback(monkeypatch)
    expected = qsa.qsa_select_paged_tokens(**inputs)
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return torch._inductor.compile(graph, example_inputs)

    def run(q, cache, table, requests, positions, lengths):
        return qsa.qsa_select_paged_tokens(
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            2048,
            max_seq_len=3000,
            use_aiter_indexer=True,
        )

    compiled = torch.compile(run, backend=backend, fullgraph=True)
    actual = compiled(
        *(
            inputs[name]
            for name in (
                "q",
                "compressed_k_cache",
                "page_table",
                "token_to_request",
                "query_positions",
                "context_lens",
            )
        )
    )
    assert torch.equal(actual, expected)
    assert any(
        "atom_qsa_indexer_select" in str(node.target)
        for graph in graphs
        for node in graph.graph.nodes
    )


def test_yarn_keeps_mrope_axes(device):
    from atom.model_ops.qwen4_exp.qsa_attention import build_qwen4_exp_rope

    config = SimpleNamespace(
        head_dim=256,
        max_position_embeddings=32768,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    with torch.device(device):
        rope = build_qwen4_exp_rope(config, 128)
    assert tuple(rope.mrope_section) == (11, 11, 10)
    assert rope.mrope_interleaved and rope.mscale > 1
    query = torch.randn((7, 4, 128), dtype=torch.bfloat16, device=device)
    positions = torch.stack(
        [torch.arange(n, n + 7, device=device) for n in (12000, 14000, 16000)]
    )
    actual, _ = qsa.qsa_apply_mrope(rope, positions, query)
    pairs = torch.arange(32, device=device)
    axes = torch.where(
        (pairs % 3 == 2) & (pairs < 30), 2, torch.where(pairs % 3 == 1, 1, 0)
    )
    selected_positions = positions.T[:, axes]
    cos = (
        rope.cos_cache.view(-1, 32)[selected_positions, pairs]
        .to(torch.bfloat16)
        .float()[:, None]
    )
    sin = (
        rope.sin_cache.view(-1, 32)[selected_positions, pairs]
        .to(torch.bfloat16)
        .float()[:, None]
    )
    left, right = query[:, :, :32].float(), query[:, :, 32:64].float()
    own_l, own_r = (left * cos).to(torch.bfloat16), (right * cos).to(torch.bfloat16)
    other_l, other_r = (right * sin).to(torch.bfloat16), (left * sin).to(torch.bfloat16)
    expected = torch.cat(
        (
            (own_l.float() - other_l.float()).to(torch.bfloat16),
            (own_r.float() + other_r.float()).to(torch.bfloat16),
            query[:, :, 64:],
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_older_aiter_keeps_native_selector(device, monkeypatch):
    inputs = _inputs(device)
    native_inputs = dict(inputs, use_aiter_indexer=False)
    expected = qsa.qsa_select_paged_tokens(**native_inputs)
    monkeypatch.setattr(aiter_indexer, "_selector", lambda: None)
    actual = qsa.qsa_select_paged_tokens(**inputs)
    assert torch.equal(actual, expected)


def test_other_page_size_keeps_native_selector(device, monkeypatch):
    inputs = _inputs(device)
    inputs["compressed_k_cache"] = inputs["compressed_k_cache"].view(-1, 4, 1, 128)
    table = inputs["page_table"]
    inputs["page_table"] = (
        (table[:, :, None] * 4 + torch.arange(4, device=device))
        .reshape(table.shape[0], -1)
        .to(torch.int32)
    )
    expected = qsa.qsa_select_paged_tokens(**dict(inputs, use_aiter_indexer=False))

    def unexpected():
        raise AssertionError("Unsupported page size must not probe the AITER backend")

    monkeypatch.setattr(aiter_indexer, "_available", unexpected)
    actual = qsa.qsa_select_paged_tokens(**inputs)
    assert torch.equal(actual, expected)


def test_plugin_indexer_lengths_do_not_need_attention_plan(monkeypatch):
    from atom.plugin.sglang.qwen4_exp_bridge import _packed_prefill_lengths

    monkeypatch.setenv("ATOM_QSA_FLYDSL_ATTN", "0")
    batch = SimpleNamespace(extend_seq_lens_cpu=(3, 6, 1), seq_lens_cpu=(8, 6, 3))
    assert _packed_prefill_lengths(batch, 3, 10) == ((3, 6, 1), (8, 6, 3))
    assert _packed_prefill_lengths(batch, 3, 12) is None


def test_plugin_refuses_device_lengths(device, monkeypatch):
    from atom.plugin.sglang.qwen4_exp_bridge import _packed_prefill_lengths

    batch = SimpleNamespace(
        extend_seq_lens_cpu=torch.tensor([3], device=device),
        seq_lens_cpu=torch.tensor([8], device=device),
    )

    def no_iteration(self):
        raise AssertionError("Device lengths must not be iterated on the CPU")

    monkeypatch.setattr(torch.Tensor, "__iter__", no_iteration)
    assert _packed_prefill_lengths(batch, 1, 3) is None
