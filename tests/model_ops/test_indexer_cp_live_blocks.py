# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 indexer-CP decode scorer and local top-k launched far wider than
the requests, as a decode graph captured for max_model_len launches them."""

import pytest
import torch

pytest.importorskip("aiter", reason="requires the AITER runtime")
triton = pytest.importorskip("triton", reason="requires Triton")

from atom.model_ops.minimax_m3 import indexer_candidate_exchange as ex
from atom.model_ops.minimax_m3 import indexer_context_parallel as cp

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

WORLD, QLEN, HEADS, TOPK = 4, 4, 4, 16
WIDE = 4096 * 128  # launch length, past every request below


def _inputs(batch, context, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    real = triton.cdiv(context, 128)
    q = torch.randn(batch * QLEN, HEADS, 128, device="cuda", generator=g)
    cache = torch.randn(batch * real, 128, 128, device="cuda", generator=g)
    table = torch.zeros(batch, WIDE // 128, device="cuda", dtype=torch.int32)
    pages = torch.randperm(batch * real, device="cuda", generator=g).int()
    table[:, :real] = pages.view(batch, real)
    lengths = [max(QLEN, context - 37 * i) for i in range(batch)]
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q.bfloat16(), cache.to(torch.float8_e4m3fn), table, lengths


def _scores(q, cache, table, lengths, max_len, rank):
    blocks = triton.cdiv(max_len, 128)
    return cp.indexer_context_scores(
        q, cache, table[:, :blocks], lengths, max_len, rank, WORLD, QLEN, 128**-0.5
    )


def _keys(scores, lengths, max_len, rank):
    return ex.local_candidate_keys(
        scores,
        lengths,
        TOPK,
        rank,
        WORLD,
        QLEN,
        triton.cdiv(max_len, 128),
        0,
        1,
    )


def _assert_live_scores_equal(actual, expected, lengths, rank):
    for request, length in enumerate(lengths.tolist()):
        blocks = triton.cdiv(length, 128)
        live = max(0, (blocks - rank + WORLD - 1) // WORLD)
        rows = slice(request * QLEN, (request + 1) * QLEN)
        assert torch.equal(actual[:, rows, :live], expected[:, rows, :live])


@needs_gpu
@pytest.mark.parametrize("batch,context", [(1, 1000), (3, 9000), (16, 40000)])
@pytest.mark.parametrize("rank", [0, 3])
def test_wide_launch_matches_real_length(batch, context, rank):
    q, cache, table, lengths = _inputs(batch, context, seed=batch * 7 + rank)
    real_len = int(lengths.max())
    narrow = _scores(q, cache, table, lengths, real_len, rank)
    wide = _scores(q, cache, table, lengths, WIDE, rank)
    _assert_live_scores_equal(wide, narrow, lengths, rank)
    assert torch.equal(
        _keys(wide, lengths, WIDE, rank),
        _keys(narrow, lengths, real_len, rank),
    )


@needs_gpu
@pytest.mark.parametrize("programs", [1, 7, 64])
def test_scores_do_not_depend_on_the_split(monkeypatch, programs):
    q, cache, table, lengths = _inputs(8, 30000, seed=11)
    expected = _scores(q, cache, table, lengths, WIDE, 1)
    monkeypatch.setattr(cp, "_context_score_programs", lambda *args: programs)
    actual = _scores(q, cache, table, lengths, WIDE, 1)
    _assert_live_scores_equal(actual, expected, lengths, 1)


@needs_gpu
def test_graph_replay_matches_eager():
    rank = 0
    q, cache, table, lengths = _inputs(4, 20000, seed=5)
    _scores(q, cache, table, lengths, WIDE, rank)  # compile outside the capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = _scores(q, cache, table, lengths, WIDE, rank)
        # Shrinking is the discriminating order: stale scores from the longer
        # replay remain in the masked tail and must not affect the live prefix.
        for length in (20000, 5000, QLEN):
            q.copy_(torch.randn_like(q))
            lengths.fill_(length)
            graph.replay()
            eager = _scores(q, cache, table, lengths, WIDE, rank)
            _assert_live_scores_equal(replayed, eager, lengths, rank)
            assert torch.equal(
                _keys(replayed, lengths, WIDE, rank),
                _keys(eager, lengths, WIDE, rank),
            )
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_grid_follows_batch_not_launch_width(batch):
    widest = cp._context_score_programs(batch, 1 << 20)
    for local_blocks in (2048, 1 << 16):
        assert cp._context_score_programs(batch, local_blocks) == widest
    for local_blocks in (1, 8, 64):
        programs = cp._context_score_programs(batch, local_blocks)
        assert 1 <= programs <= triton.cdiv(local_blocks, cp.CP_SCORE_MIN_SPAN)


@needs_gpu
@pytest.mark.parametrize("transposed", [False, True])
def test_orientations_agree(monkeypatch, transposed):
    q, cache, table, lengths = _inputs(6, 25000, seed=23)
    expected = _scores(q, cache, table, lengths, WIDE, 2)
    monkeypatch.setattr(cp, "_transposed_score", lambda: transposed)
    actual = _scores(q, cache, table, lengths, WIDE, 2)
    _assert_live_scores_equal(actual, expected, lengths, 2)
