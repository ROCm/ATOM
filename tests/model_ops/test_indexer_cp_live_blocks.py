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


@needs_gpu
@pytest.mark.parametrize("batch,context", [(1, 1000), (3, 9000), (16, 40000)])
@pytest.mark.parametrize("rank", [0, 3])
def test_wide_launch_matches_real_length(batch, context, rank):
    q, cache, table, lengths = _inputs(batch, context, seed=batch * 7 + rank)
    real_len = int(lengths.max())
    narrow = _scores(q, cache, table, lengths, real_len, rank)
    wide = _scores(q, cache, table, lengths, WIDE, rank)
    live = narrow.shape[2]
    assert torch.equal(wide[..., :live], narrow)
    assert torch.all(wide[..., live:] == float("-inf"))
    keys = [
        ex.local_candidate_keys(
            s, lengths, TOPK, rank, WORLD, QLEN, triton.cdiv(n, 128), 0, 1
        )
        for s, n in ((narrow, real_len), (wide, WIDE))
    ]
    assert torch.equal(*keys)


@needs_gpu
@pytest.mark.parametrize("batch,context", [(1, 1000), (3, 9000), (16, 40000)])
@pytest.mark.parametrize("rank", [0, 3])
def test_unfilled_dead_slots_do_not_reach_the_selection(
    monkeypatch, batch, context, rank
):
    """``fill_dead_slots=False`` leaves the slots past a request's live blocks
    unwritten. They are poisoned here with a score above every real one (the
    scorer's buffer comes back pre-filled), so a reader that stepped past the
    live blocks would select them: the shard's keys must equal the filled
    run's, bit for bit."""
    q, cache, table, lengths = _inputs(batch, context, seed=batch * 11 + rank)
    blocks = WIDE // 128
    filled = _scores(q, cache, table, lengths, WIDE, rank)

    def poisoned(shape, **kw):
        return torch.full(shape, 1e37, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(cp.torch, "empty", poisoned)
        unfilled = cp.indexer_context_scores(
            q, cache, table[:, :blocks], lengths, WIDE, rank, WORLD, QLEN,
            128**-0.5, fill_dead_slots=False,
        )  # fmt: skip
    # The poison is really there, past the longest request's blocks.
    live = triton.cdiv(triton.cdiv(int(lengths.max()), 128), WORLD)
    assert torch.all(unfilled[..., live:] == 1e37)
    assert torch.all(filled[..., live:] == float("-inf"))
    keys = [
        ex.local_candidate_keys(s, lengths, TOPK, rank, WORLD, QLEN, blocks, 0, 1)
        for s in (filled, unfilled)
    ]
    assert torch.equal(keys[0], keys[1])


@needs_gpu
@pytest.mark.parametrize("programs", [1, 7, 64])
def test_scores_do_not_depend_on_the_split(monkeypatch, programs):
    q, cache, table, lengths = _inputs(8, 30000, seed=11)
    expected = _scores(q, cache, table, lengths, WIDE, 1)
    monkeypatch.setattr(cp, "_context_score_programs", lambda *args: programs)
    assert torch.equal(_scores(q, cache, table, lengths, WIDE, 1), expected)


@needs_gpu
def test_graph_replay_matches_eager():
    q, cache, table, lengths = _inputs(4, 20000, seed=5)
    _scores(q, cache, table, lengths, WIDE, 2)  # compile outside the capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = _scores(q, cache, table, lengths, WIDE, 2)
        for length in (QLEN, 5000, 20000):  # every slot is rewritten
            q.copy_(torch.randn_like(q))
            lengths.fill_(length)
            graph.replay()
            assert torch.equal(replayed, _scores(q, cache, table, lengths, WIDE, 2))
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
    assert torch.equal(_scores(q, cache, table, lengths, WIDE, 2), expected)
