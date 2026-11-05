# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Short replicated verify ownership and Q projection boundaries."""

from dataclasses import replace
from functools import lru_cache

import pytest

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer.kernel import IxBuild, plain_layout, unfit
from atom.mono.dsa_indexer.plan import IndexerShape


def _key(batch=4, qlen=8, world=8, q_lora=2048):
    shape = IndexerShape(32, 128, 2048, q_lora, "bf16", "fp8", 128, 64)
    return IxBuild(shape, batch * qlen, world, 8192, replicated=True, qlen=qlen)


@pytest.mark.parametrize("qlen", [6, 8])
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("q_lora", [1024, 2048])
def test_short_verify_owns_one_request_per_active_rank(qlen, world, q_lora):
    for batch in range(1, world + 1):
        key = _key(batch, qlen, world, q_lora)
        assert not unfit(key)
        assert key.short_owned_verify
        assert key.own_rows
        assert key.row_keys == key.max_ctx
        # Global row numbering remains the ABI, even for rank-local Q work.
        assert plain_layout(key)["iqb"][1] == key.tokens * key.shape.qrows * 2


@pytest.mark.parametrize(
    "changes",
    [
        {"max_ctx": 262145},
        {"max_ctx": 0},
        {"replicated": False},
        {"world": 1},
        {"qlen": 1},
        {"qlen": 2},
        {"qlen": 3},
        {"qlen": 16},
        {"tokens": 64, "qlen": 6},
        {"tokens": 72},
    ],
)
def test_short_verify_specialization_does_not_escape_its_domain(changes):
    assert not replace(_key(), **changes).short_owned_verify


def test_long_window_can_use_runtime_short_verify_and_plain_decode_is_unchanged():
    for batch in (1, 4, 8):
        long_key = replace(_key(batch), max_ctx=163840)
        assert long_key.own_rows
        assert long_key.short_owned_verify
        decode = replace(_key(batch), tokens=batch, qlen=1)
        assert decode.own_rows == (batch >= decode.world)
        assert not decode.short_owned_verify


def test_short_verify_can_still_admit_multiple_requests_per_rank():
    key = _key(batch=10, qlen=6)
    assert not unfit(key)
    assert key.own_rows
    assert key.short_owned_verify


@pytest.mark.parametrize("maximum", [8192, 16384, 65536, 163840, 262144])
def test_low_batch_score_storage_covers_short_owned_and_long_split_rows(maximum):
    key = replace(_key(batch=4), max_ctx=maximum)
    assert key.row_keys == max(key.lmax, 8192)
    full_batch = replace(key, tokens=64)
    assert full_batch.row_keys == maximum


@lru_cache(None)
def _planner(batch, qlen, world, maximum, short_score=False):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int32, Int64

    from atom.mono.device.ops import rsrc
    from atom.mono.dsa_indexer import select

    key = replace(_key(batch, qlen, world), max_ctx=maximum)
    rows = key.tokens
    s = rows

    @fx.struct
    class Smem:
        rows: fx.Array[fx.Int32, 3 * s, 16]
        plan: fx.Array[fx.Int32, 4 + 2 * s, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def planner(ctx: Int64, bids: Int64, out: Int64, rank: Int32):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        select.step_table(
            {
                "S": rows,
                "shape": key.shape,
                "tid": tid,
                "lane": tid % 64,
                "rank": rank,
                "world": world,
                "inter": 128,
                "own_rows": key.own_rows,
                "short_owned_verify": key.short_owned_verify,
                "short_score": short_score,
                "max_ctx": maximum,
                "qlen": qlen,
                "rg": select.piece_rows(qlen),
                "ppr": select.pieces_a_request(qlen),
                "args": {"ctx": ctx, "bids": bids},
                "rows": lds.rows.ptr,
                "plan": lds.plan.ptr,
            }
        )
        for i in range(tid, 3 * rows, 512):
            bo.buffer_store(fx.ptr_load(lds.rows.ptr + i), rsrc(out), i)
        if short_score:
            for i in range(tid, 4 + 2 * rows, 512):
                bo.buffer_store(fx.ptr_load(lds.plan.ptr + i), rsrc(out), 3 * rows + i)

    @flyc.jit
    def launch(
        ctx: Int64, bids: Int64, out: Int64, rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL stream sentinel.
    ):
        planner(ctx, bids, out, rank).launch(grid=(1,), block=(512,), stream=stream)

    return launch


@pytest.mark.parametrize(
    "batch,qlen,world", [(1, 6, 8), (4, 8, 8), (8, 6, 8), (4, 8, 4)]
)
@pytest.mark.parametrize("maximum", [8192, 163840])
def test_gpu_runtime_ownership_matches_contexts(batch, qlen, world, maximum):
    import torch

    from atom.mono.dsa_indexer.plan import local_len

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires an AMD GPU and FlyDSL")
    rows = batch * qlen
    launch = _planner(batch, qlen, world, maximum)
    bids = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(qlen)
    out = torch.empty(3 * rows, dtype=torch.int32, device="cuda")
    patterns = [[length] * batch for length in (0, 2048, 8192, maximum)]
    if maximum > 8192:
        patterns += [
            [8193] * batch,
            [[0, 8192, 8193, maximum][r % 4] for r in range(batch)],
        ]
    for lengths in patterns:
        ctx = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        short = max(lengths) <= 8192
        for rank in range(world):
            expected = []
            for row in range(rows):
                request, draft = divmod(row, qlen)
                context = lengths[request] - (qlen - 1 - draft)
                split = batch < world and lengths[request] > 8192
                count = (
                    local_len(max(context, 0), rank, world, 128)
                    if split
                    else (max(context, 0) if request % world == rank else 0)
                )
                flags = (
                    int(context > 2048 and count > 0)
                    | (int(split) << 1)
                    | (int(short) << 2)
                )
                expected.extend((context, count, flags))
            launch(ctx.data_ptr(), bids.data_ptr(), out.data_ptr(), rank)
            assert out.tolist() == expected, (lengths, rank)


@pytest.mark.parametrize("qlen", [6, 8])
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_runtime_score_plan_keeps_segment_table_disjoint(qlen, batch):
    import torch
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from atom.mono.dsa_indexer import select

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires an AMD GPU and FlyDSL")
    world, maximum, rows = 8, 163840, batch * qlen
    launch = _planner(batch, qlen, world, maximum, short_score=True)
    ctx = torch.full((batch,), 8192, device="cuda", dtype=torch.int32)
    bids = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(qlen)
    out = torch.empty(3 * rows + 4 + 2 * rows, device="cuda", dtype=torch.int32)

    def call():
        _run_compiled(
            launch, ctx.data_ptr(), bids.data_ptr(), out.data_ptr(), 0,
            torch.cuda.current_stream(),
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    try:
        for length in (8192, 8193, maximum, 2048, 8192):
            ctx.fill_(length)
            graph.replay()
            result = out.cpu().tolist()
            kinds = [result[3 * r:3 * r + 3] for r in range(rows)]
            plan = result[3 * rows:]
            group = 2 if length <= 8192 else qlen
            counts = []
            for request in range(batch):
                for draft in range(group - 1, qlen, group):
                    _, n, flags = kinds[request * qlen + draft]
                    units = (n + 127) // 128 if flags & 1 else 0
                    lg = plan[select.PLAN_SCORE_LG]
                    counts.append((units + (1 << lg) - 1) >> lg)
            first = 0
            for i, count in enumerate(counts):
                assert plan[select.PLAN_FIRSTS + i] == first
                first += count
            assert plan[select.PLAN_SCORE_TOTAL] == first
            if length == 8192:
                assert first == 64 * (qlen // 2)
                assert plan[select.PLAN_SCORE_LG] == 0
            first = 0
            for row, (_, n, flags) in enumerate(kinds):
                assert plan[select.PLAN_FIRSTS + rows + row] == first
                chunks = (
                    (n + select.CHUNK - 1) // select.CHUNK if flags & 1 else 0
                )
                lg = plan[select.PLAN_SEG_LG]
                first += (chunks + (1 << lg) - 1) >> lg
            assert plan[select.PLAN_SEG_TOTAL] == first
    finally:
        graph.reset()
