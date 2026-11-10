# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Short/long verify transitions keep score and segment tables disjoint."""

from dataclasses import replace
from functools import lru_cache

import pytest

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer.kernel import IxBuild
from atom.mono.dsa_indexer.plan import IndexerShape


def _key(batch=4, qlen=8, world=8, q_lora=2048):
    shape = IndexerShape(32, 128, 2048, q_lora, "bf16", "fp8", 128, 64)
    return IxBuild(shape, batch * qlen, world, 8192, replicated=True, qlen=qlen)


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
        ctx: Int64,
        bids: Int64,
        out: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL stream sentinel.
    ):
        planner(ctx, bids, out, rank).launch(grid=(1,), block=(512,), stream=stream)

    return launch


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
            launch,
            ctx.data_ptr(),
            bids.data_ptr(),
            out.data_ptr(),
            0,
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
            kinds = [result[3 * r : 3 * r + 3] for r in range(rows)]
            plan = result[3 * rows :]
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
                chunks = (n + select.CHUNK - 1) // select.CHUNK if flags & 1 else 0
                lg = plan[select.PLAN_SEG_LG]
                first += (chunks + (1 << lg) - 1) >> lg
            assert plan[select.PLAN_SEG_TOTAL] == first
    finally:
        graph.reset()
