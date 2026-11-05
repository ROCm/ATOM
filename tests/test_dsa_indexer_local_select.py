# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact single-CTA short selection, including mutable graph inputs."""

from functools import lru_cache

import pytest

pytest.importorskip("flydsl")

from atom.mono.dsa_indexer import select
from atom.mono.dsa_indexer.plan import IndexerShape
from atom.mono.plan.execution import WAVES


def test_local_radix_reuses_q_reduction_storage_without_widening_peer_bins():
    assert select.DEFAULT_RADIX.bits == (8, 8, 8, 8)
    assert not select.DEFAULT_RADIX.local
    assert select.LOCAL_RADIX.bits == (11, 11, 10)
    assert select.LOCAL_RADIX.bins <= WAVES * 64 * 4
    with pytest.raises(ValueError):
        select.Radix((11, 11, 10))
    with pytest.raises(ValueError):
        select.Radix((12, 10, 10), local=True)


@lru_cache(None)
def _selector(page):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr.typing import Int64, T

    from atom.mono.device.ops import rsrc

    shape = IndexerShape(
        32, 128, 2048, 2048, "bf16", "fp4" if page == 64 else "fp8", page, 64
    )

    @fx.struct
    class Smem:
        red: fx.Array[fx.Int32, WAVES * 64 * 4, 16]
        sel: fx.Array[fx.Int32, select.SEL_ROWS, 16]
        scan: fx.Array[fx.Int32, 2 * WAVES, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel(logits: Int64, lengths: Int64, tables: Int64, output: Int64):
        tid, row = fx.thread_idx.x, fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        n = fx.Int32(bo.buffer_load(rsrc(lengths), row, vec_width=1, dtype=T.i32))
        # Scores are supplied by this test. Full-kernel graph tests exercise
        # their publication, peer winner exchange, and reader acknowledgement.
        c = {
            "tid": tid,
            "lane": tid % 64,
            "wave": tid // 64,
            "shape": shape,
            "world": 1,
            "rank": fx.Int32(0),
            "replicated": True,
            "qlen": 1,
            "inter": 128,
            "radix": select.DEFAULT_RADIX,
            "red": lds.red.ptr,
            "sel": lds.sel.ptr,
            "scan": lds.scan.ptr,
            "ilog": logits,
            "ilog_bytes": 8 * 8192 * 4,
            "row_keys": 8192,
            "sflag_row": 64,
            "sflag": None,
            "emitted": None,
            "slab": output,
            "prcnt": [],
            "poll": lambda _waits: [],
            "put": lambda *_args: None,
            "stamp": lambda _point: None,
            "tl_pass": 0,
            "args": {"bt": tables, "bt_ld": 8192 // page},
        }
        select.stage_short_selection(c, row, row, n)

    @flyc.jit
    def launch(
        logits: Int64, lengths: Int64, tables: Int64, output: Int64, stream: fx.Stream
    ):
        kernel(logits, lengths, tables, output).launch(
            grid=(8,), block=(512,), stream=stream
        )

    return launch


@pytest.mark.parametrize("page", [16, 64, 128])
def test_local_selection_exact_and_graph_replay(page):
    import torch
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires an AMD GPU and FlyDSL")
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("Mono targets gfx950")
    torch.manual_seed(4901)
    logits = torch.empty((8, 8192), dtype=torch.float32, device="cuda")
    lengths = torch.tensor(
        [2049, 2053, 4095, 4096, 4097, 8185, 8191, 8192],
        dtype=torch.int32,
        device="cuda",
    )
    tables = torch.randperm(8 * 8192 // page, device="cuda").int().view(8, -1)
    output = torch.full((8, 2048), -1, dtype=torch.int32, device="cuda")
    launch = _selector(page)

    def call():
        _run_compiled(
            launch,
            logits.data_ptr(),
            lengths.data_ptr(),
            tables.data_ptr(),
            output.data_ptr(),
            torch.cuda.current_stream(),
        )

    logits.normal_()
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    patterns = [
        torch.randn(8, 8192),
        torch.zeros(8, 8192),
        torch.full((8, 8192), float("nan")),
        torch.randint(-4, 5, (8, 8192)).float(),
        torch.arange(8192).float().expand(8, -1).clone(),
        (torch.arange(8192, dtype=torch.int32) + 0x3F800000)
        .view(torch.float32)
        .expand(8, -1)
        .clone(),
        torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0, 0.0]
        ).repeat(8, 1024),
    ]
    cpu_tables = tables.cpu().long()
    try:
        for pattern in patterns:
            logits.copy_(pattern)
            lengths.copy_(lengths.roll(1))
            for _ in range(3):
                graph.replay()
            actual = output.cpu().long()
            for row, n in enumerate(lengths.cpu().tolist()):
                values = pattern[row, :n]
                bits = values.view(torch.int32)
                keys = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
                keys[values.isnan()] = -(2**31)
                chosen = keys.argsort(descending=True, stable=True)[:2048]
                expected = cpu_tables[row, chosen // page] * page + chosen % page
                assert torch.equal(actual[row].sort().values, expected.sort().values)
    finally:
        graph.reset()
