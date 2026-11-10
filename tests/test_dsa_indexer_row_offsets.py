# SPDX-License-Identifier: MIT
"""Exercise the device row-prefix reduction, including all four row tiles."""

from functools import lru_cache

import pytest

pytest.importorskip("flydsl")


@lru_cache(None)
def _offset_kernel(rows):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from flydsl.expr import gpu
    from flydsl.expr.typing import Int64, T

    from atom.mono.device.ops import rsrc
    from atom.mono.dsa_indexer import select

    s = rows

    @fx.struct
    class Smem:
        rows: fx.Array[fx.Int32, select.ROW_WORDS * s, 16]
        sel: fx.Array[fx.Int32, select.SEL_ROWS + s, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def offsets(counts: Int64, out: Int64):
        tid, row = fx.thread_idx.x, fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        if tid < rows:
            n = fx.Int32(bo.buffer_load(rsrc(counts), tid, vec_width=1, dtype=T.i32))
            fx.ptr_store(n, lds.rows.ptr + select.ROW_WORDS * tid)
            fx.ptr_store(n, lds.rows.ptr + select.ROW_WORDS * tid + 1)
            fx.ptr_store(
                (tid % 3 == 0).select(fx.Int32(1), fx.Int32(0)),
                lds.rows.ptr + select.ROW_WORDS * tid + 2,
            )
        gpu.barrier()

        def poll(requests):
            # Long rows publish twice their local metadata count, ensuring
            # the reduction consumes the mailbox result, not the metadata.
            n = bo.buffer_load(rsrc(counts), requests[0][1], vec_width=1, dtype=T.i32)
            return [(fx.Int32(n) * 2,)]

        offset = select.rows_before(
            {
                "S": rows,
                "tid": tid,
                "lane": tid % 64,
                "own_rows": False,
                "rows": lds.rows.ptr,
                "sel": lds.sel.ptr,
                "cnt": None,
                "poll": poll,
            },
            row,
        )
        # Every wave must observe the same prefix, not just the first one.
        if tid % 64 == 0:
            bo.buffer_store(offset, rsrc(out), row * 8 + tid // 64)

    @flyc.jit
    def launch(counts: Int64, out: Int64):
        offsets(counts, out).launch(grid=(rows,), block=(512,))

    return launch


@pytest.mark.parametrize("rows", [64, 65, 128, 129, 252, 256])
def test_row_offsets_match_exclusive_sum_on_every_wave(rows):
    import torch

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires an AMD GPU and FlyDSL")
    launch = _offset_kernel(rows)
    torch.manual_seed(431)
    patterns = [
        torch.zeros(rows, dtype=torch.int32, device="cuda"),
        torch.ones(rows, dtype=torch.int32, device="cuda"),
        torch.randint(0, 1025, (rows,), dtype=torch.int32, device="cuda"),
    ]
    for counts in patterns:
        published = counts.clone()
        published[::3] *= 2
        expected = published.cumsum(0) - published
        out = torch.full((rows, 8), -1, device="cuda", dtype=torch.int32)
        launch(counts.data_ptr(), out.data_ptr())
        assert torch.equal(out.cpu(), expected.cpu()[:, None].expand(rows, 8))
