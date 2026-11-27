# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Rows read from device memory into LDS, every row set's loads issued before
any lands (``rows_to_lds_together``): ``atom.mono.device.ops.rows_to_lds`` of
several sources in one round trip."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm

from atom.mono.device.ops import ld_dev, traced
from atom.mono.plan.execution import THREADS


def rows_to_lds_together(tid, specs):
    """``rows_to_lds`` of each (src, rows, row_words, dst, dst_row) in
    ``specs``, every load issued before any lands in LDS: one round trip,
    where ``rows_to_lds`` in turn waits for one's stores before the next's
    loads. A side-effecting no-op after the loads keeps LLVM from sinking one
    past the stores before it."""
    loads = []
    for src, rows, row_words, dst, dst_row in specs:
        assert row_words % 4 == 0, row_words
        n = rows * row_words // 4
        for i in range(-(-n // THREADS)):
            e = tid + THREADS * i
            w = fx.Vector(ld_dev(src, fx.min(e, n - 1) * 4, 4))
            loads.append((e, n, w, row_words, dst, dst_row))
    llvm.InlineAsmOp(None, [], "", "", has_side_effects=True)
    for e, n, w, row_words, dst, dst_row in loads:
        _lds_put_if(e, n, w, dst + (e * 4 // row_words * dst_row + e * 4 % row_words))


@traced
def _lds_put_if(e, n, w, at):
    if e < n:
        fx.ptr_store(w, at)
