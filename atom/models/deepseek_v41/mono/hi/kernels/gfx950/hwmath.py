# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``v_rsq_f32`` / ``v_rcp_f32`` / ``v_exp_f32`` elementwise on gfx950: the hardware approximations the decode
kernels round through (``atom.mono.device.ops.hw_rsq`` / ``hw_rcp`` / ``hw_exp2``), for host code that has to reproduce
those kernels' bits (the token tables' build, ``tk.toktab``): their results are faithful, not correctly rounded, so
no CPU formula gives them.

``build(op)`` -> (symbol, launcher(src, dst, n)): f32 ``dst[i] = op(src[i])`` for i < n, a grid-stride loop of
``GRID`` x 256 threads on the current stream.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from atom.mono.device.ops import (
    hw_exp2,
    hw_rcp,
    hw_rsq,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
)

Int32, Int64, F32 = fx.Int32, fx.Int64, fx.Float32
_CURRENT_STREAM = fx.Stream(None)

OPS = {"rsq": hw_rsq, "rcp": hw_rcp, "exp2": hw_exp2}
GRID, THREADS = 1024, 256
VERSION = "v1"


@traced
def _entry(t, b, src, dst, n, fn):
    i = b * THREADS + t
    while i < n:
        x = Int32(bo.buffer_load(rsrc(src), i, vec_width=1, dtype=T.i32)).bitcast(F32)
        bo.buffer_store(F32(fn(x)).bitcast(Int32), rsrc(dst), i)
        i = i + GRID * THREADS


def build(op: str):
    """(kernel symbol, launcher(src, dst, n)) of ``op`` in ``OPS``."""
    fn = OPS[op]
    name = kernel_symbol("tk_hwmath", op=op, v=VERSION)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def hwmath(src: Int64, dst: Int64, n: Int64):
        _entry(fx.thread_idx.x, fx.block_idx.x, src, dst, Int32(n), fn)

    @launcher(name)
    def launch(src: Int64, dst: Int64, n: Int64, stream: fx.Stream = _CURRENT_STREAM):
        hwmath(src, dst, n).launch(grid=(GRID,), block=(THREADS,), stream=stream)

    return name, launch
