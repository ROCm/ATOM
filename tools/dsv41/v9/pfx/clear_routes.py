# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""PF-CLR v3: ``compile_clear_missing_routes`` with one thread per route row.

AITER's producer kernel strides 65536 threads over every dword of every row
(~5.7e7 dwords at the C256 prefill shape). v3 reads ``valid[row]`` once and
only an invalid row writes its dwords: payload words 0, the rest 0x7F7F7F7F,
byte-identical to AITER's kernel on every validity pattern. The upstream home
is ``aiter/ops/flydsl/kernels/comm_fused_moe/gfx950/a8w4/producer.py``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr
from flydsl.expr import gpu, ptrtoint
from flydsl.expr.typing import T


def _check(row_bytes, payload_bytes):
    if not 0 < payload_bytes <= row_bytes or row_bytes % 4 or payload_bytes % 4:
        raise ValueError("route payload and row sizes must be positive whole dwords")


@functools.cache
def compile_clear_missing_routes(rows: int, row_bytes: int, payload_bytes: int):
    _check(row_bytes, payload_bytes)
    words_per_row = row_bytes // 4
    payload_words = payload_bytes // 4

    @flyc.kernel(
        name=f"comm_fused_moe_clear_missing_routes_v3_r{rows}_b{row_bytes}_p{payload_bytes}",
        known_block_size=[256, 1, 1],
    )
    def kernel(route: fx.Pointer, validity: fx.Pointer):
        output = global_typed_ptr(fx.Int64(ptrtoint(route)), T.i32)
        valid = global_typed_ptr(fx.Int64(ptrtoint(validity)), T.i32)
        index = fx.Int32(gpu.block_id("x")) * 256 + fx.Int32(gpu.thread_id("x"))
        stride = fx.Int32(gpu.grid_dim.x) * 256
        for row in range(index, fx.Int32(rows), stride):
            if fx.Int32(valid[row]) == 0:
                base = row * fx.Int32(words_per_row)
                for column in range(fx.Int32(0), fx.Int32(words_per_row), fx.Int32(1)):
                    output[base + column] = (column < fx.Int32(payload_words)).select(
                        fx.Int32(0), fx.Int32(0x7F7F7F7F)
                    )

    @flyc.jit
    def launch(route, validity, stream):
        kernel(route, validity).launch(
            grid=(max(1, (rows + 255) // 256), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
