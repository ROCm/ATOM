# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-launch QSA selection while the full context fits the sparse budget.

Stable top-k returns every visible compressed group in group order in this
case. Expanding those groups therefore produces the logical tokens 0..position,
followed by padding. The index cache is still updated by the caller for later
steps; only scoring, top-k, and expansion are skipped here.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64, Stream, T

from atom.mono.device.ops import rsrc

THREADS = 256
_CURRENT_STREAM = Stream(None)


@functools.cache
def _build(rows: int, width: int):
    slots = rows * width

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def select_kernel(
        positions: Int64,
        requests: Int64,
        contexts: Int64,
        output: Int64,
        num_requests: Int32,
    ):
        index = gpu.block_idx.x * THREADS + gpu.thread_idx.x
        if index < slots:
            row, column = index // width, index % width
            request = fx.Int32(
                bo.buffer_load(rsrc(requests), row, vec_width=1, dtype=T.i32)
            )
            valid_request = (request >= 0) & (request < num_requests)
            safe_request = fx.min(fx.max(request, 0), num_requests - 1)
            context = fx.Int32(
                bo.buffer_load(rsrc(contexts), safe_request, vec_width=1, dtype=T.i32)
            )
            # The served position tensor is int64. The shortcut applies only
            # below the 2048-token budget, so a nonzero high word is invalid.
            position = fx.Int32(
                bo.buffer_load(rsrc(positions), 2 * row, vec_width=1, dtype=T.i32)
            )
            high = fx.Int32(
                bo.buffer_load(rsrc(positions), 2 * row + 1, vec_width=1, dtype=T.i32)
            )
            length = (
                valid_request & (high == 0) & (position >= 0) & (position < context)
            ).select(position + 1, fx.Int32(0))
            bo.buffer_store(
                (column < length).select(column, fx.Int32(-1)), rsrc(output), index
            )

    @flyc.jit
    def launch(
        positions: Int64,
        requests: Int64,
        contexts: Int64,
        output: Int64,
        num_requests: Int32,
        stream: Stream = _CURRENT_STREAM,
    ):
        select_kernel(positions, requests, contexts, output, num_requests).launch(
            grid=((slots + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch.func.__name__ = f"qwen38_qsa_short_r{rows}_w{width}"
    return launch


def select_all_visible(
    positions: torch.Tensor,
    requests: torch.Tensor,
    contexts: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Fill ``out`` with every visible logical token, then -1 padding."""
    rows, width = out.shape
    _build(rows, width)(
        positions.data_ptr(),
        requests.data_ptr(),
        contexts.data_ptr(),
        out.data_ptr(),
        contexts.numel(),
        stream=torch.cuda.current_stream(out.device),
    )
    return out
