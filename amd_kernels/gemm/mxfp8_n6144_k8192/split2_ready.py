# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The split-2 row's ready words (``prefill_256x192_split2``): one launch zeroes all 256 words (one per tile, each on its
own 128-byte line) before a set's first GEMM; the GEMM re-arms each word it uses."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
import torch
from amd_kernels.common import buffer_ops
from flydsl.expr.typing import Vector as Vec


@functools.cache
def _clear_launcher():
    @flyc.kernel(name="flydsl_mxfp8_split2_clear_ready", known_block_size=[256, 1, 1])
    def clear(ready: fx.Tensor):
        rsrc = buffer_ops.create_buffer_resource(ready, max_size=False, num_records_bytes=256 * 32 * 4)
        zero = Vec.filled(4, 0, fx.Int32)
        for i in range_constexpr(8):
            buffer_ops.buffer_store(zero, rsrc, (fx.thread_idx.x + i * 256) * 16, offset_is_bytes=True)

    @flyc.jit
    def launch_split2_clear_ready(ready: fx.Tensor, stream: fx.Stream):
        clear(ready).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)
    return launch_split2_clear_ready


def clear(ready: torch.Tensor) -> None:
    """Zero all 256 ready words of ``ready`` (int32 [256 * 32])."""
    from amd_kernels.common.tensor_shim import _run_compiled
    _run_compiled(_clear_launcher(), ready, torch.cuda.current_stream())
