# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense BF16 input staging for the graph-safe pull reduce-scatter."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr.typing import Vector as Vec

NT = 256
WGS = 256


def compile_bf16_stage():
    @flyc.kernel(name="bf16_stage_h6144_ranks4", known_block_size=[NT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_pool: fx.Int64, counter: fx.Tensor,
               i64_stride: fx.Int64, i32_rows: fx.Int32):
        nbytes = fx.Int64(i32_rows) * 6144 * 2
        dst = buffer_ops.create_buffer_resource_from_addr(
            arg_pool + fx.Int64(fx.Int32(counter[0]) & 1) * i64_stride, num_records_bytes=nbytes)
        src = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=nbytes)
        off = fx.block_idx.x * NT * 4 + fx.thread_idx.x * 4
        while off < i32_rows * 3072:
            val = Vec(buffer_ops.buffer_load(src, off, vec_width=4, dtype=fx.Int32))
            buffer_ops.buffer_store(val, dst, off)
            off = off + WGS * NT * 4

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_pool: fx.Int64, counter: fx.Tensor,
               i64_stride: fx.Int64, i32_rows: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_pool, counter, i64_stride, i32_rows).launch(
            grid=(WGS, 1, 1), block=(NT, 1, 1), stream=stream)
    launch.kernel_name = "bf16_stage_h6144_ranks4"
    return launch
