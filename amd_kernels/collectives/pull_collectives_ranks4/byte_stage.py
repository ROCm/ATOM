# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""16-byte-aligned input staging into alternating IPC slots."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr.typing import Vector as Vec

NT = 256
WGS = 256  # Same grid-stride staging as the dense BF16 reduce-scatter.


def compile_byte_stage():
    @flyc.kernel(name="byte_stage_aligned16_ranks4", known_block_size=[NT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_pool: fx.Int64, counter: fx.Tensor,
               i64_stride: fx.Int64, i32_bytes: fx.Int32):
        dst = buffer_ops.create_buffer_resource_from_addr(
            arg_pool + fx.Int64(fx.Int32(counter[0]) & 1) * i64_stride,
            num_records_bytes=i32_bytes)
        src = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_bytes)
        off = fx.block_idx.x * NT * 4 + fx.thread_idx.x * 4
        while off < i32_bytes // 4:
            val = Vec(buffer_ops.buffer_load(src, off, vec_width=4, dtype=fx.Int32))
            buffer_ops.buffer_store(val, dst, off)
            off = off + WGS * NT * 4

    @flyc.jit
    def launch_byte_stage_aligned16_ranks4(arg_x: fx.Int64, arg_pool: fx.Int64, counter: fx.Tensor,
                                         i64_stride: fx.Int64, i32_bytes: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_pool, counter, i64_stride, i32_bytes).launch(
            grid=(WGS, 1, 1), block=(NT, 1, 1), stream=stream)
    launch_byte_stage_aligned16_ranks4.kernel_name = "byte_stage_aligned16_ranks4"
    return launch_byte_stage_aligned16_ranks4
