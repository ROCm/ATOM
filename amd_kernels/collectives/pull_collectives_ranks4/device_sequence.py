# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-safe form of kernel.py's fence and persistent pull.

The stage reads counter c and writes slot c%2. The last fence workgroup
increments the local counter and publishes c+1. The pull reads c+1, waits
for every source, then reads slot c%2. A rank cannot reuse that slot until
all peers have completed the intervening pull. One stream per communicator.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from .kernel import (WORLD, NT, UNROLL, STEP, SYS, FENCE_WGS, PER_PEER,
                     PULL_WGS, _flag_add, _flag_read, _sel4)

def compile_device_fence_publish():
    """Fence + publish in one launch: every workgroup releases its XCD's L2 (the
    producers' dirty lines reach memory), takes a ticket on the counter word next
    to the flag; the last arriver resets the counter and stores ``seq`` into the
    flag. Run on the stream that produced the data (the comm stream, after its
    wait for the compute stream)."""

    @flyc.kernel(
        name="pull_collectives_ranks4_device_fence_publish", known_block_size=[64, 1, 1]
    )
    def kernel(arg_flag: fx.Int64, counter: fx.Tensor):
        i32_seq = fx.Int32(counter[0]) + 1
        _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="")
        if fx.thread_idx.x == 0:
            t = _flag_add(arg_flag + 4, 1)
            if t == FENCE_WGS - 1:
                _flag_add(arg_flag + 4, -FENCE_WGS)
                fr = buffer_ops.create_buffer_resource_from_addr(
                    arg_flag, num_records_bytes=64
                )
                counter[0] = i32_seq
                buffer_ops.buffer_store(i32_seq, fr, 0, cache_modifier=SYS)

    @flyc.jit
    def launch(arg_flag: fx.Int64, counter: fx.Tensor, stream: fx.Stream):
        kernel(arg_flag, counter).launch(
            grid=(FENCE_WGS, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.kernel_name = "pull_collectives_ranks4_device_fence_publish"
    return launch


def compile_device_pull(pull_wgs: int = PULL_WGS):
    assert pull_wgs > 0 and pull_wgs % WORLD == 0
    name = f"pull_collectives_ranks4_device_pull_persistent{pull_wgs}"
    """dst[p * block, (p + 1) * block) <- bytes [src_off, src_off + block) of
    rank p's region, p = 0..3 (the local rank's block copied too). Workgroup w
    serves rank w % 4: lane 0 spins until that rank's flag reaches ``seq`` (its
    own rank needs no wait), the workgroup acquires, then loops over the rank's
    chunks w // 4, w // 4 + PULL_WGS // 4, ..."""

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_dst: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_block: fx.Int64,  # bytes per rank block (multiple of 16)
        i64_src_off: fx.Int64,  # byte offset of the block inside a rank's region
        counter: fx.Tensor,
        i64_stride: fx.Int64,
        i32_me: fx.Int32,
    ):
        i32_seq = fx.Int32(counter[0])
        i64_src_off = i64_src_off + fx.Int64((i32_seq - 1) & 1) * i64_stride
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        p = bid % WORLD
        if (tx == 0) & (p != i32_me):
            fa = _sel4(p, arg_f0, arg_f1, arg_f2, arg_f3)
            f = _flag_read(fa)
            while f < i32_seq:
                f = _flag_read(fa)
        fx.gpu.barrier()
        _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
        src_a = _sel4(p, arg_s0, arg_s1, arg_s2, arg_s3) + i64_src_off
        src = buffer_ops.create_buffer_resource_from_addr(
            src_a, num_records_bytes=i64_block
        )
        dst = buffer_ops.create_buffer_resource_from_addr(
            arg_dst + fx.Int64(p) * i64_block, num_records_bytes=i64_block
        )
        n_dw = fx.Int32(i64_block // 4)
        per_block = (n_dw + (PER_PEER - 1)) // PER_PEER
        per_block = (per_block + (STEP - 1)) // STEP * STEP
        pb = bid // WORLD
        while pb < PER_PEER:
            start = pb * per_block
            end = start + per_block
            off = start + tx * 4
            while off < end:
                vals = []
                for u in range_constexpr(UNROLL):
                    o = off + u * NT * 4
                    vals.append(
                        Vec(
                            buffer_ops.buffer_load(
                                src,
                                o,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=o < n_dw,
                                cache_modifier=SYS,
                            )
                        )
                    )
                for u in range_constexpr(UNROLL):
                    o = off + u * NT * 4
                    buffer_ops.buffer_store(vals[u], dst, o, mask=o < n_dw)
                off = off + STEP
            pb = pb + (pull_wgs // WORLD)

    @flyc.jit
    def launch(
        arg_dst: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        i64_block: fx.Int64,
        i64_src_off: fx.Int64,
        counter: fx.Tensor,
        i64_stride: fx.Int64,
        i32_me: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_dst,
            arg_s0,
            arg_s1,
            arg_s2,
            arg_s3,
            arg_f0,
            arg_f1,
            arg_f2,
            arg_f3,
            i64_block,
            i64_src_off,
            counter,
            i64_stride,
            i32_me,
        ).launch(grid=(pull_wgs, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
