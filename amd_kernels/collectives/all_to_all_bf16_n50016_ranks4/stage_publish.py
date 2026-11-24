# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-safe owner row exchange into its final vocabulary layout.

Protocol derived from pull_collectives_ranks4: two IPC slots and an uncached
sequence flag. System stores publish payload bytes without an L2 fence. Stage and
pull use a device counter, so graph replay never reuses a host sequence.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

N = 50016
WORLD = 4
NT = 256
UNROLL = 8
STEP = NT * 4 * UNROLL
SYS = 0x11

def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _flag_add(addr_i64, v):
    """Atomic add at system scope on an uncached word; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(v).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def compile_stage_publish(stage_wgs):
    SW = stage_wgs  # 64 / 128 / 512, chosen by a measured sweep
    assert SW in (64, 128, 512)
    @flyc.kernel(name=f"all_to_all_bf16_n50016_ranks4_stage_publish_w{SW}", known_block_size=[NT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_pool: fx.Int64, counter: fx.Tensor,
               i32_ndw: fx.Int32, i64_slot_bytes: fx.Int64, arg_flag: fx.Int64):
        slot = fx.Int32(counter[0]) & 1
        src = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_ndw * 4)
        dst = buffer_ops.create_buffer_resource_from_addr(arg_pool + fx.Int64(slot) * i64_slot_bytes,
                                                        num_records_bytes=i32_ndw * 4)
        off = fx.block_idx.x * STEP + fx.thread_idx.x * 4
        while off < i32_ndw:
            vals = [Vec(buffer_ops.buffer_load(src, off + u * NT * 4, vec_width=4, dtype=fx.Int32))
                    for u in range_constexpr(UNROLL)]
            for u in range_constexpr(UNROLL):
                buffer_ops.buffer_store(vals[u], dst, off + u * NT * 4, cache_modifier=SYS)
            off = off + SW * STEP

        # All producing waves complete their system stores before lane 0
        # takes this workgroup's ticket. SYS stores avoid dirty XCD L2 lines.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if fx.thread_idx.x == 0:
            ticket = _flag_add(arg_flag + 4, 1)
            if ticket == SW - 1:
                _flag_add(arg_flag + 4, -SW)
                seq = fx.Int32(counter[0]) + 1
                counter[0] = seq
                fr = buffer_ops.create_buffer_resource_from_addr(arg_flag, num_records_bytes=64)
                buffer_ops.buffer_store(seq, fr, 0, cache_modifier=SYS)

    @flyc.jit
    def launch_all_to_all_bf16_stage_publish(arg_x: fx.Int64, arg_pool: fx.Int64, counter: fx.Tensor,
                                    i32_ndw: fx.Int32, i64_slot_bytes: fx.Int64, arg_flag: fx.Int64, stream: fx.Stream):
        kernel(arg_x, arg_pool, counter, i32_ndw, i64_slot_bytes, arg_flag).launch(
            grid=(SW, 1, 1), block=(NT, 1, 1), stream=stream)
    return launch_all_to_all_bf16_stage_publish
