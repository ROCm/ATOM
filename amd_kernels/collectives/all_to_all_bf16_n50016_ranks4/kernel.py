# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-safe owner row exchange into its final vocabulary layout.

Copied protocol from pull_collectives_ranks4: two IPC slots, an uncached
sequence flag and peer system loads. Stage publishes using system stores. Stage and
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


def _flag_read(addr_i64):
    """The flag's current value: an atomic add of 0 at system scope (never
    hoisted out of the spin loop, served by the owner's uncached memory)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(0).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def _sel4(p, a0, a1, a2, a3):
    return (p == 0).select(a0, (p == 1).select(a1, (p == 2).select(a2, a3)))



def compile_pull(pull_wgs=32):
    PW = pull_wgs  # 32 below 1024 send rows, 128 above (measured sweep).
    assert PW in (32, 128)
    @flyc.kernel(name=f"all_to_all_bf16_n50016_ranks4_pull_w{PW}", known_block_size=[NT, 1, 1])
    def kernel(arg_out: fx.Int64, arg_s0: fx.Int64, arg_s1: fx.Int64, arg_s2: fx.Int64, arg_s3: fx.Int64,
               arg_f0: fx.Int64, arg_f1: fx.Int64, arg_f2: fx.Int64, arg_f3: fx.Int64,
               counter: fx.Tensor, counts: fx.Tensor, i64_slot_bytes: fx.Int64, i32_rank: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        p = bid % WORLD
        seq = fx.Int32(counter[0])
        if (tx == 0) & (p != i32_rank):
            flag = _sel4(p, arg_f0, arg_f1, arg_f2, arg_f3)
            val = _flag_read(flag)
            # Modular int32 comparison remains valid across counter wrap.
            # Two slots bound the peer lead/lag to much less than 2**31.
            while (val - seq) < 0:
                val = _flag_read(flag)
        fx.gpu.barrier()
        _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
        lo = fx.Int32(0)
        for r in range_constexpr(WORLD):
            lo = lo + (i32_rank > r).select(fx.Int32(counts[r]), fx.Int32(0))
        rows = fx.Int32(counts[i32_rank])
        ndw = rows * (N // 2)
        slot_off = fx.Int64((seq - 1) & 1) * i64_slot_bytes
        src = buffer_ops.create_buffer_resource_from_addr(
            _sel4(p, arg_s0, arg_s1, arg_s2, arg_s3) + slot_off + fx.Int64(lo) * (N * 2),
            num_records_bytes=ndw * 4)
        dst = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=ndw * WORLD * 4)
        off = (bid // WORLD) * STEP + tx * 4
        while off < ndw:
            vals = [Vec(buffer_ops.buffer_load(src, off + u * NT * 4, vec_width=4, dtype=fx.Int32,
                                               cache_modifier=SYS)) for u in range_constexpr(UNROLL)]
            for u in range_constexpr(UNROLL):
                o = off + u * NT * 4
                row = o // (N // 2)
                col = o % (N // 2)
                dst_off = row * (N * WORLD // 2) + p * (N // 2) + col
                buffer_ops.buffer_store(vals[u], dst, dst_off, mask=o < ndw)
            off = off + (PW // WORLD) * STEP

    @flyc.jit
    def launch_all_to_all_bf16_pull(arg_out: fx.Int64, arg_s0: fx.Int64, arg_s1: fx.Int64, arg_s2: fx.Int64,
                                   arg_s3: fx.Int64, arg_f0: fx.Int64, arg_f1: fx.Int64, arg_f2: fx.Int64,
                                   arg_f3: fx.Int64, counter: fx.Tensor, counts: fx.Tensor,
                                   i64_slot_bytes: fx.Int64, i32_rank: fx.Int32, stream: fx.Stream):
        kernel(arg_out, arg_s0, arg_s1, arg_s2, arg_s3, arg_f0, arg_f1, arg_f2, arg_f3,
               counter, counts, i64_slot_bytes, i32_rank).launch(grid=(PW, 1, 1), block=(NT, 1, 1), stream=stream)
    return launch_all_to_all_bf16_pull
