# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""FlyDSL collectives of the sequence-parallel prefill (TP4 on MI355X, gfx950): the shared flag helpers and constants,
and the launches every communicator builds on.

Every rank owns a data pool (coarse-grained device memory the peers map by IPC)
and, per microbatch, a flag word and a ticket counter in uncached memory. A
collective publishes with ``fence_publish`` (every workgroup writes its XCD's L2
back -- a coarse-grained buffer's dirty lines are not written back for the peers'
xGMI reads at a kernel boundary -- and the last arriver on the ticket stores the
sequence number into the flag), then a small persistent grid pulls: its
workgroups spin on their rank's flag, acquire, and stream that rank's block over
its xGMI link with system-coherent loads (``SC0 | SC1``, no stale lines from the
local L2) and plain stores. Few, long-lived workgroups matter: next to the compute
stream's kernels a CU slot is rare, and a one-shot grid of 1024 workgroups took
3-10x its idle time (16 persistent: 1.3x, RCCL's 1.25x); RCCL's channels are
persistent too. A reduce-scatter pulls the four blocks into local staging and
``sum4`` adds them in fp32 in rank order (fixed, no atomics; RCCL's bf16 ring
differs by up to 0.25).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

WORLD = 4
NT = 256  # threads per workgroup
UNROLL = 8  # 16 B per thread per step
STEP = NT * 4 * UNROLL  # dwords per workgroup per loop step
SYS = 0x11  # sc0 | sc1: system-coherent loads / stores
# MI355X: workgroup w runs on XCD w % 8, each XCD has its own L2; 8 workgroups
# per XCD in case the dispatcher does not spread 8 single waves
FENCE_WGS = 64
# chunks per rank block (256 x 32 KB steps at 50 MB), and the persistent pull
# grid that loops over them: 4 workgroups per rank hold the link rate idle
# (0.87 ms vs RCCL 0.95 at 50 MB) and next to a GEMM loop match RCCL (1.16 vs
# 1.18 ms; 64 workgroups 1.45, the one-shot 1024 2.65 on a high-priority stream)
PER_PEER = 256
PULL_WGS = 16
SUM_WGS = 1024
# per microbatch set: the flag word, then the fence ticket counter (bytes)
FLAG_STRIDE = 16


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


def _sel4(p, a0, a1, a2, a3):
    return (p == 0).select(a0, (p == 1).select(a1, (p == 2).select(a2, a3)))


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _pk_bf16(a, b):
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def compile_publish():
    """The flag word <- value (a system-coherent store): the init's zeroing of
    the flags and ticket counters."""

    @flyc.kernel(name="pull_collectives_ranks4_publish", known_block_size=[64, 1, 1])
    def kernel(arg_flag: fx.Int64, i32_seq: fx.Int32):
        fr = buffer_ops.create_buffer_resource_from_addr(arg_flag, num_records_bytes=64)
        buffer_ops.buffer_store(
            i32_seq, fr, 0, mask=fx.thread_idx.x == 0, cache_modifier=SYS
        )

    @flyc.jit
    def launch(arg_flag: fx.Int64, i32_seq: fx.Int32, stream: fx.Stream):
        kernel(arg_flag, i32_seq).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.kernel_name = "pull_collectives_ranks4_publish"
    return launch


def compile_fence_publish():
    """Fence + publish in one launch: every workgroup releases its XCD's L2 (the
    producers' dirty lines reach memory), takes a ticket on the counter word next
    to the flag; the last arriver resets the counter and stores ``seq`` into the
    flag. Run on the stream that produced the data (the comm stream, after its
    wait for the compute stream)."""

    @flyc.kernel(
        name="pull_collectives_ranks4_fence_publish", known_block_size=[64, 1, 1]
    )
    def kernel(arg_flag: fx.Int64, i32_seq: fx.Int32):
        _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="")
        if fx.thread_idx.x == 0:
            t = _flag_add(arg_flag + 4, 1)
            if t == FENCE_WGS - 1:
                _flag_add(arg_flag + 4, -FENCE_WGS)
                fr = buffer_ops.create_buffer_resource_from_addr(
                    arg_flag, num_records_bytes=64
                )
                buffer_ops.buffer_store(i32_seq, fr, 0, cache_modifier=SYS)

    @flyc.jit
    def launch(arg_flag: fx.Int64, i32_seq: fx.Int32, stream: fx.Stream):
        kernel(arg_flag, i32_seq).launch(
            grid=(FENCE_WGS, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.kernel_name = "pull_collectives_ranks4_fence_publish"
    return launch


def compile_sum4():
    """The reduce-scatter's second step after the pull into staging: out =
    bf16(sum in fp32, rank order 0..3, of the four local bf16 blocks)."""

    @flyc.kernel(name="pull_collectives_ranks4_sum4", known_block_size=[NT, 1, 1])
    def kernel(
        arg_out: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        i64_block: fx.Int64,  # bytes per block (bf16, multiple of 16)
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        srcs = [
            buffer_ops.create_buffer_resource_from_addr(a, num_records_bytes=i64_block)
            for a in (arg_s0, arg_s1, arg_s2, arg_s3)
        ]
        out = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=i64_block
        )
        n_dw = fx.Int32(i64_block // 4)
        per_block = (n_dw + (SUM_WGS - 1)) // SUM_WGS
        per_block = (per_block + (STEP - 1)) // STEP * STEP
        start = bid * per_block
        end = start + per_block
        off = start + tx * 4
        while off < end:
            words = []
            for u in range_constexpr(UNROLL):
                o = off + u * NT * 4
                words.append(
                    [
                        Vec(
                            buffer_ops.buffer_load(
                                srcs[r],
                                o,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=o < n_dw,
                            )
                        )
                        for r in range_constexpr(WORLD)
                    ]
                )
            for u in range_constexpr(UNROLL):
                o = off + u * NT * 4
                packed = []
                for i in range_constexpr(4):
                    lo = fx.Float32(0.0)
                    hi = fx.Float32(0.0)
                    for r in range_constexpr(WORLD):
                        w = fx.Int32(words[u][r][i])
                        lo = lo + _as_f32(w << 16)
                        hi = hi + _as_f32(w & fx.Int32(-65536))
                    packed.append(_pk_bf16(lo, hi))
                buffer_ops.buffer_store(
                    Vec.from_elements(packed, fx.Int32), out, o, mask=o < n_dw
                )
            off = off + STEP

    @flyc.jit
    def launch(
        arg_out: fx.Int64,
        arg_s0: fx.Int64,
        arg_s1: fx.Int64,
        arg_s2: fx.Int64,
        arg_s3: fx.Int64,
        i64_block: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(arg_out, arg_s0, arg_s1, arg_s2, arg_s3, i64_block).launch(
            grid=(SUM_WGS, 1, 1), block=(NT, 1, 1), stream=stream
        )

    launch.kernel_name = "pull_collectives_ranks4_sum4"
    return launch
