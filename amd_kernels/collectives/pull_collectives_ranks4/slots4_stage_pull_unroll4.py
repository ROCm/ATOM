# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four-read pull batch: slot sum + shared add overlapped with the reduce-scatter pull, one launch.

Workgroups 0..PULLERS-1 pull, the next STAGERS stage; PULLERS + STAGERS <= 256
so no puller shares a CU with a stager (272 groups: +12% at 32768 rows).

Stagers: the slot-order FP32 sum + shared output of slots4_stage_rows, rounded
once to BF16, into the own pool slot (counter c -> slot c % 2), in chunks:
chunk k = part k // 4 of owner block k % 4, so every owner's first part comes
first. Stores are system-coherent (SC0 | SC1, through the L2). After a chunk
every wave waits for its stores (vmcnt 0), the workgroup meets at a barrier and
lane 0 takes a ticket; the last of the STAGERS stores c+1 into the chunk's flag
word. No L2 writeback per chunk: a system-scope release per chunk and stager
(buffer_wbl2) took the 32K tail from 1.86 to 2.32 ms.

Pullers: workgroup w serves source rank w % 4. Per part j its lane 0 spins on
that source's flag of chunk 4 j + me (the own rank too), the workgroup acquires,
then copies the part of its block with system-coherent 16-byte loads into
rank-major local staging (as device_pull). The last workgroup of the launch to
finish stores c+1 into the device counter.

Flag words: one uncached buffer per rank, mapped by the peers: flags at word k
(k < 4 * PARTS), tickets at byte TICKETS + 4 k, the launch's end ticket at END.
A rank rewrites slot c % 2 only in call c+2, after every peer finished the pull
of call c (each peer's call c+1 launch started after it).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.common import bf16x8, pk_bf16
from .kernel import SYS, _flag_add, _flag_read, _sel4

H = 6144
NT = 256
# Standalone TP4 4K..32K tail: 4 beats 1/2/8/16/32, including its consumer.
# Above 16K in the concurrent context the host keeps the eight-read batch.
UNROLL = 4  # 16 B reads per thread in flight, private to this fused collective
PULLERS = 32  # 8 per source rank
STAGERS = 192
PARTS = 8  # chunks per owner block
TICKETS = 256  # byte offsets in the chunk-flag buffer
END = 1024
FLAG_BYTES = 4096


def compile_slots4_stage_pull_unroll4():
    unroll = UNROLL
    step = NT * 4 * unroll  # dwords per puller per loop step
    pps = PULLERS // 4
    nwg = PULLERS + STAGERS
    name = f"slots4_shared_stage_pull_unroll{unroll}_h6144_ranks4"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_slots: fx.Int64, arg_shared: fx.Int64, arg_dst: fx.Int64,
               arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, arg_p3: fx.Int64,
               arg_c0: fx.Int64, arg_c1: fx.Int64, arg_c2: fx.Int64, arg_c3: fx.Int64,
               counter: fx.Tensor, i64_stride: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        i32_seq = fx.Int32(counter[0]) + 1
        slot_off = fx.Int64((i32_seq - 1) & 1) * i64_stride
        br = i32_rows // 4
        my_c = _sel4(i32_me, arg_c0, arg_c1, arg_c2, arg_c3)
        if bid >= PULLERS:
            sid = bid - PULLERS
            dst = buffer_ops.create_buffer_resource_from_addr(
                _sel4(i32_me, arg_p0, arg_p1, arg_p2, arg_p3) + slot_off,
                num_records_bytes=fx.Int64(i32_rows) * H * 2)
            hr = buffer_ops.create_buffer_resource_from_addr(
                arg_shared, num_records_bytes=fx.Int64(i32_rows) * H * 2)
            k = fx.Int32(0)
            while k < 4 * PARTS:
                r0 = (k // 4) * br // PARTS
                r1 = (k // 4 + 1) * br // PARTS
                piece = sid
                while piece < (r1 - r0) * 3:
                    row = (k % 4) * br + r0 + piece // 3
                    col = (piece % 3) * (NT * 4) + tx * 4
                    off = row * (H // 2) + col
                    sr = buffer_ops.create_buffer_resource_from_addr(
                        arg_slots + fx.Int64(row) * (4 * H * 2), num_records_bytes=4 * H * 2)
                    vals = [bf16x8(Vec(buffer_ops.buffer_load(
                        sr, col + q * (H // 2), vec_width=4, dtype=fx.Int32)))
                        for q in range_constexpr(4)]
                    shared = bf16x8(Vec(buffer_ops.buffer_load(
                        hr, off, vec_width=4, dtype=fx.Int32)))
                    acc = [(((vals[0][e] + vals[1][e]) + vals[2][e]) + vals[3][e]) + shared[e]
                           for e in range_constexpr(8)]
                    packed = [pk_bf16(acc[2 * e], acc[2 * e + 1]) for e in range_constexpr(4)]
                    buffer_ops.buffer_store(Vec.from_elements(packed, fx.Int32), dst, off,
                                            cache_modifier=SYS)
                    piece = piece + STAGERS
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if tx == 0:
                    tk = my_c + TICKETS + fx.Int64(k * 4)
                    if _flag_add(tk, 1) == STAGERS - 1:
                        _flag_add(tk, -STAGERS)
                        fr = buffer_ops.create_buffer_resource_from_addr(
                            my_c, num_records_bytes=TICKETS)
                        buffer_ops.buffer_store(i32_seq, fr, k, cache_modifier=SYS)
                k = k + 1
        else:
            src = bid % 4
            fa = _sel4(src, arg_c0, arg_c1, arg_c2, arg_c3)
            block = fx.Int64(br) * (H * 2)
            srs = buffer_ops.create_buffer_resource_from_addr(
                _sel4(src, arg_p0, arg_p1, arg_p2, arg_p3) + slot_off
                + fx.Int64(i32_me * br) * (H * 2), num_records_bytes=block)
            drs = buffer_ops.create_buffer_resource_from_addr(
                arg_dst + fx.Int64(src) * block, num_records_bytes=block)
            j = fx.Int32(0)
            while j < PARTS:
                if tx == 0:
                    fw = fa + fx.Int64((j * 4 + i32_me) * 4)
                    f = _flag_read(fw)
                    while f < i32_seq:
                        f = _flag_read(fw)
                fx.gpu.barrier()
                _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
                d1 = ((j + 1) * br // PARTS) * (H // 2)
                off = (j * br // PARTS) * (H // 2) + (bid // 4) * step + tx * 4
                while off < d1:
                    vals = []
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        vals.append(Vec(buffer_ops.buffer_load(
                            srs, o, vec_width=4, dtype=fx.Int32, mask=o < d1,
                            cache_modifier=SYS)))
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        buffer_ops.buffer_store(vals[u], drs, o, mask=o < d1)
                    off = off + step * pps
                j = j + 1
        fx.gpu.barrier()
        if tx == 0:
            if _flag_add(my_c + END, 1) == nwg - 1:
                _flag_add(my_c + END, -nwg)
                counter[0] = i32_seq

    @flyc.jit
    def launch_slots4_stage_pull_unroll4(
            arg_slots: fx.Int64, arg_shared: fx.Int64, arg_dst: fx.Int64,
            arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, arg_p3: fx.Int64,
            arg_c0: fx.Int64, arg_c1: fx.Int64, arg_c2: fx.Int64, arg_c3: fx.Int64,
            counter: fx.Tensor, i64_stride: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32,
            stream: fx.Stream):
        kernel(arg_slots, arg_shared, arg_dst, arg_p0, arg_p1, arg_p2, arg_p3,
               arg_c0, arg_c1, arg_c2, arg_c3, counter, i64_stride, i32_rows, i32_me).launch(
            grid=(nwg, 1, 1), block=(NT, 1, 1), stream=stream)
    launch_slots4_stage_pull_unroll4.kernel_name = name
    return launch_slots4_stage_pull_unroll4
