# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Residual add + Gemma RMSNorm, bf16 out, FlyDSL (gfx950): bit for bit
``gemma_rmsnorm._gemma_fused_add_rmsnorm_kernel`` at hidden 6144 (BLOCK_N 8192,
16 warps). One workgroup per row, Triton's lane map: thread t holds columns
8 t .. 8 t + 7, so the sum of squares has its order (per thread in sequence, the
wave tree, then the 16 wave sums xor 8 / 4 / 2 / 1). Its 4 all-masked warps add
+0 and are left out. The weight is loaded with x and r, and every wave folds the
wave sums itself after one barrier (Triton: two barriers, then the weight load)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    wave_sum_lane63,
)

NW = H // 8 // 64  # 12 waves of 8 columns per lane


def _block_sum(w):
    """Triton's fold of 16 wave sums (lane 0 of xor 8, 4, 2, 1), waves 12-15 = +0."""
    return (((w[0] + w[8]) + w[4]) + ((w[2] + w[10]) + w[6])) + (
        ((w[1] + w[9]) + w[5]) + ((w[3] + w[11]) + w[7])
    )


def compile_add_rmsnorm():
    name = "gemma_add_rmsnorm_h6144"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [m, H] bf16, row stride i32_sx
        arg_r: fx.Int64,  # residual [m, H] bf16, row stride i32_sr
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_out: fx.Int64,  # [m, H] bf16
        arg_res: fx.Int64,  # residual out [m, H] bf16
        arg_aux: fx.Int64,  # unused, kept in the launch ABI
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,
        i32_sa: fx.Int32,
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx, row = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        nb = H * 2
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x + fx.Int64(row) * fx.Int64(i32_sx * 2), num_records_bytes=nb
        )
        rr = buffer_ops.create_buffer_resource_from_addr(
            arg_r + fx.Int64(row) * fx.Int64(i32_sr * 2), num_records_bytes=nb
        )
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=nb)
        orow = fx.Int64(row) * nb
        outr = buffer_ops.create_buffer_resource_from_addr(
            arg_out + orow, num_records_bytes=nb
        )
        resr = buffer_ops.create_buffer_resource_from_addr(
            arg_res + orow, num_records_bytes=nb
        )
        off = tx * 4
        xv = bf16x8(Vec(buffer_ops.buffer_load(xr, off, vec_width=4, dtype=fx.Int32)))
        rv = bf16x8(Vec(buffer_ops.buffer_load(rr, off, vec_width=4, dtype=fx.Int32)))
        wv = bf16x8(Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32)))
        s = [xv[i] + rv[i] for i in range_constexpr(8)]
        res = [pk_bf16(s[2 * i], s[2 * i + 1]) for i in range_constexpr(4)]
        buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
        acc = wave_sum_lane63(seq_sum_sq(s))
        if lane == 63:
            part[wave] = acc
        fx.gpu.barrier()
        rstd = rsqrt_rn(
            _block_sum([part[i] for i in range_constexpr(NW)]),
            i32_eps.bitcast(fx.Float32),
        )
        y = [(s[i] * rstd) * (wv[i] + fx.Float32(1.0)) for i in range_constexpr(8)]
        out = [pk_bf16(y[2 * i], y[2 * i + 1]) for i in range_constexpr(4)]
        buffer_ops.buffer_store(Vec.from_elements(out, fx.Int32), outr, off)

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_out: fx.Int64,
        arg_res: fx.Int64,
        arg_aux: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,
        i32_sa: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_x,
            arg_r,
            arg_w,
            arg_out,
            arg_res,
            arg_aux,
            i32_sx,
            i32_sr,
            i32_sa,
            i32_eps,
        ).launch(grid=(fx.Int64(i32_m), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
