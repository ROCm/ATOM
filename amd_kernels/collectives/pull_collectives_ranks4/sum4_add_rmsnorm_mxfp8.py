# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RS rank sum + residual + Gemma RMSNorm + MXFP8 (row-major codes, AITER-shuffled scales) in the final local launch.

The first load sums four BF16 rank slabs, with the reduce-scatter's sum order and BF16 rounding. The norm and quant are
Triton's SHUFFLE_Q=False kernel's lane map, thread tx holds the 16 contiguous columns
16 tx .. 16 tx + 15 (tx < 384), and the sum of squares is its packed tree with FMA leaves, ssq = E + O,
E = (fma(q0, q0, q2^2) + fma(q4, q4, q6^2)) + (fma(q8, q8, q10^2) + fma(q12, q12, q14^2)) (O: odd columns), then
the same wave tree and a cross-warp fold equal to ``_block_sum`` with commuted operands (bitwise equal). One
16-byte code piece per thread; a 32-column group is 2 lanes (quad_perm [1, 0, 3, 2] pair max).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.add_rmsnorm_mxfp8 import (
    NW,
    TINY,
    _block_sum,
    _cvt_pk_fp8,
    _f32,
    _scale_off,
    _set_fp16_ovfl,
)
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.common import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    max_dpp,
    maxnum,
    minnum,
    pk_bf16,
    rsqrt_rn,
    wave_sum_lane63,
    xcd_grid,
    xcd_row,
)


def _e8m0_rowq(y):
    """(scale byte as fp32, 2^(127 - byte)) of a 32-column group whose 2 lanes hold 16 values each: the tiled
    variant's amax / log2 / ceil / clamp sequence."""
    a = fx.math.absf(y[0])
    for v in y[1:]:
        a = maxnum(a, fx.math.absf(v))
    a = maxnum(max_dpp(a, 0xB1), _f32(TINY))  # quad_perm [1, 0, 3, 2]
    d = a / fx.Float32(448.0)
    small = d < _f32(TINY)
    lg = fx.Float32(
        _rocdl_ops.log(T.f32, fx.as_ir_value(small.select(d * fx.Float32(2.0**32), d)))
    ) - small.select(fx.Float32(32.0), fx.Float32(0.0))
    sb = minnum(
        maxnum(fx.math.ceil(lg) + fx.Float32(127.0), fx.Float32(0.0)),
        fx.Float32(254.0),
    )
    rescale = fx.Float32(_rocdl_ops.exp2(T.f32, fx.as_ir_value(fx.Float32(127.0) - sb)))
    return sb, rescale


def _codes16(y, rescale):
    """16 e4m3 codes of y * rescale as 4 dwords (column order, low byte first)."""
    q = [v * rescale for v in y]
    return [
        _cvt_pk_fp8(
            _cvt_pk_fp8(fx.Int32(0), q[4 * d], q[4 * d + 1], False),
            q[4 * d + 2],
            q[4 * d + 3],
            True,
        )
        for d in range_constexpr(4)
    ]


def _tail_rowq(tx, row, lane, s, wv, rstd, has, qr, sr):
    """Row-major codes (one 16-byte piece per thread) + AITER scale bytes."""
    y = [(s[i] * rstd) * (wv[i] + fx.Float32(1.0)) for i in range_constexpr(16)]
    sb, rescale = _e8m0_rowq(y)
    codes = _codes16(y, rescale)
    buffer_ops.buffer_store(
        Vec.from_elements(codes, fx.Int32),
        qr,
        row * (H // 4) + tx * 4,
        mask=has,
    )
    # the even lane of each pair stores group tx // 2's scale byte (both compute it)
    sbi = fx.Int32(sb.to(fx.Int32))
    buffer_ops.buffer_store(
        sbi.to(fx.Int8),
        sr,
        _scale_off(row, tx // 2),
        mask=has & ((tx & 1) == 0),
        offset_is_bytes=True,
    )


def _slab_acc(split_k, tx, row, arg_tmp, i32_m, n_dwords, base_dword, mask):
    """Four staged BF16 rank blocks, FP32 accumulation from +0 in rank order."""
    acc = [fx.Float32(0.0)] * n_dwords
    for rank in range_constexpr(4):
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_tmp + (fx.Int64(rank) * fx.Int64(i32_m) + fx.Int64(row)) * (H * 2),
            num_records_bytes=H * 2,
        )
        for v in range_constexpr(n_dwords // 8):
            vals = bf16x8(Vec(buffer_ops.buffer_load(
                xr, base_dword // 2 + v * 4, vec_width=4, dtype=fx.Int32, mask=mask)))
            for j in range_constexpr(8):
                acc[v * 8 + j] = acc[v * 8 + j] + vals[j]
    return acc


def _pre_rowq(split_k, tx, row, arg_tmp, i32_m, rr, wr, resr):
    """The Triton row-major kernel's lane map: thread tx holds the 16
    contiguous columns 16 tx .. 16 tx + 15 (tx < 384). Returns (thread ssq, s,
    wv) and stores the residual-out."""
    has = tx < 384

    def ld(rs, v):
        return bf16x8(
            Vec(
                buffer_ops.buffer_load(
                    rs, tx * 8 + v * 4, vec_width=4, dtype=fx.Int32, mask=has
                )
            )
        )

    rv = ld(rr, 0) + ld(rr, 1)
    wv = ld(wr, 0) + ld(wr, 1)
    acc = _slab_acc(split_k, tx, row, arg_tmp, i32_m, 16, tx * 16, has)
    # the bf16 dwords the plain kernel loads from the reduced GEMM output
    packed = [pk_bf16(acc[2 * i], acc[2 * i + 1]) for i in range_constexpr(8)]
    xv = bf16x8(packed[:4]) + bf16x8(packed[4:])
    s = [xv[i] + rv[i] for i in range_constexpr(16)]
    res = [pk_bf16(s[2 * i], s[2 * i + 1]) for i in range_constexpr(8)]
    for v in range_constexpr(2):
        buffer_ops.buffer_store(
            Vec.from_elements(res[v * 4 : v * 4 + 4], fx.Int32),
            resr,
            tx * 8 + v * 4,
            mask=has,
        )
    # Triton's packed per-thread tree (FMA leaves; see the module docstring)
    fma = lambda a, b2: fx.Float32(fx.math.fma(a, a, b2 * b2))  # noqa: E731
    e = [fma(s[4 * i], s[4 * i + 2]) for i in range_constexpr(4)]
    o = [fma(s[4 * i + 1], s[4 * i + 3]) for i in range_constexpr(4)]
    ssq = ((e[0] + e[1]) + (e[2] + e[3])) + ((o[0] + o[1]) + (o[2] + o[3]))
    return wave_sum_lane63(ssq), s, wv, has


def compile_sum4_add_rmsnorm_mxfp8():
    split_k = 4
    name = "sum4_add_rmsnorm_mxfp8_rowmajor_h6144_ranks4"
    _pre = _pre_rowq
    _tail = _tail_rowq

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]

    # the first loads' arguments lead (the first 56 bytes arrive preloaded)
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_sr: fx.Int32,
        i32_so: fx.Int32,
        arg_tmp: fx.Int64,  # (4, m, H) BF16 staged rank shards
        arg_r: fx.Int64,  # residual [m, H] bf16, row stride i32_sr
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_res: fx.Int64,  # residual out [m, H] bf16, row stride i32_so
        arg_q: fx.Int64,  # codes e4m3 [m, H] row-major
        arg_s: fx.Int64,  # scales u8, AITER layout
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_eps: fx.Int32,  # fp32 bits
    ):
        _set_fp16_ovfl()
        tx = fx.thread_idx.x
        row, live = xcd_row(fx.block_idx.x, i32_m)
        if live:
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            nb = H * 2
            rr = buffer_ops.create_buffer_resource_from_addr(
                arg_r + fx.Int64(row) * fx.Int64(i32_sr * 2), num_records_bytes=nb
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=nb
            )
            resr = buffer_ops.create_buffer_resource_from_addr(
                arg_res + fx.Int64(row) * fx.Int64(i32_so * 2), num_records_bytes=nb
            )
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=i64_q_bytes
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_s, num_records_bytes=i64_s_bytes
            )
            acc2, s, wv, has = _pre(split_k, tx, row, arg_tmp, i32_m, rr, wr, resr)
            if lane == 63:
                part[wave] = acc2
            fx.gpu.barrier()
            rstd = rsqrt_rn(
                _block_sum([part[i] for i in range_constexpr(NW)]),
                i32_eps.bitcast(fx.Float32),
            )
            _tail(tx, row, lane, s, wv, rstd, has, qr, sr)

    @flyc.jit
    def launch_sum4_add_rmsnorm_mxfp8(
        arg_tmp: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_res: fx.Int64,
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sr: fx.Int32,
        i32_so: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m,
            i32_sr,
            i32_so,
            arg_tmp,
            arg_r,
            arg_w,
            arg_res,
            arg_q,
            arg_s,
            i64_q_bytes,
            i64_s_bytes,
            i32_eps,
        ).launch(
            grid=(fx.Int64(xcd_grid(i32_m)), 1, 1), block=(64 * NW, 1, 1), stream=stream
        )

    launch_sum4_add_rmsnorm_mxfp8.kernel_name = name
    launch_sum4_add_rmsnorm_mxfp8.compile_hints = PRELOAD_HINTS
    return launch_sum4_add_rmsnorm_mxfp8
