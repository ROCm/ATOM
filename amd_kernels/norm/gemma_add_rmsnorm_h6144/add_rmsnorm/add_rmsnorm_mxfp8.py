# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Residual add + Gemma RMSNorm + MXFP8 (1x32, E8M0) quant, FlyDSL (gfx950): bit for bit
``mxfp8_norm_quant._gemma_add_rmsnorm_mxfp8_quant_kernel`` at hidden 6144 (BLOCK_N 8192, 8 warps), codes in the decode
GEMM's (16, 16) layout, AITER's shuffled scales.

Triton's lane map: thread t holds columns 8 t .. 8 t + 7 and, for t < 256, 4096 + 8 t .., so the sum of squares keeps
its order (16 squares in sequence, the wave tree, the 8 wave sums xor 4 / 2 / 1); a 32-column group is 4 lanes. Scale
2^ceil(log2(RN(amax / 448))) clamped to E8M0 [0, 254], codes by v_cvt_scalef32_pk_fp8_f32 with MODE.fp16_ovfl set, as
Triton emits them. One barrier: the weight loads go out with x and r, every wave folds the wave sums itself, and lane
pairs swap 8-byte halves by DPP to store whole 16-byte code pieces (Triton: five barriers, two LDS transposes). Each
XCD takes a contiguous run of rows (``xcd_row``): a code tile's pieces of 16 rows share cache lines, written from one
L2 instead of eight."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    max_dpp,
    maxnum,
    minnum,
    pk_bf16,
    rsqrt_rn,
    seq_sum_sq,
    swap_xor1,
    wave_sum_lane63,
    xcd_grid,
    xcd_row,
)

NW = 8
SCALE_K1 = H // 256  # AITER scale layout: 256-column blocks per row
TINY = 0x00800000  # fp32 tiny, the amax floor
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)  # hwreg(HW_REG_MODE, 23, 1)
NT = 2  # buffer store cache policy: non-temporal


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _set_fp16_ovfl():
    """MODE.fp16_ovfl = 1, which Triton sets before its fp8 conversions."""
    _llvm.call_intrinsic(
        None,
        "llvm.amdgcn.s.setreg",
        [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()],
        [],
        [],
    )


def _cvt_pk_fp8(old, a, b, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32 (scale 1): a, b -> 2 e4m3 in the low / high half."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _e8m0(y):
    """(scale byte as fp32, 2^(127 - byte)) of a 32-column group whose 4 lanes hold ``y``: Triton's amax / log2 /
    ceil / clamp sequence."""
    a = fx.math.absf(y[0])
    for v in y[1:]:
        a = maxnum(a, fx.math.absf(v))
    a = max_dpp(a, 0x4E)  # quad_perm [2, 3, 0, 1]
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


def _codes(y, rescale):
    """8 e4m3 codes of y * rescale as 2 dwords (column order, low byte first)."""
    q = [v * rescale for v in y]
    lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[0], q[1], False), q[2], q[3], True)
    hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[4], q[5], False), q[6], q[7], True)
    return [lo, hi]


def _block_sum(w):
    """Triton's fold of the 8 wave sums (lane 0 of xor 4, 2, 1)."""
    return ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]))


def _scale_off(m, g):
    """Byte of (row, group) in AITER's shuffled scale layout."""
    return (
        ((((m // 32) * SCALE_K1 + g // 8) * 4 + g % 4) * 16 + m % 16) * 2 + (g % 8) // 4
    ) * 2 + (m % 32) // 16


def _code_off(m, p):
    """Byte of (row, 16-column piece) in the (16, 16) code layout."""
    return (m // 16) * (16 * H) + (p // 4) * 1024 + (p % 4) * 256 + (m % 16) * 16


def compile_add_rmsnorm_mxfp8():
    name = "gemma_add_rmsnorm_mxfp8_tiled_h6144"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]

    # the first loads' arguments lead (the first 56 bytes arrive preloaded)
    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,
        i32_so: fx.Int32,
        i32_sa: fx.Int32,  # unused, kept in the launch ABI
        arg_x: fx.Int64,  # [m, H] bf16, row stride i32_sx
        arg_r: fx.Int64,  # residual [m, H] bf16, row stride i32_sr
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_res: fx.Int64,  # residual out [m, H] bf16, row stride i32_so
        arg_aux: fx.Int64,  # unused, kept in the launch ABI
        arg_q: fx.Int64,  # codes e4m3, (16, 16) tiles of 16 rows
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
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + fx.Int64(row) * fx.Int64(i32_sx * 2), num_records_bytes=nb
            )
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
            has1 = tx < 256  # second chunk (columns 4096 + 8 t) inside the row
            offs = [tx * 4, tx * 4 + 2048]
            masks = [None, has1]

            def ld(rs, c):
                return bf16x8(
                    Vec(
                        buffer_ops.buffer_load(
                            rs, offs[c], vec_width=4, dtype=fx.Int32, mask=masks[c]
                        )
                    )
                )

            xv = [ld(xr, c) for c in range_constexpr(2)]
            rv = [ld(rr, c) for c in range_constexpr(2)]
            wv = [ld(wr, c) for c in range_constexpr(2)]
            s = [
                [xv[c][i] + rv[c][i] for i in range_constexpr(8)]
                for c in range_constexpr(2)
            ]
            for c in range_constexpr(2):
                res = [
                    pk_bf16(s[c][2 * i], s[c][2 * i + 1]) for i in range_constexpr(4)
                ]
                buffer_ops.buffer_store(
                    Vec.from_elements(res, fx.Int32), resr, offs[c], mask=masks[c]
                )
            acc = wave_sum_lane63(seq_sum_sq(s[0] + s[1]))
            if lane == 63:
                part[wave] = acc
            fx.gpu.barrier()
            rstd = rsqrt_rn(
                _block_sum([part[i] for i in range_constexpr(NW)]),
                i32_eps.bitcast(fx.Float32),
            )
            codes, sbs = [], []
            for c in range_constexpr(2):
                y = [
                    (s[c][i] * rstd) * (wv[c][i] + fx.Float32(1.0))
                    for i in range_constexpr(8)
                ]
                sb, rescale = _e8m0(y)
                codes.append(_codes(y, rescale))
                sbs.append(fx.Int32(sb.to(fx.Int32)))
            # lane pairs: the even lane stores chunk 0's 16 columns, the odd chunk 1's
            odd = (lane & 1) == 1
            recv = swap_xor1(
                [odd.select(codes[0][i], codes[1][i]) for i in range_constexpr(2)]
            )
            piece = Vec.from_elements(
                [
                    odd.select(recv[0], codes[0][0]),
                    odd.select(recv[1], codes[0][1]),
                    odd.select(codes[1][0], recv[0]),
                    odd.select(codes[1][1], recv[1]),
                ],
                fx.Int32,
            )
            p = tx // 2 + odd.select(fx.Int32(256), fx.Int32(0))  # 16-column piece
            # nt: these 16-byte pieces of 16 rows' lines are read by the next GEMM only
            buffer_ops.buffer_store(
                piece,
                qr,
                _code_off(row, p) // 4,
                mask=((lane & 1) == 0) | has1,
                cache_modifier=NT,
            )
            # one lane per 32-column group stores its scale byte
            q4 = lane % 4
            g = tx // 4 + (q4 == 1).select(fx.Int32(128), fx.Int32(0))
            sb = (q4 == 1).select(sbs[1], sbs[0])
            buffer_ops.buffer_store(
                sb.to(fx.Int8),
                sr,
                _scale_off(row, g),
                mask=(q4 == 0) | ((q4 == 1) & has1),
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_res: fx.Int64,
        arg_aux: fx.Int64,
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_sr: fx.Int32,
        i32_so: fx.Int32,
        i32_sa: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m,
            i32_sx,
            i32_sr,
            i32_so,
            i32_sa,
            arg_x,
            arg_r,
            arg_w,
            arg_res,
            arg_aux,
            arg_q,
            arg_s,
            i64_q_bytes,
            i64_s_bytes,
            i32_eps,
        ).launch(
            grid=(fx.Int64(xcd_grid(i32_m)), 1, 1), block=(64 * NW, 1, 1), stream=stream
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
