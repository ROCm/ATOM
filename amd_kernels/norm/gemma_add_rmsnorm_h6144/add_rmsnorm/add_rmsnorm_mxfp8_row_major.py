# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``add_rmsnorm_mxfp8`` with row-major codes (AITER scales): the same arithmetic and quant of the fp32 y, only the code
stores differ; the dense layers' sequence-parallel tails use it at every M."""

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
    xcd_grid,
    xcd_row,
)

from .add_rmsnorm_mxfp8 import (
    _block_sum, _codes, _e8m0, _scale_off, _set_fp16_ovfl,
)

NW = 8


def compile_add_rmsnorm_mxfp8_row_major():
    name = "gemma_add_rmsnorm_mxfp8_row_major_h6144"

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
        arg_q: fx.Int64,  # codes e4m3, row-major [m, H]
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
            for c in range_constexpr(2):
                buffer_ops.buffer_store(
                    Vec.from_elements(codes[c], fx.Int32), qr,
                    row * (H // 4) + tx * 2 + c * 1024,
                    mask=masks[c],
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
    def launch_add_rmsnorm_mxfp8_row_major(
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

    launch_add_rmsnorm_mxfp8_row_major.kernel_name = name
    launch_add_rmsnorm_mxfp8_row_major.compile_hints = PRELOAD_HINTS
    return launch_add_rmsnorm_mxfp8_row_major
