# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``kernel.py``'s norm + quant on 4 physical waves, each running three of its logical waves: the same square tree
and bf16 values, fewer waves to schedule."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .quant_math import (
    _code_off,
    _codes,
    _e8m0,
    _scale_off,
    _set_fp16_ovfl,
)
from .common import (
    PRELOAD_HINTS,
    H,
    bf16x8,
    pk_bf16,
    rsqrt_rn,
    swap_xor1,
    wave_sum_lane63,
    xcd_grid,
    xcd_row,
)

NW = 4  # three logical 512-column waves per physical wave


def _block_sum(w):
    return (((w[0]+w[8])+w[4])+((w[2]+w[10])+w[6])) + (((w[1]+w[9])+w[5])+((w[3]+w[11])+w[7]))


def compile_gemma_rmsnorm_bf16_mxfp8_fourwave(shuffled=False, with_bf16=False):
    name = f"gemma_rmsnorm_bf16_mxfp8_fourwave_{'tiled' if shuffled else 'rowmajor'}_h6144" + ("_bf16" if with_bf16 else "")

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        arg_x: fx.Int64,  # [m, H] bf16, row stride i32_sx
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_q: fx.Int64,  # codes e4m3 (row-major [m, H] or (16, 16) tiles)
        arg_s: fx.Int64,  # scales u8, AITER layout
        arg_norm: fx.Int64,  # optional norm out [m, H] bf16
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
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=nb
            )
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=i64_q_bytes
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_s, num_records_bytes=i64_s_bytes
            )
            normr = buffer_ops.create_buffer_resource_from_addr(
                arg_norm + fx.Int64(row) * nb, num_records_bytes=nb
            )
            xv=[];wv=[]
            for c in range_constexpr(3):
                off=tx*4+c*1024
                xv.append(bf16x8(Vec(buffer_ops.buffer_load(xr,off,vec_width=4,dtype=fx.Int32))))
                wv.append(bf16x8(Vec(buffer_ops.buffer_load(wr,off,vec_width=4,dtype=fx.Int32))))
            for c in range_constexpr(3):
                v=xv[c]
                a0=fx.math.fma(v[0],v[0],v[2]*v[2])
                a1=fx.math.fma(v[1],v[1],v[3]*v[3])
                b0=fx.math.fma(v[4],v[4],v[6]*v[6])
                b1=fx.math.fma(v[5],v[5],v[7]*v[7])
                acc=wave_sum_lane63((a0+b0)+(a1+b1))
                if lane==63:part[wave+c*4]=acc
            fx.gpu.barrier()
            rstd = rsqrt_rn(
                _block_sum([part[i] for i in range_constexpr(12)]),
                i32_eps.bitcast(fx.Float32),
            )
            for c in range_constexpr(3):
                y = [(xv[c][i] * rstd) * (wv[c][i] + fx.Float32(1.0)) for i in range_constexpr(8)]
                # the standalone quantizer reads the bf16-rounded normed value
                yb = bf16x8(
                    Vec.from_elements(
                        [pk_bf16(y[2 * i], y[2 * i + 1]) for i in range_constexpr(4)],
                        fx.Int32,
                    )
                )
                if const_expr(with_bf16):
                    buffer_ops.buffer_store(Vec.from_elements([pk_bf16(y[2*i],y[2*i+1]) for i in range_constexpr(4)],fx.Int32),normr,tx*4+c*1024)
                sb, rescale = _e8m0(yb)
                codes = _codes(yb, rescale)  # 2 dwords, column order
                if const_expr(shuffled):
                    # (16, 16) layout: even lanes store their piece's low 8 B plus
                    # the odd lane's high 8 B as one 16 B store; odd lanes idle
                    recv = swap_xor1(codes)
                    piece = Vec.from_elements(
                        [codes[0], codes[1], recv[0], recv[1]], fx.Int32
                    )
                    buffer_ops.buffer_store(
                        piece,
                        qr,
                        _code_off(row, tx // 2+c*128) // 4,
                        mask=(tx % 2) == 0,
                    )
                else:
                    buffer_ops.buffer_store(
                        Vec.from_elements(codes, fx.Int32),
                        qr,
                        (row * (H // 4)) + tx * 2+c*512,
                    )
                if (tx % 4) == 0:
                    # one lane per 32-column group stores its scale byte
                    buffer_ops.buffer_store(
                        fx.Int32(sb.to(fx.Int32)).to(fx.Int8),
                        sr,
                        _scale_off(row, tx // 4+c*64),
                        offset_is_bytes=True,
                    )

    @flyc.jit
    def launch_gemma_rmsnorm_bf16_mxfp8_fourwave_h6144(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_norm: fx.Int64,
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m,
            i32_sx,
            arg_x,
            arg_w,
            arg_q,
            arg_s,
            arg_norm,
            i64_q_bytes,
            i64_s_bytes,
            i32_eps,
        ).launch(
            grid=(fx.Int64(xcd_grid(i32_m)), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_gemma_rmsnorm_bf16_mxfp8_fourwave_h6144.kernel_name = name
    launch_gemma_rmsnorm_bf16_mxfp8_fourwave_h6144.compile_hints = PRELOAD_HINTS
    return launch_gemma_rmsnorm_bf16_mxfp8_fourwave_h6144
