# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144.quant_math import (
    _code_off,
    _codes,
    _e8m0,
    _scale_off,
    _set_fp16_ovfl,
)
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144.common import (
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

NW = H // 8 // 64  # 12 waves of 8 columns per lane (the plain norm's layout)


def _block_sum(w):
    return (((w[0]+w[8])+w[4])+((w[2]+w[10])+w[6])) + (((w[1]+w[9])+w[5])+((w[3]+w[11])+w[7]))


def _lookup_source(arg_ids,arg_peers,row,rows,shard_vocab,id_bits):
    ir=buffer_ops.create_buffer_resource_from_addr(arg_ids,num_records_bytes=rows*(id_bits//8))
    if id_bits==32:
        token=fx.Int64(buffer_ops.buffer_load(ir,row,vec_width=1,dtype=fx.Int32,is_scalar=True))
    else:
        lo=fx.Int64(buffer_ops.buffer_load(ir,row*2,vec_width=1,dtype=fx.Int32,is_scalar=True))
        hi=fx.Int64(buffer_ops.buffer_load(ir,row*2+1,vec_width=1,dtype=fx.Int32,is_scalar=True))
        token=(lo&fx.Int64(0xffffffff))|(hi<<32)
    valid=(token>=0)&(token<fx.Int64(shard_vocab*4))
    safe=valid.select(token,fx.Int64(0))
    owner=(safe//fx.Int64(shard_vocab)).to(fx.Int32)
    local=safe%fx.Int64(shard_vocab)
    pr=buffer_ops.create_buffer_resource_from_addr(arg_peers,num_records_bytes=32)
    # buffer_ops' scalar load returns raw i32 dwords regardless of dtype.
    lo=fx.Int64(buffer_ops.buffer_load(pr,owner*2,vec_width=1,dtype=fx.Int32,is_scalar=True))
    hi=fx.Int64(buffer_ops.buffer_load(pr,owner*2+1,vec_width=1,dtype=fx.Int32,is_scalar=True))
    base=(lo&fx.Int64(0xffffffff))|(hi<<32)
    return buffer_ops.create_buffer_resource_from_addr(base+local*(H*2),num_records_bytes=H*2),valid

def _load_clean(rsrc,off,valid):
    v=Vec(buffer_ops.buffer_load(rsrc,off,vec_width=4,dtype=fx.Int32,mask=valid))
    clean=[]
    for i in range(4):
        w=fx.Int32(v[i])
        low=((w&0x7f80)==0).select(fx.Int32(0),w&0xffff)
        high=((w&0x7f800000)==0).select(fx.Int32(0),w&fx.Int32(-65536))
        clean.append(low|high)
    return Vec.from_elements(clean,fx.Int32)


def compile_vocab_gather_gemma_mxfp8(shuffled=False, id_bits=32):
    name = f"vocab_gather_gemma_mxfp8_{'tiled' if shuffled else 'rowmajor'}_h6144" + f"_i{id_bits}"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        arg_ids: fx.Int64, arg_peers: fx.Int64, shard_vocab: fx.Int32,
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_q: fx.Int64,  # codes e4m3 (row-major [m, H] or (16, 16) tiles)
        arg_s: fx.Int64,  # scales u8, AITER layout
        arg_residual: fx.Int64,  # raw embedding residual [m, H] bf16
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
            xr,valid=_lookup_source(arg_ids,arg_peers,row,i32_m,shard_vocab,id_bits)
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=nb
            )
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=i64_q_bytes
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_s, num_records_bytes=i64_s_bytes
            )
            residual_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arg_residual + fx.Int64(row) * nb, num_records_bytes=nb
            )
            off = tx * 4
            xv = bf16x8(
                _load_clean(xr,off,valid)
            )
            wv = bf16x8(
                Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32))
            )
            s = xv
            # Triton plain norm: even/odd four-way square trees, not the
            # sequential residual-norm tree. BF16 squares are exact in fp32
            # for normal served values; explicit fma also covers tiny values.
            a0=fx.math.fma(s[0],s[0],s[2]*s[2])
            a1=fx.math.fma(s[1],s[1],s[3]*s[3])
            b0=fx.math.fma(s[4],s[4],s[6]*s[6])
            b1=fx.math.fma(s[5],s[5],s[7]*s[7])
            acc = wave_sum_lane63((a0+b0)+(a1+b1))
            if lane == 63:
                part[wave] = acc
            fx.gpu.barrier()
            rstd = rsqrt_rn(
                _block_sum([part[i] for i in range_constexpr(NW)]),
                i32_eps.bitcast(fx.Float32),
            )
            y = [(s[i] * rstd) * (wv[i] + fx.Float32(1.0)) for i in range_constexpr(8)]
            # the standalone quantizer reads the bf16-rounded normed value
            yb = bf16x8(
                Vec.from_elements(
                    [pk_bf16(y[2 * i], y[2 * i + 1]) for i in range_constexpr(4)],
                    fx.Int32,
                )
            )
            buffer_ops.buffer_store(Vec.from_elements([pk_bf16(xv[2*i],xv[2*i+1]) for i in range_constexpr(4)],fx.Int32),residual_rsrc,off)
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
                    _code_off(row, tx // 2) // 4,
                    mask=(tx % 2) == 0,
                )
            else:
                buffer_ops.buffer_store(
                    Vec.from_elements(codes, fx.Int32),
                    qr,
                    (row * (H // 4)) + tx * 2,
                )
            if (tx % 4) == 0:
                # one lane per 32-column group stores its scale byte
                buffer_ops.buffer_store(
                    fx.Int32(sb.to(fx.Int32)).to(fx.Int8),
                    sr,
                    _scale_off(row, tx // 4),
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch_vocab_gather_gemma_mxfp8_h6144(
        arg_ids: fx.Int64, arg_peers: fx.Int64, shard_vocab: fx.Int32,
        arg_w: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_residual: fx.Int64,
        i64_q_bytes: fx.Int64,
        i64_s_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m,
            arg_ids, arg_peers, shard_vocab,
            arg_w,
            arg_q,
            arg_s,
            arg_residual,
            i64_q_bytes,
            i64_s_bytes,
            i32_eps,
        ).launch(
            grid=(fx.Int64(xcd_grid(i32_m)), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_vocab_gather_gemma_mxfp8_h6144.kernel_name = name
    launch_vocab_gather_gemma_mxfp8_h6144.compile_hints = PRELOAD_HINTS
    return launch_vocab_gather_gemma_mxfp8_h6144
