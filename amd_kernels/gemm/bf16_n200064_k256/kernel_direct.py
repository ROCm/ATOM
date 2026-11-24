# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four waves, one per SIMD. Original row-major weights, packed activations.

Each wave owns 16 vocabulary rows, sharing each 256-K weight row over 16/32/64
activation rows. FP32 MFMA accumulators round once to bf16 through typed casts;
the inline-asm cvt wrapper hides the MFMA read hazard from LLVM.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T, Vector as Vec
from .reference import N as V

def compile_bf16_gemm_packed_rows(mt):
    MT=mt
    @flyc.kernel(name=f'bf16_gemm_packed_rows_n200064_k256_mt{MT}_direct',known_block_size=[256,1,1])
    def kernel(arg_x:fx.Int64,arg_w:fx.Int64,arg_y:fx.Int64,i32_n:fx.Int32):
        tx,bid=fx.thread_idx.x,fx.block_idx.x
        lane=tx%64
        wave=fx.Int32(fx.rocdl.readfirstlane(T.i32,tx//64))
        n0=bid*64+wave*16
        wi=n0+lane%16
        xr=buffer_ops.create_buffer_resource_from_addr(arg_x,num_records_bytes=MT*8192)
        wr=buffer_ops.create_buffer_resource_from_addr(arg_w,num_records_bytes=V*256*2)
        yr=buffer_ops.create_buffer_resource_from_addr(arg_y,num_records_bytes=i32_n*V*2)
        wv=[Vec(buffer_ops.buffer_load(wr,(wi*128+k*16+(lane//16)*4),vec_width=4,dtype=fx.Int32)) for k in range_constexpr(8)]
        xv=[]
        for m in range_constexpr(MT):
            xv.append([Vec(buffer_ops.buffer_load(xr,m*2048+k*256+lane*4,vec_width=4,dtype=fx.Int32,
                                                 mask=(lane%16+m*16)<i32_n)) for k in range_constexpr(8)])
        acc=[Vec.filled(4,0.0,fx.Float32) for m in range_constexpr(MT)]
        for k in range_constexpr(8):
            for m in range_constexpr(MT):
                acc[m]=Vec(fx.rocdl.mfma_f32_16x16x32_bf16(Vec.make_type(4,fx.Float32),
                    [wv[k].bitcast(fx.BFloat16),xv[m][k].bitcast(fx.BFloat16),acc[m],0,0,0]))
        for m in range_constexpr(MT):
            r=lane%16+m*16
            for e in range_constexpr(2):
                pk=fx.Int32(Vec.from_elements([fx.Float32(acc[m][2*e]),fx.Float32(acc[m][2*e+1])],fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])
                off=r*(V//2)+(n0+(lane//16)*4+e*2)//2
                buffer_ops.buffer_store(pk,yr,off,mask=r<i32_n)
    @flyc.jit
    def launch_bf16_gemm_packed_rows_direct(arg_x:fx.Int64,arg_w:fx.Int64,arg_y:fx.Int64,
                              i32_n:fx.Int32,stream:fx.Stream):
        kernel(arg_x,arg_w,arg_y,i32_n).launch(grid=(V//64,1,1),block=(256,1,1),stream=stream)
    return launch_bf16_gemm_packed_rows_direct
