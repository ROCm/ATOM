# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Five bf16 inputs read in place, whole output-column shard, fp32 split-K slabs."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

N,K,H,NW=1536,30720,6144,4


def compile_concat_linear_unroll(mt,nt,split,unroll=2):
    assert K % (32*split)==0 and N%(16*nt*NW)==0
    steps=K//32//split
    nblocks=N//(16*nt*NW)
    assert steps % unroll == 0
    name=f'bf16_concat_linear_n1536_k30720_vec4_unroll{unroll}_mt{mt}_nt{nt}_sk{split}'
    def mfma(a,b,c):
        return Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(4,fx.Float32),[a.bitcast(fx.BFloat16),b.bitcast(fx.BFloat16),c,0,0,0]))

    @flyc.kernel(name=name,known_block_size=[256,1,1])
    def kernel(x0:fx.Int64,x1:fx.Int64,x2:fx.Int64,x3:fx.Int64,x4:fx.Int64,
               weight:fx.Int64,partials:fx.Int64,rows:fx.Int32,
               sx0:fx.Int32,sx1:fx.Int32,sx2:fx.Int32,sx3:fx.Int32,sx4:fx.Int32):
        tx,bid=fx.thread_idx.x,fx.block_idx.x
        lane=tx%64
        wave=fx.Int32(fx.rocdl.readfirstlane(T.i32,tx//64))
        s=bid%split
        nb=(bid//split)%nblocks
        mb=bid//(split*nblocks)
        wr=buffer_ops.create_buffer_resource_from_addr(weight,num_records_bytes=N*K*2)
        pr=buffer_ops.create_buffer_resource_from_addr(partials,num_records_bytes=fx.Int64(rows)*(N*split*4))
        def operands(kk,valid):
            tap=kk//H
            base=(tap==0).select(x0,(tap==1).select(x1,(tap==2).select(x2,(tap==3).select(x3,x4))))
            sx=(tap==0).select(sx0,(tap==1).select(sx1,(tap==2).select(sx2,(tap==3).select(sx3,sx4))))
            xr=buffer_ops.create_buffer_resource_from_addr(base,num_records_bytes=fx.Int64(rows)*fx.Int64(sx*2))
            av=[Vec(buffer_ops.buffer_load(wr,((nb*NW*nt+wave*nt+j)*(K//32)+kk//32)*256+lane*4,
                   vec_width=4,dtype=fx.Int32,mask=valid)) for j in range_constexpr(nt)]
            bv=[Vec(buffer_ops.buffer_load(xr,((mb*mt+i)*16+lane%16)*(sx//2)+(kk%H+lane//16*8)//2,
                   vec_width=4,dtype=fx.Int32,mask=valid&(((mb*mt+i)*16+lane%16)<rows))) for i in range_constexpr(mt)]
            return av+bv
        init=[Vec.filled(4,0.0,fx.Float32) for _ in range_constexpr(mt*nt)]
        init+=operands(s*steps*32,fx.Boolean(True))
        result=init
        for it,old in range(fx.Index(0),fx.Index(steps//unroll),fx.Index(1),init=[fx.as_ir_value(v) for v in init]):
            current=old
            for u in range_constexpr(unroll):
                step=fx.Int32(it)*unroll+u
                kk=(s*steps+step+1)*32
                safe=(step+1<steps).select(kk,fx.Int32(0))
                next_ops=operands(safe,step+1<steps)
                av=[Vec(current[mt*nt+j]) for j in range_constexpr(nt)]
                bv=[Vec(current[mt*nt+nt+i]) for i in range_constexpr(mt)]
                out=[mfma(av[j],bv[i],Vec(current[i*nt+j])) for i in range_constexpr(mt) for j in range_constexpr(nt)]
                current=[fx.as_ir_value(v) for v in out+next_ops]
            result=yield current
        for i in range_constexpr(mt):
            for j in range_constexpr(nt):
                # The four accumulator components occupy adjacent FP32 columns.
                # One aligned 16-byte store keeps the exact slab layout and bits.
                row=(mb*mt+i)*16+lane%16
                col=(nb*NW*nt+wave*nt+j)*16+(lane//16)*4
                buffer_ops.buffer_store(Vec(result[i*nt+j]),pr,(s*rows+row)*N+col,mask=row<rows)

    @flyc.jit
    def launch_bf16_concat_linear_unroll_n1536_k30720(x0:fx.Int64,x1:fx.Int64,x2:fx.Int64,x3:fx.Int64,x4:fx.Int64,
                                              weight:fx.Int64,partials:fx.Int64,rows:fx.Int32,
                                              sx0:fx.Int32,sx1:fx.Int32,sx2:fx.Int32,sx3:fx.Int32,sx4:fx.Int32,stream:fx.Stream):
        kernel(x0,x1,x2,x3,x4,weight,partials,rows,sx0,sx1,sx2,sx3,sx4).launch(
            grid=(fx.Int64(((rows+mt*16-1)//(mt*16))*nblocks*split),1,1),block=(256,1,1),stream=stream)
    launch_bf16_concat_linear_unroll_n1536_k30720.kernel_name=name
    return launch_bf16_concat_linear_unroll_n1536_k30720

