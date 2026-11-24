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


def compile_concat_linear_staged_a(mt,nt,split,unroll=2):
    assert mt%4==0 and unroll==2
    assert K % (32*split)==0 and N%(16*nt*NW)==0
    steps=K//32//split
    assert steps%unroll==0 and (H//32)%unroll==0
    nblocks=N//(16*nt*NW)
    name=f'bf16_concat_linear_n1536_k30720_vec4_staged_a_u{unroll}_mt{mt}_nt{nt}_sk{split}'
    def mfma(a,b,c):
        return Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(4,fx.Float32),[a.bitcast(fx.BFloat16),b.bitcast(fx.BFloat16),c,0,0,0]))

    @fx.struct
    class Shared:
        a:fx.Array[fx.Int32,2*mt*64*4,16]

    @flyc.kernel(name=name,known_block_size=[256,1,1])
    def kernel(x0:fx.Int64,x1:fx.Int64,x2:fx.Int64,x3:fx.Int64,x4:fx.Int64,
               weight:fx.Int64,partials:fx.Int64,rows:fx.Int32,
               sx0:fx.Int32,sx1:fx.Int32,sx2:fx.Int32,sx3:fx.Int32,sx4:fx.Int32):
        tx,bid=fx.thread_idx.x,fx.block_idx.x
        sm=fx.SharedAllocator().allocate(Shared).peek()
        av16=fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32,sm.a.ptr),fx.make_layout(2*mt*64*4,1)),fx.make_layout(4,1))
        atom=fx.make_copy_atom(fx.UniversalCopy128b(),fx.Int32)
        lane=tx%64
        wave=fx.Int32(fx.rocdl.readfirstlane(T.i32,tx//64))
        s=bid%split
        nb=(bid//split)%nblocks
        mb=bid//(split*nblocks)
        wr=buffer_ops.create_buffer_resource_from_addr(weight,num_records_bytes=N*K*2)
        pr=buffer_ops.create_buffer_resource_from_addr(partials,num_records_bytes=fx.Int64(rows)*(N*split*4))
        bases=(x0,x1,x2,x3,x4)
        strides=(sx0,sx1,sx2,sx3,sx4)
        acc=[Vec.filled(4,0.0,fx.Float32) for _ in range_constexpr(mt*nt)]
        for tap in range_constexpr(5):
            sx=strides[tap]
            xr=buffer_ops.create_buffer_resource_from_addr(bases[tap],num_records_bytes=fx.Int64(rows)*fx.Int64(sx*2))
            first=(s*steps>tap*(H//32)).select(s*steps,fx.Int32(tap*(H//32)))
            last=((s+1)*steps<(tap+1)*(H//32)).select((s+1)*steps,fx.Int32((tap+1)*(H//32)))
            span=(last>first).select(last-first,fx.Int32(0))
            def operands(step,valid):
                kk=step*32
                av=[Vec(buffer_ops.buffer_load(wr,((nb*NW*nt+wave*nt+j)*(K//32)+step)*256+lane*4,
                       vec_width=4,dtype=fx.Int32,mask=valid)) for j in range_constexpr(nt)]
                local=(kk-tap*H)
                bv=[Vec(buffer_ops.buffer_load(xr,((mb*mt*16)+(tx//4)+i*64)*(sx//2)+(local+(tx%4)*8)//2,
                       vec_width=4,dtype=fx.Int32,mask=valid&(((mb*mt*16)+(tx//4)+i*64)<rows))) for i in range_constexpr(mt//4)]
                return av+bv
            init=acc+operands(first,span>0)
            result=init
            for it,old in range(fx.Index(0),fx.Index(span//unroll),fx.Index(1),init=[fx.as_ir_value(v) for v in init]):
                current=old
                for u in range_constexpr(unroll):
                    step=fx.Int32(it)*unroll+u
                    safe=(step+1<span).select(first+step+1,first)
                    next_ops=operands(safe,step+1<span)
                    av=[Vec(current[mt*nt+j]) for j in range_constexpr(nt)]
                    # Two LDS slots retain each wave's reads until the following
                    # write barrier. Every tap span is even, so segments restart in slot 0.
                    for j in range_constexpr(mt//4):
                        row=tx//4+j*64
                        piece=(row//16)*64+(tx%4)*16+row%16
                        tmp=fx.make_rmem_tensor(fx.make_layout(4,1),fx.Int32)
                        tmp.store(Vec(current[mt*nt+nt+j]))
                        fx.copy(atom,tmp,fx.slice(av16,(None,u*mt*64+piece)))
                    fx.gpu.barrier()
                    bv=[]
                    for i in range_constexpr(mt):
                        tmp=fx.make_rmem_tensor(fx.make_layout(4,1),fx.Int32)
                        fx.copy(atom,fx.slice(av16,(None,(u*mt+i)*64+lane)),tmp)
                        bv.append(Vec(tmp.load()))
                    out=[mfma(av[j],bv[i],Vec(current[i*nt+j])) for i in range_constexpr(mt) for j in range_constexpr(nt)]
                    current=[fx.as_ir_value(v) for v in out+next_ops]
                result=yield current
            acc=[Vec(result[j]) for j in range_constexpr(mt*nt)]
        for i in range_constexpr(mt):
            for j in range_constexpr(nt):
                # The four accumulator components occupy adjacent FP32 columns.
                # One aligned 16-byte store keeps the exact slab layout and bits.
                row=(mb*mt+i)*16+lane%16
                col=(nb*NW*nt+wave*nt+j)*16+(lane//16)*4
                buffer_ops.buffer_store(acc[i*nt+j],pr,(s*rows+row)*N+col,mask=row<rows)

    @flyc.jit
    def launch_bf16_concat_linear_staged_a_n1536_k30720(x0:fx.Int64,x1:fx.Int64,x2:fx.Int64,x3:fx.Int64,x4:fx.Int64,
                                              weight:fx.Int64,partials:fx.Int64,rows:fx.Int32,
                                              sx0:fx.Int32,sx1:fx.Int32,sx2:fx.Int32,sx3:fx.Int32,sx4:fx.Int32,stream:fx.Stream):
        kernel(x0,x1,x2,x3,x4,weight,partials,rows,sx0,sx1,sx2,sx3,sx4).launch(
            grid=(fx.Int64(((rows+mt*16-1)//(mt*16))*nblocks*split),1,1),block=(256,1,1),stream=stream)
    launch_bf16_concat_linear_staged_a_n1536_k30720.kernel_name=name
    return launch_bf16_concat_linear_staged_a_n1536_k30720

