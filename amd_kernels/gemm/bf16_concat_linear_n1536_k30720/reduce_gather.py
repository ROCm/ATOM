# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gather four bf16 column shards into [rows,6144], with batched epoch handshakes."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from .flags import _flag_store,_spin

LINE=32
GROUPS=256
DATA_OFF=LINE*GROUPS*4
ROW_BYTES=1536*2
SYS=0x11


def compile_reduce_gather(rpw,split):
    R=rpw
    name=f'bf16_reduce_gather_ranks4_n6144_c1536_r{R}_sk{split}'
    @flyc.kernel(name=name,known_block_size=[256,1,1])
    def kernel(partials:fx.Tensor,b0:fx.Int64,b1:fx.Int64,b2:fx.Int64,b3:fx.Int64,
               counts:fx.Int64,out:fx.Tensor,rows:fx.Int32,me:fx.Int32,
               slot:fx.Int32,slot_bytes:fx.Int64):
        tx,g=fx.thread_idx.x,fx.block_idx.x
        bs=(b0,b1,b2,b3)
        mine=(me==0).select(b0,(me==1).select(b1,(me==2).select(b2,b3)))
        cr=buffer_ops.create_buffer_resource_from_addr(counts,num_records_bytes=GROUPS*4)
        epoch=fx.Int32(buffer_ops.buffer_load(cr,g,vec_width=1,dtype=fx.Int32,is_scalar=True))+1
        # Each group owns the same rows on every rank. Finish the local
        # fixed-order fp32 slab sum into uncached staging before publication.
        pr=buffer_ops.create_buffer_resource(partials,num_records_bytes=fx.Int64(rows)*(1536*split*4))
        for r in range(fx.Int32(0),fx.Int32(R),fx.Int32(1)):
            row=g*R+r
            live=(row<rows)&(tx<192)
            sums=[fx.Float32(0.0) for _ in range_constexpr(8)]
            for slab in range_constexpr(split):
                a=Vec(buffer_ops.buffer_load(pr,(slab*rows+row)*1536+tx*8,vec_width=4,dtype=fx.Float32,mask=live))
                b=Vec(buffer_ops.buffer_load(pr,(slab*rows+row)*1536+tx*8+4,vec_width=4,dtype=fx.Float32,mask=live))
                sums=[sums[j]+fx.Float32(a[j] if j<4 else b[j-4]) for j in range_constexpr(8)]
            packed=[fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(sums[2*j]),fx.as_ir_value(sums[2*j+1]))) for j in range_constexpr(4)]
            sr=buffer_ops.create_buffer_resource_from_addr(mine+DATA_OFF+fx.Int64(slot)*slot_bytes+fx.Int64(row)*ROW_BYTES,num_records_bytes=ROW_BYTES)
            buffer_ops.buffer_store(Vec.from_elements(packed,fx.Int32),sr,tx*4,mask=live)
        # A flag cannot pass any wave's data stores. The allocation is uncached,
        # so draining every wave plus a workgroup barrier publishes all rows.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx<3:
            peer=(me+tx+1)%4
            ptr=(peer==0).select(b0,(peer==1).select(b1,(peer==2).select(b2,b3)))
            _flag_store(ptr+fx.Int64((g*LINE+me)*4),epoch)
            _spin(mine+fx.Int64((g*LINE+peer)*4),epoch)
        fx.gpu.barrier()
        dst=buffer_ops.create_buffer_resource(out,num_records_bytes=fx.Int64(rows)*(6144*2))
        for r in range(fx.Int32(0),fx.Int32(R),fx.Int32(1)):
            row=g*R+r
            live=(row<rows)&(tx<192)
            vals=[]
            for src in range_constexpr(4):
                rr=buffer_ops.create_buffer_resource_from_addr(bs[src]+DATA_OFF+fx.Int64(slot)*slot_bytes+fx.Int64(row)*ROW_BYTES,num_records_bytes=ROW_BYTES)
                vals.append(Vec(buffer_ops.buffer_load(rr,tx*4,vec_width=4,dtype=fx.Int32,mask=live,cache_modifier=SYS)))
            for src in range_constexpr(4):
                buffer_ops.buffer_store(vals[src],dst,row*3072+src*768+tx*4,mask=live)
        buffer_ops.buffer_store(epoch,cr,g,mask=tx==0)
    @flyc.jit
    def launch_bf16_reduce_gather_n6144_c1536(partials:fx.Tensor,b0:fx.Int64,b1:fx.Int64,b2:fx.Int64,b3:fx.Int64,
                                     counts:fx.Int64,out:fx.Tensor,rows:fx.Int32,me:fx.Int32,
                                     slot:fx.Int32,slot_bytes:fx.Int64,stream:fx.Stream):
        kernel(partials,b0,b1,b2,b3,counts,out,rows,me,slot,slot_bytes).launch(
            grid=(fx.Int64((rows+R-1)//R),1,1),block=(256,1,1),stream=stream)
    launch_bf16_reduce_gather_n6144_c1536.kernel_name=name
    return launch_bf16_reduce_gather_n6144_c1536
