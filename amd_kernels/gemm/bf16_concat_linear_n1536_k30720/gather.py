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


def compile_gather(rpw):
    R=rpw
    name=f'bf16_gather_ranks4_n6144_c1536_r{R}'
    @flyc.kernel(name=name,known_block_size=[256,1,1])
    def kernel(b0:fx.Int64,b1:fx.Int64,b2:fx.Int64,b3:fx.Int64,
               counts:fx.Int64,out:fx.Tensor,rows:fx.Int32,me:fx.Int32,
               slot:fx.Int32,slot_bytes:fx.Int64):
        tx,g=fx.thread_idx.x,fx.block_idx.x
        bs=(b0,b1,b2,b3)
        mine=(me==0).select(b0,(me==1).select(b1,(me==2).select(b2,b3)))
        cr=buffer_ops.create_buffer_resource_from_addr(counts,num_records_bytes=GROUPS*4)
        epoch=fx.Int32(buffer_ops.buffer_load(cr,g,vec_width=1,dtype=fx.Int32,is_scalar=True))+1
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
    def launch_bf16_gather_n6144_c1536(b0:fx.Int64,b1:fx.Int64,b2:fx.Int64,b3:fx.Int64,
                                     counts:fx.Int64,out:fx.Tensor,rows:fx.Int32,me:fx.Int32,
                                     slot:fx.Int32,slot_bytes:fx.Int64,stream:fx.Stream):
        kernel(b0,b1,b2,b3,counts,out,rows,me,slot,slot_bytes).launch(
            grid=(fx.Int64((rows+R-1)//R),1,1),block=(256,1,1),stream=stream)
    launch_bf16_gather_n6144_c1536.kernel_name=name
    return launch_bf16_gather_n6144_c1536
