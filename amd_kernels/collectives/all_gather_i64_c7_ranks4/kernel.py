# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Single-workgroup seven-column int64 all-gather, two device-sequenced slots."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from amd_kernels.collectives.all_gather_sampler_i64_c8_ranks4.kernel import _read_flag,_sel4,SLOT_BYTES,SYS
NT=256

def compile_i64_c7_gather():
    @flyc.kernel(name='all_gather_i64_c7_ranks4',known_block_size=[NT,1,1])
    def kernel(arg_x:fx.Int64,arg_src:fx.Int64,arg_out:fx.Int64,
               p0:fx.Int64,p1:fx.Int64,p2:fx.Int64,p3:fx.Int64,
               f0:fx.Int64,f1:fx.Int64,f2:fx.Int64,f3:fx.Int64,counter:fx.Tensor,
               nlocal:fx.Int32,nout:fx.Int32,src_stride:fx.Int32,x_stride:fx.Int32,rank:fx.Int32):
        tx=fx.thread_idx.x
        seq=fx.Int32(counter[0])+1;slot=fx.Int64((seq-1)&1)*SLOT_BYTES
        dst=buffer_ops.create_buffer_resource_from_addr(_sel4(rank,p0,p1,p2,p3)+slot,num_records_bytes=SLOT_BYTES)
        if tx<nlocal:
            xr=buffer_ops.create_buffer_resource_from_addr(arg_x,num_records_bytes=((nlocal-1)*x_stride+7)*8)
            vals=[Vec(buffer_ops.buffer_load(xr,tx*x_stride*2+j*4,vec_width=4,dtype=fx.Int32)) for j in range_constexpr(3)]
            tail=Vec(buffer_ops.buffer_load(xr,tx*x_stride*2+12,vec_width=2,dtype=fx.Int32))
            for j in range_constexpr(3):buffer_ops.buffer_store(vals[j],dst,tx*14+j*4,cache_modifier=SYS)
            buffer_ops.buffer_store(tail,dst,tx*14+12,cache_modifier=SYS)
        fx.rocdl.s_waitcnt(vmcnt=0);fx.gpu.barrier()
        if tx==0:
            fr=buffer_ops.create_buffer_resource_from_addr(_sel4(rank,f0,f1,f2,f3),num_records_bytes=4)
            buffer_ops.buffer_store(seq,fr,0,cache_modifier=SYS)
        fx.rocdl.s_waitcnt(vmcnt=0);fx.gpu.barrier()
        if tx<4:
            flag=_sel4(tx,f0,f1,f2,f3);val=_read_flag(flag)
            while (val-seq)<0:val=_read_flag(flag)
        fx.gpu.barrier();llvm.FenceOp(llvm.AtomicOrdering.acquire,syncscope='')
        if tx<nout:
            ptr=fx.inttoptr(fx.PointerType.get(T.i64,address_space=fx.AddressSpace.Global,alignment=8),arg_src)
            src=fx.Int64(ptr[tx]).to(fx.Int32);peer=src//src_stride;local=src%src_stride
            rr=buffer_ops.create_buffer_resource_from_addr(_sel4(peer,p0,p1,p2,p3)+slot,num_records_bytes=SLOT_BYTES)
            ot=buffer_ops.create_buffer_resource_from_addr(arg_out,num_records_bytes=nout*56)
            vals=[Vec(buffer_ops.buffer_load(rr,local*14+j*4,vec_width=4,dtype=fx.Int32,cache_modifier=SYS)) for j in range_constexpr(3)]
            tail=Vec(buffer_ops.buffer_load(rr,local*14+12,vec_width=2,dtype=fx.Int32,cache_modifier=SYS))
            for j in range_constexpr(3):buffer_ops.buffer_store(vals[j],ot,tx*14+j*4)
            buffer_ops.buffer_store(tail,ot,tx*14+12)
        fx.rocdl.s_waitcnt(vmcnt=0);fx.gpu.barrier()
        if tx==0:counter[0]=seq
    @flyc.jit
    def launch_all_gather_i64_c7_ranks4(arg_x:fx.Int64,arg_src:fx.Int64,arg_out:fx.Int64,
               p0:fx.Int64,p1:fx.Int64,p2:fx.Int64,p3:fx.Int64,
               f0:fx.Int64,f1:fx.Int64,f2:fx.Int64,f3:fx.Int64,counter:fx.Tensor,
               nlocal:fx.Int32,nout:fx.Int32,src_stride:fx.Int32,x_stride:fx.Int32,rank:fx.Int32,stream:fx.Stream):
        kernel(arg_x,arg_src,arg_out,p0,p1,p2,p3,f0,f1,f2,f3,counter,nlocal,nout,src_stride,x_stride,rank).launch(
            grid=(1,1,1),block=(NT,1,1),stream=stream)
    return launch_all_gather_i64_c7_ranks4
