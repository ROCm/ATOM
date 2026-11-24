# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One workgroup packs, publishes, waits and directly unpacks peer rows.

Two IPC slots bound peer lead/lag. System stores and loads bypass the dirty
XCD L2, as in all_to_all_bf16_n50016_ranks4. The uncached sequence flags use
atomic read/add(0); all token/count/NaN outputs use ordinary stores.
Two slots suffice: a rank cannot finish call c + 1 before every peer published c + 1, which follows that peer's
reads of slot c % 2. The flag wait compares (flag - seq) < 0 in int32, valid across the counter's wrap.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

NT=256
FIELDS=11
SLOT_BYTES=256*FIELDS*8
SYS=0x11

def _ptr(addr, dtype):
    return fx.inttoptr(fx.PointerType.get(dtype,address_space=fx.AddressSpace.Global,alignment=8 if dtype==T.i64 else 4),addr)

def _read_flag(addr):
    return fx.Int32(llvm.AtomicRMWOp(llvm.AtomicBinOp.add,_ptr(addr,T.i32).llvm_ptr,fx.Int32(0).ir_value(),
                                  llvm.AtomicOrdering.monotonic,syncscope='',alignment=4).result)

def _sel4(p,a,b,c,d):
    return (p==0).select(a,(p==1).select(b,(p==2).select(c,d)))

def compile_sampler_gather():
    @flyc.kernel(name='all_gather_sampler_i64_c8_ranks4',known_block_size=[NT,1,1])
    def kernel(arg_tok:fx.Int64,arg_ns:fx.Int64,arg_nr:fx.Int64,arg_nan:fx.Int64,arg_cu:fx.Int64,
               arg_src:fx.Int64,arg_ot:fx.Int64,arg_os:fx.Int64,arg_or:fx.Int64,arg_on:fx.Int64,
               p0:fx.Int64,p1:fx.Int64,p2:fx.Int64,p3:fx.Int64,
               f0:fx.Int64,f1:fx.Int64,f2:fx.Int64,f3:fx.Int64,counter:fx.Tensor,
               nlocal:fx.Int32,nout:fx.Int32,src_stride:fx.Int32,tok_stride:fx.Int32,ncols:fx.Int32,
               rank:fx.Int32,has_nan:fx.Int32):
        tx=fx.thread_idx.x
        seq=fx.Int32(counter[0])+1
        slot=fx.Int64((seq-1)&1)*SLOT_BYTES
        own=_sel4(rank,p0,p1,p2,p3)
        dst=buffer_ops.create_buffer_resource_from_addr(own+slot,num_records_bytes=SLOT_BYTES)
        if tx<nlocal:
            tr=buffer_ops.create_buffer_resource_from_addr(arg_tok,num_records_bytes=((nlocal-1)*tok_stride+ncols)*8)
            for col in range_constexpr(8):
                tok=Vec.filled(2,0,fx.Int32)
                if col<ncols:tok=Vec(buffer_ops.buffer_load(tr,(tx*tok_stride+col)*2,vec_width=2,dtype=fx.Int32))
                buffer_ops.buffer_store(tok,dst,(tx*FIELDS+col)*2,cache_modifier=SYS)
            ns=fx.Int32(_ptr(arg_ns,T.i32)[tx]).to(fx.Int64)
            nr=fx.Int32(_ptr(arg_nr,T.i32)[tx]).to(fx.Int64)
            nc=fx.Int64(0)
            if has_nan!=0:
                cu=_ptr(arg_cu,T.i32)
                lo=fx.Int32(cu[tx]);hi=fx.Int32(cu[tx+1])
                for j in range(lo,hi):nc=nc+fx.Int32(_ptr(arg_nan,T.i32)[fx.Int32(j)]).to(fx.Int64)
            for col,val in ((8,ns),(9,nr),(10,nc)):
                buffer_ops.buffer_store(Vec.from_elements([val.to(fx.Int32),(val>>32).to(fx.Int32)],fx.Int32),
                                        dst,(tx*FIELDS+col)*2,cache_modifier=SYS)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx==0:
            fr=buffer_ops.create_buffer_resource_from_addr(_sel4(rank,f0,f1,f2,f3),num_records_bytes=4)
            buffer_ops.buffer_store(seq,fr,0,cache_modifier=SYS)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx<4:
            flag=_sel4(tx,f0,f1,f2,f3)
            val=_read_flag(flag)
            while (val-seq)<0:val=_read_flag(flag)
        fx.gpu.barrier()
        llvm.FenceOp(llvm.AtomicOrdering.acquire,syncscope='')
        if tx<nout:
            src=fx.Int64(_ptr(arg_src,T.i64)[tx]).to(fx.Int32)
            peer=src//src_stride;local=src%src_stride
            rp=_sel4(peer,p0,p1,p2,p3)+slot
            rr=buffer_ops.create_buffer_resource_from_addr(rp,num_records_bytes=SLOT_BYTES)
            ot=buffer_ops.create_buffer_resource_from_addr(arg_ot,num_records_bytes=nout*64)
            vals=[Vec(buffer_ops.buffer_load(rr,local*FIELDS*2+j*4,vec_width=4,dtype=fx.Int32,cache_modifier=SYS)) for j in range_constexpr(5)]
            for j in range_constexpr(4):buffer_ops.buffer_store(vals[j],ot,tx*16+j*4)
            _ptr(arg_os,T.i32)[tx]=fx.Int32(vals[4][0])
            _ptr(arg_or,T.i32)[tx]=fx.Int32(vals[4][2])
            if has_nan!=0:
                _ptr(arg_on,T.i32)[tx]=fx.Int32(buffer_ops.buffer_load(rr,local*FIELDS*2+20,vec_width=1,dtype=fx.Int32,cache_modifier=SYS))
        # Retire this slot before the same stream starts the next call.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx==0:counter[0]=seq
    @flyc.jit
    def launch_all_gather_sampler_i64_c8_ranks4(
               arg_tok:fx.Int64,arg_ns:fx.Int64,arg_nr:fx.Int64,arg_nan:fx.Int64,arg_cu:fx.Int64,
               arg_src:fx.Int64,arg_ot:fx.Int64,arg_os:fx.Int64,arg_or:fx.Int64,arg_on:fx.Int64,
               p0:fx.Int64,p1:fx.Int64,p2:fx.Int64,p3:fx.Int64,
               f0:fx.Int64,f1:fx.Int64,f2:fx.Int64,f3:fx.Int64,counter:fx.Tensor,
               nlocal:fx.Int32,nout:fx.Int32,src_stride:fx.Int32,tok_stride:fx.Int32,ncols:fx.Int32,
               rank:fx.Int32,has_nan:fx.Int32,stream:fx.Stream):
        kernel(arg_tok,arg_ns,arg_nr,arg_nan,arg_cu,arg_src,arg_ot,arg_os,arg_or,arg_on,
               p0,p1,p2,p3,f0,f1,f2,f3,counter,nlocal,nout,src_stride,tok_stride,ncols,rank,has_nan).launch(
            grid=(1,1,1),block=(NT,1,1),stream=stream)
    return launch_all_gather_sampler_i64_c8_ranks4
