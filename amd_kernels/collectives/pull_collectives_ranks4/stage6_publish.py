# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Stage six tensor fields and publish their completed IPC slot in one launch."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir.dialects import llvm as _llvm
from amd_kernels.common import buffer_ops
from .kernel import SYS, _flag_add

NT = 256
WGS = 64
UNROLL = 8
STEP = NT * 4 * UNROLL


def _select_field(bid,b1,b2,b3,b4,b5,x0,x1,x2,x3,x4,x5):
    return (bid<b1).select(x0,(bid<b2).select(x1,(bid<b3).select(x2,
        (bid<b4).select(x3,(bid<b5).select(x4,x5)))))


def compile_stage6_publish(workgroups=WGS):
    assert workgroups >= 8 and workgroups % 8 == 0
    name = f'stage6_publish_leader_ranks4_wg{workgroups}'

    @flyc.kernel(name=name,known_block_size=[NT,1,1])
    def kernel(arg_s0:fx.Int64,arg_s1:fx.Int64,arg_s2:fx.Int64,arg_s3:fx.Int64,arg_s4:fx.Int64,arg_s5:fx.Int64,
               off0:fx.Int64,off1:fx.Int64,off2:fx.Int64,off3:fx.Int64,off4:fx.Int64,off5:fx.Int64,
               n0:fx.Int32,n1:fx.Int32,n2:fx.Int32,n3:fx.Int32,n4:fx.Int32,n5:fx.Int32,
               b1:fx.Int32,b2:fx.Int32,b3:fx.Int32,b4:fx.Int32,b5:fx.Int32,
               pool:fx.Int64,stride:fx.Int64,counter:fx.Tensor,flag:fx.Int64):
        tx,bid=fx.thread_idx.x,fx.block_idx.x
        seq=fx.Int32(counter[0])
        src_addr=_select_field(bid,b1,b2,b3,b4,b5,arg_s0,arg_s1,arg_s2,arg_s3,arg_s4,arg_s5)
        field_off=_select_field(bid,b1,b2,b3,b4,b5,off0,off1,off2,off3,off4,off5)
        nbytes=_select_field(bid,b1,b2,b3,b4,b5,n0,n1,n2,n3,n4,n5)
        first=_select_field(bid,b1,b2,b3,b4,b5,fx.Int32(0),b1,b2,b3,b4,b5)
        n_wg=_select_field(bid,b1,b2,b3,b4,b5,b1,b2-b1,b3-b2,b4-b3,b5-b4,workgroups-b5)
        src=buffer_ops.create_buffer_resource_from_addr(src_addr,num_records_bytes=nbytes)
        dst=buffer_ops.create_buffer_resource_from_addr(pool+fx.Int64(seq&1)*stride+field_off,num_records_bytes=nbytes)
        off=(bid-first)*STEP+tx*4
        while off < nbytes//4:
            vals=[]
            for u in range_constexpr(UNROLL):
                o=off+u*NT*4
                vals.append(Vec(buffer_ops.buffer_load(src,o,vec_width=4,dtype=fx.Int32,mask=o<nbytes//4)))
            for u in range_constexpr(UNROLL):
                o=off+u*NT*4
                buffer_ops.buffer_store(vals[u],dst,o,mask=o<nbytes//4)
            off=off+n_wg*STEP
        # gfx950 counts stores in vmcnt. All waves retire their writes before
        # lane 0 releases this workgroup's XCD and contributes its ticket.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        if tx == 0:
            _llvm.FenceOp(_llvm.AtomicOrdering.release,syncscope='')
            ticket=_flag_add(flag+4,1)
            if ticket == workgroups-1:
                _flag_add(flag+4,-workgroups)
                fr=buffer_ops.create_buffer_resource_from_addr(flag,num_records_bytes=64)
                counter[0]=seq+1
                buffer_ops.buffer_store(seq+1,fr,0,cache_modifier=SYS)

    @flyc.jit
    def launch_stage6_publish_ranks4(arg_s0:fx.Int64,arg_s1:fx.Int64,arg_s2:fx.Int64,arg_s3:fx.Int64,arg_s4:fx.Int64,arg_s5:fx.Int64,
                                    off0:fx.Int64,off1:fx.Int64,off2:fx.Int64,off3:fx.Int64,off4:fx.Int64,off5:fx.Int64,
                                    n0:fx.Int32,n1:fx.Int32,n2:fx.Int32,n3:fx.Int32,n4:fx.Int32,n5:fx.Int32,
                                    b1:fx.Int32,b2:fx.Int32,b3:fx.Int32,b4:fx.Int32,b5:fx.Int32,
                                    pool:fx.Int64,stride:fx.Int64,counter:fx.Tensor,flag:fx.Int64,stream:fx.Stream):
        kernel(arg_s0,arg_s1,arg_s2,arg_s3,arg_s4,arg_s5,off0,off1,off2,off3,off4,off5,
               n0,n1,n2,n3,n4,n5,b1,b2,b3,b4,b5,pool,stride,counter,flag).launch(
                   grid=(workgroups,1,1),block=(NT,1,1),stream=stream)
    launch_stage6_publish_ranks4.kernel_name=name
    return launch_stage6_publish_ranks4
