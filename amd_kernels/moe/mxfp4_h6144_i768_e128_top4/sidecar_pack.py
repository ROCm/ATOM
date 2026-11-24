# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The correction mask of the lossless MXFP4 sidecar: one bit per value, set where FP4 rounding of the shared MXFP8
value (with the source FP4 scale) differs from the direct MXFP4 code.

That happens only where FP8 rounding lands on an FP4 midpoint; the sign, negative zero included, stays in the FP8 code.
Both operands come straight from finite bf16: the transport changes, never the quantization rule.
"""
import functools
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from amd_kernels.common import buffer_ops
from .common.launch import _run_compiled
from .mxfp4_from_mxfp8 import _e4m3x2
from .quant import _cvt_pk_fp4
from amd_kernels.warm_up import launcher

@functools.cache
@launcher
def compile_sidecar_pack():
    name='mxfp4_mxfp8_sidecar_pack_h6144'
    @flyc.kernel(name=name,known_block_size=[256,1,1])
    def kernel(Q8:fx.Tensor,S8:fx.Tensor,S4:fx.Tensor,Q4:fx.Tensor,MASK:fx.Tensor,n:fx.Int32):
        b=fx.block_idx.x*256+fx.thread_idx.x
        q8r=buffer_ops.create_buffer_resource(Q8,max_size=False,num_records_bytes=n*32)
        s8r=buffer_ops.create_buffer_resource(S8,max_size=False,num_records_bytes=n)
        s4r=buffer_ops.create_buffer_resource(S4,max_size=False,num_records_bytes=n)
        q4r=buffer_ops.create_buffer_resource(Q4,max_size=False,num_records_bytes=n*16)
        mr=buffer_ops.create_buffer_resource(MASK,max_size=False,num_records_bytes=n*4)
        e8=fx.Int32(buffer_ops.buffer_load(s8r,b,vec_width=1,dtype=fx.Int8)).to(fx.Int32)&255
        e4=fx.Int32(buffer_ops.buffer_load(s4r,b,vec_width=1,dtype=fx.Int8)).to(fx.Int32)&255
        # unscaled FP8 values times the FP4 / FP8 scale ratio: midpoint comparisons stay normal for tiny bf16 values
        ratio=((e4-e8+127)<<23).bitcast(fx.Float32)
        mask=fx.Int32(0)
        for d in range_constexpr(4):
            words=Vec(buffer_ops.buffer_load(q8r,b*8+d*2,vec_width=2,dtype=fx.Int32))
            vals=[]
            for w in range_constexpr(2):
                for hi in (False,True):
                    vals.extend(_e4m3x2(fx.Int32(words[w]),fx.Float32(1.0),hi))
            candidate=fx.Int32(0)
            for j in range_constexpr(4):
                candidate=_cvt_pk_fp4(candidate,vals[j*2],vals[j*2+1],ratio,j)
            direct=fx.Int32(buffer_ops.buffer_load(q4r,b*4+d,vec_width=1,dtype=fx.Int32))
            for j in range_constexpr(8):
                diff=((direct>>(j*4))&15)!=((candidate>>(j*4))&15)
                mask=mask|diff.select(fx.Int32(1<<(d*8+j)),fx.Int32(0))
        buffer_ops.buffer_store(mask,mr,b)
    @flyc.jit
    def launch_mxfp4_mxfp8_sidecar_pack_h6144(Q8:fx.Tensor,S8:fx.Tensor,S4:fx.Tensor,Q4:fx.Tensor,MASK:fx.Tensor,n:fx.Int32,stream:fx.Stream):
        kernel(Q8,S8,S4,Q4,MASK,n).launch(grid=((n+255)//256,1,1),block=(256,1,1),stream=stream)
    return launch_mxfp4_mxfp8_sidecar_pack_h6144

def pack_mxfp4_sidecar(q4,s4,q8,s8):
    n=s4.numel();m=q4.shape[0]
    mask=torch.empty((m,q4.shape[1]//4),dtype=torch.uint8,device=q4.device)
    _run_compiled(compile_sidecar_pack(),q8.view(-1),s8.view(-1),s4.view(-1),q4.view(-1),mask.view(-1),n,torch.cuda.current_stream())
    return mask
