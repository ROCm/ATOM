# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The reduced output: routed slots 0..3, then the shared expert, summed in fp32 to bf16 (H 6144)."""
import functools
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from amd_kernels.warm_up import launcher


@functools.cache
@launcher
def compile_reduce():
    @flyc.kernel(name='moe_top4_shared_sum_h6144',known_block_size=[256,1,1])
    def kernel(P:fx.Tensor,S:fx.Tensor,O:fx.Tensor,m:fx.Int32):
        tx=fx.thread_idx.x
        row,chunk=fx.block_idx.x//3,fx.block_idx.x%3
        col=chunk*2048+tx*8
        pr=buffer_ops.create_buffer_resource(P,max_size=False,num_records_bytes=m*4*6144*2)
        sr=buffer_ops.create_buffer_resource(S,max_size=False,num_records_bytes=m*6144*2)
        out=buffer_ops.create_buffer_resource(O,max_size=False,num_records_bytes=m*6144*2)
        acc=Vec.filled(8,0.0,fx.Float32)
        for k in range_constexpr(4):
            v=Vec(buffer_ops.buffer_load(pr,((row*4+k)*6144+col)//2,vec_width=4,dtype=fx.Int32)).bitcast(fx.BFloat16).to(fx.Float32)
            acc=acc+v
        sh=Vec(buffer_ops.buffer_load(sr,(row*6144+col)//2,vec_width=4,dtype=fx.Int32)).bitcast(fx.BFloat16).to(fx.Float32)
        buffer_ops.buffer_store((acc+sh).to(fx.BFloat16).bitcast(fx.Int32),out,(row*6144+col)//2)
    @flyc.jit
    def launch_moe_slots_shared_reduce(P:fx.Tensor,S:fx.Tensor,O:fx.Tensor,m:fx.Int32,stream:fx.Stream):
        kernel(P,S,O,m).launch(grid=(m*3,1,1),block=(256,1,1),stream=stream)
    return launch_moe_slots_shared_reduce


def reduce_slots_shared(parts,shared):
    from .launch import _run_compiled
    out=torch.empty_like(shared)
    _run_compiled(compile_reduce(),parts.view(-1),shared.view(-1),out.view(-1),shared.shape[0],torch.cuda.current_stream())
    return out
