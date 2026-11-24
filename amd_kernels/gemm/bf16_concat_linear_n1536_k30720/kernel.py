# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The split-K slab reduce of every tier: the fp32 slabs summed in slab order, one bf16 rounding."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

N=1536


def compile_reduce(split):
    @flyc.kernel(name=f'bf16_linear_reduce_n1536_sk{split}',known_block_size=[256,1,1])
    def kernel(partials:fx.Tensor,out:fx.Tensor,rows:fx.Int32):
        t=fx.block_idx.x*256+fx.thread_idx.x
        valid=t<rows*(N//2)
        src=buffer_ops.create_buffer_resource(partials,num_records_bytes=fx.Int64(rows)*(N*split*4))
        dst=buffer_ops.create_buffer_resource(out,num_records_bytes=fx.Int64(rows)*(N*2))
        a,b=fx.Float32(0.0),fx.Float32(0.0)
        for s in range_constexpr(split):
            v=Vec(buffer_ops.buffer_load(src,s*rows*N+t*2,vec_width=2,dtype=fx.Float32,mask=valid))
            a=a+fx.Float32(v[0]);b=b+fx.Float32(v[1])
        pk=fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(a),fx.as_ir_value(b)))
        buffer_ops.buffer_store(pk,dst,t,mask=valid)
    @flyc.jit
    def launch_bf16_concat_reduce_n1536(partials:fx.Tensor,out:fx.Tensor,rows:fx.Int32,stream:fx.Stream):
        kernel(partials,out,rows).launch(grid=(fx.Int64((rows*(N//2)+255)//256),1,1),block=(256,1,1),stream=stream)
    launch_bf16_concat_reduce_n1536.kernel_name=f'bf16_linear_reduce_n1536_sk{split}'
    return launch_bf16_concat_reduce_n1536
