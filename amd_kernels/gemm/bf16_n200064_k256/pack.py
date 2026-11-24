# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gather activation rows into the MFMA input layout. Pads with exact zeros."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec


def compile_pack_bf16_rows(mt):
    MT=mt
    @flyc.kernel(name=f"pack_bf16_rows_k256_mt{MT}",known_block_size=[256,1,1])
    def kernel(arg_x:fx.Int64,arg_rows:fx.Int64,arg_out:fx.Int64,i32_n:fx.Int32,i32_nr:fx.Int32):
        tx=fx.thread_idx.x
        rr=buffer_ops.create_buffer_resource_from_addr(arg_rows,num_records_bytes=i32_n*8)
        xr=buffer_ops.create_buffer_resource_from_addr(arg_x,num_records_bytes=i32_nr*512)
        dst=buffer_ops.create_buffer_resource_from_addr(arg_out,num_records_bytes=MT*8192)
        for it in range_constexpr(MT*2):
            v=tx+it*256
            mb=v//512;ks=(v%512)//64;lane=v%64
            r=mb*16+lane%16
            src=fx.Int32(buffer_ops.buffer_load(rr,r*2,vec_width=1,dtype=fx.Int32))
            val=Vec(buffer_ops.buffer_load(xr,src*128+ks*16+(lane//16)*4,
                                           vec_width=4,dtype=fx.Int32,mask=r<i32_n))
            buffer_ops.buffer_store(val,dst,v*4)
    @flyc.jit
    def launch_pack_bf16_rows(arg_x:fx.Int64,arg_rows:fx.Int64,arg_out:fx.Int64,
                              i32_n:fx.Int32,i32_nr:fx.Int32,stream:fx.Stream):
        kernel(arg_x,arg_rows,arg_out,i32_n,i32_nr).launch(grid=(1,1,1),block=(256,1,1),stream=stream)
    return launch_pack_bf16_rows
