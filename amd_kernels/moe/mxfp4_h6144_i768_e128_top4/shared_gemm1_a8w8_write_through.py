# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared gate/up GEMM (``StoreSharedFp32``'s bits) for the composed routed + shared launch: one tile per workgroup,
codes and scales stored with cache policy 16 (write-through) for the down-GEMM consumers of the same launch."""
import functools
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from amd_kernels.common import buffer_ops
from .shared_dense import StoreSharedFp32
from .mxfp8.shared_dense_4wave import _scale_off, compile_mxfp8_gemm_4w_gateup
from .shared_quant import _fp8, _e8m0_fp8
from amd_kernels.warm_up import launcher

class StoreSharedFp32WriteThrough(StoreSharedFp32):
    def thunks_rm(self,*args):
        return [],[]
    def _quant_store(self,values,base_row,base_col,ti):
        row=base_row+ti*16+self.r
        for h in range_constexpr(self.n_b//2):
            v=values[h*8:h*8+8]
            amax=fx.math.absf(v[0])
            for j in range_constexpr(1,8):
                z=fx.math.absf(v[j]);amax=(amax>z).select(amax,z)
            for sh in range_constexpr(2):
                z=amax.shuffle_xor(16<<sh,64);amax=(amax>z).select(amax,z)
            e=_e8m0_fp8(amax);scale=(e<<23).bitcast(fx.Float32)
            col=base_col+32*h+self.qg*4
            for j in range_constexpr(2):
                pk=_fp8(fx.Int32(0),v[j*4],v[j*4+1],scale,False)
                pk=_fp8(pk,v[j*4+2],v[j*4+3],scale,True)
                buffer_ops.buffer_store(pk,self.q_rsrc,(row*self.out_n+col+j*16)//4,cache_modifier=16)
            buffer_ops.buffer_store(e.to(fx.Int8),self.s_rsrc,_scale_off(row,base_col//32+h,self.scale_k1),
                                    offset_is_bytes=True,mask=self.qg==0,cache_modifier=16)

@functools.cache
@launcher
def compile_shared_gemm1_a8w8_write_through(*,tiles):
    # the composed host issues at most 96 shared workgroups (< grid_max): one tile each
    return compile_mxfp8_gemm_4w_gateup(K=6144,alpha=1.702,beta=1.0,limit=7.0,
        epilogue_type=StoreSharedFp32WriteThrough,
        kernel_prefix='shared_gemm1_a8w8_swiglu_quant_write_through_h6144_i768_',
        tiles=tiles,grid_max=128)
