# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One MXFP8 activation rule for the shared expert at every row count."""
import functools
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .common.launch import _run_compiled
from amd_kernels.warm_up import launcher


def _e8m0_fp8(amax):
    """e8m0 exponent of ceil_pow2(amax / 448), floored at 1; 255 (NaN / Inf) is not bumped."""
    u = (amax * fx.Float32(1.0 / 448.0)).bitcast(fx.Int32)
    e = (u >> 23) & 255
    bump = ((u & 0x7FFFFF) != 0) & (e < 255)
    e = bump.select(e + 1, e)
    return (e < 1).select(fx.Int32(1), e)


def _fp8(old, a, b, scale, hi):
    """v_cvt_scalef32_pk_fp8_f32: (a, b) / scale -> 2 e4m3 into the low (``hi`` False) or high half of ``old``."""
    ty = ir.VectorType.get([2], ir.IntegerType.get_signless(16))
    old_v = llvm.bitcast(ty, fx.as_ir_value(old))
    res = llvm.call_intrinsic(ty, "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [old_v, fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(scale), fx.Boolean(hi).ir_value()], [], [])
    return fx.Int32(llvm.bitcast(T.i32, res))


@functools.cache
@launcher
def compile_quant():
    """bf16 [m, 6144] -> e4m3 codes + row-major e8m0 scales per 32 columns, a workgroup per 2048 columns of a row."""
    @flyc.kernel(name="shared_mxfp8_quant_h6144", known_block_size=[256,1,1])
    def kernel(X:fx.Tensor,Q:fx.Tensor,S:fx.Tensor,m:fx.Int32):
        tx=fx.thread_idx.x
        row,chunk=fx.block_idx.x // 3,fx.block_idx.x % 3
        col=chunk*2048+tx*8
        xr=buffer_ops.create_buffer_resource(X,max_size=False,num_records_bytes=m*6144*2)
        qr=buffer_ops.create_buffer_resource(Q,max_size=False,num_records_bytes=m*6144)
        sr=buffer_ops.create_buffer_resource(S,max_size=False,num_records_bytes=m*192)
        raw=Vec(buffer_ops.buffer_load(xr,(row*6144+col)//2,vec_width=4,dtype=fx.Int32))
        v=[]
        for j in range_constexpr(4):
            dw=fx.Int32(raw[j])
            v.append((dw << 16).bitcast(fx.Float32))
            v.append((dw & fx.Int32(-65536)).bitcast(fx.Float32))
        amax=fx.math.absf(v[0])
        for j in range_constexpr(1,8):
            z=fx.math.absf(v[j]);amax=(amax>z).select(amax,z)
        for sh in range_constexpr(2):
            z=amax.shuffle_xor(1<<sh,64);amax=(amax>z).select(amax,z)
        e=_e8m0_fp8(amax);scale=(e<<23).bitcast(fx.Float32)
        lo=_fp8(fx.Int32(0),v[0],v[1],scale,False);lo=_fp8(lo,v[2],v[3],scale,True)
        hi=_fp8(fx.Int32(0),v[4],v[5],scale,False);hi=_fp8(hi,v[6],v[7],scale,True)
        buffer_ops.buffer_store(Vec.from_elements([lo,hi],fx.Int32),qr,(row*6144+col)//4)
        buffer_ops.buffer_store(e.to(fx.Int8),sr,row*192+col//32,offset_is_bytes=True,mask=(tx%4)==0)
    @flyc.jit
    def launch_shared_mxfp8_quant(X:fx.Tensor,Q:fx.Tensor,S:fx.Tensor,m:fx.Int32,stream:fx.Stream):
        kernel(X,Q,S,m).launch(grid=(m*3,1,1),block=(256,1,1),stream=stream)
    return launch_shared_mxfp8_quant


def shared_quant(x, stream=None):
    """MXFP8 codes [m, 6144] and row-major e8m0 scales [m, 192] of bf16 ``x``.

    Args:
        stream: launch there; the codes and scales still come from the current stream's allocator pool and are
            recorded on ``stream`` (the shared expert's side stream).
    """
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] == 6144
    m=x.shape[0]
    q=torch.empty((m,6144),device=x.device,dtype=torch.uint8)
    s=torch.empty((m,192),device=x.device,dtype=torch.uint8)
    if stream is None:
        stream=torch.cuda.current_stream()
    elif stream != torch.cuda.current_stream():
        q.record_stream(stream);s.record_stream(stream)
    _run_compiled(compile_quant(),x.view(-1),q.view(-1),s.view(-1),m,stream)
    return q,s
