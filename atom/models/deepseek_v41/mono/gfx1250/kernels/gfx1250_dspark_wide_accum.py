"""M16 N128 projection: one complete K reduction in each physical wave.

Eight waves cover independent N16 output blocks. Original K residues modulo
eight accumulate independently within each wave, then sum in the original order.
There is no cross-wave reduction or RED materialization. Weight and group32 scale
prefetch distance is three K128 iterations; no partial C/old RED restoration.
The input tile remains in LDS and is shared by all eight output waves.
"""
import flydsl.expr as fx
from flydsl.expr import const_expr,gpu,range_constexpr
from aiter.ops.flydsl.kernels import buffer_ops as bo
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import (
    Q_RANK,HEAD_DIM,ROPE,IDX_ROPE,IDX_ROWS,HALF,ip,lds_row,ld_i32,ld_bf,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import _weight_step
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import scalar_address
from atom.mono.device.ops import bf16_round,rsrc,traced

@traced
def prefetch(c,weight,scale,ntask):
    wave=fx.thread_idx.x//32
    view=dict(c,lane=fx.thread_idx.x%32,wave=wave)
    rw,rs=rsrc(scalar_address(weight)),rsrc(scalar_address(scale))
    ops=[_weight_step(view,rw,rs,Q_RANK,ntask*8+wave,fx.Int32(i)) for i in range(3)]
    fx.rocdl.sched_barrier(0)
    return ops

@traced
def project(c,weight,scale,ntask,t0,n,ops,index=False):
    lane,wave=fx.thread_idx.x%32,fx.thread_idx.x//32
    token=lane%16;half=lane//16
    local=fx.min(token,n-1)
    xl=c['qxl']+local*lds_row(Q_RANK//4)
    xsl=c['qxsl']+local*lds_row(Q_RANK//32)
    view=dict(c,lane=lane,wave=wave)
    rw,rs=rsrc(scalar_address(weight)),rsrc(scalar_address(scale))
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,fx.Float8E4M3FN,fx.Float8E4M3FN,fx.Float32))
    accumulators=[fx.make_rmem_tensor(8,fx.Float32) for _ in range(8)]
    for acc in accumulators:acc.store(fx.full(8,0.0,fx.Float32))
    assert Q_RANK%128==0
    for i in range_constexpr(Q_RANK//128):
        if const_expr(i+3<Q_RANK//128):
            ops.append(_weight_step(view,rw,rs,Q_RANK,ntask*8+wave,fx.Int32(i+3)))
            fx.rocdl.sched_barrier(0)
        chunks=[fx.Vector(fx.ptr_load(xl+i*32+half*4+j*8,
            result_type=fx.Vector.make_type(4,fx.Int32))) for j in range(4)]
        x=fx.Vector.from_elements([chunks[j][q] for j in range(4) for q in range(4)],fx.Int32)
        codes=[fx.ptr_load(xsl+i*4+j) for j in range(4)]
        xs=codes[0]|(codes[1]<<8)|(codes[2]<<16)|(codes[3]<<24)
        x=fx.Vector.from_elements([(token<n).select(x[j],fx.Int32(0)) for j in range(16)],fx.Int32)
        xs=(token<n).select(xs,fx.Int32(0x7f7f7f7f))
        a,b=fx.make_rmem_tensor(16,fx.Int32),fx.make_rmem_tensor(16,fx.Int32)
        a.store(ops[i][0]);b.store(x)
        acc=accumulators[i%8]
        fx.gemm(atom,acc,a,b,acc,scale_a=ops[i][1],scale_b=xs)
    values=fx.Vector.filled(8,0.0,fx.Float32)
    for acc in accumulators:values=values+acc.load()
    if token<n:
        position=ld_i32(scalar_address(c['args']['pos']),2*(t0+token))
        for pair in range_constexpr(4):
            row=ntask*128+wave*16+half*8+2*pair
            dim=ip.DIM if index else HEAD_DIM
            rotated=IDX_ROPE if index else ROPE
            d=row%dim;live=d>=dim-rotated
            jj=fx.max(d-(dim-rotated),0)//2
            cs=ld_bf(scalar_address(c['args']['cos']),position*HALF+jj)
            sn=ld_bf(scalar_address(c['args']['sin']),position*HALF+jj)
            v0,v1=bf16_round(values[2*pair]),bf16_round(values[2*pair+1])
            e0=fx.Float32(live.select(v0*cs-v1*sn,v0))
            e1=fx.Float32(live.select(v0*sn+v1*cs,v1))
            if const_expr(index):c['put_bf'](c['iq'],(t0+token)*IDX_ROWS+row,[e0,e1])
            else:
                out=(t0+token)*c['d'].heads*HEAD_DIM+row
                bo.buffer_store(e0.to(fx.BFloat16),rsrc(scalar_address(c['args']['q_out'])),out)
                bo.buffer_store(e1.to(fx.BFloat16),rsrc(scalar_address(c['args']['q_out'])),out+1)
    gpu.barrier()
