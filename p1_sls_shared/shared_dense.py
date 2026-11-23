# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The shared expert's MXFP8 gate/up + SwiGLU + quant and down GEMMs on the 4-wave LDS-pipelined kernels
(``mxfp8/shared_dense_4wave.py``; ``shared_expert`` from 96 rows), on the weight layout the direct tiles read."""
import functools
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from .mxfp8.shared_dense_4wave import StoreSwigluQ, compile_mxfp8_gemm_4w_gateup, compile_mxfp8_gemm_4w, _acc_read, _scale_off
from .shared_gemm1_a8w8 import VIEW_UP_ROWS, _swiglu
from .shared_gemm2_a8w8 import VIEW_W_K
from .shared_quant import _fp8, _e8m0_fp8
from .common.launch import _run_compiled
from amd_kernels.warm_up import launcher


class StoreSharedFp32(StoreSwigluQ):
    """``StoreSwigluQ`` with the SwiGLU on the fp32 accumulators (no bf16 rounding), ``shared_quant``'s MXFP8 rule."""

    def _read_acts(self, cg, cu, ti):
        values=[]
        for j in range_constexpr(self.n_b):
            g=cg[self.c_idx_fn(ti,j)];u=cu[self.c_idx_fn(ti,j)]
            values.extend([_swiglu(fx.Float32(_acc_read(g,k)),fx.Float32(_acc_read(u,k))) for k in range_constexpr(4)])
        return values

    def _quant_store(self, values, base_row, base_col, ti):
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
                buffer_ops.buffer_store(pk,self.q_rsrc,(row*self.out_n+col+j*16)//4)
            buffer_ops.buffer_store(e.to(fx.Int8),self.s_rsrc,_scale_off(row,base_col//32+h,self.scale_k1),
                                    offset_is_bytes=True,mask=self.qg==0)


@functools.cache
@launcher
def _shuffle_scales(k):
    """Row-major e8m0 scales [m, k / 32] -> AITER's shuffled layout (``_scale_off``), rows padded to 32."""
    sc=k//32;kt=k//256
    @flyc.kernel(name=f'shared_scale_row_to_tiles_k{k}',known_block_size=[256,1,1])
    def kernel(S:fx.Tensor,O:fx.Tensor,m:fx.Int32):
        t=fx.thread_idx.x;b=fx.block_idx.x
        row=(b//kt)*32+(t//4)%16+(t%2)*16
        col=(b%kt)*8+(t//64)+(t%4//2)*4
        sr=buffer_ops.create_buffer_resource(S,max_size=False,num_records_bytes=m*sc)
        out=buffer_ops.create_buffer_resource(O,max_size=False,num_records_bytes=((m+31)//32*32)*sc)
        v=buffer_ops.buffer_load(sr,row*sc+col,vec_width=1,dtype=fx.Int8)
        buffer_ops.buffer_store(v,out,b*256+t,offset_is_bytes=True)
    @flyc.jit
    def launch_shared_scales_to_tiles(S:fx.Tensor,O:fx.Tensor,m:fx.Int32,stream:fx.Stream):
        kernel(S,O,m).launch(grid=(((m+31)//32)*kt,1,1),block=(256,1,1),stream=stream)
    return launch_shared_scales_to_tiles


@functools.cache
def _gemm1(tiles=(4,4), *, schedule="deferred", view=False):
    """The gate/up GEMM; ``schedule``: single (one tile per workgroup), early or deferred previous-tile epilogue;
    ``view``: the rank's shard read from the full weight (up rows ``VIEW_UP_ROWS`` after the gate rows)."""
    up_rows = VIEW_UP_ROWS if view else None
    if schedule == "single" and tiles == (4,4):
        from .shared_gemm1_a8w8_single_tile import compile_shared_gemm1_a8w8_single_tile
        return compile_shared_gemm1_a8w8_single_tile(up_rows=up_rows)
    if schedule == "early" and tiles == (4,4):
        from .shared_gemm1_a8w8_epilogue_early import compile_shared_gemm1_a8w8_epilogue_early
        return compile_shared_gemm1_a8w8_epilogue_early(up_rows=up_rows)
    return compile_mxfp8_gemm_4w_gateup(K=6144,alpha=1.702,beta=1.0,limit=7.0,
        epilogue_type=StoreSharedFp32,kernel_prefix='shared_fp32_swiglu_mxfp8_h6144_i1536_',tiles=tiles,
        up_rows=up_rows)


@functools.cache
def _gemm2(tiles=(4,4), *, nt=False, view=False):
    return compile_mxfp8_gemm_4w(K=1536,tiles=tiles,store_nt=nt,w_k=VIEW_W_K if view else None)


# nt bf16 output stores (same bits, kernel suffix _nt): entry -0.4..-1.0% at 4096-32768, -0.1..-0.5% at 1025-3072
SHARED_GEMM2_NT_FROM = 2048


def _gemm2_nt_for(m):
    return m >= SHARED_GEMM2_NT_FROM


def _gemm1_schedule_for(m):
    # twelve 256-column gate/up parent tiles per 256 rows: at most 252 tiles through 5376 rows, above that a
    # workgroup owns two
    return "single" if 1025 <= m <= 5376 else "early" if m > 5376 else "deferred"


def _tiled_scales(s, m, stream, side, keep=None):
    ss=torch.empty(((m+31)//32*32,192),device=s.device,dtype=torch.uint8)
    _hold(stream, side, keep, ss)
    _run_compiled(_shuffle_scales(6144),s.view(-1),ss.view(-1),m,stream)
    return ss


def _hold(stream, side, keep, *tensors):
    """Ties the current stream's ``tensors``, used on the side ``stream``, to its work: appended to ``keep`` (the
    caller frees them after the current stream waited for ``stream``) or else recorded on ``stream``."""
    if not side:
        return
    if keep is not None:
        keep.extend(tensors)
    else:
        for t in tensors:
            t.record_stream(stream)


def shared_dense(q,s,w,*,config=((4,4),(4,4)), scales_tiled=False, stream=None):
    """The shared expert's bf16 output [m, 6144] from ``shared_quant``'s codes ``q`` and scales ``s``: the TP shard
    (``SharedWeights.tp_operands``).

    Args:
        scales_tiled: ``s`` is already in the shuffled layout (``_shuffle_scales``).
        stream: launch there; the buffers still come from the current stream's allocator pool and are recorded on
            ``stream`` (the shared expert's side stream).
    """
    m=q.shape[0]
    side=stream is not None and stream != torch.cuda.current_stream()
    if stream is None:
        stream=torch.cuda.current_stream()
    ss=s if scales_tiled else _tiled_scales(s,m,stream,side)
    hq=torch.empty((m,1536),device=q.device,dtype=torch.uint8)
    hs=torch.empty(((m+31)//32*32,48),device=q.device,dtype=torch.uint8)
    out=torch.empty((m,6144),device=q.device,dtype=torch.bfloat16)
    if side:
        for t in (hq,hs,out):
            t.record_stream(stream)
    i8=lambda t:t.view(torch.int8).view(-1)
    w13,s13,w2,s2=w.tp_operands()
    g1=_gemm1(config[0],schedule=_gemm1_schedule_for(m),view=w.full)
    g2=_gemm2(config[1],nt=_gemm2_nt_for(m),view=w.full)
    p1=((m+255)//256)*12;p2=((m+255)//256)*24
    n1=((p1+7)//8*8 if g1.sub_split>1 else p1)*g1.sub_split
    n2=((p2+7)//8*8 if g2.sub_split>1 else p2)*g2.sub_split
    _run_compiled(g1,i8(q),i8(w13),i8(hq),hs.view(-1),ss.view(-1),s13,m,3072,0,n1,stream)
    _run_compiled(g2,i8(hq),i8(w2),out.view(-1),hs.view(-1),s2,m,6144,0,n2,stream)
    return out


# The full expert (I 3072) on its own rows (sequence-parallel prefill): gate/up N 6144 K 6144, down N 6144 K 3072.
@functools.cache
def _full_gemm1(tiles=(4,4), *, schedule="deferred"):
    if tiles == (4,4) or schedule == "deferred":
        return _gemm1(tiles, schedule=schedule)
    if schedule == "single":
        from .shared_gemm1_a8w8_single_tile import compile_shared_gemm1_a8w8_single_tile
        return compile_shared_gemm1_a8w8_single_tile(tiles=tiles)
    if schedule == "early":
        from .shared_gemm1_a8w8_epilogue_early import compile_shared_gemm1_a8w8_epilogue_early
        return compile_shared_gemm1_a8w8_epilogue_early(tiles=tiles)
    return compile_mxfp8_gemm_4w_gateup(K=6144,alpha=1.702,beta=1.0,limit=7.0,
        epilogue_type=StoreSharedFp32,kernel_prefix='shared_fp32_swiglu_mxfp8_h6144_i1536_',tiles=tiles)


@functools.cache
def _full_gemm2(tiles=(4,4), *, nt=False):
    return compile_mxfp8_gemm_4w(K=2*1536,tiles=tiles,store_nt=nt)


# rows -> ((gate/up tiles, schedule), (down tiles, nt stores)); "single" holds one work item per workgroup (at most
# 256). Sweep (benchmark_shared_full.py --sweep): vs the TP shard on 2 m rows -27% at 1024, -12% at 2048, -6% at
# 4096, -6% at 8192, -5% at 16384.
FULL_CONFIGS = (
    (1280, (((2,4),"single"),((2,4),True))),
    (2560, (((4,4),"single"),((4,4),True))),
    (4096, (((2,4),"deferred"),((2,4),True))),
    (16384, (((4,4),"early"),((4,4),True))),
)


def full_config_for(m):
    """The full expert's GEMM configuration at ``m`` rows (``FULL_CONFIGS``; beyond its last bound, the last)."""
    for bound, config in FULL_CONFIGS:
        if m <= bound:
            return config
    return FULL_CONFIGS[-1][1]


def _full_parents(m):
    return ((m+255)//256)*24


def shared_dense_full(q,s,w,*,stream=None,config=None,keep=None):
    """The full expert's bf16 output [m, 6144] from MXFP8 codes ``q`` [m, 6144] and row-major e8m0 ``s`` [m, 192]
    (``shared_dense``'s stream rule; with a list ``keep``, the buffers go there instead of being recorded)."""
    m=q.shape[0]
    side=stream is not None and stream != torch.cuda.current_stream()
    if stream is None:
        stream=torch.cuda.current_stream()
    (t1,sched),(t2,nt)=full_config_for(m) if config is None else config
    assert sched != "single" or _full_parents(m) * (4 // t1[0]) * (4 // t1[1]) <= 256, (m, t1)
    ss=_tiled_scales(s,m,stream,side,keep)
    hq=torch.empty((m,3072),device=q.device,dtype=torch.uint8)
    hs=torch.empty(((m+31)//32*32,96),device=q.device,dtype=torch.uint8)
    out=torch.empty((m,6144),device=q.device,dtype=torch.bfloat16)
    _hold(stream,side,keep,hq,hs,out)
    i8=lambda t:t.view(torch.int8).view(-1)
    g1,g2=_full_gemm1(t1,schedule=sched),_full_gemm2(t2,nt=nt)
    p1=((m+255)//256)*24;p2=((m+255)//256)*24
    n1=((p1+7)//8*8 if g1.sub_split>1 else p1)*g1.sub_split
    n2=((p2+7)//8*8 if g2.sub_split>1 else p2)*g2.sub_split
    _run_compiled(g1,i8(q),i8(w.w13),i8(hq),hs.view(-1),ss.view(-1),w.s13.view(-1),m,6144,0,n1,stream)
    _run_compiled(g2,i8(hq),i8(w.w2),out.view(-1),hs.view(-1),w.s2.view(-1),m,6144,0,n2,stream)
    return out
