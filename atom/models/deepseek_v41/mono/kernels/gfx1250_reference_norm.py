# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Original c1 slice→norm ownership and tagged dataflow, on physical wave32."""

from functools import lru_cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    MIX, KT, HIDDEN, SLICES, EPS, X8_WORDS, ld_bf, fma,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_k1 import stage_slice_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_plan import BLOCKS, THREADS, C1ReferencePlan
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, pow2
from atom.mono.device.ops import bf16_round, bf_lo, bf_hi, fp8_pack4, hw_rsq, kernel_symbol, launcher, rsrc, traced
from atom.mono.device.sync import sreg


@traced
def stage_norm_c1(c):
    """Mother's 256 active threads/24 values each, with a logical wave64 sum."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    ffn = c.get("ffn",False)
    normed_cm = c.get("normed_cm",0)
    tt = fx.min(tid,255)
    mine = tid < 256
    cols = [tt*8+2048*ch for ch in range(3)]
    # Same weights-before-LIN wait; no dependency on PMIX.
    wts = [[ld_bf(a["attn_w"],fx.min(cols[ch],HIDDEN-8)+j) for j in range(8)] for ch in range(3)]
    words = c["poll"]([(c["lin"],fx.min(cols[ch],HIDDEN-8)//2+k,1) for ch in range(3) for k in range(4)])
    xs=[]
    for ch in range_constexpr(3):
        v=[]
        for k in range_constexpr(4):
            v += [bf_lo(words[4*ch+k][0]),bf_hi(words[4*ch+k][0])]
        xs.append([(cols[ch]<HIDDEN).select(x,fx.Float32(0.0)) for x in v])
    acc=fx.Float32(0.0)
    for ch in range_constexpr(3):
        for x in xs[ch]:acc=acc+x*x
    for off in (1,2,4,8,16):acc=acc+acc.shuffle_xor(off,32)
    fx.ptr_store(acc,red+tid)
    gpu.barrier()
    acc=acc+fx.ptr_load(red+(tid^32))
    # Every bridge read completes before the first four words become sums.
    gpu.barrier()
    if (lane==0)&mine:fx.ptr_store(acc,red+wave)
    gpu.barrier()
    total=fx.ptr_load(red+lane%4)
    for off in (1,2):total=total+total.shuffle_xor(off,32)
    r=hw_rsq(fma(total,fx.Float32(1.0/HIDDEN),fx.Float32(EPS)))
    for ch in range_constexpr(3):
        col=fx.min(cols[ch],HIDDEN-8)
        ys=[bf16_round((xs[ch][j]*r)*wts[ch][j]) for j in range(8)]
        amax=fx.Float32(0.0)
        for y in ys:amax=fx.max(amax,abs(y))
        for off in (1,2):amax=fx.max(amax,amax.shuffle_xor(off,32))
        amax=fx.max(amax,fx.Float32(1e-4))
        code=code_ceil(amax*fx.Float32(1.0/FP8_MAX));step=pow2(code)
        if mine&(cols[ch]<HIDDEN):
            if const_expr(c["index"] or ffn):
                for j in range_constexpr(8):
                    bo.buffer_store(ys[j].to(fx.BFloat16),rsrc(a["normed"]),col+j,
                                    cache_modifier=normed_cm)
            if const_expr(c["index"]):
                c["put_bf"](c["nb"],col,ys[:4])
                c["put_bf"](c["nb"],col+4,ys[4:])
            if const_expr(not ffn):
                w0=fp8_pack4(*[clamp_fp8(ys[i]/step) for i in range(4)])
                w1=fp8_pack4(*[clamp_fp8(ys[4+i]/step) for i in range(4)])
                c["put_words"](c["x8"],col//4,[w0,w1])
                if tt%4==0:c["put"](c["x8s"],col//32,code)
    gpu.barrier()


def front_symbol(fold,feeds_index):
    return kernel_symbol("gfx1250_reference_k1_front",fold=int(fold),ix=int(feeds_index),b=BLOCKS,t=THREADS)


@lru_cache(None)
def build_front_c1(fold=True,feeds_index=False):
    plan=C1ReferencePlan(feeds_index=feeds_index);layout=plan.k1_scratch_layout()
    name=front_symbol(fold,feeds_index)
    @fx.struct
    class Smem:
        rl:fx.Array[fx.Float32,KT,16]
        fl:fx.Array[fx.Float32,MIX*KT,16]
        red:fx.Array[fx.Float32,512,16]
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def front(res_in:Int64,pend:Int64,post_in:Int64,comb_in:Int64,pre_in:Int64,hc_fn:Int64,
              res_out:Int64,scratch:Int64,diag:Int64,layer:Int64,attn_w:Int64,normed:Int64):
        tid,bid=fx.thread_idx.x,fx.block_idx.x
        alloc=fx.SharedAllocator();lds=alloc.allocate(Smem).peek()
        pointer=fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        failure=fx.inttoptr(pointer,diag+fx.Int64(BLOCKS*8*4))
        layerp=fx.inttoptr(pointer,layer)
        mb=reference_mailbox(fx.ptr_load(layerp),diag,{"lin":1,"pmix":2,"x8":3,"x8s":4,"nb":5},failure)
        args=dict(res_in=res_in,pend=pend,post_in=post_in,comb_in=comb_in,pre_in=pre_in,hc_fn=hc_fn,
            res_out=res_out,attn_w=attn_w,normed=normed)
        c=dict(S=1,tid=tid,lane=tid%64,wave=tid//64,rl=lds.rl.ptr,fl=lds.fl.ptr,red=lds.red.ptr,args=args,
            fold=fold,aux=False,index=feeds_index,put=mb.put,put_bf=mb.put_bf,put_words=mb.put_words,poll=mb.poll)
        for region in ("lin","pmix","x8","x8s","nb"):
            c[region]=sreg(scratch,layout[region][0],region)
        for task in range(plan.stages["k1.slice"].first(bid),SLICES,BLOCKS):stage_slice_c1(c,task)
        for task in range(plan.stages["k1.norm"].first(bid),1,BLOCKS):stage_norm_c1(c)
    @launcher(name)
    def launch(res_in:Int64,pend:Int64,post_in:Int64,comb_in:Int64,pre_in:Int64,hc_fn:Int64,
               res_out:Int64,scratch:Int64,diag:Int64,layer:Int64,attn_w:Int64,normed:Int64,stream:fx.Stream):
        front(res_in,pend,post_in,comb_in,pre_in,hc_fn,res_out,scratch,diag,layer,attn_w,normed).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
