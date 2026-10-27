# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother ROUTE and each consumer's private table on eight physical wave32s.

A physical lane carries original lanes l and l+32. The original xor32
exchange is between those registers; smaller shuffles stay within wave32.
This preserves one logical wave per 64-expert chunk, including the single
wave that owns top-k, without a global union or a second readiness protocol.
"""
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int64

from atom.models.deepseek_v41.mono.kernels.attn_pre import ld_f32
from atom.models.deepseek_v41.mono.kernels.attn_pre import lds_row
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_sync import reference_mailbox, publish_plain
from atom.models.deepseek_v41.mono.kernels.moe import _score, ROUTE_COPIES, MASK_WORDS, SCALE
from atom.models.deepseek_v41.mono.kernels.moe_shape import RouteShape
from atom.mono.device.ops import (
    CM_DEV, ballot, bf16_round, fresh, kernel_symbol, lane_gather,
    lanes_below, launcher, ld_dev, popcount, readlane, traced,
)
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task

THREADS, WAVES = 256, 8


class C1RouteShape(RouteShape):
    """TP2 c1 table: picks/weights, six sorted slots, inverse order and queue.

    ROUTE's low nine id bits retain the expert; bits 9..11 carry its sorted
    slot. The original tagged-pair region and eight-copy fanout are unchanged.
    No per-expert bitmap or expert-to-slot array is needed by these consumers.
    """
    @property
    def order0(self):
        return 25

    @property
    def first0(self):
        return 31

    @property
    def queue0(self):
        return 38


def dual_ballot(pred0, pred1):
    """Original logical wave64 mask, with both physical ballots zero extended."""
    mask = fx.Int64(0xffffffff)
    return (ballot(pred0)&mask) | ((ballot(pred1)&mask)<<32)


def dual_prefix(v0, v1, lane):
    """Original integer lane-prefix for lanes l and l+32."""
    for offset in (1,2,4,8,16):
        moved0 = lane_gather(v0, fx.max(lane-offset,0))
        moved1 = lane_gather(v1, fx.max(lane-offset,0))
        # In the upper half, the source below lane32 is in the lower register.
        cross = lane_gather(v0, (lane+32-offset)&31)
        v0 = v0+(lane>=offset).select(moved0,0)
        v1 = v1+(lane>=offset).select(moved1,cross)
    # Offset32 is the original final prefix step, not a reassociated scan.
    return v0, v1+v0


@traced
def stage_route_c1(c, token):
    """Mother six experts/lane, sequential top6 and eight tagged copies."""
    lane, wave, rs = c["lane"], c["wave"], c["rs"]
    assert rs.tokens == 1 and rs.experts == 384 and rs.topk == 6
    if wave == 0:
        # Both logical lane halves load bias before waiting for LOGIT.
        bias = [[ld_f32(c["args"]["bias"],lane+32*h+64*i)
                 for i in range(6)] for h in range(2)]
        got = c["poll"]([(c["logit"],(token*384+lane+32*h+64*i)*4+p,2)
                         for h in range(2) for i in range(6) for p in (0,2)],batch=24)
        vals, orig, ids = [], [], []
        for h in range_constexpr(2):
            vh, oh, ih = [], [], []
            for i in range_constexpr(6):
                parts = got[h*12+2*i]+got[h*12+2*i+1]
                total = fx.Float32(0.)
                for p in range_constexpr(4):
                    total = total+parts[p].bitcast(fx.Float32)
                score = _score(bf16_round(total))
                vh.append(fx.max(score+bias[h][i],fx.Float32(float("-inf"))))
                oh.append(fx.Float32(fx.isnan(score).select(fx.Float32(0.),score)))
                ih.append(fx.Int32(lane+32*h+64*i))
            for p,q in rs.sort_net:
                swap = vh[p]<vh[q]
                vh[p],vh[q] = swap.select(vh[q],vh[p]),swap.select(vh[p],vh[q])
                oh[p],oh[q] = swap.select(oh[q],oh[p]),swap.select(oh[p],oh[q])
                ih[p],ih[q] = swap.select(ih[q],ih[p]),swap.select(ih[p],ih[q])
            vals.append(vh);orig.append(oh);ids.append(ih)
        cursor = [fx.Int32(0),fx.Int32(0)]
        total, my_w, my_id = fx.Float32(0.),fx.Float32(0.),fx.Int32(0)
        for k in range_constexpr(6):
            cv,ci,co = [],[],[]
            for h in range_constexpr(2):
                v, eid, weight = fx.Float32(float("-inf")),fx.Int32(0),fx.Float32(0.)
                for i in range_constexpr(6):
                    at = cursor[h]==i
                    v,eid,weight = at.select(vals[h][i],v),at.select(ids[h][i],eid),at.select(orig[h][i],weight)
                cv.append(v);ci.append(eid);co.append(weight)
            # Original butterfly (32,16,8,4,2,1), with xor32 in registers.
            best = fx.max(cv[0],cv[1])
            for offset in (16,8,4,2,1):
                best = fx.max(best,best.shuffle_xor(offset,32))
            ball = dual_ballot(cv[0]==best,cv[1]==best)
            win = fx.Int32((ball==fx.Int64(0)).select(fx.Int32(0),fx.Int32(fmath.cttz(ball))))
            winner0, winner1 = readlane(ci[0],win&31),readlane(ci[1],win&31)
            win_id = (win<32).select(winner0,winner1)
            mine = []
            for h in range_constexpr(2):
                won = (cursor[h]<6)&(ci[h]==win_id)
                mine.append(fx.Float32(won.select(co[h],fx.Float32(0.))))
                cursor[h] = cursor[h]+won.select(1,0)
            w0,w1 = readlane(mine[0],win_id&31),readlane(mine[1],win_id&31)
            weight = ((win_id&63)<32).select(w0,w1)
            total = total+weight
            my_w = (lane==k).select(weight,my_w)
            my_id = (lane==k).select(win_id,my_id)
        scale = fx.Float32(SCALE)/fx.max(total,fx.Float32(1e-20))
        if const_expr(isinstance(rs, C1RouteShape)):
            slot = fx.Int32(0)
            for pick in range_constexpr(6):
                slot = slot + (readlane(my_id, pick) < my_id).select(1, 0)
            my_id = my_id | (slot << 9)
        if lane < 6:
            for copy in range_constexpr(ROUTE_COPIES):
                at = (copy+token)*12+lane
                c["put"](c["route"],at,my_id)
                c["put"](c["route"],at+6,my_w*scale)
    gpu.barrier()


@traced
def load_route_reference_c1(c, tab):
    """Mother bitmap/prefix construction, private to every consuming CTA."""
    tid,lane,wave,rs = c["tid"],c["lane"],c["wave"],c["rs"]
    assert rs.tokens == 1 and rs.experts == 384 and rs.topk == 6
    copy_at = c["bid"]%ROUTE_COPIES*12
    got = c["poll"]([(c["route"],copy_at+fx.min(tid,11),1)])
    if tid < 12:
        fx.ptr_store(got[0][0],tab+tid)
    # Cover the identical 768 mask words with virtual thread rounds.
    for start in range_constexpr(0,MASK_WORDS*384,THREADS):
        idx = tid+start
        if idx < MASK_WORDS*384:
            fx.ptr_store(fx.Int32(0),tab+rs.mask0+idx)
    gpu.barrier()
    if tid < 6:
        eid = fx.ptr_load(tab+tid)
        fx.llvm.atomic_or(tab+rs.mask0+eid*MASK_WORDS,fx.Int32(1),syncscope="workgroup")
    gpu.barrier()
    counts,experts = [],[]
    for h in range_constexpr(2):
        eid = lane+32*h+64*fx.min(wave,5)
        count = fx.Int32(0)
        for word in range_constexpr(MASK_WORDS):
            count = count+popcount(fx.ptr_load(tab+rs.mask0+eid*MASK_WORDS+word))
        counts.append(count);experts.append(eid)
    ball = dual_ballot(counts[0]!=0,counts[1]!=0)
    incl0,incl1 = dual_prefix(counts[0],counts[1],lane)
    inclusive = [incl0,incl1]
    totals = [popcount(ball),lane_gather(incl1,31),lane_gather(counts[1],31)]
    if (wave<6)&(lane==0):
        for k in range_constexpr(3):
            fx.ptr_store(totals[k].bitcast(fx.Float32),c["red"]+wave*3+k)
    gpu.barrier()
    below_u,below_picks = fx.Int32(0),fx.Int32(0)
    for chunk in range_constexpr(5):
        n = fx.ptr_load(c["red"]+chunk*3).bitcast(fx.Int32)
        picks = fx.ptr_load(c["red"]+chunk*3+1).bitcast(fx.Int32)
        below_u = below_u+(chunk<wave).select(n,fx.Int32(0))
        below_picks = below_picks+(chunk<wave).select(picks,fx.Int32(0))
    for h in range_constexpr(2):
        slot = below_u+lanes_below(ball,lane+32*h)
        if wave < 6:
            fx.ptr_store(slot,tab+rs.expert_slot0+fresh(experts[h]))
            if counts[h] != 0:
                fx.ptr_store(fx.Int32(experts[h]),tab+13+slot)
                fx.ptr_store(below_picks+inclusive[h]-counts[h],tab+rs.first0+slot)
    if (wave==5)&(lane==0):
        n_all = below_u+totals[0]
        fx.ptr_store(n_all,tab+12)
        fx.ptr_store(fx.Int32(6),tab+rs.first0+n_all)
    gpu.barrier()
    if tid < 6:
        eid = fx.ptr_load(tab+tid)
        slot = fx.ptr_load(tab+rs.expert_slot0+eid)
        fx.ptr_store(slot,tab+19+tid)
        # c1 has one pick in each expert's slot: the original token rank is 0.
        pos = fx.ptr_load(tab+rs.first0+slot)
        fx.ptr_store(tid,tab+rs.order0+pos)
    gpu.barrier()


@traced
def load_route_c1(c, tab):
    """Consume producer-ranked top6, one CTA join and no bitmap/prefix work."""
    if const_expr(isinstance(c["rs"], C1RouteShape)):
        tid, rs = c["tid"], c["rs"]
        copy_at = c["bid"] % ROUTE_COPIES * 12
        got = c["poll"]([(c["route"], copy_at + fx.min(tid, 11), 1)])[0][0]
        if tid < 6:
            expert, slot = got & 511, got >> 9
            fx.ptr_store(expert, tab + tid)
            fx.ptr_store(expert, tab + 13 + slot)
            fx.ptr_store(slot, tab + 19 + tid)
            fx.ptr_store(tid, tab + rs.order0 + slot)
        if (tid >= 6) & (tid < 12):
            fx.ptr_store(got, tab + tid)
        if tid < 7:
            fx.ptr_store(tid, tab + rs.first0 + tid)
        if tid == 0:
            fx.ptr_store(fx.Int32(6), tab + 12)
        gpu.barrier()
    else:
        load_route_reference_c1(c, tab)


def ffn_route_symbol_c1(tables=False):
    return kernel_symbol("gfx1250_reference_ffn_route",b=BLOCKS,t=THREADS,tab=int(tables))


@traced
def load_x8r_c1(c, xl, xsl):
    """Mother c1 UG input staging, before its first private route-table wait."""
    tid = c["tid"]
    if tid < 16:
        c["poll"]([(c["x8rdy"],0,1)])
    gpu.barrier()
    # Only row0 is live at c1; retain each original plain 16-byte load.
    for name,width,dst in (("x8r",1280,xl),("x8rs",160,xsl)):
        for start in range_constexpr(0,width//4,THREADS):
            quad = tid+start
            if quad < width//4:
                words = fx.Vector(ld_dev(c[name].value,quad*4,4))
                fx.ptr_store(words,dst+quad*4)
    gpu.barrier()


@lru_cache(None)
def build_reference_ffn_route_c1(tables=False):
    """Independent real XRDY/XQ/router/route, with optional private-table taps."""
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_ffn import ffn_input_layout_c1,run_ffn_input_c1
    layout,rs = ffn_input_layout_c1(),RouteShape(384,6,1,2,1)
    name = ffn_route_symbol_c1(tables)
    @fx.struct
    class Smem:
        red:fx.Array[fx.Float32,WAVES*64*4,16]
        tab:fx.Array[fx.Int32,rs.tab_words,16]
        xl:fx.Array[fx.Int32,lds_row(1280),16]
        xsl:fx.Array[fx.Int32,lds_row(160),16]
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def kernel(x:Int64,gate_w:Int64,scratch:Int64,diag:Int64,layer:Int64,bias:Int64,tables_out:Int64):
        tid,bid = fresh(fx.thread_idx.x),fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        iptr = fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        failure = fx.inttoptr(iptr,diag+fx.Int64(BLOCKS*8*4))
        mb = reference_mailbox(fx.ptr_load(fx.inttoptr(iptr,layer))+128,diag,
            {key:i+1 for i,key in enumerate(layout)},failure)
        # Keep the mother's wave-uniform raw argument resources. Opaque
        # VGPR addresses introduce a waterfall at every FFN buffer access.
        c = dict(S=1,tid=tid,bid=bid,lane=tid%32,wave=tid//32,rs=rs,
                 args=dict(x=x,gate_w=gate_w,bias=bias),red=lds.red.ptr,
                 put=mb.put,poll=mb.poll,x_cm=CM_DEV,x_ready=True)
        for key in ("xrdy","x8rdy","x8r","x8rs","x8","x8s","logit","route"):
            c[key] = sreg(scratch,layout[key][0],key)
        if bid == 168:
            publish_plain(c["put"],c["xrdy"],0,1,tid==0)
        run_ffn_input_c1(c,bid,True)
        for token in range(first_task(bid,rs.route0),1,BLOCKS):
            stage_route_c1(c,token)
        if tables:
            # The mother c1 UG loop has max_u*ug_parts=216 placed tasks.
            # Its first iteration stages x before loading route, once per CTA.
            task = first_task(bid,rs.ug0)
            if task < rs.max_u*rs.ug_parts:
                load_x8r_c1(c,lds.xl.ptr,lds.xsl.ptr)
                load_route_c1(c,lds.tab.ptr)
                for start in range_constexpr(0,rs.tab_words,THREADS):
                    idx = tid+start
                    if idx < rs.queue0:
                        fx.ptr_store(fx.ptr_load(lds.tab.ptr+idx),fx.inttoptr(iptr,tables_out)+bid*(rs.tab_words+1440)+idx)
                for start in range_constexpr(0,1280,THREADS):
                    idx = tid+start
                    if idx < 1280:
                        fx.ptr_store(fx.ptr_load(lds.xl.ptr+idx),fx.inttoptr(iptr,tables_out)+bid*(rs.tab_words+1440)+rs.tab_words+idx)
                if tid < 160:
                    fx.ptr_store(fx.ptr_load(lds.xsl.ptr+tid),fx.inttoptr(iptr,tables_out)+bid*(rs.tab_words+1440)+rs.tab_words+1280+tid)
    @launcher(name)
    def launch(x:Int64,gate_w:Int64,scratch:Int64,diag:Int64,layer:Int64,bias:Int64,tables_out:Int64,stream:fx.Stream):
        kernel(x,gate_w,scratch,diag,layer,bias,tables_out).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
