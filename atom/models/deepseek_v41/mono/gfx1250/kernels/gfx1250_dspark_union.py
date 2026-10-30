# SPDX-License-Identifier: MIT
"""Original private route table on wave32; both logical lane halves retained."""
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_moe_front import dual_ballot
from atom.mono.device.ops import fresh,lanes_below,lane_gather,popcount
THREADS=256
WAVES=8
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
def load_route(c,tab):
    rs,tid,lane,wave=c['rs'],c['tid'],c['lane'],c['wave']
    assert not rs.dpa and rs.experts in (128,384) and rs.topk in (3,6)
    n,mw,experts=rs.picks,rs.mask_words,rs.experts
    indices=[plus(pas,tid) for pas in range(0,2*n,THREADS)]
    copy_at=c['bid']%ROUTE_COPIES*(2*n)
    got=c['poll']([(c['route_src'],copy_at+fx.min(idx,2*n-1),1) for idx in indices])
    for i in range_constexpr(len(indices)):
        if indices[i]<2*n:fx.ptr_store(got[i][0],tab+indices[i])
    for pas in range_constexpr(0,mw*experts,THREADS):
        idx=plus(pas,tid)
        if idx<mw*experts:fx.ptr_store(fx.Int32(0),tab+rs.mask0+idx)
    gpu.barrier()
    for pas in range_constexpr(0,n,THREADS):
        pick=plus(pas,tid)
        if pick<n:
            t=pick//rs.topk
            eid=fx.ptr_load(tab+t*2*rs.topk+pick%rs.topk)
            mark_pick(tab,rs,t,eid)
    gpu.barrier()
    chunks=experts//64
    counts,es=[],[]
    for half in range_constexpr(2):
        e=lane+32*half+64*fx.min(wave,chunks-1)
        count=fx.Int32(0)
        for word in range_constexpr(mw):
            count=count+popcount(fx.ptr_load(tab+rs.mask0+e*mw+word))
        counts.append(count);es.append(e)
    ball=dual_ballot(counts[0]!=0,counts[1]!=0)
    inclusive=list(dual_prefix(counts[0],counts[1],lane))
    tiles=[mplan.pick_tiles(counts[half]) for half in range(2)]
    tinclusive=list(dual_prefix(tiles[0],tiles[1],lane)) if rs.multi else tiles
    totals=[popcount(ball),lane_gather(inclusive[1],31),lane_gather(tinclusive[1],31)]
    if (wave<chunks)&(lane==0):
        for k in range_constexpr(3):
            fx.ptr_store(totals[k].bitcast(fx.Float32),c['red']+wave*3+k)
    gpu.barrier()
    below=[fx.Int32(0),fx.Int32(0),fx.Int32(0)]
    for chunk in range_constexpr(chunks-1):
        for k in range_constexpr(3):
            v=fx.ptr_load(c['red']+chunk*3+k).bitcast(fx.Int32)
            below[k]=below[k]+(chunk<wave).select(v,fx.Int32(0))
    nu,first,tile0=below
    for half in range_constexpr(2):
        slot=nu+lanes_below(ball,lane+32*half)
        if wave<chunks:
            fx.ptr_store(slot,tab+rs.expert_slot0+fresh(es[half]))
            if counts[half]!=0:
                fx.ptr_store(fx.Int32(es[half]),tab+2*n+1+slot)
                fx.ptr_store(first+inclusive[half]-counts[half],tab+rs.first0+slot)
            if const_expr(rs.multi):
                tf=tile0+tinclusive[half]-tiles[half]
                if counts[half]!=0:fx.ptr_store(tf,tab+rs.tile_first0+slot)
                for j in range_constexpr(rs.slot_tiles):
                    if j<tiles[half]:fx.ptr_store(slot,tab+rs.tile_slot0+tf+j)
    if (wave==chunks-1)&(lane==0):
        n_all=nu+totals[0]
        fx.ptr_store(n_all,tab+2*n)
        fx.ptr_store(first+totals[1],tab+rs.first0+n_all)
        if const_expr(rs.multi):fx.ptr_store(tile0+totals[2],tab+rs.tile_first0+n_all)
    gpu.barrier()
    for pas in range_constexpr(0,n,THREADS):
        pick=plus(pas,tid)
        if pick<n:
            t=pick//rs.topk
            eid=fx.ptr_load(tab+t*2*rs.topk+pick%rs.topk)
            place_pick(c,tab,pick,t,eid)
    gpu.barrier()
