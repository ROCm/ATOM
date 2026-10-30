# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""c1 down with independent routed production and output ownership.

TP2 assigns eight expert/output units per arrival-ranked CTA, publishes its
plain contributions before any consumer wait, then combines in original top-k
order. TP4 keeps the mother paired-task path.
"""
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_moe import fp4_weight_offset
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import reference_fp8_loads, reference_fp8_matrices_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_route import load_route_c1
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import poll_copy
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import (
    HIDDEN, DOWN_TASKS, DOWN_R_MAX, DOWN_CHAINS, RED_TASKS,
    down_scale_words, down_ug_flags,
    take_down_rank, contrib_store, contrib_load, down_combine, stage_reduce,
    region_pair,
)
from .reference_layout import wsl_words
from atom.mono.device.mx import scale_index
from atom.mono.device.ops import CM_DEV, CM_NT_GFX1250, bf16_round, fresh, lane_gather, row_sum, rsrc, traced
from atom.mono.plan.execution import BLOCKS, first_task
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_route import C1RouteShape
from atom.mono.device.ops import uniform
from atom.mono.device.sync import preg
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import publish_plain


def down_scales_c1(c, e, task):
    """Original full N32 scale block and padded 256-word copy rounds."""
    d, lane = c["d"],c["lane"]
    words = d.down_scale_cols*8
    block = (e*HIDDEN+task*16)//32*words
    return [fx.Vector(bo.buffer_load(rsrc(c["args"]["w2_s"]),
              block+fx.min(256*i+(lane+32*h)*4,words-4),vec_width=4,
              dtype=T.i32,cache_modifier=CM_NT_GFX1250))
            for i in range((words+255)//256) for h in range(2)]


def down_tile_weights_c1(c, e, task):
    d,lane=c["d"],c["lane"]
    base=e*(HIDDEN*d.inter//8)
    rw=rsrc(c["args"]["w2"])
    out=[]
    for st in range(d.inter//128):
        pieces=[fx.Vector(bo.buffer_load(rw,
            base+fp4_weight_offset(task,st,lane//16,lane%16,p,d.inter)//4,
            vec_width=4,dtype=T.i32,cache_modifier=CM_NT_GFX1250)) for p in range(2)]
        out.append(pieces[0].shuffle(pieces[1],list(range(8))))
    return out


def down_weights_c1(c,tasks,slots,nu):
    es=[c["rs"].union_expert(c["tab"],1,fx.min(c["wave"]+8*i,nu-1)) for i in range(slots)]
    return ([down_scales_c1(c,e,task) for task in tasks for e in es],
            [down_tile_weights_c1(c,e,task) for task in tasks for e in es])


def down_mid_c1(c,first):
    """Original eight-word B and nine-code register streams on wave32.

    Four physical eight-lane quarters hold the four original K fragments.
    Expand only at WMMA consumption: eager sixteen-word expansion spilled
    nineteen VGPRs and four SGPRs in the full joint kernel.
    """
    lane,d=c["lane"],c["d"]
    lane_word=fresh(first*d.mid_words+lane//8*4)
    bvs=[]
    for st in range(d.inter//128):
        chunks=[fx.Vector(bo.buffer_load(rsrc(c["mid"].value),lane_word+st*32+16*j,
                vec_width=4,dtype=T.i32,cache_modifier=CM_DEV)) for j in range(2)]
        bvs.append(chunks[0].shuffle(chunks[1],list(range(8))))
    steps=d.inter//128
    codes=[]
    for start in range(0,steps,4):
        n=min(4,steps-start)
        got=bo.buffer_load(rsrc(c["mids"].value),first*(d.inter//32)+lane//8*steps+start,
                          vec_width=n,dtype=T.i32,cache_modifier=CM_DEV)
        codes.extend([fx.Int32(got)] if n==1 else [fx.Int32(fx.Vector(got)[j]) for j in range(n)])
    return bvs,codes


@traced
def down_matrix_c1(c,j,task,sl,mid,first,av,live):
    lane,d=c["lane"],c["d"]
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,fx.Float4E2M1FN,fx.Float8E4M3FN,fx.Float32))
    accs=[fx.make_rmem_tensor(8,fx.Float32) for _ in range(DOWN_CHAINS)]
    for acc in accs:acc.store(fx.full(8,0.,fx.Float32))
    bvs,sbs=mid
    for st in range_constexpr(d.inter//128):
        sw=fx.Int32(0)
        for group in range_constexpr(4):
            at=scale_index(task%2*16+lane%16,st*4+group,d.down_scale_cols)
            sw=sw|(((fx.ptr_load(sl+at//4)>>(8*(at%4)))&255)<<(8*group))
        bv=fx.Vector.from_elements([
            lane_gather(bvs[st][group//2*4+q],lane%8+(2*(group%2)+lane//16)*8)
            for group in range(4) for q in range(4)],fx.Int32)
        sx=fx.Int32(0)
        for group in range_constexpr(4):
            sx=sx|(lane_gather(sbs[st],lane%8+group*8)<<(group*8))
        a,b=fx.make_rmem_tensor(8,fx.Int32),fx.make_rmem_tensor(16,fx.Int32)
        a.store(av[st]);b.store(bv)
        acc=accs[st%DOWN_CHAINS]
        fx.gemm(atom,acc,a,b,acc,scale_a=sw,scale_b=sx)
    value=accs[0].load()
    for chain in range_constexpr(1,DOWN_CHAINS):value=value+accs[chain].load()
    pick=c["rs"].pick_at(c["tab"],first)
    if live & (lane%16==0):
        for row in range_constexpr(8):
            contrib_store(c,j,pick*16+lane//16*8+row,value[row])


@traced
def down_shared_c1(c,tasks):
    cc=dict(c,xl=c["sxl"],xsl=c["sxsl"])
    shops=[reference_fp8_loads(cc,c["args"]["sw2"],c["args"]["sw2_s"],c["d"].sh_inter,task,CM_NT_GFX1250) for task in tasks]
    # Fold the mother's 512 logical copy threads over the 256 physical
    # threads: SMQ has 288 words, so using its old stride misses the last 32.
    poll_copy(c,c["tid"],256,[
        (c["smq"],lambda u:u,c["d"].sh_inter//4,c["sxl"],c["d"].sh_inter//4),
        (c["smc"],lambda u:u,c["d"].sh_inter//32,c["sxsl"],c["d"].sh_inter//32),
    ])
    gpu.barrier()
    for j in range_constexpr(len(tasks)):
        reference_fp8_matrices_c1(cc,c["d"].sh_inter,shops[j],mother_red=True,join=False)
        gpu.barrier()
        if c["tid"]<16:
            value=row_sum(c["red"],c["tid"],0)
            fx.ptr_store(bf16_round(value).bitcast(fx.Int32),c["sdl"]+j*16+c["tid"])
        gpu.barrier()


@traced
def stage_down_reference_c1(c,tasks):
    n=len(tasks)
    slots=DOWN_R_MAX//n
    nu=c["rs"].n_union(c["tab"],1)
    # c1 has six slots; the mother's round covers 8 or 16 of them.
    down_shared_c1(c,tasks)
    ucs=[fx.min(c["wave"]+8*i,nu-1) for i in range(slots)]
    firsts=[c["rs"].slot_first(c["tab"],u) for u in ucs]
    svs,avs=down_weights_c1(c,tasks,slots,nu)
    # Both original logical lane halves retain their exact wait collection.
    flags=[f for h in range(2) for f in down_ug_flags(dict(c,lane=c["lane"]+32*h),ucs)]
    c["poll"]([(c["ugf"],f,1) for f in flags])
    mids=[down_mid_c1(c,first) for first in firsts]
    sl=c["wsl"]+c["wave"]*wsl_words(c["d"])
    sw=down_scale_words(c["d"])
    for q in range_constexpr(len(svs)):
        for i in range_constexpr(len(svs[q])//2):
            for h in range_constexpr(2):
                fx.ptr_store(svs[q][2*i+h],sl+q*sw+256*i+(c["lane"]+32*h)*4)
    for i in range_constexpr(slots):
        for j in range_constexpr(n):
            down_matrix_c1(c,j,tasks[j],sl+(j*slots+i)*sw,mids[i],firsts[i],avs[j*slots+i],c["wave"]+8*i<nu)
    gpu.barrier()
    for j in range_constexpr(n):down_combine(c,tasks[j],j,0,1)


def compact_weight_step(c, expert, tasks, step):
    """One K128 step, both dynamically assigned output tiles."""
    lane, d = c["lane"], c["d"]
    base = expert * (HIDDEN * d.inter // 8)
    rw = rsrc(c["args"]["w2"])
    result = []
    for task in tasks:
        pieces = [fx.Vector(bo.buffer_load(rw,
            base + fp4_weight_offset(task, step, lane // 16, lane % 16, p, d.inter) // 4,
            vec_width=4, dtype=T.i32, cache_modifier=CM_NT_GFX1250)) for p in range(2)]
        result.append(pieces[0].shuffle(pieces[1], list(range(8))))
    return result


def compact_mid_step(c, first, step):
    """Producer-transposed K128 operand, expanded only within two live steps."""
    lane, d = c["lane"], c["d"]
    base = first * d.mid_words + step * 32 + (lane // 16) * 16
    chunks = [fx.Vector(bo.buffer_load(rsrc(c["mid"].value), base + q * 4,
              vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)) for q in range(4)]
    values = fx.Vector.from_elements([chunks[q][i] for q in range(4) for i in range(4)], fx.Int32)
    scale = fx.Int32(bo.buffer_load(rsrc(c["mids"].value),
        first * (d.inter // 128) + step, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
    return values, scale


@traced
def compact_routed_down_live(c, tasks, slot=None, global_output=False):
    """Six useful expert waves, two-step prefetch and original two chains.

    No CTA barrier occurs in this wave-uniform branch. The caller executes
    shared down with all eight waves, then joins all eight before combine.
    """
    lane, d, rs = c["lane"], c["d"], c["rs"]
    slot = uniform(c["wave"]) if slot is None else uniform(slot)
    first = slot
    expert = rs.union_expert(c["tab"], 1, slot)
    steps = d.inter // 128
    scales = [down_scales_c1(c, expert, task) for task in tasks]
    weights = [compact_weight_step(c, expert, tasks, st) for st in range(2)]
    # Original UGF collection for this useful expert, both virtual halves.
    flags = [f for h in range(2) for f in
             down_ug_flags(dict(c, lane=lane+32*h), [slot])]
    c["poll"]([(c["ugf"], f, 1) for f in flags])
    sl = c["wsl"] + c["wave"] * wsl_words(d)
    sw = down_scale_words(d)
    for j in range_constexpr(len(tasks)):
        for i in range_constexpr(len(scales[j]) // 2):
            for h in range_constexpr(2):
                fx.ptr_store(scales[j][2*i+h], sl+j*sw+256*i+(lane+32*h)*4)
    mids = [compact_mid_step(c, first, st) for st in range(2)]
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
        16,16,128,fx.Float4E2M1FN,fx.Float8E4M3FN,fx.Float32))
    accs = [[fx.make_rmem_tensor(8,fx.Float32) for chain in range(DOWN_CHAINS)]
            for task in tasks]
    for pair in accs:
        for acc in pair:
            acc.store(fx.full(8,0.,fx.Float32))
    for st in range_constexpr(steps):
        bv, sx = mids[st]
        b = fx.make_rmem_tensor(16,fx.Int32)
        b.store(bv)
        for j in range_constexpr(len(tasks)):
            weight_scale = fx.Int32(0)
            for group in range_constexpr(4):
                at = scale_index(tasks[j]%2*16+lane%16,st*4+group,d.down_scale_cols)
                weight_scale = weight_scale | (((fx.ptr_load(sl+j*sw+at//4) >> (8*(at%4))) & 255) << (8*group))
            a = fx.make_rmem_tensor(8,fx.Int32)
            a.store(weights[st][j])
            acc = accs[j][st%DOWN_CHAINS]
            fx.gemm(atom,acc,a,b,acc,scale_a=weight_scale,scale_b=sx)
        if st+2 < steps:
            weights.append(compact_weight_step(c,expert,tasks,st+2))
            mids.append(compact_mid_step(c,first,st+2))
    pick = rs.pick_at(c["tab"],first)
    for j in range_constexpr(len(tasks)):
        value = accs[j][0].load()
        for chain in range_constexpr(1,DOWN_CHAINS):
            value = value + accs[j][chain].load()
        if lane%16 == 0:
            if const_expr(global_output):
                # This wave owns one expert/output tile. Keep FP32 values;
                # route weighting and BF16 rounding belong to the consumer.
                for q in range_constexpr(2):
                    words = fx.Vector.from_elements(
                        [value[q*4+i].bitcast(fx.Int32) for i in range(4)], fx.Int32)
                    bo.buffer_store(words, rsrc(c["dcv"].value),
                        tasks[j]*96+pick*16+(lane//16)*8+q*4,
                        cache_modifier=CM_DEV)
            else:
                for row in range_constexpr(8):
                    contrib_store(c,j,pick*16+lane//16*8+row,value[row])


@traced
def combine_outputs_c1(c, tasks):
    """Combine one/two complete tiles together, preserving each tile's order."""
    tid, n = c["tid"], len(tasks)
    if tid < n*16:
        j, row = tid//16, tid%16
        shared = fx.ptr_load(c["sdl"]+tid).bitcast(fx.Float32)
        routed = fx.Float32(0.0)
        for k in range_constexpr(6):
            weight = c["rs"].route_w(c["tab"],1,0,k)
            value = contrib_load(c,j,k*16+row)
            routed = bf16_round(routed+bf16_round(value*weight))
        fx.ptr_store(bf16_round(routed+shared),c["pair"]+tid)
    gpu.barrier()
    if tid < n*8:
        task = tasks[0] if n == 1 else (tid < 8).select(tasks[0],tasks[1])
        col = task*16+2*(tid%8)
        v0, v1 = fx.ptr_load(c["pair"]+2*tid), fx.ptr_load(c["pair"]+2*tid+1)
        for p in range_constexpr(c["d"].tp):
            dst = preg(c["peer_addr"](p),c["peer"]["ffn"],"ffn")
            c["put_bf"](dst,2*region_pair(c["rank"],0,1,col),[v0,v1])
    gpu.barrier()


@traced
def stage_down_distributed_c1(c, arrival, tasks):
    """Publish all local production before waiting for remote CTA results.

    Arrival IDs 0..239 each own eight units, unit=output*6+expert-slot.
    Every effective wave runs nine K128 steps. IDs 240..255 only consume;
    their UG work and output ownership remain intact.
    """
    down_shared_c1(c,tasks)
    if uniform(arrival) < DOWN_TASKS*6//8:
        unit = uniform(arrival)*8+uniform(c["wave"])
        task, slot = uniform(unit//6), uniform(unit%6)
        compact_routed_down_live(c,[task],slot=slot,global_output=True)
    # Every writer wave drains its stores, then all eight join. No CTA can
    # enter a contribution wait while its own production is unpublished.
    publish_plain(c["put"],c["dcf"],arrival,1,
                  (c["tid"] == 0) & (arrival < DOWN_TASKS*6//8))
    waits = []
    for task in tasks:
        # Six contiguous expert slots intersect at most two producer groups.
        waits += [(c["dcf"],task*6//8,1),(c["dcf"],(task*6+5)//8,1)]
    c["poll"](waits)
    tid = c["tid"]
    if tid < len(tasks)*96:
        task = tasks[0] if len(tasks) == 1 else (tid < 96).select(tasks[0],tasks[1])
        word = bo.buffer_load(rsrc(c["dcv"].value),task*96+tid%96,
                              vec_width=1,dtype=T.i32,cache_modifier=CM_DEV)
        fx.ptr_store(fx.Int32(word),c["contrib"]+tid)
    gpu.barrier()
    combine_outputs_c1(c,tasks)


@traced
def compact_routed_down(c,tasks):
    if uniform(c["wave"]) < 6:
        compact_routed_down_live(c,tasks)


@traced
def stage_down_c1(c,tasks):
    if const_expr(isinstance(c["rs"], C1RouteShape)):
        down_shared_c1(c,tasks)
        compact_routed_down(c,tasks)
        gpu.barrier()
        for j in range_constexpr(len(tasks)):
            down_combine(c,tasks[j],j,0,1)
    else:
        stage_down_reference_c1(c,tasks)


@traced
def run_down_c1(c,bid,reduce=True):
    assert c["rs"].tokens==1 and c["rs"].topk==6 and c["d"].tp in (2,4)
    load_route_c1(c,c["tab"])
    arrival=take_down_rank(c)
    task=first_task(bid,0)
    if arrival<DOWN_TASKS-BLOCKS:
        if const_expr(isinstance(c["rs"],C1RouteShape)):
            stage_down_distributed_c1(c,arrival,[task,BLOCKS+arrival])
        else:
            stage_down_c1(c,[task,BLOCKS+arrival])
    else:
        if const_expr(isinstance(c["rs"],C1RouteShape)):
            stage_down_distributed_c1(c,arrival,[task])
        else:
            stage_down_c1(c,[task])
    if const_expr(reduce):
        for task in range(first_task(bid,0),RED_TASKS,BLOCKS):
            stage_reduce(c,task)
