# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 down: arrival queue, paired tasks and tagged second TP.

Keep shared-first, all round weights/scales before UGF waits, register MID
operands, two matrix chains and original private-LDS BF16 composition.
"""
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.kernels.gfx1250_moe import fp4_weight_offset
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_projection import reference_fp8_loads, reference_fp8_matrices_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_route import load_route_c1
from atom.models.deepseek_v41.mono.kernels.attn_pre import poll_copy
from atom.models.deepseek_v41.mono.kernels.moe import (
    HIDDEN, DOWN_TASKS, DOWN_R_MAX, DOWN_CHAINS, RED_TASKS,
    down_scale_words, wsl_words, down_ug_flags,
    take_down_rank, contrib_store, down_combine, stage_reduce,
)
from atom.mono.device.mx import scale_index
from atom.mono.device.ops import CM_DEV, CM_NT_GFX1250, bf16_round, fresh, lane_gather, row_sum, rsrc, traced
from atom.mono.plan.execution import BLOCKS, first_task
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_route import C1RouteShape
from atom.mono.device.ops import uniform


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
def compact_routed_down_live(c, tasks):
    """Six useful expert waves, two-step prefetch and original two chains.

    No CTA barrier occurs in this wave-uniform branch. The caller executes
    shared down with all eight waves, then joins all eight before combine.
    """
    lane, d, rs = c["lane"], c["d"], c["rs"]
    slot = uniform(c["wave"])
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
            for row in range_constexpr(8):
                contrib_store(c,j,pick*16+lane//16*8+row,value[row])


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
        stage_down_c1(c,[task,BLOCKS+arrival])
    else:
        stage_down_c1(c,[task])
    if const_expr(reduce):
        for task in range(first_task(bid,0),RED_TASKS,BLOCKS):
            stage_reduce(c,task)
