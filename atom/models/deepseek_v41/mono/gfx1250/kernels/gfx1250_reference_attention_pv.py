# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 PV, split-combine and IRQ with wave32 matrix fragments.

The original eight waves own splits w+8r, prefetch valid key tiles before
statistics waits, and consume tagged AM/AL/AP. Two register streams represent
each wave64 lane half. The original alpha, statistics, tree and output LDS
layouts and software reduction order remain intact. WMMA pads the K16 PV
operands with zero to K32; IRQ uses the unchanged mother implementation.
"""
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int64, T
from aiter.ops.flydsl.kernels import buffer_ops as bo

from atom.models.deepseek_v41.mono import attention_plan as aplan
from atom.models.deepseek_v41.mono.gfx950.kernels.attention import (
    BK, COLS, GROUPS, HEAD_DIM, HEAD_TILE, LOG2E, NEG, SCORE_LDS_WORDS,
    _ap, _key_row, _keys, _ld_row4, _stat_specs, await_selection,
    scratch_pairs, tile_head, tile_task,
)
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_post import irq_group, irq_rope_loads
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_attention import _wmma
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_attention_score import stage_attn_score_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_indexer import vector_address
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox
from atom.mono.device.ops import (
    bf16_round, hw_exp2, kernel_symbol, lane_gather, launcher, row_shr, rsrc, traced, uniform,
)
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task
from atom.mono.plan.layout import pair_layout

THREADS, WAVES = 256, 8


@traced
def l_tree_c1(ls, alphas, lane):
    """Mother DPP tree over 64 splits, two 32-lane register streams."""
    vs = [fx.Float32(fmath.fma(ls[p], alphas[p],
              row_shr(ls[p] * alphas[p], lane, 8))) for p in range(2)]
    for k in (4, 2, 1):
        vs = [row_shr(v, lane, k) + v for v in vs]
    vs = [fx.Float32((lane // 16 == 1).select(
          lane_gather(v.bitcast(fx.Int32), 15).bitcast(fx.Float32) + v, v))
          for v in vs]
    b31 = lane_gather(vs[0].bitcast(fx.Int32), 31).bitcast(fx.Float32)
    upper = b31 + vs[1]
    return lane_gather(upper.bitcast(fx.Int32), 31).bitcast(fx.Float32)


@traced
def pv_stats_c1(got, act, lane, wave, att):
    for k in range_constexpr(2):
        h = 2 * wave + k
        ms = [fx.Float32((lane + p * 32 < act).select(
              got[p][k][0].bitcast(fx.Float32), fx.Float32(NEG))) for p in range(2)]
        ls = [fx.Float32((lane + p * 32 < act).select(
              got[p][2+k][0].bitcast(fx.Float32), fx.Float32(0.))) for p in range(2)]
        maximum = fx.max(ms[0], ms[1])
        for off in (16, 8, 4, 2, 1):
            maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
        alphas = [hw_exp2(m - maximum) for m in ms]
        combined = l_tree_c1(ls, alphas, lane)
        for p in range_constexpr(2):
            fx.ptr_store(alphas[p], att["alpha"] + ((lane+p*32) * HEAD_TILE + h))
        if lane == 0:
            fx.ptr_store(maximum, att["stat"] + h)
            fx.ptr_store(combined, att["stat"] + HEAD_TILE + h)



@traced
def pv_key_tile_c1(a, t, meta, kv_len, act, sp, virtual, byte):
    """Skip invalid key-row lookup and global payload load, including kv_len=0."""
    words = fx.Vector.filled(4, 0, fx.Int32)
    if sp < act:
        row = _key_row(a, t, meta, fx.min(sp*BK+virtual//4, kv_len-1))
        words = _ld_row4(a["pool"], row, byte)
    return words


@traced
def pv_split_matrix_c1(c, kv_len, act, sp, tiles, pws):
    """Unchanged valid-split math; caller enters with the complete wave32."""
    lane, wave, att = c["lane"], c["wave"], c["att"]
    head, half = lane % 16, lane // 16
    kt = att["kv"] + wave * (BK * COLS // 2)
    # Original two lane halves write their original quarter rows.
    for p in range_constexpr(2):
        virtual = lane + p * 32
        ok = sp * BK + virtual // 4 < kv_len
        fx.ptr_store(fx.Vector.from_elements(
            [ok.select(tiles[p][e], fx.Int32(0)) for e in range(4)], fx.Int32),
            kt + (virtual // 4 * (COLS // 2) + virtual % 4 * 4))
    rocdl.s_wait_dscnt(0)
    # AP virtual halves contain keys [0:8] and [8:16] for each head.
    words = [pws[p][e] for p in range(2) for e in range(2)]
    own = fx.Vector.from_elements(words, fx.Int32)
    other = fx.Vector.from_elements(
        [word.shuffle_xor(16, 32) for word in words], fx.Int32)
    # Reassemble the four mother quarters, then zero-pad the second K16.
    q0 = [fx.Int32((half == 0).select(own[e], other[e])) for e in range(4)]
    q1 = [fx.Int32((half == 0).select(other[e], own[e])) for e in range(4)]
    # q0 holds quarters0/2 and q1 quarters1/3 (logical lane l, l+32).
    ordered = [q0[0], q0[1], q1[0], q1[1], q0[2], q0[3], q1[2], q1[3]]
    pv = fx.Vector.from_elements(
        [(half == 0).select(word, fx.Int32(0)) for word in ordered],
        fx.Int32).bitcast(fx.BFloat16)
    row_accs = []
    for tt in range_constexpr(2):
        col = tt * 16 + head
        kw = fx.Vector.from_elements([
            (half == 0).select(
                fx.Int32(fx.ptr_load(kt + key * (COLS // 2) + col // 2))
                >> (col % 2 * 16) & 0xFFFF, fx.Int32(0))
            for key in range(BK)], fx.Int32).to(fx.Int16).bitcast(fx.BFloat16)
        accum = fx.make_rmem_tensor(8, fx.Float32)
        accum.store(fx.full(8, 0., fx.Float32))
        _wmma(accum, pv, kw)
        result = accum.load()
        row_accs.append(fx.Vector.from_elements([fx.Float32((sp < act).select(
            result[j], fx.Float32(0.))) for j in range(8)], fx.Float32))
    # A wave's next split may reuse kt only after all reads complete.
    rocdl.s_wait_dscnt(0)
    return row_accs


@traced
def pv_group_c1(c, kv_len, act, splits, tiles, pws, al, sink, i):
    lane, wave, tid, att = c["lane"], c["wave"], c["tid"], c["att"]
    head, half = lane % 16, lane // 16
    kt = att["kv"] + wave * (BK * COLS // 2)
    accs = []
    for r in range_constexpr(8):
        sp = splits[r]
        row_accs = [fx.Vector.filled(8, 0., fx.Float32) for _ in range(2)]
        # Both values are wave-uniform; no CTA barrier is conditional.
        if sp < act:
            row_accs = pv_split_matrix_c1(c, kv_len, act, sp, tiles[r],
                                         [pws[p][r] for p in range(2)])
        accs.append(row_accs)
    for tt in range_constexpr(2):
        for j in range_constexpr(8):
            av = [accs[r][tt][j] for r in range(8)]
            pairs = [fx.Float32(fmath.fma(av[2*m], al[2*m], av[2*m+1] * al[2*m+1]))
                     for m in range(4)]
            node = (pairs[0] + pairs[1]) + (pairs[2] + pairs[3])
            virtual = head + (half * 2 + j // 4) * 16
            k = tt * 4 + j % 4
            fx.ptr_store(node, att["tree"] + ((wave * 64 + virtual) * 8 + k))
    gpu.barrier()
    # Original owner mapping and wave-combine tree; each thread emits two cols.
    for p in range_constexpr(2):
        virtual_tid = tid + p * THREADS
        h, col = virtual_tid // COLS, virtual_tid % COLS
        owner = col % 16 // 4 * 16 + h
        k = col // 16 * 4 + col % 4
        ws = [fx.ptr_load(att["tree"] + ((ww * 64 + owner) * 8 + k)) for ww in range(8)]
        xs = [ws[2*m] + ws[2*m+1] for m in range(4)]
        total = (xs[0] + xs[2]) + (xs[1] + xs[3])
        maximum = fx.ptr_load(att["stat"] + h)
        combined = fx.ptr_load(att["stat"] + HEAD_TILE + h)
        sink_value = sink[p]
        final_m = fx.max(maximum, sink_value * fx.Float32(LOG2E))
        alpha_kv = hw_exp2(maximum - final_m)
        alpha_sink = hw_exp2(fx.Float32(fmath.fma(sink_value, fx.Float32(LOG2E), -final_m)))
        final_l = fx.Float32(fmath.fma(combined, alpha_kv, alpha_sink))
        value = (total * alpha_kv) / fx.max(final_l, fx.Float32(1e-30))
        value = fx.Float32((final_l > fx.Float32(0.)).select(value, fx.Float32(0.)))
        fx.ptr_store(bf16_round(value), att["out"] + ((i*HEAD_TILE+h)*COLS+col))
    gpu.barrier()


@traced
def stage_attn_pv_c1(c, task, groups):
    lane, wave, d, att, a = c["lane"], c["wave"], c["d"], c["att"], c["args"]
    t, tile, g0 = tile_task(task, GROUPS // groups, d.head_tiles)
    await_selection(c, t)
    kv_len, meta = _keys(a, t)
    act, last = uniform((kv_len+BK-1)//BK), aplan.last_split(kv_len)
    split_wave = uniform(wave)
    splits = [split_wave + 8*r for r in range(8)]
    def group_tiles(i):
        dc = g0 * groups + i
        return [[pv_key_tile_c1(a, t, meta, kv_len, act, splits[r],
                    lane+p*32, (dc*COLS+8*(lane%4))*2)
                 for p in range(2)] for r in range(8)]
    tiles = group_tiles(0)
    sink = [fx.Float32(bo.buffer_load(rsrc(a["sink"]),
              tile_head(d, tile, (c["tid"]+p*THREADS)//COLS)[0], vec_width=1,
              dtype=T.f32)) for p in range(2)]
    stats = [_stat_specs(c,t,tile,last,lane+p*32,wave) for p in range(2)]
    if wave == 0:
        for p in range_constexpr(2):
            c["poll"](stats[p])
    gpu.barrier()
    got = [c["poll"](stats[p] + [
        (c["ap"], (_ap(d,t,tile,aplan.polled_split(sp,last))+lane+p*32)*2, 2)
        for sp in splits], batch=12) for p in range(2)]
    pv_stats_c1(got,act,lane,wave,att)
    pws = [values[4:] for values in got]
    gpu.barrier()
    al = [fx.ptr_load(att["alpha"]+(sp*HEAD_TILE+lane%16)) for sp in splits]
    for i in range_constexpr(groups):
        nxt = group_tiles(i+1) if i+1 < groups else None
        pv_group_c1(c,kv_len,act,splits,tiles,pws,al,sink,i)
        tiles = nxt


@traced
def stage_attn_irq_c1(c, task, groups, debug=False):
    t,tile,g0 = tile_task(task,GROUPS//groups,c["d"].head_tiles)
    ropes = irq_rope_loads(c,t,[g0*groups+i for i in range(groups)])
    stage_attn_pv_c1(c,task,groups)
    for i in range_constexpr(groups):
        if const_expr(debug):
            h, pair = c["tid"]//16, c["tid"]%16
            for j in range_constexpr(2):
                value = fx.ptr_load(c["att"]["out"]+((i*HEAD_TILE+h)*COLS+pair*2+j))
                fx.ptr_store(value.to(fx.BFloat16),c["debug_out"]+
                    ((tile*HEAD_TILE+h)*HEAD_DIM+(g0*groups+i)*COLS+pair*2+j))
        irq_group(c,t,tile,g0*groups+i,c["att"]["out"]+i*HEAD_TILE*COLS,ropes[i])


def attention_pv_layout_c1():
    d=Dims(2)
    return pair_layout({**scratch_pairs(1,d),"xo":d.xo_words,
                        "xos":d.heads*HEAD_DIM//32}.items())


def attention_pv_symbol_c1(debug=False):
    return kernel_symbol("gfx1250_reference_attention_pv",b=BLOCKS,t=THREADS,db=int(debug))


@lru_cache(None)
def build_reference_attention_pv_c1(debug=False):
    """Original score and IRQ placements in one cooperative c1 kernel."""
    layout,d,name=attention_pv_layout_c1(),Dims(2),attention_pv_symbol_c1(debug)
    @fx.struct
    class ScoreLds:
        kv:fx.Array[fx.Int32,WAVES*SCORE_LDS_WORDS,16]
    @fx.struct
    class PvLds:
        tree:fx.Array[fx.Float32,8*64*8,16]
        alpha:fx.Array[fx.Float32,64*HEAD_TILE,16]
        stat:fx.Array[fx.Float32,2*HEAD_TILE,16]
        out:fx.Array[fx.Float32,HEAD_TILE*COLS,16]
        kv:fx.Array[fx.Int32,WAVES*BK*COLS//2,16]
    @fx.union
    class Smem:
        score:ScoreLds
        pv:PvLds
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def kernel(q:Int64,pool:Int64,sel:Int64,sbound:Int64,kmeta:Int64,
               sink:Int64,pos:Int64,cos:Int64,sin:Int64,scratch:Int64,
               diag:Int64,layer:Int64,config:Int64,out:Int64):
        tid,bid=fx.thread_idx.x,fx.block_idx.x
        mem=fx.SharedAllocator().allocate(Smem)
        sc,pv=mem.score.peek(),mem.pv.peek()
        iptr=fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        cfg=fx.inttoptr(iptr,config)
        failure=fx.inttoptr(iptr,diag+fx.Int64(BLOCKS*8*4))
        mb=reference_mailbox(fx.ptr_load(fx.inttoptr(iptr,layer))+64,diag,
                             {key:i+1 for i,key in enumerate(layout)},failure)
        args=dict(q=q,pool=pool,sel=sel,sbound=sbound,kmeta=kmeta,topk=512,
                  sink=sink,pos=pos,cos=cos,sin=sin,ring_off=fx.ptr_load(cfg+1),
                  ring_slots=512,iact=fx.ptr_load(cfg),qk_scale=fx.ptr_load(cfg+2))
        out=vector_address(out)
        c=dict(S=1,tid=tid,bid=bid,lane=tid%32,wave=tid//32,d=d,args=args,
               att=dict(score_kv=sc.kv.ptr,tree=pv.tree.ptr,alpha=pv.alpha.ptr,
                        stat=pv.stat.ptr,out=pv.out.ptr,kv=pv.kv.ptr),
               put=mb.put,put_words=mb.put_words,poll=mb.poll,
               debug_out=fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type,
                    fx.AddressSpace.Global,2),out))
        scratch=vector_address(scratch)
        for region in layout:
            c[region]=sreg(scratch,layout[region][0],region)
        for task in range(first_task(bid,0),d.head_tiles.count*8,BLOCKS):
            stage_attn_score_c1(c,task,8)
        for task in range(first_task(bid,64),d.head_tiles.count*GROUPS,BLOCKS):
            stage_attn_irq_c1(c,task,1,debug)
        gpu.barrier()
    @launcher(name)
    def launch(q:Int64,pool:Int64,sel:Int64,sbound:Int64,kmeta:Int64,
               sink:Int64,pos:Int64,cos:Int64,sin:Int64,scratch:Int64,
               diag:Int64,layer:Int64,config:Int64,out:Int64,stream:fx.Stream):
        kernel(q,pool,sel,sbound,kmeta,sink,pos,cos,sin,scratch,diag,layer,config,out).launch(
            grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
