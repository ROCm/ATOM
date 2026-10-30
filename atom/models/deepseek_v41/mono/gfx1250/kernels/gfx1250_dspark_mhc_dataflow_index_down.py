# SPDX-License-Identifier: MIT
"""External index/attention/WO trial, exact image K2 ABI. Full MoE not present."""
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_index_attention import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_wo import (
    WO_PTRS,run_wo,wo_context,stage_reduce,tile_rows,lds_row,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_mhc_dataflow_seam import (
    SEAM_PTRS,run_seam,run_gather,run_seam_gates,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_moe_front import run_moe_front
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_moe_front import front_context,route_shape
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_union import load_route
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_ug import run_ug,stage_lds
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_down import run_down,DOWN_PTRS
@lru_cache(None)
def build_dspark_down(key: LayerPostBuild, debug=False, reduce=False, seam=False, gates=False, front=False, shared_debug=False, union=False, ug=False, ug_debug=False, down=False, down_debug=False):
    assert key.tp==4 and key.ep==key.moe_tp==1
    assert key.tokens in (5,6,10,12,15,18,20,24,25,30,35,36,40,42,48) and not key.index_fp4 and not key.kv_fp8
    assert not key.timeline and key.diag_off==-1
    assert key.index_bound_max in (0,1024,8192,1048576)
    s=key.tokens
    assert s==48 and not key.tok_par and seam and not key.timeline
    assert not debug and not down_debug
    rs=route_shape(key.halves()[1])
    dims=Dims(4)
    offsets=stage_lds(s,rs)["offsets"]
    layout=scratch_layout(key)
    sizes=lds_words(PER_THREAD)
    name=kernel_symbol('gfx1250_dspark_mhc_dataflow_down',s=s,n=key.index_bound_max,tpar=int(key.tok_par),db=int(debug),ar=int(reduce),se=int(seam),ga=int(gates),fr=int(front),e=key.experts,k=key.topk,sd=int(shared_debug),un=int(union),ug=int(ug),ud=int(ug_debug),dn=int(down),dd=int(down_debug))
    @fx.struct
    class IndexSmem:
        keys:fx.Array[fx.Int32,sizes['keys'],16]
        hist:fx.Array[fx.Int32,sizes['hist'],16]
        buf:fx.Array[fx.Int32,sizes['buf'],16]
    @fx.struct
    class ScoreSmem:
        kv:fx.Array[fx.Int32,8*SCORE_LDS_WORDS,16]
    @fx.struct
    class PvSmem:
        tree:fx.Array[fx.Float32,8*64*8,16]
        alpha:fx.Array[fx.Float32,64*HEAD_TILE,16]
        stat:fx.Array[fx.Float32,2*HEAD_TILE,16]
        out:fx.Array[fx.Float32,pv_groups(s,dims.head_tiles.count)*HEAD_TILE*COLS,16]
        kv:fx.Array[fx.Int32,8*BK*COLS//2,16]
    @fx.struct
    class GemvSmem:
        red:fx.Array[fx.Float32,4096,16]
        xl:fx.Array[fx.Int32,tile_rows(s)*lds_row(dims.group_k//4),16]
        xsl:fx.Array[fx.Int32,tile_rows(s)*lds_row(dims.group_k//32),16]
        pend:fx.Array[fx.Float32,s*32,16]
    @fx.struct
    class SliceSmem:
        red:fx.Array[fx.Float32,4096,16]
        rl:fx.Array[fx.Float32,16*128,16]
        fl:fx.Array[fx.Float32,24*128,16]
        pend:fx.Array[fx.Float32,s*32,16]
    @fx.struct
    class MoeFrontSmem:
        red:fx.Array[fx.Float32,4096,16]
        pair:fx.Array[fx.Float32,16*s,16]
        tab:fx.Array[fx.Int32,rs.tab_words,16]
        stage:fx.Array[fx.Int32,sum(stage_lds(s,rs)["words"].values()),16]
    @fx.union
    class Stages:
        index:IndexSmem
        score:ScoreSmem
        pv:PvSmem
        gemv:GemvSmem
        slice:SliceSmem
        front:MoeFrontSmem
    @fx.struct
    class Smem:
        late_meta:fx.Array[fx.Int32,272 + (3*s+2 if s > 24 else 0),16]
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def index(iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64, pscale: Int64, iplan: Int64, itab: Int64, itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64, lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32, ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, sbound: Int64, q: Int64, pool: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64, rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32, ring_slots: Int32, sink: Int64, qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64, wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64, res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64, normed: Int64, gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, layer: Int32, moe_layer: Int32, tl: Int64):
        tid,bid=fx.thread_idx.x,fx.block_idx.x
        allocator=fx.SharedAllocator()
        allocation=allocator.allocate(Smem)
        mem=allocation.peek()
        stages=allocator.allocate(Stages)
        ix=stages.index.peek()
        sc=stages.score.peek()
        pv=stages.pv.peek()
        gm=stages.gemv.peek()
        sl=stages.slice.peek()
        mf=stages.front.peek()
        args_a=attn_post_args(iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog, lstride, isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound, q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out, pre_out, normed)
        bases=peer_bases(peers,4)
        # Explicit metadata-only storage replaces compiler-generated scratch.
        # No weight/input prefetch, task, score or selection value is moved.
        if fresh(tid)==0:
            if const_expr(key.index_bound_max>0):
                for i,name in enumerate(LATE_PTRS):
                    fx.ptr_store(fx.Int32(args_a[name]),mem.late_meta.ptr+2*i)
                    fx.ptr_store(fx.Int32(args_a[name]>>32),mem.late_meta.ptr+2*i+1)
                for i,name in enumerate(LATE_WORDS):
                    fx.ptr_store(args_a[name],mem.late_meta.ptr+2*len(LATE_PTRS)+i)
            fx.ptr_store(fx.Int32(scratch),mem.late_meta.ptr+24)
            fx.ptr_store(fx.Int32(scratch>>32),mem.late_meta.ptr+25)
            if const_expr(key.tok_par):
                for p in range_constexpr(4):
                    fx.ptr_store(fx.Int32(bases[p]),mem.late_meta.ptr+112+2*p)
                    fx.ptr_store(fx.Int32(bases[p]>>32),mem.late_meta.ptr+113+2*p)
                fx.ptr_store(fx.Int32(sym),mem.late_meta.ptr+120)
                fx.ptr_store(fx.Int32(sym>>32),mem.late_meta.ptr+121)
            for i,name in enumerate(ATT_PTRS):
                fx.ptr_store(fx.Int32(args_a[name]),mem.late_meta.ptr+128+2*i)
                fx.ptr_store(fx.Int32(args_a[name]>>32),mem.late_meta.ptr+129+2*i)
            for i,name in enumerate(ATT_WORDS):
                fx.ptr_store(fx.Int32(args_a[name]),mem.late_meta.ptr+146+i)
            if const_expr(debug):
                fx.ptr_store(fx.Int32(out),mem.late_meta.ptr+152)
                fx.ptr_store(fx.Int32(out>>32),mem.late_meta.ptr+153)
            for i,name in enumerate(WO_PTRS):
                fx.ptr_store(fx.Int32(args_a[name]),mem.late_meta.ptr+160+2*i)
                fx.ptr_store(fx.Int32(args_a[name]>>32),mem.late_meta.ptr+161+2*i)
            fx.ptr_store(fx.Int32(sym),mem.late_meta.ptr+168)
            fx.ptr_store(fx.Int32(sym>>32),mem.late_meta.ptr+169)
            for p in range_constexpr(4):
                fx.ptr_store(fx.Int32(bases[p]),mem.late_meta.ptr+170+2*p)
                fx.ptr_store(fx.Int32(bases[p]>>32),mem.late_meta.ptr+171+2*p)
            fx.ptr_store(rank,mem.late_meta.ptr+178)
            if const_expr(debug):
                fx.ptr_store(fx.Int32(tl),mem.late_meta.ptr+180)
                fx.ptr_store(fx.Int32(tl>>32),mem.late_meta.ptr+181)
            if const_expr(reduce):
                fx.ptr_store(fx.Int32(normed),mem.late_meta.ptr+182)
                fx.ptr_store(fx.Int32(normed>>32),mem.late_meta.ptr+183)
            if const_expr(seam):
                for i,name in enumerate(SEAM_PTRS):
                    fx.ptr_store(fx.Int32(args_a[name]),mem.late_meta.ptr+192+2*i)
                    fx.ptr_store(fx.Int32(args_a[name]>>32),mem.late_meta.ptr+193+2*i)
                fx.ptr_store(moe_layer,mem.late_meta.ptr+218)
                fx.ptr_store(fx.Int32(tl),mem.late_meta.ptr+220)
                fx.ptr_store(fx.Int32(tl>>32),mem.late_meta.ptr+221)
            if const_expr(front):
                for i,value in enumerate((gate_w,bias,sgu,sgu_s)):
                    fx.ptr_store(fx.Int32(value),mem.late_meta.ptr+224+2*i)
                    fx.ptr_store(fx.Int32(value>>32),mem.late_meta.ptr+225+2*i)
            if const_expr(shared_debug):
                fx.ptr_store(fx.Int32(w2),mem.late_meta.ptr+234)
                fx.ptr_store(fx.Int32(w2>>32),mem.late_meta.ptr+235)
            if const_expr(union):
                dump=out if ug else w13
                fx.ptr_store(fx.Int32(dump),mem.late_meta.ptr+232)
                fx.ptr_store(fx.Int32(dump>>32),mem.late_meta.ptr+233)
            if const_expr(ug):
                for i,value in enumerate((w13,w13_s)):
                    fx.ptr_store(fx.Int32(value),mem.late_meta.ptr+240+2*i)
                    fx.ptr_store(fx.Int32(value>>32),mem.late_meta.ptr+241+2*i)
            if const_expr(ug_debug):
                for i,value in enumerate((sw2,sw2_s)):
                    fx.ptr_store(fx.Int32(value),mem.late_meta.ptr+244+2*i)
                    fx.ptr_store(fx.Int32(value>>32),mem.late_meta.ptr+245+2*i)
            if const_expr(down):
                for i,value in enumerate((w2,w2_s,sw2,sw2_s,out)):
                    fx.ptr_store(fx.Int32(value),mem.late_meta.ptr+248+2*i)
                    fx.ptr_store(fx.Int32(value>>32),mem.late_meta.ptr+249+2*i)
            if const_expr(down_debug):
                fx.ptr_store(fx.Int32(tl),mem.late_meta.ptr+260)
                fx.ptr_store(fx.Int32(tl>>32),mem.late_meta.ptr+261)
        gpu.barrier()
        for name in SEAM_PTRS:args_a[name]=fx.Int64(0)
        for name in WO_PTRS:args_a[name]=fx.Int64(0)
        for name in ATT_PTRS:args_a[name]=fx.Int64(0)
        for name in ATT_WORDS:args_a[name]=fx.Int32(0)
        for name in LATE_PTRS:args_a[name]=fx.Int64(0)
        for name in LATE_WORDS:args_a[name]=fx.Int32(0)
        mb=Mailbox(layer,gfx1250=True)
        c=dict(S=s,tid=tid,bid=bid,lane=tid%32,wave=tid//32,args=args_a,
               put=mb.put,put_words=mb.put_words,poll=mb.poll,
               tok_par=key.tok_par,index_fp4=False,rank=rank,d=dims,
               late_meta=mem.late_meta.ptr,
               scratch_layout=layout,
               peer_addr=lambda p:late_pointer(dict(late_meta=mem.late_meta.ptr),112+2*p),peer=peer_offsets(s,dims),
               sel_lds=dict(keys=ix.keys.ptr,hist=ix.hist.ptr,buf=ix.buf.ptr),
               score_red=fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type,fx.AddressSpace.Shared,4),ix.keys.ptr))
        if const_expr(key.index_bound_max>0):
            if iact!=0:run_indexer(c,s,bid,key.index_bound_max)
        # Original per-CTA union handover, before score task placement.
        gpu.barrier()
        c['index_bound_max']=key.index_bound_max
        c['att']=dict(score_kv=sc.kv.ptr,tree=pv.tree.ptr,alpha=pv.alpha.ptr,
                      stat=pv.stat.ptr,out=pv.out.ptr,kv=pv.kv.ptr)
        c['defer_tdm_rows']=True
        per=score_tasks(s,dims.head_tiles.count)
        for task in range(first_task(bid,SCORE0),s*dims.head_tiles.count*per,BLOCKS):
            ca=attention_context(c,'score')
            stage_attn_score(ca,task,per)
        groups=pv_groups(s,dims.head_tiles.count)
        for task in range(first_task(bid,IRQ0),s*dims.head_tiles.count*GROUPS//groups,BLOCKS):
            ca=attention_context(c,'pv')
            if const_expr(debug):
                ca['debug_out']=fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type,
                                           fx.AddressSpace.Global,2),late_pointer(c,152))
            stage_attn_irq(ca,task,groups,debug)
        gpu.barrier()
        c.update(red=gm.red.ptr,xl=gm.xl.ptr,xsl=gm.xsl.ptr,pend_lds=gm.pend.ptr,
                 debug=debug,put_bf=mb.put_bf)
        if const_expr(debug):
            c['debug_woa']=fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type,fx.AddressSpace.Global,2),late_pointer(c,180))
        run_wo(c,key,bid)
        if const_expr(reduce):
            cr=wo_context(c,'reduce')
            if const_expr(key.tok_par):cr=dict(cr,S=s//4)
            output=fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type,fx.AddressSpace.Global,2),late_pointer(c,182))
            for task in range(first_task(bid,0),5120//32,BLOCKS):
                stage_reduce(cr,task)
                for begin in range_constexpr(0,cr['S']*32,THREADS):
                    at=cr['tid']+begin
                    if at<cr['S']*32:
                        fx.ptr_store(fx.ptr_load(cr['pend_lds']+at).to(fx.BFloat16),
                                     output+(at//32)*5120+task*32+at%32)
                gpu.barrier()
        if const_expr(seam):
            assert not reduce
            gpu.barrier()
            c.update(rl=sl.rl.ptr,fl=sl.fl.ptr,red=sl.red.ptr,pend_lds=sl.pend.ptr)
            run_seam(c,key,bid)
            # This standalone probe stops after XG and optionally checks gates.
            # Full MoE will own gather placement and the before_ug callback.
            gpu.barrier()
            if const_expr(front):
                c.update(front_red=mf.red.ptr,front_pair=mf.pair.ptr,front_xl=mf.stage.ptr+offsets['x8l'],front_xsl=mf.stage.ptr+offsets['x8sl'],front_tab=mf.tab.ptr,front_stage=mf.stage.ptr,shared_debug=shared_debug,ug_debug=ug_debug,down_debug=down_debug)
                run_moe_front(c,key,bid)
            else:run_gather(c,key,bid)
            if const_expr(gates):
                gpu.barrier()
                run_seam_gates(c,key,bid)
        if const_expr(ug):
            assert seam and front and gates
            gpu.barrier()
            cu=run_ug(c,key,bid)
        if const_expr(union):
            assert seam and front and gates
            gpu.barrier()
            if const_expr(not ug):
                cu=front_context(c,key,'route')
                cu['red']=mf.red.ptr
                load_route(cu,mf.tab.ptr)
            elif const_expr(not rs.multi and rs.max_u*rs.ug_parts<=BLOCKS):load_route(cu,mf.tab.ptr)
            output=fx.inttoptr(fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4),late_pointer(c,232))
            for i in range_constexpr(0,rs.tab_words,THREADS):
                at=cu['tid']+i
                if at<rs.tab_words:fx.ptr_store(fx.ptr_load(mf.tab.ptr+at),output+bid*rs.tab_words+at)
        if const_expr(down):
            assert seam and gates and front and ug
            assert not (debug or reduce or union or shared_debug or ug_debug)
            gpu.barrier()
            run_down(c,key,bid)
        if const_expr(key.tok_par and key.index_bound_max>0):await_candidates(c,s,bid)
    @launcher(name)
    def launch(iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64, pscale: Int64, iplan: Int64, itab: Int64, itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64, lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32, ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, sbound: Int64, q: Int64, pool: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64, rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32, ring_slots: Int32, sink: Int64, qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64, wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64, res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64, normed: Int64, gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, layer: Int32, moe_layer: Int32, tl: Int64, stream:fx.Stream):
        index(iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog, lstride, isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound, q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out, pre_out, normed, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out, scratch, sym, peers, rank, layer, moe_layer, tl).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
