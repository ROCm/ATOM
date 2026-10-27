# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 output projections: original input staging and K ownership.

WOA keeps a N32 task as two N16 tiles, four waves each taking a contiguous
K range. Weights load before the tagged XO input wait. Native wave32 matrix
fragments are restored to the mother's red layout before its original
wave-ordered sum, BF16 rounding and group32 output quantization.
"""
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64

from atom.models.deepseek_v41.mono.kernels.attention import (
    BK, COLS, HEAD_DIM, HEAD_TILE, SCORE_LDS_WORDS,
)
from atom.models.deepseek_v41.mono.kernels.attn_pre import lds_row, poll_copy
from atom.models.deepseek_v41.mono.kernels.attn_post import QUANT_FLOOR, WOA_SPLIT_K
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_attention_score import stage_attn_score_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_attention_pv import stage_attn_irq_c1
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_indexer import vector_address
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_projection import _weight_step
from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_sync import reference_mailbox, publish_plain, scalar_address
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, pow2
from atom.mono.device.ops import (
    bf16_round, fp8_pack4, fresh, kernel_symbol, lane_gather, launcher, row_sum, rsrc, traced, uniform,
)
from atom.mono.device.sync import sreg
from atom.mono.plan.execution import BLOCKS, first_task
from atom.mono.plan.layout import pair_layout

THREADS, WAVES = 256, 8


@traced
def woa_loads_c1(c, task):
    """Original contiguous K-split and two row tiles, all loads before wait."""
    inner = c["d"].group_k
    split = WOA_SPLIT_K
    per = inner // 128 // split
    wave = c["wave"]
    rg = task * 2 + wave // split
    loaded = [_weight_step(c,rsrc(c["args"]["woa"]),rsrc(c["args"]["woa_s"]),
                  inner,rg,(wave%split)*per+i) for i in range(per)]
    weights = fx.Vector.from_elements([loaded[i][0][j] for i in range(per) for j in range(16)],fx.Int32)
    scales = fx.Vector.from_elements([loaded[i][1] for i in range(per)],fx.Int32)
    fx.rocdl.sched_barrier(0)
    return [(fx.Vector.from_elements([weights[i*16+j] for j in range(16)],fx.Int32),scales[i])
            for i in range(per)]


@traced
def woa_matrices_c1(c, ops):
    lane,wave,inner = c["lane"],c["wave"],c["d"].group_k
    half = lane // 16
    split,per = WOA_SPLIT_K,inner//128//WOA_SPLIT_K
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
        16,16,128,fx.Float8E4M3FN,fx.Float8E4M3FN,fx.Float32))
    accum = fx.make_rmem_tensor(8,fx.Float32)
    accum.store(fx.full(8,0.,fx.Float32))
    for i in range_constexpr(per):
        st = (wave%split)*per+i
        chunks = [fx.Vector(fx.ptr_load(c["xl"]+st*32+half*4+j*8,
                    result_type=fx.Vector.make_type(4,fx.Int32))) for j in range(4)]
        xv = fx.Vector.from_elements([chunks[p][j] for p in range(4) for j in range(4)],fx.Int32)
        codes = [fx.ptr_load(c["xsl"]+st*4+j) for j in range(4)]
        sx = codes[0] | (codes[1]<<8) | (codes[2]<<16) | (codes[3]<<24)
        weight,sw = ops[i]
        a,b = fx.make_rmem_tensor(16,fx.Int32),fx.make_rmem_tensor(16,fx.Int32)
        a.store(weight)
        b.store(xv)
        fx.gemm(atom,accum,a,b,accum,scale_a=sw,scale_b=sx)
    if lane%16 == 0:
        values = accum.load()
        for j in range_constexpr(8):
            row = half*8+j
            # Mother MFMA accumulator [wave64lane][4], token0 only.
            at = (wave*64+16*(row//4))*4+row%4
            fx.ptr_store(values[j],c["red"]+at)
    gpu.barrier()


@traced
def woa_tile_c1(c, task, ops, debug=False):
    d,tid,lane = c["d"],c["tid"],c["lane"]
    group = task // (d.woa_tasks // d.groups)
    words,scales = d.group_k//4,d.group_k//32
    poll_copy(c,tid,THREADS,[
        (c["xo"],lambda u:group*words+u,words,c["xl"],words),
        (c["xos"],lambda u:group*scales+u,scales,c["xsl"],scales),
    ])
    gpu.barrier()
    woa_matrices_c1(c,ops)
    row = tid%32
    tile = row//16
    # Same mother ordered sum and BF16 rounding before quantization.
    y = bf16_round(row_sum(c["red"],row%16,0,[tile*WOA_SPLIT_K+w for w in range(WOA_SPLIT_K)]))
    maximum = abs(y)
    for off in (1,2,4,8,16):
        maximum = fx.max(maximum,maximum.shuffle_xor(off,32))
    maximum = fx.max(maximum,fx.Float32(QUANT_FLOOR))
    code = code_ceil(maximum*fx.Float32(1./FP8_MAX))
    q = clamp_fp8(y/pow2(code))
    packed = fp8_pack4(q,*[
        lane_gather(q.bitcast(fx.Int32),fx.min(lane+k,31)).bitcast(fx.Float32)
        for k in (1,2,3)])
    if tid<32:
        if const_expr(debug):
            fx.ptr_store(y.to(fx.BFloat16),c["debug_woa"]+task*32+row)
        if row%4 == 0:
            c["put_words"](c["x8b"],(task*32+row)//4,[packed])
        if row == 0:
            c["put"](c["x8bs"],task,code)
    gpu.barrier()


def attention_wo_layout_c1(tp=2):
    d=Dims(tp)
    return pair_layout({"am":64*d.heads,"al":64*d.heads,
        "ap":d.head_tiles.count*64*64*2,"xo":d.xo_words,
        "xos":d.heads*HEAD_DIM//32,"x8b":d.yb_words,"x8bs":d.o_rows//32}.items())


def attention_wo_symbol_c1(debug=False):
    return kernel_symbol("gfx1250_reference_attention_woa",b=BLOCKS,t=THREADS,db=int(debug))


@lru_cache(None)
def build_reference_attention_woa_c1(debug=False):
    """Original score→IRQ→WOA binding; WOB/TP integration follows separately."""
    layout,d,name=attention_wo_layout_c1(),Dims(2),attention_wo_symbol_c1(debug)
    @fx.struct
    class Shared:
        red:fx.Array[fx.Float32,WAVES*2*64*4,16]
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
    @fx.struct
    class GemvLds:
        xl:fx.Array[fx.Int32,lds_row(d.group_k//4),16]
        xsl:fx.Array[fx.Int32,lds_row(d.group_k//32),16]
    @fx.union
    class Smem:
        score:ScoreLds
        pv:PvLds
        gemv:GemvLds
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def kernel(q:Int64,pool:Int64,sel:Int64,sbound:Int64,kmeta:Int64,
               sink:Int64,pos:Int64,cos:Int64,sin:Int64,scratch:Int64,
               diag:Int64,layer:Int64,config:Int64,out:Int64,
               woa:Int64,woa_s:Int64,woa_out:Int64):
        tid,bid=fx.thread_idx.x,fx.block_idx.x
        allocator=fx.SharedAllocator()
        shared=allocator.allocate(Shared).peek()
        mem=allocator.allocate(Smem)
        sc,pv,gm=mem.score.peek(),mem.pv.peek(),mem.gemv.peek()
        iptr=fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        cfg=fx.inttoptr(iptr,config)
        failure=fx.inttoptr(iptr,diag+fx.Int64(BLOCKS*8*4))
        mb=reference_mailbox(fx.ptr_load(fx.inttoptr(iptr,layer))+64,diag,
                             {key:i+1 for i,key in enumerate(layout)},failure)
        # These late-stage addresses must not pin scalar descriptors through
        # the whole score/PV control flow (the initial binding spilled9SGPRs).
        woa,woa_s=vector_address(woa),vector_address(woa_s)
        sink,pos=vector_address(sink),vector_address(pos)
        cos,sin=vector_address(cos),vector_address(sin)
        args=dict(q=q,pool=pool,sel=sel,sbound=sbound,kmeta=kmeta,topk=512,
                  sink=sink,pos=pos,cos=cos,sin=sin,woa=woa,woa_s=woa_s,
                  ring_off=fx.ptr_load(cfg+1),ring_slots=512,iact=fx.ptr_load(cfg),
                  qk_scale=fx.ptr_load(cfg+2))
        out,woa_out=vector_address(out),vector_address(woa_out)
        bptr=fx.PointerType.get(fx.BFloat16.ir_type,fx.AddressSpace.Global,2)
        c=dict(S=1,tid=tid,bid=bid,lane=tid%32,wave=tid//32,d=d,args=args,
               att=dict(score_kv=sc.kv.ptr,tree=pv.tree.ptr,alpha=pv.alpha.ptr,
                        stat=pv.stat.ptr,out=pv.out.ptr,kv=pv.kv.ptr),
               red=shared.red.ptr,xl=gm.xl.ptr,xsl=gm.xsl.ptr,
               put=mb.put,put_words=mb.put_words,poll=mb.poll,
               debug_out=fx.inttoptr(bptr,out),debug_woa=fx.inttoptr(bptr,woa_out))
        scratch=vector_address(scratch)
        for region in layout:
            c[region]=sreg(scratch,layout[region][0],region)
        for task in range(first_task(bid,0),d.head_tiles.count*8,BLOCKS):
            stage_attn_score_c1(c,task,8)
        for task in range(first_task(bid,64),d.head_tiles.count*16,BLOCKS):
            stage_attn_irq_c1(c,task,1,debug)
        gpu.barrier()
        task=first_task(bid,160)
        if task<d.woa_tasks:
            ops=woa_loads_c1(c,task)
            woa_tile_c1(c,task,ops,debug)
    @launcher(name)
    def launch(q:Int64,pool:Int64,sel:Int64,sbound:Int64,kmeta:Int64,
               sink:Int64,pos:Int64,cos:Int64,sin:Int64,scratch:Int64,
               diag:Int64,layer:Int64,config:Int64,out:Int64,
               woa:Int64,woa_s:Int64,woa_out:Int64,stream:fx.Stream):
        kernel(q,pool,sel,sbound,kmeta,sink,pos,cos,sin,scratch,diag,layer,config,
               out,woa,woa_s,woa_out).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch


@traced
def wob_loads_c1(c, task):
    """Mother N16 task: all eight interleaved K partials before input wait."""
    inner, wave = c["d"].o_rows, c["wave"]
    per = inner // 128 // WAVES
    # Both bases are immutable, wave-uniform layer parameters. Restore that
    # fact where the scalar buffer descriptors are built; their earlier
    # vector_address lifetime must not turn each prefetch into an exec loop.
    rw, rs = rsrc(scalar_address(c["args"]["wob"])), rsrc(scalar_address(c["args"]["wob_s"]))
    loaded = [_weight_step(c, rw, rs, inner, task, wave + WAVES*i)
              for i in range(per)]
    weights = fx.Vector.from_elements(
        [loaded[i][0][j] for i in range(per) for j in range(16)], fx.Int32)
    scales = fx.Vector.from_elements([loaded[i][1] for i in range(per)], fx.Int32)
    fx.rocdl.sched_barrier(0)
    return [(fx.Vector.from_elements([weights[i*16+j] for j in range(16)], fx.Int32), scales[i])
            for i in range(per)]


@traced
def wob_matrices_c1(c, ops):
    """One staged input, eight original K streams and original red addresses."""
    lane, wave, inner = c["lane"], c["wave"], c["d"].o_rows
    half = lane // 16
    atom = fx.make_mma_atom(fx.rocdl.WMMAScale(
        16, 16, 128, fx.Float8E4M3FN, fx.Float8E4M3FN, fx.Float32))
    accum = fx.make_rmem_tensor(8, fx.Float32)
    accum.store(fx.full(8, 0., fx.Float32))
    for i in range_constexpr(inner // 128 // WAVES):
        st = wave + WAVES*i
        chunks = [fx.Vector(fx.ptr_load(c["xl"]+st*32+half*4+j*8,
                    result_type=fx.Vector.make_type(4, fx.Int32))) for j in range(4)]
        xv = fx.Vector.from_elements([chunks[p][j] for p in range(4) for j in range(4)], fx.Int32)
        codes = [fx.ptr_load(c["xsl"]+st*4+j) for j in range(4)]
        sx = codes[0] | (codes[1]<<8) | (codes[2]<<16) | (codes[3]<<24)
        weight, sw = ops[i]
        a, b = fx.make_rmem_tensor(16, fx.Int32), fx.make_rmem_tensor(16, fx.Int32)
        a.store(weight)
        b.store(xv)
        fx.gemm(atom, accum, a, b, accum, scale_a=sw, scale_b=sx)
    if lane%16 == 0:
        values = accum.load()
        for j in range_constexpr(8):
            row = half*8+j
            fx.ptr_store(values[j], c["red"]+(wave*64+16*(row//4))*4+row%4)
    gpu.barrier()


@traced
def stage_wob_c1(c, task, ops):
    """Rank-local BF16 pairs directly to every original peer attention region."""
    from atom.models.deepseek_v41.mono.kernels.attn_post import attn_region_pair
    from atom.mono.device.sync import preg
    wob_matrices_c1(c, ops)
    if c["tid"] < 8:
        col = task*16+2*c["tid"]
        v0 = row_sum(c["red"], 2*c["tid"], 0)
        v1 = row_sum(c["red"], 2*c["tid"]+1, 0)
        for p in range_constexpr(c["d"].tp):
            dst = preg(c["peer_addr"](p), c["peer"]["attn"], "attn")
            c["put_bf"](dst, 2*attn_region_pair(c["rank"], 0, 1, col), [v0, v1])
    gpu.barrier()


@traced
def stage_attn_reduce_c1(c, task):
    """Original rank-order FP32 sum and BF16 attention slice in private LDS.

    This can be isolated before connecting slice_prefetch/stage_slice. A full
    K2 binding must issue that original prefetch before entering this wait.
    """
    from atom.models.deepseek_v41.mono.kernels.attn_post import attn_region_pair
    from atom.mono.device.ranks import sum_partials
    from atom.mono.device.sync import preg
    if c["tid"] < 16:
        col = task*32+2*c["tid"]
        own = preg(c["sym"], c["peer"]["attn"], "attn")
        v0, v1 = sum_partials(c["poll"], own,
            lambda src: attn_region_pair(src, 0, 1, col), c["d"].tp)
        fx.ptr_store(bf16_round(v0), c["pend_lds"]+2*c["tid"])
        fx.ptr_store(bf16_round(v1), c["pend_lds"]+2*c["tid"]+1)
    gpu.barrier()


def attention_ffn_front_layout_c1(tp=2):
    """WO fields followed by original seam partials and norm readiness."""
    wo = attention_wo_layout_c1(tp)
    fields = {name:count//8 for name,(_,count) in wo.items()}
    fields.update(lin=5120//2, pmix=160*25, xrdy=1)
    return pair_layout(fields.items())


FFN_FRONT_ARGS = ("res_in", "post_in", "comb_in", "pre_in", "hc_fn",
                  "res_out", "attn_w", "normed", "reduce_out")
FFN_INPUT_ARGS = FFN_FRONT_ARGS+("gate_w", "bias")
FFN_SHARED_ARGS = FFN_INPUT_ARGS+("sgu", "sgu_s")
FFN_GATE_ARGS = FFN_SHARED_ARGS+("hc_scale", "hc_base", "pre_out", "post_out", "comb_out")
FFN_UG_ARGS = FFN_GATE_ARGS+("w13", "w13_s")
FFN_DOWN_ARGS = FFN_UG_ARGS+("w2", "w2_s", "sw2", "sw2_s", "out")
INDEX_ARGS = ("plane", "itab", "iq", "iw", "ibound", "icand", "table",
              "ilog", "ibmax", "isel", "irow", "icout")
# Immutable binding addresses restored after the index's large tree live set.
# These are parameter bases, not payloads or an extra inter-stage mailbox.
INDEX_ATTENTION_ARGS = ("q", "pool", "sbound", "kmeta", "sink", "pos", "cos", "sin",
                        "out", "woa", "woa_s", "woa_out", "wob", "wob_s",
                        "sym", "peers", "rank", "config")


def attention_wob_symbol_c1(debug=False, reduce=True, seam=False, ffn=False, router=False, route=False, shared=False, gate=False, ug=False, down=False, ffn_reduce=True, index_bound_max=0, ring_slots=512, tp=2):
    fields = dict(b=BLOCKS, t=THREADS, db=int(debug), ar=int(reduce))
    if tp != 2:
        fields["tp"] = tp
    if seam:
        fields["sm"] = 1
    if ffn:
        fields.update(xq=1,rt=int(router))
    if route:
        fields["route"] = 1
    if shared:
        fields["shared"] = 1
    if gate:
        fields["gate"] = 1
    if ug:
        fields["ug"] = 1
    if down:
        fields.update(down=1,fr=int(ffn_reduce))
    if index_bound_max:
        fields["ix"] = index_bound_max
    if ring_slots != 512:
        fields["rs"] = ring_slots
    return kernel_symbol("gfx1250_reference_attention_wob", **fields)


@lru_cache(None)
def build_reference_attention_wob_c1(debug=False, reduce=True, seam=False, ffn=False, router=False, route=False, shared=False, gate=False, ug=False, down=False, ffn_reduce=True, index_bound_max=0, ring_slots=512, tp=2):
    """Mother score→IRQ→WOA→WOB binding, with an isolated first TP consumer.

    The existing PeerBuffer/StepMailboxes owns peer mappings and step lifetime;
    this builder only consumes its unchanged sym/peers/rank arguments.
    """
    from atom.models.deepseek_v41.mono.kernels.dims import HIDDEN
    from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
    from atom.mono.device.ranks import peer_bases
    from atom.mono.device.ops import CM_DEV, load_ptr64
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_k1 import slice_prefetch_c1, stage_slice_c1
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_norm import stage_norm_c1
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_ffn import ffn_input_layout_c1, run_ffn_input_c1
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_route import stage_route_c1
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_shared import run_shared_c1
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_gate import stage_gate_c1
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_ug import run_ug_c1
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_down import run_down_c1
    from atom.models.deepseek_v41.mono.kernels.moe import wsl_words, lds_tail
    from atom.models.deepseek_v41.mono.kernels.moe_shape import RouteShape
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_route import C1RouteShape
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_indexer import run_indexer_c1, PER_THREAD
    from atom.models.deepseek_v41.mono.kernels.gfx1250_reference_topk import lds_words
    if seam and not reduce:
        raise ValueError("the FFN seam requires the first TP reduction")
    if (ffn and not seam) or (router and not ffn):
        raise ValueError("FFN input requires the seam; router requires FFN input")
    if route and not router:
        raise ValueError("route requires router")
    if (shared and not route) or (gate and not shared):
        raise ValueError("shared requires route; after-shared gate requires shared")
    if ug and not gate:
        raise ValueError("UG requires the original after-shared gate")
    if down and not ug:
        raise ValueError("down requires original UG")
    if index_bound_max and (not ffn or not 0<index_bound_max<=16384 or index_bound_max%128):
        raise ValueError("joint c1 FP8 index requires FFN and a multiple-of128 bound up to16384")
    if not isinstance(ring_slots, int) or ring_slots < 1:
        raise ValueError("positive ring geometry required")
    layout = ffn_input_layout_c1(index_bound_max, tp) if ffn else attention_ffn_front_layout_c1(tp) if seam else attention_wo_layout_c1(tp)
    d, name = Dims(tp), attention_wob_symbol_c1(debug, reduce, seam, ffn, router, route, shared, gate, ug, down, ffn_reduce, index_bound_max, ring_slots, tp)
    rs = (C1RouteShape if tp == 2 else RouteShape)(384,6,1,tp,1)
    index_sizes = lds_words(PER_THREAD)
    ffn_table_args = FFN_DOWN_ARGS if down else FFN_UG_ARGS if ug else FFN_GATE_ARGS if gate else FFN_SHARED_ARGS if shared else FFN_INPUT_ARGS
    @fx.struct
    class Shared:
        red:fx.Array[fx.Float32, WAVES*2*64*4, 16]
    @fx.struct
    class ScoreLds:
        kv:fx.Array[fx.Int32, WAVES*SCORE_LDS_WORDS, 16]
    @fx.struct
    class PvLds:
        tree:fx.Array[fx.Float32, 8*64*8, 16]
        alpha:fx.Array[fx.Float32, 64*HEAD_TILE, 16]
        stat:fx.Array[fx.Float32, 2*HEAD_TILE, 16]
        out:fx.Array[fx.Float32, HEAD_TILE*COLS, 16]
        kv:fx.Array[fx.Int32, WAVES*BK*COLS//2, 16]
    @fx.struct
    class GemvLds:
        xl:fx.Array[fx.Int32, lds_row(d.group_k//4), 16]
        xsl:fx.Array[fx.Int32, lds_row(d.group_k//32), 16]
        pend:fx.Array[fx.Float32, 32, 16]
    @fx.struct
    class SliceLds:
        rl:fx.Array[fx.Float32, 128, 16]
        fl:fx.Array[fx.Float32, 24*128, 16]
        pend:fx.Array[fx.Float32, 32, 16]
    @fx.struct
    class SharedFfnLds:
        xl:fx.Array[fx.Int32, lds_row(HIDDEN//4), 16]
        xsl:fx.Array[fx.Int32, lds_row(HIDDEN//32), 16]
        pair:fx.Array[fx.Float32, 16, 16]
    @fx.struct
    class UgLds:
        tab:fx.Array[fx.Int32, rs.tab_words, 16]
        xl:fx.Array[fx.Int32, lds_row(HIDDEN//4), 16]
        xsl:fx.Array[fx.Int32, lds_row(HIDDEN//32), 16]
        wsl:fx.Array[fx.Int32, WAVES*wsl_words(d), 16]
    @fx.struct
    class DownLds:
        tab:fx.Array[fx.Int32, rs.tab_words, 16]
        wsl:fx.Array[fx.Int32, WAVES*wsl_words(d), 16]
        sxl:fx.Array[fx.Int32, lds_row(d.sh_inter//4)+lds_tail(d.sh_inter), 16]
        sxsl:fx.Array[fx.Int32, lds_row(d.sh_inter//32), 16]
        contrib:fx.Array[fx.Int32, 2*rs.picks*16, 16]
        sdl:fx.Array[fx.Int32, 32, 16]
        pair:fx.Array[fx.Float32, 16, 16]
    @fx.struct
    class IndexLds:
        keys:fx.Array[fx.Int32,index_sizes["keys"],16]
        hist:fx.Array[fx.Int32,index_sizes["hist"],16]
        buf:fx.Array[fx.Int32,index_sizes["buf"],16]
    @fx.union
    class Smem:
        score:ScoreLds
        pv:PvLds
        gemv:GemvLds
        slice:SliceLds
        shared_ffn:SharedFfnLds
        ug:UgLds
        down:DownLds
        index:IndexLds
    @flyc.kernel(name=name, known_block_size=[THREADS,1,1])
    def kernel(q:Int64, pool:Int64, sel:Int64, sbound:Int64, kmeta:Int64,
               sink:Int64, pos:Int64, cos:Int64, sin:Int64, scratch:Int64,
               diag:Int64, layer:Int64, config:Int64, out:Int64,
               woa:Int64, woa_s:Int64, woa_out:Int64, wob:Int64, wob_s:Int64,
               sym:Int64, peers:Int64, rank:Int64, reduce_out:Int64):
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        allocator = fx.SharedAllocator()
        shared_mem, mem = allocator.allocate(Shared).peek(), allocator.allocate(Smem)
        sc, pv, gm = mem.score.peek(), mem.pv.peek(), mem.gemv.peek()
        sl = mem.slice.peek()
        sf = mem.shared_ffn.peek()
        ul = mem.ug.peek()
        dl = mem.down.peek()
        iptr = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        cfg = fx.inttoptr(iptr, config)
        failure = fx.inttoptr(iptr, diag+fx.Int64(BLOCKS*8*4))
        regions = {key:i+1 for i,key in enumerate(layout)}
        regions["attn"] = len(regions)+1
        layer_value = fx.ptr_load(fx.inttoptr(iptr, layer))
        mb = reference_mailbox(layer_value+64,
                               diag, regions, failure)
        if const_expr(index_bound_max>0):
            # The index tree and later-stage kernargs cannot share a pinned
            # live set on this target. SGPR-only and VGPR-held bindings both
            # spilled in CPU compilation. Read later parameter bases from
            # the same immutable table at the attention boundary instead.
            reduce_out = vector_address(reduce_out)
            layer_value = fresh(layer_value)
            scratch = vector_address(scratch)
            ix = mem.index.peek()
            index_table = scalar_address(reduce_out)
            # Match the accepted standalone index binding: scoring inputs
            # are uniform scalar bases; later selection/output bases live
            # in VGPRs until their descriptor is consumed. Making every
            # scoring base opaque adds waterfall masks to the tree live set.
            index_args = {key:(scalar_address if i<5 else vector_address)(
                              load_ptr64(index_table,len(ffn_table_args)+i))
                          for i,key in enumerate(INDEX_ARGS)}
            index_args.update(kmeta=kmeta,iact=fx.ptr_load(cfg),
                itab_stride=index_bound_max//8,itab_len=fx.ptr_load(cfg+5),
                lstride=index_bound_max,bstride=index_bound_max//8,
                iprod=fx.ptr_load(cfg+3),ilift=fx.ptr_load(cfg+4),
                rows_per_page=fresh(fx.ptr_load(cfg+6)),page_rows=fx.ptr_load(cfg+7),
                main_off=fx.ptr_load(cfg+8))
            ci = dict(S=1,tid=tid,bid=bid,lane=tid%32,wave=tid//32,
                args=index_args,put=mb.put,put_words=mb.put_words,poll=mb.poll,
                sel_lds=dict(keys=ix.keys.ptr,hist=ix.hist.ptr,buf=ix.buf.ptr),
                score_red=fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type,
                    fx.AddressSpace.Shared,4),ix.keys.ptr))
            for key in layout:
                if key=="isf" or key=="irdy" or key.startswith("isel") or key.startswith("iblk"):
                    ci[key] = sreg(scratch,layout[key][0],key)
            if ci["args"]["iact"]!=0:
                run_indexer_c1(ci,bid,index_bound_max)
            # Original run_attn_post barrier transfers the LDS union, including
            # CTAs with no index task. It is a CTA join, never a grid barrier.
            gpu.barrier()
            attention_table = scalar_address(reduce_out)
            restored = {key:load_ptr64(attention_table,len(ffn_table_args)+len(INDEX_ARGS)+i)
                        for i,key in enumerate(INDEX_ATTENTION_ARGS)}
            q, pool, sbound, kmeta = (restored[key] for key in ("q","pool","sbound","kmeta"))
            sink, pos, cos, sin = (restored[key] for key in ("sink","pos","cos","sin"))
            out, woa, woa_s, woa_out = (restored[key] for key in ("out","woa","woa_s","woa_out"))
            wob, wob_s, sym, peers = (restored[key] for key in ("wob","wob_s","sym","peers"))
            rank, config = restored["rank"], restored["config"]
            cfg = fx.inttoptr(iptr,config)
            sel = load_ptr64(attention_table,len(ffn_table_args)+INDEX_ARGS.index("irow"))
        if const_expr(down):
            ffn_layer = fresh(layer_value)
        if const_expr(ffn and not down):
            fmb = reference_mailbox(layer_value+128, diag, regions, failure)
        woa, woa_s = vector_address(woa), vector_address(woa_s)
        wob, wob_s = vector_address(wob), vector_address(wob_s)
        sink, pos = vector_address(sink), vector_address(pos)
        cos, sin = vector_address(cos), vector_address(sin)
        sym, peers = vector_address(sym), vector_address(peers)
        if const_expr(not down):
            bases = [vector_address(base) for base in peer_bases(peers, tp)]
        args = dict(q=q, pool=pool, sel=sel, sbound=sbound, kmeta=kmeta, topk=512,
                    sink=sink, pos=pos, cos=cos, sin=sin, woa=woa, woa_s=woa_s,
                    wob=wob, wob_s=wob_s, ring_off=fx.ptr_load(cfg+1), ring_slots=ring_slots,
                    iact=fx.ptr_load(cfg), qk_scale=fx.ptr_load(cfg+2))
        out, woa_out, reduce_out = map(vector_address, (out, woa_out, reduce_out))
        bptr = fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 2)
        rank_value = fx.ptr_load(fx.inttoptr(iptr,rank))
        if const_expr(down):
            rank_value = fresh(rank_value)
        c = dict(S=1, tid=tid, bid=bid, lane=tid%32, wave=tid//32, d=d, args=args,
                 defer_tdm_rows=index_bound_max>0,
                 att=dict(score_kv=sc.kv.ptr, tree=pv.tree.ptr, alpha=pv.alpha.ptr,
                          stat=pv.stat.ptr, out=pv.out.ptr, kv=pv.kv.ptr),
                 red=shared_mem.red.ptr, xl=gm.xl.ptr, xsl=gm.xsl.ptr, pend_lds=gm.pend.ptr,
                 put=mb.put, put_bf=mb.put_bf, put_words=mb.put_words, poll=mb.poll,
                 sym=sym, peer=peer_offsets(1,tp),
                 rank=rank_value, debug_out=fx.inttoptr(bptr,out),
                 debug_woa=fx.inttoptr(bptr,woa_out))
        if const_expr(not down):
            c["peer_addr"] = lambda p:bases[p]
        if const_expr(seam):
            # This binding's last argument is the immutable FFN_FRONT_ARGS
            # pointer table, rather than the isolated TP output address.
            arg_table = reduce_out
            args.update({key:vector_address(load_ptr64(arg_table,i))
                         for i,key in enumerate(FFN_INPUT_ARGS if ffn else FFN_FRONT_ARGS)})
            c.update(rl=sl.rl.ptr, fl=sl.fl.ptr, pend_lds=sl.pend.ptr,
                     threads=THREADS, fold=True, aux=False, index=False,
                     ffn=True, normed_cm=CM_DEV)
            reduce_out = args["reduce_out"]
        scratch = vector_address(scratch)
        for region in layout:
            c[region] = sreg(scratch, layout[region][0], region)
        for task in range(first_task(bid,0), d.head_tiles.count*8, BLOCKS):
            stage_attn_score_c1(c, task, 8)
        for task in range(first_task(bid,64), d.head_tiles.count*16, BLOCKS):
            stage_attn_irq_c1(c, task, 1, debug)
        gpu.barrier()
        task = first_task(bid,160)
        if task < d.woa_tasks:
            ops = woa_loads_c1(c, task)
            woa_tile_c1(c, task, ops, debug)
        if const_expr(down):
            # Keep the mother's single immutable-table read, but materialize
            # its bases before the first publishing stage. Keeping these
            # opaque VGPR addresses over score/PV spilled four registers in
            # the joint down binary. Every base is ready before WOB loads
            # and input waits, so publication never depends on a table load.
            bases = [scalar_address(base) for base in peer_bases(scalar_address(peers),tp)]
            c["peer_addr"] = lambda p:bases[p]
            c["rank"] = uniform(fresh(rank_value))
        wob_task = first_task(bid,160+d.woa_tasks)
        wob_second = wob_task+BLOCKS
        has_second = wob_second < HIDDEN//16
        ops = wob_loads_c1(c, wob_task)
        ops2 = wob_loads_c1(c, has_second.select(wob_second,wob_task))
        poll_copy(c, tid, THREADS, [
            (c["x8b"], lambda u:u, d.yb_words, c["xl"], d.yb_words),
            (c["x8bs"], lambda u:u, d.o_rows//32, c["xsl"], d.o_rows//32),
        ])
        gpu.barrier()
        stage_wob_c1(c, wob_task, ops)
        if has_second:
            stage_wob_c1(c, wob_second, ops2)
        if const_expr(reduce):
            for task in range(first_task(bid,0), HIDDEN//32, BLOCKS):
                if const_expr(seam):
                    pre = slice_prefetch_c1(c, task)
                stage_attn_reduce_c1(c, task)
                if const_expr(not seam or debug):
                    if tid < 32:
                        fx.ptr_store(fx.ptr_load(c["pend_lds"]+tid).to(fx.BFloat16),
                                     fx.inttoptr(bptr,reduce_out)+task*32+tid)
                gpu.barrier()
                if const_expr(seam):
                    stage_slice_c1(c, task, pre)
        if const_expr(seam):
            if const_expr(down):
                # Defer only scalar restoration; the epoch was captured once
                # at entry. Score's scalar live set otherwise spills four
                # SGPRs after adding down's rank and epoch consumers.
                fmb = reference_mailbox(uniform(fresh(ffn_layer))+128,
                                       diag,regions,failure)
            for task in range(first_task(bid,168), 1, BLOCKS):
                stage_norm_c1(dict(c, lane=tid%64, wave=tid//64))
                if const_expr(ffn):
                    publish_plain(fmb.put, c["xrdy"], 0, 1, tid==0)
                else:
                    publish_plain(c["put"], c["xrdy"], 0, 1, tid==0)
        if const_expr(ffn):
            ftid = fresh(fx.thread_idx.x)
            fscratch = scalar_address(scratch)
            cb = dict(S=1, tid=ftid, bid=bid, lane=ftid%32, wave=ftid//32,
                      d=d,
                      rs=rs,
                      args=dict(x=scalar_address(args["normed"]),
                                gate_w=scalar_address(args["gate_w"]),
                                bias=scalar_address(args["bias"])), red=shared_mem.red.ptr,
                      put=fmb.put, put_words=fmb.put_words, poll=fmb.poll, x_cm=CM_DEV, x_ready=True)
            for key in ("xrdy","x8rdy","x8r","x8rs","x8","x8s","logit","route"):
                cb[key] = sreg(fscratch,layout[key][0],key)
            run_ffn_input_c1(cb, bid, router)
            if const_expr(route):
                for token in range(first_task(bid,104),1,BLOCKS):
                    stage_route_c1(cb,token)
            if const_expr(shared):
                # Mother shared input is wider than the attention GEMV input.
                # Load the immutable pointers at this seam to bound liveness.
                cb["args"].update({key:scalar_address(load_ptr64(arg_table,i))
                                   for i,key in enumerate(FFN_SHARED_ARGS)
                                   if i >= len(FFN_INPUT_ARGS)})
                cb.update(xl=sf.xl.ptr, xsl=sf.xsl.ptr, pair=sf.pair.ptr)
                for key in ("smid","smq","smc"):
                    cb[key] = sreg(fscratch,layout[key][0],key)
                run_shared_c1(cb, bid)
            if const_expr(gate):
                # Original late caller: hand the shared LDS union back before
                # gate; PMIX belongs to K2a, so consume with mb, not fmb.
                gpu.barrier()
                for token in range(first_task(bid,160),1,BLOCKS):
                    gate_args = {key:scalar_address(load_ptr64(arg_table,i))
                                 for i,key in enumerate(FFN_GATE_ARGS)
                                 if i >= len(FFN_SHARED_ARGS)}
                    cg = dict(S=1, tid=ftid, lane=ftid%32, wave=ftid//32,
                              threads=THREADS, red=shared_mem.red.ptr, args=gate_args,
                              poll=mb.poll, pmix=sreg(fscratch,layout["pmix"][0],"pmix"))
                    stage_gate_c1(cg,token)
            if const_expr(ug):
                cb["args"].update({key:scalar_address(load_ptr64(arg_table,i))
                                   for i,key in enumerate(FFN_UG_ARGS)
                                   if i >= len(FFN_GATE_ARGS)})
                cb.update(xl=ul.xl.ptr,xsl=ul.xsl.ptr,tab=ul.tab.ptr,wsl=ul.wsl.ptr)
                for key in ("mid","mids","ugf"):
                    cb[key] = sreg(fscratch,layout[key][0],key)
                run_ug_c1(cb,bid)
            if const_expr(down):
                cb["args"].update({key:scalar_address(load_ptr64(arg_table,i))
                                   for i,key in enumerate(FFN_DOWN_ARGS)
                                   if i >= len(FFN_UG_ARGS)})
                cb.update(tab=dl.tab.ptr,wsl=dl.wsl.ptr,sxl=dl.sxl.ptr,sxsl=dl.sxsl.ptr,
                          contrib=dl.contrib.ptr,sdl=dl.sdl.ptr,pair=dl.pair.ptr,
                          put_bf=fmb.put_bf,tag=fmb.tag,tok_par=False,
                          sym=scalar_address(c["sym"]),peer=c["peer"],peer_addr=c["peer_addr"],rank=c["rank"])
                cb["dq"] = sreg(fscratch,layout["dq"][0],"dq")
                run_down_c1(cb,bid,ffn_reduce)
    @launcher(name)
    def launch(q:Int64, pool:Int64, sel:Int64, sbound:Int64, kmeta:Int64,
               sink:Int64, pos:Int64, cos:Int64, sin:Int64, scratch:Int64,
               diag:Int64, layer:Int64, config:Int64, out:Int64,
               woa:Int64, woa_s:Int64, woa_out:Int64, wob:Int64, wob_s:Int64,
               sym:Int64, peers:Int64, rank:Int64, reduce_out:Int64, stream:fx.Stream):
        kernel(q,pool,sel,sbound,kmeta,sink,pos,cos,sin,scratch,diag,layer,config,out,
               woa,woa_s,woa_out,wob,wob_s,sym,peers,rank,reduce_out).launch(
                   grid=(BLOCKS,), block=(THREADS,), stream=stream)
    return launch
