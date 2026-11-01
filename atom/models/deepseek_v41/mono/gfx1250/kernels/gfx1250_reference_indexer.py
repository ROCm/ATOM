# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mother c1 FP8 index task, selection-tree and tagged publication structure.

FP8 operands are represented exactly in BF16 for wave32 WMMA. The 16-key
logical tile pair stays with its original one-of-eight K1-independent waves.
All task key loads precede query and matrix work. WMMA's different C fragment
uses private LDS to restore the mother's grouped head-weight FMA reduction.
"""
from functools import lru_cache
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T
from aiter.ops.flydsl.kernels import buffer_ops as bo
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.gfx950.kernels.index_score import (
    TILE, TILE_BYTES, TASK_CHUNKS, TASK_CHUNKS_LOG2, NEG, FLT_MAX,
    _selected_rows, scratch_pairs,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_indexer import _fp8_bf16
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_attention import _wmma
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_topk import key_bits, lds_topk, lds_words
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import reference_mailbox, publish_plain
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_plan import BLOCKS
from atom.models.deepseek_v41.mono.step_meta import KMETA
from atom.mono.device.ops import CM_DEV, fresh, ld_dev, ld_i32, rsrc, traced, lane_gather, launcher, kernel_symbol
from atom.mono.device.sync import sreg
from atom.mono.plan.arith import pick, sel
from atom.mono.plan.execution import first_task
from atom.mono.plan.layout import pair_layout
from atom.mono.plan.trace import enter_stage

THREADS, PER_THREAD = 256, 32


def plane_ptr(plane, tile, byte):
    return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
                      plane + fx.Int64(tile)*TILE_BYTES + fx.Int64(byte))


def st_dev(value, ptr, index):
    # Materialize this descriptor at its use, rather than holding descriptors
    # for every tree output across the score and histogram stages.
    scalar = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(ptr).ir_value()))
    address = fx.Int64(llvm.inline_asm(T.i64, [scalar.ir_value()],
                                     "s_mov_b64 $0, $1", "=s,s", has_side_effects=True))
    bo.buffer_store(value,rsrc(address),index,cache_modifier=CM_DEV)


def vector_address(ptr):
    """Keep a wave-uniform address in two VGPRs until its descriptor is used."""
    low, high = fresh(fx.Int32(ptr)), fresh(fx.Int32(ptr >> 32))
    return (fx.Int64(low) & fx.Int64(0xffffffff)) | (fx.Int64(high) << 32)


def final_c1(c, kind, t, pos, ident, rows=None):
    a = c["args"]
    if const_expr(kind == "sel"):
        row, pool_row = rows
        st_dev(row, a["isel"], t * ip.TOPK + pos)
        st_dev(pool_row, a["irow"], t * ip.TOPK + pos)
    else:
        st_dev(Int32(ident), a["icout"], t * ip.TOPK_BLOCKS + pos)


@traced
def score_keys_c1(c, lead, ch0, width):
    a = c["args"]
    lane, wave = c["lane"], c["wave"]
    lane16, half = lane % 16, lane//16
    blocks = [(ch0+fx.min(width-1,m))*16+wave*2+lane16//8 for m in range(TASK_CHUNKS)]
    tiles = [fx.Int32(bo.buffer_load(rsrc(a["itab"]),lead*a["itab_stride"]+fx.min(b,a["itab_len"]-1),
                                    vec_width=1,dtype=T.i32)) for b in blocks]
    # Keep raw FP8 chunks in registers; convert one matrix operand at a time.
    return [(fx.ptr_load(plane_ptr(a["plane"],tile,1024+(lane16%8)*4)).bitcast(fx.Float32),
             [fx.Vector(fx.ptr_load(plane_ptr(a["plane"],tile,((kt*2+half)*8+lane16%8)*16),
                                    result_type=fx.Vector.make_type(4,fx.Int32))) for kt in range(4)])
            for tile in tiles]


@traced
def score_query_c1(c, tok):
    a, lane = c["args"], c["lane"]
    return [([fx.Vector(bo.buffer_load(rsrc(a["iq"]),
                     ((tok*32+h*16+lane%16)*128+kt*32+(lane//16)*16)//4,
                     vec_width=4,dtype=T.i32)) for kt in range(4)],
             fx.Float32(bo.buffer_load(rsrc(a["iw"]),tok*32+h*16+lane%16,
                                      vec_width=1,dtype=T.f32))) for h in range(2)]


@traced
def score_one_chunk_c1(c,tok,bound,ch,keys,query):
    tid,lane,wave = c["tid"],c["lane"],c["wave"]
    a,red = c["args"],c["score_red"]
    ks,key_words = keys
    # Query weights were loaded before the matrix work. The physical C layout
    # assigns two heads per lane; private LDS joins them for the original sum.
    for h in range_constexpr(2):
        words,weight = query[h]
        if (wave==0)&(lane<16):
            fx.ptr_store(weight,red+4096+h*16+lane)
        acc=fx.make_rmem_tensor(8,fx.Float32)
        acc.store(fx.full(8,0.,fx.Float32))
        for kt in range_constexpr(4):
            _wmma(acc,_fp8_bf16(words[kt]),_fp8_bf16(key_words[kt]))
        result=acc.load()
        for j in range_constexpr(8):
            key=lane//16*8+j
            scale=lane_gather(ks.bitcast(fx.Int32),key).bitcast(fx.Float32)
            fx.ptr_store(fx.max(result[j]*scale,fx.Float32(0)),
                         red+(wave*16+key)*32+h*16+lane%16)
    gpu.barrier()
    if tid<128:
        quarters=[]
        for group in range_constexpr(4):
            terms=[(fx.ptr_load(red+tid*32+h*16+group*4+e),
                    fx.ptr_load(red+4096+h*16+group*4+e))
                   for h in range(2) for e in range(4)]
            firsts=[fx.Float32(fmath.fma(terms[x][0],terms[x][1],terms[y][0]*terms[y][1]))
                    for x,y in ((0,2),(1,3),(4,6),(5,7))]
            quarters.append((firsts[0]+firsts[2])+(firsts[1]+firsts[3]))
        score=(quarters[0]+quarters[2])+(quarters[1]+quarters[3])
        col=ch*128+tid
        live=col<bound
        score=fx.Float32(live.select(score,fx.Float32(NEG)))
        st_dev(score,a["ilog"],tok*a["lstride"]+col)
        if a["iprod"]!=0:
            best=fx.Float32(fx.isnan(score).select(fx.Float32(NEG),score))
            for off in (1,2,4):
                best=fx.max(best,best.shuffle_xor(off,32))
            best=fx.min(best,fx.Float32(FLT_MAX))
            block=ch*16+tid//8
            count=ip.blocks(bound)
            best=fx.Float32((block==count-1).select(fx.Float32(float("inf")),best))
            if (tid%8==0)&(block<count):
                st_dev(best,a["ibmax"],tok*a["bstride"]+block)
    gpu.barrier()


@traced
def stage_iscore_c1(c,lead,span,ch0,width,max_chunks):
    tid,a=c["tid"],c["args"]
    row=lead+tid//TASK_CHUNKS
    m=tid%TASK_CHUNKS
    row_bound=ld_i32(a["ibound"],fx.min(row,c["S"]-1))
    keys=score_keys_c1(c,lead,ch0,width)
    for r in range(span):
        tok=lead+r
        bound=ld_i32(a["ibound"],tok)
        query=score_query_c1(c,tok)
        for k in range_constexpr(TASK_CHUNKS):
            if ip.task_writes(bound,ch0,width,k):
                score_one_chunk_c1(c,tok,bound,ch0+k,keys[k],query)
    who=(tid<span*TASK_CHUNKS)&ip.task_writes(row_bound,ch0,width,m)
    publish_plain(c["put"],c["isf"],row*max_chunks+ch0+m,1,who)



@traced
def stage_ilist_c1(c, t, j, raw, kind, level, bound_max, max_chunks):
    """List j of ``level`` for token t (``raw`` its column bound); see the
    module. Only the lists ``index_plan.list_count`` gives a token run."""
    tid = fresh(c["tid"])
    j, raw = fresh(j), fresh(raw)
    a = c["args"]
    k = ip.KINDS[kind]
    slots = ip.list_slots(kind, level, bound_max)
    bound = ip.kind_bound(kind, raw)
    if j < ip.list_count(bound, level, k):
        n = ip.list_inputs(bound, level, j, k)
        final = ip.is_final(bound, level, k)
        # the top level has one list, a token's last: it has no region above it
        to_final = True if level == ip.top_level(kind, bound_max) else final
        got = []
        if const_expr(level == 0):
            if const_expr(kind == "sel"):
                first, count = ip.CHUNKS_PER_SEG * j, ip.seg_chunks(bound, j)
            else:
                first, count = ip.block_chunks(raw, j)
            if tid < count:
                c["poll"]([(c["isf"], t * max_chunks + first + tid, 1)])
            gpu.barrier()
        else:
            below = ip.list_slots(kind, level - 1, bound_max)
            base = tid * PER_THREAD
            got = c["poll"](
                [
                    (
                        c[f"i{kind}{level - 1}"],
                        (
                            (t * below + j * ip.fan(k) + fx.min(base + d, n - 1) // k)
                            * k
                            + fx.min(base + d, n - 1) % k
                        )
                        * 2,
                        2,
                    )
                    for d in range(PER_THREAD)
                ]
            )
        src, stride = (
            (a["ilog"], a["lstride"]) if kind == "sel" else (a["ibmax"], a["bstride"])
        )

        def load(q, i):
            if const_expr(level == 0):
                # a quad stays in the row: strides are whole chunks / blocks
                v = fx.Vector(ld_dev(src, t * stride + j * ip.SEG + i, 4))
                return [fx.Int32(v[e]) for e in range(4)]
            return [got[4 * q + e][0] for e in range(4)]

        keys = c["sel_lds"]["keys"]
        first_i = tid * PER_THREAD

        def ident_of(d, i):
            return j * ip.SEG + i if level == 0 else got[d][1]

        # a selection's rows (``_selected_rows``), in flight during the passes
        def selected_rows():
            picked = [
                _selected_rows(c, t, ident_of(d, fx.min(first_i + d, n - 1)))
                for d in range(PER_THREAD)
            ]
            return fx.Vector.from_elements([word for pair in picked for word in pair],fx.Int32)

        def emit(pos, d, i, rows):
            if const_expr(d is None):
                ident, bits = fx.Int32(-1), fx.Float32(NEG).bitcast(fx.Int32)
                picked = (fx.Int32(-1), fx.Int32(0))
            else:
                ident = ident_of(d, i)
                bits = key_bits(fx.ptr_load(keys + i))
                picked = (rows[2*d],rows[2*d+1]) if kind == "sel" else None
            if to_final:
                final_c1(c, kind, t, pos, ident, picked)
            else:
                c["put_words"](
                    c[f"i{kind}{level}"], ((t * slots + j) * k + pos) * 2, [bits, ident]
                )

        lds_topk(
            n, k, load, emit, c["sel_lds"], tid, PER_THREAD,
            selected_rows if kind == "sel" else None,
        )  # fmt: skip
        if const_expr(kind == "sel"):  # noqa: SIM102
            if final:
                publish_plain(c["put"], c["irdy"], t, 1, tid == 0)

@traced
def run_scores_c1(c, s, bid, bounds, max_chunks):
    """The score stage's tasks (``index_plan.score_tasks``). A request's tokens
    share its tile table (key metadata word 2: its block table offset); a
    REINDEX layer's are each token's own candidates."""
    assert s * TASK_CHUNKS <= THREADS, s  # a thread a (token, chunk) flag
    a = c["args"]
    reindex = a["ilift"] != 0
    owners = [sel(reindex, t, ld_i32(a["kmeta"], t * KMETA + 2)) for t in range(s)]
    counts, spans, width_log2 = ip.score_tasks(bounds, owners, BLOCKS, TASK_CHUNKS_LOG2)
    width = Int32(1) << width_log2
    enter_stage("k2a.iscore")
    for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
        lead, i = ip.flat_task(counts, task)
        stage_iscore_c1(c, lead, pick(spans, lead), i << width_log2, width, max_chunks)

@traced
def run_tree_c1(c, s, bid, kind, bounds, bound_max, max_chunks):
    k = ip.KINDS[kind]
    for level in range_constexpr(ip.top_level(kind, bound_max) + 1):
        counts = [ip.list_count(ip.kind_bound(kind, b), level, k) for b in bounds]
        enter_stage(f"k2a.i{kind}{level}")
        for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
            t, j = ip.flat_task(counts, task)
            stage_ilist_c1(c, t, j, pick(bounds, t), kind, level, bound_max, max_chunks)

@traced
def run_indexer_c1(c,bid,bound_max):
    a=c["args"]
    bound=ld_i32(a["ibound"],0)
    bounds=[bound]
    chunks=ip.chunks(bound_max)
    run_scores_c1(c,1,bid,bounds,chunks)
    run_tree_c1(c,1,bid,"sel",bounds,bound_max,chunks)
    if a["iprod"]!=0:
        run_tree_c1(c,1,bid,"blk",bounds,bound_max,chunks)


def indexer_layout(bound_max):
    return pair_layout(scratch_pairs(1,bound_max).items())


def indexer_symbol(bound_max, scope="all"):
    return kernel_symbol("gfx1250_reference_indexer",n=bound_max,b=BLOCKS,t=THREADS,scope=scope)


@lru_cache(None)
def build_reference_indexer_c1(bound_max, scope="all"):
    """Standalone correctness binding of original K2a index task/epoch schema."""
    if not 0<bound_max<=65536 or bound_max%128:
        raise ValueError("c1 reference FP8 index bound must be a multiple of128 up to65536")
    if scope not in ("all", "score", "select"):
        raise ValueError("index compile isolation scope must be all/score/select")
    layout=indexer_layout(bound_max)
    sizes=lds_words(PER_THREAD)
    name=indexer_symbol(bound_max,scope)
    @fx.struct
    class Smem:
        keys:fx.Array[fx.Int32,sizes["keys"],16]
        hist:fx.Array[fx.Int32,sizes["hist"],16]
        buf:fx.Array[fx.Int32,sizes["buf"],16]
    @flyc.kernel(name=name,known_block_size=[THREADS,1,1])
    def kernel(plane:Int64,itab:Int64,iq:Int64,iw:Int64,ibound:Int64,kmeta:Int64,
               icand:Int64,table:Int64,ilog:Int64,ibmax:Int64,isel:Int64,irow:Int64,
               icout:Int64,scratch:Int64,diag:Int64,layer:Int64,config:Int64):
        tid,bid=fx.thread_idx.x,fx.block_idx.x
        lds=fx.SharedAllocator().allocate(Smem).peek()
        pointer=fx.PointerType.get(fx.Int32.ir_type,fx.AddressSpace.Global,4)
        lp=fx.inttoptr(pointer,layer)
        cp=fx.inttoptr(pointer,config)
        cfg=[fx.ptr_load(cp+i) for i in range(7)]
        failure=fx.inttoptr(pointer,diag+fx.Int64(BLOCKS*8*4))
        # Mother K2a epoch65+layer; local K1 epoch is never reused here.
        mb=reference_mailbox(fx.ptr_load(lp)+64,diag,
                             {name:i+1 for i,name in enumerate(layout)},failure)
        args=dict(plane=plane,itab=itab,iq=iq,iw=iw,ibound=ibound,kmeta=kmeta,
                  icand=vector_address(icand),table=vector_address(table),ilog=vector_address(ilog),ibmax=vector_address(ibmax),isel=vector_address(isel),
                  irow=vector_address(irow),icout=vector_address(icout),itab_stride=bound_max//8,itab_len=cfg[3],
                  lstride=bound_max,bstride=bound_max//8,iprod=cfg[1],ilift=cfg[2],
                  rows_per_page=fresh(cfg[4]),page_rows=cfg[5],main_off=cfg[6])
        c=dict(S=1,tid=tid,bid=bid,lane=tid%32,wave=tid//32,args=args,
               put=mb.put,put_words=mb.put_words,poll=mb.poll,
               sel_lds=dict(keys=lds.keys.ptr,hist=lds.hist.ptr,buf=lds.buf.ptr),
               score_red=fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type,fx.AddressSpace.Shared,4),lds.keys.ptr))
        scratch = vector_address(scratch)
        for region in layout:
            c[region]=sreg(scratch,layout[region][0],region)
        if cfg[0]!=0:
            if const_expr(scope=="all"):
                run_indexer_c1(c,bid,bound_max)
            elif const_expr(scope=="score"):
                run_scores_c1(c,1,bid,[ld_i32(ibound,0)],ip.chunks(bound_max))
            else:
                run_tree_c1(c,1,bid,"sel",[ld_i32(ibound,0)],bound_max,ip.chunks(bound_max))
    @launcher(name)
    def launch(plane:Int64,itab:Int64,iq:Int64,iw:Int64,ibound:Int64,kmeta:Int64,
               icand:Int64,table:Int64,ilog:Int64,ibmax:Int64,isel:Int64,irow:Int64,
               icout:Int64,scratch:Int64,diag:Int64,layer:Int64,config:Int64,stream:fx.Stream):
        kernel(plane,itab,iq,iw,ibound,kmeta,icand,table,ilog,ibmax,isel,irow,icout,
               scratch,diag,layer,config).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
