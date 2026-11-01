# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""External TP4 DSpark FP8 index stages, not an enabled K2 implementation."""
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
    _selected_rows as image_selected_rows,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_topk import key_bits, lds_topk, lds_words
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import publish_plain, uniform_stage_context
from atom.models.deepseek_v41.mono.gfx950.kernels.layer_post import LayerPostBuild, scratch_layout, ABI
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_post import attn_post_args
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.kernels.token_par import own_rows
from atom.models.deepseek_v41.mono.step_meta import KMETA
from atom.mono.device.ops import CM_DEV, CM_SYS_GFX1250, fresh, uniform, ld_dev, ld_i32, rsrc, traced, lane_gather, launcher, kernel_symbol
from atom.mono.device.ranks import peer_bases
from atom.mono.device.sync import Mailbox, POLL_MAX, preg, sreg
from atom.mono.plan.arith import pick, sel
from atom.mono.plan.execution import BLOCKS, first_task
from atom.mono.plan.trace import enter_stage
THREADS, PER_THREAD = 256, 32
LATE_PTRS=('icand','table','ilog','ibmax','isel','irow','icout','kmeta')
LATE_WORDS=('rows_per_page','page_rows','main_off','ilift','iprod','lstride','bstride')

def late_pointer(c,offset):
    low=scalar_word(fx.ptr_load(c['late_meta']+offset))
    high=scalar_word(fx.ptr_load(c['late_meta']+offset+1))
    return (fx.Int64(low)&fx.Int64(0xffffffff))|(fx.Int64(high)<<32)

def late_word(c,name):
    return scalar_word(fx.ptr_load(c['late_meta']+2*len(LATE_PTRS)+LATE_WORDS.index(name)))

def restore_late(c, pointers=LATE_PTRS, words=LATE_WORDS, scratch=True):
    out=dict(c)
    out['args']=dict(c['args'])
    for name in pointers:
        offset=2*LATE_PTRS.index(name)
        out['args'][name]=late_pointer(c,offset)
    for name in words:
        out['args'][name]=late_word(c,name)
    if scratch:
        base=late_pointer(c,24)
        for region,(off,_) in c['scratch_layout'].items():out[region]=sreg(base,off,region)
    return out

def mhtjstest_index_row_offset(c, part):
    # Preserve every accepted <=24-row address. Wide index arrays live after
    # all 272 fixed metadata words used by attention, WO, shared and down.
    return 272 + part*c["S"] if c["S"] > 24 else (32,64,88)[part]

def mhtjstest_index_row_value(c, part, row):
    value=fx.ptr_load(c['late_meta']+mhtjstest_index_row_offset(c,part)+row)
    return scalar_word(value)

def restored_bounds(c,n):
    return [mhtjstest_index_row_value(c,0,t) for t in range(n)]


@traced
def mhtjstest_plan_wide_scores(c, t0, n):
    """Original score_tasks plan, streamed through LDS with bounded live state.

    Lane 0 forms the original owner runs in reverse, then applies the same
    together/width/count rules. Row arrays do not stay live in registers.
    """
    reindex = c['args']['ilift'] != 0
    run, widest = fx.Int32(0), fx.Int32(0)
    shared_total, own_total = fx.Int32(0), fx.Int32(0)
    next_owner = fx.Int32(0)
    for reverse in range(fx.Int32(n)):
        row = n - 1 - reverse
        bound = fx.Int32(fx.ptr_load(c['late_meta']+mhtjstest_index_row_offset(c,0)+row))
        own = ip.score_chunks(bound)
        owner = sel(reindex,t0+row,ld_i32(c['args']['kmeta'],(t0+row)*KMETA+2))
        same_next = (reverse != 0) & (owner == next_owner)
        run = sel(same_next,run+1,fx.Int32(1))
        widest = sel(same_next,fx.max(own,widest),own)
        previous_row = fx.max(row-1,0)
        previous_owner = sel(reindex,t0+previous_row,
                             ld_i32(c['args']['kmeta'],(t0+previous_row)*KMETA+2))
        shared = sel((row == 0) | (owner != previous_owner),widest,fx.Int32(0))
        fx.ptr_store(shared,c['late_meta']+mhtjstest_index_row_offset(c,1)+row)
        fx.ptr_store(run,c['late_meta']+mhtjstest_index_row_offset(c,2)+row)
        shared_total = shared_total+shared
        own_total = own_total+own
        next_owner = owner
    together = shared_total >= BLOCKS
    need_total = sel(together,shared_total,own_total)
    width_log2 = fx.Int32(0)
    for group in range_constexpr(1,TASK_CHUNKS_LOG2+1):
        width_log2 = sel((need_total >> group) >= BLOCKS,group,width_log2)
    total = fx.Int32(0)
    for row in range(fx.Int32(n)):
        bound = fx.Int32(fx.ptr_load(c['late_meta']+mhtjstest_index_row_offset(c,0)+row))
        shared = fx.Int32(fx.ptr_load(c['late_meta']+mhtjstest_index_row_offset(c,1)+row))
        run = fx.Int32(fx.ptr_load(c['late_meta']+mhtjstest_index_row_offset(c,2)+row))
        need = sel(together,shared,ip.score_chunks(bound))
        count = (need+(fx.Int32(1)<<width_log2)-1)>>width_log2
        fx.ptr_store(count,c['late_meta']+mhtjstest_index_row_offset(c,1)+row)
        fx.ptr_store(sel(together,run,1),c['late_meta']+mhtjstest_index_row_offset(c,2)+row)
        total = total+count
    fx.ptr_store(total,c['late_meta']+272+3*c['S'])
    fx.ptr_store(width_log2,c['late_meta']+273+3*c['S'])


@traced
def mhtjstest_wide_flat_task(c,n,task):
    row,start,running = task*0,task*0,fx.Int32(0)
    for index in range(fx.Int32(n-1)):
        running = running+mhtjstest_index_row_value(c,1,index)
        past = task >= running
        row = sel(past,index+1,row)
        start = sel(past,running,start)
    return row,task-start


@traced
def mhtjstest_plan_wide_tree(c,n,kind,level,k):
    total = fx.Int32(0)
    for row in range(fx.Int32(n)):
        bound = fx.Int32(fx.ptr_load(c['late_meta']+mhtjstest_index_row_offset(c,0)+row))
        count = ip.list_count(ip.kind_bound(kind,bound),level,k)
        fx.ptr_store(count,c['late_meta']+mhtjstest_index_row_offset(c,1)+row)
        total = total+count
    fx.ptr_store(total,c['late_meta']+272+3*c['S'])

def scalar_word(value):
    lane=rocdl.readfirstlane(T.i32,fx.Int32(value).ir_value())
    return fx.Int32(llvm.inline_asm(T.i32,[lane],"s_mov_b32 $0, $1","=s,s",has_side_effects=True))
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


@traced
def score_keys(c, lead, ch0, width):
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
def score_query(c, tok):
    a, lane = c["args"], c["lane"]
    return [([fx.Vector(bo.buffer_load(rsrc(a["iq"]),
                     ((tok*32+h*16+lane%16)*128+kt*32+(lane//16)*16)//4,
                     vec_width=4,dtype=T.i32)) for kt in range(4)],
             fx.Float32(bo.buffer_load(rsrc(a["iw"]),tok*32+h*16+lane%16,
                                      vec_width=1,dtype=T.f32))) for h in range(2)]


def mhtjstest_fp8_wmma(accum, query_chunks, key_chunks):
    """Same K order and C layout as four BF16 K32 ops, using raw FP8 K128.

    Each lane owns four 16-byte pieces at k = 32*chunk + 16*half.
    Keep keys as A and queries as B, matching the existing _wmma wrapper:
    output heads remain lane%16 and keys remain half*8 + accumulator index.
    """
    atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 128, fx.Float8E4M3FN, fx.Float32))
    def operand(chunks):
        lo = chunks[0].shuffle(chunks[1], list(range(8)))
        hi = chunks[2].shuffle(chunks[3], list(range(8)))
        packed = lo.shuffle(hi, list(range(16)))
        fragment = fx.make_rmem_tensor(16, fx.Int32)
        fragment.store(packed)
        return fragment
    fx.gemm(atom, accum, operand(key_chunks), operand(query_chunks), accum)


@traced
def score_one_chunk(c,tok,bound,ch,keys,query):
    """A chunk's 128 keys, wave-local: in the WMMA C layout a key's 16 heads
    (of each head half) are one half-wave's 16 lanes, so the weighted head
    sum is four xor shuffles and a block's 8 keys are that half-wave's -- no
    LDS round trip, no barrier (the mother's layout parked 4096 products in
    LDS and summed them on 128 threads between two CTA barriers a chunk).
    Products, NEG/inf/NaN rules and the stores are the original's; only the
    order of the 32-head sum changes (logits keep their tolerance)."""
    lane,wave = c["lane"],c["wave"]
    a = c["args"]
    ks,key_words = keys
    totals=[fx.Float32(0.0) for _ in range(8)]
    for h in range_constexpr(2):
        words,weight = query[h]
        acc=fx.make_rmem_tensor(8,fx.Float32)
        acc.store(fx.full(8,0.,fx.Float32))
        mhtjstest_fp8_wmma(acc, words, key_words)
        result=acc.load()
        for j in range_constexpr(8):
            key=lane//16*8+j
            scale=lane_gather(ks.bitcast(fx.Int32),key).bitcast(fx.Float32)
            totals[j]=totals[j]+fx.max(result[j]*scale,fx.Float32(0))*weight
    for j in range_constexpr(8):
        for off in (1,2,4,8):
            totals[j]=totals[j]+totals[j].shuffle_xor(off,32)
    l16=lane%16
    half_keys=ch*128+wave*16+lane//16*8
    best=fx.Float32(NEG)
    mine=fx.Float32(NEG)
    for j in range_constexpr(8):
        live=(half_keys+j)<bound
        score=fx.Float32(live.select(totals[j],fx.Float32(NEG)))
        mine=fx.Float32((l16==j).select(score,mine))
        best=fx.max(best,fx.Float32(fx.isnan(score).select(fx.Float32(NEG),score)))
    if l16<8:
        st_dev(mine,a["ilog"],tok*a["lstride"]+half_keys+l16)
    if a["iprod"]!=0:
        best=fx.min(best,fx.Float32(FLT_MAX))
        block=ch*16+wave*2+lane//16
        count=ip.blocks(bound)
        best=fx.Float32((block==count-1).select(fx.Float32(float("inf")),best))
        if (l16==0)&(block<count):
            st_dev(best,a["ibmax"],tok*a["bstride"]+block)


@traced
def stage_iscore(c,lead,span,ch0,width,max_chunks):
    c=restore_late(c,pointers=('ilog','ibmax'),words=('iprod','lstride','bstride'))
    tid,a=c["tid"],c["args"]
    row=lead+tid//TASK_CHUNKS
    m=tid%TASK_CHUNKS
    row_bound=ld_i32(a["ibound"],fx.min(row,c["S"]-1))
    keys=score_keys(c,lead,ch0,width)
    for r in range(span):
        tok=lead+r
        bound=ld_i32(a["ibound"],tok)
        query=score_query(c,tok)
        for k in range_constexpr(TASK_CHUNKS):
            if ip.task_writes(bound,ch0,width,k):
                score_one_chunk(c,tok,bound,ch0+k,keys[k],query)
    # every wave's logits of the task stored before its flags
    gpu.barrier()
    if const_expr(c["S"] <= THREADS // TASK_CHUNKS):
        who=(tid<span*TASK_CHUNKS)&ip.task_writes(row_bound,ch0,width,m)
        publish_plain(c["put"],c["isf"],row*max_chunks+ch0+m,1,who)
    else:
        # Each logical (row, chunk) publisher still owns exactly one flag.
        # gfx1250 has 256 threads; a 48-row run requires two publisher rounds.
        for mhtjstest_round in range_constexpr((c["S"]*TASK_CHUNKS+THREADS-1)//THREADS):
            mhtjstest_tid=tid+mhtjstest_round*THREADS
            mhtjstest_row=lead+mhtjstest_tid//TASK_CHUNKS
            mhtjstest_m=mhtjstest_tid%TASK_CHUNKS
            mhtjstest_bound=ld_i32(a["ibound"],fx.min(mhtjstest_row,c["S"]-1))
            who=(mhtjstest_tid<span*TASK_CHUNKS)&ip.task_writes(mhtjstest_bound,ch0,width,mhtjstest_m)
            publish_plain(c["put"],c["isf"],mhtjstest_row*max_chunks+ch0+mhtjstest_m,1,who)


@traced
def stage_ilist(c, t, j, raw, kind, level, bound_max, max_chunks):
    """List j of ``level`` for token t (``raw`` its column bound); see the
    module. Only the lists ``index_plan.list_count`` gives a token run."""
    c=restore_late(c,pointers=('ilog',) if kind=='sel' else ('ibmax',),
                   words=('lstride',) if kind=='sel' else ('bstride',))
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
            # Restore uniform bases once for the original prefetch batch.
            # Do not keep one opaque scalar copy per selected element.
            c_sel=restore_late(c,pointers=('icand','table','kmeta'),
                               words=('rows_per_page','page_rows','main_off','ilift'),scratch=False)
            picked = [
                image_selected_rows(c_sel, t, ident_of(d, fx.min(first_i + d, n - 1)))
                for d in range(PER_THREAD)
            ]
            return fx.Vector.from_elements([word for pair in picked for word in pair],fx.Int32)

        # A final selection's winners: their ids by output place in LDS (the
        # radix histograms, free once the passes end), then two places a
        # thread map to rows and store. The mother prefetched every
        # candidate's rows before the passes (per_thread table lookups and
        # divisions a thread, 32 at 256 threads) for k winners of n.
        win = c["sel_lds"]["hist"]

        def emit(pos, d, i, rows):
            if const_expr(d is None):
                ident, bits = fx.Int32(-1), fx.Float32(NEG).bitcast(fx.Int32)
                picked = (fx.Int32(-1), fx.Int32(0))
            else:
                ident = ident_of(d, i)
                bits = key_bits(fx.ptr_load(keys + i))
                picked = None
            if to_final:
                if const_expr(kind == "sel"):
                    fx.ptr_store(ident, win + pos)
                else:
                    _final(c, kind, t, pos, ident, picked)
            else:
                c["put_words"](
                    c[f"i{kind}{level}"], ((t * slots + j) * k + pos) * 2, [bits, ident]
                )

        lds_topk(n, k, load, emit, c["sel_lds"], tid, PER_THREAD, None)
        if const_expr(kind == "sel"):
            if to_final:
                c_sel=restore_late(c,pointers=('icand','table','kmeta'),
                                   words=('rows_per_page','page_rows','main_off','ilift'),scratch=False)
                for r in range_constexpr(k // THREADS):
                    pos = fresh(c["tid"]) + THREADS * r
                    ident = fx.ptr_load(win + pos)
                    row, pool_row = image_selected_rows(c_sel, t, ident)
                    valid = ident >= 0
                    _final(c, kind, t, pos, ident,
                           (valid.select(row, fx.Int32(-1)), valid.select(pool_row, fx.Int32(0))))
            gpu.barrier()
        region = "irdy" if kind == "sel" else "icrdy" if c["tok_par"] else None
        if const_expr(region is not None):
            if final:
                _publish_all(c, region, t)


def _final(c, kind, t, pos, ident, rows=None):
    c=restore_late(c,pointers=('isel','irow') if kind=='sel' else ('icout',),words=(),scratch=False)
    if const_expr(kind == "sel"):
        row, pool_row = rows
        # the attention reads the pool rows; the selection only check mode
        # compares, which never runs token parallel: this rank's alone
        st_dev(row, c["args"]["isel"], t * ip.TOPK + pos)
        _st_out(c, pool_row, "irow", t * ip.TOPK + pos)
    else:
        _st_out(c, Int32(ident), "icout", t * ip.TOPK_BLOCKS + pos)


def _st_out(c, v, name, i):
    """Word i of output ``name``: this rank's, or in a token-parallel launch,
    which selects only its own rows (``_own_rows``), every rank's copy. The
    outputs live in the peer buffer's plain part (``peer_layout.plain_layout``),
    at one offset from every rank's base."""
    a = c["args"]
    if const_expr(c["tok_par"]):
        off = a[name] - late_pointer(c,120)
        for p in range_constexpr(c["d"].tp):
            bo.buffer_store(v, rsrc(c["peer_addr"](p) + off), i, cache_modifier=CM_SYS_GFX1250)
    else:
        st_dev(v, a[name], i)


def _publish_all(c, region, t):
    """Token t's flag in ``region`` after this CTA's outputs: this rank's
    scratch, or in a token-parallel launch every rank's peer region."""
    if const_expr(c["tok_par"]):
        flags = [
            preg(c["peer_addr"](p), c["peer"][region], region) for p in range(c["d"].tp)
        ]

        def put_each(dsts, i, v):
            for dst in dsts:
                c["put"](dst, i, v)

        publish_plain(put_each, flags, t, 1, fresh(c["tid"]) == 0)
    else:
        publish_plain(c["put"], c[region], t, 1, fresh(c["tid"]) == 0)


@traced
def await_candidates(c, s, bid):
    """A token-parallel launch of the candidate-producing layer ends only once
    every rank's rows' candidate blocks are in this rank's copy: the host's
    candidate table reads them after it (one CTA waits, at the launch's end)."""
    c=restore_late(c,pointers=(),words=('iprod',),scratch=False)
    enter_stage("k2a.icwait")
    if (c["args"]["iact"] != 0) & (c["args"]["iprod"] != 0) & (bid == 0):
        flags = preg(late_pointer(c,120), c["peer"]["icrdy"], "icrdy")
        for t0 in range_constexpr(0, s, POLL_MAX):
            c["poll"]([(flags, t, 1) for t in range(t0, min(t0 + POLL_MAX, s))])


def _own_rows(c, s):
    """(first row, rows) this rank scores and selects: every row, or in a
    token-parallel launch its own (``token_par.own_rows``) -- a row's
    selection depends on that row alone, and ``_st_out`` hands it to every rank."""
    if const_expr(c["tok_par"]):
        return own_rows(s, c["d"].tp, c["rank"])
    return 0, s


@traced
def run_indexer(c, s, bid, bound_max):
    """Every indexer stage of a selecting layer, the tasks of each laid out from
    the step's bounds (``index_plan``): only live tasks. The bounds were written
    before this launch: plain loads."""
    a = c["args"]
    max_chunks = ip.chunks(bound_max)
    t0, n = _own_rows(c, s)
    if fresh(c['tid'])==0:
        for t in range(n):
            fx.ptr_store(ld_i32(a['ibound'],t0+t),c['late_meta']+mhtjstest_index_row_offset(c,0)+t)
    gpu.barrier()
    if const_expr(c["index_fp4"]):
        run_scores4(c, t0, n, bid, max_chunks)
    else:
        run_scores(c, t0, n, bid, None if s > 24 else restored_bounds(c,n), max_chunks)
    _run_tree(c, t0, n, bid, "sel", None, bound_max, max_chunks)
    if late_word(c,'iprod') != 0:
        _run_tree(c, t0, n, bid, "blk", None, bound_max, max_chunks)


@traced
def run_scores(c, t0, n, bid, bounds, max_chunks):
    """The score stage's tasks (``index_plan.score_tasks``) over rows t0 ..
    t0 + n. A request's tokens share its tile table (key metadata word 2: its
    block table offset); a REINDEX layer's are each token's own candidates."""
    assert 0 < n <= c["S"] <= 48  # stage_iscore publishes all logical flags in thread-sized rounds
    c=restore_late(c,pointers=('kmeta',),words=('ilift',),scratch=False)
    a = c["args"]
    if const_expr(c['S'] > 24):
        if fresh(c['tid'])==0:
            mhtjstest_plan_wide_scores(c,t0,n)
        gpu.barrier()
        total=scalar_word(fx.ptr_load(c['late_meta']+272+3*c['S']))
        width_log2=scalar_word(fx.ptr_load(c['late_meta']+273+3*c['S']))
    else:
        reindex = a["ilift"] != 0
        owners = [
            sel(reindex, t0 + t, ld_i32(a["kmeta"], (t0 + t) * KMETA + 2)) for t in range(n)
        ]
        counts, spans, width_log2 = ip.score_tasks(bounds, owners, BLOCKS, TASK_CHUNKS_LOG2)
        total=ip.flat_total(counts)
        if fresh(c['tid'])==0:
            for i in range_constexpr(n):
                fx.ptr_store(fx.Int32(counts[i]),c['late_meta']+mhtjstest_index_row_offset(c,1)+i)
                fx.ptr_store(fx.Int32(spans[i]),c['late_meta']+mhtjstest_index_row_offset(c,2)+i)
        gpu.barrier()
    width = Int32(1) << width_log2
    enter_stage("k2a.iscore")
    for task in range(first_task(bid, 0), total, BLOCKS):
        if const_expr(c['S'] > 24):
            lead,i=mhtjstest_wide_flat_task(c,n,task)
        else:
            current_counts=[mhtjstest_index_row_value(c,1,t) for t in range(n)]
            lead, i = ip.flat_task(current_counts, task)
        stage_iscore(
            c, t0 + lead, scalar_word(fx.ptr_load(c['late_meta']+fresh(mhtjstest_index_row_offset(c,2))+lead)), i << width_log2, width, max_chunks
        )


@traced
def _run_tree(c, t0, n, bid, kind, bounds, bound_max, max_chunks):
    k = ip.KINDS[kind]
    for level in range_constexpr(ip.top_level(kind, bound_max) + 1):
        if const_expr(c['S'] > 24):
            if fresh(c['tid'])==0:
                mhtjstest_plan_wide_tree(c,n,kind,level,k)
            gpu.barrier()
            total=scalar_word(fx.ptr_load(c['late_meta']+272+3*c['S']))
        else:
            bounds=restored_bounds(c,n)
            counts = [ip.list_count(ip.kind_bound(kind, b), level, k) for b in bounds]
            total=ip.flat_total(counts)
            if fresh(c['tid'])==0:
                for i in range_constexpr(n):
                    fx.ptr_store(fx.Int32(counts[i]),c['late_meta']+mhtjstest_index_row_offset(c,1)+i)
            gpu.barrier()
        enter_stage(f"k2a.i{kind}{level}")
        # the candidate-block tree from the CTAs half a grid past the row
        # tree's: both wait only on the score flags, so they run side by side
        # (from CTA 0 both, a row's CTA ran its blk list after its sel list)
        for task in range(first_task(bid, 0 if kind == "sel" else BLOCKS // 2), total, BLOCKS):
            if const_expr(c['S'] > 24):
                t,j=mhtjstest_wide_flat_task(c,n,task)
            else:
                current_counts=[mhtjstest_index_row_value(c,1,i) for i in range(n)]
                t, j = ip.flat_task(current_counts, task)
            stage_ilist(
                c, t0 + t, j, scalar_word(fx.ptr_load(c['late_meta']+fresh(mhtjstest_index_row_offset(c,0))+t)), kind, level, bound_max, max_chunks
            )


from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_attention import (
    stage_attn_score, stage_attn_irq, BK, COLS, HEAD_TILE, SCORE_LDS_WORDS,
)
from atom.models.deepseek_v41.mono.gfx950.kernels.attention import score_tasks, pv_groups, GROUPS
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_post import SCORE0, IRQ0
ATT_PTRS=('q','pool','sel','sbound','kmeta','sink','pos','cos','sin')
ATT_WORDS=('ring_off','ring_slots','qk_scale','iact','topk')

def attention_context(c,stage):
    out=dict(c)
    out['tid']=fresh(fx.thread_idx.x)
    out['lane']=out['tid']%32
    out['wave']=out['tid']//32
    names=('q','pool','sel','sbound','kmeta') if stage=='score' else ('pool','sel','sbound','kmeta','sink','pos','cos','sin')
    out['args']={name:late_pointer(c,128+2*ATT_PTRS.index(name)) for name in names}
    for i,name in enumerate(ATT_WORDS):
        out['args'][name]=scalar_word(fx.ptr_load(c['late_meta']+146+i))
    scratch=late_pointer(c,24)
    for region,(off,_) in c['scratch_layout'].items():out[region]=sreg(scratch,off,region)
    if const_expr(c['tok_par'] and c['index_bound_max']>0):
        out['irdy']=preg(late_pointer(c,120),c['peer']['irdy'],'irdy')
    return out
@lru_cache(None)
def build_dspark_attention(key: LayerPostBuild, debug=False):
    assert key.tp==4 and key.ep==key.moe_tp==1
    assert key.tokens in (5,6,10,12,15,18,20,24,25,30,35,36,40,42,48) and not key.index_fp4 and not key.kv_fp8
    assert not key.timeline and key.diag_off==-1
    assert key.index_bound_max in (0,1024,8192,1048576)
    s=key.tokens
    dims=Dims(4)
    layout=scratch_layout(key)
    sizes=lds_words(PER_THREAD)
    name=kernel_symbol('gfx1250_dspark_attention',s=s,n=key.index_bound_max,tpar=int(key.tok_par),db=int(debug))
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
    @fx.union
    class Stages:
        index:IndexSmem
        score:ScoreSmem
        pv:PvSmem
    @fx.struct
    class Smem:
        late_meta:fx.Array[fx.Int32,272 + 3*s+2 if s > 24 else 160,16]
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
        args_a=attn_post_args(iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog, lstride, isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound, q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out, pre_out, normed)
        bases=peer_bases(peers,4) if const_expr(key.tok_par) else []
        # Explicit metadata-only storage replaces compiler-generated scratch.
        # No weight/input prefetch, task, score or selection value is moved.
        if fresh(tid)==0:
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
        gpu.barrier()
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
        if const_expr(key.tok_par and key.index_bound_max>0):await_candidates(c,s,bid)
    @launcher(name)
    def launch(iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64, pscale: Int64, iplan: Int64, itab: Int64, itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64, lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32, ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, sbound: Int64, q: Int64, pool: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64, rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32, ring_slots: Int32, sink: Int64, qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64, wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64, res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64, normed: Int64, gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64, w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32, layer: Int32, moe_layer: Int32, tl: Int64, stream:fx.Stream):
        index(iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog, lstride, isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound, q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off, ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob, wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out, pre_out, normed, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out, scratch, sym, peers, rank, layer, moe_layer, tl).launch(grid=(BLOCKS,),block=(THREADS,),stream=stream)
    return launch
