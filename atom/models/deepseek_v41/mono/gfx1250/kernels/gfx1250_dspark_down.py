# SPDX-License-Identifier: MIT
"""Mother TP4 down, with wave32 WMMA register operands and packed contributions."""
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import *
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import _down_ug_flag
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_ug import ug_context
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_union import load_route
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_index import late_pointer,scalar_word
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_moe import fp4_weight_offset
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_k1 import gemv_fp8_loads
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_multirow_projection import reference_fp8_matrices_rows
from atom.mono.device.ops import CM_NT_GFX1250
from atom.mono.device.sync import Mailbox
THREADS=256
WAVES=8
CM_NT=CM_NT_GFX1250
DOWN_PTRS=('w2','w2_s','sw2','sw2_s','out')

def down_context(root,key):
    c=ug_context(root,key)
    if key.tokens > 16:
        # This is the same physical 0..255 ID as front_context's opaque
        # fresh copy. Limit the identity replacement to wide down stages.
        mhtjstest_tid = fx.thread_idx.x
        c.update(tid=mhtjstest_tid, lane=mhtjstest_tid % 32, wave=mhtjstest_tid // 32)
    c['args']={name:late_pointer(root,248+2*i) for i,name in enumerate(DOWN_PTRS)}
    mb=Mailbox(scalar_word(fx.ptr_load(root['late_meta']+218)),gfx1250=True)
    c.update(put=mb.put,put_bf=mb.put_bf,put_words=mb.put_words,poll=mb.poll,
             out_local=False,down_debug=root.get('down_debug',False))
    for name in ('dq',):
        c[name]=sreg(late_pointer(root,24),root['scratch_layout'][name][0],name)
    return c

def observation(c,offset=0):
    # Debug addresses start at consumption, never live across weight prefetch.
    return fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type,fx.AddressSpace.Global,4),late_pointer(c,260))+offset

def scale_block_loads(c,ws,first_row,scale_cols):
    # Fold both halves of the original 64-lane load/store block onto wave32.
    words=scale_cols*8
    block=first_row//32*words
    return [fx.Vector(bo.buffer_load(rsrc(ws),
        block+fx.min(256*i+(c['lane']+32*h)*4,words-4),
        vec_width=4,dtype=T.i32,cache_modifier=CM_NT))
        for i in range((words+255)//256) for h in range(2)]

def scale_block_store(c,svs,sl):
    for i in range_constexpr(len(svs)//2):
        for h in range_constexpr(2):
            fx.ptr_store(svs[2*i+h],sl+256*i+(c['lane']+32*h)*4)

def fp4_tile_weights(c,w,e,n_rows,k,rg,st0,blk,k_real=None):
    lane=c['lane'];base=e*(n_rows*k//8);rw=rsrc(w)
    result=[]
    for i in range_constexpr(blk):
        parts=[]
        for h in range_constexpr(2):
            # K tail maps to an in-bounds tile address, then zeroes that
            # fragment exactly as the mother's fp4_tile_weights does.
            group=lane//16+2*h
            real=True if k_real is None else (st0+i)*128+group*32<k_real
            half=lane//16 if isinstance(real,bool) else real.select(lane//16,fx.Int32(0))
            piece=h if isinstance(real,bool) else real.select(fx.Int32(h),fx.Int32(0))
            off=fp4_weight_offset(rg,st0+i,half,lane%16,piece,k)//4
            value=fx.Vector(bo.buffer_load(rw,base+off,vec_width=4,dtype=T.i32,cache_modifier=CM_NT))
            if const_expr(k_real is not None):
                value=fx.Vector.from_elements([real.select(value[q],fx.Int32(0)) for q in range(4)],fx.Int32)
            parts.append(value)
        result.append(parts[0].shuffle(parts[1],list(range(8))))
    return result

def mid_operands(c,first,count,st0=0,nst=None):
    """Same full round register prefetch, both logical wave64 halves at once."""
    lane,d=c['lane'],c['d']
    row=first+fx.min(lane%16,count-1)
    nst=d.inter//128 if nst is None else nst
    lane_word=fresh(row*d.mid_words+lane//16*4)
    bvs=[]
    for st in range(st0,st0+nst):
        chunks=[fx.Vector(bo.buffer_load(rsrc(c['mid'].value),lane_word+st*32+8*h,
            vec_width=4,dtype=T.i32,cache_modifier=CM_DEV)) for h in range(4)]
        bvs.append(chunks[0].shuffle(chunks[1],list(range(8))).shuffle(
            chunks[2].shuffle(chunks[3],list(range(8))),list(range(16))))
    groups=[]
    for group in range(4):
        base=mid_code_index(d,row,group)
        codes=[]
        for w0 in range_constexpr(0,nst,4):
            n=min(4,nst-w0)
            value=bo.buffer_load(rsrc(c['mids'].value),base+st0+w0,
                vec_width=n,dtype=T.i32,cache_modifier=CM_DEV)
            codes.extend([fx.Int32(value)] if n==1 else [fx.Int32(fx.Vector(value)[j]) for j in range(n)])
        groups.append(codes)
    return bvs,[groups[0][i]|(groups[1][i]<<8)|(groups[2][i]<<16)|(groups[3][i]<<24) for i in range(nst)]

def down_ug_flags(c,ucs):
    rs=c['rs']
    per_slot=rs.slot_tiles*rs.ug_parts if rs.multi else rs.ug_parts
    n=len(ucs)*per_slot
    return [_down_ug_flag(c,ucs,fx.min(c['lane']+32*h+64*j,n-1),per_slot)
        for h in range(2) for j in range(-(-n//64))]

def down_zero():
    return [fx.Vector.filled(8,0.,fx.Float32) for _ in range(DOWN_CHAINS)]

def down_chains(c,task,sl,mid,av,accs,st0=0):
    lane,d=c['lane'],c['d']
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,fx.Float4E2M1FN,fx.Float8E4M3FN,fx.Float32))
    regs=[fx.make_rmem_tensor(8,fx.Float32) for _ in range(DOWN_CHAINS)]
    for reg,value in zip(regs,accs):reg.store(value)
    bvs,sbs=mid
    for i in range_constexpr(len(av)):
        sw=fx.Int32(0)
        for group in range_constexpr(4):
            at=scale_index(task%2*16+lane%16,(st0+i)*4+group,d.down_scale_cols)
            sw=sw|(((fx.ptr_load(sl+at//4)>>(8*(at%4)))&255)<<(8*group))
        a,b=fx.make_rmem_tensor(8,fx.Int32),fx.make_rmem_tensor(16,fx.Int32)
        a.store(av[i]);b.store(bvs[i])
        acc=regs[(st0+i)%DOWN_CHAINS]
        fx.gemm(atom,acc,a,b,acc,scale_a=sw,scale_b=sbs[i])
    return [reg.load() for reg in regs]

@traced
def down_store(c,j,accs,first,count,live,task):
    lane,rs=c['lane'],c['rs']
    jj=lane%16
    pick=rs.pick_at(c['tab'],first+fx.min(jj,count-1))
    weight=rs.route_w(c['tab'],c['S'],pick//rs.topk,pick%rs.topk)
    value=accs[0]
    for h in range_constexpr(1,DOWN_CHAINS):value=value+accs[h]
    if live & (jj<count):
        for q in range_constexpr(0,8,2):
            row=8*(lane//16)+q
            word=bf16_pair(value[q]*weight,value[q+1]*weight).bitcast(fx.Int32)
            contrib_store(c,j,pick*8+row//2,word)

@traced
def down_shared(c,tasks):
    s,tid,d=c['S'],c['tid'],c['d']
    shops=[shared_down_loads(c,task) for task in tasks]
    for t0,nn in token_tiles(s):
        load_smid_tile(c,t0,nn,d.sh_inter//32)
        cc=dict(c,S=nn,xl=c['sxl'],xsl=c['sxsl'])
        for j in range_constexpr(len(tasks)):
            reference_fp8_matrices_rows(cc,d.sh_inter,shops[j],join=False)
            gpu.barrier()
            if tid<ROWS*nn:
                value=bf16_round(row_sum(c['red'],tid%ROWS,tid//ROWS))
                at=(j*s+plus(t0,tid//ROWS))*ROWS+tid%ROWS
                fx.ptr_store(value.bitcast(fx.Int32),c['sdl']+at)
            gpu.barrier()
def down_weights(c, tasks, rnd, slots, nu, st0=0, nst=None):
    """Round ``rnd``'s weights of ``stage_down``: each (task, slot)'s scale block
    and weight operands of K steps st0 .. st0 + nst (all of K by default), this
    wave's slots of U (a slot past U: the last's). Loads only."""
    s, wave, tab, a, d = c["S"], c["wave"], c["tab"], c["args"], c["d"]
    nst = d.inter // 128 if nst is None else nst
    us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
    es = [local_expert(c, c["rs"].union_expert(tab, s, fx.min(u, nu - 1))) for u in us]
    svs = [
        scale_block_loads(c, a["w2_s"], e * HIDDEN + task * 16, d.down_scale_cols)
        for task in tasks
        for e in es
    ]
    avs = [
        fp4_tile_weights(c, a["w2"], e, HIDDEN, d.inter, task, st0, nst, d.inter_real)
        for task in tasks
        for e in es
    ]
    return svs, avs
@traced
def down_finish(c, tasks):
    """Every contribution of ``tasks`` stored: combine them, token tile by
    token tile."""
    gpu.barrier()
    repeated = "second_live" in c  # ``RouteShape.down_one_call``
    for t0, nn in token_tiles(c["S"]):
        for j in range_constexpr(len(tasks)):
            if const_expr(j == 1 and repeated):
                if c["second_live"]:
                    down_combine(c, tasks[j], j, t0, nn)
            else:
                down_combine(c, tasks[j], j, t0, nn)
def shared_down_loads(c, task):
    """The shared expert's down rows 16 task .. over this rank's part of its
    intermediate: the replicated weight's columns of it under DP
    attention."""
    a, d = c["args"], c["d"]
    if const_expr(not c["rs"].dpa):
        return gemv_fp8_loads(c, a["sw2"], a["sw2_s"], d.sh_inter, task)
    return gemv_fp8_loads(
        c, a["sw2"], a["sw2_s"], d.sh_inter, task,
        k_row=SHARED_INTER, st0=c["rank"] * (d.sh_inter // 128),
    )  # fmt: skip
@traced
def load_smid_tile(c, t0, n, groups):
    """The shared expert's MXFP8 intermediate of tokens t0 .. t0 + n (SMQ /
    SMC, ``smid_quant``) into LDS ``sxl`` / ``sxsl`` rows 0 .., by every
    thread."""
    tid, words = c["tid"], c["d"].sh_inter // 4
    poll_copy(
        c, tid, THREADS,
        [
            (c["smq"], lambda u: plus(t0 * words, u), n * words, c["sxl"], words),
            (c["smc"], lambda u: plus(t0 * groups, u), n * groups, c["sxsl"], groups),
        ],
    )  # fmt: skip
    gpu.barrier()
def contrib_store(c, j, i, word):
    """Task j's routed contribution word i ([pick (t, k)][8 words]: rows 2i,
    2i + 1 of bf16(v w), packed)."""
    fx.ptr_store(word, c["contrib"] + (j * c["rs"].picks * 8 + i))
def contrib_load(c, j, pick, r):
    """Row r of task j's contribution of ``pick``: bf16(v w) as f32."""
    word = fx.ptr_load(c["contrib"] + (j * c["rs"].picks * 8 + pick * 8 + (r >> 1)))
    return ((r & 1) == 0).select(bf_lo(word), bf_hi(word))
@traced
def down_combine(c, task, j, t0, n):
    """Down task ``task`` (the CTA's j-th) for tokens t0 .. t0 + n: the routed
    contributions (each a bf16, as the reference's experts return) summed in
    f32 in each token's top-k order (expert parallel: of this rank's experts),
    plus the shared expert's (``down_shared``), rounded to bf16 once and pushed
    to every rank (token parallel: to the token's owner)."""
    s, tid, d, tab = c["S"], c["tid"], c["d"], c["tab"]
    topk = c["rs"].topk
    if tid < ROWS * n:
        r = tid & (ROWS - 1)
        tl = tid >> 4
        t = plus(t0, tl)
        shared = fx.ptr_load(c["sdl"] + ((j * s + t) * ROWS + r)).bitcast(fx.Float32)
        if const_expr(c['down_debug']):
            fx.ptr_store(shared,observation(c,c['rs'].picks*HIDDEN)+t*HIDDEN+task*ROWS+r)
        # the reference sums its experts' bf16 outputs in f32
        # (``inference/model.py`` ``MoE.forward``), not aiter's atomic stage 2
        # (a bf16 rounding an add)
        routed = fx.Float32(0.0)
        for k in range_constexpr(topk):
            contribution=contrib_load(c,j,t*topk+k,r)
            added=routed+contribution
            if const_expr(c['down_debug']):
                fx.ptr_store(contribution,observation(c)+(t*topk+k)*HIDDEN+task*ROWS+r)
            if const_expr(c["rs"].ep > 1):
                # another rank's pick: no contribution here (its LDS word
                # never written: selected away, not multiplied by zero)
                local = is_local(c, c["rs"].route_id(tab, s, t, k))
                added = local.select(added, routed)
            routed = added
        # the address afresh: hoisted, it was held a VGPR across every tile's
        # and task's combine (expert parallel, 12 rows a rank: a spill)
        fx.ptr_store(bf16_round(routed + shared), c["pair"] + fresh(tid))
    gpu.barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        v0 = fx.ptr_load(c["pair"] + (tl * ROWS + 2 * rp))
        v1 = fx.ptr_load(c["pair"] + (tl * ROWS + 2 * rp + 1))
        col = task * ROWS + 2 * rp
        if const_expr(c["tok_par"]):
            # to the token's owner only, at its own token index there
            own_n = own_count(s, d.ranks)
            owner = t // own_n
            for p in range_constexpr(d.ranks):
                if owner == p:
                    dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                    pair = region_pair(c["rank"], t - p * own_n, own_n, col)
                    c["put_bf"](dst, 2 * pair, [v0, v1])
        else:
            for p in range_constexpr(d.ranks):
                dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                c["put_bf"](dst, 2 * region_pair(c["rank"], t, s, col), [v0, v1])
    gpu.barrier()
@traced
def stage_reduce(c, task):
    """Columns 128 task ..: the TP ranks' partials in rank order 0 .. TP - 1, in
    fp32 -> bf16 -> ``out``."""
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * RED_COLS // 2, THREADS):
        reduce_pair(c, task, pas + tid)
@traced
def reduce_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 64."""
    s, tp = c["S"], c["d"].ranks
    a = c["args"]
    if idx < s * RED_COLS // 2:
        t = idx // (RED_COLS // 2)
        col = task * RED_COLS + 2 * (idx % (RED_COLS // 2))
        own = preg(c["sym"], c["peer"]["ffn"], "ffn")
        acc0, acc1 = sum_partials(
            c["poll"], own, lambda src: region_pair(src, t, s, col), tp
        )
        bo.buffer_store(acc0.to(fx.BFloat16), rsrc(a["out"]), t * HIDDEN + col)
        bo.buffer_store(acc1.to(fx.BFloat16), rsrc(a["out"]), t * HIDDEN + col + 1)
@traced
def take_down_rank(c):
    """This CTA's arrival rank at the down stage (DQ, a word a tag): which
    CTAs take the extra tiles. A tile's result is the same whichever CTA
    computes it."""
    if c["tid"] == 0:
        at = c["dq"].value + fx.Int64(c["tag"] * 4)
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), at)
        rank = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        fx.ptr_store(fx.Int32(rank), c["tab"] + c["rs"].queue0)
    gpu.barrier()
    return fx.ptr_load(c["tab"] + c["rs"].queue0)
@traced
def down_mfma(c, j, task, sl, mid, first, count, av, live):
    """Down task ``task`` (the CTA's j-th) on a slot of U: the routed MFMA of its
    picks first .. first + min(count, 16) (their B operands ``mid``:
    ``mid_operands``; the scale block in ``sl``), each pick's contribution."""
    accs = down_chains(c, task, sl, mid, av, down_zero())
    down_store(c, j, accs, first, count, live, task)
@traced
def stage_down(c, tasks):
    """Rows 16 task .. of the hidden output for each of ``tasks`` (a CTA's one or
    two, run together: a second pass would double the CTA's time): routed
    experts of U (a wave each, round robin), the shared expert, their sum ->
    pushed to every rank (token parallel: to the token's owner)."""
    assert down_parts(c["d"])==1  # TP4; EP/TP2 part branches remain out of scope
    s, wave, tab, d = c["S"], c["wave"], c["tab"], c["d"]
    n = len(tasks)
    nu = c["rs"].n_union(tab, s)
    sl = c["wsl"] + wave * down_wsl_words(d)
    sw = down_scale_words(d)
    # slots of U a round: the loads of a round stay the same
    slots = DOWN_R_MAX // n
    per_round = WAVES * slots
    # the shared expert's down projection first: its input is ready long
    # before the routed results, which a CTA mostly waits on here; after the
    # rounds it sat on the path past the last ug task
    down_shared(c, tasks)
    if const_expr(down_parts(d) > 1):
        stage_down_parts(c, tasks, nu, sl, sw, slots, per_round)
        return
    for rnd in range(0, (nu + per_round - 1) // per_round, 1):
        # the round's slots of U (wave, wave + WAVES, ...), every load of them
        # and of every task in flight before the first wait: the weights and
        # their scale blocks (no ug result: before the round's ug flags), then
        # the slots' picks' intermediate rows, each lane its pick's straight
        # from MID (``mid_operands``). A slot past U loads the last slot again
        # (L2 hits) and contributes nothing
        us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
        ucs = [fx.min(u, nu - 1) for u in us]
        firsts = [c["rs"].slot_first(tab, u) for u in ucs]
        svs, avs = down_weights(c, tasks, rnd, slots, nu)
        # this round's slots' ug tasks done, not every slot of the wave: the
        # round starts while later slots' ug run
        c["poll"]([(c["ugf"], f, 1) for f in down_ug_flags(c, ucs)])
        counts = [c["rs"].slot_first(tab, u + 1) - f for u, f in zip(ucs, firsts)]
        mids = [mid_operands(c, f, k) for f, k in zip(firsts, counts)]
        for q in range_constexpr(len(svs)):
            scale_block_store(c, svs[q], sl + q * sw)
        for i in range_constexpr(slots):
            for j in range_constexpr(n):
                down_mfma(
                    c, j, tasks[j], sl + (j * slots + i) * sw, mids[i], firsts[i],
                    counts[i], avs[j * slots + i], us[i] < nu,
                )  # fmt: skip
            # a slot of more than 16 picks: its next pick tile's rows
            if const_expr(s > TILE):
                for j0 in range_constexpr(TILE, s, TILE):
                    if j0 < counts[i]:
                        more = mid_operands(c, firsts[i] + j0, counts[i] - j0)
                        for j in range_constexpr(n):
                            down_mfma(
                                c, j, tasks[j], sl + (j * slots + i) * sw, more,
                                firsts[i] + j0, counts[i] - j0, avs[j * slots + i],
                                us[i] < nu,
                            )  # fmt: skip
    down_finish(c, tasks)

@traced
def run_down(root,key,bid):
    c=down_context(root,key)
    s=c['S']
    tid=c['tid']
    multi=len(token_tiles(s))>1
    ug_tasks=c['rs'].max_u*c['rs'].ug_parts
    assert not down_by_unit(c)
    # the route table is in LDS already past one token tile (every CTA loads it
    # for the ug queue) or when every CTA ran an ug iteration (more ug task
    # slots than CTAs), whose first loads it; else load it
    if const_expr(not multi and ug_tasks <= BLOCKS):
        load_route(c, c["tab"])
    # every CTA its tile; the first DOWN_TASKS - BLOCKS to get here an extra
    # one too: the CTAs whose ug ended first (by bid, those ending last held
    # two tiles and ended the stage)
    down_task = first_task(bid, DOWN0)
    rank = take_down_rank(c)
    if const_expr(c['down_debug']):
        if c['tid']==0:fx.ptr_store(rank.to(fx.Float32),observation(c,(c['rs'].picks+c['S'])*HIDDEN)+bid)
    has_second = rank < DOWN_TASKS - BLOCKS
    by_unit = down_by_unit(c)
    if const_expr(by_unit):
        # the down stamp at the units' end: the combine is timed in the reduce
        second = has_second.select(BLOCKS + rank, down_task)
        down_by_units(
            {**c, "second_live": has_second}, [down_task, second],
            lambda: stamp(timeline, tls, tid, tl0 + shift + 6),
        )  # fmt: skip
    elif const_expr(c["rs"].down_one_call):
        # a CTA without a second runs its own again (L2 hits), pushed once
        second = has_second.select(BLOCKS + rank, down_task)
        stage_down({**c, "second_live": has_second}, [down_task, second])
    elif has_second:
        stage_down(c, [down_task, BLOCKS + rank])
    else:
        stage_down(c, [down_task])
    # token parallel: this rank's tokens' sums only (``out`` from its first
    # token's row, or from row 0: ``out_local``)
    if const_expr(c["tok_par"]):
        cr = own_view(c, () if c["out_local"] else ("out",))
    else:
        cr = c
    for task in range(first_task(fresh(bid), RED0), RED_TASKS, BLOCKS):
        stage_reduce(cr, task)
