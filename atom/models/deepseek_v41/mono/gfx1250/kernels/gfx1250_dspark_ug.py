# SPDX-License-Identifier: MIT
"""Original DSpark UG structure with gfx1250 wave32 scaled WMMA operands."""
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_moe_front import front_context,load_x8_rows
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_union import load_route
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_index import late_pointer,scalar_word
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_moe import fp4_weight_offset
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import publish_plain as publish
from atom.mono.device.ops import CM_NT_GFX1250
THREADS=256
WAVES=8
CM_NT=CM_NT_GFX1250

def ug_context(root,key):
    c=front_context(root,key,'route')
    if key.tokens > 16:
        # front_context's fresh ID is the same physical thread ID. Express
        # its existing nonnegative range only in the wide routed UG stage.
        mhtjstest_tid = fx.thread_idx.x
        c.update(tid=mhtjstest_tid, lane=mhtjstest_tid % 32, wave=mhtjstest_tid // 32)
    c['args']={name:late_pointer(root,240+2*i) for i,name in enumerate(('w13','w13_s'))}
    c['red'],c['tab']=root['front_red'],root['front_tab']
    for name,offset in stage_lds(key.tokens,c['rs'])['offsets'].items():
        c[name]=root['front_stage']+offset
    for name in ('mid','mids','ugf','ugq','ugc'):
        c[name]=sreg(late_pointer(root,24),root['scratch_layout'][name][0],name)
    c['tag']=scalar_word(fx.ptr_load(root['late_meta']+218))+1
    c['ug_debug']=root.get('ug_debug',False)
    if c['ug_debug']:
        c['ug_gu']=fx.inttoptr(fx.PointerType.get(fx.Float32.ir_type,fx.AddressSpace.Global,4),late_pointer(root,244))
        c['ug_mid']=fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type,fx.AddressSpace.Global,2),late_pointer(root,246))
    return c

@traced
def ug_pair_loads(c,e,rg0,sc0,st0,n,sc_steps):
    """One mother round: both logical scale halves and paired projection tiles."""
    lane,d=c['lane'],c['d']
    rw,rs=rsrc(c['args']['w13']),rsrc(c['args']['w13_s'])
    # Keep round-address arithmetic local to the prefetch instead of retaining
    # all unrolled target K-step scalar offsets through the matrix loop.
    if c['rs'].ug_k_slices==2 or c['S']==6:
        sc0,st0=fresh(sc0),fresh(st0)
    block=(e*(2*d.inter)+rg0*16)//32*(HIDDEN//32*8)
    words=sc_steps//2
    sv=[]
    for half in range_constexpr(2):
        v=bo.buffer_load(rs,block+32*sc0+(lane+32*half)*words,
                         vec_width=words,dtype=T.i32,cache_modifier=CM_NT)
        sv.append(fx.Vector(v) if words>1 else fx.Int32(v))
    # Dword base before dynamic multiply avoids signed32 byte-base overflow.
    start=e*((2*d.inter)*(HIDDEN//2)//4)
    weights=[]
    for tile in range_constexpr(2):
        pieces=[]
        for i in range_constexpr(n):
            halves=[fx.Vector(bo.buffer_load(rw,start+fp4_weight_offset(
                rg0+tile,st0+i,lane//16,lane%16,j,HIDDEN)//4,
                vec_width=4,dtype=T.i32,cache_modifier=CM_NT)) for j in range(2)]
            pieces.append(halves[0].shuffle(halves[1],list(range(8))))
        weights.append(pieces)
    return sv,weights

@traced
def ug_pair_matrix(c,e,rg0,ks,col):
    """Mother K partitions and prefetch depth; one input serves gate and up."""
    lane,wave=c['lane'],c['wave']
    sl=c['wsl']+wave*UG_WSL_WORDS
    xl=c['xl']+col*lds_row(HIDDEN//4)
    xsl=c['xsl']+col*lds_row(HIDDEN//32)
    slices=c['rs'].ug_k_slices
    # A physical wave still owns the same uniform K slice.  Materialize its
    # address operand here so LLVM cannot retain every later round's scalar
    # offsets through route construction and the persistent task loop.
    ks=fresh(ks)
    rounds=ug_rounds(HIDDEN,ks,slices)
    ahead=slices//2
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,fx.Float4E2M1FN,fx.Float8E4M3FN,fx.Float32))
    accs=[fx.make_rmem_tensor(8,fx.Float32) for _ in range(2)]
    for acc in accs:acc.store(fx.full(8,0.,fx.Float32))
    ops=[ug_pair_loads(c,e,rg0,*rounds[q]) for q in range(min(ahead,len(rounds)))]
    for r in range_constexpr(len(rounds)):
        sc0,st0,n,sc_steps=rounds[r]
        sv,weights=ops[r]
        for half in range_constexpr(2):
            fx.ptr_store(sv[half],sl+(lane+32*half)*(sc_steps//2))
        for i in range_constexpr(n):
            # The target's eight step offsets otherwise remain SGPR-live
            # from weight prefetch through later activation consumption.
            step=fresh(st0+i) if slices==2 else st0+i
            chunks=[fx.Vector(fx.ptr_load(xl+step*32+lane//16*4+j*8,
                result_type=fx.Vector.make_type(4,fx.Int32))) for j in range(4)]
            xv=chunks[0].shuffle(chunks[1],list(range(8))).shuffle(
                chunks[2].shuffle(chunks[3],list(range(8))),list(range(16)))
            sx=fx.Int32(0)
            for j in range_constexpr(4):sx=sx|(fx.ptr_load(xsl+step*4+j)<<(8*j))
            x=fx.make_rmem_tensor(16,fx.Int32);x.store(xv)
            for tile in range_constexpr(2):
                sw=fx.Int32(0)
                for j in range_constexpr(4):
                    # sc0 and st0 are even K128 steps in every mother round.
                    # Exact shuffle_scale algebra removes signed div/mod
                    # address temporaries otherwise live across all K steps.
                    word=32*(st0-sc0)+(i//2)*64+j*16+lane%16
                    shift=8*(2*(i%2)+tile)
                    sw=sw|(((fx.ptr_load(sl+word)>>shift)&255)<<(8*j))
                w=fx.make_rmem_tensor(8,fx.Int32);w.store(weights[tile][i])
                fx.gemm(atom,accs[tile],w,x,accs[tile],scale_a=sw,scale_b=sx)
        if const_expr(r+ahead<len(rounds)):
            ops.append(ug_pair_loads(c,e,rg0,*rounds[r+ahead]))
    return [acc.load() for acc in accs]

@traced
def ug_pass(c,e,part0,first,count,col):
    tid,lane,wave=c['tid'],c['lane'],c['wave']
    rs=c['rs']
    p,rg0,ks,real=ug_wave_pair(c,part0)
    accs=[fx.Vector.filled(8,0.,fx.Float32) for _ in range(2)]
    if real:accs=ug_pair_matrix(c,local_expert(c,e),rg0,ks,col)
    # Restore the original logical64 RED fragment before its pairwise reduction.
    if lane%16<rs.tile_picks:
        for tile in range_constexpr(2):
            slot=ug_red_slot(rs,p,wave%2,ks,tile)
            for j in range_constexpr(8):
                row=lane//16*8+j
                at=(16*(row//4)+lane%16)*4+row%4
                fx.ptr_store(accs[tile][j],c['red']+slot*64*4+at)
    gpu.barrier()
    for pas in range_constexpr(0,rs.ug_groups*rs.tile_picks*UG_PART,THREADS):
        ug_epilogue(c,part0,pas+tid,first,count)
def ug_rounds(k, ks, slices):
    """K slice ``ks`` of ``slices``' rounds (first step of the scales loaded,
    first step, steps, steps of scales loaded); ``ks`` traced: the steps are.
    Halves: the 8-step scale blocks 2 r + ks, the odd block left split 4 + 4,
    so both run k / 256 steps. Quarters: steps 8 i + 2 ks, + 1 of each block
    i, a round loading only their scales (the shuffled layout keeps 2 steps'
    together: a dword a lane, where the whole block was 4)."""
    blocks = k // 128 // 8
    if slices == 4:
        return [(fx.Int32(8 * i) + ks * 2,) * 2 + (2, 2) for i in range(blocks)]
    assert slices == 2
    rounds = [
        (fx.Int32(16 * r) + ks * 8, fx.Int32(16 * r) + ks * 8, 8, 8)
        for r in range(blocks // 2)
    ]
    if blocks % 2:
        last = (blocks - 1) * 8
        rounds.append((fx.Int32(last), fx.Int32(last) + ks * 4, 4, 8))
    return rounds
def ug_wave_pair(c, part0):
    """This wave's share of an ug task from group ``part0``: group p of its
    ``ug_groups`` (wave / (2 slices)), its 16-column half bb = wave % 2 -- w13's
    gate tile 2 b and up tile 2 b + 1 (aiter GUGU interleaves them a 16-row
    tile; b = 2 (part0 + p) + bb), one 32-row scale block -- over K slice ks =
    wave / 2 % slices (``gemv_fp4_pair``): (p, 2 b, ks, whether the group is in
    the real width)."""
    wave, slices = c["wave"], c["rs"].ug_k_slices
    p = wave // (2 * slices)
    b = (part0 + p) * 2 + wave % 2
    return p, 2 * b, wave // 2 % slices, part0 + p < c["d"].inter_real // UG_PART
def ug_red_slot(rs, p, bb, ks, t):
    """RED's 64-lane slot of an ug task's partial C: group p, half bb, K slice
    ks, gate (t 0) or up (t 1)."""
    return ((p * 2 + bb) * rs.ug_k_slices + ks) * 2 + t
def stage_lds(s, rs):
    """Word offsets into ``Smem.stage`` of the buffers only one stage reads: the
    shared stage's MXFP8 x tile; the waves' scale blocks (``wsl``: the ug
    stage's ``UG_WSL_WORDS`` a wave, the down stage's ``down_wsl_words``), then
    the ug stage's routed MXFP8 x tile, or the down stage's token tile of the
    shared expert's MXFP8 intermediate (``load_smid_tile``), each down task's
    routed contributions (``contrib_store``) and its shared expert rows
    (``down_shared``). A barrier ends every stage."""
    d = rs.dims
    # the x tile the shared / ug stages hold (every token's up to one tile)
    rows = tile_rows(s)
    x8 = rows * lds_row(X8_WORDS)
    groups = rows * lds_row(HIDDEN // 32)
    ug_wsl = WAVES * UG_WSL_WORDS
    down_wsl = WAVES * down_wsl_words(d)
    sxl = rows * lds_row(d.sh_inter // 4) + lds_tail(d.sh_inter)
    sxsl = rows * lds_row(d.sh_inter // 32)
    # a CTA's two down tasks' contributions, a pick's 16 rows of bf16 a task
    contrib = 2 * rs.picks * 8
    offsets = {
        "x8l": 0, "x8sl": x8,
        "wsl": 0, "x8rl": ug_wsl, "x8rsl": ug_wsl + x8,
        "sxl": down_wsl, "sxsl": down_wsl + sxl,
        "contrib": down_wsl + sxl + sxsl,
        "sdl": down_wsl + sxl + sxsl + contrib,
    }  # fmt: skip
    words = {
        "shared": x8 + groups,
        "ug": ug_wsl + x8 + groups,
        "down": down_wsl + sxl + sxsl + contrib + 2 * s * ROWS,
    }
    return {"offsets": offsets, "words": {"all": max(words.values())}}
def gather_x8r(c, first, count):
    """The X8 rows and codes of the slot's picks first .. first + min(count, 16)
    into LDS rows 0 .. (only those rows: an ug pass reads a pick's row, the
    columns past them repeat the last one's)."""
    tab, topk = c["tab"], c["rs"].topk

    def token(j):
        return c["rs"].pick_at(tab, first + fx.min(j, count - 1)) // topk

    load_x8_rows(c, "x8", "x8s", c["x8rl"], c["x8rsl"], token, fx.min(count, TILE))
def ug_tile_task(c, task):
    """Past one token tile, ug task ``task``: its slot's expert, its first
    group, and its pick tile's first pick and the picks from it."""
    tab, rs, s = c["tab"], c["rs"], c["S"]
    ug_parts = rs.ug_parts
    tile = task // ug_parts
    u = rs.tile_slot(tab, tile)
    first = mplan.tile_pick0(rs.slot_first(tab, u), tile, rs.tile_first(tab, u))
    count = rs.slot_first(tab, u + 1) - first
    return rs.union_expert(tab, s, u), task % ug_parts * rs.ug_groups, first, count
@traced
def stage_ug(c, task):
    """One token tile: slot task // ug_parts of U (``RouteShape.ug_parts``), the
    ``ug_groups`` FP4 groups of intermediate columns from 32 ug_groups (task %
    ug_parts) (``ug_wave_pair``); then silu x up, rounded to bf16, and the
    MXFP8 requant of each token's 32 values of a group -> MID (the stage-1
    epilogue and the stage-2 input quant). Past one: ``stage_ug_tile``."""
    s, tid, lane, tab = c["S"], c["tid"], c["lane"], c["tab"]
    rs = c["rs"]
    ug_parts = rs.ug_parts
    u = task // ug_parts
    if u < c["rs"].n_union(tab, s):
        e = c["rs"].union_expert(tab, s, u)
        # the slot's picks, a column each (a column past them repeats the last);
        # every token's row is in LDS: a column reads its pick's token's
        first = c["rs"].slot_first(tab, u)
        count = c["rs"].slot_first(tab, u + 1) - first
        pick = c["rs"].pick_at(tab, first + fx.min(lane % 16, count - 1))
        ug_pass(c, e, task % ug_parts * rs.ug_groups, first, count, pick // rs.topk)
        publish(c["put"], c["ugf"], task, 1, tid == 0)
    gpu.barrier()
@traced
def ug_epilogue(c, part0, idx, first, count):
    """Thread ``idx``'s value of an ug task: silu x up of its (group, pick j,
    column), the group's MXFP8 scale, its byte of MID (the pick's row, ``first``
    + j in the pick order)."""
    lane, red, d, rs = c["lane"], c["red"], c["d"], c["rs"]
    idx=fresh(idx)  # epilogue address lifetimes start after the matrix
    cols = rs.tile_picks
    if idx < rs.ug_groups * cols * UG_PART:
        p = idx // (cols * UG_PART)
        part = part0 + p
        t = idx % (cols * UG_PART) // UG_PART
        i = idx % UG_PART
        tile = i // 16
        r = i % 16
        at = (16 * (r // 4) + t) * 4 + r % 4
        g, up = [
            pairwise_sum(
                [
                    fx.ptr_load(red + ug_red_slot(rs, p, tile, ks, t) * 64 * 4 + at)
                    for ks in range(rs.ug_k_slices)
                ]
            )
            for t in range(2)
        ]
        if const_expr(c['ug_debug']):
            if t<count:
                fx.ptr_store(g,c['ug_gu']+(first+t)*(2*d.inter)+part*UG_PART+i)
                fx.ptr_store(up,c['ug_gu']+(first+t)*(2*d.inter)+d.inter+part*UG_PART+i)
        gc = fx.min(g, fx.Float32(LIMIT))
        uc = fx.max(fx.min(up, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
        sig = hw_rcp(fx.Float32(1.0) + hw_exp2(gc * fx.Float32(-LOG2E)))
        # the stage-1 kernel stores the intermediate as bf16; the stage-2 input
        # quant (fused_mx_quant_moe_sort) makes it MXFP8: amax floored
        # AMAX_FLOOR, the ceil code, x / scale clamped, RNE
        v = bf16_round(gc * sig * uc)
        if const_expr(c['ug_debug']):
            if t<count:fx.ptr_store(v.to(fx.BFloat16),c['ug_mid']+(first+t)*d.inter+part*UG_PART+i)
        amax = fx.max(
            butterfly(abs(v), (1, 2, 4, 8, 16), fx.max), fx.Float32(AMAX_FLOOR)
        )
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        q8 = clamp_fp8(v * rcp_pow2(code))
        nb = [
            lane_gather(q8.bitcast(fx.Int32), fx.min(lane + q, 31)).bitcast(fx.Float32)
            for q in range(1, 4)
        ]
        # 4 threads' bytes -> one word
        word = fp8_pack4(q8, *nb)
        row = first + t
        if (i % 4 == 0) & (t < count):
            st_dev(word, c["mid"], row * d.mid_words + part * 8 + i // 4)
        if (i == 0) & (t < count):
            st_dev(code, c["mids"], mid_code_index(d, row, part))
@traced
def stage_ug_tile(c, task, total):
    """Past one token tile: ug task ``task`` of ``total`` (a pick tile of its
    slot), then the next task from the queue (``take_ug_task``). No next task's
    weights prefetched under this one: held across its epilogue and gather they
    spilled the merged K2 (S = 24 / 48), and without them it ran faster."""
    tid, lane = c["tid"], c["lane"]
    e, part0, first, count = ug_tile_task(c, task)
    gather_x8r(c, first, count)
    ug_pass(c, e, part0, first, count, fx.min(lane % 16, count - 1))
    nxt = (total > BLOCKS).select(take_ug_task(c) + BLOCKS, total)
    if const_expr(not down_by_unit(c)):
        # the epilogue's MID stores done before the task's flag (down by unit:
        # the CTA's flag after its last task, ``run_moe``)
        publish(c["put"],c["ugf"],task,1,tid==0)
    return nxt

@traced
def run_ug(root,key,bid):
    c=ug_context(root,key)
    s,tid=key.tokens,c['tid']
    multi=len(token_tiles(s))>1
    ug_tasks=c['rs'].max_u*c['rs'].ug_parts
    enter_stage("k2b.ug")
    c["xl"], c["xsl"] = c["x8rl"], c["x8rsl"]
    # the ug stage's wave an SGPR: the signed // of an id LLVM cannot see is
    # non-negative kept its fix-up live in VGPRs through the ug loop (TP 2,
    # 12 rows: a spill). Elsewhere a VGPR: as an SGPR it slowed down (TP 4,
    # 48 rows +4 us)
    cu = {**c, "wave": uniform(c["tid"] >> 5)}
    if const_expr(multi):
        # past one token tile a task gathers its pick tile's rows itself. The
        # first round placed (the shared units' CTAs take its last tasks); the
        # rest, past a round, from the queue: a CTA a shared or router task
        # delayed takes fewer of them
        load_route(c, c["tab"])
        n_tiles = c["rs"].tile_first(c["tab"], c["rs"].n_union(c["tab"], s))
        ug_total = mplan.ug_tasks(n_tiles, c["rs"].ug_parts)
        task = first_task(bid, c["rs"].ug0)
        while task < ug_total:
            task = stage_ug_tile(cu, task, ug_total)
        if const_expr(down_by_unit(c)):
            publish(c["put"], c["ugc"], bid, 1, tid == 0)
    else:
        # one token tile: every task placed (its x in LDS once a CTA)
        loaded = fx.Int32(0)
        for task in range(first_task(bid, c["rs"].ug0), ug_tasks, BLOCKS):
            if loaded == 0:
                # X8 (ready long before the routing) first: after the routing
                # lands only its own poll remains
                load_x8_rows(c, "x8", "x8s", c["x8rl"], c["x8rsl"], lambda j: j, s)
                load_route(c, c["tab"])
            loaded = fx.Int32(1)
            stage_ug(cu, task)
    return c
