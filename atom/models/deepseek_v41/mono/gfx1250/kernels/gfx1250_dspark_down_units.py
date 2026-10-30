"""Expert-row-tile x N256 routed down, within the full mono K2.

Port the existing gfx950 down-unit scheduling/DCON protocol to TP4 gfx1250,
with per-tile UGF dependencies, native wave32 WMMA and shared MID operands.
"""
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_down import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_down import down_context as base_context
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import publish_plain as publish
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import take_down_unit

UNIT_N = 256
UNIT_BLOCKS = HIDDEN // UNIT_N


def unit_context(root, key):
    c=base_context(root,key)
    assert key.tokens==48 and key.topk==6 and key.ep==1 and not key.tok_par
    for name in ('dnq','dcon','dnf'):
        c[name]=sreg(late_pointer(root,24),root['scratch_layout'][name][0],name)
    d=c['d']
    # The old per-CTA routed contribution array is dead in this organization.
    # Reuse it for one compact pick tile, without increasing resident LDS.
    words=16*lds_row(d.mid_words)
    assert words+16*(d.inter//128)<=2*c['rs'].picks*8
    c['rd_x']=c['contrib']
    c['rd_s']=c['contrib']+words
    return c


@traced
def load_mid_tile(c,first,count):
    tid,d=c['tid'],c['d']
    stride=lds_row(d.mid_words)
    quads=d.mid_words//4
    for begin in range_constexpr(0,16*quads,THREADS):
        q=tid+begin
        if q<count*quads:
            row,col=q//quads,4*(q%quads)
            value=fx.Vector(bo.buffer_load(rsrc(c['mid'].value),
                (first+row)*d.mid_words+col,vec_width=4,dtype=T.i32,cache_modifier=CM_DEV))
            fx.ptr_store(value,c['rd_x']+row*stride+col)
    steps=d.inter//128
    for begin in range_constexpr(0,16*steps,THREADS):
        q=tid+begin
        if q<count*steps:
            row,st=q//steps,q%steps
            code=fx.Int32(0)
            for group in range_constexpr(4):
                part=fx.Int32(bo.buffer_load(rsrc(c['mids'].value),
                    mid_code_index(d,first+row,st*4+group),vec_width=1,dtype=T.i32,cache_modifier=CM_DEV))
                code=code|(part<<(8*group))
            fx.ptr_store(code,c['rd_s']+row*steps+st)
    gpu.barrier()


@traced
def unit_matrix(c,e,task0,first,count,sl):
    lane,d=c['lane'],c['d']
    row=fx.min(lane%16,count-1)
    xl=c['rd_x']+row*lds_row(d.mid_words)
    xs=c['rd_s']+row*(d.inter//128)
    tasks=[task0,task0+1]
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,fx.Float4E2M1FN,fx.Float8E4M3FN,fx.Float32))
    accs=[[fx.make_rmem_tensor(8,fx.Float32) for _ in range(DOWN_CHAINS)] for _ in tasks]
    for chain in accs:
        for acc in chain:acc.store(fx.full(8,0.,fx.Float32))
    # Two weight slots; all 8 waves reuse the CTA's single MID/scale tile.
    weights=[[fp4_tile_weights(c,c['args']['w2'],e,HIDDEN,d.inter,t,i,1,d.inter_real)[0]
              for t in tasks] for i in range(2)]
    a,b=fx.make_rmem_tensor(8,fx.Int32),fx.make_rmem_tensor(16,fx.Int32)
    for st in range_constexpr(d.inter//128):
        current=weights[st%2]
        if const_expr(st+2<d.inter//128):
            weights[st%2]=[fp4_tile_weights(c,c['args']['w2'],e,HIDDEN,d.inter,t,st+2,1,d.inter_real)[0] for t in tasks]
        chunks=[fx.Vector(fx.ptr_load(xl+st*32+lane//16*4+j*8,
                result_type=fx.Vector.make_type(4,fx.Int32))) for j in range(4)]
        b.store(fx.Vector.from_elements([chunks[j][q] for j in range(4) for q in range(4)],fx.Int32))
        sx=fx.ptr_load(xs+st)
        for j in range_constexpr(2):
            sw=fx.Int32(0)
            for group in range_constexpr(4):
                at=scale_index(j*16+lane%16,st*4+group,d.down_scale_cols)
                sw=sw|(((fx.ptr_load(sl+at//4)>>(8*(at%4)))&255)<<(8*group))
            a.store(current[j])
            acc=accs[j][st%DOWN_CHAINS]
            fx.gemm(atom,acc,a,b,acc,scale_a=sw,scale_b=sx)
    pick=c['rs'].pick_at(c['tab'],first+row)
    w=c['rs'].route_w(c['tab'],c['S'],pick//c['rs'].topk,pick%c['rs'].topk)
    for j in range_constexpr(2):
        value=accs[j][0].load()
        for chain in range_constexpr(1,DOWN_CHAINS):value=value+accs[j][chain].load()
        if lane%16<count:
            for q in range_constexpr(0,8,2):
                col=tasks[j]*16+(lane//16)*8+q
                word=bf16_pair(value[q]*w,value[q+1]*w).bitcast(fx.Int32)
                st_dev(word,c['dcon'],pick*(HIDDEN//2)+col//2)
    gpu.barrier()


@traced
def unit(c,work):
    tile,block=work//UNIT_BLOCKS,work%UNIT_BLOCKS
    rs,tab=c['rs'],c['tab']
    slot=rs.tile_slot(tab,tile)
    e=local_expert(c,rs.union_expert(tab,c['S'],slot))
    first=mplan.tile_pick0(rs.slot_first(tab,slot),tile,rs.tile_first(tab,slot))
    count=fx.min(fx.Int32(16),rs.slot_first(tab,slot+1)-first)
    task0=(block*WAVES+c['wave'])*2
    sl=c['wsl']+c['wave']*down_wsl_words(c['d'])
    scales=scale_block_loads(c,c['args']['w2_s'],e*HIDDEN+task0*16,c['d'].down_scale_cols)
    # Only this pick tile's UG producers are dependencies; retain their
    # original per-task publication rather than introducing a global UG join.
    c['poll']([(c['ugf'],tile*rs.ug_parts+fx.min(c['tid'],rs.ug_parts-1),1)])
    load_mid_tile(c,first,count)
    scale_block_store(c,scales,sl)
    unit_matrix(c,e,task0,first,count,sl)


@traced
def finish(c,tasks):
    for f in range(c['tid'],BLOCKS,THREADS):
        c['poll']([(c['dnf'],f,1)])
    gpu.barrier()
    for begin in range_constexpr(0,len(tasks)*c['S']*8,THREADS):
        idx=begin+c['tid']
        if idx<len(tasks)*c['S']*8:dcon_pair(c,tasks,idx)
    gpu.barrier()


@traced
def run_down(root,key,bid):
    c=unit_context(root,key)
    # Existing route table, arrival rank and final column ownership are kept.
    task=first_task(bid,DOWN0)
    arrival=take_down_rank(c)
    second=BLOCKS+arrival
    live=arrival<DOWN_TASKS-BLOCKS
    nu=c['rs'].n_union(c['tab'],c['S'])
    total=c['rs'].tile_first(c['tab'],nu)*UNIT_BLOCKS
    work=first_task(bid,DOWN0)
    while work<total:
        unit(c,work)
        work=take_down_unit(c)+BLOCKS
    publish(c['put'],c['dnf'],bid,1,c['tid']==0)
    # Useful shared work occupies the interval before the final DCON join.
    if live:
        down_shared(c,[task,second])
        finish(dict(c,second_live=live),[task,second])
    else:
        down_shared(c,[task])
        finish(c,[task])
    for t in range(first_task(fresh(bid),RED0),RED_TASKS,BLOCKS):
        stage_reduce(c,t)


# EP1 specialization: every routed pick is local, enforced by unit_context.
@traced
def dcon_pair(c, tasks, idx):
    """``down_units_combine``'s pair ``idx``: rows 2 rp, 2 rp + 1 of a task's
    16 for one token, summed as ``down_combine`` (f32 in top-k order, local
    picks only, then the shared expert's, one bf16 rounding), every routed
    contribution's load issued together."""
    s, tab, d = c["S"], c["tab"], c["d"]
    topk = c["rs"].topk
    per_task = s * (ROWS // 2)
    j = idx // per_task
    rem = idx % per_task
    t = rem // (ROWS // 2)
    rp = rem % (ROWS // 2)
    task = tasks[0]
    live = j == 0
    if const_expr(len(tasks) > 1):
        task = (j == 0).select(tasks[0], tasks[1])
        live = (j == 0) | c["second_live"]
    col = task * ROWS + 2 * rp
    words = [
        fx.Int32(
            bo.buffer_load(
                rsrc(c["dcon"].value),
                (t * topk + k) * (HIDDEN // 2) + col // 2,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        for k in range(topk)
    ]
    r0, r1 = fx.Float32(0.0), fx.Float32(0.0)
    for k in range_constexpr(topk):
        r0 = r0 + bf_lo(words[k])
        r1 = r1 + bf_hi(words[k])
    base = (j * s + t) * ROWS + 2 * rp
    sh0 = fx.ptr_load(c["sdl"] + base).bitcast(fx.Float32)
    sh1 = fx.ptr_load(c["sdl"] + (base + 1)).bitcast(fx.Float32)
    v0 = bf16_round(r0 + sh0)
    v1 = bf16_round(r1 + sh1)
    if const_expr(c["tok_par"]):
        # to the token's owner only, at its own token index there
        own_n = own_count(s, d.ranks)
        owner = t // own_n
        for p in range_constexpr(d.ranks):
            if live & (owner == p):
                dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                pair = region_pair(c["rank"], t - p * own_n, own_n, col)
                c["put_bf"](dst, 2 * pair, [v0, v1])
    else:
        for p in range_constexpr(d.ranks):
            if live:
                dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                c["put_bf"](dst, 2 * region_pair(c["rank"], t, s, col), [v0, v1])
