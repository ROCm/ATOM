# SPDX-License-Identifier: MIT
"""Original XQ/router/route/shared dependencies for TP4 DSpark target/draft."""
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_seam import x_context,run_gather
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_index import late_pointer
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import publish_plain as publish
from flydsl._mlir.dialects import llvm
from atom.mono.device.ops import ballot
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_k1 import gemv_fp8_loads
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_multirow_projection import reference_fp8_matrices_rows
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_attention import _wmma
from atom.model_ops.deepseek_v41.router import router_x_loads as mother_x_loads,router_weight_loads as mother_weight_loads
from atom.models.deepseek_v41.mono.gfx950.kernels.moe import _score,_silu_mul
from atom.mono.device.ops import CM_NT_GFX1250,readlane,fresh
THREADS=256
WAVES=8
CM_NT=CM_NT_GFX1250
MOE_FRONT_PTRS=('gate_w','bias','sgu','sgu_s')

def front_context(c,key,stage):
    out=x_context(c)
    tid=fresh(fx.thread_idx.x)
    out.update(tid=tid,lane=tid%32,wave=tid//32,rs=route_shape(key.halves()[1]),
               x_cm=CM_DEV,x_ready=True,red=c['front_red'],pair=c['front_pair'],
               x8l=c['front_xl'],x8sl=c['front_xsl'],sh_gate0=0,sh_up0=c['d'].shared_tasks)
    names={'xq':(), 'router':('gate_w',), 'route':('bias',), 'shared':('sgu','sgu_s')}[stage]
    out['args'].update({name:late_pointer(c,224+2*MOE_FRONT_PTRS.index(name)) for name in names})
    for name in ('x8','x8s','x8rdy','logit','route','smid','smq','smc'):
        out[name]=sreg(late_pointer(c,24),c['scratch_layout'][name][0],name)
    out.update(route_dsts=[out['route']],route_src=out['route'],route_rows=c['S'],route_t0=0)
    out['shared_debug']=c.get('shared_debug',False)
    if out['shared_debug'] and stage=='shared':
        out['shared_gu']=fx.inttoptr(fx.PointerType.get(fx.BFloat16.ir_type,fx.AddressSpace.Global,2),late_pointer(c,234))
    return out

def combined(lo,hi):
    return [fx.Vector.from_elements([lo[i][j] for j in range(4)]+[hi[i][j] for j in range(4)],fx.Int32)
            for i in range(5)]

def router_weight_loads(w,tile,part,lane,wave,cm):
    vl=lane%16+(lane//16)*32
    return combined(mother_weight_loads(w,tile,part,vl,wave,cm),
                    mother_weight_loads(w,tile,part,vl+16,wave,cm))

def router_x_loads(x,part,n,lane,wave,cm,t0):
    vl=lane%16+(lane//16)*32
    return combined(mother_x_loads(x,part,n,vl,wave,cm,t0),
                    mother_x_loads(x,part,n,vl+16,wave,cm,t0))

def router_tile_mfma(weights,inputs,lane,wave,red):
    accum=fx.make_rmem_tensor(8,fx.Float32)
    accum.store(fx.full(8,0.,fx.Float32))
    for i in range_constexpr(5):
        _wmma(accum,inputs[i].bitcast(fx.BFloat16),weights[i].bitcast(fx.BFloat16))
    values=accum.load()
    for j in range_constexpr(8):
        row=lane//16*8+j
        at=(wave*64+16*(row//4)+lane%16)*4+row%4
        fx.ptr_store(values[j],red+at)
def dual_ballot(pred0, pred1):
    """Original logical wave64 mask, with both physical ballots zero extended."""
    mask = fx.Int64(0xffffffff)
    return (ballot(pred0)&mask) | ((ballot(pred1)&mask)<<32)
@traced
def stage_xq(c, t):
    """Token t, a group of 32 a thread."""
    # thread g < 160 owns the 32 values of group g; the rest mirror thread 159
    g = fx.min(c["tid"], HIDDEN // 32 - 1)
    # 32 bf16 as four 16-byte loads (a load an element made the stage issue bound)
    words = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["x"]),
                (t * HIDDEN + g * 32) // 2 + 4 * q,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=c["x_cm"],
            )
        )
        for q in range(4)
    ]
    xq_quant(c, t, g, [words[i // 4][i % 4] for i in range(16)])
@traced
def xq_quant(c, t, g, words):
    """Token t's group g (16 words of bf16 pairs) as MXFP8: the routed
    experts' and the shared one's (one rule)."""
    mine = c["tid"] < HIDDEN // 32
    v = [(bf_lo if i % 2 == 0 else bf_hi)(words[i // 2]) for i in range(32)]
    amax = fx.Float32(AMAX_FLOOR)
    for y in v:
        amax = fx.max(amax, abs(y))
    if mine:
        put_mxfp8_group(c, "x8", "x8s", t, g, v, amax)
def put_mxfp8_group(c, words_mb, codes_mb, t, g, v, amax):
    """Token t's group g of 32 values ``v`` as MXFP8 (the ceil code of
    ``amax`` / 448, x / scale clamped, RNE) -> scratch ``words_mb`` /
    ``codes_mb`` (plain words: the token's X8RDY announces them)."""
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    inv = rcp_pow2(code)
    words = []
    for w4 in range_constexpr(4):
        ws = [
            fp8_pack4(*[clamp_fp8(v[8 * w4 + 4 * h + i] * inv) for i in range(4)])
            for h in range(2)
        ]
        words.append(ws)
    for h in range_constexpr(2):
        st_dev(
            fx.Vector.from_elements(words[2 * h] + words[2 * h + 1], fx.Int32),
            c[words_mb],
            t * X8_WORDS + g * 8 + 4 * h,
        )
    st_dev(fx.Int32(code), c[codes_mb], t * (HIDDEN // 32) + g)
def router_weights(c, task):
    """``stage_router``'s gate weight loads for ``task``: they do not wait for
    x, so a CTA issues them before x's wait."""
    tile = task % (c["rs"].experts // ROWS * ROUTER_PARTS) // ROUTER_PARTS
    part = task % ROUTER_PARTS
    return router_weight_loads(
        c["args"]["gate_w"], tile, part, c["lane"], c["wave"], CM_NT
    )
@traced
def stage_router(c, task, wvs):
    """Router task (token group, row tile, K part): K part ``part`` of rows
    16 ``tile`` .. of the bf16 gate weight [experts, 5120] times the group's
    tokens (``RouteShape.router_groups``), a token tile at a time -- the
    original path's decode router (``atom.model_ops.deepseek_v41.router``), the
    same tiles in the same order -> the part's f32 sum (``route`` sums the
    parts in order, rounds to bf16). Every tile's x loads issue first: a task
    over every token's tiles in series read their x serially (48 rows: a router
    CTA 9 us, 8 of them x)."""
    s, tid, red, lane, wave = c["S"], c["tid"], c["red"], c["lane"], c["wave"]
    a = c["args"]
    experts = c["rs"].experts
    groups, per = c["rs"].router_groups
    per_group = experts // ROWS * ROUTER_PARTS
    tile, part = task % per_group // ROUTER_PARTS, task % ROUTER_PARTS
    g0 = 0 if groups == 1 else task // per_group * per
    # the group's token tiles: (first token, tokens), tokens traced past one group
    tiles = [
        (plus(g0, j0), min(TILE, s - j0) if groups == 1 else fx.min(
            fx.Int32(min(TILE, per - j0)), s - plus(g0, j0)))
        for j0 in range(0, per, TILE)
    ]  # fmt: skip
    xs = [router_x_loads(a["x"], part, n, lane, wave, c["x_cm"], t0) for t0, n in tiles]
    for (t0, n), xvs in zip(tiles, xs):
        router_tile_mfma(wvs, xvs, lane, wave, red)
        gpu.barrier()
        if tid < ROWS * n:
            r = tid % ROWS
            tl = tid // ROWS
            v = row_sum(red, r, tl)
            c["put"](
                c["logit"],
                (plus(t0, tl) * experts + tile * ROWS + r) * ROUTER_PARTS + part,
                v,
            )
        gpu.barrier()
@traced
def load_x8_rows(c, words_mb, codes_mb, xl, xsl, token_of, live):
    """MXFP8 rows ``token_of(j)``, j < ``live`` (<= TILE; traced or int), of
    scratch ``words_mb`` / ``codes_mb`` into LDS rows j of ``xl`` / ``xsl``:
    the first ``live`` lanes wait on their token's X8RDY, then plain 16-byte
    loads (half the bytes of the tagged pairs a poll would read)."""
    tid = c["tid"]
    if tid < TILE:
        c["poll"]([(c["x8rdy"], token_of(fx.min(tid, live - 1)), 1)])
    gpu.barrier()
    for mb, width, dst in ((words_mb, X8_WORDS, xl), (codes_mb, HIDDEN // 32, xsl)):
        quads = width // 4
        for i in range_constexpr(-(-TILE * quads // THREADS)):
            q = tid + THREADS * i
            row = q // quads
            if (q < TILE * quads) & (row < live):
                v = fx.Vector(
                    bo.buffer_load(
                        rsrc(c[mb].value),
                        token_of(row) * width + q % quads * 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                fx.ptr_store(v, dst + (row * lds_row(width) + q % quads * 4))
    gpu.barrier()
@traced
def smid_quant(c, task, t0, n):
    """Odd shared task ``task``: its 16 columns (LDS ``pair``) and the even
    task's (SMID) are a quantize_fp8 group of 32 (amax floored AMAX_FLOOR, the
    ceil code) -> SMQ / SMC, a thread a token. Once here, not in each of the 256 CTAs
    whose down stage reads them: that took a down CTA 14 us at 48 rows."""
    tid, sh = c["tid"], c["d"].sh_inter
    assert c["d"].shared_tasks % 2 == 0  # tasks pair into groups
    if tid < n:
        t = plus(t0, tid)
        got = c["poll"](
            [(c["smid"], t * sh + (task - 1) * ROWS + r, 1) for r in range(ROWS)]
        )
        vals = [got[r][0].bitcast(fx.Float32) for r in range(ROWS)] + [
            fx.ptr_load(c["pair"] + (tid * ROWS + r)) for r in range(ROWS)
        ]
        amax = fx.Float32(AMAX_FLOOR)
        for y in vals:
            amax = fx.max(amax, abs(y))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        inv = rcp_pow2(code)
        g = task // 2
        for w in range_constexpr(8):
            word = fp8_pack4(*[clamp_fp8(vals[4 * w + i] * inv) for i in range(4)])
            c["put_words"](c["smq"], t * (sh // 4) + g * 8 + w, [word])
        c["put_words"](c["smc"], t * (sh // 32) + g, [code])
@traced
def shared_unit(c, unit, multi):
    """Shared unit ``unit`` (task, token tile): its gate weights in flight
    before x's wait, the tile's rows, ``stage_shared``."""
    s, shared_tasks, a = c["S"], c["d"].shared_tasks, c["args"]
    gops = gemv_fp8_loads(
        c, a["sgu"], a["sgu_s"], HIDDEN, plus(c["sh_gate0"], unit % shared_tasks)
    )
    if const_expr(not multi):
        load_x8_rows(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: j, s)
        stage_shared(c, unit, gops, 0, s)
    else:
        t0 = unit // shared_tasks * TILE
        n = fx.min(fx.Int32(TILE), s - t0)
        load_x8_rows(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: t0 + j, n)
        stage_shared(c, unit % shared_tasks, gops, t0, n)
@traced
def stage_route(c, token):
    """Mother six experts/lane, sequential top6 and eight tagged copies."""
    lane, wave, rs = c["lane"], c["wave"], c["rs"]
    assert rs.experts in (128,384) and rs.topk in (3,6) and not rs.dpa
    ept=rs.experts//64
    if wave == 0:
        # Both logical lane halves load bias before waiting for LOGIT.
        bias = [[ld_f32(c["args"]["bias"],lane+32*h+64*i)
                 for i in range(ept)] for h in range(2)]
        got = c["poll"]([(c["logit"],(token*rs.experts+lane+32*h+64*i)*4+p,2)
                         for h in range(2) for i in range(ept) for p in (0,2)],batch=24)
        vals, orig, ids = [], [], []
        for h in range_constexpr(2):
            vh, oh, ih = [], [], []
            for i in range_constexpr(ept):
                parts = got[h*(2*ept)+2*i]+got[h*(2*ept)+2*i+1]
                total = fx.Float32(0.)
                for p in range_constexpr(4):
                    total = total+parts[p].bitcast(fx.Float32)
                score = _score(bf16_round(total))
                vh.append(fx.max(score+bias[h][i],fx.Float32(float("-inf"))))
                oh.append(fx.Float32(fx.isnan(score).select(fx.Float32(0.),score)))
                ih.append(fx.Int32(lane+32*h+64*i))
            for p,q in rs.sort_net:
                swap = vh[p]<vh[q]
                vh[p],vh[q] = swap.select(vh[q],vh[p]),swap.select(vh[p],vh[q])
                oh[p],oh[q] = swap.select(oh[q],oh[p]),swap.select(oh[p],oh[q])
                ih[p],ih[q] = swap.select(ih[q],ih[p]),swap.select(ih[p],ih[q])
            vals.append(vh);orig.append(oh);ids.append(ih)
        cursor = [fx.Int32(0),fx.Int32(0)]
        total, my_w, my_id = fx.Float32(0.),fx.Float32(0.),fx.Int32(0)
        for k in range_constexpr(rs.topk):
            cv,ci,co = [],[],[]
            for h in range_constexpr(2):
                v, eid, weight = fx.Float32(float("-inf")),fx.Int32(0),fx.Float32(0.)
                for i in range_constexpr(ept):
                    at = cursor[h]==i
                    v,eid,weight = at.select(vals[h][i],v),at.select(ids[h][i],eid),at.select(orig[h][i],weight)
                cv.append(v);ci.append(eid);co.append(weight)
            # Original butterfly (32,16,8,4,2,1), with xor32 in registers.
            best = fx.max(cv[0],cv[1])
            for offset in (16,8,4,2,1):
                best = fx.max(best,best.shuffle_xor(offset,32))
            ball = dual_ballot(cv[0]==best,cv[1]==best)
            win = fx.Int32((ball==fx.Int64(0)).select(fx.Int32(0),fx.Int32(fmath.cttz(ball))))
            winner0, winner1 = readlane(ci[0],win&31),readlane(ci[1],win&31)
            win_id = (win<32).select(winner0,winner1)
            mine = []
            for h in range_constexpr(2):
                won = (cursor[h]<ept)&(ci[h]==win_id)
                mine.append(fx.Float32(won.select(co[h],fx.Float32(0.))))
                cursor[h] = cursor[h]+won.select(1,0)
            w0,w1 = readlane(mine[0],win_id&31),readlane(mine[1],win_id&31)
            weight = ((win_id&63)<32).select(w0,w1)
            total = total+weight
            my_w = (lane==k).select(weight,my_w)
            my_id = (lane==k).select(win_id,my_id)
        scale = fx.Float32(SCALE)/fx.max(total,fx.Float32(1e-20))
        if lane < rs.topk:
            for copy in range_constexpr(ROUTE_COPIES):
                at = (copy*c["S"]+token)*(2*rs.topk)+lane
                c["put"](c["route"],at,my_id)
                c["put"](c["route"],at+rs.topk,my_w*scale)
    gpu.barrier()
@traced
def stage_shared(c, task, gops, t0, n):
    """Gate rows 16 task .. (weights ``gops``), up rows 576 + 16 task .. against
    tokens t0 .. t0 + n (their x in LDS; ``n`` traced: a tile a CTA picks at run
    time) -> bf16 -> silu x up -> SMID; an odd task then quantizes its group of
    32 (``smid_quant``)."""
    tid, red, d = c["tid"], c["red"], c["d"]
    a = c["args"]
    matrix=dict(c,S=n)
    rows=n
    reference_fp8_matrices_rows(matrix,HIDDEN,gops,join=False)
    gpu.barrier()
    gate = fx.Float32(0.0)
    if tid < ROWS * n:
        gate = row_sum(red, tid % ROWS, tid // ROWS)
        if const_expr(c['shared_debug']):
            fx.ptr_store(gate.to(fx.BFloat16),c['shared_gu']+(t0+tid//ROWS)*(2*d.sh_inter)+task*ROWS+tid%ROWS)
    gpu.barrier()
    uops = gemv_fp8_loads(c, a["sgu"], a["sgu_s"], HIDDEN, plus(c["sh_up0"], task))
    reference_fp8_matrices_rows(matrix,HIDDEN,uops,join=False)
    gpu.barrier()
    if tid < ROWS * n:
        r = tid % ROWS
        t = plus(t0, tid // ROWS)
        up = row_sum(red, r, tid // ROWS)
        if const_expr(c['shared_debug']):
            fx.ptr_store(up.to(fx.BFloat16),c['shared_gu']+t*(2*d.sh_inter)+d.sh_inter+task*ROWS+r)
        v = fx.Float32(_silu_mul(bf16_round(gate), bf16_round(up)).to(fx.BFloat16))
        c["put"](c["smid"], t * d.sh_inter + task * ROWS + r, v)
        fx.ptr_store(v, c["pair"] + tid)
    gpu.barrier()
    if task % 2 == 1:
        smid_quant(c, task, t0, n)
    gpu.barrier()

@traced
def run_moe_front(root,key,bid):
    run_gather(root,key,bid)
    c=front_context(root,key,'xq')
    s,tid=c['S'],c['tid']
    router_tasks=c['rs'].router_tasks
    route0,shared0=c['rs'].route0,c['rs'].shared0
    c=front_context(root,key,'xq')
    enter_stage("k2b.xq")
    for t in range(first_task(bid, XQ0), s, BLOCKS):
        if const_expr(c["rs"].dpa):
            # another rank's token: quantized straight from its push
            xq_token_dpa(c, t)
        else:
            await_x(c, [t])
            stage_xq(c, t)
        publish(c["put"], c["x8rdy"], t, 1, tid == 0)
    multi = len(token_tiles(s)) > 1
    c=front_context(root,key,'router')
    enter_stage("k2b.router")
    # DP attention: the router and route on this rank's own rows alone (their
    # rows' x ready without the other ranks'), the routes pushed to every rank
    cr = c
    if const_expr(c["rs"].dpa):
        cr = own_view(c, ("x",))
        cr["route_t0"] = cr["t0"]
    rows = cr["S"]
    mine = [plus(cr.get("t0", 0), t) for t in range(rows)]
    for task in range(first_task(bid, c["rs"].router0), router_tasks, BLOCKS):
        if const_expr(rows >= ROUTER_EARLY_FROM):
            # the task's weights in flight through x's wait
            wvs = router_weights(cr, task)
            await_x(c, mine)
            stage_router(cr, task, wvs)
        else:
            await_x(c, mine)
            stage_router(cr, task, router_weights(cr, task))
    c=front_context(root,key,'route')
    enter_stage("k2b.route")
    cr=c
    for t in range(first_task(bid, route0), rows, BLOCKS):
        stage_route(cr, t)
    c=front_context(root,key,'shared')
    enter_stage("k2b.shared")
    c["xl"], c["xsl"] = c["x8l"], c["x8sl"]
    # a CTA's one shared unit (task, token tile) a round: straight-line code,
    # whose tile loads LLVM does not hoist into one live set as it would out of
    # a task loop (spills). A task's tiles in series on one CTA put three tiles'
    # latency on the path down waits on (the weights read again a tile: L2 hits).
    # Past BLOCKS units (DP attention's MoE over every rank's rows) a second round
    units = c["rs"].shared_units
    for k in range_constexpr(-(-units // BLOCKS)):
        unit = first_task(bid, shared0) + k * BLOCKS
        if unit < units:
            shared_unit(c, unit, multi)
