# SPDX-License-Identifier: MIT
"""TP4 DSpark output projections; image task/prefetch/publish order retained."""
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_post import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_index import scalar_word,late_pointer
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import _weight_step
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_multirow_projection import reference_fp8_matrices_rows
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import scalar_address
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_k1 import gemv_fp8_loads
from atom.mono.device.ops import fresh,rsrc
from atom.mono.device.sync import sreg,preg
THREADS=256
WAVES=8
WO_PTRS=('woa','woa_s','wob','wob_s')

def wo_context(c,stage):
    out=dict(c)
    tid=fresh(fx.thread_idx.x)
    out.update(tid=tid,lane=tid%32,wave=tid//32)
    names=('woa','woa_s') if stage=='woa' else ('wob','wob_s') if stage=='wob' else ()
    out['args']={name:late_pointer(c,160+2*WO_PTRS.index(name)) for name in names}
    scratch=late_pointer(c,24)
    for name in ('xo','xos','x8b','x8bs'):
        off,_=c['scratch_layout'][name];out[name]=sreg(scratch,off,name)
    out['sym']=late_pointer(c,168)
    out['rank']=scalar_word(fx.ptr_load(c['late_meta']+178))
    out['peer_addr']=lambda p:late_pointer(c,170+2*p)
    return out
@traced
def woa_loads(c, task):
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
def woa_matrices_rows(c,ops):
    lane,wave,inner=c['lane'],c['wave'],c['d'].group_k
    half=lane//16;token=fx.min(lane%16,c['S']-1)
    xl=c['xl']+token*lds_row(inner//4)
    xsl=c['xsl']+token*lds_row(inner//32)
    split,per=WOA_SPLIT_K,inner//128//WOA_SPLIT_K
    atom=fx.make_mma_atom(fx.rocdl.WMMAScale(16,16,128,fx.Float8E4M3FN,fx.Float8E4M3FN,fx.Float32))
    accum=fx.make_rmem_tensor(8,fx.Float32);accum.store(fx.full(8,0.,fx.Float32))
    for i in range_constexpr(per):
        st=(wave%split)*per+i
        chunks=[fx.Vector(fx.ptr_load(xl+st*32+half*4+j*8,result_type=fx.Vector.make_type(4,fx.Int32))) for j in range(4)]
        live=lane%16<c['S']
        xv=fx.Vector.from_elements([live.select(chunks[p][j],fx.Int32(0)) for p in range(4) for j in range(4)],fx.Int32)
        codes=[fx.ptr_load(xsl+st*4+j) for j in range(4)]
        sx=live.select(codes[0]|(codes[1]<<8)|(codes[2]<<16)|(codes[3]<<24),fx.Int32(0x7f7f7f7f))
        weight,sw=ops[i]
        a,b=fx.make_rmem_tensor(16,fx.Int32),fx.make_rmem_tensor(16,fx.Int32)
        a.store(weight);b.store(xv)
        fx.gemm(atom,accum,a,b,accum,scale_a=sw,scale_b=sx)
    if lane%16<c['S']:
        values=accum.load()
        for j in range_constexpr(8):
            row=half*8+j
            fx.ptr_store(values[j],c['red']+(wave*64+16*(row//4)+lane%16)*4+row%4)
    gpu.barrier()
@traced
def load_rows(
    c, words_mb, codes_mb, stride_w, off_w, words, stride_g, off_g, groups, xl, xsl,
    t0=0, n=None,
):  # fmt: skip
    """Tokens t0 .. t0 + n's (every token's by default) slice ``[off, off +
    words)`` of a mailbox of rows ``stride`` words apart (and its codes), into
    LDS rows at the ``lds_row`` stride. ``n`` traced (a tile a CTA picks at run
    time): a TILE-row copy of which only n rows are polled."""
    n = c["S"] if n is None else n
    rows = n if isinstance(n, int) else TILE
    poll_copy(
        c,
        c["tid"],
        THREADS,
        [
            (
                words_mb,
                lambda u: plus(t0, u // words) * stride_w + off_w + u % words,
                rows * words,
                xl,
                words,
            ),
            (
                codes_mb,
                lambda u: plus(t0, u // groups) * stride_g + off_g + u % groups,
                rows * groups,
                xsl,
                groups,
            ),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
    )
    gpu.barrier()
@traced
def woa_tile(c, task, ops, t0, n):
    """wo_a task ``task`` (``woa_loads``' ``ops``) on tokens t0 .. t0 + n: its
    group's attention output rows into LDS, then ``stage_woa``."""
    d = c["d"]
    g = task // (d.woa_tasks // d.groups)
    load_rows(
        c, c["xo"], c["xos"], d.xo_words, g * d.group_k // 4, d.group_k // 4,
        d.heads * HEAD_DIM // 32, g * d.group_k // 32, d.group_k // 32,
        c["xl"], c["xsl"], t0, n,
    )  # fmt: skip
    stage_woa(c, task, ops, t0, n)
def load_wob_rows(c, part, t0, n):
    """wo_b's input over K part ``part`` (X8B / X8BS) of tokens t0 .. t0 + n
    into LDS ``xl`` / ``xsl``."""
    d = c["d"]
    load_rows(
        c, c["x8b"], c["x8bs"], d.yb_words, part * d.wob_k // 4, d.wob_k // 4,
        d.o_rows // 32, part * d.wob_k // 32, d.wob_k // 32, c["xl"], c["xsl"], t0, n,
    )  # fmt: skip
@traced
def wob_push(c, task, t0=0, n=None, part=0):
    """``stage_wob``'s row sums of ``red`` pushed as source ``rank parts +
    part``."""
    s, tid, red, d = c["S"], c["tid"], c["red"], c["d"]
    n = s if n is None else n
    src = c["rank"] * d.wob_parts + part if d.wob_parts > 1 else c["rank"]
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        col = task * ROWS + 2 * rp
        v0 = row_sum(red, 2 * rp, tl)
        v1 = row_sum(red, 2 * rp + 1, tl)
        if const_expr(c["tok_par"]):
            # to the token's owner only, at its own token index there
            own_n = own_count(s, d.tp)
            owner = t // own_n
            for p in range_constexpr(d.tp):
                if owner == p:
                    dst = preg(c["peer_addr"](p), c["peer"]["attn"], "attn")
                    pair = attn_region_pair(src, t - p * own_n, own_n, col)
                    c["put_bf"](dst, 2 * pair, [v0, v1])
        else:
            for p in range_constexpr(d.tp):
                dst = preg(c["peer_addr"](p), c["peer"]["attn"], "attn")
                c["put_bf"](dst, 2 * attn_region_pair(src, t, s, col), [v0, v1])
    gpu.barrier()
def stage_wob(c, task, ops, t0=0, n=None, part=0):
    """Rows 16 task .. of wo_b over K part ``part`` (``wob_loads``' operands)
    -> this rank's bf16 partial of the part, pushed to every rank (token
    parallel: to the token's owner) as source ``rank parts + part``; the token
    tile from ``t0`` (its x in LDS)."""
    wob_mfma(c, ops, n)
    wob_push(c, task, t0, n, part)
@traced
def stage_reduce(c, task):
    """The attention output at this slice's 32 columns: the TP ranks' partials
    in the all-reduce's order (``sum_partials``), in fp32 -> bf16 ->
    ``pend_lds``."""
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * k1.COLS // 2, THREADS):
        reduce_attn_pair(c, task, plus(pas, tid))
    gpu.barrier()
@traced
def reduce_attn_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 16 (``stage_reduce``)."""
    s = c["S"]
    c0 = task * k1.COLS
    if idx < s * k1.COLS // 2:
        t = idx // (k1.COLS // 2)
        col = c0 + 2 * (idx % (k1.COLS // 2))
        own = preg(c["sym"], c["peer"]["attn"], "attn")
        acc0, acc1 = sum_partials(
            c["poll"],
            own,
            lambda src: attn_region_pair(src, t, s, col),
            c["d"].attn_sources,
        )
        lds = c["pend_lds"] + (t * k1.COLS + col - c0)
        fx.ptr_store(bf16_round(acc0), lds)
        fx.ptr_store(bf16_round(acc1), lds + 1)

@traced
def stage_woa(c,task,ops,t0=0,n=None):
    n=c['S'] if n is None else n
    woa_matrices_rows(dict(c,S=n),ops)
    # Mother 512 logical lanes: two 256-lane output rounds for an N16 tile.
    for begin in range_constexpr(0,tile_rows(c['S'])*WOA_ROWS,THREADS):
        tid=c['tid']+begin
        row=tid%WOA_ROWS
        tile=row//ROWS
        tl=fx.min(tid//WOA_ROWS,n-1)
        t=plus(t0,tl)
        y=bf16_round(row_sum(c['red'],row%ROWS,tl,
                    [tile*WOA_SPLIT_K+w for w in range(WOA_SPLIT_K)]))
        amax=fx.max(butterfly(abs(y),(1,2,4,8,16),fx.max),fx.Float32(AMAX_FLOOR))
        code=code_ceil(amax*fx.Float32(1./FP8_MAX))
        q=clamp_fp8(y/pow2(code))
        lane=c['lane']
        word=fp8_pack4(q,*[lane_gather(q.bitcast(fx.Int32),fx.min(lane+k,31)).bitcast(fx.Float32) for k in (1,2,3)])
        if tid<WOA_ROWS*n:
            if const_expr(c['debug']):fx.ptr_store(y.to(fx.BFloat16),c['debug_woa']+t*c['d'].o_rows+task*WOA_ROWS+row)
            if row%4==0:c['put_words'](c['x8b'],t*c['d'].yb_words+(task*WOA_ROWS+row)//4,[word])
            if row==0:c['put'](c['x8bs'],t*(c['d'].o_rows//32)+task,code)
    gpu.barrier()

def wob_loads(c,task,part=0):
    assert c['d'].wob_parts==1 and part==0
    return gemv_fp8_loads(c,c['args']['wob'],c['args']['wob_s'],c['d'].o_rows,task)

def wob_mfma(c,ops,n=None):
    n=c['S'] if n is None else n
    reference_fp8_matrices_rows(dict(c,S=n),c['d'].wob_k,ops)

@traced
def run_wo(c,key,bid):
    s,d,tid=key.tokens,c['d'],c['tid']
    c=wo_context(c,'woa')
    enter_stage("k2a.wo_a")
    # a CTA's wo_a task (at most one): its weights in flight before the
    # attention output's wait
    assert d.woa_tasks <= BLOCKS
    tiles = token_tiles(s)
    if const_expr(len(tiles) == 1):
        woa_task = first_task(bid, WOA0)
        if woa_task < d.woa_tasks:
            woa_ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, woa_ops, 0, s)
    elif const_expr(d.woa_tasks == BLOCKS):
        # TP 1 (DP attention): a task a CTA fills every CTA already, so a CTA
        # runs its task's token tiles in series on its weights, loaded once
        # (EP 24 / 48 rows -3.4 / -5.1 us)
        woa_task = first_task(bid, WOA0)
        woa_ops = woa_loads(c, woa_task)
        for t0, n in tiles:
            woa_tile(c, woa_task, woa_ops, t0, n)
    else:
        # a (task, token tile) a unit, a CTA each: a task's tiles in series on
        # one CTA left three quarters of the CTAs idle through wo_a (its weights
        # read again a tile: L2 hits)
        for unit in range(first_task(bid, WOA0), d.woa_tasks * len(tiles), BLOCKS):
            woa_task = unit % d.woa_tasks
            t0 = unit // d.woa_tasks * TILE
            woa_ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, woa_ops, t0, fx.min(fx.Int32(TILE), s - t0))
    c=wo_context(c,'wob')
    enter_stage("k2a.wo_b")
    assert d.wob_k // 4 <= d.group_k // 4  # xl holds a part
    assert BLOCKS < WOB_TASKS <= 2 * BLOCKS
    if const_expr(len(tiles) == 1 and d.wob_parts == 2):
        # TP 1 (DP attention): wo_a takes every CTA, so wo_b's weights wait for
        # it; a CTA a K part of 2 or 3 tasks (half the CTAs a part), every
        # task's weights in flight at once and the part's rows loaded once: one
        # HBM round, not a part's after the other's
        unit = first_task(bid, WOA0 + d.woa_tasks)
        part, slot = unit % 2, unit // 2
        per = BLOCKS // 2
        tasks = [plus(slot, per * k) for k in range(-(-WOB_TASKS // per))]
        ops = [wob_loads(c, fx.min(t, WOB_TASKS - 1), part) for t in tasks]
        load_wob_rows(c, part, 0, s)
        for k in range_constexpr(len(tasks)):
            if tasks[k] < WOB_TASKS:
                stage_wob(c, tasks[k], ops[k], 0, s, part)
    elif const_expr(len(tiles) == 1):
        # every CTA has one or two wo_b tasks: both tasks' weights in flight
        # before y's wait (a CTA with one loads its own again: L2 hits); a
        # later K part's after the part before it
        wob_task = first_task(bid, WOA0 + d.woa_tasks)
        wob_second = wob_task + BLOCKS
        has_second = wob_second < WOB_TASKS
        second = has_second.select(wob_second, wob_task)
        wob_ops = wob_loads(c, wob_task)
        wob_ops2 = wob_loads(c, second)
        for part in range_constexpr(d.wob_parts):
            if const_expr(part > 0):
                wob_ops = wob_loads(c, wob_task, part)
                wob_ops2 = wob_loads(c, second, part)
            load_wob_rows(c, part, 0, s)
            stage_wob(c, wob_task, wob_ops, 0, s, part)
            if has_second:
                stage_wob(c, wob_second, wob_ops2, 0, s, part)
    elif const_expr(d.wob_parts == 2):
        # TP 1 past one tile: a CTA a (tile, K part) and its tasks, as above
        vt, wob_tasks = tile_tasks(
            bid, WOA0 + d.woa_tasks * len(tiles), 2 * len(tiles), WOB_TASKS
        )
        part, tile = vt % 2, vt // 2
        t0 = tile * TILE
        n = fx.min(fx.Int32(TILE), s - t0)

        def loads(k):
            return wob_loads(c, fx.min(wob_tasks[k], WOB_TASKS - 1), part)

        # WOB_LIVE tasks' weights in flight, the next issued as a GEMV frees
        # its registers (all of a CTA's five at once spilled)
        live = [loads(k) for k in range(min(WOB_LIVE, len(wob_tasks)))]
        load_wob_rows(c, part, t0, n)
        for k in range_constexpr(len(wob_tasks)):
            ops = live[k % WOB_LIVE]
            wob_mfma(c, ops, n)
            if const_expr(k + WOB_LIVE < len(wob_tasks)):
                live[k % WOB_LIVE] = loads(k + WOB_LIVE)
            if wob_tasks[k] < WOB_TASKS:
                wob_push(c, wob_tasks[k], t0, n, part)
    else:
        tile, wob_tasks = tile_tasks(
            bid, WOA0 + d.woa_tasks * len(tiles), len(tiles), WOB_TASKS
        )
        t0 = tile * TILE
        n = fx.min(fx.Int32(TILE), s - t0)
        for part in range_constexpr(d.wob_parts):
            wob_ops = [wob_loads(c, fx.min(t, WOB_TASKS - 1), part) for t in wob_tasks]
            load_wob_rows(c, part, t0, n)
            for k in range_constexpr(len(wob_tasks)):
                if wob_tasks[k] < WOB_TASKS:
                    stage_wob(c, wob_tasks[k], wob_ops[k], t0, n, part)
