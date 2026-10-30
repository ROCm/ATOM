from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_dynamic_mhc import row_view,tile_prefetch,stage_tile
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_wo import reduce_attn_pair
# SPDX-License-Identifier: MIT
"""Image FFN seam; original prefetch/wait, own-row views and XG publication."""
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_dynamic_k1 import *
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_index import scalar_word,late_pointer
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_dspark_wo import wo_context,stage_reduce
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_post import K2A_OWN
from atom.models.deepseek_v41.mono.kernels.token_par import TOKEN_ROW_BYTES,own_view,own_count
from atom.mono.device.ops import CM_DEV
from atom.mono.device.sync import Mailbox,preg
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import publish_plain as publish

SEAM_PTRS=('res_in','post_in','comb_in','pre_in','hc_fn','hc_scale','hc_base',
           'attn_w','res_out','post_out','comb_out','pre_out','normed')
THREADS=256
WAVES=8
_XG_QUADS=HIDDEN//4

def seam_context(c,stage):
    # Restore only the consuming stage's addresses, after WO has freed its registers.
    names={'slice':('res_in','post_in','comb_in','pre_in','hc_fn','res_out'),
           'norm':('attn_w','normed'),
           'gate':('hc_scale','hc_base','post_out','comb_out','pre_out'),
           'publish':('normed',)}[stage]
    out=wo_context(c,'reduce')
    tid=fresh(fx.thread_idx.x)
    out.update(tid=tid,lane=tid%64,wave=tid//64,threads=THREADS,
               fold=True,index=False,aux=False,ffn=True,normed_cm=CM_DEV)
    out['args']={name:late_pointer(c,192+2*SEAM_PTRS.index(name)) for name in names}
    for name in ('lin','pmix','xrdy'):
        out[name]=sreg(late_pointer(c,24),c['scratch_layout'][name][0],name)
    out['mhc_sq']=sreg(late_pointer(c,220),0,'mhc_sq')
    return out

def owned(c):
    return own_view(c,tuple(n for n in K2A_OWN if n in c['args'])) if c['tok_par'] else c

def x_context(c):
    out=seam_context(c,'publish')
    mb=Mailbox(scalar_word(fx.ptr_load(c['late_meta']+218)),gfx1250=True)
    out.update(put=mb.put,put_words=mb.put_words,poll=mb.poll,
               args=dict(x=out['args']['normed']))
    return out

def slice_fn_loads(c,task):
    return [ld_f32(c['args']['hc_fn'],(e//KT)*K_MIX+((e%KT)//COLS)*HIDDEN+task*COLS+e%COLS)
            for e in [c['tid']+THREADS*i for i in range(MIX*KT//THREADS)]]

def slice_prefetch(c,task):
    rounds=(c['S']*KT+THREADS-1)//THREADS
    rows=[slice_loads(c,task*COLS,fx.min(c['tid']+THREADS*i,c['S']*KT-1))
          for i in range(min(8,rounds))]
    return slice_fn_loads(c,task) if c['S']<=24 else None,rows
@traced
def stage_slice(c, task, pre=None):
    """Original batched residual loads, per-token reduction and N16 mix tiles."""
    s, tid = c['S'], c['tid']
    lane, wave = tid % 32, tid // 32
    fns,first_rows=(slice_fn_loads(c,task),None) if pre is None else pre
    rounds = (s * KT + THREADS - 1) // THREADS
    # Original batch=4 at 512 lanes -> batch=8 at 256 lanes.
    for begin in range_constexpr(0, rounds, 8):
        indices = range(begin, min(begin + 8, rounds))
        got = first_rows if begin==0 and first_rows is not None else [slice_loads(c, task * COLS, fx.min(tid + THREADS * i, s * KT - 1)) for i in indices]
        for i, value in zip(indices, got):
            at = tid + THREADS * i
            if at < s * KT:
                slice_row(c, task * COLS, at, value)
    if const_expr(fns is None):fns=slice_fn_loads(c,task)
    for i in range_constexpr(MIX * KT // THREADS):
        hi = bf16_round(fns[i])
        fx.ptr_store(hi + bf16_round(fns[i] - hi), c['fl'] + tid + THREADS * i)
    gpu.barrier()
    for begin in range_constexpr(0, s * COLS // 2, THREADS):
        slice_input(c, task, tid + begin)
    # Four logical wave64 groups per physical CTA; preserve xor32,16,..,1.
    for begin in range_constexpr(0, s, 4):
        token = begin + tid // 64
        if token < s:
            x0 = fx.ptr_load(c['rl'] + token * KT + 2 * (tid % 64))
            x1 = fx.ptr_load(c['rl'] + token * KT + 2 * (tid % 64) + 1)
            fx.ptr_store(x0 * x0 + x1 * x1, c['red'] + tid)
        gpu.barrier()
        if token < s:
            sq = fx.ptr_load(c['red'] + tid) + fx.ptr_load(c['red'] + (tid ^ 32))
            for off in (16, 8, 4, 2, 1):
                sq = sq + sq.shuffle_xor(off, 32)
            if tid % 64 == 0:
                c['put'](c['pmix'], (task * s + token) * PART + MIX, sq)
        gpu.barrier()
    # Eight K16 partials from the accepted FP32 instruction translation.
    # Red is reused for each original N16 tile, including the final short tile.
    for t0, n in token_tiles(s):
        if lane < MIX:
            for tl in range_constexpr(n):
                accum = fx.Float32(0.0)
                for k in range_constexpr(KT // WAVES):
                    col = wave * (KT // WAVES) + k
                    accum = fx.Float32(fmath.fma(fx.ptr_load(c['fl'] + lane * KT + col),
                                      fx.ptr_load(c['rl'] + (t0 + tl) * KT + col), accum))
                fx.ptr_store(accum, c['red'] + (wave * n + tl) * 32 + lane)
        gpu.barrier()
        for begin in range_constexpr(0, MIX * n, THREADS):
            at = tid + begin
            if at < MIX * n:
                tl, row = at // MIX, at % MIX
                total = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    total = total + fx.ptr_load(c['red'] + (w * n + tl) * 32 + row)
                c['put'](c['pmix'], (task * s + t0 + tl) * PART + row, total)
        gpu.barrier()
@traced
def publish_x(ca, cb, t):
    """Token t's ``normed`` row, written at device scope by this CTA's norm, is
    readable by every CTA: the stores drained, then its flag."""
    publish(cb["put"], cb["xrdy"], t, 1, ca["tid"] == 0)
@traced
def publish_x_tok_par(ca, cb, t):
    """Token t of this rank's (its row ``rank n + t``): ``push_x``."""
    tg = cb["rank"] * own_count(ca["S"], ca["d"].tp) + t
    push_x(ca, cb, tg, tg)
@traced
def push_x(ca, cb, tg, row):
    """MoE row ``tg``, K2a's ``normed`` row ``row``: the flag here, as
    ``publish_x``, then the row pushed to every other rank's XG."""
    tid, rank = ca["tid"], cb["rank"]
    publish(cb["put"], cb["xrdy"], tg, 1, tid == 0)
    xg = cb["peer"]["xg"]
    for k in range_constexpr(-(-_XG_QUADS // THREADS)):
        q = tid + THREADS * k
        if q < _XG_QUADS:
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(ca["args"]["normed"]),
                    row * (HIDDEN // 2) + 2 * q,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            for p in range_constexpr(cb["d"].ranks):
                if rank != p:
                    dst = preg(cb["peer_addr"](p), xg, "xg")
                    cb["put_words"](dst, tg * (HIDDEN // 2) + 2 * q, [w[0], w[1]])
@traced
def gather_x(cb, t):
    """Token t, another rank's: its row from XG -> ``normed`` here (device
    scope), then its flag (the router's wait)."""
    tid = cb["tid"]
    own = preg(cb["sym"], cb["peer"]["xg"], "xg")
    for k in range_constexpr(-(-_XG_QUADS // THREADS)):
        q = tid + THREADS * k
        if q < _XG_QUADS:
            (w,) = cb["poll"]([(own, t * (HIDDEN // 2) + 2 * q, 2)])
            bo.buffer_store(
                fx.Vector.from_elements([w[0], w[1]], fx.Int32),
                rsrc(cb["args"]["x"]),
                t * (HIDDEN // 2) + 2 * q,
                cache_modifier=CM_DEV,
            )
    publish(cb["put"], cb["xrdy"], t, 1, tid == 0)

@traced
def run_seam(c,key,bid):
    co=owned(seam_context(c,'slice'))
    enter_stage('k2a.slice')
    for unit in range(first_task(bid,0),SLICES*((key.tokens+15)//16),BLOCKS):
        task,rowbase=unit%SLICES,(unit//SLICES)*16
        local=row_view(co,rowbase)
        pre=tile_prefetch(local,task)
        # Preserve global peer layout and masked tail; stage only this tile's 16 rows.
        reduce_attn_pair(co,task,co['tid']+rowbase*16)
        gpu.barrier()
        stage_tile(local,task,pre)
    # Callback deliberately uses the full S and unshifted normed pointer.
    co=owned(seam_context(c,'norm'))
    enter_stage('k2a.norm')
    for t in range(first_task(bid,NORM0),co['S'],BLOCKS):
        stage_norm(co,t)
        ca=seam_context(c,'publish')
        cb=x_context(c)
        if const_expr(key.tok_par):publish_x_tok_par(ca,cb,t)
        else:publish_x(ca,cb,t)

@traced
def run_gather(c,key,bid):
    if const_expr(key.tok_par):
        cb=x_context(c)
        own=own_count(c['S'],4)
        for t in range(first_task(bid,0),c['S'],BLOCKS):
            if t//own!=cb['rank']:gather_x(cb,t)

@traced
def run_seam_gates(c,key,bid):
    # Final layer_post calls this only as run_moe(before_ug=...).
    co=owned(seam_context(c,'gate'))
    enter_stage('k2a.gate')
    for t in range(first_task(bid,GATE0),co['S'],BLOCKS):
        stage_gate(co,t)
