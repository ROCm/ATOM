"""Complete 16-row mHC producer, including residual, collapse and both sums.

The private candidate uses the disabled timeline argument as a separate tagged
48*160-cell sq_x mailbox. The host must clear it with step_begin every step.
"""
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import *

THREADS = 256
ROWS_MHC = 16


def row_view(c, rowbase):
    assert c['S'] in (6,12,18,24,30,36,42) and not c['tok_par']
    args = dict(c['args'])
    strides = dict(res_in=4*5120*2, res_out=4*5120*2,
                   pend=5120*2, aux=5120*2, pre_in=4*4,
                   post_in=4*4, comb_in=4*4*4)
    for name, stride in strides.items():
        if name in args:
            args[name] = args[name] + fx.Int64(rowbase) * stride
    out = dict(c, S=16, args=args, rowbase=rowbase, global_s=c['S'], valid=fx.min(fx.Int32(16), c['S']-rowbase))
    if 'pend_lds' in c:
        out['pend_lds'] = c['pend_lds'] + rowbase * COLS
    return out


def tile_prefetch(c, task):
    # A full M16 tile: exactly eight rounds of settled residual loads.
    fns = [ld_f32(c['args']['hc_fn'], (e // KT) * K_MIX +
                  ((e % KT) // COLS) * HIDDEN + task * COLS + e % COLS)
           for e in [c['tid'] + THREADS*i for i in range(MIX*KT//THREADS)]]
    rows = [slice_loads(c, task*COLS, fx.min((c['tid']+THREADS*i)//KT,c['valid']-1)*KT+(c['tid']+THREADS*i)%KT) for i in range(8)]
    return fns, rows


@traced
def stage_tile(c, task, pre=None):
    assert c['S'] == 16
    tid, rb = c['tid'], c['rowbase']
    lane, wave = tid % 32, tid // 32
    fns, rows = tile_prefetch(c, task) if pre is None else pre
    for i in range_constexpr(8):
        if tid+THREADS*i < c['valid']*KT:
            slice_row(c, task*COLS, tid+THREADS*i, rows[i])
    for i in range_constexpr(MIX*KT//THREADS):
        # Keep raw FP32 weights; the matrix operands split them once below.
        fx.ptr_store(fns[i], c['fl']+tid+THREADS*i)
    gpu.barrier()

    # Sixteen lanes per row collapse all 32 columns, then publish the BF16
    # input and its square sum together. Each pair and partial has one owner.
    token, cc = tid // 16, 2*(tid % 16)
    safe_token = fx.min(token,c['valid']-1)
    outs = []
    for d in range_constexpr(2):
        acc = fx.Float32(0.0)
        for h in range_constexpr(HC):
            x = fx.ptr_load(c['rl']+safe_token*KT+h*COLS+cc+d)
            acc = acc + ld_f32(c['args']['pre_in'], safe_token*HC+h)*x
        outs.append(bf16_round(acc))
    if token < c['valid']:
        c['put_bf'](c['lin'], (rb+token)*HIDDEN+task*COLS+cc, outs)
    sqx = outs[0]*outs[0]+outs[1]*outs[1]
    for off in (1, 2, 4, 8):
        sqx = sqx + sqx.shuffle_xor(off, 32)
    if (tid % 16 == 0) & (token < c['valid']):
        c['put'](c['mhc_sq'], (rb+token)*SLICES+task, sqx)
    if const_expr(c['aux']):
        for d in range_constexpr(2):
            acc = fx.Float32(0.0)
            for h in range_constexpr(HC):
                acc = acc + fx.ptr_load(c['rl']+safe_token*KT+h*COLS+cc+d)
            if token < c['valid']:
                bo.buffer_store((acc*fx.Float32(1.0/HC)).to(fx.BFloat16),
                                rsrc(c['args']['aux']), token*HIDDEN+task*COLS+cc+d)

    # Each physical wave owns a row; two rounds cover M16 without LDS
    # reduction or barriers between rows.
    for begin in range_constexpr(0, 16, 8):
        token = begin+wave
        vals = [fx.ptr_load(c['rl']+fx.min(token,c['valid']-1)*KT+4*lane+j) for j in range(4)]
        sq = vals[0]*vals[0]+vals[1]*vals[1]
        sq = sq + (vals[2]*vals[2]+vals[3]*vals[3])
        for off in (16, 8, 4, 2, 1):
            sq = sq + sq.shuffle_xor(off, 32)
        if (lane == 0) & (token < c['valid']):
            c['put'](c['pmix'], (task*c['global_s']+rb+token)*PART+MIX, sq)

    # R_new is already BF16. Only the FP32 weights need hi/lo decomposition.
    # Two waves own M16 x N24, each with the complete K128 accumulation.
    if wave < 2:
        half, token = lane//16, lane%16
        row = fx.min(wave*16+lane%16, 23)
        atom = fx.make_mma_atom(fx.rocdl.WMMA(16,16,32,fx.BFloat16,fx.Float32))
        acc = fx.make_rmem_tensor(8,fx.Float32)
        acc.store(fx.full(8,0.0,fx.Float32))
        a,b = fx.make_rmem_tensor(16,fx.BFloat16),fx.make_rmem_tensor(16,fx.BFloat16)
        for k0 in range_constexpr(0,128,32):
            xs = [fx.ptr_load(c['rl']+fx.min(token,c['valid']-1)*KT+k0+half*16+k).to(fx.BFloat16) for k in range(16)]
            ws = [fx.ptr_load(c['fl']+row*KT+k0+half*16+k) for k in range(16)]
            hi = [v.to(fx.BFloat16) for v in ws]
            lo = [(v-h.to(fx.Float32)).to(fx.BFloat16) for v,h in zip(ws,hi)]
            b.store(fx.Vector.from_elements(xs,fx.BFloat16))
            for vals in (lo,hi):
                a.store(fx.Vector.from_elements(vals,fx.BFloat16))
                fx.gemm(atom,acc,a,b,acc)
        values = acc.load()
        for j in range_constexpr(8):
            output = wave*16+half*8+j
            if (output < MIX) & (token < c['valid']):
                c['put'](c['pmix'], (task*c['global_s']+rb+token)*PART+output, values[j])
    gpu.barrier()


@traced
def norm_square_total(c, t):
    # Eight waves independently consume the 160 small partials. This avoids
    # recomputing 5120 squares and the old cross-wave LDS reduction barriers.
    lane = c['tid'] % 32
    words = c['poll']([(c['mhc_sq'], t*SLICES+lane+32*i, 1) for i in range(5)])
    total = fx.Float32(0.0)
    for i in range_constexpr(5):
        total = total + words[i][0].bitcast(fx.Float32)
    for off in (1,2,4,8,16):
        total = total + total.shuffle_xor(off,32)
    return total
