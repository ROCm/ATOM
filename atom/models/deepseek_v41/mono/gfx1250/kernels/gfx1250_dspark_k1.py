# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 DSpark K1, image stage/ABI/tiles with gfx1250 instruction translation.

External trial: no runner gate is enabled by this module.
"""
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import *
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import _CURRENT_STREAM
from atom.models.deepseek_v41.mono.gfx950.kernels.attn_pre import wqb_rope as image_wqb_rope, iq_rope as image_iq_rope
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_projection import reference_fp8_loads
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_multirow_projection import reference_fp8_matrices_rows
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_gate import stage_gate_c1
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_index_query import wv_reduce_c1, stage_iquant_c1
from atom.models.deepseek_v41.mono.gfx950.kernels.index_query import AMAX_SEED, IDX_WEIGHTS_SCALE
from atom.mono.device.sync import Mailbox, Addr
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_sync import scalar_address
from atom.mono.device.ops import CM_NT_GFX1250, bf2_f32, row_shr, lane_gather
from flydsl.expr.typing import T

THREADS = 256
# Eight K streams are unchanged; physical waves are 32 lanes.
WAVES = 8
def vector_address(ptr):
    """Accepted address-lifetime translation; value/space do not change."""
    low, high = fresh(fx.Int32(ptr)), fresh(fx.Int32(ptr >> 32))
    return (fx.Int64(low) & fx.Int64(0xffffffff)) | (fx.Int64(high) << 32)


def scalar_region(region):
    return Addr(region.region, region.space, scalar_address(region.value))


def peer_stage(c, names):
    view = dict(c)
    for name in names:
        if name in c:
            view[name] = [tuple(scalar_region(x) for x in a) if isinstance(a, tuple)
                          else scalar_region(a) for a in c[name]]
    return view


def argument_stage(c, names):
    args = dict(c['args'])
    for name in names:
        args[name] = scalar_address(args[name])
    return dict(c, args=args)


@traced
def wqb_rope(c, task, t0, n):
    return image_wqb_rope(argument_stage(c, ('cos', 'sin', 'pos')), task, t0, n)


@traced
def iq_rope(c, task, t0, n):
    return image_iq_rope(argument_stage(c, ('cos', 'sin', 'pos')), task, t0, n)


@traced
def stage_slice(c, task, pre=None):
    """Original batched residual loads, per-token reduction and N16 mix tiles."""
    assert pre is None  # K1 has no preceding peer reduction.
    s, tid = c['S'], c['tid']
    lane, wave = tid % 32, tid // 32
    # Each original logical CTA's 3072 weights, all issued before residuals.
    fns = [ld_f32(c['args']['hc_fn'], (e // KT) * K_MIX +
                  ((e % KT) // COLS) * HIDDEN + task * COLS + e % COLS)
           for e in [tid + THREADS * i for i in range(MIX * KT // THREADS)]]
    rounds = (s * KT + THREADS - 1) // THREADS
    # Original batch=4 at 512 lanes -> batch=8 at 256 lanes.
    for begin in range_constexpr(0, rounds, 8):
        indices = range(begin, min(begin + 8, rounds))
        got = [slice_loads(c, task * COLS, fx.min(tid + THREADS * i, s * KT - 1)) for i in indices]
        for i, value in zip(indices, got):
            at = tid + THREADS * i
            if at < s * KT:
                slice_row(c, task * COLS, at, value)
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
def stage_gate(c, token):
    stage_gate_c1(dict(c, lane=c['tid'] % 32, wave=c['tid'] // 32), token)


@traced
def gemv_fp8_loads(c, weight, scales, inner, task):
    return reference_fp8_loads(dict(c, lane=c['tid'] % 32, wave=c['tid'] // 32),
                               scalar_address(weight), scalar_address(scales), inner, task)


@traced
def stage_wqkv(c, task, ops, t0, n):
    matrix = dict(c, S=n, lane=c['tid'] % 32, wave=c['tid'] // 32)
    reference_fp8_matrices_rows(matrix, HIDDEN, ops)
    if c['tid'] < ROWS * n:
        row, tl = c['tid'] % ROWS, c['tid'] // ROWS
        value = row_sum(c['red'], row, tl)
        c['put'](c['qkv'], (t0 + tl) * QKV_ROWS + task * ROWS + row, bf16_round(value))
    gpu.barrier()


@traced
def qprojection(c, task, ops, t0, n, rope, index=False):
    c = dict(c, args=dict(c['args'], q_out=scalar_address(c['args']['q_out'])))
    matrix = dict(c, S=n, lane=c['tid'] % 32, wave=c['tid'] // 32,
                  xl=c['qxl'], xsl=c['qxsl'])
    reference_fp8_matrices_rows(matrix, Q_RANK, ops)
    if c['tid'] < (ROWS // 2) * n:
        tl, pair = c['tid'] // (ROWS // 2), c['tid'] % (ROWS // 2)
        row = task * ROWS + 2 * pair
        v0 = bf16_round(row_sum(c['red'], 2 * pair, tl))
        v1 = bf16_round(row_sum(c['red'], 2 * pair + 1, tl))
        dim = ip.DIM if index else HEAD_DIM
        rotated = IDX_ROPE if index else ROPE
        live = row % dim >= dim - rotated
        cs, sn = rope
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        if const_expr(index):
            c['put_bf'](c['iq'], (t0 + tl) * IDX_ROWS + row, [e0, e1])
        else:
            at = (t0 + tl) * c['d'].heads * HEAD_DIM + row
            bo.buffer_store(e0.to(fx.BFloat16), rsrc(c['args']['q_out']), at)
            bo.buffer_store(e1.to(fx.BFloat16), rsrc(c['args']['q_out']), at + 1)
    gpu.barrier()


@traced
def stage_wqb(c, task, ops, t0, n, rope):
    qprojection(c, task, ops, t0, n, rope)


@traced
def stage_iq(c, task, ops, t0, n, rope):
    qprojection(c, task, ops, t0, n, rope, index=True)


@traced
def stage_iquant(c, task):
    args = dict(c['args'])
    for name in ('iq_out', 'iqs_out', 'iw_out'):
        args[name] = scalar_address(args[name])
    stage_iquant_c1(dict(c, args=args, iw=scalar_region(c['iw'])), task)


@traced
def stage_norm(c, t):
    """The seam's RMSNorm (rstd = rsq(fma(sum, 1/H, eps)), bf16((x rstd) w); 256
    threads, 3 chunks of 8 at 8 t + 2048 c), then ATOM's quantize_fp8 of the bf16
    row (amax floored at AMAX_FLOOR, the ceil code, x / scale) -> x8 words and codes;
    the FFN seam (``ffn``, whose x8 K2b quantizes itself) stores the bf16 row as
    ``normed`` instead. A layer that feeds its indexer (``index``) stores it as
    ``normed`` too and puts it in NB. Threads 256.. mirror thread 255 and store
    nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    c = peer_stage(c, ("nb_dsts",))
    a = c["args"]
    # the row's store policy: device scope when this launch also reads it
    normed_cm = c.get("normed_cm", 0)
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + 2048 * ch for ch in range(3)]
    # the norm weights, which the row does not decide: loaded before its wait
    wts = [
        [ld_bf(a["attn_w"], fx.min(cols[ch], HIDDEN - 8) + j) for j in range(8)]
        for ch in range(3)
    ]
    bases = [t * HIDDEN + fx.min(cols[ch], HIDDEN - 8) for ch in range(3)]
    words = c["poll"](
        [(c["lin"], bases[ch] // 2 + k, 1) for ch in range(3) for k in range(4)]
    )
    xs = []
    for ch in range_constexpr(3):
        v = []
        for k in range_constexpr(4):
            v += [bf_lo(words[4 * ch + k][0]), bf_hi(words[4 * ch + k][0])]
        xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    acc = fx.Float32(0.0)
    for ch in range_constexpr(3):
        for x in xs[ch]:
            acc = acc + x * x
    for off in (1, 2, 4, 8, 16):
        acc = acc + acc.shuffle_xor(off, 32)
    fx.ptr_store(acc, red + tid)
    gpu.barrier()
    acc = acc + fx.ptr_load(red + (tid ^ 32))
    gpu.barrier()
    if (lane == 0) & mine:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    tot = butterfly(fx.ptr_load(red + lane % 4), (1, 2))
    r = hw_rsq(fma(tot, fx.Float32(1.0 / HIDDEN), fx.Float32(EPS)))
    for ch in range_constexpr(3):
        col = fx.min(cols[ch], HIDDEN - 8)
        ys = [bf16_round((xs[ch][j] * r) * wts[ch][j]) for j in range(8)]
        amax = fx.Float32(0.0)
        for y in ys:
            amax = fx.max(amax, abs(y))
        amax = fx.max(butterfly(amax, (1, 2), fx.max), fx.Float32(AMAX_FLOOR))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        step = pow2(code)
        if mine & (cols[ch] < HIDDEN):
            if const_expr(c["index"] or c["ffn"]):
                for j in range_constexpr(8):
                    bo.buffer_store(
                        ys[j].to(fx.BFloat16),
                        rsrc(a["normed"]),
                        t * HIDDEN + col + j,
                        cache_modifier=normed_cm,
                    )
            if const_expr(c["index"]):
                # the indexer's weights_proj reads the bf16 row
                c["put_bf"](c["nb"], t * HIDDEN + col, ys[0:4])
                c["put_bf"](c["nb"], t * HIDDEN + col + 4, ys[4:8])
                if const_expr(c["tok_par"]):
                    # token parallel: the row for every other rank's compressor
                    tg = c["t0"] + t
                    for p in range_constexpr(c["d"].tp):
                        if c["rank"] != p:
                            dst = c["nb_dsts"][p]
                            c["put_bf"](dst, tg * HIDDEN + col, ys[0:4])
                            c["put_bf"](dst, tg * HIDDEN + col + 4, ys[4:8])
            if const_expr(not c["ffn"]):
                w0 = fp8_pack4(*[clamp_fp8(ys[i] / step) for i in range(4)])
                w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] / step) for i in range(4)])
                c["put_words"](c["x8"], t * X8_WORDS + col // 4, [w0, w1])
                if tt % 4 == 0:
                    c["put"](c["x8s"], t * (HIDDEN // 32) + col // 32, code)
    gpu.barrier()


@traced
def stage_qkv(c, t):
    """Token t: q_norm -> MXFP8 (the Triton dual norm, 4 warps x 8 columns) and
    kv_norm -> GPT-J RoPE of the tail -> bf16 -> group-32 FP8 QAT -> the ring.

    Threads 256.. mirror thread 255's loads (so every wave reaches the barrier)
    and store nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    c = peer_stage(c, ("q_dsts", "kv_dsts"))
    c = argument_stage(c, ("qn_w", "kvn_w", "cos", "sin", "pos", "ring", "ring_rows", "qr_out", "qrs_out"))
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    # every weight and table value the token's rows do not decide, loaded
    # before their wait
    qnw = [ld_bf(a["qn_w"], qc[j]) for j in range(8)]
    kvnw = [ld_bf(a["kvn_w"], tt * 2 + j) for j in range(2)]
    pos = ld_i32(a["pos"], 2 * t)
    jr = tt - (KV_DIM - ROPE) // 2
    live = jr >= 0
    jj = fx.max(jr, 0)
    cs = live.select(ld_bf(a["cos"], pos * HALF + jj), fx.Float32(1.0))
    sn = live.select(ld_bf(a["sin"], pos * HALF + jj), fx.Float32(0.0))
    rel = ld_i32(a["ring_rows"], t)
    got = c["poll"](
        [(c["qkv"], t * QKV_ROWS + qc[j], 1) for j in range(8)]
        + [(c["qkv"], t * QKV_ROWS + Q_RANK + tt * 2 + j, 1) for j in range(2)]
    )
    qv, kvv = got[:8], got[8:]
    xq = [
        ((tt * 8 + j) < Q_RANK).select(qv[j][0].bitcast(fx.Float32), fx.Float32(0.0))
        for j in range(8)
    ]
    xk = [kvv[j][0].bitcast(fx.Float32) for j in range(2)]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    for off in (8, 4, 2, 1, 16):
        ssq = ssq + ssq.shuffle_xor(off, 32)
    ssk = xk[0] * xk[0] + xk[1] * xk[1]
    for off in (8, 4, 2, 1, 16):
        ssk = ssk + ssk.shuffle_xor(off, 32)
    fx.ptr_store(ssq, red + tid)
    fx.ptr_store(ssk, red + 256 + tid)
    gpu.barrier()
    ssq = ssq + fx.ptr_load(red + (tid ^ 32))
    ssk = ssk + fx.ptr_load(red + 256 + (tid ^ 32))
    gpu.barrier()
    if (lane == 63) & mine:
        fx.ptr_store(ssq, red + wave)
        fx.ptr_store(ssk, red + 4 + wave)
    gpu.barrier()
    tq = butterfly(fx.ptr_load(red + lane % 4), (2, 1))
    tk = butterfly(fx.ptr_load(red + 4 + lane % 4), (2, 1))
    nq = hw_rsq(
        div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
        + fx.Float32(EPS)
    )
    nk = hw_rsq(
        div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
        + fx.Float32(EPS)
    )
    # q: MXFP8 by _mxfp8_quant_op, a group of 32 = 4 threads
    ys = [(xq[j] * nq) * qnw[j] for j in range(8)]
    amax = fx.Float32(0.0)
    for y in ys:
        amax = fx.max(amax, abs(y))
    code = code_round(fx.max(butterfly(amax, (1, 2), fx.max), fx.Float32(AMAX_FLOOR)))
    mul = pow2(254 - code)
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[ys[i] * mul for i in range(4)])
        w1 = fp8_pack4(*[ys[4 + i] * mul for i in range(4)])
        if const_expr(c["tok_par"]):
            # token parallel: to every rank, at the token's row of the step
            tg = c["t0"] + t
            for p in range_constexpr(c["d"].tp):
                qw, qs = c["q_dsts"][p]
                c["put_words"](qw, tg * QX8_WORDS + 2 * tt, [w0, w1])
                if tt % 4 == 0:
                    c["put"](qs, tg * (Q_RANK // 32) + tt // 4, code)
        else:
            c["put_words"](c["qx8"], t * QX8_WORDS + 2 * tt, [w0, w1])
            if tt % 4 == 0:
                c["put"](c["qx8s"], t * (Q_RANK // 32) + tt // 4, code)
        if const_expr(c["index"]):
            # the (qr, qr_scale) pair the original indexer reads
            bo.buffer_store(
                fx.Vector.from_elements([w0, w1], fx.Int32),
                rsrc(a["qr_out"]),
                t * QX8_WORDS + 2 * tt,
            )
            if tt % 4 == 0:
                bo.buffer_store(
                    fx.Int8(code), rsrc(a["qrs_out"]), t * (Q_RANK // 32) + tt // 4
                )
    # kv: this thread's pair (2 tt, 2 tt + 1), RoPE on the last 64 columns
    kn = [bf16_round((xk[j] * nk) * kvnw[j]) for j in range(2)]
    e0 = bf16_round(kn[0] * cs - kn[1] * sn)
    e1 = bf16_round(kn[0] * sn + kn[1] * cs)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8), fx.max)
    code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    step = pow2(code)
    scaled = [clamp_fp8(v / step) for v in (e0, e1)]
    # the step's row of this token (int32, -1: none) past the layer's ring start
    row = rel + a["ring_off"]
    if const_expr(c["kv_fp8"]):
        _ring_row_fp8(c, t, tt, mine, rel, row, scaled, code)
    else:
        qat = [fp8_value(v) * step for v in scaled]
        # a [rows, 512] bf16 pool's row: the QAT values
        if mine & (rel >= 0):
            dst = rsrc(a["ring"] + fx.Int64(row) * (KV_DIM * 2))
            for j in range_constexpr(2):
                bo.buffer_store(qat[j].to(fx.BFloat16), dst, tt * 2 + j)
        if const_expr(c["tok_par"]):  # noqa: SIM102
            # token parallel: the row to every other rank (``ring_row_from``)
            if mine:
                tg = c["t0"] + t
                for p in range_constexpr(c["d"].tp):
                    if c["rank"] != p:
                        c["put_bf"](c["kv_dsts"][p], tg * KV_DIM + 2 * tt, qat)
    gpu.barrier()


@traced
def stage_iw(c, task, hidden, nbl, load_group):
    """Row ``task`` of weights_proj; wave w its tokens w, w + WAVES, ..., a group
    of WAVES tokens' normed rows at a time in LDS ``nbl`` (bf16 pairs,
    row-packed; ``load_group(first token, tokens)`` puts them there) -> IW. The
    weights load once; each (row, token) sum keeps wv_splitk's per-lane order,
    so spreading the tokens over waves changes nothing but the latency."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    c = peer_stage(c, ("iw_dsts",))
    a = dict(c["args"], iw_w=scalar_address(c["args"]["iw_w"]))
    m = task
    ks = [
        k1 + k2 * 512
        for k1 in range(0, hidden, 2048)
        for k2 in range(4)
        if k1 + k2 * 512 < hidden
    ]
    wvs = []
    for k0 in ks:
        kk = fx.min(k0 + lane * 8, hidden - 8)
        wvs.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["iw_w"]),
                    (m * hidden + kk) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT_GFX1250,
                )
            )
        )
    for g in range_constexpr(0, s, 4):
        n = min(4, s - g)
        load_group(g, n)
        tl = fx.min(wave, n - 1)
        t = tl if g == 0 else tl + g
        total = fx.Float32(0.0)
        for i in range_constexpr(len(ks)):
            k = ks[i] + lane * 8
            live = k < hidden
            kk = fx.min(k, hidden - 8)
            xv = fx.Vector(
                fx.ptr_load(
                    nbl + ((tl * hidden + kk) // 2),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            wv = wvs[i]
            ax = [bf_lo(xv[j]) for j in range(4)]
            ay = [bf_hi(xv[j]) for j in range(4)]
            bx = [bf_lo(wv[j]) for j in range(4)]
            by = [bf_hi(wv[j]) for j in range(4)]
            accx = ax[0] * bx[0]
            accy = ay[0] * by[0]
            for j in range_constexpr(1, 4):
                accx = fx.Float32(fmath.fma(ax[j], bx[j], accx))
                accy = fx.Float32(fmath.fma(ay[j], by[j], accy))
            total = fx.Float32(live.select(total + (accx + accy), total))
        w = wv_reduce_c1(total, lane, c["tid"], c["red"])
        if (lane == 0) & (wave < n):
            if const_expr(c["tok_par"]):
                # token parallel: to every rank, at the token's row of the step
                for p in range_constexpr(c["d"].tp):
                    dst = c["iw_dsts"][p]
                    c["put"](dst, (c["t0"] + t) * ip.HEADS + m, bf16_round(w))
            else:
                c["put"](c["iw"], t * ip.HEADS + m, bf16_round(w))
        gpu.barrier()


@traced
def load_x8(c, words_mb, codes_mb, words, groups, xl, xsl, t0=0, n=None):
    """Tokens t0 .. t0 + n's MXFP8 rows (every token's by default) from the
    mailbox into LDS rows 0 .. (codes as i32). ``n`` traced (a tile a CTA picks
    at run time): a TILE-row copy of which only n rows are polled."""
    words_mb, codes_mb = scalar_region(words_mb), scalar_region(codes_mb)
    n = c["S"] if n is None else n
    rows = n if isinstance(n, int) else TILE
    poll_copy(
        c,
        c["tid"],
        THREADS,
        [
            (words_mb, lambda u: plus(t0 * words, u), rows * words, xl, words),
            (codes_mb, lambda u: plus(t0 * groups, u), rows * groups, xsl, groups),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
    )
    gpu.barrier()


@traced
def normed_row_from(c, nbg, t, n):
    """Token t, when another rank owns it: its normed row from NBG -> ``normed``
    (``stage_norm``'s store)."""
    nbg = scalar_region(nbg)
    tid = c["tid"]
    a = c["args"]
    if t // n != c["rank"]:
        for k in range_constexpr(-(-(HIDDEN // 4) // THREADS)):
            q = tid + THREADS * k
            if q < HIDDEN // 4:
                (w,) = c["poll"]([(nbg, t * (HIDDEN // 2) + 2 * q, 2)])
                bo.buffer_store(
                    fx.Vector.from_elements([w[0], w[1]], fx.Int32),
                    rsrc(a["normed"]),
                    t * (HIDDEN // 2) + 2 * q,
                )


@traced
def ring_row_from(c, kg, t, n):
    """Token t, when another rank owns it: its window row from KG -> the ring
    row (``stage_qkv``'s store): a BF16 row's words, or an FP8 row's data
    words and then its codes, a word each (``_ring_row_fp8``)."""
    kg = scalar_region(kg)
    c = argument_stage(c, ("ring", "ring_rows"))
    tid = c["tid"]
    a = c["args"]
    if t // n != c["rank"]:
        rel = ld_i32(a["ring_rows"], t)
        row = rel + a["ring_off"]
        words = KV_DIM // 4 + KV_DIM // 32 if const_expr(c["kv_fp8"]) else KV_DIM // 2
        if tid < words:
            (w,) = c["poll"]([(kg, t * (KV_DIM // 2) + tid, 1)])
            if rel >= 0:
                if const_expr(c["kv_fp8"]):
                    dst = rsrc(a["ring"] + fx.Int64(row) * KV_FP8_ROW_BYTES)
                    if tid < KV_DIM // 4:
                        bo.buffer_store(w[0], dst, tid)
                    else:
                        bo.buffer_store(fx.Int8(w[0]), dst, KV_DIM + tid - KV_DIM // 4)
                else:
                    bo.buffer_store(
                        w[0],
                        rsrc(a["ring"] + fx.Int64(row) * (KV_DIM * 2)),
                        tid,
                    )


def build_dspark_k1(key: AttnPreBuild):
    # Explicit scope gate until each branch has GPU acceptance.
    assert key.tp == 4 and key.tokens in (3, 5, 6, 10, 12, 15, 18, 20, 24, 25, 30, 35, 36, 40, 42, 48)
    assert not key.index_fp4 and not key.kv_fp8 and not key.timeline
    assert key.diag_off == -1
    assert not key.tok_par or key.tokens in (12, 24)
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s)
    tiles, rows = token_tiles(s), tile_rows(s)
    keyed = key_tuple(key, SOURCES)
    dims = Dims(key.tp)
    assert dims.wqb_tasks % BLOCKS == 0
    if key.tok_par:
        assert s % key.tp == 0, f"{s} rows do not split over {key.tp} ranks"
    own_n = s // key.tp if key.tok_par else s
    own_tiles, own_rows_lds = token_tiles(own_n), tile_rows(own_n)
    wqb_per_cta = dims.wqb_tasks // BLOCKS

    # LDS: what several stages read, then one union of what only one stage
    # does -- slice's, wqkv_a's and index_w's buffers; a barrier ends each
    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        qxl: fx.Array[fx.Int32, rows * lds_row(QX8_WORDS), 16]
        qxsl: fx.Array[fx.Int32, rows * lds_row(Q_RANK // 32), 16]
        tls: fx.Array[fx.Int64, TL_POINTS if key.timeline else 1, 16]

    @fx.struct
    class SliceLds:
        rl: fx.Array[fx.Float32, s * KT, 16]
        fl: fx.Array[fx.Float32, MIX * KT, 16]

    @fx.struct
    class WqkvLds:
        xl: fx.Array[fx.Int32, rows * lds_row(X8_WORDS), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(HIDDEN // 32), 16]

    @fx.struct
    class IwLds:
        # a group of WAVES tokens' normed rows (``stage_iw``)
        nbl: fx.Array[
            fx.Int32, min(own_n, 4) * HIDDEN // 2 if key.feeds_index else 1, 16
        ]

    @fx.union
    class StageLds:
        slice: SliceLds
        wqkv: WqkvLds
        iw: IwLds

    name = kernel_symbol(
        "gfx1250_dspark_k1",
        s=s,
        f=int(key.fold),
        ix=int(key.feeds_index),
        i4=int(key.index_fp4),
        a=int(key.aux),
        tl=int(key.timeline),
        tpar=int(key.tok_par),
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_pre(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64,
        iw_out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, tl: Int64,
    ):  # fmt: skip
        wqb, wqb_s, q_out = map(vector_address, (wqb, wqb_s, q_out))
        if const_expr(s == 48 and key.feeds_index and key.fold):
            iw_w = vector_address(iw_w)
            # This uniform scale address otherwise spans K1 in two VGPRs;
            # its low word spills at the index-query weight load for c8.
            iq_w, iq_ws = map(scalar_address, (iq_w, iq_ws))
        else:
            iq_w, iq_ws, iw_w = map(vector_address, (iq_w, iq_ws, iw_w))
        iq_out, iqs_out, iw_out = map(vector_address, (iq_out, iqs_out, iw_out))
        cos, sin, pos = map(vector_address, (cos, sin, pos))
        ring, ring_rows = vector_address(ring), vector_address(ring_rows)
        qn_w, kvn_w = vector_address(qn_w), vector_address(kvn_w)
        qr_out, qrs_out = vector_address(qr_out), vector_address(qrs_out)
        sym, peers = vector_address(sym), vector_address(peers)
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        stage_lds = alloc.allocate(StageLds)
        sl, wl, il = stage_lds.slice.peek(), stage_lds.wqkv.peek(), stage_lds.iw.peek()
        mb = Mailbox(layer, gfx1250=True)
        c = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64,
            "rl": sl.rl.ptr, "fl": sl.fl.ptr, "red": lds.red.ptr, "threads": THREADS,
            "xl": wl.xl.ptr, "xsl": wl.xsl.ptr, "qxl": lds.qxl.ptr, "qxsl": lds.qxsl.ptr,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
            "args": {
                "res_in": res_in, "pend": pend, "post_in": post_in, "comb_in": comb_in,
                "pre_in": pre_in, "hc_fn": hc_fn, "hc_scale": hc_scale, "hc_base": hc_base,
                "attn_w": attn_w, "wqkv": wqkv, "wqkv_s": wqkv_s, "qn_w": qn_w,
                "kvn_w": kvn_w, "wqb": wqb, "wqb_s": wqb_s, "cos": cos, "sin": sin,
                "pos": pos, "ring": ring, "ring_rows": ring_rows, "ring_off": ring_off,
                "res_out": res_out,
                "post_out": post_out, "comb_out": comb_out, "pre_out": pre_out,
                "q_out": q_out, "normed": normed, "qr_out": qr_out, "qrs_out": qrs_out,
                "aux": aux, "iq_w": iq_w, "iq_ws": iq_ws, "iw_w": iw_w,
                "iq_out": iq_out, "iqs_out": iqs_out, "iw_out": iw_out,
            },
            "fold": key.fold, "index": key.feeds_index, "aux": key.aux, "ffn": False,
            "d": dims, "tok_par": False, "kv_fp8": key.kv_fp8,
        }  # fmt: skip
        for region, (off, _) in layout.items():
            c[region] = sreg(scratch, off, region)
        c["rank"] = rank
        if const_expr(key.tok_par):
            bases = [vector_address(b) for b in peer_bases(scalar_address(peers), key.tp)]
            c["peer_addr"] = lambda p: bases[p]
            pl = peer_offsets(s, dims)
            co = own_view(c, K1_OWN)
            co["tok_par"] = True
            # qkv pushes its token's q latent to every rank's QG / QGS and its
            # window row to every other rank's KG
            co["q_dsts"] = [
                (preg(bases[p], pl["qg"], "qg"), preg(bases[p], pl["qgs"], "qgs"))
                for p in range(key.tp)
            ]
            co["kv_dsts"] = [preg(bases[p], pl["kg"], "kg") for p in range(key.tp)]
            co["nb_dsts"] = [preg(bases[p], pl["nbg"], "nbg") for p in range(key.tp)]
            co["iw_dsts"] = [preg(bases[p], pl["iwg"], "iwg") for p in range(key.tp)]
        else:
            co = c
        timeline = key.timeline
        tls = lds.tls.ptr
        stamp_begin(timeline, tls, tid, TL_POINTS)
        enter_stage("k1.slice")
        for task in range(first_task(bid, 0), SLICES, BLOCKS):
            stage_slice(co, task)
        stamp(timeline, tls, tid, 1)
        # the gate's outputs are the next seam's: it runs after the stages
        # this launch waits on that its CTAs may also hold (norm, wqkv_a)
        enter_stage("k1.norm")
        for t in range(first_task(bid, NORM0), own_n, BLOCKS):
            stage_norm(co, t)
        stamp(timeline, tls, tid, 2)
        enter_stage("k1.wqkv_a")
        # a CTA's wqkv task (at most one): its weights in flight before the
        # normed row's wait
        assert WQKV_TASKS <= BLOCKS
        if const_expr(len(own_tiles) == 1):
            wqkv_task = first_task(bid, WQKV0)
            if wqkv_task < WQKV_TASKS:
                wqkv_ops = gemv_fp8_loads(co, wqkv, wqkv_s, HIDDEN, wqkv_task)
                load_x8(
                    co, co["x8"], co["x8s"], X8_WORDS, HIDDEN // 32, co["xl"],
                    co["xsl"], 0, own_rows_lds,
                )  # fmt: skip
                stage_wqkv(co, wqkv_task, wqkv_ops, 0, own_n)
        else:
            # a task's tiles in series on one CTA left more than half the CTAs
            # idle through it (the weights read again a tile: L2 hits)
            tile, wqkv_tasks = tile_tasks(bid, WQKV0, len(own_tiles), WQKV_TASKS)
            wqkv_ops = [
                gemv_fp8_loads(co, wqkv, wqkv_s, HIDDEN, fx.min(t, WQKV_TASKS - 1))
                for t in wqkv_tasks
            ]
            t0 = tile * TILE
            n = fx.min(fx.Int32(TILE), own_n - t0)
            load_x8(
                co, co["x8"], co["x8s"], X8_WORDS, HIDDEN // 32, co["xl"], co["xsl"],
                t0, n,
            )  # fmt: skip
            for k in range_constexpr(len(wqkv_tasks)):
                if wqkv_tasks[k] < WQKV_TASKS:
                    stage_wqkv(co, wqkv_tasks[k], wqkv_ops[k], t0, n)
        stamp(timeline, tls, tid, 3)
        enter_stage("k1.gate")
        for t in range(first_task(bid, GATE0), own_n, BLOCKS):
            stage_gate(co, t)
        stamp(timeline, tls, tid, 4)
        enter_stage("k1.qkv")
        for t in range(first_task(bid, QKV0), own_n, BLOCKS):
            stage_qkv(co, t)
        if const_expr(key.tok_par):
            # wq_b reads every token's q latent where qkv pushed it
            c["qx8"] = preg(sym, pl["qg"], "qg")
            c["qx8s"] = preg(sym, pl["qgs"], "qgs")
        stamp(timeline, tls, tid, 5)
        enter_stage("k1.wq_b")
        # every CTA's wq_b tasks (and an indexer layer's wq task), two at a
        # time: the first two's weights in flight before the q latent's wait;
        # the weights stay in registers for the later token tiles (and the iq
        # task runs in each: held on past them, its weights spill in index_w)
        assert IQ_TASKS == BLOCKS
        wqb_task = first_task(bid, WQB0)
        wqb_ops = [
            gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, nth_task(wqb_task, i))
            for i in range(min(2, wqb_per_cta))
        ]
        iq_task = first_task(bid, IQ0)
        if const_expr(key.feeds_index):
            iq_ops = gemv_fp8_loads(c, iq_w, iq_ws, Q_RANK, iq_task)
        wqb_all = list(wqb_ops)
        for t0, n in tiles:
            # the epilogues' RoPE tables, loaded before the q latent's wait
            ropes = [
                wqb_rope(c, nth_task(wqb_task, b), t0, n) for b in range(wqb_per_cta)
            ]
            if const_expr(key.feeds_index):
                iq_ropes = iq_rope(c, iq_task, t0, n)
            load_x8(
                c, c["qx8"], c["qx8s"], QX8_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"],
                t0, n,
            )  # fmt: skip
            for b in range_constexpr(0, wqb_per_cta, 2):
                if const_expr(b >= WQB_HELD):
                    # past the held tasks: loaded again each token tile (L2 hits)
                    wqb_all[b:] = [
                        gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, nth_task(wqb_task, b + i))
                        for i in range(min(2, wqb_per_cta - b))
                    ]
                elif const_expr(b > 0 and t0 == 0):
                    wqb_all += [
                        gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, nth_task(wqb_task, b + i))
                        for i in range(min(2, wqb_per_cta - b))
                    ]
                for i in range_constexpr(min(2, wqb_per_cta - b)):
                    stage_wqb(
                        c, nth_task(wqb_task, b + i), wqb_all[b + i], t0, n,
                        ropes[b + i],
                    )  # fmt: skip
            if const_expr(key.feeds_index):
                stage_iq(c, iq_task, iq_ops, t0, n, iq_ropes)
        stamp(timeline, tls, tid, 6)
        if const_expr(key.feeds_index):
            enter_stage("k1.index_w")

            def load_group(g, n):
                poll_copy(
                    co, tid, THREADS,
                    [(co["nb"], lambda u: plus(g * HIDDEN // 2, u), n * HIDDEN // 2,
                      il.nbl.ptr, None)],
                )  # fmt: skip
                gpu.barrier()

            for task in range(first_task(bid, IW0), IW_TASKS, BLOCKS):
                stage_iw(co, task, HIDDEN, il.nbl.ptr, load_group)
            if const_expr(key.tok_par):
                # the quantization reads every token's head weights where iw
                # pushed them
                c["iw"] = preg(sym, pl["iwg"], "iwg")
            enter_stage("k1.index_quant")
            for task in range(first_task(bid, IQUANT0), s * ip.HEADS, BLOCKS):
                if const_expr(key.index_fp4):
                    stage_iquant_fp4(c, task)
                else:
                    stage_iquant(c, task)
        stamp(timeline, tls, tid, TL_POINTS - 1)
        if const_expr(key.tok_par):
            kg = preg(sym, pl["kg"], "kg")
            for t in range(first_task(bid, QKV0), s, BLOCKS):
                ring_row_from(c, kg, t, own_n)
            if const_expr(key.feeds_index):
                # the compressor reads every token's normed row
                nbg = preg(sym, pl["nbg"], "nbg")
                for t in range(first_task(bid, NORM0), s, BLOCKS):
                    normed_row_from(c, nbg, t, own_n)
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @launcher(name)
    def launch(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64,
        iw_out: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, tl: Int64, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        attn_pre(
            res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_w,
            wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s, cos, sin, pos, ring, ring_rows, ring_off,
            res_out, post_out, comb_out, pre_out, q_out, normed, qr_out, qrs_out, aux,
            iq_w, iq_ws, iw_w, iq_out, iqs_out, iw_out, scratch, sym, peers, rank,
            layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(attn_pre, launch)
    return launch
