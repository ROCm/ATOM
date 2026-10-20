"""Wide builds' grouped B1/B3 (``layout_wide.for_wide``): delta_group.py's
stages with the mixed rows polled in rounds, every request row's MFMA result
stored from its lanes (``common_wide``), and B3's per-row work past the
CTA's threads in rounds: the qkvz loads (:func:`_qkvz_round`), the b and a
chains 4 request rows at a time (:func:`_ba_chains`, on the mixed rows held
as bf16 words: all rows fit one pass) and the q and k norms
(:func:`_qk_norms`). Each row's q and k load with its recurrence step, which
stores its state at once. Past ``WAVES`` rows B1's and B3's epilogues run in
passes of ``WAVES`` request rows (:func:`_b1_out`, :func:`_b3_out`).

Derived from delta_group.py; delta_group.py stays the source.
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import THREADS
from atom.models.qwen4_exp.mono.kernels.device_ops import (
    read_lane_i32,
    rsq,
    xred,
    xshfl,
)
from atom.models.qwen4_exp.mono.kernels.layout import (
    RW_PART,
    RW_ROW,
    WAVES,
    b3_tiles,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    add,
    bf16r,
    bf16x8,
    delta_state,
    lds_f4,
    lds_row,
    mb_ix,
    mfma_operands,
    p2_issue,
    p2_mma,
    row_ix,
    sigmoid_fast,
    softplus_fast,
    sum16,
    task_index,
)
from atom.models.qwen4_exp.mono.kernels.stages.common_wide import (
    poll_mixed_wide,
    store_rows,
)
from atom.models.qwen4_exp.mono.kernels.stages.delta import (
    B1_GROUPS,
    B1_KI,
    B3_TILES,
    QKV,
    RSQ128,
    _theirs,
    sumsq64,
)
from atom.models.qwen4_exp.mono.kernels.stages.delta_group import _group_conv
from atom.models.qwen4_exp.mono.kernels.stages.hc import OOB, fma
from atom.models.qwen4_exp.mono.kernels.stages.moe import (
    CHAIN_STEPS,
    _chain_step,
    _part,
)


def _b1_taps(c, p, ch, ek):
    """Request row ``rw = w + 8 ek``'s conv state and taps (``b1``'s
    ``conv_out`` and ``cs`` of row ``rw``) for :func:`_b1_out`."""
    rw = fx.min(c.wave + WAVES * ek, c.rows - 1)
    conv_in, conv_out = delta_state(c, p, rw)[:2]
    cs = [
        fx.Int32(
            bo.buffer_load(conv_in, k * (QKV // 2) + ch // 2, vec_width=1, dtype=T.i32)
        )
        for k in range(3)
    ]
    return rw, conv_out, cs


def _b1_out(c, red, rs, t, ch, cw, row, ek):
    """Pass ``ek`` of B1's epilogue past ``WAVES`` rows: wave ``w`` runs
    request row ``w + 8 ek``'s on its :func:`_b1_taps` ``row``, as ``b1``'s
    wave ``r`` row ``r``'s."""
    lane = c.lane
    rw, conv_out, cs = row
    q = c.wave + WAVES * ek
    if q < c.rows:
        proj = bf16r(fx.ptr_load(red + row_ix(lane, rw, rs)))
        out = proj
        if t < QKV // 64:
            # conv taps are oldest first. bf16 x bf16 products are exact, so
            # mul + add matches the native fma chain.
            w0, w1 = mbx.bf2_f32(cw[0])
            w2, w3 = mbx.bf2_f32(cw[1])
            old = [(w >> 16 * (lane % 2)) & 0xFFFF for w in cs]
            taps, nxt = _group_conv(c, red, rs, lane, rw, old, proj)
            words = [n | (xshfl(n, 1) << 16) for n in nxt]
            conv = taps[0] * w0
            conv = conv + taps[1] * w1
            conv = conv + taps[2] * w2
            conv = conv + proj * w3
            out = bf16r(conv * sigmoid_fast(conv))
            if lane % 2 == 0:
                for k in range_constexpr(3):
                    bo.buffer_store(words[k], conv_out, k * (QKV // 2) + ch // 2)
        other = xred(out, 1, _theirs)
        if lane % 2 == 0:
            mbx.put(
                c.mb_qkvz,
                mb_ix(t * 32 + lane // 2, rw),
                mbx.bf16_pair(out, other),
                c.tag,
            )


_b1_out = ASTRewriter.transform(_b1_out)


def b1(c, p):
    lane, wave = c.lane, c.wave
    t = task_index(c, "b1")
    x8 = c.lds_i + c.L.lds["x8"]
    xe = c.lds_i + c.L.lds["xe"]
    red = c.lds_f + c.L.lds["red"]
    rs = c.L.stride["red"]
    # Batched builds: wave ``r < rows`` runs request row r's epilogue.
    rw = 0 if c.rows == 1 else fx.min(wave, c.rows - 1)
    # Every row's table entry holds its group's input state (the group's
    # first row's) and its own snapshot.
    conv_in, conv_out = delta_state(c, p, rw)[:2]

    # Weights and the conv operands do not depend on the mixed row: issue
    # them before waiting for it. Wave ``w < 4`` owns n-block ``4 t + w``: one
    # chain over the 20 k_iters in order. Waves 4-7 load out of range and skip
    # the GEMM.
    nb = t * 4 + wave % 4
    off = (wave < 4).select(fx.Int32(0), fx.Int32(OOB))
    units = [
        p2_issue(p.w_qkvz, p.s_qkvz, nb, B1_KI, g, lane, off) for g in range(B1_GROUPS)
    ]
    ch = fx.min(t * 64 + lane, QKV - 1)
    cw = fx.Vector(bo.buffer_load(p.conv_w, ch * 2, vec_width=2, dtype=T.i32))
    cs = [
        fx.Int32(
            bo.buffer_load(conv_in, k * (QKV // 2) + ch // 2, vec_width=1, dtype=T.i32)
        )
        for k in range(3)
    ]
    more = [_b1_taps(c, p, ch, ek) for ek in range(1, -(-c.rows // WAVES))]

    poll_mixed_wide(c, x8, xe)

    if wave < 4:
        xr, er = mfma_operands(c, x8, xe)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for g in range_constexpr(B1_GROUPS):
            acc = p2_mma(acc, units[g], g, xr, er, lane)
        store_rows(c, acc, red, wave * 16 + lane % 16, rs)
    # ATOM's verify window: a group's snapshots overwrite its input taps in
    # place (``conv_out[r]`` overlaps ``conv_in``), so every wave's ``cs``
    # loads land before any wave stores past the barrier.
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()

    epilogue = (wave == 0) if c.rows == 1 else (wave < c.rows)
    if epilogue:
        proj = bf16r(fx.ptr_load(red + row_ix(lane, rw, rs)))
        out = proj
        if t < QKV // 64:
            # conv taps are oldest first. bf16 x bf16 products are exact, so
            # mul + add matches the native fma chain.
            w0, w1 = mbx.bf2_f32(cw[0])
            w2, w3 = mbx.bf2_f32(cw[1])
            old = [(w >> 16 * (lane % 2)) & 0xFFFF for w in cs]
            taps, nxt = _group_conv(c, red, rs, lane, rw, old, proj)
            words = [n | (xshfl(n, 1) << 16) for n in nxt]
            conv = taps[0] * w0
            conv = conv + taps[1] * w1
            conv = conv + taps[2] * w2
            conv = conv + proj * w3
            out = bf16r(conv * sigmoid_fast(conv))
            if lane % 2 == 0:
                for k in range_constexpr(3):
                    bo.buffer_store(words[k], conv_out, k * (QKV // 2) + ch // 2)
        other = xred(out, 1, _theirs)
        if lane % 2 == 0:
            mbx.put(
                c.mb_qkvz,
                mb_ix(t * 32 + lane // 2, rw),
                mbx.bf16_pair(out, other),
                c.tag,
            )
    for ek in range_constexpr(1, -(-c.rows // WAVES)):
        _b1_out(c, red, rs, t, ch, cw, more[ek - 1], ek)


b1 = ASTRewriter.transform(b1)


def _qkvz_round(c, kh, vh, s, m):
    """Round ``m`` of B3's qkvz loads: thread ``tid + 512 m = 80 r + i`` loads
    request row ``r``'s 4 consecutive values of ``delta.b3``'s loader thread
    ``i < 80`` (q 32 threads, k 32, the split's v rows 8 and z rows 8) into
    the row's ``qk`` / ``vz``."""
    tid = c.tid
    li = fx.min(tid + THREADS * m, 80 * c.rows - 1)
    i, ri = li % 80, li // 80
    src = (i < 72).select(
        (i < 64).select(
            (i < 32).select(kh * 64 + 2 * i, C.delta_key // 2 + kh * 64 + 2 * (i - 32)),
            C.delta_key + vh * 64 + 16 * s + 2 * (i - 64),
        ),
        QKV // 2 + vh * 64 + 16 * s + 2 * (i - 72),
    )
    if tid + THREADS * m < 80 * c.rows:
        v = mbx.poll([(c.mb_qkvz, mb_ix(src, ri), 2)], c.tag)[0]
        dw = (i < 64).select(
            c.L.lds["qk"] + ri * c.L.stride["qk"] + i * 4,
            c.L.lds["vz"] + ri * c.L.stride["vz"] + (i - 64) * 4,
        )
        dst = [c.lds_f + (dw + k) for k in range(4)]
        q0, q1 = mbx.bf2_f32(v[0])
        q2, q3 = mbx.bf2_f32(v[1])
        fx.ptr_store(q0, dst[0])
        fx.ptr_store(q1, dst[1])
        fx.ptr_store(q2, dst[2])
        fx.ptr_store(q3, dst[3])


_qkvz_round = ASTRewriter.transform(_qkvz_round)


def _ba_chains(c, baw, bax, tile):
    """B3's b and a chains of request rows ``4 tile ..`` (4 at most): waves
    ``1 + 2 tile`` (b) and ``2 + 2 tile`` (a), mod ``WAVES``, lane ``16 i + q``
    part ``q`` of request row ``4 tile + i``, as ``delta_group.b3``'s waves
    1-2, lane ``16 r + q``; ``bax`` holds the mixed rows as bf16 words."""
    lane, wave = c.lane, c.wave
    n = min(4, c.rows - 4 * tile)
    ab = (wave + (WAVES - 1 - 2 * tile)) % WAVES
    if (ab < 2) & (lane < 16 * n):
        q = lane % 16
        big = q < 8
        wrow = baw + (ab * RW_ROW + q * RW_PART)
        xrow = bax + row_ix(q * RW_PART, 4 * tile + lane // 16, c.L.stride["bax"])
        v4i = fx.Vector.make_type(4, fx.Int32)
        acc = fx.Float32(0.0)
        for k in range_constexpr(CHAIN_STEPS):
            wv = fx.Vector(fx.ptr_load(wrow + k * 4, result_type=v4i))
            w8 = bf16x8([wv[e] for e in range(4)])
            xv = fx.Vector(fx.ptr_load(xrow + k * 4, result_type=v4i))
            x8 = bf16x8([xv[e] for e in range(4)])
            new = acc
            for e in range_constexpr(8):
                new = fma(w8[e], x8[e], new)
            acc = _chain_step(k, big, new, acc)
        bits = acc.bitcast(fx.Int32)
        for r in range_constexpr(n):
            total = read_lane_i32(bits, 16 * r).bitcast(fx.Float32)
            for q in range_constexpr(1, 16):
                total = total + read_lane_i32(bits, 16 * r + q).bitcast(fx.Float32)
            if lane == 0:
                fx.ptr_store(
                    bf16r(total), lds_row(c, c.lds_f, "sums", 4 * tile + r) + ab
                )


_ba_chains = ASTRewriter.transform(_ba_chains)


def _qk_norms(c, m):
    """Round ``m`` of B3's q and k L2 norms: wave ``w + 8 m = 2 r + n`` takes
    request row r's q (n = 0) or k, as ``delta_group.b3``'s wave ``2 r + n``."""
    lane = c.lane
    q = c.wave + WAVES * m
    if q < 2 * c.rows:
        nk = q % 2
        halves = sumsq64(
            lds_row(c, c.lds_f, "qk", q // 2) + (nk * 128 + (lane // 32) * 64), lane
        )
        rs_n = rsq(xred(halves, 32, add) + C.eps)
        rs_n = (nk == 0).select(rs_n * fx.Int32(RSQ128).bitcast(fx.Float32), rs_n)
        if lane == 0:
            fx.ptr_store(rs_n, lds_row(c, c.lds_f, "sums", q // 2) + (2 + nk))


_qk_norms = ASTRewriter.transform(_qk_norms)


def _b3_out(c, nw, t, vh, s, ek):
    """Pass ``ek`` of B3's epilogue past ``WAVES`` rows: wave ``w`` runs
    request row ``w + 8 ek``'s, as ``b3``'s wave ``r`` row ``r``'s."""
    lane = c.lane
    q = c.wave + WAVES * ek
    rw = fx.min(q, c.rows - 1)
    if q < c.rows:
        cores_r, core_r, vz_r = [
            lds_row(c, c.lds_f, n, rw) for n in ("cores", "core", "vz")
        ]
        got = mbx.poll([(c.mb_core, mb_ix(vh * 128 + 2 * lane, rw), 2)], c.tag)[0]
        fx.ptr_store(got[0].bitcast(fx.Float32), cores_r + 2 * lane)
        fx.ptr_store(got[1].bitcast(fx.Float32), cores_r + (2 * lane + 1))
        tot = sumsq64(cores_r, lane) + sumsq64(cores_r + 64, lane)
        inv_rms = rsq(tot * (1.0 / C.delta_v_dim) + C.eps)
        r0 = (lane % 4) * 8
        cv = lds_f4(core_r, r0) + lds_f4(core_r, r0 + 4)
        zv = lds_f4(vz_r, 32 + r0) + lds_f4(vz_r, 32 + r0 + 4)
        wv = bf16x8([nw[e] for e in range(4)])
        out = [
            bf16r(((cv[e] * inv_rms) * wv[e]) * sigmoid_fast(zv[e])) for e in range(8)
        ]
        amax = fmath.absf(out[0])
        for e in range_constexpr(1, 8):
            amax = fx.max(amax, fmath.absf(out[e]))
        amax = xred(xred(amax, 1, fx.max), 2, fx.max)
        ex, scale = p2.e8m0_fp8(amax)
        lo = p2.fp8_word(out[0], out[1], out[2], out[3], scale)
        hi = p2.fp8_word(out[4], out[5], out[6], out[7], scale)
        if lane < 4:
            mbx.put2(c.mb_d8, mb_ix(vh * 32 + s * 8 + lane * 2, rw), lo, hi, c.tag)
        if lane == 0:
            mbx.put(c.mb_de, mb_ix(t, rw), ex, c.tag)


_b3_out = ASTRewriter.transform(_b3_out)


def b3(c, p):
    tid, lane, wave = c.tid, c.lane, c.wave
    rows = c.rows
    t = task_index(c, "b3")
    vh = t // 4
    s = t % 4
    kh = vh // C.delta_gqa
    row = tid // 16
    kl = tid % 16
    # Per request row LDS pointers (``at``); the one-row build's are made
    # up front as before.
    ptrs = {}
    for name in ("qk", "vz", "core", "cores", "sums", "baw", "bax"):
        view = c.lds_i if name in ("baw", "bax") else c.lds_f
        ptrs[name] = [
            lds_row(c, view, name, r) for r in range(1 if name == "baw" else rows)
        ]

    def at(name, r):
        return ptrs[name][r] if isinstance(r, int) else lds_row(c, c.lds_f, name, r)

    core_l, baw, bax = at("core", 0), at("baw", 0), at("bax", 0)
    tiles = b3_tiles(kh, vh, s)

    # Batched builds: producer ``m`` of a hint is request row ``m // n`` of the
    # one-row hint's ``n`` producers.
    def hint(m):
        k = m if rows == 1 else m % B3_TILES
        tile = tiles[5]
        for i in (4, 3, 2, 1, 0):
            tile = (k == i).select(tiles[i], tile)
        return c.mb_qkvz, mb_ix(tile * 32 + 31, 0 if rows == 1 else m // B3_TILES)

    bj = fx.min(fx.max(tid - 128, 0), C.hidden // 8 - 1)  # BA loader threads 128..447

    if t < c.L.tasks["b3"]:
        # Prefetch what does not depend on B1: the gate parameters, norm
        # weights (and below, ahead of the mixed row's poll, the b and a
        # weight rows). This row's state (every request row's) follows that
        # poll, which would wait for it.
        a_log = fx.Float32(bo.buffer_load(p.a_log, vh, vec_width=1, dtype=T.f32))
        dt_bias = fx.Float32(bo.buffer_load(p.dt_bias, vh, vec_width=1, dtype=T.f32))
        nw = fx.Vector(
            bo.buffer_load(
                p.norm_w, (s * 32 + (lane % 4) * 8) // 2, vec_width=4, dtype=T.i32
            )
        )

        # b and a: the bf16 mixed row and both weight rows into the
        # router's 16-part LDS layout, then part ``q`` of row b (wave 1) or a
        # (wave 2) on lane ``q``: one fma chain from +0, the 16 parts summed in
        # order. This overlaps wave 0's wait for B1. Batched builds: request
        # row r's chains run on lanes 16 r + q.
        if (tid >= 128) & (tid < 128 + C.hidden // 8):
            wb = fx.Vector(
                bo.buffer_load(
                    p.ba_w, vh * (C.hidden // 2) + bj * 4, vec_width=4, dtype=T.i32
                )
            )
            wa = fx.Vector(
                bo.buffer_load(
                    p.ba_w,
                    (C.delta_v_heads + vh) * (C.hidden // 2) + bj * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            rocdl.sched_barrier(0)
            specs = []
            for r in range_constexpr(rows):
                specs += [
                    (c.mb_xbf, mb_ix(bj * 4, r), 2),
                    (c.mb_xbf, mb_ix(bj * 4 + 2, r), 2),
                ]
            xv = mbx.poll(specs, c.tag)
            part, first = _part(bj * 8)
            off = part * RW_PART + (bj * 8 - first) // 2
            for r in range_constexpr(rows):
                words = [xv[2 * r][0], xv[2 * r][1], xv[2 * r + 1][0], xv[2 * r + 1][1]]
                fx.ptr_store(
                    fx.Vector.from_elements(words, fx.Int32), at("bax", r) + off
                )
            fx.ptr_store(wb, baw + off)
            fx.ptr_store(wa, baw + (RW_ROW + off))
        gpu.barrier()
        sbase = (vh * C.delta_v_dim + s * 32 + row) * C.delta_k_dim + kl
        # Every row's state addresses are read here: a table read
        # issued after a row's state stores would wait for them (vmcnt counts
        # stores too), and serialize the rows.
        # Each group's first row loads the group's state; the later rows
        # chain on it.
        st, rec_out = [], []
        for r in range_constexpr(rows):
            rec_in = delta_state(c, p, r)[2]
            rec_out.append(delta_state(c, p, r)[3])
            if const_expr(r % c.group == 0):
                st.append(
                    [
                        fx.Float32(
                            bo.buffer_load(
                                rec_in, sbase + 16 * e, vec_width=1, dtype=T.f32
                            )
                        )
                        for e in range(8)
                    ]
                )
        for k in range_constexpr(-(-rows // 4)):
            _ba_chains(c, baw, bax, k)

        mbx.pre_poll(hint, B3_TILES * rows, c.tag)
        for m in range_constexpr(-(-80 * rows // THREADS)):
            _qkvz_round(c, kh, vh, s, m)
        gpu.barrier()

        # The request rows step by step, interleaved so that their latencies
        # overlap; every row runs the operations of the one-row build.
        R = range(rows)
        qk_r = [at("qk", r) for r in R]

        # The gates, identical in all four siblings.
        decay, beta = [], []
        for r in range_constexpr(rows):
            b = fx.ptr_load(at("sums", r))
            a = fx.ptr_load(at("sums", r) + 1)
            sp = softplus_fast(a + dt_bias)
            decay.append(fmath.exp(sp * (fx.Float32(0.0) - fmath.exp(a_log))))
            beta.append(sigmoid_fast(b))

        # L2 norms of q and k. The one-row build: redundantly in every
        # wave. Batched builds: once per CTA, wave ``2 r + n`` takes request
        # row r's q (n = 0) or k, its lanes ``32 h + j`` the sumsq64 of half h
        # (the tree stays within 32 lanes), and leaves the factor in ``sums``
        # word ``2 + n`` (after b and a).
        if const_expr(rows == 1):
            sq = [sumsq64(qk_r[r], lane) + sumsq64(qk_r[r] + 64, lane) for r in R]
            sk = [
                sumsq64(qk_r[r] + 128, lane) + sumsq64(qk_r[r] + 192, lane) for r in R
            ]
            inv_q = [
                rsq(sq[r] + C.eps) * fx.Int32(RSQ128).bitcast(fx.Float32) for r in R
            ]
            inv_k = [rsq(sk[r] + C.eps) for r in R]
        else:
            for m in range_constexpr(-(-2 * rows // WAVES)):
                _qk_norms(c, m)
            gpu.barrier()
            inv_q = [fx.ptr_load(at("sums", r) + 2) for r in R]
            inv_k = [fx.ptr_load(at("sums", r) + 3) for r in R]

        # Gated delta rule on this row: S <- decay S; delta = beta (v - S k);
        # S <- S + delta k^T; core = S q.
        # Row r's state is row r - 1's result, one row after the other; a
        # group's first row starts from the group's state.
        sn, core = [], []
        for r in range_constexpr(rows):
            prev = st[r // c.group] if r % c.group == 0 else sn[r - 1]
            qh_r = [fx.ptr_load(qk_r[r] + (kl + 16 * e)) * inv_q[r] for e in range(8)]
            kh_r = [
                fx.ptr_load(qk_r[r] + (128 + kl + 16 * e)) * inv_k[r] for e in range(8)
            ]
            sv_r = [prev[e] * decay[r] for e in range(8)]
            pr_r = fx.Float32(0.0)
            for e in range_constexpr(8):
                pr_r = fma(sv_r[e], kh_r[e], pr_r)
            delta_r = beta[r] * (fx.ptr_load(at("vz", r) + row) - sum16(pr_r))
            sn.append([fma(delta_r, kh_r[e], sv_r[e]) for e in range(8)])
            cr_r = fx.Float32(0.0)
            for e in range_constexpr(8):
                cr_r = fma(sn[r][e], qh_r[e], cr_r)
            core.append(sum16(cr_r))
            for e in range_constexpr(8):
                bo.buffer_store(sn[r][e], rec_out[r], sbase + 16 * e)
        if kl == 0:
            for r in range_constexpr(rows):
                fx.ptr_store(core[r], at("core", r) + row)
        gpu.barrier()

        # The head's 128 core rows from the four siblings (this split's own
        # included), their square sum in the native order, then the
        # gated RMSNorm of the split's 32 rows and their MXFP8 block: lane
        # ``l < 4`` packs rows 8 l .. 8 l + 7 (the other lanes repeat them).
        # Batched builds: threads 16 r + j publish row r's, wave r finishes it.
        if tid < 16 * rows:
            j, rj = (tid, 0) if rows == 1 else (tid % 16, tid // 16)
            cs = c.L.stride["core"]
            mbx.put2(
                c.mb_core,
                mb_ix(t * 32 + 2 * j, rj),
                fx.ptr_load(core_l + row_ix(2 * j, rj, cs)),
                fx.ptr_load(core_l + row_ix(2 * j + 1, rj, cs)),
                c.tag,
            )
        rw = 0 if rows == 1 else fx.min(wave, rows - 1)
        epilogue = (wave == 0) if rows == 1 else (wave < rows)
        if epilogue:
            cores_r, core_r, vz_r = at("cores", rw), at("core", rw), at("vz", rw)
            got = mbx.poll([(c.mb_core, mb_ix(vh * 128 + 2 * lane, rw), 2)], c.tag)[0]
            fx.ptr_store(got[0].bitcast(fx.Float32), cores_r + 2 * lane)
            fx.ptr_store(got[1].bitcast(fx.Float32), cores_r + (2 * lane + 1))
            tot = sumsq64(cores_r, lane) + sumsq64(cores_r + 64, lane)
            inv_rms = rsq(tot * (1.0 / C.delta_v_dim) + C.eps)
            r0 = (lane % 4) * 8
            cv = lds_f4(core_r, r0) + lds_f4(core_r, r0 + 4)
            zv = lds_f4(vz_r, 32 + r0) + lds_f4(vz_r, 32 + r0 + 4)
            wv = bf16x8([nw[e] for e in range(4)])
            out = [
                bf16r(((cv[e] * inv_rms) * wv[e]) * sigmoid_fast(zv[e]))
                for e in range(8)
            ]
            amax = fmath.absf(out[0])
            for e in range_constexpr(1, 8):
                amax = fx.max(amax, fmath.absf(out[e]))
            amax = xred(xred(amax, 1, fx.max), 2, fx.max)
            ex, scale = p2.e8m0_fp8(amax)
            lo = p2.fp8_word(out[0], out[1], out[2], out[3], scale)
            hi = p2.fp8_word(out[4], out[5], out[6], out[7], scale)
            if lane < 4:
                mbx.put2(c.mb_d8, mb_ix(vh * 32 + s * 8 + lane * 2, rw), lo, hi, c.tag)
            if lane == 0:
                mbx.put(c.mb_de, mb_ix(t, rw), ex, c.tag)
        for ek in range_constexpr(1, -(-rows // WAVES)):
            _b3_out(c, nw, t, vh, s, ek)


b3 = ASTRewriter.transform(b3)
