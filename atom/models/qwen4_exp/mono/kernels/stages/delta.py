"""DeltaNet stages: B1 (QKVZ projection + causal conv) and B3 (gated delta rule).

B1, task ``t`` (256): rows ``64 t .. 64 t + 63`` of the fused QKVZ weight over
K 2560. Waves 0-3 each own n-block ``w`` and run one chain over all 20 k_iters
in order; waves 4-7 load out of range and skip the GEMM. QKV tiles (``t < 160``) run the conv + SiLU epilogue and shift the conv
state; z tiles publish the raw bf16 projection. The conv state is ATOM's,
time-major ``[3, QKV]`` bf16, oldest first: row ``k`` holds tap ``k`` of every
channel. A lane loads the word its channel shares with its pair's and keeps its
half; the even lane of a pair stores the pair's words of the next state.

B3, task ``u`` (192) = value head ``vh = u // 4``, rows ``32 s .. 32 s + 31``
(``s = u % 4``). Thread ``tid`` owns state row ``tid // 16`` and keys
``l + 16 i`` (``l = tid % 16``, ``i = 0..7``), so the fma chains and 16-lane
trees of the recurrence are the native ones. It also computes ``b[vh]`` and
``a[vh]`` itself from the bf16 mixed row in the native 16-part order (the
router's split), on waves 1 and 2 while wave 0 waits for B1, so there is no
separate BA stage. The four siblings of a head exchange their 32 core rows:
each reduces the head's 128 in the native order.
One split is exactly one MXFP8 block of the gated output, so its exponent
needs no other task.
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
from atom.models.qwen4_exp.mono.kernels.device_ops import (
    read_lane_i32,
    rsq,
    xred,
    xshfl,
)
from atom.models.qwen4_exp.mono.kernels.layout import (
    RW_PART,
    RW_ROW,
    RX_PART,
    b3_tiles,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    add,
    bf16_bits,
    bf16r,
    bf16x8,
    delta_state,
    lds_f4,
    lds_row,
    mb_ix,
    mfma_operands,
    p2_issue,
    p2_mma,
    pick_row,
    poll_mixed,
    row_ix,
    sigmoid_fast,
    softplus_fast,
    sum16,
    task_index,
)
from atom.models.qwen4_exp.mono.kernels.stages.hc import OOB, fma
from atom.models.qwen4_exp.mono.kernels.stages.moe import (
    CHAIN_STEPS,
    _chain_step,
    _part,
    _store_x,
)

QKV = C.delta_qkv
B1_KI = p2.k_iters(C.hidden)  # 20
B1_GROUPS = B1_KI // 4  # 5
RSQ128 = 0x3DB504F3  # v_rsq_f32(128.0)
B3_TILES = 6  # B1 tiles one B3 task reads


def _theirs(mine, theirs):
    return theirs


def sumsq64(base, lane):
    """Sum of squares of the 64 f32 at LDS ``base``:
    ``fma(x_j, x_j, fl(x_{j+32}^2))`` on ``j = lane % 32``, then the trees over
    16 and 8, 4, 2, 1 lanes. Every lane gets the bits of the native lane 0
    (the adds are commutative, so the xor butterflies pair the same terms)."""
    j = lane % 32
    x0 = fx.ptr_load(base + j)
    x1 = fx.ptr_load(base + (j + 32))
    v = fma(x0, x0, x1 * x1)
    for off in (16, 8, 4, 2, 1):
        v = xred(v, off, add)
    return v


def _bf(bits):
    """bf16 bits (low half) -> f32."""
    return (bits << 16).bitcast(fx.Float32)


def _seq_conv(c, red, rs, lane, rw, old, proj):
    """Seq builds: row ``rw``'s three older conv taps (f32) and its state
    snapshot (three bf16, low halves). Row ``r``'s history is ``old`` (row 0's
    state) then the projections of rows ``0 .. r - 1``, all in the CTA's ``red``."""
    hist = [bf16r(fx.ptr_load(red + row_ix(lane, q, rs))) for q in range(c.rows - 1)]
    bits = old + [bf16_bits(v) for v in hist]
    hist = [_bf(b) for b in old] + hist
    taps = [pick_row(rw, [hist[r + k] for r in range(c.rows)]) for k in range(3)]
    keep = [pick_row(rw, [bits[r + 1 + k] for r in range(c.rows)]) for k in range(2)]
    return taps, keep + [bf16_bits(proj)]


def b1(c, p):
    lane, wave = c.lane, c.wave
    t = task_index(c, "b1")
    x8 = c.lds_i + c.L.lds["x8"]
    xe = c.lds_i + c.L.lds["xe"]
    red = c.lds_f + c.L.lds["red"]
    rs = c.L.stride["red"]
    # Batched builds: wave ``r < rows`` runs request row r's epilogue.
    rw = 0 if c.rows == 1 else fx.min(wave, c.rows - 1)
    if const_expr(c.seq):
        # Every row starts from row 0's state and stores its own snapshot.
        conv_in = delta_state(c, p, 0)[0]
        conv_out = delta_state(c, p, rw)[1]
    else:
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

    poll_mixed(c, x8, xe)

    if wave < 4:
        xr, er = mfma_operands(c, x8, xe)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for g in range_constexpr(B1_GROUPS):
            acc = p2_mma(acc, units[g], g, xr, er, lane)
        if lane < 16:
            for r in range_constexpr(c.rows):
                fx.ptr_store(acc[r], red + row_ix(wave * 16 + lane, r, rs))
    if const_expr(c.seq):
        # ATOM's verify window: the rows' snapshots overwrite row 0's input
        # taps in place (``conv_out[r]`` overlaps ``conv_in``), so every
        # wave's ``cs`` loads land before any wave stores past the barrier.
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
            if const_expr(c.seq):
                taps, nxt = _seq_conv(c, red, rs, lane, rw, old, proj)
                words = [n | (xshfl(n, 1) << 16) for n in nxt]
            else:
                taps = [_bf(b) for b in old]
                # The state shifts by one row: the pair's words of rows 1-2.
                words = [cs[1], cs[2], mbx.bf16_pair(proj, xshfl(proj, 1))]
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


b1 = ASTRewriter.transform(b1)


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
        view = c.lds_i if name == "baw" else c.lds_f
        ptrs[name] = [
            lds_row(c, view, name, r) for r in range(1 if name == "baw" else rows)
        ]

    def at(name, r):
        return ptrs[name][r] if isinstance(r, int) else lds_row(c, c.lds_f, name, r)

    qk, core_l, baw, bax = at("qk", 0), at("core", 0), at("baw", 0), at("bax", 0)
    tiles = b3_tiles(kh, vh, s)

    # Batched builds: producer ``m`` of a hint is request row ``m // n`` of the
    # one-row hint's ``n`` producers.
    def hint(m):
        k = m if rows == 1 else m % B3_TILES
        tile = tiles[5]
        for i in (4, 3, 2, 1, 0):
            tile = (k == i).select(tiles[i], tile)
        return c.mb_qkvz, mb_ix(tile * 32 + 31, 0 if rows == 1 else m // B3_TILES)

    # qkvz pair index of 4 consecutive values for loader thread i < 80:
    # q (32 threads), k (32), the split's v rows (8) and z rows (8). Batched
    # builds: thread ``80 r + i`` loads row r's.
    if const_expr(rows == 1):
        i, ri = fx.min(tid, 79), 0
    else:
        li = fx.min(tid, 80 * rows - 1)
        i, ri = li % 80, li // 80
    src = (i < 72).select(
        (i < 64).select(
            (i < 32).select(kh * 64 + 2 * i, C.delta_key // 2 + kh * 64 + 2 * (i - 32)),
            C.delta_key + vh * 64 + 16 * s + 2 * (i - 64),
        ),
        QKV // 2 + vh * 64 + 16 * s + 2 * (i - 72),
    )
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
            for r in range_constexpr(rows):
                _store_x(at("bax", r), xv[2 * r], 2 * bj)
                _store_x(at("bax", r), xv[2 * r + 1], 2 * bj + 1)
            part, first = _part(bj * 8)
            off = part * RW_PART + (bj * 8 - first) // 2
            fx.ptr_store(wb, baw + off)
            fx.ptr_store(wa, baw + (RW_ROW + off))
        gpu.barrier()
        sbase = (vh * C.delta_v_dim + s * 32 + row) * C.delta_k_dim + kl
        # Every row's state addresses are read here: a table read
        # issued after a row's state stores would wait for them (vmcnt counts
        # stores too), and serialize the rows.
        # Seq builds load row 0's state only; the later rows chain on it.
        st, rec_out = [], []
        for r in range_constexpr(rows):
            rec_in = delta_state(c, p, r)[2]
            rec_out.append(delta_state(c, p, r)[3])
            if const_expr(r == 0 or not c.seq):
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
        if (wave >= 1) & (wave < 3) & (lane < 16 * rows):
            q = lane if rows == 1 else lane % 16
            big = q < 8
            wrow = baw + ((wave - 1) * RW_ROW + q * RW_PART)
            xrow = bax + row_ix(
                q * RX_PART, 0 if rows == 1 else lane // 16, c.L.stride["bax"]
            )
            v4i = fx.Vector.make_type(4, fx.Int32)
            acc = fx.Float32(0.0)
            for k in range_constexpr(CHAIN_STEPS):
                wv = fx.Vector(fx.ptr_load(wrow + k * 4, result_type=v4i))
                w8 = bf16x8([wv[e] for e in range(4)])
                x8 = lds_f4(xrow, k * 8) + lds_f4(xrow, k * 8 + 4)
                new = acc
                for e in range_constexpr(8):
                    new = fma(w8[e], x8[e], new)
                acc = _chain_step(k, big, new, acc)
            bits = acc.bitcast(fx.Int32)
            for r in range_constexpr(rows):
                total = read_lane_i32(bits, 16 * r).bitcast(fx.Float32)
                for q in range_constexpr(1, 16):
                    total = total + read_lane_i32(bits, 16 * r + q).bitcast(fx.Float32)
                if lane == 0:
                    fx.ptr_store(bf16r(total), at("sums", r) + (wave - 1))

        mbx.pre_poll(hint, B3_TILES * rows, c.tag)
        if tid < 80 * rows:
            v = mbx.poll([(c.mb_qkvz, mb_ix(src, ri), 2)], c.tag)[0]
            if const_expr(rows == 1):
                dst = [qk + tid * 4] + [
                    qk + (tid * 4 + k) for k in (1, 2, 3)
                ]  # vz follows qk
            else:
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
            if wave < 2 * rows:
                nk = wave % 2
                halves = sumsq64(
                    at("qk", wave // 2) + (nk * 128 + (lane // 32) * 64), lane
                )
                rs_n = rsq(xred(halves, 32, add) + C.eps)
                rs_n = (nk == 0).select(
                    rs_n * fx.Int32(RSQ128).bitcast(fx.Float32), rs_n
                )
                if lane == 0:
                    fx.ptr_store(rs_n, at("sums", wave // 2) + (2 + nk))
            gpu.barrier()
            inv_q = [fx.ptr_load(at("sums", r) + 2) for r in R]
            inv_k = [fx.ptr_load(at("sums", r) + 3) for r in R]
        qh = [
            [fx.ptr_load(qk_r[r] + (kl + 16 * e)) * inv_q[r] for e in range(8)]
            for r in R
        ]
        kh_ = [
            [fx.ptr_load(qk_r[r] + (128 + kl + 16 * e)) * inv_k[r] for e in range(8)]
            for r in R
        ]

        # Gated delta rule on this row: S <- decay S; delta = beta (v - S k);
        # S <- S + delta k^T; core = S q.
        if const_expr(c.seq):
            # Seq builds: row r's state is row r - 1's result, one row after
            # the other.
            sn, core = [], []
            for r in range_constexpr(rows):
                sv_r = [
                    (st[0] if r == 0 else sn[r - 1])[e] * decay[r] for e in range(8)
                ]
                pr_r = fx.Float32(0.0)
                for e in range_constexpr(8):
                    pr_r = fma(sv_r[e], kh_[r][e], pr_r)
                delta_r = beta[r] * (fx.ptr_load(at("vz", r) + row) - sum16(pr_r))
                sn.append([fma(delta_r, kh_[r][e], sv_r[e]) for e in range(8)])
                cr_r = fx.Float32(0.0)
                for e in range_constexpr(8):
                    cr_r = fma(sn[r][e], qh[r][e], cr_r)
                core.append(sum16(cr_r))
        else:
            sv = [[st[r][e] * decay[r] for e in range(8)] for r in R]
            pr = [fx.Float32(0.0) for r in R]
            for e in range_constexpr(8):
                pr = [fma(sv[r][e], kh_[r][e], pr[r]) for r in R]
            pred = [sum16(x) for x in pr]
            delta = [beta[r] * (fx.ptr_load(at("vz", r) + row) - pred[r]) for r in R]
            sn = [[fma(delta[r], kh_[r][e], sv[r][e]) for e in range(8)] for r in R]
            cr = [fx.Float32(0.0) for r in R]
            for e in range_constexpr(8):
                cr = [fma(sn[r][e], qh[r][e], cr[r]) for r in R]
            core = [sum16(x) for x in cr]
        for r in range_constexpr(rows):
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


b3 = ASTRewriter.transform(b3)
