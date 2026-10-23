"""Grouped verify QSA stages (``build_layer(..., group=S)``): the MTP verify
rows of several requests in one launch.

The launch's ``rows`` are ``rows / S`` groups of ``S`` rows, group ``g`` rows
``g S .. g S + S - 1``: request ``g``'s tokens ``L_g .. L_g + S - 1``, as a
seq build's rows (``qsa`` module docstring), and row ``r`` is the ``j = r %
S``-th of its group. The ``qsa`` table holds every row's entry (the group's
pages and caches, each row's ``pos``); only each group's first ``pos`` is read
(:func:`qsa_len`). Tasks are a batched build's (one task set per row).

Q1, Q2, Q4 and Q6 run the batched stages. Q3, Q5 and Q7 here are the seq
stages restricted to the row's group: they take the group's tokens' index
keys, keys and values from the group's rows' mailboxes instead of memory,
matched by the group's cache rows, and nothing reads another group's. So
every group's rows are bit-identical to its own seq launch's.
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels import p2
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import GRID, THREADS
from atom.models.qwen4_exp.mono.kernels.device_ops import (
    opaque,
    rsrc,
    uniform,
    xred,
    xshfl,
)
from atom.models.qwen4_exp.mono.kernels.layout import (
    K_STRIDE,
    SCORE_TILE,
    SS_ROW,
    TASKS,
    WAVES,
)
from atom.models.qwen4_exp.mono.kernels.stages import qsa_fast
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    add,
    bf16r,
    bf16x8,
    mb_ix,
    row_ix,
    task_index,
    task_rounds,
)
from atom.models.qwen4_exp.mono.kernels.stages.common_wide import fresh_round
from atom.models.qwen4_exp.mono.kernels.stages.hc import _f32, fma, sigmoid_acc
from atom.models.qwen4_exp.mono.kernels.stages.qsa import (
    GROUP,
    HEAD_WORDS,
    INDEX_WORDS,
    NEG_MAX,
    P_IK,
    P_Q,
    P_V,
    PP_ROW,
    Q1_INDEX_TASKS,
    Q7_HEADS,
    Q7_ROUNDS,
    QN_SEQ,
    R128,
    ROPE_WORDS,
    SCALE,
    SEL,
    SELH_FROM_QN,
    STATE,
    _cache_row,
    _entry,
    _lds_i4,
    _request,
    list_len,
    norm,
    page_block,
    qsa_row,
    row_len,
    unpack,
    word,
)
from atom.models.qwen4_exp.mono.kernels.stages.qsa_fast import (
    _block_key,
    _cached_scores,
    _rope16,
    _score,
)


def qsa_len(p, rows, group):
    """``qsa.qsa_len`` of a grouped build: ``p.Ls`` every row's ``L``, row
    ``g S + j``'s ``L_g + j`` from its group's first row's ``pos``; ``p.L``
    row 0's. Call at kernel entry (a wide build: at the start of each stage
    that reads it), outside runtime branches."""
    r = rsrc(p.a_qsa)
    p.seq = False
    p.Ls = []
    for g in range(rows // group):
        pos = _entry(r, row_ix(STATE.index("pos"), g * group, len(STATE)))
        Lg = uniform(bo.buffer_load(rsrc(pos), 0, vec_width=1, dtype=T.i32))
        p.Ls += [Lg] + [Lg + j for j in range(1, group)]
    p.L = p.Ls[0]


def _group_of(c, rq):
    """``(gb, j)``: the first row of request row ``rq``'s group and ``rq``'s
    index in it (``rq`` wave-uniform)."""
    j = rq % c.group
    return rq - j, j


def _now_rows(c, r_pg, L, j):
    """``(tail, now)``: ``qsa._now_rows`` of the ``j``-th row of a group (cache
    length ``L``): ``now`` lists the cache rows of the group's tokens ``L - j
    .. L - j + S - 1``."""
    tail = page_block(r_pg, L // 4, True)
    now = []
    for s in range(c.group):
        Ls = L - j + s
        now.append(4 * page_block(r_pg, Ls // 4, True) + Ls % 4)
    return tail, now


def q3(c, p):
    qsa_fast._mailboxes(c)
    lane, wave = c.lane, c.wave
    t = task_index(c, "q3")
    iqn = c.lds_f + c.L.lds["iqn"]
    per = c.L.tasks["q3"] // c.rows
    rq, tq = _request(c, t, per)
    L = row_len(p, rq)
    gb, j = _group_of(c, rq)
    a_pages = qsa_row(p, rq, "pages")
    a_index_key = qsa_row(p, rq, "index_key")
    a_index_cmp = qsa_row(p, rq, "index_cmp")
    a_cos = qsa_row(p, rq, "cos")
    a_sin = qsa_row(p, rq, "sin")
    cb = (L + 1) // 4

    # Tasks with no block skip the stage; Q4 waits on the others' hints, and
    # on task 0's at any context (a row at positions 0-2 has no block).
    if (t < c.L.tasks["q3"]) & (tq * WAVES < fx.max(cb, 1)):
        r_pg = rsrc(a_pages)
        r_ik = rsrc(a_index_key)
        r_cmp = rsrc(a_index_cmp)
        r_cos = rsrc(a_cos)
        r_sin = rsrc(a_sin)
        rope_word = fx.min(lane, ROPE_WORDS - 1)
        wq = unpack([word(rsrc(p.a_iq_norm), lane)])
        wk = unpack([word(rsrc(p.a_ik_norm), lane)])
        cs = unpack([word(r_cos, L * ROPE_WORDS + rope_word)])
        sn = unpack([word(r_sin, L * ROPE_WORDS + rope_word)])
        mbx.pre_poll(lambda m: (c.mb_qp, mb_ix(32 * m, rq)), Q1_INDEX_TASKS, c.tag)

        if wave < C.index_heads:
            x = mbx.poll([(c.mb_qp, mb_ix(INDEX_WORDS * wave + lane, rq), 1)], c.tag)[0]
            y = norm(unpack(x), wq, R128, True)
            y = _rope16(y, cs, sn, lane, lane < ROPE_WORDS)
            for k in range_constexpr(2):
                fx.ptr_store(y[k], iqn + (wave * C.index_dim + 2 * lane + k))
        got = mbx.poll(
            [(c.mb_qp, mb_ix(P_IK + lane, gb + s), 1) for s in range(c.group)], c.tag
        )
        iks = [v[0] for v in got]
        gpu.barrier()
        q = [
            fx.ptr_load(iqn + (h * C.index_dim + 2 * lane + e))
            for h in range(4)
            for e in range(2)
        ]

        # As ``qsa_fast.q3``: the blocks below the group's first token's from
        # ``index_cmp``, the rest from their raw keys.
        nw = per * WAVES
        first = tq * WAVES + wave
        nc = (L - j) // 4
        _cached_scores(c, rq, q, r_pg, r_cmp, first, nc, nw)
        b = nc + first
        if b < cb:
            g = page_block(r_pg, b, True)
            kws = [word(r_ik, (4 * g + r) * INDEX_WORDS + lane) for r in range(4)]
            at = 4 * b * ROPE_WORDS + rope_word
            kc, ks = word(r_cos, at), word(r_sin, at)
            raw = []
            for r in range_constexpr(4):
                row = 4 * b + r
                kw = kws[r]
                for s in range_constexpr(c.group):
                    kw = (row == L - j + s).select(iks[s], kw)
                raw.append(unpack([fx.Int32(kw)]))
            kn = _block_key(raw, wk, kc, ks, lane)
            score = _score(q, kn)
            if lane == 0:
                mbx.put(c.mb_isc, mb_ix(b, rq), score, c.tag)
            # Row L completes block b: its key for ATOM's compressed cache.
            if 4 * b + 3 == L:
                bo.buffer_store(
                    mbx.bf16_pair(kn[0], kn[1]), r_cmp, g * INDEX_WORDS + lane
                )

        gpu.barrier()
        if c.tid == 0:
            mbx.put(c.mb_fisch, mb_ix(tq, rq), 1, c.tag)


q3 = ASTRewriter.transform(q3)


def _q5(c, p, tid, t, turn):
    qs = c.lds_i + c.L.lds["q5_q"]
    ks = c.lds_i + c.L.lds["q5_k"]
    sl = c.lds_i + c.L.lds["q5_sel"]
    # Tasks (tile, request row, kv head), tile-major, as a batched build's,
    # the rows turned ``turn`` on (``q5``).
    rq = (t // 2) % c.rows if turn is None else opaque((t // 2 + turn) % c.rows)
    L = row_len(p, rq)
    gb, j = _group_of(c, rq)
    a_pages = qsa_row(p, rq, "pages")
    a_key = qsa_row(p, rq, "key")

    if t < c.L.tasks["q5"]:
        kv = t % 2
        tile = t // (2 * c.rows)
        _, n, ak = list_len(L)
        if tile * SCORE_TILE < ak:
            r_pg = rsrc(a_pages)
            tail, now_row = _now_rows(c, r_pg, L, j)

            def hint(m):
                # The q heads, the group's rows' k heads, then selh.
                head = (m < GROUP).select(GROUP * kv + m, C.q_heads + kv)
                kq = (m < GROUP).select(
                    rq, gb + fx.min(fx.max(m - GROUP, 0), c.group - 1)
                )
                return c.mb_qn, (m < GROUP + c.group).select(
                    mb_ix(HEAD_WORDS * head, kq), mb_ix(SELH_FROM_QN, rq)
                )

            mbx.pre_poll(hint, GROUP + c.group + 1, c.tag)
            if (tid < SCORE_TILE // 4) & (tile * (SCORE_TILE // 4) + tid < n):
                b = mbx.poll(
                    [(c.mb_sel, mb_ix(tile * (SCORE_TILE // 4) + tid, rq), 1)], c.tag
                )[0][0]
                fx.ptr_store(page_block(r_pg, b), sl + tid)
            specs, dst = [], []
            for r in range_constexpr(2):
                s = fx.min(tid + THREADS * r, QN_SEQ[c.group] - 1)
                hh = s // (HEAD_WORDS // 2)
                w = 2 * (s % (HEAD_WORDS // 2))
                head = (hh < GROUP).select(GROUP * kv + hh, C.q_heads + kv)
                # Head GROUP + q: the group's row q's k head.
                qn_at = mb_ix(
                    head * HEAD_WORDS + w, (hh < GROUP).select(rq, gb + hh - GROUP)
                )
                specs.append((c.mb_qn, qn_at, 2))
                dst.append(hh * HEAD_WORDS + w)
            got = mbx.poll(specs, c.tag)
            for r in range_constexpr(2):
                fx.ptr_store(fx.Vector.from_elements(got[r], fx.Int32), qs + dst[r])
            gpu.barrier()

            r_k = rsrc(a_key)
            quad = tid % 32
            curs = [
                _lds_i4(qs, (GROUP + q) * HEAD_WORDS + 4 * quad) for q in range(c.group)
            ]
            for r in range_constexpr(2):
                row = tid // 32 + 16 * r
                kr = _cache_row(
                    fx.ptr_load(sl + row // 4), tile * SCORE_TILE + row, n, ak, tail
                )
                at = (kr * C.kv_heads + kv) * HEAD_WORDS + 4 * quad
                kw = fx.Vector(bo.buffer_load(r_k, at, vec_width=4, dtype=T.i32))
                # The group's tokens from its rows' Q2s, not memory.
                w4 = [kw[e] for e in range(4)]
                for q in range_constexpr(c.group):
                    hit = kr == now_row[q]
                    w4 = [fx.Int32(hit.select(curs[q][e], w4[e])) for e in range(4)]
                fx.ptr_store(
                    fx.Vector.from_elements(w4, fx.Int32),
                    ks + (row * K_STRIDE + 4 * quad),
                )
            gpu.barrier()

            if tid < GROUP * SCORE_TILE:
                h = tid // SCORE_TILE
                i = tid % SCORE_TILE
                acc = fx.Float32(0.0)
                for d in range_constexpr(HEAD_WORDS // 4):
                    qf = bf16x8(_lds_i4(qs, h * HEAD_WORDS + 4 * d))
                    kf = bf16x8(_lds_i4(ks, i * K_STRIDE + 4 * d))
                    for e in range_constexpr(8):
                        acc = fma(qf[e], kf[e], acc)
                j = tile * SCORE_TILE + i
                score = (j < ak).select(acc * _f32(SCALE), fx.Float32(NEG_MAX))
                other = xshfl(score, 1)
                if i % 2 == 0:
                    mbx.put2(
                        c.mb_ss,
                        mb_ix((GROUP * kv + h) * SS_ROW + j, rq),
                        score,
                        other,
                        c.tag,
                    )
            gpu.barrier()
            if tid == 0:
                mbx.put(c.mb_ssh, mb_ix(2 * tile + kv, rq), 1, c.tag)


_q5 = ASTRewriter.transform(_q5)


def q5(c, p):
    tid = fx.Int32(c.tid)
    t = task_index(c, "q5")
    # A CTA's rounds run tasks GRID apart: where 2 rows divides GRID, all of
    # one request row, so a long request's tiles would fall on the same CTAs
    # every round. Round k turns the rows k groups on instead (a tile's tasks
    # lie in one round, so the turn permutes its rows), each round's row
    # ``opaque``: a row's addresses kept for its next round spill SGPRs at
    # 16 rows.
    turn = c.group if GRID % (2 * c.rows) == 0 else None
    for k in range_constexpr(task_rounds(c, "q5")):
        rot = None if turn is None else k * turn % c.rows
        _q5(c, p, tid, t if k == 0 else t + k * GRID, rot)


def _q7(c, p, tid, lane, wave, t):
    sl = c.lds_i + c.L.lds["q7_sel"]
    pb = c.lds_f + c.L.lds["q7_p"]
    cur = c.lds_i + c.L.lds["q7_cur"]
    part = c.lds_f + c.L.lds["q7_part"]
    # Tasks (request row, x), as a batched build's.
    rq, x = _request(c, t, TASKS["q7"])
    L = row_len(p, rq)
    gb, j = _group_of(c, rq)
    a_pages = qsa_row(p, rq, "pages")
    a_value = qsa_row(p, rq, "value")

    if t < c.L.tasks["q7"]:
        dt = x % 8
        kv = x // 24
        qh0 = GROUP * kv + Q7_HEADS * ((x // 8) % 3)
        _, n, ak = list_len(L)
        r_pg = rsrc(a_pages)
        tail, now_row = _now_rows(c, r_pg, L, j)
        nt = (ak + SCORE_TILE - 1) // SCORE_TILE
        own = tid % 128  # the owner this thread stands in for
        hd = own // 32
        dim = own % 32

        mbx.pre_poll(lambda m: (c.mb_selh, mb_ix(0, rq)), 1, c.tag)
        if tid < n:
            b = mbx.poll([(c.mb_sel, mb_ix(tid, rq), 1)], c.tag)[0][0]
            fx.ptr_store(page_block(r_pg, b), sl + tid)
        # The current token's values and the gate, in one poll outside any
        # divergent branch (one inside turns qp's resource into VGPRs and
        # the gate poll into a waterfall loop).
        # Thread 8 q + i polls the group's row q's (token L - j + q's) values.
        ct = fx.min(tid, 8 * c.group - 1)
        cur_at = P_V + HEAD_WORDS * kv + 16 * dt + 2 * (ct % 8)
        cur_row = gb + ct // 8
        gate_at = P_Q + 2 * HEAD_WORDS * (qh0 + hd) + HEAD_WORDS + 16 * dt + dim // 2
        v, g = mbx.poll(
            [(c.mb_qp, mb_ix(cur_at, cur_row), 2), (c.mb_qp, mb_ix(gate_at, rq), 1)],
            c.tag,
        )
        gw = g[0]
        if tid < 8 * c.group:
            fx.ptr_store(fx.Vector.from_elements(v, fx.Int32), cur + 2 * tid)
        gpu.barrier()

        # Values of positions 128 i + slot, dims 8 dq .. 8 dq + 7 of the tile
        # (zero from ``ak`` on), ready before the probabilities are waited on.
        slot = 16 * wave + lane // 4
        dq = lane % 4
        r_v = rsrc(a_value)
        vq = []
        for i in range_constexpr(Q7_ROUNDS):
            j = 128 * i + slot
            vr = _cache_row(fx.ptr_load(sl + fx.min(j // 4, SEL - 1)), j, n, ak, tail)
            at = (vr * C.kv_heads + kv) * HEAD_WORDS + 16 * dt + 4 * dq
            vw = fx.Vector(bo.buffer_load(r_v, at, vec_width=4, dtype=T.i32))
            vq.append((vw, vr))
        words = []
        now_vs = [_lds_i4(cur, 16 * q + 4 * dq) for q in range(c.group)]
        for i in range_constexpr(Q7_ROUNDS):
            w, tk = vq[i]
            live = 128 * i + slot < ak
            pick = [w[e] for e in range(4)]
            for q in range_constexpr(c.group):
                hit = tk == now_row[q]
                pick = [fx.Int32(hit.select(now_vs[q][e], pick[e])) for e in range(4)]
            words.append(
                [fx.Int32(live.select(pick[e], fx.Int32(0))) for e in range(4)]
            )

        mbx.pre_poll(
            lambda m: (c.mb_pph, mb_ix(qh0 + fx.min(m, Q7_HEADS - 1), rq)),
            Q7_HEADS,
            c.tag,
        )
        specs, dst = [], []
        for r in range_constexpr(-(-Q7_HEADS * PP_ROW // (2 * THREADS))):
            s = fx.min(tid + THREADS * r, Q7_HEADS * PP_ROW // 2 - 1)
            h4 = s // (PP_ROW // 2)
            w0 = 2 * (s % (PP_ROW // 2))
            at = (qh0 + h4) * PP_ROW + fx.min(w0, SCORE_TILE // 2 * nt - 2)
            specs.append((c.mb_pp, mb_ix(at, rq), 2))
            dst.append(h4 * SS_ROW + 2 * w0)
        got = mbx.poll(specs, c.tag)
        for r in range_constexpr(len(dst)):
            fx.ptr_store(
                fx.Vector.from_elements(unpack(got[r]), fx.Float32), pb + dst[r]
            )
        gpu.barrier()

        acc = [fx.Float32(0.0)] * (8 * Q7_HEADS)
        for i in range_constexpr(Q7_ROUNDS):
            j = 128 * i + slot
            vf = bf16x8(words[i])
            for h4 in range_constexpr(Q7_HEADS):
                pj = fx.ptr_load(pb + (h4 * SS_ROW + fx.min(j, SS_ROW - 1)))
                pj = (j < ak).select(pj, fx.Float32(0.0))
                for e in range_constexpr(8):
                    acc[8 * h4 + e] = fma(pj, vf[e], acc[8 * h4 + e])
        for off in (4, 8, 16, 32):
            acc = [xred(x, off, add) for x in acc]
        if lane < 4:
            for k in range_constexpr(2 * Q7_HEADS):
                at = wave * 128 + (k // 2) * 32 + 8 * lane + 4 * (k % 2)
                fx.ptr_store(
                    fx.Vector.from_elements(acc[4 * k : 4 * k + 4], fx.Float32),
                    part + at,
                )
        gpu.barrier()
        a = fx.ptr_load(part + own)
        for w in range_constexpr(1, WAVES):
            a = a + fx.ptr_load(part + (w * 128 + own))

        if wave < 2:
            qh = qh0 + hd
            gl, gh = mbx.bf2_f32(gw)
            sig = bf16r(sigmoid_acc((dim % 2 == 0).select(gl, gh)))
            out = bf16r(bf16r(a) * sig)
            amax = fmath.absf(out)
            for off in (1, 2, 4, 8, 16):
                amax = xred(amax, off, fx.max)
            ex, scale = p2.e8m0_fp8(amax)
            x1 = xshfl(out, 1)
            wd = p2.fp8_word(out, x1, xshfl(out, 2), xshfl(x1, 2), scale)
            if dim % 4 == 0:
                mbx.put(c.mb_d8, mb_ix(64 * qh + 8 * dt + dim // 4, rq), wd, c.tag)
            if dim == 0:
                mbx.put(c.mb_de, mb_ix(8 * qh + dt, rq), ex, c.tag)


_q7 = ASTRewriter.transform(_q7)


def q7(c, p):
    tid, lane, wave = fx.Int32(c.tid), c.lane, c.wave
    t = task_index(c, "q7")
    for k in range_constexpr(task_rounds(c, "q7")):
        if const_expr(k == 0):
            _q7(c, p, tid, lane, wave, t)
        else:
            # Wide builds' second round, on its own ids (``fresh_round``).
            ck = fresh_round(c)
            _q7(ck, p, fx.Int32(ck.tid), ck.lane, ck.wave, t + k * GRID)
