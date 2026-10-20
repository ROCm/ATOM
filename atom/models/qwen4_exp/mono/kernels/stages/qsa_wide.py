"""Wide builds' Q1 (``layout_wide.for_wide``): qsa.py's ``q1`` with the
mixed rows polled in rounds and every request row's MFMA result stored from
its lanes (``common_wide``). Past ``WAVES`` rows: Q1's epilogue and Q2 in
passes of ``WAVES`` request rows (:func:`_q1_out`, :func:`q2`), and Q6 in
task rounds (:func:`q6`).

Derived from qsa.py; qsa.py stays the source.
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp.mono.kernels import buffer_ops as bo
from atom.models.qwen4_exp.mono.kernels import mailbox as mbx
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.models.qwen4_exp.mono.kernels.config import GRID, THREADS
from atom.models.qwen4_exp.mono.kernels.device_ops import rsrc, xred
from atom.models.qwen4_exp.mono.kernels.layout import (
    SCORE_TILE,
    SCORE_TILES,
    SS_ROW,
    WAVES,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    bf16r,
    mb_ix,
    mfma_operands,
    p2_issue,
    p2_mma,
    row_ix,
    task_index,
    task_rounds,
    wave_max,
    wave_sum,
)
from atom.models.qwen4_exp.mono.kernels.stages.common_wide import (
    fresh_round,
    poll_mixed_wide,
    store_rows,
)
from atom.models.qwen4_exp.mono.kernels.stages.hc import OOB
from atom.models.qwen4_exp.mono.kernels.stages.moe import mexp_fma
from atom.models.qwen4_exp.mono.kernels.stages.qsa import (
    GROUP,
    HEAD_WORDS,
    INDEX_WORDS,
    NEG_MAX,
    P_IK,
    P_K,
    P_Q,
    P_V,
    PP_ROW,
    Q1_GROUPS,
    Q1_KI,
    R256,
    ROPE_WORDS,
    _request,
    _theirs,
    list_len,
    norm,
    page_block,
    qsa_row,
    rope,
    row_len,
    unpack,
)


def _q1_out(c, red, rs, t, ek):
    """Pass ``ek`` of Q1's epilogue past ``WAVES`` rows: wave ``w`` runs
    request row ``w + 8 ek``'s, as ``q1``'s wave ``r`` row ``r``'s."""
    lane = c.lane
    q = c.wave + WAVES * ek
    rw = fx.min(q, c.rows - 1)
    if q < c.rows:
        proj = bf16r(fx.ptr_load(red + row_ix(lane, rw, rs)))
        other = xred(proj, 1, _theirs)
        if lane % 2 == 0:
            at = mb_ix(t * 32 + lane // 2, rw)
            mbx.put(c.mb_qp, at, mbx.bf16_pair(proj, other), c.tag)


_q1_out = ASTRewriter.transform(_q1_out)


def q1(c, p):
    lane, wave = c.lane, c.wave
    t = task_index(c, "q1")
    x8 = c.lds_i + c.L.lds["x8"]
    xe = c.lds_i + c.L.lds["xe"]
    red = c.lds_f + c.L.lds["red"]
    rs = c.L.stride["red"]
    # Batched builds: wave ``r < rows`` runs request row r's epilogue.
    rw = 0 if c.rows == 1 else fx.min(wave, c.rows - 1)

    if t < c.L.tasks["q1"]:
        # Wave ``w < 4``: n-block ``4 t + w``, one chain over the 20 k_iters;
        # waves 4-7 load out of range.
        nb = t * 4 + wave % 4
        off = (wave < 4).select(fx.Int32(0), fx.Int32(OOB))
        units = [
            p2_issue(p.w_qsa, p.s_qsa, nb, Q1_KI, g, lane, off)
            for g in range(Q1_GROUPS)
        ]
        poll_mixed_wide(c, x8, xe)

        if wave < 4:
            xr, er = mfma_operands(c, x8, xe)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for g in range_constexpr(Q1_GROUPS):
                acc = p2_mma(acc, units[g], g, xr, er, lane)
            store_rows(c, acc, red, wave * 16 + lane % 16, rs)
        gpu.barrier()

        epilogue = (wave == 0) if c.rows == 1 else (wave < c.rows)
        if epilogue:
            proj = bf16r(fx.ptr_load(red + row_ix(lane, rw, rs)))
            other = xred(proj, 1, _theirs)
            if lane % 2 == 0:
                at = mb_ix(t * 32 + lane // 2, rw)
                mbx.put(c.mb_qp, at, mbx.bf16_pair(proj, other), c.tag)
        for ek in range_constexpr(1, -(-c.rows // WAVES)):
            _q1_out(c, red, rs, t, ek)


q1 = ASTRewriter.transform(q1)


def _q2(c, p, t, ek):
    lane = c.lane
    # Wave ``w``: request row ``w + 8 ek``.
    q = c.wave + WAVES * ek
    rw = fx.min(q, c.rows - 1)
    L = row_len(p, rw)
    a_pages = qsa_row(p, rw, "pages")
    a_key = qsa_row(p, rw, "key")
    a_value = qsa_row(p, rw, "value")
    a_index_key = qsa_row(p, rw, "index_key")
    a_cos = qsa_row(p, rw, "cos")
    a_sin = qsa_row(p, rw, "sin")

    if t < c.L.tasks["q2"]:
        mine = q < c.rows
        if mine:
            row_l = 4 * page_block(rsrc(a_pages), L // 4, True) + L % 4
            is_q = t < C.q_heads
            kv = fx.max(t - C.q_heads, 0)
            first = is_q.select(P_Q + 2 * HEAD_WORDS * t, P_K + HEAD_WORDS * kv)
            r_w = rsrc(is_q.select(p.a_q_norm, p.a_k_norm))
            w = fx.Vector(bo.buffer_load(r_w, 2 * lane, vec_width=2, dtype=T.i32))
            tab = L * ROPE_WORDS + 2 * fx.min(lane, C.rotary_dim // 4 - 1)
            cs = fx.Vector(bo.buffer_load(rsrc(a_cos), tab, vec_width=2, dtype=T.i32))
            sn = fx.Vector(bo.buffer_load(rsrc(a_sin), tab, vec_width=2, dtype=T.i32))
            x = mbx.poll([(c.mb_qp, mb_ix(first + 2 * lane, rw), 2)], c.tag)[0]

            y = norm(unpack(x), unpack([w[0], w[1]]), R256, False)
            y = rope(
                y,
                unpack([cs[0], cs[1]]),
                unpack([sn[0], sn[1]]),
                lane,
                8,
                lane < C.rotary_dim // 4,
            )
            out = [mbx.bf16_pair(y[0], y[1]), mbx.bf16_pair(y[2], y[3])]
            mbx.put2(
                c.mb_qn, mb_ix(t * HEAD_WORDS + 2 * lane, rw), out[0], out[1], c.tag
            )

            if t >= C.q_heads:
                at = (row_l * C.kv_heads + kv) * HEAD_WORDS + 2 * lane
                bo.buffer_store(fx.Vector.from_elements(out, fx.Int32), rsrc(a_key), at)
                v = mbx.poll(
                    [(c.mb_qp, mb_ix(P_V + HEAD_WORDS * kv + 2 * lane, rw), 2)], c.tag
                )[0]
                bo.buffer_store(fx.Vector.from_elements(v, fx.Int32), rsrc(a_value), at)
            if t == C.q_heads:
                ik = mbx.poll([(c.mb_qp, mb_ix(P_IK + lane, rw), 1)], c.tag)[0][0]
                bo.buffer_store(ik, rsrc(a_index_key), row_l * INDEX_WORDS + lane)


_q2 = ASTRewriter.transform(_q2)


def q2(c, p):
    """``qsa.q2`` past ``WAVES`` rows: pass ``ek`` (request rows ``8 ek ..``)
    on fresh ids and mailbox addresses (``common_wide.fresh_round``)."""
    t = task_index(c, "q2")
    for ek in range(-(-c.rows // WAVES)):
        _q2(fresh_round(c) if ek else c, p, t, ek)


def _q6(c, p, t):
    tid, lane, wave = fx.Int32(c.tid), c.lane, c.wave
    eb = c.lds_f + c.L.lds["q6_e"]
    red = c.lds_f + c.L.lds["q6_red"]

    # Batched builds: tasks (request row, head).
    rq, hq = _request(c, t, C.q_heads)
    L = row_len(p, rq)

    if t < c.L.tasks["q6"]:
        kv = hq // GROUP
        cb, n, ak = list_len(L)
        nt = (ak + SCORE_TILE - 1) // SCORE_TILE
        mbx.pre_poll(
            lambda m: (c.mb_ssh, mb_ix(2 * fx.min(m, nt - 1) + kv, rq)),
            SCORE_TILES,
            c.tag,
        )
        # Thread ``tid`` holds positions 2 (tid + 512 r) and the next, r < 3.
        top = SCORE_TILE * nt - 2
        got = mbx.poll(
            [
                (
                    c.mb_ss,
                    mb_ix(hq * SS_ROW + fx.min(2 * (tid + THREADS * r), top), rq),
                    2,
                )
                for r in range(3)
            ],
            c.tag,
        )
        s = []
        for r in range_constexpr(3):
            for e in range_constexpr(2):
                j = 2 * (tid + THREADS * r) + e
                s.append(
                    (j < ak).select(got[r][e].bitcast(fx.Float32), fx.Float32(NEG_MAX))
                )

        m = s[0]
        for k in range_constexpr(1, 6):
            m = fx.max(m, s[k])
        m = wave_max(m)
        if lane == 0:
            fx.ptr_store(m, red + wave)
        gpu.barrier()
        m = fx.ptr_load(red + 0)
        for w in range_constexpr(1, WAVES):
            m = fx.max(m, fx.ptr_load(red + w))
        e = [(x <= NEG_MAX).select(fx.Float32(0.0), mexp_fma(x - m)) for x in s]
        for r in range_constexpr(2):
            at = 2 * (tid + THREADS * r)
            fx.ptr_store(
                fx.Vector.from_elements(e[2 * r : 2 * r + 2], fx.Float32), eb + at
            )
        if tid < 128:
            fx.ptr_store(
                fx.Vector.from_elements(e[4:6], fx.Float32),
                eb + (2 * THREADS * 2 + 2 * tid),
            )
        gpu.barrier()

        if wave < 4:
            loc = fx.Float32(0.0)
            for k in range_constexpr(9):
                loc = loc + fx.ptr_load(eb + (tid + 256 * k))
            loc = wave_sum(loc)
            if lane == 0:
                fx.ptr_store(loc, red + (WAVES + wave))
        gpu.barrier()
        den = fx.Float32(0.0)
        for w in range_constexpr(4):
            den = den + fx.ptr_load(red + (WAVES + w))
        inv = fx.Float32(1.0) / den
        for r in range_constexpr(3):
            j = 2 * (tid + THREADS * r)
            if j < SCORE_TILE * nt:
                pw = mbx.bf16_pair(e[2 * r] * inv, e[2 * r + 1] * inv)
                mbx.put(c.mb_pp, mb_ix(hq * PP_ROW + j // 2, rq), pw, c.tag)
        gpu.barrier()
        if tid == 0:
            mbx.put(c.mb_pph, mb_ix(hq, rq), 1, c.tag)


_q6 = ASTRewriter.transform(_q6)


def q6(c, p):
    """``qsa.q6`` past ``WAVES`` rows, more tasks than CTAs: round ``k`` runs
    task ``task_index + k GRID`` on fresh ids and mailbox addresses; a
    round's last CTA barrier is past its LDS reads."""
    t = task_index(c, "q6")
    for k in range(task_rounds(c, "q6")):
        _q6(fresh_round(c) if k else c, p, t if k == 0 else t + k * GRID)
