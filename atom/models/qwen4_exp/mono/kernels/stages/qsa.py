"""QSA stages (sparse full attention with the block indexer).

Q1, task ``t`` (218): rows ``64 t .. 64 t + 63`` of the fused projection
``q_index_kv_weight`` over K 2560, in this row order: index q (4 heads of
128) and index k (128), then per query head its q and gate (256 each), then k
and v (2 heads of 256 each). Waves 0-3 each own n-block ``w``
over all k_iters in order and waves 4-7 load out of range, as B1 does. The
bf16 outputs go out as pairs.

Q2, task ``t`` (26): q head ``t``, or k head ``t - 24``. Wave 0 holds the head,
lane ``l`` dims ``4 l .. 4 l + 3``: the direct-scale norm, then
RoPE at row ``L`` on lanes < 16. The head goes out as pairs (``qn``).
The k tasks also append row ``L`` of the key and value caches, and task 24
the raw index key's, in their pages. Nothing in this launch reads row ``L`` back: the
later stages take the current token from the mailboxes.

Q3 and Q4 (``qsa_fast``): every complete index block's score (``isc``),
then the top ``min(cb, 512)`` blocks by score, ties to the lower block,
published in rank order (``sel``) and then a hint (``selh``).

The selected list: position ``j < 4 n`` is token ``4 sel[j / 4] + j %
4`` (``n = min(cb, 512)``), then the incomplete block's tokens, ``ak`` in all.
Every stage takes the current token's key and value from the mailboxes.

Q5, task ``t`` (130): positions ``32 (t / 2) ..`` of the list for kv head ``t %
2``. The 12 q heads and the 32 key rows go to LDS; thread ``32 h + i`` (``h <
12``) takes the sequential fma chain over the 256 dims and publishes
score ``(12 kv + h, j)`` (``ss``; ``NEG_MAX`` past ``ak``). Each task ends with
its hint (``ssh``).

Q6, task ``h`` (24): the softmax of head ``h``: the max, ``e_j =
mexp(s_j - m)``, the 256 strided partial sums ``loc_t``, their halving sums
in waves 0-3 and the denominator in wave order; ``p_j = bf16(e_j / den)``
goes out as bf16 words (``pp``), then a hint (``pph``).

Q7, task ``x`` (48): kv head ``x / 24``,
heads ``4 ((x / 8) % 3) ..`` of its group (4), dims ``32 (x % 8) ..`` of each
head. Thread ``(w, l)`` loads the 32 values of list positions ``128 i + 16 w +
l / 4`` (8 per lane, ``l % 4``; zero from ``ak`` on) into registers before
waiting on the probabilities. Each thread sums its positions, the 16
position groups of a wave sum by butterfly and the 8 waves in wave order.
Lane ``l`` of wave ``w < 2`` owns head ``2 w + l / 32``, dim ``l % 32``: the
output gate and the MXFP8 pack into ``d8``/``de``, B3's regions, for C1.

The QSA state (cache length ``L``, caches, RoPE tables, norm weights) comes
from the table behind the ``qsa`` argument: ``L`` is loaded at kernel entry by
:func:`qsa_len`, the addresses at the start of each stage that reads them by
:func:`qsa_state`, so that they do not hold SGPRs across the other stages.

The caches are ATOM's paged ones, one page of :data:`PAGE` tokens per entry
of the request's block-table row (``pages``): token ``i`` is row ``PAGE
pages[i / PAGE] + i % PAGE``. A page holds 4 whole index blocks, so block
``b`` is row ``g = 4 pages[b / 4] + b % 4`` of the compressed key cache and
its tokens are rows ``4 g .. 4 g + 3`` of the others (:func:`page_block`).
Q2 and Q3 look up the page of ``L`` or of their block, wave-uniform; Q5 and
Q7 translate the selected blocks as they take them from ``sel``, and the
incomplete block and the current token through the page of ``L / 4``, which
always exists.

Batched builds (``c.rows`` > 1): the ``qsa`` table is ``[rows][STATE]`` and
every row's ``L`` is loaded at kernel entry (``p.Ls``). Q1 takes the rows in
its MFMA A rows; in Q2 wave ``r`` runs request row ``r``'s head. Every other
task belongs to one request row and runs the one-request code on that row's
mailboxes and caches: Q3 ``GRID / rows`` tasks per row (their blocks'
stride follows), Q4 ``qsa_fast.Q4_TASKS`` per row, Q5 tasks ``(tile, row, kv)``
(tile-major, so the tiles only long contexts use come last), Q6 ``(row,
head)``, Q7 ``(row, x)``. A stage with more tasks than CTAs runs them in
rounds (:func:`common.task_rounds`).
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
from atom.models.qwen4_exp.mono.kernels.device_ops import rsrc, uniform, xred, xshfl
from atom.models.qwen4_exp.mono.kernels.layout import (
    K_STRIDE,
    MAX_ROWS,
    PAIR,
    QSA_LIST,
    SC,
    SCORE_TILE,
    SCORE_TILES,
    SS_ROW,
    TASKS,
    WAVES,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    add,
    bf16r,
    bf16x8,
    mb_ix,
    mfma_operands,
    p2_issue,
    p2_mma,
    poll_mixed,
    row_ix,
    task_index,
    task_rounds,
    wave_max,
    wave_sum,
)
from atom.models.qwen4_exp.mono.kernels.stages.hc import (
    EPS,
    OOB,
    _f32,
    fma,
    natural_sum,
    sigmoid_acc,
)
from atom.models.qwen4_exp.mono.kernels.stages.moe import mexp_fma

Q1_KI = p2.k_iters(C.hidden)  # 20
Q1_GROUPS = Q1_KI // 4  # 5

# Pairs of the fused projection (``qp``): the 4 index q heads, the index key,
# then per head its q and gate, then the k heads and the v heads.
P_IK = C.index_heads * C.index_dim // 2  # 256
P_Q = P_IK + C.index_dim // 2  # 320: q of head h at P_Q + 256 h, its gate 128 pairs on
P_K = P_Q + C.q_heads * C.head_dim  # 6464: k head j at P_K + 128 j
P_V = P_K + C.kv_heads * C.head_dim // 2  # 6720
HEAD_WORDS = C.head_dim // 2  # 128 bf16 words per head
INDEX_WORDS = C.index_dim // 2  # 64
ROPE_WORDS = C.rotary_dim // 2  # 32 bf16 words per RoPE table row
PAGE = 16  # tokens per page of ATOM's caches (``kv_cache_block_size``)
PAGE_BLOCKS = PAGE // C.index_compress  # 4 index blocks per page
R256 = 0x3B800000  # 1/256
R128 = 0x3C000000  # 1/128
SQRT128 = 0x413504F3  # fl(sqrt(128)), a true divide
SIGN = -(2**31)
Q1_INDEX_TASKS = (P_Q + 31) // 32  # 10: the Q1 tasks holding the index q heads and key
SEL = C.index_budget // C.index_compress  # 512 selected blocks

GROUP = C.q_heads // C.kv_heads  # 12 q heads per kv head
SCALE = 0x3D800000  # 1/16
NEG_MAX = -3.4028234663852886e38  # the padding score and max seed
PP_ROW = SS_ROW // 2  # words of probabilities per head
QN_SPECS = (GROUP + 1) * HEAD_WORDS // 2  # Q5: 2-pair polls of its q heads and k head
# Seq builds: every row's k head (``rows`` of them) after the q heads.
QN_SEQ = {rows: (GROUP + rows) * HEAD_WORDS // 2 for rows in range(2, MAX_ROWS + 1)}
assert max(QN_SEQ.values()) <= 2 * THREADS  # two poll rounds
SELH_FROM_QN = (SC["selh"] - SC["qn"]) // PAIR
Q7_ROUNDS = -(-QSA_LIST // 128)  # 17 rounds of 128 list positions

# The table behind the ``qsa`` argument: one int64 per entry. ``pos`` points
# at the int32 cache length ``L``, ``pages`` at the request's int32 block-table
# row; ``key``/``value`` are the layer's ``[pages, PAGE, 2, 256]``,
# ``index_key`` its ``[pages, PAGE, 128]`` and ``index_cmp`` its ``[pages,
# PAGE_BLOCKS, 128]`` bf16 caches, ``cos``/``sin`` ``[>= L + 1, 64]`` bf16,
# then the 4 norm weights (bf16) as the host binds them.
STATE = (
    "pos",
    "pages",
    "key",
    "value",
    "index_key",
    "index_cmp",
    "cos",
    "sin",
    "q_norm",
    "k_norm",
    "iq_norm",
    "ik_norm",
)
QSA_STATE = ("q2", "q3", "q4", "q5", "q6", "q7")  # the stages that read it


def _entry(r, i):
    """Entry ``i`` of the state table ``r`` as a wave-uniform int64."""
    v = fx.Vector(bo.buffer_load(r, 2 * i, vec_width=2, dtype=T.i32))
    words = fx.Vector.from_elements([uniform(v[0]), uniform(v[1])], fx.Int32)
    return words.bitcast(fx.Int64)[0]


def qsa_len(p, rows=1, seq=False):
    """Set ``p.Ls`` (each request row's ``L``) and ``p.L`` (row 0's). Call
    once at kernel entry, outside runtime branches. Seq builds: the table
    has one row, the request's, and row ``r`` is token ``L + r``."""
    r = rsrc(p.a_qsa)
    p.seq = seq
    p.Ls = []
    for q in range(1 if seq else rows):
        pos = _entry(r, row_ix(STATE.index("pos"), q, len(STATE)))
        p.Ls.append(uniform(bo.buffer_load(rsrc(pos), 0, vec_width=1, dtype=T.i32)))
    if seq:
        p.Ls += [p.Ls[0] + q for q in range(1, rows)]
    p.L = p.Ls[0]


def row_len(p, r):
    """Request row ``r``'s ``L`` (``r`` a Python int or wave-uniform)."""
    if isinstance(r, int):
        return p.Ls[r]
    L = p.Ls[0]
    for q in range(1, len(p.Ls)):
        L = (r == q).select(p.Ls[q], L)
    return L


def qsa_row(p, r, name):
    """Request row ``r``'s address ``name`` (``r`` wave-uniform); row 0's is
    ``p.a_<name>`` from :func:`qsa_state`. Rows share the norm weights, not
    the caches or the RoPE tables (sized by each row's cache). Seq builds:
    every row is the request's."""
    if p.seq or (isinstance(r, int) and r == 0):
        return getattr(p, "a_" + name)
    return _entry(rsrc(p.a_qsa), r * len(STATE) + STATE.index(name))


def _request(c, t, per):
    """``(request row, its task)`` of task ``t`` of a stage with ``per`` tasks
    per row, rows one after another."""
    if c.rows == 1:
        return 0, t
    return fx.min(t // per, c.rows - 1), t % per


def qsa_state(p):
    """Set ``p.a_<entry>`` (wave-uniform addresses) for the :data:`STATE`
    entries after ``pos``. Call at the start of each stage that reads them,
    outside runtime branches; the loads a stage does not use are dropped."""
    r = rsrc(p.a_qsa)
    for i, name in enumerate(STATE):
        if name != "pos":
            setattr(p, "a_" + name, _entry(r, i))


def word(r, i):
    """One int32 word at index ``i`` of buffer resource ``r``."""
    return fx.Int32(bo.buffer_load(r, i, vec_width=1, dtype=T.i32))


def page_block(r_pages, b, scalar=False):
    """Index block ``b``'s row ``g`` in the paged caches, in blocks: page
    ``pages[b / 4]`` (``r_pages``), block ``b % 4`` of it. Its tokens are rows
    ``4 g .. 4 g + 3``; its compressed key is row ``g``. ``scalar`` when ``b``
    is wave-uniform."""
    page = bo.buffer_load(
        r_pages, b // PAGE_BLOCKS, vec_width=1, dtype=T.i32, is_scalar=scalar
    )
    return fx.Int32(page) * PAGE_BLOCKS + b % PAGE_BLOCKS


def unpack(words):
    """bf16 words -> f32 values, low half first."""
    out = []
    for w in words:
        out += list(mbx.bf2_f32(w))
    return out


def norm(x, w, r_bits, zero_centered):
    """``bf16((x inv) scale)`` of a head held by one wave, ``len(x)``
    consecutive dims per lane: the square sum is the pairtree over the
    head, ``inv`` goes through f64. ``scale`` is ``w`` (direct) or ``1 + w``."""
    sq = [v * v for v in x]
    while len(sq) > 1:
        sq = [sq[2 * i] + sq[2 * i + 1] for i in range(len(sq) // 2)]
    stab = natural_sum(sq[0]) * _f32(r_bits) + _f32(EPS)
    inv = (fx.Float64(1.0) / fx.Float64(fmath.sqrt(stab.to(fx.Float64)))).to(fx.Float32)
    if zero_centered:
        w = [fx.Float32(1.0) + v for v in w]
    return [bf16r((v * inv) * s) for v, s in zip(x, w)]


def rope(y, cs, sn, lane, off, active):
    """RoPE on bf16 values where the partner of a value sits in the
    same slot of lane ``lane ^ off``, negated on the lanes with ``lane & off``
    clear. Lanes where ``active`` is false keep ``y``."""
    neg = ((lane & off) == 0).select(fx.Int32(SIGN), fx.Int32(0))
    out = []
    for v, c, s in zip(y, cs, sn):
        partner = (xshfl(v, off).bitcast(fx.Int32) ^ neg).bitcast(fx.Float32)
        out.append(active.select(bf16r(bf16r(v * c) + bf16r(partner * s)), v))
    return out


def _theirs(mine, theirs):
    return theirs


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
        poll_mixed(c, x8, xe)

        if wave < 4:
            xr, er = mfma_operands(c, x8, xe)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for g in range_constexpr(Q1_GROUPS):
                acc = p2_mma(acc, units[g], g, xr, er, lane)
            if lane < 16:
                for q in range_constexpr(c.rows):
                    fx.ptr_store(acc[q], red + row_ix(wave * 16 + lane, q, rs))
        gpu.barrier()

        epilogue = (wave == 0) if c.rows == 1 else (wave < c.rows)
        if epilogue:
            proj = bf16r(fx.ptr_load(red + row_ix(lane, rw, rs)))
            other = xred(proj, 1, _theirs)
            if lane % 2 == 0:
                at = mb_ix(t * 32 + lane // 2, rw)
                mbx.put(c.mb_qp, at, mbx.bf16_pair(proj, other), c.tag)


q1 = ASTRewriter.transform(q1)


def q2(c, p):
    lane = c.lane
    t = task_index(c, "q2")
    # Batched builds: wave ``r < rows`` runs request row r.
    rw = 0 if c.rows == 1 else fx.min(c.wave, c.rows - 1)
    L = row_len(p, rw)
    a_pages = qsa_row(p, rw, "pages")
    a_key = qsa_row(p, rw, "key")
    a_value = qsa_row(p, rw, "value")
    a_index_key = qsa_row(p, rw, "index_key")
    a_cos = qsa_row(p, rw, "cos")
    a_sin = qsa_row(p, rw, "sin")

    if t < c.L.tasks["q2"]:
        mine = (c.wave == 0) if c.rows == 1 else (c.wave < c.rows)
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


q2 = ASTRewriter.transform(q2)


def _i32(m):
    return fx.Int32(m - (1 << 32) if m >= 1 << 31 else m)


def _lds_i4(ptr, off):
    v = fx.Vector(fx.ptr_load(ptr + off, result_type=fx.Vector.make_type(4, fx.Int32)))
    return [v[e] for e in range(4)]


def list_len(L):
    """``(cb, n, ak)``: complete blocks, selected blocks, active keys."""
    n1 = L + 1
    cb = n1 // 4
    n = fx.min(cb, SEL)
    return cb, n, 4 * n + n1 % 4


def _cache_row(g, j, n, ak, tail):
    """The paged cache row of list position ``j``: ``g`` is the row in blocks
    (:func:`page_block`) of its block if it is selected, ``tail`` that of the
    incomplete block, which the rest of the list is; 0 past the list."""
    row = (j < 4 * n).select(4 * g + j % 4, 4 * tail + j - 4 * n)
    return (j < ak).select(row, fx.Int32(0))


def _now_rows(p, r_pg, L, seq):
    """``(tail, now)``: the incomplete block's row in blocks, which is ``L /
    4``'s when it holds a list position, and the cache row of token ``L``.
    Seq builds: ``now`` lists the rows of every launch token."""
    tail = page_block(r_pg, L // 4, True)
    if seq:
        return tail, [4 * page_block(r_pg, Lq // 4, True) + Lq % 4 for Lq in p.Ls]
    return tail, 4 * tail + L % 4


def _q5(c, p, tid, t, turn):
    qs = c.lds_i + c.L.lds["q5_q"]
    ks = c.lds_i + c.L.lds["q5_k"]
    sl = c.lds_i + c.L.lds["q5_sel"]
    # Batched builds: tasks (tile, request row, kv head), tile-major, the
    # rows turned ``turn`` on (``q5``).
    rq = 0 if c.rows == 1 else (t // 2 + turn) % c.rows
    L = row_len(p, rq)
    a_pages = qsa_row(p, rq, "pages")
    a_key = qsa_row(p, rq, "key")

    if t < c.L.tasks["q5"]:
        kv = t % 2
        tile = t // 2 if c.rows == 1 else t // (2 * c.rows)
        _, n, ak = list_len(L)
        if tile * SCORE_TILE < ak:
            r_pg = rsrc(a_pages)
            tail, now_row = _now_rows(p, r_pg, L, c.seq)

            def hint(m):
                head = (m < GROUP).select(GROUP * kv + m, C.q_heads + kv)
                return c.mb_qn, mb_ix(
                    (m <= GROUP).select(HEAD_WORDS * head, fx.Int32(SELH_FROM_QN)), rq
                )

            def hint_seq(m):
                # Seq builds: the q heads, every row's k head, then selh.
                head = (m < GROUP).select(GROUP * kv + m, C.q_heads + kv)
                kq = (m < GROUP).select(rq, fx.min(fx.max(m - GROUP, 0), c.rows - 1))
                return c.mb_qn, (m < GROUP + c.rows).select(
                    mb_ix(HEAD_WORDS * head, kq), mb_ix(SELH_FROM_QN, rq)
                )

            if const_expr(c.seq):
                mbx.pre_poll(hint_seq, GROUP + c.rows + 1, c.tag)
            else:
                mbx.pre_poll(hint, GROUP + 2, c.tag)
            if (tid < SCORE_TILE // 4) & (tile * (SCORE_TILE // 4) + tid < n):
                b = mbx.poll(
                    [(c.mb_sel, mb_ix(tile * (SCORE_TILE // 4) + tid, rq), 1)], c.tag
                )[0][0]
                fx.ptr_store(page_block(r_pg, b), sl + tid)
            specs, dst = [], []
            for r in range_constexpr(2):
                s = fx.min(
                    tid + THREADS * r, (QN_SEQ[c.rows] if c.seq else QN_SPECS) - 1
                )
                hh = s // (HEAD_WORDS // 2)
                w = 2 * (s % (HEAD_WORDS // 2))
                head = (hh < GROUP).select(GROUP * kv + hh, C.q_heads + kv)
                if const_expr(c.seq):
                    # Head GROUP + q: row q's k head.
                    qn_at = mb_ix(
                        head * HEAD_WORDS + w, (hh < GROUP).select(rq, hh - GROUP)
                    )
                else:
                    qn_at = mb_ix(head * HEAD_WORDS + w, rq)
                specs.append((c.mb_qn, qn_at, 2))
                dst.append(hh * HEAD_WORDS + w)
            got = mbx.poll(specs, c.tag)
            for r in range_constexpr(2):
                fx.ptr_store(fx.Vector.from_elements(got[r], fx.Int32), qs + dst[r])
            gpu.barrier()

            r_k = rsrc(a_key)
            quad = tid % 32
            if const_expr(c.seq):
                curs = [
                    _lds_i4(qs, (GROUP + q) * HEAD_WORDS + 4 * quad)
                    for q in range(c.rows)
                ]
            else:
                cur = _lds_i4(qs, GROUP * HEAD_WORDS + 4 * quad)
            for r in range_constexpr(2):
                row = tid // 32 + 16 * r
                kr = _cache_row(
                    fx.ptr_load(sl + row // 4), tile * SCORE_TILE + row, n, ak, tail
                )
                at = (kr * C.kv_heads + kv) * HEAD_WORDS + 4 * quad
                kw = fx.Vector(bo.buffer_load(r_k, at, vec_width=4, dtype=T.i32))
                if const_expr(c.seq):
                    # Tokens L .. L + rows - 1 from the rows' Q2s, not memory.
                    w4 = [kw[e] for e in range(4)]
                    for q in range_constexpr(c.rows):
                        hit = kr == now_row[q]
                        w4 = [fx.Int32(hit.select(curs[q][e], w4[e])) for e in range(4)]
                else:
                    now = kr == now_row
                    w4 = [fx.Int32(now.select(cur[e], kw[e])) for e in range(4)]
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
                mbx.put(
                    c.mb_ssh, mb_ix(2 * tile + kv, rq) if c.rows > 1 else t, 1, c.tag
                )


_q5 = ASTRewriter.transform(_q5)


def q5(c, p):
    tid = fx.Int32(c.tid)
    t = task_index(c, "q5")
    # A CTA's rounds run tasks GRID apart: where 2 rows divides GRID, all of
    # one request row, so a long request's tiles would fall on the same CTAs
    # every round. Round k turns the rows k on instead (a tile's tasks lie in
    # one round, so the turn permutes its rows). A seq build's rows are one
    # request's.
    turn = 0 if c.seq or GRID % (2 * c.rows) else 1
    for k in range_constexpr(task_rounds(c, "q5")):
        _q5(c, p, tid, t if k == 0 else t + k * GRID, k * turn % c.rows)


def q6(c, p):
    tid, lane, wave = fx.Int32(c.tid), c.lane, c.wave
    t = task_index(c, "q6")
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


q6 = ASTRewriter.transform(q6)


Q7_HEADS = 4  # q heads of a Q7 task


def _q7(c, p, tid, lane, wave, t):
    sl = c.lds_i + c.L.lds["q7_sel"]
    pb = c.lds_f + c.L.lds["q7_p"]
    cur = c.lds_i + c.L.lds["q7_cur"]
    part = c.lds_f + c.L.lds["q7_part"]
    # Batched builds: tasks (request row, x).
    rq, x = _request(c, t, TASKS["q7"])
    L = row_len(p, rq)
    a_pages = qsa_row(p, rq, "pages")
    a_value = qsa_row(p, rq, "value")

    if t < c.L.tasks["q7"]:
        dt = x % 8
        kv = x // 24
        qh0 = GROUP * kv + Q7_HEADS * ((x // 8) % 3)
        _, n, ak = list_len(L)
        r_pg = rsrc(a_pages)
        tail, now_row = _now_rows(p, r_pg, L, c.seq)
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
        if const_expr(c.seq):
            # Seq builds: thread 8 q + i polls row q's (token L + q's) values.
            ct = fx.min(tid, 8 * c.rows - 1)
            cur_at = P_V + HEAD_WORDS * kv + 16 * dt + 2 * (ct % 8)
            cur_row = ct // 8
        else:
            cur_at = P_V + HEAD_WORDS * kv + 16 * dt + 2 * fx.min(tid, 7)
            cur_row = rq
        gate_at = P_Q + 2 * HEAD_WORDS * (qh0 + hd) + HEAD_WORDS + 16 * dt + dim // 2
        v, g = mbx.poll(
            [(c.mb_qp, mb_ix(cur_at, cur_row), 2), (c.mb_qp, mb_ix(gate_at, rq), 1)],
            c.tag,
        )
        gw = g[0]
        if tid < 8 * (c.rows if c.seq else 1):
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
            vq.append((vw, vr if c.seq else vr == now_row))
        words = []
        if const_expr(c.seq):
            now_vs = [_lds_i4(cur, 16 * q + 4 * dq) for q in range(c.rows)]
            for i in range_constexpr(Q7_ROUNDS):
                w, tk = vq[i]
                live = 128 * i + slot < ak
                pick = [w[e] for e in range(4)]
                for q in range_constexpr(c.rows):
                    hit = tk == now_row[q]
                    pick = [
                        fx.Int32(hit.select(now_vs[q][e], pick[e])) for e in range(4)
                    ]
                words.append(
                    [fx.Int32(live.select(pick[e], fx.Int32(0))) for e in range(4)]
                )
        else:
            now_v = _lds_i4(cur, 4 * dq)
            for i in range_constexpr(Q7_ROUNDS):
                w, now = vq[i]
                live = 128 * i + slot < ak
                words.append(
                    [
                        fx.Int32(live.select(now.select(now_v[e], w[e]), fx.Int32(0)))
                        for e in range(4)
                    ]
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
        _q7(c, p, tid, lane, wave, t if k == 0 else t + k * GRID)
