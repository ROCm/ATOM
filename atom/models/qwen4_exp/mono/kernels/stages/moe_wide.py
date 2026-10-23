"""Wide builds' M3-M5 (``layout_wide.for_wide``): moe.py's stages with
every request row's MFMA result taken from its lanes (``common_wide``):

- M3 runs a router block's request rows (an even count) in halves on two CTAs
  (``layout_wide``'s ``m3_task_rows``), its router chains 4 request rows at
  a time, past ``m3_rows`` rows in passes of ``m3_rows`` request rows.
- M4 (two slices per task) fills the shared expert's LDS rows on waves
  ``0 .. rows - 1`` (past ``WAVES`` rows in passes, :func:`_m4_fill`) and runs
  its epilogue's rows ``ws``, ``ws + 4``, ... on wave ``ws`` of each slice.
- M5's routing past ``WAVES`` rows: in passes (:func:`_m5_pre_route`,
  :func:`_m5_own_route`).
- M5 streams each wave's ``ceil((10 rows + 1) / 8)`` down chains with
  :data:`M5_INFLIGHT` units of weights in flight, and polls its rows in
  blocks (:func:`_m5_blocks`): on the CTAs past M4's last partial round
  (``layout_wide``), a block's chains run once M4's earlier rounds end.

Derived from moe.py; moe.py stays the source.
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
from atom.models.qwen4_exp.mono.kernels.device_ops import read_lane_i32, uniform, xred
from atom.models.qwen4_exp.mono.kernels.layout import (
    DN_ROW,
    RW_PART,
    RW_ROW,
    RX_PART,
    WAVES,
)
from atom.models.qwen4_exp.mono.kernels.layout_wide import M5_BLOCK, WIDE_ROWS
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    HC_WORDS,
    bf16r,
    bf16x8,
    lds_f4,
    lds_row,
    mb_ix,
    mfma_operands,
    mfma_row,
    row_ix,
    task_index,
    task_rounds,
)
from atom.models.qwen4_exp.mono.kernels.stages.common_wide import (
    fresh_round,
    pre_poll_wide,
    store_rows,
)
from atom.models.qwen4_exp.mono.kernels.stages.hc import (
    OOB,
    fma,
    lane_gather,
    sigmoid_acc,
)
from atom.models.qwen4_exp.mono.kernels.stages.moe import (
    CHAIN_STEPS,
    DN_LOADS,
    DN_S_BYTES,
    DN_W_BYTES,
    DN_X8,
    GU_KI,
    GU_S_BYTES,
    GU_TILES,
    GU_W_BYTES,
    ROW_CHUNKS,
    ROW_WORDS,
    RW_LOADS,
    RX_LOADS,
    TOP_IDS,
    TOP_INJ,
    TOP_M4,
    TOP_RES,
    TOP_RG,
    TOP_W,
    X8_LOADS,
    _chain_step,
    _dn_chain,
    _dn_issue,
    _dn_mma,
    _gu_issue,
    _gu_mma,
    _m5_route,
    _part,
    _store_x,
    key_expert,
    logit_specs,
    sigmoid_m,
    top10,
)

M5_INFLIGHT = 5  # M5's down-weight units in flight per wave
M5_PF_STEPS = 3  # M5: chain steps from a poll block's prefetch to its first chain
M5_ROUTED_LOADS = C.top_k * DN_ROW // 2  # 2-pair loads of a row's routed ranks
# M5's poll-block votes: words of ``top``'s row 0 that the routing leaves free.
TOP_VOTE = TOP_IDS + C.top_k
M5_VOTES = WIDE_ROWS // M5_BLOCK
assert TOP_VOTE + M5_VOTES <= TOP_M4


def _m3_chains(c, rw, rx, lgl, tile, q0, rows):
    """M3's router chains of a pass's request rows ``q0 + 4 tile ..`` (4 at
    most of its ``rows``, in LDS rows ``4 tile ..`` of ``rx``): lane ``16 i +
    q`` runs part ``q`` of weight row ``wave`` for request row ``q0 + 4 tile +
    i``, as ``moe.m3``'s lane ``16 r + q``."""
    lane, wave = c.lane, c.wave
    n = min(4, rows - 4 * tile)
    slg = c.L.stride["lgl"]
    if lane < 16 * n:
        lp = lane % 16
        big = lp < 8
        wrow = rw + (wave * RW_ROW + lp * RW_PART)
        xrow = rx + row_ix(lp * RX_PART, 4 * tile + lane // 16, c.L.stride["rx"])
        v4i = fx.Vector.make_type(4, fx.Int32)
        acc = fx.Float32(0.0)
        for s in range_constexpr(CHAIN_STEPS):
            wv = fx.Vector(fx.ptr_load(wrow + s * 4, result_type=v4i))
            w8 = bf16x8([wv[e] for e in range(4)])
            x8 = lds_f4(xrow, s * 8) + lds_f4(xrow, s * 8 + 4)
            new = acc
            for e in range_constexpr(8):
                new = fma(w8[e], x8[e], new)
            acc = _chain_step(s, big, new, acc)
        bits = acc.bitcast(fx.Int32)
        for rt in range_constexpr(n):
            total = read_lane_i32(bits, 16 * rt).bitcast(fx.Float32)
            for q in range_constexpr(1, 16):
                total = total + read_lane_i32(bits, 16 * rt + q).bitcast(fx.Float32)
            if lane == 0:
                fx.ptr_store(bf16r(total), lgl + row_ix(wave, q0 + 4 * tile + rt, slg))


_m3_chains = ASTRewriter.transform(_m3_chains)


def m3(c, p):
    tid = c.tid
    experts = c.experts
    # Wide builds: task ``task`` runs router block ``t`` (``moe.m3``'s task)
    # for request rows ``h0 .. h0 + rows - 1``, a half of the rows when even
    # (``layout_wide``).
    task = task_index(c, "m3")
    rows = c.L.m3_task_rows
    t, h0 = task % (experts // 8 + 1), task // (experts // 8 + 1) * rows
    rw = c.lds_i + c.L.lds["rw"]
    rx = c.lds_f + c.L.lds["rx"]
    lgl = c.lds_f + c.L.lds["lgl"]
    rxs = [rx] + [lds_row(c, c.lds_f, "rx", r) for r in range(1, c.L.m3_rows)]
    slg = c.L.stride["lgl"]

    # Batched builds: hint ``m`` is M2 task ``m % 80`` of request row ``m // 80``.
    def hint(m):
        k, r = (m, 0) if rows == 1 else (m % (C.hidden // 32), m // (C.hidden // 32))
        return c.mb_mxe, mb_ix(k, h0 + r)

    if task < (experts // 8 + 1) * (c.rows // rows):
        # Chunk ``j``: 8 bf16 of row ``j // 320``; the gate task reads row E 8 times.
        ws = []
        for m in range_constexpr(RW_LOADS):
            j = tid + m * THREADS
            g = fx.min(t * 8 + j // ROW_CHUNKS, experts)
            off = g * ROW_WORDS + (j % ROW_CHUNKS) * 4
            ws.append(fx.Vector(bo.buffer_load(p.r_w, off, vec_width=4, dtype=T.i32)))

        pre_poll_wide(hint, C.hidden // 32 * rows, c.tag)
        hi = fx.min(tid + THREADS, RX_LOADS - 1)
        # Wide builds: the task's rows ``q0 ..`` (``m3_rows`` a pass) through
        # LDS rows ``0 ..`` of ``rx``; the router weights in the first pass.
        for qp in range_constexpr(-(-rows // c.L.m3_rows)):
            q0 = qp * c.L.m3_rows
            n = min(c.L.m3_rows, rows - q0)
            v = mbx.poll(
                [
                    (c.mb_mxbf, mb_ix(j * 2, h0 + q0 + r), 2)
                    for r in range(n)
                    for j in (tid, hi)
                ],
                c.tag,
            )
            for r in range_constexpr(n):
                _store_x(rxs[r], v[2 * r], tid)
                _store_x(rxs[r], v[2 * r + 1], hi)
            if const_expr(qp == 0):
                for m in range_constexpr(RW_LOADS):
                    j = tid + m * THREADS
                    part, first = _part((j % ROW_CHUNKS) * 8)
                    off = (
                        (j // ROW_CHUNKS) * RW_ROW
                        + part * RW_PART
                        + ((j % ROW_CHUNKS) * 8 - first) // 2
                    )
                    fx.ptr_store(ws[m], rw + off)
            gpu.barrier()

            # Lane ``q < 16``: part ``q`` of row ``wave``, one fma chain from +0.
            # Wide builds: 4 request rows at a time (:func:`_m3_chains`).
            for k in range_constexpr(-(-n // 4)):
                _m3_chains(c, rw, rx, lgl, k, q0, n)
            gpu.barrier()

        # Thread ``2 r + i``: logits ``8 t + 4 i ..`` of request row ``h0 + r``.
        li, lr = (tid, 0) if rows == 1 else (tid % 2, tid // 2)
        if (t < experts // 8) & (tid < 2 * rows):
            lg = lds_f4(lgl, row_ix(li * 4, lr, slg))
            mbx.put2(
                c.mb_lg,
                mb_ix(t * 4 + li * 2, h0 + lr),
                mbx.bf16_pair(lg[0], lg[1]),
                mbx.bf16_pair(lg[2], lg[3]),
                c.tag,
            )
        if (t == experts // 8) & ((tid == 0) if rows == 1 else (tid < rows)):
            gl = lgl if rows == 1 else lgl + tid * slg
            mbx.put(
                c.mb_sg,
                0 if rows == 1 else mb_ix(0, h0 + tid),
                bf16r(sigmoid_acc(fx.ptr_load(gl))),
                c.tag,
            )


m3 = ASTRewriter.transform(m3)


def m4(c, p):
    # Batched builds from 3 rows on have more tasks than CTAs: rounds, each
    # starting with a CTA barrier ahead of its LDS writes.
    assert c.L.m4_slices == 2, c.L.m4_slices
    t = task_index(c, "m4")
    for k in range_constexpr(task_rounds(c, "m4")):
        _m4(fresh_round(c) if k else c, p, t if k == 0 else t + k * GRID, k == 0)


def _m5_pre_route(c, t5, ek):
    """Pass ``ek`` of M4's routing of M5's request rows past ``WAVES`` rows:
    wave ``w`` routes row ``WAVES - 1 - w + 8 ek``."""
    j = WAVES - 1 - c.wave + WAVES * ek
    if (t5 < c.L.tasks["m5"]) & (j < c.rows):
        _m5_route(c, t5, j)


_m5_pre_route = ASTRewriter.transform(_m5_pre_route)


def _m4_fill(c, shared, x8, xe, sx8, sxe, ek):
    """Pass ``ek`` of M4's LDS fill past ``WAVES`` rows: on the shared
    expert's tasks, wave ``w`` fills LDS row ``w + 8 ek`` with that request row."""
    lane = c.lane
    q = c.wave + WAVES * ek
    s = fx.min(q, c.rows - 1)
    src = s
    if shared & (q < c.rows):
        xw = mbx.poll(
            [
                (c.mb_mxe, mb_ix(lane, src), 1),
                (c.mb_mxe, mb_ix(fx.min(lane + 64, C.hidden // 32 - 1), src), 1),
            ],
            c.tag,
        )
        fx.ptr_store(xw[0][0], xe + row_ix(lane, s, sxe))
        if lane < 16:
            fx.ptr_store(xw[1][0], xe + row_ix(lane + 64, s, sxe))
        fw = mbx.poll(
            [(c.mb_mx8, mb_ix((lane + 64 * i) * 2, src), 2) for i in range(X8_LOADS)],
            c.tag,
        )
        for i in range_constexpr(X8_LOADS):
            fx.ptr_store(
                fx.Vector.from_elements(fw[i], fx.Int32),
                x8 + row_ix((lane + 64 * i) * 2, s, sx8),
            )


_m4_fill = ASTRewriter.transform(_m4_fill)


def _m4(c, p, t, first):
    lane, wave = c.lane, c.wave
    rows = c.rows
    # Slices ``q0 .. q0 + S - 1`` of the task's (row, rank): slice ``j`` on
    # waves ``4 j ..`` when S = 2 (each wave then runs two chains, block pairs
    # ``4 q0 + w % 4`` and 4 on, the block as in a one-slice task).
    S = c.L.m4_slices
    tiles = GU_TILES // S
    top = c.lds_i + c.L.lds["top"]
    x8 = c.lds_i + c.L.lds["gx8"]
    xe = c.lds_i + c.L.lds["gxe"]
    red = c.lds_f + c.L.lds["gred"]
    sx8, sxe, rs = c.L.stride["gx8"], c.L.stride["gxe"], c.L.stride["gred"]

    if t < c.L.tasks["m4"]:
        # Batched builds: request row ``rr``'s routed tasks ``100 rr ..``, then
        # the shared expert's tasks (``rr`` 0, every row in MFMA rows).
        if const_expr(rows == 1):
            r = t // GU_TILES
            q0 = t % GU_TILES
            rr = 0
        else:
            routed = C.top_k * tiles
            in_sh = t >= routed * rows
            tt = in_sh.select(t - routed * (rows - 1), t % routed)
            r = tt // tiles
            q0 = tt % tiles * S
            rr = in_sh.select(0, t // routed)
        shared = r == C.top_k
        gpu.barrier()
        if (wave == 0) & (r < C.top_k):
            got = mbx.poll(logit_specs(c, lane, c.experts, rr), c.tag)
            picks = top10(got[0] + got[1], lane, c.experts)
            pick = picks[0]
            for k in range_constexpr(1, C.top_k):
                pick = (r == k).select(picks[k], pick)
            if lane == 0:
                fx.ptr_store(key_expert(pick), top + (TOP_IDS if rows == 1 else TOP_M4))
        # Batched builds, a CTA whose M5 task follows: wave ``WAVES - 1 - j``
        # routes M5's request row ``j`` meanwhile, so that M5 can issue its down
        # weights once this task ends.
        if const_expr(rows > 1 and first):
            t5 = task_index(c, "m5")
            if (t5 < c.L.tasks["m5"]) & (wave >= WAVES - rows):
                _m5_route(c, t5, WAVES - 1 - wave)
            for e5 in range_constexpr(1, -(-rows // WAVES)):
                _m5_pre_route(c, t5, e5)
        # The exponents are each M2 task's last store; the fp8 words follow.
        # Wide builds: wave 1 fills LDS row 0 with request row ``rr``; the
        # shared expert's tasks: wave ``s < rows`` fills row ``s``.
        if const_expr(rows == 1):
            s, src, fill = 0, 0, wave == 1
        else:
            s = shared.select(fx.min(wave, rows - 1), fx.Int32(0))
            src = shared.select(s, rr)
            fill = (wave == 1) | (shared & (wave < rows))
        if fill:
            xw = mbx.poll(
                [
                    (c.mb_mxe, mb_ix(lane, src), 1),
                    (c.mb_mxe, mb_ix(fx.min(lane + 64, C.hidden // 32 - 1), src), 1),
                ],
                c.tag,
            )
            fx.ptr_store(xw[0][0], xe + row_ix(lane, s, sxe))
            if lane < 16:
                fx.ptr_store(xw[1][0], xe + row_ix(lane + 64, s, sxe))
            fw = mbx.poll(
                [
                    (c.mb_mx8, mb_ix((lane + 64 * i) * 2, src), 2)
                    for i in range(X8_LOADS)
                ],
                c.tag,
            )
            for i in range_constexpr(X8_LOADS):
                fx.ptr_store(
                    fx.Vector.from_elements(fw[i], fx.Int32),
                    x8 + row_ix((lane + 64 * i) * 2, s, sx8),
                )
        for ef in range_constexpr(1, -(-rows // WAVES)):
            _m4_fill(c, shared, x8, xe, sx8, sxe, ef)
        gpu.barrier()

        pick = top + (TOP_IDS if rows == 1 else TOP_M4)
        expert = shared.select(fx.Int32(0), uniform(fx.ptr_load(pick)))
        aw = shared.select(p.a_sh_w_gu, p.a_e_w_gu + fx.Int64(expert) * GU_W_BYTES)
        a_s = shared.select(p.a_sh_s_gu, p.a_e_s_gu + fx.Int64(expert) * GU_S_BYTES)
        r_w = bo.create_buffer_resource_from_addr(aw, num_records_bytes=GU_W_BYTES)
        r_s = bo.create_buffer_resource_from_addr(a_s, num_records_bytes=GU_S_BYTES)
        pair, part = q0 * 4 + wave % 4, wave // 4
        units = [
            [
                _gu_issue(r_w, r_s, pair + 4 * j if j else pair, part, g, lane)
                for g in range(GU_KI // 4)
            ]
            for j in range(S)
        ]
        xr, er = mfma_operands(c, x8, xe, "gx8", "gxe")
        for j in range_constexpr(S):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for g in range_constexpr(GU_KI // 4):
                acc = _gu_mma(acc, units[j][g], g, part, xr, er, lane)
            ix = wave * 16 + lane % 16
            store_rows(c, acc, red, ix + j * WAVES * 16 if j else ix, rs)
        gpu.barrier()

        # Lane L: column 64 q + L of gate and up. Wide builds: wave ``ws`` of
        # the slice's 4 runs LDS rows ``ws``, ``ws + 4`` (:func:`_m4_out`).
        ws = wave % 4
        sl = (wave // 4) * (WAVES * 16)
        q = q0 + wave // 4
        for k in range_constexpr(-(-rows // 4)):
            _m4_out(c, red, rs, shared, rr, r, q, sl, ws, k)


_m4 = ASTRewriter.transform(_m4)


def _m4_out(c, red, rs, shared, rr, r, q, sl, ws, k):
    """M4's epilogue of LDS row ``ws + 4 k`` on wave ``ws`` of the slice:
    ``moe._m4``'s of row ``rw``, for request row ``out``. A routed task has
    row 0 only (``ws`` 0); the shared expert's tasks run every row."""
    lane = c.lane
    rows = c.rows
    rw = fx.min(ws + 4 * k, rows - 1)
    out = shared.select(rw, rr)
    go = (
        ((ws == 0) | (shared & (ws < rows)))
        if k == 0
        else (shared & (ws + 4 * k < rows))
    )
    if go:
        x = bf16r(fx.ptr_load(red + row_ix(sl + lane, rw, rs)))
        y = bf16r(fx.ptr_load(red + row_ix(sl + 64 + lane, rw, rs)))
        # x gate, y up. Routed: split SiLU; shared: standard.
        h_routed = bf16r(bf16r(x * y) * bf16r(sigmoid_m(x)))
        h_shared = bf16r(bf16r(x * sigmoid_acc(x)) * y)
        h = shared.select(h_shared, h_routed)
        amax = fmath.absf(h)
        for off in (1, 2, 4, 8, 16):
            amax = xred(amax, off, fx.max)
        ex, scale = p2.e8m0_fp8(amax)
        v8 = lane_gather(h, (lane // 32) * 32 + (lane % 4) * 8, 8)
        lo = p2.fp8_word(v8[0], v8[1], v8[2], v8[3], scale)
        hi = p2.fp8_word(v8[4], v8[5], v8[6], v8[7], scale)
        ex_hi = read_lane_i32(ex, 32)
        if lane % 32 < 4:
            slot = r * DN_ROW + q * 16 + (lane // 32) * 8 + (lane % 4) * 2
            mbx.put2(c.mb_dn, mb_ix(slot, out), lo, hi, c.tag)
        if lane == 0:
            mbx.put2(c.mb_dn, mb_ix(r * DN_ROW + DN_X8 + q * 2, out), ex, ex_hi, c.tag)


_m4_out = ASTRewriter.transform(_m4_out)


def _m5_issue(c, p, t, top, s):
    """Unit ``s`` of ``moe._m5_issue_rows``: the down weights of this wave's
    chain ``k = wave + 8 s`` (zero loads past the last chain)."""
    rows, lane, wave = c.rows, c.lane, c.wave
    n = C.top_k * rows  # the shared expert's chain
    k = wave + WAVES * s
    row = fx.min(k // C.top_k, rows - 1)
    e = uniform(
        fx.ptr_load(top + row_ix(TOP_IDS + k % C.top_k, row, c.L.stride["top"]))
    )
    sh = k == n
    return _dn_issue(
        bo.create_buffer_resource_from_addr(
            sh.select(p.a_sh_w_dn, p.a_e_w_dn + fx.Int64(e) * DN_W_BYTES),
            num_records_bytes=DN_W_BYTES,
        ),
        bo.create_buffer_resource_from_addr(
            sh.select(p.a_sh_s_dn, p.a_e_s_dn + fx.Int64(e) * DN_S_BYTES),
            num_records_bytes=DN_S_BYTES,
        ),
        t,
        lane,
        (k <= n).select(fx.Int32(0), fx.Int32(OOB)),
    )


def _m5_chain(c, unit, s, dn, ys):
    """Chain ``k = wave + 8 s`` of ``moe._m5_chains_rows`` on ``unit``
    (:func:`_m5_issue`); the shared expert's chain stores every row."""
    rows, lane, wave = c.rows, c.lane, c.wave
    sdn, sys_ = c.L.stride["dn"], c.L.stride["ys"]
    n = C.top_k * rows
    k = wave + WAVES * s
    if k < n:
        row, rank = k // C.top_k, k % C.top_k
        acc = _dn_chain(unit, dn + row_ix(rank * DN_ROW, row, sdn), lane)
        if lane < 16:
            fx.ptr_store(bf16r(acc[0]), ys + row_ix(rank * 16 + lane, row, sys_))
    if k == n:
        # Partial L over k_iters L and L + 4, reduced in order.
        row_b = dn + row_ix(C.top_k * DN_ROW, mfma_row(c), sdn)
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        w0 = _dn_mma(_dn_mma(zero, unit, 0, row_b, lane), unit, 4, row_b, lane)
        w1 = _dn_mma(zero, unit, 1, row_b, lane)
        w2 = _dn_mma(zero, unit, 2, row_b, lane)
        w3 = _dn_mma(zero, unit, 3, row_b, lane)
        tot = [bf16r(((w0[i] + w1[i]) + w2[i]) + w3[i]) for i in range(4)]
        store_rows(c, tot, ys, C.top_k * 16 + lane % 16, sys_)


_m5_chain = ASTRewriter.transform(_m5_chain)


def _m5_own_route(c, t, m5_only, ek):
    """Pass ``ek`` of M5's own routing (a CTA without an M4 task) past
    ``WAVES`` rows: wave ``w``, request row ``w + 8 ek``."""
    q = c.wave + WAVES * ek
    if (q < c.rows) & m5_only:
        _m5_route(c, t, fx.min(q, c.rows - 1))


_m5_own_route = ASTRewriter.transform(_m5_own_route)


def _m5_blocks(rows, block):
    """M5's poll blocks of ``block`` request rows (``layout_wide.M5_BLOCK``,
    else one block): (first row, end row, last) of each, the last also every
    row's shared rank; and the chain step (:func:`_m5_chain`) that first reads
    each. A row block's routed ranks come from M4 tasks of the earlier rounds,
    every shared rank from the last round's last tasks."""
    n = C.top_k * rows
    nb = -(-rows // block)
    blocks = [(b * block, min(b * block + block, rows), b == nb - 1) for b in range(nb)]
    first = [None] * nb
    for s in range(-(-(n + 1) // WAVES)):
        hi = min(WAVES * s + WAVES - 1, n)
        need = nb - 1 if hi == n else hi // C.top_k // block
        for b in range(need + 1):
            if first[b] is None:
                first[b] = s
    return blocks, first


def _m5_loads(blk):
    """(per-row 2-pair loads of rows ``r0 ..``, loads of the earlier rows'
    shared ranks) of poll block ``blk``."""
    r0, _, last = blk
    return (DN_LOADS if last else M5_ROUTED_LOADS), (DN_ROW // 2 * r0 if last else 0)


def _m5_specs(c, blk):
    """Poll block ``blk``'s payload loads of this thread: pairs ``2 tid`` and
    ``2 hi`` of each of its rows, then its share of the earlier rows' shared
    ranks."""
    r0, r1, _ = blk
    tid = c.tid
    n, ns = _m5_loads(blk)
    hi = fx.min(tid + THREADS, n - 1)
    specs = [(c.mb_dn, mb_ix(j * 2, r), 2) for r in range(r0, r1) for j in (tid, hi)]
    for i in range(-(-ns // THREADS)):
        m = fx.min(tid + THREADS * i, ns - 1)
        j = M5_ROUTED_LOADS + m % (DN_ROW // 2)
        specs.append((c.mb_dn, mb_ix(j * 2, m // (DN_ROW // 2)), 2))
    return specs


def _m5_hint(c, blk, m):
    """Hint ``m`` of poll block ``blk``: the M4 exponent pairs of its rows'
    ranks, then (the last block) of the earlier rows' shared ranks."""
    r0, r1, last = blk
    nk = (C.top_k + 1 if last else C.top_k) * GU_TILES
    k, r = m % nk, m // nk
    if r0:
        r = r + r0
        if last:
            q = m - nk * (r1 - r0)
            own = q < 0
            k = own.select(k, C.top_k * GU_TILES + q % GU_TILES)
            r = own.select(r, q // GU_TILES)
    return c.mb_dn, mb_ix((k // GU_TILES) * DN_ROW + DN_X8 + (k % GU_TILES) * 2, r)


def _m5_hints(blk):
    r0, r1, last = blk
    return (C.top_k + 1 if last else C.top_k) * GU_TILES * (
        r1 - r0
    ) + GU_TILES * r0 * last


def _m5_store(c, dn, blk, v):
    """Store poll block ``blk``'s payload ``v`` (one list of words per
    :func:`_m5_specs` load) into LDS."""
    r0, r1, _ = blk
    tid, sdn = c.tid, c.L.stride["dn"]
    n, ns = _m5_loads(blk)
    hi = fx.min(tid + THREADS, n - 1)
    for r in range_constexpr(r0, r1):
        i = 2 * (r - r0)
        fx.ptr_store(
            fx.Vector.from_elements(v[i], fx.Int32), dn + row_ix(tid * 2, r, sdn)
        )
        if tid + THREADS < n:
            fx.ptr_store(
                fx.Vector.from_elements(v[i + 1], fx.Int32), dn + row_ix(hi * 2, r, sdn)
            )
    for i in range_constexpr(-(-ns // THREADS)):
        m = fx.min(tid + THREADS * i, ns - 1)
        j = M5_ROUTED_LOADS + m % (DN_ROW // 2)
        dst = dn + row_ix(j * 2, m // (DN_ROW // 2), sdn)
        fx.ptr_store(fx.Vector.from_elements(v[2 * (r1 - r0) + i], fx.Int32), dst)


_m5_store = ASTRewriter.transform(_m5_store)


def _m5_poll_block(c, dn, blk):
    """Poll block ``blk`` into LDS: wave 0 waits on its hints, then every
    thread polls its payload loads."""
    pre_poll_wide(lambda m: _m5_hint(c, blk, m), _m5_hints(blk), c.tag)
    _m5_store(c, dn, blk, mbx.poll(_m5_specs(c, blk), c.tag))
    gpu.barrier()


def _m5_take(c, dn, top, blk, b, v):
    """Poll block ``blk`` (``b``) from its prefetch ``v`` (one load each,
    ``M5_PF_STEPS`` chain steps ahead): store it, and the CTA votes whether
    any load came early (``TOP_VOTE + b``); if one did, poll the block again.
    The prefetch keeps the weight units issued after it in flight (a poll's
    loads would wait for them)."""
    specs = _m5_specs(c, blk)
    _m5_store(c, dn, blk, mbx.split_values(v, specs))
    if mbx.pending(v, c.tag):
        fx.ptr_store(fx.Int32(1), top + (TOP_VOTE + b))
    gpu.barrier()
    if uniform(fx.ptr_load(top + (TOP_VOTE + b))) != 0:
        _m5_poll_block(c, dn, blk)


_m5_take = ASTRewriter.transform(_m5_take)


def m5(c, p):
    tid, lane, wave = c.tid, c.lane, c.wave
    rows = c.rows
    t = task_index(c, "m5")
    top = c.lds_i + c.L.lds["top"]
    topf = c.lds_f + c.L.lds["top"]
    dn = c.lds_i + c.L.lds["dn"]
    ys = c.lds_f + c.L.lds["ys"]
    stop, sys_ = c.L.stride["top"], c.L.stride["ys"]
    nu = -(-(C.top_k * rows + 1) // WAVES)  # chains per wave (:func:`_m5_chain`)

    # Batched builds: hint ``m`` is M4 exponent pair ``m % 110`` of request row ``m // 110``.
    def hint(m):
        n = (C.top_k + 1) * GU_TILES
        k, r = (m, 0) if rows == 1 else (m % n, m // n)
        return c.mb_dn, mb_ix((k // GU_TILES) * DN_ROW + DN_X8 + (k % GU_TILES) * 2, r)

    if t < c.L.tasks["m5"]:
        gpu.barrier()
        # Wave 0 routes (:func:`_m5_route`). Batched builds: wave r, request row
        # r, unless the CTA's M4 task did already.
        if const_expr(rows == 1):
            if wave == 0:
                _m5_route(c, t, 0)
        else:
            m5_only = task_index(c, "m4") >= c.L.tasks["m4"]
            if (wave < rows) & m5_only:
                _m5_route(c, t, fx.min(wave, rows - 1))
            for ek in range_constexpr(1, -(-rows // WAVES)):
                _m5_own_route(c, t, m5_only, ek)
            if const_expr(c.L.m5_block < rows):
                if tid < M5_VOTES:
                    fx.ptr_store(fx.Int32(0), top + (TOP_VOTE + tid))
        gpu.barrier()

        # Issue the down weights once M4's first routed tile is out. Issued
        # earlier, they compete for HBM with M4's gate_up weights, which are on
        # the critical path (flushed caches: ~1.2 us off the layer). Batched
        # builds: at once on a CTA whose M4 task has ended.
        if const_expr(rows == 1):
            mbx.pre_poll(hint, 1, c.tag)
        else:
            if m5_only:
                mbx.pre_poll(hint, 1, c.tag)
        if const_expr(rows == 1):
            # Chain A: rank ``wave``. Chain B: ranks 8-9 on waves 0-1, the shared
            # expert on wave 2, nothing (zero loads) on waves 3-7.
            e_a = uniform(fx.ptr_load(top + (TOP_IDS + wave)))
            e_b = uniform(fx.ptr_load(top + (TOP_IDS + 8 + fx.min(wave, 1))))
            sh = wave == 2
            unit_a = _dn_issue(
                bo.create_buffer_resource_from_addr(
                    p.a_e_w_dn + fx.Int64(e_a) * DN_W_BYTES,
                    num_records_bytes=DN_W_BYTES,
                ),
                bo.create_buffer_resource_from_addr(
                    p.a_e_s_dn + fx.Int64(e_a) * DN_S_BYTES,
                    num_records_bytes=DN_S_BYTES,
                ),
                t,
                lane,
                0,
            )
            unit_b = _dn_issue(
                bo.create_buffer_resource_from_addr(
                    sh.select(p.a_sh_w_dn, p.a_e_w_dn + fx.Int64(e_b) * DN_W_BYTES),
                    num_records_bytes=DN_W_BYTES,
                ),
                bo.create_buffer_resource_from_addr(
                    sh.select(p.a_sh_s_dn, p.a_e_s_dn + fx.Int64(e_b) * DN_S_BYTES),
                    num_records_bytes=DN_S_BYTES,
                ),
                t,
                lane,
                (wave < 3).select(fx.Int32(0), fx.Int32(OOB)),
            )
        else:
            units = [_m5_issue(c, p, t, top, s) for s in range(min(M5_INFLIGHT, nu))]

        # The exponent pair is each M4 task's last store. The rows in poll
        # blocks (:func:`_m5_blocks`): the first now, the others in the chains.
        blocks, first = _m5_blocks(rows, c.L.m5_block)
        _m5_poll_block(c, dn, blocks[0])

        if const_expr(rows == 1):
            acc = _dn_chain(unit_a, dn + wave * DN_ROW, lane)
            if lane < 16:
                fx.ptr_store(bf16r(acc[0]), ys + (wave * 16 + lane))
            row_b = dn + (wave + 8) * DN_ROW
            if wave < 2:
                acc_b = _dn_chain(unit_b, row_b, lane)
                if lane < 16:
                    fx.ptr_store(bf16r(acc_b[0]), ys + ((wave + 8) * 16 + lane))
            if sh:
                # Partial L over k_iters L and L + 4, reduced in order.
                zero = fx.Vector.filled(4, 0.0, fx.Float32)
                w0 = _dn_mma(
                    _dn_mma(zero, unit_b, 0, row_b, lane), unit_b, 4, row_b, lane
                )
                w1 = _dn_mma(zero, unit_b, 1, row_b, lane)
                w2 = _dn_mma(zero, unit_b, 2, row_b, lane)
                w3 = _dn_mma(zero, unit_b, 3, row_b, lane)
                if lane < 16:
                    shared = bf16r(((w0[0] + w1[0]) + w2[0]) + w3[0])
                    fx.ptr_store(shared, ys + (C.top_k * 16 + lane))
        else:
            pf = [None] * len(blocks)
            for s in range_constexpr(nu):
                if const_expr(s + M5_INFLIGHT < nu):
                    units.append(_m5_issue(c, p, t, top, s + M5_INFLIGHT))
                for b in range_constexpr(1, len(blocks)):
                    if const_expr(s == first[b]):
                        _m5_take(c, dn, top, blocks[b], b, pf[b])
                    if const_expr(s == max(first[b] - M5_PF_STEPS, first[b - 1])):
                        pf[b] = mbx.load_pairs(_m5_specs(c, blocks[b]))
                _m5_chain(c, units[s], s, dn, ys)
        gpu.barrier()

        # Thread i < 32: stream i // 8, columns 16 t + 2 (i % 8) and the next.
        # Batched builds: thread 32 r + i, of request row r.
        i, rt = (tid, 0) if rows == 1 else (tid % 32, tid // 32)
        if tid < 32 * rows:
            j = (i % 8) * 2
            acc0 = fx.Float32(0.0)
            acc1 = fx.Float32(0.0)
            for k in range_constexpr(C.top_k):
                wk = fx.ptr_load(topf + row_ix(TOP_W + k, rt, stop))
                acc0 = bf16r(
                    fma(fx.ptr_load(ys + row_ix(k * 16 + j, rt, sys_)), wk, acc0)
                )
                acc1 = bf16r(
                    fma(fx.ptr_load(ys + row_ix(k * 16 + j + 1, rt, sys_)), wk, acc1)
                )
            rg = fx.ptr_load(topf + row_ix(TOP_RG, rt, stop))
            sh0 = fx.ptr_load(ys + row_ix(C.top_k * 16 + j, rt, sys_))
            sh1 = fx.ptr_load(ys + row_ix(C.top_k * 16 + j + 1, rt, sys_))
            blk0 = bf16r(acc0 + bf16r(rg * sh0))
            blk1 = bf16r(acc1 + bf16r(rg * sh1))
            inj = fx.ptr_load(topf + row_ix(TOP_INJ + i, rt, stop))
            r0, r1 = mbx.bf2_f32(fx.ptr_load(top + row_ix(TOP_RES + i, rt, stop)))
            out = mbx.bf16_pair(r0 + bf16r(blk0 * inj), r1 + bf16r(blk1 * inj))
            bo.buffer_store(
                out, p.h_out, row_ix((i // 8) * ROW_WORDS + t * 8 + i % 8, rt, HC_WORDS)
            )


m5 = ASTRewriter.transform(m5)
