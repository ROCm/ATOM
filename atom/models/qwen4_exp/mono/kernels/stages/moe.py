"""MoE stages: M3 (router), M4 (routed and shared gate_up) and M5 (down,
combine and the MLP HC write).

The expert count ``E`` (``c.experts``) is a build parameter, 512 for the model
and 16 for the CI fixture. Every stage reads the MLP HC's mixed row (M2's
``mx8``/``mxe``/``mxbf`` mailboxes).

M3, task ``t < E / 8``: router logits ``8 t .. 8 t + 7``. Task ``E / 8``
computes the shared expert gate the same way and publishes ``bf16(σ_o(.))``.
The task's 8 weight rows are prefetched into LDS before the mixed row is
polled; wave ``w`` then runs row ``w``'s 16 K-part chains on lanes 0-15.

M4, task ``t`` (110) = (rank ``r = t // 10``, 64-column slice ``q = t % 10``);
rank 10 is the shared expert. Wave 0 selects the top 10 itself from the logits
and picks rank ``r``'s expert. Wave ``w`` runs one n-block's chain over
k_iters 0-19: block pair ``4 q + w % 4``'s gate block for ``w < 4``, its up
block otherwise. Wave 0 applies the SiLU and publishes the slice's two MXFP8
blocks into row ``r`` of ``dn``.

The experts are ATOM's A8W4 tensors, read in place (the shared expert is the
last one). Gate / up rows are interleaved a 16-row block at a time (block
``2 j + part``, part 0 gate), the intermediate padded to 768. A scale dword
(aiter ``shuffle_scale``) holds k_iters ``2 k1`` and ``2 k1 + 1`` of a block
pair, byte ``2 kp + np``; it is shifted right by ``8 np`` where it is used,
since the MFMA's byte select ``2 kp`` must be a constant.

M5, task ``t`` (160): hidden columns ``16 t .. 16 t + 15``. Wave 0 selects
the top 10 again and computes the router weights. Wave ``w`` runs the down
chain of rank ``w``; waves 0-1 also run ranks 8-9 and wave 2 the shared
expert. Threads 0-31 combine in rank order and write the layer output,
``h_out[b] = bf16(w_b + bf16(block * inj_b))``, with ``w`` C1's streams and
``inj`` M1's injection weights.

Batched builds (``c.rows`` request rows) keep every row's arithmetic that of
one request. M3 runs request row ``r``'s chains on lanes ``16 r ..``. M4 runs
each row's 100 routed tasks as one request's, then the shared expert's 10
tasks over all rows (MFMA rows, wave ``r`` publishing row r). M5 routes row
``r`` on wave ``r`` and spreads the ``10 rows + 1`` down chains (chain ``k``:
rank ``k % 10`` of row ``k // 10``, the last one the shared expert over all
rows) over the waves, chain ``k`` on wave ``k % 8``; threads ``32 r ..``
combine row r.
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
    read_lane_i32,
    uniform,
    wave_umax_dpp,
    xred,
)
from atom.models.qwen4_exp.mono.kernels.layout import (
    DN_ROW,
    ROUTER_PARTS,
    RW_PART,
    RW_ROW,
    RX_PART,
    WAVES,
)
from atom.models.qwen4_exp.mono.kernels.stages.common import (
    HC_WORDS,
    a_operand,
    bf16r,
    bf16x8,
    exp2_hw,
    lds_f4,
    lds_row,
    mb_ix,
    mfma_operands,
    mfma_row,
    payload_wait,
    row_ix,
    task_index,
    task_rounds,
)
from atom.models.qwen4_exp.mono.kernels.stages.hc import (
    OOB,
    fma,
    lane_gather,
    sigmoid_acc,
)

assert C.shared_inter == C.moe_inter

L2E = 1.4426950408889634
TWO_M64 = 2.0**-64
INT_MIN = -(2**31)
ROW_WORDS = C.hidden // 2

# M3: a weight row is 320 chunks of 8 bf16; the task's 8 rows are 5 chunks per thread.
ROW_CHUNKS = C.hidden // 8
RW_LOADS = 8 * ROW_CHUNKS // THREADS
assert RW_LOADS * THREADS == 8 * ROW_CHUNKS
RX_LOADS = C.hidden // 4  # 2-pair loads of the bf16 mixed row
assert THREADS < RX_LOADS <= 2 * THREADS
CHAIN_STEPS = max(n for _, n in ROUTER_PARTS) // 8  # 24 steps of 8 values
SHORT_STEPS = min(n for _, n in ROUTER_PARTS) // 8  # 16

# M4: gate_up is [48 x (gate 16, up 16), 2560]; a task covers 64 columns of each.
INTER_PAD = p2.k_iters(C.moe_inter) * 128  # 768, aiter's padded intermediate
GU_KI = p2.k_iters(C.hidden)  # 20
GU_K1 = GU_KI // 2  # scale dwords per lane of a block pair
GU_TILES = C.moe_inter // 64  # 10
GU_PAIRS = INTER_PAD // 16  # 48, 40 used
GU_W_BYTES = 2 * GU_PAIRS * GU_KI * 1024
GU_S_BYTES = GU_PAIRS * GU_K1 * 64 * 4
X8_LOADS = C.hidden // 8 // 64  # 2-pair loads per lane of the fp8 mixed row
assert X8_LOADS * 64 * 8 == C.hidden

# M5: down is [2560, 640], K padded to 6 k_iters; k_iter 5 is zero and skipped.
DN_KI = p2.k_iters(C.moe_inter)  # 6
DN_STEPS = C.moe_inter // 128  # 5
DN_K1 = (DN_STEPS + 1) // 2  # 3 scale dwords per lane of a block pair
DN_W_BYTES = C.hidden // 16 * DN_KI * 1024
DN_S_BYTES = C.hidden // 32 * (DN_KI // 2) * 64 * 4
DN_X8 = C.moe_inter // 4  # a dn row: 160 fp8 words, then 20 exponents
DN_LOADS = (C.top_k + 1) * DN_ROW // 2  # 2-pair loads of all 11 rows
assert THREADS < DN_LOADS <= 2 * THREADS

# LDS ``top``: expert ids by rank, bf16 router weights, the shared gate, and
# (M5) thread ``i``'s residual word and injection weight.
TOP_IDS = 0
TOP_M4 = 15  # batched builds: M4's pick, apart from M5's routing (see _m5_route)
TOP_W = 16
TOP_RG = 31
TOP_RES = 32
TOP_INJ = 64

# Batcher's odd-even merge sort of 8 keys (19 comparators).
SORT8 = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (1, 2), (5, 6),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (2, 4), (3, 5),
    (1, 2), (3, 4), (5, 6),
)  # fmt: skip


def mexp_fma(x):
    """``exp(x)`` as the router softmax computes it: the hardware ``exp2`` of
    ``x * log2(e)``; below 2^-126 of ``fma(x, log2(e), 64)``, scaled by 2^-64."""
    p = x * L2E
    return (p < -126.0).select(exp2_hw(fma(x, L2E, 64.0)) * TWO_M64, exp2_hw(p))


def mexp_mul(x):
    """``exp(x)`` as the routed SiLU computes it: as :func:`mexp_fma`, but the
    shifted argument is the rounded product plus 64, not an fma."""
    p = x * L2E
    return (p < -126.0).select(exp2_hw(p + 64.0) * TWO_M64, exp2_hw(p))


def sigmoid_m(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + mexp_mul(fx.Float32(0.0) - x))


# ------------------------------------------------------------- routing


def logit_specs(c, lane, experts, row=0):
    """Poll specs of lane ``lane``'s 8 logits of request row ``row`` (lanes past
    ``E / 8`` repeat the last)."""
    j = fx.min(lane, experts // 8 - 1)
    return [(c.mb_lg, mb_ix(j * 4, row), 2), (c.mb_lg, mb_ix(j * 4 + 2, row), 2)]


def top10(words, lane, experts):
    """The top 10 of the logits as 10 wave-uniform keys, in rank order.

    ``words`` are lane ``j``'s 4 logit words (:func:`logit_specs`): logits
    ``8 j .. 8 j + 7``, low half first. A key is the logit's monotone bits with
    ``511 - idx`` in the 9 low bits, which bf16 values leave clear, so equal
    logits go to the lower expert. Each lane sorts its 8 keys, then every round
    takes the wave maximum of the lane heads and shifts the winner's list.
    """
    ks = []
    for i in range(8):
        w = words[i // 2]
        bits = w << 16 if i % 2 == 0 else w & fx.Int32(-65536)
        mono = (bits >= 0).select(bits ^ fx.Int32(INT_MIN), ~bits)
        key = (mono & fx.Int32(-512)) | (511 - (lane * 8 + i))
        ks.append(fx.Uint32((lane < experts // 8).select(key, fx.Int32(0))))
    for a, b in SORT8:
        first = ks[a] > ks[b]
        ka, kb = ks[a], ks[b]
        ks[a], ks[b] = first.select(ka, kb), first.select(kb, ka)
    ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
    picks = []
    for _ in range(C.top_k):
        m = wave_umax_dpp(ks[0])
        hit = ks[0] == m
        ks = [hit.select(ks[i + 1], ks[i]) for i in range(8)] + [ks[8]]
        picks.append(m)
    return picks


def key_expert(m):
    return 511 - (m & 0x1FF)


def key_logit(m):
    bits = (m < 0).select((m & fx.Int32(-512)) ^ fx.Int32(INT_MIN), ~(m | 0x1FF))
    return bits.bitcast(fx.Float32)


def router_weights(picks):
    """bf16 router weights of the 10 picks, rank order."""
    v = [key_logit(m) for m in picks]
    e = [mexp_fma(v[k] - v[0]) for k in range(C.top_k)]
    s = fx.Float32(0.0)
    for k in range(C.top_k):
        s = s + e[k]
    return [bf16r(e[k] / s) for k in range(C.top_k)]


# ------------------------------------------------------------- M3


def _part(k):
    """K part of the router's split holding ``k`` and that part's first k."""
    big = k < ROUTER_PARTS[8][0]
    p = big.select(k // 192, (k - 512) // 128)
    return p, big.select(p * 192, p * 128 + 512)


def _store_x(rx, words, j):
    """Mixed-row values ``4 j .. 4 j + 3`` (2 bf16 words) as f32 into their part."""
    x0, x1 = mbx.bf2_f32(words[0])
    x2, x3 = mbx.bf2_f32(words[1])
    part, first = _part(j * 4)
    fx.ptr_store(
        fx.Vector.from_elements([x0, x1, x2, x3], fx.Float32),
        rx + (part * RX_PART + j * 4 - first),
    )


def _chain_step(s, big, new, acc):
    """Steps past the short parts' length only advance the long parts."""
    return new if s < SHORT_STEPS else big.select(new, acc)


def m3(c, p):
    tid, lane, wave = c.tid, c.lane, c.wave
    experts = c.experts
    # Batched builds: task ``task`` runs router block ``t`` (the one-request
    # task) for request rows ``h0 .. h0 + rows - 1``: every row, or a part of
    # them per task (``layout``'s ``m3_task_rows``).
    task = task_index(c, "m3")
    rows = c.L.m3_task_rows
    split = rows < c.rows
    t, h0 = (
        (task % (experts // 8 + 1), task // (experts // 8 + 1) * rows)
        if split
        else (task, 0)
    )
    rw = c.lds_i + c.L.lds["rw"]
    rx = c.lds_f + c.L.lds["rx"]
    lgl = c.lds_f + c.L.lds["lgl"]
    rxs = [rx] + [lds_row(c, c.lds_f, "rx", r) for r in range(1, rows)]
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

        payload_wait(c, hint, C.hidden // 32 * rows)
        hi = fx.min(tid + THREADS, RX_LOADS - 1)
        v = mbx.poll(
            [
                (c.mb_mxbf, mb_ix(j * 2, h0 + r), 2)
                for r in range(rows)
                for j in (tid, hi)
            ],
            c.tag,
        )
        for r in range_constexpr(rows):
            _store_x(rxs[r], v[2 * r], tid)
            _store_x(rxs[r], v[2 * r + 1], hi)
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
        # Batched builds: lane ``16 r + q``, of request row ``r``.
        if lane < 16 * rows:
            lp = lane if rows == 1 else lane % 16
            big = lp < 8
            wrow = rw + (wave * RW_ROW + lp * RW_PART)
            xrow = rx + row_ix(
                lp * RX_PART, 0 if rows == 1 else lane // 16, c.L.stride["rx"]
            )
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
            for rt in range_constexpr(rows):
                total = read_lane_i32(bits, 16 * rt).bitcast(fx.Float32)
                for q in range_constexpr(1, 16):
                    total = total + read_lane_i32(bits, 16 * rt + q).bitcast(fx.Float32)
                if lane == 0:
                    fx.ptr_store(bf16r(total), lgl + row_ix(wave, rt, slg))
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
                mb_ix(0, h0) if rows == 1 else mb_ix(0, h0 + tid),
                bf16r(sigmoid_acc(fx.ptr_load(gl))),
                c.tag,
            )


m3 = ASTRewriter.transform(m3)


# ------------------------------------------------------------- M4


def _gu_issue(r_w, r_s, pair, part, g, lane):
    """Issue the weight and scale loads of k_iter group ``g`` of block
    ``part`` of gate / up pair ``pair``: 4 weight tiles, 2 scale dwords."""
    blk = pair * 2 + part
    ws = [
        fx.Vector(
            bo.buffer_load(
                r_w,
                ((blk * GU_KI + g * 4 + q) * 64 + lane) * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for q in range_constexpr(4)
    ]
    ss = [
        fx.Int32(
            bo.buffer_load(
                r_s, (pair * GU_K1 + g * 2 + h) * 64 + lane, vec_width=1, dtype=T.i32
            )
        )
        for h in range_constexpr(2)
    ]
    return ws, ss


def _gu_mma(acc, unit, g, part, x8, xe, lane):
    """``acc += W[group g] @ x`` for a unit from :func:`_gu_issue`."""
    ws, ss = unit
    for q in range_constexpr(4):
        kit = g * 4 + q
        ea = fx.ptr_load(xe + (kit * 4 + lane // 16))
        sw = ss[q // 2] >> (part * 8)
        acc = p2.mfma_p2(acc, a_operand(x8, kit, lane), ws[q], ea, sw, 0, (q % 2) * 2)
    return acc


def m4(c, p):
    # Batched builds from 3 rows on have more tasks than CTAs: rounds, each
    # starting with a CTA barrier ahead of its LDS writes.
    t = task_index(c, "m4")
    for k in range_constexpr(task_rounds(c, "m4")):
        _m4(c, p, t if k == 0 else t + k * GRID, k == 0)


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
        # The exponents are each M2 task's last store; the fp8 words follow.
        # Batched builds: wave ``1 + s`` fills LDS row ``s``, with request row
        # ``rr`` (s = 0) or ``s`` (the shared expert's tasks).
        if const_expr(rows == 1):
            s, src, fill = 0, 0, wave == 1
        else:
            s = fx.min(fx.max(wave - 1, 0), rows - 1)
            src = (wave == 1).select(rr, s)
            fill = (wave == 1) | (shared & (wave > 1) & (wave <= rows))
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
            ix = wave * 16 + lane
            if lane < 16:
                for rt in range_constexpr(rows):
                    fx.ptr_store(
                        acc[rt], red + row_ix(ix + j * WAVES * 16 if j else ix, rt, rs)
                    )
        gpu.barrier()

        # Lane L: column 64 q + L of gate and up. Batched builds: LDS row
        # ``rw`` on wave ``rw`` of the slice's waves (the shared expert's
        # tasks: all rows), for request row ``out``.
        if const_expr(rows == 1):
            q, rw, out, epilogue = q0, 0, 0, wave == 0
            sl = 0
        else:
            ws = wave if S == 1 else wave % 4
            sl = 0 if S == 1 else (wave // 4) * (WAVES * 16)
            q = q0 if S == 1 else q0 + wave // 4
            rw = fx.min(ws, rows - 1)
            out = shared.select(rw, rr)
            epilogue = (ws == 0) | (shared & (ws < rows))
        if epilogue:
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
                mbx.put2(
                    c.mb_dn, mb_ix(r * DN_ROW + DN_X8 + q * 2, out), ex, ex_hi, c.tag
                )


_m4 = ASTRewriter.transform(_m4)


# ------------------------------------------------------------- M5


def _dn_issue(r_w, r_s, nb, lane, off):
    """Down weights of n-block ``nb``, k_iters 0-4, its pair's 3 scale dwords
    and their shift (block ``nb % 2`` of the pair); ``off`` = ``OOB`` loads
    zeros without touching memory."""
    ws = [
        fx.Vector(
            bo.buffer_load(
                r_w, ((nb * DN_KI + k) * 64 + lane) * 4 + off, vec_width=4, dtype=T.i32
            )
        )
        for k in range(DN_STEPS)
    ]
    ss = [
        fx.Int32(
            bo.buffer_load(
                r_s,
                ((nb // 2) * DN_K1 + k1) * 64 + lane + off,
                vec_width=1,
                dtype=T.i32,
            )
        )
        for k1 in range(DN_K1)
    ]
    return ws, ss, (nb % 2) * 8


def _dn_mma(acc, unit, k, row, lane):
    """``acc += W[k_iter k] @ x`` with the activation row at LDS ``row``."""
    ws, ss, shift = unit
    ea = fx.ptr_load(row + (DN_X8 + k * 4 + lane // 16))
    sw = ss[k // 2] >> shift
    return p2.mfma_p2(acc, a_operand(row, k, lane), ws[k], ea, sw, 0, (k % 2) * 2)


def _dn_chain(unit, row, lane):
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k in range(DN_STEPS):
        acc = _dn_mma(acc, unit, k, row, lane)
    return acc


def _m5_issue_rows(c, p, t, top):
    """Batched builds: the down weights of this wave's chains ``k = wave + 8 s``
    (zero loads past the last chain), by ``s``."""
    rows, lane, wave = c.rows, c.lane, c.wave
    n = C.top_k * rows  # the shared expert's chain
    units = []
    for s in range(-(-(n + 1) // WAVES)):
        k = wave + WAVES * s
        row = fx.min(k // C.top_k, rows - 1)
        e = uniform(
            fx.ptr_load(top + row_ix(TOP_IDS + k % C.top_k, row, c.L.stride["top"]))
        )
        sh = k == n
        units.append(
            _dn_issue(
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
        )
    return units


def _m5_chains_rows(c, units, dn, ys):
    """Batched builds: run this wave's chains (:func:`_m5_issue_rows`) into
    each row's ``ys``; the shared expert's chain runs every row in MFMA rows."""
    rows, lane, wave = c.rows, c.lane, c.wave
    sdn, sys_ = c.L.stride["dn"], c.L.stride["ys"]
    n = C.top_k * rows
    for s in range_constexpr(len(units)):
        k = wave + WAVES * s
        if k < n:
            row, rank = k // C.top_k, k % C.top_k
            acc = _dn_chain(units[s], dn + row_ix(rank * DN_ROW, row, sdn), lane)
            if lane < 16:
                fx.ptr_store(bf16r(acc[0]), ys + row_ix(rank * 16 + lane, row, sys_))
        if k == n:
            # Partial L over k_iters L and L + 4, reduced in order.
            row_b = dn + row_ix(C.top_k * DN_ROW, mfma_row(c), sdn)
            unit = units[s]
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            w0 = _dn_mma(_dn_mma(zero, unit, 0, row_b, lane), unit, 4, row_b, lane)
            w1 = _dn_mma(zero, unit, 1, row_b, lane)
            w2 = _dn_mma(zero, unit, 2, row_b, lane)
            w3 = _dn_mma(zero, unit, 3, row_b, lane)
            if lane < 16:
                for rt in range_constexpr(rows):
                    shared = bf16r(((w0[rt] + w1[rt]) + w2[rt]) + w3[rt])
                    fx.ptr_store(shared, ys + row_ix(C.top_k * 16 + lane, rt, sys_))


_m5_chains_rows = ASTRewriter.transform(_m5_chains_rows)


def _m5_route(c, t, rr):
    """M5 task ``t``'s routing of request row ``rr``, on this wave: the top 10
    and their router weights, the shared gate, and lane i < 32's residual word
    (stream i // 8) and injection weight, all ready once the logits are."""
    lane = c.lane
    top = c.lds_i + c.L.lds["top"]
    topf = c.lds_f + c.L.lds["top"]
    stop = c.L.stride["top"]
    b = fx.min(lane, 31) // 8
    word = b * ROW_WORDS + t * 8 + lane % 8
    got = mbx.poll(
        logit_specs(c, lane, c.experts, rr)
        + [
            (c.mb_sg, mb_ix(0, rr), 1),
            (c.mb_w, mb_ix(word, rr), 1),
            (c.mb_minj, mb_ix(b, rr), 1),
        ],
        c.tag,
    )
    picks = top10(got[0] + got[1], lane, c.experts)
    w = router_weights(picks)
    if lane == 0:
        for k in range_constexpr(C.top_k):
            fx.ptr_store(key_expert(picks[k]), top + row_ix(TOP_IDS + k, rr, stop))
            fx.ptr_store(w[k], topf + row_ix(TOP_W + k, rr, stop))
        fx.ptr_store(got[2][0], top + row_ix(TOP_RG, rr, stop))
    if lane < 32:
        fx.ptr_store(got[3][0], top + row_ix(TOP_RES + lane, rr, stop))
        fx.ptr_store(got[4][0], top + row_ix(TOP_INJ + lane, rr, stop))


_m5_route = ASTRewriter.transform(_m5_route)


def m5(c, p):
    tid, lane, wave = c.tid, c.lane, c.wave
    rows = c.rows
    t = task_index(c, "m5")
    top = c.lds_i + c.L.lds["top"]
    topf = c.lds_f + c.L.lds["top"]
    dn = c.lds_i + c.L.lds["dn"]
    ys = c.lds_f + c.L.lds["ys"]
    stop, sdn, sys_ = c.L.stride["top"], c.L.stride["dn"], c.L.stride["ys"]

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
            units = _m5_issue_rows(c, p, t, top)

        # The exponent pair is each M4 task's last store.
        mbx.pre_poll(hint, (C.top_k + 1) * GU_TILES * rows, c.tag)
        hi = fx.min(tid + THREADS, DN_LOADS - 1)
        v = mbx.poll(
            [(c.mb_dn, mb_ix(j * 2, r), 2) for r in range(rows) for j in (tid, hi)],
            c.tag,
        )
        for r in range_constexpr(rows):
            fx.ptr_store(
                fx.Vector.from_elements(v[2 * r], fx.Int32),
                dn + row_ix(tid * 2, r, sdn),
            )
            if tid + THREADS < DN_LOADS:
                fx.ptr_store(
                    fx.Vector.from_elements(v[2 * r + 1], fx.Int32),
                    dn + row_ix(hi * 2, r, sdn),
                )
        gpu.barrier()

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
            _m5_chains_rows(c, units, dn, ys)
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
