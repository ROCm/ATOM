# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""K1's indexer projections (FULL / REINDEX layers): the index query and the
head weights the scorer reads, bit for bit the original chain
(``Indexer.project`` then ``quantize_query_rows`` and ``scale_indexer_weights``,
``p31_indexer_plan.md``). The indexer is replicated: every rank computes all 32
heads, as the original does.

    iw      (32 tasks, a row of weights_proj [32, 5120] each, a wave a token): aiter
            ``wv_splitk_small<bf16, 64, 1, 1, 8, 4, 6>`` -- a lane's 8-element
            chunks at k = k1 + 512 k2 + 8 lane in (k1, k2) order, each
            (a0 b0, a1 b1) then fma chains on the even and odd elements, the
            chunk's x + y added to the lane's sum; across lanes v_add_f32 dpp
            row_shr 8, 4, 2, wave_shr 1, row_bcast 15, row_bcast 31, lane 63
            (``harness/p31_wvsplitk_emul.py``) -> bf16 -> IW
    iq      (256 tasks, 16 rows of wq_b [4096, 1280], ``attn_pre.stage_iq``):
            the FP8 GEMV of the q latent -> bf16 -> GPT-J RoPE on each head's
            dims 64..127 -> IQ
    iquant  (S x 32 tasks, a (token, head) row of 128): aiter
            ``dynamic_per_token_scaled_quant``: amax (seeded 1e-10),
            scale = rn(amax x f32(1/448)), inv = 1 / scale (IEEE), e4m3 of
            clamp(rn(x inv), +-448); then the head weight rn(rn(w q_scale) x
            weights_scale) -> global ``iq_out`` / ``iqs_out`` / ``iw_out``;
            on the FP4 plane (``stage_iquant_fp4``) ``quantize_fp4``'s group-32
            E8M0 E2M1 instead, the scales in the row-group scorer's packed
            order and the weight rn(w x weights_scale)
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.mono.device.mx import FP8_MAX, clamp_fp8
from atom.mono.device.ops import (
    CM_NT,
    bf2_f32,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    lane_gather,
    ld_i32,
    row_shr,
    rsrc,
    traced,
    wave_umax_dpp,
)
from atom.mono.device.sync import Quad
from atom.mono.dsa_indexer.paged.fused import plan as ip
from atom.mono.plan.execution import THREADS, WAVES

IDX_ROWS = ip.HEADS * ip.DIM  # 4096
IDX_ROPE = 64  # a head's rotated tail
IDX_WEIGHTS_SCALE = ip.DIM**-0.5 * ip.HEADS**-0.5  # 1/64: exact
IW_TASKS = ip.HEADS  # a row a task, a wave a token
IQ_GROUPS = ip.DIM // 32  # FP4 groups a head's query: IQDONE flags a (token, head)
# the widest step whose FP4 query its GEMV quantizes in place: one token tile
# (``attn_pre.TILE``, an MFMA's N)
IQ_IN_PLACE_MAX = 16


def iq_in_place(s: int, index_fp4: bool) -> bool:
    """Whether the layer kernel's index query of ``s`` rows is quantized by
    its GEMV's group tasks (``iq_group_fp4``: no quantization stage, a
    flag a (token, head, group)) rather than by ``stage_iquant_fp4`` (a flag a
    (token, head)). The producer (K1) and the scorer's wait (K2a) both ask."""
    return index_fp4 and s <= IQ_IN_PLACE_MAX


AMAX_SEED = 1e-10


def scratch_pairs(tokens: int, hidden: int) -> dict[str, int]:
    return {
        "nb": tokens * hidden // 2,  # the normed row, bf16 pairs
        "iq": tokens * IDX_ROWS // 2,  # the rotated index query, bf16 pairs
        "iw": tokens * ip.HEADS,  # weights_proj out, f32 of bf16
    }


def _wv_reduce(v, lane):
    """wv_splitk's cross-lane v_add_f32 dpp chain, the value lane 63 ends with."""
    for k in (8, 4, 2):
        v = row_shr(v, lane, k) + v
    moved = lane_gather(v.bitcast(fx.Int32), fx.max(lane - 1, 0)).bitcast(fx.Float32)
    v = fx.Float32((lane >= 1).select(moved + v, v))  # wave_shr:1
    row = lane // 16
    b15 = lane_gather(v.bitcast(fx.Int32), fx.max(row * 16 - 1, 0)).bitcast(fx.Float32)
    v = fx.Float32((row % 2 == 1).select(b15 + v, v))
    b31 = lane_gather(v.bitcast(fx.Int32), 31).bitcast(fx.Float32)
    v = fx.Float32((row >= 2).select(b31 + v, v))
    return lane_gather(v.bitcast(fx.Int32), 63).bitcast(fx.Float32)


def iw_tasks(s):
    """``stage_iw``'s tasks for ``s`` tokens: a (token group of WAVES, row of
    weights_proj) each, the group major."""
    return IW_TASKS * -(-s // WAVES)


@traced
def stage_iw(c, task, hidden, nbl, load_group):
    """Row ``task`` % 32 of weights_proj for token group ``task`` // 32: wave w
    its token w of the group, the group's normed rows in LDS ``nbl`` (bf16
    pairs, row-packed; ``load_group(first token, tokens)`` puts them there) ->
    IW. A task a group, not a row's every group in series: each (row, token)
    sum keeps wv_splitk's per-lane order, so spreading the tokens over tasks
    and waves changes nothing but the latency."""
    s, lane = c["S"], c["lane"]
    a = c["args"]
    m = task % IW_TASKS
    group = task // IW_TASKS
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
                    cache_modifier=CM_NT,
                )
            )
        )
    for g in range_constexpr(0, s, WAVES):
        if group == g // WAVES:
            _iw_group(c, m, g, ks, wvs, hidden, nbl, load_group)


@traced
def _iw_group(c, m, g, ks, wvs, hidden, nbl, load_group):
    """``stage_iw``'s row ``m`` for tokens g .. g + WAVES - 1 (``wvs`` its
    weights, at ``ks``)."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    n = min(WAVES, s - g)
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
    w = _wv_reduce(total, lane)
    if (lane == 0) & (wave < n):
        if const_expr(c["tok_par"]):
            # token parallel: to every rank, at the token's row of the step
            for p in range_constexpr(c["d"].tp):
                dst = c["iw_dsts"][p]
                c["put"](dst, (c["t0"] + t) * ip.HEADS + m, bf16_round(w))
        else:
            c["put"](c["iw"], t * ip.HEADS + m, bf16_round(w))
    gpu.barrier()


def poll_query(c, t, h, lane):
    """Token t, head h: the lane's pair of the index query and the head's weight,
    one poll batch (IW in the peer buffer in a token-parallel launch: IWG)."""
    q = (c["iq"], (t * IDX_ROWS + h * ip.DIM) // 2 + lane, 1)
    w = (c["iw"], t * ip.HEADS + h, 1)
    return c["poll"]([q, w])


@traced
def stage_iquant(c, task):
    """Token task // 32, head task % 32: a wave, 2 dims a lane (wave 0)."""
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    t = task // ip.HEADS
    h = task % ip.HEADS
    if wave == 0:
        got = poll_query(c, t, h, lane)
        x0, x1 = bf2_f32(got[0][0])
        amax = butterfly(
            fx.max(fx.max(abs(x0), abs(x1)), fx.Float32(AMAX_SEED)),
            (1, 2, 4, 8, 16, 32),
            fx.max,
        )
        scale = amax * fx.Float32(1.0 / FP8_MAX)
        inv = fx.Float32(1.0) / scale
        q = fp8_pack4(
            clamp_fp8(x0 * inv), clamp_fp8(x1 * inv), fx.Float32(0.0), fx.Float32(0.0)
        )
        row = t * ip.HEADS + h
        bo.buffer_store(
            fx.Int16(q & 0xFFFF), rsrc(a["iq_out"]), row * (ip.DIM // 2) + lane
        )
        if lane == 0:
            w = got[1][0].bitcast(fx.Float32)
            bo.buffer_store(scale, rsrc(a["iqs_out"]), row)
            bo.buffer_store(
                (w * scale) * fx.Float32(IDX_WEIGHTS_SCALE), rsrc(a["iw_out"]), row
            )
        query_done(c, row)


@traced
def step_queried(c):
    """Whether this step scores a row (``index_plan.scored`` on its widest
    bound, ``ibound``: the bounds the scorer's tasks come from, written before
    the launch). The query, the head weights and their quantization feed the
    scorer alone: a step whose every row selects all it sees needs none of
    them. Alike on every CTA and rank."""
    # a row a lane (S <= 64), one wave maximum read into a scalar: a load a
    # lane whatever S is (S unrolled loads held at once spill K1 at large S),
    # and a uniform branch on an SGPR, not a VGPR live through K1
    assert c["S"] <= 64, c["S"]
    row = fx.min(c["lane"], fx.Int32(c["S"] - 1))
    return ip.scored(wave_umax_dpp(ld_i32(c["args"]["ibound"], row)))


@traced
def await_queries(c, lead, span, rows):
    """The layer kernel (``fused``): every head's query of rows lead .. lead
    + span - 1 (at most ``rows``) written (``query_done``'s IQDONE), a thread
    a (row, head), then the CTA past the waits; a step whose query is
    quantized in place (``iq_in_place``) its 4 group flags, two 16-byte polls.
    Every thread calls it."""
    if const_expr(c["fused"]):
        # a thread a (row, head): THREADS // HEADS rows a pass
        per = THREADS // ip.HEADS
        for first in range(0, rows, per):
            n = min(per, rows - first)
            k = fx.min(c["tid"], fx.Int32(n * ip.HEADS - 1))
            row = lead + fx.min(first + k // ip.HEADS, span - 1)
            f = row * ip.HEADS + k % ip.HEADS
            if const_expr(iq_in_place(c["S"], c["index_fp4"])):
                c["k1_poll"](
                    [
                        (c["iqdone"], Quad(f * (IQ_GROUPS // 2) + i), 2)
                        for i in range(IQ_GROUPS // 2)
                    ]
                )
            else:
                c["k1_poll"]([(c["iqdone"], f, 1)])
        gpu.barrier()


@traced
def query_done(c, row):
    """The layer kernel (``fused``): the wave's stores of (token, head)
    ``row`` drained, then its IQDONE flag (K2a's scorer reads them in this
    launch)."""
    if const_expr(c["fused"]):
        rocdl.s_waitcnt(vmcnt=0)
        if c["lane"] == 0:
            c["put"](c["iqdone"], row, 1)


@traced
def iq_group_weight(c, group, t0, n):
    """``iq_group_fp4``'s head weight, polled before the GEMV's wait (the head
    weights land well before the q latent): thread (token, 0) of a head's
    group 0, 0 elsewhere."""
    tid = c["tid"]
    w = fx.Int32(0)
    if (tid < 16 * n) & (tid % 16 == 0) & (group % IQ_GROUPS == 0):
        row = (t0 + tid // 16) * ip.HEADS + group // IQ_GROUPS
        w = c["poll"]([(c["iw"], row, 1)])[0][0]
    return w.bitcast(fx.Float32)


@traced
def iq_group_fp4(c, group, t0, n, rope, iw, red, red_task):
    """The layer kernel's FP4 index query in place (``iq_in_place``), group
    task ``group`` (head group // 4, dims 32 (group % 4) ..): its two GEMV
    tasks' rows in ``red`` (parts ``red_task`` apart) -> bf16 -> RoPE ->
    ``stage_iquant_fp4``'s code and scale, a thread (token, dim pair), a group
    16 lanes; group 0 of a head its weight ``iw`` too (``iq_group_weight``);
    the wave's stores drained, then the (token, head, group) IQDONE flag."""
    tid = c["tid"]
    a = c["args"]
    if tid < 16 * n:
        rp = tid % 16
        tl = tid // 16
        t = t0 + tl
        part = red + rp // 8 * red_task
        r = 2 * (rp % 8)
        v0 = bf16_round(acc_sum(part, r, tl))
        v1 = bf16_round(acc_sum(part, r + 1, tl))
        d = (group * 32 + 2 * rp) % ip.DIM
        live = d >= ip.DIM - IDX_ROPE
        cs, sn = rope
        x0 = bf16_round(live.select(v0 * cs - v1 * sn, v0))
        x1 = bf16_round(live.select(v0 * sn + v1 * cs, v1))
        amax = butterfly(fx.max(abs(x0), abs(x1)), (1, 2, 4, 8), fx.max)
        bits = (fx.max(amax, fx.Float32(FP4_MIN_AMAX)) * fx.Float32(1.0 / 6.0)).bitcast(
            fx.Int32
        )
        code = (bits >> 23) + fx.Int32(
            ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
        )
        inv = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        h = group // IQ_GROUPS
        g = group % IQ_GROUPS
        row = t * ip.HEADS + h
        byte = _e2m1(x0 * inv) | (_e2m1(x1 * inv) << 4)
        bo.buffer_store(
            fx.Int8(byte), rsrc(a["iq_out"]), row * (ip.DIM // 2) + g * 16 + rp
        )
        if rp == 0:
            bo.buffer_store(
                fx.Int8(code),
                rsrc(a["iqs_out"]),
                t * QS_BYTES + g * (16 * QS_M_TILES) + h % 16 * QS_M_TILES + h // 16,
            )
        if (rp == 0) & (g == 0):
            bo.buffer_store(iw * fx.Float32(IDX_WEIGHTS_SCALE), rsrc(a["iw_out"]), row)
        rocdl.s_waitcnt(vmcnt=0)
        if rp == 0:
            c["put"](c["iqdone"], row * IQ_GROUPS + g, 1)


# the FP4 query's E8M0 scales a token, as the row-group scorer packs them:
# [32-dim chunk][head % 16][head // 16, the M tiles padded to a dword]
QS_M_TILES = -(-(ip.HEADS // 16) // 4) * 4
QS_BYTES = ip.DIM // 32 * 16 * QS_M_TILES
FP4_MIN_AMAX = 6.0 * 2.0**-126  # the smallest normal times E2M1's max


def _e2m1(x):
    """The E2M1 nibble of ``x`` (a value over its scale): nearest-even with
    ``quantize_fp4``'s asymmetric midpoints, the sign bit x's."""
    m = abs(x)
    code = fx.Int32(0)
    for edge, value, inclusive in (
        (0.25, 1, False), (0.75, 2, True), (1.25, 3, False), (1.75, 4, True),
        (2.5, 5, False), (3.5, 6, True), (5.0, 7, False),
    ):  # fmt: skip
        past = (m >= fx.Float32(edge)) if inclusive else (m > fx.Float32(edge))
        code = fx.Int32(past.select(fx.Int32(value), code))
    sign = (x.bitcast(fx.Int32) >> 31) & 1
    return code | (sign << 3)


@traced
def stage_iquant_fp4(c, task, live):
    """``stage_iquant`` on the FP4 plane: token task // 32, head task % 32, a
    wave (each wave its own task, where ``live`` says it has one), 2 dims a
    lane and a 32-dim group 16 lanes. ``quantize_fp4`` to the bit: code =
    ceil(log2(max(amax, FP4_MIN_AMAX) x f32(1/6))), the value times 2^(127 -
    code) (its scale's exact inverse) to E2M1, the even dim the low nibble."""
    lane = c["lane"]
    a = c["args"]
    t = task // ip.HEADS
    h = task % ip.HEADS
    if live:
        got = poll_query(c, t, h, lane)
        x0, x1 = bf2_f32(got[0][0])
        amax = butterfly(fx.max(abs(x0), abs(x1)), (1, 2, 4, 8), fx.max)
        bits = (fx.max(amax, fx.Float32(FP4_MIN_AMAX)) * fx.Float32(1.0 / 6.0)).bitcast(
            fx.Int32
        )
        code = (bits >> 23) + fx.Int32(
            ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
        )
        inv = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        row = t * ip.HEADS + h
        byte = _e2m1(x0 * inv) | (_e2m1(x1 * inv) << 4)
        bo.buffer_store(fx.Int8(byte), rsrc(a["iq_out"]), row * (ip.DIM // 2) + lane)
        if lane % 16 == 0:
            bo.buffer_store(
                fx.Int8(code),
                rsrc(a["iqs_out"]),
                t * QS_BYTES
                + lane // 16 * (16 * QS_M_TILES)
                + h % 16 * QS_M_TILES
                + h // 16,
            )
        if lane == 0:
            w = got[1][0].bitcast(fx.Float32)
            bo.buffer_store(w * fx.Float32(IDX_WEIGHTS_SCALE), rsrc(a["iw_out"]), row)
        query_done(c, row)


def acc_word(w, g, col):
    """RED's word of wave ``w``'s lane (g, col) C quad."""
    return (w * 64 + g * 16 + (col ^ (4 * g))) * 4


def acc_sum(red, r, t, waves=None):
    """Row r, token t of the waves' accumulators at ``red``, summed in wave
    order from 0 (``waves``: which, in order; all of them by default)."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + (acc_word(w, r // 4, t) + r % 4))
    return tot
