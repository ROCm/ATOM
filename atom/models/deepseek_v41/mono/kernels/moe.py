# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""K2b ``moe``: one V4.1 layer's FFN, from ``ffn_normed`` to the reduced output.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S = ``tokens`` rows,
E routed experts and top-k (``MoeBuild``: the target's 384 / 6, the DSpark
draft's 128 / 3). The rank holds every routed expert's 1/TP of the
intermediate width (TP-sharded MoE) and the shared expert's 1/TP (``Dims``).
Stages, with TP4's task counts:

    xq      (S): the row as MXFP8 (amax floored ``AMAX_FLOOR``, the ceil
            code) for the routed experts (aiter ``fused_mx_quant_moe_sort``) and
            for the shared one (ATOM ``quantize_fp8``): one rule, so one copy
            -> X8 (plain words, then the token's flag X8RDY)
    router  (E / 16 / ``router_tiles`` x 4, 16-row tiles x a quarter of K):
            bf16 GEMV -> the part's f32 sum -> LOGIT
    route   (S): the parts summed in order -> bf16 logits; aiter
            ``topk_gating_kernel_opt`` (sqrtsoftplus + bias, top-k, renormalized
            x 1.5), the scores an expert a thread, the top-k on wave 0 -> ROUTE
            (DP attention: router and route on
            the rank's own step's rows alone, the routes pushed to every rank's
            RT, ``peer_layout``)
    shared  (36, 16 gate + 16 up rows): FP8 GEMV -> bf16 -> aiter silu_and_mul
            (gate <= 10, up in [-10, 10], ocml exp) -> bf16 -> SMID
    ug      (|U| x 9, 2 x (32 gate + 32 up) rows of one expert of the union U of
            every token's top-k -- the draft's |U| x 18 of 1 x (32 + 32):
            ``RouteShape.ug_groups``; w13 gate / up interleaved a 16-row tile):
            the MXFP4 x MXFP8 GEMV, a wave the gate and up 16-row tiles of one
            32-row scale block over a half (the draft's: a quarter) of K (each
            B operand from LDS read once for both), the slices summed
            pairwise; then its epilogue:
            clamp, silu x up, bf16,
            and the stage-2 input quant to MXFP8 (amax floored AMAX_FLOOR, the
            ceil code) -> MID (plain words at device scope, then the task's
            flag, UGF; down by unit: the CTA's after its last task, UGC;
            ``RouteShape.down_y``: tagged pairs, no flag)
    down    (``RouteShape.down_y``: ``run_down_y`` -- queue units of the
            shared expert's and the slots' tasks, each pick's bf16
            contribution to DY as tagged pairs, then a combine polling them;
            the arithmetic below unchanged)
            (320, 16 rows; a CTA's two run together): per expert of U the MXFP4
            x MXFP8 GEMV of its intermediate (x its routing weight -> bf16),
            summed in f32 in each token's top-k order (the reference's sum;
            aiter's atomic stage 2 rounds to bf16 at every add); the shared
            expert's FP8 down projection (quantize_fp8 of SMID) -> bf16;
            routed + shared, one bf16 rounding -> pushed to every rank's FFN region (a
            token-parallel launch: the token's owner's alone)
    reduce  (160, 32 columns): the TP partials in rank order 0 .. TP - 1
            (``sum_partials``), fp32 -> bf16 -> ``out`` (a token-parallel
            launch: this rank's tokens' rows alone, ``own_view``)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.model_ops.deepseek_v41.router import PARTS as ROUTER_PARTS
from atom.model_ops.deepseek_v41.router import (
    router_k_offsets,
    router_tile_mfma,
    router_weight_loads,
    router_x_loads,
)
from atom.models.deepseek_v41.mono import moe_plan as mplan
from atom.models.deepseek_v41.mono.config import MAX_MOE_ROWS
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1

# helpers the kernel calls are imported by name: FlyDSL keys a build by the
# source of the same-directory functions it names, never through a module
from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    TILE,
    flat_span,
    gemv_fp8_loads,
    gemv_fp8_mfmas,
    lds_row,
    lds_tail,
    lin_rstd,
    lssq_specs,
    plus,
    poll_copy,
    tile_rows,
    token_tiles,
)
from atom.models.deepseek_v41.mono.kernels.debug import mailbox
from atom.models.deepseek_v41.mono.kernels.dims import (
    ROUTE_COPIES,
    SHARED_INTER,
    UG_PART,
)
from atom.models.deepseek_v41.mono.kernels.mailbox import epoch_of, preg, region_addrs
from atom.models.deepseek_v41.mono.kernels.moe_shape import (
    SORT_NETS,
    MoeBuild,
    route_shape,
)
from atom.models.deepseek_v41.mono.kernels.mx import AMAX_FLOOR
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.kernels.red import acc_sum
from atom.models.deepseek_v41.mono.kernels.token_par import own_count, own_view
from atom.models.deepseek_v41.mono.kernels.wave_max import wave_max_key
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device import queue, route
from atom.mono.device.mx import (
    FP4,
    FP8,
    FP8_MAX,
    clamp_fp8,
    code_ceil,
    fp4_tile_load,
    mfma_scaled,
    rcp_pow2,
    scale_index,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    _hw_f32,
    ballot,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_rcp,
    kernel_symbol,
    lane_gather,
    lanes_below,
    launcher,
    ld_f32,
    lds_bytes,
    permlane_swap,
    popcount,
    readlane,
    row_sum,
    rsrc,
    st_dev,
    traced,
    uniform,
)
from atom.mono.device.ranks import peer_bases, sum_partials
from atom.mono.device.select import KEY_LEAST, f32_key
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Quad, publish
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, LDS_BYTES, THREADS, WAVES, first_task
from atom.mono.plan.layout import pair_layout
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi

HIDDEN = k1.HIDDEN
# a pick tile is a token tile's MFMA N: ``gather_x8r`` fills one from the other
assert mplan.TILE == TILE
SCALE = 1.5
LIMIT = 10.0  # swiglu_limit
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
ROWS = k1.ROWS
DOWN_TASKS = HIDDEN // ROWS  # 320
# a down unit's hidden block (``down_units``): a wave DOWN_WAVE_TASKS adjacent
# 16-row tasks (one 32-row scale block), WAVES waves
DOWN_WAVE_TASKS = 2
DOWN_UNIT_BLOCKS = DOWN_TASKS // (WAVES * DOWN_WAVE_TASKS)  # 20
# (task, slot) scale blocks a down wave holds a round (``stage_down``): 3 took
# no less time, 4 spills
DOWN_R_MAX = 2
# a down slot's K steps on this many independent MFMA accumulators
DOWN_CHAINS = 2
# a reduce task's columns: at 128 a thread polled 6 pairs at 48 rows, each
# pass a round trip in series (32: 1-2 passes, S=48 -1 us)
RED_COLS = 32
RED_TASKS = HIDDEN // RED_COLS  # 160
X8_WORDS = HIDDEN // 4
TAGS = 256  # mailbox tags (a layer's K1 / K2a / K2b each own one)
# The task queues: a word each a launch, counting up from the step's clear --
# the ug tasks past the first round (``take_ug_task``), the down arrivals
# (``take_down_rank``), the down units (``take_down_unit``). Their own words,
# not the scratch's: the layer builds of a step lay the scratch out apart, and
# one build's pairs at another's queue offset start its count past zero.
QUEUES = ("ugq", "dq", "dnq")
# a queue's words, TAGS + 1 slots (tag t's at ``queue.tag_offset(t)``): one
# region a queue, so a launch's three words sit in three cache lines, not one
# its CTAs' atomics on all three contend for
QUEUE_REGION = (TAGS + 1) * 4
# in the step zeroing's 16 B stores
QUEUE_BYTES = -(-len(QUEUES) * QUEUE_REGION // 16) * 16


def queue_offset(layer: int) -> int:
    """The byte offset of the launch ``layer``'s word in the first queue's
    region (``queue.tag_offset``): the launch is handed its address
    (``queue_words``), queue i's ``i * QUEUE_REGION`` on."""
    return queue.tag_offset(layer)


XQ0 = 0
DOWN0 = 0
RED0 = 0
_CURRENT_STREAM = fx.Stream(None)


# ``timeline`` stamps: kernel start, then the end of each stage's task loop
TL_STAGES = ("xq", "router", "route", "shared", "ug", "down", "reduce")
TL_POINTS = 1 + len(TL_STAGES)


ABI = KernelAbi(
    (
        "x", "gate_w", "bias", "w13", "w13_s", "w2", "w2_s", "sgu", "sgu_s", "sw2",
        "sw2_s", "out", "queue_words", "scratch", "numa_word", "sym", "peers", "rank",
        "layer", "tl", "epoch",
    )
)  # fmt: skip


def scratch_layout(key: MoeBuild) -> dict[str, tuple[int, int]]:
    tokens, experts, topk = key.tokens, key.experts, key.topk
    rs = route_shape(key)
    d = rs.dims
    pairs = {
        # the MXFP8 x of the routed experts (aiter fused_mx_quant_moe_sort) and
        # of the shared one (ATOM quantize_fp8): one rule (``xq_quant``)
        "x8": tokens * X8_WORDS // 2,
        "x8s": tokens * HIDDEN // 32 // 2,
        # a token's two MXFP8 regions written (plain words): ``stage_xq``
        "x8rdy": tokens,
        "logit": tokens * experts * ROUTER_PARTS,
        "route": ROUTE_COPIES * tokens * 2 * topk,
        "smid": tokens * d.sh_inter,
        # SMID quantize_fp8'd (``smid_quant``): the down stage's MXFP8 rows
        "smq": tokens * d.sh_inter // 4,
        "smc": tokens * d.sh_inter // 32,
        # a pick a row, in the slot order (``RouteShape.order0``)
        "mid": rs.picks * d.mid_words,
        "mids": rs.picks * d.inter // 32,
        # an ug task's done flag (``stage_ug``)
        "ugf": (rs.max_tiles if rs.multi else rs.max_u) * rs.ug_parts,
        # down by unit: each pick's bf16 contribution [pick][HIDDEN] (packed
        # pairs), and a CTA's units-done flag (``down_units``)
        "dcon": rs.picks * HIDDEN // 4,
        # down by queue units (``run_down_y``): each pick's bf16 contribution
        # [pick][HIDDEN] and the shared expert's rows [task][token][16], both
        # tagged bf16 pairs
        "dy": rs.picks * HIDDEN // 2,
        "sdo": DOWN_TASKS * tokens * ROWS // 2,
        "dnf": BLOCKS,
        # down by unit: a CTA's ug-tasks-done flag (``down_by_unit``)
        "ugc": BLOCKS,
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: MoeBuild) -> int:
    return sum(n for _, n in scratch_layout(key).values())


def region_pair(src, t, s, col):
    return (src * s + t) * (HIDDEN // 2) + col // 2


# ---------------------------------------------------------------- xq
@traced
def stage_xq(c, t):
    """Token t, a group of 32 a thread."""
    # thread g < 160 owns the 32 values of group g; the rest mirror thread 159
    g = fx.min(c["tid"], HIDDEN // 32 - 1)
    # 32 bf16 as four 16-byte loads (a load an element made the stage issue bound)
    words = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["x"]),
                (t * HIDDEN + g * 32) // 2 + 4 * q,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=c["x_cm"],
            )
        )
        for q in range(4)
    ]
    xq_quant(c, t, g, [words[i // 4][i % 4] for i in range(16)])


@traced
def stage_xq_pushed(c, t):
    """DP attention: token t another rank's, its row read where the owner
    pushed it (XG) -- no copy into x, no flag between (``layer_post.push_x``)."""
    g = fx.min(c["tid"], HIDDEN // 32 - 1)
    xg = preg(c["sym"], c["peer"]["xg"], "xg")
    got = c["poll"]([(xg, t * (HIDDEN // 2) + g * 16 + 2 * k, 2) for k in range(8)])
    xq_quant(c, t, g, [got[i // 2][i % 2] for i in range(16)])


@traced
def xq_quant(c, t, g, words):
    """Token t's group g (16 words of bf16 pairs) as MXFP8: the routed
    experts' and the shared one's (one rule)."""
    mine = c["tid"] < HIDDEN // 32
    v = [(bf_lo if i % 2 == 0 else bf_hi)(words[i // 2]) for i in range(32)]
    amax = fx.Float32(AMAX_FLOOR)
    for y in v:
        amax = fx.max(amax, abs(y))
    if mine:
        put_mxfp8_group(c, "x8", "x8s", t, g, v, amax)


def put_mxfp8_group(c, words_mb, codes_mb, t, g, v, amax):
    """Token t's group g of 32 values ``v`` as MXFP8 (the ceil code of
    ``amax`` / 448, x / scale clamped, RNE) -> scratch ``words_mb`` /
    ``codes_mb`` (plain words: the token's X8RDY announces them)."""
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    inv = rcp_pow2(code)
    words = []
    for w4 in range_constexpr(4):
        ws = [
            fp8_pack4(*[clamp_fp8(v[8 * w4 + 4 * h + i] * inv) for i in range(4)])
            for h in range(2)
        ]
        words.append(ws)
    for h in range_constexpr(2):
        st_dev(
            fx.Vector.from_elements(words[2 * h] + words[2 * h + 1], fx.Int32),
            c[words_mb].value,
            t * X8_WORDS + g * 8 + 4 * h,
        )
    st_dev(fx.Int32(code), c[codes_mb].value, t * (HIDDEN // 32) + g)


# ---------------------------------------------------------------- router
# a router task's K part, and its units (4 bf16, one poll) a token
# (``router_x_normed``)
ROUTER_K = HIDDEN // ROUTER_PARTS
ROUTER_UNITS = ROUTER_K // 4


def router_weights(c, task):
    """``stage_router``'s gate weight loads for ``task``: they do not wait for
    x, so a CTA issues them before x's wait. A list a row tile."""
    rs = c["rs"]
    tile0 = task % rs.router_group_tasks // ROUTER_PARTS * rs.router_tiles
    part = task % ROUTER_PARTS
    return [
        router_weight_loads(
            c["args"]["gate_w"], tile0 + i, part, c["lane"], c["wave"], CM_NT
        )
        for i in range(rs.router_tiles)
    ]


@traced
def stage_router(c, task, wvs):
    """Router task (token group, row tiles, K part): K part ``part`` of the
    ``router_tiles`` 16-row tiles from ``tile0`` of the bf16 gate weight
    [experts, 5120] times the group's
    tokens (``RouteShape.router_groups``), a token tile at a time -- the
    original path's decode router (``atom.model_ops.deepseek_v41.router``), the
    same tiles in the same order -> the part's f32 sum (``route`` sums the
    parts in order, rounds to bf16). Every tile's x loads issue first: a task
    over every token's tiles in series read their x serially (48 rows: a router
    CTA 9 us, 8 of them x)."""
    s, tid, red, lane, wave = c["S"], c["tid"], c["red"], c["lane"], c["wave"]
    a = c["args"]
    experts = c["rs"].experts
    groups, per = c["rs"].router_groups
    per_group = c["rs"].router_group_tasks
    tiles_a_task = c["rs"].router_tiles
    tile0 = task % per_group // ROUTER_PARTS * tiles_a_task
    part = task % ROUTER_PARTS
    g0 = 0 if groups == 1 else task // per_group * per
    # the group's token tiles: (first token, tokens), tokens traced past one group
    tiles = [
        (plus(g0, j0), min(TILE, s - j0) if groups == 1 else fx.min(
            fx.Int32(min(TILE, per - j0)), s - plus(g0, j0)))
        for j0 in range(0, per, TILE)
    ]  # fmt: skip
    if const_expr(c.get("dot_norm") is None):
        xs = [
            router_x_loads(a["x"], part, n, lane, wave, c["x_cm"], t0)
            for t0, n in tiles
        ]
    else:
        xs = router_x_normed(c, part, g0, per, tiles)
    for (t0, n), xvs in zip(tiles, xs):
        for i in range_constexpr(tiles_a_task):
            router_tile_mfma(wvs[i], xvs, lane, wave, red + i * WAVES * 256)
        gpu.barrier()
        # thread (row tile, token, row): a tile's 256 threads, TILE tokens
        i, q = tid // (ROWS * TILE), tid % (ROWS * TILE)
        r, tl = q % ROWS, q // ROWS
        if (tid < tiles_a_task * ROWS * TILE) & (tl < n):
            v = row_sum(red + i * WAVES * 256, r, tl)
            row = (tile0 + i) * ROWS + r
            c["put"](
                c["logit"], (plus(t0, tl) * experts + row) * ROUTER_PARTS + part, v
            )
        gpu.barrier()


def router_x_normed(c, part, g0, per, tiles):
    """``stage_router``'s x operands (``router_x_loads``' layout) from the FFN
    seam's LIN (``dot_norm``: the seam mailbox's poll, its LIN and LSSQ, the
    norm weight), the task normalizing its own K part with the norm stage's
    bits (``lin_rstd``, bf16((x rstd) w)): the router waits for the slice,
    not for the norm. Every LIN word of
    the part polled once (a thread 4 words a unit) and landed normalized in LDS
    (``rxl``, a token a row), wave w taking the rstd of the group's tokens
    w, w + 8, .. (LDS past the x tile) in the same poll; the MFMA
    operands then read from LDS."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    dn = c["dot_norm"]
    xl = c["rxl"]
    rs_lds = xl + per * lds_row(ROUTER_K // 2)
    units = per * ROUTER_UNITS
    rounds = -(-units // THREADS)
    us = [fx.min(tid + THREADS * i, units - 1) for i in range(rounds)]
    cols = [part * ROUTER_K + u % ROUTER_UNITS * 4 for u in us]
    # the norm weights, which no wait decides: issued first
    gws = [
        fx.Vector(bo.buffer_load(rsrc(dn["w"]), col // 2, vec_width=2, dtype=T.i32))
        for col in cols
    ]
    waves_tokens = -(-per // WAVES)
    specs = []
    for k in range_constexpr(waves_tokens):
        tok = fx.min(plus(g0, wave + WAVES * k), s - 1)
        specs += lssq_specs(dn["lssq"], tok, lane)
    for u, col in zip(us, cols):
        tok = fx.min(plus(g0, u // ROUTER_UNITS), s - 1)
        specs.append((dn["lin"], Quad((tok * HIDDEN + col) // 4), 2))
    # a retry re-loads only a lane's stale specs: past the first pass the
    # 48 router CTAs' 2.2 MB a pass shrinks to the pairs still outstanding
    got = dn["poll"](specs, stale_only=True)
    per_tok = len(lssq_specs(dn["lssq"], 0, lane))
    for k in range_constexpr(waves_tokens):
        r = lin_rstd(lane, got[k * per_tok : (k + 1) * per_tok])
        j = wave + WAVES * k
        put_lds_live(rs_lds + j, r.bitcast(fx.Int32), (lane == 0) & (j < per))
    gpu.barrier()
    e = waves_tokens * per_tok
    for i in range_constexpr(rounds):
        u = us[i]
        r = fx.Int32(fx.ptr_load(rs_lds + u // ROUTER_UNITS)).bitcast(fx.Float32)
        words = got[e + i]
        out = [
            bf16_pair(
                (bf_lo(words[q]) * r) * bf_lo(gws[i][q]),
                (bf_hi(words[q]) * r) * bf_hi(gws[i][q]),
            ).bitcast(fx.Int32)
            for q in range(2)
        ]
        at = u // ROUTER_UNITS * lds_row(ROUTER_K // 2) + u % ROUTER_UNITS * 2
        fx.ptr_store(fx.Vector.from_elements(out, fx.Int32), xl + at)
    gpu.barrier()
    xs = []
    for j0 in range(0, per, TILE):
        n = min(TILE, per - j0)
        row = j0 + fx.min(lane % 16, n - 1)
        xs.append(
            [
                fx.ptr_load(
                    xl + (row * lds_row(ROUTER_K // 2) + (k0 - part * ROUTER_K) // 2),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                for k0 in router_k_offsets(part, lane, wave)
            ]
        )
    return xs


@traced
def put_lds_live(ptr, v, live):
    if live:
        fx.ptr_store(v, ptr)


# ---------------------------------------------------------------- route
def _score(x):
    """sqrt(log2(1 + exp2(x log2e)) ln2), x itself past 20 (topk_gating)."""
    sp = fx.Float32((x > fx.Float32(20.0)).select(
        x, _hw_f32("llvm.amdgcn.log.f32", fx.Float32(1.0) + hw_exp2(x * fx.Float32(LOG2E)))
        * fx.Float32(LN2),
    ))  # fmt: skip
    return fx.Float32(fmath.sqrt(sp))


@traced
def route_scores(c, t):
    """Token t's scores, an expert a thread over every wave: its router parts
    polled and summed in part
    order -> bf16 logit -> score (unbiased, NaN as 0) and its biased value ->
    LDS ``red`` (values at the expert, unbiased scores ``experts`` past)."""
    tid, red = c["tid"], c["red"]
    experts = c["rs"].experts
    assert experts <= THREADS
    if tid < experts:
        # the bias, which the logits do not decide: loaded before their wait
        bias = ld_f32(c["args"]["bias"], tid)
        assert ROUTER_PARTS == 4
        got = c["poll"](
            [(c["logit"], Quad((t * experts + tid) * 2 + h), 2) for h in (0, 1)]
        )
        parts = got[0] + got[1]
        tot = fx.Float32(0.0)
        for p in range_constexpr(ROUTER_PARTS):
            tot = tot + parts[p].bitcast(fx.Float32)
        sc = _score(bf16_round(tot))
        # aiter's NaN guards: a NaN score weighs 0 and never wins (a select:
        # the max propagates a NaN, whose order key ranks above +inf)
        biased = sc + bias
        fx.ptr_store(
            fx.Float32(fx.isnan(biased).select(fx.Float32(float("-inf")), biased)),
            red + tid,
        )
        fx.ptr_store(
            fx.Float32(fx.isnan(sc).select(fx.Float32(0.0), sc)), red + (experts + tid)
        )


@traced
def stage_route(c, t):
    """aiter topk_gating_kernel_opt for token t: the scores on every wave
    (``route_scores``), the top-k on wave 0."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    experts, topk = c["rs"].experts, c["rs"].topk
    route_scores(c, t)
    gpu.barrier()
    if wave == 0:
        ept = experts // 64
        # a lane's keys sorted with their experts only: each winner's
        # unbiased score is read back from LDS by its expert at the end
        vals = [f32_key(fx.ptr_load(red + (lane + 64 * i))) for i in range(ept)]
        idxs = [fx.Int32(lane + 64 * i) for i in range(ept)]
        for p, q in c["rs"].sort_net:
            sw = vals[p] < vals[q]
            vals[p], vals[q] = sw.select(vals[q], vals[p]), sw.select(vals[p], vals[q])
            idxs[p], idxs[q] = sw.select(idxs[q], idxs[p]), sw.select(idxs[p], idxs[q])
        # round k's winner read off the winning lane (the lowest lane holding
        # the wave's max), which then pops its head: a round reads every
        # lane's head only.
        # A lane whose list runs out (ept < topk: the draft's 2 a lane, top 3)
        # takes KEY_LEAST, below every score's key: 64 ept >= topk, so a live
        # entry wins every round
        assert 64 * ept >= topk
        ids = []
        for k in range_constexpr(topk):
            best = wave_max_key(vals[0])
            win = fx.Int32(fmath.cttz(ballot(vals[0] == best)))
            ids.append(readlane(idxs[0], win))
            popped = lane == win
            # only the entries a later round can still reach
            reach = min(ept, topk - k)
            for j in range_constexpr(reach - 1):
                vals[j] = popped.select(vals[j + 1], vals[j])
                idxs[j] = popped.select(idxs[j + 1], idxs[j])
            if const_expr(reach == ept):
                vals[ept - 1] = popped.select(fx.Int32(KEY_LEAST), vals[ept - 1])
        my_id = ids[0]
        for k in range_constexpr(1, topk):
            my_id = (lane == k).select(ids[k], my_id)
        # lane k's winner's unbiased score (``route_scores``' LDS, NaN as 0),
        # the k winners' summed in top-k order
        my_w = fx.Float32(fx.ptr_load(red + (experts + my_id)))
        total = fx.Float32(0.0)
        for k in range_constexpr(topk):
            total = total + readlane(my_w, k)
        scale = fx.Float32(SCALE) / fx.max(total, fx.Float32(1e-20))
        if lane < topk:
            # token t of the routed rows is the step's row ``route_t0`` + t
            tg = plus(c["route_t0"], t)
            for dst in c["route_dsts"]:
                for r in range_constexpr(ROUTE_COPIES):
                    at = (r * c["route_rows"] + tg) * 2 * topk + lane
                    c["put"](dst, at, my_id)
                    c["put"](dst, at + topk, my_w * scale)
    gpu.barrier()


# a step's picks at most for ``load_route_wave``: one a lane of wave 0
WAVE_ROUTE_PICKS = 64


def wave_route(rs) -> bool:
    """Whether ``load_route`` builds the table on wave 0 (``load_route_wave``):
    every pick a lane of one wave, under TP (no expert parallel)."""
    return rs.ep == 1 and rs.picks <= WAVE_ROUTE_PICKS


def load_route(c, tab):
    """The route table in LDS ``tab`` (``RouteShape``): ``load_route_wave`` for
    a step's picks a lane of one wave, else ``load_route_bitmaps``. Both lay
    out the same words the kernel reads."""
    if const_expr(wave_route(c["rs"])):
        load_route_wave(c, tab)
    else:
        load_route_bitmaps(c, tab)


def _dpp(x, ctrl):
    """DPP ``ctrl`` of x (every row and bank; a lane reading outside its row
    takes 0: no lane here uses such a value)."""
    return fx.Int32(update_dpp_i32(fx.Int32(0), x, ctrl, 0xF, 0xF, True))


def _swap_min_max(x, off):
    """(min, max) of x and lane ``lane ^ off``'s x, off 16 or 32: a
    v_permlane*_swap gives a lane its own value and its partner's."""
    a, b = permlane_swap(off, x, x)
    return fx.min(a, b), fx.max(a, b)


def _wave_sort(x, lane, size):
    """The wave's i32 keys sorted ascending in lane order within each run of
    ``size`` (32 or 64) lanes: a bitonic network in its flip form, every
    compare ascending (a block's first step compares lane i with i ^ (k - 1),
    the lower lane keeping the min, then the half-cleaners i ^ j). Partners by
    DPP within a row (quad perms, row mirrors, row shifts) and the permlane
    swaps across rows: no LDS, no barrier."""
    k = 2
    while k <= size:
        low = (lane & (k // 2)) == 0
        if k == 2:
            p = _dpp(x, 0xB1)  # lane ^ 1
        elif k == 4:
            p = _dpp(x, 0x1B)  # quad mirror: lane ^ 3
        elif k == 8:
            p = _dpp(x, 0x141)  # half-row mirror: lane ^ 7
        else:
            p = _dpp(x, 0x140)  # row mirror: lane ^ 15
            if k >= 32:
                a, b = permlane_swap(16, p, p)
                p = ((lane & 16) != 0).select(a, b)  # lane ^ 31
            if k == 64:
                a, b = permlane_swap(32, p, p)
                p = ((lane & 32) != 0).select(a, b)  # lane ^ 63
        x = low.select(fx.min(x, p), fx.max(x, p))
        j = k // 4
        while j >= 1:
            low = (lane & j) == 0
            if j >= 16:
                mn, mx = _swap_min_max(x, j)
            elif j >= 4:
                # the lower lane's partner j above it (row_shl), the upper's
                # j below (row_shr)
                mn = fx.min(x, _dpp(x, 0x100 + j))
                mx = fx.max(x, _dpp(x, 0x110 + j))
            else:
                p = _dpp(x, 0xB1 if j == 1 else 0x4E)  # lane ^ 1, lane ^ 2
                mn, mx = fx.min(x, p), fx.max(x, p)
            x = low.select(mn, mx)
            j //= 2
        k *= 2
    return x


@traced
def load_route_wave(c, tab):
    """``load_route_bitmaps``' table for a step's picks a lane of wave 0
    (``wave_route``), with one barrier: wave 0 polls the picks' ids, a pick a
    lane, wave 1 their weights (every route word in one batch, as there), and
    wave 0 alone lays out the rest from its lanes.

    The picks' keys (expert, pick) sorted ascending (``_wave_sort``) are the
    pick order, a slot's picks by token ascending as the bitmaps' ranks give
    them (a token picks an expert once): lane r holds the pick at position r.
    A slot starts where the expert changes; its slot is the starts below it
    (a ballot), and U, the slots' first positions and, past one token tile,
    their pick tiles follow from the starts' ballots. Each expert of U gets its
    slot (``RouteShape.expert_slot0``); the other experts' entries, and the
    token bitmaps, are read by ``load_route_bitmaps`` alone, and not written."""
    s, wave = c["S"], c["wave"]
    # tid and lane afresh: the lane masks and addresses built on them stay
    # here, not hoisted out of the ug loop into registers held across stages
    tid = fresh(c["tid"])
    lane = tid & 63
    rs = c["rs"]
    topk = rs.topk
    n = s * topk
    assert wave_route(rs) and 2 * WAVE_ROUTE_PICKS <= THREADS
    # thread tid's route word: wave 0 lane p pick p's id, wave 1 lane p its
    # weight (a lane past the picks the last pick's); past wave 1 the last
    # word, re-read and not stored (``load_route_bitmaps``' poll's)
    pk = fx.min(lane, n - 1)
    word = pk // topk * (2 * topk) + pk % topk
    word = (tid < 64).select(word, (tid < 128).select(word + topk, fx.Int32(2 * n - 1)))
    copy_at = c["bid"] % ROUTE_COPIES * (2 * n)
    got = c["poll"]([(c["route_src"], copy_at + word, 1)])
    if (tid < 128) & (lane < n):
        fx.ptr_store(got[0][0], tab + word)
    if wave == 0:
        mine = lane < n
        # (expert, pick) keys, past the picks above every key
        key = mine.select(got[0][0] * 64 + lane, fx.Int32(rs.experts * 64))
        key = _wave_sort(key, lane, 32 if n <= 32 else 64)
        e, pick = key >> 6, key & 63
        # the previous position's expert (DPP wave_shr:1; lane 0 takes none)
        e_prev = _dpp(key, 0x138) >> 6
        start = mine & ((lane == 0) | (e != e_prev))
        starts = ballot(start)
        n_u = popcount(starts)
        below = lanes_below(starts, lane)
        slot = start.select(below, below - 1)
        if mine:
            fx.ptr_store(pick, tab + (rs.order0 + lane))
            fx.ptr_store(slot, tab + (2 * rs.picks + 1 + rs.max_u + pick))
        if start:
            fx.ptr_store(e, tab + (2 * n + 1 + slot))
            fx.ptr_store(lane, tab + (rs.first0 + slot))
            fx.ptr_store(slot, tab + (rs.expert_slot0 + e))
        if lane == 0:
            fx.ptr_store(n_u, tab + 2 * n)
            # the picks of U: every pick under TP
            fx.ptr_store(fx.Int32(n), tab + (rs.first0 + n_u))
        if const_expr(rs.multi):
            # a slot's picks run to its last position: the next start's below,
            # or the last pick
            last = (starts >> 1) | (fx.Int64(1) << (n - 1))
            count = fx.Int32(fmath.cttz(last >> fx.Int64(lane))) + 1
            tiles = mplan.pick_tiles(count)
            # the slots' pick tiles below each start: a slot's 1 + its tiles
            # past each further TILE picks
            tf = below
            total = n_u
            for t in range_constexpr(1, rs.slot_tiles):
                more = ballot(start & (tiles > t))
                tf = tf + lanes_below(more, lane)
                total = total + popcount(more)
            if start:
                fx.ptr_store(tf, tab + (rs.tile_first0 + slot))
                for j in range_constexpr(rs.slot_tiles):
                    if j < tiles:
                        fx.ptr_store(slot, tab + (rs.tile_slot0 + tf + j))
            if lane == 0:
                fx.ptr_store(total, tab + (rs.tile_first0 + n_u))
    gpu.barrier()


@traced
def load_route_bitmaps(c, tab):
    """Every token's (ids, weights) into LDS ``tab``, then the union U of the ids
    (ascending): tab[2 S topk ..] = |U|, the experts of U, each (t, k)'s slot in
    U, each expert's slot, the picks in slot order (a slot's ascending) and each
    slot's first position (``RouteShape``).

    A pick a thread, no scan in series: each pick ORs its token's bit into its
    expert's token bitmap, so an expert's pick count is its bitmap's popcount,
    a slot's first position the counts of the experts below it (a wave's prefix
    sum; slots follow the experts ascending), and a pick's rank in its slot the
    bitmap's tokens below its own. The bitmaps are the same whatever order the
    ORs land in, so every CTA derives the same pick order."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rs = c["rs"]
    topk, experts = rs.topk, rs.experts
    n = s * topk
    mw = rs.mask_words
    # a thread's route words in one poll batch: a pass each was a round trip
    # each in series past 256 picks (48 rows: 5 us from the last route to the
    # table); a word past 2n re-reads the last and is not stored
    idxs = [plus(pas, tid) for pas in range(0, 2 * n, THREADS)]
    copy_at = c["bid"] % ROUTE_COPIES * (2 * n)
    got = c["poll"](
        [(c["route_src"], copy_at + fx.min(idx, 2 * n - 1), 1) for idx in idxs]
    )
    for i in range_constexpr(len(idxs)):
        if idxs[i] < 2 * n:
            fx.ptr_store(got[i][0], tab + idxs[i])
    for pas in range_constexpr(0, mw * experts, THREADS):
        idx = plus(pas, tid)
        if idx < mw * experts:
            fx.ptr_store(fx.Int32(0), tab + (rs.mask0 + idx))
    gpu.barrier()
    # a pick a thread, past THREADS picks a pass each
    for p0 in range_constexpr(0, n, THREADS):
        pick = plus(p0, tid)
        if pick < n:
            t = pick // topk
            eid = fx.ptr_load(tab + (t * 2 * topk + pick % topk))
            # expert parallel: another rank's experts keep an empty bitmap, so
            # they never enter U and their picks no slot
            if const_expr(rs.ep == 1) or is_local(c, eid):
                mark_pick(tab, rs, t, eid)
    gpu.barrier()
    base = 2 * n
    # a chunk of 64 experts a wave (wave 0 alone through 6 chunks in series
    # was 3.3 us of the table's 5 at 48 rows): each chunk's own counts and
    # prefixes, its totals through LDS ``red`` (free here), then each chunk
    # past the totals of the chunks below it -- the serial walk's values
    chunks = experts // 64
    assert chunks <= WAVES
    sums = c["red"]
    e = lane + 64 * fx.min(wave, chunks - 1)
    count = fx.Int32(0)
    for w in range_constexpr(mw):
        word = fx.ptr_load(tab + (rs.mask0 + e * mw + w))
        count = count + popcount(word)
    ball = ballot(count != 0)
    # this chunk's inclusive prefixes, lanes ascending: picks, and pick tiles
    incl = lane_prefix(count, lane)
    tiles = mplan.pick_tiles(count)
    tincl = lane_prefix(tiles, lane) if rs.multi else tiles
    totals = [popcount(ball), lane_gather(incl, 63), lane_gather(tincl, 63)]
    if (wave < chunks) & (lane == 0):
        for k in range_constexpr(3):
            fx.ptr_store(totals[k].bitcast(fx.Float32), sums + (wave * 3 + k))
    gpu.barrier()
    below_c = [fx.Int32(0), fx.Int32(0), fx.Int32(0)]  # n_u, first, tile0
    for j in range_constexpr(chunks - 1):
        for k in range_constexpr(3):
            v = fx.ptr_load(sums + (j * 3 + k)).bitcast(fx.Int32)
            below_c[k] = below_c[k] + (j < wave).select(v, fx.Int32(0))
    n_u, first, tile0 = below_c
    slot = n_u + lanes_below(ball, lane)
    # e afresh: hoisted, its address held a VGPR across the prefix sums (spill)
    e_at = fresh(e)
    if wave < chunks:
        fx.ptr_store(slot, tab + (rs.expert_slot0 + e_at))
        if count != 0:
            fx.ptr_store(fx.Int32(e), tab + (base + 1 + slot))
            fx.ptr_store(first + incl - count, tab + (rs.first0 + slot))
        if const_expr(rs.multi):
            # the slot's pick tiles: their first index, and each one's slot
            tf = tile0 + tincl - tiles
            if count != 0:
                fx.ptr_store(tf, tab + (rs.tile_first0 + slot))
            for j in range_constexpr(rs.slot_tiles):
                if j < tiles:
                    fx.ptr_store(slot, tab + (rs.tile_slot0 + tf + j))
    if (wave == chunks - 1) & (lane == 0):
        n_all = n_u + totals[0]
        fx.ptr_store(n_all, tab + base)
        # the picks of U: every pick under TP
        fx.ptr_store(first + totals[1], tab + (rs.first0 + n_all))
        if const_expr(rs.multi):
            fx.ptr_store(tile0 + totals[2], tab + (rs.tile_first0 + n_all))
    gpu.barrier()
    for p0 in range_constexpr(0, n, THREADS):
        pick = plus(p0, tid)
        if pick < n:
            t = pick // topk
            eid = fx.ptr_load(tab + (t * 2 * topk + pick % topk))
            if const_expr(rs.ep == 1) or is_local(c, eid):
                place_pick(c, tab, pick, t, eid)
    gpu.barrier()


def mark_pick(tab, rs, t, eid):
    """Token t into expert eid's token bitmap (``load_route``)."""
    route.mark_token(tab + rs.mask0, rs.mask_words, eid, t)


@traced
def place_pick(c, tab, tid, t, eid):
    """Pick ``tid`` (token t, expert eid of U): its slot of U, and its position
    in the pick order (``load_route``)."""
    rs = c["rs"]
    slot = fx.ptr_load(tab + (rs.expert_slot0 + eid))
    fx.ptr_store(slot, tab + (2 * rs.picks + 1 + rs.max_u + tid))
    # the tokens below t that picked this expert: the picks before tid in it
    rank = route.tokens_below(tab + rs.mask0, rs.mask_words, eid, t)
    pos = fx.ptr_load(tab + (rs.first0 + slot)) + rank
    fx.ptr_store(tid, tab + (rs.order0 + pos))


def local_expert(c, e):
    """Routed expert e's index among this rank's weights (expert parallel:
    from its first, ``e0``)."""
    return e if const_expr(c["rs"].ep == 1) else e - c["e0"]


def is_local(c, eid):
    """Whether routed expert ``eid`` is this rank's (expert parallel)."""
    e0 = c["e0"]
    return (eid >= e0) & (eid < e0 + c["rs"].local_experts)


def lane_prefix(v, lane):
    """The wave's inclusive prefix sum of v, lanes ascending."""
    incl = v
    for d in (1, 2, 4, 8, 16, 32):
        moved = lane_gather(incl, fx.max(lane - d, 0))
        incl = incl + (lane >= d).select(moved, fx.Int32(0))
    return incl


# ---------------------------------------------------------------- shared ug
def _silu_mul(g, u):
    """aiter silu_and_mul with a limit: gate min limit (through bf16), up med3;
    x * rcp(1 + exp(-x)) (ocml exp)."""
    g = bf16_round(fx.min(g, fx.Float32(LIMIT)))
    u = fx.max(fx.min(u, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
    act = g * hw_rcp(fx.Float32(1.0) + fx.Float32(fmath.exp(-g)))
    return act * u


@traced
def stage_shared(c, task, gops, uops, t0, n):
    """Gate rows 16 task .. (weights ``gops``), up rows 576 + 16 task ..
    (``uops``, or loaded here after the gate's GEMV when None) against tokens
    t0 .. t0 + n (their x in LDS; ``n`` traced: a tile a CTA picks at run time)
    -> bf16 -> silu x up -> SMID; an odd task then quantizes its group of 32
    (``smid_quant``)."""
    tid, red, d = c["tid"], c["red"], c["d"]
    a = c["args"]
    rows = None if isinstance(n, int) else n
    gemv_fp8_mfmas(c, HIDDEN, c["xl"], c["xsl"], red, gops, rows=rows)
    gpu.barrier()
    gate = fx.Float32(0.0)
    if tid < ROWS * n:
        gate = acc_sum(red, tid % ROWS, tid // ROWS)
    gpu.barrier()
    if const_expr(uops is None):
        uops = gemv_fp8_loads(c, a["sgu"], a["sgu_s"], HIDDEN, plus(c["sh_up0"], task))
    gemv_fp8_mfmas(c, HIDDEN, c["xl"], c["xsl"], red, uops, rows=rows)
    gpu.barrier()
    if tid < ROWS * n:
        r = tid % ROWS
        t = plus(t0, tid // ROWS)
        up = acc_sum(red, r, tid // ROWS)
        v = fx.Float32(_silu_mul(bf16_round(gate), bf16_round(up)).to(fx.BFloat16))
        c["put"](c["smid"], t * d.sh_inter + task * ROWS + r, v)
        fx.ptr_store(v, c["pair"] + tid)
    gpu.barrier()
    if task % 2 == 1:
        smid_quant(c, task, t0, n)
    gpu.barrier()


@traced
def smid_quant(c, task, t0, n):
    """Odd shared task ``task``: its 16 columns (LDS ``pair``) and the even
    task's (SMID) are a quantize_fp8 group of 32 (amax floored AMAX_FLOOR, the
    ceil code) -> SMQ / SMC, a thread a token. Once here, not in each of the 256 CTAs
    whose down stage reads them: that took a down CTA 14 us at 48 rows."""
    tid, sh = c["tid"], c["d"].sh_inter
    assert c["d"].shared_tasks % 2 == 0  # tasks pair into groups
    if tid < n:
        t = plus(t0, tid)
        # the even task's 16 columns: 16-byte polls of two pairs (the base even)
        got = c["poll"](
            [(c["smid"], t * sh + (task - 1) * ROWS + r, 2) for r in range(0, ROWS, 2)]
        )
        vals = [got[r // 2][r % 2].bitcast(fx.Float32) for r in range(ROWS)] + [
            fx.ptr_load(c["pair"] + (tid * ROWS + r)) for r in range(ROWS)
        ]
        amax = fx.Float32(AMAX_FLOOR)
        for y in vals:
            amax = fx.max(amax, abs(y))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        inv = rcp_pow2(code)
        g = task // 2
        words = [
            fp8_pack4(*[clamp_fp8(vals[4 * w + i] * inv) for i in range(4)])
            for w in range(8)
        ]
        # two words a 16-byte put
        for w in range_constexpr(0, 8, 2):
            c["put_words"](c["smq"], t * (sh // 4) + g * 8 + w, words[w : w + 2])
        c["put_words"](c["smc"], t * (sh // 32) + g, [code])


# ---------------------------------------------------------------- routed ug
@traced
def gemv_fp4_pair(c, w, ws, e, n_rows, k, rg0, ks, xl, xsl, scale_cols, col):
    """Tiles rg0 and rg0 + 1 (16 rows each, one 32-row scale block) of expert
    e's (16, 16)-preshuffled MXFP4 weight [n_rows, k] times 16 MXFP8 rows in
    LDS, over this wave's K slice ``ks`` (``ug_rounds``) -> both tiles' partial
    C, a Vector of 8 (rows 4 (lane / 16) .., column lane % 16, whose LDS row is
    ``col``). Each B operand read from LDS feeds both tiles' MFMAs: the waves of
    a task share x, so a wave a tile over all of K read it once a tile.
    """
    lane, wave = c["lane"], c["wave"]
    sl = c["wsl"] + wave * UG_WSL_WORDS
    slices = c["rs"].ug_k_slices
    rounds = ug_rounds(k, ks, slices)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    # rounds whose loads are in flight past the one computing: a quarter's
    # 2-step rounds two (the draft's 20 rows: -0.5 us)
    ahead = slices // 2
    ops = [
        fp4_pair_loads(c, w, ws, e, n_rows, k, rg0, scale_cols, *rounds[q])
        for q in range(min(ahead, len(rounds)))
    ]
    for r in range_constexpr(len(rounds)):
        sc0, st0, n, sc_steps = rounds[r]
        sv, avs = ops[r]
        fx.ptr_store(sv, sl + lane * (sc_steps // 2))
        sas = [
            [
                lds_u8(
                    sl,
                    scale_index(
                        t * 16 + lane % 16, (st0 + i) * 4 + lane // 16, scale_cols
                    )
                    - 128 * sc0,
                )
                for i in range(n)
            ]
            for t in range(2)
        ]
        accs = fp4_pair_mfma(c, k, xl, xsl, st0, n, avs, sas, accs, col)
        if const_expr(r + ahead < len(rounds)):
            ops.append(
                fp4_pair_loads(
                    c, w, ws, e, n_rows, k, rg0, scale_cols, *rounds[r + ahead]
                )
            )
    return fx.Vector.from_elements(
        [accs[t][q] for t in range(2) for q in range(4)], fx.Float32
    )


def ug_rounds(k, ks, slices):
    """K slice ``ks`` of ``slices``' rounds (first step of the scales loaded,
    first step, steps, steps of scales loaded); ``ks`` traced: the steps are.
    Halves: the 8-step scale blocks 2 r + ks, the odd block left split 4 + 4,
    so both run k / 256 steps. Quarters: steps 8 i + 2 ks, + 1 of each block
    i, a round loading only their scales (the shuffled layout keeps 2 steps'
    together: a dword a lane, where the whole block was 4)."""
    blocks = k // 128 // 8
    if slices == 4:
        return [(fx.Int32(8 * i) + ks * 2,) * 2 + (2, 2) for i in range(blocks)]
    assert slices == 2
    rounds = [
        (fx.Int32(16 * r) + ks * 8, fx.Int32(16 * r) + ks * 8, 8, 8)
        for r in range(blocks // 2)
    ]
    if blocks % 2:
        last = (blocks - 1) * 8
        rounds.append((fx.Int32(last), fx.Int32(last) + ks * 4, 4, 8))
    return rounds


def fp4_pair_loads(c, w, ws, e, n_rows, k, rg0, scale_cols, sc0, st0, n, sc_steps):
    """``gemv_fp4_pair``'s loads of a round: the scales of steps sc0 .. sc0 +
    sc_steps (of the tiles' 32-row block: 128 bytes a step, sc_steps / 2 dwords
    a lane) and both tiles' weight operands of steps st0 .. st0 + n. Loads
    only."""
    lane = c["lane"]
    block = (e * n_rows + rg0 * 16) // 32 * (scale_cols * 8)
    words = sc_steps // 2
    sv = bo.buffer_load(
        rsrc(ws),
        block + 32 * sc0 + lane * words,
        vec_width=words,
        dtype=T.i32,
        cache_modifier=CM_NT,
    )
    sv = fx.Vector(sv) if words > 1 else fx.Int32(sv)
    return sv, [fp4_tile_weights(c, w, e, n_rows, k, rg0 + t, st0, n) for t in range(2)]


def scale_block_loads(c, ws, first_row, scale_cols):
    """The loads of the 32-row e8m0 block holding ``first_row``: in aiter's
    shuffle_scale order a block is contiguous (both 16-row halves, all of K), so
    a few 16-byte loads a lane fetch it, where a byte load a K step doubled the
    stage's load instructions and bound it by their issue (2026-09-29, ug). A
    lane past the block re-reads its last 16 bytes."""
    lane = c["lane"]
    words = scale_cols * 8
    block = first_row // 32 * words
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(ws),
                block + fx.min(256 * i + lane * 4, words - 4),
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for i in range((words + 255) // 256)
    ]


def down_scale_words(d) -> int:
    """A down tile's 32-row scale block in LDS: ``scale_block_store`` writes
    whole 256-word rounds."""
    return down_lds_words(d.down_scale_cols * 8)


# a wave's LDS scale words in the ug stage: a batch's 256
UG_WSL_WORDS = 256
# the widest step whose every x row the ug stage holds in LDS (``ug_x_resident``)
UG_X_RESIDENT_MAX = 24
# a resident row's codes in LDS, a byte each (``load_x8_rows(code_bytes=)``):
# a word each took 24 rows past the LDS
UG_CODE_ROW = HIDDEN // 32 + 4


def down_wsl_words(d) -> int:
    """A wave's LDS scale words in the down stage: a round's blocks -- a (task,
    slot) each, ``DOWN_R_MAX`` at most (``stage_down``)."""
    return DOWN_R_MAX * down_scale_words(d)


def scale_block_store(c, svs, sl):
    """``scale_block_loads``' words -> this wave's LDS block ``sl``
    (``down_scale_words``; a lane past the block writes past it, inside it)."""
    lane = c["lane"]
    for i in range_constexpr(len(svs)):
        fx.ptr_store(svs[i], sl + (256 * i + lane * 4))


def lds_scales(sl, local_row, lane, st0, blk, scale_cols):
    """K steps st0 .. st0 + blk's scale bytes of this lane's row from a 32-row
    scale block in LDS (``local_row`` its row in the block). An even step and
    the next are bytes 2 apart of one word (``scale_index``'s d): a read a
    pair, not a step (down: 3 reads a slot, not 5)."""
    out, word = [], None
    for i in range(blk):
        idx = scale_index(local_row, (st0 + i) * 4 + lane // 16, scale_cols)
        if (st0 + i) % 2 == 0 or word is None:
            word = fx.ptr_load(sl + idx // 4)
        out.append((word >> (idx % 4 * 8)) & 0xFF)
    return out


def lds_u8(ptr, i):
    return (fx.ptr_load(ptr + i // 4) >> (i % 4 * 8)) & 0xFF


def fp4_tile_weights(c, w, e, n_rows, k, rg, st0, blk, k_real=None):
    """The weight operands of expert e's 16-row tile rg, K steps st0 .. st0 +
    blk: loads only, nothing waits on them here. ``k_real``: a lane whose 32
    columns lie past it (zero weights) reads its row group's first lane's
    address instead (no extra traffic) and holds zero."""
    lane = c["lane"]
    base = e * n_rows * (k // 2) // 4  # dwords: expert e
    out = []
    for i in range_constexpr(blk):
        past = k_real is not None and ((st0 + i) * 128 + 96) >= k_real
        if const_expr(past):
            real = (st0 + i) * 128 + lane // 16 * 32 < k_real
            at = real.select(lane * 4, lane % 16 * 4)
        else:
            at = lane * 4
        v = fp4_tile_load(rsrc(w), base, rg, k, st0 + i, at)
        if const_expr(past):
            v = fx.Vector.from_elements(
                [real.select(v[q], fx.Int32(0)) for q in range(4)], fx.Int32
            )
        out.append(v)
    return out


def fp4_pair_mfma(c, k, xl, xsl, st0, n, avs, sas, accs, col):
    """``accs`` (two tiles') chained through ``n`` K steps from st0 against MXFP8
    rows in LDS (column lane % 16: row ``col``): a step's B operand loaded once
    for both tiles' MFMAs."""
    lane = c["lane"]
    x_row = lds_row(k // 4)
    s_row = lds_row(k // 32)
    j = lane // 16
    accs = list(accs)
    for i in range_constexpr(n):
        bv = fp8_operand(
            [
                fx.ptr_load(
                    xl + (col * x_row + (st0 + i) * 32 + 16 * h + j * 4),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                for h in range(2)
            ]
        )
        if const_expr(c.get("x_code_bytes", False)):
            sb = lds_u8(xsl, col * UG_CODE_ROW + (st0 + i) * 4 + j)
        else:
            sb = fx.ptr_load(xsl + (col * s_row + (st0 + i) * 4 + j))
        for t in range_constexpr(2):
            accs[t] = mfma_scaled(avs[t][i], bv, accs[t], sas[t][i], sb, FP4, FP8)
    return accs


def fp8_operand(halves):
    """A lane's MXFP8 MFMA operand of a K step: its two 16-byte halves (K
    16 j .. and 64 + 16 j .. of the step, j = lane / 16)."""
    return fx.Vector.from_elements(
        [fx.Vector(h).bitcast(fx.Int32)[d] for h in halves for d in range(4)],
        fx.Int32,
    )


def down_lds_words(n):
    """A down wave's LDS block of n words: ``scale_block_store`` writes whole
    256-word rounds."""
    return -(-n // 256) * 256


@traced
def take_ug_task(c):
    """This CTA's next ug task past the first round, from the launch's queue
    (``ugq``, a word a tag, counting from 0): the CTAs that ran a shared,
    router or route task come to it late and so take fewer, where a fixed
    placement gave every CTA the same count and down waited on theirs. A
    task's result is the same whichever CTA runs it."""
    return take_launch_queue(c, "ugq")


def take_launch_queue(c, name):
    """This launch's word of queue ``name`` (``QUEUES``, from the launch's
    ``queue_words``: ``queue_offset``), taken for the whole CTA."""
    at = c["args"]["queue_words"] + fx.Int64(QUEUES.index(name) * QUEUE_REGION)
    return queue.take_shared(queue.word(at), c["tab"] + c["rs"].queue0, c["tid"])


@traced
def stage_ug(c, task):
    """One token tile: slot task // ug_parts of U (``RouteShape.ug_parts``), the
    ``ug_groups`` FP4 groups of intermediate columns from 32 ug_groups (task %
    ug_parts) (``ug_wave_pair``); then silu x up, rounded to bf16, and the
    MXFP8 requant of each token's 32 values of a group -> MID (the stage-1
    epilogue and the stage-2 input quant). Past one: ``stage_ug_tile``."""
    s, tid, lane, tab = c["S"], c["tid"], c["lane"], c["tab"]
    rs = c["rs"]
    ug_parts = rs.ug_parts
    u = task // ug_parts
    if u < c["rs"].n_union(tab, s):
        e = c["rs"].union_expert(tab, s, u)
        # the slot's picks, a column each (a column past them repeats the last);
        # every token's row is in LDS: a column reads its pick's token's
        first = c["rs"].slot_first(tab, u)
        count = c["rs"].slot_first(tab, u + 1) - first
        pick = c["rs"].pick_at(tab, first + fx.min(lane % 16, count - 1))
        ug_pass(c, e, task % ug_parts * rs.ug_groups, first, count, pick // rs.topk)
        if const_expr(not rs.down_y):
            publish(c["put"], c["ugf"], task, 1, tid == 0)
    gpu.barrier()


def ug_wave_pair(c, part0):
    """This wave's share of an ug task from group ``part0``: group p of its
    ``ug_groups`` (wave / (2 slices)), its 16-column half bb = wave % 2 -- w13's
    gate tile 2 b and up tile 2 b + 1 (aiter GUGU interleaves them a 16-row
    tile; b = 2 (part0 + p) + bb), one 32-row scale block -- over K slice ks =
    wave / 2 % slices (``gemv_fp4_pair``): (p, 2 b, ks, whether the group is in
    the real width)."""
    wave, slices = c["wave"], c["rs"].ug_k_slices
    p = wave // (2 * slices)
    b = (part0 + p) * 2 + wave % 2
    return p, 2 * b, wave // 2 % slices, part0 + p < c["d"].inter_real // UG_PART


def ug_red_slot(rs, p, bb, ks, t):
    """RED's 64-lane slot of an ug task's partial C: group p, half bb, K slice
    ks, gate (t 0) or up (t 1)."""
    return ((p * 2 + bb) * rs.ug_k_slices + ks) * 2 + t


def ug_red_word(slot, g, col, bb):
    """RED's word of lane (g, col)'s C quad in an ug slot (the MFMA lane 16 g +
    col): the column swizzled within its lane group by the group and the half
    ``bb``, so the epilogue's read -- a (row, pick) a lane, its rows' groups and
    halves in one wave -- meets every bank once."""
    return (slot * 64 + g * 16 + (col ^ (2 * g + 8 * bb))) * 4


def pairwise_sum(vs):
    """vs summed as a balanced tree, neighbours first: (v0 + v1) + (v2 + v3)."""
    while len(vs) > 1:
        vs = [vs[i] + vs[i + 1] for i in range(0, len(vs), 2)]
    return vs[0]


def ug_tile_task(c, task):
    """Past one token tile, ug task ``task``: its slot's expert, its first
    group, and its pick tile's first pick and the picks from it."""
    tab, rs, s = c["tab"], c["rs"], c["S"]
    ug_parts = rs.ug_parts
    tile = task // ug_parts
    u = rs.tile_slot(tab, tile)
    first = mplan.tile_pick0(rs.slot_first(tab, u), tile, rs.tile_first(tab, u))
    count = rs.slot_first(tab, u + 1) - first
    return rs.union_expert(tab, s, u), task % ug_parts * rs.ug_groups, first, count


@traced
def stage_ug_tile(c, task, total):
    """Past one token tile: ug task ``task`` of ``total`` (a pick tile of its
    slot), then the next task from the queue (``take_ug_task``). No next task's
    weights prefetched under this one: held across its epilogue and gather they
    spilled the merged K2 (S = 24 / 48), and without them it ran faster."""
    placed = ug_placed(c["rs"])
    # a CTA that ran a shared unit (before ug0, turned) holds no first-round
    # task: it starts at the queue
    if (task < placed) | (task >= BLOCKS):
        ug_tile_run(c, ug_task_id(task, placed))
    return (total > placed).select(take_ug_task(c) + BLOCKS, BLOCKS + total)


@traced
def ug_tile_run(c, task):
    """Past one token tile, ug task ``task``'s pass (and its flag)."""
    tid, lane, rs = c["tid"], c["lane"], c["rs"]
    e, part0, first, count = ug_tile_task(c, task)
    if const_expr(ug_x_resident(c["S"], rs.own_routes)):
        # every token's row in LDS (``run_ug``): a column reads its pick's
        pick = rs.pick_at(c["tab"], first + fx.min(lane % 16, count - 1))
        ug_pass(c, e, part0, first, count, pick // rs.topk)
    else:
        gather_x8r(c, first, count)
        ug_pass(c, e, part0, first, count, fx.min(lane % 16, count - 1))
    if const_expr(not down_by_unit(rs) and not rs.down_y):
        # the epilogue's MID stores done before the task's flag (down by unit:
        # the CTA's flag after its last task, ``run_moe``)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid == 0:
            c["put"](c["ugf"], task, 1)


def ug_placed(rs):
    """Past one token tile, the first round's ug tasks: one a CTA from ug0 up to
    the shared units' CTAs."""
    return BLOCKS - rs.shared_units % BLOCKS


def ug_task_id(task, placed):
    """``stage_ug_tile``'s task number -> its ug task: a first-round one below
    ``placed``, a queue's past BLOCKS; a CTA before ug0's first (between) 0."""
    return (task < placed).select(
        task, (task >= BLOCKS).select(task - BLOCKS + placed, 0)
    )


@traced
def ug_pass(c, e, part0, first, count, col):
    """An ug task's GEMV and epilogue for the slot's picks first .. first +
    min(count, 16) (column lane % 16's x in LDS row ``col``)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a, d, rs = c["args"], c["d"], c["rs"]
    p, rg0, ks, real = ug_wave_pair(c, part0)
    # a group past the real width (the padding to inter) has zero weights:
    # its GEMV is zero unread, and the epilogue still writes its zero MID
    acc = fx.Vector.filled(8, 0.0, fx.Float32)
    if real:
        acc = gemv_fp4_pair(
            c, a["w13"], a["w13_s"], local_expert(c, e), 2 * d.inter, HIDDEN, rg0, ks, c["xl"],
            c["xsl"], HIDDEN // 32, col,
        )  # fmt: skip
    for t in range_constexpr(2):
        slot = ug_red_slot(rs, p, wave % 2, ks, t)
        fx.ptr_store(
            fx.Vector.from_elements([acc[4 * t + q] for q in range(4)], fx.Float32),
            red + ug_red_word(slot, lane // 16, lane % 16, wave % 2),
        )
    gpu.barrier()
    # thread (p, j, i): group p, the slot's pick j, intermediate column
    # 32 part + i, a pass of THREADS at a time; a group's 32 threads stay
    # lane-aligned
    cols = rs.tile_picks
    for pas in range_constexpr(0, rs.ug_groups * cols * UG_PART, THREADS):
        ug_epilogue(c, part0, pas + tid, first, count)


@traced
def load_x8_rows(
    c, words_mb, codes_mb, xl, xsl, token_of, live, rows=TILE, code_bytes=False
):
    """MXFP8 rows ``token_of(j)``, j < ``live`` (<= ``rows``; traced or int), of
    scratch ``words_mb`` / ``codes_mb`` into LDS rows j of ``xl`` / ``xsl``:
    the first ``live`` lanes wait on their token's X8RDY, then plain 16-byte
    loads (half the bytes of the tagged pairs a poll would read). ``code_bytes``:
    the codes a byte each, ``UG_CODE_ROW`` bytes a row (``ug_x_resident``),
    else a word each at the ``lds_row`` stride."""
    tid = c["tid"]
    if tid < rows:
        c["poll"]([(c["x8rdy"], token_of(fx.min(tid, live - 1)), 1)])
    gpu.barrier()
    for mb, width, dst in ((words_mb, X8_WORDS, xl), (codes_mb, HIDDEN // 32, xsl)):
        quads = width // 4
        packed = code_bytes and mb == codes_mb
        for i in range_constexpr(-(-rows * quads // THREADS)):
            q = tid + THREADS * i
            row = q // quads
            if (q < rows * quads) & (row < live):
                v = fx.Vector(
                    bo.buffer_load(
                        rsrc(c[mb].value),
                        token_of(row) * width + q % quads * 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                if const_expr(packed):
                    word = v[0] | v[1] << 8 | v[2] << 16 | v[3] << 24
                    fx.ptr_store(word, dst + (row * (UG_CODE_ROW // 4) + q % quads))
                else:
                    fx.ptr_store(v, dst + (row * lds_row(width) + q % quads * 4))
    gpu.barrier()


def gather_x8r(c, first, count):
    """The X8 rows and codes of the slot's picks first .. first + min(count, 16)
    into LDS rows 0 .. (only those rows: an ug pass reads a pick's row, the
    columns past them repeat the last one's)."""
    tab, topk = c["tab"], c["rs"].topk

    def token(j):
        return c["rs"].pick_at(tab, first + fx.min(j, count - 1)) // topk

    load_x8_rows(c, "x8", "x8s", c["x8rl"], c["x8rsl"], token, fx.min(count, TILE))


@traced
def ug_epilogue(c, part0, idx, first, count):
    """Thread ``idx``'s value of an ug task: silu x up of its (group, pick j,
    column), the group's MXFP8 scale, its byte of MID (the pick's row, ``first``
    + j in the pick order)."""
    lane, red, d, rs = c["lane"], c["red"], c["d"], c["rs"]
    cols = rs.tile_picks
    if idx < rs.ug_groups * cols * UG_PART:
        p = idx // (cols * UG_PART)
        part = part0 + p
        t = idx % (cols * UG_PART) // UG_PART
        i = idx % UG_PART
        tile = i // 16
        r = i % 16
        g, up = [
            pairwise_sum(
                [
                    fx.ptr_load(
                        red
                        + ug_red_word(ug_red_slot(rs, p, tile, ks, gu), r // 4, t, tile)
                        + r % 4
                    )
                    for ks in range(rs.ug_k_slices)
                ]
            )
            for gu in range(2)
        ]
        gc = fx.min(g, fx.Float32(LIMIT))
        uc = fx.max(fx.min(up, fx.Float32(LIMIT)), fx.Float32(-LIMIT))
        sig = hw_rcp(fx.Float32(1.0) + hw_exp2(gc * fx.Float32(-LOG2E)))
        # the stage-1 kernel stores the intermediate as bf16; the stage-2 input
        # quant (fused_mx_quant_moe_sort) makes it MXFP8: amax floored
        # AMAX_FLOOR, the ceil code, x / scale clamped, RNE
        v = bf16_round(gc * sig * uc)
        amax = fx.max(
            butterfly(abs(v), (1, 2, 4, 8, 16), fx.max), fx.Float32(AMAX_FLOOR)
        )
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        q8 = clamp_fp8(v * rcp_pow2(code))
        nb = [
            lane_gather(q8.bitcast(fx.Int32), fx.min(lane + q, 63)).bitcast(fx.Float32)
            for q in range(1, 4)
        ]
        # 4 threads' bytes -> one word
        word = fp8_pack4(q8, *nb)
        row = first + t
        at = row * d.mid_words + part * 8 + i // 4
        if const_expr(rs.down_y):
            # tagged pairs: a down unit polls its slot's rows themselves
            # (``run_down_y``), no flag and no store drain before one
            if (i % 4 == 0) & (t < count):
                c["put_words"](c["mid"], at, [word])
            if (i == 0) & (t < count):
                c["put"](c["mids"], mid_code_index(d, row, part), code)
        else:
            if (i % 4 == 0) & (t < count):
                st_dev(word, c["mid"].value, at)
            if (i == 0) & (t < count):
                st_dev(code, c["mids"].value, mid_code_index(d, row, part))


# ---------------------------------------------------------------- down
@traced
def stage_down(c, tasks):
    """Rows 16 task .. of the hidden output for each of ``tasks`` (a CTA's one or
    two, run together: a second pass would double the CTA's time): routed
    experts of U (a wave each, round robin), the shared expert, their sum ->
    pushed to every rank (token parallel: to the token's owner)."""
    s, wave, tab, d = c["S"], c["wave"], c["tab"], c["d"]
    n = len(tasks)
    nu = c["rs"].n_union(tab, s)
    sl = c["wsl"] + wave * down_wsl_words(d)
    sw = down_scale_words(d)
    # slots of U a round: the loads of a round stay the same
    slots = DOWN_R_MAX // n
    per_round = WAVES * slots
    # the shared expert's down projection first: its input is ready long
    # before the routed results, which a CTA mostly waits on here; after the
    # rounds it sat on the path past the last ug task
    down_shared(c, tasks)
    if const_expr(down_parts(d) > 1):
        stage_down_parts(c, tasks, nu, sl, sw, slots, per_round)
        return
    for rnd in range(0, (nu + per_round - 1) // per_round, 1):
        # the round's slots of U (wave, wave + WAVES, ...), every load of them
        # and of every task in flight before the first wait: the weights and
        # their scale blocks (no ug result: before the round's ug flags), then
        # the slots' picks' intermediate rows, each lane its pick's straight
        # from MID (``mid_operands``). A slot past U loads the last slot again
        # (L2 hits) and contributes nothing
        us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
        ucs = [fx.min(u, nu - 1) for u in us]
        firsts = [c["rs"].slot_first(tab, u) for u in ucs]
        svs, avs = down_weights(c, tasks, rnd, slots, nu)
        # this round's slots' ug tasks done, not every slot of the wave: the
        # round starts while later slots' ug run
        c["poll"]([(c["ugf"], f, 1) for f in down_ug_flags(c, ucs)])
        counts = [c["rs"].slot_first(tab, u + 1) - f for u, f in zip(ucs, firsts)]
        mids = [mid_operands(c, f, k) for f, k in zip(firsts, counts)]
        for q in range_constexpr(len(svs)):
            scale_block_store(c, svs[q], sl + q * sw)
        for i in range_constexpr(slots):
            for j in range_constexpr(n):
                down_mfma(
                    c, j, tasks[j], sl + (j * slots + i) * sw, mids[i], firsts[i],
                    counts[i], avs[j * slots + i], us[i] < nu,
                )  # fmt: skip
            # a slot of more than 16 picks: its next pick tile's rows
            if const_expr(s > TILE):
                for j0 in range_constexpr(TILE, s, TILE):
                    if j0 < counts[i]:
                        more = mid_operands(c, firsts[i] + j0, counts[i] - j0)
                        for j in range_constexpr(n):
                            down_mfma(
                                c, j, tasks[j], sl + (j * slots + i) * sw, more,
                                firsts[i] + j0, counts[i] - j0, avs[j * slots + i],
                                us[i] < nu,
                            )  # fmt: skip
    down_finish(c, tasks)


@traced
def down_finish(c, tasks):
    """Every contribution of ``tasks`` stored: combine them, token tile by
    token tile."""
    gpu.barrier()
    repeated = "second_live" in c  # ``RouteShape.down_one_call``
    for t0, nn in token_tiles(c["S"]):
        for j in range_constexpr(len(tasks)):
            if const_expr(j == 1 and repeated):
                if c["second_live"]:
                    down_combine(c, tasks[j], j, t0, nn)
            else:
                down_combine(c, tasks[j], j, t0, nn)


def down_by_unit(rs) -> bool:
    """Whether the down stage runs by unit (``down_by_units``): a CTA's down
    tasks in one call (``RouteShape.down_one_call``) with K in parts. It waits
    for every ug task at once, so the ug stage flags a CTA (UGC), not a task
    (EP 32 / 48 rows -1.3 / -1.5 us)."""
    return rs.down_one_call and down_parts(rs.dims) > 1


@traced
def down_by_units(c, tasks, after_units):
    """Down past MAX_ROWS with K in parts: once every CTA's ug tasks are done
    (UGC), the units of every CTA (``down_units``), ``after_units()``, the
    shared expert's rows of the CTA's ``tasks`` (in the time it would wait for
    the other CTAs' units: its input ready long before), then those tasks
    combined (``down_units_combine``)."""
    nu = c["rs"].n_union(c["tab"], c["S"])
    for f in range(c["tid"], BLOCKS, THREADS):
        c["poll"]([(c["ugc"], f, 1)])
    gpu.barrier()
    down_units(c, nu)
    after_units()
    down_shared(c, tasks)
    down_units_combine(c, tasks)


@traced
def down_units(c, nu):
    """Down by unit: a unit is (pick tile of U, hidden block of
    DOWN_UNIT_BLOCKS), its cost at most one tile's, CTAs a unit each then from
    a queue (``take_down_unit``: a fixed placement left CTAs 64 us apart); a
    wave its DOWN_WAVE_TASKS tasks' chains over K, each pick's contribution to
    DCON. Then the CTA's flag (DNF)."""
    d = c["d"]
    parts = down_parts(d, True)
    nst = d.inter // 128 // parts
    sl = c["wsl"] + c["wave"] * down_wsl_words(d)
    total = c["rs"].tile_first(c["tab"], nu) * DOWN_UNIT_BLOCKS
    unit = first_task(c["bid"], DOWN0)
    while unit < total:
        unit = down_unit_next(c, unit, sl, parts, nst, total)
    publish(c["put"], c["dnf"], c["bid"], 1, c["tid"] == 0)


@traced
def down_unit_next(c, unit, sl, parts, nst, total):
    """Down unit ``unit``, then this CTA's next from the queue (past the first
    round)."""
    down_unit(c, unit, sl, parts, nst)
    return (total > BLOCKS).select(take_down_unit(c) + BLOCKS, total)


@traced
def take_down_unit(c):
    """This CTA's next down unit past the first round, from the launch's queue
    (``dnq``, a word a tag, counting from 0), as ``take_ug_task``."""
    return take_launch_queue(c, "dnq")


@traced
def down_unit(c, unit, sl, parts, nst):
    """A wave's share of down unit ``unit``: its DOWN_WAVE_TASKS adjacent
    16-row tasks (one 32-row scale block) on the unit's pick tile."""
    v = unit // DOWN_UNIT_BLOCKS
    task0 = (unit % DOWN_UNIT_BLOCKS * WAVES + c["wave"]) * DOWN_WAVE_TASKS
    u = c["rs"].tile_slot(c["tab"], v)
    e = local_expert(c, c["rs"].union_expert(c["tab"], c["S"], u))
    first = mplan.tile_pick0(
        c["rs"].slot_first(c["tab"], u), v, c["rs"].tile_first(c["tab"], u)
    )
    count = c["rs"].slot_first(c["tab"], u + 1) - first
    sv = scale_block_loads(
        c, c["args"]["w2_s"], e * HIDDEN + task0 * 16, c["d"].down_scale_cols
    )
    scale_block_store(c, sv, sl)
    down_unit_tile(c, e, task0, sl, first, fx.min(fx.Int32(TILE), count), parts, nst)


@traced
def down_unit_tile(c, e, task0, sl, first, count, parts, nst):
    """Tasks task0 .. task0 + DOWN_WAVE_TASKS of expert e on picks first ..
    first + count: their chains over K in parts (the picks' MID operands
    shared), part p + 1's loads before part p's MFMAs, each pick's
    contribution to DCON."""
    d, a = c["d"], c["args"]
    n = DOWN_WAVE_TASKS
    tasks = [task0 + j for j in range(n)]

    def weights(st0):
        return [
            fp4_tile_weights(c, a["w2"], e, HIDDEN, d.inter, t, st0, nst, d.inter_real)
            for t in tasks
        ]

    accs = [down_zero() for _ in range(n)]
    avs = weights(0)
    mids = mid_operands(c, first, count, 0, nst)
    for part in range_constexpr(parts):
        st0 = part * nst
        if const_expr(part + 1 < parts):
            avs_n = weights(st0 + nst)
            mids_n = mid_operands(c, first, count, st0 + nst, nst)
        for j in range_constexpr(n):
            accs[j] = down_chains(c, tasks[j], sl, mids, avs[j], accs[j], st0)
        if const_expr(part + 1 < parts):
            avs, mids = avs_n, mids_n
    for j in range_constexpr(n):
        down_store_dcon(c, accs[j], tasks[j], first, count)


@traced
def down_store_dcon(c, accs, task, first, count):
    """``down_store`` into DCON: each of picks first .. first + count's
    contribution to rows 16 task .. (bf16(v w), packed pairs)."""
    lane, tab, s, rs = c["lane"], c["tab"], c["S"], c["rs"]
    jj = lane % 16
    col = fx.min(jj, count - 1)
    acc = accs[0]
    for h in range_constexpr(1, DOWN_CHAINS):
        acc = acc + accs[h]
    pick = rs.pick_at(tab, first + col)
    w = rs.route_w(tab, s, pick // rs.topk, pick % rs.topk)
    for q in range_constexpr(0, 4, 2):
        r = 4 * (lane // 16) + q
        if jj < count:
            word = bf16_pair(acc[q] * w, acc[q + 1] * w).bitcast(fx.Int32)
            st_dev(word, c["dcon"].value, pick * (HIDDEN // 2) + (task * ROWS + r) // 2)


@traced
def down_units_combine(c, tasks):
    """Every CTA's units done (DNF), then ``down_finish`` from DCON in one
    pass: a thread a (task, token, row pair), pushed straight (no LDS pair
    round, no barrier a token tile)."""
    for f in range(c["tid"], BLOCKS, THREADS):
        c["poll"]([(c["dnf"], f, 1)])
    gpu.barrier()
    n_pairs = len(tasks) * c["S"] * (ROWS // 2)
    for pas in range_constexpr(0, n_pairs, THREADS):
        idx = plus(pas, c["tid"])
        if idx < n_pairs:
            dcon_pair(c, tasks, idx)
    gpu.barrier()


@traced
def dcon_pair(c, tasks, idx):
    """``down_units_combine``'s pair ``idx``: rows 2 rp, 2 rp + 1 of a task's
    16 for one token, summed as ``down_combine`` (f32 in top-k order, local
    picks only, then the shared expert's, one bf16 rounding), every routed
    contribution's load issued together."""
    s, tab, d = c["S"], c["tab"], c["d"]
    topk = c["rs"].topk
    per_task = s * (ROWS // 2)
    j = idx // per_task
    rem = idx % per_task
    t = rem // (ROWS // 2)
    rp = rem % (ROWS // 2)
    task = tasks[0]
    live = j == 0
    if const_expr(len(tasks) > 1):
        task = (j == 0).select(tasks[0], tasks[1])
        live = (j == 0) | c["second_live"]
    col = task * ROWS + 2 * rp
    words = [
        fx.Int32(
            bo.buffer_load(
                rsrc(c["dcon"].value),
                (t * topk + k) * (HIDDEN // 2) + col // 2,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        for k in range(topk)
    ]
    r0, r1 = fx.Float32(0.0), fx.Float32(0.0)
    for k in range_constexpr(topk):
        local = is_local(c, c["rs"].route_id(tab, s, t, k))
        r0 = local.select(r0 + bf_lo(words[k]), r0)
        r1 = local.select(r1 + bf_hi(words[k]), r1)
    base = (j * s + t) * ROWS + 2 * rp
    sh0 = fx.ptr_load(c["sdl"] + base).bitcast(fx.Float32)
    sh1 = fx.ptr_load(c["sdl"] + (base + 1)).bitcast(fx.Float32)
    v0 = bf16_round(r0 + sh0)
    v1 = bf16_round(r1 + sh1)
    if const_expr(c["shard_f"]):
        # a sharded residual: the column's owner only (a reduce-scatter)
        for p in range_constexpr(d.ranks):
            if live & (col // (HIDDEN // d.ranks) == p):
                c["put_bf"](
                    preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn"),
                    2 * region_pair(c["rank"], t, s, col), [v0, v1],
                )  # fmt: skip
    elif const_expr(c["tok_par"]):
        # to the token's owner only, at its own token index there
        own_n = own_count(s, d.ranks)
        owner = t // own_n
        for p in range_constexpr(d.ranks):
            if live & (owner == p):
                dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                pair = region_pair(c["rank"], t - p * own_n, own_n, col)
                c["put_bf"](dst, 2 * pair, [v0, v1])
    else:
        for p in range_constexpr(d.ranks):
            if live:
                dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                c["put_bf"](dst, 2 * region_pair(c["rank"], t, s, col), [v0, v1])


# ------------------------------------------------------- down by queue units
# (``RouteShape.down_y``): the shared expert's down tasks a queue unit
# (``shared_down_unit``), their intermediate tile loaded once: where the
# rank's share is narrow (``shared_down_wide``) 256 rows a unit, the queue's
# first, its weights a rolling SHARED_DOWN_AHEAD tasks in
# flight; else two tasks a unit past the slots'
SHARED_DOWN_TASKS_TILE = 16
SHARED_DOWN_TASKS = 2
SHARED_DOWN_AHEAD = 4
# a shared expert share a rank narrower than this: its 2-task units' fixed
# cost outweighs their bytes (the down there bound by hand-off latency, not by
# its bytes)
SHARED_DOWN_WIDE_BELOW = 512


def shared_down_wide(rs) -> bool:
    """Whether the shared expert's down runs as 256-row units, first in
    the queue: one token tile, a narrow share (``SHARED_DOWN_WIDE_BELOW``)."""
    return not rs.multi and rs.dims.sh_inter < SHARED_DOWN_WIDE_BELOW


def shared_down_tasks(rs) -> int:
    """The shared expert's down tasks a queue unit."""
    return SHARED_DOWN_TASKS_TILE if shared_down_wide(rs) else SHARED_DOWN_TASKS


def shared_down_units(rs) -> int:
    """The shared expert's down queue units."""
    return DOWN_TASKS // shared_down_tasks(rs)


# a routed unit's weight tiles a wave holds (a 16-row task's K steps each): at
# TP4 four tasks (512 rows); a wider K takes fewer tasks
DOWN_Y_WAVE_TILES = 20


def down_y_wave_tasks(d) -> int:
    """A routed unit's adjacent 16-row tasks a wave: whole 32-row scale blocks,
    ``DOWN_R_MAX`` at most, their weights within DOWN_Y_WAVE_TILES."""
    fit = DOWN_Y_WAVE_TILES // (d.inter // 128) // 2 * 2
    return max(2, min(2 * DOWN_R_MAX, fit))


def down_y_unit_blocks(d) -> int:
    """A slot's routed units (hidden blocks of ``down_y_wave_tasks`` tasks a
    wave)."""
    return DOWN_TASKS // (WAVES * down_y_wave_tasks(d))


@traced
def run_down_y(c, tasks, unit):
    """Down by queue units (``RouteShape.down_y``): every unit from the
    launch's queue (DNQ) -- each slot's (slot, hidden block) units
    (``down_y_unit``) and the shared expert's (``shared_down_unit``), in
    ``down_y_run``'s order -- by the CTAs as they reach down, so a CTA whose
    ug ended last finds the queue empty, from ``unit`` (this CTA's first take,
    ``take_down_rank_unit``); then the combine of this CTA's ``tasks``
    (``down_y_combine``). A unit's next is taken after it: one taken ahead let
    the first CTAs to arrive claim two units each and the last none (+6%)."""
    blocks = down_y_unit_blocks(c["d"])
    rs, tab = c["rs"], c["tab"]
    # a slot's units, or past one token tile a pick tile's
    n_union = rs.n_union(tab, c["S"])
    routed = (rs.tile_first(tab, n_union) if rs.multi else n_union) * blocks
    shared = shared_down_units(rs)
    total = routed + shared
    sl = c["wsl"] + c["wave"] * down_wsl_words(c["d"])
    unit = fx.Int32(unit)
    while unit < total:
        down_y_run(c, unit, sl, routed, shared)
        unit = take_down_unit(c)
    for pas in range_constexpr(0, len(tasks) * c["S"] * (ROWS // 2), THREADS):
        down_y_combine(c, tasks, plus(pas, c["tid"]))
    gpu.barrier()


@traced
def down_y_run(c, unit, sl, routed, shared):
    """Unit ``unit`` of ``run_down_y``: with wide shared units
    (``shared_down_wide``) the shared expert's ``shared`` first (from the CTAs
    that reach down first, while the slots' MID is still being written), then
    the slots' ``routed``; else the slots' first."""
    if const_expr(shared_down_wide(c["rs"])):
        if unit < shared:
            shared_down_unit(c, unit)
        if unit >= shared:
            down_y_unit(c, unit - shared, sl)
    else:
        if unit < routed:
            down_y_unit(c, unit, sl)
        if unit >= routed:
            shared_down_unit(c, unit - routed)


@traced
def down_y_unit(c, unit, sl):
    """Down unit ``unit`` (slot ``unit // down_y_unit_blocks`` of U, a hidden
    block): a wave its ``down_y_wave_tasks`` adjacent 16-row tasks -- the
    slot's MID rows polled into LDS (``load_mid_rows``: tagged, no flag), the
    poll's first pass issued before the scale blocks and every task's
    weights and checked after them -- the chains over K, then each pick's
    contribution bf16(v w) to DY as tagged pairs (``down_store_y``).

    vmcnt counts loads and stores in issue order: a poll issued behind the
    weights waits for them, and a weight wait issued behind a task's DY stores
    waits for their acks (G306 / G307 / G312: -16 us/step with all three
    orders of this and ``shared_down_unit``, none alone)."""
    tab, d, a, rs = c["tab"], c["d"], c["args"], c["rs"]
    n_tasks, blocks = down_y_wave_tasks(d), down_y_unit_blocks(d)
    task0 = (unit % blocks * WAVES + c["wave"]) * n_tasks
    if const_expr(rs.multi):
        # past one token tile: pick tile v of its slot (a TILE of picks at most)
        v = unit // blocks
        u = rs.tile_slot(tab, v)
        first = mplan.tile_pick0(rs.slot_first(tab, u), v, rs.tile_first(tab, u))
        count = fx.min(fx.Int32(TILE), rs.slot_first(tab, u + 1) - first)
    else:
        u = unit // blocks
        first = rs.slot_first(tab, u)
        count = rs.slot_first(tab, u + 1) - first
    e = local_expert(c, rs.union_expert(tab, c["S"], u))
    nst = d.inter // 128
    block_words = down_scale_words(d)
    tasks = [task0 + j for j in range(n_tasks)]

    def weights(t):
        return fp4_tile_weights(c, a["w2"], e, HIDDEN, d.inter, t, 0, nst, d.inter_real)

    def issue():
        return [
            scale_block_loads(c, a["w2_s"], e * HIDDEN + (task0 + 2 * b) * 16, d.down_scale_cols)
            for b in range(n_tasks // 2)
        ], [weights(t) for t in tasks]  # fmt: skip

    svs, avs = load_mid_rows(c, first, count, between=issue)
    for b in range_constexpr(len(svs)):
        scale_block_store(c, svs[b], sl + b * block_words)
    mids = mid_lds_operands(c, count, nst)
    # every task's chains before any DY store
    accss = [
        down_chains(c, tasks[j], sl + j // 2 * block_words, mids, avs[j], down_zero())
        for j in range(n_tasks)
    ]  # fmt: skip
    for j in range_constexpr(n_tasks):
        down_store_y(c, accss[j], tasks[j], first, count)
    gpu.barrier()


@traced
def load_mid_rows(c, first, count, between=None):
    """The MID rows of a slot's picks first .. first + count into LDS rows 0 ..
    of ``dml`` / ``dmsl`` at the ``lds_row`` stride: what ug writes polled
    (tagged pairs) -- a row's words of the real width (two a spec) and its
    real groups' codes (``mid_code_index`` puts a padding group's among them)
    -- and the padding to ``inter`` (ug writes none) zeroed: a zero weight does
    not mask a NaN byte. ``between``: loads issued after the poll's first pass
    and before its check (``Mailbox.poll``), their values returned."""
    d, tid = c["d"], c["tid"]
    words, real_words = d.mid_words, d.inter_real // 4
    groups, steps = d.inter // 32, d.inter // 128
    real_groups = d.inter_real // 32
    assert real_words % 2 == 0
    specs, stores = [], []
    units = TILE * real_words // 2
    for i in range_constexpr(-(-units // THREADS)):
        unit = fx.min(tid + THREADS * i, units - 1)
        r, w = unit // (real_words // 2), unit % (real_words // 2) * 2
        row = first + fx.min(r, count - 1)
        specs.append((c["mid"], Quad((row * words + w) // 2), 2))
        stores.append(
            (c["dml"] + (r * lds_row(words) + w), (tid + THREADS * i < units) & (r < count), None)
        )  # fmt: skip
    for i in range_constexpr(-(-TILE * groups // THREADS)):
        e = fx.min(tid + THREADS * i, TILE * groups - 1)
        r, p = e // groups, e % groups
        real = p % steps * 4 + p // steps < real_groups
        row = first + fx.min(r, count - 1)
        # a padding group's position polls the row's first code instead
        specs.append((c["mids"], row * groups + real.select(p, 0), 1))
        stores.append(
            (c["dmsl"] + (r * lds_row(groups) + p), (tid + THREADS * i < TILE * groups) & (r < count), real)
        )  # fmt: skip
    if const_expr(between is not None):
        got, extra = c["poll"](specs, batch=len(specs), between=between)
    else:
        got = c["poll"](specs, batch=len(specs))
    for (at, live, real), v in zip(stores, got):
        if real is None:
            put_lds_live(at, fx.Vector.from_elements(v, fx.Int32), live)
        else:
            put_lds_live(at, real.select(v[0], fx.Int32(0)), live)
    # the words past the real width
    pad = words - real_words
    for i in range_constexpr(-(-TILE * pad // THREADS)):
        e = fx.min(tid + THREADS * i, TILE * pad - 1)
        at = c["dml"] + (e // pad * lds_row(words) + real_words + e % pad)
        put_lds_live(at, fx.Int32(0), tid + THREADS * i < TILE * pad)
    gpu.barrier()
    if const_expr(between is not None):
        return extra


def mid_lds_operands(c, count, nst):
    """``mid_operands``' B operands of the picks in LDS (``load_mid_rows``):
    column lane % 16 the pick (past the slot's, the last), each K step's two
    16-byte halves and its group's code (``mid_code_index``' order)."""
    lane, d = c["lane"], c["d"]
    j = lane // 16
    row = fx.min(lane % 16, count - 1)
    words = fresh(row * lds_row(d.mid_words) + j * 4)
    vec4 = fx.Vector.make_type(4, fx.Int32)
    bvs = [
        fp8_operand(
            [
                fx.ptr_load(c["dml"] + (words + (st * 32 + 16 * h)), result_type=vec4)
                for h in range(2)
            ]
        )
        for st in range(nst)
    ]
    codes = row * lds_row(d.inter // 32) + j * (d.inter // 128)
    sbs = [fx.Int32(fx.ptr_load(c["dmsl"] + (codes + st))) for st in range(nst)]
    return bvs, sbs


@traced
def down_store_y(c, accs, task, first, count):
    """``down_store`` into DY as tagged pairs: each of picks first .. first +
    count's contribution bf16(v w) to rows 16 task .. (``down_y_combine``
    polls them)."""
    lane, tab, s, rs = c["lane"], c["tab"], c["S"], c["rs"]
    jj = lane % 16
    col = fx.min(jj, count - 1)
    acc = accs[0]
    for h in range_constexpr(1, DOWN_CHAINS):
        acc = acc + accs[h]
    pick = rs.pick_at(tab, first + col)
    w = rs.route_w(tab, s, pick // rs.topk, pick % rs.topk)
    # rows r .. r + 3 (r = 4 (lane // 16)): two words, one 16-byte put
    words = [bf16_pair(acc[q] * w, acc[q + 1] * w).bitcast(fx.Int32) for q in (0, 2)]
    if jj < count:
        at = pick * (HIDDEN // 2) + (task * ROWS + 4 * (lane // 16)) // 2
        c["put_words"](c["dy"], Quad(at // 2), words)


@traced
def shared_down_unit(c, k):
    """Down tasks ``shared_down_tasks`` k .. of the shared expert's down
    projection (its MXFP8 intermediate, ``load_smid_tile``) for every token,
    bf16 -> SDO [task][token][16 rows] as tagged pairs, the intermediate loaded
    once for them. Every task's weights in flight first; wide units
    (``shared_down_wide``) SHARED_DOWN_AHEAD tasks' at a time, the next issued
    before this one's MFMAs. A task's SDO word is held and stored at the
    unit's end: a later task's weight wait would else wait for its store's
    ack (vmcnt counts in issue order; ``down_y_unit``)."""
    s, tid, red, d = c["S"], c["tid"], c["red"], c["d"]
    per = shared_down_tasks(c["rs"])
    tasks = [k * per + j for j in range(per)]
    tiles = token_tiles(s)
    ahead = min(per, SHARED_DOWN_AHEAD) if shared_down_wide(c["rs"]) else per
    shops = [shared_down_loads(c, tasks[j]) for j in range(ahead)]
    rp, tl = tid % (ROWS // 2), tid // (ROWS // 2)
    held = []
    for t0, nn in tiles:
        load_smid_tile(c, t0, nn, d.sh_inter // 32)
        for j in range_constexpr(len(tasks)):
            if const_expr(j + ahead < len(tasks)):
                shops.append(shared_down_loads(c, tasks[j + ahead]))
            gemv_fp8_mfmas(c, d.sh_inter, c["sxl"], c["sxsl"], red, shops[j], rows=nn)
            gpu.barrier()
            word = fx.Int32(0)
            if tid < ROWS * nn // 2:
                vs = [bf16_round(acc_sum(red, 2 * rp + h, tl)) for h in range(2)]
                word = bf16_pair(vs[0], vs[1]).bitcast(fx.Int32)
            held.append((j, t0, nn, word))
            gpu.barrier()
    for j, t0, nn, word in held:
        if tid < ROWS * nn // 2:
            # pair (element ``(task s + t) ROWS + 2 rp``) // 2
            at = (tasks[j] * s + plus(t0, tl)) * (ROWS // 2) + rp
            c["put_words"](c["sdo"], at, [word])


@traced
def down_y_combine(c, tasks, idx):
    """``run_down_y``'s combine, pair ``idx``: rows 2 rp, 2 rp + 1 of one of
    this CTA's ``tasks`` for one token, its top-k picks' contributions polled
    from DY and summed in f32 in top-k order, then the shared expert's (SDO),
    one bf16 rounding, pushed to every rank (token parallel: the token's
    owner) -- ``down_combine``'s arithmetic."""
    s, d = c["S"], c["d"]
    topk = c["rs"].topk
    per_task = s * (ROWS // 2)
    if idx < len(tasks) * per_task:
        j = idx // per_task
        rem = idx % per_task
        t, rp = rem // (ROWS // 2), rem % (ROWS // 2)
        task = tasks[0]
        live = j == 0
        if const_expr(len(tasks) > 1):
            task = (j == 0).select(tasks[0], tasks[1])
            live = (j == 0) | c["second_live"]
        col = task * ROWS + 2 * rp
        got = c["poll"](
            [
                (c["dy"], (t * topk + k) * (HIDDEN // 2) + col // 2, 1)
                for k in range(topk)
            ]
            + [(c["sdo"], ((task * s + t) * ROWS + 2 * rp) // 2, 1)]
        )
        r0, r1 = fx.Float32(0.0), fx.Float32(0.0)
        for k in range_constexpr(topk):
            r0 = r0 + bf_lo(got[k][0])
            r1 = r1 + bf_hi(got[k][0])
        v0 = bf16_round(r0 + bf_lo(got[topk][0]))
        v1 = bf16_round(r1 + bf_hi(got[topk][0]))
        if const_expr(c["shard_f"]):
            # a sharded residual: the column's owner only (a reduce-scatter)
            for p in range_constexpr(d.ranks):
                if live & (col // (HIDDEN // d.ranks) == p):
                    c["put_bf"](
                        preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn"),
                        2 * region_pair(c["rank"], t, s, col), [v0, v1],
                    )  # fmt: skip
        elif const_expr(c["tok_par"]):
            # to the token's owner only, at its own token index there
            own_n = own_count(s, d.ranks)
            owner = t // own_n
            for p in range_constexpr(d.ranks):
                if live & (owner == p):
                    dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                    pair = region_pair(c["rank"], t - p * own_n, own_n, col)
                    c["put_bf"](dst, 2 * pair, [v0, v1])
        else:
            for p in range_constexpr(d.ranks):
                if live:
                    dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                    c["put_bf"](dst, 2 * region_pair(c["rank"], t, s, col), [v0, v1])


# a down pass's K steps at most: TP2's whole intermediate. Past it the slots'
# operands outgrow the VGPRs (expert parallel's whole width: 18 steps spilled
# 624 B a lane at 12 rows), so K runs in parts of DOWN_PART_STEPS, one after
# another (two parts of 9 still spilled 84 B: the accumulators and the next
# part's loads stay live across a part)
DOWN_PASS_STEPS = 9
DOWN_PART_STEPS = 6
# pipelined parts (``down_pipelined``): part p + 1's weights and MID loaded
# before part p's MFMAs. Two parts in flight at once, so half the steps a part
# (6 a part spilled 48-472 B)
DOWN_PIPE_STEPS = 3


def down_pipelined(n) -> bool:
    """Whether a down call of ``n`` tasks pipelines its K parts: for a CTA's
    two tasks (EP 12 / 24 rows -1.8 / -5.9 us). A one-task call holds twice the
    slots a round: pipelined, it spilled 20-56 B."""
    return n == 2


def down_parts(d, pipelined=False) -> int:
    """The K parts of a down slot (``DOWN_PASS_STEPS``, ``DOWN_PART_STEPS``,
    ``DOWN_PIPE_STEPS``)."""
    steps = d.inter // 128
    if steps <= DOWN_PASS_STEPS:
        return 1
    parts = -(-steps // (DOWN_PIPE_STEPS if pipelined else DOWN_PART_STEPS))
    assert steps % parts == 0, f"{steps} K steps in {parts} parts"
    return parts


@traced
def stage_down_parts(c, tasks, nu, sl, sw, slots, per_round):
    """``stage_down``'s rounds with K in ``down_parts``: each (task, slot)'s
    accumulators carried through the parts in step order (the chains as one
    pass over K keeps them: the same sums), each part's weights and MID
    operands loaded at its start, or pipelined (``down_pipelined``) at the
    start of the part before. The first part's weights and every scale block
    load before the round's ug flags, as one pass's do."""
    s, wave, tab, d = c["S"], c["wave"], c["tab"], c["d"]
    n = len(tasks)
    pipe = down_pipelined(n)
    parts = down_parts(d, pipe)
    nst = d.inter // 128 // parts
    for rnd in range(0, (nu + per_round - 1) // per_round, 1):
        us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
        ucs = [fx.min(u, nu - 1) for u in us]
        firsts = [c["rs"].slot_first(tab, u) for u in ucs]
        svs, avs = down_weights(c, tasks, rnd, slots, nu, 0, nst)
        c["poll"]([(c["ugf"], f, 1) for f in down_ug_flags(c, ucs)])
        counts = [c["rs"].slot_first(tab, u + 1) - f for u, f in zip(ucs, firsts)]
        for q in range_constexpr(len(svs)):
            scale_block_store(c, svs[q], sl + q * sw)
        accs = [down_zero() for _ in range(n * slots)]
        mids = [mid_operands(c, f, k, 0, nst) for f, k in zip(firsts, counts)]
        for part in range_constexpr(parts):
            st0 = part * nst
            # a part's loads at its start, or (pipelined) at the start of the
            # part before
            at = part + 1 if pipe else part
            if const_expr(0 < at < parts):
                _, avs_n = down_weights(c, tasks, rnd, slots, nu, at * nst, nst)
                mids_n = [
                    mid_operands(c, f, k, at * nst, nst) for f, k in zip(firsts, counts)
                ]
                if const_expr(not pipe):
                    avs, mids = avs_n, mids_n
            for i in range_constexpr(slots):
                for j in range_constexpr(n):
                    q = j * slots + i
                    accs[q] = down_chains(
                        c, tasks[j], sl + q * sw, mids[i], avs[q], accs[q], st0
                    )
            if const_expr(pipe and part + 1 < parts):
                avs, mids = avs_n, mids_n
        for i in range_constexpr(slots):
            for j in range_constexpr(n):
                down_store(c, j, accs[j * slots + i], firsts[i], counts[i], us[i] < nu)
            # a slot of more than 16 picks: its next pick tile's rows, every
            # part's weights again (L2 hits)
            if const_expr(s > TILE):
                for j0 in range(TILE, counts[i], TILE):
                    down_tile_parts(
                        c, tasks, rnd, slots, nu, i, sl, sw, firsts[i] + j0,
                        counts[i] - j0, us[i] < nu,
                    )  # fmt: skip
    down_finish(c, tasks)


@traced
def down_tile_parts(c, tasks, rnd, slots, nu, i, sl, sw, first, count, live):
    """``stage_down_parts``' pick tile past a slot's first: slot i of the round,
    picks first .., through every K part."""
    n = len(tasks)
    pipe = down_pipelined(n)
    parts = down_parts(c["d"], pipe)
    nst = c["d"].inter // 128 // parts
    accs = [down_zero() for _ in range(n)]
    _, avs = down_weights(c, tasks, rnd, slots, nu, 0, nst)
    more = mid_operands(c, first, count, 0, nst)
    for part in range_constexpr(parts):
        st0 = part * nst
        at = part + 1 if pipe else part
        if const_expr(0 < at < parts):
            _, avs_n = down_weights(c, tasks, rnd, slots, nu, at * nst, nst)
            more_n = mid_operands(c, first, count, at * nst, nst)
            if const_expr(not pipe):
                avs, more = avs_n, more_n
        for j in range_constexpr(n):
            q = j * slots + i
            accs[j] = down_chains(c, tasks[j], sl + q * sw, more, avs[q], accs[j], st0)
        if const_expr(pipe and part + 1 < parts):
            avs, more = avs_n, more_n
    for j in range_constexpr(n):
        down_store(c, j, accs[j], first, count, live)


@traced
def take_down_rank(c):
    """This CTA's arrival rank at the down stage (DQ, a word a tag): which
    CTAs take the extra tiles. A tile's result is the same whichever CTA
    computes it."""
    return take_launch_queue(c, "dq")


@traced
def take_down_rank_unit(c):
    """``take_down_rank`` and the first ``take_down_unit`` (``run_down_y``) in
    one round trip: thread 0's two atomics back to back, both handed over in
    one LDS word (rank | unit << 16). Their words' addresses go through an
    opaque move: on a uniform address LLVM's atomic optimizer reads each
    result back at once, two round trips in series."""
    slot = c["tab"] + c["rs"].queue0
    gpu.barrier()
    if c["tid"] == 0:
        base = c["args"]["queue_words"] + fx.Int64(fresh(fx.Int32(0)))
        rank = fx.Int32(
            queue.take(queue.word(base + fx.Int64(QUEUES.index("dq") * QUEUE_REGION)))
        )
        unit = fx.Int32(
            queue.take(queue.word(base + fx.Int64(QUEUES.index("dnq") * QUEUE_REGION)))
        )
        fx.ptr_store((rank & fx.Int32(0xFFFF)) | (unit << fx.Int32(16)), slot)
    gpu.barrier()
    v = fx.Int32(fx.ptr_load(slot))
    return v & fx.Int32(0xFFFF), v >> fx.Int32(16)


def down_weights(c, tasks, rnd, slots, nu, st0=0, nst=None):
    """Round ``rnd``'s weights of ``stage_down``: each (task, slot)'s scale block
    and weight operands of K steps st0 .. st0 + nst (all of K by default), this
    wave's slots of U (a slot past U: the last's). Loads only."""
    s, wave, tab, a, d = c["S"], c["wave"], c["tab"], c["args"], c["d"]
    nst = d.inter // 128 if nst is None else nst
    us = [wave + WAVES * (slots * rnd + i) for i in range(slots)]
    es = [local_expert(c, c["rs"].union_expert(tab, s, fx.min(u, nu - 1))) for u in us]
    svs = [
        scale_block_loads(c, a["w2_s"], e * HIDDEN + task * 16, d.down_scale_cols)
        for task in tasks
        for e in es
    ]
    avs = [
        fp4_tile_weights(c, a["w2"], e, HIDDEN, d.inter, task, st0, nst, d.inter_real)
        for task in tasks
        for e in es
    ]
    return svs, avs


def mid_operands(c, first, count, st0=0, nst=None):
    """The MFMA B operands of picks first .. first + min(count, 16) (column
    lane % 16 the pick, past them the last) over K steps st0 .. st0 + nst (all
    of K by default): each K step's two 16-byte halves of the lane's MID row
    and its group's code, loaded at device scope straight into registers -- no
    LDS block a slot, so a round holds more slots."""
    lane, d = c["lane"], c["d"]
    j = lane // 16
    row = first + fx.min(lane % 16, count - 1)
    nst = d.inter // 128 if nst is None else nst
    # the step's constant after an opaque lane word folds into the immediate
    # offset; summed with j * 4 it was a hoisted VGPR a step (spill)
    lane_word = fresh(row * d.mid_words + j * 4)
    bvs = [
        fp8_operand(
            [
                bo.buffer_load(
                    rsrc(c["mid"].value),
                    lane_word + (st * 32 + 16 * h),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
                for h in range(2)
            ]
        )
        for st in range(st0, st0 + nst)
    ]
    # this lane's codes, one a K step: contiguous in MIDS (``mid_code_index``),
    # 4 a load -- a load a step was 5 of a slot's 16 load instructions, and the
    # stage is load-issue bound (probe ``harness/down_probe.py``: -8 %)
    base = mid_code_index(d, row, j)
    sbs = []
    for w0 in range_constexpr(0, nst, 4):
        n = min(4, nst - w0)
        v = bo.buffer_load(
            rsrc(c["mids"].value),
            base + (st0 + w0),
            vec_width=n,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
        if const_expr(n == 1):
            sbs.append(fx.Int32(v))
        else:
            v = fx.Vector(v)
            sbs += [fx.Int32(v[k]) for k in range(n)]
    return bvs, sbs


def mid_code_index(d, row, group):
    """MIDS word of pick row ``row``'s MXFP8 group ``group`` (32 columns): a K
    step's 4 groups apart, so the steps of one group % 4 lie together
    (``mid_operands``' lane j reads groups j, j + 4, ..)."""
    steps = d.inter // 128
    return row * (d.inter // 32) + group % 4 * steps + group // 4


def down_ug_flags(c, ucs):
    """This lane's ug flags of a down round's slots ``ucs``: flag f a (slot,
    pick tile past one token tile, part), a tile past the slot's repeating its
    last; lane l takes flags l, l + 64, .. (TP2's 18 parts a tile, 48 rows:
    108 flags a round)."""
    rs = c["rs"]
    ug_parts = rs.ug_parts
    per_slot = rs.slot_tiles * ug_parts if rs.multi else ug_parts
    n = len(ucs) * per_slot
    return [
        _down_ug_flag(c, ucs, fx.min(c["lane"] + 64 * j, n - 1), per_slot)
        for j in range(-(-n // 64))
    ]


def _down_ug_flag(c, ucs, f, per_slot):
    rs, tab = c["rs"], c["tab"]
    ug_parts = rs.ug_parts
    mine = ucs[0]
    for i in range_constexpr(1, len(ucs)):
        mine = (f // per_slot == i).select(ucs[i], mine)
    if const_expr(not rs.multi):
        return mplan.ug_flag(mine, f % ug_parts, ug_parts)
    tile0 = rs.tile_first(tab, mine)
    tiles = rs.tile_first(tab, mine + 1) - tile0
    return mplan.ug_flag(
        mplan.polled_tile(tile0, tiles, f % per_slot // ug_parts),
        f % ug_parts,
        ug_parts,
    )


@traced
def down_shared(c, tasks):
    """Each of ``tasks``' rows of the shared expert's down projection
    (its MXFP8 intermediate, ``load_smid_tile``) for every
    token, bf16 -> LDS ``sdl`` [task][token][16 rows] (``down_combine``):
    every task's weights in flight first."""
    s, tid, red, d = c["S"], c["tid"], c["red"], c["d"]
    shops = [shared_down_loads(c, task) for task in tasks]
    for t0, nn in token_tiles(s):
        load_smid_tile(c, t0, nn, d.sh_inter // 32)
        for j in range_constexpr(len(tasks)):
            gemv_fp8_mfmas(c, d.sh_inter, c["sxl"], c["sxsl"], red, shops[j], rows=nn)
            gpu.barrier()
            if tid < ROWS * nn:
                v = acc_sum(red, tid % ROWS, tid // ROWS)
                at = (j * s + plus(t0, tid // ROWS)) * ROWS + tid % ROWS
                fx.ptr_store(bf16_round(v).bitcast(fx.Int32), c["sdl"] + at)
            gpu.barrier()


def shared_down_loads(c, task):
    """The shared expert's down rows 16 task .. over this rank's part of its
    intermediate: the replicated weight's columns of it under DP
    attention."""
    a, d = c["args"], c["d"]
    if const_expr(not c["rs"].dpa):
        return gemv_fp8_loads(c, a["sw2"], a["sw2_s"], d.sh_inter, task)
    return gemv_fp8_loads(
        c, a["sw2"], a["sw2_s"], d.sh_inter, task,
        k_row=SHARED_INTER, k0=c["rank"] * d.sh_inter,
    )  # fmt: skip


@traced
def down_combine(c, task, j, t0, n):
    """Down task ``task`` (the CTA's j-th) for tokens t0 .. t0 + n: the routed
    contributions (each a bf16, as the reference's experts return) summed in
    f32 in each token's top-k order (expert parallel: of this rank's experts),
    plus the shared expert's (``down_shared``), rounded to bf16 once and pushed
    to every rank (token parallel: to the token's owner)."""
    s, tid, d, tab = c["S"], c["tid"], c["d"], c["tab"]
    topk = c["rs"].topk
    if tid < ROWS * n:
        r = tid & (ROWS - 1)
        tl = tid >> 4
        t = plus(t0, tl)
        shared = fx.ptr_load(c["sdl"] + ((j * s + t) * ROWS + r)).bitcast(fx.Float32)
        # the reference sums its experts' bf16 outputs in f32
        # (``inference/model.py`` ``MoE.forward``), not aiter's atomic stage 2
        # (a bf16 rounding an add)
        routed = fx.Float32(0.0)
        for k in range_constexpr(topk):
            added = routed + contrib_load(c, j, t * topk + k, r)
            if const_expr(c["rs"].ep > 1):
                # another rank's pick: no contribution here (its LDS word
                # never written: selected away, not multiplied by zero)
                local = is_local(c, c["rs"].route_id(tab, s, t, k))
                added = local.select(added, routed)
            routed = added
        # the address afresh: hoisted, it was held a VGPR across every tile's
        # and task's combine (expert parallel, 12 rows a rank: a spill)
        fx.ptr_store(bf16_round(routed + shared), c["pair"] + fresh(tid))
    gpu.barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        v0 = fx.ptr_load(c["pair"] + (tl * ROWS + 2 * rp))
        v1 = fx.ptr_load(c["pair"] + (tl * ROWS + 2 * rp + 1))
        col = task * ROWS + 2 * rp
        if const_expr(c["shard_f"]):
            # a sharded residual: the column's owner only (a reduce-scatter)
            for p in range_constexpr(d.ranks):
                if col // (HIDDEN // d.ranks) == p:
                    c["put_bf"](
                        preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn"),
                        2 * region_pair(c["rank"], t, s, col), [v0, v1],
                    )  # fmt: skip
        elif const_expr(c["tok_par"]):
            # to the token's owner only, at its own token index there
            own_n = own_count(s, d.ranks)
            owner = t // own_n
            for p in range_constexpr(d.ranks):
                if owner == p:
                    dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                    pair = region_pair(c["rank"], t - p * own_n, own_n, col)
                    c["put_bf"](dst, 2 * pair, [v0, v1])
        else:
            for p in range_constexpr(d.ranks):
                dst = preg(c["peer_addr"](p), c["peer"]["ffn"], "ffn")
                c["put_bf"](dst, 2 * region_pair(c["rank"], t, s, col), [v0, v1])
    gpu.barrier()


@traced
def down_mfma(c, j, task, sl, mid, first, count, av, live):
    """Down task ``task`` (the CTA's j-th) on a slot of U: the routed MFMA of its
    picks first .. first + min(count, 16) (their B operands ``mid``:
    ``mid_operands``; the scale block in ``sl``), each pick's contribution."""
    accs = down_chains(c, task, sl, mid, av, down_zero())
    down_store(c, j, accs, first, count, live)


def down_zero():
    """A down slot's accumulators: the K steps on DOWN_CHAINS of them in turn,
    summed in order (``down_store``): one chain put every FP8-rate MFMA's
    latency in series."""
    return [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(DOWN_CHAINS)]


def down_chains(c, task, sl, mid, av, accs, st0=0):
    """``accs`` through K steps st0 .. st0 + len(av) of down task ``task`` on a
    slot (weights ``av``, B operands ``mid``, the scale block in ``sl``): step
    i on chain i % DOWN_CHAINS, wherever a part of K starts."""
    lane, d = c["lane"], c["d"]
    sas = lds_scales(
        sl, task % 2 * 16 + lane % 16, lane, st0, len(av), d.down_scale_cols
    )
    bvs, sbs = mid
    accs = list(accs)
    for i in range_constexpr(len(av)):
        h = (st0 + i) % DOWN_CHAINS
        accs[h] = mfma_scaled(av[i], bvs[i], accs[h], sas[i], sbs[i], FP4, FP8)
    return accs


@traced
def down_store(c, j, accs, first, count, live):
    """The chains summed in order: each of picks first .. first + min(count,
    16)'s contribution (task j's)."""
    lane, tab, s, rs = c["lane"], c["tab"], c["S"], c["rs"]
    jj = lane % 16
    col = fx.min(jj, count - 1)
    acc = accs[0]
    for h in range_constexpr(1, DOWN_CHAINS):
        acc = acc + accs[h]
    pick = rs.pick_at(tab, first + col)
    w = rs.route_w(tab, s, pick // rs.topk, pick % rs.topk)
    for q in range_constexpr(0, 4, 2):
        r = 4 * (lane // 16) + q
        if live & (jj < count):
            word = bf16_pair(acc[q] * w, acc[q + 1] * w).bitcast(fx.Int32)
            contrib_store(c, j, pick * 8 + r // 2, word)


def contrib_store(c, j, i, word):
    """Task j's routed contribution word i ([pick (t, k)][8 words]: rows 2i,
    2i + 1 of bf16(v w), packed)."""
    fx.ptr_store(word, c["contrib"] + (j * c["rs"].picks * 8 + i))


def contrib_load(c, j, pick, r):
    """Row r of task j's contribution of ``pick``: bf16(v w) as f32."""
    word = fx.ptr_load(c["contrib"] + (j * c["rs"].picks * 8 + pick * 8 + (r >> 1)))
    return ((r & 1) == 0).select(bf_lo(word), bf_hi(word))


# ---------------------------------------------------------------- loaders


@traced
def load_smid_tile(c, t0, n, groups):
    """The shared expert's MXFP8 intermediate of tokens t0 .. t0 + n (SMQ /
    SMC, ``smid_quant``) into LDS ``sxl`` / ``sxsl`` rows 0 .., by every
    thread."""
    tid, words = c["tid"], c["d"].sh_inter // 4
    poll_copy(
        c, tid, THREADS,
        [
            (c["smq"], flat_span(t0, words), n * words, c["sxl"], words),
            (c["smc"], flat_span(t0, groups), n * groups, c["sxsl"], groups),
        ],
    )  # fmt: skip
    gpu.barrier()


# ---------------------------------------------------------------- reduce
@traced
def stage_reduce(c, task):
    """Columns 128 task ..: the TP ranks' partials in rank order 0 .. TP - 1, in
    fp32 -> bf16 -> ``out``."""
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * RED_COLS // 2, THREADS):
        reduce_pair(c, task, pas + tid)


@traced
def reduce_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 64."""
    s, tp = c["S"], c["d"].ranks
    a = c["args"]
    # (pair) the pair's body a: the columns to PND too (``layer_pair``)
    pnd = c.get("pnd")
    if idx < s * RED_COLS // 2:
        t = idx // (RED_COLS // 2)
        col = task * RED_COLS + 2 * (idx % (RED_COLS // 2))
        own = preg(c["sym"], c["peer"]["ffn"], "ffn")
        acc0, acc1 = sum_partials(
            c["poll"], own, lambda src: region_pair(src, t, s, col), tp
        )
        bo.buffer_store(acc0.to(fx.BFloat16), rsrc(a["out"]), t * HIDDEN + col)
        bo.buffer_store(acc1.to(fx.BFloat16), rsrc(a["out"]), t * HIDDEN + col + 1)
        if const_expr(pnd is not None):
            # this rank's columns, a tagged bf16 pair a (row, column pair):
            # put_bf's index is the pair's element (= its first word)
            per = RED_TASKS // tp * RED_COLS
            c["put_bf"](pnd, t * per + (col - c["rank"] * per), [acc0, acc1])


def moe_smem(s, rs, timeline):
    """K2b's LDS (``timeline``: with its own stamp record)."""

    @fx.struct
    class Smem:
        # an ug task's partial C: 2 tiles a wave (``ug_red_slot``)
        red: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]
        tab: fx.Array[fx.Int32, rs.tab_words, 16]
        pair: fx.Array[fx.Float32, ROWS * s, 16]
        # one stage's buffers at a time (``stage_lds``)
        stage: fx.Array[fx.Int32, sum(stage_lds(s, rs)["words"].values()), 16]
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]

    return Smem


def ug_x_resident(s, own_routes):
    """Past one token tile up to ``UG_X_RESIDENT_MAX`` rows, every token's x
    row in LDS once a CTA (as one tile's): an ug task reads its picks' rows
    there instead of gathering them. Not token parallel (``own_routes``),
    where the other ranks' rows land late: a task gathers its picks' rows."""
    return TILE < s <= UG_X_RESIDENT_MAX and not own_routes


def stage_lds(s, rs):
    """Word offsets into ``Smem.stage`` of the buffers only one stage reads: the
    router's normalized bf16 x of its K part (``rxl``, ``router_x_normed``);
    the shared stage's MXFP8 x tile; the waves' scale blocks (``wsl``: the ug
    stage's ``UG_WSL_WORDS`` a wave, the down stage's ``down_wsl_words``), then
    the ug stage's routed MXFP8 x tile, or the down stage's token tile of the
    shared expert's MXFP8 intermediate (``load_smid_tile``), each down task's
    routed contributions (``contrib_store``) and its shared expert rows
    (``down_shared``). A barrier ends every stage."""
    d = rs.dims
    # the x tile the shared / ug stages hold (every token's up to one tile)
    rows = tile_rows(s)
    x8 = rows * lds_row(X8_WORDS)
    groups = rows * lds_row(HIDDEN // 32)
    resident = ug_x_resident(s, rs.own_routes)
    ug_rows = s if resident else rows
    # resident rows' codes a byte each (``load_x8_rows(code_bytes=)``)
    ug_codes = (
        ug_rows * UG_CODE_ROW // 4 if resident else ug_rows * lds_row(HIDDEN // 32)
    )
    ug_x8 = ug_rows * lds_row(X8_WORDS)
    ug_wsl = WAVES * UG_WSL_WORDS
    down_wsl = WAVES * down_wsl_words(d)
    sxl = rows * lds_row(d.sh_inter // 4) + lds_tail(d.sh_inter)
    sxsl = rows * lds_row(d.sh_inter // 32)
    # a CTA's two down tasks' contributions, a pick's 16 rows of bf16 a task
    contrib = 2 * rs.picks * 8
    # down by queue units (``run_down_y``): a slot's MID rows and codes, in place of
    # the contributions
    dml = TILE * lds_row(d.mid_words)
    dmsl = TILE * lds_row(d.inter // 32)
    offsets = {
        "rxl": 0, "x8l": 0, "x8sl": x8,
        "wsl": 0, "x8rl": ug_wsl, "x8rsl": ug_wsl + ug_x8,
        "sxl": down_wsl, "sxsl": down_wsl + sxl,
        "contrib": down_wsl + sxl + sxsl,
        "sdl": down_wsl + sxl + sxsl + contrib,
        "dml": down_wsl + sxl + sxsl, "dmsl": down_wsl + sxl + sxsl + dml,
    }  # fmt: skip
    words = {
        "router": rs.router_groups[1] * (lds_row(ROUTER_K // 2) + 1),
        "shared": x8 + groups,
        "ug": ug_wsl + ug_rows * lds_row(X8_WORDS) + ug_codes,
        "down": down_wsl + sxl + sxsl + contrib + 2 * s * ROWS,
        "down_y": down_wsl + sxl + sxsl + dml + dmsl,
    }
    return {"offsets": offsets, "words": {"all": max(words.values())}}


def moe_context(s, rs, lds, mb, bases, layout, scratch, args, rank, sym, numa=""):
    """K2b's stage context: LDS, mailbox, scratch regions, arguments. Its lane math
    starts from a ``fresh`` thread id: in the merged K2 it is not K2a's, held
    from K2a's stages on. The lane by a mask: the signed ``%`` of an id LLVM
    cannot see is non-negative kept its sign fix-up live in VGPRs through the
    ug loop (expert parallel: a spill). The wave a VGPR here, an SGPR in the ug
    stage only (``run_moe``)."""
    tid = fresh(fx.thread_idx.x)
    c = {
        "S": s, "tid": tid, "bid": fx.block_idx.x, "lane": tid & 63,
        "wave": tid // 64,
        # a sharded residual (``layer``): the output reduce-scattered
        "shard_f": False,
        "red": lds.red.ptr, "tab": lds.tab.ptr, "pair": lds.pair.ptr,
        "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words,
        "poll": mb.poll, "peer_addr": lambda p: bases[p],
        "rank": rank, "sym": sym, "peer": peer_offsets(s, rs.dims), "rs": rs,
        "d": rs.dims, "args": args,
        # x (``ffn_normed``) before this launch: plain loads, no readiness wait
        "x_cm": 0, "x_ready": False,
        # token parallel (``layer_post``): the FFN sums reduce-scattered
        "tok_par": False,
        # ... into ``out`` from its first row (DP attention: a rank's own rows
        # are its own step's), not from the rank's first token's
        "out_local": False,
    }  # fmt: skip
    d = rs.dims
    # expert parallel: this rank's first routed expert
    c["e0"] = rank * rs.local_experts if const_expr(rs.ep > 1) else 0
    if const_expr(rs.dpa):
        # DP attention: its part of the replicated shared expert, gate row
        # tiles from rank part, up tiles past every gate tile
        c["sh_gate0"] = rank * d.shared_tasks
        c["sh_up0"] = SHARED_INTER // ROWS + rank * d.shared_tasks
    else:
        c["sh_gate0"], c["sh_up0"] = 0, d.shared_tasks
    for name, off in stage_lds(s, rs)["offsets"].items():
        c[name] = lds.stage.ptr + off
    c.update(region_addrs(scratch, layout, numa, args["numa_word"]))
    # where routes go and are read: this launch's scratch, every row routed here
    c["route_dsts"], c["route_src"] = [c["route"]], c["route"]
    c["route_rows"], c["route_t0"] = s, 0
    if const_expr(rs.routes_pushed):
        # each rank routes its own rows (DP attention: its own step's) and
        # pushes them to every rank's RT (``run_moe``)
        rt = c["peer"]["rt"]
        c["route_src"] = preg(sym, rt, "rt")
        c["route_dsts"] = [preg(bases[p], rt, "rt") for p in range(rs.dims.ranks)]
    return c


@traced
def run_moe(c, key, bid, tls, tl0, before_ug=None):
    """K2b's stages, in order; ``timeline`` stamps from point ``tl0`` + 1.
    ``before_ug()``: a caller's stage between shared and ug, stamped there (the
    later stamps one point on)."""
    s, timeline, tid = key.tokens, key.timeline, c["tid"]
    router_tasks = c["rs"].router_tasks
    route0, shared0 = c["rs"].route0, c["rs"].shared0
    ug_tasks = c["rs"].max_u * c["rs"].ug_parts
    if const_expr(c.get("gather_x") is not None and not c["rs"].dpa):
        # a token-parallel launch: the other ranks' rows land first, each on
        # the CTA whose xq then takes it
        enter_stage("k2b.gather")
        for t in range(first_task(bid, XQ0), s, BLOCKS):
            gather_rows(c, t)
    enter_stage("k2b.xq")
    for t in range(first_task(bid, XQ0), s, BLOCKS):
        if const_expr(c["rs"].dpa):
            # another rank's token: quantized straight from its push
            xq_token_dpa(c, t)
        else:
            await_x(c, [t])
            stage_xq(c, t)
        publish(c["put"], c["x8rdy"], t, 1, tid == 0)
    stamp(timeline, tls, tid, tl0 + 1)
    multi = len(token_tiles(s)) > 1
    enter_stage("k2b.router")
    # DP attention or token parallel: the router and route on this rank's own
    # rows alone (their rows' x ready without the other ranks'), the routes
    # pushed to every rank
    cr = c
    if const_expr(c["rs"].routes_pushed):
        cr = own_view(c, ("x",))
        cr["route_t0"] = cr["t0"]
    rows = cr["S"]
    mine = [plus(cr.get("t0", 0), t) for t in range(rows)]
    for task in range(first_task(bid, c["rs"].router0), router_tasks, BLOCKS):
        # the task's weights in flight through x's wait
        wvs = router_weights(cr, task)
        await_router_x(c, mine)
        stage_router(cr, task, wvs)
    stamp(timeline, tls, tid, tl0 + 2)
    enter_stage("k2b.route")
    for t in range(first_task(bid, route0), rows, BLOCKS):
        stage_route(cr, t)
    stamp(timeline, tls, tid, tl0 + 3)
    enter_stage("k2b.shared")
    c["xl"], c["xsl"] = c["x8l"], c["x8sl"]
    # a CTA's one shared unit (task, token tile) a round: straight-line code,
    # whose tile loads LLVM does not hoist into one live set as it would out of
    # a task loop (spills). A task's tiles in series on one CTA put three tiles'
    # latency on the path down waits on (the weights read again a tile: L2 hits).
    # Past BLOCKS units (DP attention's MoE over every rank's rows) a second round
    units = c["rs"].shared_units
    for k in range_constexpr(-(-units // BLOCKS)):
        unit = first_task(bid, shared0) + k * BLOCKS
        if unit < units:
            shared_unit(c, unit, multi)
    stamp(timeline, tls, tid, tl0 + 4)
    late = before_ug is not None
    if const_expr(late):
        # the caller's LDS is the same union: a barrier hands it over
        gpu.barrier()
        before_ug()
        stamp(timeline, tls, tid, tl0 + 5)
    shift = int(late)
    enter_stage("k2b.ug")
    c["xl"], c["xsl"] = c["x8rl"], c["x8rsl"]
    # the ug stage's wave an SGPR: the signed // of an id LLVM cannot see is
    # non-negative kept its fix-up live in VGPRs through the ug loop (TP 2,
    # 12 rows: a spill). Elsewhere a VGPR: as an SGPR it slowed down (TP 4,
    # 48 rows +4 us)
    cu = {**c, "wave": uniform(c["tid"] >> 6)}
    if const_expr(multi):
        # past one token tile a task gathers its pick tile's rows itself, or up
        # to ``UG_X_RESIDENT_MAX`` rows reads them from every token's x loaded
        # once. The first round placed on the CTAs from ug0 up to the shared
        # units' (those come late), the rest from the queue: a CTA a shared or
        # router task delayed takes fewer of them
        if const_expr(ug_x_resident(s, c["rs"].own_routes)):
            load_x8_rows(
                c, "x8", "x8s", c["x8rl"], c["x8rsl"], lambda j: j, s, s,
                code_bytes=True,
            )  # fmt: skip
            cu["x_code_bytes"] = True
        load_route(c, c["tab"])
        n_tiles = c["rs"].tile_first(c["tab"], c["rs"].n_union(c["tab"], s))
        ug_total = mplan.ug_tasks(n_tiles, c["rs"].ug_parts)
        task = first_task(bid, c["rs"].ug0)
        while ug_task_id(task, ug_placed(c["rs"])) < ug_total:
            task = stage_ug_tile(cu, task, ug_total)
        if const_expr(down_by_unit(c["rs"])):
            publish(c["put"], c["ugc"], bid, 1, tid == 0)
    else:
        # one token tile: every task placed (its x in LDS once a CTA)
        loaded = fx.Int32(0)
        for task in range(first_task(bid, c["rs"].ug0), ug_tasks, BLOCKS):
            if loaded == 0:
                # X8 (ready long before the routing) first: after the routing
                # lands only its own poll remains
                load_x8_rows(c, "x8", "x8s", c["x8rl"], c["x8rsl"], lambda j: j, s)
                load_route(c, c["tab"])
            loaded = fx.Int32(1)
            stage_ug(cu, task)
    stamp(timeline, tls, tid, tl0 + shift + 5)
    enter_stage("k2b.down")
    assert BLOCKS < DOWN_TASKS <= 2 * BLOCKS
    # the route table is in LDS already past one token tile (every CTA loads it
    # for the ug queue) or when every CTA ran an ug iteration (more ug task
    # slots than CTAs), whose first loads it; else load it
    if const_expr(not multi and ug_tasks <= BLOCKS):
        load_route(c, c["tab"])
    # every CTA its tile; the first DOWN_TASKS - BLOCKS to get here an extra
    # one too: the CTAs whose ug ended first (by bid, those ending last held
    # two tiles and ended the stage)
    down_task = first_task(bid, DOWN0)
    if const_expr(c["rs"].down_y):
        rank, unit0 = take_down_rank_unit(c)
    else:
        rank = take_down_rank(c)
    has_second = rank < DOWN_TASKS - BLOCKS
    by_unit = down_by_unit(c["rs"])
    if const_expr(c["rs"].down_y):
        # the combine's second task: live for the first arrivals only
        cy = dict(c)
        cy["second_live"] = has_second
        run_down_y(cy, [down_task, has_second.select(BLOCKS + rank, down_task)], unit0)
    elif const_expr(by_unit):
        # the down stamp at the units' end: the combine is timed in the reduce
        second = has_second.select(BLOCKS + rank, down_task)
        down_by_units(
            {**c, "second_live": has_second}, [down_task, second],
            lambda: stamp(timeline, tls, tid, tl0 + shift + 6),
        )  # fmt: skip
    elif const_expr(c["rs"].down_one_call):
        # a CTA without a second runs its own again (L2 hits), pushed once
        second = has_second.select(BLOCKS + rank, down_task)
        stage_down({**c, "second_live": has_second}, [down_task, second])
    elif has_second:
        stage_down(c, [down_task, BLOCKS + rank])
    else:
        stage_down(c, [down_task])
    if const_expr(not by_unit):
        stamp(timeline, tls, tid, tl0 + shift + 6)
    enter_stage("k2b.reduce")
    # (RQW) a caller's side work on the reduce stage's idle waves (``layer_run``)
    reduce_side = c.get("reduce_side")
    if const_expr(reduce_side is not None):
        reduce_side()
    # token parallel: this rank's tokens' sums only (``out`` from its first
    # token's row, or from row 0: ``out_local``)
    if const_expr(c["tok_par"]):
        cr = own_view(c, () if c["out_local"] else ("out",))
    else:
        cr = c
    # a sharded residual: this rank's columns only (their partials' owner)
    red_own = RED_TASKS // c["d"].ranks if c["shard_f"] else RED_TASKS
    for unit in range(first_task(bid, RED0), red_own, BLOCKS):
        task = unit
        if const_expr(c["shard_f"]):
            task = c["rank"] * red_own + unit
        stage_reduce(cr, task)
    stamp(timeline, tls, tid, tl0 + shift + 7)


@traced
def shared_unit(c, unit, multi):
    """Shared unit ``unit`` (task, token tile): its gate weights in flight
    before x's wait, and its up weights too in one token tile (past it they
    spill a wide build: loaded after the gate's GEMV), the tile's rows,
    ``stage_shared``."""
    s, shared_tasks, a = c["S"], c["d"].shared_tasks, c["args"]
    task = unit % shared_tasks
    gops = gemv_fp8_loads(c, a["sgu"], a["sgu_s"], HIDDEN, plus(c["sh_gate0"], task))
    if const_expr(not multi):
        uops = gemv_fp8_loads(c, a["sgu"], a["sgu_s"], HIDDEN, plus(c["sh_up0"], task))
        load_x8_rows(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: j, s)
        stage_shared(c, task, gops, uops, 0, s)
    else:
        t0 = unit // shared_tasks * TILE
        n = fx.min(fx.Int32(TILE), s - t0)
        load_x8_rows(c, "x8", "x8s", c["x8l"], c["x8sl"], lambda j: t0 + j, n)
        stage_shared(c, task, gops, None, t0, n)


@traced
def xq_token_dpa(c, t):
    """DP attention's xq of token t: this rank's own after its norm, another
    rank's from its push."""
    if t // c["rs"].router_rows == c["rank"]:
        await_x(c, [t])
        stage_xq(c, t)
    else:
        stage_xq_pushed(c, t)


@traced
def gather_rows(c, t):
    """Token t, when another rank owns it: its row here (``gather_x``)."""
    if t // own_count(c["S"], c["d"].ranks) != c["rank"]:
        c["gather_x"](t)


def await_x(c, tokens):
    """In a launch that also writes x (``layer_post``): tokens' rows ready."""
    if const_expr(c["x_ready"]):
        c["poll"]([(c["xrdy"], t, 1) for t in tokens])


def await_router_x(c, tokens):
    """The router's wait for x: none where it normalizes the seam's LIN
    itself (``dot_norm``)."""
    if const_expr(c.get("dot_norm") is None):
        await_x(c, tokens)


def moe_args(*values) -> dict:
    """K2b's kernel arguments by name (``ABI`` order, less the shared tail)."""
    names = ABI.names[: ABI.names.index("scratch")]
    assert len(values) == len(names)
    return dict(zip(names, values))


def build_moe(key: MoeBuild):
    s = key.tokens
    timeline = key.timeline
    assert 1 <= s <= MAX_MOE_ROWS
    rs = route_shape(key)
    assert rs.experts // 64 in SORT_NETS
    layout = scratch_layout(key)
    keyed = key_tuple(key, SOURCES)
    Smem = moe_smem(s, rs, timeline)
    assert lds_bytes(Smem) <= LDS_BYTES, f"K2b's LDS outgrows a CU's at {rs}"

    name = kernel_symbol("v41_moe", s=s, e=rs.experts, k=rs.topk, tl=int(timeline))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def moe(
        x: Int64, gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64,
        w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        queue_words: Int64,
        scratch: Int64, numa_word: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, tl: Int64, epoch: Int64,
    ):  # fmt: skip
        # the step's epoch: every hand-off of the launch tagged with it (``kernels.mailbox``)
        ep = epoch_of(epoch)
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        args = moe_args(
            x,
            gate_w,
            bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            out,
            queue_words,
        )
        args["numa_word"] = numa_word
        c = moe_context(
            s, rs, lds, mailbox(layer, scratch, key.diag_off, layout, epoch=ep),
            peer_bases(peers, rs.dims.ranks), layout, scratch, args,
            rank, sym, key.numa,
        )  # fmt: skip
        tls = lds.tls.ptr
        stamp_begin(timeline, tls, tid, TL_POINTS)
        run_moe(c, key, bid, tls, 0)
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @launcher(name)
    def launch(
        x: Int64, gate_w: Int64, bias: Int64, w13: Int64, w13_s: Int64, w2: Int64,
        w2_s: Int64, sgu: Int64, sgu_s: Int64, sw2: Int64, sw2_s: Int64, out: Int64,
        queue_words: Int64,
        scratch: Int64, numa_word: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, tl: Int64, epoch: Int64, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        moe(
            x, gate_w, bias, w13, w13_s, w2, w2_s, sgu, sgu_s, sw2, sw2_s, out,
            queue_words,
            scratch, numa_word, sym, peers, rank, layer, tl, epoch,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(moe, launch)
    return launch
