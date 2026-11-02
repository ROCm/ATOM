# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from vLLM PR #60683, vllm/models/kimi_k3/amd/mono/stages/moe.py, whose own device layer was a
# fork of ATOM's atom/mono (MIT, Copyright (C) 2026, Advanced Micro
# Devices, Inc.). This port reads those primitives from atom.mono
# directly and takes ATOM's mailbox discipline in place of the fork's.
"""K2b ``moe``: one Kimi-K3 latent-MoE layer, from the MoE input row x to the
layer's output, every TP reduction inside the kernel.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, up to ``MAX_ROWS``
rows: the MFMA's B operand holds 16 token rows, so past 16 the GEMVs after
lnorm and the final reduce run a token group at a time (``group_view``). The original path (vLLM ROCm, see MOE_SPEC.md) replicates the
router and the latent down projection on every rank; here each rank computes
its eighth and the rows are all-gathered over the peer buffer.

    early (CTAs 0 .. 130, one task each, x from the input row; weights first):
      router  (7 x 16 rows of this rank's 112 experts): fp32 logits -> every
              rank's RLOG
      latent  (28 x 16 rows of this rank's 448): bf16, quantized once a
              32-column block (two row groups) to MXFP4 as aiter's inline
              quant -> every rank's LATQ / LATSC
      sgu     (96 x 16 rows of the shared gate | up): bf16 -> SGU
    route   (S, CTAs 131 ..): sigmoid (v_rcp), + bias, top-16, the pre-bias
            sigmoids renormalized (IEEE 1 / sum) -> ROUTE (every rank alike)
    sdown   (448 x 16 rows, K 768): SiTU of SGU (vLLM's kernel) -> bf16 ->
            GEMV -> bf16 partial; this rank's up columns -> SHP (plain, SHRDY
            flags), the rest -> every rank's FINAL; from a queue on CTAs 84 ..
            as each is free (``run_sdown``)
    ug      (|U| x 12, every CTA): the union U of the step's picks; an expert's
            32 gate + 32 up rows against the MXFP4 latent of its picks
            (aiter's inline quant), SiTUv2 (aiter's fp32 epilogue), MXFP4 ->
            INTER / ISC a pick (plain rows in the slot order, a task's UGF flag)
    down    (224 x 16 latent rows): w2 of every expert of U against its picks'
            INTER, bf16(w acc) folded in each token's top-k order with a bf16
            rounding an add -> every rank's LATR
    lnorm   (28 x 128 columns): the ranks' LATR in rank order, bf16, RMSNorm
            (the fused AR + RMSNorm form) -> LN (plain, LNRDY flags)
    up      (56 x 16 rows of this rank's 896 a token group): LN GEMV + SHP ->
            bf16 -> every rank's FINAL
    reduce  (56 x 128 columns, ``reduce``): FINAL in rank order -> bf16 out

Peer regions are reused by every layer: the mailbox tag tells a layer's
pairs from an earlier layer's, and the step's first kernel finds them
zeroed and peer-fenced (``StepMailboxes.begin_step``).
"""

import math
from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.kimi_k3.mono.config import group_rows, groups
from atom.models.kimi_k3.mono.kernels import token_groups
from atom.models.kimi_k3.mono.kernels.attention.kda import (
    _SOURCES,
    TG,
)
from atom.mono.device import gemv, queue, route
from atom.mono.device.mx import FP4, UNIT_SCALE, fp4_tile_load, mfma_scaled, scale_index
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    butterfly,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    kernel_symbol,
    ld_f32,
    load_ptr64,
    popcount,
    row_sum,
    rsrc,
    st_dev,
    traced,
    uniform,
    wave_sum,
)
from atom.mono.device.ranks import peer_bases, sum_partials
from atom.mono.device.select import (
    BATCH,
    count_beaten,
    entry_key,
    f32_key,
    wave_compact,
    wave_floor,
    wave_kth,
)
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, preg, publish, sreg
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import PAIR_BYTES, byte_layout, pair_layout
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.debug import region_ids

REGIONS = (
    "sgu",
    "h",
    "rpart",
    "lpart",
    "route",
    "ugf",
    "lmsq",
    "lnrdy",
    "shrdy",
    "rlog",
    "latq",
    "latsc",
    "latr",
    "latg",
    "fin",
    "fing",
)
# the regions every rank pushes into (every peer's copy)
PEER_REGIONS = ("rlog", "latq", "latsc", "latr", "latg", "fin", "fing")


TP = 8
HIDDEN = 7168
LAT = 3584
E = 896
TOPK = 16
RI = 384  # routed intermediate a rank
SI = 768  # shared intermediate a rank
ROWS = 16
LOG2E = 1.0 / math.log(2.0)
BETA, LBETA = 4.0, 25.0

R_GROUPS = E // TP // ROWS  # 7 router row groups a rank
R_PARTS = 4  # K quarters: the router is on the path to the route
L_GROUPS = LAT // TP // ROWS  # 28 latent row groups a rank
L_PARTS = 2  # K halves: the latent is on the path to ug
R_TASKS = R_GROUPS * R_PARTS  # 28
L_TASKS = L_GROUPS * L_PARTS  # 56
G_TASKS = 2 * SI // ROWS  # 96
EARLY = R_TASKS + L_TASKS + G_TASKS  # 180
LATE0 = EARLY  # route from here
H_CTA = LATE0 + 8  # the shared experts' SiTU (standalone: route takes LATE0 ..)
SD_TASKS = HIDDEN // ROWS  # 448
UP_N = HIDDEN // TP  # 896 columns a rank up-projects
UP_TASKS = UP_N // ROWS  # 56
LN_TASKS = LAT // 128  # 28
LN0 = 56  # lnorm's CTAs (a task's second half, ``ln_halves``: from 0)
LN_HALVES_MAX = 2
RED_TASKS = HIDDEN // 128  # 56
RED0 = 200
UG_GROUPS = RI // 32  # 12
W13_K = LAT  # 3584
W13_STEPS = W13_K // 128  # 28
W2_STEPS = RI // 128  # 3
DOWN_TASKS = LAT // ROWS  # 224
# a wave's experts whose w2 is in flight together; every one's B operands are
# issued before the next batch's w2 (``stage_down``), and at 5 those spilled
DOWN_BATCH = 4
INTER_WORDS = RI // 8  # a pick's MXFP4 intermediate: 48 words
TAGS = 258  # mailbox tags a step: a launch's ``layer`` + 1 (K2's 129 .. 256)
# The task queues: a word each a launch, counting up from the step's clear --
# ug past its first round (``claim_ug_task``), then the shared down
# (``run_sdown``). Their own buffer: the scratch every launch shares holds
# other kernels' words at any offset, and a count must start from zero.
QUEUES = ("ug", "sdown")
QUEUE_BYTES = -(-len(QUEUES) * TAGS * 4 // 16) * 16  # step zeroing's 16 B stores


def queue_offset(layer: int) -> int:
    """The byte offset of the launch ``layer``'s words in the queue buffer
    (``queue.tag_offset``): the launch is handed the queue there, so its
    kernel adds constants only."""
    return queue.tag_offset(layer, len(QUEUES))


def queue_word(c, name):
    """Queue ``name``'s word of this launch (``queue_offset``)."""
    return queue.word(c["queue"] + fx.Int64(QUEUES.index(name) * 4))


def max_u(s: int) -> int:
    """The expert union's bound for an ``s``-row step: every token picks
    ``TOPK``, and there are only ``E`` experts to pick from. ``load_route``
    writes slot ``off`` from a scan over all ``E`` flags and nothing clamps
    it: ``expert`` must hold this many."""
    return min(s * TOPK, E)


def mask_words(s: int) -> int:
    """Words of an expert's token bitmap (``load_route``)."""
    return -(-s // 32)


# The latent's all-reduce in two halves: lnorm task j's 128 columns' partials
# go to rank ``lat_owner(j)`` (ranks own 4, 3, 4, 3, .. of the 28 tasks), whose
# sum, rank order, goes to every rank's LATG (``stage_lnorm``).
LAT_OWN = -(-LN_TASKS // TP) * 128  # the columns a rank owns, at most


def lat_owner(j):
    """The rank owning lnorm task j's columns (int or traced)."""
    return j * TP // LN_TASKS


def lat_first(r):
    """Rank r's first lnorm task (int or traced)."""
    return (r * LN_TASKS + TP - 1) // TP


def group_row(src, t, s):
    """Token t's row of rank ``src``'s block in a group-major peer region (LATR,
    FINAL: [token group][src][the group's tokens], ``group_rows``); one group
    is [src][s tokens]. ``t`` an int or traced; ``s`` the step's rows."""
    g = t // TG
    if s <= TG or s % TG == 0:  # every group whole: one row count
        rows = min(s, TG)
    elif isinstance(t, int):
        rows = group_rows(s, g)
    else:
        rows = fx.min(fx.Int32(s) - g * TG, fx.Int32(TG))
    return g * TP * TG + src * rows + t % TG


TL_POINTS = 8  # start, early, sdown, ug, down, lnorm, up, reduce

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class MoeBuild:
    # ``sym``: the field's short name in the kernel symbol (``symbol_params``)
    tokens: int = field(metadata={"sym": "s"})
    eps: float = 1e-5  # the latent RMSNorm
    # the final reduce in this launch (standalone)
    reduce: bool = field(default=True, metadata={"sym": "r"})
    timeline: bool = field(default=False, metadata={"sym": "t"})
    # bounded waits recording into ``diag`` (debug)
    diag: bool = field(default=False, metadata={"sym": "g"})
    # debug: run phases < stop only (1 early, 2 route/sdown, 3 ug, 4 down,
    # 5 lnorm, 6 up)
    stop: int = field(default=99, metadata={"sym": "st"})


ABI = KernelAbi(
    (
        "x",
        "w_gate",
        "bias",
        "w_ld",
        "w_sgu",
        "w_sd",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "ln_w",
        "w_up",
        "out",
        "scratch",
        "plain",
        "queue",
        "peers",
        "rank",
        "layer",
        "diag",
        "tl",
    )
)


def peer_pairs(s):
    """The peer regions, pairs: name -> count."""
    return {
        "rlog": TP * s * (E // TP),
        "latq": s * LAT // 8,  # MXFP4 words, 8 latent columns each
        "latsc": s * LAT // 32,  # E8M0 codes, one a 32-column block
        # the owner's partials of its latent columns (``lat_owner``):
        # [token group][src][token][LAT_OWN]; the owners' sums: [token][LAT]
        "latr": TP * s * LAT_OWN // 2,
        "latg": s * LAT // 2,
        # the layer output's all-reduce in two halves (``final_owner``): a
        # column's partials go to the rank owning its 896 (its up columns) --
        # [src rank][token][UP_N], group-major as LATR -- and the owner's sum
        # to every rank: [token][HIDDEN]. A rank pushes each peer an eighth
        # of what pushing every partial to every rank would.
        "fin": TP * s * UP_N // 2,
        "fing": s * HIDDEN // 2,
    }


def peer_layout(s):
    return pair_layout(peer_pairs(s).items())


def peer_bytes(s):
    lay = peer_layout(s)
    return max(o + n for o, n in lay.values())


def scratch_layout(key: MoeBuild) -> dict:
    s = key.tokens
    pairs = {
        "sgu": s * SI,
        "h": s * SI // 2,
        "rpart": R_GROUPS * R_PARTS * s * ROWS,
        "lpart": L_GROUPS * L_PARTS * s * ROWS,
        "route": s * 2 * TOPK,
        "ugf": max_u(s) * UG_GROUPS,
        "lmsq": LN_TASKS * s,
        "lnrdy": LN_TASKS * LN_HALVES_MAX,
        "shrdy": UP_TASKS,
    }
    return pair_layout(pairs.items())


def scratch_bytes(key: MoeBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def plain_layout(key: MoeBuild) -> dict:
    """The plain words, in the step's ``plain`` buffer (``byte_layout``): INTER
    and ISC, a pick a row in the slot order (``load_route``), its MXFP4
    intermediate and its codes (an ug task's flag announces them); the bf16 LN
    and SHP rows."""
    s = key.tokens
    return byte_layout(
        (
            ("inter", s * TOPK * INTER_WORDS * 4),
            ("isc", s * TOPK * UG_GROUPS * 4),
            ("ln", s * LAT * 2),
            ("shp", s * UP_N * 2),
        )
    )


def plain_bytes(key: MoeBuild) -> int:
    return max(o + n for o, n in plain_layout(key).values())


def group_view(c, g: int):
    """``c`` seen from token group g by the GEMVs past lnorm (shared down, up)
    and the final reduce: ``S`` its 16 rows, their LN and SHP rows and output
    rows, the group's block of FIN (group-major, ``group_row``, so a group's
    stage indexes it as a 16-row step does) and its rows of FING. One group
    is ``c`` itself (``token_groups.group_view``)."""
    return token_groups.group_view(c, g, GROUP_ROWS, GROUP_REGIONS)


# ``group_view``'s moves: per-token buffers (bytes a row), FIN / FING (bytes a
# group, every peer's copy and this rank's)
GROUP_ROWS = (("ln", LAT * 2), ("shp", UP_N * 2), (("args", "out"), HIDDEN * 2))
GROUP_REGIONS = tuple(
    (("peer", region + side), pairs * PAIR_BYTES)
    for region, pairs in (("fin", TP * TG * UP_N // 2), ("fing", TG * HIDDEN // 2))
    for side in ("", "_own")
)


def f32_of(w):
    return w.bitcast(fx.Float32)


def tanh_libm(x):
    """The ``tanhf`` of x to a few ulp: (1 - e) / (1 + e), e = exp(-2|x|), x's sign."""
    e = hw_exp2(abs(x) * fx.Float32(-2.0 * LOG2E))
    t = (fx.Float32(1.0) - e) / (fx.Float32(1.0) + e)
    return (x < fx.Float32(0.0)).select(-t, t)


def situ_vllm(g, u):
    """The ``situ_and_mul`` of vLLM on bf16 g, u:
    ((2 tanh(g/4)) (1 + tanh(g/2))) (25 tanh(u/25))."""
    a = fx.Float32(0.5 * BETA) * tanh_libm(g * fx.Float32(1.0 / BETA))
    a = a * (fx.Float32(1.0) + tanh_libm(g * fx.Float32(0.5)))
    return a * (fx.Float32(LBETA) * tanh_libm(u * fx.Float32(1.0 / LBETA)))


def tanh_aiter(x):
    """The ``act.tanh_f32`` of aiter: (1 - e) rcp(1 + e), e = exp2(|x| (-2 log2 e))."""
    e = hw_exp2(abs(x) * fx.Float32(-2.0 * LOG2E))
    t = (fx.Float32(1.0) - e) * hw_rcp(fx.Float32(1.0) + e)
    return (x > fx.Float32(0.0)).select(t, -t)


def situ_aiter(g, u):
    """The situv2 epilogue of aiter on fp32 accumulators."""
    sg = hw_rcp(fx.Float32(1.0) + hw_exp2(g * fx.Float32(-LOG2E)))
    a = (fx.Float32(BETA) * tanh_aiter(g * fx.Float32(1.0 / BETA))) * sg
    return (a * fx.Float32(LBETA)) * tanh_aiter(u * fx.Float32(1.0 / LBETA))


def e8m0_roundup(amax):
    """The ``_e8m0_roundup(amax, 6)`` of aiter: ceil code of amax * f32(1/6),
    at most 254."""
    wi = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    b = fx.Int32(fx.Uint32(wi + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)) & 0xFF
    return fx.min(b, fx.Int32(254))


def pk_fp4_f32(old, a, b, scale, j):
    """Two f32 / scale -> E2M1 nibbles into byte j of ``old`` (RNE, saturating)."""
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_f32(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(fx.Float32(a)),
            as_ir_value(fx.Float32(b)),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


def pk_fp4_bf16(old, word, scale, j):
    """A packed bf16 pair word / scale -> E2M1 nibbles into byte j of ``old``."""
    v = fx.Vector.from_elements([fx.Int32(word)], fx.Int32).bitcast(fx.BFloat16)
    return fx.Int32(
        rocdl.cvt_scalef32_pk_fp4_bf16(
            T.i32,
            as_ir_value(fx.Int32(old)),
            as_ir_value(v),
            as_ir_value(fx.Float32(scale)),
            j,
        )
    )


# ---------------------------------------------------------------- early
@traced
def early_out(c, kind, i, q, red, t0=0):
    """Task (i, q) of ``kind`` (0 router, 1 latent, 2 sgu): its 16 rows for
    tokens t0 .. (``red`` holds that token group). The router's parts q > 0
    publish fp32 partials, part 0 adds them in part order and sends the fp32
    rows to every rank's RLOG; the latent: ``latent_out``; sgu bf16 to SGU."""
    s, tid = c["S"], c["tid"]
    sg = group_rows(s, t0 // TG)
    if const_expr(kind == 1):
        latent_out(c, i, q, red, t0)
        return
    if const_expr(kind == 2):
        if tid < (ROWS // 2) * sg:
            rp = tid % (ROWS // 2)
            t = tid // (ROWS // 2)
            v0 = bf16_round(row_sum(red, 2 * rp, t))
            v1 = bf16_round(row_sum(red, 2 * rp + 1, t))
            c["put_bf"](c["sgu"], (t0 + t) * 2 * SI + i * ROWS + 2 * rp, [v0, v1])
    else:
        if tid < ROWS * sg:
            r = tid % ROWS
            t = tid // ROWS
            v = row_sum(red, r, t)
            at = lambda qq: ((i * R_PARTS + qq) * s + t0 + t) * ROWS + r
            if q > 0:
                c["put"](c["rpart"], at(q), v)
            if q == 0:
                got = c["poll"]([(c["rpart"], at(qq), 1) for qq in range(1, R_PARTS)])
                for qq in range_constexpr(R_PARTS - 1):
                    v = v + f32_of(got[qq][0])
                fx.ptr_store(v, c["av"] + tid)
        gpu.barrier()
        if (q == 0) & (tid < ROWS * sg):
            r = tid % ROWS
            t = tid // ROWS
            v = fx.ptr_load(c["av"] + tid)
            for p in range_constexpr(TP):
                c["put"](
                    c["peer"]["rlog"][p],
                    (c["rank"] * s + t0 + t) * (E // TP) + i * ROWS + r,
                    v,
                )
    gpu.barrier()


@traced
def latent_out(c, i, q, red, t0=0):
    """Latent task (i, q), tokens t0 ..: the MXFP4 block of 32 columns is two
    row groups (2b, 2b + 1) of one rank, so part 0 of the even group gathers
    both groups' partials (its own part 1, the odd group's parts 0 and 1; each
    group summed in part order, as one group alone), rounds them to bf16,
    quantizes the block as aiter's inline quant (``quant_block``) and sends the
    four FP4 words and the E8M0 code to every rank's LATQ / LATSC. Every other
    part only publishes its fp32 partial: the latent is quantized once, here,
    not by every CTA of every rank."""
    s, tid = c["S"], c["tid"]
    sg = group_rows(s, t0 // TG)
    lead = (q == 0) & (i % 2 == 0)
    at = lambda g, qq, t, r: ((g * L_PARTS + qq) * s + t0 + t) * ROWS + r
    if tid < ROWS * sg:
        r = tid % ROWS
        t = tid // ROWS
        v = row_sum(red, r, t)
        if lead:
            got = c["poll"](
                [
                    (c["lpart"], at(i, 1, t, r), 1),
                    (c["lpart"], at(i + 1, 0, t, r), 1),
                    (c["lpart"], at(i + 1, 1, t, r), 1),
                ]
            )
            fx.ptr_store(v + f32_of(got[0][0]), c["av"] + tid)
            fx.ptr_store(
                f32_of(got[1][0]) + f32_of(got[2][0]), c["av"] + (ROWS * sg + tid)
            )
        else:
            c["put"](c["lpart"], at(i, q, t, r), v)
    gpu.barrier()
    if lead & (tid < sg):
        # token t0 + tid's block: 32 columns, the even group's 16 then the odd's
        words = []
        for g in range_constexpr(2):
            for rp in range_constexpr(ROWS // 2):
                base = c["av"] + (g * ROWS * sg + tid * ROWS + 2 * rp)
                v0 = bf16_round(fx.ptr_load(base))
                v1 = bf16_round(fx.ptr_load(base + 1))
                words.append(
                    fx.Vector.from_elements([v0, v1], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
        q4, code = quant_block(words)
        kb = c["rank"] * (LAT // TP // 32) + i // 2
        tt = t0 + tid
        for p in range_constexpr(TP):
            c["put_words"](c["peer"]["latq"][p], tt * (LAT // 8) + kb * 4, q4[:2])
            c["put_words"](c["peer"]["latq"][p], tt * (LAT // 8) + kb * 4 + 2, q4[2:])
            c["put"](c["peer"]["latsc"][p], tt * (LAT // 32) + kb, code)
    gpu.barrier()


def quant_block(ws):
    """16 packed bf16 pair words (32 values) -> MXFP4 as aiter's inline quant:
    the amax of |bf16| bits, the ceil E8M0 code of amax / 6, then
    v_cvt_scalef32_pk_fp4_bf16 -> (4 words, code)."""
    am = fx.Int32(0)
    for w in ws:
        am = fx.max(am, w & 0x7FFF)
        am = fx.max(am, (w >> 16) & 0x7FFF)
    amax = (am << 16).bitcast(fx.Float32)
    code = e8m0_roundup(amax)
    scale = (code << 23).bitcast(fx.Float32)
    out = []
    for q in range(4):
        pk = fx.Int32(0)
        for j in range(4):
            pk = pk_fp4_bf16(pk, ws[4 * q + j], scale, j)
        out.append(pk)
    return out, code


def early_shape(kind):
    """(K a part, parts) of an early kind: router a quarter, latent a half, sgu all."""
    return {
        0: (HIDDEN // R_PARTS, R_PARTS),
        1: (HIDDEN // L_PARTS, L_PARTS),
        2: (HIDDEN, 1),
    }[kind]


def early_loads(c, kind, i, q):
    """Early task (i, q) of ``kind``'s weights for this wave: issued as early as
    the CTA can, then ``early_compute`` takes them."""
    a = c["args"]
    klen, _ = early_shape(kind)
    if kind == 0:
        w, row0 = a["w_gate"], c["rank"] * (E // TP) + i * ROWS
    elif kind == 1:
        w, row0 = a["w_ld"], c["rank"] * (LAT // TP) + i * ROWS
    else:
        w, row0 = a["w_sgu"], i * ROWS
    return [
        gemv.loads(
            c["lane"],
            c["wave"],
            w,
            row0,
            n,
            ldk=HIDDEN,
            k0=q * klen + k0,
            paired=True,
        )
        for k0, n in gemv.slabs(klen, min(c["S"], TG))
    ]


@traced
def early_compute(c, kind, i, q, ops):
    """Early task (i, q): x's K slice (the input row, or -- ``xflags`` -- the
    row this launch's AttnRes stage publishes) against ``ops`` -> ``early_out``.

    The slice is staged a slab at a time (``gemv.slabs``). At every row count up
    to 8 that is one slab and the whole slice, as it always was; a wider step
    splits it on group boundaries, which leaves the accumulation order alone.
    """
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    sg = min(s, TG)
    klen, _ = early_shape(kind)
    plan = gemv.slabs(klen, sg)
    xrow = gemv.slab_width(klen, sg) + gemv.LDS_PAD
    if const_expr(c.get("xflags") is not None):
        if tid < c["xflags"]:
            c["poll"]([(c["xrdy"], tid, 1)])
        gpu.barrier()
    if const_expr(groups(s) > 1):
        acc = early_units(c, q, klen, [op for sl_ops in ops for op in sl_ops])
        for g in range_constexpr(groups(s)):
            fx.ptr_store(acc[g], red + (wave * 64 + lane) * 4)
            gpu.barrier()
            early_out(c, kind, i, q, red, TG * g)
        return
    # one token group: its rows staged a slab at a time against the slab's
    # weights. ``range_constexpr`` + indexing, not ``enumerate``: this body is
    # AST-rewritten (``@traced``), and a ``for`` over an arbitrary Python
    # iterable with tuple unpacking is not a form the rewriter is meant to see.
    src = c["xbuf"] if const_expr(c.get("xflags") is not None) else c["args"]["x"]
    cm = CM_DEV if const_expr(c.get("xflags") is not None) else 0
    acc = None
    for sl in range_constexpr(len(plan)):
        k0 = plan[sl][0]
        n = plan[sl][1]
        if const_expr(sl):
            # the previous slab's readers before its LDS is overwritten
            gpu.barrier()
        gemv.rows_to_lds(
            tid, src, n, sg, c["xl"], xrow, cm, ldk=HIDDEN, k0=q * klen + k0
        )
        acc = gemv.mfmas(lane, wave, c["xl"], xrow, n, sg, ops[sl], acc)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    early_out(c, kind, i, q, red, 0)


def early_units(c, q, klen, ops):
    """``early_compute`` past one token group: ``gemv.staged_units`` over the
    units -- a column group (``gemv.KGROUP``) of a token group -- with x one
    unit ahead (``ops``: ``gemv.loads``' entries, one a column group, all in
    flight already). Column groups in order per token group: ``gemv.mfmas``'
    accumulation order. -> an accumulator a group."""
    tid, lane, wave, G = c["tid"], c["lane"], c["wave"], groups(c["S"])
    xrow = gemv.KGROUP + gemv.LDS_PAD
    bufs = [c["xl"], c["xl"] + TG * xrow // 2]
    src = c["xbuf"] if const_expr(c.get("xflags") is not None) else c["args"]["x"]
    cm = CM_DEV if const_expr(c.get("xflags") is not None) else 0

    def width(i):
        return min(gemv.KGROUP, klen - i * gemv.KGROUP)

    def rows(g):
        return group_rows(c["S"], g)

    def unit_loads(i, g):
        return gemv.rows_loads(
            tid,
            src + fx.Int64(TG * g * HIDDEN * 2),
            rows(g),
            width(i),
            HIDDEN,
            k0=q * klen + i * gemv.KGROUP,
            cm=cm,
        )

    units = [(i, g) for i in range(len(ops)) for g in range(G)]
    acc = [fx.Vector.filled(4, 0.0, fx.Float32)] * G

    def store(u, xs, buf):
        i, g = units[u]
        gemv.rows_store(tid, xs, rows(g), width(i), buf, xrow)

    def multiply(u, buf):
        i, g = units[u]
        acc[g] = gemv.mfmas_unit(
            lane, wave, buf, xrow, i, klen, rows(g), ops[i], acc[g]
        )

    gemv.staged_units(
        len(units), 1, bufs, lambda u: unit_loads(*units[u]), store, multiply
    )
    return acc


@traced
def stage_early(c, bid):
    """K2b alone: CTA bid's early task (router parts, latent parts, sgu)."""
    lb = bid - R_TASKS
    gb = bid - R_TASKS - L_TASKS
    if bid < R_TASKS:
        early_compute(
            c,
            0,
            bid // R_PARTS,
            bid % R_PARTS,
            early_loads(c, 0, bid // R_PARTS, bid % R_PARTS),
        )
    if (bid >= R_TASKS) & (bid < R_TASKS + L_TASKS):
        early_compute(
            c,
            1,
            lb // L_PARTS,
            lb % L_PARTS,
            early_loads(c, 1, lb // L_PARTS, lb % L_PARTS),
        )
    if bid >= R_TASKS + L_TASKS:
        early_compute(c, 2, gb, 0, early_loads(c, 2, gb, 0))


# ---------------------------------------------------------------- route
@traced
def stage_route(c, t):
    """Token t (wave 0): 896 logits, 14 a lane (expert lane + 64 i) -> sigmoid
    (v_rcp) + bias -> the top-16 ranked by score, descending, equal scores the
    lower expert first (aiter's slot order) -> the pre-bias sigmoids
    renormalized in that order (IEEE 1 / sum).

    The top 16 by counting (``device.select``'s one-wave form): the keys at
    least a floor of the 16th largest (``wave_floor``) compacted, each ranked
    by how many of them beat it; past ``ROUTE_CAP`` candidates a radix bit
    search over all 896 keys (``wave_kth``), with the same order."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    a = c["args"]
    if wave == 0:
        s = c["S"]
        per = E // 64
        specs = []
        for i in range_constexpr(per):
            e = lane + 64 * i
            src = e // (E // TP)
            specs.append(
                (c["peer"]["rlog_own"], (src * s + t) * (E // TP) + e % (E // TP), 1)
            )
        # the bias does not depend on the logits: in flight under the poll
        bias = [ld_f32(a["bias"], lane + 64 * i) for i in range(per)]
        got = c["poll"](specs)
        stamp(c["tl_on"], c["tls"], c["tid"], 14)
        keys, sigs = [], []
        for i in range_constexpr(per):
            lg = f32_of(got[i][0])
            sg = hw_rcp(fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
            sc = sg + bias[i]
            # a NaN logit (a warmup's garbage row) ranks last
            # x == x is false only for a NaN: no isnan intrinsic here
            live = sc == sc  # noqa: PLR0124
            keys.append(f32_key(live.select(sc, fx.Float32(-3.0e38))))
            sigs.append((sg == sg).select(sg, fx.Float32(0.0)))  # noqa: PLR0124
        # an entry's id: the lower expert the larger, so it wins a tie
        ids = [fx.Int32(E - 1) - (lane + 64 * i) for i in range(per)]
        cand, csig = c["rcand"], c["rsig"]
        floor = wave_floor(keys, TOPK)
        n = wave_compact(
            cand,
            ROUTE_CAP,
            [(keys[i] >= floor, keys[i], ids[i]) for i in range(per)],
            lane,
            [(csig, sigs)],
        )
        if n <= ROUTE_CAP:
            _route_rank(cand, csig, n, lane, red)
        else:
            sel = wave_kth(keys, TOPK, lane)
            wave_compact(
                cand,
                ROUTE_CAP,
                [(sel[i], keys[i], ids[i]) for i in range(per)],
                lane,
                [(csig, sigs)],
            )
            _route_rank(cand, csig, fx.Int32(TOPK), lane, red)
        # the sigmoids in rank order, summed serially (aiter's renorm)
        total = fx.Float32(0.0)
        for j in range_constexpr(TOPK):
            total = total + fx.ptr_load(red + (R_SIG + j))
        inv = fx.Float32(1.0) / total
        if lane < TOPK:
            me = fx.ptr_load(red + (R_EXP + lane)).bitcast(fx.Int32)
            ms = fx.ptr_load(red + (R_SIG + lane))
            c["put"](
                c["route"], t * 2 * TOPK + lane, fx.min(fx.max(me, 0), fx.Int32(E - 1))
            )
            c["put"](c["route"], t * 2 * TOPK + TOPK + lane, ms * inv)
    gpu.barrier()


# The route's words of ``red`` (one wave's): the picks' sigmoids and experts
# by rank. The candidates: ``rcand`` (``put_key`` entries) and their sigmoids
# ``rsig``, an LDS region of the stage union.
R_SIG = 64
R_EXP = R_SIG + TOPK
# Candidates, two a lane: past this many the route takes the radix path.
ROUTE_CAP = 128
ROUTE_SLOTS = ROUTE_CAP + BATCH  # ``wave_compact``'s dead tail included


@fx.struct
class RouteSelLds:
    """The route's candidates: a member of a build's stage union (the route
    runs between stages, past a barrier on each side)."""

    cand: fx.Array[fx.Int32, 2 * ROUTE_SLOTS, 16]
    sig: fx.Array[fx.Float32, ROUTE_SLOTS, 16]


@traced
def _route_rank(cand, csig, n, lane, red):
    """Each of the n (<= ``ROUTE_CAP``) candidates' rank: how many beat it
    (``count_beaten``, candidates ``lane`` and ``lane + 64`` in one pass). The
    sort keys are distinct, so the ranks are; ranks 0 .. 15 are the top 16,
    stored by rank: the sigmoid at red[R_SIG + r], the expert at red[R_EXP + r]."""
    qs = [lane, lane + 64]
    ranks = count_beaten(cand, [entry_key(cand, q) for q in qs], n)
    for i in range_constexpr(2):
        q = qs[i]
        if (q < n) & (ranks[i] < TOPK):
            expert = fx.Int32(E - 1) - fx.ptr_load(cand + 2 * q)
            fx.ptr_store(fx.ptr_load(csig + q), red + (R_SIG + ranks[i]))
            fx.ptr_store(expert.bitcast(fx.Float32), red + (R_EXP + ranks[i]))


def passes(n):
    """A CTA's passes over n work items, a thread an item a pass."""
    return (n + THREADS - 1) // THREADS


def pick_expert(rt, p):
    """Pick p's (token p / 16, rank p % 16) expert from the route table."""
    t, k = p // TOPK, p % TOPK
    return fx.max(fx.min(fx.ptr_load(rt["route"] + (t * 2 * TOPK + k)), E - 1), 0)


@traced
def load_route(c):
    """ROUTE -> LDS (ids, weights); the union U (ascending): expert[slot],
    slot[e], first[slot] (``first[|U|]`` = every pick) and the picks in slot
    order, a slot's tokens ascending (``order``: pick ids t 16 + k), the row
    order of INTER. Returns |U|.

    Every CTA must derive the same order (ug writes a pick's INTER row, down
    reads it): each pick ORs its token's bit into its expert's bitmap
    (``rt["mask"]``, the stage union: only this stage reads it), so an
    expert's pick count is a popcount and a pick's rank in its slot the
    bitmap's tokens below its own, whatever order the ORs land in."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    mw = mask_words(s)
    n = s * 2 * TOPK
    for i in range_constexpr(passes(n)):
        j = tid + THREADS * i
        if j < n:
            v = c["poll"]([(c["route"], j, 1)])[0][0]
            fx.ptr_store(v, rt["route"] + j)
    for i in range_constexpr(passes(E * mw)):
        j = tid + THREADS * i
        if j < E * mw:
            fx.ptr_store(fx.Int32(0), rt["mask"] + j)
    gpu.barrier()
    for i in range_constexpr(passes(s * TOPK)):
        p = tid + THREADS * i
        if p < s * TOPK:
            route.mark_token(rt["mask"], mw, pick_expert(rt, p), p // TOPK)
    gpu.barrier()
    # two experts a thread: their pick counts; slots and first picks by scans
    ea, eb = fx.min(2 * tid, E - 1), fx.min(2 * tid + 1, E - 1)
    na, nb = fx.Int32(0), fx.Int32(0)
    for w in range_constexpr(mw):
        na = na + popcount(fx.ptr_load(rt["mask"] + (ea * mw + w)))
        nb = nb + popcount(fx.ptr_load(rt["mask"] + (eb * mw + w)))
    na = (2 * tid < E).select(na, fx.Int32(0))
    nb = (2 * tid + 1 < E).select(nb, fx.Int32(0))
    ua, ub = (na > 0).select(fx.Int32(1), fx.Int32(0)), (nb > 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    off, tot = block_excl_scan(ua + ub, lane, wave, c["scan"])
    poff, ptot = block_excl_scan(na + nb, lane, wave, c["scan"])
    if ua == 1:
        fx.ptr_store(off, rt["slot"] + ea)
        fx.ptr_store(ea, rt["expert"] + off)
        fx.ptr_store(poff, rt["first"] + off)
    if ub == 1:
        fx.ptr_store(off + ua, rt["slot"] + eb)
        fx.ptr_store(eb, rt["expert"] + off + ua)
        fx.ptr_store(poff + na, rt["first"] + off + ua)
    if tid == THREADS - 1:
        fx.ptr_store(ptot, rt["first"] + tot)
    gpu.barrier()
    for i in range_constexpr(passes(s * TOPK)):
        p = tid + THREADS * i
        if p < s * TOPK:
            t = p // TOPK
            e = pick_expert(rt, p)
            # the tokens below t that picked e: the picks before p in its slot
            rank = route.tokens_below(rt["mask"], mw, e, t)
            slot = fx.ptr_load(rt["slot"] + e)
            fx.ptr_store(p, rt["order"] + (fx.ptr_load(rt["first"] + slot) + rank))
    gpu.barrier()
    return uniform(tot)


@traced
def load_latq(c):
    """Every rank's quantized latent (LATQ words, LATSC codes, ``latent_out``)
    -> LDS: a thread a (token, 32-block), its four words and its code; the
    codes packed four a word (a 128-column step's, block 4 st + b in byte b),
    so that 64 rows of the latent fit LDS."""
    s, tid = c["S"], c["tid"]
    nblk = s * (LAT // 32)
    for i in range_constexpr((nblk + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nblk - 1)
        t = b // (LAT // 32)
        kb = b % (LAT // 32)
        w0 = t * (LAT // 8) + kb * 4
        got = c["poll"](
            [(c["peer"]["latq_own"], w0 + q, 1) for q in range(4)]
            + [(c["peer"]["latsc_own"], t * (LAT // 32) + kb, 1)]
        )
        if const_expr(i == 0):
            stamp(c["tl_on"], c["tls"], tid, 17)
        # a step's four blocks are four adjacent lanes (nblk and THREADS are
        # multiples of 4): their codes, each in its byte, summed across them
        packed = butterfly(got[4][0] << ((kb % 4) * 8), (1, 2))
        if tid + THREADS * i < nblk:
            for q in range_constexpr(4):
                fx.ptr_store(got[q][0], c["latq"] + (w0 + q))
            if kb % 4 == 0:
                fx.ptr_store(packed, c["latsc"] + (t * (LAT // 128) + kb // 4))
    gpu.barrier()


# ---------------------------------------------------------------- ug
def w13_scale_word(e, rg, st, lane):
    """The dword of aiter's gate_up=False ``shuffle_scale`` holding (row rg 16 +
    lane % 16, K block 4 st + lane / 16) for st's pair and rg's pair: byte
    (st % 2) 2 + rg % 2."""
    n1 = rg // 2
    k1 = st // 2
    return (
        e * (2 * RI * (W13_K // 32) // 4)
        + ((n1 * (W13_STEPS // 2) + k1) * 4 + lane // 16) * 16
        + lane % 16
    )


@traced
def claim_ug_task(c, total):
    """Claim an ug task past the first round from the launch's ug queue
    (``queue_word``) into LDS ``scan[0]`` (tid 0; read it past a barrier). A
    CTA its early or route work delayed takes fewer tasks, where round-robin
    gave every CTA the same count and the stage ended with the latest
    starter; a task's result is the same whichever CTA runs it. One round's
    worth of tasks or fewer: no claim (``total``)."""
    if c["tid"] == 0:
        task = total
        if total > BLOCKS:
            task = BLOCKS + queue.take(queue_word(c, "ug"))
        fx.ptr_store(fx.Int32(task), c["scan"])


@traced
def run_ug(c, bid, nu):
    """Every ug task of the step's union: the first round placed (reversed:
    its leftover tasks go to the high CTAs, whose early work ends first), the
    rest from the queue. A task's weights are in flight under the one before
    it (past the end: the last task's, cache hits), and its claim one task
    further ahead: the atomic's round trip under a whole task."""
    total = nu * UG_GROUPS
    last = total - 1
    task = BLOCKS - 1 - bid
    claim_ug_task(c, total)
    ops = ug_loads(c, fx.min(task, last))
    gpu.barrier()
    nxt = fx.ptr_load(c["scan"])
    while task < total:
        gpu.barrier()  # every thread read the claim before the next one lands
        claim_ug_task(c, total)
        ops_n = ug_loads(c, fx.min(nxt, last))
        stage_ug(c, task, ops)
        # ``stage_ug`` ends at a barrier: the claim is visible
        task = nxt
        nxt = fx.ptr_load(c["scan"])
        ops = ops_n


def ug_loads(c, task):
    """Task ``task``'s w13 operands for this wave (its row group, its K half)
    and their scale words: issued a task ahead (``stage_ug`` takes them)."""
    lane, wave = c["lane"], c["wave"]
    a, rt = c["args"], c["rt"]
    slot = task // UG_GROUPS
    g = task % UG_GROUPS
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    rgl = wave % 4  # 0, 1 gate; 2, 3 up
    half = wave // 4
    rg = (rgl // 2) * (RI // ROWS) + 2 * g + rgl % 2
    r_w = rsrc(a["w13"])
    base = e * (2 * RI * W13_K // 8)
    steps = [half * (W13_STEPS // 2) + i for i in range(W13_STEPS // 2)]
    av = [fp4_tile_load(r_w, base, rg, W13_K, st, lane * 4) for st in steps]
    r_s = rsrc(a["w13s"])
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s, w13_scale_word(e, rg, st, lane), vec_width=1, dtype=T.i32
            )
        )
        for st in steps[::2]
    ]
    return av + sw


@traced
def stage_ug(c, task, ops):
    """Slot task / 12, 32-column group task % 12: gate rows 32 g .. and up rows
    384 + 32 g .. of the slot's expert (``ops``: ``ug_loads``) against the MXFP4
    latent of the slot's picks, 16 a pass (``ug_tile``); then the task's UGF
    flag, past its INTER stores."""
    rt = c["rt"]
    slot = task // UG_GROUPS
    first = fx.ptr_load(rt["first"] + slot)
    count = uniform(fx.ptr_load(rt["first"] + slot + 1) - first)
    if const_expr(c["S"] <= TG):
        # a token picks an expert once: at most S picks a slot
        ug_tile(c, task, ops, first, count)
    else:
        j0 = fx.Int32(0)
        while j0 < count:
            ug_tile(c, task, ops, first + j0, count - j0)
            j0 = j0 + TG
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        c["put"](c["ugf"], task, 1)


@traced
def ug_tile(c, task, ops, first, count):
    """``stage_ug``'s pass over the picks at first .. first + min(count, 16)
    (pick order, ``load_route``): a pick an MFMA column, its token's latent row
    from LDS; the epilogue writes each pick's INTER row."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    g = task % UG_GROUPS
    rgl = wave % 4
    half = wave // 4
    rg = (rgl // 2) * (RI // ROWS) + 2 * g + rgl % 2
    steps = [half * (W13_STEPS // 2) + i for i in range(W13_STEPS // 2)]
    av = ops[: len(steps)]
    sw = ops[len(steps) :]
    cnt = fx.min(count, TG)
    t = fx.ptr_load(rt["order"] + first + fx.min(lane % 16, cnt - 1)) // TOPK
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(steps)):
        st = steps[i]
        bw = fx.Vector(
            fx.ptr_load(
                c["latq"] + (t * (LAT // 8) + st * 16 + kb * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        sb = (fx.ptr_load(c["latsc"] + (t * (LAT // 128) + st)) >> (kb * 8)) & 0xFF
        # the code's byte (st % 2) 2 + rg % 2 of the word; rg % 2 is the wave's
        sa = (sw[i // 2] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
        acc = mfma_scaled(av[i], bw, acc, sa, sb, FP4, FP4, 0, 0)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # thread (pick column, intermediate column): g / u summed over the K
    # halves, SiTUv2
    tt = tid // 32
    col = tid % 32
    live = tt < cnt
    tt = fx.min(tt, cnt - 1)
    first_rg = (
        col < 16
    )  # gate / up row group 0 (waves 0, 2) or 1 (1, 3); +4: K's 2nd half
    r = col % 16
    gv = first_rg.select(
        row_sum(red, r, tt, waves=(0, 4)), row_sum(red, r, tt, waves=(1, 5))
    )
    uv = first_rg.select(
        row_sum(red, r, tt, waves=(2, 6)), row_sum(red, r, tt, waves=(3, 7))
    )
    v = situ_aiter(gv, uv)
    amax = butterfly(abs(v), (16, 8, 4, 2, 1), fx.max)
    code = e8m0_roundup(amax)
    gpu.barrier()
    fx.ptr_store(v, c["av"] + tid)
    gpu.barrier()
    if live & (col < 4):
        scale = (code << 23).bitcast(fx.Float32)
        pk = fx.Int32(0)
        for j in range_constexpr(4):
            x0 = fx.ptr_load(c["av"] + (tt * 32 + col * 8 + 2 * j))
            x1 = fx.ptr_load(c["av"] + (tt * 32 + col * 8 + 2 * j + 1))
            pk = pk_fp4_f32(pk, x0, x1, scale, j)
        row = first + tt
        st_dev(pk, c["inter"], row * INTER_WORDS + g * 4 + col)
        if col == 0:
            st_dev(code, c["isc"], row * UG_GROUPS + g)
    gpu.barrier()


# ---------------------------------------------------------------- down
def w2_scale_words(c, e, rgd, lane):
    """The dwords of w2's e8m0 codes (row e 3584 + rgd 16 + lane % 16, K block
    4 st + lane / 16) in ``e8m0_shuffle``'s layout (16 codes a row) for the 3
    K steps: steps 0 and 1 share a dword (``scale_index``'s d bit), step 2 has
    its own. Raw: the byte is picked at its use (``w2_code``), so a batch
    loaded ahead is not waited for to shift it."""
    row = e * LAT + rgd * ROWS + lane % 16
    return [
        fx.Int32(
            bo.buffer_load(
                rsrc(c["args"]["w2s"]),
                scale_index(row, col + lane // 16, 16) // 4,
                vec_width=1,
                dtype=T.i32,
            )
        )
        for col in (0, 8)
    ]


def w2_code(words, rgd, st):
    """Step st's code of ``w2_scale_words``: byte 2 (st % 2) + rgd % 2 (the
    layout's d and a bits; a = row / 16 % 2 = rgd % 2, LAT being a multiple of
    32) of word st / 2."""
    return (words[st // 2] >> ((st % 2) * 16 + (rgd % 2) * 8)) & 0xFF


def inter_operands(c, row, live):
    """A lane's B operands of INTER row ``row`` (its 3 K steps' words at lane
    group kb = lane / 16, and their codes), plain loads at device scope (the
    slot's ug flags polled first); a dead column multiplies zero."""
    kb = c["lane"] // 16
    out = []
    for st in range_constexpr(W2_STEPS):
        bw = fx.Vector(
            bo.buffer_load(
                rsrc(c["inter"]),
                row * INTER_WORDS + st * 16 + kb * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        code = fx.Int32(
            bo.buffer_load(
                rsrc(c["isc"]),
                row * UG_GROUPS + st * 4 + kb,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        bw = fx.Vector.from_elements(
            [live.select(bw[d], fx.Int32(0)) for d in range(4)], fx.Int32
        )
        out.append((bw, live.select(code, fx.Int32(UNIT_SCALE))))
    return out


def tile_row(c, first, count):
    """A lane's pick row of the tile at first (``count`` picks from it) and
    whether its column is live; a dead column (past the picks, or a slot past
    U: count 0) reads row first."""
    col = c["lane"] % 16
    return first + fx.max(fx.min(col, fx.min(count, TG) - 1), 0), col < count


@traced
def down_tile(c, rgd, ops, j, first, count, bs):
    """Batch slot j's w2 rows 16 rgd .. (``ops``) against its picks first .. first +
    min(count, 16) (their B operands ``bs``: ``inter_operands``), a pick a
    column -> bf16(w acc) of each pick into LDS ``contrib`` (lane: rows
    4 (lane / 16) + i of pick column lane % 16)."""
    lane, rt = c["lane"], c["rt"]
    nbs = DOWN_BATCH * W2_STEPS
    row, live = tile_row(c, first, count)
    words = ops[nbs + 2 * j : nbs + 2 * j + 2]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for st in range_constexpr(W2_STEPS):
        acc = mfma_scaled(
            ops[j * W2_STEPS + st],
            bs[st][0],
            acc,
            w2_code(words, rgd, st),
            bs[st][1],
            FP4,
            FP4,
            0,
            0,
        )
    if live:
        pid = fx.ptr_load(rt["order"] + row)
        t, k = pid // TOPK, pid % TOPK
        wt = fx.ptr_load(rt["route"] + (t * 2 * TOPK + TOPK + k)).bitcast(fx.Float32)
        for i in range_constexpr(4):
            fx.ptr_store(
                bf16_round(acc[i] * wt),
                c["contrib"] + (pid * ROWS + 4 * (lane // 16) + i),
            )


def down_loads(c, rgd, nu, b0):
    """The w2 tiles and scale codes of a wave's batch from slot b0 (slots b0,
    b0 + 8, ...; past |U| the last slot's): flat, ``stage_down`` unpacks."""
    lane = c["lane"]
    r_w = rsrc(c["args"]["w2"])
    rt = c["rt"]
    slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(DOWN_BATCH)]
    es = [uniform(fx.ptr_load(rt["expert"] + sl)) for sl in slots]
    av = [
        fp4_tile_load(r_w, es[j] * (LAT * RI // 8), rgd, RI, st, lane * 4)
        for j in range(DOWN_BATCH)
        for st in range(W2_STEPS)
    ]
    sw = [w for j in range(DOWN_BATCH) for w in w2_scale_words(c, es[j], rgd, lane)]
    return av + sw


@traced
def stage_down(c, rgd, nu, ops0):
    """Latent rows 16 rgd ..: every slot's w2 against its picks (wave w the slots
    w, w + 8, ...; a batch of DOWN_BATCH in flight, the next issued under this
    one's MFMAs; ``ops0`` the first, issued before any ug flag was waited for;
    a batch waits only for its own slots' ug tasks) -> bf16(w acc) a (token,
    k) in LDS -> folded in top-k order with a bf16 rounding an add -> every
    rank's LATR."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * DOWN_BATCH
        slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(DOWN_BATCH)]
        # the batch's ug flags, lanes l, l + 64, .. (DOWN_BATCH x 12 of them)
        nf = DOWN_BATCH * UG_GROUPS
        specs = []
        for q in range_constexpr(-(-nf // 64)):
            f = fx.min(lane + 64 * q, nf - 1)
            mine = slots[0]
            for j in range_constexpr(1, DOWN_BATCH):
                mine = (f // UG_GROUPS == j).select(slots[j], mine)
            specs.append((c["ugf"], mine * UG_GROUPS + f % UG_GROUPS, 1))
        c["poll"](specs)
        firsts, counts, bss = [], [], []
        for j in range_constexpr(DOWN_BATCH):
            first = fx.ptr_load(rt["first"] + slots[j])
            count = fx.ptr_load(rt["first"] + slots[j] + 1) - first
            # a slot past U repeats the last: its picks are written once
            firsts.append(first)
            counts.append(((b0 + WAVES * j) < nu).select(count, fx.Int32(0)))
        # every slot's B operands, then the next batch's weights: a B load
        # issued after them waited for them (vmcnt counts in order)
        for j in range_constexpr(DOWN_BATCH):
            row, live = tile_row(c, firsts[j], counts[j])
            bss.append(inter_operands(c, row, live))
        ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        for j in range_constexpr(DOWN_BATCH):
            down_tile(c, rgd, ops, j, firsts[j], counts[j], bss[j])
            if const_expr(s > TG):
                # past 16 picks: the next tiles against the same weights
                j0 = fx.Int32(TG)
                while j0 < counts[j]:
                    row, live = tile_row(c, firsts[j] + j0, counts[j] - j0)
                    down_tile(
                        c,
                        rgd,
                        ops,
                        j,
                        firsts[j] + j0,
                        counts[j] - j0,
                        inter_operands(c, row, live),
                    )
                    j0 = j0 + TG
        b0 = nb
        ops = ops_n
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        y0 = fx.Float32(0.0)
        y1 = fx.Float32(0.0)
        for k in range_constexpr(TOPK):
            y0 = bf16_round(
                y0 + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * ROWS + 2 * rp))
            )
            y1 = bf16_round(
                y1 + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * ROWS + 2 * rp + 1))
            )
        # to the owner of these columns' lnorm task alone (``lat_owner``)
        task = rgd // (128 // ROWS)
        for p in range_constexpr(TP):
            if lat_owner(task) == p:
                c["put_bf"](
                    c["peer"]["latr"][p],
                    group_row(c["rank"], tt, s) * LAT_OWN
                    + (rgd - lat_first(p) * (128 // ROWS)) * ROWS
                    + 2 * rp,
                    [y0, y1],
                )
    gpu.barrier()


# ---------------------------------------------------------------- shared down
@traced
def stage_h(c, t):
    """Token t's shared h = SiTU(g, u) (vLLM's situ_and_mul, bf16) of SGU, once
    -> H; a thread a column pair, its polls in one batch."""
    tid = c["tid"]
    n = SI // 2
    if tid < n:
        got = c["poll"](
            [
                (c["sgu"], (t * 2 * SI) // 2 + tid, 1),
                (c["sgu"], (t * 2 * SI + SI) // 2 + tid, 1),
            ]
        )
        g0, g1 = bf_lo(got[0][0]), bf_hi(got[0][0])
        u0, u1 = bf_lo(got[1][0]), bf_hi(got[1][0])
        c["put_bf"](c["h"], t * SI + 2 * tid, [situ_vllm(g0, u0), situ_vllm(g1, u1)])
    gpu.barrier()


@traced
def run_h(c, bid):
    """The shared SiTU of every token on the CTAs with no down task (the
    shared down's, ``run_sdown``): tokens bid - 224, + 32, .."""
    s, n = c["S"], BLOCKS - DOWN_TASKS
    for i in range_constexpr(-(-s // n)):
        if (bid >= DOWN_TASKS) & (bid - DOWN_TASKS + n * i < s):
            stage_h(c, bid - DOWN_TASKS + n * i)


@traced
def load_h(c):
    """H -> LDS rows (for the shared down GEMV), once a CTA."""
    s, tid = c["S"], c["tid"]
    hrow = SI + gemv.LDS_PAD
    n = s * SI // 2
    per = (n + THREADS - 1) // THREADS
    got = c["poll"]([(c["h"], fx.min(tid + THREADS * i, n - 1), 1) for i in range(per)])
    for i in range_constexpr(per):
        pidx = tid + THREADS * i
        t = fx.min(pidx, n - 1) // (SI // 2)
        cp = fx.min(pidx, n - 1) % (SI // 2)
        if pidx < n:
            fx.ptr_store(got[i][0], c["xl"] + (t * hrow + 2 * cp) // 2)
    gpu.barrier()


@traced
def stage_sdown(c, j, ops):
    """Shared down rows 16 j .. (K 768) against h in LDS (``load_h``), a token
    group at a time on the same weights: bf16 partial -> SHP (this rank's up
    columns) or every rank's FINAL; then SHRDY."""
    tid = c["tid"]
    lo = c["rank"] * UP_TASKS
    mine = (j >= lo) & (j < lo + UP_TASKS)
    for g in range_constexpr(groups(c["S"])):
        if const_expr(g):
            gpu.barrier()  # the previous group's readers of ``red``
        sdown_rows(group_view(c, g), j, ops, TG * g)
    publish(
        c["put"],
        c["shrdy"],
        fx.max(fx.min(j - lo, UP_TASKS - 1), 0),
        fx.Int32(1),
        (tid == 0) & mine,
    )
    gpu.barrier()


@traced
def sdown_rows(c, j, ops, t0):
    """``stage_sdown`` for one token group: h rows t0 .. of LDS."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    hrow = SI + gemv.LDS_PAD
    acc = gemv.mfmas(lane, wave, c["xl"] + t0 * hrow // 2, hrow, SI, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    lo = c["rank"] * UP_TASKS
    mine = (j >= lo) & (j < lo + UP_TASKS)
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        if mine:
            w = (
                fx.Vector.from_elements([v0, v1], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
            bo.buffer_store(
                w,
                rsrc(c["shp"]),
                (tt * UP_N + (j - lo) * ROWS + 2 * rp) // 2,
                cache_modifier=CM_DEV,
            )
        else:
            # to the columns' owner alone (``final_owner``)
            for p in range_constexpr(TP):
                if j // UP_TASKS == p:
                    c["put_bf"](
                        c["peer"]["fin"][p],
                        (c["rank"] * s + tt) * UP_N
                        + (j - p * UP_TASKS) * ROWS
                        + 2 * rp,
                        [v0, v1],
                    )


SD0 = LN0 + LN_TASKS  # 84: the shared down's CTAs, past lnorm / up's


@traced
def claim_sdown(c):
    """Claim the next shared-down task from the launch's sdown queue
    (``queue_word``) into LDS ``scan[0]`` (tid 0; read it past a barrier)."""
    if c["tid"] == 0:
        k = queue.take(queue_word(c, "sdown"))
        fx.ptr_store(fx.Int32(k), c["scan"])


@traced
def run_sdown(c, bid, w_sd):
    """The 448 shared-down tasks from the launch's sdown queue, on CTAs 84 ..
    (past lnorm / up's) as each is free: those with no down task (224 ..)
    right after ug, the rest after their down. Task k is row group (k + 56
    rank) % 448, so the first 56 claimed -- by the CTAs free first -- are the
    rank's up columns, the ones up waits on (SHRDY); only the final reduce
    waits on the rest. A task's result is the same whichever CTA runs it.

    One claim site, at the top of the loop's body: a claim before the loop
    and one at its end were merged into one block entered from both, an
    irreducible loop whose waves split at its barriers and hung."""
    live = bid >= SD0
    if live:
        load_h(c)
    lo = c["rank"] * UP_TASKS
    more = live
    while more:
        gpu.barrier()  # every thread read the previous claim
        claim_sdown(c)
        gpu.barrier()
        k = fx.ptr_load(c["scan"])
        if k < SD_TASKS:
            j = (k + lo) % SD_TASKS
            stage_sdown(
                c,
                j,
                gemv.loads(c["lane"], c["wave"], w_sd, j * ROWS, SI, paired=True),
            )
        more = k < SD_TASKS


# ---------------------------------------------------------------- lnorm / up / reduce
def ln_halves(s: int) -> int:
    """lnorm's token halves a task: two (on two CTAs, ``layer``) once a task
    takes three passes or more (S > 16), else one."""
    return LN_HALVES_MAX if passes(s * 64) >= 3 else 1


def ln_passes(npass: int, halves: int, h: int) -> list:
    """Half h's passes of a task's ``npass``: ceil(npass / halves) a half, the
    last half the rest."""
    cut = -(-npass // halves)
    return list(range(npass))[h * cut : (h + 1) * cut]


def ln_halves_of(s: int, g: int) -> range:
    """The lnorm halves holding token group g's rows (a pass: a wave a token,
    ``WAVES`` tokens): the group's up waits for those alone."""
    halves, npass = ln_halves(s), passes(s * 64)
    first, last = TG * g, min(TG * (g + 1), s) - 1
    hs = [
        h
        for h in range(halves)
        if (ps := ln_passes(npass, halves, h))
        and ps[0] * WAVES <= last
        and (ps[-1] + 1) * WAVES > first
    ]
    return range(hs[0], hs[-1] + 1)


@traced
def stage_lnorm(c, j, h=0):
    """Columns 128 j ..: the ranks' latent partials summed in rank order (fp32
    -> bf16; on their owner, ``lat_owner``, which sends the sum to every
    rank's LATG -- the others read it there), every task's sum of squares ->
    rstd -> bf16(acc rstd w) -> LN; then LNRDY (j, h).

    One wave a token (64 column pairs): a pass of 8 tokens, ``s / 8`` passes;
    half h of them (``ln_passes``: all of them for one half, ``ln_halves``).
    Every pass's sums of squares go out before any is polled, so the step
    takes one task-to-task hop, not one a pass. The values wait in registers
    across it.
    """
    s, tid, lane, red = c["S"], c["tid"], c["lane"], c["red"]
    a = c["args"]
    n = s * 64
    owner = lat_owner(j) == c["rank"]
    base = lat_first(c["rank"]) * 128
    # this half's passes (``ln_passes``)
    ps = ln_passes(passes(n), ln_halves(s), h)
    # a pass's two norm weights, one word, issued before any poll
    lnw = [
        fx.Int32(
            bo.buffer_load(
                rsrc(a["ln_w"]),
                (j * 128 + 2 * ((tid + THREADS * i) % 64)) // 2,
                vec_width=1,
                dtype=T.i32,
            )
        )
        for i in ps
    ]
    held = []
    for k in range_constexpr(len(ps)):
        e = tid + THREADS * ps[k]
        mine = e < n
        t = fx.min(e // 64, s - 1)
        col = j * 128 + 2 * (e % 64)
        # defined before the traced ``if``: a name its branches bind is not
        # seen past it
        lo, hi = fx.Float32(0.0), fx.Float32(0.0)
        if owner:
            # the owner: the ranks' partials in rank order -> every rank
            lo, hi = sum_partials(
                c["poll"],
                c["peer"]["latr_own"],
                lambda src, t=t, col=col: (group_row(src, t, s) * LAT_OWN + col - base)
                // 2,
                TP,
            )
            lo, hi = bf16_round(lo), bf16_round(hi)
            for p in range_constexpr(TP):
                c["put_bf"](c["peer"]["latg"][p], t * LAT + col, [lo, hi])
        else:
            w = c["poll"]([(c["peer"]["latg_own"], (t * LAT + col) // 2, 1)])[0][0]
            lo, hi = bf2_f32(w)
        sq = wave_sum(lo * lo + hi * hi)
        if mine & (lane == 0):
            c["put"](c["lmsq"], j * s + t, sq)
        held.append((mine, t, col, lo, hi))
    stamp(c["tl_on"], c["tls"], tid, 15)
    for k in range_constexpr(len(ps)):
        mine, t = held[k][0], held[k][1]
        if mine & (lane < LN_TASKS):
            v = c["poll"]([(c["lmsq"], lane * s + t, 1)])[0][0]
            fx.ptr_store(f32_of(v), red + (t * 32 + lane))
    gpu.barrier()
    for k in range_constexpr(len(ps)):
        mine, t, col, lo, hi = held[k]
        tot = fx.ptr_load(red + t * 32)
        for q in range_constexpr(1, LN_TASKS):
            tot = tot + fx.ptr_load(red + (t * 32 + q))
        r = hw_rsq(tot * fx.Float32(1.0 / LAT) + fx.Float32(c["ln_eps"]))
        w0, w1 = bf2_f32(lnw[k])
        y = (
            fx.Vector.from_elements([lo * r * w0, hi * r * w1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        if mine:
            bo.buffer_store(
                y, rsrc(c["ln"]), (t * LAT + col) // 2, cache_modifier=CM_DEV
            )
    publish(c["put"], c["lnrdy"], h * LN_TASKS + j, fx.Int32(1), tid == 0)
    gpu.barrier()


@traced
def run_up(c, bid):
    """Up task j of token group g on CTA 56 g + j: a step's groups side by
    side, on CTAs that wait for the latent norm anyway."""
    for g in range_constexpr(groups(c["S"])):
        if (bid >= UP_TASKS * g) & (bid < UP_TASKS * (g + 1)):
            stage_up(group_view(c, g), bid - UP_TASKS * g, ln_halves_of(c["S"], g))


@traced
def stage_up(c, j, hs):
    """This rank's up rows 16 j .. of its shard (hidden rows 896 rank ..) for
    the (group's) tokens, once the lnorm halves ``hs`` holding them are in:
    LN GEMV + SHP -> bf16 -> ``final_owner``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    ops = gemv.loads(lane, wave, c["args"]["w_up"], j * ROWS, LAT, paired=True)
    # every lnorm task's halves of these tokens (LNRDY h LN_TASKS + task)
    if tid < LN_TASKS * len(hs):
        c["poll"]([(c["lnrdy"], hs[0] * LN_TASKS + tid, 1)])
    if tid == 0:
        c["poll"]([(c["shrdy"], j, 1)])
    gpu.barrier()
    lrow = LAT + gemv.LDS_PAD
    gemv.dev_rows_to_lds(tid, c["ln"], LAT, s, c["xl"], lrow)
    acc = gemv.mfmas(lane, wave, c["xl"], lrow, LAT, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (ROWS // 2) * s:
        rp = tid % (ROWS // 2)
        tt = tid // (ROWS // 2)
        w = fx.Int32(
            bo.buffer_load(
                rsrc(c["shp"]),
                (tt * UP_N + j * ROWS + 2 * rp) // 2,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        v0 = bf16_round(bf_lo(w) + row_sum(red, 2 * rp, tt))
        v1 = bf16_round(bf_hi(w) + row_sum(red, 2 * rp + 1, tt))
        final_owner(c, tt, j * ROWS + 2 * rp, v0, v1)
    gpu.barrier()


def final_owner(c, t, col, v0, v1):
    """The owner's half of the layer output's all-reduce, for its column pair
    ``col`` (of its UP_N) of token t: its own partial (``v0``, ``v1``) beside
    the peers' (FIN), summed in rank order -> bf16 -> this rank's ``out`` and
    every rank's FING
    (the bits as one word)."""
    s = c["S"]
    c["put_bf"](c["peer"]["fin_own"], (c["rank"] * s + t) * UP_N + col, [v0, v1])
    lo, hi = sum_partials(
        c["poll"],
        c["peer"]["fin_own"],
        lambda src: ((src * s + t) * UP_N + col) // 2,
        TP,
    )
    w = (
        fx.Vector.from_elements([lo, hi], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    at = (t * HIDDEN + c["rank"] * UP_N + col) // 2
    bo.buffer_store(w, rsrc(c["args"]["out"]), at)
    for p in range_constexpr(TP):
        c["put_words"](c["peer"]["fing"][p], at, [w])


@traced
def stage_reduce(c, j):
    """Columns 128 j .. of every token group (``reduce_rows``)."""
    for g in range_constexpr(groups(c["S"])):
        reduce_rows(group_view(c, g), j)


@traced
def reduce_rows(c, j):
    """Columns 128 j ..: another rank's (their owner's, ``final_owner``) bf16
    sums from FING -> ``out``; this rank's own columns its up wrote.

    ``s * 64`` work items against ``THREADS`` threads: one pass covers them only
    up to 8 rows, so the pass is strided.
    """
    s, tid = c["S"], c["tid"]
    n = s * 64
    if j // (UP_N // 128) != c["rank"]:
        for i in range_constexpr((n + THREADS - 1) // THREADS):
            e = tid + THREADS * i
            if e < n:
                at = ((e // 64) * HIDDEN + j * 128 + 2 * (e % 64)) // 2
                w = c["poll"]([(c["peer"]["fing_own"], at, 1)])[0][0]
                bo.buffer_store(w, rsrc(c["args"]["out"]), at)
    gpu.barrier()


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def route_lds(s: int):
    """LDS of the route table (``load_route``) for an s-row step; ug and down
    read it. The bitmaps it is built from: ``mask_lds``."""

    @fx.struct
    class RouteLds:
        route: fx.Array[fx.Int32, s * 2 * TOPK, 16]
        slot: fx.Array[fx.Int32, E, 16]
        expert: fx.Array[fx.Int32, max_u(s), 16]
        first: fx.Array[fx.Int32, max_u(s) + 1, 16]
        order: fx.Array[fx.Int32, s * TOPK, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    return RouteLds


def mask_lds(s: int):
    """The experts' token bitmaps: a member of a build's stage union, read by
    ``load_route`` alone (before ``load_latq`` fills the ug member)."""

    @fx.struct
    class MaskLds:
        mask: fx.Array[fx.Int32, E * mask_words(s), 16]

    return MaskLds


def ug_lds(s: int):
    @fx.struct
    class UgLds:
        latq: fx.Array[fx.Int32, s * LAT // 8, 16]
        latsc: fx.Array[fx.Int32, s * LAT // 128, 16]
        av: fx.Array[fx.Float32, THREADS, 16]

    return UgLds


def down_lds(s: int):
    @fx.struct
    class DownLds:
        contrib: fx.Array[fx.Float32, s * TOPK * ROWS, 16]

    return DownLds


def route_table(rl, ml):
    """The route table's LDS pointers (``c["rt"]``)."""
    return {
        "route": rl.route.ptr,
        "slot": rl.slot.ptr,
        "expert": rl.expert.ptr,
        "first": rl.first.ptr,
        "order": rl.order.ptr,
        "mask": ml.mask.ptr,
    }


def build(key: MoeBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    # the standalone K2b launch; the shipped path is ``layer.build``'s fused K2
    assert 1 <= s <= 16
    lay = scratch_layout(key)
    plain_lay = plain_layout(key)
    play = peer_layout(s)
    RouteLds = route_lds(s)

    @fx.struct
    class EarlyLds:
        xl: fx.Array[fx.Int32, s * (HIDDEN + gemv.LDS_PAD) // 2, 16]

    @fx.union
    class StageLds:
        early: EarlyLds
        ug: ug_lds(s)
        down: down_lds(s)
        rmask: mask_lds(s)
        rsel: RouteSelLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = kernel_symbol("k3_mono_moe", **symbol_params(key))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def moe(
        x: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        plain: Int64,
        queue: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        diag: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        rl = alloc.allocate(RouteLds).peek()
        st_lds = alloc.allocate(StageLds)
        el, ul, dl = st_lds.early.peek(), st_lds.ug.peek(), st_lds.down.peek()
        rsel = st_lds.rsel.peek()
        if const_expr(key.diag):
            mb = Mailbox(layer, diag, region_ids(REGIONS))
        else:
            mb = Mailbox(layer)
        bases = peer_bases(peers, TP)
        own = load_ptr64(peers, rank)
        peer = {}
        for region in PEER_REGIONS:
            off = play[region][0]
            peer[region] = [preg(b, off, region) for b in bases]
            peer[region + "_own"] = preg(own, off, region)
        c = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "ln_eps": key.eps,
            "put": mb.put,
            "put_bf": mb.put_bf,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "queue": queue,
            "red": lds.red.ptr,
            "peer": peer,
            "xl": el.xl.ptr,
            "rcand": rsel.cand.ptr,
            "rsig": rsel.sig.ptr,
            "latq": ul.latq.ptr,
            "latsc": ul.latsc.ptr,
            "av": ul.av.ptr,
            "contrib": dl.contrib.ptr,
            "scan": rl.scan.ptr,
            "rt": route_table(rl, st_lds.rmask.peek()),
            "inter": plain + fx.Int64(plain_lay["inter"][0]),
            "isc": plain + fx.Int64(plain_lay["isc"][0]),
            "ln": plain + fx.Int64(plain_lay["ln"][0]),
            "shp": plain + fx.Int64(plain_lay["shp"][0]),
            "args": {
                "x": x,
                "w_gate": w_gate,
                "bias": bias,
                "w_ld": w_ld,
                "w_sgu": w_sgu,
                "w_sd": w_sd,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "ln_w": ln_w,
                "w_up": w_up,
                "out": out,
            },
        }
        for region in (
            "sgu",
            "h",
            "rpart",
            "lpart",
            "route",
            "ugf",
            "lmsq",
            "lnrdy",
            "shrdy",
        ):
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        stamp_begin(on, tls, tid, TL_POINTS)
        # compile-time gate outside, traced condition inside
        if const_expr(key.stop > 1):  # noqa: SIM102
            if bid < EARLY:
                stage_early(c, bid)
        stamp(on, tls, tid, 1)
        if const_expr(key.stop <= 2):
            stamp_flush(on, tls, tl, tid, bid, TL_POINTS)
            return
        if (bid >= LATE0) & (bid < LATE0 + s):
            stage_route(c, bid - LATE0)
        if (bid >= H_CTA) & (bid < H_CTA + s):
            stage_h(c, bid - H_CTA)
        stamp(on, tls, tid, 2)
        if const_expr(key.stop <= 3):
            return
        nu = load_route(c)
        if const_expr(key.stop == 35):
            # debug: the route table, |U| and U into ``out`` (as int32), CTA 0
            if bid == 0:
                n = s * 2 * TOPK
                if tid < n:
                    bo.buffer_store(fx.ptr_load(c["rt"]["route"] + tid), rsrc(out), tid)
                if tid < max_u(s):
                    bo.buffer_store(
                        fx.ptr_load(c["rt"]["expert"] + tid), rsrc(out), n + 1 + tid
                    )
                if tid == 0:
                    bo.buffer_store(nu, rsrc(out), n)
            return
        load_latq(c)
        if const_expr(key.stop == 36):
            return
        run_ug(c, bid, nu)
        stamp(on, tls, tid, 3)
        if const_expr(key.stop <= 4):
            stamp_flush(on, tls, tl, tid, bid, TL_POINTS)
            return
        if bid < DOWN_TASKS:
            # the first w2 batch in flight before any ug flag is waited for
            dops = down_loads(c, bid, nu, fx.min(c["wave"], nu - 1))
            stage_down(c, bid, nu, dops)
        run_sdown(c, bid, w_sd)
        stamp(on, tls, tid, 4)
        if const_expr(key.stop <= 5):
            return
        if (bid >= LN0) & (bid < LN0 + LN_TASKS):
            stage_lnorm(c, bid - LN0)
        stamp(on, tls, tid, 5)
        if const_expr(key.stop <= 6):
            return
        run_up(c, bid)
        stamp(on, tls, tid, 6)
        if const_expr(key.reduce):  # noqa: SIM102
            if (bid >= RED0) & (bid < RED0 + RED_TASKS):
                stage_reduce(c, bid - RED0)
        stamp(on, tls, tid, 7)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        x: Int64,
        w_gate: Int64,
        bias: Int64,
        w_ld: Int64,
        w_sgu: Int64,
        w_sd: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        ln_w: Int64,
        w_up: Int64,
        out: Int64,
        scratch: Int64,
        plain: Int64,
        queue: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        diag: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        moe(
            x,
            w_gate,
            bias,
            w_ld,
            w_sgu,
            w_sd,
            w13,
            w13s,
            w2,
            w2s,
            ln_w,
            w_up,
            out,
            scratch,
            plain,
            queue,
            peers,
            rank,
            layer,
            diag,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(moe, launch)
    _BUILDS[key] = launch
    return launch
