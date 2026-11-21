# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 MLA back half in one resident launch: a full-attention layer from
the decode attention's partials (``o [rows, heads, 512]`` BF16 and
``lse [rows, heads]`` f32 over this rank's KV shard, the DCP group's heads)
to the attention output (``[rows, HIDDEN]`` BF16, summed over the TP ranks),
in place of the ``W_UV`` batched GEMM, aiter's DCP merge
(``dcp_merge_ll_gfx1250``), the output gate (``fused_sigmoid_mul``) and
``k2_oproj_ar_gfx1250``. Each step is that kernel's arithmetic, so the output
is the sequence's bit for bit.

``mla_back_layout``: waves 0..2 of every CTA project one 16-row block of one
head's ``W_UV`` each (FP8, one per-tensor scale; weights and token rows
loaded straight to registers) as the gluon ``batched_gemm_a16w8`` does: the
FP8 weights widened to BF16, sixteen K32 WMMAs chained from zero,
``bf16(acc * scale)``. A lane holds eight values of one token: it pushes them
as two 16 B lines -- four BF16, the row's LSE, the launch's epoch -- to the
rank owning the head (its own heads' to itself), stored at system scope, once
every token tile is projected.

Waves 3..7 merge: a wave takes one ``(token, local head)`` row, lane ``j``
polling line ``j`` of every shard until all carry the epoch, then
``dcp_merge_ll_gfx1250``'s combine in its order of operations: a non-finite
LSE counts as -inf; ``m`` the largest (0 if none), ``g = log(sum_s exp(lse_s
- m)) + m`` summed in rank order; ``out = sum_s exp(lse_s - g) * o_s`` as a
chain of fused multiply-adds from +0 in rank order, a shard whose weight is 0
skipped; rounded to BF16 once. ``exp`` and ``log`` are the device libm's
(``_expf`` / ``_logf``: the instructions the merge kernel compiles to). The
gate follows in the same lanes -- ``bf16(x * (1 / (1 + exp2(-(g *
log2e)))))``, AITER's ``_sigmoid_exp2`` with a true division -- into ``core``
(``[rows, proj]``, o_proj's input) at device scope, then the row's flag.

The 224 o_proj CTAs are ``k2_oproj_ar_gfx1250``'s one-shot build: their FP8
weight slabs are copied to LDS before any of the above (``gemv_tdm.project``'s
prologue: the copies fly while the merge waits for the fabric), each waits
for every row's flag, then projects ``core`` and all-reduces in one more hop.
CTAs 224..255 only project ``W_UV`` and merge.

``dcp`` 1: no merge (a rank's own heads, as without DCP); the lines pass
through the rank's own bank. ``bound``: polls give up past it (the harness's
builds; None waits like the kernels it replaces).
``ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER`` (off by default): the merge's and
the all-reduce's polls issue a round's loads together and await them once,
each where its group has more than one rank.
"""

import struct
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250, tdm_timeline
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_gfx1250 as absorb
from atom.models.kimi_k3.mono.kernels.stages import mla_back_layout as lay
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    ROWS,
    TDM_FP8_PAIR_DEPTH,
    tdm_lds_words,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    hw_exp2,
    hw_log2,
    ld_dev,
    load_ptr64,
    mma_bf16,
    rsrc,
    spin_pause,
    st_dev,
    traced,
    uniform,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.utils import envs

THREADS = lay.THREADS
WAVES = lay.WAVES
LANES = lay.LANES
ACC = 8
CTA_PAIRS = oar.CTA_PAIRS
LINE_TOKENS = oar.LINE_TOKENS
ACCUMULATOR_WORDS = oar.ACCUMULATOR_WORDS
assert oar.LINE_WORDS == lay.LINE_WORDS and oar.CTAS == lay.OPROJ_CTAS

MLA_BACK_ABI = KernelAbi(
    (
        "o",
        "lse",
        "w_uv",
        "s_uv",
        "gate",
        "w_o",
        "w_o_scale",
        "out",
        "core",
        "flags",
        "peers",
        "merge_peers",
        "epochs",
        "rank",
        "dcp_rank",
    )
)
MLA_BACK_PROFILE_ABI = KernelAbi((*MLA_BACK_ABI.names, "stamps"))
# a profile build's i32 stamps a CTA, the low half of the 100 MHz clock:
# thread 0's start, its W_UV block pushed, (wave 3's) merged row's lines seen
# and its gated row acknowledged, every row's flag seen, the columns staged,
# the ranks' partials seen, the end
STAMPS = 8
STAMP_START, STAMP_UV_PUSHED, STAMP_LINES_SEEN, STAMP_CORE_STORED = 0, 1, 2, 3
STAMP_CORE_SEEN, STAMP_STAGED, STAMP_SUMMED, STAMP_END = 4, 5, 6, 7
# the merging thread that stamps: lane 0 of the CTA's first merging wave
MERGE_THREAD = lay.WUV_WAVES * LANES


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# the device libm's expf / logf constants (ROCm OCML), as dcp_merge_ll_gfx1250
# compiles them: log2(e) and ln(2) each as a high and a low float
LOG2E_HI = _f32(0x3FB8AA3B)
LOG2E_LO = _f32(0x32A5705F)
LN2_HI = _f32(0x3F317217)
LN2_LO = _f32(0x3377D1CF)
EXP_UNDERFLOW = _f32(0xC2CE8ED0)
EXP_OVERFLOW = _f32(0x42B17218)
# +-inf as the i32 holding their bits
POS_INF_BITS = 0x7F800000
NEG_INF_BITS = -0x00800000
# AITER's _sigmoid_exp2: f32(1.44269504089)
GATE_LOG2E = _f32(0x3FB8AA3B)


@dataclass(frozen=True)
class MlaBackKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    plan: lay.BackPlan
    abi: KernelAbi = MLA_BACK_ABI
    launch_count: int = 1


def _inf(negative: bool = False):
    """+-inf from its bits (no float literal for it crosses the DSL)."""
    return fx.Int32(NEG_INF_BITS if negative else POS_INF_BITS).bitcast(fx.Float32)


def _fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def _expf(x):
    """The device libm's ``expf`` (OCML): ``2 ** (x * log2e)`` with the
    product in double-float, the hardware ``2 ** fraction`` scaled by the
    integer part, then the range fixups -- instruction for instruction what
    ``dcp_merge_ll_gfx1250`` runs."""
    ph = x * fx.Float32(LOG2E_HI)
    pl = _fma(x, fx.Float32(LOG2E_HI), -ph)
    whole = fx.Float32(fmath.roundeven(ph))
    pl = _fma(x, fx.Float32(LOG2E_LO), pl)
    r = hw_exp2((ph - whole) + pl)
    exponent = fx.Int32(
        llvm.call_intrinsic(
            T.i32, "llvm.fptosi.sat.i32.f32", [whole.ir_value()], [], []
        )
    )
    r = fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.ldexp.f32.i32", [r.ir_value(), exponent.ir_value()], [], []
        )
    )
    r = (x < fx.Float32(EXP_UNDERFLOW)).select(fx.Float32(0.0), r)
    return (x > fx.Float32(EXP_OVERFLOW)).select(_inf(), r)


def _logf(x):
    """The device libm's ``logf`` (OCML): the hardware ``log2`` times ln 2 in
    double-float; a non-finite ``log2`` passes through."""
    y = hw_log2(x)
    ph = y * fx.Float32(LN2_HI)
    pl = _fma(y, fx.Float32(LN2_HI), -ph)
    pl = _fma(y, fx.Float32(LN2_LO), pl)
    return (fx.Float32(fmath.absf(y)) < _inf()).select(ph + pl, y)


def _sigmoid(g):
    """AITER's ``_sigmoid_exp2`` as Triton compiles it: the hardware ``2 **
    x`` and a correctly rounded division."""
    return fx.Float32(1.0) / (hw_exp2(-(g * fx.Float32(GATE_LOG2E))) + fx.Float32(1.0))


def _weights(lse):
    """A merged row's weight a shard, ``exp(lse_s - g)`` (0 where it is not a
    number): ``dcp_merge_ll_gfx1250``'s, in its order."""
    neg_inf = _inf(negative=True)
    clean = [
        ((v != v) | (v == _inf())).select(neg_inf, v) for v in lse  # noqa: PLR0124
    ]
    top = neg_inf
    for v in clean:
        top = fx.max(top, v)
    top = (top == neg_inf).select(fx.Float32(0.0), top)
    total = None
    for v in clean:
        term = _expf(v - top)
        total = term if total is None else total + term
    g = _logf(total) + top
    out = []
    for v in clean:
        f = _expf(v - g)
        out.append((f != f).select(fx.Float32(0.0), f))  # noqa: PLR0124
    return out


@traced
def _poll(
    mine, words, epoch, bound, modifier=ar.POLL_VOLATILE, marks=None, bid=None,
    tid=None,
):  # fmt: skip
    """The lines at ``words`` of this rank's bank once all carry ``epoch``
    (re-loaded together while any is stale), as their four words.
    ``modifier``: ``ar.POLL_VOLATILE`` (each load waited out before the
    next) or ``ar.POLL_TOGETHER``. ``marks`` (tool builds, with ``bound``):
    a poll that gives up marks ``marks[ar.GAVE_UP_WORD]``, a wave's first
    lane its polls, where any, in ``marks[ar.POLLS_WORD + CTA]``."""
    lines = oar._load_lines(mine, words, modifier)
    if const_expr(bound is None):
        while oar._stale(lines, epoch):
            spin_pause()
            lines = oar._load_lines(mine, words, modifier)
    else:
        polls = fx.Int32(0)
        while oar._stale(lines, epoch) & (polls < bound):
            spin_pause()
            lines = oar._load_lines(mine, words, modifier)
            polls = polls + 1
        if const_expr(marks is not None):
            if polls >= bound:
                st_dev(fx.Int32(ar.GAVE_UP) | bid, marks, ar.GAVE_UP_WORD)
            if (tid % LANES == 0) & (polls > 0):
                st_dev(polls, marks, ar.POLLS_WORD + bid)
    return [[fx.Int32(line[j]) for j in range(lay.LINE_WORDS)] for line in lines]


@traced
def _project_uv(c, slot, lane):
    """This wave's W_UV block ``slot`` (``head * 8 + block``) over every
    token tile, each lane's eight values pushed to the head's owner as two
    lines."""
    plan = c["plan"]
    head = slot // lay.N_BLOCKS
    block = slot % lay.N_BLOCKS
    lane_group = lane // ROWS
    n = block * ROWS + lane % ROWS
    weights = [
        [
            absorb._words(
                c["w_uv"], lay.uv_weight_word(head, n, s, half, lane_group), 2
            )
            for half in range(2)
        ]
        for s in range(lay.K_STEPS)
    ]
    scale = fx.Int32(
        bo.buffer_load(rsrc(c["s_uv"]), 0, vec_width=1, dtype=T.i32, is_scalar=True)
    ).bitcast(fx.Float32)
    owner = head // plan.local_heads
    local = head % plan.local_heads
    base = load_ptr64(c["merge_peers"], owner)
    one = absorb._splat(fx.Int32(absorb.E8M0_ONE))
    # Every tile's rows are projected before any is pushed: a wave's loads
    # issued after its system-scope stores wait for those stores (about 3 us
    # a tile, measured), so the pushes go out behind the last tile's loads.
    projected = []
    for tile in range_constexpr(plan.tiles):
        token = fx.min(fx.Int32(tile * ROWS) + lane % ROWS, fx.Int32(plan.rows - 1))
        operands = [
            [
                absorb._words(
                    c["o"],
                    lay.uv_token_word(plan, token, head, s, half, lane_group),
                    4,
                )
                for half in range(2)
            ]
            for s in range(lay.K_STEPS)
        ]
        lse = fx.Int32(
            bo.buffer_load(
                rsrc(c["lse"]), token * plan.heads + head, vec_width=1, dtype=T.i32
            )
        )
        acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
        for s in range_constexpr(lay.K_STEPS):
            acc = mma_bf16(
                absorb._fp8_operand(weights[s], one),
                absorb._bf16_operand(operands[s]),
                acc,
                GFX1250,
            )
        words = [
            absorb._pack(acc[2 * i] * scale, acc[2 * i + 1] * scale)
            for i in range_constexpr(ACC // 2)
        ]
        out_token = tile * ROWS + lane % ROWS
        projected.append((token, out_token < plan.rows, words, lse))
    if const_expr(c["late"]):  # noqa: SIM102
        # TEST ONLY: the odd CTAs' lines land while the merges poll them
        if c["bid"] % 2 == 1:
            ar._sleeps(c["late"])
    for token, live, words, lse in projected:
        for half in range_constexpr(2):
            word = c["bank"] + lay.uv_line_word(
                plan, c["dcp_rank"], token, local, block, lane_group, half
            )
            oar._push_line(
                live,
                base + fx.Int64(word) * 4,
                [words[2 * half], words[2 * half + 1], lse],
                c["epoch"],
            )
    c["stamp"](STAMP_UV_PUSHED)


@traced
def _merge_row(c, row, lane):
    """Merged row ``row`` (wave-uniform) on this wave: lane ``j`` its line
    ``j`` of every shard, combined by their LSE, gated, into ``core``; then
    the row's flag."""
    plan = c["plan"]
    lines = _poll(
        c["merge_mine"],
        [c["bank"] + lay.merge_line_word(plan, s, row, lane) for s in range(plan.dcp)],
        c["epoch"],
        c["bound"],
        *c["merge_poll"],
    )
    c["stamp"](STAMP_LINES_SEEN, MERGE_THREAD)
    if const_expr(plan.dcp == 1):
        merged = [lines[0][0], lines[0][1]]
    else:
        factors = _weights([line[2].bitcast(fx.Float32) for line in lines])
        sums = [fx.Float32(0.0) for _ in range(lay.LINE_VALUES)]
        for s in range_constexpr(plan.dcp):
            values = [*bf2_f32(lines[s][0]), *bf2_f32(lines[s][1])]
            used = factors[s] != fx.Float32(0.0)
            sums = [
                used.select(_fma(factors[s], values[j], sums[j]), sums[j])
                for j in range(lay.LINE_VALUES)
            ]
        merged = [absorb._pack(sums[0], sums[1]), absorb._pack(sums[2], sums[3])]
    # gate and core are both [rows, proj] BF16: the same word of each
    at = lay.core_word(plan, row, lane)
    gates = absorb._words(c["gate"], at, lay.LINE_VALUES // 2)
    gated = []
    for word, gate in zip(merged, gates):
        x = bf2_f32(word)
        g = bf2_f32(gate)
        gated.append(absorb._pack(x[0] * _sigmoid(g[0]), x[1] * _sigmoid(g[1])))
    bo.buffer_store(
        fx.Vector.from_elements(gated, fx.Int32),
        rsrc(c["core"]),
        at,
        cache_modifier=CM_DEV,
    )
    # the flag behind the row: its stores acknowledged first
    rocdl.s_wait_storecnt(0)
    c["stamp"](STAMP_CORE_STORED, MERGE_THREAD)
    if lane == 0:
        st_dev(c["epoch"], c["flags"], row)


@traced
def _back(c, bid, lane, wave):
    """A CTA's share of W_UV and of the merge (``mla_back_layout``)."""
    plan = c["plan"]
    role = uniform(wave)
    if role < lay.WUV_WAVES:
        slot = uniform(bid) * lay.WUV_WAVES + role
        if slot < plan.heads * lay.N_BLOCKS:
            _project_uv(c, slot, lane)
    else:
        first = uniform(bid) * lay.MERGE_WAVES + role - lay.WUV_WAVES
        for round_index in range_constexpr(plan.merge_rounds):
            row = first + round_index * lay.MERGE_SLOTS
            if row < plan.merge_rows:
                _merge_row(c, row, lane)


def _flags(c, tid):
    plan = c["plan"]
    return [
        fx.Vector(
            ld_dev(
                c["flags"],
                tid * lay.FLAG_VECTOR + v * THREADS * lay.FLAG_VECTOR,
                lay.FLAG_VECTOR,
            )
        )
        for v in range(plan.flag_vectors)
    ]


def _flags_stale(c, tid, loaded):
    plan = c["plan"]
    stale = None
    for v, vector in enumerate(loaded):
        for i in range(lay.FLAG_VECTOR):
            index = tid * lay.FLAG_VECTOR + v * THREADS * lay.FLAG_VECTOR + i
            late = (fx.Int32(vector[i]) != c["epoch"]) & (index < plan.merge_rows)
            stale = late if stale is None else stale | late
    return stale


@traced
def _await_core(c, tid):
    """Until every merged row's flag carries the epoch: ``core`` is whole.
    Read at device scope, so no fence follows."""
    loaded = _flags(c, tid)
    if const_expr(c["bound"] is None):
        while _flags_stale(c, tid, loaded):
            rocdl.s_sleep(1)
            loaded = _flags(c, tid)
    else:
        polls = fx.Int32(0)
        while _flags_stale(c, tid, loaded) & (polls < c["bound"]):
            rocdl.s_sleep(1)
            loaded = _flags(c, tid)
            polls = polls + 1


@traced
def _push_partials(c, stage, bases, bid, item):
    """Staged line ``item`` (``[triple][pair]``) of this CTA's columns to
    every rank: ``k2_oproj_ar_gfx1250._push_oneshot``, the line placed by
    ``mla_back_layout.ar_line_word``."""
    plan = c["plan"]
    if item < oar.triples(plan.rows) * CTA_PAIRS:
        triple = item // CTA_PAIRS
        pair = item % CTA_PAIRS
        words = [
            fx.ptr_load(stage + (triple * LINE_TOKENS + j) * CTA_PAIRS + pair)
            for j in range(LINE_TOKENS)
        ]
        line = fx.Vector.from_elements([*words, c["epoch"]], fx.Int32)
        word = c["bank"] + lay.ar_line_word(
            plan, c["rank"], triple, bid * CTA_PAIRS + pair
        )
        for base in bases:
            ar._store_line_sys(base + fx.Int64(word) * 4, line)


@traced
def _reduce(c, bid, item, first=True):
    """Staged line ``item`` of this CTA's columns from every rank's partials,
    summed in rank order in fp32 and landed (``k2_oproj_ar_gfx1250``'s
    one-shot sums)."""
    plan = c["plan"]
    count = oar.triples(plan.rows)
    if item < count * CTA_PAIRS:
        triple = item // CTA_PAIRS
        column_pair = bid * CTA_PAIRS + item % CTA_PAIRS
        lines = _poll(
            c["mine"],
            [
                c["bank"] + lay.ar_line_word(plan, p, triple, column_pair)
                for p in range(plan.tp)
            ],
            c["epoch"],
            c["bound"],
            *c["ar_poll"],
        )
        if const_expr(first):
            c["stamp"](STAMP_SUMMED)
        oar._land_line(
            fx.Int32(1) == 1,
            c["out"],
            plan.rows,
            triple,
            column_pair,
            oar._sum_lines([line[:LINE_TOKENS] for line in lines]),
        )


@traced
def _oproj(
    c, gemv, red, stage, w_o, w_o_scale, depth, paired, bid, tid, lane, wave
):  # fmt: skip
    """An o_proj CTA: its weight slabs in flight, then its share of W_UV and
    of the merge, the wait for ``core``, and ``k2_oproj_ar_gfx1250``'s
    projection, staging and one-shot all-reduce of its 32 columns."""
    plan = c["plan"]
    rows, proj, tiles = plan.rows, plan.proj, plan.tiles
    bases = [load_ptr64(c["peers"], p) for p in range(plan.tp)]
    row0 = bid * oar.CTA_COLUMNS

    def prologue():
        _back(c, bid, lane, wave)
        _await_core(c, tid)
        gpu.barrier()
        c["stamp"](STAMP_CORE_SEEN)

    accumulators = gemv_tdm.project(
        gemv,
        c["core"],
        proj,
        rows,
        w_o,
        HIDDEN,
        proj,
        [row0, row0 + ROWS],
        lane,
        wave,
        w_scale=w_o_scale,
        depth=depth,
        paired=paired,
        prologue=prologue,
        x_cache_policy=CM_DEV,
    )
    for tile in range_constexpr(tiles):
        for index in range_constexpr(2):
            gemv_gfx1250.store_accumulator(
                accumulators[index][tile],
                red + (2 * tile + index) * ACCUMULATOR_WORDS,
                wave,
                lane,
            )
    gpu.barrier()
    task = tid // (ROWS * 8)
    local_token = tid % (ROWS * 8) // 8
    row_pair = tid % 8
    for tile in range_constexpr(tiles):
        slot = red + (2 * tile + task) * ACCUMULATOR_WORDS
        token = tile * ROWS + local_token
        low = bf16_round(gemv_gfx1250.row_sum(slot, 2 * row_pair, local_token, WAVES))
        high = bf16_round(
            gemv_gfx1250.row_sum(slot, 2 * row_pair + 1, local_token, WAVES)
        )
        word = (
            fx.Vector.from_elements([low, high], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        oar._store_stage(
            local_token < min(ROWS, rows - tile * ROWS),
            stage + token * CTA_PAIRS + task * ROWS // 2 + row_pair,
            word,
        )
    # the last triple's spare tokens carry zeros
    spare = (oar.triples(rows) * LINE_TOKENS - rows) * CTA_PAIRS
    if const_expr(spare > 0):  # noqa: SIM102
        if tid < spare:
            fx.ptr_store(fx.Int32(0), stage + rows * CTA_PAIRS + tid)
    gpu.barrier()
    c["stamp"](STAMP_STAGED)
    # past 48 rows a thread pushes a line a round, every round's before any
    # poll
    if const_expr(c["late"]):  # noqa: SIM102
        if bid % 2 == 1:
            ar._sleeps(c["late"])
    for k in range_constexpr(oar.line_rounds(rows)):
        _push_partials(c, stage, bases, bid, tid if k == 0 else tid + k * THREADS)
    # TEST ONLY (a late-push build): a CTA sums and lands its partner's columns
    summed = bid if c["late"] is None else bid ^ 1
    for k in range_constexpr(oar.line_rounds(rows)):
        _reduce(c, summed, tid if k == 0 else tid + k * THREADS, k == 0)


def build_mla_back(
    plan: Gfx1250KimiBuildPlan,
    tp: int,
    dcp: int,
    local_heads: int,
    bound: int | None = None,
    profile: bool = False,
    poll_together: bool | None = None,
    poll_marks: bool = False,
    test_swap_late: int | None = None,
) -> MlaBackKernel:
    """The back half at ``plan.tokens`` rows (``mla_back_layout.BackPlan``).
    ``o`` [rows, heads, 512] BF16 and ``lse`` [rows, heads] f32; ``w_uv``
    [heads, 128, 512] E4M3 and ``s_uv`` its f32 scale; ``gate`` [rows, proj]
    BF16; ``w_o`` / ``w_o_scale`` o_proj's preshuffled FP8 [HIDDEN, proj] and
    E8M0 block scales; ``peers`` / ``merge_peers`` every TP / DCP rank's
    receive memory (``BackPlan.peer_bytes``; the same table where the groups
    are the same) and ``rank`` / ``dcp_rank`` this rank's index in each;
    ``epochs`` ``k2_allreduce_gfx1250``'s counters. Out: ``out`` [rows,
    HIDDEN] BF16. Scratch: ``core`` [rows, proj] BF16 (the gated rows) and
    ``flags`` (``BackPlan.flag_words`` i32, zeroed once). ``profile``: ``STAMPS``
    i32 a CTA to a trailing ``stamps`` argument. ``poll_together`` (default
    ``ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER`` where a group has more than
    one rank): a poll's loads -- the merge's of the DCP shards' lines, the
    all-reduce's of the TP ranks', each where its group has more than one
    rank -- go out together and are awaited once (``ar.POLL_TOGETHER``; with
    the default each is waited out before the next: a memory round trip a
    rank); the same lines at the same addresses. Tool builds only, with
    ``bound``: ``poll_marks`` (``_poll``'s marks in ``epochs``, which then
    holds ``ar.GAVE_UP_WORD + 1`` words) and ``test_swap_late`` (the odd
    CTAs push their W_UV lines and their o_proj partials that many sleeps
    late, and an o_proj CTA sums and lands its partner CTA's columns: lines
    that land while they are polled)."""
    if plan.architecture is not GFX1250:
        raise ValueError("the MLA back requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("the MLA back requires 256 threads/eight waves")
    if plan.blocks < lay.GRID:
        raise ValueError(f"the MLA back needs {lay.GRID} CTAs")
    back = lay.BackPlan(plan.tokens, tp, dcp, local_heads)
    if poll_together is None:
        poll_together = envs.ATOM_KIMI_MONO_MLA_BACK_POLL_TOGETHER
    poll_together = bool(poll_together) and max(tp, dcp) > 1
    if (poll_marks or test_swap_late is not None) and bound is None:
        raise ValueError("the MLA back's poll marks and late pushes take a bound")
    if test_swap_late is not None and test_swap_late < 0:
        raise ValueError("test_swap_late counts sleeps")
    merge_modifier = ar.POLL_TOGETHER if poll_together and dcp > 1 else ar.POLL_VOLATILE
    ar_modifier = ar.POLL_TOGETHER if poll_together and tp > 1 else ar.POLL_VOLATILE
    rows, proj = back.rows, back.proj
    tiles = back.tiles
    rest = 2 * tiles * ACCUMULATOR_WORDS + oar.triples(rows) * LINE_TOKENS * CTA_PAIRS
    paired = (
        "oproj" in gemv_tdm.paired_sites()
        and proj >= 2 * gemv_tdm.SLAB
        and sum(tdm_lds_words(rows, 2, True, TDM_FP8_PAIR_DEPTH, 2)) + rest
        <= plan.lds_bytes // 4
    )
    depth = TDM_FP8_PAIR_DEPTH if paired else gemv_tdm.FP8_DEPTH
    if (
        sum(tdm_lds_words(rows, 2, True, depth, 2 if paired else 1)) + rest
        > plan.lds_bytes // 4
    ):
        raise ValueError(f"the MLA back's o_proj does not fit the LDS at {rows} rows")

    @fx.struct
    class Lds:
        gemv: fx.Array[
            fx.Int32,
            sum(tdm_lds_words(rows, 2, True, depth, 2 if paired else 1)),
            16,
        ]
        red: fx.Array[fx.Float32, 2 * tiles * ACCUMULATOR_WORDS, 16]
        stage: fx.Array[fx.Int32, oar.triples(rows) * LINE_TOKENS * CTA_PAIRS, 16]

    name = (
        f"k3_mla_back_gfx1250_tp{tp}_d{dcp}_h{local_heads}_s{rows}"
        f"{'_ps' if paired else ''}"
        f"{'_pt' if poll_together else ''}"
        f"{'' if bound is None else f'_b{bound}'}"
        f"{'_gm' if poll_marks else ''}"
        f"{'' if test_swap_late is None else f'_sw{test_swap_late}'}"
        f"{'_profile' if profile else ''}"
        f"{tdm_timeline.suffix()}"
    )
    sources = SOURCES
    timeline = tdm_timeline.key()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        o: Int64,
        lse: Int64,
        w_uv: Int64,
        s_uv: Int64,
        gate: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        out: Int64,
        core: Int64,
        flags: Int64,
        peers: Int64,
        merge_peers: Int64,
        epochs: Int64,
        rank: Int32,
        dcp_rank: Int32,
        stamps: Int64,
    ):
        _ = sources, timeline
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        tdm_timeline.open_cta(bid, wave, timeline, allocator)

        def stamp(index, thread=0):
            if const_expr(profile):  # noqa: SIM102
                if tid == thread:
                    st_dev(fx.Int32(_realtime()), stamps, bid * STAMPS + index)

        stamp(STAMP_START)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        bank = lay.bank_base(back, epoch)
        c = {
            "plan": back,
            "bound": bound,
            "o": o,
            "lse": lse,
            "w_uv": w_uv,
            "s_uv": s_uv,
            "gate": gate,
            "out": out,
            "core": core,
            "flags": flags,
            "merge_peers": merge_peers,
            "dcp_rank": dcp_rank,
            "epoch": epoch,
            "bank": bank,
            "merge_mine": rsrc(load_ptr64(merge_peers, dcp_rank)),
            "mine": rsrc(load_ptr64(peers, rank)),
        }
        c["peers"] = peers
        c["rank"] = rank
        c["stamp"] = stamp
        c["bid"] = bid
        c["late"] = test_swap_late
        marks = epochs if poll_marks else None
        c["merge_poll"] = (merge_modifier, marks, bid, tid)
        c["ar_poll"] = (ar_modifier, marks, bid, tid)
        if bid < lay.OPROJ_CTAS:
            _oproj(
                c, lds.gemv.ptr, lds.red.ptr, lds.stage.ptr, w_o, w_o_scale,
                depth, paired, bid, tid, lane, wave,
            )  # fmt: skip
        else:
            _back(c, bid, lane, wave)
        gpu.barrier()
        stamp(STAMP_END)
        if tid == 0:
            bo.buffer_store(epoch, rsrc(epochs), bid)
        tdm_timeline.close_cta()

    @flyc.jit
    def launch(
        o: Int64,
        lse: Int64,
        w_uv: Int64,
        s_uv: Int64,
        gate: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        out: Int64,
        core: Int64,
        flags: Int64,
        peers: Int64,
        merge_peers: Int64,
        epochs: Int64,
        rank: Int32,
        dcp_rank: Int32,
        stamps: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources, timeline
        kernel(
            o,
            lse,
            w_uv,
            s_uv,
            gate,
            w_o,
            w_o_scale,
            out,
            core,
            flags,
            peers,
            merge_peers,
            epochs,
            rank,
            dcp_rank,
            stamps,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(lay.GRID,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    MLA_BACK_PROFILE_ABI.check(kernel, launch)
    if profile:
        return MlaBackKernel(
            name, launch, (lay.GRID,), (THREADS,), back, MLA_BACK_PROFILE_ABI
        )
    launcher = ar.TailedLauncher(launch, (0,))
    return MlaBackKernel(name, launcher, (lay.GRID,), (THREADS,), back)
