# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 K2 tail in one resident launch: the gathering all-reduce
(``k2_allreduce_gfx1250.build_gather_allreduce``'s seven-element lines), the
routed experts' RMSNorm and the replicated finalize
(``k2_finalize_tdm_gfx1250``), bit for bit the three launches' output.

Every CTA first issues its finalize share's first up-weight slabs through
the TDM (``gemv_tdm_gfx1250.project``), then, in the GEMV's prologue, does
its all-reduce share: CTA ``b < 6 rows`` pushes and reduces lines
``256 b ..``, a line a thread, as the gathering all-reduce does. A routed row
is 512 lines, two CTAs'; each of the two polls its partner's lines from the
receive bank as well and reduces them alike, so it holds the whole reduced
row and normalizes its own half with no hand-off. Past 42 rows the shares
outnumber the grid's CTAs: CTA ``b`` also takes the shared rows' shares
``b + k grid`` (``tail_rounds``), gathering all its shares' lines before
pushing any, and publishes its flag once all are reduced. The norm is AITER
``rmsnorm2d_fwd``'s gfx1250 kernel's arithmetic: its 256-thread reduction
order (a thread's two 8-element runs 256 apart, the waves' butterflies, then
the eight wave sums pairwise), the mean as the sum times ``v_rcp`` of the
width, eps added in double, ``v_rsq``, then ``bf16(x * w * rstd)`` rounded
to fp32 at each product. The normed routed rows replace the reduced ones in
``data``; the shared rows are reduced in place, both at device scope. A CTA
then publishes its flag (``flags[b] = epoch``) and every CTA waits on all of
them before its GEMV reads the normed rows, every token slab the LDS holds
in flight at once, and adds the reduced shared rows.

Epochs: one a launch, ``k2_allreduce_gfx1250``'s counters, so this launch
keeps them equal with the other kernels sharing them; the flags carry the
whole epoch, unique to the launch.

Early gather (``ATOM_KIMI_MONO_TAIL_EARLY_GATHER``, off by default; a share a
CTA, no split waves): the CTA gathers its lines before it issues the first
up-weight slabs, so its gathers' loads do not queue behind them; it pushes
them in the prologue as before.

Polls together (``ATOM_KIMI_MONO_TAIL_POLL_TOGETHER``, off by default): a
poll's loads of the ranks' copies of a line (and of a routed line's partner)
carry no VOLATILE, so the compiler issues them back to back and waits once;
with it, it waits out each load before it issues the next, and a poll takes
a memory round trip a rank (about 0.4 us each: 2.0 to 2.5 us a poll at four
ranks, 3.5 to 4.7 us at eight, one GPU, the peers' banks local). The polled
lines and addresses are the same; the pause between polls keeps the loads in
the loop. More than one rank only: at one rank the first poll, issued right
behind the CTA's own local pushes, would return before they land and pay a
retry (the whole TP1 layer 0.5 us slower at 16 rows).

Resident GEMV (``ATOM_KIMI_MONO_TAIL_RESIDENT``, off by default; up to
``tail_resident_rows()`` rows, where every up-weight slab and token slab fits
the LDS): TAIL_DEPTH weight slabs are issued ahead of the gathers and the
rest once the CTA's lines are pushed, under the all-reduce; once the flags
are seen the token slabs are issued, every copy is awaited once, and the
slabs are multiplied with no barrier between them (the ring's two barriers a
slab were most of the finalize's time at these widths).

Split waves (``ATOM_KIMI_MONO_TAIL_SPLIT``): the CTA's lines go through LDS
to waves [0, 4), which push them, two a thread, while waves [4, 8) poll and
reduce them, two a thread, and store the reduced (or normalized) lines. A
wave's ``s_wait_storecnt`` counts its remote pushes until the fabric acks
them; split, the flag's waits only cover the reducing waves' local stores,
and every barrier before the flags are seen is an LDS one.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT, MAX_ROWS, TOPK
from atom.models.kimi_k3.mono.kernels.stages import fresh_prefetch_gfx1250 as fresh
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250, tdm_timeline
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    TDM_CLUSTER,
    TDM_SLAB,
    tdm_lds_words,
    tdm_task_words,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_allreduce_gfx1250 import (
    LINE7_ELEMENTS,
    LINE7_ROUTED,
    LINE_WORDS,
    MAX_CTAS,
    SCOPE_SYS,
    THREADS,
    VOLATILE,
    _load_lines,
    _pack,
    _poll_lines7,
    _seven,
    _stale7,
    _store_line_sys,
    allreduce7_lines,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import (
    _lds_barrier,
    _realtime,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    hw_rsq,
    ld_dev,
    load_ptr64,
    rsrc,
    spin_pause,
    st_dev,
    traced,
    uniform,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.utils import envs

WAVES = THREADS // GFX1250.wave_size
ROWS = 16
PAIRS = HIDDEN // (2 * ROWS)
ACCUMULATOR_WORDS = WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS
SLABS = LATENT // TDM_SLAB
HALF = LATENT // 2
# AITER's rmsnorm2d_fwd at LATENT: 256 threads of two 8-element runs, the
# second 256 elements past the first, a wave's 32 threads 512 apart.
NORM_RUN = 8
NORM_THREADS = LATENT // (2 * NORM_RUN)
assert LINE7_ROUTED == 2 * THREADS and NORM_THREADS % GFX1250.wave_size == 0
assert THREADS == 2 * ROWS * 8
# the AR's LDS: the reduced routed row, the wave sums, the routes
AR_WORDS = LATENT + WAVES + 2 * TOPK
# split waves: waves [0, PUSH_WAVES) push, the rest reduce, SPLIT_LINES lines
# a thread; past AR_WORDS the CTA's lines and its routed lines' gains
PUSH_WAVES = WAVES // 2
PUSH_THREADS = PUSH_WAVES * GFX1250.wave_size
SPLIT_LINES = THREADS // PUSH_THREADS
assert THREADS - PUSH_THREADS == PUSH_THREADS
SPLIT_AR_WORDS = AR_WORDS + THREADS * LINE_WORDS + THREADS * LINE7_ELEMENTS
STAMPS = 20
# a profile build's points, thread 0's shader cycle counter (low word), kept
# in LDS and copied out at the CTA's end: entry, the first weight slabs
# issued, every thread's lines gathered, the pushes issued, thread 0's lines
# polled, a routed row in LDS (every thread's lines polled, the pushes
# acked), its rstd, the reduced (normed) lines' stores issued, drained, the
# flag issued, every flag seen, the GEMV done, the end; the 100 MHz clock at
# entry (P_RT0) and at the end (P_RT1), which convert the cycles; a spy
# CTA's (one with no all-reduce share polling a routed CTA's lines): thread
# 0's lines seen, every thread's; the later weight slabs issued (P_MORE)
(
    P_START,
    P_ISSUED,
    P_GATHERED,
    P_PUSHED,
    P_POLLED,
    P_ROW,
    P_RSTD,
    P_STORED,
    P_DRAINED,
    P_FLAG,
    P_SEEN,
    P_GEMV,
    P_END,
    P_RT0,
    P_SPY,
    P_SPY_ALL,
    P_RT1,
    P_MORE,
    P_SPARE0,
    P_SPARE1,
) = range(STAMPS)
SPY_BOUND = 1 << 20
TAIL_DEPTH = 3
# a bounded (scratch) build's give-up mark: flags[GAVE_UP_WORD] = GAVE_UP | CTA,
# past every CTA's flag and stamps
GAVE_UP_WORD = MAX_CTAS * (1 + STAMPS)
GAVE_UP = 0x7AB00000

TAIL_ABI = KernelAbi(
    (
        "data",
        "grouped",
        "route_rows",
        "route_weights",
        "norm_weight",
        "up_weight",
        "output",
        "flags",
        "peers",
        "epochs",
        "rank",
    )
)


def tail_ar_ctas(rows: int) -> int:
    """The all-reduce's shares, 256 lines each: a CTA's, or past
    ``tail_line_rows()`` rows a CTA's in each of ``tail_rounds`` rounds."""
    return allreduce7_lines(rows) // THREADS


def tail_line_rows() -> int:
    """The widest whose all-reduce takes a line a thread."""
    rows = 1
    while tail_ar_ctas(rows + 1) <= MAX_CTAS:
        rows += 1
    return rows


def tail_max_rows() -> int:
    return MAX_ROWS


def tail_grid(rows: int) -> int:
    """The finalize's PAIRS CTAs, or the all-reduce's where more (at most
    MAX_CTAS), in whole TDM clusters."""
    ctas = max(PAIRS, min(tail_ar_ctas(rows), MAX_CTAS))
    return -(-ctas // TDM_CLUSTER) * TDM_CLUSTER


def tail_rounds(rows: int) -> int:
    """All-reduce shares a CTA: share ``b + k grid`` its round ``k``'s."""
    return -(-tail_ar_ctas(rows) // tail_grid(rows))


def tail_resident_rows() -> int:
    """The widest whose every up-weight slab and token slab fit the LDS."""
    rows = 0
    while rows < tail_line_rows():
        try:
            if tail_layout(rows + 1, SLABS - 1).tokens != SLABS:
                break
        except ValueError:
            break
        rows += 1
    return rows


def tail_flag_words(profile: bool = False, bounded: bool = False) -> int:
    """``flags``: a word a CTA, then with ``profile`` STAMPS a CTA; a
    ``bounded`` build's through ``GAVE_UP_WORD``."""
    if bounded:
        return GAVE_UP_WORD + 1 + MAX_CTAS
    return MAX_CTAS * (1 + (STAMPS if profile else 0))


@dataclass(frozen=True)
class TailLayout:
    """i32 word offsets of the one LDS block: the AR's scratch at 0 aliasing
    the GEMV's token slabs, the weight stages past those, and the epilogue's
    partial sums over weight stages the last slab does not read."""

    gemv: int
    red: int
    words: int
    depth: int
    tokens: int


def tail_split() -> bool:
    return envs.ATOM_KIMI_MONO_TAIL_SPLIT


def tail_early_gather() -> bool:
    return envs.ATOM_KIMI_MONO_TAIL_EARLY_GATHER


def tail_poll_together() -> bool:
    return envs.ATOM_KIMI_MONO_TAIL_POLL_TOGETHER


def tail_resident() -> bool:
    return envs.ATOM_KIMI_MONO_TAIL_RESIDENT


def tail_layout(
    rows: int,
    depth: int | None = None,
    tokens: int | None = None,
    split: bool = False,
) -> TailLayout:
    """``depth``: up-weight slabs issued ahead of the all-reduce, at most
    TAIL_DEPTH by default: more slow its gathers more than they save.
    ``tokens``: token slabs resident, by default as many as the rest of the
    LDS holds: all in flight at once once the flags are seen. ``split``: the
    split waves' AR scratch."""
    ar_words = SPLIT_AR_WORDS if split else AR_WORDS
    tiles = (rows + ROWS - 1) // ROWS
    token_words = tdm_lds_words(rows, 2)[0] // 2
    stage_words = 2 * tdm_task_words()
    red_words = 2 * tiles * ACCUMULATOR_WORDS
    limit = GFX1250.max_workgroup_lds_bytes // 4
    for d in [depth] if depth is not None else range(TAIL_DEPTH, 0, -1):
        stages = d + 1
        last = (SLABS - 1) % stages
        need = -(-red_words // stage_words)
        start = last + 1 if last + 1 + need <= stages else 0
        if start == 0 and need > last:
            continue
        for b in [tokens] if tokens is not None else range(SLABS, 1, -1):
            x_words = b * token_words
            gemv = -(-max(0, ar_words - x_words) // 64) * 64
            words = gemv + x_words + stages * stage_words
            if words <= limit:
                red = gemv + x_words + start * stage_words
                return TailLayout(gemv, red, words, d, b)
    raise ValueError(f"the K2 tail's LDS does not fit at {rows} rows")


@dataclass(frozen=True)
class K2TailKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    ar_ctas: int
    eps: float
    abi: KernelAbi = TAIL_ABI
    launch_count: int = 1
    # its LDS union and kernel body, for a launch that runs it as a part
    lds: object = None
    body: object = None


def _rcp(value):
    """``v_rcp_f32`` itself: LLVM folds ``llvm.amdgcn.rcp`` of a constant to
    the correctly rounded quotient, which AITER's runtime division is not."""
    return fx.Float32(
        llvm.inline_asm(
            T.f32,
            [fx.Float32(value).ir_value()],
            "v_rcp_f32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def _normalize(x, gain, rstd):
    """``(x * (gain + 0)) * rstd`` rounded to fp32 at each step, as AITER's
    rmsnorm2d computes it: LLVM would fold the product's BF16 conversion into
    one ``v_fma_mixlo_bf16``, whose single rounding differs."""
    return fx.Float32(
        llvm.inline_asm(
            T.f32,
            [fx.Float32(v).ir_value() for v in (x, gain, rstd)],
            "v_add_f32 $0, 0, $2\n v_mul_f32 $0, $1, $0\n v_mul_f32 $0, $0, $3",
            "=&v,v,v,v",
            has_side_effects=False,
        )
    )


@traced
def _stamp(record, tid, index):
    """Thread 0's cycle counter at point ``index`` into the CTA's LDS record
    (a global store a point would make every barrier wait for it)."""
    if tid == 0:
        fx.ptr_store(tdm_timeline._cycles(), record + index)


@traced
def _stamps_open(record, tid):
    """The record zeroed (a point a CTA never passes reads 0), then the
    100 MHz clock."""
    if tid < STAMPS:
        fx.ptr_store(fx.Int32(0), record + tid)
    if tid == 0:
        fx.ptr_store(fx.Int32(_realtime()), record + P_RT0)


@traced
def _stamps_close(record, flags, bid, tid):
    if tid == 0:
        fx.ptr_store(fx.Int32(_realtime()), record + P_RT1)
    if tid < STAMPS:
        st_dev(
            fx.Int32(fx.ptr_load(record + tid)), flags, MAX_CTAS + bid * STAMPS + tid
        )


def _push_delay(units):
    """TEST ONLY: hold the pushes back ``units`` sleeps of 31 * 64 clocks
    (about a microsecond each at 2 GHz), as a longer fabric hop would the
    lines."""
    for _ in range(units):
        llvm.InlineAsmOp(None, [], "s_sleep 31", "", has_side_effects=True)


def tail_helpers_max(rows: int) -> int:
    """The most helper CTAs a routed share's lines can be dealt to (8, 4, 2,
    1; 0: none): the grid's CTAs with no all-reduce share."""
    if tail_rounds(rows) > 1:
        return 0
    spare = tail_grid(rows) - tail_ar_ctas(rows)
    for n in (8, 4, 2, 1):
        if 2 * rows * n <= spare:
            return n
    return 0


def tail_push_lines(rows: int, helpers: int = 0) -> dict:
    """``{(CTA, thread): line}`` of the all-reduce lines the grid pushes (a
    share a CTA): the kernel's own arithmetic, for a host-side audit."""
    ar_ctas, grid = tail_ar_ctas(rows), tail_grid(rows)
    out = {}
    for bid in range(grid):
        for tid in range(THREADS):
            if not helpers:
                if bid < ar_ctas:
                    out[bid, tid] = bid * THREADS + tid
                continue
            per = THREADS // helpers
            j = bid - ar_ctas
            if 0 <= j < 2 * rows * helpers:
                if tid < per:
                    share = j % (2 * rows)
                    out[bid, tid] = share * THREADS + j // (2 * rows) * per + tid
            elif 2 * rows <= bid < ar_ctas:
                out[bid, tid] = bid * THREADS + tid
    return out


@traced
def _send_line(c, line, values):
    """Line ``line``'s seven values, tagged, to every rank."""
    words = [_pack(values[2 * w], values[2 * w + 1]) for w in range(3)]
    words.append(
        (_pack(values[6], fx.Float32(0.0)) & fx.Int32(0xFFFF)) | (c["tag"] << 16)
    )
    out = fx.Vector.from_elements(words, fx.Int32)
    at = fx.Int64(c["bank"] + (c["rank"] * c["lines"] + line) * LINE_WORDS) * 4
    for base in c["bases"]:
        _store_line_sys(base + at, out)


@traced
def _ar_push_helped(c):
    """``_ar_push`` with helpers (identity routes): a routed share's lines
    are gathered and pushed by ``c["helpers"]`` CTAs with no all-reduce share
    (CTA ``ar_ctas + j``: share ``j % (2 rows)``'s lines from
    ``j // (2 rows) * THREADS / helpers``, a line a thread of its first
    waves), so a routed CTA pushes nothing -- it loads its norm weights and
    polls -- and a CU gathers ``THREADS / helpers`` lines' TOPK picks. The
    shared CTAs push their own lines as before. The same lines, values and
    addresses: who pushes a routed line is all that changes."""
    tid, bid = c["tid"], c["bid"]
    rows, ar_ctas, n = c["rows"], c["ar_ctas"], c["helpers"]
    per = THREADS // n
    j = bid - ar_ctas
    helper = (j >= 0) & (j < 2 * rows * n)
    share = helper.select(j % (2 * rows), fx.min(bid, 2 * rows - 1))
    at = helper.select(j // (2 * rows) * per, fx.Int32(0)) + tid
    token = share // 2
    gather = helper & (tid < per)
    shared = (bid >= 2 * rows) & (bid < ar_ctas)
    if const_expr(c["fused"]):
        weights = []
        for quad in range_constexpr(TOPK // 4):
            loaded = fx.Vector(ld_dev(c["route_weights"], token * TOPK + 4 * quad, 4))
            weights.extend(fx.Int32(loaded[k]).bitcast(fx.Float32) for k in range(4))
    else:
        loaded = fx.Vector(
            bo.buffer_load(
                rsrc(c["route_weights"]),
                token * TOPK,
                vec_width=TOPK,
                dtype=T.i32,
                is_scalar=True,
            )
        )
        weights = [fx.Int32(loaded[k]).bitcast(fx.Float32) for k in range(TOPK)]
    v0 = v1 = v2 = v3 = v4 = v5 = v6 = fx.Float32(0.0)
    g0 = g1 = g2 = g3 = g4 = g5 = g6 = fx.Float32(0.0)
    if gather:
        column = (share % 2) * HALF + at * LINE7_ELEMENTS
        source = rsrc(c["grouped"])
        picks = [
            _seven(source, (token * TOPK + k) * LATENT + column, _load_cm(c))
            for k in range(TOPK)
        ]
        sums = []
        for i in range_constexpr(LINE7_ELEMENTS):
            total = fx.Float32(0.0)
            for pick, weight in zip(picks, weights):
                total = total + weight * (fx.Float32(0.0) + pick[i])
            sums.append(bf16_round(total))
        v0, v1, v2, v3, v4, v5, v6 = sums
    elif shared:
        v0, v1, v2, v3, v4, v5, v6 = _seven(
            c["local"], (bid * THREADS + tid) * LINE7_ELEMENTS, _load_cm(c)
        )
    if bid < 2 * rows:
        g0, g1, g2, g3, g4, g5, g6 = _seven(
            rsrc(c["norm_weight"]), (bid % 2) * HALF + tid * LINE7_ELEMENTS
        )
    gpu.barrier()
    if const_expr(c["profile"]):
        _stamp(c["record"], tid, P_GATHERED)
    # TEST ONLY delay: the pushing threads' alone, so a routed CTA polls
    # while its lines are still to come
    if gather | shared:
        if const_expr(c["push_delay"]):
            _push_delay(c["push_delay"])
        if const_expr(not c["skip_push"]):
            _send_line(
                c,
                gather.select(share * THREADS + at, bid * THREADS + tid),
                [v0, v1, v2, v3, v4, v5, v6],
            )
    if const_expr(c["profile"]):
        _stamp(c["record"], tid, P_PUSHED)
    return [g0, g1, g2, g3, g4, g5, g6]


@traced
def _spy(c):
    """Profile only: a CTA with no all-reduce share polls routed CTA
    ``bid % (2 rows)``'s lines and their partners' as that CTA does (bounded),
    stamping thread 0's and every thread's."""
    tp, lines, bank = c["tp"], c["lines"], c["bank"]
    tid, bid = c["tid"], c["bid"]
    if bid >= c["ar_ctas"]:
        line = (bid % (2 * c["rows"])) * THREADS + tid
        partner = line ^ THREADS
        ats = [bank + (p * lines + line) * LINE_WORDS for p in range(tp)] + [
            bank + (p * lines + partner) * LINE_WORDS for p in range(tp)
        ]
        words = _load_lines(c["mine"], ats)
        polls = fx.Int32(0)
        while _stale7(words, len(ats), c["tag"]) & (polls < SPY_BOUND):
            spin_pause()
            words = _load_lines(c["mine"], ats)
            polls = polls + 1
        _stamp(c["record"], tid, P_SPY)
        _lds_barrier()
        _stamp(c["record"], tid, P_SPY_ALL)


@traced
def _store_seven_dev(local, element, values):
    """Seven BF16 at ``element`` at device scope: the whole pairs as words,
    the odd end alone."""
    if element % 2 == 0:
        for w in range_constexpr(3):
            bo.buffer_store(
                _pack(values[2 * w], values[2 * w + 1]),
                local,
                element // 2 + w,
                cache_modifier=CM_DEV,
            )
        bo.buffer_store(
            values[6].to(fx.BFloat16), local, element + 6, cache_modifier=CM_DEV
        )
    else:
        bo.buffer_store(
            values[0].to(fx.BFloat16), local, element, cache_modifier=CM_DEV
        )
        for w in range_constexpr(3):
            bo.buffer_store(
                _pack(values[2 * w + 1], values[2 * w + 2]),
                local,
                (element + 1) // 2 + w,
                cache_modifier=CM_DEV,
            )


def _reduced(got, first, tp):
    """The sources' copies of a line (``got``'s from line ``first``) summed in
    rank order in fp32, rounded to BF16."""
    total = None
    for p in range_constexpr(tp):
        at = (first + p) * LINE_WORDS
        copy = []
        for w in range_constexpr(3):
            copy.extend(bf2_f32(fx.Int32(got[at + w])))
        copy.append(bf_lo(fx.Int32(got[at + 3])))
        total = copy if p == 0 else [a + b for a, b in zip(total, copy)]
    return [bf16_round(v) for v in total]


@traced
def _row_square_sum(row, squares, tid, lane, wave, lds_only=False):
    """The sum of squares of the LATENT fp32 at ``row`` in AITER rmsnorm2d's
    order: thread t of 256 its two runs from ``base`` and ``base + 256``
    (``base = 8 (t % 32) + 512 (t // 32)``, past the row for the last wave),
    in order from the first; each wave's 32 sums by an xor butterfly 1, 2, 4,
    8, 16; the eight wave sums pairwise. ``lds_only``: its barrier waits for
    no global store."""
    total = fx.Float32(0.0)
    if tid < NORM_THREADS:
        base = (tid % GFX1250.wave_size) * NORM_RUN + (tid // GFX1250.wave_size) * (
            2 * NORM_RUN * GFX1250.wave_size
        )
        values = []
        for run in range_constexpr(2):
            for quad in range_constexpr(NORM_RUN // 4):
                loaded = fx.Vector(
                    fx.ptr_load(
                        row + base + run * (NORM_RUN * GFX1250.wave_size) + quad * 4,
                        result_type=fx.Vector.make_type(4, fx.Float32),
                    )
                )
                values.extend(fx.Float32(loaded[j]) for j in range_constexpr(4))
        square = values[0] * values[0]
        for value in values[1:]:
            square = square + value * value
        total = square
    total = butterfly(total, (1, 2, 4, 8, 16), architecture=GFX1250)
    if lane == 0:
        fx.ptr_store(total, squares + wave)
    if const_expr(lds_only):
        _lds_barrier()
    else:
        gpu.barrier()
    sums = [fx.ptr_load(squares + w) for w in range_constexpr(WAVES)]
    pairs = [sums[2 * i] + sums[2 * i + 1] for i in range_constexpr(WAVES // 2)]
    quads = [pairs[2 * i] + pairs[2 * i + 1] for i in range_constexpr(WAVES // 4)]
    return quads[0] + quads[1]


@traced
def _poll_lines7_bounded(mine, ats, tag, flags, bid, bound, modifier, tid):
    """``_poll_lines7`` giving up past ``bound`` polls, marking
    ``flags[GAVE_UP_WORD]``; thread 0's polls into
    ``flags[GAVE_UP_WORD + 1 + CTA]`` (scratch builds only)."""
    words = _load_lines(mine, ats, modifier)
    polls = fx.Int32(0)
    while _stale7(words, len(ats), tag) & (polls < bound):
        spin_pause()
        words = _load_lines(mine, ats, modifier)
        polls = polls + 1
    if polls >= bound:
        st_dev(fx.Int32(GAVE_UP) | bid, flags, GAVE_UP_WORD)
    if (tid == 0) & (polls > 0):
        st_dev(polls, flags, GAVE_UP_WORD + 1 + bid)
    return words


@traced
def _poll_lines7_counted(mine, ats, tag, record, tid, modifier):
    """``_poll_lines7`` counting its loads (thread 0's into P_SPARE0)."""
    words = _load_lines(mine, ats, modifier)
    polls = fx.Int32(1)
    while _stale7(words, len(ats), tag):
        spin_pause()
        words = _load_lines(mine, ats, modifier)
        polls = polls + 1
    if tid == 0:
        fx.ptr_store(polls, record + P_SPARE0)
    return words


def _poll7(c, ats):
    """Every source's copy of the lines at ``ats`` once all carry the tag."""
    if const_expr(c["profile"] and c["bound"] is None):
        return _poll_lines7_counted(
            c["mine"], ats, c["tag"], c["record"], c["tid"], c["poll_cm"]
        )
    if const_expr(c["bound"] is None):
        return _poll_lines7(c["mine"], ats, c["tag"], c["poll_cm"])
    return _poll_lines7_bounded(
        c["mine"], ats, c["tag"], c["flags"], c["bid"], c["bound"], c["poll_cm"],
        c["tid"],
    )  # fmt: skip


@traced
def _routed_finish(c, line, gains):
    """A routed line's copies and its partner's (the other half of the row,
    the partner CTA's line ``tid``) once all carry the tag, both reduced; the
    whole row into LDS, its norm, and this line normalized into ``data``."""
    tp, lines, bank = c["tp"], c["lines"], c["bank"]
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    partner = line ^ THREADS
    got = _poll7(
        c,
        [bank + (p * lines + line) * LINE_WORDS for p in range(tp)]
        + [bank + (p * lines + partner) * LINE_WORDS for p in range(tp)],
    )
    own = _reduced(got, 0, tp)
    other = _reduced(got, tp, tp)
    if const_expr(c["profile"]):
        _stamp(c["record"], tid, P_POLLED)
    half = bid % 2
    row = c["row"]
    for k in range_constexpr(LINE7_ELEMENTS):
        fx.ptr_store(own[k], row + half * HALF + tid * LINE7_ELEMENTS + k)
        fx.ptr_store(other[k], row + (1 - half) * HALF + tid * LINE7_ELEMENTS + k)
    if const_expr(c["lds_barriers"]):
        _lds_barrier()
    else:
        gpu.barrier()
    if const_expr(c["profile"]):
        _stamp(c["record"], tid, P_ROW)
    total = _row_square_sum(
        row, c["squares"], tid, lane, wave, lds_only=c["lds_barriers"]
    )
    mean = fx.Float32(
        fx.Float64(total * _rcp(fx.Float32(float(LATENT)))) + fx.Float64(c["eps"])
    )
    rstd = hw_rsq(mean)
    if const_expr(c["profile"]):
        _stamp(c["record"], tid, P_RSTD)
    _store_seven_dev(
        c["local"],
        line * LINE7_ELEMENTS,
        [_normalize(x, g, rstd) for x, g in zip(own, gains)],
    )
    if const_expr(c["profile"]):
        _stamp(c["record"], tid, P_STORED)


@traced
def _shared_finish(c, line):
    tp, lines, bank = c["tp"], c["lines"], c["bank"]
    got = _poll7(c, [bank + (p * lines + line) * LINE_WORDS for p in range(tp)])
    if const_expr(c["profile"]):
        _stamp(c["record"], c["tid"], P_POLLED)
    _store_seven_dev(c["local"], line * LINE7_ELEMENTS, _reduced(got, 0, tp))
    if const_expr(c["profile"]):
        _stamp(c["record"], c["tid"], P_STORED)


def _load_cm(c):
    """Inputs stored in this launch (a fused one's) load at device scope."""
    return CM_DEV if c["fused"] else 0


@traced
def _ar_push(c, gathered=None):
    """This CTA's all-reduce lines (routed ones gathered from ``grouped``)
    pushed to every rank; returns a routed line's norm weights. With
    ``identity`` routes, route ``k`` of token ``t`` is row ``t * TOPK + k``
    of ``grouped`` (the fused MoE's), its gathers issued at once.
    ``gathered``: ``_ar_gather``'s, done already."""
    if const_expr(c["helpers"]):
        return _ar_push_helped(c)
    values, gains = _ar_gather(c) if gathered is None else gathered
    if const_expr(c["split"]):
        _split_push(c, values, gains)
        return gains
    # every thread has its words before any pushes: a CU with remote stores
    # in flight holds back the gathers' loads behind them
    gpu.barrier()
    if const_expr(c["profile"]):
        _stamp(c["record"], c["tid"], P_GATHERED)
    if const_expr(c["push_delay"]):
        _push_delay(c["push_delay"])
    if const_expr(c["late_odd"]):  # noqa: SIM102
        # TEST ONLY: a routed row's even CTA polls its partner's lines while
        # they land
        if c["bid"] % 2 == 1:
            _push_delay(c["late_odd"])
    if const_expr(not c["skip_push"]):
        _ar_send(c, values)
    if const_expr(c["profile"]):
        _stamp(c["record"], c["tid"], P_PUSHED)
    return gains


@traced
def _ar_gather(c):
    """``_ar_push``'s line values and norm weights, ``c["bid"]`` the share."""
    tid, bid = c["tid"], c["bid"]
    routes = c["routes"]
    routed = bid < 2 * c["rows"]
    line = bid * THREADS + tid
    token = bid // 2
    if const_expr(not c["identity"]):
        if routed & (tid < TOPK):
            at = token * TOPK + tid
            route_row = bo.buffer_load(
                rsrc(c["route_rows"]), at, vec_width=1, dtype=T.i32
            )
            route_weight = bo.buffer_load(
                rsrc(c["route_weights"]), at, vec_width=1, dtype=T.f32
            )
            fx.ptr_store(fx.Int32(route_row).bitcast(fx.Float32), routes + 2 * tid)
            fx.ptr_store(fx.Float32(route_weight), routes + 2 * tid + 1)
        gpu.barrier()
    v0 = v1 = v2 = v3 = v4 = v5 = v6 = fx.Float32(0.0)
    g0 = g1 = g2 = g3 = g4 = g5 = g6 = fx.Float32(0.0)
    if routed:
        column = (bid % 2) * HALF + tid * LINE7_ELEMENTS
        source = rsrc(c["grouped"])
        if const_expr(c["identity"] and c["fused"]):
            # Stored in this launch: at device scope, not through the
            # scalar cache.
            rows_of = [token * TOPK + k for k in range(TOPK)]
            weights = []
            for quad in range_constexpr(TOPK // 4):
                loaded = fx.Vector(
                    ld_dev(c["route_weights"], token * TOPK + 4 * quad, 4)
                )
                weights.extend(
                    fx.Int32(loaded[k]).bitcast(fx.Float32) for k in range(4)
                )
        elif const_expr(c["identity"] and c["vector_weights"]):
            rows_of = [token * TOPK + k for k in range(TOPK)]
            weights = []
            for quad in range_constexpr(TOPK // 4):
                loaded = fx.Vector(
                    bo.buffer_load(
                        rsrc(c["route_weights"]),
                        token * TOPK + 4 * quad,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                weights.extend(
                    fx.Int32(loaded[k]).bitcast(fx.Float32) for k in range(4)
                )
        elif const_expr(c["identity"]):
            rows_of = [token * TOPK + k for k in range(TOPK)]
            loaded = fx.Vector(
                bo.buffer_load(
                    rsrc(c["route_weights"]),
                    token * TOPK,
                    vec_width=TOPK,
                    dtype=T.i32,
                    is_scalar=True,
                )
            )
            weights = [fx.Int32(loaded[k]).bitcast(fx.Float32) for k in range(TOPK)]
        else:
            rows_of = [
                fx.ptr_load(routes + 2 * k).bitcast(fx.Int32) for k in range(TOPK)
            ]
            weights = [fx.ptr_load(routes + 2 * k + 1) for k in range(TOPK)]
        picks = [_seven(source, row * LATENT + column, _load_cm(c)) for row in rows_of]
        g0, g1, g2, g3, g4, g5, g6 = _seven(rsrc(c["norm_weight"]), column)
        sums = []
        for j in range_constexpr(LINE7_ELEMENTS):
            total = fx.Float32(0.0)
            for pick, weight in zip(picks, weights):
                total = total + weight * (fx.Float32(0.0) + pick[j])
            sums.append(bf16_round(total))
        v0, v1, v2, v3, v4, v5, v6 = sums
    else:
        v0, v1, v2, v3, v4, v5, v6 = _seven(
            c["local"], fx.min(line, c["lines"] - 1) * LINE7_ELEMENTS, _load_cm(c)
        )
    return [v0, v1, v2, v3, v4, v5, v6], [g0, g1, g2, g3, g4, g5, g6]


@traced
def _ar_send(c, values):
    """``_ar_push``'s line of share ``c["bid"]``, if one, to every rank."""
    tid, bid = c["tid"], c["bid"]
    live = bid < c["ar_ctas"]
    line = bid * THREADS + tid
    if live:
        words = [_pack(values[2 * w], values[2 * w + 1]) for w in range(3)]
        words.append(
            (_pack(values[6], fx.Float32(0.0)) & fx.Int32(0xFFFF)) | (c["tag"] << 16)
        )
        out = fx.Vector.from_elements(words, fx.Int32)
        at = fx.Int64(c["bank"] + (c["rank"] * c["lines"] + line) * LINE_WORDS) * 4
        for base in c["bases"]:
            _store_line_sys(base + at, out)


@traced
def _split_push(c, values, gains):
    """Split waves: the thread's tagged line (and a routed line's gains) to
    LDS, then the pushing waves push the CTA's lines, SPLIT_LINES a thread."""
    tid, bid = c["tid"], c["bid"]
    words = [_pack(values[2 * w], values[2 * w + 1]) for w in range(3)]
    words.append(
        (_pack(values[6], fx.Float32(0.0)) & fx.Int32(0xFFFF)) | (c["tag"] << 16)
    )
    for w in range_constexpr(LINE_WORDS):
        fx.ptr_store(words[w], c["stage"] + tid * LINE_WORDS + w)
    if bid < 2 * c["rows"]:
        for k in range_constexpr(LINE7_ELEMENTS):
            fx.ptr_store(gains[k], c["gains"] + tid * LINE7_ELEMENTS + k)
    _lds_barrier()
    if const_expr(c["profile"]):
        _stamp(c["record"], tid, P_GATHERED)
    if (bid < c["ar_ctas"]) & c["pusher"]:
        for k in range_constexpr(SPLIT_LINES):
            t = tid + k * PUSH_THREADS
            out = fx.Vector.from_elements(
                [
                    fx.Int32(fx.ptr_load(c["stage"] + t * LINE_WORDS + w))
                    for w in range(LINE_WORDS)
                ],
                fx.Int32,
            )
            at = (
                fx.Int64(
                    c["bank"]
                    + (c["rank"] * c["lines"] + bid * THREADS + t) * LINE_WORDS
                )
                * 4
            )
            for base in c["bases"]:
                _store_line_sys(base + at, out)


def _split_lines(c):
    """A reducing thread's lines of its CTA, as indices into the CTA's."""
    first = c["tid"] - PUSH_THREADS
    return [first + k * PUSH_THREADS for k in range(SPLIT_LINES)]


@traced
def _split_routed_reduce(c):
    """A reducing thread's routed lines and their partners' once all carry
    the tag, reduced into the LDS row."""
    tp, lines, bank, bid = c["tp"], c["lines"], c["bank"], c["bid"]
    own = [bid * THREADS + t for t in _split_lines(c)]
    ats = [bank + (p * lines + line) * LINE_WORDS for line in own for p in range(tp)]
    ats += [
        bank + (p * lines + (line ^ THREADS)) * LINE_WORDS
        for line in own
        for p in range(tp)
    ]
    got = _poll7(c, ats)
    if const_expr(c["profile"]):
        _stamp(c["record"], c["tid"] - PUSH_THREADS, P_POLLED)
    half = bid % 2
    row = c["row"]
    for k, t in enumerate(_split_lines(c)):
        mine = _reduced(got, k * tp, tp)
        other = _reduced(got, (SPLIT_LINES + k) * tp, tp)
        for j in range_constexpr(LINE7_ELEMENTS):
            fx.ptr_store(mine[j], row + half * HALF + t * LINE7_ELEMENTS + j)
            fx.ptr_store(other[j], row + (1 - half) * HALF + t * LINE7_ELEMENTS + j)


@traced
def _split_normalize(c, rstd):
    """A reducing thread's routed lines normalized from the LDS row into
    ``data``, its stores drained."""
    bid = c["bid"]
    half = bid % 2
    row = c["row"]
    for t in _split_lines(c):
        values = [
            fx.ptr_load(row + half * HALF + t * LINE7_ELEMENTS + j)
            for j in range(LINE7_ELEMENTS)
        ]
        gains = [
            fx.ptr_load(c["gains"] + t * LINE7_ELEMENTS + j)
            for j in range(LINE7_ELEMENTS)
        ]
        _store_seven_dev(
            c["local"],
            (bid * THREADS + t) * LINE7_ELEMENTS,
            [_normalize(x, g, rstd) for x, g in zip(values, gains)],
        )
    rocdl.s_wait_storecnt(0)


@traced
def _split_shared(c):
    """A reducing thread's shared lines once all carry the tag, reduced into
    ``data``, its stores drained."""
    tp, lines, bank, bid = c["tp"], c["lines"], c["bank"], c["bid"]
    own = [bid * THREADS + t for t in _split_lines(c)]
    got = _poll7(
        c, [bank + (p * lines + line) * LINE_WORDS for line in own for p in range(tp)]
    )
    for k, line in enumerate(own):
        _store_seven_dev(c["local"], line * LINE7_ELEMENTS, _reduced(got, k * tp, tp))
    rocdl.s_wait_storecnt(0)


@traced
def _split_finish(c):
    """Split waves: this CTA's lines reduced by the reducing waves, the
    routed ones normalized (the whole CTA's norm), then its flag once the
    reducing waves' stores drain; no barrier waits for a global store."""
    tid, bid, lane, wave = c["tid"], c["bid"], c["lane"], c["wave"]
    if bid < c["ar_ctas"]:
        if bid < 2 * c["rows"]:
            if c["reducer"]:
                _split_routed_reduce(c)
            _lds_barrier()
            total = _row_square_sum(
                c["row"], c["squares"], tid, lane, wave, lds_only=True
            )
            mean = fx.Float32(
                fx.Float64(total * _rcp(fx.Float32(float(LATENT))))
                + fx.Float64(c["eps"])
            )
            rstd = hw_rsq(mean)
            if const_expr(c["profile"]):
                _stamp(c["record"], tid, P_RSTD)
            if c["reducer"]:
                _split_normalize(c, rstd)
        else:
            if c["reducer"]:
                _split_shared(c)
        _lds_barrier()
        if tid == PUSH_THREADS:
            st_dev(c["epoch"], c["flags"], bid)


@traced
def _ar_finish(c, gains):
    """This CTA's lines reduced, the routed ones normalized, then its flag."""
    tid, bid = c["tid"], c["bid"]
    line = bid * THREADS + tid
    if const_expr(c["poll_delay"]):
        _push_delay(c["poll_delay"])
    if bid < c["ar_ctas"]:
        if bid < 2 * c["rows"]:
            _routed_finish(c, line, gains)
        else:
            _shared_finish(c, line)
        rocdl.s_wait_storecnt(0)
        if const_expr(c["profile"]):
            _stamp(c["record"], tid, P_DRAINED)
        gpu.barrier()
        if const_expr(c["profile"]):
            _stamp(c["record"], tid, P_FLAG)
        if tid == 0:
            st_dev(c["epoch"], c["flags"], bid)


@traced
def _ar_rounds(c, grid, rounds):
    """``_ar_push`` then ``_ar_finish`` of shares ``bid + k grid``, ``k <
    rounds``: every share's lines gathered, then all pushed, then each
    reduced; the flag once all are. Past round 0 the shares are shared
    rows'."""
    tid, bid = c["tid"], c["bid"]
    views = [c] + [{**c, "bid": bid + k * grid} for k in range(1, rounds)]
    gathered = [_ar_gather(v) for v in views]
    gpu.barrier()
    for v, (values, _) in zip(views, gathered):
        _ar_send(v, values)
    if bid < 2 * c["rows"]:
        _routed_finish(c, bid * THREADS + tid, gathered[0][1])
    elif bid < c["ar_ctas"]:
        _shared_finish(c, bid * THREADS + tid)
    for k in range_constexpr(1, rounds):
        _ar_share_finish(views[k])
    if bid < c["ar_ctas"]:
        rocdl.s_wait_storecnt(0)
        gpu.barrier()
        if tid == 0:
            st_dev(c["epoch"], c["flags"], bid)


@traced
def _ar_share_finish(c):
    if c["bid"] < c["ar_ctas"]:
        _shared_finish(c, c["bid"] * THREADS + c["tid"])


@traced
def _wait_flags(flags, tid, ctas, epoch):
    """Thread ``t < ctas`` until CTA ``t``'s flag carries ``epoch``."""
    if tid < ctas:
        flag = fx.Int32(ld_dev(flags, tid))
        while flag != epoch:
            rocdl.s_sleep(1)
            flag = fx.Int32(ld_dev(flags, tid))


@traced
def _prefetch_share(data, grouped, route_weights, rows, ar_ctas, bid, tid):
    """``ATOM_KIMI_MONO_XPF=tail``: the lines this CTA's all-reduce share
    gathers, ahead of its first weight copies: a routed CTA's TOPK half rows
    of ``grouped`` and its token's route weights, a shared CTA's lines of
    ``data``'s shared rows."""
    per_row = HALF * 2 // 128
    if bid < 2 * rows:
        token = bid // 2
        half = bid % 2
        fresh.prefetch_own(
            "tail",
            lambda i: grouped
            + fx.Int64((token * TOPK + i // per_row) * LATENT + half * HALF) * 2
            + fx.Int64(i % per_row * 128),
            TOPK * per_row,
            tid,
            THREADS,
        )
        fresh.prefetch_own(
            "tail",
            lambda i: route_weights + fx.Int64(token * TOPK * 4),
            1,
            tid,
            THREADS,
        )
    if (bid >= 2 * rows) & (bid < ar_ctas):
        fresh.prefetch_own(
            "tail",
            lambda i: data + fx.Int64(bid * THREADS * LINE7_ELEMENTS * 2 + i * 128),
            THREADS * LINE7_ELEMENTS * 2 // 128,
            tid,
            THREADS,
        )


@traced
def _wait_flags_bounded(flags, tid, bid, ctas, epoch, bound):
    """``_wait_flags`` giving up past ``bound`` polls (scratch builds only)."""
    if tid < ctas:
        flag = fx.Int32(ld_dev(flags, tid))
        polls = fx.Int32(0)
        while (flag != epoch) & (polls < bound):
            rocdl.s_sleep(1)
            flag = fx.Int32(ld_dev(flags, tid))
            polls = polls + 1
        if polls >= bound:
            st_dev(fx.Int32(GAVE_UP) | (bid << 8) | tid, flags, GAVE_UP_WORD)


@traced
def _store_epochs(epochs, bid, tid, grid, epoch):
    counter = bid + tid * grid
    if counter < MAX_CTAS:
        bo.buffer_store(epoch, rsrc(epochs), counter)


def build_k2_tail(
    tp: int,
    rows: int,
    eps: float,
    depth: int | None = None,
    tokens: int | None = None,
    identity_routes: bool = False,
    profile: bool = False,
    split: bool | None = None,
    poll_bound: int | None = None,
    early_gather: bool | None = None,
    spy: bool = False,
    push_delay_us: float = 0.0,
    first_depth: int | None = None,
    helpers: int = 0,
    lds_barriers: bool = False,
    vector_weights: bool = False,
    poll_delay: int = 0,
    poll_together: bool | None = None,
    resident: bool | None = None,
    test_skip_push: bool = False,
    test_late_odd: int = 0,
) -> K2TailKernel:
    """The gathering all-reduce, routed RMSNorm (``norm_weight`` [LATENT]
    BF16, ``eps``) and replicated finalize (``up_weight`` [HIDDEN, LATENT]
    BF16 into ``output`` [rows, HIDDEN]) in one launch, on ``data``
    ([rows, LATENT] routed then [rows, HIDDEN] shared BF16) as the gathering
    all-reduce takes it. ``flags``: ``tail_flag_words(profile)`` i32, zeroed
    once. ``depth``, ``tokens``: ``tail_layout``'s. ``identity_routes``:
    ``route_rows`` is the identity (the fused MoE's) and goes unread.
    ``profile``: each CTA's 100 MHz stamps in ``flags`` past its flags,
    STAMPS a CTA: start, all-reduce done, flags seen, GEMV done, end, lines
    gathered, routed lines polled, rstd. ``split``: split waves (default
    ``ATOM_KIMI_MONO_TAIL_SPLIT``). ``poll_bound``: scratch builds only --
    every poll gives up past that many tries and marks
    ``flags[GAVE_UP_WORD]``; production polls are unbounded, since the
    ranks' skew has no bound. ``early_gather`` (default
    ``ATOM_KIMI_MONO_TAIL_EARLY_GATHER``; a share a CTA, no split waves): the
    lines gathered before the first up-weight slabs are issued. ``spy``
    (profile builds): ``_spy``. ``push_delay_us``: TEST ONLY, never set by
    the runner -- every CTA holds its pushes back that long, as a longer
    fabric hop would the lines. ``first_depth`` (a share a CTA, no split
    waves): only that many of the ``depth`` up-weight slabs are issued ahead
    of the all-reduce's gathers, the rest once the CTA's lines are pushed,
    under the fabric hop. ``helpers`` (identity routes, a share a CTA, no
    split waves; at most ``tail_helpers_max``): ``_ar_push_helped``.
    ``lds_barriers``: a routed CTA's two barriers ahead of its norm sync its
    LDS alone (they wait for no store, so not for its pushes' acks).
    ``vector_weights``: a routed CTA loads its token's route weights as its
    picks, not through the scalar cache. ``poll_delay``: TEST ONLY -- the
    first poll held back that many sleeps. ``poll_together`` (default
    ``ATOM_KIMI_MONO_TAIL_POLL_TOGETHER`` at more than one rank): a poll's
    loads of the sources' copies go out together and are awaited once (not
    VOLATILE, which makes the compiler wait out each before the next: a round
    trip a source); the same lines at the same addresses. ``resident``
    (``depth`` every later slab, every token slab resident: up to
    ``tail_resident_rows()`` rows): the GEMV awaits all its copies once and
    multiplies the slabs with no barrier between them. ``test_skip_push``:
    TEST ONLY, with ``poll_bound`` -- no line is pushed. By default
    (``ATOM_KIMI_MONO_TAIL_RESIDENT``, a share a CTA, no split waves, no
    ``depth`` / ``tokens`` / ``first_depth`` given) on at those widths, with
    TAIL_DEPTH slabs issued ahead of the gathers and the rest once the lines
    are pushed. ``test_late_odd``: TEST ONLY, with ``poll_bound`` (a share a
    CTA, no split waves, no helpers) -- the odd CTAs hold their pushes back
    that many sleeps: a routed row's even CTA polls its partner's lines
    while they land."""
    ar_ctas = tail_ar_ctas(rows)
    if test_skip_push and (poll_bound is None or tail_rounds(rows) > 1):
        raise ValueError("a tail that pushes nothing takes a poll bound, a share a CTA")
    if spy and not (profile and tail_rounds(rows) == 1):
        raise ValueError("a spy K2 tail is a profile build, a share a CTA")
    push_delay = round(push_delay_us)
    if not 1 <= rows <= tail_max_rows():
        raise ValueError(f"the K2 tail takes 1..{tail_max_rows()} rows, got {rows}")
    grid = tail_grid(rows)
    rounds = tail_rounds(rows)
    # every routed share in round 0, its row's partner on a CTA of its own
    assert 2 * rows <= grid
    if profile and rounds > 1:
        raise ValueError("a profile K2 tail takes a share a CTA")
    if split is None:
        split = tail_split() and rounds == 1
    elif split and rounds > 1:
        raise ValueError(
            f"split waves take a share a CTA: up to {tail_line_rows()} rows"
        )
    if early_gather is None:
        early_gather = tail_early_gather()
    early_gather = early_gather and rounds == 1 and not split
    if test_late_odd and (poll_bound is None or rounds > 1 or split or helpers):
        raise ValueError(
            "a late-odd tail takes a poll bound, a share a CTA, no split waves"
            " and no helpers"
        )
    if first_depth is not None and (rounds > 1 or split):
        raise ValueError("first_depth takes a share a CTA and no split waves")
    if helpers:
        if not identity_routes or split or helpers not in (1, 2, 4, 8):
            raise ValueError("tail helpers take identity routes and no split waves")
        if helpers > tail_helpers_max(rows):
            raise ValueError(
                f"{helpers} tail helpers a share at {rows} rows: at most"
                f" {tail_helpers_max(rows)}"
            )
        early_gather = False
    if poll_together is None:
        # one rank: a poll issued right behind the CTA's own local pushes
        # returns before they land and pays a retry
        poll_together = tail_poll_together() and tp > 1
    if resident is None:
        resident = (
            tail_resident()
            and depth is None
            and tokens is None
            and first_depth is None
            and rounds == 1
            and not split
            and rows <= tail_resident_rows()
        )
        if resident:
            depth, first_depth = SLABS - 1, TAIL_DEPTH
    layout = tail_layout(rows, depth, tokens, split)
    if resident and (
        layout.depth != SLABS - 1 or layout.tokens != SLABS or rounds > 1 or split
    ):
        raise ValueError(
            f"a resident K2 tail GEMV keeps every slab in LDS: depth {SLABS - 1},"
            f" up to {tail_resident_rows()} rows"
        )
    lines = allreduce7_lines(rows)
    bank_words = tp * lines * LINE_WORDS
    tiles = (rows + ROWS - 1) // ROWS
    name = (
        f"k3_k2_gfx1250_tail_tp{tp}_s{rows}_d{layout.depth}_t{layout.tokens}"
        + ("_id" if identity_routes else "")
        + ("_sw" if split else "")
        + ("_eg" if early_gather else "")
        + ("" if first_depth is None else f"_f{first_depth}")
        + (f"_h{helpers}" if helpers else "")
        + ("_lb" if lds_barriers else "")
        + ("_wv" if vector_weights else "")
        + (f"_qd{poll_delay}" if poll_delay else "")
        + ("_pt" if poll_together else "")
        + ("_np" if test_skip_push else "")
        + (f"_lo{test_late_odd}" if test_late_odd else "")
        + ("_rs" if resident else "")
        + ("" if poll_bound is None else f"_b{poll_bound}")
        + (f"_pd{push_delay}" if push_delay else "")
        + ("_profile" if profile else "")
        + ("_spy" if spy else "")
        + fresh.suffix("tail")
    )
    sources = SOURCES

    @fx.union
    class Lds:
        words: fx.Array[fx.Int32, layout.words + (STAMPS if profile else 0), 16]
        floats: fx.Array[fx.Float32, layout.words + (STAMPS if profile else 0), 16]

    @traced
    def body(
        storage,
        data,
        grouped,
        route_rows,
        route_weights,
        norm_weight,
        up_weight,
        output,
        flags,
        peers,
        epochs,
        rank,
        fused=None,
    ):
        """The kernel on ``storage`` (an ``Lds``). ``fused``: the launch's
        last part (``k2_fused_gfx1250``): ``wait_inputs()`` before the
        all-reduce reads ``data``, ``grouped`` and ``route_weights``, then
        at device scope."""
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        if const_expr(fused is None and fresh.on("tail")):
            _prefetch_share(data, grouped, route_weights, rows, ar_ctas, bid, tid)
        words = storage.words.peek().ptr
        floats = storage.floats.peek().ptr
        record = words + layout.words
        if const_expr(profile):
            _stamps_open(record, tid)
            _stamp(record, tid, P_START)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        row0 = bid * (2 * ROWS)
        owner = bid < PAIRS
        # Threads [0, 128) finish a tile's first task, the rest its second.
        task = tid // (ROWS * 8)
        local_token = tid % (ROWS * 8) // 8
        row_pair = tid % 8
        locations = [
            (
                fx.min(tile * ROWS + local_token, rows - 1) * HIDDEN
                + fx.min(row0, HIDDEN - 2 * ROWS)
                + task * ROWS
            )
            // 2
            + row_pair
            for tile in range_constexpr(tiles)
        ]
        shared = data + fx.Int64(rows * LATENT * 2)
        c = {
            "tp": tp,
            "rows": rows,
            "lines": lines,
            "ar_ctas": ar_ctas,
            "eps": eps,
            "identity": identity_routes,
            "profile": profile,
            "tid": tid,
            "bid": bid,
            "lane": lane,
            "wave": wave,
            "epoch": epoch,
            "tag": epoch & fx.Int32(0xFFFF),
            "bank": (epoch & 1) * bank_words,
            "rank": rank,
            "bases": [load_ptr64(peers, p) for p in range(tp)],
            "mine": rsrc(load_ptr64(peers, rank)),
            "local": rsrc(data),
            "grouped": grouped,
            "route_rows": route_rows,
            "route_weights": route_weights,
            "norm_weight": norm_weight,
            "flags": flags,
            "row": floats,
            "squares": floats + LATENT,
            "routes": floats + LATENT + WAVES,
            "fused": fused is not None,
            "split": split,
            "bound": poll_bound,
            "record": record,
            "push_delay": push_delay,
            "helpers": helpers,
            "lds_barriers": lds_barriers,
            "vector_weights": vector_weights,
            "poll_delay": poll_delay,
            "skip_push": test_skip_push,
            "late_odd": test_late_odd,
            "poll_cm": SCOPE_SYS if poll_together else SCOPE_SYS | VOLATILE,
        }
        if const_expr(split):
            role = uniform(wave)
            c["pusher"] = role < PUSH_WAVES
            c["reducer"] = role >= PUSH_WAVES
            c["stage"] = words + AR_WORDS
            c["gains"] = floats + AR_WORDS + THREADS * LINE_WORDS
        bases = []
        early = None
        if const_expr(early_gather and fused is None):
            early = _ar_gather(c)

        def prologue(more=None):
            if const_expr(profile):
                _stamp(record, tid, P_ISSUED)
            if const_expr(fused is not None):
                fused["wait_inputs"]()
            if const_expr(split):
                _ar_push(c)
                _split_finish(c)
            elif const_expr(rounds == 1):
                gains = _ar_push(c, early)
                if const_expr(more is not None):
                    more()
                    if const_expr(profile):
                        _stamp(record, tid, P_MORE)
                _ar_finish(c, gains)
                if const_expr(spy):
                    _spy(c)
            else:
                _ar_rounds(c, grid, rounds)
            _store_epochs(epochs, bid, tid, grid, epoch)
            # Every reader of what the flags publish reads at device scope:
            # no acquire fence (its SCOPE_DEV cache invalidate, one a CTA,
            # held the last CTAs' GEMVs back by microseconds).
            if const_expr(poll_bound is None):
                _wait_flags(flags, tid, min(ar_ctas, grid), epoch)
            else:
                _wait_flags_bounded(
                    flags, tid, bid, min(ar_ctas, grid), epoch, poll_bound
                )
            if const_expr(split):
                _lds_barrier()
            else:
                gpu.barrier()
            if const_expr(profile):
                _stamp(record, tid, P_SEEN)
            bases.extend(fx.Int32(ld_dev(shared, location)) for location in locations)

        accumulators = gemv_tdm.project(
            words + layout.gemv,
            data,
            LATENT,
            rows,
            up_weight,
            HIDDEN,
            LATENT,
            [row0, row0 + ROWS],
            lane,
            wave,
            depth=layout.depth,
            prologue=prologue,
            x_cache_policy=CM_DEV,
            token_buffers=layout.tokens,
            prologue_first=first_depth,
            resident=resident,
        )
        if const_expr(profile):
            _stamp(record, tid, P_GEMV)
        red = floats + layout.red
        for tile in range_constexpr(tiles):
            for index in range_constexpr(2):
                gemv_gfx1250.store_accumulator(
                    accumulators[index][tile],
                    red + (2 * tile + index) * ACCUMULATOR_WORDS,
                    wave,
                    lane,
                )
        gpu.barrier()
        for tile in range_constexpr(tiles):
            slot = red + (2 * tile + task) * ACCUMULATOR_WORDS
            if owner & (local_token < min(ROWS, rows - tile * ROWS)):
                low = bf16_round(
                    bf_lo(bases[tile])
                    + gemv_gfx1250.row_sum(slot, 2 * row_pair, local_token, WAVES)
                )
                high = bf16_round(
                    bf_hi(bases[tile])
                    + gemv_gfx1250.row_sum(slot, 2 * row_pair + 1, local_token, WAVES)
                )
                bo.buffer_store(
                    fx.Vector.from_elements([low, high], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0],
                    rsrc(output),
                    locations[tile],
                )
        if const_expr(profile):
            _stamp(record, tid, P_END)
            _stamps_close(record, flags, bid, tid)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        data: Int64,
        grouped: Int64,
        route_rows: Int64,
        route_weights: Int64,
        norm_weight: Int64,
        up_weight: Int64,
        output: Int64,
        flags: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
    ):
        _ = sources
        body(
            fx.SharedAllocator().allocate(Lds),
            data,
            grouped,
            route_rows,
            route_weights,
            norm_weight,
            up_weight,
            output,
            flags,
            peers,
            epochs,
            rank,
        )

    @flyc.jit
    def launch(
        data: Int64,
        grouped: Int64,
        route_rows: Int64,
        route_weights: Int64,
        norm_weight: Int64,
        up_weight: Int64,
        output: Int64,
        flags: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            data,
            grouped,
            route_rows,
            route_weights,
            norm_weight,
            up_weight,
            output,
            flags,
            peers,
            epochs,
            rank,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    TAIL_ABI.check(kernel, launch)
    return K2TailKernel(
        name, launch, (grid,), (THREADS,), ar_ctas, eps, lds=Lds, body=body
    )
