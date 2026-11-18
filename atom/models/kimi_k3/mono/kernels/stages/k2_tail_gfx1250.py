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
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250
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
    THREADS,
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
STAMPS = 8
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


def tail_flag_words(profile: bool = False, bounded: bool = False) -> int:
    """``flags``: a word a CTA, then with ``profile`` STAMPS a CTA; a
    ``bounded`` build's through ``GAVE_UP_WORD``."""
    if bounded:
        return GAVE_UP_WORD + 1
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
def _stamp(flags, bid, tid, index):
    if tid == 0:
        st_dev(fx.Int32(_realtime()), flags, MAX_CTAS + bid * STAMPS + index)


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
def _poll_lines7_bounded(mine, ats, tag, flags, bid, bound):
    """``_poll_lines7`` giving up past ``bound`` polls, marking
    ``flags[GAVE_UP_WORD]`` (scratch builds only)."""
    words = _load_lines(mine, ats)
    polls = fx.Int32(0)
    while _stale7(words, len(ats), tag) & (polls < bound):
        spin_pause()
        words = _load_lines(mine, ats)
        polls = polls + 1
    if polls >= bound:
        st_dev(fx.Int32(GAVE_UP) | bid, flags, GAVE_UP_WORD)
    return words


def _poll7(c, ats):
    """Every source's copy of the lines at ``ats`` once all carry the tag."""
    if const_expr(c["bound"] is None):
        return _poll_lines7(c["mine"], ats, c["tag"])
    return _poll_lines7_bounded(
        c["mine"], ats, c["tag"], c["flags"], c["bid"], c["bound"]
    )


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
        _stamp(c["flags"], bid, tid, 6)
    half = bid % 2
    row = c["row"]
    for k in range_constexpr(LINE7_ELEMENTS):
        fx.ptr_store(own[k], row + half * HALF + tid * LINE7_ELEMENTS + k)
        fx.ptr_store(other[k], row + (1 - half) * HALF + tid * LINE7_ELEMENTS + k)
    gpu.barrier()
    total = _row_square_sum(row, c["squares"], tid, lane, wave)
    mean = fx.Float32(
        fx.Float64(total * _rcp(fx.Float32(float(LATENT)))) + fx.Float64(c["eps"])
    )
    rstd = hw_rsq(mean)
    if const_expr(c["profile"]):
        _stamp(c["flags"], bid, tid, 7)
    _store_seven_dev(
        c["local"],
        line * LINE7_ELEMENTS,
        [_normalize(x, g, rstd) for x, g in zip(own, gains)],
    )


@traced
def _shared_finish(c, line):
    tp, lines, bank = c["tp"], c["lines"], c["bank"]
    got = _poll7(c, [bank + (p * lines + line) * LINE_WORDS for p in range(tp)])
    _store_seven_dev(c["local"], line * LINE7_ELEMENTS, _reduced(got, 0, tp))


def _load_cm(c):
    """Inputs stored in this launch (a fused one's) load at device scope."""
    return CM_DEV if c["fused"] else 0


@traced
def _ar_push(c):
    """This CTA's all-reduce lines (routed ones gathered from ``grouped``)
    pushed to every rank; returns a routed line's norm weights. With
    ``identity`` routes, route ``k`` of token ``t`` is row ``t * TOPK + k``
    of ``grouped`` (the fused MoE's), its gathers issued at once."""
    values, gains = _ar_gather(c)
    if const_expr(c["split"]):
        _split_push(c, values, gains)
        return gains
    # every thread has its words before any pushes: a CU with remote stores
    # in flight holds back the gathers' loads behind them
    gpu.barrier()
    if const_expr(c["profile"]):
        _stamp(c["flags"], c["bid"], c["tid"], 5)
    _ar_send(c, values)
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
                loaded = fx.Vector(ld_dev(c["route_weights"], token * TOPK + 4 * quad, 4))
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
        _stamp(c["flags"], bid, tid, 5)
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
        _stamp(c["flags"], bid, c["tid"] - PUSH_THREADS, 6)
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
                _stamp(c["flags"], bid, tid, 7)
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
    if bid < c["ar_ctas"]:
        if bid < 2 * c["rows"]:
            _routed_finish(c, line, gains)
        else:
            _shared_finish(c, line)
        rocdl.s_wait_storecnt(0)
        gpu.barrier()
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
    ranks' skew has no bound."""
    ar_ctas = tail_ar_ctas(rows)
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
    layout = tail_layout(rows, depth, tokens, split)
    lines = allreduce7_lines(rows)
    bank_words = tp * lines * LINE_WORDS
    tiles = (rows + ROWS - 1) // ROWS
    name = (
        f"k3_k2_gfx1250_tail_tp{tp}_s{rows}_d{layout.depth}_t{layout.tokens}"
        + ("_id" if identity_routes else "")
        + ("_sw" if split else "")
        + ("" if poll_bound is None else f"_b{poll_bound}")
        + ("_profile" if profile else "")
        + fresh.suffix("tail")
    )
    sources = SOURCES

    @fx.union
    class Lds:
        words: fx.Array[fx.Int32, layout.words, 16]
        floats: fx.Array[fx.Float32, layout.words, 16]

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
        if const_expr(profile):
            _stamp(flags, bid, tid, 0)
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
        }
        if const_expr(split):
            role = uniform(wave)
            c["pusher"] = role < PUSH_WAVES
            c["reducer"] = role >= PUSH_WAVES
            c["stage"] = words + AR_WORDS
            c["gains"] = floats + AR_WORDS + THREADS * LINE_WORDS
        bases = []

        def prologue():
            if const_expr(fused is not None):
                fused["wait_inputs"]()
            if const_expr(split):
                _ar_push(c)
                _split_finish(c)
            elif const_expr(rounds == 1):
                _ar_finish(c, _ar_push(c))
            else:
                _ar_rounds(c, grid, rounds)
            _store_epochs(epochs, bid, tid, grid, epoch)
            if const_expr(profile):
                _stamp(flags, bid, tid, 1)
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
                _stamp(flags, bid, tid, 2)
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
        )
        if const_expr(profile):
            _stamp(flags, bid, tid, 3)
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
            _stamp(flags, bid, tid, 4)

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
