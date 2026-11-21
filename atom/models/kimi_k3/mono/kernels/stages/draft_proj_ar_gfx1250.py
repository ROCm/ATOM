# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A DSpark draft layer's row-parallel projection and its TP all-reduce in
one gfx1250 launch: o_proj, or the MLP's down_proj, from the BF16 activation
``[rows, k]`` to the layer's ``[rows, HIDDEN]`` BF16 output summed over the
ranks -- in place of ``dynamic_per_group_scaled_quant``, the FlyDSL
``gemm_a8w8_mxfp8_128`` GEMM (split_k 1) and aiter's custom all-reduce, bit
for bit.

CTA ``b`` owns output columns ``32 b .. 32 b + 32`` (``draft_mlp_layout``).
Its two 16-row tasks' FP8 weights (AITER's 16x16 preshuffle, 128x128 E8M0
block scales) reach LDS in four TDM copies, and the BF16 activation in eight
(a wave its rows; every CTA reads all of it, and plain loads of it top out
near 30 GB/s a CU) -- all of K where the LDS holds it, else K's halves in
turn. Every thread then quantizes its share from LDS as AITER's per-1x128
E8M0 ``dynamic_per_group_scaled_quant`` does: a 128-element group's largest
BF16 magnitude (NaNs left out) floored at ``kAmaxFloor``, the code
``ceil_pow2(amax * f32(1 / 448))`` capped at 254, the values divided by its
power of two with ``v_cvt_scalef32_pk8_fp8_bf16`` -- eight lanes a group, the
FP8 bytes stored in the WMMA operand layout, the codes a word each. A wave
then chains the scaled FP8 WMMA (16x16x128, the E8M0 codes on both operands)
over K in group order for one (task, token tile): the GEMM's sum, rounded to
BF16 once.

The all-reduce is ``k2_oproj_ar_gfx1250``'s one-shot, its bank layout and
epochs unchanged: the CTA stages its BF16 pairs in LDS, pushes them as 16 B
lines (three tokens' pair of a column pair and the epoch) to every rank's
bank at system scope, polls every rank's line of its own columns until all
carry the epoch and sums them in rank order in fp32 -- aiter's custom
all-reduce's arithmetic, its f32 denormals flushed as that kernel's float
mode flushes them -- into the output. A serving build polls without a
bound, as that kernel does; ``poll_bound`` (tools and tests) gives a poll up
past that many tries and marks ``flags[0]``.
``ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER`` (off by default, more than one
rank): a poll's loads of the ranks' lines are issued together and awaited
once, in place of a memory round trip a rank.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS
from atom.models.kimi_k3.mono.kernels.stages.mla_absorb_gfx1250 import (
    _mma_fp8,
    _pack,
    _scalar_byte,
    _splat,
)
from atom.models.kimi_k3.mono.kernels.stages.mla_draft_front_gfx1250 import (
    _weight_operand,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    bf2_f32,
    load_ptr64,
    rsrc,
    spin_pause,
    st_dev,
    traced,
    uniform,
    xshfl,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.consensus import MonoUnsupported
from atom.utils import envs

THREADS = lay.THREADS
WAVES = lay.WAVES
LANES = lay.LANES
ACC = 8
CTAS = lay.CTAS
CTA_PAIRS = lay.CTA_PAIRS
LINE_TOKENS = lay.LINE_TOKENS
# rounds of the activation quant whose loads go out together
QUANT_BATCH = 4
# the bank layout is k2_oproj_ar_gfx1250's one-shot
assert (LINE_TOKENS, lay.LINE_WORDS) == (oar.LINE_TOKENS, oar.LINE_WORDS)
assert (CTAS, CTA_PAIRS, THREADS) == (oar.CTAS, oar.CTA_PAIRS, oar.THREADS)
assert CTAS <= ar.MAX_CTAS == lay.EPOCH_COUNTERS

DRAFT_PROJ_AR_ABI = KernelAbi(("x", "w", "w_scale", "out", "peers", "epochs", "rank"))
# a bounded build's trailing argument: flags[0] = GAVE_UP | CTA once a poll
# gave up
DRAFT_PROJ_AR_BOUNDED_ABI = KernelAbi((*DRAFT_PROJ_AR_ABI.names, "flags"))
# a profile build's: STAMPS i32 a CTA, thread 0's low half of the 100 MHz
# clock at the launch's phases
DRAFT_PROJ_AR_PROFILE_ABI = KernelAbi((*DRAFT_PROJ_AR_BOUNDED_ABI.names, "stamps"))
STAMPS = 8
STAMP_START, STAMP_LANDED, STAMP_QUANTIZED, STAMP_STAGED = 0, 1, 2, 3
STAMP_PUSHED, STAMP_SUMMED, STAMP_END = 4, 5, 6


@dataclass(frozen=True)
class DraftProjArKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    rows: int
    tp: int
    k: int
    abi: KernelAbi = DRAFT_PROJ_AR_ABI
    launch_count: int = 1


@traced
def _copy_weights(c, lds, wave):
    """Waves [0, 4): a half of a task's K groups each, one TDM copy."""
    if wave < lay.TASKS * lay.WEIGHT_PARTS:
        task, part = wave % lay.TASKS, wave // lay.TASKS
        part_groups = lay.groups(c["k"]) // lay.WEIGHT_PARTS
        source = fx.Int64(c["bid"] * lay.CTA_COLUMNS + task * ROWS) * fx.Int64(
            c["k"]
        ) + fx.Int64(part * (part_groups * lay.GROUP_BYTES))
        gemv_tdm.load_rows(
            c["w"] + source,
            lay.GROUP_BYTES // 2,
            lds + (task * lay.groups(c["k"]) + part * part_groups) * lay.GROUP_WORDS,
            part_groups,
            part_groups,
            padded=False,
            cache_policy=gemv_tdm._WEIGHT_CACHE_POLICY,
            whole=True,
            width=lay.GROUP_BYTES // 2,
            uniform=True,
        )


def _magnitude_no_nan(words):
    """The largest BF16 magnitude of the pair words' elements, NaNs left
    out (the kernel's amax skips them)."""
    best = fx.Int32(0)
    for word in words:
        for half in (word & 0x7FFF, (word >> 16) & 0x7FFF):
            best = fx.max(best, (half > lay.BF16_INF).select(fx.Int32(0), half))
    return best


@traced
def _magnitude(words):
    """The largest BF16 magnitude of the pair words' elements: the high and
    the low halves' maxima side by side (a word's masked high half decides
    an unsigned compare), then, only where a NaN is among them, the
    element-wise maximum that leaves NaNs out -- as the kernel's two paths."""
    high, low = fx.Int32(0), fx.Int32(0)
    for word in words:
        masked = word & 0x7FFF7FFF
        high = fx.max(high, masked)
        low = fx.max(low, masked << 16)
    magnitude = fx.max(high, low) >> 16
    if magnitude > lay.BF16_INF:
        magnitude = _magnitude_no_nan(words)
    return magnitude


@traced
def _quantize(c, lds, words, token, group, unit, live):
    """A lane's 16 elements of a (token, group) as FP8 into the WMMA operand
    layout, the group's E8M0 code from its eight lanes."""
    rows, k = c["rows"], c["k"]
    magnitude = _magnitude(words)
    for off in (1, 2, 4):
        magnitude = fx.max(magnitude, xshfl(magnitude, off, GFX1250))
    amax = fx.max(
        fx.Int32(lay.AMAX_FLOOR_BITS).bitcast(fx.Float32),
        (magnitude << 16).bitcast(fx.Float32),
    )
    bits = (amax * fx.Int32(lay.INV_FP8_MAX_BITS).bitcast(fx.Float32)).bitcast(fx.Int32)
    code = fx.Int32(
        fx.Uint32((bits & 0x7FFFFFFF) + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)
    )
    code = fx.min(code, fx.Int32(lay.MAX_CODE))
    scale = (code << 23).bitcast(fx.Float32)
    packed = []
    for half in range_constexpr(2):
        eight = fx.Vector.from_elements(
            words[4 * half : 4 * half + 4], fx.Int32
        ).bitcast(fx.BFloat16)
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk8_fp8_bf16(
                T.vec(2, T.i32), as_ir_value(eight), as_ir_value(scale)
            )
        )
        packed.extend(fx.Int32(pair[w]) for w in range_constexpr(2))
    if live:
        at = (
            ((token // ROWS * lay.groups(k) + group) * 8 + unit) * ROWS + token % ROWS
        ) * 4
        fx.ptr_store(
            fx.Vector.from_elements(packed, fx.Int32), lds + lay.token_offset(k) + at
        )
        if unit == 0:
            fx.ptr_store(code, lds + lay.code_offset(rows, k) + group * rows + token)


def _copy_tokens(c, lds, phase):
    """K phase ``phase`` of the activation to the LDS scratch: a wave its
    rows, one TDM copy."""
    rows, k = c["rows"], c["k"]
    width = k // c["phases"]
    gemv_tdm.load_rows(
        c["x"] + fx.Int64(phase * width * 2),
        k,
        lds + lay.scratch_offset(rows, k),
        rows,
        lay.token_rows_a_wave(rows),
        padded=False,
        width=width,
        uniform=True,
    )


def _stage_tokens(c, lds, tid, phase):
    """Every thread's share of the scratch's activation (16-element runs,
    eight consecutive lanes a 128-element group), quantized into the WMMA
    operand layout."""
    rows, k, phases = c["rows"], c["k"], c["phases"]
    count = lay.groups(k) // phases
    total = lay.quant_items(rows, k, phases)
    rounds = lay.quant_rounds(rows, k, phases)
    unit = tid % lay.QUANT_LANES
    scratch = lds + lay.scratch_offset(rows, k)
    quad = fx.Vector.make_type(4, fx.Int32)
    for first in range_constexpr(0, rounds, QUANT_BATCH):
        loaded = []
        for r in range_constexpr(first, min(first + QUANT_BATCH, rounds)):
            item = fx.min(tid + THREADS * r, total - 1)
            run = item // lay.QUANT_LANES
            # a wave's four groups are consecutive tokens of one K group:
            # their LDS stores fall 16 B apart, on distinct banks
            local, token = run // rows, run % rows
            at = (
                token * (k // phases)
                + local * lay.K_GROUP
                + unit * lay.QUANT_LANE_ELEMENTS
            ) // 2
            words = []
            for half in range_constexpr(2):
                raw = fx.Vector(fx.ptr_load(scratch + at + 4 * half, result_type=quad))
                words.extend(fx.Int32(raw[i]) for i in range_constexpr(4))
            loaded.append(
                (words, token, phase * count + local, tid + THREADS * r < total)
            )
        for words, token, group, live in loaded:
            _quantize(c, lds, words, token, group, unit, live)


def _chain_body(c, lds, lane, task, tile):
    rows, k = c["rows"], c["k"]
    count = lay.groups(k)
    tokens = lds + lay.token_offset(k)
    codes = lds + lay.code_offset(rows, k)
    token = fx.min(fx.Int32(tile * ROWS) + lane % ROWS, fx.Int32(rows - 1))
    acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
    for g in range_constexpr(count):
        a = _weight_operand(lds, task * count + g, lane)
        words = []
        for q in range_constexpr(4):
            at = (
                ((tile * count + g) * 8 + q * 2 + lane // ROWS) * ROWS + lane % ROWS
            ) * 4
            raw = fx.Vector(
                fx.ptr_load(tokens + at, result_type=fx.Vector.make_type(4, fx.Int32))
            )
            words.extend(fx.Int32(raw[i]) for i in range_constexpr(4))
        b = fx.Vector.from_elements(words, fx.Int32)
        x_code = fx.Int32(fx.ptr_load(codes + g * rows + token))
        acc = _mma_fp8(a, b, acc, _splat(c["w_codes"][g]), _splat(x_code))
    return acc


@traced
def _chain(c, lds, lane, task, tile, live):
    """A (task, token tile)'s scaled WMMAs over K, the groups in order."""
    acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
    if live:
        acc = _chain_body(c, lds, lane, task, tile)
    return acc


@traced
def _store_stage(c, lds, acc, task, tile, lane, live):
    """A chain's 16 columns of its tile, BF16, a lane its token's eight
    columns: the CTA's ``[token][pair]`` stage."""
    rows, k = c["rows"], c["k"]
    token = tile * ROWS + lane % ROWS
    if live & (token < rows):
        words = [_pack(acc[2 * i], acc[2 * i + 1]) for i in range_constexpr(ACC // 2)]
        fx.ptr_store(
            fx.Vector.from_elements(words, fx.Int32),
            lds
            + lay.stage_offset(rows, k)
            + token * CTA_PAIRS
            + task * (ROWS // 2)
            + (lane // ROWS) * (ACC // 2),
        )


@traced
def _push(c, lds, item):
    """Staged line ``item`` (``[triple][pair]``) of this CTA to every rank."""
    rows = c["rows"]
    if lay.line_live(rows, item):
        triple = item // CTA_PAIRS
        pair = item % CTA_PAIRS
        stage = lds + lay.stage_offset(rows, c["k"])
        words = [
            fx.ptr_load(stage + (triple * LINE_TOKENS + j) * CTA_PAIRS + pair)
            for j in range(LINE_TOKENS)
        ]
        line = fx.Vector.from_elements([*words, c["epoch"]], fx.Int32)
        at = fx.Int64(lay.line_word(c["rank"], rows, c["bank"], c["bid"], item)) * 4
        for base in c["bases"]:
            ar._store_line_sys(base + at, line)


@traced
def _poll_lines_bounded(
    mine, words, epoch, flags, bid, bound, modifier=ar.POLL_VOLATILE, counted=None
):
    """``k2_oproj_ar_gfx1250._poll_lines`` giving up past ``bound`` polls,
    marking ``flags[0]`` (tools and tests only). ``counted`` (a late-push
    build's thread index): thread 0's polls, where any, into ``flags[1]`` (an
    even CTA's) or ``flags[2]`` (an odd one's)."""
    lines = oar._load_lines(mine, words, modifier)
    polls = fx.Int32(0)
    while oar._stale(lines, epoch) & (polls < bound):
        spin_pause()
        lines = oar._load_lines(mine, words, modifier)
        polls = polls + 1
    if polls >= bound:
        st_dev(fx.Int32(lay.GAVE_UP) | bid, flags, 0)
    if const_expr(counted is not None):  # noqa: SIM102
        if (counted == 0) & (polls > 0):
            st_dev(polls, flags, 1 + bid % 2)
    return [[fx.Int32(line[j]) for j in range(LINE_TOKENS)] for line in lines]


def _poll(c, words):
    """The ranks' lines at ``words`` once all carry the epoch. ``together``
    (``ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER``): the loads go out back to
    back and are awaited once (``ar.POLL_TOGETHER``); else each is waited
    out before the next, as ever."""
    if const_expr(c["bound"] is None):
        if const_expr(c["together"]):
            return oar._poll_lines(c["mine"], words, c["epoch"], ar.POLL_TOGETHER)
        return oar._poll_lines(c["mine"], words, c["epoch"])
    return _poll_lines_bounded(
        c["mine"], words, c["epoch"], c["flags"], c["bid"], c["bound"],
        ar.POLL_TOGETHER if c["together"] else ar.POLL_VOLATILE, c["counted"],
    )  # fmt: skip


def _sum_ranks(words):
    """BF16 pair words from every rank summed in rank order in fp32, to a
    pair word, as aiter's all-reduce kernels sum them: their code objects
    run FLOAT_DENORM_MODE_32 0 (f32 denormals flushed, operands and results;
    this kernel runs 3), so the ``v_pk_add_f32`` chain sits between two
    ``s_denorm_mode`` in one asm block. A flushed sum is zero or normal: its
    rounding to BF16 does not depend on the mode. One rank: production runs
    no all-reduce, and the pair passes through."""
    if const_expr(len(words) == 1):
        return oar._sum_ranks(words)
    pairs = [
        fx.Vector.from_elements(list(bf2_f32(word)), fx.Float32).ir_value()
        for word in words
    ]
    asm = "\n ".join(
        [
            "s_denorm_mode 12",
            "v_pk_add_f32 $0, $1, $2",
            *[f"v_pk_add_f32 $0, $0, ${i}" for i in range(3, len(words) + 1)],
            "s_denorm_mode 15",
        ]
    )
    out = fx.Vector(
        llvm.inline_asm(
            T.vec(2, T.f32),
            pairs,
            asm,
            "=&v" + ",v" * len(words),
            has_side_effects=True,
        )
    )
    return _pack(fx.Float32(out[0]), fx.Float32(out[1]))


@traced
def _sum(c, item):
    """This CTA's line ``item`` from every rank's partials, summed in rank
    order in fp32 and landed in the output."""
    rows = c["rows"]
    if lay.line_live(rows, item):
        triple = item // CTA_PAIRS
        # the CTA whose columns this one sums and lands: itself (TEST ONLY:
        # its partner, a late-push build's)
        bid = c["sum_bid"]
        column_pair = bid * CTA_PAIRS + item % CTA_PAIRS
        partials = _poll(
            c,
            [lay.line_word(p, rows, c["bank"], bid, item) for p in range(c["tp"])],
        )
        oar._land_line(
            fx.Int32(1) == 1,
            c["out"],
            rows,
            triple,
            column_pair,
            [_sum_ranks([line[j] for line in partials]) for j in range(LINE_TOKENS)],
        )


def build_draft_proj_ar(
    rows: int,
    tp: int,
    k: int,
    poll_bound: int | None = None,
    profile: bool = False,
    poll_together: bool | None = None,
    test_swap_late: int | None = None,
) -> DraftProjArKernel:
    """The projection and its all-reduce at ``rows`` token rows, ``tp`` ranks
    and input width ``k`` (the rank's shard: o_proj's heads, down_proj's
    intermediate). ``x`` [rows, k] BF16; ``w``/``w_scale`` the rank's
    preshuffled FP8 [HIDDEN, k] and its [HIDDEN / 128, k / 128] E8M0 block
    scales; ``out`` [rows, HIDDEN] BF16; ``peers`` every rank's bank base
    (``draft_mlp_layout.peer_bytes(tp, rows)`` mapped at each), ``epochs``
    ``k2_allreduce_gfx1250``'s counters, ``rank`` this rank. ``poll_bound``:
    a trailing ``flags`` argument takes the give-up mark. ``profile``:
    ``STAMPS`` i32 a CTA to a further trailing ``stamps`` argument.
    ``poll_together`` (default ``ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER`` at
    more than one rank): a poll's loads of the ranks' lines go out together
    and are awaited once (``ar.POLL_TOGETHER``; with the default each is
    waited out before the next: a memory round trip a rank); the same lines
    at the same addresses. ``test_swap_late``: TEST ONLY, with
    ``poll_bound`` -- a CTA sums and lands its partner CTA's columns, the
    odd CTAs pushing that many sleeps late (lines that land while they are
    polled), thread 0's polls into ``flags[1]`` / ``flags[2]``."""
    why = lay.refusal(rows, k, tp)
    if why is not None:
        raise MonoUnsupported(f"draft projection all-reduce: {why}")
    assert lay.bank_words(tp, rows) == oar.oneshot_bank_words(tp, rows)
    count = lay.groups(k)
    tile_count = lay.tiles(rows)
    words_a_bank = lay.bank_words(tp, rows)
    phases = lay.token_phases(rows, k)
    lds_count = lay.lds_words(rows, k, phases)
    bounded = poll_bound is not None
    if poll_together is None:
        poll_together = envs.ATOM_KIMI_MONO_DRAFT_AR_POLL_TOGETHER and tp > 1
    if test_swap_late is not None and (not bounded or test_swap_late < 0):
        raise ValueError("a late-push build counts sleeps and takes a poll bound")
    name = (
        f"k3_draft_proj_ar_gfx1250_tp{tp}_s{rows}_k{k}"
        + ("_pt" if poll_together else "")
        + (f"_b{poll_bound}" if bounded else "")
        + ("" if test_swap_late is None else f"_sw{test_swap_late}")
        + ("_profile" if profile else "")
    )
    sources = SOURCES

    @fx.struct
    class Lds:
        words: fx.Array[fx.Int32, lds_count, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        w: Int64,
        w_scale: Int64,
        out: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        flags: Int64,
        stamps: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % LANES
        wave = uniform(tid // LANES)
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek().words.ptr

        def stamp(index):
            if const_expr(profile):  # noqa: SIM102
                if tid == 0:
                    st_dev(fx.Int32(_realtime()), stamps, bid * STAMPS + index)

        stamp(STAMP_START)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        c = {
            "rows": rows,
            "k": k,
            "tp": tp,
            "phases": phases,
            "bound": poll_bound,
            "together": poll_together,
            "counted": None if test_swap_late is None else tid,
            "bid": bid,
            "sum_bid": bid if test_swap_late is None else bid ^ 1,
            "x": x,
            "w": w,
            "out": out,
            "rank": rank,
            "flags": flags,
            "epoch": epoch,
            "bank": (epoch & 1) * words_a_bank,
        }
        _copy_weights(c, lds, wave)
        _copy_tokens(c, lds, 0)
        c["bases"] = [load_ptr64(peers, p) for p in range(tp)]
        c["mine"] = rsrc(load_ptr64(peers, rank))
        block = uniform(bid) * lay.CTA_COLUMNS // 128
        c["w_codes"] = [_scalar_byte(w_scale, block * count + g) for g in range(count)]
        for phase in range_constexpr(phases):
            rocdl.s_wait_tensorcnt(0)
            gpu.barrier()
            if const_expr(phase == 0):
                stamp(STAMP_LANDED)
            _stage_tokens(c, lds, tid, phase)
            if const_expr(phase + 1 < phases):
                # every wave done reading the scratch before the next
                # phase's copies land in it
                gpu.barrier()
                _copy_tokens(c, lds, phase + 1)
        gpu.barrier()
        stamp(STAMP_QUANTIZED)
        chains = lay.TASKS * tile_count
        for r in range_constexpr(-(-chains // WAVES)):
            index = wave + WAVES * r
            live = index < chains
            task, tile = index % lay.TASKS, index // lay.TASKS
            acc = _chain(c, lds, lane, task, tile, live)
            _store_stage(c, lds, acc, task, tile, lane, live)
        # the last triple's spare tokens carry zeros
        spare = (lay.triples(rows) * LINE_TOKENS - rows) * CTA_PAIRS
        if const_expr(spare > 0):  # noqa: SIM102
            if tid < spare:
                fx.ptr_store(
                    fx.Int32(0),
                    lds + lay.stage_offset(rows, k) + rows * CTA_PAIRS + tid,
                )
        gpu.barrier()
        stamp(STAMP_STAGED)
        # every round's lines pushed before any poll: a load issued after a
        # remote store waits out that store's round trip
        if const_expr(test_swap_late):  # noqa: SIM102
            if bid % 2 == 1:
                ar._sleeps(test_swap_late)
        for r in range_constexpr(lay.line_rounds(rows)):
            _push(c, lds, tid + r * THREADS)
        stamp(STAMP_PUSHED)
        for r in range_constexpr(lay.line_rounds(rows)):
            # a late-push build counts its first line round's polls: the
            # later rounds' lines have landed by then, and their few polls
            # would overwrite the count
            _sum(c if r == 0 else {**c, "counted": None}, tid + r * THREADS)
        stamp(STAMP_SUMMED)
        gpu.barrier()
        stamp(STAMP_END)
        counter = bid + tid * CTAS
        if counter < ar.MAX_CTAS:
            bo.buffer_store(epoch, rsrc(epochs), counter)

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        w_scale: Int64,
        out: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        flags: Int64,
        stamps: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(x, w, w_scale, out, peers, epochs, rank, flags, stamps).launch(
            grid=(CTAS,), block=(THREADS,), stream=stream
        )

    DRAFT_PROJ_AR_PROFILE_ABI.check(kernel, launch)
    if profile:
        abi, launcher = DRAFT_PROJ_AR_PROFILE_ABI, launch
    elif bounded:
        abi, launcher = DRAFT_PROJ_AR_BOUNDED_ABI, ar.TailedLauncher(launch, (0,))
    else:
        abi, launcher = DRAFT_PROJ_AR_ABI, ar.TailedLauncher(launch, (0, 0))
    return DraftProjArKernel(name, launcher, (CTAS,), (THREADS,), rows, tp, k, abi)
