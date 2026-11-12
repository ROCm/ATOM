# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2a's o_proj and its TP all-reduce in one gfx1250 launch: the gated core rows
(``[rows, proj]`` BF16, this rank's heads) to the attention output
(``[rows, HIDDEN]`` BF16, summed over the ranks), as gfx950's ``attn_post``
takes them.

CTA ``b`` owns o_proj rows ``32 b .. 32 b + 32`` -- HIDDEN columns of the
output -- through ``gemv_tdm_gfx1250`` (FP8 weights in AITER's preshuffle and
their E8M0 block scales). A rank owns ``HIDDEN / tp`` columns: the CTA pushes
its BF16 partials to the owning rank only (one eighth of an all-to-all push),
where the same CTA sums the ranks' partials in rank order in fp32 -- AITER's
custom all-reduce's arithmetic -- and pushes the sum to every rank, whose CTA
``b`` lands it in the output.

Each push is an 8 B pair -- a BF16 pair and the launch's epoch -- stored at
system scope, polled until the tag is current: no fence, no barrier between
the CTAs. Epochs are ``k2_allreduce_gfx1250``'s counters, so a launch of
either kernel takes the next epoch; regions of epoch ``e`` live in bank
``e & 1``.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_words
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    load_ptr64,
    rsrc,
    spin_pause,
    traced,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = 256
WAVES = 8
ROWS = 16
CTA_COLUMNS = 2 * ROWS
CTAS = HIDDEN // CTA_COLUMNS
CTA_PAIRS = CTA_COLUMNS // 2
# Past SPREAD_ROWS rows the sums are dealt to every CTA of the owning rank on
# waves [0, 4) while waves [4, 8) land them -- a wave's loads issued after its
# remote stores wait out their round trip; below, the CTA owning the columns
# sums them on its owning rank and lands them directly.
SPREAD_ROWS = 24
SUM_THREADS = THREADS // 2
LAND_THREADS = THREADS - SUM_THREADS
ACCUMULATOR_WORDS = WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS
assert THREADS == 2 * ROWS * 8
assert CTAS <= ar.MAX_CTAS

OPROJ_AR_ABI = KernelAbi(("core", "w_o", "w_o_scale", "out", "peers", "epochs", "rank"))


def own_pairs(tp: int) -> int:
    """A rank's owned output columns, as BF16 pairs."""
    return HIDDEN // tp // 2


def bank_words(tp: int, rows: int) -> int:
    """A bank: the partials ``[source rank][row][owned pair]``, then the sums
    ``[row][pair]``, two words a pair."""
    return 2 * (tp * rows * own_pairs(tp) + rows * HIDDEN // 2)


def oproj_peer_bytes(tp: int, rows: int) -> int:
    return 2 * bank_words(tp, rows) * 4


@dataclass(frozen=True)
class K2OprojAllReduceKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = OPROJ_AR_ABI
    launch_count: int = 1


def _store_pair_sys(address, pair):
    llvm.InlineAsmOp(
        None,
        [as_ir_value(fx.Int64(address)), as_ir_value(pair)],
        "global_store_b64 $0, $1, off scope:SCOPE_SYS",
        "v,v",
        has_side_effects=True,
    )


def _load_pairs(mine, words):
    return [
        fx.Vector(
            bo.buffer_load(
                mine,
                word,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=ar.SCOPE_SYS | ar.VOLATILE,
            )
        )
        for word in words
    ]


def _stale(pairs, epoch):
    stale = None
    for pair in pairs:
        late = fx.Int32(pair[1]) != epoch
        stale = late if stale is None else stale | late
    return stale


@traced
def _poll_pairs(mine, words, epoch):
    """The pairs at ``words`` of this rank's region once all carry ``epoch``,
    re-loaded together while any is stale."""
    pairs = _load_pairs(mine, words)
    while _stale(pairs, epoch):
        spin_pause()
        pairs = _load_pairs(mine, words)
    return [fx.Int32(pair[0]) for pair in pairs]


def _sum_ranks(words):
    """BF16 pairs from every rank summed in rank order in fp32, to a pair."""
    sums = None
    for word in words:
        values = bf2_f32(word)
        sums = values if sums is None else [a + b for a, b in zip(sums, values)]
    return (
        fx.Vector.from_elements(sums, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


@traced
def _push_partial(live, address, pair):
    if live:
        _store_pair_sys(address, pair)


@traced
def _reduce_share(mine, bases, rank, rows, tp, own, bank, epoch, bid, tid):
    """This CTA's share of the columns this rank owns, ``[row][owned pair]``
    dealt evenly to the grid: every rank's partials summed and pushed to every
    rank."""
    items = rows * own
    per_cta = -(-items // CTAS)
    sums_base = bank + 2 * tp * rows * own
    picks = []
    for k in range_constexpr(-(-per_cta // SUM_THREADS)):
        index = tid + k * SUM_THREADS
        item = fx.min(bid * per_cta + index, items - 1)
        picks.append(((index < per_cta) & (bid * per_cta + index < items), item))
    words = [
        bank + 2 * ((p * rows + item // own) * own + item % own)
        for _, item in picks
        for p in range(tp)
    ]
    partials = _poll_pairs(mine, words, epoch)
    for k, (live, item) in enumerate(picks):
        total = _sum_ranks(partials[k * tp : (k + 1) * tp])
        token = item // own
        column_pair = rank * own + item % own
        at = fx.Int64(sums_base + 2 * (token * (HIDDEN // 2) + column_pair)) * 4
        out_pair = fx.Vector.from_elements([total, epoch], fx.Int32)
        for base in bases:
            _push_partial(live, base + at, out_pair)


@traced
def _reduce_owned(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid):
    """This CTA's own columns on the rank owning them: every rank's partials
    summed, pushed to every rank and landed here directly."""
    rounds = -(-rows * CTA_PAIRS // THREADS)
    first = (bid * CTA_COLUMNS - rank * own * 2) // 2
    sums_base = bank + 2 * tp * rows * own
    items = []
    for k in range_constexpr(rounds):
        item = fx.min(tid + k * THREADS, rows * CTA_PAIRS - 1)
        items.append(
            (tid + k * THREADS < rows * CTA_PAIRS, item // CTA_PAIRS, item % CTA_PAIRS)
        )
    words = [
        bank + 2 * ((p * rows + token) * own + first + pair)
        for _, token, pair in items
        for p in range(tp)
    ]
    partials = _poll_pairs(mine, words, epoch)
    for k, (live, token, pair) in enumerate(items):
        total = _sum_ranks(partials[k * tp : (k + 1) * tp])
        column_pair = bid * CTA_PAIRS + pair
        at = fx.Int64(sums_base + 2 * (token * (HIDDEN // 2) + column_pair)) * 4
        out_pair = fx.Vector.from_elements([total, epoch], fx.Int32)
        for base in bases:
            _push_partial(live, base + at, out_pair)
        _store_out(live, out, token * (HIDDEN // 2) + column_pair, total)


@traced
def _owner_sums(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid):
    if bid // (CTAS // tp) == rank:
        _reduce_owned(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid)
    else:
        _land_sums(mine, out, rows, tp, own, bank, epoch, bid, tid, THREADS)


@traced
def _split_waves(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid):
    if tid < SUM_THREADS:
        _reduce_share(mine, bases, rank, rows, tp, own, bank, epoch, bid, tid)
    else:
        _land_sums(
            mine, out, rows, tp, own, bank, epoch, bid, tid - SUM_THREADS, LAND_THREADS
        )


@traced
def _store_out(live, out, word, value):
    if live:
        bo.buffer_store(value, rsrc(out), word)


@traced
def _land_sums(mine, out, rows, tp, own, bank, epoch, bid, tid, threads):
    """This CTA's columns from their owner's sums, on ``threads`` lanes."""
    rounds = -(-rows * CTA_PAIRS // threads)
    sums_base = bank + 2 * tp * rows * own
    items = []
    for k in range_constexpr(rounds):
        item = fx.min(tid + k * threads, rows * CTA_PAIRS - 1)
        items.append(
            (
                tid + k * threads < rows * CTA_PAIRS,
                item // CTA_PAIRS * (HIDDEN // 2) + bid * CTA_PAIRS + item % CTA_PAIRS,
            )
        )
    sums = _poll_pairs(mine, [sums_base + 2 * word for _, word in items], epoch)
    for (live, word), value in zip(items, sums):
        _store_out(live, out, word, value)


def build_oproj_allreduce(
    plan: Gfx1250KimiBuildPlan, tp: int, proj: int
) -> K2OprojAllReduceKernel:
    if plan.architecture is not GFX1250:
        raise ValueError("the o_proj all-reduce requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("the o_proj all-reduce requires 256 threads/eight waves")
    if CTAS > plan.blocks:
        raise ValueError(f"the o_proj all-reduce needs {CTAS} CTAs")
    if HIDDEN % (tp * CTA_COLUMNS):
        raise ValueError(f"a rank's columns must be whole CTAs at TP{tp}")
    rows = plan.tokens
    tiles = (rows + ROWS - 1) // ROWS
    own = own_pairs(tp)
    words_a_bank = bank_words(tp, rows)

    @fx.struct
    class Lds:
        gemv: fx.Array[
            fx.Int32,
            sum(tdm_lds_words(rows, 2, True, depth=gemv_tdm.FP8_DEPTH)),
            16,
        ]
        red: fx.Array[fx.Float32, 2 * tiles * ACCUMULATOR_WORDS, 16]

    name = f"k3_k2_gfx1250_oproj_ar_tp{tp}_s{rows}"
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        out: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        lds = fx.SharedAllocator().allocate(Lds).peek()
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        bank = (epoch & 1) * words_a_bank
        bases = [load_ptr64(peers, p) for p in range(tp)]
        mine = rsrc(load_ptr64(peers, rank))
        row0 = bid * CTA_COLUMNS
        owner = bid // (CTAS // tp)
        owner_base = bases[0]
        for p in range_constexpr(1, tp):
            owner_base = (owner == p).select(bases[p], owner_base)
        accumulators = gemv_tdm.project(
            lds.gemv.ptr,
            core,
            proj,
            rows,
            w_o,
            HIDDEN,
            proj,
            [row0, row0 + ROWS],
            lane,
            wave,
            w_scale=w_o_scale,
            depth=gemv_tdm.FP8_DEPTH,
        )
        for tile in range_constexpr(tiles):
            for index in range_constexpr(2):
                gemv_gfx1250.store_accumulator(
                    accumulators[index][tile],
                    lds.red.ptr + (2 * tile + index) * ACCUMULATOR_WORDS,
                    wave,
                    lane,
                )
        gpu.barrier()
        task = tid // (ROWS * 8)
        local_token = tid % (ROWS * 8) // 8
        row_pair = tid % 8
        for tile in range_constexpr(tiles):
            slot = lds.red.ptr + (2 * tile + task) * ACCUMULATOR_WORDS
            token = tile * ROWS + local_token
            low = bf16_round(
                gemv_gfx1250.row_sum(slot, 2 * row_pair, local_token, WAVES)
            )
            high = bf16_round(
                gemv_gfx1250.row_sum(slot, 2 * row_pair + 1, local_token, WAVES)
            )
            word = (
                fx.Vector.from_elements([low, high], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
            )
            pair = (row0 - owner * (2 * own)) // 2 + task * ROWS // 2 + row_pair
            at = fx.Int64(bank + 2 * ((rank * rows + token) * own + pair)) * 4
            _push_partial(
                local_token < min(ROWS, rows - tile * ROWS),
                owner_base + at,
                fx.Vector.from_elements([word, epoch], fx.Int32),
            )
        if const_expr(rows > SPREAD_ROWS):
            _split_waves(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid)
        else:
            _owner_sums(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid)
        gpu.barrier()
        counter = bid + tid * CTAS
        if counter < ar.MAX_CTAS:
            bo.buffer_store(epoch, rsrc(epochs), counter)

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        out: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            core,
            w_o,
            w_o_scale,
            out,
            peers,
            epochs,
            rank,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(CTAS,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    OPROJ_AR_ABI.check(kernel, launch)
    return K2OprojAllReduceKernel(name, launch, (CTAS,), (THREADS,))
