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

Each push is a 16 B line -- one BF16 column pair of three consecutive tokens
and the launch's epoch -- stored at system scope, polled until the tag is
current: no fence, no barrier between the CTAs. A CTA stages its pairs in LDS
to cut them into lines; the last token triple's spare tokens carry zeros.
Epochs are ``k2_allreduce_gfx1250``'s counters, so a launch of either kernel
takes the next epoch; regions of epoch ``e`` live in bank ``e & 1``.

One-shot (``ATOM_KIMI_MONO_OPROJ_AR_ONESHOT``): the CTA pushes its partials to
every rank instead of the owner, and every rank's CTA ``b`` sums its columns'
partials in rank order itself -- the owner's arithmetic, so the output is
bit for bit the same -- with one fabric hop, not two, and no remote store
left to drain at the launch's end. Its banks hold every rank's partials of
every column.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import fresh_prefetch_gfx1250 as fresh
from atom.models.kimi_k3.mono.kernels.stages import gemv_gfx1250, tdm_timeline
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    TDM_CLUSTER,
    tdm_lds_words,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    load_ptr64,
    rsrc,
    spin_pause,
    st_dev,
    traced,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.utils import envs

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
OPROJ_AR_PROFILE_ABI = KernelAbi((*OPROJ_AR_ABI.names, "stamps"))
# a profile build's i32 stamps a CTA: the low half of the 100 MHz clock
STAMPS = 8
STAMP_START, STAMP_STAGED, STAMP_PUSHED = 0, 1, 2
# the partials summed (the owner, or every rank one-shot), the sums pushed,
# the sums landed (a rank not owning the columns), the end
STAMP_SUMMED, STAMP_SUMS_PUSHED, STAMP_LANDED, STAMP_END = 3, 4, 5, 6


def own_pairs(tp: int) -> int:
    """A rank's owned output columns, as BF16 pairs."""
    return HIDDEN // tp // 2


# a line: three tokens' BF16 pair of one column pair, then the epoch
LINE_TOKENS = 3
LINE_WORDS = LINE_TOKENS + 1


def triples(rows: int) -> int:
    return -(-rows // LINE_TOKENS)


def line_rounds(rows: int) -> int:
    """A CTA's ``[triple][pair]`` lines a thread: one up to 48 rows."""
    return -(-triples(rows) * CTA_PAIRS // THREADS)


def bank_words(tp: int, rows: int) -> int:
    """A bank: the partials ``[source rank][token triple][owned pair]``, then
    the sums ``[token triple][pair]``, a line each."""
    return LINE_WORDS * triples(rows) * (tp * own_pairs(tp) + HIDDEN // 2)


def oneshot_bank_words(tp: int, rows: int) -> int:
    """A one-shot bank: the partials ``[source rank][token triple][pair]``."""
    return LINE_WORDS * triples(rows) * tp * (HIDDEN // 2)


def _oneshot(oneshot: bool | None) -> bool:
    return envs.ATOM_KIMI_MONO_OPROJ_AR_ONESHOT if oneshot is None else oneshot


def oproj_peer_bytes(tp: int, rows: int, oneshot: bool | None = None) -> int:
    words = bank_words(tp, rows)
    if _oneshot(oneshot):
        words = max(words, oneshot_bank_words(tp, rows))
    return 2 * words * 4


@dataclass(frozen=True)
class K2OprojAllReduceKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = OPROJ_AR_ABI
    launch_count: int = 1


def _load_lines(mine, words):
    return [
        fx.Vector(
            bo.buffer_load(
                mine,
                word,
                vec_width=LINE_WORDS,
                dtype=T.i32,
                cache_modifier=ar.SCOPE_SYS | ar.VOLATILE,
            )
        )
        for word in words
    ]


def _stale(lines, epoch):
    stale = None
    for line in lines:
        late = fx.Int32(line[LINE_TOKENS]) != epoch
        stale = late if stale is None else stale | late
    return stale


@traced
def _poll_lines(mine, words, epoch):
    """The lines at ``words`` of this rank's region once all carry ``epoch``,
    re-loaded together while any is stale, as their tokens' pair words."""
    lines = _load_lines(mine, words)
    while _stale(lines, epoch):
        spin_pause()
        lines = _load_lines(mine, words)
    return [[fx.Int32(line[j]) for j in range(LINE_TOKENS)] for line in lines]


def _sum_ranks(words):
    """BF16 pairs from every rank summed in rank order in fp32, to a pair."""
    sums = None
    for word in words:
        values = bf2_f32(word)
        sums = values if sums is None else [a + b for a, b in zip(sums, values)]
    return (
        fx.Vector.from_elements(sums, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


def _sum_lines(lines):
    """Every rank's line of the same pair, summed token by token."""
    return [_sum_ranks([line[j] for line in lines]) for j in range(LINE_TOKENS)]


@traced
def _push_line(live, address, words, epoch):
    if live:
        ar._store_line_sys(address, fx.Vector.from_elements([*words, epoch], fx.Int32))


@traced
def _store_out(live, out, word, value):
    if live:
        bo.buffer_store(value, rsrc(out), word)


def _land_line(live, out, rows, triple, column_pair, words):
    for j in range(LINE_TOKENS):
        token = triple * LINE_TOKENS + j
        _store_out(
            live & (token < rows), out, token * (HIDDEN // 2) + column_pair, words[j]
        )


@traced
def _reduce_share(mine, bases, rank, rows, tp, own, bank, epoch, bid, tid, stamp):
    """This CTA's share of the lines this rank owns, ``[triple][owned pair]``
    dealt evenly to the grid: every rank's partials summed and pushed to every
    rank."""
    count = triples(rows)
    items = count * own
    per_cta = -(-items // CTAS)
    sums_base = bank + LINE_WORDS * tp * count * own
    picks = []
    for k in range_constexpr(-(-per_cta // SUM_THREADS)):
        index = tid + k * SUM_THREADS
        item = fx.min(bid * per_cta + index, items - 1)
        picks.append(((index < per_cta) & (bid * per_cta + index < items), item))
    words = [
        bank + LINE_WORDS * ((p * count + item // own) * own + item % own)
        for _, item in picks
        for p in range(tp)
    ]
    partials = _poll_lines(mine, words, epoch)
    stamp(STAMP_SUMMED)
    for k, (live, item) in enumerate(picks):
        totals = _sum_lines(partials[k * tp : (k + 1) * tp])
        column_pair = rank * own + item % own
        at = (
            fx.Int64(
                sums_base + LINE_WORDS * (item // own * (HIDDEN // 2) + column_pair)
            )
            * 4
        )
        for base in bases:
            _push_line(live, base + at, totals, epoch)
    stamp(STAMP_SUMS_PUSHED)


@traced
def _oneshot_sums(mine, out, rows, tp, bank, epoch, bid, tid, stamp):
    """This CTA's columns from every rank's partials (``_reduce_owned``'s
    sums), landed here: no sum leaves the rank. Past 48 rows a thread takes
    a line a round, the rounds in turn."""
    for k in range_constexpr(line_rounds(rows)):
        _oneshot_sum(
            mine, out, rows, tp, bank, epoch, bid, tid if k == 0 else tid + k * THREADS,
            stamp if k == 0 else (lambda *_: None),
        )  # fmt: skip


@traced
def _oneshot_sum(mine, out, rows, tp, bank, epoch, bid, item, stamp):
    count = triples(rows)
    if item < count * CTA_PAIRS:
        triple = item // CTA_PAIRS
        column_pair = bid * CTA_PAIRS + item % CTA_PAIRS
        partials = _poll_lines(
            mine,
            [
                bank + LINE_WORDS * ((p * count + triple) * (HIDDEN // 2) + column_pair)
                for p in range(tp)
            ],
            epoch,
        )
        stamp(STAMP_SUMMED)
        _land_line(
            fx.Int32(1) == 1, out, rows, triple, column_pair, _sum_lines(partials)
        )


@traced
def _reduce_owned(
    mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid, stamp
):  # fmt: skip
    """This CTA's own columns on the rank owning them: every rank's partials
    summed, pushed to every rank and landed here directly."""
    count = triples(rows)
    rounds = -(-count * CTA_PAIRS // THREADS)
    first = (bid * CTA_COLUMNS - rank * own * 2) // 2
    sums_base = bank + LINE_WORDS * tp * count * own
    items = []
    for k in range_constexpr(rounds):
        item = fx.min(tid + k * THREADS, count * CTA_PAIRS - 1)
        items.append(
            (tid + k * THREADS < count * CTA_PAIRS, item // CTA_PAIRS, item % CTA_PAIRS)
        )
    words = [
        bank + LINE_WORDS * ((p * count + triple) * own + first + pair)
        for _, triple, pair in items
        for p in range(tp)
    ]
    partials = _poll_lines(mine, words, epoch)
    stamp(STAMP_SUMMED)
    for k, (live, triple, pair) in enumerate(items):
        totals = _sum_lines(partials[k * tp : (k + 1) * tp])
        column_pair = bid * CTA_PAIRS + pair
        at = (
            fx.Int64(sums_base + LINE_WORDS * (triple * (HIDDEN // 2) + column_pair))
            * 4
        )
        for base in bases:
            _push_line(live, base + at, totals, epoch)
        _land_line(live, out, rows, triple, column_pair, totals)
    stamp(STAMP_SUMS_PUSHED)


@traced
def _owner_sums(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid, stamp):
    if bid // (CTAS // tp) == rank:
        _reduce_owned(
            mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid, stamp
        )  # fmt: skip
    else:
        _land_sums(mine, out, rows, tp, own, bank, epoch, bid, tid, THREADS, stamp)


@traced
def _split_waves(mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid, stamp):
    if tid < SUM_THREADS:
        _reduce_share(
            mine, bases, rank, rows, tp, own, bank, epoch, bid, tid, stamp
        )  # fmt: skip
    else:
        _land_sums(
            mine, out, rows, tp, own, bank, epoch, bid, tid - SUM_THREADS,
            LAND_THREADS, stamp,
        )  # fmt: skip


@traced
def _land_sums(mine, out, rows, tp, own, bank, epoch, bid, tid, threads, stamp=None):
    """This CTA's columns from their owner's sums, on ``threads`` lanes."""
    count = triples(rows)
    rounds = -(-count * CTA_PAIRS // threads)
    sums_base = bank + LINE_WORDS * tp * count * own
    items = []
    for k in range_constexpr(rounds):
        item = fx.min(tid + k * threads, count * CTA_PAIRS - 1)
        items.append(
            (
                tid + k * threads < count * CTA_PAIRS,
                item // CTA_PAIRS,
                bid * CTA_PAIRS + item % CTA_PAIRS,
            )
        )
    sums = _poll_lines(
        mine,
        [
            sums_base + LINE_WORDS * (triple * (HIDDEN // 2) + column_pair)
            for _, triple, column_pair in items
        ],
        epoch,
    )
    if const_expr(stamp is not None):
        stamp(STAMP_LANDED, tid)
    for (live, triple, column_pair), words in zip(items, sums):
        _land_line(live, out, rows, triple, column_pair, words)


@traced
def _store_stage(live, address, word):
    if live:
        fx.ptr_store(word, address)


@traced
def _push_oneshot(stage, bases, rank, rows, bank, epoch, bid, item):
    """Staged line ``item`` (``[triple][pair]``) of this CTA to every rank."""
    if item < triples(rows) * CTA_PAIRS:
        triple = item // CTA_PAIRS
        words = [
            fx.ptr_load(
                stage + (triple * LINE_TOKENS + j) * CTA_PAIRS + item % CTA_PAIRS
            )
            for j in range(LINE_TOKENS)
        ]
        line = fx.Vector.from_elements([*words, epoch], fx.Int32)
        at = (
            fx.Int64(
                bank
                + LINE_WORDS
                * (
                    (rank * triples(rows) + triple) * (HIDDEN // 2)
                    + bid * CTA_PAIRS
                    + item % CTA_PAIRS
                )
            )
            * 4
        )
        for base in bases:
            ar._store_line_sys(base + at, line)


@traced
def _push_owner(stage, owner_base, rank, rows, own, bank, epoch, row0, owner, item):
    """Staged line ``item`` (``[triple][pair]``) of this CTA, its first column
    ``row0``, to rank ``owner``, owning its columns."""
    if item < triples(rows) * CTA_PAIRS:
        triple = item // CTA_PAIRS
        pair = item % CTA_PAIRS
        words = [
            fx.ptr_load(stage + (triple * LINE_TOKENS + j) * CTA_PAIRS + pair)
            for j in range(LINE_TOKENS)
        ]
        owned = (row0 - owner * (2 * own)) // 2 + pair
        at = (
            fx.Int64(
                bank + LINE_WORDS * ((rank * triples(rows) + triple) * own + owned)
            )
            * 4
        )
        ar._store_line_sys(
            owner_base + at, fx.Vector.from_elements([*words, epoch], fx.Int32)
        )


def build_oproj_allreduce(
    plan: Gfx1250KimiBuildPlan,
    tp: int,
    proj: int,
    oneshot: bool | None = None,
    profile: bool = False,
) -> K2OprojAllReduceKernel:
    """``oneshot``: one fabric hop (default ``ATOM_KIMI_MONO_OPROJ_AR_ONESHOT``;
    the banks then need ``oproj_peer_bytes(tp, rows, True)``). ``profile``:
    ``STAMPS`` i32 a CTA to a trailing ``stamps`` argument."""
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
    oneshot = _oneshot(oneshot)
    words_a_bank = oneshot_bank_words(tp, rows) if oneshot else bank_words(tp, rows)
    abi = OPROJ_AR_PROFILE_ABI if profile else OPROJ_AR_ABI

    @fx.struct
    class Lds:
        gemv: fx.Array[
            fx.Int32,
            sum(tdm_lds_words(rows, 2, True, depth=gemv_tdm.FP8_DEPTH)),
            16,
        ]
        red: fx.Array[fx.Float32, 2 * tiles * ACCUMULATOR_WORDS, 16]
        # the CTA's pairs ``[token][pair]``, cut into lines once all are in
        stage: fx.Array[fx.Int32, triples(rows) * LINE_TOKENS * CTA_PAIRS, 16]

    name = (
        f"k3_k2_gfx1250_oproj_ar_tp{tp}_s{rows}{'_os' if oneshot else ''}"
        f"{'_profile' if profile else ''}{fresh.suffix('oproj')}"
        f"{tdm_timeline.suffix()}"
    )
    sources = SOURCES
    timeline = tdm_timeline.key()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        out: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stamps: Int64,
    ):
        _ = sources, timeline
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        fresh.prefetch(
            "oproj", core, rows * proj * 2, bid, tid, CTAS, THREADS, TDM_CLUSTER
        )
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        tdm_timeline.open_cta(bid, wave, timeline, allocator)

        def stamp(index, local=None):
            # thread 0's view (``local``: the thread's index in its group)
            if const_expr(profile):  # noqa: SIM102
                if (tid if local is None else local) == 0:
                    st_dev(fx.Int32(_realtime()), stamps, bid * STAMPS + index)

        stamp(STAMP_START)
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
            _store_stage(
                local_token < min(ROWS, rows - tile * ROWS),
                lds.stage.ptr + token * CTA_PAIRS + task * ROWS // 2 + row_pair,
                word,
            )
        # the last triple's spare tokens carry zeros
        spare = (triples(rows) * LINE_TOKENS - rows) * CTA_PAIRS
        if const_expr(spare > 0):  # noqa: SIM102
            if tid < spare:
                fx.ptr_store(fx.Int32(0), lds.stage.ptr + rows * CTA_PAIRS + tid)
        gpu.barrier()
        stamp(STAMP_STAGED)
        # past 48 rows a thread pushes a line a round, every round's before
        # any poll
        if const_expr(oneshot):
            for k in range_constexpr(line_rounds(rows)):
                _push_oneshot(
                    lds.stage.ptr, bases, rank, rows, bank, epoch, bid,
                    tid if k == 0 else tid + k * THREADS,
                )  # fmt: skip
            stamp(STAMP_PUSHED)
            _oneshot_sums(mine, out, rows, tp, bank, epoch, bid, tid, stamp)
        else:
            for k in range_constexpr(line_rounds(rows)):
                _push_owner(
                    lds.stage.ptr, owner_base, rank, rows, own, bank, epoch,
                    row0, owner, tid if k == 0 else tid + k * THREADS,
                )  # fmt: skip
        if const_expr(not oneshot):
            stamp(STAMP_PUSHED)
            if const_expr(rows > SPREAD_ROWS):
                _split_waves(
                    mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid, stamp
                )
            else:
                _owner_sums(
                    mine, bases, out, rank, rows, tp, own, bank, epoch, bid, tid, stamp
                )
        gpu.barrier()
        stamp(STAMP_END)
        counter = bid + tid * CTAS
        if counter < ar.MAX_CTAS:
            bo.buffer_store(epoch, rsrc(epochs), counter)
        tdm_timeline.close_cta()

    @flyc.jit
    def launch(
        core: Int64,
        w_o: Int64,
        w_o_scale: Int64,
        out: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stamps: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources, timeline
        kernel(
            core,
            w_o,
            w_o_scale,
            out,
            peers,
            epochs,
            rank,
            stamps,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(CTAS,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    OPROJ_AR_PROFILE_ABI.check(kernel, launch)
    launcher = launch if profile else ar.TailedLauncher(launch, (0,))
    return K2OprojAllReduceKernel(name, launcher, (CTAS,), (THREADS,), abi)
