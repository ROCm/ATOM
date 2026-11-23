# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A TP all-reduce of BF16 rows in two shots over the TP group's symmetric
(torch.symm_mem) buffers, from a gfx1250 kernel's resident grid.

``Layout``: rows of ``row_pairs`` BF16 pairs; CTA ``b`` holds pairs
``b * cta_pairs ..`` of every row, and rank ``r`` owns pairs ``r * own ..``
(``own = row_pairs / tp``), whole CTAs of them. Each partial pair goes to its
owner only (one eighth of an all-to-all push), the owner sums the ranks'
partials in rank order in fp32 -- AITER's custom all-reduce's arithmetic -- and
pushes the sum to every rank, where the CTA holding it lands it.

Past ``SPREAD_ROWS`` rows the sums are dealt to every CTA of the owning rank on
waves ``[0, 4)`` while waves ``[4, 8)`` land them -- a wave's loads issued
after its remote stores wait out their round trip; below, the CTA holding the
columns sums them on their owner and lands them directly.

Every push is an 8 B pair -- a BF16 pair and the launch's epoch -- stored at
system scope and polled until its tag is current: no fence, no barrier between
the CTAs. A bank: the partials ``[source rank][row][owned pair]``, then the
sums ``[row][pair]``, two words a pair.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.mono.device.ops import bf2_f32, load_ptr64, rsrc, spin_pause, traced
from atom.mono.runtime.abi import KernelAbi

THREADS = 256
SPREAD_ROWS = 24
SUM_THREADS = THREADS // 2
LAND_THREADS = THREADS - SUM_THREADS


@dataclass(frozen=True)
class Layout:
    tp: int
    rows: int
    row_pairs: int
    cta_pairs: int
    ctas: int

    def __post_init__(self):
        if self.ctas * self.cta_pairs != self.row_pairs:
            raise ValueError("the CTAs must hold every pair of a row once")
        if self.ctas % self.tp:
            raise ValueError(f"a rank's pairs must be whole CTAs at TP{self.tp}")

    @property
    def own(self) -> int:
        return self.row_pairs // self.tp

    @property
    def bank_words(self) -> int:
        return 2 * (self.tp * self.rows * self.own + self.rows * self.row_pairs)

    @property
    def peer_bytes(self) -> int:
        return 2 * self.bank_words * 4


def store_pair_sys(address, pair):
    """One 8 B pair to a peer: a global b64 store at system scope."""
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
def poll_pairs(mine, words, epoch):
    """The pairs at ``words`` of this rank's region once all carry ``epoch``,
    re-loaded together while any is stale."""
    pairs = _load_pairs(mine, words)
    while _stale(pairs, epoch):
        spin_pause()
        pairs = _load_pairs(mine, words)
    return [fx.Int32(pair[0]) for pair in pairs]


def sum_ranks(words):
    """BF16 pairs from every rank summed in rank order in fp32, to a pair."""
    sums = None
    for word in words:
        values = bf2_f32(word)
        sums = values if sums is None else [a + b for a, b in zip(sums, values)]
    return (
        fx.Vector.from_elements(sums, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


@traced
def push_live(live, address, pair):
    if live:
        store_pair_sys(address, pair)


@traced
def store_live(live, out, word, value):
    if live:
        bo.buffer_store(value, rsrc(out), word)


def partial_address(layout, bank, owner_base, rank, row, cta, pair):
    """Where pair ``pair`` of CTA ``cta``'s row ``row`` goes on its owner."""
    owned = cta * layout.cta_pairs - cta // (layout.ctas // layout.tp) * layout.own
    return (
        owner_base
        + fx.Int64(bank + 2 * ((rank * layout.rows + row) * layout.own + owned + pair))
        * 4
    )


def owner_base(layout, bases, cta):
    owner = cta // (layout.ctas // layout.tp)
    base = bases[0]
    for p in range_constexpr(1, layout.tp):
        base = (owner == p).select(bases[p], base)
    return base


@traced
def _reduce_share(layout, mine, bases, rank, bank, epoch, bid, tid):
    """This CTA's share of the pairs this rank owns, ``[row][owned pair]``
    dealt evenly to the grid: summed and pushed to every rank."""
    items = layout.rows * layout.own
    per_cta = -(-items // layout.ctas)
    sums = bank + 2 * layout.tp * layout.rows * layout.own
    picks = []
    for k in range_constexpr(-(-per_cta // SUM_THREADS)):
        index = tid + k * SUM_THREADS
        item = fx.min(bid * per_cta + index, items - 1)
        picks.append(((index < per_cta) & (bid * per_cta + index < items), item))
    words = [
        bank
        + 2 * ((p * layout.rows + item // layout.own) * layout.own + item % layout.own)
        for _, item in picks
        for p in range(layout.tp)
    ]
    partials = poll_pairs(mine, words, epoch)
    for k, (live, item) in enumerate(picks):
        total = sum_ranks(partials[k * layout.tp : (k + 1) * layout.tp])
        pair = rank * layout.own + item % layout.own
        at = fx.Int64(sums + 2 * (item // layout.own * layout.row_pairs + pair)) * 4
        out_pair = fx.Vector.from_elements([total, epoch], fx.Int32)
        for base in bases:
            push_live(live, base + at, out_pair)


@traced
def _reduce_owned(layout, mine, bases, out, rank, bank, epoch, bid, tid):
    """This CTA's own pairs on the rank owning them: summed, pushed to every
    rank and landed here directly."""
    count = layout.rows * layout.cta_pairs
    first = bid * layout.cta_pairs - rank * layout.own
    sums = bank + 2 * layout.tp * layout.rows * layout.own
    items = []
    for k in range_constexpr(-(-count // THREADS)):
        item = fx.min(tid + k * THREADS, count - 1)
        items.append(
            (
                tid + k * THREADS < count,
                item // layout.cta_pairs,
                item % layout.cta_pairs,
            )
        )
    words = [
        bank + 2 * ((p * layout.rows + row) * layout.own + first + pair)
        for _, row, pair in items
        for p in range(layout.tp)
    ]
    partials = poll_pairs(mine, words, epoch)
    for k, (live, row, pair) in enumerate(items):
        total = sum_ranks(partials[k * layout.tp : (k + 1) * layout.tp])
        word = row * layout.row_pairs + bid * layout.cta_pairs + pair
        at = fx.Int64(sums + 2 * word) * 4
        out_pair = fx.Vector.from_elements([total, epoch], fx.Int32)
        for base in bases:
            push_live(live, base + at, out_pair)
        store_live(live, out, word, total)


@traced
def _land_sums(layout, mine, out, bank, epoch, bid, tid, threads):
    """This CTA's pairs from their owner's sums, on ``threads`` lanes."""
    count = layout.rows * layout.cta_pairs
    sums = bank + 2 * layout.tp * layout.rows * layout.own
    items = []
    for k in range_constexpr(-(-count // threads)):
        item = fx.min(tid + k * threads, count - 1)
        items.append(
            (
                tid + k * threads < count,
                item // layout.cta_pairs * layout.row_pairs
                + bid * layout.cta_pairs
                + item % layout.cta_pairs,
            )
        )
    values = poll_pairs(mine, [sums + 2 * word for _, word in items], epoch)
    for (live, word), value in zip(items, values):
        store_live(live, out, word, value)


@traced
def _owner_sums(layout, mine, bases, out, rank, bank, epoch, bid, tid):
    if bid // (layout.ctas // layout.tp) == rank:
        _reduce_owned(layout, mine, bases, out, rank, bank, epoch, bid, tid)
    else:
        _land_sums(layout, mine, out, bank, epoch, bid, tid, THREADS)


@traced
def _split_waves(layout, mine, bases, out, rank, bank, epoch, bid, tid):
    if tid < SUM_THREADS:
        _reduce_share(layout, mine, bases, rank, bank, epoch, bid, tid)
    else:
        _land_sums(layout, mine, out, bank, epoch, bid, tid - SUM_THREADS, LAND_THREADS)


def sum_and_land(layout, mine, bases, out, rank, bank, epoch, bid, tid):
    """Once the partials are pushed: the owner's sums, landed in ``out`` (a
    ``[rows][row_pairs]`` word array) at this CTA's pairs."""
    if const_expr(layout.rows > SPREAD_ROWS):
        _split_waves(layout, mine, bases, out, rank, bank, epoch, bid, tid)
    else:
        _owner_sums(layout, mine, bases, out, rank, bank, epoch, bid, tid)


PAIR_ALLREDUCE_PAIRS = ar.REDUCE_WIDTH // 2
PAIR_ALLREDUCE_CTAS = 224


def pair_allreduce_layout(tp: int, rows: int) -> Layout:
    """K2's ``[routed | shared]`` rows, 48 columns a CTA."""
    return Layout(
        tp,
        rows,
        PAIR_ALLREDUCE_PAIRS,
        PAIR_ALLREDUCE_PAIRS // PAIR_ALLREDUCE_CTAS,
        PAIR_ALLREDUCE_CTAS,
    )


@dataclass(frozen=True)
class K2PairAllReduceKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    ctas: int
    abi: KernelAbi = ar.ALLREDUCE_ABI
    launch_count: int = 1


def build_pair_allreduce(tp: int, rows: int) -> K2PairAllReduceKernel:
    """``k2_allreduce_gfx1250.build_allreduce``'s sum in place, in two shots."""
    layout = pair_allreduce_layout(tp, rows)
    assert layout.ctas <= ar.MAX_CTAS
    count = rows * layout.cta_pairs
    name = f"k3_k2_gfx1250_pair_allreduce_tp{tp}_s{rows}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(data: Int64, peers: Int64, epochs: Int64, rank: Int32):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        bank = (epoch & 1) * layout.bank_words
        bases = [load_ptr64(peers, p) for p in range(tp)]
        mine = rsrc(load_ptr64(peers, rank))
        target = owner_base(layout, bases, bid)
        local = rsrc(data)
        held = []
        for k in range_constexpr(-(-count // THREADS)):
            item = fx.min(tid + k * THREADS, count - 1)
            row = item // layout.cta_pairs
            pair = item % layout.cta_pairs
            word = fx.Int32(
                bo.buffer_load(
                    local,
                    row * layout.row_pairs + bid * layout.cta_pairs + pair,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            held.append((tid + k * THREADS < count, row, pair, word))
        for live, row, pair, word in held:
            push_live(
                live,
                partial_address(layout, bank, target, rank, row, bid, pair),
                fx.Vector.from_elements([word, epoch], fx.Int32),
            )
        sum_and_land(layout, mine, bases, data, rank, bank, epoch, bid, tid)
        gpu.barrier()
        counter = bid + tid * layout.ctas
        if counter < ar.MAX_CTAS:
            bo.buffer_store(epoch, rsrc(epochs), counter)

    @flyc.jit
    def launch(
        data: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(data, peers, epochs, rank).launch(
            grid=(layout.ctas,), block=(THREADS,), stream=stream
        )

    ar.ALLREDUCE_ABI.check(kernel, launch)
    return K2PairAllReduceKernel(name, launch, (layout.ctas,), (THREADS,), layout.ctas)
