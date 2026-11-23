# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's ``[routed | shared]`` all-reduce inside a gfx1250 kernel, one shot over
the TP group's symmetric (torch.symm_mem) buffers.

Each rank pushes its partial as 16 B lines -- three words of payload, six
bf16, and the launch's epoch -- to every rank's receive bank with SCOPE_SYS
stores, then reduces the lines it pushed: it polls every source's copy of a
line until the copy carries the epoch, sums the sources in rank order in fp32
from rank 0's value, as aiter's custom all-reduce does, and rounds to bf16 into
the input in place. A thread writes back only lines it pushed itself, so none
overwrites a line another has yet to read.

Epochs: ``MAX_CTAS`` counters in device memory, all equal: CTA ``b`` reads
counter ``b`` at launch and stores it back one higher into every counter
congruent to ``b`` modulo the grid, as aiter's custom all-reduce keeps its
block flags, so the epoch counts launches whatever the grid. Every rank
launches the same kernels in the same order, so epochs agree across ranks and
CUDA graph replays with no host step. Lines of epoch ``e`` land in bank
``e & 1``: a rank writes that bank again only in launch ``e + 2``, which needs
every peer's launch ``e + 1`` pushes, issued after that peer finished reading
launch ``e``.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T, as_ir_value

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT, TOPK
from atom.models.kimi_k3.mono.kernels.stages import fresh_prefetch_gfx1250 as fresh
from atom.mono.device.ops import (
    bf2_f32,
    bf16_round,
    load_ptr64,
    rsrc,
    spin_pause,
    st_dev,
    traced,
)
from atom.mono.runtime.abi import KernelAbi
from atom.utils import envs

THREADS = 256
MAX_CTAS = 256
# a line: three payload words (six bf16) and the epoch
LINE_WORDS = 4
PAYLOAD_WORDS = 3
LINE_ELEMENTS = 2 * PAYLOAD_WORDS
# gfx12 CPol SCOPE field (bits 4:3): peers on either tray
SCOPE_SYS = 3 << 3
VOLATILE = 1 << 31
# A poll's loads' cache modifier. POLL_VOLATILE, every helper's default: the
# compiler waits out each load before it issues the next, a memory round trip
# a source. POLL_TOGETHER: the loads go out back to back and are awaited
# once; the pause between polls (side-effecting asm) keeps them in the loop.
# A caller opts in by passing ``modifier=POLL_TOGETHER`` to ``_load_lines``,
# ``_poll_lines`` or ``_poll_lines7`` (``k2_oproj_ar_gfx1250``'s alike).
POLL_VOLATILE = SCOPE_SYS | VOLATILE
POLL_TOGETHER = SCOPE_SYS
# a tool build's marks in ``epochs`` past its MAX_CTAS counters: thread 0's
# polls a CTA from POLLS_WORD, then the give-up mark (GAVE_UP | CTA)
POLLS_WORD = 256
GAVE_UP_WORD = 512
GAVE_UP = 0x7AB00000
REDUCE_WIDTH = LATENT + HIDDEN
assert REDUCE_WIDTH % LINE_ELEMENTS == 0

ALLREDUCE_ABI = KernelAbi(("data", "peers", "epochs", "rank"))
GATHER_ALLREDUCE_ABI = KernelAbi(
    ("data", "grouped", "route_rows", "route_weights", "peers", "epochs", "rank")
)
GATHER_ALLREDUCE_PROFILE_ABI = KernelAbi((*GATHER_ALLREDUCE_ABI.names, "stamps"))
# a routed row's words
ROUTED_WORDS = LATENT // 2


def allreduce_lines(rows: int) -> int:
    return rows * REDUCE_WIDTH // LINE_ELEMENTS


def allreduce_threads(rows: int) -> int:
    """A CTA's threads: twice as many once ``MAX_CTAS`` CTAs would leave some
    thread a second line, whose loads would wait out the first line's remote
    stores."""
    return THREADS if allreduce_lines(rows) <= MAX_CTAS * THREADS else 2 * THREADS


def allreduce_ctas(rows: int) -> int:
    return min(MAX_CTAS, -(-allreduce_lines(rows) // allreduce_threads(rows)))


def allreduce_peer_bytes(tp: int, rows: int) -> int:
    """One rank's receive banks: two of ``[source rank][line]``."""
    return 2 * tp * allreduce_lines(rows) * LINE_WORDS * 4


@dataclass(frozen=True)
class K2AllReduceKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    ctas: int
    abi: KernelAbi = ALLREDUCE_ABI
    launch_count: int = 1


class TailedLauncher:
    """``launch`` with trailing arguments the caller does not pass (a
    non-profile build's ``stamps``); its ``_mem_cache`` is ``launch``'s, for
    ``launcher_resources``."""

    def __init__(self, launch, tail):
        self.launch = launch
        self.tail = tail

    def __call__(self, *args, **kwargs):
        return self.launch(*args, *self.tail, **kwargs)

    @property
    def _mem_cache(self):
        return self.launch._mem_cache


# A gathering all-reduce profile build's i32 stamps a CTA (thread 0's line):
# the low half of the 100 MHz clock.
GATHER_STAMPS = 8
GATHER_START, GATHER_ROUTED, GATHER_GATHERED, GATHER_PUSHED = 0, 1, 2, 3
GATHER_SEEN, GATHER_END = 4, 5
# a profile build drains its stores (its pushes' fabric acks, as the tail's
# flag does) after the poll, stamped
GATHER_DRAINED = 6


def _load_lines(mine, ats, modifier=SCOPE_SYS | VOLATILE):
    """Every source's copy of a line. ``modifier``: with VOLATILE (the
    default) the compiler waits out each load before it issues the next, a
    round trip a copy; without, the loads are in flight together."""
    words = []
    for at in ats:
        line = fx.Vector(
            bo.buffer_load(
                mine,
                at,
                vec_width=LINE_WORDS,
                dtype=T.i32,
                cache_modifier=modifier,
            )
        )
        words.extend(line[w] for w in range(LINE_WORDS))
    return fx.Vector.from_elements(words, fx.Int32)


def _stale(words, sources, epoch):
    stale = None
    for p in range(sources):
        late = fx.Int32(words[p * LINE_WORDS + PAYLOAD_WORDS]) != epoch
        stale = late if stale is None else stale | late
    return stale


@traced
def _poll_lines(mine, ats, epoch, modifier=SCOPE_SYS | VOLATILE):
    """Every source's copy of a line (words ``ats`` of this rank's bank) once
    all carry ``epoch``, re-loaded together while any is stale: one round trip
    after the last source lands, not one a source. A copy stays until launch
    ``epoch + 2``, and each 16 B load sees a copy whole."""
    words = _load_lines(mine, ats, modifier)
    while _stale(words, len(ats), epoch):
        spin_pause()
        words = _load_lines(mine, ats, modifier)
    return words


@traced
def _poll_lines_test(mine, ats, key, seven, modifier, bound, marks, bid, tid):
    """Tool builds: ``_poll_lines`` (``seven``: ``_poll_lines7``) giving up
    past ``bound`` re-loads, marking ``marks[GAVE_UP_WORD]``; thread 0's
    re-loads, where any, into ``marks[POLLS_WORD + CTA]``."""
    words = _load_lines(mine, ats, modifier)
    polls = fx.Int32(0)
    while _stale_of(words, len(ats), key, seven) & (polls < bound):
        spin_pause()
        words = _load_lines(mine, ats, modifier)
        polls = polls + 1
    if polls >= bound:
        st_dev(fx.Int32(GAVE_UP) | bid, marks, GAVE_UP_WORD)
    if (tid == 0) & (polls > 0):
        st_dev(polls, marks, POLLS_WORD + bid)
    return words


def _stale_of(words, sources, key, seven):
    return _stale7(words, sources, key) if seven else _stale(words, sources, key)


def poll_context(poll_together, poll_bound, marks, bid, tid):
    """What ``_poll`` takes: None for the default polls (their ISA as ever),
    else the modifier and, for a tool build, the bound and its marks."""
    if not poll_together and poll_bound is None:
        return None
    return {
        "modifier": POLL_TOGETHER if poll_together else POLL_VOLATILE,
        "bound": poll_bound,
        "marks": marks,
        "bid": bid,
        "tid": tid,
    }


def _poll(pc, mine, ats, key, seven=False):
    if pc is None:
        return _poll_lines7(mine, ats, key) if seven else _poll_lines(mine, ats, key)
    if pc["bound"] is None:
        if seven:
            return _poll_lines7(mine, ats, key, pc["modifier"])
        return _poll_lines(mine, ats, key, pc["modifier"])
    return _poll_lines_test(
        mine, ats, key, seven, pc["modifier"], pc["bound"], pc["marks"], pc["bid"],
        pc["tid"],
    )  # fmt: skip


def _sleeps(units):
    """TEST ONLY: ``units`` sleeps of 31 * 64 clocks (about 0.9 us each)."""
    for _ in range(units):
        llvm.InlineAsmOp(None, [], "s_sleep 31", "", has_side_effects=True)


def _swapped(line, lines, threads):
    """TEST ONLY: the line a thread polls and reduces in place of its own --
    its partner CTA's, where both are lines -- so every line is still reduced
    once, by a CTA that did not push it."""
    partner = line ^ threads
    return ((partner < lines) & (line < lines)).select(partner, line)


def _store_line_sys(address, line):
    """One 16 B line to a peer, as aiter's custom all-reduce writes it: a global
    (not buffer) b128 store at system scope."""
    llvm.InlineAsmOp(
        None,
        [as_ir_value(fx.Int64(address)), as_ir_value(line)],
        "global_store_b128 $0, $1, off scope:SCOPE_SYS",
        "v,v",
        has_side_effects=True,
    )


def _push_line(bases, rank, lines, bank, epoch, line, words):
    out = fx.Vector.from_elements(words + [epoch], fx.Int32)
    at = fx.Int64(bank + (rank * lines + line) * LINE_WORDS) * 4
    for base in bases:
        _store_line_sys(base + at, out)


def _local_words(local, line):
    return [
        fx.Int32(
            bo.buffer_load(local, line * PAYLOAD_WORDS + w, vec_width=1, dtype=T.i32)
        )
        for w in range_constexpr(PAYLOAD_WORDS)
    ]


@traced
def _push_lines(local, bases, rank, lines, bank, epoch, line, stride):
    """Lines ``line``, ``line + stride``, .. of this rank's partials, tagged
    ``epoch``, to every rank's bank (words from ``bank``) at ``bases``: loaded
    before the first store, since a load issued after a remote store waits for
    that store's fabric round trip."""
    while line < lines:
        _push_line(bases, rank, lines, bank, epoch, line, _local_words(local, line))
        line = line + stride


def _gather_sum(values, weights):
    """BF16 pair words weighted and summed in route order in fp32 from zero, to
    a bf16 pair: ``moe_gather_reduce``'s arithmetic."""
    low, high = fx.Float32(0.0), fx.Float32(0.0)
    for value, weight in zip(values, weights):
        value_low, value_high = bf2_f32(value)
        low = low + weight * (fx.Float32(0.0) + value_low)
        high = high + weight * (fx.Float32(0.0) + value_high)
    return fx.Int32(
        fx.Vector.from_elements([low, high], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def _routes_of(routes, first_token, token):
    slot = (token - first_token) * (2 * TOPK)
    rows = [fx.ptr_load(routes + slot + 2 * k).bitcast(fx.Int32) for k in range(TOPK)]
    weights = [fx.ptr_load(routes + slot + 2 * k + 1) for k in range(TOPK)]
    return rows, weights


@traced
def _gathered_words(local, grouped, routes, rows, first_token, line):
    """A routed line's words, as AITER's ``moe_gather_reduce`` stores them (the
    last routed line's shared words read as they are). ``routes``: the LDS
    (row, weight) pairs of tokens ``first_token`` and the next. A line within
    one token's row takes a 12 B load a route; one across two re-gathers its
    second token's words."""
    word = line * PAYLOAD_WORDS
    token = fx.min(word // ROUTED_WORDS, first_token + 1)
    column = word % ROUTED_WORDS
    # ROUTED_WORDS is no multiple of PAYLOAD_WORDS: a line at a row's end
    # loads from that end back, never past it (past the last route row's end
    # lies whatever the allocator placed there, or an unmapped page), and its
    # words shift back into place; the words past the row are re-gathered
    # below or are the shared rows'
    start = fx.min(column, ROUTED_WORDS - PAYLOAD_WORDS)
    shift = column - start
    route_rows, weights = _routes_of(routes, first_token, token)
    loaded = [
        fx.Vector(
            bo.buffer_load(
                grouped,
                row * ROUTED_WORDS + start,
                vec_width=PAYLOAD_WORDS,
                dtype=T.i32,
            )
        )
        for row in route_rows
    ]
    plain = _local_words(local, line)
    sums = [
        _gather_sum([fx.Int32(v[w]) for v in loaded], weights)
        for w in range(PAYLOAD_WORDS)
    ]
    words = []
    for w in range_constexpr(PAYLOAD_WORDS):
        value = sums[w]
        for k in range_constexpr(1, PAYLOAD_WORDS):
            value = (shift == k).select(sums[min(w + k, PAYLOAD_WORDS - 1)], value)
        words.append(value)
    second = word + PAYLOAD_WORDS - 1
    if (second // ROUTED_WORDS != word // ROUTED_WORDS) & (
        second < rows * ROUTED_WORDS
    ):
        next_rows, next_weights = _routes_of(routes, first_token, token + 1)
        for w in range_constexpr(1, PAYLOAD_WORDS):
            at = (word + w) % ROUTED_WORDS
            values = [
                fx.Int32(
                    bo.buffer_load(
                        grouped, row * ROUTED_WORDS + at, vec_width=1, dtype=T.i32
                    )
                )
                for row in next_rows
            ]
            crossed = (word + w) // ROUTED_WORDS != word // ROUTED_WORDS
            words[w] = crossed.select(_gather_sum(values, next_weights), words[w])
    return [
        (word + w < rows * ROUTED_WORDS).select(words[w], plain[w])
        for w in range(PAYLOAD_WORDS)
    ]


@traced
def _stage_routes(routes, route_rows, route_weights, rows, first_token, tid):
    """Tokens ``first_token`` and the next's (row, weight) route pairs to LDS."""
    if tid < 2 * TOPK:
        at = fx.min(first_token + tid // TOPK, rows - 1) * TOPK + tid % TOPK
        row = bo.buffer_load(rsrc(route_rows), at, vec_width=1, dtype=T.i32)
        weight = bo.buffer_load(rsrc(route_weights), at, vec_width=1, dtype=T.f32)
        fx.ptr_store(fx.Int32(row).bitcast(fx.Float32), routes + 2 * tid)
        fx.ptr_store(fx.Float32(weight), routes + 2 * tid + 1)
    gpu.barrier()


@traced
def _push_gathered(c, line, routed, live, first_token):
    """``_push_line`` of this thread's line, if ``live``: a ``routed`` one
    gathered. Every thread of the CTA has its words before any pushes: a CU
    with remote stores in flight holds back the gathers' loads behind them."""
    first = second = third = fx.Int32(0)
    if live & routed:
        words = _gathered_words(
            c["local"], c["grouped"], c["routes"], c["rows"], first_token, line
        )
        first = words[0]
        second = words[1]
        third = words[2]
    elif live:
        words = _local_words(c["local"], line)
        first = words[0]
        second = words[1]
        third = words[2]
    gpu.barrier()
    if const_expr(c.get("late")):  # noqa: SIM102
        if c["bid"] % 2 == 1:
            _sleeps(c["late"])
    if const_expr(not c.get("skip_push")):  # noqa: SIM102
        if live:
            _push_line(
                c["bases"],
                c["rank"],
                c["lines"],
                c["bank"],
                c["epoch"],
                line,
                [first, second, third],
            )


@traced
def _reduce_lines(data, mine, tp, lines, bank, epoch, line, last, stride, pc=None):
    """Lines ``line``, ``line + stride``, .. below ``last``: every source's copy
    once it carries ``epoch``, summed in rank order in fp32 from rank 0's, to
    bf16 in this rank's partials. ``pc``: ``poll_context``'s."""
    local = rsrc(data)
    while line < last:
        got = _poll(
            pc,
            mine,
            [bank + (p * lines + line) * LINE_WORDS for p in range(tp)],
            epoch,
        )
        sums = []
        for p in range_constexpr(tp):
            values = []
            for w in range_constexpr(PAYLOAD_WORDS):
                values.extend(bf2_f32(fx.Int32(got[p * LINE_WORDS + w])))
            sums = values if p == 0 else [a + b for a, b in zip(sums, values)]
        for w in range_constexpr(PAYLOAD_WORDS):
            bo.buffer_store(
                fx.Vector.from_elements([sums[2 * w], sums[2 * w + 1]], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0],
                local,
                line * PAYLOAD_WORDS + w,
            )
        line = line + stride


def build_allreduce(tp: int, rows: int) -> K2AllReduceKernel:
    """Lines are dealt to the grid's threads in turn; a thread reduces in place
    only lines it pushed itself."""
    lines = allreduce_lines(rows)
    threads = allreduce_threads(rows)
    ctas = allreduce_ctas(rows)
    stride = ctas * threads
    bank_words = tp * lines * LINE_WORDS
    name = f"k3_k2_gfx1250_allreduce_tp{tp}_s{rows}"

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(data: Int64, peers: Int64, epochs: Int64, rank: Int32):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        bank = (epoch & 1) * bank_words
        bases = [load_ptr64(peers, p) for p in range(tp)]
        mine = rsrc(load_ptr64(peers, rank))
        line = bid * threads + tid
        _push_lines(rsrc(data), bases, rank, lines, bank, epoch, line, stride)
        _reduce_lines(data, mine, tp, lines, bank, epoch, line, fx.Int32(lines), stride)
        gpu.barrier()
        # Every counter, not just this launch's: a launch with more CTAs reads
        # the rest, and tags must not repeat across the CTAs and widths that
        # share the banks.
        counter = bid + tid * ctas
        if counter < MAX_CTAS:
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
            grid=(ctas,), block=(threads,), stream=stream
        )

    ALLREDUCE_ABI.check(kernel, launch)
    return K2AllReduceKernel(name, launch, (ctas,), (threads,), ctas)


def _build_gather_allreduce6(
    tp: int,
    rows: int,
    poll_together: bool = False,
    poll_bound: int | None = None,
    test_skip_push: bool = False,
    test_swap_late: int | None = None,
) -> K2AllReduceKernel:
    """``build_allreduce`` whose routed lines this rank computes itself: AITER's
    MoE gather-reduce of ``grouped`` (the grouped MoE's gemm2 rows, BF16) by
    the routes' rows and weights (``[rows, TOPK]`` i32 and f32), bit for bit,
    in place of reading them -- one launch fewer. A thread takes one line, so
    a CTA's routed lines span at most two tokens, whose routes it stages in
    LDS."""
    lines = allreduce_lines(rows)
    threads = allreduce_threads(rows)
    ctas = allreduce_ctas(rows)
    if lines > ctas * threads or threads * PAYLOAD_WORDS > ROUTED_WORDS:
        raise ValueError(f"the gathering all-reduce takes a line a thread at {rows}")
    routed_lines = -(-rows * LATENT // LINE_ELEMENTS)
    bank_words = tp * lines * LINE_WORDS
    name = f"k3_k2_gfx1250_gather_allreduce_tp{tp}_s{rows}" + _poll_suffix(
        poll_together, poll_bound, test_skip_push, test_swap_late
    )

    @fx.struct
    class Lds:
        routes: fx.Array[fx.Float32, 4 * TOPK, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        data: Int64,
        grouped: Int64,
        route_rows: Int64,
        route_weights: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lds = fx.SharedAllocator().allocate(Lds).peek()
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        bank = (epoch & 1) * bank_words
        bases = [load_ptr64(peers, p) for p in range(tp)]
        mine = rsrc(load_ptr64(peers, rank))
        line = bid * threads + tid
        first_token = bid * threads * PAYLOAD_WORDS // ROUTED_WORDS
        if bid * threads < routed_lines:
            _stage_routes(
                lds.routes.ptr, route_rows, route_weights, rows, first_token, tid
            )
        _push_gathered(
            {
                "lines": lines,
                "rows": rows,
                "bases": bases,
                "rank": rank,
                "bank": bank,
                "epoch": epoch,
                "grouped": rsrc(grouped),
                "routes": lds.routes.ptr,
                "local": rsrc(data),
                "bid": bid,
                "skip_push": test_skip_push,
                "late": test_swap_late,
            },
            line,
            line < routed_lines,
            line < lines,
            first_token,
        )
        _reduce_lines(
            data,
            mine,
            tp,
            lines,
            bank,
            epoch,
            line if test_swap_late is None else _swapped(line, lines, threads),
            fx.Int32(lines),
            lines,
            poll_context(poll_together, poll_bound, epochs, bid, tid),
        )
        gpu.barrier()
        counter = bid + tid * ctas
        if counter < MAX_CTAS:
            bo.buffer_store(epoch, rsrc(epochs), counter)

    @flyc.jit
    def launch(
        data: Int64,
        grouped: Int64,
        route_rows: Int64,
        route_weights: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(data, grouped, route_rows, route_weights, peers, epochs, rank).launch(
            grid=(ctas,), block=(threads,), stream=stream
        )

    GATHER_ALLREDUCE_ABI.check(kernel, launch)
    return K2AllReduceKernel(
        name, launch, (ctas,), (threads,), ctas, GATHER_ALLREDUCE_ABI
    )


# Seven-element lines: seven BF16 and the epoch's low 16 bits. LATENT and
# HIDDEN are multiples of seven, so a line never straddles a row and a routed
# row is two CTAs' lines. A line's slot is rewritten every second launch, so a
# stale copy's tag is two behind, never the current one modulo 2^16. A seventh
# fewer lines than six-element ones keep a CU's pushes at a line a thread
# further: past THREADS lines a CU, the cross-tray pushes slow down markedly.
LINE7_ELEMENTS = 7
LINE7_ROUTED = LATENT // LINE7_ELEMENTS
assert LATENT % LINE7_ELEMENTS == 0 and HIDDEN % LINE7_ELEMENTS == 0
assert LINE7_ROUTED % THREADS == 0


def allreduce7_lines(rows: int) -> int:
    return rows * REDUCE_WIDTH // LINE7_ELEMENTS


def _seven(pointer, element, cache_modifier=0):
    """BF16 ``element .. element + 7`` of ``pointer`` as fp32: one 16 B load
    from the BF16 pair at or before ``element``, shifted down one where the
    line starts on an odd element. It never reads past the line's last pair."""
    loaded = fx.Vector(
        bo.buffer_load(
            pointer,
            element // 2,
            vec_width=LINE_WORDS,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    )
    values = []
    for w in range(LINE_WORDS):
        values.extend(bf2_f32(fx.Int32(loaded[w])))
    odd = element % 2 == 1
    return [odd.select(values[j + 1], values[j]) for j in range(LINE7_ELEMENTS)]


def _pack(low, high):
    return fx.Int32(
        fx.Vector.from_elements([low, high], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def _stale7(words, sources, tag):
    stale = None
    for p in range(sources):
        late = (fx.Int32(words[p * LINE_WORDS + 3]) >> 16) & fx.Int32(0xFFFF) != tag
        stale = late if stale is None else stale | late
    return stale


@traced
def _poll_lines7(mine, ats, tag, modifier=SCOPE_SYS | VOLATILE):
    """``_poll_lines`` of seven-element lines, tagged in their top 16 bits."""
    words = _load_lines(mine, ats, modifier)
    while _stale7(words, len(ats), tag):
        spin_pause()
        words = _load_lines(mine, ats, modifier)
    return words


@traced
def _store_seven(local, element, values):
    """Seven BF16 at ``element``: the whole pairs as words, the odd end alone."""
    if element % 2 == 0:
        for w in range_constexpr(3):
            bo.buffer_store(
                _pack(values[2 * w], values[2 * w + 1]), local, element // 2 + w
            )
        bo.buffer_store(
            values[6].to(fx.BFloat16), local, element + 6, offset_is_bytes=False
        )
    else:
        bo.buffer_store(
            values[0].to(fx.BFloat16), local, element, offset_is_bytes=False
        )
        for w in range_constexpr(3):
            bo.buffer_store(
                _pack(values[2 * w + 1], values[2 * w + 2]),
                local,
                (element + 1) // 2 + w,
            )


@traced
def _push_reduce7(
    local, mine, bases, rank, tp, lines, bank, tag, line, values, stamp, pc=None,
    push=True, target=None,
):  # fmt: skip
    """Line ``line``'s seven values pushed to every rank, then every rank's
    copy reduced in rank order in fp32 and stored back as BF16. ``pc``:
    ``poll_context``'s. TEST ONLY: ``push`` False pushes nothing; ``target``
    is the line polled and reduced in place of ``line``."""
    words = [_pack(values[2 * w], values[2 * w + 1]) for w in range(3)]
    words.append((_pack(values[6], fx.Float32(0.0)) & fx.Int32(0xFFFF)) | (tag << 16))
    out = fx.Vector.from_elements(words, fx.Int32)
    at = fx.Int64(bank + (rank * lines + line) * LINE_WORDS) * 4
    if const_expr(push):
        for base in bases:
            _store_line_sys(base + at, out)
    stamp(GATHER_PUSHED)
    if const_expr(target is not None):
        line = target
    got = _poll(
        pc,
        mine,
        [bank + (p * lines + line) * LINE_WORDS for p in range(tp)],
        tag,
        seven=True,
    )
    stamp(GATHER_SEEN)
    reduced = None
    for p in range_constexpr(tp):
        copy = []
        for w in range_constexpr(3):
            copy.extend(bf2_f32(fx.Int32(got[p * LINE_WORDS + w])))
        copy.append(bf2_f32(fx.Int32(got[p * LINE_WORDS + 3]))[0])
        reduced = copy if p == 0 else [a + b for a, b in zip(reduced, copy)]
    _store_seven(local, line * LINE7_ELEMENTS, [bf16_round(v) for v in reduced])


@traced
def _prefetch_gather7(data, grouped, route_rows, route_weights, rows, lines, bid, tid):
    """``ATOM_KIMI_MONO_XPF=gather``: the lines this CTA gathers, ahead of its
    routes: a routed CTA's TOPK half rows of ``grouped`` at the fused MoE's
    identity routes (row ``token * TOPK + k``; other routes only miss) and its
    token's routes, a shared CTA's lines of ``data``."""
    half = LINE7_ROUTED // 2 * LINE7_ELEMENTS
    per_row = half * 2 // 128
    if bid * THREADS < rows * LINE7_ROUTED:
        token = bid * THREADS // LINE7_ROUTED
        column = (bid * THREADS % LINE7_ROUTED) * LINE7_ELEMENTS
        fresh.prefetch_own(
            "gather",
            lambda i: grouped
            + fx.Int64((token * TOPK + i // per_row) * LATENT + column) * 2
            + fx.Int64(i % per_row * 128),
            TOPK * per_row,
            tid,
            THREADS,
        )
        fresh.prefetch_own(
            "gather",
            lambda i: route_rows + fx.Int64(token * TOPK * 4),
            1,
            tid,
            THREADS,
        )
        fresh.prefetch_own(
            "gather",
            lambda i: route_weights + fx.Int64(token * TOPK * 4),
            1,
            tid,
            THREADS,
        )
    else:
        end = lines * LINE7_ELEMENTS * 2 - 1
        fresh.prefetch_own(
            "gather",
            lambda i: data
            + fx.Int64(fx.min(bid * THREADS * LINE7_ELEMENTS * 2 + i * 128, end)),
            THREADS * LINE7_ELEMENTS * 2 // 128,
            tid,
            THREADS,
        )


def _poll_suffix(poll_together, poll_bound, test_skip_push, test_swap_late):
    """A build's name past the default's: the polls' and the tool knobs'."""
    return (
        ("_pt" if poll_together else "")
        + ("" if poll_bound is None else f"_b{poll_bound}")
        + ("_np" if test_skip_push else "")
        + ("" if test_swap_late is None else f"_sw{test_swap_late}")
    )


def _check_test_knobs(poll_bound, test_skip_push, test_swap_late):
    if test_skip_push and poll_bound is None:
        raise ValueError("a build that pushes nothing takes a poll bound")
    if test_swap_late is not None and test_swap_late < 0:
        raise ValueError("test_swap_late counts sleeps")


def build_gather_allreduce(
    tp: int,
    rows: int,
    profile: bool = False,
    prepush_barrier: bool | None = None,
    poll_together: bool | None = None,
    poll_bound: int | None = None,
    test_skip_push: bool = False,
    test_swap_late: int | None = None,
) -> K2AllReduceKernel:
    """``build_allreduce`` whose routed lines this rank computes itself: AITER's
    MoE gather-reduce of ``grouped`` (the grouped MoE's gemm2 rows, BF16) by
    the routes' rows and weights (``[rows, TOPK]`` i32 and f32), bit for bit,
    in place of reading them -- one launch fewer. Seven-element lines, a line
    a thread, while those fit MAX_CTAS CTAs; six-element ones past that.
    ``profile``: ``GATHER_STAMPS`` i32 a CTA to a trailing ``stamps``
    argument. ``prepush_barrier`` (``ATOM_KIMI_MONO_GATHER_AR_PREPUSH_BARRIER``):
    the CTA's barrier between the gathers and the pushes. ``poll_together``
    (default ``ATOM_KIMI_MONO_GATHER_AR_POLL_TOGETHER`` at more than one
    rank): a poll's loads of the ranks' copies go out together and are
    awaited once (``POLL_TOGETHER``); the same lines at the same addresses.
    Tool builds only, never the runner's: ``poll_bound`` (polls give up past
    it and mark ``epochs[GAVE_UP_WORD]``, thread 0's polls a CTA from
    ``epochs[POLLS_WORD]``: ``epochs`` then holds GAVE_UP_WORD + 1 words),
    ``test_skip_push`` (no line is pushed) and ``test_swap_late`` (a CTA
    reduces its partner CTA's lines, the odd CTAs pushing that many sleeps
    late: lines that land while they are polled)."""
    if poll_together is None:
        poll_together = envs.ATOM_KIMI_MONO_GATHER_AR_POLL_TOGETHER and tp > 1
    _check_test_knobs(poll_bound, test_skip_push, test_swap_late)
    lines = allreduce7_lines(rows)
    if lines > MAX_CTAS * THREADS:
        assert not profile
        return _build_gather_allreduce6(
            tp, rows, poll_together, poll_bound, test_skip_push, test_swap_late
        )
    if prepush_barrier is None:
        prepush_barrier = envs.ATOM_KIMI_MONO_GATHER_AR_PREPUSH_BARRIER
    from atom.models.kimi_k3.mono.kernels.stages.moe_tdm_gfx1250 import _realtime

    ctas = -(-lines // THREADS)
    routed_lines = rows * LINE7_ROUTED
    bank_words = tp * lines * LINE_WORDS
    name = (
        f"k3_k2_gfx1250_gather_allreduce7_tp{tp}_s{rows}"
        f"{'' if prepush_barrier else '_nb'}{'_profile' if profile else ''}"
        f"{fresh.suffix('gather')}"
    ) + _poll_suffix(poll_together, poll_bound, test_skip_push, test_swap_late)

    @fx.struct
    class Lds:
        routes: fx.Array[fx.Float32, 2 * TOPK, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        data: Int64,
        grouped: Int64,
        route_rows: Int64,
        route_weights: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stamps: Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lds = fx.SharedAllocator().allocate(Lds).peek()
        if const_expr(fresh.on("gather")):
            _prefetch_gather7(
                data, grouped, route_rows, route_weights, rows, lines, bid, tid
            )

        def stamp(index):
            if const_expr(profile):
                if tid == 0:
                    st_dev(fx.Int32(_realtime()), stamps, bid * GATHER_STAMPS + index)
                if const_expr(index == GATHER_SEEN):
                    rocdl.s_wait_storecnt(0)
                    if tid == 0:
                        st_dev(
                            fx.Int32(_realtime()),
                            stamps,
                            bid * GATHER_STAMPS + GATHER_DRAINED,
                        )

        stamp(GATHER_START)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        tag = epoch & fx.Int32(0xFFFF)
        bank = (epoch & 1) * bank_words
        bases = [load_ptr64(peers, p) for p in range(tp)]
        mine = rsrc(load_ptr64(peers, rank))
        local = rsrc(data)
        line = bid * THREADS + tid
        live = line < lines
        # a CTA's routed lines are half of one token's row
        routed = bid * THREADS < routed_lines
        token = bid * THREADS // LINE7_ROUTED
        if routed & (tid < TOPK):
            at = token * TOPK + tid
            route_row = bo.buffer_load(rsrc(route_rows), at, vec_width=1, dtype=T.i32)
            route_weight = bo.buffer_load(
                rsrc(route_weights), at, vec_width=1, dtype=T.f32
            )
            fx.ptr_store(
                fx.Int32(route_row).bitcast(fx.Float32), lds.routes.ptr + 2 * tid
            )
            fx.ptr_store(fx.Float32(route_weight), lds.routes.ptr + 2 * tid + 1)
        gpu.barrier()
        stamp(GATHER_ROUTED)
        v0 = v1 = v2 = v3 = v4 = v5 = v6 = fx.Float32(0.0)
        if routed:
            source = rsrc(grouped)
            column = (line % LINE7_ROUTED) * LINE7_ELEMENTS
            picks = [
                _seven(
                    source,
                    fx.ptr_load(lds.routes.ptr + 2 * k).bitcast(fx.Int32) * LATENT
                    + column,
                )
                for k in range(TOPK)
            ]
            weights = [fx.ptr_load(lds.routes.ptr + 2 * k + 1) for k in range(TOPK)]
            sums = []
            for j in range_constexpr(LINE7_ELEMENTS):
                total = fx.Float32(0.0)
                for pick, weight in zip(picks, weights):
                    total = total + weight * (fx.Float32(0.0) + pick[j])
                sums.append(bf16_round(total))
            v0, v1, v2, v3, v4, v5, v6 = sums
        else:
            plain = _seven(local, fx.min(line, lines - 1) * LINE7_ELEMENTS)
            v0, v1, v2, v3, v4, v5, v6 = plain
        # every thread has its words before any pushes: a CU with remote
        # stores in flight holds back the gathers' loads behind them
        if const_expr(prepush_barrier):
            gpu.barrier()
        stamp(GATHER_GATHERED)
        if const_expr(test_swap_late):  # noqa: SIM102
            if bid % 2 == 1:
                _sleeps(test_swap_late)
        if live:
            _push_reduce7(
                local,
                mine,
                bases,
                rank,
                tp,
                lines,
                bank,
                tag,
                line,
                [v0, v1, v2, v3, v4, v5, v6],
                stamp,
                poll_context(poll_together, poll_bound, epochs, bid, tid),
                not test_skip_push,
                None if test_swap_late is None else _swapped(line, lines, THREADS),
            )
        gpu.barrier()
        stamp(GATHER_END)
        counter = bid + tid * ctas
        if counter < MAX_CTAS:
            bo.buffer_store(epoch, rsrc(epochs), counter)

    @flyc.jit
    def launch(
        data: Int64,
        grouped: Int64,
        route_rows: Int64,
        route_weights: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stamps: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            data, grouped, route_rows, route_weights, peers, epochs, rank, stamps
        ).launch(grid=(ctas,), block=(THREADS,), stream=stream)

    abi = GATHER_ALLREDUCE_PROFILE_ABI if profile else GATHER_ALLREDUCE_ABI
    GATHER_ALLREDUCE_PROFILE_ABI.check(kernel, launch)
    launcher = launch if profile else TailedLauncher(launch, (0,))
    return K2AllReduceKernel(name, launcher, (ctas,), (THREADS,), ctas, abi)
