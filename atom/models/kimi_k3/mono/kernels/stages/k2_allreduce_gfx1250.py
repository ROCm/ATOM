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
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN, LATENT
from atom.mono.device.ops import bf2_f32, load_ptr64, rsrc, spin_pause, traced
from atom.mono.runtime.abi import KernelAbi

THREADS = 256
MAX_CTAS = 256
# a line: three payload words (six bf16) and the epoch
LINE_WORDS = 4
PAYLOAD_WORDS = 3
LINE_ELEMENTS = 2 * PAYLOAD_WORDS
# gfx12 CPol SCOPE field (bits 4:3): peers on either tray
SCOPE_SYS = 3 << 3
VOLATILE = 1 << 31
REDUCE_WIDTH = LATENT + HIDDEN
assert REDUCE_WIDTH % LINE_ELEMENTS == 0

ALLREDUCE_ABI = KernelAbi(("data", "peers", "epochs", "rank"))


def allreduce_lines(rows: int) -> int:
    return rows * REDUCE_WIDTH // LINE_ELEMENTS


def allreduce_ctas(rows: int) -> int:
    return min(MAX_CTAS, -(-allreduce_lines(rows) // THREADS))


def shared_line0(rows: int) -> int:
    """The first line wholly inside the shared partials (``build_shared_push``)."""
    return -(-rows * LATENT // LINE_ELEMENTS)


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


def _load_lines(mine, ats):
    """Every source's copy of a line, all loads in flight together."""
    words = []
    for at in ats:
        line = fx.Vector(
            bo.buffer_load(
                mine,
                at,
                vec_width=LINE_WORDS,
                dtype=T.i32,
                cache_modifier=SCOPE_SYS | VOLATILE,
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
def _poll_lines(mine, ats, epoch):
    """Every source's copy of a line (words ``ats`` of this rank's bank) once
    all carry ``epoch``, re-loaded together while any is stale: one round trip
    after the last source lands, not one a source. A copy stays until launch
    ``epoch + 2``, and each 16 B load sees a copy whole."""
    words = _load_lines(mine, ats)
    while _stale(words, len(ats), epoch):
        spin_pause()
        words = _load_lines(mine, ats)
    return words


@traced
def _push_lines(data, peers, rank, tp, lines, bank, epoch, line, last, stride):
    """Lines ``line``, ``line + stride``, .. below ``last`` of this rank's
    partials, tagged ``epoch``, to every rank's bank (words from ``bank``)."""
    local = rsrc(data)
    while line < last:
        words = [
            fx.Int32(
                bo.buffer_load(
                    local, line * PAYLOAD_WORDS + w, vec_width=1, dtype=T.i32
                )
            )
            for w in range_constexpr(PAYLOAD_WORDS)
        ]
        out = fx.Vector.from_elements(words + [epoch], fx.Int32)
        for p in range_constexpr(tp):
            bo.buffer_store(
                out,
                rsrc(load_ptr64(peers, p)),
                bank + (rank * lines + line) * LINE_WORDS,
                cache_modifier=SCOPE_SYS,
            )
        line = line + stride


@traced
def _reduce_lines(data, mine, tp, lines, bank, epoch, line, last, stride):
    """Lines ``line``, ``line + stride``, .. below ``last``: every source's copy
    once it carries ``epoch``, summed in rank order in fp32 from rank 0's, to
    bf16 in this rank's partials."""
    local = rsrc(data)
    while line < last:
        got = _poll_lines(
            mine, [bank + (p * lines + line) * LINE_WORDS for p in range(tp)], epoch
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


def build_allreduce(tp: int, rows: int, routed_only: bool = False) -> K2AllReduceKernel:
    """Lines are dealt to the grid's threads in turn, and a thread reduces in
    place only lines it pushed itself, or lines no CTA of this launch pushes.

    ``routed_only``: push only the lines below ``shared_line0`` -- the rest
    ``build_shared_push`` pushed with this launch's epoch already -- reduce
    those already-landed shared lines first, while the routed ones cross, then
    the routed lines."""
    lines = allreduce_lines(rows)
    ctas = allreduce_ctas(rows)
    stride = ctas * THREADS
    bank_words = tp * lines * LINE_WORDS
    pushed = shared_line0(rows) if routed_only else lines
    name = f"k3_k2_gfx1250_allreduce_tp{tp}_s{rows}{'_r' if routed_only else ''}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(data: Int64, peers: Int64, epochs: Int64, rank: Int32):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        bank = (epoch & 1) * bank_words
        mine = rsrc(load_ptr64(peers, rank))
        line = bid * THREADS + tid
        _push_lines(
            data, peers, rank, tp, lines, bank, epoch, line, fx.Int32(pushed), stride
        )
        if const_expr(routed_only):
            _reduce_lines(
                data,
                mine,
                tp,
                lines,
                bank,
                epoch,
                fx.Int32(pushed) + line,
                fx.Int32(lines),
                stride,
            )
        _reduce_lines(
            data, mine, tp, lines, bank, epoch, line, fx.Int32(pushed), stride
        )
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
            grid=(ctas,), block=(THREADS,), stream=stream
        )

    ALLREDUCE_ABI.check(kernel, launch)
    return K2AllReduceKernel(name, launch, (ctas,), (THREADS,), ctas)


def build_shared_push(tp: int, rows: int, ctas: int = 32) -> K2AllReduceKernel:
    """The shared partials' lines (from ``shared_line0``) to every rank, tagged
    the epoch the next ``build_allreduce(routed_only=True)`` launch reads:
    counter 0 plus one, as the counters only move at that launch's end. Run
    as soon as the shared partials are written, beside the routed MoE: it
    stores and never waits."""
    lines = allreduce_lines(rows)
    first = shared_line0(rows)
    bank_words = tp * lines * LINE_WORDS
    name = f"k3_k2_gfx1250_shared_push_tp{tp}_s{rows}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(data: Int64, peers: Int64, epochs: Int64, rank: Int32):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        epoch = fx.Int32(bo.buffer_load(rsrc(epochs), 0, vec_width=1, dtype=T.i32)) + 1
        _push_lines(
            data,
            peers,
            rank,
            tp,
            lines,
            (epoch & 1) * bank_words,
            epoch,
            first + bid * THREADS + tid,
            fx.Int32(lines),
            ctas * THREADS,
        )

    @flyc.jit
    def launch(
        data: Int64,
        peers: Int64,
        epochs: Int64,
        rank: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(data, peers, epochs, rank).launch(
            grid=(ctas,), block=(THREADS,), stream=stream
        )

    ALLREDUCE_ABI.check(kernel, launch)
    return K2AllReduceKernel(name, launch, (ctas,), (THREADS,), ctas)
