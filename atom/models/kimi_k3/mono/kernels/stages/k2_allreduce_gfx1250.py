# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2's ``[routed | shared]`` all-reduce inside a gfx1250 kernel, one shot over
the TP group's symmetric (torch.symm_mem) buffers.

Each rank pushes its partial as 16 B lines -- three words of payload, six
bf16, and the launch's epoch -- to every rank's receive bank with SCOPE_SYS
stores, then reduces the lines its CTA pushed: it polls every source's copy of
a line until the copy carries the epoch, sums the sources in rank order in
fp32 from rank 0's value, as aiter's custom all-reduce does, and rounds to bf16
into the input in place. A CTA writes back only the lines it pushed, so none
overwrites a line another CTA has yet to read.

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
from flydsl.expr import gpu, range_constexpr
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


def _line_tag(mine, at):
    return fx.Int32(
        fx.Vector(
            bo.buffer_load(
                mine,
                at,
                vec_width=LINE_WORDS,
                dtype=T.i32,
                cache_modifier=SCOPE_SYS | VOLATILE,
            )
        )[PAYLOAD_WORDS]
    )


@traced
def _poll_line(mine, at, epoch):
    """The line at word ``at`` of this rank's receive bank once it carries
    ``epoch``; it stays there until launch ``epoch + 2``."""
    seen = _line_tag(mine, at)
    while seen != epoch:
        spin_pause()
        seen = _line_tag(mine, at)
    spin_pause()
    return fx.Vector(
        bo.buffer_load(
            mine,
            at,
            vec_width=LINE_WORDS,
            dtype=T.i32,
            cache_modifier=SCOPE_SYS | VOLATILE,
        )
    )


def build_allreduce(tp: int, rows: int) -> K2AllReduceKernel:
    lines = allreduce_lines(rows)
    ctas = allreduce_ctas(rows)
    per = -(-lines // ctas)
    bank_words = tp * lines * LINE_WORDS
    name = f"k3_k2_gfx1250_allreduce_tp{tp}_s{rows}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(data: Int64, peers: Int64, epochs: Int64, rank: Int32):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        epoch = (
            fx.Int32(bo.buffer_load(rsrc(epochs), bid, vec_width=1, dtype=T.i32)) + 1
        )
        bank = (epoch & 1) * bank_words
        first = bid * per
        last = fx.min(first + per, fx.Int32(lines))
        local = rsrc(data)
        line = first + tid
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
            line = line + THREADS
        mine = rsrc(load_ptr64(peers, rank))
        line = first + tid
        while line < last:
            sums = []
            for p in range_constexpr(tp):
                got = _poll_line(mine, bank + (p * lines + line) * LINE_WORDS, epoch)
                values = []
                for w in range_constexpr(PAYLOAD_WORDS):
                    values.extend(bf2_f32(fx.Int32(got[w])))
                sums = values if p == 0 else [a + b for a, b in zip(sums, values)]
            for w in range_constexpr(PAYLOAD_WORDS):
                bo.buffer_store(
                    fx.Vector.from_elements([sums[2 * w], sums[2 * w + 1]], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0],
                    local,
                    line * PAYLOAD_WORDS + w,
                )
            line = line + THREADS
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
