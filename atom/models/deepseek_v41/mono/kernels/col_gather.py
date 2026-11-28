# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A column-sharded residual's columns gathered across the TP ranks, in place,
one launch.

A sharded layer (``runner._shard``) leaves on each rank only its own
``hidden / tp`` columns of the residual, of the owed FFN output and of the
drafter's taps. Where a reader needs every column (an Engram layer, the end
of the step), each rank first finishes its own columns -- the mHC math is per
column (``mhc_cols``) -- and pushes them to every rank's CG region as tagged
pairs; then every rank polls every column and writes it in place:

- ``SETTLE``: the owed post settled into the residual streams (``res``, HC
  planes), what an Engram layer reads;
- ``COLLAPSE``: settled and collapsed by the next pre (``out``, one plane),
  the step's last hidden state;
- ``NONE``: no residual (the taps alone);

then the taps (``taps`` buffers [rows, hidden], a plane each), as they are.
A column is read only by the rank that owns it, before that rank pushes it,
and written by every rank only from the pushes: in place is safe.

The pairs carry ``GATHER_TAG + call`` (a runtime argument: one build serves
every gather of a step), so two gathers of one step never see each other's.
"""

import enum
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from atom.models.deepseek_v41.mono.kernels.dims import HC, HIDDEN, Dims
from atom.models.deepseek_v41.mono.kernels.mailbox import Mailbox, epoch_of, preg
from atom.models.deepseek_v41.mono.kernels.mhc_cols import (
    collapsed8,
    seam_mixes,
    settled8,
)
from atom.models.deepseek_v41.mono.kernels.peer_layout import CG_PLANES, peer_offsets
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import (
    bf16_words,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
    words4,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.plan.arith import pick
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

# CG's planes a row: the residual's HC streams at most, then the taps
PLANES = CG_PLANES
MAX_TAPS = PLANES - HC
# the gathers' tags: past every layer's (K1 / K2a / K2b, the draft head's)
GATHER_TAG = 200

ABI = KernelAbi(
    (
        "res", "sub", "post", "comb", "pre", "out", "tap0", "tap1", "tap2", "tap3",
        "call", "sym", "peers", "rank", "epoch",
    )
)  # fmt: skip
TAPS = ("tap0", "tap1", "tap2", "tap3")


class State(enum.IntEnum):
    """What the gather makes of the residual (module docstring)."""

    NONE = 0
    SETTLE = 1
    COLLAPSE = 2


@dataclass(frozen=True)
class ColGatherBuild:
    rows: int
    taps: int
    state: State = State.COLLAPSE
    tp: int = 4

    def __post_init__(self):
        assert 0 <= self.taps <= MAX_TAPS
        assert self.state != State.NONE or self.taps, "nothing to gather"
        assert HIDDEN % (8 * self.tp) == 0
        # SETTLE lands into ``res``, which its pushes read: a (row, quad)'s
        # pushes (``push_task``: every plane to every rank) one wave's lanes,
        # so all of them read before any of that wave's lands writes
        lanes = THREADS // WAVES
        assert (
            self.state != State.SETTLE or lanes % (self.planes * self.tp) == 0
        ), "a settle's (row, quad) pushes past one wave"

    @property
    def state_planes(self) -> int:
        return {State.NONE: 0, State.SETTLE: HC, State.COLLAPSE: 1}[self.state]

    @property
    def planes(self) -> int:
        return self.state_planes + self.taps

    @property
    def quads(self) -> int:
        """8-column quads a row (the whole width)."""
        return HIDDEN // 8

    @property
    def own_quads(self) -> int:
        return self.quads // self.tp

    @property
    def pushes(self) -> int:
        """``push_own``'s tasks: an own quad of a row, a plane, a rank."""
        return self.rows * self.own_quads * self.planes * self.tp

    @property
    def lands(self) -> int:
        """``land_all``'s tasks: a quad of a row, a plane."""
        return self.rows * self.quads * self.planes

    @property
    def ctas(self) -> int:
        """Every task a thread of its own, up to every CTA."""
        return min(BLOCKS, -(-max(self.pushes, self.lands) // THREADS))

    @property
    def threads(self) -> int:
        return self.ctas * THREADS


def cg_pair(row, plane, col):
    """CG's pair of row ``row``, plane ``plane``, columns col, col + 1."""
    return (row * PLANES + plane) * (HIDDEN // 2) + col // 2


def push_task(g, p: ColGatherBuild, rank):
    """Push task ``g`` -> (destination rank, plane, row, first column of
    ``rank``'s own quad): the rank fastest, then the plane, the row, the quad
    (ints or traced values alike)."""
    dst, rest = g % p.tp, g // p.tp
    pl, rest = rest % p.planes, rest // p.planes
    t, q = rest // p.own_quads, rest % p.own_quads
    return dst, pl, t, rank * (HIDDEN // p.tp) + 8 * q


def land_task(g, p: ColGatherBuild):
    """Land task ``g`` -> (row, plane, first column): the quad fastest, then
    the row, the plane."""
    col, rest = 8 * (g % p.quads), g // p.quads
    return rest % p.rows, rest // p.rows, col


def plane_words(c, t, col, pl):
    """Plane ``pl`` of row t at columns col .. + 7 (this rank's): the
    residual finished (``State``: a settled stream, or the collapse), then
    the taps as they are; four bf16 pair words."""
    p, a = c["key"], c["args"]
    planes = []
    if const_expr(p.state != State.NONE):
        mix = seam_mixes(a["post"], a["comb"], a["pre"], t)
        streams = settled8(a["res"], a["sub"], HIDDEN, t, col, mix)
        if const_expr(p.state == State.SETTLE):
            planes += [bf16_words(s) for s in streams]
        else:
            planes.append(bf16_words(collapsed8(streams, mix)))
    planes += [words4(a[TAPS[k]], (t * HIDDEN + col) // 2) for k in range(p.taps)]
    return pick(planes, pl)


def plane_dst(c, t, pl, col):
    """Where row t's plane ``pl`` at col lands: (buffer, word)."""
    p, a = c["key"], c["args"]
    row = (t * HIDDEN + col) // 2
    dsts = []
    if const_expr(p.state == State.SETTLE):
        dsts += [(a["res"], ((t * HC + h) * HIDDEN + col) // 2) for h in range(HC)]
    elif const_expr(p.state == State.COLLAPSE):
        dsts.append((a["out"], row))
    dsts += [(a[TAPS[k]], row) for k in range(p.taps)]
    return pick([d for d, _ in dsts], pl), pick([w for _, w in dsts], pl)


def push_own(c, g):
    """Push tasks g, g + threads, .. (``push_one``)."""
    p = c["key"]
    for k in range_constexpr(-(-p.pushes // p.threads)):
        push_one(c, g + k * p.threads)


@traced
def push_one(c, g):
    """Push task ``g`` (``push_task``): the plane's 8 columns to that rank's
    CG."""
    p = c["key"]
    if g < p.pushes:
        dst, pl, t, col = push_task(g, p, c["rank"])
        w = plane_words(c, t, col, pl)
        # the lane's own rank's base (``dst`` varies by lane): not fresh
        cg = preg(pick(c["bases"], dst), c["cg_off"], "cg", fresh=False)
        at = cg_pair(t, pl, col)
        c["mb"].put_words(cg, at, [w[0], w[1]])
        c["mb"].put_words(cg, at + 2, [w[2], w[3]])


@traced
def land_all(c, g):
    """Land tasks g, g + threads, .. (``land_task``): every rank's columns of
    a row's plane from CG, one batched poll, into place."""
    p = c["key"]
    tasks = [g + k * p.threads for k in range(-(-p.lands // p.threads))]
    where = []
    specs = []
    for gk in tasks:
        t, pl, col = land_task(fx.min(gk, fx.Int32(p.lands - 1)), p)
        where.append((t, pl, col))
        at = cg_pair(t, pl, col)
        specs += [(c["cg"], at, 2), (c["cg"], at + 2, 2)]
    got = c["mb"].poll(specs)
    for k in range_constexpr(len(tasks)):
        lo, hi = got[2 * k], got[2 * k + 1]
        words = fx.Vector.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int32)
        land_one(c, tasks[k] < p.lands, *where[k], words)


@traced
def land_one(c, live, t, pl, col, words):
    if live:
        dst, at = plane_dst(c, t, pl, col)
        bo.buffer_store(words, rsrc(dst), at)


def build_col_gather(key: ColGatherBuild):
    keyed = key_tuple(key, SOURCES)
    peer = peer_offsets(key.rows, Dims(key.tp))
    name = kernel_symbol(
        "v41_col_gather", s=key.rows, st=int(key.state), tap=key.taps, tp=key.tp
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def col_gather(
        res: Int64, sub: Int64, post: Int64, comb: Int64, pre: Int64, out: Int64,
        tap0: Int64, tap1: Int64, tap2: Int64, tap3: Int64, call: Int32,
        sym: Int64, peers: Int64, rank: Int32, epoch: Int64,
    ):  # fmt: skip
        # the step's epoch: every hand-off of the launch tagged with it (``kernels.mailbox``)
        ep = epoch_of(epoch)
        bases = peer_bases(peers, key.tp)
        c = {
            "key": key, "rank": rank,
            "args": {"res": res, "sub": sub, "post": post, "comb": comb,
                     "pre": pre, "out": out, "tap0": tap0, "tap1": tap1,
                     "tap2": tap2, "tap3": tap3},
            "mb": Mailbox(GATHER_TAG + call, epoch=ep),
            "cg": preg(sym, peer["cg"], "cg"), "cg_off": peer["cg"],
            "bases": [bases[r] for r in range(key.tp)],
        }  # fmt: skip
        g = fx.block_idx.x * THREADS + fx.thread_idx.x
        push_own(c, g)
        land_all(c, g)
        _ = keyed

    @launcher(name)
    def launch(
        res: Int64, sub: Int64, post: Int64, comb: Int64, pre: Int64, out: Int64,
        tap0: Int64, tap1: Int64, tap2: Int64, tap3: Int64, call: Int32,
        sym: Int64, peers: Int64, rank: Int32, epoch: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        col_gather(
            res, sub, post, comb, pre, out, tap0, tap1, tap2, tap3, call, sym,
            peers, rank, epoch,
        ).launch(grid=(key.ctas,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(col_gather, launch)
    return launch
