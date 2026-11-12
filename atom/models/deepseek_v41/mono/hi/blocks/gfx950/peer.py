# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``PeerAllReduce``: y = the sum over the TP ranks of a per-row partial, the same bits on every rank.

The TP all-reduce of a row-parallel projection (``plan.program``: tensor parallelism). Input
``x``: a channel this rank's earlier stages put (``BF16`` or ``F32``); output: a channel of the same
format (``out``) and / or a plain arg (``out_arg``: [rows, width] bf16 for BF16, f32 for F32).

One stage, one thread per 32-bit word of the step's partial (``x.words_per_row() * rows``): poll the
word, put it into this rank's slot of every rank's peer copy (``to_rank``, system scope), poll this
rank's ``tp`` slots (``from_rank``) and add them in rank order 0 .. tp - 1 in fp32; BF16 words are
two bf16 values, each summed this way and rounded once at the end (the reference's row-parallel
linear: bf16 partials, an fp32 all-reduce, then bf16). Every rank adds the same values in the same
order, so they agree to the bit. Each task waits only for the same task's puts on the other ranks,
which come before those ranks' polls: with every CTA resident, deadlock-free.

``proto`` (per edge): ``"tagged"`` is the one-shot exchange
above (every rank sends its whole partial to every rank: tp - 1 partials out), one stage. ``"spread"`` is the
same exchange in two stages: ``<name>`` puts chunk c to rank d as its own task (tp x the tasks: a CTA's remote
puts were the limit, not the link), ``<name>.sum`` polls and sums. ``"two_shot"`` (reduce-scatter + all-gather):
rank o owns the words ``o * Q ..`` of the partial (Q = ceil(words / tp));

1. ``<name>``: every rank puts its partial's words into their owner's peer copy (its own slot);
2. ``<name>.own``: the owner polls the ``tp`` ranks' words, sums them in rank order (the same fp32 sum, the
   same bits) and puts the result into every rank's copy;
3. ``<name>.out``: every rank polls the owners' results into ``out`` / ``out_arg``.

(tp - 1) / tp of a partial out per shot: 2 (tp - 1) / tp partials in all vs tp - 1, for one more peer hop.
The owner's puts are spread one destination a task too. Words travel as tagged pairs: a dense + flag variant
(half the bytes) paid a store drain and a flag round trip per hop. ``"auto"`` (default): ``spread`` up to
``AUTO_SPREAD_WORDS`` words a step (latency-bound), ``two_shot`` above (bandwidth-bound). Every protocol gives
the same bits.
"""

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import bf2_f32, bf16_pair, rsrc, traced
from ...device.gfx950.peer import from_rank, to_rank
from ...plan.program import Region, Scenario, Stage
from ...plan.trace import Space
from ...target import MI355X
from ..base import Block, Channel, Fmt

THREADS = MI355X.threads
AUTO_SPREAD_WORDS = (
    16384  # proto "auto": "spread" up to this many words a step, "two_shot" above
)


@dataclass(frozen=True)
class _Plan:
    name: str
    x: str
    fmt: str  # "bf16" | "f32"
    words: int  # 32-bit words of the step's partial
    tp: int
    peer: str  # the peer region
    out: str  # output channel ("" if none)
    out_arg: str  # output arg ("" if none)
    proto: str = "tagged"
    chunk: int = (
        THREADS  # spread / two_shot: words a task (threads past it idle): more, smaller tasks
    )

    @property
    def q(self) -> int:  # two_shot: words an owner reduces
        return -(-self.words // self.tp)

    @property
    def tpo(self) -> int:  # two_shot: tasks (flags) an owner's share takes
        return -(-self.q // self.chunk)


class PeerAllReduce(Block):
    def __init__(self, name: str, x: Channel, out: Channel | None = None, out_arg: str | None = None,
                 proto: str = "auto", chunk: int = THREADS):  # fmt: skip
        if x.fmt not in (Fmt.BF16, Fmt.F32) or x.flagged:
            raise ValueError(
                f"PeerAllReduce {name}: input must be a tagged BF16 or F32 channel"
            )
        if out is not None and (
            out.fmt is not x.fmt or out.width != x.width or out.flagged
        ):
            raise ValueError(
                f"PeerAllReduce {name}: output channel must match the input's format and width"
            )
        if out is None and not out_arg:
            raise ValueError(f"PeerAllReduce {name}: give an output channel or arg")
        if proto not in ("auto", "tagged", "spread", "two_shot"):
            raise ValueError(
                f"PeerAllReduce {name}: proto auto / tagged / spread / two_shot, got {proto!r}"
            )
        if not 0 < chunk <= THREADS:
            raise ValueError(f"PeerAllReduce {name}: chunk in 1 .. {THREADS}")
        self.name, self.x, self.out, self.out_arg, self.proto, self.chunk = (
            name,
            x,
            out,
            out_arg or "",
            proto,
            chunk,
        )

    def plan(self, sc: Scenario) -> _Plan:
        if sc.tp < 2:
            raise ValueError(
                f"PeerAllReduce {self.name}: needs a TP scenario (tp > 1), got {sc.name}"
            )
        words = sc.rows * self.x.words_per_row()
        proto = self.proto
        if proto == "auto":  # latency-bound below ~16K words, bandwidth-bound above
            proto = "spread" if words <= AUTO_SPREAD_WORDS else "two_shot"
        return _Plan(
            self.name,
            self.x.name,
            self.x.fmt.value,
            words,
            sc.tp,
            f"{self.name}.peer",
            self.out.name if self.out else "",
            self.out_arg,
            proto,
            self.chunk,
        )

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        if p.proto == "two_shot":
            return self._two_shot(sc, p)
        if p.proto == "spread":
            return self._spread(sc, p)
        writes = [
            Region(p.peer, p.words, (self.name,), space=Space.PEER, exchange=True)
        ]
        if self.out is not None:
            writes += list(self.out.regions(sc.rows, (self.name,)))
        args = (p.out_arg,) if p.out_arg else ()
        tasks = -(-p.words // THREADS)
        return [
            Stage(
                self.name,
                tasks,
                partial(_reduce, p=p),
                writes=tuple(writes),
                args=args,
                key=(p,),
            )
        ]

    def _spread(self, sc: Scenario, p: _Plan) -> list[Stage]:
        chunks = -(-p.words // p.chunk)
        writes = (
            tuple(self.out.regions(sc.rows, (f"{self.name}.sum",)))
            if self.out is not None
            else ()
        )
        args = (p.out_arg,) if p.out_arg else ()
        return [
            Stage(self.name, p.tp * chunks, partial(_spread_put, p=p),
                  writes=(Region(p.peer, p.words, (self.name,), space=Space.PEER),), key=(p,)),
            Stage(f"{self.name}.sum", chunks, partial(_spread_sum, p=p), writes=writes, args=args, key=(p,)),
        ]  # fmt: skip

    def _two_shot(self, sc: Scenario, p: _Plan) -> list[Stage]:
        own, out = f"{self.name}.own", f"{self.name}.out"
        scatter = (Region(f"{p.peer}.rs", p.q, (self.name,), space=Space.PEER),)
        gather = (Region(f"{p.peer}.ag", p.q, (own,), space=Space.PEER),)
        writes = (
            tuple(self.out.regions(sc.rows, (out,))) if self.out is not None else ()
        )
        args = (p.out_arg,) if p.out_arg else ()
        return [
            Stage(
                self.name,
                p.tp * p.tpo,
                partial(_scatter, p=p),
                writes=scatter,
                key=(p,),
            ),
            Stage(own, p.tp * p.tpo, partial(_own, p=p), writes=gather, key=(p,)),
            Stage(
                out,
                p.tp * p.tpo,
                partial(_gather, p=p),
                writes=writes,
                args=args,
                key=(p,),
            ),
        ]


def _sum_word(p, words):
    """The ``tp`` ranks' words of one position, summed in rank order (fp32) -> the output word."""
    if p.fmt == "f32":
        acc = words[0].bitcast(fx.Float32)
        for src in range_constexpr(1, p.tp):
            acc = acc + words[src].bitcast(fx.Float32)
        return acc.bitcast(fx.Int32)
    lo, hi = bf2_f32(words[0])
    for src in range_constexpr(1, p.tp):
        a, b = bf2_f32(words[src])
        lo, hi = lo + a, hi + b
    return bf16_pair(lo, hi).bitcast(fx.Int32)


@traced
def _reduce(k, t, p):
    w = t * THREADS + k.tid
    valid = w < p.words
    wc = fx.min(
        w, fx.Int32(p.words - 1)
    )  # an idle thread polls a real word, puts nothing
    mine = mb.poll([(k.mb[p.x], wc, 1)], k.tag)[0][0]
    if valid:
        for dst in range_constexpr(p.tp):
            mb.put(to_rank(k, p.peer, dst), w, mine, k.tag)
    got = mb.poll([(from_rank(k, p.peer, src), wc, 1) for src in range(p.tp)], k.tag)
    y = _sum_word(p, [g[0] for g in got])
    if valid:
        if p.out:
            mb.put(k.mb[p.out], w, y, k.tag)
        if p.out_arg:
            bo.buffer_store(y, rsrc(k.args[p.out_arg]), w, cache_modifier=0)


# ------------------------------------------------------------------ spread


@traced
def _spread_put(k, t, p):
    """One-shot, its puts spread over tp x as many tasks: task t puts chunk t % chunks to rank t // chunks."""
    chunks = -(-p.words // p.chunk)
    dst, w = t // chunks, (t % chunks) * p.chunk + k.tid
    mine = mb.poll([(k.mb[p.x], fx.min(w, fx.Int32(p.words - 1)), 1)], k.tag)[0][0]
    if (w < p.words) & (k.tid < p.chunk):
        mb.put(to_rank(k, p.peer, dst), w, mine, k.tag)


@traced
def _spread_sum(k, t, p):
    w = t * p.chunk + k.tid
    wc = fx.min(w, fx.Int32(p.words - 1))
    got = mb.poll([(from_rank(k, p.peer, src), wc, 1) for src in range(p.tp)], k.tag)
    y = _sum_word(p, [g[0] for g in got])
    if (w < p.words) & (k.tid < p.chunk):
        if p.out:
            mb.put(k.mb[p.out], w, y, k.tag)
        if p.out_arg:
            bo.buffer_store(y, rsrc(k.args[p.out_arg]), w, cache_modifier=0)


# ------------------------------------------------------------------ two_shot


def _chunk(p, t):
    """Task ``t`` of the scatter / gather stages: (owner, the first word of its chunk in the owner's share)."""
    return t // p.tpo, (t % p.tpo) * p.chunk


@traced
def _scatter(k, t, p):
    o, i0 = _chunk(p, t)
    i = i0 + k.tid
    w = o * p.q + i
    valid = (i < p.q) & (w < p.words) & (k.tid < p.chunk)
    mine = mb.poll([(k.mb[p.x], fx.min(w, fx.Int32(p.words - 1)), 1)], k.tag)[0][0]
    if valid:
        mb.put(to_rank(k, f"{p.peer}.rs", o), i, mine, k.tag)


@traced
def _own(k, t, p):
    """Owner (this rank) chunk ``t % tpo``: every rank's words, summed in rank order, to rank ``t // tpo``'s copy
    (each destination its own task: remote puts spread over the CTAs, as ``spread``)."""
    dst, c = t // p.tpo, t % p.tpo
    i = c * p.chunk + k.tid
    w = k.rank * p.q + i
    valid = (i < p.q) & (w < p.words) & (k.tid < p.chunk)
    ic = fx.min(i, fx.Int32(p.q - 1))
    got = mb.poll(
        [(from_rank(k, f"{p.peer}.rs", src), ic, 1) for src in range(p.tp)], k.tag
    )
    y = _sum_word(p, [g[0] for g in got])
    if valid:
        mb.put(to_rank(k, f"{p.peer}.ag", dst), i, y, k.tag)


@traced
def _gather(k, t, p):
    o, i0 = _chunk(p, t)
    i = i0 + k.tid
    w = o * p.q + i
    valid = (i < p.q) & (w < p.words) & (k.tid < p.chunk)
    y = mb.poll(
        [(from_rank(k, f"{p.peer}.ag", o), fx.min(i, fx.Int32(p.q - 1)), 1)], k.tag
    )[0][0]
    if valid:
        if p.out:
            mb.put(k.mb[p.out], w, y, k.tag)
        if p.out_arg:
            bo.buffer_store(y, rsrc(k.args[p.out_arg]), w, cache_modifier=0)
