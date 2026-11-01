# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tagged mailbox: hand-offs between the CTAs of one GPU (device scope) and
between GPUs (system scope), every address a named region (``Addr``) recorded
when the kernel is traced (``atom.mono.plan``)."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.mono.device.ops import (
    CM_DEV,
    CM_SYS,
    CM_SYS_GFX1250,
    ballot,
    bf16_pair,
    memrealtime,
    rsrc,
    traced,
)
from atom.mono.plan.layout import DIAG_WORDS
from atom.mono.plan.trace import Space, note

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)


@traced
def _spin(load_all, pending):
    """Re-issue ``load_all`` until ``pending`` clears; a side-effecting op in the
    loop keeps the loads from being hoisted."""
    v = load_all()
    while pending(v):
        rocdl.s_sleep(1)
        v = load_all()
    return v


# a debug build's mailbox wait gives up after this long (100 MHz ticks: 10 s)
DIAG_TICKS = 10 * 100_000_000


@traced
def _spin_bounded(
    load_all,
    pending,
    diag,
    region_id,
    index,
    tag,
    steady=False,
    failure=None,
    wait_ticks=DIAG_TICKS,
):
    """``_spin`` giving up after wait_ticks: the CTA's record in ``diag`` names the
    wait (the runner raises with it after the step), and the kernel runs on."""
    t0 = memrealtime(steady)
    v = load_all()
    late = fx.Int32(0)

    def waiting(words):
        bad = pending(words)
        if const_expr(steady):
            # All wave32 lanes reissue loads and leave the wait together.
            bad = ballot(bad) != 0
        return bad

    while waiting(v) & (late == 0):
        if const_expr(steady):
            # On gfx1250 the original one-unit polling interval can delay
            # producer progress when many wave32 consumers are waiting.
            # Retain the same wait set/publication protocol, but back off
            # stale reads. Verified with the full front and 216 consumers.
            rocdl.s_sleep(64)
        else:
            rocdl.s_sleep(1)
        v = load_all()
        timed_out = memrealtime(steady) - t0 > wait_ticks
        if const_expr(failure is not None):
            stopped = (
                fx.generic_load(
                    failure, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"
                )
                != 0
            )
            timed_out = timed_out | (ballot(stopped) != 0)
        late = timed_out.select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        bid = fx.block_idx.x
        if const_expr(failure is not None):
            fx.atomic_cas(
                failure,
                fx.Int32(0),
                fx.Int32(30_000_000) + fx.Int32(bid),
                success_ordering=fx.AtomicOrdering.Release,
                failure_ordering=fx.AtomicOrdering.Monotonic,
                syncscope="agent",
            )
        # the CTA's first give-up of the step: the later ones wait on its fallout
        seen = bo.buffer_load(
            rsrc(diag),
            bid * DIAG_WORDS,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_SYS_GFX1250 if steady else CM_SYS,
        )
        if fx.Int32(seen) == 0:
            at = fx.Int32(memrealtime(steady) & 0x7FFFFFFF)
            record = [1, region_id, index, tag, v[1], bid, at, 0]
            for h in range_constexpr(2):  # 16 B a store
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [fx.Int32(w) for w in record[4 * h : 4 * h + 4]], fx.Int32
                    ),
                    rsrc(diag),
                    bid * DIAG_WORDS + 4 * h,
                    cache_modifier=CM_SYS_GFX1250 if steady else CM_SYS,
                )
    return v


@dataclass(frozen=True)
class Addr:
    """A mailbox region's address: the region's name and space, recorded when the
    kernel is traced (``atom.mono.plan``), and the traced address. The space also
    sets the hand-off's scope: device for SCRATCH, system for PEER."""

    region: str
    space: Space
    value: object


def sreg(base, offset, region):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset))


def preg(base, offset, region):
    """Peer-symmetric region ``region`` at byte ``offset`` of a rank's buffer ``base``."""
    return Addr(region, Space.PEER, base + fx.Int64(offset))


def shift(a, delta):
    """``a`` moved by ``delta`` bytes (a traced or Python integer), same region."""
    return Addr(a.region, a.space, a.value + delta)


# the store / load cache policy of a hand-off in each space
_SCOPE_CM = {Space.SCRATCH: CM_DEV, Space.PEER: CM_SYS}


def _noted(a, kind):
    note(a.region, a.space, kind)
    return a.value


class Mailbox:
    """Tagged-pair hand-off between CTAs (device scope) and GPUs (system scope).

    Every 32-bit value is stored next to this launch's tag; a consumer
    polls the payload until all tags match, so a hand-off costs one round trip
    (no store drain, no separate flag). The tag is ``layer + 1``: it tells the
    layers of one step apart, and the runner zeroes every mailbox at a step's
    start (``StepMailboxes.begin_step``), so a pair from an earlier step never
    carries a live tag. Every address is an ``Addr``: its space picks the scope.
    """

    def __init__(
        self,
        layer,
        diag=None,
        region_ids=None,
        *,
        steady=False,
        failure=None,
        wait_ticks=DIAG_TICKS,
        gfx1250=False,
    ):
        """``diag`` (a debug build): the address of the per-CTA wait records, every
        wait bounded (``_spin_bounded``); ``region_ids``: region name -> the id a
        record names it by."""
        self.tag = layer + 1
        self.diag = diag
        self.region_ids = region_ids or {}
        self.steady = steady
        self.failure = failure
        if not isinstance(wait_ticks, int) or wait_ticks <= 0:
            raise ValueError("mailbox wait_ticks must be a positive integer")
        self.wait_ticks = wait_ticks
        self.scope_cm = {
            Space.SCRATCH: CM_DEV,
            Space.PEER: CM_SYS_GFX1250 if (gfx1250 or steady) else CM_SYS,
        }
        if steady and diag is None:
            raise ValueError("gfx1250 peer waits require bounded diagnostics")

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(_noted(a, "put")),
            i * 2,
            cache_modifier=self.scope_cm[a.space],
        )

    def put_bf(self, a, i, vs):
        """Elements i .. i+len(vs) (2 or 4, i aligned) as packed bf16 pairs, one store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(_noted(a, "put")),
            i,
            cache_modifier=self.scope_cm[a.space],
        )

    def put_words(self, a, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. as (word, tag). Integer payloads
        (fp8 words) must come this way: through an f32 value a NaN bit pattern is
        not kept."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(_noted(a, "put")),
            i * 2,
            cache_modifier=self.scope_cm[a.space],
        )

    def poll(self, specs, batch=POLL_MAX):
        """Batched poll of pairs ``specs`` = [(Addr, pair index, npairs in {1, 2})],
        each loaded at its own space's scope.

        All pairs are re-loaded together while any tag is stale, so a batch costs
        one round trip after its last producer lands. Returns the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        cms = [self.scope_cm[a.space] for a, _, _ in specs]
        bases = [_noted(a, "poll") for a, _, _ in specs]
        tag = self.tag

        def load_all():
            words = []
            for b, cm, (_, i, n) in zip(bases, cms, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        if const_expr(self.diag is None):
            v = _spin(load_all, pending)
        else:
            rid = self.region_ids.get(specs[0][0].region, -1)
            v = _spin_bounded(
                load_all,
                pending,
                self.diag,
                rid,
                specs[0][1],
                tag,
                self.steady,
                self.failure,
                self.wait_ticks,
            )
        outs, e = [], 0
        for _, _, n in specs:
            outs.append(
                [
                    (
                        (v[e + 2 * q + 1] == tag).select(v[e + 2 * q], fx.Int32(0))
                        if self.steady
                        else v[e + 2 * q]
                    )
                    for q in range(n)
                ]
            )
            e += 2 * n
        return outs


@traced
def publish(put, a, i, v, who):
    """This CTA's plain stores made readable by every CTA, then (thread ``who``)
    the mailbox pair (``a``, i) := v. The stores are drained first: a flag may
    not overtake the data it announces, and a workgroup-scope release fence does
    not wait for them (LLVM's AMDGPU memory model omits vmcnt outside tgsplit
    mode: the workgroup's waves share one CU, the readers here do not)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put(a, i, v)
