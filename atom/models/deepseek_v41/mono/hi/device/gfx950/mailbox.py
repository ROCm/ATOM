# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ATOM atom/mono/device/sync.py (MIT, Copyright (C) 2026 Advanced
# Micro Devices, Inc.) and the FlyDSL GLM-5 layer kernel.
"""Tagged-pair hand-offs between CTAs.

Every 32-bit payload is stored next to the launch's tag as one 8- or 16-byte
coherent store; a consumer polls the payload itself until every tag matches, so
a hand-off costs one round trip. Tags are ``step * LAYER_SLOTS + layer + 1`` and
scratch is never cleared: a pair from an earlier launch does not carry the
current tag. Every address is an ``Addr`` naming its region, recorded at
trace time for the contract check (``plan.check``).

All functions take the tag explicitly (``k.tag`` of the stage context).

Tags are compared for equality only: a launch's tag must differ from the tags
the previous launch on the same scratch left behind (every polled pair is
rewritten by every launch), not be larger than them. Nothing depends on tag
order, so the int32 tag wrapping around is harmless.

"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...plan import trace
from ...plan.trace import Space, note
from . import buffer_ops as bo
from .ops import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    rsrc,
    spin_pause,
    traced,
)

POLL_MAX = 12  # pair specs per poll batch: bounds live registers
_SCOPE_CM = {Space.SCRATCH: CM_DEV, Space.PEER: CM_SYS}


@dataclass(frozen=True)
class Addr:
    """A region's traced base address, with its name and space for the record.

    A dual-copy region (``Region.home``; ``device.gfx950.dual``) is addressed as copy 0 here: its buffer resources
    route every load to the CTA's copy and every store to both."""

    region: str
    space: Space
    value: object
    delta: object = 0  # bytes ``value`` lies past the region's start (``shift``)
    # ``Program.dyn_rows``: a row-major region's pairs a row (``Region.rows``): its pair p is row p // row_pairs, and
    # polls / puts of rows >= the live rows (``live_rows``) are masked. 0: not row-major, or a static program.
    row_pairs: int = 0
    nbytes: int = (
        0  # (dyn_rows, row-major) the region's bytes: a masked put's resource is bounded by them
    )

    def __post_init__(
        self,
    ):  # the memory-model record: accesses through this address are the region's
        trace.root(
            self.value,
            trace.PEER if self.space is Space.PEER else trace.REGION,
            self.region,
        )


# ---- symbolic rows (Program.dyn_rows): the stage's live rows while it is traced

_live = None  # the stage's ``k.n`` (traced) while a dyn_rows stage is traced, else None


@contextmanager
def live_rows(n):
    """Trace the enclosed stage with live rows ``n`` (a traced Int32; None: a static program): ``poll`` / ``put*``
    on row-major regions (``Addr.row_pairs``) then skip the rows >= n."""
    global _live
    prev, _live = _live, n
    try:
        yield
    finally:
        _live = prev


def row_live(a: Addr, pair):
    """Whether pair ``pair`` of ``a`` (counted from ``a``'s shifted base) lies in a live row: True (a Python bool)
    unless a dyn_rows stage is traced and ``a`` is row-major. (A compare with the live pairs' bound: no division.)
    """
    if _live is None or not a.row_pairs:
        return True
    d = a.delta // 8 if isinstance(a.delta, int) else fx.Int32(a.delta) // 8
    return fx.Int32(pair) + d < _live * a.row_pairs


def clamp_live(a: Addr, pair, width: int = 1):
    """A poll's pair index with a dead row's pair moved onto the last live pair (a dyn_rows stage, row-major ``a``):
    the poll then waits for a pair that this launch does put, and needs no mask held through its spin (the tuned
    stages have no registers to spare). The words it returns for a dead row are another row's: garbage, as intended.
    Unchanged otherwise."""
    if _live is None or not a.row_pairs:
        return pair
    d = a.delta // 8 if isinstance(a.delta, int) else fx.Int32(a.delta) // 8
    return fx.min(fx.Int32(pair) + d, _live * a.row_pairs - width) - d


def _clamped(a: Addr, pair, width):
    """(Addr, pair, width) of a poll spec with its pair clamped onto the live rows (``clamp_live``), addressed from
    the region's start: an Addr shifted into a dead row would otherwise need a negative index.
    """
    if isinstance(a.delta, int) and a.delta == 0:
        return a, clamp_live(a, pair, width), width
    d = a.delta // 8 if isinstance(a.delta, int) else fx.Int32(a.delta) // 8
    back = a.delta if isinstance(a.delta, int) else fx.Int64(a.delta)
    start = replace(a, value=fx.Int64(a.value) - back, delta=0)
    return start, fx.min(fx.Int32(pair) + d, _live * a.row_pairs - width), width


def shift(a: Addr, delta) -> Addr:
    """``a`` moved by ``delta`` bytes, same region."""
    return replace(a, value=a.value + delta, delta=a.delta + delta)


def _noted(a: Addr, kind: str):
    """The buffer resource of ``a`` (its own), the access recorded as ``kind``."""
    note(a.region, a.space, kind)
    return rsrc(a.value)


def _store(a: Addr, words, i):
    """Store the int32 ``words`` at word ``i`` of ``a`` (scope from its space); a put."""
    store(
        a,
        (
            fx.Vector.from_elements(words, fx.Int32)
            if len(words) > 1
            else fx.Int32(words[0])
        ),
        i,
        _SCOPE_CM[a.space],
    )


def store(a: Addr, data, i, cm=CM_DEV, scale=None):
    """Store ``data`` (an int32 or a vector of them) at word ``i`` (``scale``: ``i * scale``) of ``a``; recorded as
    a put."""
    r = _noted(a, "put")
    if scale is not None:  # the word index after the resource
        i = i * scale
    bo.buffer_store(data, r, i, cache_modifier=cm)


def load(a: Addr, i, n, cm=CM_DEV, scale=None):
    """``n`` int32 words (an Int32, or a vector for n > 1) at word ``i`` (``scale``: ``i * scale``) of ``a``, issued
    now; recorded as a poll."""
    r = _noted(a, "poll")
    if scale is not None:  # the word index after the resource
        i = fx.Int32(i) * scale
    return bo.buffer_load(r, i, vec_width=n, dtype=T.i32, cache_modifier=cm)


def load_pairs(a: Addr, i, n, cm=CM_DEV):
    """``n`` words from pair ``i`` of ``a`` (``load`` at word ``2 i``: the resource first, then the index)."""
    return load(a, i, n, cm, scale=2)


def _put(a: Addr, words, i, scale):
    """A put's words at word ``i * scale`` (``scale`` None: ``i``). In a dyn_rows stage a dead row's put is skipped
    (``row_live``)."""
    ok = (
        row_live(a, i if scale == 2 else fx.Int32(i) * (scale or 1) // 2)
        if _live is not None
        else True
    )
    if ok is not True:
        _put_masked(a, words, i if scale is None else fx.Int32(i) * scale, ok)
        return
    _put_now(a, words, i, scale)


def _put_masked(a: Addr, words, w, ok):
    """A dead row's put dropped by its address (out of the buffer's range), not by a branch: a value from a cross-lane
    reduction must not feed a store inside a lane-conditional branch. The store goes
    through a resource bounded by the region (``Addr.nbytes``: the regions' own resources are unbounded, so an
    out-of-range offset there would land 2 GiB on)."""
    data = (
        fx.Vector.from_elements([fx.Int32(x) for x in words], fx.Int32)
        if len(words) > 1
        else fx.Int32(words[0])
    )
    note(a.region, a.space, "put")
    cm = _SCOPE_CM[a.space]
    start = fx.Int64(a.value) - fx.Int64(a.delta)
    d = a.delta // 4 if isinstance(a.delta, int) else fx.Int32(a.delta) // 4
    bo.buffer_store(
        data, rsrc(start, a.nbytes), fx.Int32(w) + d, mask=ok, cache_modifier=cm
    )


def _put_now(a: Addr, words, i, scale):
    store(
        a, fx.Vector.from_elements(words, fx.Int32), i, _SCOPE_CM[a.space], scale=scale
    )


def put(a: Addr, i, v, tag):
    """Pair ``i`` := (v, tag); ``v`` an f32 or int32 bits."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    _put(a, [bits, tag], i, 2)


def put_fill(a: Addr, i, tag):
    """Pair ``i`` := (0, tag), never masked: a dyn_rows stage's dead task filling a dead row (``emit``'s fill)."""
    _put_now(a, [fx.Int32(0), tag], i, 2)


def put2(a: Addr, i, v0, v1, tag):
    """Pairs ``i``, ``i + 1`` (i even) in one 16-byte store."""
    words = []
    for v in (v0, v1):
        words += [
            v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v),
            tag,
        ]
    _put(a, words, i, 2)


def put_bf(a: Addr, i, vs, tag):
    """Elements ``i .. i + len(vs)`` (2 or 4, i aligned) as packed bf16 pairs:
    pair ``i // 2 + j`` := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one store."""
    words = []
    for j in range_constexpr(len(vs) // 2):
        words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
    _put(a, words, i, None)


def put_words(a: Addr, i, words, tag):
    """Raw 32-bit words (1 or 2) at pairs ``i ..`` as (word, tag): integer payloads
    (fp8 words) must come this way, an f32 round trip would not keep NaN patterns."""
    vals = []
    for w in words:
        vals += [fx.Int32(w), tag]
    _put(a, vals, i, 2)


@traced
def _spin(load_all, pending):
    v = load_all()
    while pending(v):
        spin_pause()
        v = load_all()
    return v


def poll(specs, tag, batch=POLL_MAX, live=None):
    """Batched poll of ``specs`` = [(Addr, pair index, npairs in {1, 2})], one space.

    All pairs are re-loaded together while any tag is stale, so a batch costs one
    round trip after its last producer lands. Returns each spec's value words.

    ``live`` (``Program.dyn_rows``): per spec, whether it is awaited (a Python bool or a uniform traced Bool); a dead
    spec's pair is loaded with the rest but its tag is not checked (a dead row's pair is never put, so it holds an
    earlier launch's tag): its words are garbage. None: every spec (the code of before).
    """
    if (
        live is None and _live is not None
    ):  # dyn_rows: dead rows' pairs polled as the last live pair (clamp_live)
        # (a 2-pair load moves by 2: the live pairs' bound is even when a row's pairs are)
        fits = [not a.row_pairs or n == 1 or a.row_pairs % 2 == 0 for a, _, n in specs]
        live = [True if f else row_live(a, i) for f, (a, i, n) in zip(fits, specs)]
        specs = [
            _clamped(a, i, n) if f and a.row_pairs else (a, i, n)
            for f, (a, i, n) in zip(fits, specs)
        ]
    if const_expr(len(specs) > batch):
        lo = None if live is None else live[:batch]
        hi = None if live is None else live[batch:]
        return poll(specs[:batch], tag, batch, lo) + poll(specs[batch:], tag, batch, hi)
    spaces = {a.space for a, _, _ in specs}
    assert len(spaces) == 1, f"one poll batch spans {spaces}"
    cm = _SCOPE_CM[spaces.pop()]

    def load_all():
        words = []
        for a, i, n in specs:
            w = fx.Vector(load_pairs(a, i, 2 * n, cm))
            words += [w[e] for e in range(2 * n)]
        return fx.Vector.from_elements(words, fx.Int32)

    nw = sum(2 * n for _, _, n in specs)
    # each tag word's spec (dead specs' tags are not checked); all of them: the code of before
    if live is not None and all(live[j] is True for j in range(len(specs))):
        live = None
    if live is None:

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

    v = _spin(load_all, pending)
    outs, e = [], 0
    for _, _, n in specs:
        outs.append([v[e + 2 * q] for q in range(n)])
        e += 2 * n
    return outs


@traced
def publish(a: Addr, i, v, tag, who):
    """This CTA's plain stores made visible device-wide, then (thread ``who``) the
    pair (a, i) := v. The stores drain first: a flag must not overtake its data."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put(a, i, v, tag)
