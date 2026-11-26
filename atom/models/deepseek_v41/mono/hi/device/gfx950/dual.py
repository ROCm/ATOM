# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dual-copy regions at trace time (``Region.home`` "numa2").

MI355X homes every 8 KiB of a 2 MiB page on one of its two I/O dies, alternating: home = address bit 13 XOR a bit of
the page, and a poll of memory homed on the other IOD costs 1.3-1.5x a local round trip. A dual
build keeps two copies of a region in one 2 MiB page of scratch, copy ``i`` in the 8 KiB runs of one parity: logical
byte ``o`` of a region's copy lies at ``2 (o & ~8191) + (o & 8191)`` from the region's start, and copy 1 ``HALF``
(8 KiB) after copy 0 (``plan.program``: regions up to 8 KiB sit inside one run, larger ones start on a run). Copy 0's
runs are all on one IOD and copy 1's on the other (``dsv41hi.runtime.dual`` checks it at startup and per scratch).
A run area larger than a page continues in the next pages (a region never straddles two): each page's copy is chosen
on its own (its bit of the page differs between pages), so a CTA has a ``local`` offset per page.

A stage's ``k.mb[region]`` is copy 0. The emitter registers each region's address (``root``) and, per stage, the
CTA's ``local`` offset: 0 or ``HALF``, the copy on its IOD (``emit.kernel``: the CTA's XCD from HW_REG_XCC_ID,
the copy of each XCD a word of the scratch header). Every buffer resource built from an address is noted
(``buffer_ops.create_buffer_resource_from_addr`` -> ``note``); ``buffer_ops.buffer_load`` reads through
``load_offset`` (the run layout of a larger region, ``place``, then the local copy), so every poll and load is local;
``buffer_ops.buffer_store`` stores at ``place``'s offset and again ``HALF`` bytes on (``other``): the same data and
cache bits into both copies, so the tag in every 8 B half too; ordering (``s_waitcnt vmcnt(0)`` before a flag) covers
both, since both stores are issued before it. Plain global loads / stores of a dual region are refused
(``global_access``). The CTA's copy is chosen once a stage (``local``).

The address walk follows the IR's def chain (``plan.trace._preds``), independent of the memory-model record.
Nothing here emits IR unless a dual program's stages are being traced.
"""

from __future__ import annotations

from contextlib import contextmanager

HALF = 8192  # bytes from a region's copy 0 to its copy 1 (one run of the home rule)
RUN_MASK = 0x3FFFE000  # the run bits of a byte offset that ``place`` spreads (offsets >= 1 GiB stay out of range)

_state: dict | None = (
    None  # while a dual program's stages are traced: roots, resources, memo
)


@contextmanager
def emitting(on: bool):
    """The stages traced inside belong to a dual build (``on``) or not."""
    global _state
    before = _state
    _state = (
        {"roots": {}, "rsrc": {}, "memo": {}, "local": {}, "names": {}} if on else None
    )
    try:
        yield
    finally:
        _state = before


def active() -> bool:
    return _state is not None


def local(value, page: int = 0) -> None:
    """``value`` (Int32: 0 or ``HALF``): the byte offset of this CTA's copy in run-area page ``page``, for the stage
    being traced."""
    if _state is not None:
        _state["local"][page] = value


def root(value, spread: bool, page: int = 0, name: str = "") -> None:
    """``value`` (Int64) is copy 0 of dual region ``name``; ``spread``: the region spans runs (> 8 KiB); ``page``: the
    run-area page it lies in."""
    from ...plan.trace import _ir

    if _state is not None:
        v = _ir(value)
        if v is not None:
            _state["roots"][v] = (value, spread, page)
            _state["names"][v] = name


def spread_byte(x, spread: bool):
    """Copy 0's byte offset of logical byte ``x`` (int or Int32) of a region (``spread``: larger than a run): ``x + (x &
    ~8191)``, the run layout ``place`` gives a resource's accesses."""
    if not spread:
        return x
    if isinstance(x, int):
        return 2 * (x & ~(HALF - 1)) + (x & (HALF - 1))
    return (x & ~(HALF - 1)) + x  # 2 (x & ~8191) + (x & 8191) = (x & ~8191) + x


def note(rsrc, addr) -> None:
    """A buffer resource built over ``addr``."""
    from ...plan.trace import _ir

    if _state is not None:
        v = _ir(rsrc)
        if v is not None:
            _state["rsrc"][v] = addr


def _root(v):
    """The dual root the value ``v`` derives from: (root, (root value, spread, page)) or None; raises on two
    roots."""
    from ...plan.trace import _preds

    memo, roots = _state["memo"], _state["roots"]
    stack, busy = [(v, False)], set()
    while stack:
        x, done = stack.pop()
        if x in memo:
            continue
        if x in roots:
            memo[x] = frozenset([x])
            continue
        preds = _preds(x)
        if not done:
            busy.add(x)
            stack.append((x, True))
            stack.extend((p, False) for p in preds if p not in memo and p not in busy)
            continue
        busy.discard(x)
        acc = set()
        for p in preds:
            acc |= memo.get(p, frozenset())
        memo[x] = frozenset(acc)
    found = memo.get(v, frozenset())
    if not found:
        return None
    if len(found) > 1:
        raise ValueError("an address derives from two dual region copies")
    (r,) = found
    return r, roots[r]


def _of_addr(addr):
    from ...plan.trace import _ir

    v = _ir(addr)
    return None if v is None else _root(v)


def bound(addr, nbytes):
    """The byte bound of a buffer resource over ``addr``: over a dual region it covers the other copy (``HALF`` on)
    and, for a spread region, the run layout (twice the bytes); offsets used to drop an access (>= 1 GiB) stay out of
    range."""
    if nbytes is None:
        return None
    hit = _of_addr(addr)
    if hit is None:
        return nbytes
    _r, (rv, spread, _) = hit
    import flydsl.expr as fx

    if not spread:
        return (
            nbytes + HALF
            if isinstance(nbytes, int)
            else fx.Int64(nbytes) + fx.Int64(HALF)
        )
    # a resource from inside the region: its bytes lie up to twice their logical distance from the copy's start on
    return (fx.Int64(nbytes) + (fx.Int64(addr) - fx.Int64(rv))) * 2 + fx.Int64(2 * HALF)


def _hit(rsrc):
    from ...plan.trace import _ir

    addr = _state["rsrc"].get(_ir(rsrc))
    hit = _of_addr(addr)
    return None if hit is None else (addr, *hit)


def over(rsrc) -> bool:
    """``rsrc`` is over a dual region (its accesses go through ``load_offset`` / ``place`` and ``other``)."""
    return _hit(rsrc) is not None


def place(rsrc, offset):
    """The byte offset (i32 ir.Value) of an access through ``rsrc`` at byte ``offset``: unchanged unless the resource
    is over a spread dual region, where logical byte o (from the region copy's start) lies at o + (o & ~8191).
    """
    hit = _hit(rsrc)
    if hit is None or not hit[2][1]:
        return offset
    import flydsl.expr as fx

    from ...plan.trace import _ir

    addr, r, (rv, _, _) = hit
    rel = fx.Int32(offset)
    if (
        _ir(addr) != r
    ):  # a resource over an address inside the region: its distance from the copy's start
        rel = rel + fx.Int32(fx.Int64(addr) - fx.Int64(rv))
    return (fx.Int32(offset) + (rel & RUN_MASK)).ir_value()


def load_offset(rsrc, offset):
    """The byte offset of a load through ``rsrc`` at byte ``offset``: ``place``'s, in this CTA's copy."""
    import flydsl.expr as fx

    off = place(rsrc, offset)
    if _hit(rsrc) is None:
        return off
    return (fx.Int32(off) + fx.Int32(_state["local"][_hit(rsrc)[2][2]])).ir_value()


def other(rsrc):
    """The extra byte offset that takes a store through ``rsrc`` from copy 0 to copy 1 (``HALF``), or None (not a
    dual region / build)."""
    return None if _hit(rsrc) is None else HALF


def global_access(addr) -> None:
    """Plain global loads / stores do not go through ``place`` / ``other``: refuse them on a dual region."""
    if _state is not None and _of_addr(addr) is not None:
        raise ValueError(
            "a global load / store of a dual region: use buffer_load / buffer_store"
        )
