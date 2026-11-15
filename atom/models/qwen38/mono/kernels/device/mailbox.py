# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ATOM atom/mono/device/sync.py (MIT, Copyright (C) 2026 Advanced
# Micro Devices, Inc.) and the FlyDSL GLM-5 layer kernel.
"""Tagged-pair hand-offs between CTAs.

Every 32-bit payload is stored next to the launch's tag as one 8- or 16-byte
coherent store; a consumer polls the payload itself until every tag matches, so
a hand-off costs one round trip. Tags are ``step * LAYER_SLOTS + layer + 1`` and
scratch is never cleared: a pair from an earlier launch does not carry the
current tag. Every address is an ``Addr`` naming its region, recorded at trace
time for the contract check (``atom.mono.plan.check``).

All functions take the tag explicitly (``k.tag`` of the stage context).

Tags are compared for equality only: a launch's tag must differ from the tags
the previous launch on the same scratch left behind (every polled pair is
rewritten by every launch), not be larger than them. Nothing depends on tag
order, so the int32 tag wrapping around is harmless.

**Dual-copy regions** (``Program.dual``): MI355X homes every 8 KiB of a 2 MiB page on one of its two I/O dies,
alternating. A dual build keeps two copies of a region in the run area at the start of scratch
(``program._dual_layout``): logical byte ``o`` of copy 0 lies at ``2 (o & ~8191) + (o & 8191)`` from the
region's start for a region larger than a run (``spread``), at ``o`` otherwise, and copy 1 ``HALF`` bytes after copy
0. ``store`` writes both copies with the same data and cache bits; ``load`` reads the copy of the CTA's I/O die (the
header word's bit for its XCD, ``dual``), recomputed at each load.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.device.ops import (
    CM_DEV,
    bf16_pair,
    rsrc,
    spin_pause,
    traced,
)
from atom.mono.plan.trace import Space, note

POLL_MAX = 12  # pair specs per poll batch: bounds live registers
_SCOPE_CM = {Space.SCRATCH: CM_DEV}
# bytes from a dual region's copy 0 to its copy 1 (one run of the home rule)
HALF = 8192


@dataclass(frozen=True)
class Addr:
    """A region's traced base address (``value``), with its name and space for the record. Every region of a stage
    is reached through one buffer resource over scratch (``base``), the region's byte offset ``off`` the
    instruction's scalar offset (a constant LLVM rematerializes, not an address held in SGPRs).

    ``dual``: a dual region's header word (its byte offset in scratch; bit x: the copy XCD x reads), None for one
    copy; ``spread``: the region spans runs (its bytes are placed, ``spread_byte``)."""

    region: str
    space: Space
    value: object
    base: object
    off: int
    dual: object = None
    spread: bool = False


def spread_byte(x, spread: bool):
    """Copy 0's byte offset of logical byte ``x`` (Int32) of a region (``spread``: larger than a run)."""
    if not spread:
        return x
    return (x & ~(HALF - 1)) + x  # 2 (x & ~8191) + (x & 8191) = (x & ~8191) + x


def _dual_byte(a: Addr, i):
    """Copy 0's byte offset, from the region's start, of word ``i`` of a dual ``a``."""
    return fx.Int32(spread_byte(0 + fx.Int32(i) * 4, a.spread))


def _local(a: Addr):
    """This CTA's copy offset (0 or ``HALF``) of a dual ``a``: the header word's bit for the CTA's XCD. The XCC
    read has side effects so it is redone at each use, not hoisted into an SGPR held across the stage.
    """
    from flydsl._mlir.dialects import llvm

    hdr = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<4>"), (fx.Int64(a.base) + fx.Int64(a.dual)).ir_value()
    )
    w = fx.Int32(llvm.LoadOp(T.i32, hdr.result, alignment=4, invariant=True).result)
    xcc = llvm.InlineAsmOp(
        T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s", has_side_effects=True
    )
    return ((w >> (fx.Int32(xcc.result) & 7)) & 1) * HALF


def _noted(a: Addr, kind: str):
    """Record the access; -> (the shared resource, the region's offset in it)."""
    note(a.region, a.space, kind)
    return rsrc(a.base), a.off


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _aux(cm):
    """The buffer instruction's cache-policy bits (``cache_modifier``'s encoding) as an attribute."""
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cm)


def store(a: Addr, data, i, cm=CM_DEV):
    """Store ``data`` (an int32 or a vector of them) at word ``i`` of ``a``; recorded as a put. A dual region: both
    copies, the same data and cache bits."""
    r, so = _noted(a, "put")
    if a.dual is not None:
        x = _dual_byte(a, i)
        for c in (0, HALF):
            rocdl.raw_ptr_buffer_store(
                _ir(data),
                rocdl.get_buffer_rsrc(r),
                _ir(fx.Int32(x + c)),
                _ir(fx.Int32(a.off)),
                aux=_aux(cm),
            )
        return
    rocdl.raw_ptr_buffer_store(
        _ir(data),
        rocdl.get_buffer_rsrc(r),
        _ir(fx.Int32(i) * 4),
        _ir(fx.Int32(so)),
        aux=_aux(cm),
    )


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


def load(a: Addr, i, n, cm=CM_DEV):
    """``n`` int32 words (an Int32, or a vector for n > 1) at word ``i`` of ``a``, issued now; recorded as a poll.
    A dual region: this CTA's copy."""
    r, so = _noted(a, "poll")
    ty = T.i32 if n == 1 else ir.VectorType.get([n], T.i32)
    off, soff = _ir(fx.Int32(i) * 4), _ir(fx.Int32(so))
    if a.dual is not None:  # this CTA's copy
        off, soff = _ir(_dual_byte(a, i) + _local(a)), _ir(fx.Int32(a.off))
    return rocdl.raw_ptr_buffer_load(
        ty, rocdl.get_buffer_rsrc(r), off, soff, aux=_aux(cm)
    )


def load_pairs(a: Addr, i, n, cm=CM_DEV):
    """``n`` words from pair ``i`` of ``a`` (``load`` at word ``2 i``)."""
    return load(a, fx.Int32(i) * 2, n, cm)


def put(a: Addr, i, v, tag):
    """Pair ``i`` := (v, tag); ``v`` an f32 or int32 bits."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    _store(a, [bits, tag], i * 2)


def put2(a: Addr, i, v0, v1, tag):
    """Pairs ``i``, ``i + 1`` (i even) in one 16-byte store."""
    words = []
    for v in (v0, v1):
        words += [
            v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v),
            tag,
        ]
    _store(a, words, i * 2)


def put_bf(a: Addr, i, vs, tag):
    """Elements ``i .. i + len(vs)`` (2 or 4, i aligned) as packed bf16 pairs:
    pair ``i // 2 + j`` := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one store."""
    words = []
    for j in range_constexpr(len(vs) // 2):
        words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
    _store(a, words, i)


def put_words(a: Addr, i, words, tag):
    """Raw 32-bit words (1 or 2) at pairs ``i ..`` as (word, tag): integer payloads
    (fp8 words) must come this way, an f32 round trip would not keep NaN patterns."""
    vals = []
    for w in words:
        vals += [fx.Int32(w), tag]
    _store(a, vals, i * 2)


@traced
def _spin(load_all, pending):
    v = load_all()
    while pending(v):
        spin_pause()
        v = load_all()
    return v


def poll(specs, tag, batch=POLL_MAX):
    """Batched poll of ``specs`` = [(Addr, pair index, npairs in {1, 2})], one space.

    All pairs are re-loaded together while any tag is stale, so a batch costs one
    round trip after its last producer lands. Returns each spec's value words."""
    if const_expr(len(specs) == 0):
        return []
    if const_expr(len(specs) > batch):
        return poll(specs[:batch], tag, batch) + poll(specs[batch:], tag, batch)
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


def poll_f32(specs, tag):
    """[(Addr, pair index)] single pairs -> list of f32."""
    return [v[0].bitcast(fx.Float32) for v in poll([(a, i, 1) for a, i in specs], tag)]
