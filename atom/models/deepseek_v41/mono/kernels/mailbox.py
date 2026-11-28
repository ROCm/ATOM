# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The V4.1 kernels' tagged mailbox: ``atom.mono.device.sync``'s, every region
address (``FreshAddr``) computed at its use from a ``fresh_s64`` base, and the
waits only these kernels use: a 16-bit code in a pair's tag (``put_coded`` /
``poll_coded``), a poll that holds only the specs a lane needs
(``poll_need``), a poll whose retries re-load only the stale specs
(``poll(stale_only=True)``), and other loads issued between a poll's first
pass and its check (``poll(between=...)``). The other models' kernels keep the
shared module, its addresses made once."""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.mono.device.numa import local_copy, xcc_id
from atom.mono.device.ops import rsrc, traced
from atom.mono.device.sync import (
    _SCOPE_CM,
    POLL_MAX,
    Quad,
    _pair_word,
    _spin,
    _spin_bounded,
    _word,
)
from atom.mono.device.sync import Mailbox as SharedMailbox
from atom.mono.plan.layout import dual_places
from atom.mono.plan.trace import Space, note
from atom.mono.polling import POLL_SLEEP_UNITS

# a stale-only poll's buffer records (``Mailbox.poll(stale_only=True)``):
# a lane's settled spec loads at word STALE_WORD, past them -- out of range, so
# no memory access and zeros -- its stale specs in range (a mailbox region is
# far smaller)
STALE_RECORDS = 1 << 30
STALE_WORD = (1 << 29) - 4
assert STALE_WORD * 4 >= STALE_RECORDS


@traced
def _spin_stale(load_all, load_stale, pending, merge):
    """``_spin`` re-loading only each lane's stale specs: a pass's loads
    of a spec whose tags matched are masked out of range (no memory access)
    and its words kept, a tagged pair once current staying so for the layer
    (the same words, the same tag checks). The retry's loads all issue before
    any merge reads one (in-order vmcnt: a merge between them would wait for
    each in turn)."""
    v = load_all()
    while pending(v):
        rocdl.s_sleep(POLL_SLEEP_UNITS)
        w = load_stale(v)
        rocdl.sched_barrier(0)
        v = merge(v, w)
    return v


@traced
def _spin_first(load_all, pending, between):
    """``_spin`` with ``between()`` (other loads) issued after the first pass
    and before its check: vmcnt counts in issue order, so the first check
    waits for the first pass alone, not for ``between``'s loads. The first
    check stays out of the retry loop (a loop phi of a load still in flight
    waits for every load). -> (values, ``between()``'s result)."""
    v = load_all()
    # side-effecting empty asm on both sides of ``between``: the scheduler
    # moves no load across it (it hoisted the weights above the first pass)
    llvm.InlineAsmOp(None, [], "", "", has_side_effects=True)
    extra = between()
    llvm.InlineAsmOp(None, [], "", "", has_side_effects=True)
    if pending(v):
        v = _spin(load_all, pending)
    return v, extra


# ``fresh_s64``'s asm strings: a counter's, or a call site's (``_site``), both
# restarted at each trace. FlyDSL traces a launcher in an MLIR context of its
# own every time, and keys its cache by the traced IR: process-wide counters
# gave a retrace (or another process) other strings, so a rebuild never hit the
# cache. Restarted, a kernel's strings depend on its own trace alone
_FRESH = itertools.count()
_SITES: dict = {}
_TRACE = [None]


def _this_trace():
    """Restart the strings' counters when the current MLIR context is not the
    previous call's: a new trace."""
    global _FRESH
    ctx = ir.Context.current
    if ctx is not _TRACE[0]:
        _TRACE[0], _FRESH = ctx, itertools.count()
        _SITES.clear()


def _site():
    """An id of the traced call's Python stack (every frame's file and line):
    the same in a ``range_constexpr`` loop's unrolled iterations, another at
    each other call."""
    _this_trace()
    f, key = sys._getframe(2), []
    while f is not None:
        key.append((f.f_code.co_filename, f.f_lineno))
        f = f.f_back
    return _SITES.setdefault(tuple(key), len(_SITES))


def fresh_s64(base, site=None):
    """``base`` (a wave-uniform i64) through an ``s_mov_b64`` LLVM takes for a
    new value wherever its asm string differs (no side effects, so it schedules
    freely): ``site`` names it (None: a string of its own). A region's 64-bit
    address (``FreshAddr.value``) is so computed at its use: made once at the
    context's creation, the sums of a layer kernel's ~100 regions were held from
    there to their last use, at the 104-SGPR cap (lane spills; a little more and
    scratch), and computed at the use from a plain base, LLVM would merge a
    region's uses across stages into one held value again. An unrolled loop's
    iterations share their site's string. (SimplifyCFG sinks no inline asm: a
    per-owner put's copies, ``for p``: ``if owner == p``, keep their payload
    code each, code size only where the owner is uniform in the wave.) A kernel
    argument used as a base goes through it once at the kernel's start: one
    value of its own, not a part of a merged kernarg load (a 16-dword tuple
    reloaded whole)."""
    _this_trace()
    tag = f"f{next(_FRESH)}" if site is None else f"s{site}"
    return fx.Int64(
        llvm.inline_asm(
            T.i64, [fx.Int64(base).ir_value()], f"s_mov_b64 $0, $1 ; {tag}", "=s,s",
        )
    )  # fmt: skip


def fresh_s32(v):
    """``v`` (a wave-uniform i32) through an ``s_mov_b32`` of a string of its
    own (``fresh_s64``'s, 32 bits): a kernel argument a value of its own."""
    _this_trace()
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(v).ir_value()], f"s_mov_b32 $0, $1 ; f{next(_FRESH)}", "=s,s",
        )
    )  # fmt: skip


@dataclass(frozen=True)
class FreshAddr:
    """A mailbox region's address, ``sync.Addr``'s with its value computed at
    each use: the region's name and space, recorded when the kernel is traced
    (``atom.mono.plan``), and the traced address (``value``: ``base`` +
    ``offset`` bytes, from a ``fresh_s64`` base). The space also sets the
    hand-off's scope: device for SCRATCH, system for PEER."""

    region: str
    space: Space
    base: object
    offset: object = 0
    # a dual region's (``dreg``): the word offset of the copy this CTA reads
    copy: object = None
    # ``base`` wave-uniform, taken through ``fresh_s64``; False: a lane's own
    # base (a select by lane), used as it is -- an SGPR asm operand takes a
    # lane-varying value through v_readfirstlane, silently: one lane's base
    # for every lane (col_gather's pushes to the lane's rank hung, G303)
    fresh: bool = True

    @property
    def value(self):
        base = fresh_s64(self.base, _site()) if self.fresh else self.base
        return base + fx.Int64(self.offset)


def sreg(base, offset, region):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return FreshAddr(region, Space.SCRATCH, base, offset)


def dreg(base, offset, region, copy):
    """Scratch region ``region`` kept in two copies (``plan.layout.dual_layout``)
    at byte ``offset`` of ``base``: a put writes both, a poll reads the one at
    word offset ``copy`` (``device.numa.local_copy``)."""
    return FreshAddr(region, Space.SCRATCH, base, offset, copy)


def preg(base, offset, region, fresh=True):
    """Peer-symmetric region ``region`` at byte ``offset`` of a rank's buffer
    ``base``; ``fresh`` False where ``base`` varies by lane
    (``FreshAddr.fresh``)."""
    return FreshAddr(region, Space.PEER, base, offset, fresh=fresh)


def region_addrs(scratch, layout, dual: str = "", word=0) -> dict:
    """``atom.mono.device.numa.region_addrs`` as ``FreshAddr`` regions: each
    region of ``layout`` (name -> (byte offset, bytes)) in ``scratch``; one
    named in ``dual`` (``plan.layout.dual_spec``) kept in two copies at its
    offset there (``dreg``), the CTA reading the copy ``word`` gives its XCD --
    a page's copy found once for all its regions."""
    places = dual_places(dual)
    xcc = xcc_id() if places else None
    copies = {}
    out = {}
    for region, (off, _) in layout.items():
        if region in places:
            at, page = places[region]
            if page not in copies:
                copies[page] = local_copy(word, page, xcc)
            out[region] = dreg(scratch, at, region, copies[page])
        else:
            out[region] = sreg(scratch, off, region)
    return out


def _batch_values(specs, kind):
    """``_noted`` of a batch's specs, the address of a region the batch names
    more than once (one ``FreshAddr``) computed once: one buffer resource for its
    loads, as when the value was made once for the kernel."""
    seen, out = {}, []
    for a, _, _ in specs:
        note(a.region, a.space, kind)
        if id(a) not in seen:
            seen[id(a)] = a.value
        out.append(seen[id(a)])
    return out


class Mailbox(SharedMailbox):
    """``sync.Mailbox`` over ``FreshAddr`` regions, with the waits only the
    V4.1 kernels use (module docstring); its puts are the shared class's."""

    def put_coded(self, a, i, word, code):
        """Raw 32-bit ``word`` at pair i as (word, tag | code << 16): a 16-bit
        ``code`` in the tag's high half, taken with the word by one
        ``poll_coded`` (no second pair, no second wait). The tag must fit 16
        bits: a layer's (``layer + 1``), not a launch counter's."""
        vals = [fx.Int32(word), fx.Int32(self.tag) | (code << 16)]
        self._store(a, _word(a, i, 1), fx.Vector.from_elements(vals, fx.Int32))

    def poll_coded(self, specs):
        """``poll`` of ``specs`` in one batch, pairs ``put_coded`` stored: stale
        while any tag's low half is not this launch's tag. -> each spec's values
        and its first pair's code. The stale test ORs every tag's XOR with the
        launch's in VGPRs and compares once (``poll``'s compare a tag, a lane
        mask each, would hold a large batch's SGPRs)."""
        cms = [_SCOPE_CM[a.space] for a, _, _ in specs]
        bases = _batch_values(specs, "poll")
        tag = self.tag

        def load_all():
            words = []
            for b, cm, (a, i, n) in zip(bases, cms, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        _pair_word(a, i, n),
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            acc = v[1] ^ tag
            for e in range_constexpr(3, nw, 2):
                acc = acc | (v[e] ^ tag)
            return (acc & 0xFFFF) != 0

        first = specs[0][1]
        first = first.q * 2 if isinstance(first, Quad) else first
        v = self.spin(specs[0][0], first, load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append(([v[e + 2 * q] for q in range(n)], v[e + 1] >> 16))
            e += 2 * n
        return outs

    def poll_need(self, specs, need, coded=False):
        """``poll`` of one batch whose spec k is waited on only where
        ``need[k]`` (a lane's traced bool): an unneeded spec loads out of
        range (``STALE_WORD`` past a bounded record: no memory access, zeros)
        and never holds the batch. ``coded``: pairs ``put_coded`` stored,
        stale while a tag's low half is not this launch's tag -> each spec's
        values and its first pair's code (``poll_coded``'s)."""
        assert 0 < len(specs) <= POLL_MAX and len(need) == len(specs)
        cms = [_SCOPE_CM[a.space] for a, _, _ in specs]
        bases = _batch_values(specs, "poll")
        srs = [rsrc(b, STALE_RECORDS) for b in bases]
        at = [
            need[k].select(_pair_word(a, i, n), fx.Int32(STALE_WORD))
            for k, (a, i, n) in enumerate(specs)
        ]
        ends = list(itertools.accumulate(2 * n for _, _, n in specs))
        tag = self.tag

        def load_all():
            words = []
            for k in range_constexpr(len(specs)):
                n = specs[k][2]
                w = fx.Vector(
                    bo.buffer_load(
                        srs[k], at[k], vec_width=2 * n, dtype=T.i32,
                        cache_modifier=cms[k],
                    )
                )  # fmt: skip
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        def pending(v):
            bad = None
            for k in range_constexpr(len(specs)):
                e0 = ends[k] - 2 * specs[k][2]
                acc = v[e0 + 1] ^ tag
                for q in range_constexpr(1, specs[k][2]):
                    acc = acc | (v[e0 + 2 * q + 1] ^ tag)
                stale = ((acc & 0xFFFF) if coded else acc) != 0
                bad = need[k] & stale if bad is None else bad | (need[k] & stale)
            return bad

        first = specs[0][1]
        first = first.q * 2 if isinstance(first, Quad) else first
        v = self.spin(specs[0][0], first, load_all, pending)
        outs = []
        for k in range_constexpr(len(specs)):
            e0 = ends[k] - 2 * specs[k][2]
            words = [v[e0 + 2 * q] for q in range(specs[k][2])]
            outs.append((words, v[e0 + 1] >> 16) if coded else words)
        return outs

    def spin(self, a, index, load_all, pending, between=None):
        """Re-issue ``load_all`` until ``pending`` clears: a wait in region
        ``a`` (its word ``index``), bounded and named in a debug build's
        report like every pair's (``poll``). A wait on plain words (counts
        that say themselves when they are in) comes this way too. With
        ``between``: its loads issued after the first pass, before the first
        check (``_spin_first``; a debug build issues them after the wait) ->
        (values, its result)."""
        if const_expr(between is not None and self.diag is None):
            return _spin_first(load_all, pending, between)
        if const_expr(self.diag is None):
            v = _spin(load_all, pending)
        else:
            rid = self.region_ids.get(a.region, -1)
            v = _spin_bounded(load_all, pending, self.diag, rid, index, self.tag)
        if const_expr(between is not None):
            return v, between()
        return v

    def poll(self, specs, batch=POLL_MAX, between=None, stale_only=False):
        """Batched poll of pairs ``specs`` = [(FreshAddr, pair index, npairs
        in {1, 2})], each loaded at its own space's scope; two pairs of a dual
        region at a ``Quad`` (``_word``).

        All pairs are re-loaded together while any tag is stale, so a batch costs
        one round trip after its last producer lands. Returns the value words;
        with ``between`` (one batch) also its result, its loads issued after the
        first pass (``spin``)."""
        assert between is None or 0 < len(specs) <= batch
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        cms = [_SCOPE_CM[a.space] for a, _, _ in specs]
        bases = _batch_values(specs, "poll")
        tag = self.tag

        def load_all():
            words = []
            for b, cm, (a, i, n) in zip(bases, cms, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        _pair_word(a, i, n),
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

        first = specs[0][1]
        first = first.q * 2 if isinstance(first, Quad) else first
        if const_expr(stale_only):
            # one batch whose retries re-load each lane's stale specs
            # alone (``_spin_stale``): the router's LIN / LSSQ poll
            assert between is None and self.diag is None and len(specs) <= batch
            srs = [rsrc(b, STALE_RECORDS) for b in bases]
            at = [_pair_word(a, i, n) for a, i, n in specs]
            ends = list(itertools.accumulate(2 * n for _, _, n in specs))

            def stale(v, k):
                # spec k's pairs not all at this launch's tag (``pending``'s
                # test, a spec at a time)
                e = ends[k] - 2 * specs[k][2]
                bad = v[e + 1] != tag
                for q in range_constexpr(1, specs[k][2]):
                    bad = bad | (v[e + 2 * q + 1] != tag)
                return bad

            def load_masked(live):
                # ``live[k]``: spec k loaded in range where true, else at
                # STALE_WORD (dropped); None: in range
                words = []
                for k in range_constexpr(len(specs)):
                    n = specs[k][2]
                    off = at[k]
                    if const_expr(live[k] is not None):
                        off = live[k].select(off, fx.Int32(STALE_WORD))
                    w = fx.Vector(
                        bo.buffer_load(
                            srs[k], off, vec_width=2 * n, dtype=T.i32,
                            cache_modifier=cms[k],
                        )
                    )  # fmt: skip
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            def load_first():
                return load_masked([None] * len(specs))

            def load_stale(v):
                return load_masked([stale(v, k) for k in range(len(specs))])

            def merge(v, w):
                # a stale spec's words from this pass, a settled one's kept
                words = []
                for k in range_constexpr(len(specs)):
                    bad = stale(v, k)
                    e0 = ends[k] - 2 * specs[k][2]
                    words += [bad.select(w[e], v[e]) for e in range(e0, ends[k])]
                return fx.Vector.from_elements(words, fx.Int32)

            v = _spin_stale(load_first, load_stale, pending, merge)
        else:
            v = self.spin(specs[0][0], first, load_all, pending, between)
        if const_expr(between is not None):
            v, extra = v
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        if const_expr(between is not None):
            return outs, extra
        return outs
