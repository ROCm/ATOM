# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Peer regions of a TP program, seen from a stage (``plan.program``: tensor parallelism).

A peer region holds ``PEER_SLOTS`` x ``tp`` copies of its ``pairs`` pairs in every rank's peer buffer:
[launch-tag parity][source rank]. Rank ``r`` puts its data into slot ``(parity, r)`` of every rank's
buffer (``to_rank``, its own included) and polls slots ``(parity, src)`` of its own (``from_rank``).
Puts and polls of these ``Addr``s use system scope (``mailbox``: ``Space.PEER``). A shared program
(``Program.peer_shared``) holds a region's parity-1 slots mirrored from the group's top (``k.peer_top``), so programs
of other layouts on its group never write where its launch's slots lie (``plan.program``).

Every helper is a free function of explicit values.
"""

import flydsl.expr as fx
from flydsl.expr.typing import T

from ...plan.program import PAIR_BYTES
from ...plan.trace import Space, note
from .mailbox import Addr, shift
from .ops import load_ptr64, traced, uniform


def _slot_bytes(k, region: str, src, tag=None) -> object:
    """Byte offset of slot (the launch's parity, ``src``) within the region: this launch's, or (``tag``) the one
    whose tag is ``tag`` (a ``prev_reads`` region: the previous launch's)."""
    pairs = k.peer_pairs[region]
    parity = fx.Int32(k.tag if tag is None else tag) & 1
    if (
        k.program.peer_shared
    ):  # parity 1: the region's tp slots at top - offset - tp slots
        # (the group's step phase on top of the tag: a step of an odd number of launches flips it, so the next
        # step's first launch has the other parity than this one's last: ``PeerGroup.end_step``)
        parity = (fx.Int32(k.tag if tag is None else tag) + k.peer_phase) & 1
        mirror = k.peer_top - fx.Int64(
            2 * k.program.peer[region] + k.tp * pairs * PAIR_BYTES
        )
        return fx.Int64(parity) * mirror + fx.Int64(src * pairs) * PAIR_BYTES
    return fx.Int64((parity * k.tp + src) * pairs) * PAIR_BYTES


def from_rank(k, region: str, src, tag=None) -> Addr:
    """This rank's copy of ``region``, the slot rank ``src`` puts into (polled here); ``tag``: of that launch."""
    return shift(k.mb[region], _slot_bytes(k, region, src, tag))


def peer_base(k, dst):
    """Rank ``dst``'s peer buffer, from the peer table in memory."""
    return load_ptr64(k.peers, dst)


def to_rank(k, region: str, dst) -> Addr:
    """Rank ``dst``'s copy of ``region``, the slot this rank puts into."""
    base = peer_base(k, dst) + fx.Int64(k.program.peer[region])
    a = Addr(region, Space.PEER, base)
    return shift(a, _slot_bytes(k, region, k.rank))


# ------------------------------------------------------------------ TP partials (the MoE's; the producer pushes, a
# consumer sums)
#
# A partial region holds a [rows, width] row-major partial per slot. ``fmt`` "bf16": a pair = two packed bf16 values
# (pair (row * width + col) / 2); "f32": two planes, values 0-1 then 2-3 of every 4-element item (pair
# plane * pairs / 2 + (row * width + col) / 2), so each store instruction of a wave stays contiguous.


def partial_pairs(fmt: str, rows: int, width: int) -> int:
    """Pairs a slot of a [rows, width] partial takes."""
    if (
        fmt == "f32p"
    ):  # 3 pairs a 4-element item (a _pack3 quad + a (value, tag) pair), a whole 16 B a slot
        items = rows * width // 4
        return 3 * items + (items & 1)
    return rows * width // (2 if fmt == "bf16" else 1)


def push_partial(k, region: str, fmt: str, e, vals4, dsts=None) -> None:
    """This rank's partial of elements ``e .. e + 3`` (``e`` a multiple of 4) into every rank's slot (``dsts``: only
    these ranks, e.g. one a task: a CTA's remote puts are the limit)."""
    from . import mailbox as mb

    half = k.peer_pairs[region] // 2  # f32: the second plane's first pair
    dl = list(range(k.tp)) if dsts is None else list(dsts)
    for i, dst in enumerate(dl):
        a = to_rank(k, region, dst)
        if fmt == "bf16":
            mb.put_bf(a, e, vals4, k.tag)
        else:
            mb.put2(a, e // 2, vals4[0], vals4[1], k.tag)
            mb.put2(a, half + e // 2, vals4[2], vals4[3], k.tag)


def sum_partials(k, region: str, fmt: str, e, tp: int, tag=None) -> list:
    """Elements ``e .. e + 3`` summed over the ``tp`` ranks' partials in rank order (fp32), one batched poll.
    ``tag``: the producing launch's (default this launch's; a ``Stage.prev_reads`` region: the previous launch's,
    its parity slot and its tag)."""
    tag = k.tag if tag is None else tag
    from . import mailbox as mb

    if fmt == "f32p":
        return _sum_p(k, region, e, tp, tag)
    half = k.peer_pairs[region] // 2
    specs = []
    for src in range(tp):
        specs += [
            (from_rank(k, region, src, tag), e // 2, 2),
            (from_rank(k, region, src, tag), half + e // 2, 2),
        ]
    got = mb.poll(specs, tag)
    parts = [
        [w.bitcast(fx.Float32) for w in got[2 * s] + got[2 * s + 1]] for s in range(tp)
    ]
    acc = parts[0]
    for s in range(1, tp):
        acc = [acc[q] + parts[s][q] for q in range(4)]
    return acc


# ---- two-shot TP sum (reduce-scatter, then all-gather). Rank q owns columns q * width / tp ..
# + width / tp - 1 of every row. Shot 1 (``push_rs``): each rank pushes its fp32 partial of those columns to q only,
# region ``rs`` (a slot a source rank, [rows, width / tp] in the "f32" two-plane layout, or packed: "f32p", a lane's
# tagged quad of 3 values + a pair, or "f32q", 3 values + tags a 16 B quad across lanes). The
# owner sums them in rank order (``sum_partials(k, rs, fmt, el, tp)``: the one-shot sum's bits, any format) and shot 2
# (``push_partial(k, ag, fmt, el, sums)``) puts the sums into every rank's region ``ag`` (a slot an owner; fmt "bf16"
# when the consumer rounds the sum to bf16 anyway: half the bytes, the same bits). ``read_ag`` reads element e from its
# owner's slot. A link carries 2 x 1 / tp of the partial instead of all of it, for one more hop.


def shot2_pairs(rows: int, width: int, tp: int, fmt: str = "f32") -> int:
    """Pairs a slot of a two-shot region (``rs``: "f32" or "f32q"; ``ag``: "f32" or "bf16") takes. "f32q" (the
    rs partials packed, ``push_rs(fmt="f32q")``): an owner's columns a whole number of 16-element groups.
    """
    if fmt == "f32p":
        if (width // tp) % 4 or width % tp:
            raise ValueError(
                f"rs 'f32p': width / tp a multiple of 4 (4-element items an owner), got {width} / {tp}"
            )
        return partial_pairs("f32p", rows, width // tp)
    return rows * width // tp // (2 if fmt == "bf16" else 1)


def owned(e, width: int, tp: int):
    """(owner rank, element index in the owner's [rows, width / tp] slice) of element ``e`` of [rows, width]."""
    q = width // tp
    col = e % width
    return col // q, (e // width) * q + col % q


def push_rs(
    k, region: str, e, vals4, width: int, fmt: str = "f32", two_owners: bool = False
) -> None:
    """Shot 1: this rank's partial of elements ``e .. e + 3`` (one owner's columns) to their owner only. The owner must
    be the same for a wave's lanes (its slot address is wave-uniform): a wave's elements within one owner's columns.
    ``two_owners``: a wave's elements may cross one owner boundary (an owner's ``width / tp`` columns at least a
    wave's span, not a multiple of it: e.g. 640 at 8 ranks against a 256-column wave): the lanes of the first active
    lane's owner put there, the rest to the next owner (two wave-uniform addresses, predicated stores).
    ``fmt`` "f32p" / "f32q": the owner's slot in a packed layout (``_push_p``: 6 B a value instead of 8, lane-local,
    with ``two_owners`` too; "f32q": 5.5 B, lanes 4m .. 4m + 3 one 16-element group in order, every lane of a
    group pushing, width / tp a multiple of 16, one owner a wave); the owner sums with ``sum_partials(fmt=)``, the same
    bits."""
    from . import mailbox as mb

    owner, el = owned(e, width, k.tp)
    if fmt == "f32p" and not two_owners:
        return _push_p(k, region, el, vals4, [owner])
    half = k.peer_pairs[region] // 2
    if not two_owners:
        a = to_rank(k, region, owner)
        mb.put2(a, el // 2, vals4[0], vals4[1], k.tag)
        mb.put2(a, half + el // 2, vals4[2], vals4[3], k.tag)
        return
    lo = uniform(owner)
    hi = fx.min(lo + 1, fx.Int32(k.tp - 1))
    a_lo, a_hi = to_rank(k, region, lo), to_rank(k, region, hi)
    _put_rs(a_lo, owner == lo, el, half, vals4, k.tag)
    _put_rs(a_hi, owner != lo, el, half, vals4, k.tag)


# ---- gathered regions (a sharded residual's all-gather, ``HCSeam(residual="sharded")``): rank q computes columns
# q * width / tp .. of a row-major [rows, width] tensor and puts them into every rank's slot q; a reader takes any
# element from its owner's slot (from slot (parity, 0): ``from_rank(k, region, 0)``). A slot is tiled
# [tile][row][``tile`` pairs] so a producer task's tile (``tile`` pairs of every row) is one contiguous run of puts
# (a store instruction's peer stores contiguous).


@traced
def _put_rs(a, on, el, half, vals4, tag):
    from . import mailbox as mb

    if on:
        mb.put2(a, el // 2, vals4[0], vals4[1], tag)
        mb.put2(a, half + el // 2, vals4[2], vals4[3], tag)


def tile_local(row, cl, rows: int, tile: int):
    """Pair index within an owner's slot of the pair at row ``row``, column pair ``cl`` of the owner's columns."""
    return ((cl // tile) * rows + row) * tile + cl % tile


def tile_index(w, wp: int, tp: int, rows: int, tile: int):
    """Pair ``w`` of a row-major [rows, wp] pair tensor -> its index from slot (parity, 0) of a gathered region (its
    owner's slot, then ``tile_local``; a slot is rows x wp / tp pairs)."""
    q = wp // tp
    c = w % wp
    return (c // q) * (rows * q) + tile_local(w // wp, c % q, rows, tile)


def read_ag(k, region: str, e, width: int, fmt: str = "f32", tag=None) -> list:
    """Shot 2's sums of elements ``e .. e + 3`` (one owner's columns; the owner may differ per lane), one poll."""
    tag = k.tag if tag is None else tag
    from . import mailbox as mb
    from .ops import bf2_f32

    if (
        mb._live is not None
    ):  # dyn_rows: a dead row reads the last live row's sums (its owner put no others)
        e = fx.min(fx.Int32(e), mb._live * width - 4)
    owner, el = owned(e, width, k.tp)
    pairs = k.peer_pairs[region]
    a = from_rank(
        k, region, 0, tag
    )  # slot (parity, 0); the owner's slot follows at owner x pairs
    if fmt == "bf16":
        got = mb.poll([(a, owner * pairs + el // 2, 2)], tag)[0]
        return list(bf2_f32(got[0])) + list(bf2_f32(got[1]))
    got = mb.poll(
        [(a, owner * pairs + el // 2, 2), (a, owner * pairs + pairs // 2 + el // 2, 2)],
        tag,
    )
    return [w.bitcast(fx.Float32) for w in got[0] + got[1]]


# ---- two-field partials, 3 values + a tag in every 16 B: a thread's 4 elements of two fp32
# fields (a, b) are 3 dwordx4 stores [a0 a1 a2 tag] [a3 b0 b1 tag] [b2 b3 0 tag], 12 B an element instead of two tagged
# pairs' 16 B, in 3 planes (quad v of item j at word (v * items + j) * 4: each store instruction of a wave
# contiguous). A 16 B store reaches a peer as two 8 B halves that can land apart (measured: words 0-1 stale
# with words 2-3 current), so each half carries the tag's low 16 bits (``_pack3``).


def _pack3(x, y, z, tag):
    """Three 32-bit words + a tag as 4 words in two self-tagged 8 B halves: [x, y.lo16 << 16 | t] [y.hi16 << 16 | t, z]
    (t = the tag's low 16 bits: stale data is at most two launches old, so 16 bits never alias).
    """
    t = fx.Int32(tag) & 0xFFFF
    y = fx.Int32(y)
    return [fx.Int32(x), (y << 16) | t, (y & fx.Int32(-65536)) | t, fx.Int32(z)]


def _unpack3(w):
    """(x, y, z) of a ``_pack3`` quad's words."""
    y = ((w[1] >> 16) & 0xFFFF) | (w[2] & fx.Int32(-65536))
    return w[0], y, w[3]


def _tags_ok(w, tag):
    t = fx.Int32(tag) & 0xFFFF
    return ((w[1] & 0xFFFF) == t) & ((w[2] & 0xFFFF) == t)


# ---- "f32q": one fp32 field, 3 values + a tag a 16 B quad (5.5 B a value, f32 pairs' 8)
#
# A producer's 4 lanes 4m .. 4m + 3 must hold one 16-element group's elements in order (lane j: 4 j .. 4 j + 3, the
# group 16-aligned: Linear's epilogue items and the MoE combine are). Group G's 16 values:
#   plane A, quad 4 G + j (lane j): j 0: (0 1 2), j 1: (3 4 5), j 2: (6 7 8), j 3: (12 13 14)
#   plane B, quad G (lane 2): (9 10 11);  plane C, pair G (lane 3): 15
# lane j > 0 takes elements 4 j - 2, 4 j - 1 from lane j - 1 (ds_bpermute; every lane of a group calls the push).
# Each store instruction of a wave stays contiguous. A reader's 4 elements are 2 loads: quad A (4 G + j) and the
# next one (j 0, 1: A 4 G + j + 1; j 2: B G; j 3: C G).


# ---- "f32p": a lane's own 4 fp32 values as a ``_pack3`` quad (values 0-2, a tag in each 8 B half) and a (value 3,
# tag) pair: 6 B a value (f32 pairs' 8, f32q's 5.5), no lane exchange and no lane-dependent store (the peer measurements
# P16). Plane A: item i's quad at word 4 i; plane C: its pair at pair 2 items + i. Each store instruction of a wave
# stays contiguous; a reader's 4 elements are one 16 B and one 8 B load.


def partial3_pairs(rows: int, width: int) -> int:
    """Pairs a slot of a [rows, width] two-field partial takes (12 words a 4-element item)."""
    return rows * width * 3 // 2


def _quad_word(k, region, e, v):
    """Word of quad ``v`` of the item holding element ``e`` (planes of ``items`` quads)."""
    items = k.peer_pairs[region] * 2 // 12
    return (v * items + e // 4) * 4


def sum_partials3(k, region: str, e, tp: int, tag=None):
    """The ``tp`` ranks' elements ``e .. e + 3`` of both fields, re-loaded together until every quad's tag is
    current, each field summed in rank order (fp32). -> (a4, b4). ``tag``: the producing launch's (default this
    launch's; a ``Stage.prev_reads`` region: the previous launch's, its parity slot and its tag).
    """
    tag = k.tag if tag is None else tag
    from . import buffer_ops as bo
    from . import mailbox as mb
    from .ops import CM_SYS, rsrc

    bases = []
    for src in range(tp):
        a = from_rank(k, region, src, tag)
        note(a.region, a.space, "poll")
        bases.append(a.value)

    def load_all():
        words = []
        for b in bases:
            for v in range(3):
                w = fx.Vector(bo.buffer_load(rsrc(b), _quad_word(k, region, e, v), vec_width=4, dtype=T.i32,
                                             cache_modifier=CM_SYS))  # fmt: skip
                words += [w[i] for i in range(4)]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = None
        for q in range(3 * tp):
            # (`== False`: an fx boolean, which `not` would test as a Python object)
            b = _tags_ok([v[4 * q + i] for i in range(4)], tag) == False
            bad = b if bad is None else bad | b
        return bad

    v = mb._spin(load_all, pending)
    a4, b4 = None, None
    for s in range(tp):
        w = [
            x
            for q in range(3)
            for x in _unpack3([v[12 * s + 4 * q + i] for i in range(4)])
        ]  # the 9 values
        av = [w[i].bitcast(fx.Float32) for i in range(4)]
        bv = [w[4 + i].bitcast(fx.Float32) for i in range(4)]
        a4 = av if a4 is None else [a4[q] + av[q] for q in range(4)]
        b4 = bv if b4 is None else [b4[q] + bv[q] for q in range(4)]
    return a4, b4


def _p_items(k, region):
    return k.peer_pairs[region] // 3  # (the pad pair of an odd item count rounds away)


def _push_p(k, region, e, v, dsts, on=None):
    """``on``: per destination, this lane's predicate (``push_rs``' two owners)."""
    items, i = _p_items(k, region), e // 4
    bits = [fx.Float32(x).bitcast(fx.Int32) for x in v]
    quad = fx.Vector.from_elements(_pack3(bits[0], bits[1], bits[2], k.tag), fx.Int32)
    dl = list(range(k.tp)) if dsts is None else list(dsts)
    for n, dst in enumerate(dl):
        a = to_rank(k, region, dst)
        note(a.region, a.space, "put")
        if on is None:
            _p_put(a, items, i, quad, bits[3], k.tag)


def _p_put(a, items, i, quad, v3, tag):
    from . import buffer_ops as bo
    from . import mailbox as mb
    from .ops import CM_SYS, rsrc

    bo.buffer_store(quad, rsrc(a.value), i * 4, cache_modifier=CM_SYS)
    mb.put(a, 2 * items + i, v3, tag)


def _sum_p(k, region, e, tp, tag):
    from . import buffer_ops as bo
    from . import mailbox as mb
    from .ops import CM_SYS, rsrc

    items, i = _p_items(k, region), e // 4
    bases = []
    for src in range(tp):
        a = from_rank(k, region, src, tag)
        note(a.region, a.space, "poll")
        bases.append(a.value)

    def load_all():
        words = []
        for b in bases:
            q = fx.Vector(
                bo.buffer_load(
                    rsrc(b), i * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_SYS
                )
            )
            c = fx.Vector(
                bo.buffer_load(
                    rsrc(b),
                    (2 * items + i) * 2,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
            )
            words += [q[j] for j in range(4)] + [c[0], c[1]]
        return fx.Vector.from_elements(words, fx.Int32)

    def pending(v):
        bad = None
        for s_ in range(tp):
            # (`== False`: an fx boolean, which `not` would test as a Python object)
            stale = _tags_ok([v[6 * s_ + j] for j in range(4)], tag) == False
            b = stale | (v[6 * s_ + 5] != fx.Int32(tag))
            bad = b if bad is None else bad | b
        return bad

    v = mb._spin(load_all, pending)
    acc = None
    for s_ in range(tp):
        x, y, z = _unpack3([v[6 * s_ + j] for j in range(4)])
        f = [w.bitcast(fx.Float32) for w in (x, y, z, v[6 * s_ + 4])]
        acc = f if acc is None else [acc[q] + f[q] for q in range(4)]
    return acc
