# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Merge retained rank-local TopK roots with K3's consumed-slab lifetime.

Split ranks retain every possible global winner, then select the exact TopK
from their union. Request owners select once and publish their root. Readers
acknowledge consumption before the next layer can reuse a producer's slab.
Contiguous rank key ranges preserve the native ascending-global-index ties.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.ops import (
    CM_SYS,
    rsrc,
    traced,
)
from atom.mono.device.sync import preg, publish
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.dsa_indexer.paged.score import _final, _selected_rows
from atom.mono.dsa_indexer.paged.topk import lds_topk
from atom.mono.plan.execution import THREADS


def _peer_region(c, rank, name):
    return preg(c["peer_addr"](rank), c["ix_peer_layout"][name][0], name)


def _pair_ptr(c, rank, kind):
    return (
        c["peer_addr"](rank)
        + c["peer_plain"]
        + c["ix_plain_layout"][f"ix_{kind}_pairs"][0]
    )


@traced
def _acknowledge(c, t, code, owner, split):
    destinations = [_peer_region(c, p, "ix_done") for p in range(c["world"])]
    at = tp.row_rank_at(code, t, c["rank"], c["S"], c["world"])

    def put_each(dsts, index, value):
        for source, dst in enumerate(dsts):
            active = fx.Boolean(True) if split else source == owner
            if active:
                c["put"](dst, index, value)

    publish(put_each, destinations, at, fx.Int32(1), c["tid"] == 0)
    if const_expr("irdy" in c):  # noqa: SIM102
        if code == 0:
            publish(c["put"], c["irdy"], t, 1, c["tid"] == 0)
    producer = fx.Boolean(True) if split else c["rank"] == owner
    if producer:
        c["poll"](
            [
                (
                    _peer_region(c, c["rank"], "ix_done"),
                    tp.row_rank_at(code, t, p, c["S"], c["world"]),
                    1,
                )
                for p in range(c["world"])
            ]
        )


@traced
def _merge_roots(c, t, kind, code):
    """One peer-read union of retained TopK roots, stable in global-ID order."""
    k = ip.KINDS[kind]
    flags = _peer_region(c, c["rank"], "ix_count")
    counts = c["poll"](
        [
            (flags, tp.row_rank_at(code, t, p, c["S"], c["world"]), 1)
            for p in range(c["world"])
        ]
    )
    starts, ends = [], []
    n = fx.Int32(0)
    for source in range_constexpr(c["world"]):
        starts.append(n)
        n = n + counts[source][0]
        ends.append(n)
    if n == k * c["world"]:
        _merge_root_lists(c, t, kind, n, starts, ends, True)
    else:
        _merge_root_lists(c, t, kind, n, starts, ends, False)
    _acknowledge(c, t, code, fx.Int32(0), True)


@traced
def _merge_root_lists(c, t, kind, n, starts, ends, dense):
    k = ip.KINDS[kind]

    def pair_at(i):
        base = _pair_ptr(c, 0, kind)
        local = i
        for source in range_constexpr(1, c["world"]):
            hit = (i >= starts[source]) & (i < ends[source])
            base = hit.select(_pair_ptr(c, source, kind), base)
            local = hit.select(i - starts[source], local)
        return base, (t * k + local) * 2

    def load(q, i):
        words = []
        if const_expr(dense):
            # Full roots have quad-aligned boundaries. Read each quad in
            # two 16-byte transactions instead of four independent pairs.
            base = _pair_ptr(c, i // k, kind)
            at = (t * k + i % k) * 2
            for half in range_constexpr(2):
                pairs = fx.Vector(
                    bo.buffer_load(
                        rsrc(base),
                        at + half * 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_SYS,
                    )
                )
                for e in range_constexpr(2):
                    fx.ptr_store(
                        pairs[e * 2 + 1], c["sel_lds"]["identities"] + i + half * 2 + e
                    )
                    words.append(pairs[e * 2])
        else:
            for e in range_constexpr(4):
                base, at = pair_at(fx.min(i + e, fx.max(n - 1, 0)))
                pair = fx.Vector(
                    bo.buffer_load(
                        rsrc(base), at, vec_width=2, dtype=T.i32, cache_modifier=CM_SYS
                    )
                )
                fx.ptr_store(pair[1], c["sel_lds"]["identities"] + i + e)
                words.append(pair[0])
        return words

    def emit(pos, d, i, extra, value_bits=None):
        if const_expr(d is None):
            ident = fx.Int32(-1)
        else:
            ident = fx.ptr_load(c["sel_lds"]["identities"] + i)
        if const_expr(kind == "sel"):
            _final(c, kind, t, pos, ident, _selected_rows(c, t, ident))
        else:
            _final(c, kind, t, pos, ident)

    pt = max(4, -(-(k * c["world"]) // (THREADS * 4)) * 4)
    lds_topk(
        n, k, load, emit, c["sel_lds"], c["tid"], pt, fast_hist=True, emit_key=True
    )


@traced
def _read_owned_root(c, t, kind, code, owner):
    flags = _peer_region(c, c["rank"], "ix_count")
    count = c["poll"]([(flags, tp.row_rank_at(code, t, owner, c["S"], c["world"]), 1)])[
        0
    ][0]
    pairs = _pair_ptr(c, owner, kind)
    k = ip.KINDS[kind]
    for rep in range_constexpr(-(-k // THREADS)):
        i = c["tid"] + rep * THREADS
        ident = fx.Int32(-1)
        if i < count:
            ident = fx.Int32(
                bo.buffer_load(
                    rsrc(pairs),
                    (t * k + i) * 2 + 1,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
            )
        if const_expr(kind == "sel"):
            _final(c, kind, t, i, ident, _selected_rows(c, t, ident))
        else:
            _final(c, kind, t, i, ident)
    _acknowledge(c, t, code, owner, False)


@traced
def _finish_merged(c, t, kind, mode, owner):
    code = tp.KINDS.index(kind)
    if mode != tp.LOCAL:
        if (mode == tp.SPLIT) | (c["rank"] == owner):
            count = c["poll"]([(c["ix_root_ready"], code * c["S"] + t, 1)])[0][0]
            _publish_count(c, t, code, count)
        if mode == tp.SPLIT:
            _merge_roots(c, t, kind, code)
        else:
            _read_owned_root(c, t, kind, code, owner)


@traced
def _publish_count(c, t, code, count):
    flags = [_peer_region(c, p, "ix_count") for p in range(c["world"])]
    at = tp.row_rank_at(code, t, c["rank"], c["S"], c["world"])

    def put_each(dsts, i, value):
        for dst in dsts:
            c["put"](dst, i, value)

    publish(put_each, flags, at, count, c["tid"] == 0)


@traced
def finish_row(c, t, kind, mode, owner):
    if const_expr(c.get("tp_direct", False)):
        if mode == tp.SPLIT:
            _read_direct(c, t, kind)
        else:
            _finish_merged(c, t, kind, mode, owner)
    else:
        _finish_merged(c, t, kind, mode, owner)


@traced
def _read_direct(c, t, kind):
    code = tp.KINDS.index(kind)
    flags = _peer_region(c, c["rank"], "ix_count")
    counts = c["poll"](
        [
            (flags, tp.row_rank_at(code, t, p, c["S"], c["world"]), 1)
            for p in range(c["world"])
        ]
    )
    k = ip.KINDS[kind]
    offset = fx.Int32(0)
    for source in range_constexpr(c["world"]):
        count = counts[source][0]
        pairs = _pair_ptr(c, source, kind)
        for quad in range(c["tid"], ip.cdiv(count, 4), THREADS):
            first = quad * 4
            ids = fx.Vector(
                bo.buffer_load(
                    rsrc(pairs),
                    t * k * 2 + fx.min(first, fx.max(count - 4, 0)),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_SYS,
                )
            )
            # A short tail uses a scalar load because winner storage is
            # packed, so count need not end on a quad boundary.
            for e in range_constexpr(4):
                if first + e < count:
                    ident = ids[e]
                    if count % 4 != 0:
                        ident = fx.Int32(
                            bo.buffer_load(
                                rsrc(pairs),
                                t * k * 2 + first + e,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=CM_SYS,
                            )
                        )
                    if const_expr(kind == "sel"):
                        _final(
                            c,
                            kind,
                            t,
                            offset + first + e,
                            ident,
                            _selected_rows(c, t, ident),
                        )
                    else:
                        _final(c, kind, t, offset + first + e, ident)
        offset = offset + count
    for pos in range(c["tid"], k, THREADS):
        if pos >= offset:
            if const_expr(kind == "sel"):
                _final(
                    c, kind, t, pos, fx.Int32(-1), _selected_rows(c, t, fx.Int32(-1))
                )
            else:
                _final(c, kind, t, pos, fx.Int32(-1))
    _acknowledge(c, t, code, fx.Int32(0), True)
