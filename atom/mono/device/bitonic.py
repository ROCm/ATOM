# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Bitonic networks over one wave's 64 lanes, an entry a lane: a signed
64-bit sort key (larger first), exchanged by xor shuffles -- no LDS, no
barrier, the same compare-exchanges whatever the data.

- ``wave_sort``: the wave's 64 entries in descending order (lane 0 the
  largest);
- ``wave_merge_top``: the 64 largest of two such sorted waves' entries,
  sorted: c_i = max(a_i, b_{63 - i}) is bitonic and holds them, then one
  bitonic merge.
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from atom.mono.device.ops import i64_of, i64_words, lane_gather, xshfl

LANES = 64


def _partner(v, j):
    """Lane ``lane ^ j``'s 64-bit value."""
    lo, hi = i64_words(v)
    return i64_of(xshfl(lo, j), xshfl(hi, j))


def _exchange(v, lane, j, desc):
    """One compare-exchange of the pair (lane, lane ^ j): the lower lane keeps
    the larger where ``desc`` (the pair's block sorts descending), else the
    smaller."""
    other = _partner(v, j)
    lower = (lane & j) == 0
    keep_max = (lower & desc) | (~lower & ~desc)
    return keep_max.select(fx.max(v, other), fx.min(v, other))


def _merge(v, lane):
    """A bitonic wave sorted descending: strides 32 .. 1, the lower lane of
    each pair the larger."""
    for j in (32, 16, 8, 4, 2, 1):
        other = _partner(v, j)
        v = ((lane & j) == 0).select(fx.max(v, other), fx.min(v, other))
    return v


def wave_sort(v, lane):
    """The wave's 64 entries (``v``, a lane's) sorted descending."""
    for k in (2, 4, 8, 16, 32):
        # blocks of k alternate in direction, then the whole wave descends
        desc = (lane & k) == 0
        j = k // 2
        for _ in range_constexpr(k.bit_length() - 1):
            v = _exchange(v, lane, j, desc)
            j //= 2
    return _merge(v, lane)


def wave_merge_top(a, b, lane):
    """The 64 largest of two waves' entries, each sorted descending, sorted
    descending."""
    lo, hi = i64_words(b)
    rev = fx.Int32(LANES - 1) - lane
    b_rev = i64_of(lane_gather(lo, rev), lane_gather(hi, rev))
    return _merge(fx.max(a, b_rev), lane)
