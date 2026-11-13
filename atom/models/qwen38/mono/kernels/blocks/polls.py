# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-phase, maskable polls of tagged pairs (the attention blocks' input and exchange polls).

``mpoll_start(specs)`` issues one attempt of every spec ``(Addr, pair index, npairs in {1, 2},
ok)`` (``ok``: None, one condition, or one per pair) without waiting, so a block can put other
loads behind it (a CU returns loads in order); ``mpoll_finish(specs, first, tag)`` re-polls
until every pair whose ``ok`` holds (or is None) carries ``tag``. A spec's index must be in
range even when ``ok`` is false: it is loaded and its tag ignored (no out-of-range offsets, so
no region bound is needed).
"""

import flydsl.expr as fx

from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.ops import spin_pause, traced
from atom.mono.plan.trace import note


def _mpoll_load(specs):
    words = []
    for a, i, n, _ in specs:
        v = fx.Vector(mb.load(a, fx.Int32(i) * 2, 2 * n))
        words += [v[e] for e in range(2 * n)]
    return fx.Vector.from_elements(words, fx.Int32)


def _mpoll_stale(v, specs, tag):
    bad, e = None, 0
    for _, _, n, ok in specs:
        for q in range(n):
            x = v[e + 2 * q + 1] != tag
            okq = ok[q] if isinstance(ok, (tuple, list)) else ok
            if okq is not None:
                x = x & okq
            bad = x if bad is None else bad | x
        e += 2 * n
    return bad


def mpoll_start(specs):
    """Issue a masked poll [(Addr, pair index (in range), npairs in {1, 2}, ok | None)]."""
    for a, *_ in specs:
        note(a.region, a.space, "poll")
    return _mpoll_load(specs)


@traced
def _mpoll_spin(v, specs, tag):
    bad = _mpoll_stale(v, specs, tag)
    while bad:
        spin_pause()
        v = _mpoll_load(specs)
        bad = _mpoll_stale(v, specs, tag)
    return v


def mpoll_finish(specs, first, tag):
    """Wait until every valid spec of ``first`` carries ``tag``; each spec's value words."""
    v = _mpoll_spin(first, specs, tag)
    outs, e = [], 0
    for _, _, n, _ in specs:
        outs.append([v[e + 2 * q] for q in range(n)])
        e += 2 * n
    return outs
