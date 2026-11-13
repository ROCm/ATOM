# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A served step's per-ratio visible counts and compress / write plans, on the device from the step's positions.

The host builds both from its own positions (``metadata`` ``visible``: ``(position + 1) // ratio``;
``compress_plan.make_compress_plans``). Derived here from the positions the step's graph reads, they hold for a step
whose positions the device advanced from the previous verdict (``device_round``) as well. Same rows as the host's:
a request's ``rows`` tokens in order, each ``[ragged row, request, position, window_len]`` (``window_len =
max(K - 1 - j, 0)``, ``K = ratio``: CSA2 never overlaps); the compress plan keeps the rows at a compression boundary
(``(position + 1) % ratio == 0``), the write plan the rows within ``K + extra_write`` of the request's end; the
rest -1."""

import triton
import triton.language as tl


@triton.jit
def _plan_rows(
    pos,
    i,
    j,
    b,
    end,
    live,
    cplan,
    wplan,
    R: tl.constexpr,
    EXTRA: tl.constexpr,
    CAP: tl.constexpr,
):
    """One ratio's compress and write plan rows (each plan already -1)."""
    win = tl.maximum(R - 1 - j, 0)
    cm = live & ((pos + 1) % R == 0)
    wm = live & (pos + R + EXTRA >= end)
    cidx = tl.cumsum(cm.to(tl.int32), 0) - 1
    widx = tl.cumsum(wm.to(tl.int32), 0) - 1
    cm = cm & (cidx < CAP)
    wm = wm & (widx < CAP)
    tl.store(cplan + cidx * 4, i, mask=cm)
    tl.store(cplan + cidx * 4 + 1, b, mask=cm)
    tl.store(cplan + cidx * 4 + 2, pos, mask=cm)
    tl.store(cplan + cidx * 4 + 3, win, mask=cm)
    tl.store(wplan + widx * 4, i, mask=wm)
    tl.store(wplan + widx * 4 + 1, b, mask=wm)
    tl.store(wplan + widx * 4 + 2, pos, mask=wm)
    tl.store(wplan + widx * 4 + 3, win, mask=wm)


@triton.jit
def _plans_kernel(
    positions,
    vis_a,
    vis_b,
    cplan_a,
    wplan_a,
    cplan_b,
    wplan_b,
    n,
    ROWS: tl.constexpr,
    RA: tl.constexpr,
    RB: tl.constexpr,
    EXTRA: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    live = i < n
    pos = tl.load(positions + i, mask=live, other=0).to(tl.int32)
    j = i % ROWS
    b = i // ROWS
    end = pos - j + ROWS  # the request's end: its rows are consecutive positions
    tl.store(vis_a + i, (pos + 1) // RA, mask=live)
    tl.store(vis_b + i, (pos + 1) // RB, mask=live)
    # every plan row -1 first (the kernels skip a negative position), then the live rows over them
    f = tl.arange(0, 4 * BLOCK)
    fm = f < 4 * CAP
    neg = tl.full([4 * BLOCK], -1, tl.int32)
    tl.store(cplan_a + f, neg, mask=fm)
    tl.store(wplan_a + f, neg, mask=fm)
    tl.store(cplan_b + f, neg, mask=fm)
    tl.store(wplan_b + f, neg, mask=fm)
    tl.debug_barrier()
    _plan_rows(pos, i, j, b, end, live, cplan_a, wplan_a, RA, EXTRA, CAP)
    _plan_rows(pos, i, j, b, end, live, cplan_b, wplan_b, RB, EXTRA, CAP)


def fill_step_plans(
    positions, rows_per_request, ratios, extra_write, vis, cplans, wplans, cap
):
    """``positions`` [n] (int32 / int64, the step's: ``n`` a multiple of ``rows_per_request``); ``ratios`` the two
    compress ratios; ``vis`` / ``cplans`` / ``wplans`` per ratio: [>= n] int32 and [cap x 4] int32 (flat), written on
    the current stream."""
    n = positions.numel()
    ra, rb = ratios
    _plans_kernel[(1,)](
        positions,
        vis[ra],
        vis[rb],
        cplans[ra],
        wplans[ra],
        cplans[rb],
        wplans[rb],
        n,
        ROWS=rows_per_request,
        RA=ra,
        RB=rb,
        EXTRA=extra_write,
        CAP=cap,
        BLOCK=triton.next_power_of_2(max(cap, 2)),
    )
