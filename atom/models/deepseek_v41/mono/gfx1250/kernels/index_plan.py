# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The indexer's per-step arithmetic, defined once: the selection tree (lists a level, a list's inputs, task counts),
the REINDEX layer's compacted context and key rows, and the scorer's task split.

The kernel calls these on traced Int32 (``fx`` / ``I32`` below are FlyDSL's); the CPU test rebinds ``fx`` / ``I32`` to a
plain-integer backend and runs the same code on Python ints, so it checks what the kernel computes. Only non-negative
operands reach ``//`` (Python's floor and the device's truncation agree there)."""

import flydsl.expr as fx

I32 = fx.Int32

SEG = 8192  # values a level-0 list's input (one CTA's LDS keys)
KSEL, KBLK = 512, 2048  # the selection's k, the candidate blocks' k (layer 20)
NCAND = KBLK  # REINDEX candidate blocks a row
NLMAX = 3  # the deepest tree (IDX_LEVELS <= 3: n <= 1M)
CHUNK = 128  # scorer columns a chunk
SUB = 4  # chunks a scorer task at most
TOPK = 512  # rows a selection keeps (rows with a context <= TOPK are not scored)
LANES = 32  # a wave's lanes: the task prefix holds row 32 g + lane in lane `lane` of group g


def cdiv(a, b):
    return (a + (b - 1)) // b


def tree_levels(n, k):
    """Keep the original two passes; use a third only when two cannot merge all lists."""
    return (n > SEG * (SEG // k)).select(I32(3), I32(2))


def lists(n, lev, fan):
    """Lists at level lev (Python int) of a token with n inputs: segments, then / fan a level."""
    x = cdiv(fx.max(n, I32(0)), I32(SEG))
    for _ in range(lev):
        x = cdiv(x, I32(fan))
    return x


def lists_rt(n, lev, fan):
    """lists with a traced level (<= NLMAX) and fan."""
    x = cdiv(fx.max(n, I32(0)), I32(SEG))
    for i in range(NLMAX):
        x = (I32(i) < lev).select(cdiv(x, fan), x)
    return x


def list_inputs_rt(n, lev, j, k, fan):
    """Inputs of list j at level lev: down the chain of last children to level 0, then back up."""
    jc = j
    cnts = [None] * (NLMAX + 1)
    for l in range(NLMAX, 0, -1):
        on = I32(l) <= lev
        cnt = fx.min(lists(n, l - 1, fan) - jc * fan, fan)
        cnts[l] = cnt
        jc = on.select(jc * fan + cnt - 1, jc)
    v = fx.min(fx.max(n - jc * SEG, I32(0)), I32(SEG))
    for l in range(1, NLMAX + 1):
        v = (I32(l) <= lev).select((cnts[l] - 1) * k + fx.min(v, k), v)
    return v


def list_count_rt(n, lev, fan):
    """Tasks (lists) of a token at level lev: level 0 at least one; a higher level only while the one below has > 1."""
    return (lev == 0).select(
        fx.max(lists(n, 0, fan), I32(1)),
        (lists_rt(n, fx.max(lev - 1, I32(0)), fan) > 1).select(
            lists_rt(n, lev, fan), I32(0)
        ),
    )


def rx_kept(bd):
    """A REINDEX token's kept candidate blocks: min(blocks(visible), NCAND)."""
    return fx.min(cdiv(fx.max(bd, I32(0)), I32(8)), I32(NCAND))


def rx_context(kept, last, bd):
    """Its compacted columns (candidate_table.py): (kept - 1) 8 + clamp(visible - 8 last, 0, 8), last = its last kept
    block; 0 without candidates."""
    ctx = (kept - 1) * 8 + fx.min(fx.max(bd - last * 8, I32(0)), I32(8))
    return (kept > 0).select(ctx, I32(0))


def key_row(krow, bd):
    """The index row a scorer key reads: its real row clamped to the token's bound -- not to the compacted column count
    (past 16384 index rows a REINDEX token's real rows exceed it)."""
    return fx.min(krow, fx.max(bd - 1, I32(0)))


def score_sub(TR, ctas):
    """Chunks a scorer task (Python int): ~16 a row at a ~2k context; more a task only when TR x 16 exceeds one round of
    CTAs (c8: 3 rounds -> 1); c1 / c2 keep one (a longer task only adds latency there).
    """
    return next((s_ for s_ in range(1, SUB + 1) if TR * -(-16 // s_) <= ctas), SUB)


def score_tasks(cx, tcols):
    """Scorer tasks of a token with cx columns: none while the selection takes every column (cx <= TOPK)."""
    return (cx > TOPK).select((cx + (tcols - 1)) // tcols, I32(0))


def row_groups(TR):
    """Lane groups of the rows' task prefix (Python int): 2 up to 64 rows (the narrow builds' code), 3 at 96."""
    return max(2, cdiv(TR, LANES))


def task_prefix(cnt, TR, ln, scan, last):
    """The rows' inclusive task prefix in every wave's lanes (no serial LDS walk): incs[g] in lane ln = the tasks of the
    rows <= 32 g + ln (rows past TR: 0). cnt(row): a row's task count (row < TR); scan(v): the wave's inclusive scan;
    last(v): lane 31's value. Returns (incs, the step's tasks)."""
    incs, carry = [], I32(0)
    for g in range(row_groups(TR)):
        r = ln + LANES * g
        cg = (r < TR).select(cnt(fx.min(r, I32(TR - 1))), I32(0))
        inc = scan(cg) + carry
        carry = last(inc)
        incs.append(inc)
    return incs, carry


def task_of(incs, i, TR, count_true, wave_max):
    """Task i's (row, index in the row): the rows whose inclusive prefix is <= i come before it (count_true(p): the
    wave's lanes where p holds; wave_max: the wave's max)."""
    r, before = I32(0), I32(0)
    for inc in incs:
        p = inc <= i
        r = r + count_true(p)
        before = fx.max(before, wave_max(p.select(inc, I32(0))))
    return fx.min(r, I32(TR - 1)), i - before
