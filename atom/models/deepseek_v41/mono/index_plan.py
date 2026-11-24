# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Every per-step decision of which mailbox cells the indexer writes and reads,
defined once (DESIGN_v3 4.9, I3).

Each function takes a token's ``bound`` -- how many columns its scorer may read:
``step.visible[ratio]``, or the candidate table's context on a REINDEX layer --
and is called both by the kernel (on traced Int32) and by the CPU property test
(on Python ints), so the test checks what the kernel runs. Only non-negative
operands reach ``//`` and ``%``, where Python's floor and the device's
truncation agree; ``i32`` asserts the device's range on the CPU side.
"""

import numpy as np

from atom.mono.plan.arith import sel

CHUNK = 128  # columns a score task: 16 tiles of 8 rows
SEG = 8192  # values a select task ranks: its LDS-resident keys (K2a's LDS budget)
CHUNKS_PER_SEG = SEG // CHUNK
TOPK = 512
TOPK_BLOCKS = 2048
BLOCK_ROWS = 8  # a candidate block, and an index tile (an FP4 plane's page)
STEP = 64  # keys an FP4 walk step: a wave's 4 MFMA tiles, 8 pages
WALK_ROWS = 6  # query rows an FP4 walk scores off each key load
HEADS = 32  # the indexer's heads (``index_n_heads``)
DIM = 128  # an index head's dims (``index_head_dim``)
# the two selections: a token's top rows, and on a candidate-producing layer its
# top blocks -- each a tree over its own bound (``kind_bound``)
KINDS = {"sel": TOPK, "blk": TOPK_BLOCKS}


def imin(a, b):
    return sel(a < b, a, b)


def imax(a, b):
    return sel(a > b, a, b)


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


def chunks(bound):
    """Chunks of a token's columns: chunk ch covers [CHUNK ch, CHUNK (ch + 1))."""
    return cdiv(imax(bound, 0), CHUNK)


def scored(bound):
    """Does the score stage run for a token? Not while its selection is every
    column it sees (bound <= TOPK: the top-k takes them all, in column order,
    and so does the candidate-block top-k, its blocks <= TOPK_BLOCKS)."""
    return bound > TOPK


def score_chunks(bound):
    """The chunks the score stage scores for a token."""
    return sel(scored(bound), chunks(bound), 0)


def chunk_written(bound, ch):
    """Does the score stage write chunk ch's flag (and logits)?"""
    return scored(bound) & (ch * CHUNK < bound)


def segments(bound, seg=SEG):
    """Select segments of a token: segment s covers columns [seg s, seg (s + 1))."""
    return cdiv(imax(bound, 0), seg)


def seg_len(bound, s, seg=SEG):
    """Columns in segment s."""
    return imax(imin(bound - s * seg, seg), 0)


def share_chunks(any_scored, cols, s, seg=SEG):
    """The chunk flags segment s of a row's share polls, ``cols`` its columns
    (``row_shares``): ``any_scored`` the row's, by its whole bound (a split
    row's always)."""
    return sel(any_scored, cdiv(seg_len(cols, s, seg), CHUNK), 0)


def winners(n, k):
    """Valid entries a top-k of n values emits (the rest are -1)."""
    return imin(imax(n, 0), k)


# The selection is a tree. Level 0: segment s of ``seg`` values -> its top-k
# list. Level l > 0: list j merges the level l - 1 lists [FAN j, FAN (j + 1)),
# FAN = seg // k, laid end to end: every list but a level's last holds exactly
# k valid entries (its inputs held >= k), so the merged input is dense and in
# index order. A global winner is in its own list's top-k (fewer than k beat it
# there), so each level keeps every winner. The tree ends at a level of one list.
# ``seg`` is SEG but for a REINDEX row (REINDEX_SEG): its at most REINDEX_CHUNKS
# x CHUNK columns as four lists and one merge, each a quarter of SEG's keys.

# a REINDEX row's segment
REINDEX_SEG = 4096


def fan(k, seg=SEG):
    return seg // k


def lists(bound, level, k, seg=SEG):
    """Lists at ``level`` (static unroll: callers loop over levels in Python)."""
    n = segments(bound, seg)
    for _ in range(level):
        n = cdiv(n, fan(k, seg))
    return n


def list_inputs(bound, level, j, k, seg=SEG):
    """Values list j of ``level`` ranks: a segment's columns at level 0, else the
    valid entries of the lists it merges."""
    if level == 0:
        return seg_len(bound, j, seg)
    below = lists(bound, level - 1, k, seg)
    first = j * fan(k, seg)
    count = imin(below - first, fan(k, seg))
    last = first + count - 1
    return i32(imax(count - 1, 0) * k + list_valid(bound, level - 1, last, k, seg))


def list_valid(bound, level, j, k, seg=SEG):
    """Valid entries of list j at ``level``."""
    return winners(list_inputs(bound, level, j, k, seg), k)


def levels(bound_max, k, seg=SEG):
    """Merge levels a build needs for bounds up to ``bound_max`` (Python ints):
    the depth at which one list remains."""
    level, n = 0, segments(bound_max, seg)
    while n > 1:
        n = cdiv(n, fan(k, seg))
        level += 1
    return level


def per_thread(level, bound_max, k, threads, seg=SEG):
    """Values a thread holds in a ``level`` list of a build for bounds up to
    ``bound_max`` (Python ints): a segment at level 0, else the most the list
    can merge (min(lists below, FAN) x k), over ``threads`` in whole quads. A
    select's cost follows its per-thread capacity, not its inputs."""
    if level == 0:
        most = seg
    else:
        most = min(lists(bound_max, level - 1, k, seg), fan(k, seg)) * k
    return cdiv(cdiv(most, threads), 4) * 4


def is_final(bound, level, k, seg=SEG):
    """Is ``level``'s list the token's answer (one list or none): the selection
    publishes IRDY from it (a split row's share: its answer to every rank's
    merge)."""
    return lists(bound, level, k, seg) <= 1


def final_level(bound, depth, k, seg=SEG):
    """The level whose single list is the token's answer: the first that
    ``is_final`` (a token below the build's maximum ends early)."""
    at = depth
    for level in range(depth, -1, -1):
        at = sel(is_final(bound, level, k, seg), level, at)
    return at


def selected(bound, k=TOPK):
    """Valid ids in the final selection, which the attention reads as its first
    keys (the rest of the ``k`` slots are -1)."""
    return winners(bound, k)


def candidate_offset(t, ident):
    """Candidate address inside token t's row, including FULL and padding.

    A live REINDEX column is in [0, TOPK_BLOCKS * BLOCK_ROWS), so its block
    slot is unchanged. FULL history rows and negative padding may lie outside
    that range: clamp the speculative address, not the selected row. The
    caller discards the candidate value for FULL and padding.
    """
    slot = imin(imax(ident, 0) // BLOCK_ROWS, TOPK_BLOCKS - 1)
    return i32(t * TOPK_BLOCKS + slot)


def blocks(bound):
    """Candidate blocks a candidate-producing layer ranks: the blocks holding
    visible rows, the newest pinned."""
    return cdiv(imax(bound, 0), BLOCK_ROWS)


def compacts(cols):
    """Does a candidate-producing layer's row rank only its candidate blocks'
    rows, compacted (``index_score._compact``)? Once its newest block is block
    ``TOPK_BLOCKS - 1`` or later; ``cols`` its bound or its
    ``candidate_bound``, the same answer. Its selection is the same for finite
    scores: every top-``TOPK`` row lies in a block whose maximum is at least
    that row's score, so among the top-``TOPK_BLOCKS`` blocks (ties: the lower
    column, the lower block, in both trees). With exactly ``TOPK_BLOCKS``
    blocks the compaction moves nothing."""
    return cols > (TOPK_BLOCKS - 1) * BLOCK_ROWS


def candidate_bound(bound):
    """The columns a candidate-producing layer's row ranks: its candidate
    blocks' rows moved to the row's front in block order, the newest block
    (pinned, partly visible: its ``(bound - 1) % BLOCK_ROWS + 1`` rows) last;
    every column while it has at most ``TOPK_BLOCKS`` blocks."""
    newest = (TOPK_BLOCKS - 1) * BLOCK_ROWS + ((bound - 1) & (BLOCK_ROWS - 1)) + 1
    return imin(bound, newest)


def block_chunks(bound, j):
    """(first chunk, chunk count) whose logits hold the rows of block segment j
    (blocks [SEG j, SEG (j + 1))): the flags that segment's task polls."""
    rows = SEG * BLOCK_ROWS
    first = j * (rows // CHUNK)
    count = cdiv(imax(imin(bound, (j + 1) * rows) - j * rows, 0), CHUNK)
    return first, sel(scored(bound), count, 0)


def flat_total(counts):
    """Tasks of a stage whose token t has ``counts[t]``: laid token after token."""
    total = 0
    for n in counts:
        total = total + n
    return total


def flat_task(counts, task):
    """(token, its index) of flat task ``task`` < ``flat_total(counts)``."""
    # task's own type for the accumulators: a select needs a traced operand
    t, start, running = task * 0, task * 0, 0
    for i, n in enumerate(counts[:-1]):
        running = running + n
        past = task >= running
        t = sel(past, i + 1, t)
        start = sel(past, running, start)
    return t, task - start


def score_tasks(bounds, owners, ctas, width_log2_max):
    """(counts, spans, width_log2): flat task (t, i) scores chunks
    [i << width_log2, (i + 1) << width_log2) of the ``spans[t]`` tokens from t.
    Adjacent tokens of one owner (one tile table) share a task, then chunk
    groups widen, each only while the step keeps a task for each of ``ctas``."""
    n = len(bounds)
    own = [score_chunks(b) for b in bounds]
    same = [owners[t] == owners[t + 1] for t in range(n - 1)]
    one = bounds[0] * 0 + 1  # the bounds' type: a select operand
    run, widest = [one] * n, list(own)
    for t in range(n - 2, -1, -1):
        run[t] = sel(same[t], run[t + 1] + 1, one)
        widest[t] = sel(same[t], imax(own[t], widest[t + 1]), own[t])
    shared = widest[:1] + [sel(s, 0, w) for s, w in zip(same, widest[1:])]
    together = flat_total(shared) >= ctas
    need = [sel(together, s, o) for s, o in zip(shared, own)]
    spans = [sel(together, r, 1) for r in run]
    total, width_log2 = flat_total(need), one - 1
    for g in range(1, width_log2_max + 1):
        width_log2 = sel((total >> g) >= ctas, g, width_log2)
    counts = [(c + (one << width_log2) - 1) >> width_log2 for c in need]
    return counts, spans, width_log2


def walk_pieces(owners):
    """A step's rows as pieces, ``(lead, span)``: at most WALK_ROWS adjacent
    tokens of one owner (one page table), scored off each key load. ``owners``
    is the step's rows' requests (int numpy); every ratio's plan shares them."""
    n = owners.shape[0]
    t = np.arange(n)
    run_start = np.ones(n, dtype=bool)
    run_start[1:] = owners[1:] != owners[:-1]
    run_first = np.maximum.accumulate(np.where(run_start, t, 0))
    lead = np.flatnonzero(run_start | ((t - run_first) % WALK_ROWS == 0))
    return lead, np.diff(lead, append=n)


# A FULL layer's request whose rows' largest bound m is past SPLIT_FROM columns
# is scored and ranked in shares over
# the TP ranks (``splits``): rank r its columns [r Q, r Q + Q), Q =
# ``share_len(m)``. Each rank's top-k goes to every rank as (value, column)
# pairs, and every rank's merge (``index_score.stage_imerge``) ranks the tp x k
# of them: a winner is in its own rank's top-k (fewer than k beat it there,
# ties to the lower column in both, and rank order is column order), so the
# same ids in the same order as one tree over every column. A shorter request
# is ranked whole on every rank, with no hand-off.
SPLIT_FROM = 32768
SPLIT_UNIT = CHUNK
# a step of this many index rows splits a request even where a share's tree
# then takes a level more (``split_gains``)
SPLIT_DEEP_ROWS = 16


def split_tp(tp):
    """Can a selection split over ``tp`` ranks? The merge's tp x TOPK keys
    must be whole quads a thread (tp 4 or 8)."""
    return tp > 1 and tp % 4 == 0


def splits(tp, ratio):
    """Do a FULL layer's long requests split over ``tp`` ranks? The candidate
    producer (ratio 1) ranks blocks over every column, so it stays whole."""
    return ratio > 1 and split_tp(tp)


def share_len(m, tp):
    """A split request's columns a rank, its rows' largest bound ``m``."""
    return cdiv(m, tp * SPLIT_UNIT) * SPLIT_UNIT


def tree_depth(n):
    """Merge levels of a selection tree over ``n`` columns (int numpy): 0 for
    one segment."""
    m = (np.asarray(n) + SEG - 1) // SEG
    depth = np.zeros_like(m)
    while (m > 1).any():
        deeper = m > 1
        depth = depth + deeper
        m = np.where(deeper, (m + fan(TOPK) - 1) // fan(TOPK), m)
    return depth


def split_gains(m, tp, rows):
    """Does splitting a request whose rows' largest bound is ``m`` pay, in a
    step of ``rows`` index rows? Where it keeps its selection's levels (a
    split share's tree, then the merge, against the whole row's tree); where
    it adds one (a hand-off and a select), only from ``SPLIT_DEEP_ROWS`` rows:
    every rank scoring every row's columns then costs more than the level."""
    keeps = tree_depth(share_len(m, tp)) + 1 <= tree_depth(m)
    return keeps | (rows >= SPLIT_DEEP_ROWS)


def row_shares(bounds, owners, tp, ratio, rank):
    """Each row's (first column, columns, split) on ``rank``, on the host --
    the reference ``step_meta._walk_plan`` writes them by on the device: a
    split request's share (``share_len`` from rank x it, within the row's
    bound), a whole one's every column. ``owners``: the rows' requests, a
    request's rows adjacent (int numpy)."""
    bounds = np.asarray(bounds, dtype=np.int64)
    owners = np.asarray(owners)
    top = np.zeros_like(bounds)
    for owner in np.unique(owners):
        mine = owners == owner
        top[mine] = bounds[mine].max()
    split = splits(tp, ratio) & (top > SPLIT_FROM) & split_gains(top, tp, len(bounds))
    q = share_len(top, tp)
    first = np.where(split, rank * q, 0)
    n = np.where(split, np.clip(bounds - first, 0, q), bounds)
    return first, n, split


def fill_walk_plan(bounds, pieces, plan, shares=None):
    """The FP4 plane's FULL-layer score tasks of a step, on the host -- the
    reference ``step_meta._walk_plan`` lays them out by on the device: ``plan``
    [ctas, 4] int32 gets a (lead, span, ch0, ch1) a CTA -- chunks [ch0, ch1)
    of the ``span`` tokens from ``lead``, one of ``pieces`` (`walk_pieces`) --
    and zeros past the last task. ``bounds`` is the step's rows (int numpy);
    ``shares`` their (first column, columns) on this rank (``row_shares``;
    None: every column). Every piece gets ceil(need / width) tasks sharing its
    chunks evenly, at the smallest width that fits them all in ``ctas``: a
    width one chunk too wide left a wave per CTA a step behind the rest."""
    ctas = plan.shape[0]
    plan.fill(0)
    lead, span = pieces
    first, cols = (np.zeros_like(bounds), bounds) if shares is None else shares
    # A piece's chunks are its widest row's: `scored` and `cdiv` are monotone;
    # whether it is scored is its rows' own (a share may be short or empty).
    top = np.maximum.reduceat(bounds, lead)
    need = np.where(scored(top), cdiv(np.maximum.reduceat(cols, lead), CHUNK), 0)
    # (lead, span, need, tasks, first chunk) a piece; tasks is need at width 1
    record = np.array((lead, span, need, need, first[lead] // CHUNK))
    if not need.all():
        record = record[:, need > 0]
    need, k = record[2], record[3]
    if not need.size:
        return
    assert need.size < ctas, (need.size, ctas)
    total = int(need.sum())
    if total > ctas:
        # the width search's range: ceil(total / ctas), the least that could
        # fit, to ceil(total / (ctas - pieces)), which does (a piece's ceil adds
        # under one task); feasibility is monotone, so the first width that fits
        lo = cdiv(total, ctas)
        widths = np.arange(lo, max(cdiv(total, ctas - need.size), lo) + 1)[:, None]
        width = lo + int(np.argmax(cdiv(need, widths).sum(1) <= ctas))
        k[:] = cdiv(need, width)
    piece = np.repeat(np.arange(need.size), k)
    i = np.arange(piece.size) - np.repeat(np.cumsum(k) - k, k)
    lead, span, need, k, ch = record[:, piece]
    tasks = plan[: piece.size]
    tasks[:, 0] = lead
    tasks[:, 1] = span
    tasks[:, 2] = ch + i * need // k
    tasks[:, 3] = ch + (i + 1) * need // k


SPLIT_BIT = 30  # a share word's split flag, above any column count


def share_word(cols, split):
    """A row's columns on this rank and split flag in one word (the plan's
    share row, word 1): what a selection tree reads of every row."""
    return cols | (split << SPLIT_BIT)


def word_cols(word):
    return word & ((1 << SPLIT_BIT) - 1)


def word_split(word):
    return word >> SPLIT_BIT


def fill_shares(shares, rows):
    """``rows`` [S, 4] int32 (the plan's rows past its tasks) := each row's
    (first column, ``share_word``, 0, 0) of ``row_shares``."""
    first, cols, split = shares
    rows.fill(0)
    rows[:, 0] = first
    rows[:, 1] = share_word(cols.astype(np.int64), split.astype(np.int64))


# a REINDEX token's chunks at most: its candidate blocks' rows
REINDEX_CHUNKS = TOPK_BLOCKS * BLOCK_ROWS // CHUNK


def reindex_split(tokens, ctas, least):
    """FP4 plane, a REINDEX layer: tasks a token (each its own piece on its own
    candidate table), REINDEX_CHUNKS shared over them -- fixed by the step's
    width, so a CTA finds its task from its index alone. At least ``least``
    chunks a task."""
    return max(1, min(ctas // tokens, REINDEX_CHUNKS // least))


def reindex_chunks(i, per):
    """Chunks [ch0, ch1) of a REINDEX token's task i of ``per``."""
    return i * REINDEX_CHUNKS // per, (i + 1) * REINDEX_CHUNKS // per


def task_writes(bound, ch0, width, m):
    """Does a score task over chunks [ch0, ch0 + width) write its chunk ch0 + m
    for a token whose scorer reads ``bound`` columns?"""
    return (m < width) & chunk_written(bound, ch0 + m)


def list_count(bound, level, k, seg=SEG):
    """Lists a token runs at ``level``: level 0 always one (so a token that sees
    nothing still emits its -1s), a higher level only while the one below had
    more than one."""
    if level == 0:
        return imax(lists(bound, 0, k, seg), 1)
    return sel(lists(bound, level - 1, k, seg) > 1, lists(bound, level, k, seg), 0)


def kind_bound(kind, bound):
    """The values a ``kind`` selection ranks for a token whose scorer reads
    ``bound`` columns: the columns, or the blocks holding them."""
    return bound if kind == "sel" else blocks(bound)


def top_level(kind, bound_max, seg=SEG):
    """A ``kind`` selection tree's last level for bounds up to ``bound_max``."""
    return levels(kind_bound(kind, bound_max), KINDS[kind], seg)


def list_slots(kind, level, bound_max, seg=SEG):
    """Lists a token can have at ``level`` of one tree."""
    return lists(kind_bound(kind, bound_max), level, KINDS[kind], seg)


def reindex_bound(bound_max):
    """A REINDEX row's columns at most, in a build for bounds up to
    ``bound_max``: its candidate blocks' rows."""
    return min(bound_max, REINDEX_CHUNKS * CHUNK)


def trees(kind, bound_max):
    """The (bound, seg) of each ``kind`` tree a build for bounds up to
    ``bound_max`` runs: the columns', and for the selection a REINDEX row's."""
    if kind == "blk":
        return ((bound_max, SEG),)
    return ((bound_max, SEG), (reindex_bound(bound_max), REINDEX_SEG))


def region_levels(kind, bound_max):
    """Levels with a ``kind`` region: those below some tree's top level."""
    return max(top_level(kind, b, s) for b, s in trees(kind, bound_max))


def region_slots(kind, level, bound_max):
    """Lists a token's ``level`` region holds: the most of the trees that put
    lists at that level (a tree's top level puts none)."""
    return max(
        list_slots(kind, level, b, s)
        for b, s in trees(kind, bound_max)
        if level < top_level(kind, b, s)
    )
