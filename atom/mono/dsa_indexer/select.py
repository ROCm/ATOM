# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The indexer's rows: a long row's scores and its share of the global top-K
under DCP, and every row's KV slots at its CSR offset (``kernel`` places
the stages).

    score   (a long row's block of local keys, wave w the keys 16 w ..): the cache
            block's preshuffled rows as the MFMA's B operand (two 16 B loads a
            lane), the row's FP8 q (IQ8) as A, 4 chained 16x16x32 FP8 MFMAs a
            16-head half, relu(dot k_scale) w_h summed in gluon's head order
            -> ILOG (device scope), SFLAG[row, block]
    select  (a long row, one CTA; every rank the same rows): the exact global
            top-K by (score desc, rank asc, local index asc) without moving a
            candidate -- 4 radix passes of 8 bits over the merge's ordinal,
            each a 256-bin histogram of this rank's keys under the prefix,
            pushed to every rank (HIST) and summed in rank order, so every rank
            picks the same digit. The last pass's per-rank counts of the
            threshold bin are the lower ranks' share of the ties. Then this
            rank's keys above the threshold and its share of the equal ones,
            in local order
    row     (a row, after its count): the count to CNT, the earlier rows'
            counts polled, the slots written at the sum -- the MLA's packed
            indices, its indptr and the owned counts

A short row (``plan.short``) skips score and select: every key the
rank holds, in local order (the original merge's answer whenever ctx <= K).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.ops import (
    CM_DEV,
    CM_SYS,
    block_excl_scan,
    butterfly,
    head_sum,
    ld_dev,
    mfma_fp8,
    rsrc,
    traced,
    wave_incl_scan,
    wave_sum,
)
from atom.mono.device.queue import word
from atom.mono.device.select import INT_MIN, f32_key, u32_gt
from atom.mono.device.sync import publish, sleep_ticks
from atom.mono.dsa_indexer.plan import (
    SCORE_KEYS,
    TILE,
    cache_byte,
    key_pos,
    local_len,
    scale_byte,
    short,
)
from atom.mono.plan.arith import pick
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES

# radix passes over the 32-bit key, most significant first; a pass is a round
# trip to every rank whose cost grows with the bins pushed and polled
DIGIT_BITS = (8, 8, 8, 8)
PASSES = len(DIGIT_BITS)
SHIFTS = tuple(32 - sum(DIGIT_BITS[: p + 1]) for p in range(PASSES))
BINS = 1 << max(DIGIT_BITS)
# thread t holds bins t + k THREADS; fewer bins than threads leave the last
# threads none (their polls repeat the last bin: nothing assigned under a
# traced branch)
BINS_A_THREAD = max(1, BINS // THREADS)


def _bin(tid, k):
    return fx.min(tid + k * THREADS, BINS - 1)


# the digit's choice (``_decide``): a lane's bins, over the first DEC_WAVES
DEC_WAVES = min(WAVES, BINS // 64)
DEC_A_LANE = BINS // (64 * DEC_WAVES)


# a selection chunk: the keys a CTA counts at once, 8 a thread in registers;
# its above / equal counts share one 32-bit scan, 16 bits each
CHUNK = 4096
CHUNK_LG = CHUNK.bit_length() - 1
PT = CHUNK // THREADS
# a segment's chunks at most, log2: 64 rows of 2^22 local keys fit
CPS_LG_MAX = 8
# the waves holding a step's segments, a thread each (``_sum_before``)
BEFORE_WAVES = BLOCKS // 64
# a row histogram word (``_row_add``): its count in the low COUNT_BITS (a
# row's keys at most, ``kernel.unfit``), the segments added in above
COUNT_BITS = 22
ARRIVED = 1 << COUNT_BITS
# a row histogram's copies: segment g adds into copy g % ROW_COPIES (a word
# takes fewer atomics), its readers sum them
ROW_COPIES = 4
# LDS ``rows`` words a row (``step_table``): context, keys here, long | split
# << 1
ROW_WORDS = 3
# ``own_rows``: a request's keys split over the ranks only past this many
# (``split_keys``)
SPLIT_MIN = 262144
# a piece's rows at most (``piece_rows``)
RG_MAX = 8
TASK_UNITS_LG = 3
TASK_UNITS = 1 << TASK_UNITS_LG  # a score task's units at most


def peer_pairs(tokens: int, world: int, replicated: bool) -> dict:
    return {
        "hist": tokens * PASSES * world * BINS,
        # replicated: (row, rank)'s winners in this rank's slab, their count
        "rcnt": tokens * world if replicated else 0,
        # this rank's own, plain words (two a pair): a row's histogram of
        # each pass, its segments' added (``_row_add``). It must be zero at
        # every launch's start: the indexer's own peer region, zeroed a step
        # and then by each row's last segment (``_row_clear``). Not the step
        # scratch: Kimi-K3's K1 / K2 leave their pairs there between two DSA
        # layers (counts on them: a wrong top-k or a fault)
        "rowhist": tokens * PASSES * ROW_COPIES * BINS // 2,
    }


def hist_at(r, p, src, b, world):
    return ((r * PASSES + p) * world + src) * BINS + b


def radix_key(v):
    """The DCP merge's ordinal of an f32 (``dcp_topk_merge._f32_to_ord``:
    larger float, larger ordinal; +0 above -0; NaN the very bottom) with its
    sign bit flipped: as unsigned bits, larger is better and the radix digits
    are plain bytes. Held in an Int32 -- FlyDSL's integer compares and shifts
    are signed -- so digits are shifted then masked (``digit``), prefixes
    masked (``under``) and orders compared through ``device.select.u32_gt``."""
    o = fx.Int32(fx.isnan(v).select(fx.Int32(INT_MIN), f32_key(v)))
    return o ^ fx.Int32(INT_MIN)


def digit(key, p):
    """Pass p's digit of a key."""
    return (key >> SHIFTS[p]) & ((1 << DIGIT_BITS[p]) - 1)


def under(key, p, prefix):
    """Whether the key's bits above pass p's digit are ``prefix``."""
    used = 32 - SHIFTS[p] - DIGIT_BITS[p]
    return ((key >> (32 - used)) & ((1 << used) - 1)) == prefix


def row_kind(c, r):
    """``_row_kind`` of row r, from the CTA's table (``step_table``)."""
    at = c["rows"] + ROW_WORDS * r
    flags = fx.ptr_load(at + 2)
    split = fx.Boolean(True)
    if const_expr(c["own_rows"]):
        split = (flags & 2) != 0
    return fx.ptr_load(at), fx.ptr_load(at + 1), (flags & 1) != 0, split


def ilog_at(c, r):
    """Row r's first ILOG word (``IxBuild.row_keys``)."""
    return r * c["row_keys"]


def sflag_at(c, r):
    """Row r's first SFLAG pair, a ``SCORE_KEYS`` unit each
    (``IxBuild.sflag_row``)."""
    return r * c["sflag_row"]


@traced
def step_table(c):
    """The step's rows and tasks once a CTA, on wave 0, a lane a row
    (``kernel.MAX_TOKENS`` is a wave), behind one barrier:

    - every row's ``_row_kind`` into LDS ``rows`` (context, keys here,
      long | split << 1): the step's later reads an LDS word each;
    - the score and segment tasks into LDS ``plan`` from the same lanes: the
      totals a sum over the rows' lanes, the firsts a scan.

    Score tasks, pieces in order (a piece's units on its last row's lane): a
    task is a run of up to ``TASK_UNITS`` of a piece's units, its loads out
    together, its flags one drain; the narrowest run whose tasks fit one CTA
    round, else the widest.

    Segment tasks, rows in order: a row's segments wait on each other every
    pass, so all of them are resident at once: a segment is one chunk while
    the long rows' chunks fit the CTAs one each, else the fewest chunks, a
    power of two (a shift, not a division by a runtime value), that make them
    fit (sum_r ceil(a_r / cps) <= sum a_r / cps + rows <= BLOCKS). The step's
    rows decide it, not max_model_len -- V4.1's tree the same way."""
    tid = c["tid"]
    s, ql, rg, ppr = c["S"], c["qlen"], c["rg"], c["ppr"]
    if tid < 64:
        r = fx.min(tid, fx.Int32(s - 1))
        live = tid < s
        rctx = request_ctx(c, r)
        total = None
        if const_expr(c["own_rows"]):
            # the step's requests' contexts: each counted on its first row's lane
            first = live & (r % ql == 0)
            total = _rows_sum(first.select(fx.max(rctx, fx.Int32(0)), fx.Int32(0)), s)
        ctx, n, long, split = _row_kind(c, r, rctx, total)
        if live:
            flags = long.select(fx.Int32(1), fx.Int32(0))
            if const_expr(c["own_rows"]):
                flags = flags | split.select(fx.Int32(2), fx.Int32(0))
            at = c["rows"] + ROW_WORDS * tid
            fx.ptr_store(ctx, at)
            fx.ptr_store(n, at + 1)
            fx.ptr_store(flags, at + 2)
        long = long & live
        # the row's piece, and its units on the piece's last row
        # (``score_units``)
        pc = (r // ql) * ppr + r % ql // rg
        lead, span = piece(c, pc)
        last = live & (r == lead + span - 1)
        nu = row_units(n, last & long)
        lg = fx.Int32(TASK_UNITS_LG)
        for w in reversed(range(TASK_UNITS_LG)):
            tasks = _rows_sum((nu + (1 << w) - 1) >> w, s)
            lg = (tasks <= BLOCKS).select(fx.Int32(w), lg)
        cnt = (nu + (fx.Int32(1) << lg) - 1) >> lg
        _lay_out(
            c,
            _score_firsts(c) + pc,
            last,
            cnt,
            s,
            (PLAN_SCORE_TOTAL, PLAN_SCORE_LG, lg),
        )
        # a long row's chunks (0 for any other)
        chunks = long.select((n + CHUNK - 1) // CHUNK, fx.Int32(0))
        total = _rows_sum(chunks, s)
        room = BLOCKS - _rows_sum(long.select(fx.Int32(1), fx.Int32(0)), s)
        slg = fx.Int32(0)
        for w in range(1, CPS_LG_MAX + 1):
            slg = (total > (room << (w - 1))).select(fx.Int32(w), slg)
        slg = (total <= BLOCKS).select(fx.Int32(0), slg)
        segs = (chunks + (fx.Int32(1) << slg) - 1) >> slg
        _lay_out(
            c, _seg_firsts(c) + r, live, segs, s, (PLAN_SEG_TOTAL, PLAN_SEG_LG, slg)
        )
    gpu.barrier()


def _row_kind(c, r, rctx, total):
    """(context, keys held here, long?, keys split over the ranks?) of row r
    of a request of context ``rctx`` (``request_ctx``), the step's requests'
    contexts ``total`` (``own_rows``: ``split_keys``).
    The keys held here follow from the context (``plan.local_len``), never
    from a second published count: a padded graph row's context (0, a
    verify's draft past it below) holds none and is not long, and no row can
    be long over fewer keys than its top-K (a capture's warmup published
    counts its contexts disagree with). Split: every row without
    ``own_rows`` (a constant: its branches fold away), else ``split_keys``
    of its request."""
    ql = c["qlen"]
    # row r is its request's draft r % ql, the qlen - 1 - that drafts past it
    # not yet in it
    ctx = rctx if const_expr(ql == 1) else rctx - (ql - 1 - r % ql)
    cz = fx.max(ctx, fx.Int32(0))
    n = local_len(cz, c["rank"], c["world"], c["inter"])
    split = fx.Boolean(True)
    if const_expr(c["own_rows"]):
        split = split_keys(c, rctx, total)
        n = split.select(n, (owner(c, r) == c["rank"]).select(cz, fx.Int32(0)))
    return ctx, n, ~short(c["shape"], ctx) & (n > 0), split


def request_ctx(c, r):
    """Row r's request's context (a padded token's: none, 0)."""
    a = c["args"]
    if const_expr(c["qlen"] == 1):
        return fx.Int32(bo.buffer_load(rsrc(a["ctx"]), r, vec_width=1, dtype=T.i32))
    bid = _batch_id(c, r)
    req = fx.Int32(
        bo.buffer_load(
            rsrc(a["ctx"]), fx.max(bid, fx.Int32(0)), vec_width=1, dtype=T.i32
        )
    )
    return (bid >= 0).select(req, fx.Int32(0))


def split_keys(c, rctx, total):
    """``own_rows``: whether a request of context ``rctx`` has its keys split
    over the ranks instead of owned: past SPLIT_MIN keys and more than half
    a rank's share of the step's ``total`` -- its owner would score it
    alone while the others idle. Every rank reads the same contexts, so every
    rank decides alike."""
    return (rctx > SPLIT_MIN) & (2 * c["world"] * rctx > total)


def owner(c, r):
    """``own_rows``: the rank that scores and selects row r -- its request's,
    the requests dealt over the ranks in turn (a request's rows share their
    key loads, ``piece``)."""
    return (r // c["qlen"]) % c["world"]


def stored_at(c, j, split):
    """This rank's local key j's position in its cache: j itself under DCP
    and for a row this rank owns, its global position for a replicated row
    whose keys are ``split`` (``plan.key_pos``)."""
    if const_expr(not c["replicated"]):
        return j
    return split.select(key_pos(j, c["rank"], c["world"], c["inter"]), j)


def unit_runs(c, split):
    """A score unit's stored run: local unit u of a row is the cache's run
    u * mul + add of SCORE_KEYS keys (``plan.key_pos`` with its interleave a
    unit, ``REPLICA_INTERLEAVE``) -- (W, rank) for a replicated row whose
    keys are ``split``, else (1, 0): two selects a task, not one a unit's
    first key (``stored_at``)."""
    if const_expr(not c["replicated"]):
        return 1, 0
    return split.select(fx.Int32(c["world"]), fx.Int32(1)), split.select(
        fx.Int32(c["rank"]), fx.Int32(0)
    )


def key_block(c, req, at, last=None):
    """The cache position ``at`` (``stored_at``) of a row of request ``req``
    (``request_of``): (its block, its row in the block), the block clamped to
    the one holding position ``last`` if given (a key past the row's stays in
    bounds; never stored)."""
    a, rows = c["args"], c["shape"].block_rows
    blk = at // rows if last is None else fx.min(at // rows, last // rows)
    e = fx.Int32(
        bo.buffer_load(
            rsrc(a["bt"]),
            req * a["bt_ld"] + blk,
            vec_width=1,
            dtype=T.i32,
        )
    )
    return e, at % rows


def _batch_id(c, r):
    """A verify's query token r's ``batch_id_per_q_token`` (-1: padded)."""
    return fx.Int32(
        bo.buffer_load(rsrc(c["args"]["bids"]), r, vec_width=1, dtype=T.i32)
    )


def request_of(c, r):
    """Query token r's request, its block table row: its batch id on a verify
    (a padded token's -1 reads request 0's row: its context holds no key,
    nothing it reads is stored), token r itself on a plain decode."""
    if const_expr(c["qlen"] == 1):
        return r
    return fx.max(_batch_id(c, r), fx.Int32(0))


def piece_rows(qlen: int) -> int:
    """A piece's rows at most, the rows scored off each key load: V4.1's FP4
    walk pieces (``walk_pieces``), at most ``RG_MAX`` adjacent rows of one
    request -- one block table -- a request cut every that many rows."""
    return min(qlen, RG_MAX)


def pieces_a_request(qlen: int) -> int:
    return -(-qlen // piece_rows(qlen))


def piece(c, p):
    """Piece p: (its first row, its rows). A request's rows are cut every
    ``rg`` rows, its last piece the rest (static when a request is one
    piece)."""
    rg, ql, ppr = c["rg"], c["qlen"], c["ppr"]
    if const_expr(ppr == 1):
        return p * ql, rg
    k = p % ppr
    return (p // ppr) * ql + k * rg, fx.min(fx.Int32(rg), ql - k * rg)


def row_units(n, long):
    """A row's score units: its ``n`` local keys' if long, else none."""
    return long.select((n + SCORE_KEYS - 1) // SCORE_KEYS, fx.Int32(0))


def score_units(c, p):
    """Piece p's score units: its last row's (the piece's widest: a verify's
    later rows see more)."""
    lead, span = piece(c, p)
    _, n, long, _ = row_kind(c, lead + span - 1)
    return row_units(n, long)


# LDS ``plan`` words (``step_table``): the step's score tasks (total, log2 of
# the units a task) and segment tasks (total, log2 of the chunks a segment),
# then a word a piece (its first score task) and a word a row (its first
# segment)
PLAN_SCORE_TOTAL, PLAN_SCORE_LG, PLAN_SEG_TOTAL, PLAN_SEG_LG = range(4)
PLAN_FIRSTS = 4


def pieces(c):
    return c["S"] // c["qlen"] * c["ppr"]


def _score_firsts(c):
    return c["plan"] + PLAN_FIRSTS


def _seg_firsts(c):
    return c["plan"] + PLAN_FIRSTS + pieces(c)


def _lanes(s):
    """The lanes a plan over S rows reduces over: the least power of two >= S
    (S is the build's, so S=1 reduces over none)."""
    return 1 << (s - 1).bit_length()


def _rows_sum(v, s):
    """v summed over the rows' lanes (lanes past S hold 0), on those lanes."""
    return butterfly(v, tuple(o for o in (32, 16, 8, 4, 2, 1) if o < _lanes(s)))


@traced
def _lay_out(c, at, mine, cnt, s, words):
    """A row lane's ``cnt`` tasks: on a lane that owns an item (``mine``) the
    item's first (an exclusive scan, rows in order) to ``at``; the total and
    ``words`` to the plan's head."""
    lane = c["lane"]
    incl = wave_incl_scan(cnt, lane, _lanes(s))
    if mine:
        fx.ptr_store(incl - cnt, at)
    if lane == _lanes(s) - 1:
        fx.ptr_store(incl, c["plan"] + words[0])
        fx.ptr_store(words[2], c["plan"] + words[1])


def score_plan(c):
    """(the step's score tasks, log2 of the units a task)."""
    return fx.ptr_load(c["plan"] + PLAN_SCORE_TOTAL), fx.ptr_load(
        c["plan"] + PLAN_SCORE_LG
    )


def segment_plan(c):
    """(the step's segment tasks, log2 of the chunks a segment)."""
    return fx.ptr_load(c["plan"] + PLAN_SEG_TOTAL), fx.ptr_load(c["plan"] + PLAN_SEG_LG)


def _locate(firsts, n, task):
    """Task ``task`` -> (item, task of the item): the last of n items whose
    first task is at or before it (an item with no tasks shares its first
    with the next), by bisection over the nondecreasing firsts in LDS."""
    at = fx.Int32(0)
    step = 1 << (n - 1).bit_length() >> 1 if n > 1 else 0
    while step:
        probe = at + step
        ok = (probe < n) & (
            fx.ptr_load(firsts + fx.min(probe, fx.Int32(n - 1))) <= task
        )
        at = ok.select(probe, at)
        step >>= 1
    return at, task - fx.ptr_load(firsts + at)


def locate_score(c, task):
    """Score task ``task`` -> (piece, task of the piece)."""
    return _locate(_score_firsts(c), pieces(c), task)


def locate_segment(c, sid):
    """Segment task ``sid`` -> (row, segment of the row)."""
    return _locate(_seg_firsts(c), c["S"], sid)


# ---------------------------------------------------------------- score
def _block_keys(c, blk, key, j):
    """A lane's key of one cache block: dims j*16 .. +16 and (4 + j)*16 ..
    +16 (two 16 B loads) and its scale, at device scope (the newest key was
    written by this launch, ``stage_k``)."""
    sh = c["shape"]
    kv = [
        fx.Vector(
            bo.buffer_load(
                blk,
                cache_byte(sh, key, (j + 4 * p) * TILE) // 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        for p in range(2)
    ]
    ks = fx.Float32(
        bo.buffer_load(
            blk,
            scale_byte(sh, key) // 4,
            vec_width=1,
            dtype=T.f32,
            cache_modifier=CM_DEV,
        )
    )
    b64 = fx.Vector.from_elements(
        [kv[p][d] for p in range(2) for d in range(4)], fx.Int32
    ).bitcast(fx.Int64)
    return b64, ks


# LDS ``qrows``: a score task's rows 1 .. rg - 1 (row 0 is read straight to
# registers), each its FP8 query -- a head's head_dim bytes at a stride of
# QPAD words more (16 lanes reading 16 B at heads lane % 16 then meet the 64
# banks once) -- then its folded head weights
QPAD = 4


def head_words(sh):
    return sh.head_dim // 4 + QPAD


def qrow_words(sh):
    return sh.heads * head_words(sh) + sh.heads


def _query(sh, j, lane, q_at, w_at):
    """A row's FP8 query (an i64 of 8 E4M3 a 16-head half and MFMA step) and
    folded head weights, a lane's share: ``q_at(head, word)`` / ``w_at(head)``
    load 16 B of the row's query / weights from where it is held."""
    qs, ws = [], []
    for hh in range_constexpr(sh.heads // TILE):
        head = TILE * hh + lane % 16
        qa = [q_at(head, (j + 4 * p) * (TILE // 4)) for p in range(2)]
        qs.append(
            fx.Vector.from_elements(
                [qa[p][d] for p in range(2) for d in range(4)], fx.Int32
            ).bitcast(fx.Int64)
        )
        w = w_at(TILE * hh + 4 * j)
        ws.append([fx.Int32(w[i]).bitcast(fx.Float32) for i in range(4)])
    return qs, ws


def _row_query(c, q, j, lane):
    """Row q's query and head weights from where ``stage_iquant`` wrote them
    (device scope: this launch's)."""
    sh = c["shape"]
    return _query(
        sh, j, lane,
        lambda h, x: fx.Vector(ld_dev(c["iq8"], (q * sh.qrows + h * sh.head_dim) // 4 + x, 4)),
        lambda h: fx.Vector(ld_dev(c["iw"], q * sh.heads + h, 4)),
    )  # fmt: skip


def _staged_query(c, r, j, lane):
    """The task's row r (>= 1)'s query and head weights from LDS ``qrows``."""
    sh = c["shape"]
    base = c["qrows"] + (r - 1) * qrow_words(sh)
    vec4 = fx.Vector.make_type(4, fx.Int32)
    return _query(
        sh, j, lane,
        lambda h, x: fx.Vector(fx.ptr_load(base + h * head_words(sh) + x, result_type=vec4)),
        lambda h: fx.Vector(fx.ptr_load(base + sh.heads * head_words(sh) + h, result_type=vec4)),
    )  # fmt: skip


def _rows_loads(c, lead, span, rg):
    """A thread's 16 B chunks of the piece's rows 1 .. rg - 1's queries and
    head weights (``_row_query``'s words), all issued at once, each with its
    LDS ``qrows`` word and whether it is one: a task waits one load's latency
    for its rows, not one a row. Rows past the span repeat the last."""
    sh, tid = c["shape"], c["tid"]
    hc, wc = sh.head_dim // 16, sh.heads // 4
    out = []
    # the queries' chunks, then the weights': a base a loop, not one a lane
    for base, per, at0 in (
        (c["iq8"], sh.heads * hc, 0),
        (c["iw"], wc, sh.heads * head_words(sh)),
    ):
        n = (rg - 1) * per
        for k in range_constexpr(-(-n // THREADS)):
            e = fx.min(tid + THREADS * k, n - 1)
            r, i = e // per, e % per
            row = lead + fx.min(r + 1, span - 1)
            if const_expr(at0 == 0):
                h, x = i // hc, i % hc
                src = (row * sh.qrows + h * sh.head_dim) // 4 + 4 * x
                at = h * head_words(sh) + 4 * x
            else:
                src, at = row * sh.heads + 4 * i, at0 + 4 * i
            v = fx.Vector(ld_dev(base, src, 4))
            out.append((v, r * qrow_words(sh) + at, tid + THREADS * k < n))
    return out


@traced
def _rows_store(c, vs):
    """``_rows_loads``' chunks into LDS ``qrows`` (no barrier: the caller's)."""
    for v, at, live in vs:
        if live:
            fx.ptr_store(v, c["qrows"] + at)


@traced
def stage_score(c, p, u0, nu, width):
    """Piece p's score units u0 .. u0 + nu - 1 (nu <= ``width``, a build
    constant; wave w: local keys 16 w .. of each, from the cache block they
    sit in; lane: key lane % 16, dim group lane // 16): every unit's key
    loads issued before the first MFMA, the keys scored for every row of the
    piece off the one load (each row its query and its own key count), one
    drain for all the rows' and units' flags. ``width`` units are loaded and
    scored whatever nu is: a step's tasks run the build of their width
    (``score_task``). Rows past a short piece's span repeat its last and
    write nothing (V4.1's pieces)."""
    tid, lane, wave, sh = c["tid"], c["lane"], c["wave"], c["shape"]
    a, rg, ql = c["args"], c["rg"], c["qlen"]
    c["stamp"](c["tl_score"])
    lead, span = piece(c, p)
    last = lead + span - 1
    static = isinstance(span, int)
    rows_g = [
        lead + (r if static else fx.min(fx.Int32(r), span - 1)) for r in range(rg)
    ]
    # the rows' quantized queries, and the new keys this rank wrote of the
    # request's rows up to the piece's last (a verify row sees its request's
    # earlier drafts' new keys, each written by its own row's task here)
    if tid < span * sh.heads:
        c["poll"]([(c["qq"], lead * sh.heads + tid, 1)])
    first = lead - lead % ql
    for d in range_constexpr(ql):
        jrow = first + d
        slot = fx.Int32(
            bo.buffer_load(
                rsrc(a["slots"]), 2 * fx.min(jrow, last), vec_width=1, dtype=T.i32
            )
        )
        if (tid == rg * sh.heads + d) & (slot >= 0) & (jrow <= last):
            c["poll"]([(c["kdone"], jrow, 1)])
    kinds = [row_kind(c, r) for r in rows_g]
    ns = [n for _, n, _, _ in kinds]
    logs = [ilog_at(c, row) for row in rows_g]
    # the piece's last key (its widest row's): local, or global replicated
    last_key = kinds[-1][0] - 1 if c["replicated"] else ns[-1] - 1
    dg = lane // 16
    # this wave's first key of each unit, where it is stored: the task's
    # units past nu repeat its last, and a block past the group's last is its
    # last (in bounds; their keys are past every row's n, never stored)
    mul, add = unit_runs(c, kinds[-1][3])
    firsts = [
        ((u0 + fx.min(fx.Int32(i), nu - 1)) * mul + add) * SCORE_KEYS + TILE * wave
        for i in range(width)
    ]
    req = request_of(c, lead)
    blocks = [key_block(c, req, f, last_key) for f in firsts]
    gpu.barrier()
    c["stamp"](c["tl_score"] + 1)
    # row 0's query and head weights to registers and the other rows' to LDS
    # (their loads all out before the keys'; the stores wait on them alone),
    # a row's LDS reads a row ahead of its MFMAs
    nxt = _row_query(c, rows_g[0], dg, lane)
    staged = _rows_loads(c, lead, span, rg) if const_expr(rg > 1) else []
    keys = [
        _block_keys(
            c,
            rsrc(a["cache"] + fx.Int64(e) * fx.Int64(sh.block_bytes)),
            row + lane % 16,
            dg,
        )
        for e, row in blocks
    ]
    if const_expr(rg > 1):
        _rows_store(c, staged)
        gpu.barrier()
    for r in range_constexpr(rg):
        qs, ws = nxt
        if const_expr(r + 1 < rg):
            nxt = _staged_query(c, r + 1, dg, lane)
        for i in range_constexpr(width):
            b64, ks = keys[i]
            terms = {}
            for hh in range_constexpr(sh.heads // TILE):
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                for x in range_constexpr(4):
                    acc = mfma_fp8(qs[hh][x], b64[x], acc)
                for e in range_constexpr(4):
                    terms[hh, e] = (fx.max(acc[e] * ks, fx.Float32(0.0)), ws[hh][e])
            col = SCORE_KEYS * (u0 + i) + TILE * wave + lane % 16
            score = fx.Float32(head_sum(terms))
            # the row's keys alone: the next row's ILOG starts past its last
            stored = (lane < 16) & (i < nu) & (col < ns[r])
            if const_expr(not static):
                stored = stored & (r < span)
            if stored:
                bo.buffer_store(
                    score,
                    rsrc(c["ilog"]),
                    logs[r] + col,
                    cache_modifier=CM_DEV,
                )
    c["stamp"](c["tl_score"] + 2)
    # thread k: the piece's row k // TASK_UNITS, unit k % TASK_UNITS
    fr, fu = tid // TASK_UNITS, tid % TASK_UNITS
    flag = sflag_at(c, lead + fx.min(fr, span - 1)) + u0 + fu
    publish(
        c["put"], c["sflag"], flag, fx.Int32(1), (fu < nu) & (fr < span)
    )  # fmt: skip
    c["stamp"](c["tl_score"] + 3)


@traced
def score_task(c, p, t, lg):
    """Score task t of piece p (``step_table``): up to 2^lg units, on the
    build of that width."""
    u0 = t << lg
    nu = fx.min(score_units(c, p) - u0, 1 << lg)
    for w in range_constexpr(TASK_UNITS_LG + 1):
        if lg == w:
            stage_score(c, p, u0, nu, 1 << w)


# ---------------------------------------------------------------- select
# LDS ``sel`` words: the pass's digit, the keys still needed, this rank's ties
# taken; from ``SEL_ROWS`` a word a row (``rows_before``)
SEL_DIGIT, SEL_NEED, SEL_TAKE = range(3)
SEL_ROWS = 4


def _chunk_keys(c, r, at, here):
    """This thread's keys of the chunk at local key ``at`` (keys at + tid PT
    ..; ``here`` of them live from ``at``), as radix keys, quad loads;
    (live, keys). A load past the row reads the next row's or, past the
    last, 0 (the ILOG's bound); neither is live."""
    tid = c["tid"]
    live, keys = [], []
    base = ilog_at(c, r)
    src = rsrc(c["ilog"], c["ilog_bytes"])
    for qd in range_constexpr(PT // 4):
        w = fx.Vector(
            bo.buffer_load(
                src,
                base + at + tid * PT + 4 * qd,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        for e in range_constexpr(4):
            ok = tid * PT + 4 * qd + e < here
            live.append(ok)
            v = radix_key(fx.Int32(w[e]).bitcast(fx.Float32))
            keys.append(fx.Int32(ok.select(v, fx.Int32(0))))
    return live, keys


def _zero_hist(c):
    """LDS ``hist`` cleared for a pass's counts."""
    for k in range_constexpr(BINS_A_THREAD):
        fx.ptr_store(fx.Int32(0), c["hist"] + _bin(c["tid"], k))
    gpu.barrier()


@traced
def _count(c, live, keys, p, prefix):
    """One chunk's keys into pass ``p``'s LDS ``hist`` (cleared before the
    segment's first chunk, read after a barrier past its last). Pass 0 adds a
    thread's run of one bin once (real logits share their top byte); later
    passes add each key under the prefix."""
    hist, pt = c["hist"], len(keys)
    if const_expr(p == 0):
        ds = [digit(k, p) for k in keys]
        run = fx.Int32(0)
        for i in range_constexpr(pt):
            run = run + live[i].select(fx.Int32(1), fx.Int32(0))
            last = live[i]
            if const_expr(i + 1 < pt):
                last = last & (~live[i + 1] | (ds[i + 1] != ds[i]))
            if last:
                fx.llvm.atomic_add(hist + ds[i], run, syncscope="workgroup")
            run = last.select(fx.Int32(0), run)
    else:
        for i in range_constexpr(pt):
            if live[i] & under(keys[i], p, prefix):
                fx.llvm.atomic_add(
                    hist + digit(keys[i], p), fx.Int32(1), syncscope="workgroup"
                )


def _row_word(base, i):
    """Word ``i`` of the plain region at device address ``base``."""
    return word(base + fx.Int64(i) * fx.Int64(4))


def _row_at(r, p, copy, b):
    """Row r's histogram of pass ``p``, copy ``copy``, bin ``b``: its ROWHIST word."""
    return ((r * PASSES + p) * ROW_COPIES + copy) * BINS + b


@traced
def _row_add(c, r, p, g):
    """Segment g of a row of several: its histogram of pass ``p`` added into
    the row's copy g % ROW_COPIES (ROWHIST, device-scope integer atomics: one
    sum in any order), every bin also counting the segment in (``ARRIVED``):
    a bin's word says itself when its segments are all in."""
    tid = c["tid"]
    for k in range_constexpr(BINS_A_THREAD):
        b = _bin(tid, k)
        v = fx.ptr_load(c["hist"] + b)
        if tid + k * THREADS < BINS:
            fx.llvm.atomic_add(
                _row_word(c["rowhist"].value, _row_at(r, p, g % ROW_COPIES, b)),
                v + ARRIVED, syncscope="agent",
            )  # fmt: skip


@traced
def _wait_bin(c, idxs, needs):
    """ROWHIST words ``idxs`` (device scope) once each holds its ``needs``
    segments (the mailbox's wait: bounded and named in a debug build): their
    counts summed."""
    base = c["rowhist"].value

    def load_all():
        return fx.Vector.from_elements(
            [fx.Int32(ld_dev(base, i)) for i in idxs], fx.Int32
        )

    def pending(v):
        bad = (v[0] >> COUNT_BITS) < needs[0]
        for e in range_constexpr(1, len(idxs)):
            bad = bad | ((v[e] >> COUNT_BITS) < needs[e])
        return bad

    v = c["spin"](c["rowhist"], idxs[0], load_all, pending)
    tot = v[0] & (ARRIVED - 1)
    for e in range_constexpr(1, len(idxs)):
        tot = tot + (v[e] & (ARRIVED - 1))
    return tot


@traced
def _row_counts(c, r, p, nseg):
    """Once the row's ``nseg`` segments have added theirs (``_row_add``): the
    row's histogram of pass ``p`` over LDS ``hist``."""
    tid = c["tid"]
    # copy k holds the segments k, k + ROW_COPIES, ..
    needs = [(nseg - k + ROW_COPIES - 1) // ROW_COPIES for k in range(ROW_COPIES)]
    # every thread's own counts read (``_row_add``) before any is overwritten
    gpu.barrier()
    for k in range_constexpr(BINS_A_THREAD):
        b = _bin(tid, k)
        if tid + k * THREADS < BINS:
            idxs = [_row_at(r, p, cp, b) for cp in range(ROW_COPIES)]
            fx.ptr_store(_wait_bin(c, idxs, needs), c["hist"] + b)
    gpu.barrier()


@traced
def _row_clear(c, r):
    """Row r's ROWHIST back to zero, 16 B a store, by its last segment once
    every other one is past its passes and after the row's own publications
    (their drain does not wait on these): the step's next launch (the peer
    region is zeroed a step, not a launch) finds it as the step's first did."""
    tid = c["tid"]
    quads = PASSES * ROW_COPIES * BINS // 4
    zero = fx.Vector.filled(4, 0, fx.Int32)
    for k in range_constexpr(-(-quads // THREADS)):
        i = tid + k * THREADS
        if i < quads:
            bo.buffer_store(
                zero, rsrc(c["rowhist"].value), (r * quads + i) * 4,
                cache_modifier=CM_DEV,
            )  # fmt: skip


@traced
def _lead_push(c, r, p):
    """The row's leader (its segment 0) under the key split: the row's
    histogram of pass ``p`` here (LDS ``hist``) pushed to every rank."""
    tid, world, rank = c["tid"], c["world"], c["rank"]
    for k in range_constexpr(BINS_A_THREAD):
        b = _bin(tid, k)
        if tid + k * THREADS < BINS:
            v = fx.ptr_load(c["hist"] + b)
            for dst in range_constexpr(world):
                c["put"](c["phist"][dst], hist_at(r, p, rank, b, world), v)


@traced
def _decide(c, r, p, nseg, split):
    """Every rank's histogram of pass ``p`` (this rank's peer region, where
    every rank's leader pushed it) summed in rank order -> the digit, the
    keys still needed and, on the last pass, this rank's ties taken (LDS
    ``sel``). Every segment of the row decides alike, from the same counts:
    nothing waits on the leader but the counts themselves. Own rows: no
    rank's push, the row's counts are this CTA's LDS ``hist`` (one segment)
    or its ``nseg`` segments' added (``_row_counts``)."""
    # the bins from the top, DEC_A_LANE a lane, over the first DEC_WAVES
    # waves: one poll batch a lane, a wave scan, one exchange of the waves'
    # totals. The waves past them poll the last wave's bins again and decide
    # nothing (no value assigned under a traced branch).
    at = fx.min(c["wave"], DEC_WAVES - 1) * 64 + c["lane"]
    bins = [BINS - 1 - (DEC_A_LANE * at + k) for k in range(DEC_A_LANE)]
    if split:
        _choose(c, p, bins, *_rank_counts(c, r, p, bins))
    else:
        # no other rank holds a key of the row
        if nseg > 1:
            _row_counts(c, r, p, nseg)
        vs = [fx.ptr_load(c["hist"] + b) for b in bins]
        _choose(c, p, bins, vs, [fx.Int32(0)] * DEC_A_LANE, vs)


def _rank_counts(c, r, p, bins):
    """Every rank's counts of ``bins`` in pass ``p`` (this rank's peer region,
    where every rank's leader pushed them): (their sums in rank order, the
    lower ranks', this rank's)."""
    world, rank = c["world"], c["rank"]
    got = c["poll"](
        [
            (c["phist_own"], hist_at(r, p, src, b, world), 1)
            for b in bins
            for src in range(world)
        ]
    )
    vs, lowers, heres = [], [], []
    for k in range(len(bins)):
        per_src = [fx.Int32(got[k * world + i][0]) for i in range(world)]
        tot, lower = per_src[0], fx.Int32(0)
        for i in range(1, world):
            tot = tot + per_src[i]
        for src in range(world):
            lower = lower + (src < rank).select(per_src[src], fx.Int32(0))
        vs.append(tot)
        lowers.append(lower)
        heres.append(pick(per_src, rank))
    return vs, lowers, heres


@traced
def _choose(c, p, bins, vs, lowers, heres):
    """The row's counts of ``bins`` (``vs``; the lower ranks' and this rank's
    for the last pass's ties) -> the digit, the keys still needed and, on the
    last pass, this rank's ties taken (LDS ``sel``)."""
    lane, wave = c["lane"], c["wave"]
    sel, scan = c["sel"], c["scan"]
    need = fx.ptr_load(sel + SEL_NEED)
    c["stamp"](c["tl_pass"] + 3 * p + 2)
    mine = vs[0]
    for k in range_constexpr(1, DEC_A_LANE):
        mine = mine + vs[k]
    incl = wave_incl_scan(mine, lane)
    deciding = wave < DEC_WAVES
    if deciding & (lane == 63):
        fx.ptr_store(incl, scan + wave)
    gpu.barrier()
    above = incl - mine
    for w in range_constexpr(DEC_WAVES - 1):
        above = above + (w < wave).select(fx.ptr_load(scan + w), fx.Int32(0))
    for k in range_constexpr(DEC_A_LANE):
        if deciding & (above < need) & (need <= above + vs[k]):
            fx.ptr_store(bins[k], sel + SEL_DIGIT)
            fx.ptr_store(need - above, sel + SEL_NEED)
            # the last pass: the threshold bin's count on the lower ranks
            # and here
            if const_expr(p == PASSES - 1):
                take = fx.min(need - above - lowers[k], heres[k])
                fx.ptr_store(fx.max(take, 0), sel + SEL_TAKE)
        above = above + vs[k]
    gpu.barrier()


def _marks(live, keys, prefix):
    """A thread's keys above the threshold and on it: (gt, eq, their counts
    packed 16 bits each -- a chunk's keys fit)."""
    gt = [live[i] & u32_gt(keys[i], prefix) for i in range(len(keys))]
    eq = [live[i] & (keys[i] == prefix) for i in range(len(keys))]
    n_gt, n_eq = fx.Int32(0), fx.Int32(0)
    for i in range_constexpr(len(keys)):
        n_gt = n_gt + gt[i].select(fx.Int32(1), fx.Int32(0))
        n_eq = n_eq + eq[i].select(fx.Int32(1), fx.Int32(0))
    return gt, eq, (n_gt << 16) | n_eq


@traced
def _emit_chunk(c, r, slots, gt, eq, at, run, ties):
    """A chunk's chosen keys (``thread_slots``' ``slots``) as slots from
    ``at``: every key above the threshold, a tie while ``run`` (the
    segment's equal keys before it) is under ``ties``."""
    chosen = []
    for i in range_constexpr(PT):
        tie = eq[i] & (run < ties)
        run = run + eq[i].select(fx.Int32(1), fx.Int32(0))
        chosen.append(gt[i] | tie)
    for i in range_constexpr(PT):
        if chosen[i]:
            _put_slot(c, r, slots[i], at)
        # advanced outside the branch: no value carried out of a traced if
        at = at + chosen[i].select(fx.Int32(1), fx.Int32(0))


@traced
def _sum_before(c, g):
    """The row's earlier segments' (above, equal) counts summed: segment s's
    in LDS ``before`` (2 s, 2 s + 1), s < g < BLOCKS, so the first BLOCKS /
    64 waves hold them: a wave sum each, one barrier, the waves' sums added."""
    tid, lane, wave, scan = c["tid"], c["lane"], c["wave"], c["scan"]
    at = c["before"] + 2 * fx.min(tid, fx.Int32(BLOCKS - 1))
    earlier = tid < g
    w_gt = wave_sum(earlier.select(fx.ptr_load(at), fx.Int32(0)))
    w_eq = wave_sum(earlier.select(fx.ptr_load(at + 1), fx.Int32(0)))
    if (lane == 0) & (wave < BEFORE_WAVES):
        fx.ptr_store(w_gt, scan + wave)
        fx.ptr_store(w_eq, scan + WAVES + wave)
    gpu.barrier()
    gt, eq = fx.ptr_load(scan), fx.ptr_load(scan + WAVES)
    for w in range_constexpr(1, BEFORE_WAVES):
        gt = gt + fx.ptr_load(scan + w)
        eq = eq + fx.ptr_load(scan + WAVES + w)
    return gt, eq


@traced
def stage_segment(c, r, g, sid, lg):
    """Segment g of long row r (the step's segment ``sid``): 2^lg chunks of
    local keys from g 2^lg CHUNK (thread i: PT of a chunk's). The exact global
    threshold over every segment and rank (radix passes: the row's segments
    add their counts into the row's histogram, ``_row_add``; under the key
    split its segment 0 pushes the row's to every rank, every segment reads
    all ranks' and decides alike), then this segment's winners -> their
    slots at the row's offset plus the earlier segments' winners. The first
    chunk's keys stay in registers across the passes; a later chunk's (a step
    whose long rows' chunks outnumber the CTAs) are re-read from ILOG."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    sel, before, scan = c["sel"], c["before"], c["scan"]
    _, n, _, split = row_kind(c, r)
    # the segment that pushes the row's counts to every rank: a split row's
    # first (an owned row's stay here)
    lead = (g == 0) & split
    seglen = fx.Int32(CHUNK) << lg
    nseg = (n + seglen - 1) >> (lg + CHUNK_LG)
    lo = g << (lg + CHUNK_LG)
    here = fx.min(n - lo, seglen)
    nck = (here + CHUNK - 1) // CHUNK
    # its score units' flags (a chunk is whole units)
    units = (here + SCORE_KEYS - 1) // SCORE_KEYS
    for ku in range((units + THREADS - 1) // THREADS):
        u = tid + ku * THREADS
        if u < units:
            c["poll"]([(c["sflag"], sflag_at(c, r) + lo // SCORE_KEYS + u, 1)])
    if tid == 0:
        fx.ptr_store(fx.Int32(c["shape"].topk), sel + SEL_NEED)
    gpu.barrier()
    c["stamp"](4)
    live, keys = _chunk_keys(c, r, lo, here)
    req = request_of(c, r)
    slots = thread_slots(c, req, lo, n, split)
    prefix = fx.Int32(0)
    for p in range_constexpr(PASSES):
        _zero_hist(c)
        _count(c, live, keys, p, prefix)
        for kc in range(1, nck):
            c_live, c_keys = _chunk_keys(c, r, lo + kc * CHUNK, here - kc * CHUNK)
            _count(c, c_live, c_keys, p, prefix)
        gpu.barrier()
        c["stamp"](c["tl_pass"] + 3 * p)
        # the row's segments' counts into one histogram; under the key split
        # its leader pushes the row's (this rank's) to every rank
        if nseg > 1:
            _row_add(c, r, p, g)
        if lead:
            if nseg > 1:
                _row_counts(c, r, p, nseg)
            _lead_push(c, r, p)
        c["stamp"](c["tl_pass"] + 3 * p + 1)
        if const_expr(p == PASSES - 1 and c["delay_rank"] >= 0):  # noqa: SIM102
            # harness only (``IxBuild.delay_rank``): the last pass's pushes
            # have landed on every rank; this one reads its own late
            if c["rank"] == c["delay_rank"]:
                sleep_ticks(c["delay_ticks"])
        _decide(c, r, p, nseg, split)
        prefix = (prefix << DIGIT_BITS[p]) | fx.ptr_load(sel + SEL_DIGIT)
    take = fx.ptr_load(sel + SEL_TAKE)
    c["stamp"](5)
    # this segment's keys above the threshold and on it, a chunk's counts in
    # one scan: the slots before a thread's are the earlier threads' (and
    # chunks') above-threshold keys and as many of their ties as it takes
    gt, eq, packed = _marks(live, keys, prefix)
    excl, tot = block_excl_scan(packed, lane, wave, scan)
    seg_gt, seg_eq = (tot >> 16) & 0xFFFF, tot & 0xFFFF
    for kt in range(1, nck):
        t_live, t_keys = _chunk_keys(c, r, lo + kt * CHUNK, here - kt * CHUNK)
        _, _, t_packed = _marks(t_live, t_keys, prefix)
        _, t_tot = block_excl_scan(t_packed, lane, wave, scan)
        seg_gt = seg_gt + ((t_tot >> 16) & 0xFFFF)
        seg_eq = seg_eq + (t_tot & 0xFFFF)
    # the earlier segments' (gt, eq) summed, this one's to the later ones
    if tid == 0:
        c["put_words"](c["sge"], 2 * sid, [seg_gt, seg_eq])
    if tid < g:
        got = c["poll"]([(c["sge"], 2 * (sid - g + tid), 2)])
        fx.ptr_store(fx.Int32(got[0][0]), before + 2 * tid)
        fx.ptr_store(fx.Int32(got[0][1]), before + 2 * tid + 1)
    gpu.barrier()
    gt_before, eq_before = _sum_before(c, g)
    # the row's ties go to its segments in local order
    ties = fx.min(fx.max(take - eq_before, 0), seg_eq)
    total = gt_before + seg_gt + take
    last = g == nseg - 1
    if const_expr(c["replicated"]):
        # this rank's winners from 0 of its slab row (``_put_slot``)
        off = fx.Int32(0)
    else:
        if (tid == 0) & last:
            c["put"](c["cnt"], r, total)
        off = rows_before(c, r)
    base = off + gt_before + fx.min(take, eq_before)
    gt_excl, run = (excl >> 16) & 0xFFFF, excl & 0xFFFF
    _emit_chunk(
        c, r, slots, gt, eq, base + gt_excl + fx.min(run, ties), run, ties
    )  # fmt: skip
    gt_seen, eq_seen = (tot >> 16) & 0xFFFF, tot & 0xFFFF
    for ke in range(1, nck):
        e_live, e_keys = _chunk_keys(c, r, lo + ke * CHUNK, here - ke * CHUNK)
        e_gt, e_eq, e_packed = _marks(e_live, e_keys, prefix)
        e_excl, e_tot = block_excl_scan(e_packed, lane, wave, scan)
        e_gx = gt_seen + ((e_excl >> 16) & 0xFFFF)
        e_run = eq_seen + (e_excl & 0xFFFF)
        _emit_chunk(
            c, r, thread_slots(c, req, lo + ke * CHUNK, n, split), e_gt, e_eq,
            base + e_gx + fx.min(e_run, ties), e_run, ties,
        )  # fmt: skip
        gt_seen = gt_seen + ((e_tot >> 16) & 0xFFFF)
        eq_seen = eq_seen + (e_tot & 0xFFFF)
    if const_expr(c["replicated"]):
        _announce_row(c, r, g, sid, last, total)
    elif last:
        finish_row(c, r, off, total)
    # every earlier segment is past its passes (its SGE above): the row's last
    # leaves ROWHIST as it found it
    if last & (nseg > 1):
        _row_clear(c, r)


@traced
def _announce_row(c, r, g, sid, last, total):
    """Replicated: once every segment of row r has written its winners into
    this rank's own slab (``_put_slot``; the other ranks read them from it),
    their count to every rank (RCNT[r, rank]). A segment's stores drained,
    then its EMITTED; the row's last segment waits on the earlier ones' and
    announces."""
    tid = c["tid"]
    if ~last:
        publish(c["put"], c["emitted"], sid, fx.Int32(1), tid == 0)
    else:
        if tid < g:
            c["poll"]([(c["emitted"], sid - g + tid, 1)])

        def put_each(dsts, i, v):
            for dst in dsts:
                c["put"](dst, i, v)

        publish(put_each, c["prcnt"], r * c["world"] + c["rank"], total, tid == 0)


# ---------------------------------------------------------------- rows
def key_slot(c, req, jj, split):
    """Local key jj's KV slot (of request ``req``, its keys ``split`` or not:
    ``row_kind``)."""
    e, row = key_block(c, req, stored_at(c, jj, split))
    return e * c["shape"].block_rows + row


def thread_slots(c, req, at_key, n, split):
    """A thread's PT keys of the chunk at local key ``at_key`` (``_chunk_keys``'
    keys) as KV slots: PT consecutive keys from a multiple of PT share their
    block (PT divides a block's rows and a replicated run), so one block
    table load, issued with the keys -- not one a chosen key after the row's
    waits (``rows_before``)."""
    first = at_key + c["tid"] * PT
    rows = c["shape"].block_rows
    e, _ = key_block(c, req, stored_at(c, fx.min(first, n - 1), split))
    return [e * rows + stored_at(c, first + i, split) % rows for i in range(PT)]


def _put_slot(c, r, slot, at):
    """A key's KV slot of row r -> the packed indices at ``at``; replicated,
    ``at`` of this rank's winners of the row, into this rank's own slab only:
    every rank reads them from it (``_replicated_long``), 16 B a load, not a
    4 B store a rank here."""
    if const_expr(c["replicated"]):
        at = (r * c["world"] + c["rank"]) * c["shape"].topk + at
        bo.buffer_store(slot, rsrc(c["slab"]), at, cache_modifier=CM_SYS)
    else:
        bo.buffer_store(slot, rsrc(c["args"]["out_idx"]), at)


@traced
def rows_before(c, r):
    """The slots of the rows before r: a short row's are its keys here (from
    its context, no wait), a long row's its CNT (its last segment's). A short
    row never waits on work a CTA runs after it: its count needs none."""
    tid, s, sel = c["tid"], c["S"], c["sel"]
    if tid < r:
        _, n, long, _ = row_kind(c, tid)
        fx.ptr_store(n, sel + SEL_ROWS + tid)
        if long:
            got = c["poll"]([(c["cnt"], tid, 1)])
            fx.ptr_store(fx.Int32(got[0][0]), sel + SEL_ROWS + tid)
    gpu.barrier()
    off = fx.Int32(0)
    for q in range_constexpr(s):
        off = off + (q < r).select(fx.ptr_load(sel + SEL_ROWS + q), fx.Int32(0))
    gpu.barrier()
    return off


@traced
def finish_row(c, r, off, count):
    a = c["args"]
    if c["tid"] == 0:
        bo.buffer_store(count, rsrc(a["out_cnt"]), r)
        bo.buffer_store(off + count, rsrc(a["out_ptr"]), r + 1)
        if r == 0:
            bo.buffer_store(fx.Int32(0), rsrc(a["out_ptr"]), 0)


@traced
def stage_row(c, r):
    """Row r's slots but a long row's own winners (its segments write
    those): under DCP a short row's every key held here, in local order;
    replicated, at the row's ``out_ptr`` (the MLA's indptr, an input), a
    short row's every key in position order, a long row's every rank's
    winners from the slabs in rank order."""
    if const_expr(c["replicated"]):
        _replicated_row(c, r)
        return
    tid = c["tid"]
    _, n, long, _ = row_kind(c, r)
    if ~long:
        # a short row's keys here: at most top-K / DCP + 1, a thread each,
        # their slots loaded before the earlier rows' counts are waited for
        req = request_of(c, r)
        each = range(-(-(c["shape"].topk // c["world"] + 1) // THREADS))
        slots = [
            key_slot(
                c,
                req,
                fx.max(fx.min(tid + k * THREADS, n - 1), 0),
                fx.Boolean(True),
            )
            for k in each
        ]
        off = rows_before(c, r)
        for k in range_constexpr(len(slots)):
            jj = tid + k * THREADS
            if jj < n:
                _put_slot(c, r, slots[k], off + jj)
        finish_row(c, r, off, n)


@traced
def _replicated_row(c, r):
    tid, topk, a = c["tid"], c["shape"].topk, c["args"]
    # the row's output is long by its context: this rank may hold none of
    # its keys (``own_rows``: another rank's row)
    ctx, _, _, split = row_kind(c, r)
    long = ~short(c["shape"], ctx) & (ctx > 0)
    off = fx.Int32(bo.buffer_load(rsrc(a["out_ptr"]), r, vec_width=1, dtype=T.i32))
    out = rsrc(a["out_idx"])
    if ~long:
        req = request_of(c, r)
        for k in range_constexpr(-(-topk // THREADS)):
            j = tid + k * THREADS
            if j < ctx:
                e, row = key_block(c, req, j)
                bo.buffer_store(e * c["shape"].block_rows + row, out, off + j)
    else:
        _replicated_long(c, r, off, out, split)


@traced
def _replicated_long(c, r, off, out, split):
    """A long row's winners, replicated: read from the slabs of the ranks
    that selected them -- an owned row's owner alone, a row whose keys are
    ``split`` every rank in rank order -- once each one's count says they are
    all there. A rank writes its winners into its own slab only
    (``_put_slot``) and the readers pull 16 B a load."""
    if split:
        _split_winners(c, r, off, out)
    else:
        world, topk, sel = c["world"], c["shape"].topk, c["sel"]
        src = owner(c, r)
        # one thread polls the owner's count, the CTA reads it from LDS
        if c["tid"] == 0:
            got = c["poll"]([(c["prcnt_own"], r * world + src, 1)])
            fx.ptr_store(fx.Int32(got[0][0]), sel + SEL_ROWS)
        gpu.barrier()
        cnt = fx.ptr_load(sel + SEL_ROWS)
        _copy_winners(c, pick(c["pslab"], src), (r * world + src) * topk, cnt, out, off)


@traced
def _split_winners(c, r, off, out):
    """A row whose keys are split over the ranks: every rank's winners in
    rank order."""
    tid, world, topk, sel = c["tid"], c["world"], c["shape"].topk, c["sel"]
    if tid < world:
        got = c["poll"]([(c["prcnt_own"], r * world + tid, 1)])
        fx.ptr_store(fx.Int32(got[0][0]), sel + SEL_ROWS + tid)
    gpu.barrier()
    at = off
    for src in range_constexpr(world):
        cnt = fx.ptr_load(sel + SEL_ROWS + src)
        _copy_winners(c, c["pslab"][src], (r * world + src) * topk, cnt, out, at)
        at = at + cnt


@traced
def _copy_winners(c, slab_at, row0, cnt, out, at):
    """``cnt`` slots from a rank's slab (``slab_at``, its base) at ``row0`` to
    the packed indices at ``at``: 4 slots a 16 B load."""
    tid, topk = c["tid"], c["shape"].topk
    slab = rsrc(slab_at)
    for k in range_constexpr(-(-topk // (4 * THREADS))):
        i = 4 * (tid + k * THREADS)
        v = fx.Vector(
            bo.buffer_load(
                slab, row0 + fx.min(i, topk - 4), vec_width=4, dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )  # fmt: skip
        for e in range_constexpr(4):
            if i + e < cnt:
                bo.buffer_store(v[e], out, at + i + e)
