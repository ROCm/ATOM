# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A top-p row's draw over its largest logits (``K`` a rank), one exchange
between the ranks: the case of a top-p set within them; past them, the
two-level scan (``top_p_scan``) instead, every rank alike.

A row's ``helpers`` CTAs (``top_p_logits``: slices of its logits) each take
their slice's K largest (a candidate: a signed 64-bit sort key, the bf16's
16-bit order key high, 0x7FFFFFFF - id low: larger key, then lower id,
first) by two 256-bin counts (the key's high byte, then the low byte within
the K-th's bin), its largest key and sum of exp2((b - max) log2(e) / T),
sort them across a wave (``bitonic.wave_sort``) and hand them to the row's
lead (this rank's CAND). The lead merges them into the rank's K largest
(``bitonic.wave_merge_top``, a tree over waves) and puts them, with the
rank's largest key, sum and bound (every id above it is in the list), into
every rank's CANDX; then merges the ranks' lists alike on every rank:
M, Z = sum of the ranks' sums rescaled to M, and a candidate kept when the
mass of the candidates with a larger logit is below top_p Z (ATOM's rule).
The list decides the set when a candidate above every rank's bound is not
kept (the set ends inside the exact part of the list); then the id is the
race m / E over the kept (the scan's score), else every rank takes the
scan (``GO``: the lead's decision, to its helpers).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr

from atom.mono.device import race
from atom.mono.device.bitonic import wave_merge_top, wave_sort
from atom.mono.device.ops import (
    ballot,
    hw_exp2,
    hw_rcp,
    i64_of,
    i64_words,
    lane_gather,
    readlane,
    rsrc,
    traced,
    uniform,
    wave_prefix_dpp,
    xshfl,
)
from atom.mono.device.select import INT_MIN, key_bits, order_key
from atom.mono.lm_head import argmax as va
from atom.mono.lm_head import top_p_logits as tl
from atom.mono.plan.execution import WAVES

K = 64  # candidates a slice, a rank, the row (a wave's lanes)
BINS = 256
# pairs a list record: K candidates (two pairs each), then the header: the
# largest key, the sum (f32 bits), the bound key (-1: every id), the count
REC = 2 * K + 4
HEAD = 2 * K
EMPTY = (0, INT_MIN)  # an empty candidate's words: below every one
# LDS words of the select and merges (from word 0, the launch's LDS free):
# the two counts, broadcast words (bin, the count above it, the cut key,
# the count above the cut, the slice's count), the waves' maxima and sums,
# the two slot counters, the slice's list, the merges' lists
W_HI = 0
W_LO = W_HI + BINS
W_BC = W_LO + BINS
W_RED = W_BC + 8
W_CTR = W_RED + 2 * WAVES
W_LIST = W_CTR + 4
W_MERGE = W_LIST + 2 * K
# a junk word a lane, and a junk entry a lane: an inactive lane's atomics
# and stores (no branch around them)
W_JUNK = W_MERGE + 2 * K * WAVES
W_JUNK64 = W_JUNK + 64
WORDS = W_JUNK64 + 2 * 64


def region_pairs(tasks: int, max_rows: int, tp: int) -> tuple[int, int, int]:
    """Pairs of this rank's CAND (a slice's list, [CTA]), one copy of CANDX
    (a rank's list, [source rank][row]) and GO ([row])."""
    return tasks * REC, tp * max_rows * REC, max_rows


def _empty():
    return i64_of(EMPTY[0], EMPTY[1])


def _value(key):
    """The f32 of a 16-bit order key's bf16."""
    return (key_bits(key << 16) & fx.Int32(-65536)).bitcast(fx.Float32)


def _wave_sum(v):
    for off in (32, 16, 8, 4, 2, 1):
        v = v + xshfl(v, off)
    return v


def _wave_max(v):
    for off in (32, 16, 8, 4, 2, 1):
        v = fx.max(v, xshfl(v, off))
    return v


def _wave_max_i64(v):
    for off in (32, 16, 8, 4, 2, 1):
        lo, hi = i64_words(v)
        v = fx.max(v, i64_of(xshfl(lo, off), xshfl(hi, off)))
    return v


def _slice_loads(c, r, span):
    """This thread's loads of the slice ``span`` of row r, batch -> quads
    (``top_p_logits``): a slice of one batch loaded once, held over the
    passes; of more, a batch loaded again each pass."""
    if const_expr(tl.batches(c["key"]) == 1):
        held = tl.slice_batch(c, r, 0, span)
        return lambda bi: held
    return lambda bi: tl.slice_batch(c, r, bi, span)


def _for_values(c, loads, use):
    """``use(b, key, gid, ok)`` for every logit of this thread's ``loads``
    (``_slice_loads``)."""
    for bi in range_constexpr(tl.batches(c["key"])):
        quads = loads(bi)
        for qi in range_constexpr(len(quads)):
            w, q, ok = quads[qi]
            for wi in range_constexpr(4):
                for h in range_constexpr(2):
                    b, key = tl.key16(w[wi], h == 1)
                    use(b, key, c["first"] + q * 8 + wi * 2 + h, ok)


@traced
def _clear(c):
    """The counts and slot counters to 0, the slice's list to empty."""
    tid, lds = c["tid"], c["lds"]
    fx.ptr_store(fx.Int32(0), lds + (W_HI + tid))
    if tid < 4:
        fx.ptr_store(fx.Int32(0), lds + (W_CTR + tid))
    if tid < K:
        fx.ptr_store(_empty(), fx.recast_iter(fx.Int64, lds) + (W_LIST // 2 + tid))
    gpu.barrier()


@traced
def _count_hi(c, loads):
    """The slice's keys by high byte (LDS counts) -> its largest key."""
    lds = c["lds"]
    junk = fx.Int32(W_JUNK) + c["lane"]
    top = [fx.Int32(-1)]

    def use(b, key, gid, ok):
        at = ok.select(fx.Int32(W_HI) + (key >> 8), junk)
        fx.atomic_add(lds + at, fx.Int32(1), syncscope="workgroup")
        top[0] = fx.max(top[0], ok.select(key, fx.Int32(-1)))

    _for_values(c, loads, use)
    m = _wave_max(top[0])
    if c["lane"] == 0:
        fx.ptr_store(m, lds + (W_RED + c["wave"]))
    gpu.barrier()
    m = fx.Int32(fx.ptr_load(lds + W_RED))
    for w in range_constexpr(1, WAVES):
        m = fx.max(m, fx.Int32(fx.ptr_load(lds + (W_RED + w))))
    return m


def _bin_of_kth(cnt, lane, above0):
    """The bin (lane l: bins 255 - 4 l - j, ``cnt[j]``) where the count from
    the top, from ``above0``, first reaches K, the count above it (-1 on
    every lane: fewer than K) and the total. Not ``@traced``: under that
    rewrite the running count loses its updates within the unrolled bins,
    and the count above the hit bin misses the lane's higher bins."""
    tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
    incl = wave_prefix_dpp(tot)
    above = above0 + incl - tot
    hit_b, hit_above = fx.Int32(-1), fx.Int32(-1)
    for j in range_constexpr(4):
        hit = (above < K) & (above + cnt[j] >= K)
        hit_b = hit.select(fx.Int32(BINS - 1 - j) - lane * 4, hit_b)
        hit_above = hit.select(above, hit_above)
        above = above + cnt[j]
    # one lane at most hits
    return (
        uniform(_wave_max(hit_b)),
        uniform(_wave_max(hit_above)),
        readlane(incl, 63) + above0,
    )


@traced
def _cut(c, counts, base, above0, out):
    """Wave 0: the key (``base`` + byte) of ``counts`` (256 LDS counts, the
    highest byte first) where the count from the top first reaches K, the
    count above it (from ``above0``), and the total -> broadcast words
    ``out`` .. + 2 (byte -1: the counts hold fewer than K)."""
    if c["wave"] == 0:
        lane, lds = c["lane"], c["lds"]
        cnt = [
            fx.Int32(fx.ptr_load(lds + (counts + (BINS - 1 - j) - lane * 4)))
            for j in range(4)
        ]
        b, above, total = _bin_of_kth(cnt, lane, above0)
        if lane == 0:
            fx.ptr_store((b >= 0).select(base + b, fx.Int32(-1)), lds + out)
            fx.ptr_store(above, lds + (out + 1))
            fx.ptr_store(total, lds + (out + 2))
    gpu.barrier()


@traced
def _count_lo(c, loads, hb):
    """The slice's keys of high byte ``hb`` by low byte (LDS counts)."""
    lds = c["lds"]
    junk = fx.Int32(W_JUNK) + c["lane"]

    def use(b, key, gid, ok):
        at = (ok & ((key >> 8) == hb)).select(fx.Int32(W_LO) + (key & 255), junk)
        fx.atomic_add(lds + at, fx.Int32(1), syncscope="workgroup")

    _for_values(c, loads, use)
    gpu.barrier()


@traced
def _gather(c, loads, cut, n_gt, top_b, cr):
    """The slice's candidates -- every key above ``cut``, then keys equal to
    it to K (``n_gt`` above it: fewer than K) -- into the slice's list (LDS
    slot counters), and its sum of exp2((b - max) log2(e) / T) (summed in a
    fixed order: a thread's values, its wave, the waves) -> the sum."""
    lds = c["lds"]
    lds64 = fx.recast_iter(fx.Int64, lds)
    lane = c["lane"]
    acc = [fx.Float32(0.0)]

    def use(b, key, gid, ok):
        v = hw_exp2((fx.Float32(b) - top_b) * cr)
        acc[0] = acc[0] + ok.select(v, fx.Float32(0.0))
        gt = ok & (key > cut)
        eq = ok & (key == cut)
        ctr = (gt | eq).select(
            gt.select(fx.Int32(W_CTR), fx.Int32(W_CTR + 1)), fx.Int32(W_JUNK) + lane
        )
        slot = fx.atomic_add(lds + ctr, fx.Int32(1), syncscope="workgroup")
        slot = gt.select(slot, n_gt + slot)
        keep = (gt | eq) & (slot < K)
        at = keep.select(fx.Int32(W_LIST // 2) + slot, fx.Int32(W_JUNK64 // 2) + lane)
        fx.ptr_store(i64_of(fx.Int32(0x7FFFFFFF) - gid, key), lds64 + at)

    _for_values(c, loads, use)
    s = _wave_sum(acc[0])
    if c["lane"] == 0:
        fx.ptr_store(s, c["ldsf"] + (W_RED + WAVES + c["wave"]))
    gpu.barrier()
    s = fx.ptr_load(c["ldsf"] + (W_RED + WAVES))
    for w in range_constexpr(1, WAVES):
        s = s + fx.ptr_load(c["ldsf"] + (W_RED + WAVES + w))
    return s


@traced
def _put_list(c, a, at, entry, head):
    """Wave 0: its list (``entry`` a lane) and ``head`` (four words) into
    record ``at`` of region ``a``."""
    lo, hi = i64_words(entry)
    c["mb"].put_words(a, at + 2 * c["lane"], [lo, hi])
    if c["lane"] == 0:
        c["mb"].put_words(a, at + HEAD, head[:2])
        c["mb"].put_words(a, at + HEAD + 2, head[2:])


@traced
def _await_logits(c, r, span):
    """The logits of row r in ``lg`` of the slice ``span``: the MTOP pairs
    of the tasks that wrote them, a thread a task (a task drains its logits
    before its pair, ``kernel.task_bests``)."""
    p = c["key"]
    first, end = span
    lo = first * 8 // p.task_rows
    hi = (end * 8 - 1) // p.task_rows
    c["mb"].poll([(c["mtop"], va.task_pair(r, fx.min(lo + c["tid"], hi), p.tasks), 2)])
    gpu.barrier()


@traced
def select_slice(c, r, span, cr):
    """This CTA's slice of row r, once every task's logits of it are in: its
    K largest, sorted, and header (largest key, sum, bound, count) into its
    CAND record."""
    lds = c["lds"]
    _await_logits(c, r, span)
    loads = _slice_loads(c, r, span)
    _clear(c)
    top = _count_hi(c, loads)
    _cut(c, W_HI, fx.Int32(0), fx.Int32(0), W_BC)
    hb = fx.Int32(fx.ptr_load(lds + W_BC))
    total = fx.Int32(fx.ptr_load(lds + (W_BC + 2)))
    if hb >= 0:
        _count_lo(c, loads, hb)
        _cut(c, W_LO, hb << 8, fx.Int32(fx.ptr_load(lds + (W_BC + 1))), W_BC + 4)
    # fewer than K in the slice: every key (the cut below them all)
    cut = (hb >= 0).select(fx.Int32(fx.ptr_load(lds + (W_BC + 4))), fx.Int32(-1))
    n_gt = (hb >= 0).select(fx.Int32(fx.ptr_load(lds + (W_BC + 5))), fx.Int32(0))
    s = _gather(c, loads, cut, n_gt, _value(top), cr)
    if c["wave"] == 0:
        lds64 = fx.recast_iter(fx.Int64, lds)
        entry = wave_sort(fx.ptr_load(lds64 + (W_LIST // 2 + c["lane"])), c["lane"])
        bound = (total > K).select(cut, fx.Int32(-1))
        head = [top, s.bitcast(fx.Int32), bound, fx.min(total, fx.Int32(K))]
        _put_list(c, c["cand"], c["t"] * REC, entry, head)


@traced
def _merge_tree(c, n, region, slot):
    """The K largest of ``n`` sorted lists (list i: record ``slot(i)`` of
    ``region``), a tree of ``wave_merge_top`` over the waves through LDS ->
    the merged list, a lane's entry (in every wave)."""
    lane, wave = c["lane"], c["wave"]
    lds64 = fx.recast_iter(fx.Int64, c["lds"])
    width = 1 << max(1, (n - 1).bit_length())  # lists, a power of two

    def at(i):
        return W_MERGE // 2 + fx.min(i, fx.Int32(WAVES - 1)) * K + lane

    # level 1: wave w merges lists 2w, 2w + 1 from their records
    if wave < width // 2:
        a = _poll_entry(c, region, slot, 2 * wave, n)
        b = _poll_entry(c, region, slot, 2 * wave + 1, n)
        fx.ptr_store(wave_merge_top(a, b, lane), lds64 + at(wave))
    gpu.barrier()
    level = width // 2
    while level > 1:
        # every wave merges (past the level's: a copy no one stores)
        a = fx.ptr_load(lds64 + at(2 * wave))
        b = fx.ptr_load(lds64 + at(2 * wave + 1))
        merged = wave_merge_top(a, b, lane)
        gpu.barrier()  # every wave's reads past
        if wave < level // 2:
            fx.ptr_store(merged, lds64 + at(wave))
        gpu.barrier()
        level //= 2
    return fx.ptr_load(lds64 + at(fx.Int32(0)))


def _poll_entry(c, region, slot, i, n):
    """This lane's entry of list i (record ``slot(i)`` of ``region``), empty
    past the ``n`` lists."""
    rec = slot(fx.min(i, fx.Int32(n - 1)))
    w = c["mb"].poll([(region, rec + 2 * c["lane"], 2)])[0]
    return (i < n).select(i64_of(w[0], w[1]), _empty())


@traced
def _rank_list(c, r, cr):
    """The lead: the row's helpers' lists merged into the rank's K largest,
    put with its header (largest key, sum rescaled to it, bound, count) into
    every rank's CANDX."""
    p = c["key"]
    n = p.helpers

    def slot(j):
        return (r + p.rows * j) * REC

    if const_expr(n == 1):
        entry = _poll_entry(c, c["cand"], slot, fx.Int32(0), n)
    else:
        entry = _merge_tree(c, n, c["cand"], slot)
    if c["wave"] == 0:
        lane = c["lane"]
        live = lane < n
        hw = c["mb"].poll(
            [
                (c["cand"], slot(fx.min(lane, fx.Int32(n - 1))) + HEAD + f, 2)
                for f in (0, 2)
            ]
        )
        top = live.select(hw[0][0], fx.Int32(-1))
        m = uniform(_wave_max(top))
        s = live.select(
            hw[0][1].bitcast(fx.Float32) * hw_exp2((_value(top) - _value(m)) * cr),
            fx.Float32(0.0),
        )
        s = readlane(_wave_sum(s), 0)
        bound = uniform(_wave_max(live.select(hw[1][0], fx.Int32(-1))))
        count = readlane(_wave_sum(live.select(hw[1][1], fx.Int32(0))), 0)
        # past K over the slices: the ids below the K-th may be missing
        kth = readlane(i64_words(entry)[1], K - 1)
        bound = (count > K).select(fx.max(bound, kth), bound)
        head = [m, s.bitcast(fx.Int32), bound, fx.min(count, fx.Int32(K))]
        at = (c["rank"] * p.max_rows + r) * REC
        for dst in range_constexpr(p.tp):
            _put_list(c, c["candx_fan"][dst], at, entry, head)


def _mass_above(key, mass, lane):
    """A lane's candidate (the wave's sorted, larger key first): the mass of
    the candidates with a larger key -- the masses summed up to the lane
    before its key's first lane (equal keys share it)."""
    first = (lane == 0) | (key != lane_gather(key, fx.max(lane - 1, fx.Int32(0))))
    start = wave_prefix_dpp(first.select(lane, fx.Int32(0)), fx.max)
    incl = wave_prefix_dpp(mass).bitcast(fx.Int32)
    before = lane_gather(incl, fx.max(start - 1, fx.Int32(0))).bitcast(fx.Float32)
    return (start > 0).select(before, fx.Float32(0.0))


@traced
def _decide(c, r, cr, top_p_v):
    """Every rank's lead alike: the ranks' lists merged; M, Z; the kept
    candidates; where the list decides the set (a candidate above every
    bound not kept), the race over the kept into ``ids`` -> whether it
    decided (uniform)."""
    p = c["key"]

    def slot(q):
        return (q * p.max_rows + r) * REC

    entry = _merge_tree(c, p.tp, c["candx"], slot)
    lds = c["lds"]
    if c["wave"] == 0:
        lane = c["lane"]
        live = lane < p.tp
        q = fx.min(lane, fx.Int32(p.tp - 1))
        hw = c["mb"].poll([(c["candx"], slot(q) + HEAD + f, 2) for f in (0, 2)])
        top = live.select(hw[0][0], fx.Int32(-1))
        m = uniform(_wave_max(top))
        mv = _value(m)
        z = live.select(
            hw[0][1].bitcast(fx.Float32) * hw_exp2((_value(top) - mv) * cr),
            fx.Float32(0.0),
        )
        z = readlane(_wave_sum(z), 0)
        bound = uniform(_wave_max(live.select(hw[1][0], fx.Int32(-1))))
        count = readlane(_wave_sum(live.select(hw[1][1], fx.Int32(0))), 0)
        lo, key = i64_words(entry)
        kth = readlane(key, K - 1)
        bound = (count > K).select(fx.max(bound, kth), bound)
        valid = key >= 0
        mass = valid.select(hw_exp2((_value(key) - mv) * cr), fx.Float32(0.0))
        kept = valid & (_mass_above(key, mass, lane) < top_p_v * z)
        decided = ballot(valid & ~kept & (key > bound)) != fx.Int64(0)
        if decided:
            gid = fx.Int32(0x7FFFFFFF) - lo
            u0, u1 = race.philox(
                (gid >> 1) | (r << race.counter_shift(p.vocab * p.tp)),
                c["draw_tag"] ^ race.seed_words(p.seed)[1], race.seed_words(p.seed)[0],
            )  # fmt: skip
            word = ((gid & 1) == 0).select(u0, u1)
            score = mass * hw_rcp(race.exp_var(word))
            g = i64_of(lo, order_key(score.bitcast(fx.Int32)) ^ fx.Int32(INT_MIN))
            win = _wave_max_i64(kept.select(g, _empty()))
            if lane == 0:
                ids = rsrc(c["args"]["ids"], p.rows * 4)
                bo.buffer_store(fx.Int32(0x7FFFFFFF) - i64_words(win)[0], ids, r)
        if lane == 0:
            fx.ptr_store(decided.select(fx.Int32(1), fx.Int32(0)), lds + W_BC)
    gpu.barrier()
    return fx.Int32(fx.ptr_load(lds + W_BC)) != 0


@traced
def draw(c, r, j, cr, top_p_v, span):
    """Helper j of row r: its slice's candidates; the lead (j 0) also the
    rank's and the decision, handed to the helpers (GO: 1 decided, 2 the
    scan) -> whether the row is decided (uniform over the CTA)."""
    select_slice(c, r, span, cr)
    c["stamp"]("select")
    if j == 0:
        _rank_list(c, r, cr)
        c["stamp"]("rank_list")
        decided = _decide(c, r, cr, top_p_v)
        if c["tid"] == 0:
            c["mb"].put(c["go"], r, decided.select(fx.Int32(1), fx.Int32(2)))
    # the lead's decision, read alike by the lead and its helpers
    decided = c["mb"].poll([(c["go"], r, 1)])[0][0] == 1
    c["stamp"]("decide")
    return decided
