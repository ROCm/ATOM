# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A top-p row's draw by the two-level scan, where its largest logits do not
decide its top-p set (``top_p_candidates``): one draw of softmax(b / T)
renormalized over the set ATOM's sampler keeps -- id v kept when the
probability of the ids strictly more likely than v is below top_p (equal
logits all kept or all dropped) -- b the bf16 logits over every rank's
vocab shard. DeepSeek-V4.1 hi's two-level scan (``hi/blocks/gfx950/head.py``,
``_sample_row``).

The row's ``helpers`` CTAs of every rank (``top_p_logits``: each a slice of
the row) take it once the row's largest logit M is known (the ranks' greedy
bests, MRANK: the same on every rank); a helper hands its slice's level-1
and level-2 records to the lead (this rank's L1 / L2), and the lead hands
B1 to the helpers (B1):

- masses m_v = round(exp2((b_v - M) log2(e) / T) 2^FIX), integers: every sum
  of them is exact in any order and the same on every rank (``fix_log2``: a
  row's sum stays below 2^53);
- level 1: the rank's masses by key16 >> 8 (key16 the bf16's 16-bit order
  key) into every rank's H1; the ranks' summed: Z, and B1 -- the bin whose
  keys hold the top-p set's edge (the mass of the bins above it < top_p Z <=
  that plus its own);
- level 2: the rank's masses and best (score, id) of B1's 256 keys, and its
  best above B1, into every rank's H2; the ranks' summed: a key of B1 is kept
  when the mass above B1 plus that of B1's keys above it is below top_p Z,
  and the id is the best (score, id) over the kept keys and the bests above
  B1 (equal scores: the lower id).

The score is the exponential race m_v / E_v (E_v = -log2 U_v, U_v the id's
Philox word as the temperature race's: ``race``): its argmax draws v with
probability m_v / sum m over the candidates, the same id on every rank.
The masses are integers and the bests a max, so the split changes no id.
The context is the kernel's, with ``h1`` / ``h2`` (this rank's regions, the
launch's copy), ``h1_fan`` / ``h2_fan`` (every rank's) and ``l1`` / ``l2``
/ ``b1`` (this rank's).
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

from atom.mono.device import race
from atom.mono.device.ops import (
    fresh,
    hw_exp2,
    hw_rcp,
    readlane,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from atom.mono.device.select import INT_MIN, key_bits, order_key
from atom.mono.lm_head import argmax as va
from atom.mono.lm_head import top_p_logits as tl
from atom.mono.plan.execution import THREADS

BINS = 256
# level 1 counts into copies (lane l: copy l % COPIES): a few bins (sign and
# exponent) take most ids
COPIES = 16
# LDS words of a row's draw (the launch's LDS, free once the row's rank best
# is put): broadcast words (B1, its pad, the mass above B1 f64, top_p Z f64),
# level 1's 256 u64 masses, level 2's 256 u64 masses and 256 i64 bests of
# B1's keys, the best above B1, the level-1 copies, a junk word a lane (the
# zero masses' atomics); the gathers of the ranks' records reuse the copies
S_BC = 0
S_H1 = 8
S_H2 = S_H1 + 2 * BINS
S_G = S_H2 + 2 * BINS
S_GA = S_G + 2 * BINS
S_H1C = S_GA + 4
S_JUNK = S_H1C + COPIES * 2 * BINS
WORDS = S_JUNK + 2 * 64
S_T = S_H1C
# pairs a row's record: H1 a (mass lo, hi) a bin; H2 a (mass lo, hi, best
# lo, hi) a key of B1, then the best above B1 and a pad pair
H1_ROW = 2 * BINS
H2_ROW = 4 * BINS + 4
H2_ABOVE = 4 * BINS
EMPTY_HI = INT_MIN  # an empty best: (lo 0, hi INT_MIN), below every score


def fix_log2(vocab: int) -> int:
    """The masses' fixed point (2^this a mass of 1): a row's ``vocab``
    masses, each at most 1, sum below 2^53, so f64 sums of them are exact."""
    return 53 - (vocab - 1).bit_length()


def region_pairs(max_rows: int, tp: int) -> tuple[int, int]:
    """One copy's pairs of H1 and of H2, [source rank][row] records."""
    return tp * max_rows * H1_ROW, tp * max_rows * H2_ROW


def local_pairs(tasks: int, max_rows: int) -> tuple[int, int, int]:
    """Pairs of this rank's L1 and L2 (a helper's records, [CTA]) and B1 (a
    row's, [row]): a launch's own, read on this rank only."""
    return tasks * H1_ROW, tasks * H2_ROW, max_rows


def _i64(lo, hi):
    return fx.Vector.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32).bitcast(
        fx.Int64
    )[0]


def _words(v):
    w = fx.Vector.from_elements([fx.Int64(v)], fx.Int64).bitcast(fx.Int32)
    return w[0], w[1]


def _f64_words(x):
    w = fx.Vector.from_elements([fx.Float64(x)], fx.Float64).bitcast(fx.Int32)
    return w[0], w[1]


def _f64_of(lo, hi):
    return fx.Vector.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32).bitcast(
        fx.Float64
    )[0]


def _wave_prefix_f64(x):
    """Inclusive prefix sum of f64 ``x`` over the wave's lanes (lane 0 first):
    the DPP row shifts and row broadcasts on both words (shifted-in zeros are
    +0.0). Exact while every partial sum is an integer below 2^53."""
    for ctrl, rmask in (
        (0x111, 0xF), (0x112, 0xF), (0x114, 0xF), (0x118, 0xF),
        (0x142, 0xA), (0x143, 0xC),
    ):  # fmt: skip
        lo, hi = _f64_words(x)
        slo = fx.Int32(update_dpp_i32(fx.Int32(0), lo, ctrl, rmask, 0xF, False))
        shi = fx.Int32(update_dpp_i32(fx.Int32(0), hi, ctrl, rmask, 0xF, False))
        x = x + _f64_of(slo, shi)
    return x


def _readlane_f64(x, lane):
    lo, hi = _f64_words(x)
    return _f64_of(readlane(lo, lane), readlane(hi, lane))


def _wave_max_i32(v):
    for off in (32, 16, 8, 4, 2, 1):
        v = fx.max(v, xshfl(v, off))
    return v


def _wave_max_i64(v):
    for off in (32, 16, 8, 4, 2, 1):
        lo, hi = _words(v)
        v = fx.max(v, _i64(xshfl(lo, off), xshfl(hi, off)))
    return v


def _mass(b, m, c, fix):
    """(the fixed-point mass of bf16 logit ``b`` as Int64, its f32 value):
    exp2((b - M) c + fix), c = log2(e) / T, to the nearest integer (the f32's
    high part v / 2^32, then the exact rest)."""
    from_m = fx.Float32(b) - fx.Float32(m)
    v = hw_exp2(fmath.fma(from_m, fx.Float32(c), fx.Float32(float(fix))))
    hi = (v * fx.Float32(2.0**-32)).to(fx.Int32)
    rest = v - hi.to(fx.Float32) * fx.Float32(2.0**32) + fx.Float32(0.5)
    return _i64(rest.to(fx.Uint32), hi), v


def _score(v, word, gid):
    """The (score, id) of a candidate as one Int64 (larger: better; equal
    scores: the lower id): score = v / E, E = -log2 U of the id's Philox word
    (``race.exp_var``)."""
    score = v * hw_rcp(race.exp_var(word))
    return _i64(
        fx.Int32(0x7FFFFFFF) - gid,
        order_key(score.bitcast(fx.Int32)) ^ fx.Int32(INT_MIN),
    )


@traced
def _clear(c):
    """Level 2's masses to 0 and its bests to empty, the level-1 copies to 0."""
    lds64, tid = fx.recast_iter(fx.Int64, c["lds"]), c["tid"]
    for i in range_constexpr(-(-(2 * BINS + 1) // THREADS)):
        e = fx.min(fx.Int32(i * THREADS) + tid, fx.Int32(2 * BINS))
        fx.ptr_store(
            (e < BINS).select(fx.Int64(0), _i64(0, EMPTY_HI)), lds64 + (S_H2 // 2 + e)
        )
    for i in range_constexpr(COPIES * BINS // THREADS):
        fx.ptr_store(fx.Int64(0), lds64 + (S_H1C // 2 + i * THREADS + tid))
    gpu.barrier()


@traced
def _level1_scan(c, r, m, cr, span):
    """This CTA's slice ``span`` of row r: its masses by bin (key16 >> 8)
    into the lane's copy (a zero mass, or a quad past the slice: the lane's
    junk word), then the copies summed (integers: any order) into ``S_H1``."""
    p, lane, tid = c["key"], c["lane"], c["tid"]
    lds64 = fx.recast_iter(fx.Int64, c["lds"])
    copy = S_H1C // 2 + (lane % COPIES) * BINS
    junk = fx.Int32(S_JUNK // 2) + lane
    for bi in range_constexpr(tl.batches(p)):
        quads = tl.slice_batch(c, r, bi, span)
        for qi in range_constexpr(len(quads)):
            w, _q, ok = quads[qi]
            for wi in range_constexpr(4):
                for h in range_constexpr(2):
                    b, key = tl.key16(w[wi], h == 1)
                    q, _v = _mass(b, m, cr, p.fix_log2)
                    live = (q != fx.Int64(0)) & ok
                    at = live.select(copy + (key >> 8), junk)
                    fx.atomic_add(lds64 + at, q, syncscope="workgroup")
    gpu.barrier()
    if fresh(tid) < BINS:
        s = fx.Int64(0)
        for i in range_constexpr(COPIES):
            s = s + fx.ptr_load(lds64 + (S_H1C // 2 + i * BINS + tid))
        fx.ptr_store(s, lds64 + (S_H1 // 2 + tid))
    gpu.barrier()


@traced
def _put_l1(c):
    """A helper, threads 0 .. 255: its slice's bin ``tid`` (``S_H1``) into
    this rank's L1 record of its CTA, for the row's lead."""
    tid = c["tid"]
    if fresh(tid) < BINS:
        lds64 = fx.recast_iter(fx.Int64, c["lds"])
        lo, hi = _words(fx.ptr_load(lds64 + (S_H1 // 2 + tid)))
        c["mb"].put_words(c["l1"], c["t"] * H1_ROW + tid * 2, [lo, hi])


@traced
def _combine_l1(c, r):
    """The lead, threads 0 .. 255: bin ``tid`` of the row's helpers' L1
    records added to its own (``S_H1``)."""
    p, tid = c["key"], c["tid"]
    if const_expr(p.helpers > 1):
        if fresh(tid) < BINS:
            lds64 = fx.recast_iter(fx.Int64, c["lds"])
            specs = [
                (c["l1"], (r + p.rows * j) * H1_ROW + tid * 2, 2)
                for j in range(1, p.helpers)
            ]
            got = c["mb"].poll(specs)
            s = fx.ptr_load(lds64 + (S_H1 // 2 + tid))
            for w in got:
                s = s + _i64(w[0], w[1])
            fx.ptr_store(s, lds64 + (S_H1 // 2 + tid))
        gpu.barrier()


@traced
def _put_h1(c, r):
    """The lead, threads 0 .. 255: the row's bin ``tid`` (``S_H1``) into
    every rank's H1 record (this rank, row r)."""
    p, tid = c["key"], c["tid"]
    if fresh(tid) < BINS:
        lds64 = fx.recast_iter(fx.Int64, c["lds"])
        lo, hi = _words(fx.ptr_load(lds64 + (S_H1 // 2 + tid)))
        at = (c["rank"] * p.max_rows + r) * H1_ROW + tid * 2
        for dst in range_constexpr(p.tp):
            c["mb"].put_words(c["h1_fan"][dst], at, [lo, hi])


@traced
def _gather(c, region, r, row_pairs, fields):
    """Every thread: entry e = tid % 256 of the ranks' row-r records in this
    rank's ``region`` (``fields`` u64 at pairs 2 fields e: 1 a mass; 2 a mass
    and a best), over the ranks of half tid // 256 (rank 2 j + half): the
    masses summed, the bests' max, into LDS ``S_T`` (masses [2][256] i64,
    then bests [2][256])."""
    p, tid = c["key"], c["tid"]
    half, e = tid // BINS, tid % BINS
    n = -(-p.tp // 2)
    specs, oks = [], []
    for j in range_constexpr(n):
        src = fx.min(fx.Int32(2 * j) + half, fx.Int32(p.tp - 1))
        oks.append(fx.Int32(2 * j) + half < p.tp)
        at = (src * p.max_rows + r) * row_pairs + e * 2 * fields
        for f in range_constexpr(fields):
            specs.append((region, at + f * 2, 2))
    got = c["mb"].poll(specs)
    s, g = fx.Int64(0), _i64(0, EMPTY_HI)
    for j in range_constexpr(n):
        w = got[j * fields]
        s = s + oks[j].select(_i64(w[0], w[1]), fx.Int64(0))
        if const_expr(fields == 2):
            w = got[j * fields + 1]
            g = fx.max(g, oks[j].select(_i64(w[0], w[1]), _i64(0, EMPTY_HI)))
    lds64 = fx.recast_iter(fx.Int64, c["lds"])
    fx.ptr_store(s, lds64 + (S_T // 2 + tid))
    if const_expr(fields == 2):
        fx.ptr_store(g, lds64 + (S_T // 2 + 2 * BINS + tid))


@traced
def _level1(c, r, top_p):
    """Wave 0: the ranks' level-1 masses summed (lane l: bins 255 - 4 l - j,
    j = 0 .. 3), Z, and B1 -> broadcast words: B1, the mass above B1, top_p
    Z."""
    _gather(c, c["h1"], r, H1_ROW, 1)
    gpu.barrier()
    if c["wave"] == 0:
        lane = c["lane"]
        lds64 = fx.recast_iter(fx.Int64, c["lds"])
        cnt = []
        for j in range_constexpr(4):
            b = fx.Int32(BINS - 1 - j) - lane * 4
            both = fx.ptr_load(lds64 + (S_T // 2 + b)) + fx.ptr_load(
                lds64 + (S_T // 2 + BINS + b)
            )
            cnt.append(fx.Int64(both).to(fx.Float64))
        tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
        incl = _wave_prefix_f64(tot)
        pz = top_p.to(fx.Float64) * _readlane_f64(incl, 63)
        above = incl - tot
        hit_b, s_hit = fx.Int32(-1), fx.Float64(0.0)
        for j in range_constexpr(4):
            hit = (above < pz) & (above + cnt[j] >= pz)
            hit_b = hit.select(fx.Int32(BINS - 1 - j) - lane * 4, hit_b)
            s_hit = hit.select(above, s_hit)
            above = above + cnt[j]
        b1 = uniform(_wave_max_i32(hit_b))
        who = uniform(_wave_max_i32((hit_b >= 0).select(lane, fx.Int32(-1))))
        s_b1 = _readlane_f64(s_hit, fx.max(who, fx.Int32(0)))
        if lane == 0:
            slo, shi = _f64_words(s_b1)
            plo, phi = _f64_words(pz)
            fx.ptr_store(
                fx.Vector.from_elements([b1, b1, slo, shi, plo, phi], fx.Int32),
                c["lds"] + (S_BC + 2),
            )
    gpu.barrier()


@traced
def _candidates(c, w, gid0, b1, m, cr, r, seeds, best):
    """Word ``w``'s two ids (gid0, gid0 + 1), when either is a candidate (its
    bin >= B1): an id above B1 into ``best`` (returned), one in B1 into its
    key's mass and best (LDS atomics; the other id's to the lane's junk
    word)."""
    p = c["key"]
    b_lo, key_lo = tl.key16(w, False)
    b_hi, key_hi = tl.key16(w, True)
    if ((key_lo >> 8) >= b1) | ((key_hi >> 8) >= b1):
        lds64 = fx.recast_iter(fx.Int64, c["lds"])
        gid0 = fresh(gid0)  # computed here, not hoisted and held across the scan
        junk = fx.Int32(S_JUNK // 2) + (fresh(c["tid"]) & 63)
        u0, u1 = race.philox(
            (gid0 >> 1) | (r << race.counter_shift(p.vocab * p.tp)),
            c["draw_tag"] ^ seeds[1], seeds[0],
        )  # fmt: skip
        for h, (b, key, u) in enumerate(((b_lo, key_lo, u0), (b_hi, key_hi, u1))):
            q, v = _mass(b, m, cr, p.fix_log2)
            g = _score(v, u, gid0 + h)
            bin_ = key >> 8
            best = fx.max(best, (bin_ > b1).select(g, _i64(0, EMPTY_HI)))
            inb = bin_ == b1
            at = inb.select(fx.Int32(S_H2 // 2) + (key & 255), junk)
            fx.atomic_add(lds64 + at, inb.select(q, fx.Int64(0)), syncscope="workgroup")
            at = inb.select(fx.Int32(S_G // 2) + (key & 255), junk)
            fx.atomic_max(
                lds64 + at, inb.select(g, _i64(0, EMPTY_HI)), syncscope="workgroup"
            )
    return best


@traced
def _level2_scan(c, r, m, cr, b1, span):
    """This CTA's candidates of its slice ``span`` of row r
    (``_candidates``), the best above B1 into ``S_GA``."""
    p = c["key"]
    seeds = race.seed_words(p.seed)
    best = _i64(0, EMPTY_HI)
    for bi in range_constexpr(tl.batches(p)):
        quads = tl.slice_batch(c, r, bi, span)
        for qi in range_constexpr(len(quads)):
            w, q, ok = quads[qi]
            # a quad past the slice (clamped) holds no candidate
            b1q = ok.select(b1, fx.Int32(1 << 20))
            for wi in range_constexpr(4):
                gid0 = c["first"] + q * 8 + wi * 2
                best = _candidates(c, w[wi], gid0, b1q, m, cr, r, seeds, best)
    fx.atomic_max(
        fx.recast_iter(fx.Int64, c["lds"]) + S_GA // 2, best, syncscope="workgroup"
    )
    gpu.barrier()


def _record_words(c, e, last):
    """Pair words of an H2 / L2 record's entry: thread ``last`` the best
    above B1 (``S_GA``) and a pad, else key e's (mass, best)."""
    lds64 = fx.recast_iter(fx.Int64, c["lds"])
    mlo, mhi = _words(fx.ptr_load(lds64 + (S_H2 // 2 + e)))
    glo, ghi = _words(fx.ptr_load(lds64 + (S_G // 2 + e)))
    alo, ahi = _words(fx.ptr_load(lds64 + S_GA // 2))
    zero = fx.Int32(0)
    return (
        [last.select(alo, mlo), last.select(ahi, mhi)],
        [last.select(zero, glo), last.select(zero, ghi)],
    )


@traced
def _put_l2(c):
    """A helper, threads 0 .. 256: its slice's level-2 record (key ``tid``
    of B1's (mass, best); thread 256 the best above B1) into this rank's L2
    record of its CTA, for the row's lead."""
    tid = c["tid"]
    if fresh(tid) < BINS + 1:
        e = fx.min(tid, fx.Int32(BINS - 1))
        last = tid == BINS
        at = c["t"] * H2_ROW + last.select(fx.Int32(H2_ABOVE), e * 4)
        mass, best = _record_words(c, e, last)
        c["mb"].put_words(c["l2"], at, mass)
        c["mb"].put_words(c["l2"], at + 2, best)


@traced
def _combine_l2(c, r):
    """The lead, threads 0 .. 256: the row's helpers' L2 records merged into
    its own (masses added, bests' max): key ``tid`` of B1, thread 256 the
    best above B1."""
    p, tid = c["key"], c["tid"]
    if const_expr(p.helpers > 1):
        if fresh(tid) < BINS + 1:
            lds64 = fx.recast_iter(fx.Int64, c["lds"])
            e = fx.min(tid, fx.Int32(BINS - 1))
            last = tid == BINS
            off = last.select(fx.Int32(H2_ABOVE), e * 4)
            ats = [(r + p.rows * j) * H2_ROW + off for j in range(1, p.helpers)]
            specs = [(c["l2"], at + f, 2) for at in ats for f in (0, 2)]
            got = c["mb"].poll(specs)
            # thread 256's "mass" pair is the best above B1 (``_record_words``)
            best_at = last.select(fx.Int32(S_GA // 2), S_G // 2 + e)
            mass = fx.ptr_load(lds64 + (S_H2 // 2 + e))
            best = fx.ptr_load(lds64 + best_at)
            for j in range_constexpr(p.helpers - 1):
                wm, wb = got[2 * j], got[2 * j + 1]
                mass = mass + _i64(wm[0], wm[1])
                best = fx.max(best, last.select(_i64(wm[0], wm[1]), _i64(wb[0], wb[1])))
            if tid < BINS:
                fx.ptr_store(mass, lds64 + (S_H2 // 2 + e))
            fx.ptr_store(best, lds64 + best_at)
        gpu.barrier()


@traced
def _put_h2(c, r):
    """The lead, threads 0 .. 256: the row's level-2 record (key ``tid`` of
    B1's (mass, best); thread 256 the best above B1) into every rank's H2
    record (this rank, row r)."""
    p, tid = c["key"], c["tid"]
    if fresh(tid) < BINS + 1:
        e = fx.min(tid, fx.Int32(BINS - 1))
        last = tid == BINS
        at = (c["rank"] * p.max_rows + r) * H2_ROW + last.select(
            fx.Int32(H2_ABOVE), e * 4
        )
        mass, best = _record_words(c, e, last)
        for dst in range_constexpr(p.tp):
            c["mb"].put_words(c["h2_fan"][dst], at, mass)
            c["mb"].put_words(c["h2_fan"][dst], at + 2, best)


@traced
def _level2(c, r):
    """Wave 0: the ranks' level-2 records summed (lane l: keys 255 - 4 l - j
    of B1); a key is kept when the mass above B1 plus that of B1's keys above
    it is below top_p Z; the id = the best (score, id) over the kept keys'
    bests and the ranks' bests above B1 -> ``ids``."""
    p = c["key"]
    _gather(c, c["h2"], r, H2_ROW, 2)
    gpu.barrier()
    if c["wave"] == 0:
        lane = c["lane"]
        lds, lds64 = c["lds"], fx.recast_iter(fx.Int64, c["lds"])
        s_b1 = _f64_of(fx.ptr_load(lds + (S_BC + 4)), fx.ptr_load(lds + (S_BC + 5)))
        pz = _f64_of(fx.ptr_load(lds + (S_BC + 6)), fx.ptr_load(lds + (S_BC + 7)))
        # the ranks' bests above B1: lane q < tp polls rank q's
        src = fx.min(lane, fx.Int32(p.tp - 1))
        at = (src * p.max_rows + r) * H2_ROW + H2_ABOVE
        wa = c["mb"].poll([(c["h2"], at, 2)])[0]
        win = (lane < p.tp).select(_i64(wa[0], wa[1]), _i64(0, EMPTY_HI))
        cnt, bst = [], []
        for j in range_constexpr(4):
            f = fx.Int32(BINS - 1 - j) - lane * 4
            both = fx.ptr_load(lds64 + (S_T // 2 + f)) + fx.ptr_load(
                lds64 + (S_T // 2 + BINS + f)
            )
            cnt.append(fx.Int64(both).to(fx.Float64))
            bst.append(
                fx.max(
                    fx.ptr_load(lds64 + (S_T // 2 + 2 * BINS + f)),
                    fx.ptr_load(lds64 + (S_T // 2 + 3 * BINS + f)),
                )
            )
        tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
        above = s_b1 + (_wave_prefix_f64(tot) - tot)
        for j in range_constexpr(4):
            keep = above < pz
            win = fx.max(win, keep.select(bst[j], _i64(0, EMPTY_HI)))
            above = above + cnt[j]
        lo, _hi = _words(_wave_max_i64(win))
        if lane == 0:
            ids = rsrc(c["args"]["ids"], p.rows * 4)
            bo.buffer_store(fx.Int32(0x7FFFFFFF) - lo, ids, r)


@traced
def _row_max(c, r):
    """Row r's largest bf16 logit over every rank (the ranks' greedy bests in
    MRANK, reduced alike in every thread), as f32."""
    p = c["key"]
    got = c["mb"].poll(
        [(c["mrank"], va.rank_pair(src, r, p.max_rows), 2) for src in range(p.tp)]
    )
    kk, ii = got[0][0], got[0][1]
    for g in got[1:]:
        kk, ii = va.take(kk, ii, g[0], g[1])
    return key_bits(kk).bitcast(fx.Float32)


@traced
def _lead(c, r, m, cr, top_p, span):
    """The row's lead (helper 0): its slice's level 1 with the helpers',
    the ranks' -> B1, broadcast to the helpers; its slice's level 2 with
    the helpers', the ranks' -> the row's id."""
    _level1_scan(c, r, m, cr, span)
    _combine_l1(c, r)
    _put_h1(c, r)
    _level1(c, r, top_p)
    b1 = fx.Int32(fx.ptr_load(c["lds"] + (S_BC + 2)))
    if c["tid"] == 0:
        c["mb"].put(c["b1"], r, b1)
    _level2_scan(c, r, m, cr, b1, span)
    _combine_l2(c, r)
    _put_h2(c, r)
    _level2(c, r)


@traced
def _helper(c, r, m, cr, span):
    """A helper of the row: its slice's level 1 to the lead, B1 from it, its
    slice's level 2 to the lead."""
    _level1_scan(c, r, m, cr, span)
    _put_l1(c)
    b1 = c["mb"].poll([(c["b1"], r, 1)])[0][0]
    _level2_scan(c, r, m, cr, b1, span)
    _put_l2(c)


@traced
def scan_draw(c, r, j, cr, top_p, span):
    """Helper j of row r (j 0 its lead) by the two-level scan, over its
    slice ``span``: the lead puts the row's id into ``ids``."""
    gpu.barrier()  # the LDS's earlier readers past it
    m = _row_max(c, r)
    _clear(c)
    if j == 0:
        _lead(c, r, m, cr, top_p, span)
    else:
        _helper(c, r, m, cr, span)
