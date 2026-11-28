# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A CTA's exact top-k of up to ``LDS_KEYS`` fp32 values, the answer of aiter's
``top_k_per_row_decode(stable=True)``: ranked by the fp32 bits' u32 order key
(-0 below +0, NaN above +inf), ties to the lowest index, emitted ascending by
index, -1 padded. P0.7 pinned the key and the tie rule against aiter;
``lds_topk_timing.py CHECK=1`` (skill mono-indexer) pins this routine against
that rule on ties, +-0, NaN and +-inf at every edge n.

Thread i holds ``per_thread`` consecutive indices from i x per_thread, as
keys in LDS, so thread order is index order and one block scan places the
winners. Four 8-bit radix passes find the k-th key; integer counts, so the
answer does not depend on arrival order.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.mono.device.ops import (
    ballot,
    block_excl_scan,
    fresh,
    ld_i32,
    st_dev,
    traced,
    uniform,
    wave_incl_scan,
    wave_max,
    wave_prefix_dpp,
    wave_umax_dpp,
)
from atom.mono.device.select import order_key, u32_lt
from atom.mono.plan.execution import THREADS, WAVES

PER_THREAD = 64  # the largest a CTA's LDS takes alone: 32768 keys
LDS_KEYS = THREADS * PER_THREAD


# a histogram: 256 bins, then a dummy bin a lane: a key outside the pass's
# prefix counts there, so the adds need no branch and no two lanes share one
HIST_WORDS = 256 + 64


def lds_words(per_thread: int = PER_THREAD) -> dict[str, int]:
    """The LDS the routine needs, in 32-bit words."""
    return {"keys": THREADS * per_thread, "hist": 3 * HIST_WORDS, "buf": WAVES}


@traced
def lds_topk(n, k, load, emit, lds, tid, per_thread=PER_THREAD):
    """Top ``k`` of ``n`` <= THREADS x per_thread values, given a quad at a
    time: ``load(q, i)`` -> the 4 values (fp32 bits as Int32) at indices i ..
    i + 3, for i = per_thread tid + 4 q clamped to the quad holding index n - 1
    (values past n are read but never counted) -> ``emit(pos, d, i)`` by the
    thread holding each winner (index i = per_thread tid + d), pos its place
    ascending, and ``emit(pos, None, -1)`` for pos in [min(n, k), k). The keys
    stay in ``lds["keys"]`` (``key_bits`` of one is its value); ``lds["hist"]``
    is free once it returns. Barriers included; every thread of the CTA calls
    it."""
    lane, wave = tid % 64, tid // 64
    keys, hist, buf = lds["keys"], lds["hist"], lds["buf"]
    base = tid * per_thread
    # quads past the last value's re-read it: in the row, never counted
    last = fx.max(n - 1, 0) & ~3
    kv = []
    for q in range_constexpr(per_thread // 4):
        quad = load(q, fx.min(base + 4 * q, last))
        for e in range_constexpr(4):
            kv.append(order_key(quad[e]))
            fx.ptr_store(kv[-1], keys + (base + 4 * q + e))
    for r in range_constexpr((2 * HIST_WORDS + THREADS - 1) // THREADS):
        if tid + THREADS * r < 2 * HIST_WORDS:  # histograms 0 and 1
            fx.ptr_store(Int32(0), hist + (tid + THREADS * r))
    gpu.barrier()
    prefix = Int32(0)
    mask = Int32(0)
    need = Int32(k)
    n = fx.max(n, 0)
    # n <= k takes every key: no k-th to find
    if n > k:
        # three histograms in rotation: pass p counts into p % 3 and clears
        # (p + 2) % 3, which every wave finished reading before this pass's barrier
        for p in range_constexpr(4):
            shift = 24 - 8 * p
            h = hist + (p % 3) * HIST_WORDS
            dummy = 256 + lane
            if const_expr(p == 0):
                # a thread's keys are consecutive and logits share their top byte:
                # a run of one bin goes in as one add at its end (the others add 0
                # to the lane's own dummy bin), not an add a key on one hot bin
                run_bin, run = Int32(dummy), Int32(0)
                for d in range_constexpr(per_thread):
                    bin_ = (base + d < n).select((kv[d] >> shift) & 255, dummy)
                    same = bin_ == run_bin
                    fx.llvm.atomic_add(
                        h + same.select(dummy, run_bin), same.select(0, run),
                        syncscope="workgroup",
                    )  # fmt: skip
                    run = same.select(run + 1, 1)
                    run_bin = bin_
                fx.llvm.atomic_add(h + run_bin, run, syncscope="workgroup")
            else:
                for d in range_constexpr(per_thread):
                    key = kv[d]
                    counted = (base + d < n) & ((key & mask) == prefix)
                    bin_ = counted.select((key >> shift) & 255, dummy)
                    fx.llvm.atomic_add(h + bin_, Int32(1), syncscope="workgroup")
            gpu.barrier()
            if (p + 2 < 4) & (tid < HIST_WORDS):
                fx.ptr_store(Int32(0), hist + ((p + 2) % 3) * HIST_WORDS + tid)
            # lane l: bins 255 - 4 l - q, highest first
            cs = [fx.ptr_load(h + (255 - 4 * lane - q)) for q in range(4)]
            s4 = cs[0] + cs[1] + cs[2] + cs[3]
            before = wave_incl_scan(s4, lane) - s4
            found = Int32(-1)
            for q in range_constexpr(4):
                hit = (before < need) & (need <= before + cs[q])
                found = hit.select(
                    ((255 - 4 * lane - q) << 16) | (need - before), found
                )
                before = before + cs[q]
            found = wave_max(found)
            prefix = prefix | (((found >> 16) & 255) << shift)
            mask = mask | Int32(
                (255 << shift) - (1 << 32) if shift == 24 else 255 << shift
            )
            need = found & 0xFFFF
    take_all = n <= k
    # winners: keys above the k-th, then its ties lowest index first; one scan of
    # (above, ties) packed in 16-bit halves gives both prefixes
    gts = [(base + d < n) & u32_lt(prefix, kv[d]) for d in range(per_thread)]
    eqs = [(base + d < n) & (kv[d] == prefix) for d in range(per_thread)]
    above, ties = Int32(0), Int32(0)
    for d in range_constexpr(per_thread):
        above = above + gts[d].select(1, 0)
        ties = ties + eqs[d].select(1, 0)
    packed, _ = block_excl_scan(above | (ties << 16), lane, wave, buf)
    ab = packed & 0xFFFF
    tb = packed >> 16
    for d in range_constexpr(per_thread):
        i = base + d
        gt, tie = gts[d], eqs[d]
        win = (i < n) & (take_all | gt | (tie & (tb < need)))
        at = take_all.select(i, ab + fx.min(tb, need))
        if win:
            emit(at, d, i)
        ab = ab + gt.select(1, 0)
        tb = tb + tie.select(1, 0)
    n_out = fx.min(n, k)
    for r in range_constexpr((k + THREADS - 1) // THREADS):
        pos = tid + THREADS * r
        if (pos >= n_out) & (pos < k):
            emit(pos, None, Int32(-1))
    gpu.barrier()


# ---------------------------------------------------------------- radix select
# The same answer as ``lds_topk``, in order: keys in registers only, 8-bit
# passes over the u32 order key that
# end at the first pass whose digit's keys are all taken (the passes below
# decide nothing), the winners into slots by one CTA prefix sum in place order.
BINS = 256
COPIES = 8  # pass 0's histograms: lane l adds into copy l % COPIES
H0_STRIDE = BINS + 4  # the copies 4 words apart in bank
PASSES = 4
# The anchored pass (``radix_select``'s ``anchor``, ``index_plan.ANCHOR``): a
# key's top 16 bits (passes 0 and 1's digits together) placed in a window of
# WINDOW values around the anchor's -- bin 1 + (top - lo16), 255 above it, a
# key below it uncounted. Its k-th in a window bin stands for passes 0 and 1.
WINDOW = BINS - 2
# passes 2 and 3 issue a key's add only where some lane of the wave counts (a
# scalar branch on the ballot): past the top 16 bits few keys still count
RADIX_SPARSE = True


def _layout(k: int) -> dict[str, int]:
    """Word offsets of ``radix_select``'s LDS for top ``k``."""
    hist = COPIES * H0_STRIDE  # passes 1 ..: a histogram each
    win = hist + (PASSES - 1) * BINS  # the anchored pass's (``ip.ANCHOR``)
    cnt = win + (BINS if ip.ANCHOR else 0)  # the waves' prefix-sum totals
    slots = cnt + WAVES  # a winner a slot: (place, key)
    junk = slots + 2 * k  # a word a lane: the uncounted keys' adds
    return {
        "hist": hist, "win": win, "cnt": cnt, "slots": slots, "junk": junk,
        "words": junk + 64,
    }  # fmt: skip


def select_words(k: int) -> int:
    """The LDS words ``radix_select`` needs for top ``k``."""
    return _layout(k)["words"]


def _lds4(ptr, off):
    v = fx.Vector(fx.ptr_load(ptr + off, result_type=fx.Vector.make_type(4, Int32)))
    return [v[e] for e in range(4)]


def _count(
    lds, hoff, keys, base, n, lane, prefix, mask, shift, copies, junk, sparse=False
):
    """Count the keys at places base .. below n matching ``prefix`` under the
    static ``mask`` by their digit at ``shift`` (``copies`` > 1: lane l's
    copy), then the barrier. A key that does not count adds to its lane's junk
    word: no branch a key, no two lanes on one word. ``sparse``: a key's add
    only where some lane of the wave counts (``_add_any``)."""
    if const_expr(copies > 1):
        hoff = hoff + (lane % copies) * H0_STRIDE
    for e in range_constexpr(len(keys)):
        u = fresh(keys[e])  # the masks rebuilt a pass, not held across passes
        m = (base + e) < n
        if const_expr(mask != 0):
            m = m & ((u & Int32(_signed(mask))) == prefix)
        at = m.select(hoff + ((u >> shift) & 255), junk + lane)
        if const_expr(sparse):
            _add_any(m, lds + at)
        else:
            fx.llvm.atomic_add(lds + at, Int32(1), syncscope="workgroup")
    gpu.barrier()


@traced
def _add_any(m, at):
    """Every lane's add at ``at`` (a junk word where ``m`` fails), issued only
    when some lane's ``m`` holds: the ballot is wave-uniform, so a scalar
    branch skips the instruction (an exec mask alone still issues it)."""
    if ballot(m) != fx.Int64(0):
        fx.llvm.atomic_add(at, Int32(1), syncscope="workgroup")


def _count_window(lds, woff, keys, base, n, lane, lo16, junk):
    """The anchored pass: count the keys at places base .. below n by their
    top 16 bits' place in the window from ``lo16`` -- bin 1 + (top - lo16)
    within it, bin 255 above it; a key below it adds to its lane's junk word
    --, then the barrier. Monotone in the key: a bin above holds only greater
    keys."""
    for e in range_constexpr(len(keys)):
        u = fresh(keys[e])
        q = ((u >> 16) & 0xFFFF) - lo16
        m = ((base + e) < n) & (q >= 0)
        at = m.select(woff + fx.min(q + 1, Int32(BINS - 1)), junk + lane)
        fx.llvm.atomic_add(lds + at, Int32(1), syncscope="workgroup")
    gpu.barrier()


def _window_lo(word):
    """The window's lowest top-16 value around an anchor word (an order key:
    any word, a window that misses only costs the passes): its top 16 bits
    centred, the window kept inside 16 bits."""
    a16 = (word >> 16) & 0xFFFF
    return fx.min(fx.max(a16 - (WINDOW // 2), Int32(0)), Int32(0xFFFF - (WINDOW - 1)))


def _signed(m: int) -> int:
    return m - (1 << 32) if m >= 1 << 31 else m


def _digit(h, lane, copies, remain):
    """From histogram ``h`` (the sum of its ``copies``): the digit holding the
    ``remain``-th largest counted key, one uniform word ``hit << 25 | whole <<
    24 | digit << 16 | remain'`` (``remain'`` still to take in the digit,
    ``whole`` 1 if that is all of its keys; 0: fewer than ``remain`` counted)."""
    v = _lds4(h, 252 - 4 * lane)
    for c in range_constexpr(1, copies):
        w = _lds4(h + c * H0_STRIDE, 252 - 4 * lane)
        v = [a + b for a, b in zip(v, w)]
    cnt = [v[3], v[2], v[1], v[0]]  # digits 255 - 4 l .. 252 - 4 l
    tot = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3])
    above = wave_prefix_dpp(tot) - tot
    pick = Int32(0)
    for j in range_constexpr(4):
        here = (above < remain) & (above + cnt[j] >= remain)
        whole = (above + cnt[j] == remain).select(Int32(1 << 24), Int32(0))
        word = Int32(1 << 25) | whole | ((255 - 4 * lane - j) << 16) | (remain - above)
        pick = here.select(word, pick)
        above = above + cnt[j]
    return wave_umax_dpp(pick)  # one lane at most holds the digit (>= 0)


@traced
def _cta_prefix(v, lane, wave, cnt):
    """(exclusive prefix of ``v`` in thread order, CTA total): one barrier. ``cnt``
    (WAVES words) is ``radix_select``'s own, read last before its ending
    barrier, so no barrier is needed before the waves' totals land there."""
    incl = wave_prefix_dpp(v)
    if lane == 63:
        fx.ptr_store(incl, cnt + wave)
    gpu.barrier()
    before = Int32(0)
    total = Int32(0)
    for w in range_constexpr(WAVES):
        c = fx.ptr_load(cnt + w)
        before = before + (wave > w).select(c, Int32(0))
        total = total + c
    return before + incl - v, total


@traced
def radix_select(lds, tid, keys, base, n, k, anchor=None):
    """The best min(k, n) of the CTA's order keys (thread ``tid``'s ``keys``
    at places ``base`` .., below ``n``; places in thread order, so ties go to
    the lower place) into LDS slots 0 .. n_out in place order; returns
    ``n_out`` (uniform). Every thread calls it; ``lds``: ``select_words(k)``
    words. Ends with a barrier (slots readable, ``radix_slot``).

    ``anchor`` (``index_plan.ANCHOR``; (pointer, word)): this list's k-th key
    of the step before. The anchored pass (``_count_window``) first: a k-th
    in a window bin gives the top 16 bits of the k-th and the count still to
    take there -- what passes 0 and 1 give --, so only passes 2 and 3 run;
    else the four passes, as without it. Either way the same k-th key and
    count, so the same slots. The k-th's word for the next step after."""
    lay = _layout(k)
    tid = fresh(tid)
    lane = tid % 64
    wave = uniform(tid // 64)
    if const_expr(anchor is not None):
        # out before the zeroing's barrier; waited on with the keys
        aword = ld_i32(anchor[0], anchor[1])
    for i in range_constexpr(-(-lay["cnt"] // THREADS)):
        if tid + i * THREADS < lay["cnt"]:
            fx.ptr_store(Int32(0), lds + (tid + i * THREADS))
    gpu.barrier()
    prefix, remain, live = Int32(0), Int32(k), Int32(1)
    # 1: the anchored pass stands for passes 0 and 1
    took = Int32(0)
    if const_expr(anchor is not None):
        lo16 = _window_lo(uniform(aword))
        # n <= k takes every key: no k-th to find (the passes find none)
        if uniform((fx.Int32(n) > k).select(Int32(1), Int32(0))) != 0:
            _count_window(lds, lay["win"], keys, base, n, lane, lo16, lay["junk"])
            gw = _digit(lds + lay["win"], lane, 1, remain)
            dw = (gw >> 16) & 255
            took = ((((gw >> 25) & 1) != 0) & (dw >= 1) & (dw <= WINDOW)).select(
                Int32(1), Int32(0)
            )
            hit = took != 0
            prefix = hit.select(((lo16 + dw - 1) & 0xFFFF) << 16, prefix)
            remain = hit.select(gw & 0xFFFF, remain)
            live = hit.select(((gw >> 24) & 1) ^ 1, live)
    mask = 0
    for p in range_constexpr(PASSES):
        shift = 24 - 8 * p
        copies = COPIES if p == 0 else 1
        hoff = 0 if p == 0 else lay["hist"] + BINS * (p - 1)
        run = live != 0
        if const_expr(anchor is not None and p < 2):
            run = run & (took == 0)
        if run:
            _count(
                lds, hoff, keys, base, n, lane, prefix, mask, shift, copies,
                lay["junk"], RADIX_SPARSE and p >= 2,
            )  # fmt: skip
            got = _digit(lds + hoff, lane, copies, remain)
            prefix = prefix | (((got >> 16) & 255) << shift)
            remain = (((got >> 25) & 1) != 0).select(got & 0xFFFF, remain)
            # none hit (fewer than k keys): prefix 0 keeps them all
            live = ((got >> 25) & 1) & (((got >> 24) & 1) ^ 1)
        mask = mask | (0xFF << shift)
    # a key above the k-th's prefix, or equal to it and among the first
    # ``remain`` such in place order: one prefix sum of the two counts
    # (16-bit halves: fewer than 2^16 keys a CTA)
    assert len(keys) * THREADS < 1 << 16
    count = Int32(0)
    for e in range_constexpr(len(keys)):
        ok = (base + e) < n
        gt = (ok & u32_lt(prefix, keys[e])).select(Int32(1), Int32(0))
        count = count + gt + (ok & (keys[e] == prefix)).select(Int32(1 << 16), Int32(0))
    before, total = _cta_prefix(count, lane, wave, lds + lay["cnt"])
    ab, tb = before & 0xFFFF, (before >> 16) & 0xFFFF
    for e in range_constexpr(len(keys)):
        u = fresh(keys[e])  # the masks again, not held across the scan
        ok = (base + e) < n
        gt = ok & u32_lt(prefix, u)
        tie = ok & (u == prefix)
        if gt | (tie & (tb < remain)):
            fx.ptr_store(
                fx.Vector.from_elements([base + e, u], Int32),
                lds + (lay["slots"] + 2 * (ab + fx.min(tb, remain))),
            )
        ab = ab + gt.select(Int32(1), Int32(0))
        tb = tb + tie.select(Int32(1), Int32(0))
    gpu.barrier()
    if const_expr(anchor is not None):  # noqa: SIM102
        # the k-th's key (a pass that took its digit whole leaves the bits
        # below it 0): the next step's anchor; none while every key is taken
        if (tid == 0) & (fx.Int32(n) > k):
            st_dev(prefix, anchor[0], anchor[1])
    return (total & 0xFFFF) + fx.min((total >> 16) & 0xFFFF, remain)


@traced
def radix_slot(lds, place, n_out, k):
    """Slot ``place`` of ``radix_select``: (order key, place), (0, -1) past
    ``n_out``."""
    lay = _layout(k)
    v = fx.Vector(
        fx.ptr_load(
            lds + (lay["slots"] + 2 * fx.min(place, k - 1)),
            result_type=fx.Vector.make_type(2, Int32),
        )
    )
    have = place < n_out
    return have.select(v[1], Int32(0)), have.select(v[0], Int32(-1))
