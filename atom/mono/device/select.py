# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""An exact top-k for a small k, by counting: entry e of an LDS array is a
signed 64-bit sort key (a 32-bit key high, the entry's id low: key, then id,
descending), and an entry's rank is the count of entries that beat it -- no
order of arrival enters, so every CTA and rank agrees.

Counting every entry against every other is VALU-bound (one 64-bit compare an
entry pair). Past a few hundred entries, a floor first narrows the field to
the entries that can still be among the top k, and only those are counted
against each other. Two forms of the narrowing:

- a CTA's entries: ``group_floor`` and ``compact`` (barriers, an LDS slot
  counter);
- one wave's entries (an MoE route: a few hundred router logits, a few a
  lane): ``wave_floor`` and ``wave_compact`` (butterflies and ballots, no
  barrier), with ``wave_kth`` for the case of too many candidates.
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from atom.mono.device.ops import ballot, butterfly, lanes_below, popcount, traced
from atom.mono.plan.execution import WAVES

INT_MIN = -(2**31)

KEY_LEAST = INT_MIN  # the 32-bit key of a dead entry: below every live one


def u32_lt(a, b):
    """a < b as unsigned 32-bit keys held in Int32s (FlyDSL's integer
    compares are signed): both sign bits flipped, compared signed."""
    return (a ^ fx.Int32(INT_MIN)) < (b ^ fx.Int32(INT_MIN))


def u32_gt(a, b):
    """a > b as unsigned 32-bit keys held in Int32s (``u32_lt``)."""
    return u32_lt(b, a)


BATCH = 16  # entries every lane reads a round of ``count_beaten``


def put_key(keys, e, key, idx, live):
    """Entry e of ``keys`` (an Int32 LDS pointer, two words an entry: id, key);
    a dead one the least, beating nothing. -> this entry's 64-bit sort key."""
    fx.ptr_store(live.select(idx, fx.Int32(0)), keys + 2 * e)
    fx.ptr_store(live.select(key, fx.Int32(KEY_LEAST)), keys + (2 * e + 1))
    return (fx.Int64(key) << 32) | fx.Int64(idx)


def entry_key(keys, e):
    """Entry e's 64-bit sort key, as ``put_key`` returned it."""
    idx = fx.ptr_load(keys + 2 * e)
    key = fx.ptr_load(keys + (2 * e + 1))
    return (fx.Int64(key) << 32) | fx.Int64(idx)


@traced
def count_beaten(keys, mines, m):
    """For each of up to two 64-bit sort keys ``mines``, how many of the first
    m entries of ``keys`` beat it. Entries up to m rounded up to BATCH must
    hold ``put_key``'s (a dead one beats nothing). Every lane reads the same
    entry (an LDS broadcast), BATCH a round so the reads overlap, and a
    compare is one instruction; the counts share one register, 16 bits each,
    so the dynamic loop carries one value."""
    assert len(mines) <= 2, "two 16-bit counts a register"
    v4i = fx.Vector.make_type(4, fx.Int32)
    packed = fx.Int32(0)
    for j0 in range(0, m, BATCH):
        j0 = fx.Int32(j0)
        pairs = [
            fx.Vector(fx.ptr_load(keys + (2 * j0 + 4 * q), result_type=v4i)).bitcast(
                fx.Int64
            )
            for q in range(BATCH // 2)
        ]
        for q in range_constexpr(BATCH // 2):
            for e in range_constexpr(2):
                other = fx.Int64(pairs[q][e])
                for i, mine in enumerate(mines):
                    packed = packed + (other > mine).select(1 << (16 * i), 0)
    return [(packed >> (16 * i)) & 0xFFFF for i in range(len(mines))]


@traced
def group_floor(keys32, k, red, lane, wave):
    """A lower bound of the k-th largest of the CTA's 32-bit keys (each
    thread's ``keys32``, dead ones ``KEY_LEAST``): k disjoint groups -- the
    entries of the lanes with one ``lane % k`` -- each group's largest key, the
    least of those; k distinct entries are at least it. Live entries packed
    from entry 0 (k of them at least) leave no group empty: a group of
    contiguous lanes would hold only dead ones past the live, and its least key
    makes the floor no floor. ``red``: WAVES * k + 1 words of LDS. Barriers
    included."""
    assert k in (1, 2, 4, 8, 16, 32, 64), k
    top = keys32[0]
    for v in keys32[1:]:
        top = fx.max(top, v)
    top = butterfly(top, tuple(o for o in (32, 16, 8, 4, 2, 1) if o >= k), fx.max)
    if lane < k:
        fx.ptr_store(top, red + (wave * k + lane))
    gpu.barrier()
    if (wave == 0) & (lane < k):
        g = fx.ptr_load(red + lane)
        for w in range_constexpr(1, WAVES):
            g = fx.max(g, fx.ptr_load(red + (w * k + lane)))
        g = butterfly(g, tuple(o for o in (32, 16, 8, 4, 2, 1) if o < k), fx.min)
        if lane == 0:
            fx.ptr_store(g, red + WAVES * k)
    gpu.barrier()
    return fx.ptr_load(red + WAVES * k)


@traced
def compact(cand, counter, cap, entries, floor):
    """The entries of ``entries`` ((key32, id, live) a thread) that can be in
    the top k -- live, key at least ``floor`` (ties kept: a superset) -- to
    ``cand`` as ``put_key`` entries, an LDS slot each from ``counter`` (zeroed,
    and ``cand`` filled with dead entries, before the barrier ``group_floor``
    holds). Slots past ``cap`` are dropped; the caller reads the count after a
    barrier and counts every entry instead when it is past ``cap``."""
    for key, idx, live in entries:
        if live & (key >= floor):
            slot = fx.llvm.atomic_add(counter, fx.Int32(1), syncscope="workgroup")
            if slot < cap:
                put_key(cand, slot, key, idx, live)


# ---------------------------------------------------------------- one wave
# A wave's entries: each lane holds a list of 32-bit keys (signed order), entry
# i of lane l is the wave's entry l + 64 i, which is also its tie order.


def f32_key(v):
    """An order-preserving signed 32-bit key of an f32 (larger float, larger
    key): a negative float's magnitude bits flipped."""
    b = v.bitcast(fx.Int32)
    return (b < fx.Int32(0)).select(b ^ fx.Int32(0x7FFFFFFF), b)


def wave_floor(keys32, k):
    """A lower bound of the k-th largest of the wave's keys: the k disjoint
    groups of lanes with one ``lane % k``, each group's largest, the least of
    those. No LDS, no barrier."""
    assert k in (1, 2, 4, 8, 16, 32, 64), k
    top = keys32[0]
    for v in keys32[1:]:
        top = fx.max(top, v)
    top = butterfly(top, tuple(o for o in (32, 16, 8, 4, 2, 1) if o >= k), fx.max)
    return butterfly(top, tuple(o for o in (32, 16, 8, 4, 2, 1) if o < k), fx.min)


@traced
def wave_compact(cand, cap, entries, lane, carry=()):
    """The wave's ``entries`` ((selected, key32, id) each) that are selected,
    to ``cand`` as ``put_key`` entries in entry order (a ballot prefix a list
    position: no atomics), the first ``cap`` of them; ``carry``: (LDS pointer,
    one value an entry) pairs stored at each entry's slot too. Then BATCH dead
    entries past the last stored one, the most ``count_beaten`` reads past it
    (``cand`` holds 2 (cap + BATCH) words). -> how many were selected (all,
    past ``cap`` too: the caller decides what that means)."""
    n = fx.Int32(0)
    for i in range_constexpr(len(entries)):
        sel, key, idx = entries[i]
        bal = ballot(sel)
        slot = n + lanes_below(bal, lane)
        if sel & (slot < cap):
            put_key(cand, slot, key, idx, sel)
            for j in range_constexpr(len(carry)):
                fx.ptr_store(carry[j][1][i], carry[j][0] + slot)
        n = n + popcount(bal)
    if lane < BATCH:
        e = fx.min(n, cap) + lane
        fx.ptr_store(fx.Int32(0), cand + 2 * e)
        fx.ptr_store(fx.Int32(KEY_LEAST), cand + (2 * e + 1))
    return n


@traced
def wave_kth(keys32, k, lane):
    """Exactly the k largest of the wave's keys, equal keys the lower entry
    first: the k-th largest by a 32-step bit search of ballot counts, then a
    mask an entry (as ``keys32``). For the case of too many candidates for
    ``wave_floor`` to narrow (many equal keys)."""
    per = len(keys32)
    keys = [fx.Uint32(v ^ fx.Int32(INT_MIN)) for v in keys32]

    def count_ge(th):
        n = fx.Int32(0)
        for i in range_constexpr(per):
            n = n + popcount(ballot(keys[i] >= th))
        return n

    th = fx.Uint32(0)
    for bit in range_constexpr(31, -1, -1):
        trial = th | fx.Uint32(1 << bit)
        th = (count_ge(trial) >= k).select(trial, th)
    n_gt = fx.Int32(0)
    for i in range_constexpr(per):
        n_gt = n_gt + popcount(ballot(keys[i] > th))
    need = fx.Int32(k) - n_gt
    eq_before = fx.Int32(0)
    sels = []
    for i in range_constexpr(per):
        eqb = ballot(keys[i] == th)
        eq_rank = eq_before + lanes_below(eqb, lane)
        sels.append((keys[i] > th) | ((keys[i] == th) & (eq_rank < need)))
        eq_before = eq_before + popcount(eqb)
    return sels
