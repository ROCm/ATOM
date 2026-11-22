# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The argmax of a row over every TP rank's vocab shard, by tagged pairs in two
hops (the LM head's, ``lm_head.kernel``; DeepSeek-V4.1's draft head's,
``deepseek_v41/mono/kernels/draft_head``).

A key is aiter's u32 order of a logit's fp32 bits (-0 counted as +0), held in an
Int32; (key, id) pairs are compared in a total order -- the larger key, then the
lower id -- so any reduction order, on any rank, picks the same id: the lowest
id among the largest logits, as ``torch.argmax`` / ``topk_select(tie="low")``.

1. ``task_push``: a task's (CTA's) best of a row into this rank's MTOP
   [row][task].
2. ``rank_reduce``: one CTA a row polls the row's task bests on its rank and puts
   the rank's best into every rank's MRANK [source rank][row].
3. ``pick``: a lane a (row, source rank) polls MRANK; the row's ``tp`` lanes
   reduce to its id.

The context holds ``mb`` (the launch's mailbox), ``mtop`` / ``mrank`` (this rank's
regions), ``mrank_fan`` (every rank's MRANK), ``rank``, ``tid``, ``lane``,
``wave`` and ``lds`` (Int32 words).
"""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from atom.mono.device.ops import traced, xshfl
from atom.mono.device.select import INT_MIN, order_key, u32_lt
from atom.mono.plan.execution import THREADS, WAVES

INT_MAX = 2**31 - 1


def value_key(v):
    """The order key of fp32 ``v`` (``select.order_key``: aiter's u32 key in an
    Int32), -0 as +0."""
    bits = v.bitcast(fx.Int32)
    return order_key((bits == fx.Int32(INT_MIN)).select(fx.Int32(0), bits))


def take(ka, ia, kb, ib):
    """The better of two (key, id): a larger key (u32), then a lower id -- a
    total order, so any reduction order picks the same."""
    b = u32_lt(ka, kb) | ((ka == kb) & (ib < ia))
    return b.select(kb, ka), b.select(ib, ia)


def lane_best(kk, ii):
    """The wave's best (key, id), in every lane."""
    for off in (1, 2, 4, 8, 16, 32):
        kk, ii = take(kk, ii, xshfl(kk, off), xshfl(ii, off))
    return kk, ii


def task_pair(row, task, tasks):
    """A task's best of ``row`` in this rank's MTOP [row][task]."""
    return (row * tasks + task) * 2


def rank_pair(src, row, rows):
    """Rank ``src``'s best of ``row`` in MRANK [source rank][row]."""
    return (src * rows + row) * 2


def red_waves(c, at):
    """The waves' bests at LDS words ``at`` .. (a (key, id) pair a wave),
    reduced in wave order."""
    lds = c["lds"]
    kb = fx.Int32(fx.ptr_load(lds + at))
    ib = fx.Int32(fx.ptr_load(lds + (at + 1)))
    for w in range_constexpr(1, WAVES):
        kb, ib = take(
            kb, ib,
            fx.Int32(fx.ptr_load(lds + (at + 2 * w))),
            fx.Int32(fx.ptr_load(lds + (at + 2 * w + 1))),
        )  # fmt: skip
    return kb, ib


@traced
def task_push(c, row, task, tasks, kb, ib):
    """``row``'s best of this task (key ``kb``, id ``ib``) into this rank's
    MTOP; the caller picks the thread."""
    c["mb"].put_words(c["mtop"], task_pair(row, task, tasks), [kb, ib])


@traced
def rank_reduce(c, row, tasks, rows, tp, red_at):
    """The whole CTA: ``row``'s task bests on this rank -> the rank's best, to
    every rank's MRANK. ``red_at``: LDS words of a pair a wave, free from the
    leading barrier on."""
    tid = c["tid"]
    got = c["mb"].poll(
        [
            (c["mtop"], task_pair(row, fx.min(tid + k * THREADS, tasks - 1), tasks), 2)
            for k in range(-(-tasks // THREADS))
        ]
    )
    kk, ii = got[0][0], got[0][1]
    for g in got[1:]:
        kk, ii = take(kk, ii, g[0], g[1])
    kk, ii = lane_best(kk, ii)
    gpu.barrier()  # the area's earlier readers past it
    if c["lane"] == 0:
        at = red_at + c["wave"] * 2
        fx.ptr_store(kk, c["lds"] + at)
        fx.ptr_store(ii, c["lds"] + (at + 1))
    gpu.barrier()
    if tid == 0:
        kb, ib = red_waves(c, red_at)
        at = rank_pair(c["rank"], row, rows)
        for dst in range_constexpr(tp):
            c["mb"].put_words(c["mrank_fan"][dst], at, [kb, ib])


@traced
def pick(c, n, tp, rows, row_of):
    """Lane ``tid`` < ``n`` tp: row ``row_of(tid // tp)``'s best from rank
    ``tid % tp`` (MRANK), the row's ``tp`` lanes reduced. -> (the row's index
    b, its id in every one of its lanes, whether this thread is the row's
    first lane)."""
    tid = c["tid"]
    i = fx.min(tid, fx.Int32(n * tp - 1))
    b = i // tp
    got = c["mb"].poll([(c["mrank"], rank_pair(i % tp, row_of(b), rows), 2)])
    kk, ii = got[0][0], got[0][1]
    # a row's tp lanes: xor offsets below tp
    for k in range_constexpr(tp.bit_length() - 1):
        kk, ii = take(kk, ii, xshfl(kk, 1 << k), xshfl(ii, 1 << k))
    return b, ii, (tid < n * tp) & (tid % tp == 0)
