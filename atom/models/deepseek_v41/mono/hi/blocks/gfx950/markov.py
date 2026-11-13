# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``MarkovHead``: a DSpark draft's block sampler. Per block position ``j`` (a row of the step) the id

    x_j = argmax_v ( bf16(x_row_j . W[v]) + W1[x_{j-1}] . W2[v] )        (x_{-1}: the anchor)

over the vocab: the LM head's logit rounded to bf16 (a bf16 GEMM's output, as the engine's draft head computes it),
plus the Markov bias in fp32, ties to the lowest id; ``W`` vocab-sharded bf16 (``ArgmaxHead``'s weight and layout:
the target head's packed tensor), ``W1`` / ``W2`` [V, r] bf16 (the whole tables; this rank reads ``W2``'s rows of its
vocab). The positions are sequential: each one's bias needs the previous one's id, the argmax over every rank.

One stage, ``ceil(n / 128)`` tasks (a task a CTA: every task must be resident, each waits for all of them; where a
rank's blocks outnumber the CUs, two 128-id blocks a task, ``MarkovHead(blocks=)``):

1. the dot (``ArgmaxHead``'s: 8 row groups of 16 vocab rows a task over the whole K, bf16 MFMAs), the logits of the
   task's 128 ids for every row rounded to bf16 into LDS; the task's 128 ``W2`` rows into LDS;
2. per position j in order: ``W1[x_{j-1}]`` into LDS; per id the bias (4 lanes, 64 products each in rank order, the
   parts summed (0 + 1) + (2 + 3)); val = base + bias; the task's best (largest val, then lowest id); put to every
   rank (tagged pairs, a slot per (position, task)); poll every rank's every task's best of position j and reduce
   (the same total order: every CTA of every rank picks the same id); task 0 writes it to ``<name>.ids``.

Numerics: the engine's ``dspark_markov_argmax`` sums the same exact bf16 products in another fp32 order (its MFMA
K-blocking), and its base logits come from another GEMM's order: ids agree except where two candidates' values are
within that rounding (near-ties).

**B requests** (``Scenario(B, spec)``, B > 1): B independent blocks of ``spec`` positions (row ``b spec + j`` is
request b's position j). The dot runs once for every row (``ArgmaxHead``'s 16-row MFMA chunks and, where x does not
fit the LDS, its K slices); then per position j the B requests' steps run side by side: their B ``W1`` rows, B biases
a thread, B task bests pushed and polled in one round trip, B picks. A request's ids are the bits of its block alone.

Args: ``<name>.w`` (``ArgmaxHead.pack``), ``<name>.w1`` / ``<name>.w2`` (bf16 [V, r] tables), ``<name>.anchor``
(int32 [B]: each request's x_{-1}), ``<name>.ids`` (int32 [rows]: out).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import (
    bf2_f32,
    bf16_round,
    ld_i32,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from ...device.gfx950.peer import from_rank, to_rank
from ...device.gfx950.topk import INT_MIN, order_key
from ...plan.program import Region, Scenario, Stage
from ...plan.trace import Space
from ...target import MI355X
from ..base import Block, Channel, Fmt
from . import head as ah
from .linear import _land_batch, _try_batch

THREADS, WAVES = MI355X.threads, MI355X.waves
VT = WAVES * ah.RG  # ids a task: 128
LAND_GROUP = 2  # B requests: x's poll batches in flight at once (each 12 two-pair specs a thread)
INT_MAX = 2**31 - 1
# several blocks a task: W1 words between the bias pass's scheduling fences
BIAS_FENCE = 8


@dataclass(frozen=True)
class _MPlan(ah._Plan):
    rank_r: int = 256  # the Markov tables' rank r
    batch: int = 1  # requests: blocks of ``rows / batch`` positions, side by side
    blocks: int = (
        1  # 128-id blocks a task (``MarkovHead(blocks=)``: a rank's vocab past one block a CU)
    )

    @property
    def single(
        self,
    ) -> (
        bool
    ):  # the one-request path (x whole in the LDS); else the B-request path, which also takes
        return (
            self.batch == 1 and self.kparts == 1 and self.blocks == 1
        )  # one request whose x lands in K slices
        # (or a task of several blocks: their W2 rows over x)

    @property
    def vt(self) -> int:  # ids a task: VT a block
        return VT * self.blocks

    @property
    def mtasks(
        self,
    ) -> (
        int
    ):  # the stage's tasks (a CTA each); ``tasks``: 128-id blocks, the weight's packing
        return -(-self.tasks // self.blocks)

    @property
    def rw(self) -> int:  # 32-bit words a table row
        return self.rank_r // 2

    @property
    def w2pitch(self) -> int:
        return self.rw + 4

    @property
    def a_base(
        self,
    ) -> (
        int
    ):  # base logits [rows][VT] f32, after x (B requests: after x or the W2 rows over it)
        if self.single:
            return self.rows * self.pitch
        return max(self.rows * self.pitch, self.vt * self.w2pitch)

    @property
    def a_w2(
        self,
    ) -> (
        int
    ):  # the task's W2 rows [VT][w2pitch] (B requests: over x, loaded once the dot is done)
        return self.a_base + self.rows * VT if self.single else 0

    @property
    def a_w1(self) -> int:  # W1[x_{j-1}] [rw] (B requests: one a request)
        return (
            self.a_w2 + VT * self.w2pitch
            if self.single
            else self.a_base + self.rows * self.vt
        )

    @property
    def a_red(
        self,
    ) -> int:  # the waves' (key, id) [request][WAVES][2], then x_j a request
        return self.a_w1 + self.rw * self.batch

    @property
    def lds_words(self) -> int:
        return self.a_red + (
            2 * WAVES + 4 if self.single else (2 * WAVES + 1) * self.batch + 4
        )

    @property
    def pos(self) -> int:  # positions a request
        return self.rows // self.batch

    @property
    def mtop(self) -> str:
        return f"{self.name}.top"

    @property
    def mtop_pairs(
        self,
    ) -> int:  # a source rank's slot: (key, id) pairs [position][task]
        return self.rows * self.mtasks * 2


class MarkovHead(Block):
    """The DSpark block sampler (module docstring). ``n``: this rank's vocab rows (a multiple of 16)."""

    def __init__(self, name: str, x: Channel, n: int, layout: str = "packed", prefetch: int = 16, lead: int = 4,
                 rank_r: int = 256, logits: Channel | None = None, kparts: int = 0,
                 blocks: int | None = None):  # fmt: skip
        # blocks: 128-id blocks a task; None: 1 while a rank's blocks fit one a CU, else 2 (V4.1 at TP2: 505 blocks of
        # its 64,640 ids, 253 tasks of 2)
        if blocks not in (None, 1, 2) or (blocks == 2 and logits is not None):
            raise ValueError(
                f"MarkovHead {name}: blocks None, 1 or 2 (2: the head's own dot, no logits channel)"
            )
        self.blocks = blocks
        # logits: the base logits from that BF16 channel [rows, n] (an LM head stage before this one, e.g. on every
        # CU), not this block's dot (no weight, x unread); the rest unchanged
        if logits is not None and (
            logits.fmt is not Fmt.BF16 or logits.flagged or logits.width != n
        ):
            raise ValueError(
                f"MarkovHead {name}: logits a tagged BF16 channel of width n"
            )
        self.logits = logits
        if logits is not None:
            x = Channel(
                x.name, 2 * ah.KSTEP, Fmt.BF16
            )  # (x unread: the plan's LDS without it)
        if x.fmt is not Fmt.BF16 or x.flagged or x.width % (2 * ah.KSTEP):
            raise ValueError(
                f"MarkovHead {name}: x is a tagged BF16 channel, width a multiple of {2 * ah.KSTEP}"
            )
        if n % ah.RG or layout not in ah.LAYOUTS or rank_r % 8:
            raise ValueError(
                f"MarkovHead {name}: n a multiple of {ah.RG}, layout in {ah.LAYOUTS}, r of 8"
            )
        self.name, self.x, self.n, self.layout, self.prefetch, self.lead = (
            name,
            x,
            n,
            layout,
            prefetch,
            lead,
        )
        self.rank_r = rank_r
        # x's K slices at least this many (B requests; 0: the fewest that fit the LDS), as ``ArgmaxHead(kparts=)``: the
        # same products and K order
        if kparts < 0:
            raise ValueError(f"MarkovHead {name}: kparts >= 0")
        self.kparts = kparts

    def plan(self, sc: Scenario) -> _MPlan:
        p = _MPlan(self.name, sc.rows, self.x.width, self.n, sc.tp, self.x.name, "bf16", self.layout, self.prefetch,
                   False, "", self.lead, False, rank_r=self.rank_r, batch=sc.batch)  # fmt: skip
        blocks = self.blocks or (
            1 if p.tasks <= MI355X.cus or self.logits is not None else 2
        )
        if blocks > 1:
            p = replace(p, blocks=blocks)
        if (
            sc.batch > 1 or p.lds_words * 4 > MI355X.lds_bytes
        ):  # x in K slices where it does not fit whole
            if self.kparts > 1:
                if p.steps % self.kparts:
                    raise ValueError(
                        f"MarkovHead {self.name}: kparts {self.kparts} does not divide {p.steps} K steps"
                    )
                p = replace(p, kparts=self.kparts)
            # (two blocks: slices of a fifth of K or less, so x's landing batches fit beside the blocks' streams:
            # V4.1 TP2 at 15 / 20 rows spilled 37 / 4 VGPRs at 2 slices, none at 5)
            for kp in (
                q
                for q in (range(5, 11) if p.blocks > 1 else range(2, 9))
                if p.steps % q == 0 and q > p.kparts
            ):
                if p.lds_words * 4 <= MI355X.lds_bytes:
                    break
                p = replace(p, kparts=kp)
        if p.lds_words * 4 > MI355X.lds_bytes:
            raise ValueError(
                f"MarkovHead {self.name}: {sc.rows} rows do not fit the LDS"
            )
        if p.mtasks > MI355X.cus:
            raise ValueError(
                f"MarkovHead {self.name}: {p.mtasks} tasks, more than one a CTA"
            )
        if p.single and sc.rows > 16:
            raise ValueError(
                f"MarkovHead {self.name}: at most 16 positions (the MFMA's N)"
            )
        return p

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        space = Space.PEER if p.tp > 1 else Space.SCRATCH
        top = Region(p.mtop, p.mtop_pairs, (self.name,), space=space, exchange=True)
        lg = self.logits.name if self.logits is not None else ""
        args = tuple(
            f"{self.name}.{a}"
            for a in (("w",) if not lg else ()) + ("w1", "w2", "anchor", "ids")
        )
        key = (p, "markov") + ((("logits", lg),) if lg else ())
        return [Stage(self.name, p.mtasks, partial(_mk, p=p, lg=lg), lds_words=p.lds_words, writes=(top,), args=args,
                      key=key)]  # fmt: skip


# ------------------------------------------------------------------ device


def _mk(k, t, p: _MPlan, lg: str = ""):
    lane, wave = k.lane, k.wave
    r_w = rsrc(k.args[f"{p.name}.w"], p.w_bytes)

    tb = (
        t * p.blocks if p.blocks > 1 else t
    )  # the task's first 128-id block in the weight's packing
    rg = tb * WAVES + wave
    issue = partial(ah._issue, p, r_w, tb, wave, rg, lane, rg < p.n_rg)
    n_pre = min(p.prefetch, p.steps)
    lead = min(p.lead, n_pre)
    q = [issue(i) for i in range(lead)]
    if const_expr(p.single):
        batches = ah._x_batches(k, p, k.tid, THREADS)
        rocdl.sched_barrier(0)
        tries = [_try_batch(specs) for specs, _ in batches]
        rocdl.sched_barrier(0)
        q += [issue(i) for i in range(lead, n_pre)]
        rocdl.sched_barrier(0)
        for (specs, dests), words in zip(batches, tries, strict=True):
            _land_batch(k, p, specs, dests, words)
    else:  # B requests: x's first K slice two batches at a time, then the rest of the prefetch (no words beside it)
        batches = ah._x_batches(k, p, k.tid, THREADS)
        for g in range(0, len(batches), LAND_GROUP):
            rocdl.sched_barrier(0)
            grp = batches[g : g + LAND_GROUP]
            tries = [_try_batch(specs) for specs, _ in grp]
            rocdl.sched_barrier(0)
            for (specs, dests), words in zip(grp, tries, strict=True):
                _land_batch(k, p, specs, dests, words)
        rocdl.sched_barrier(0)
        q += [issue(i) for i in range(lead, n_pre)]
        # the task's further blocks: their weight streams, one block after another
        issues = [issue]
        for blk in range_constexpr(1, p.blocks):
            rg_b = (tb + blk) * WAVES + wave
            issues.append(
                partial(ah._issue, p, r_w, tb + blk, wave, rg_b, lane, rg_b < p.n_rg)
            )
    gpu.barrier()
    first = k.rank * p.n if p.tp > 1 else fx.Int32(0)
    if const_expr(p.single):
        _stream(k, p, lane, issue, q)
        _w2_tile(k, p, t, first)
        gpu.barrier()
        xp = uniform(ld_i32(k.args[f"{p.name}.anchor"], 0))
        for j in range_constexpr(p.rows):
            xp = _position(k, p, t, j, xp, first)
    else:  # B requests: the dot over every row, then the requests' positions side by side
        _stream_b(k, p, lane, issues, q)
        _w2_tile(k, p, t, first)
        gpu.barrier()
        _anchors(
            k, p
        )  # x_{-1} of every request into LDS (where each position leaves its x_j)
        gpu.barrier()
        _position_loop(k, p, t, first)


def _stream(k, p: _MPlan, lane, issue, q):
    """``ArgmaxHead._stream``'s MFMAs; the logits rounded to bf16 into LDS ``a_base`` [row][id in the task]."""
    n_pre = len(q)
    row = fx.min(lane % 16, fx.Int32(p.rows - 1))
    xrow = k.ldsf + row * p.pitch + (lane // 16) * 4
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range(p.steps):
        rocdl.sched_barrier(0)
        if i + n_pre < p.steps:
            q.append(issue(i + n_pre))
        b = fx.ptr_load(
            xrow + i * (ah.KSTEP // 2), result_type=fx.Vector.make_type(4, fx.Float32)
        )
        acc = mfma_bf16(q[i].bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), acc)
    rocdl.sched_barrier(0)
    _base_put(k, p, lane, [bf16_round(acc[e]) for e in range(4)])


@traced
def _base_put(k, p: _MPlan, lane, vals):
    if lane % 16 < p.rows:
        at = p.a_base + (lane % 16) * VT + k.wave * ah.RG + (lane // 16) * 4
        for e in range_constexpr(4):
            fx.ptr_store(vals[e], k.ldsf + (at + e))


def _stream_b(k, p: _MPlan, lane, issues, q):
    """B requests: ``ArgmaxHead._stream``'s MFMAs over 16-row chunks (x's next K slice landing between its steps), the
    task's blocks one after another (``issues``: a block's weight stream; ``q``: block 0's prefetch; a later block
    lands x's first K slice again: a block's registers, not two, are live); the logits rounded to bf16 into LDS
    ``a_base`` [row][id in the task]."""
    n_pre = len(q)
    xrows = [k.ldsf + fx.min(lane % 16 + ah.RG * ch, fx.Int32(p.rows - 1)) * p.pitch + (lane // 16) * 4
             for ch in range(p.rc)]  # fmt: skip
    for blk in range(p.blocks):
        if blk:
            if const_expr(p.kparts > 1):
                gpu.barrier()
                ah._land_x(k, p, 0)
                gpu.barrier()
            q = [issues[blk](i) for i in range(n_pre)]
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(p.rc)]
        for i in range(p.steps):
            rocdl.sched_barrier(0)
            if (
                i and i % p.psteps == 0
            ):  # the next K slice of x, once every wave is past the last one
                gpu.barrier()
                ah._land_x(k, p, i // p.psteps)
                gpu.barrier()
            if i + n_pre < p.steps:
                q.append(issues[blk](i + n_pre))
            for ch in range(p.rc):
                b = fx.ptr_load(
                    xrows[ch] + (i % p.psteps) * (ah.KSTEP // 2),
                    result_type=fx.Vector.make_type(4, fx.Float32),
                )
                accs[ch] = mfma_bf16(
                    q[i].bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), accs[ch]
                )
        rocdl.sched_barrier(0)
        gpu.barrier()  # (x's LDS is not the base logits', but the last slice's readers finish before any reuse)
        for ch, acc in enumerate(accs):
            _base_put_ch(k, p, lane, ch, [bf16_round(acc[e]) for e in range(4)], blk)


@traced
def _base_put_ch(k, p: _MPlan, lane, ch, vals, blk: int = 0):
    row = lane % 16 + ah.RG * ch
    if row < p.rows:
        at = p.a_base + row * p.vt + k.wave * ah.RG + (lane // 16) * 4
        at = (
            at + blk * VT if not isinstance(blk, int) or blk else at
        )  # (blk: a runtime value past block 0)
        for e in range_constexpr(4):
            fx.ptr_store(vals[e], k.ldsf + (at + e))


def _w2_tile(k, p: _MPlan, t, first):
    """The task's ``W2`` rows (ids ``first + 128 t ..``; a row past this rank's ``n`` reads the last one) into LDS:
    16 B a load, a row's ``rw`` words by ``rw / 4`` lanes."""
    r_w2 = rsrc(k.args[f"{p.name}.w2"])
    per = p.rw // 4  # 16 B loads a row
    total = p.vt * per
    for c in range(-(-total // THREADS)):
        i = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(total - 1))
        v, w4 = i // per, i % per
        gid = first + fx.min(t * p.vt + v, fx.Int32(p.n - 1))
        got = fx.Vector(
            bo.buffer_load(r_w2, gid * p.rw + w4 * 4, vec_width=4, dtype=T.i32)
        )
        fx.ptr_store(
            got.bitcast(fx.Float32), k.ldsf + (p.a_w2 + v * p.w2pitch + w4 * 4)
        )


def _position(k, p: _MPlan, t, j, xp, first):
    """Position ``j`` given x_{j-1} = ``xp`` (module docstring, 2.) -> x_j (wave-uniform)."""
    _w1_row(k, p, xp)
    gpu.barrier()
    v, part = k.tid // 4, k.tid % 4
    seg = p.rw // 4  # words a part
    w1a, w2a = p.a_w1 + part * seg, p.a_w2 + v * p.w2pitch + part * seg
    acc = fx.Float32(0.0)
    for w in range_constexpr(seg):
        a0, a1 = bf2_f32(fx.ptr_load(k.ldsf + (w1a + w)).bitcast(fx.Int32))
        b0, b1 = bf2_f32(fx.ptr_load(k.ldsf + (w2a + w)).bitcast(fx.Int32))
        acc = acc + a0 * b0
        acc = acc + a1 * b1
    acc = acc + xshfl(acc, 1)
    acc = acc + xshfl(acc, 2)
    val = fx.ptr_load(k.ldsf + (p.a_base + j * VT + v)) + acc
    lid = t * VT + v
    ok = (part == 0) & (lid < p.n)
    bits = val.bitcast(fx.Int32)
    bits = (bits == fx.Int32(INT_MIN)).select(fx.Int32(0), bits)  # -0 as +0
    kk = ok.select(order_key(bits), fx.Int32(0))
    ii = ok.select(lid + first, fx.Int32(INT_MAX))
    for off in (1, 2, 4, 8, 16, 32):
        kk, ii = ah._take(kk, ii, xshfl(kk, off), xshfl(ii, off))
    _wave_best(k, p, kk, ii)
    gpu.barrier()
    _task_push(k, p, t, j)
    # every rank's every task's best of position j
    ents = p.tp * p.mtasks
    a = from_rank(k, p.mtop, 0) if p.tp > 1 else k.mb[p.mtop]
    specs = []
    for c in range(-(-ents // THREADS)):
        e = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(ents - 1))
        src, tt = e // p.mtasks, e % p.mtasks
        specs.append((a, src * p.mtop_pairs + (j * p.mtasks + tt) * 2, 2))
    got = mb.poll(specs, k.tag)
    kk, ii = got[0][0], got[0][1]
    for g in got[1:]:
        kk, ii = ah._take(kk, ii, g[0], g[1])
    for off in (1, 2, 4, 8, 16, 32):
        kk, ii = ah._take(kk, ii, xshfl(kk, off), xshfl(ii, off))
    _pick(k, p, t, j, kk, ii)
    gpu.barrier()
    return uniform(fx.Int32(fx.ptr_load(k.lds + (p.a_red + 2 * WAVES))))


@traced
def _anchors(k, p: _MPlan):
    if k.tid < p.batch:
        fx.ptr_store(
            ld_i32(k.args[f"{p.name}.anchor"], k.tid),
            k.lds + (p.a_red + 2 * WAVES * p.batch + k.tid),
        )


@traced
def _position_loop(k, p: _MPlan, t, first):
    """B requests: the positions in a loop (not unrolled: a position's registers are not live into the next)."""
    for j in range(p.pos):
        _positions(k, p, t, j, first)


def _positions(k, p: _MPlan, t, j, first):
    """B requests: position ``j`` of every request given its x_{j-1} (in LDS: the anchor, then the last position's
    pick) -> their x_j into LDS (module docstring, "B requests": one push and one poll round trip for all of them).
    """
    xps = [
        uniform(fx.Int32(fx.ptr_load(k.lds + (p.a_red + 2 * WAVES * p.batch + b))))
        for b in range(p.batch)
    ]
    gpu.barrier()  # every thread has read x_{j-1} before this position's pick overwrites it
    for b in range(p.batch):
        _w1_row(k, p, xps[b], b)
    gpu.barrier()
    v, part = k.tid // 4, k.tid % 4
    seg = p.rw // 4  # words a part
    w2a = p.a_w2 + v * p.w2pitch + part * seg
    lid = t * p.vt + v
    ok = (part == 0) & (lid < p.n)
    _bias_loop(k, p, j, first, w2a, part, seg, v, lid, ok)
    rocdl.sched_barrier(0)
    gpu.barrier()
    _task_push_b(k, p, t, j)
    # every rank's every task's best of position j, every request
    ents = p.tp * p.mtasks
    a = from_rank(k, p.mtop, 0) if p.tp > 1 else k.mb[p.mtop]
    per = -(-ents // THREADS)
    specs = []
    for b in range(p.batch):
        for c in range(per):
            e = fx.min(fx.Int32(c * THREADS) + k.tid, fx.Int32(ents - 1))
            src, tt = e // p.mtasks, e % p.mtasks
            specs.append(
                (a, src * p.mtop_pairs + ((b * p.pos + j) * p.mtasks + tt) * 2, 2)
            )
    got = mb.poll(specs, k.tag)
    best = []
    for b in range(p.batch):
        gb = got[b * per : (b + 1) * per]
        kk, ii = gb[0][0], gb[0][1]
        for g in gb[1:]:
            kk, ii = ah._take(kk, ii, g[0], g[1])
        for off in (1, 2, 4, 8, 16, 32):
            kk, ii = ah._take(kk, ii, xshfl(kk, off), xshfl(ii, off))
        best.append((kk, ii))
    _pick_b(k, p, t, j, best)
    gpu.barrier()


@traced
def _bias_loop(k, p: _MPlan, j, first, w2a, part, seg, v, lid, ok):
    """B requests: each request's biases and the wave's best of position ``j``, a request at a time (a runtime loop:
    one request's registers live at once)."""
    for b in range(p.batch):
        _bias_one(k, p, j, b, first, w2a, part, seg, v, lid, ok)


def _bias_one(k, p: _MPlan, j, b, first, w2a, part, seg, v, lid, ok):
    """Request ``b``'s position ``j``: the thread's id ``v`` of every block of the task (``w2a`` / ``lid`` / ``ok``:
    block 0's), the best of them, then the wave's best."""
    w1a = p.a_w1 + b * p.rw + part * seg
    # every block's dot in one pass over the W1 row: a W1 word read once, never held for a later block
    accs = [fx.Float32(0.0) for _ in range(p.blocks)]
    for w in range_constexpr(seg):
        # (two blocks' LDS loads hoisted over the whole row spill 10 VGPRs: TP2)
        if const_expr(p.blocks > 1 and w and w % BIAS_FENCE == 0):
            rocdl.sched_barrier(0)
        a0, a1 = bf2_f32(fx.ptr_load(k.ldsf + (w1a + w)).bitcast(fx.Int32))
        for blk in range_constexpr(p.blocks):
            b0, b1 = bf2_f32(
                fx.ptr_load(k.ldsf + (w2a + blk * VT * p.w2pitch + w)).bitcast(fx.Int32)
            )
            accs[blk] = accs[blk] + a0 * b0
            accs[blk] = accs[blk] + a1 * b1
    kk, ii = None, None
    for blk in range_constexpr(p.blocks):
        acc = accs[blk]
        acc = acc + xshfl(acc, 1)
        acc = acc + xshfl(acc, 2)
        val = (
            fx.ptr_load(k.ldsf + (p.a_base + (b * p.pos + j) * p.vt + blk * VT + v))
            + acc
        )
        bits = val.bitcast(fx.Int32)
        bits = (bits == fx.Int32(INT_MIN)).select(fx.Int32(0), bits)  # -0 as +0
        lid_b = lid + blk * VT
        ok_b = ok if blk == 0 else (part == 0) & (lid_b < p.n)
        kb = ok_b.select(order_key(bits), fx.Int32(0))
        ib = ok_b.select(lid_b + first, fx.Int32(INT_MAX))
        kk, ii = (kb, ib) if blk == 0 else ah._take(kk, ii, kb, ib)
    for off in (1, 2, 4, 8, 16, 32):
        kk, ii = ah._take(kk, ii, xshfl(kk, off), xshfl(ii, off))
    _wave_best_dyn(k, p, kk, ii, b)


@traced
def _wave_best_dyn(k, p: _MPlan, kk, ii, b):
    """Lane 0: the wave's best into request ``b``'s slot of ``a_red`` (``b`` a runtime value)."""
    if k.lane == 0:
        fx.ptr_store(kk, k.lds + (p.a_red + (b * WAVES + k.wave) * 2))
        fx.ptr_store(ii, k.lds + (p.a_red + (b * WAVES + k.wave) * 2 + 1))


@traced
def _task_push_b(k, p: _MPlan, t, j):
    """Thread b < B: request b's task best of position j (its waves in order) to every rank's slot (row b spec + j,
    task t)."""
    if k.tid < p.batch:
        base = p.a_red + k.tid * 2 * WAVES
        kb = fx.Int32(fx.ptr_load(k.lds + base))
        ib = fx.Int32(fx.ptr_load(k.lds + (base + 1)))
        for w in range_constexpr(1, WAVES):
            kw = fx.Int32(fx.ptr_load(k.lds + (base + w * 2)))
            iw = fx.Int32(fx.ptr_load(k.lds + (base + w * 2 + 1)))
            kb, ib = ah._take(kb, ib, kw, iw)
        pair = ((k.tid * p.pos + j) * p.mtasks + t) * 2
        if const_expr(p.tp > 1):
            for dst in range_constexpr(p.tp):
                mb.put2(to_rank(k, p.mtop, dst), pair, kb, ib, k.tag)


@traced
def _pick_b(k, p: _MPlan, t, j, best):
    """B requests: the waves' bests a request (after the previous barrier's reads) -> each request's x_j into LDS;
    task 0 also writes them to ``<name>.ids``."""
    gpu.barrier()  # every wave has read ``a_red`` (the task push)
    if k.lane == 0:
        for b in range_constexpr(p.batch):
            fx.ptr_store(best[b][0], k.lds + (p.a_red + (b * WAVES + k.wave) * 2))
            fx.ptr_store(best[b][1], k.lds + (p.a_red + (b * WAVES + k.wave) * 2 + 1))
    gpu.barrier()
    if k.tid < p.batch:
        base = p.a_red + k.tid * 2 * WAVES
        kb = fx.Int32(fx.ptr_load(k.lds + base))
        ib = fx.Int32(fx.ptr_load(k.lds + (base + 1)))
        for w in range_constexpr(1, WAVES):
            kb, ib = ah._take(kb, ib, fx.Int32(fx.ptr_load(k.lds + (base + w * 2))),
                              fx.Int32(fx.ptr_load(k.lds + (base + w * 2 + 1))))  # fmt: skip
        fx.ptr_store(ib, k.lds + (p.a_red + 2 * WAVES * p.batch + k.tid))
        if t == 0:
            bo.buffer_store(ib, rsrc(k.args[f"{p.name}.ids"]), k.tid * p.pos + j)


@traced
def _w1_row(k, p: _MPlan, xp, b=0):
    """``W1[xp]`` into LDS (request ``b``'s row of ``a_w1``)."""
    if k.tid < p.rw // 4:
        got = fx.Vector(
            bo.buffer_load(
                rsrc(k.args[f"{p.name}.w1"]),
                xp * p.rw + k.tid * 4,
                vec_width=4,
                dtype=T.i32,
            )
        )
        if const_expr(b == 0):
            fx.ptr_store(got.bitcast(fx.Float32), k.ldsf + (p.a_w1 + k.tid * 4))
        else:
            fx.ptr_store(
                got.bitcast(fx.Float32), k.ldsf + (p.a_w1 + b * p.rw + k.tid * 4)
            )


@traced
def _wave_best(k, p: _MPlan, kk, ii, b=0):
    """Lane 0: the wave's best (request ``b``'s slot of ``a_red``)."""
    if k.lane == 0:  # noqa: SIM102 (traced)
        if const_expr(b == 0):
            fx.ptr_store(kk, k.lds + (p.a_red + k.wave * 2))
            fx.ptr_store(ii, k.lds + (p.a_red + k.wave * 2 + 1))


@traced
def _task_push(k, p: _MPlan, t, j):
    """Thread 0: the task's best of position j (its waves in order) to every rank's slot (position j, task t)."""
    if k.tid == 0:
        kb = fx.Int32(fx.ptr_load(k.lds + p.a_red))
        ib = fx.Int32(fx.ptr_load(k.lds + (p.a_red + 1)))
        for w in range_constexpr(1, WAVES):
            kw = fx.Int32(fx.ptr_load(k.lds + (p.a_red + w * 2)))
            iw = fx.Int32(fx.ptr_load(k.lds + (p.a_red + w * 2 + 1)))
            kb, ib = ah._take(kb, ib, kw, iw)
        pair = (j * p.mtasks + t) * 2
        if const_expr(p.tp > 1):
            for dst in range_constexpr(p.tp):
                mb.put2(to_rank(k, p.mtop, dst), pair, kb, ib, k.tag)


@traced
def _pick(k, p: _MPlan, t, j, kk, ii):
    """The waves' bests (after the previous barrier's reads: a second slot set) -> x_j into LDS; task 0 also writes
    it to ``<name>.ids``."""
    gpu.barrier()  # every wave has read ``a_red`` (the task push)
    if k.lane == 0:
        fx.ptr_store(kk, k.lds + (p.a_red + k.wave * 2))
        fx.ptr_store(ii, k.lds + (p.a_red + k.wave * 2 + 1))
    gpu.barrier()
    if k.tid == 0:
        kb = fx.Int32(fx.ptr_load(k.lds + p.a_red))
        ib = fx.Int32(fx.ptr_load(k.lds + (p.a_red + 1)))
        for w in range_constexpr(1, WAVES):
            kb, ib = ah._take(kb, ib, fx.Int32(fx.ptr_load(k.lds + (p.a_red + w * 2))),
                              fx.Int32(fx.ptr_load(k.lds + (p.a_red + w * 2 + 1))))  # fmt: skip
        fx.ptr_store(ib, k.lds + (p.a_red + 2 * WAVES))
        if t == 0:
            bo.buffer_store(ib, rsrc(k.args[f"{p.name}.ids"]), j)
