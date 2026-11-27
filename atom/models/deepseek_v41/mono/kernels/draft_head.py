# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The DSpark draft head in one launch: the LM head's logits of every block
position, then the block sampled position by position with the Markov head,
the argmax taken over every rank's vocab shard by tagged pairs.

Per request b and block position j (``x_{-1}``: the request's anchor)::

    x_j = argmax_v ( bf16(x[b, j] . W[v]) + W1[x_{j-1}] . W2[v] )

``W`` this rank's vocab rows ``n`` (ids ``first .. first + n``, row-major
[n, K] bf16: the target's LM head shard as loaded), ``W1`` / ``W2`` the whole
[V, r] bf16 Markov tables. The positions are sequential: each one's bias needs
the previous one's id, the argmax over every rank.

One stage, a task a CTA of ``blocks`` blocks of ``VT`` ids (a block 8 row
groups of 16, a wave each); every task resident, each waits on all of them:

0. ``from_state``: CTA t < rows makes row t of x from the backbone's last
   stage (``tail_row``: the owed post, the collapse and the final norm, each
   rounded as the original) and flags it; the other rows are awaited past
   each CTA's first weight loads;
1. the dot, a block at a time: x [rows, K] (``x``, plain bf16, every
   request's positions) in LDS -- whole, or in ``kparts`` K slices landed in
   turn -- the wave's 16 W rows streamed over K in units of 128 elements
   (whole cache lines a load, through the wave's LDS tile), one bf16 MFMA a
   32-element step and 16-row chunk of x (A = the 16 ids, B = the rows: the same
   products and K order at any slicing); each logit rounded to bf16 into LDS
   [row][id of the task];
2. per position j: each request's ``W1[x_{j-1}]`` into LDS (task 0 also into
   ``embeds`` [rows][r], the confidence head's input); the biases of a
   wave's 16 ids, every request at once: a bf16 MFMA a 32-wide Markov K step
   (A = the ids' ``W2`` rows, from LDS or L2; B = the requests' W1 rows), val
   = base + bias; the task's best (largest val, then lowest id: a total
   order) to its rank, CTA b of every rank reduces request b's and puts the
   rank's best to every rank, and every CTA of every rank reduces the ranks'
   bests alike, so all pick the same id; after the last position task 0
   writes every row's to ``ids`` [rows] (int32, as the anchors).

Numerics: the original path (``dspark_markov_argmax_tp``) sums the same exact
bf16 products in another fp32 order and its base logits come from another
GEMM's order: ids agree except where two candidates are within that rounding.
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.kernels.mailbox import Mailbox, preg
from atom.models.deepseek_v41.mono.kernels.mhc_cols import (
    collapsed8,
    seam_mixes,
    settled8,
)
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    LOG2E,
    bf2_f32,
    bf16_words,
    butterfly,
    hw_exp2,
    kernel_symbol,
    launcher,
    mfma_bf16,
    rsrc,
    traced,
    words4,
    xshfl,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import publish
from atom.mono.lm_head import argmax as va
from atom.mono.lm_head.dot import (
    KSTEP,
    LDS_BYTES,
    RG,
    STAGE_WORDS,
    UNIT,
    VT,
    dots,
    lds_quad,
    row_rstd,
)
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

# the head launch's mailbox tag: past every layer's K1 / K2a / K2b
HEAD_TAG = 192

# ``confidence``'s arguments ``cw`` / ``conf``: the confidence head's weight
# [hidden + r] (bf16, no bias) and [rows] its fp32 output. Else 0.
# ``from_state``'s ``res`` .. ``hidden``: the last stage's residual [rows, HC,
# K] (before its owed post), its FFN output [rows, K] (bf16), the post [rows,
# HC], comb [rows, HC, HC] and next pre [rows, HC] mixes (fp32), the final
# norm's weight [K] (bf16), and [rows, K] the collapsed rows out (bf16); x is
# then where the normed rows land (device scope). Else 0.
ABI = KernelAbi(
    (
        "x", "w", "w1", "w2", "anchor", "ids", "embeds", "res", "sub", "post",
        "comb", "pre", "norm_w", "hidden", "cw", "conf", "first", "sym", "peers",
        "rank", "tl",
    )
)  # fmt: skip
# ``timeline`` stamps: the start, the tail rows (``from_state``), the dot, the
# W2 rows, then per position its W1 rows, its bias and waves' bests, its push,
# its pick
TL_HEAD = ("tail", "dot", "w2")
TL_POSITION = ("w1", "bias", "push", "reduce", "pick")


def tl_points(positions: int) -> int:
    return 1 + len(TL_HEAD) + positions * len(TL_POSITION)


@dataclass(frozen=True)
class DraftHeadBuild:
    requests: int
    positions: int
    # this rank's vocab rows (a multiple of RG), the hidden width, the Markov rank
    vocab: int
    hidden: int
    rank_r: int
    tp: int = 4
    # weight K steps in flight a wave (whole ``UNIT``s)
    prefetch: int = 16
    # x from the backbone's last stage (``tail_row``: its owed post settled,
    # the streams collapsed, the final RMSNorm with epsilon ``eps``), not given
    from_state: bool = False
    eps: float = 0.0
    # the confidence head after the last position (``confidence_row``):
    # sigmoid(cw . [hidden ; embeds]) a row, from ``from_state``'s rows
    confidence: bool = False
    timeline: bool = field(default=False, metadata={"sym": "tl"})

    def refusal(self) -> str | None:
        """Why the launch is not built for this key (None: it is)."""
        if self.vocab % RG or self.hidden % (KSTEP * UNIT):
            return "vocab rows a row group, hidden whole weight units"
        if self.prefetch % UNIT or self.rank_r % KSTEP:
            return "prefetch whole units, the Markov rank whole K steps"
        if self.requests > RG:
            return "a request a column of the bias MFMA"
        if self.tp & (self.tp - 1):
            return "a request's ranks: xor lanes"
        if self.requests * self.rank_r // 8 > THREADS:
            return "a W1 load a thread"
        if self.lds_words * 4 > LDS_BYTES:
            return "no K slicing fits the LDS"
        if self.from_state and (self.hidden % 8 or self.rows > self.tasks):
            return "a tail_row thread 8 columns a round, a tail row a CTA"
        if self.confidence and not self.from_state:
            return "a row's hidden on its CTA"
        return None

    @property
    def rows(self) -> int:
        return self.requests * self.positions

    @property
    def tail_rounds(self) -> int:
        """``tail_row``'s rounds of 8 columns a thread."""
        return -(-self.hidden // (8 * THREADS))

    @property
    def blocks(self) -> int:
        """VT-id blocks a task: the fewest that keep a task a CTA."""
        return -(-self.vocab // (VT * BLOCKS))

    @property
    def tasks(self) -> int:
        return -(-self.vocab // (VT * self.blocks))

    @property
    def chunks(self) -> int:
        return -(-self.rows // RG)

    @property
    def steps(self) -> int:
        return self.hidden // KSTEP

    @property
    def w2_lds(self) -> bool:
        """The task's W2 rows held in LDS over x once the dot is done (where
        some K slicing fits them); else read from L2 at every position."""
        return self._kparts(True) is not None

    @property
    def kparts(self) -> int:
        """K slices x lands in: the fewest (dividing the K steps) that fit."""
        return self._kparts(self.w2_lds)

    def _kparts(self, w2_lds):
        units = self.steps // UNIT
        for kp in range(1, units + 1):
            if units % kp == 0 and self._words(kp, w2_lds) * 4 <= LDS_BYTES:
                return kp
        return None if w2_lds else units

    @property
    def w2pitch(self) -> int:
        """LDS words a W2 row (+4: bank spread)."""
        return self.rw + 4

    @property
    def psteps(self) -> int:
        return self.steps // self.kparts

    @property
    def pitch(self) -> int:
        """LDS words a row of x's slice (+4: the 16 rows of an operand load on
        distinct banks)."""
        return self.psteps * KSTEP // 2 + 4

    @property
    def rw(self) -> int:
        """32-bit words a Markov table row."""
        return self.rank_r // 2

    # LDS words: x's slice (the W2 rows over it, ``w2_lds``); the base logits
    # [row][id of the task]; each request's W1 row; the waves' (key, id) a
    # request; each request's x_{j-1}; every row's id
    def _words(self, kp, w2_lds) -> int:
        x = self.rows * (self.steps // kp * KSTEP // 2 + 4)
        if w2_lds:
            x = max(x, self.blocks * VT * (self.rw + 4))
        tail = -(-(x + self.rows * VT * self.blocks + self._tail) // 4) * 4
        return tail + STAGE_WORDS

    @property
    def _tail(self) -> int:
        return self.requests * (self.rw + WAVES * 2 + 1) + self.rows

    @property
    def a_base(self) -> int:
        w2 = self.blocks * VT * self.w2pitch if self.w2_lds else 0
        return max(self.rows * self.pitch, w2)

    @property
    def a_w1(self) -> int:
        return self.a_base + self.rows * VT * self.blocks

    @property
    def a_red(self) -> int:
        return self.a_w1 + self.requests * self.rw

    @property
    def a_prev(self) -> int:
        return self.a_red + self.requests * WAVES * 2

    @property
    def a_ids(self) -> int:
        """Every row's picked id, written out at the end (``ids_out``)."""
        return self.a_prev + self.requests

    @property
    def a_stage(self) -> int:
        """Each wave's staging tile of a weight unit (``stage_unit``), 16 B
        aligned, past everything else."""
        return self.lds_words - STAGE_WORDS

    @property
    def x_in_launch(self) -> bool:
        """x made by this launch (``tail_row``): ``lm_head.dot`` reads it after
        the rows' flags, at device scope."""
        return self.from_state

    @property
    def lds_words(self) -> int:
        return self._words(self.kparts, self.w2_lds)


# ---------------------------------------------------------------- the tail
@traced
def tail_row(c, t):
    """``from_state``: row t of the backbone's last stage -> its owed post
    settled and the streams collapsed (``mhc_cols``, to ``hidden``) and the
    final RMSNorm (bf16((h rstd) w), to x at device scope), then the row's
    flag. A thread 8 columns a round (``cols``)."""
    p, tid = c["key"], c["tid"]
    a = c["args"]
    mix = seam_mixes(a["post"], a["comb"], a["pre"], t)
    cols = [tid * 8 + 8 * THREADS * k for k in range(p.tail_rounds)]
    lives = [col < p.hidden for col in cols]
    cols = [fx.min(col, fx.Int32(p.hidden - 8)) for col in cols]
    hss = [
        collapsed8(settled8(a["res"], a["sub"], p.hidden, t, col, mix), mix)
        for col in cols
    ]
    rstd = row_rstd(c, hss, lives)
    for k in range_constexpr(p.tail_rounds):
        tail_out(c, t, cols[k], lives[k], hss[k], rstd)
    publish(c["mb"].put, c["hxr"], t, 1, tid == 0)


@traced
def tail_out(c, t, col, live, hs, rstd):
    """Row t's columns col .. + 7: collapsed to ``hidden``, normed to x."""
    if live:
        nw = words4(c["args"]["norm_w"], col // 2)
        ys = [(hs[k] * rstd) * bf2_f32(nw[k // 2])[k % 2] for k in range(8)]
        at = (t * c["key"].hidden + col) // 2
        bo.buffer_store(bf16_words(hs), rsrc(c["args"]["hidden"]), at)
        x = rsrc(c["args"]["x"])
        bo.buffer_store(bf16_words(ys), x, at, cache_modifier=CM_DEV)


@traced
def tail(c):
    """``from_state``: CTA t < rows its row t (``tail_row``)."""
    if c["t"] < c["key"].rows:
        tail_row(c, c["t"])


# ---------------------------------------------------------------- the dot
@traced
def base_put(c, blk, ch, vals):
    """``lm_head.dot``'s logits of a chunk into LDS [row][id of the task]."""
    p, lane = c["key"], c["lane"]
    row = lane % 16 + RG * ch
    if row < p.rows:
        at = (
            p.a_base + row * (VT * p.blocks) + blk * VT + c["wave"] * RG
            + (lane // 16) * 4
        )  # fmt: skip
        for e in range_constexpr(4):
            fx.ptr_store(vals[e], c["ldsf"] + (at + e))


# ---------------------------------------------------------------- positions
@traced
def anchors(c):
    """Each request's anchor (int32) into LDS ``a_prev``, four a 16-byte load
    (the words past the buffer: 0, not stored)."""
    p, tid = c["key"], c["tid"]
    if tid < -(-p.requests // 4):
        a = fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["anchor"], p.requests * 4), tid * 4, vec_width=4,
                dtype=T.i32,
            )
        )  # fmt: skip
        for e in range_constexpr(4):
            put_live(
                c["lds"] + (p.a_prev + 4 * tid + e), a[e], 4 * tid + e < p.requests
            )


@traced
def put_live(ptr, v, live):
    if live:
        fx.ptr_store(v, ptr)


@traced
def w1_rows(c, j):
    """Every request's ``W1[x_{j-1}]`` (its x_{j-1} from LDS ``a_prev``) into
    its LDS row, one 16-byte load a thread, every request's in one round
    trip; task 0 also into ``embeds`` at row (b, j)."""
    p, tid = c["key"], c["tid"]
    per = p.rw // 4
    if tid < p.requests * per:
        b, k = tid // per, tid % per
        xp = fx.Int32(fx.ptr_load(c["lds"] + (p.a_prev + b)))
        got = fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["w1"]), xp * p.rw + k * 4, vec_width=4, dtype=T.i32
            )
        )
        fx.ptr_store(got.bitcast(fx.Float32), c["ldsf"] + (p.a_w1 + b * p.rw + k * 4))
        if c["t"] == 0:
            # device scope: the confidence stage reads it on every CTA
            row = b * p.positions + j
            bo.buffer_store(
                got, rsrc(c["args"]["embeds"]), row * p.rw + k * 4,
                cache_modifier=CM_DEV,
            )  # fmt: skip


def w2_quads(c, blk):
    """Lane ``lane``'s 16 B of each Markov K step of ``W2`` row ``lane % 16``
    of its wave's row group in block ``blk`` (the MFMA's A): from LDS
    (``w2_lds``), else from L2 (a row past ``vocab``: the last one)."""
    p, lane = c["key"], c["lane"]
    v = blk * VT + c["wave"] * RG + lane % 16
    kq = (lane // 16) * 4
    steps = range(0, p.rw, KSTEP // 2)
    if const_expr(p.w2_lds):
        return [lds_quad(c, v * p.w2pitch + s + kq) for s in steps]
    lid = fx.min(c["t"] * (p.blocks * VT) + v, fx.Int32(p.vocab - 1))
    at = (c["first"] + lid) * p.rw + kq
    return [
        fx.Vector(bo.buffer_load(c["r_w2"], at + s, vec_width=4, dtype=T.i32))
        for s in steps
    ]


@traced
def w2_tile(c):
    """``w2_lds``: the task's W2 rows (a row past ``vocab``: the last one)
    into LDS over x, 16 B a load, every load of a round issued before its
    stores."""
    p, tid = c["key"], c["tid"]
    per = p.rw // 4
    total = p.blocks * VT * per
    for k0 in range_constexpr(0, -(-total // THREADS), 8):
        got = []
        for k in range_constexpr(k0, min(k0 + 8, -(-total // THREADS))):
            i = fx.min(fx.Int32(k * THREADS) + tid, fx.Int32(total - 1))
            v, w4 = i // per, i % per
            lid = c["t"] * (p.blocks * VT) + v
            gid = c["first"] + fx.min(lid, fx.Int32(p.vocab - 1))
            load = bo.buffer_load(
                c["r_w2"], gid * p.rw + w4 * 4, vec_width=4, dtype=T.i32
            )
            got.append((v, w4, fx.Vector(load)))
        for v, w4, g in got:
            fx.ptr_store(g, c["lds"] + (v * p.w2pitch + w4 * 4))


def bias(c, w2):
    """The block's biases ``W1[x_{j-1}] . W2[id]``: one bf16 MFMA a Markov K
    step (A = the row group's W2 rows, B = the requests' W1 rows, a column a
    request), lane ``lane``'s ids 4 (lane // 16) .. + 3 of request lane % 16."""
    p, lane = c["key"], c["lane"]
    b = fx.min(lane % 16, fx.Int32(p.requests - 1))
    w1 = p.a_w1 + b * p.rw + (lane // 16) * 4
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for k in range_constexpr(len(w2)):
        bq = lds_quad(c, w1 + k * (KSTEP // 2))
        acc = mfma_bf16(w2[k].bitcast(fx.BFloat16), bq.bitcast(fx.BFloat16), acc)
    return acc


@traced
def wave_bests(c, kk, ii):
    """Lane b < requests: request b's wave best into LDS ``a_red``."""
    if c["lane"] < c["key"].requests:
        at = c["key"].a_red + (c["lane"] * WAVES + c["wave"]) * 2
        fx.ptr_store(kk, c["lds"] + at)
        fx.ptr_store(ii, c["lds"] + (at + 1))


@traced
def task_push(c, j):
    """Thread b < requests: request b's task best of position j into this
    rank's MTOP (row (b, j), task t)."""
    p, tid = c["key"], c["tid"]
    if tid < p.requests:
        kb, ib = va.red_waves(c, p.a_red + tid * WAVES * 2)
        va.task_push(c, tid * p.positions + j, c["t"], p.tasks, kb, ib)


@traced
def rank_reduce(c, j):
    """CTA b < requests: request b's task bests of position j on this rank
    -> the rank's best, to every rank's MRANK (this rank, row (b, j)); its
    waves' bests in ``a_red``, once every request's was read (``task_push``)."""
    p = c["key"]
    if c["t"] < p.requests:
        row = c["t"] * p.positions + j
        va.rank_reduce(c, row, p.tasks, p.rows, p.tp, p.a_red)


@traced
def pick(c, j):
    """Every rank's best of each request (MRANK, lane b tp + src) -> each
    request's x_j into LDS (``a_prev``, and its row of ``a_ids``): the same
    on every CTA of every rank."""
    p = c["key"]
    b, ii, lead = va.pick(c, p.requests, p.tp, p.rows, lambda b: b * p.positions + j)
    if lead:
        fx.ptr_store(ii, c["lds"] + (p.a_prev + b))
        fx.ptr_store(ii, c["lds"] + (p.a_ids + b * p.positions + j))


@traced
def position(c, j):
    """Position j of every request given its x_{j-1} (LDS ``a_prev``) -> its
    x_j there (module docstring, 2.)."""
    p = c["key"]
    k0 = 1 + len(TL_HEAD) + j * len(TL_POSITION)
    w1_rows(c, j)
    gpu.barrier()
    c["stamp"](k0)
    lane = c["lane"]
    b = fx.min(lane % 16, fx.Int32(p.requests - 1))
    live = lane % 16 < p.requests
    kk, ii = None, None
    for blk in range_constexpr(p.blocks):
        acc = bias(c, w2_quads(c, blk))
        v0 = blk * VT + c["wave"] * RG + (lane // 16) * 4
        row = b * p.positions + j
        base = lds_quad(c, p.a_base + row * (VT * p.blocks) + v0).bitcast(fx.Float32)
        for e in range_constexpr(4):
            lid = c["t"] * (p.blocks * VT) + v0 + e
            ok = live & (lid < p.vocab)
            ke = ok.select(va.value_key(base[e] + acc[e]), fx.Int32(0))
            ie = ok.select(lid + c["first"], fx.Int32(va.INT_MAX))
            if const_expr(kk is None):
                kk, ii = ke, ie
            else:
                kk, ii = va.take(kk, ii, ke, ie)
    # the lanes of a request: xor 16, 32
    for off in (16, 32):
        kk, ii = va.take(kk, ii, xshfl(kk, off), xshfl(ii, off))
    wave_bests(c, kk, ii)
    if const_expr(p.confidence):
        # task 0's ``embeds`` stores drained before its bests are pushed: the
        # confidence stage on every CTA reads them once the picks are in
        # (``publish``: a barrier does not wait for them)
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    c["stamp"](k0 + 1)
    # the argmax in two hops: the tasks' bests to their rank, a rank's best
    # (one CTA a request) to every rank
    task_push(c, j)
    c["stamp"](k0 + 2)
    rank_reduce(c, j)
    c["stamp"](k0 + 3)
    pick(c, j)
    gpu.barrier()
    c["stamp"](k0 + 4)


@traced
def positions(c):
    """The positions in a loop (a position's registers not live into the
    next)."""
    for j in range(c["key"].positions):
        position(c, j)


@traced
def ids_out(c):
    """Task 0: every row's id into ``ids`` (int32), four rows a 16-byte store
    (the words past the buffer: dropped)."""
    p, tid = c["key"], c["tid"]
    if (c["t"] == 0) & (tid < -(-p.rows // 4)):
        word = fx.Vector.from_elements(
            [
                fx.Int32(
                    fx.ptr_load(c["lds"] + (p.a_ids + fx.min(4 * tid + e, p.rows - 1)))
                )
                for e in range(4)
            ],
            fx.Int32,
        )
        bo.buffer_store(word, rsrc(c["args"]["ids"], p.rows * 4), tid * 4)


@traced
def confidence_row(c, t):
    """Row t's confidence: sigmoid(cw . [hidden[t] ; embeds[t]]), fp32 over
    the bf16 operands (``DSparkConfidenceHead`` without a bias), to ``conf``.
    ``hidden[t]`` is this CTA's own (``tail_row``); ``embeds[t]`` task 0's, at
    device scope, before the pushes every later position's pick waited on."""
    p, tid, lane = c["key"], c["tid"], c["lane"]
    a = c["args"]
    s = fx.Float32(0.0)
    for k in range_constexpr(p.tail_rounds):
        col = tid * 8 + 8 * THREADS * k
        live = col < p.hidden
        col = fx.min(col, fx.Int32(p.hidden - 8))
        hs = words4(a["hidden"], (t * p.hidden + col) // 2)
        ws = words4(a["cw"], col // 2)
        for e in range_constexpr(8):
            h, w = bf2_f32(hs[e // 2])[e % 2], bf2_f32(ws[e // 2])[e % 2]
            s = s + live.select(h * w, fx.Float32(0.0))
    # the Markov embedding: 8 values a thread of the first rank_r / 8
    mine = tid < p.rank_r // 8
    ec = fx.min(tid, fx.Int32(p.rank_r // 8 - 1)) * 8
    es = fx.Vector(
        bo.buffer_load(
            rsrc(a["embeds"]), (t * p.rank_r + ec) // 2, vec_width=4,
            dtype=T.i32, cache_modifier=CM_DEV,
        )
    )  # fmt: skip
    wes = words4(a["cw"], (p.hidden + ec) // 2)
    for e in range_constexpr(8):
        v, w = bf2_f32(es[e // 2])[e % 2], bf2_f32(wes[e // 2])[e % 2]
        s = s + mine.select(v * w, fx.Float32(0.0))
    s = butterfly(s, (1, 2, 4, 8, 16, 32))
    if lane == 0:
        fx.ptr_store(s, c["ldsf"] + (p.a_red + c["wave"]))
    gpu.barrier()
    if tid == 0:
        z = fx.ptr_load(c["ldsf"] + p.a_red)
        for w in range_constexpr(1, WAVES):
            z = z + fx.ptr_load(c["ldsf"] + (p.a_red + w))
        e = hw_exp2(z * fx.Float32(-LOG2E))
        bo.buffer_store(fx.Float32(1.0) / (fx.Float32(1.0) + e), rsrc(a["conf"]), t)


@traced
def confidence(c):
    """``confidence``: CTA t < rows its row t (``confidence_row``), past the
    last position's pick (every row's embedding written)."""
    if c["t"] < c["key"].rows:
        confidence_row(c, c["t"])


def build_draft_head(key: DraftHeadBuild):
    assert key.refusal() is None, key.refusal()
    keyed = key_tuple(key, SOURCES)
    peer = peer_offsets(key.rows, Dims(key.tp))

    points = tl_points(key.positions)

    @fx.struct
    class Smem:
        words: fx.Array[fx.Int32, key.lds_words, 16]
        tls: fx.Array[fx.Int64, points if key.timeline else 1, 16]

    name = kernel_symbol(
        "v41_draft_head", b=key.requests, t=key.positions, n=key.vocab, tp=key.tp,
        fs=int(key.from_state), cf=int(key.confidence), tl=int(key.timeline),
    )  # fmt: skip

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def draft_head(
        x: Int64, w: Int64, w1: Int64, w2: Int64, anchor: Int64, ids: Int64,
        embeds: Int64, res: Int64, sub: Int64, post: Int64, comb: Int64,
        pre: Int64, norm_w: Int64, hidden: Int64, cw: Int64, conf: Int64,
        first: Int32, sym: Int64, peers: Int64, rank: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        smem = fx.SharedAllocator().allocate(Smem).peek()
        lds, tls = smem.words.ptr, smem.tls.ptr
        bases = peer_bases(peers, key.tp)
        c = {
            "key": key, "tid": tid, "lane": tid % 64, "wave": tid // 64,
            "t": fx.block_idx.x, "first": first, "rank": rank,
            "lds": lds, "ldsf": fx.recast_iter(fx.Float32, lds),
            "args": {"x": x, "w": w, "w1": w1, "anchor": anchor, "ids": ids,
                     "embeds": embeds, "res": res, "sub": sub, "post": post,
                     "comb": comb, "pre": pre, "norm_w": norm_w,
                     "hidden": hidden, "cw": cw, "conf": conf},
            "r_w": rsrc(w, key.vocab * key.hidden * 2),
            "r_w2": rsrc(w2),
            "mb": Mailbox(HEAD_TAG),
            "mtop": preg(sym, peer["mtop"], "mtop"),
            "mrank": preg(sym, peer["mrank"], "mrank"),
            "mrank_fan": [
                preg(bases[r], peer["mrank"], "mrank") for r in range(key.tp)
            ],
            "hxr": preg(sym, peer["hxr"], "hxr"),
        }  # fmt: skip
        c["stamp"] = lambda k: stamp(key.timeline, tls, tid, k)
        stamp_begin(key.timeline, tls, tid, points)
        if const_expr(key.from_state):
            tail(c)
        c["stamp"](1)
        dots(c, base_put)
        c["stamp"](2)
        if const_expr(key.w2_lds):
            gpu.barrier()  # every wave past x
            w2_tile(c)
        anchors(c)
        gpu.barrier()
        c["stamp"](3)
        positions(c)
        ids_out(c)
        if const_expr(key.confidence):
            confidence(c)
        stamp_flush(key.timeline, tls, tl, tid, fx.block_idx.x, points)
        _ = keyed

    @launcher(name)
    def launch(
        x: Int64, w: Int64, w1: Int64, w2: Int64, anchor: Int64, ids: Int64,
        embeds: Int64, res: Int64, sub: Int64, post: Int64, comb: Int64,
        pre: Int64, norm_w: Int64, hidden: Int64, cw: Int64, conf: Int64,
        first: Int32, sym: Int64, peers: Int64, rank: Int32, tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        draft_head(
            x, w, w1, w2, anchor, ids, embeds, res, sub, post, comb, pre, norm_w,
            hidden, cw, conf, first, sym, peers, rank, tl,
        ).launch(grid=(key.tasks,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(draft_head, launch)
    return launch
