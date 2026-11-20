# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The target's LM head on this rank's vocab shard in one launch: the final
RMSNorm of the step's hidden rows, then ``logits = bf16(x . W[v])``.

CTA t < rows norms row t (``norm_row``: bf16((h rstd) w), rstd =
rsq(sum h^2 / K + eps), the original norm's rounding; x at device scope) and
flags it; every CTA streams its vocab rows' weights (``head_dot``: whole
cache lines a load) while it awaits the flags, lands x and puts each logit
to ``logits`` [rows, vocab] (bf16; the caller gathers the ranks' shards).
The sums run in another order than the original GEMM's: the logits agree up
to that rounding.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from atom.models.deepseek_v41.mono.kernels import vocab_argmax as va
from atom.models.deepseek_v41.mono.kernels.dims import Dims
from atom.models.deepseek_v41.mono.kernels.head_dot import (
    KSTEP,
    LDS_BYTES,
    RG,
    STAGE_WORDS,
    UNIT,
    VT,
    dots,
    row_rstd,
)
from atom.models.deepseek_v41.mono.kernels.mhc_cols import bf16_words, words4
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
    xshfl,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.sync import Mailbox, preg, publish
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

# the LM head's mailbox slot (its rows' flags): past every layer's
LM_SLOT = 195

ABI = KernelAbi(("hidden", "norm_w", "x", "w", "logits", "ids", "sym", "peers", "rank"))


@dataclass(frozen=True)
class LmHeadBuild:
    rows: int
    # this rank's vocab rows (a multiple of RG), the hidden width
    vocab: int
    hidden: int
    eps: float
    tp: int = 4
    # weight K steps in flight a wave (whole ``UNIT``s)
    prefetch: int = 16
    # each row's greedy id over every rank's vocab into ``ids`` instead of the
    # logits (``vocab_argmax`` on the bf16 logits: the lowest id of the
    # largest, as the sampler's argmax of the gathered logits)
    argmax: bool = False

    def __post_init__(self):
        assert self.vocab % RG == 0 and self.hidden % (KSTEP * UNIT) == 0
        assert self.prefetch % UNIT == 0 and self.hidden % 8 == 0
        assert self.rows <= self.tasks, "a norm row a CTA"
        assert self.lds_words * 4 <= LDS_BYTES, "no K slicing fits the LDS"
        assert (
            not self.argmax or self.rows * self.tp <= THREADS
        ), "a pick lane a (row, rank)"

    # ``head_dot``'s layout
    x_in_launch = True

    @property
    def blocks(self) -> int:
        """VT-row blocks a task: the fewest that keep a task a CTA."""
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
    def norm_rounds(self) -> int:
        """``norm_row``'s rounds of 8 columns a thread."""
        return -(-self.hidden // (8 * THREADS))

    @property
    def kparts(self) -> int:
        """K slices x lands in: the fewest (dividing the units) that fit."""
        units = self.steps // UNIT
        for kp in range(1, units + 1):
            if units % kp == 0 and self._words(kp) * 4 <= LDS_BYTES:
                return kp
        return units

    @property
    def psteps(self) -> int:
        return self.steps // self.kparts

    @property
    def pitch(self) -> int:
        """LDS words a row of x's slice (+4: bank spread)."""
        return self.psteps * KSTEP // 2 + 4

    # LDS words: x's slice, the norm's waves' sums, (``argmax``) each row's
    # waves' (key, id) bests, the staging tiles
    def _words(self, kp) -> int:
        x = self.rows * (self.steps // kp * KSTEP // 2 + 4)
        best = self.rows * WAVES * 2 if self.argmax else 0
        return -(-(x + WAVES + best) // 4) * 4 + STAGE_WORDS

    @property
    def a_red(self) -> int:
        return self.rows * self.pitch

    @property
    def a_best(self) -> int:
        return self.a_red + WAVES

    @property
    def a_stage(self) -> int:
        return self.lds_words - STAGE_WORDS

    @property
    def lds_words(self) -> int:
        return self._words(self.kparts)


@traced
def norm_row(c, t):
    """Row t of ``hidden`` -> bf16((h rstd) w) into x (device scope), then
    the row's flag. A thread 8 columns a round."""
    p, tid = c["key"], c["tid"]
    a = c["args"]
    cols = [tid * 8 + 8 * THREADS * k for k in range(p.norm_rounds)]
    lives = [col < p.hidden for col in cols]
    cols = [fx.min(col, fx.Int32(p.hidden - 8)) for col in cols]
    hws = [words4(a["hidden"], (t * p.hidden + col) // 2) for col in cols]
    hss = [[bf2_f32(hw[k // 2])[k % 2] for k in range(8)] for hw in hws]
    rstd = row_rstd(c, hss, lives)
    for k in range_constexpr(p.norm_rounds):
        norm_out(c, t, cols[k], lives[k], hss[k], rstd)
    publish(c["mb"].put, c["hxr"], t, 1, tid == 0)


@traced
def norm_out(c, t, col, live, hs, rstd):
    if live:
        nw = words4(c["args"]["norm_w"], col // 2)
        ys = [(hs[k] * rstd) * bf2_f32(nw[k // 2])[k % 2] for k in range(8)]
        at = (t * c["key"].hidden + col) // 2
        bo.buffer_store(bf16_words(ys), rsrc(c["args"]["x"]), at, cache_modifier=CM_DEV)


@traced
def norms(c):
    """CTA t < rows: row t (``norm_row``)."""
    if c["t"] < c["key"].rows:
        norm_row(c, c["t"])


@traced
def logit_put(c, blk, ch, vals):
    """``head_dot``'s 4 logits of a lane to ``logits`` (row lane % 16 + 16 ch,
    vocab rows 4 (lane // 16) .. + 3 of the wave's group), 8 bytes."""
    p, lane = c["key"], c["lane"]
    row = lane % 16 + RG * ch
    v0 = ((c["t"] * p.blocks + blk) * WAVES + c["wave"]) * RG + (lane // 16) * 4
    if (row < p.rows) & (v0 < p.vocab):
        words = fx.Vector.from_elements(
            [
                bf16_pair(vals[0], vals[1]).bitcast(fx.Int32),
                bf16_pair(vals[2], vals[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        bo.buffer_store(words, c["r_logits"], (row * p.vocab + v0) // 2)


def best_put(c, blk, ch, vals):
    """``head_dot``'s 4 logits of a lane into its running best of chunk ch
    (``c["best"][ch]``: blocks in vocab order, so a tie keeps the earlier,
    lower id)."""
    p, lane = c["key"], c["lane"]
    v0 = ((c["t"] * p.blocks + blk) * WAVES + c["wave"]) * RG + (lane // 16) * 4
    for e in range_constexpr(4):
        ok = v0 + e < p.vocab
        ke = ok.select(va.value_key(vals[e]), fx.Int32(0))
        ie = ok.select(c["first"] + v0 + e, fx.Int32(va.INT_MAX))
        got = c["best"][ch]
        c["best"][ch] = (ke, ie) if got is None else va.take(*got, ke, ie)


@traced
def task_bests(c):
    """Each row's best of this task: a chunk's lanes (xor 16, 32) -> the
    wave's, into LDS ``a_best`` [row][wave], then thread r < rows the waves'
    -> MTOP (row r, this task)."""
    p, lane = c["key"], c["lane"]
    for ch in range_constexpr(p.chunks):
        kk, ii = c["best"][ch]
        for off in (16, 32):
            kk, ii = va.take(kk, ii, xshfl(kk, off), xshfl(ii, off))
        row = lane + RG * ch
        if (lane < RG) & (row < p.rows):
            at = p.a_best + (row * WAVES + c["wave"]) * 2
            fx.ptr_store(kk, c["lds"] + at)
            fx.ptr_store(ii, c["lds"] + (at + 1))
    gpu.barrier()
    if c["tid"] < p.rows:
        kb, ib = va.red_waves(c, p.a_best + c["tid"] * WAVES * 2)
        va.task_push(c, c["tid"], c["t"], p.tasks, kb, ib)


@traced
def argmax(c):
    """The rows' ids over every rank (``vocab_argmax``): this task's bests,
    CTA r < rows row r's rank best, CTA 0 the pick into ``ids``."""
    p = c["key"]
    task_bests(c)
    if c["t"] < p.rows:
        va.rank_reduce(c, c["t"], p.tasks, p.rows, p.tp, p.a_best)
    if c["t"] == 0:
        b, ii, lead = va.pick(c, p.rows, p.tp, p.rows, lambda b: b)
        if lead:
            bo.buffer_store(ii, rsrc(c["args"]["ids"], p.rows * 4), b)


def build_lm_head(key: LmHeadBuild):
    keyed = key_tuple(key, SOURCES)
    peer = peer_offsets(key.rows, Dims(key.tp))
    name = kernel_symbol(
        "v41_lm_head", s=key.rows, n=key.vocab, tp=key.tp, am=int(key.argmax)
    )

    @fx.struct
    class Smem:
        words: fx.Array[fx.Int32, key.lds_words, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def lm_head(
        hidden: Int64, norm_w: Int64, x: Int64, w: Int64, logits: Int64,
        ids: Int64, sym: Int64, peers: Int64, rank: Int32,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek().words.ptr
        c = {
            "key": key, "tid": tid, "lane": tid % 64, "wave": tid // 64,
            "t": fx.block_idx.x, "lds": lds, "ldsf": fx.recast_iter(fx.Float32, lds),
            "args": {"hidden": hidden, "norm_w": norm_w, "x": x, "ids": ids},
            "r_w": rsrc(w, key.vocab * key.hidden * 2),
            "r_logits": rsrc(logits, key.rows * key.vocab * 2),
            "mb": Mailbox(LM_SLOT),
            "hxr": preg(sym, peer["hxr"], "hxr"),
        }  # fmt: skip
        norms(c)
        if const_expr(key.argmax):
            bases = peer_bases(peers, key.tp)
            c.update(
                rank=rank,
                first=rank * key.vocab,
                best=[None] * key.chunks,
                mtop=preg(sym, peer["mtop"], "mtop"),
                mrank=preg(sym, peer["mrank"], "mrank"),
                mrank_fan=[
                    preg(bases[r], peer["mrank"], "mrank") for r in range(key.tp)
                ],
            )
            dots(c, best_put)
            argmax(c)
        else:
            dots(c, logit_put)
        _ = keyed

    @launcher(name)
    def launch(
        hidden: Int64, norm_w: Int64, x: Int64, w: Int64, logits: Int64,
        ids: Int64, sym: Int64, peers: Int64, rank: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        lm_head(hidden, norm_w, x, w, logits, ids, sym, peers, rank).launch(
            grid=(key.tasks,), block=(THREADS,), stream=stream
        )

    ABI.check(lm_head, launch)
    return launch
