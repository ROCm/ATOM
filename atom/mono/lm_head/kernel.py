# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A model's LM head on this rank's vocab shard picking each row's id in one
launch: ``x`` the step's rows (``norm``: the final RMSNorm of ``hidden``,
else ``hidden`` itself), each row's id over every rank's vocab of
``bf16(x . W[v])`` -- the lowest id of the largest, as the sampler's argmax
of the gathered logits; a row of temperature T > ``greedy_t`` its draw of
softmax(logits / T) instead (``rq``), within its top-p set where top_p <
1 (``top_p``: its logits kept in the launch's ``lg``, drawn by ``helpers``
CTAs once its largest logit is known).

``norm``: CTA t < rows norms row t (``norm_row``: bf16((h rstd) w), rstd =
rsq(sum h^2 / K + eps), the original norm's rounding; x at device scope)
and flags it. Every CTA streams its vocab rows' weights (``dot``: whole
cache lines a load) while it awaits x, lands it, and keeps each row's best
(``argmax``: its task's, then its rank's, then every rank's).

The launch's hand-offs are its own buffer's (``peer_layout``), tagged with
the launch (``Mailbox.of_launch``: ``tag``, the verify launch count, + 1),
so nothing zeroes it between launches. Every region peers push into (MRANK;
a top-p row's CANDX, H1 and H2) is kept in two copies, a launch's by its
tag's parity: a rank's next launch pushes into the other copy, never into
one a slower peer may still be reading.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64

from atom.mono.device import race
from atom.mono.device.ops import (
    CM_DEV,
    LOG2E,
    bf2_f32,
    bf16_pair,
    bf16_words,
    hw_rcp,
    kernel_symbol,
    launcher,
    ld_i32,
    rsrc,
    traced,
    words4,
    xshfl,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, preg, publish, shift
from atom.mono.lm_head import argmax as va
from atom.mono.lm_head import top_p, top_p_candidates, top_p_scan
from atom.mono.lm_head import top_p_logits as tl
from atom.mono.lm_head.dot import (
    KSTEP,
    LDS_BYTES,
    RG,
    STAGE_WORDS,
    UNIT,
    VT,
    dots,
    row_rstd,
)
from atom.mono.plan.build_key import key_tuple, source_digest
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.step_begin import ZERO_ALIGN

_CURRENT_STREAM = fx.Stream(None)

# the kernel's sources, in every build's JIT key: the shared mono framework
SOURCES = source_digest("mono")

ABI = KernelAbi(
    (
        "hidden",
        "norm_w",
        "x",
        "w",
        "ids",
        "temps",
        "top_ps",
        "lg",
        "tag",
        "sym",
        "peers",
        "rank",
        "tl",
    )
)

# a ``timeline`` build's stamps, a CTA's point after each (a top-p row's
# steps, ``top_p_candidates``, on its lead and helpers)
TL_POINTS = (
    "start", "dot", "task_best", "rank_best", "select", "rank_list", "decide", "end",
)  # fmt: skip


def task_blocks(vocab: int) -> int:
    """VT-row blocks a task of a ``vocab``-row shard: the fewest that keep a
    task a CTA."""
    return -(-vocab // (VT * BLOCKS))


def task_count(vocab: int) -> int:
    """The launch's tasks (CTAs) over a ``vocab``-row shard."""
    return -(-vocab // (VT * task_blocks(vocab)))


@dataclass(frozen=True)
class LmHeadBuild:
    rows: int
    # this rank's vocab rows (a multiple of RG), the hidden width
    vocab: int
    hidden: int
    tp: int
    # the rows its buffer holds (``peer_layout``): every width's alike
    max_rows: int
    # the final RMSNorm of ``hidden`` first (its epsilon); else x is ``hidden``
    norm: bool = False
    eps: float = 0.0
    # weight K steps in flight a wave (whole ``UNIT``s)
    prefetch: int = 16
    # a row of temperature T > ``greedy_t`` draws its id
    # (``atom.mono.device.race``: the launch's ``tag``, key ``seed``), within
    # its top-p set where top_p < 1 (``top_p``), T and top_p its
    # request's ``temps`` / ``top_ps`` f32 entries, a request ``rq`` rows; 0:
    # every row greedy
    rq: int = 0
    greedy_t: float = 0.0
    seed: int = 0
    # per-CTA stamps at ``TL_POINTS`` into ``tl`` [CTA][points] (debug only)
    timeline: bool = False

    def __post_init__(self):
        assert self.vocab % RG == 0 and self.hidden % (KSTEP * UNIT) == 0
        assert self.prefetch % UNIT == 0 and self.hidden % 8 == 0
        assert self.rows <= self.max_rows
        assert self.rows <= self.tasks, "a norm row a CTA"
        assert self.lds_words * 4 <= LDS_BYTES, "no K slicing fits the LDS"
        assert self.rows * self.tp <= THREADS, "a pick lane a (row, rank)"

    @property
    def x_in_launch(self) -> bool:
        """``dot``'s layout: x made by this launch (``norm``)."""
        return self.norm

    @property
    def blocks(self) -> int:
        return task_blocks(self.vocab)

    @property
    def tasks(self) -> int:
        return task_count(self.vocab)

    @property
    def task_rows(self) -> int:
        """Vocab rows a task's dot covers (task t: from t ``task_rows``)."""
        return VT * self.blocks

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

    # LDS words: x's slice, the norm's waves' sums, each row's waves' (key,
    # id) bests, the staging tiles
    def _words(self, kp) -> int:
        x = self.rows * (self.steps // kp * KSTEP // 2 + 4)
        best = self.rows * WAVES * 2
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
        """The dot's words, or a top-p row's draws' (``top_p_scan.WORDS``,
        ``top_p_candidates.WORDS``, from word 0 once the dot is done) where
        more."""
        dot = self._words(self.kparts)
        if not self.rq:
            return dot
        draw_words = max(top_p_scan.WORDS, top_p_candidates.WORDS)
        return max(dot, -(-draw_words // 4) * 4)

    @property
    def fix_log2(self) -> int:
        """``top_p_scan``'s mass fixed point over every rank's vocab."""
        return top_p_scan.fix_log2(self.vocab * self.tp)

    @property
    def helpers(self) -> int:
        """CTAs a top-p row's draw is split over (``top_p_logits``)."""
        return tl.helpers(self.rows, self.tasks)

    @property
    def scan_loads(self) -> int:
        """16-byte loads a thread of a helper's slice of a top-p row's logits
        (``top_p_logits``)."""
        slice_quads = -(-(self.vocab // 8) // self.helpers)
        return -(-slice_quads // THREADS)


def peer_layout(
    max_rows: int, tasks: int, tp: int, draws: bool
) -> dict[str, tuple[int, int]]:
    """Region -> (byte offset, bytes) of the head's own peer buffer, 8 B a
    (word, tag) pair: HXR a row's x landed (``norm_row``), MTOP the tasks'
    (key, id) bests on this rank [row][task], MRANK every rank's [copy]
    [source rank][row]; ``draws`` (a drawing head), a top-p row's draw's:
    CANDX every rank's candidate lists [copy][source rank][row], this rank's
    CAND (a helper's list [CTA]) and GO ([row]) (``top_p_candidates``); H1 /
    H2, every rank's level-1 / level-2 records [copy][source rank][row],
    this rank's L1 / L2 (a helper's records [CTA]) and B1 ([row])
    (``top_p_scan``). A launch's copy of a region peers push into is its
    tag's parity."""
    cand, candx, go = top_p_candidates.region_pairs(tasks, max_rows, tp)
    h1, h2 = top_p_scan.region_pairs(max_rows, tp)
    l1, l2, b1 = top_p_scan.local_pairs(tasks, max_rows)
    sizes = [
        ("candx", 2 * candx), ("cand", cand), ("go", go),
        ("h1", 2 * h1), ("h2", 2 * h2), ("l1", l1), ("l2", l2), ("b1", b1),
    ]  # fmt: skip
    return pair_layout(
        [
            ("hxr", max_rows),
            ("mtop", max_rows * tasks * 2),
            ("mrank", 2 * tp * max_rows * 2),
            *((name, n if draws else 0) for name, n in sizes),
        ]
    )


def peer_bytes(max_rows: int, tasks: int, tp: int, draws: bool) -> int:
    """The buffer's bytes: past its last region, in whole ``ZERO_ALIGN``
    units."""
    layout = peer_layout(max_rows, tasks, tp, draws)
    end = max(off + n for off, n in layout.values())
    return -(-end // ZERO_ALIGN) * ZERO_ALIGN


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
def logit_put(c, ch, v0, vals):
    """A top-p row's 4 bf16 logits (``c["keep"][ch]``) to ``lg`` [rows]
    [vocab] at device scope (the row's draw reads them, ``top_p``)."""
    p = c["key"]
    row = c["lane"] % 16 + RG * ch
    if c["keep"][ch] & (v0 < p.vocab):
        words = fx.Vector.from_elements(
            [
                bf16_pair(vals[0], vals[1]).bitcast(fx.Int32),
                bf16_pair(vals[2], vals[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        bo.buffer_store(
            words, c["r_lg"], (row * p.vocab + v0) // 2, cache_modifier=CM_DEV
        )


def best_put(c, blk, ch, vals):
    """``dot``'s 4 logits of a lane into its running best of chunk ch
    (``c["best"][ch]``: blocks in vocab order, so a tie keeps the earlier,
    lower id); a drawing row's (``draw_params``) by their race scores, a
    top-p row's logits kept for its draw (``logit_put``)."""
    p, lane = c["key"], c["lane"]
    v0 = ((c["t"] * p.blocks + blk) * WAVES + c["wave"]) * RG + (lane // 16) * 4
    keys = [va.value_key(v) for v in vals]
    if const_expr(p.rq):
        logit_put(c, ch, v0, vals)
        rc = c["race"][ch]
        scores = race.race_scores(
            vals, c["first"] + v0, lane % 16 + RG * ch, rc, c["draw_tag"],
            *race.seed_words(p.seed), race.counter_shift(p.vocab * p.tp),
        )  # fmt: skip
        keys = [
            (rc > fx.Float32(0.0)).select(va.value_key(s), k)
            for s, k in zip(scores, keys)
        ]
    for e in range_constexpr(4):
        ok = v0 + e < p.vocab
        ke = ok.select(keys[e], fx.Int32(0))
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
    if const_expr(p.rq):
        # a top-p row's logits (``logit_put``) drained before its MTOP pair:
        # the draw reads them once every task's pair is in
        rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] < p.rows:
        kb, ib = va.red_waves(c, p.a_best + c["tid"] * WAVES * 2)
        va.task_push(c, c["tid"], c["t"], p.tasks, kb, ib)


@traced
def picks(c):
    """The rows' ids over every rank (``argmax``): this task's bests, CTA
    r < rows row r's rank best; a row that draws within its top-p, by its
    helpers (``top_p.draw_slot``); CTA 0 the other rows' picks into
    ``ids``."""
    p = c["key"]
    task_bests(c)
    c["stamp"]("task_best")
    if c["t"] < p.rows:
        va.rank_reduce(c, c["t"], p.tasks, p.max_rows, p.tp, p.a_best)
    c["stamp"]("rank_best")
    if const_expr(p.rq):  # noqa: SIM102 (traced)
        if c["t"] < p.rows * p.helpers:
            top_p.draw_slot(c)
    if c["t"] == 0:
        b, ii, lead = va.pick(c, p.rows, p.tp, p.max_rows, lambda b: b)
        if const_expr(p.rq):
            lead = lead & ~tl.draws_top_p(*tl.row_sampling(c, b), p.greedy_t)
        if lead:
            bo.buffer_store(ii, rsrc(c["args"]["ids"], p.rows * 4), b)


def draw_params(c):
    """A drawing build's per-lane inputs, loaded before the norms: per chunk
    the lane's row's race factor log2(e) / T (``c["race"]``, 0: greedy, T
    <= ``greedy_t``, a top-p row or past the rows) and whether the row draws
    within its top-p (``c["keep"]``: its logits kept, ``logit_put``)."""
    p, lane = c["key"], c["lane"]
    c["race"], c["keep"] = [], []
    for ch in range_constexpr(p.chunks):
        row = lane % 16 + RG * ch
        t, top = tl.row_sampling(c, fx.min(row, fx.Int32(p.rows - 1)))
        live = row < p.rows
        keep = live & tl.draws_top_p(t, top, p.greedy_t)
        races = live & (t > fx.Float32(p.greedy_t)) & ~keep
        c["race"].append(races.select(fx.Float32(LOG2E) * hw_rcp(t), fx.Float32(0.0)))
        c["keep"].append(keep)


def build_lm_head(key: LmHeadBuild):
    keyed = key_tuple(key, SOURCES)
    draws = key.rq > 0
    layout = peer_layout(key.max_rows, key.tasks, key.tp, draws)
    peer = {name: off for name, (off, _) in layout.items()}
    # one copy's bytes of each two-copy region
    mrank_copy = key.tp * key.max_rows * 2 * 8
    h1_copy, h2_copy = (8 * n for n in top_p_scan.region_pairs(key.max_rows, key.tp))
    candx_copy = 8 * top_p_candidates.region_pairs(key.tasks, key.max_rows, key.tp)[1]
    name = kernel_symbol(
        "mono_lm_head", s=key.rows, n=key.vocab, h=key.hidden, tp=key.tp,
        nm=int(key.norm), rq=key.rq, tl=int(key.timeline),
    )  # fmt: skip
    points = len(TL_POINTS)

    @fx.struct
    class Smem:
        words: fx.Array[fx.Int32, key.lds_words, 16]
        tls: fx.Array[fx.Int64, points if key.timeline else 1, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def lm_head(
        hidden: Int64, norm_w: Int64, x: Int64, w: Int64, ids: Int64,
        temps: Int64, top_ps: Int64, lg: Int64, tag: Int64, sym: Int64,
        peers: Int64, rank: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        smem = fx.SharedAllocator().allocate(Smem).peek()
        lds, tls = smem.words.ptr, smem.tls.ptr
        stamp_begin(key.timeline, tls, tid, points)
        # the verify launch count (``verify_ids.IdVerifiers.launches``): the
        # draws' tag, and + 1 (nonzero) the hand-offs'
        draw_tag = ld_i32(tag, 0)
        launch_tag = (draw_tag & fx.Int32(0x7FFFFFFF)) + fx.Int32(1)
        parity = launch_tag & fx.Int32(1)
        bases = peer_bases(peers, key.tp)

        def copies(region, copy_bytes):
            """This rank's and every rank's ``region``, the launch's copy."""
            at = fx.Int64(parity * fx.Int32(copy_bytes))
            own = shift(preg(sym, peer[region], region), at)
            fan = [shift(preg(b, peer[region], region), at) for b in bases]
            return own, fan

        mrank, mrank_fan = copies("mrank", mrank_copy)
        c = {
            "key": key, "tid": tid, "lane": tid % 64, "wave": tid // 64,
            "t": fx.block_idx.x, "lds": lds, "ldsf": fx.recast_iter(fx.Float32, lds),
            "args": {
                "hidden": hidden, "norm_w": norm_w, "ids": ids, "temps": temps,
                "top_ps": top_ps, "x": x if const_expr(key.norm) else hidden,
            },
            "r_w": rsrc(w, key.vocab * key.hidden * 2),
            "mb": Mailbox.of_launch(launch_tag),
            "hxr": preg(sym, peer["hxr"], "hxr"),
            "draw_tag": draw_tag, "rank": rank, "first": rank * key.vocab,
            "best": [None] * key.chunks,
            "mtop": preg(sym, peer["mtop"], "mtop"),
            "mrank": mrank, "mrank_fan": mrank_fan,
        }  # fmt: skip
        if const_expr(draws):
            c["r_lg"] = rsrc(lg, key.rows * key.vocab * 2)
            c["h1"], c["h1_fan"] = copies("h1", h1_copy)
            c["h2"], c["h2_fan"] = copies("h2", h2_copy)
            c["candx"], c["candx_fan"] = copies("candx", candx_copy)
            for region in ("l1", "l2", "b1", "cand", "go"):
                c[region] = preg(sym, peer[region], region)
            draw_params(c)
        c["stamp"] = lambda k: stamp(key.timeline, tls, tid, TL_POINTS.index(k))
        if const_expr(key.norm):
            norms(c)
        dots(c, best_put)
        c["stamp"]("dot")
        picks(c)
        c["stamp"]("end")
        stamp_flush(key.timeline, tls, tl, tid, fx.block_idx.x, points)
        _ = keyed

    @launcher(name)
    def launch(
        hidden: Int64, norm_w: Int64, x: Int64, w: Int64, ids: Int64,
        temps: Int64, top_ps: Int64, lg: Int64, tag: Int64, sym: Int64,
        peers: Int64, rank: Int32, tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        lm_head(
            hidden, norm_w, x, w, ids, temps, top_ps, lg, tag, sym, peers, rank,
            tl,
        ).launch(grid=(key.tasks,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(lm_head, launch)
    return launch
