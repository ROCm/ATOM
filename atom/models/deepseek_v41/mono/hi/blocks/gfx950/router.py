# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``MoERouter``: DeepSeek-V4.1's MoE gate (checkpoint ``inference/model.py`` ``Gate``), per row of the step.

    scores  = x . W^T                    fp32 (``GatheredDot``, dense: bf16 x and W, fp32 products and sums;
                                         ``kslices``: its K-split MFMA mode, 0 for the per-row lane loop)
    s       = sqrt(softplus(scores))     (``score_func = "sqrtsoftplus"``; softplus = x past 20, as torch;
                                         ``"sigmoid"``: s = 1 / (1 + exp(-scores)))
    indices = top-k of s + bias          (the bias steers the choice only; ``wave_topk_won``: ties to the lowest index)
    weights = s[indices] / (sum + 1e-20) * route_scale

``score_func = "softmax_topk"`` (the bias is part of the logit, top-k first, then the softmax over the winners only,
both rounded to bf16 as a bf16 ``F.linear`` / ``softmax(dtype=bf16)``)::

    logits  = bf16(scores + bias)
    indices = top-k of logits            (ties to the lowest index)
    weights = bf16(exp(l_i - max) / sum_won exp(l_j - max))      (fp32 inside, i over the winners)

``norm`` (a ``gathered.DotNorm``): x is the raw row and the dot normalizes its K slices itself (RMSNorm in the
score task's prologue, the norm stage's bits; GatheredDot's module docstring).

``scores`` (the dense K-split mode): "wave0" (wave 0 computes the row's n scores, n / 64 a lane, then the top-k)
or "waves" (wave w < n / 64 computes columns 64 w .., one a lane, s and the biased bits into LDS; wave 0 reads its
n / 64 a lane back and runs the same top-k and puts: the same operations on the same values, so the same bits; the
sqrt(softplus) is most of the route's compute on one wave). "fused": as "waves", but
wave w polls its own columns' slice partials and adds them in slice order in registers (no LDS staging of the
partials, no column-sum pass and its barriers): a wave computes its scores as soon as its columns have landed; the
same operations on the same values.

**TP column split** (``tp_cols=True``, ``Scenario.tp`` > 1): rank r's dot computes only experts ``r n / tp ..`` (its
rows of the table: ``pack_tp_table``; the dense K-split mode at C = n / tp), stage ``<name>.ag`` puts them into every
rank's peer region ``<name>.ag`` (slot r) and fills this rank's ``scores`` channel from the tp slots; the route stage
then runs as usual on every rank (the same scores, the same picks). The table is read once a step, by its tp owners.

**Top-k** (``topk``): "heads" (``wave_topk_won_full``: each lane sorts its n / 64, then k serial wave rounds of
the lanes' heads; ``wave_topk_won`` when k > n / 64) or "thresh" (``wave_topk_won_thresh``: a bound from the lanes'
maxima, the candidates above it ranked by a count; the same winners, ties to the lowest index; its own LDS words).
With ``tp_cols`` "thresh" also moves the score function to the gather's landing threads (each its expert's s and
biased bits, the 8 waves) instead of wave 0's n / 64 a lane: the same operations on the same values.

**In a consumer** (``MoEExperts(route_from=)``, softmax_topk with the dot's slice partials): the
consumer's CTA computes every row's picks itself from the partials, the route stage's arithmetic on the same values
(the same bits), into its LDS; the route stage still runs (its channels for other readers), off the consumer's path.

Outputs: ``idx`` (an F32 channel of width k carrying int32 expert ids, ascending: the order the reference's
combine adds experts in) and ``weights`` (an F32 channel of width k, the same order). Two stages: the scores
(``GatheredDot``) and ``<name>.route``: one wave a row (8 rows a task), each lane holding ``n / 64`` scores (in the
dense K-split mode a task a row: its 8 waves land the dot's slice partials in LDS and add them, no ``.sum``
stage and its round trip).
"""

from dataclasses import dataclass
from functools import partial

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import (
    fresh,
    hw_log2,
    popcount,
    rsrc,
    traced,
    wave_incl_scan,
    wave_sum,
)
from ...device.gfx950.topk import (
    WAVE_MAX_K,
    wave_topk_won,
    wave_topk_won_full,
)
from ...plan.program import Scenario, Stage
from ...target import MI355X
from ..base import Block, Channel, Fmt
from .gathered import DotNorm, GatheredDot

WAVES = MI355X.waves
LN2 = 0.6931471805599453


@dataclass(frozen=True)
class _Plan:
    name: str
    rows: int
    n: int  # experts
    k: int
    scores: str  # the F32 scores channel
    bias: str  # f32 [n] arg
    idx: str
    weights: str
    route_scale: float
    score_func: str
    part: str  # "": scores from the ``scores`` channel; else GatheredDot's slice partials (``slices`` of them)
    slices: int
    scores_by: str = (
        "wave0"  # the dense mode's s and biased scores: "wave0", "waves" or "fused" (module docstring)
    )
    thresh: bool = (
        False  # topk "thresh": ``wave_topk_won_thresh`` (the module docstring)
    )
    landed: bool = (
        False  # tp_cols gather, topk "thresh": s and the biased bits computed at landing, read from LDS
    )
    consumer: bool = (
        False  # the route in the consumer's rt (``MoEExperts(route_from=)``), a wave a row
    )
    hoist: bool = (
        False  # tp_cols gather: the peer bases before the remote puts (``MoERouter(tp_hoist=)``)
    )
    split_land: bool = (
        False  # tp_cols gather: the landing by the waves that did not put
    )


class MoERouter(Block):
    def __init__(self, name: str, x: Channel, weight: str, bias: str, idx: Channel, weights: Channel, n: int, k: int,
                 route_scale: float = 1.0, score_func: str = "sqrtsoftplus", kslices: int = 10,
                 dot_tiles: int = 1, norm: DotNorm | None = None, scores: str = "wave0",
                 k_dot: int | None = None, tp_cols: bool = False, tp_local: bool = False,
                 tp_push: bool = False, topk: str = "heads", consumer: bool = False,
                 tp_hoist: bool = False, tp_split_land: bool = False):  # fmt: skip
        if score_func not in ("sqrtsoftplus", "sigmoid", "softmax_topk"):
            raise NotImplementedError(f"MoERouter {name}: score_func {score_func!r} (sqrtsoftplus, sigmoid, "
                                      "softmax_topk)")  # fmt: skip
        if score_func == "softmax_topk" and scores != "wave0":
            raise ValueError(
                f"MoERouter {name}: softmax_topk computes its scores on wave 0 (scores='wave0')"
            )
        if n % 64 or n > 1024 or k > WAVE_MAX_K:
            raise ValueError(
                f"MoERouter {name}: n a multiple of 64 up to 1024, k <= {WAVE_MAX_K} (n {n}, k {k})"
            )
        for ch in (idx, weights):
            if ch.fmt is not Fmt.F32 or ch.width != k or ch.flagged:
                raise ValueError(
                    f"MoERouter {name}: idx / weights are tagged F32 channels of width {k}"
                )
        if scores not in ("wave0", "waves", "fused") or (
            scores != "wave0" and (not kslices or n // 64 > WAVES)
        ):
            raise ValueError(f"MoERouter {name}: scores 'wave0', 'waves' or 'fused' (kslices, n / 64 <= {WAVES}): "
                             f"{scores!r}")  # fmt: skip
        self.name, self.n, self.k, self.scores_by = name, n, k, scores
        self.scores = Channel(f"{name}.scores", n, Fmt.F32)
        self.tp_cols = tp_cols
        # tp_cols, sigmoid: each rank's top-k of its experts, then a top-k of the tp x k candidates
        self.tp_local = tp_local
        # tp_cols: the scores' puts by their own stage, a task a (row, destination rank): a CTA's remote puts no longer
        # go out one destination after another; the route stage then only gathers
        self.tp_push = tp_push
        # the top-k (module docstring): "heads" (sorted heads / scanning rounds) or "thresh" (a bound + a rank count)
        if topk not in ("heads", "thresh") or (
            topk == "thresh" and (score_func == "softmax_topk" or tp_local)
        ):
            raise ValueError(
                f"MoERouter {name}: topk 'heads' or 'thresh' (not softmax_topk / tp_local): {topk!r}"
            )
        self.topk = topk
        # the route computed by its consumer (``MoEExperts(route_from=)``): the dot's stages only
        # (every rank all n experts at TP; no route stage, idx / weights unwritten)
        self.consumer = consumer
        self.tp_hoist = tp_hoist  # tp_cols gather: the peer bases loaded before the remote puts (``_Plan.hoist``)
        self.tp_split_land = (
            tp_split_land  # tp_cols gather: the landing by the waves that did not put
        )
        if consumer and (tp_cols or not kslices or score_func == "softmax_topk"):
            raise ValueError(f"MoERouter {name}: consumer is a plain dot's (kslices, no tp_cols) sigmoid / "
                             "sqrtsoftplus route")  # fmt: skip
        if tp_local and (not tp_cols or score_func != "sigmoid" or WAVE_MAX_K < k):
            raise ValueError(f"MoERouter {name}: tp_local is a sigmoid tp_cols route")
        if tp_cols and (scores != "wave0" or k_dot is not None):
            raise ValueError(
                f"MoERouter {name}: tp_cols computes the scores on wave 0 from a plain dot"
            )
        self.norm = norm
        self.x, self.weight, self.kslices, self.dot_tiles = (
            x,
            weight,
            kslices,
            dot_tiles,
        )
        # k_dot: the weight's K when x is narrower (``GatheredDot(k=)``: x's elements past its width read as zero)
        self.dot = None if tp_cols else GatheredDot(f"{name}.dot", x, weight, self.scores, kslices=kslices,
                                                    sum_stage=False, tiles=dot_tiles, norm=norm, k=k_dot)  # fmt: skip
        self.bias, self.idx, self.weights = bias, idx, weights
        self.route_scale, self.score_func = route_scale, score_func

    def route_plan(self, sc: Scenario) -> _Plan:
        """The route's plan for ``sc`` (the route stage's, and a consumer's)."""
        part = self.dot.part.name if self.dot is not None and self.dot.part else ""
        return _Plan(self.name, sc.rows, self.n, self.k, self.scores.name, self.bias, self.idx.name, self.weights.name,
                     self.route_scale, self.score_func, part, self.dot.kslices if self.dot is not None else 0,
                     self.scores_by,
                     self.topk == "thresh", consumer=self.consumer)  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.route_plan(sc)
        part = p.part
        if p.scores_by != "wave0" and not part:
            raise ValueError(
                f"MoERouter {self.name}: scores {p.scores_by!r} needs the dot's slice partials (kslices)"
            )
        rname = f"{self.name}.route"
        writes = tuple(self.idx.regions(sc.rows, (rname,))) + tuple(
            self.weights.regions(sc.rows, (rname,))
        )
        tasks, lds = (
            (sc.rows, _slots_at(p) + 2 * self.k) if part else (-(-sc.rows // WAVES), 0)
        )
        args = (self.bias,)
        route = Stage(rname, tasks, partial(_route, p=p), lds_words=lds, writes=writes, args=args,
                      key=(p,), row_tasks=1 if part else 0)  # fmt: skip
        return self.dot.stages(sc) + [route]

    def pack_tp_table(self, table, rank: int, tp: int):
        """tp_cols: rank ``rank``'s rows of the [n, K] table (its experts)."""
        c = self.n // tp
        return table[rank * c : (rank + 1) * c].contiguous()


THREADS_ = 512


def _log1p(u):
    """log(1 + u) for u >= 0 to ~1 ulp: log(w) u / (w - 1), w = 1 + u (the rounding of w cancels); math.log1p has
    no lowering here."""
    w = fx.Float32(1.0) + u
    lw = hw_log2(w) * fx.Float32(LN2)
    return (w == fx.Float32(1.0)).select(u, lw * (u / (w - fx.Float32(1.0))))


def _sqrtsoftplus(x):
    """torch: softplus(x) = x past threshold 20, else log1p(exp(x)); then sqrt."""
    sp = (x > fx.Float32(20.0)).select(x, _log1p(fx.Float32(fmath.exp(x))))
    return fx.Float32(fmath.sqrt(sp))


def _sigmoid(x):
    """torch's fp32 sigmoid: 1 / (1 + exp(-x))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + fx.Float32(fmath.exp(-x)))


def _score(p, x):
    """s of a score (``score_func``; softmax_topk has none: its winners' softmax)."""
    return _sigmoid(x) if p.score_func == "sigmoid" else _sqrtsoftplus(x)


def _per(p) -> int:
    """Scores a lane holds in the route: n / 64; a sigmoid route with k > n / 64 pads to k a lane (the indices past
    n never win), so the sorted-heads top-k (``wave_topk_won_full``, k <= a lane's count) serves it.
    """
    per = p.n // 64
    return max(per, p.k) if p.score_func == "sigmoid" and not p.part else per


def _route(k, t, p):
    """Row t x WAVES + wave (a row past the step re-reads the last row, puts nothing). With the dot's slice
    partials (``p.part``): row t, its partials polled by all 8 waves (slices w, w + 8, ..) into LDS, then every
    wave adds them in slice order (``GatheredDot``'s ``.sum`` order: the same bits) and wave 0 puts.
    """
    per = _per(p)
    if const_expr(
        p.scores_by != "wave0"
    ):  # this lane's column's bias (wave w < n / 64: column 64 w + lane)
        bias = fx.Float32(
            bo.buffer_load(rsrc(k.args[p.bias]), _col(k, p), vec_width=1, dtype=T.f32)
        )
    if const_expr(
        p.scores_by == "fused"
    ):  # each wave its columns' partials, sum and scores; wave 0 the rest
        _fused_scores(k, p, t, bias)
        gpu.barrier()
        _route_tail(k, p, t, k.wave == 0, None, None, per)
        gpu.barrier()  # the next task reuses the LDS
        return
    if p.part:
        row = rowc = t
        live = k.wave == 0
        vals = _land_partials(k, p, t, per)
    else:
        row = t * WAVES + k.wave
        rowc = fx.min(row, p.rows - 1)
        live = row < p.rows
        cl = (
            [fx.min(k.lane * per + d, fx.Int32(p.n - 1)) for d in range(per)]
            if per * 64 > p.n
            else None
        )
        got = mb.poll([(k.mb[p.scores], rowc * p.n + (cl[d] if cl else k.lane * per + d), 1) for d in range(per)],
                      k.tag)  # fmt: skip
        vals = [g[0].bitcast(fx.Float32) for g in got]
    if const_expr(p.scores_by == "waves"):
        _waves_scores(k, p, bias)
        gpu.barrier()
        _route_tail(k, p, rowc, live, None, None, per)
    else:
        _route_tail(k, p, rowc, live, vals, bias, per)
    if p.part:
        gpu.barrier()  # the next task reuses the LDS


@traced
def _route_tail(k, p, rowc, live, vals, bias, per):
    """The scores, the top-k and the puts, by the waves whose results are used: with the slice partials (a task a
    row) wave 0 only (the other waves only competed for its SIMDs); else a row a wave.
    """
    if const_expr(p.part != ""):  # noqa: SIM102 (traced)
        if k.wave == 0:
            _route_rows(k, p, rowc, live, vals, bias, per)


def _route_rows(k, p, rowc, live, vals, bias, per):
    if const_expr(
        p.scores_by != "wave0"
    ):  # the lane's per columns' s and biased bits, from ``_waves_scores``
        at = (p.slices + 1) * p.n + k.lane * per
        s = [fx.Float32(fx.ptr_load(k.ldsf + (at + d))) for d in range(per)]
        biased = [
            fresh(fx.Int32(fx.ptr_load(k.lds + (at + p.n + d)))) for d in range(per)
        ]
    # every lane holds n / 64 valid scores: the sorted-heads top-k when k <= n / 64 (the same winners)
    if const_expr(p.k <= per):
        won = wave_topk_won_full(p.k, biased, k.lane)
    else:
        won = wave_topk_won(fx.Int32(p.n), p.k, biased, k.lane)
    # the winners' s summed: each lane its own in index order, then the wave butterfly
    part = fx.Float32(0.0)
    for d in range(per):
        part = part + (((won >> d) & 1) != 0).select(s[d], fx.Float32(0.0))
    tot = wave_sum(part)
    cnt = popcount(won)
    pos = wave_incl_scan(cnt, k.lane) - cnt
    if const_expr(
        p.part != ""
    ):  # a task a row (LDS): the winners to their slots, then lane j puts slot j
        _route_slots(k, p, won, s, pos)
        _route_put_lanes(k, p, rowc, live, tot)


def _land_partials(k, p, row, per):
    """Row ``row``'s slice partials -> LDS [slice][n] (wave w polls slices w, w + 8, ..), summed per column in slice
    order into LDS row ``slices``; returns this lane's ``per`` scores."""
    n2 = 2 if per % 2 == 0 else 1
    for j in range(-(-p.slices // WAVES)):
        sl = fx.min(
            k.wave + j * WAVES, fx.Int32(p.slices - 1)
        )  # a wave past the last slice re-lands it
        base = row * (p.slices * p.n) + sl * p.n + k.lane * per
        got = mb.poll([(k.mb[p.part], base + d, n2) for d in range(0, per, n2)], k.tag)
        for d, g in zip(range(0, per, n2), got, strict=True):
            fx.ptr_store(
                fx.Vector.from_elements(g, fx.Int32).bitcast(fx.Float32),
                k.ldsf + (sl * p.n + k.lane * per + d),
            )
    gpu.barrier()
    # thread c: columns c, c + 512, .. < n, each one's partials added in slice order -> LDS row ``slices`` (few live
    # registers a thread; past 512 experts (e.g. 896 unsplit) a second column a thread: before, columns
    # 512 .. n - 1 were never summed and the route read stale LDS)
    for c0 in range(0, p.n, THREADS_):
        c = k.tid if c0 == 0 else k.tid + c0  # (n <= 512: the code of before)
        col = fx.min(c, fx.Int32(p.n - 1))
        v = fx.Float32(0.0)
        for sl in range(p.slices):
            v = v + fx.Float32(fx.ptr_load(k.ldsf + (sl * p.n + col)))
        _store_if(k.ldsf + (p.slices * p.n + col), v, c < p.n)
    gpu.barrier()
    return [
        fx.Float32(fx.ptr_load(k.ldsf + (p.slices * p.n + k.lane * per + d)))
        for d in range(per)
    ]


@traced
def _store_if(addr, v, ok):
    if ok:
        fx.ptr_store(v, addr)


def _slots_at(p) -> int:
    """LDS words: the winners' ids [k] then their s [k], after the route's other words."""
    return (p.slices + 1 + 2 * (p.scores_by != "wave0")) * p.n


@traced
def _route_slots(k, p, won, s, pos):
    """Each lane's winners (ascending expert id) -> LDS slot ``pos``: the id and s (one wave: its LDS writes land
    before its reads)."""
    per = p.n // 64
    at = _slots_at(p)
    for d in range_constexpr(per):
        if ((won >> d) & 1) != 0:
            fx.ptr_store(k.lane * per + d, k.lds + (at + pos))
            fx.ptr_store(s[d], k.ldsf + (at + p.k + pos))
        pos = pos + ((won >> d) & 1)


@traced
def _route_put_lanes(k, p, row, live, tot):
    """Lane j < k: slot j's id and weight (put by k lanes at once)."""
    if live & (k.lane < p.k):
        at = _slots_at(p)
        e = fx.min(k.lane, fx.Int32(p.k - 1))
        w = (
            fx.Float32(fx.ptr_load(k.ldsf + (at + p.k + e)))
            / (tot + fx.Float32(1e-20))
            * fx.Float32(p.route_scale)
        )
        mb.put(
            k.mb[p.idx], row * p.k + e, fx.Int32(fx.ptr_load(k.lds + (at + e))), k.tag
        )
        mb.put(k.mb[p.weights], row * p.k + e, w, k.tag)


def _col(k, p):
    """scores "waves": this lane's column (wave w < n / 64: 64 w + lane; the waves past compute wave n / 64 - 1's)."""
    return fx.min(k.wave, fx.Int32(p.n // 64 - 1)) * 64 + k.lane


@traced
def _fused_scores(k, p, row, bias):
    """scores "fused": wave w < n / 64 polls its column's ``slices`` partials (lane: column 64 w + lane), adds them in
    slice order (``_land_partials``' sum: from +0, slice order) and puts s and the biased bits into LDS as
    ``_waves_scores`` does."""
    if k.wave < p.n // 64:
        col = _col(k, p)
        got = mb.poll(
            [
                (k.mb[p.part], row * (p.slices * p.n) + sl * p.n + col, 1)
                for sl in range(p.slices)
            ],
            k.tag,
        )
        v = fx.Float32(0.0)
        for g in got:
            v = v + g[0].bitcast(fx.Float32)
        s = _score(p, v)
        fx.ptr_store(s, k.ldsf + ((p.slices + 1) * p.n + col))
        fx.ptr_store(
            fresh((s + bias).bitcast(fx.Int32)), k.lds + ((p.slices + 2) * p.n + col)
        )


@traced
def _waves_scores(k, p, bias):
    """scores "waves": s and the biased score's bits of this lane's column -> LDS after the summed row."""
    col = _col(k, p)
    s = _score(p, fx.Float32(fx.ptr_load(k.ldsf + (p.slices * p.n + col))))
    biased = fresh((s + bias).bitcast(fx.Int32))
    if k.wave < p.n // 64:
        fx.ptr_store(s, k.ldsf + ((p.slices + 1) * p.n + col))
        fx.ptr_store(biased, k.lds + ((p.slices + 2) * p.n + col))
