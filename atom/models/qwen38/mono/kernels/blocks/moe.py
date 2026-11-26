# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen MoE's router and routing: ``Router`` (target and drafter), the routing and expert-addressing device
helpers ``moe_line.LineMoE`` runs on (its unit plans, ``_GPlan``), and ``moe_blocks`` (the target's MoE: the Router,
then ``LineMoE``).

**Router**:

- ``<name>`` (the GEMV): ``logits = x @ W^T`` with ``x`` the step's rows (a BF16 channel of
  width H) and ``W`` [E + shared, H] bf16 (the routed rows, then the shared experts' gate rows:
  Atom's merged ``mlp.gate.weight``). Task = (16-row block of W, K slice):
  ``mfma 16x16x32 bf16`` with W's rows as M and the step's rows as N, at most two K steps (64
  products) chained per wave, the waves summed in wave order through LDS; each task publishes
  its fp32 partials (F32 region ``<name>.part`` [k_split, rows, E + shared]). The longest
  sequential chain is 64 terms.
- ``<name>.select`` (``select``: the drafter), one task per row: the partials summed in slice order, ``bf16``, then
  ``_topk``: ids -> I32 channel ``ids`` [rows, top_k] (rank order), weights -> F32 channel ``weights`` [rows, top_k
  + shared]. The target has no select stage: every ``LineMoE`` CTA selects from the partials itself (``_route``).

**The selection** (``_topk``, one wave per row; the select stage and ``_route`` run the
same code): keys = the bf16 logit's order bits, then the lower expert on equal logits (as
``torch.sort(stable)``); the top ``top_k`` by sorted per-lane lists and wave-max rounds;
weights ``p_k / sum_top p`` (rank order) with ``p = softmax`` over every routed expert, the
shared experts' ``sigmoid(logit)``.

**The routing in LDS** (``_route``): a step's (row, rank) ids are reduced to the distinct experts in
first-occurrence order (row-major), so an expert picked by several rows is read once per row chunk; the MFMA's N
dimension is every row of the chunk, and only the rows that picked the expert keep the result. A step of more than
``ROW_BLOCK`` rows is cut into equal row chunks (the last may be shorter: its missing rows repeat the step's last row
and store nothing), each with its own routing and union. The slots of a row are its ``top_k`` routed ranks, then the
fixed experts ``shared`` (the fused shared expert, slot E of the tensors).

The experts are Atom's tensors in aiter's A8W4 MoE layout, read **in place** (``Experts``;
``weights.source.bind`` checks the layout): one buffer resource per tensor, the expert's offset in
the per-lane offset. Expert ids are clamped to the tensors' slots (a corrupt id reads a wrong expert, never out of
bounds).
"""

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.qwen38.mono.kernels.blocks.base import Block, Channel, Fmt
from atom.models.qwen38.mono.kernels.blocks.linear import (
    _land_batch,
    _try_batch,
    _x_batches,
)
from atom.models.qwen38.mono.kernels.device import buffer_ops as bo
from atom.models.qwen38.mono.kernels.device import channels as chan
from atom.models.qwen38.mono.kernels.device import gemv
from atom.models.qwen38.mono.kernels.device import mailbox as mb
from atom.models.qwen38.mono.kernels.device.dpp_utils import update_dpp_i32
from atom.models.qwen38.mono.kernels.device.ops import (
    ballot,
    bf16_round,
    fresh,
    hw_exp2,
    rsrc,
    traced,
    uniform,
    xred,
)
from atom.models.qwen38.mono.kernels.geometry import MoEGeometry
from atom.models.qwen38.mono.kernels.program import PlanError, Region, Scenario, Stage
from atom.models.qwen38.mono.kernels.target import MI355X

WAVES = MI355X.waves
THREADS = MI355X.threads
LOG2E = 1.4426950408889634
INT_MIN = -(2**31)


# ------------------------------------------------------------------ specs


@dataclass(frozen=True)
class Experts:
    """Expert weights in aiter's A8W4 MoE layout, read in place: ``w`` [slots, n_stored,
    k_stored / 2] (fp4x2) and ``s`` [slots * n_stored, k_stored / 32] (E8M0) pointer-table args.
    ``n`` x ``k`` is the used weight of an expert (``gate_up``: 2 x inter rows, gate / up
    interleaved per 16-row block; the padding is zero and never read)."""

    w: str
    s: str
    slots: int
    n: int
    k: int
    n_stored: int
    k_stored: int
    gate_up: bool

    def __post_init__(self):
        if self.n % 16 or self.k % 128 or self.n_stored % 32 or self.k_stored % 256:
            raise ValueError(
                f"{self}: n % 16, k % 128, n_stored % 32 and k_stored % 256 must be 0"
            )
        if self.n > self.n_stored or self.k > self.k_stored:
            raise ValueError(
                f"{self}: used [{self.n}, {self.k}] exceeds the stored slot"
            )

    @property
    def w_bytes(self) -> int:  # one expert's weights
        return self.n_stored * self.k_stored // 2

    def weight_specs(self) -> tuple:
        from atom.models.qwen38.mono.kernels.weights.source import (
            E8M0,
            FP4X2,
            WeightSpec,
        )

        params = (("gate_up", self.gate_up), ("n", self.n), ("k", self.k))
        return (
            WeightSpec(
                self.w,
                "aiter.a8w4.moe",
                (self.slots, self.n_stored, self.k_stored // 2),
                FP4X2,
                attrs=(("is_shuffled", True),),
                params=params,
            ),
            WeightSpec(
                self.s,
                "aiter.e8m0.moe",
                (self.slots * self.n_stored, self.k_stored // 32),
                E8M0,
                params=params + (("experts", self.slots),),
            ),
        )


@dataclass(frozen=True)
class RouterLogits:
    """The Router's fp32 partial logits, for ``LineMoE`` (it selects its experts itself): region ``part``
    [k_split, rows, total] (``total``: the routed experts, then the shared gates)."""

    part: str
    k_split: int
    total: int


@dataclass(frozen=True)
class Combine:
    """The MoE's combine epilogue: ``hc``, an ``blocks.hc.HCWrite`` (``y = sum_s w_s y_s``
    into the hyper-connection streams)."""

    hc: object


# ------------------------------------------------------------------ a line unit's plan


@dataclass(frozen=True)
class _GPlan:
    """Every Python constant of a ``LineMoE`` unit (gate_up or down) and of the step's routing that the device
    helpers close over (built by ``LineMoE.plan``). Names shared with ``linear._Plan`` (rows, n, k, g, wr, wk, kse,
    pitch, ...) mean the same, so the input poll, K-wave reduction and MXFP8 epilogue helpers of
    ``blocks.linear`` apply."""

    name: str
    rows: int  # rows of a task: a chunk of the step's rows (``ROW_BLOCK``)
    n: int  # used output rows of an expert
    ek: int  # used K of an expert
    k: int  # the input row width (gate_up: ek; down: slots * ek)
    g: int  # row groups per unit
    wr: int
    wk: int
    x: str
    top_k: int
    experts: int  # routed experts (ids range)
    shared: tuple
    umax: int  # union slots of routed experts
    tslots: int  # expert slots in the tensors
    n_stored: int
    k_stored: int
    w: str
    s: str
    out_ch: str  # gate_up: the act channel (``linear._quant_out``)
    lg: str  # the Router's partials region (``RouterLogits``): the routing is selected in-task
    lg_ks: int
    lg_total: int
    rows_all: int = 0  # the step's rows (chunks of ``rows``; 0: one chunk)
    kind: str = "swiglu"  # linear._items' epilogue kind (gate_up)

    @property
    def xk(
        self,
    ):  # the input channel's width (``linear._Plan.xk``; no block-diagonal groups here)
        return self.k

    @property
    def n_rg(self):
        return self.n // 16

    @property
    def chunks(self):  # row chunks of the step (each with its own routing and union)
        return -(-self.rows_all // self.rows) if self.rows_all else 1

    @property
    def nrows(self):  # the step's rows
        return self.rows_all or self.rows

    @property
    def w_bytes(self):  # one expert's weights
        return self.n_stored * self.k_stored // 2

    @property
    def s_bytes(self):  # one expert's scales
        return self.n_stored * self.k_stored // 32

    @property
    def steps(self):  # stored K steps of an expert
        return self.k_stored // 128

    @property
    def eks(self):  # used K steps of an expert
        return self.ek // 128

    @property
    def c(self):  # K steps per wave (gate_up)
        return -(-self.eks // self.wk)

    @property
    def gw(self):
        return self.g // self.wr

    @property
    def slots(self):
        return self.top_k + len(self.shared)

    @property
    def utot(self):
        return self.umax + len(self.shared)

    @property
    def n_sel(self):
        return self.rows * self.top_k

    @property
    def kse(self):
        return self.k

    @property
    def pitch(self):
        return self.k // 4 + 4

    @property
    def spitch(self):
        return self.k // 32 + 1

    # LDS: the route area first (ids, union ranks, wave counts, count, union, slot table, weights), then from
    # ``x0`` the selection's logits (dead once the routing is in LDS) and the input [rows][pitch] words and its scale
    # codes (``moe_line``: past the logits, and the gate_up's K-wave partials after them). The route area's words
    # stay below 64 KB, in the ds instructions' offset field (above it, LLVM holds each address in a VGPR).
    # ``xs_base`` is relative to ``x0`` (the linear helpers see the input area at word 0)
    @property
    def r_ids(self):
        return 0

    @property
    def r_rank(self):
        return self.r_ids + self.n_sel

    @property
    def r_wc(self):
        return self.r_rank + self.n_sel

    @property
    def r_cnt(self):
        return self.r_wc + WAVES

    @property
    def r_uniq(self):
        return self.r_cnt + 1

    @property
    def r_sl(self):
        return self.r_uniq + self.umax

    @property
    def r_wts(self):
        return self.r_sl + self.umax * self.rows

    @property
    def x0(self):  # the input area (16 B aligned)
        return -(-(self.r_wts + self.rows * self.slots) // 4) * 4

    @property
    def xs_base(self):  # relative to x0
        return self.rows * self.pitch

    @property
    def r_lg(self):  # in-task selection: the rows' logits [rows][total] f32
        return self.x0

    @property
    def out_quads(self):  # gate_up epilogue items (linear._items)
        return self.rows * (self.g // 2) * 4

    @property
    def mid(self):  # gate_up: features of one slot of act
        return self.n // 2


def _x_issue(k, p, row0):
    """The input's poll batches (the chunk's rows), every one attempted now (one batch at 4 to 8 rows a chunk)."""
    batches = _x_batches(
        k, p, 0, None if p.chunks == 1 else lambda r: _grow(p, row0, r)[0]
    )
    rocdl.sched_barrier(0)
    tries = [_try_batch(specs) for specs, _ in batches]
    rocdl.sched_barrier(0)
    return batches, tries


def _x_land(k, p, batches, tries):
    """Land the input in LDS: the attempted batches (re-polled if stale)."""
    k = _x_view(k, p)
    for (specs, dests), words in zip(batches, tries, strict=True):
        _land_batch(k, p, specs, dests, words)


ROW_BLOCK = 8  # rows a task routes and computes at most (more: row chunks)


# ------------------------------------------------------------------ device: the step's routing in LDS


@traced
def _store_if(ok, v, ptr):
    if ok:
        fx.ptr_store(v, ptr)


def _x_view(k, p):
    """``k`` with its LDS pointers at the input area (``p.x0``): the linear helpers' view."""
    return SimpleNamespace(**{**vars(k), "lds": k.lds + p.x0, "ldsf": k.ldsf + p.x0})


def _lds_i(k, w):
    return fx.Int32(fx.ptr_load(k.lds + w))


def _grow(p, row0, r):
    """Chunk row ``r`` (in range) -> (the step's row (clamped), whether it exists; None: always)."""
    if p.chunks == 1:
        return r, None
    g = row0 + r
    return fx.min(g, fx.Int32(p.nrows - 1)), g < p.nrows


def _route(k, p, row0):
    """Every thread: the step's ids (thread i < rows x top_k: id i, row-major), selected here
    from the Router's partial logits (with the weights), then the
    union in LDS: ``uniq[u]`` (first-occurrence order), ``cnt``, and the slot table ``sl[u][r]``
    (the rank at which row r picked union expert u, -1 if it did not). Ends with a barrier.
    Lane and thread ids are opaque copies: LLVM would hoist every mask built on them to the
    kernel top, where they cost SGPR pairs for the whole stage."""
    tid, lane, wave = fresh(k.tid), fresh(k.lane), k.wave
    n_sel = p.n_sel
    i = fx.min(tid, fx.Int32(n_sel - 1))
    live = tid < n_sel
    _select_rows(k, p, row0)  # (the routing in this task, from the router's partials)
    idv = _lds_i(k, p.r_ids + i)
    idv = fx.max(fx.min(idv, fx.Int32(p.experts - 1)), fx.Int32(0))
    _store_if(live, idv, k.lds + (p.r_ids + i))
    for c in range_constexpr(-(-(p.umax * p.rows) // THREADS)):
        idx = fx.Int32(c * THREADS) + tid
        _store_if(
            idx < p.umax * p.rows,
            fx.Int32(-1),
            k.lds + (p.r_sl + fx.min(idx, fx.Int32(p.umax * p.rows - 1))),
        )
    gpu.barrier()
    # the first occurrence m of this thread's id (ids of a row are distinct: only earlier rows)
    m = i
    for j0 in range_constexpr(0, n_sel - n_sel % 4, 4):
        v = fx.Vector(
            fx.ptr_load(
                k.lds + (p.r_ids + j0), result_type=fx.Vector.make_type(4, fx.Int32)
            )
        )
        for q in range_constexpr(4):
            hit = (v[q] == idv) & (fx.Int32(j0 + q) < m)
            m = hit.select(fx.Int32(j0 + q), m)
    for j in range_constexpr(n_sel - n_sel % 4, n_sel):
        hit = (_lds_i(k, p.r_ids + j) == idv) & (fx.Int32(j) < m)
        m = hit.select(fx.Int32(j), m)
    first = live & (m == i)
    bal = ballot(first)
    below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
    prefix = fx.Int32(fx.ctpop(bal & below))
    waves = -(-n_sel // 64)
    wc = k.lds + (p.r_wc + fx.min(wave, fx.Int32(waves - 1)))
    _store_if((lane == 0) & (wave < waves), fx.Int32(fx.ctpop(bal)), wc)
    gpu.barrier()
    off = fx.Int32(0)
    cnt = fx.Int32(0)
    for w in range_constexpr(waves):
        cw = _lds_i(k, p.r_wc + w)
        off = off + (fx.Int32(w) < wave).select(cw, fx.Int32(0))
        cnt = cnt + cw
    rank = off + prefix
    _store_if(first, rank, k.lds + (p.r_rank + i))
    _store_if(first, idv, k.lds + (p.r_uniq + fx.min(rank, fx.Int32(p.umax - 1))))
    _store_if(tid == 0, cnt, k.lds + p.r_cnt)
    gpu.barrier()
    u = _lds_i(k, p.r_rank + m)
    slot = (u * p.rows + i // p.top_k) if p.rows > 1 else u
    _store_if(
        live,
        i % p.top_k,
        k.lds + (p.r_sl + fx.min(slot, fx.Int32(p.umax * p.rows - 1))),
    )
    gpu.barrier()


def _select_rows(k, p, row0):
    """The rows' logits (the Router's partials summed in slice order, bf16) into LDS, then
    wave w selects rows w, w + 8, ... into LDS ``r_ids`` and ``r_wts``. Ends with a barrier.
    """
    # the rows' logits, flat [rows][total]: two per 16-byte poll
    n = p.rows * p.lg_total
    n_all = p.nrows * p.lg_total
    for c in range(-(-n // (2 * THREADS))):
        # an odd tail overlaps
        j = fx.min((fx.Int32(c * THREADS) + k.tid) * 2, fx.Int32(n - 2))
        gj = j
        if (
            p.chunks > 1
        ):  # the chunk's rows that exist (the pair at the step's end overlaps)
            gj = fx.min(row0 * p.lg_total + j, fx.Int32(n_all - 2))
            j = gj - row0 * p.lg_total
        got = mb.poll([(k.mb[p.lg], s * n_all + gj, 2) for s in range(p.lg_ks)], k.tag)
        for h in range(2):
            acc = got[0][h].bitcast(fx.Float32)
            for s in range(1, p.lg_ks):
                acc = acc + got[s][h].bitcast(fx.Float32)
            fx.ptr_store(bf16_round(acc), k.ldsf + (p.r_lg + j + h))
    gpu.barrier()
    for rr in range(-(-p.rows // WAVES)):
        r = k.wave + WAVES * rr
        rc = fx.min(r, fx.Int32(p.rows - 1))
        # missing rows: the last's
        lr = rc if p.chunks == 1 else fx.min(rc, fx.Int32(p.nrows - 1) - row0)
        _topk_if(
            k,
            r < p.rows,
            rc,
            p.r_lg + lr * p.lg_total,
            p.experts,
            p.top_k,
            len(p.shared),
            p.r_ids,
            p.r_wts,
        )
    gpu.barrier()


def _expert(k, p, u):
    """Union slot ``u`` (traced, may differ per wave): its expert id, wave-uniform; a slot that
    does not exist this step gives ``tslots`` (one past the last expert: its loads fall past
    the tensors' buffer resources and return zeros without touching memory). Slots past the
    routed union are the fixed experts ``shared``."""
    cnt = _lds_i(k, p.r_cnt)
    routed = _lds_i(k, p.r_uniq + fx.min(fx.max(u, fx.Int32(0)), fx.Int32(p.umax - 1)))
    e = fx.max(fx.min(routed, fx.Int32(p.tslots - 1)), fx.Int32(0))
    e = (u < cnt).select(e, fx.Int32(p.tslots))
    for j, sh in enumerate(p.shared):
        e = (u == p.umax + j).select(fx.Int32(sh), e)
    return uniform(e)


def _slot(k, p, u, row):
    """The slot (rank) at which ``row`` picked union slot ``u``, -1 if it did not."""
    routed = _lds_i(
        k, p.r_sl + fx.min(fx.max(u, fx.Int32(0)), fx.Int32(p.umax - 1)) * p.rows + row
    )
    s = (u < p.umax).select(routed, fx.Int32(-1))
    for j in range(len(p.shared)):
        s = (u == p.umax + j).select(fx.Int32(p.top_k + j), s)
    return s


def _tensor_rsrc(k, p):
    """One buffer resource per tensor (every expert): the expert's offset goes into the
    per-lane offset, so a task holds two descriptors however many experts it reads."""
    return rsrc(k.args[p.w], p.tslots * p.w_bytes), rsrc(
        k.args[p.s], p.tslots * p.s_bytes
    )


def _w_issue(p, r_w, e, rg, st, lane, valid, cm=gemv.CM_NT):
    """The 16 B of expert ``e``'s weight unit (row group ``rg``, K step ``st``) of this lane (0 if
    not valid)."""
    return gemv.load_weight(
        r_w, e * (p.w_bytes // 1024) + rg * p.steps + st, lane, valid, cm=cm
    )


def _s_issue(p, r_s, e, rg, k1, lane):
    """Expert ``e``'s scale dword of row-group pair ``rg // 2``, K-step pair ``k1`` of this lane:
    codes of (rg pair, st pair) at byte ``2 (st % 2) + rg % 2``."""
    off = e * (p.s_bytes // 4) + ((rg // 2) * (p.steps // 2) + k1) * 64 + lane
    return fx.Int32(
        bo.buffer_load(r_s, off, vec_width=1, dtype=T.i32, cache_modifier=0)
    )


def _code(dw, rg, st):
    return dw >> ((st % 2) * 16 + (rg % 2) * 8)


# ------------------------------------------------------------------ device: gate_up


def _gu_kstep(p, wk, i):
    return wk + p.wk * i


def _silu(g):
    """``g * sigmoid(g)`` in fp32 (the hardware exp2)."""
    return g * (fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(g * fx.Float32(-LOG2E))))


# ------------------------------------------------------------------ Router


@dataclass(frozen=True)
class _RPlan:
    name: str
    rows: int
    hidden: int
    experts: int
    shared: int
    top_k: int
    k_split: int
    x: str
    w: str
    ids: str
    wts: str

    @property
    def total(self):  # weight rows: routed + shared gates
        return self.experts + self.shared

    @property
    def groups(self):
        return -(-self.total // 16)

    @property
    def kslice(self):
        return self.hidden // self.k_split

    @property
    def steps(self):  # 32-K MFMA steps of a slice
        return self.kslice // 32

    @property
    def c(self):
        return -(-self.steps // WAVES)

    @property
    def xpitch(self):  # LDS words per x row (bf16 pairs) of a slice, +4
        return self.kslice // 2 + 4

    @property
    def red(self):
        return self.rows * self.xpitch

    @property
    def gemv_lds(self):
        return self.red + WAVES * 16 * self.rows

    @property
    def sel_tmp(self):  # select: the top-k's scratch words, after the row's logits
        return self.total

    @property
    def sel_lds(self):
        return -(-(self.total + self.top_k) // 4) * 4


class Router(Block):
    """Router logits, top-k ids and combine weights (module docstring). ``k_split``: the GEMV's K slices (5: the most
    within one grid round at H 2560 with every wave two K steps at least); ``select`` False (the target): the GEMV
    stage only."""

    def __init__(
        self,
        name: str,
        x: Channel,
        w: str,
        experts: int,
        top_k: int,
        ids: Channel,
        weights: Channel,
        shared: int = 1,
        k_split: int = 5,
        select: bool = True,
        base: int | None = None,
    ):
        if x.fmt is not Fmt.BF16:
            raise ValueError(f"Router {name}: x must be a tagged BF16 channel")
        if ids.fmt is not Fmt.I32 or ids.width != top_k:
            raise ValueError(f"Router {name}: ids must be I32 [rows, top_k]")
        if weights.fmt is not Fmt.F32 or weights.width != top_k + shared:
            raise ValueError(
                f"Router {name}: weights must be F32 [rows, top_k + shared]"
            )
        if x.width % 32 or top_k > 64 or experts > 64 * 64 or experts >= 1 << 16:
            raise ValueError(
                f"Router {name}: unsupported sizes (hidden {x.width}, experts {experts}, top_k {top_k})"
            )
        self.name, self.x, self.w, self.experts, self.top_k, self.shared = (
            name,
            x,
            w,
            experts,
            top_k,
            shared,
        )
        self.ids, self.weights, self.k_split, self.select, self.base = (
            ids,
            weights,
            k_split,
            select,
            base,
        )

    @property
    def source(self) -> RouterLogits:
        """The partial logits, for ``LineMoE`` (it selects its experts itself)."""
        p = self.plan(Scenario())
        return RouterLogits(f"{self.name}.part", p.k_split, p.total)

    def plan(self, sc: Scenario) -> _RPlan:
        steps = self.x.width // 32
        if steps % self.k_split:
            raise PlanError(
                f"Router {self.name}: k_split {self.k_split} does not divide {steps} K steps"
            )
        return _RPlan(
            self.name,
            sc.rows,
            self.x.width,
            self.experts,
            self.shared,
            self.top_k,
            self.k_split,
            self.x.name,
            self.w,
            self.ids.name,
            self.weights.name,
        )

    def stages(self, sc: Scenario) -> list[Stage]:
        p = self.plan(sc)
        part = Region(f"{self.name}.part", p.k_split * p.rows * p.total, (self.name,))
        sel = f"{self.name}.select"
        writes = list(self.ids.regions(p.rows, (sel,))) + list(
            self.weights.regions(p.rows, (sel,))
        )
        gemv = Stage(
            self.name,
            p.groups * p.k_split,
            partial(_router, p=p),
            lds_words=-(-p.gemv_lds // 4) * 4,
            writes=(part,),
            args=(p.w,),
            base=self.base,
            key=(p,),
        )
        if not self.select:
            return [gemv]
        return [
            gemv,
            Stage(
                sel,
                p.rows,
                partial(_select, p=p),
                lds_words=p.sel_lds,
                writes=tuple(writes),
                key=(p,),
            ),
        ]

    def weight_specs(self) -> tuple:
        from atom.models.qwen38.mono.kernels.weights.source import WeightSpec

        return (
            WeightSpec(
                self.w,
                "bf16",
                (self.experts + self.shared, self.x.width),
                (torch.bfloat16,),
            ),
        )

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """The router weight [experts + shared, H]: ``mlp.gate`` then ``mlp.shared_expert_gate`` from
        ``layer.retained``."""
        w = torch.cat(
            [layer.retained["mlp.gate"], layer.retained["mlp.shared_expert_gate"]]
        )
        return {self.w: w.to(torch.bfloat16).contiguous()}


def _v4i():
    return fx.Vector.make_type(4, fx.Int32)


def _router(k, t, p):
    """Task ``t``: weight rows 16 gi .., K slice s (t = s * groups + gi)."""
    gi, s = t % p.groups, t // p.groups
    lane, wave, tid = k.lane, k.wave, k.tid
    r_w = rsrc(k.args[p.w], p.total * p.hidden * 2)
    wrow = gi * 16 + lane % 16
    j = lane // 16
    k0 = s * p.kslice
    ws = []
    for i in range(p.c):
        st = wave + WAVES * i
        ok = wrow < p.total
        if p.steps % WAVES:
            ok = ok & (st < p.steps)
        off = (wrow * p.hidden + k0 + st * 32 + j * 8) // 2
        off = ok.select(off, fx.Int32(gemv.OOB // 4))
        ws.append(
            fx.Vector(
                bo.buffer_load(r_w, off, vec_width=4, dtype=T.i32, cache_modifier=2)
            )
        )
    # the slice of every row of x: (rows x kslice / 2) bf16 pairs
    npairs = p.rows * p.kslice // 2
    for c0 in range(0, -(-npairs // THREADS), mb.POLL_MAX):
        specs, dests = [], []
        for c in range(c0, min(c0 + mb.POLL_MAX, -(-npairs // THREADS))):
            idx = fx.min(fx.Int32(c * THREADS) + tid, fx.Int32(npairs - 1))
            r, w2 = idx // (p.kslice // 2), idx % (p.kslice // 2)
            specs.append((k.mb[p.x], r * (p.hidden // 2) + s * (p.kslice // 2) + w2, 1))
            dests.append(r * p.xpitch + w2)
        got = mb.poll(specs, k.tag)
        for words, d in zip(got, dests, strict=True):
            fx.ptr_store(words[0], k.lds + d)
    gpu.barrier()
    row = fx.min(lane % 16, fx.Int32(p.rows - 1))
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range(p.c):
        st = fx.min(wave + WAVES * i, fx.Int32(p.steps - 1))
        xb = fx.Vector(
            fx.ptr_load(k.lds + (row * p.xpitch + st * 16 + j * 4), result_type=_v4i())
        )
        a = ws[i].bitcast(fx.BFloat16)
        bv = xb.bitcast(fx.BFloat16)
        acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, bv, acc]))
    # acc lane l: weight rows 4 (l // 16) + c, x row l % 16 -> LDS [wave][16 rows][x rows]
    _red_rows(k, p, acc)
    gpu.barrier()
    n = 16 * p.rows
    for c in range(-(-n // THREADS)):
        idx = fx.min(fx.Int32(c * THREADS) + tid, fx.Int32(n - 1))
        er, r = idx // p.rows, idx % p.rows
        tot = fx.Float32(fx.ptr_load(k.ldsf + (p.red + er * p.rows + r)))
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.Float32(
                fx.ptr_load(k.ldsf + (p.red + (w * 16 + er) * p.rows + r))
            )
        e = gi * 16 + er
        _put_part(
            k,
            p,
            (fx.Int32(c * THREADS) + tid < n) & (e < p.total),
            (s * p.rows + r) * p.total + e,
            tot,
        )
    gpu.barrier()


@traced
def _red_rows(k, p, acc):
    if k.lane % 16 < p.rows:
        for c in range_constexpr(4):
            er = (k.lane // 16) * 4 + c
            fx.ptr_store(
                acc[c], k.ldsf + (p.red + (k.wave * 16 + er) * p.rows + k.lane % 16)
            )


@traced
def _put_part(k, p, ok, i, v):
    if ok:
        mb.put(k.mb[f"{p.name}.part"], i, v, k.tag)


def _select(k, t, p):
    """Task ``t``: row t. Logits (slice partials summed in slice order, bf16) into LDS, then
    wave 0 selects and publishes."""
    tid = fresh(k.tid)
    r = t
    for c in range(-(-p.total // THREADS)):
        e = fx.min(fx.Int32(c * THREADS) + tid, fx.Int32(p.total - 1))
        got = mb.poll(
            [
                (k.mb[f"{p.name}.part"], (s * p.rows + r) * p.total + e, 1)
                for s in range(p.k_split)
            ],
            k.tag,
        )
        acc = got[0][0].bitcast(fx.Float32)
        for s in range_constexpr(1, p.k_split):
            acc = acc + got[s][0].bitcast(fx.Float32)
        lg = bf16_round(acc)
        fx.ptr_store(lg, k.ldsf + e)
    gpu.barrier()
    _select_wave(k, p, r)
    gpu.barrier()


def _key(lg, e, valid):
    """Signed sort key: the bf16 logit's order (-0 as +0) in the high half, ``0xFFFF - e`` low."""
    b = (lg.bitcast(fx.Int32) >> 16) & 0xFFFF
    b = (b == 0x8000).select(fx.Int32(0), b)
    v = ((b & 0x8000) != 0).select(b ^ 0x7FFF, b)  # monotone as a signed 16-bit value
    v = (v << 16) >> 16  # sign-extend
    key = (v << 16) | (0xFFFF - e)
    return valid.select(key, fx.Int32(INT_MIN))


def _key_expert(m):
    return 0xFFFF - (m & 0xFFFF)


def _key_logit(m):
    v = m >> 16  # arithmetic: the signed 16-bit order value
    b = (v < 0).select(v ^ 0x7FFF, v) & 0xFFFF
    return (b << 16).bitcast(fx.Float32)


def _wave_imax(v):
    """The wave's max of an Int32: row rotations 8, 4, 2, 1 (one DPP move each: every lane then
    holds its row's max), then the 16- and 32-lane swaps."""
    for ctrl in (0x128, 0x124, 0x122, 0x121):  # DPP row_ror:8, 4, 2, 1
        v = fx.max(v, fx.Int32(update_dpp_i32(v, v, ctrl, 0xF, 0xF, False)))
    return xred(xred(v, 16, fx.max), 32, fx.max)


@traced
def _select_wave(k, p, r):
    if k.wave == 0:
        e, w = _topk(k, 0, p.experts, p.top_k, p.shared, p.sel_tmp)
        _publish_route(k, r, e, w, p.top_k, p.shared, p.ids, p.wts)


@traced
def _topk_if(k, ok, r, lgw, experts, top_k, nshared, ids_lds, wts_lds):
    if ok:
        slots = top_k + nshared
        e, w = _topk(k, lgw, experts, top_k, nshared, wts_lds + r * slots)
        lane = fresh(k.lane)
        _store_if(
            lane < top_k,
            e,
            k.lds + (ids_lds + r * top_k + fx.min(lane, fx.Int32(top_k - 1))),
        )
        _store_if(
            lane < slots,
            w,
            k.ldsf + (wts_lds + r * slots + fx.min(lane, fx.Int32(slots - 1))),
        )


def _topk(k, lgw, experts, top_k, nshared, tmp):
    """One wave selects a row from its bf16-valued logits at LDS f32 word ``lgw`` (``experts``
    routed, then ``nshared`` shared gates): the top ``top_k`` (equal logits to the lower expert),
    weights ``p_k / sum_top p`` with ``p = softmax`` over the routed logits, the shared gates'
    ``sigmoid``. -> (lane c < top_k: rank c's expert; lane c < top_k + nshared: slot c's
    weight). ``tmp``: ``top_k`` LDS f32 words of this wave's (lane c's probability goes there,
    then every lane sums them in rank order: no all-lane exp / divide per rank)."""
    # opaque: LLVM would hoist the per-rank lane masks to the kernel top (SGPR pairs)
    lane = fresh(k.lane)
    n = -(-experts // 64)
    keys, lgs = [], []
    for i in range(n):
        e = lane * n + i
        ok = e < experts
        lg = fx.Float32(fx.ptr_load(k.ldsf + (lgw + fx.min(e, fx.Int32(experts - 1)))))
        keys.append(_key(lg, e, ok))
        lgs.append((lg, ok))
    for rnd in range(n):  # each lane's keys descending (odd-even transposition)
        for a in range(rnd % 2, n - 1, 2):
            keys[a], keys[a + 1] = fx.max(keys[a], keys[a + 1]), fx.min(
                keys[a], keys[a + 1]
            )
    picks = []
    for _ in range(top_k):
        m = _wave_imax(keys[0])
        hit = keys[0] == m
        keys = [hit.select(keys[i + 1], keys[i]) for i in range(n - 1)] + [
            hit.select(fx.Int32(INT_MIN), keys[n - 1])
        ]
        picks.append(m)
    top = _key_logit(picks[0])
    # the softmax denominator over every routed expert: lane order, then the butterfly
    z = fx.Float32(0.0)
    for lg, ok in lgs:
        z = z + ok.select(hw_exp2((lg - top) * fx.Float32(LOG2E)), fx.Float32(0.0))
    for off in (32, 16, 8, 4, 2, 1):
        z = xred(z, off, lambda a, b: a + b)
    # lane c: rank c's key (lane 0 writes the picks, every lane reads its own: no per-rank lane
    # masks, which cost SGPR pairs) and probability; the rank-order sum through LDS
    for c in range(top_k):
        _store_if(lane == 0, picks[c], k.lds + (tmp + c))
    mine = fx.Int32(fx.ptr_load(k.lds + (tmp + fx.min(lane, fx.Int32(top_k - 1)))))
    mine_e = _key_expert(mine)
    mine_p = hw_exp2((_key_logit(mine) - top) * fx.Float32(LOG2E)) / z
    _store_if(lane < top_k, mine_p, k.ldsf + (tmp + fx.min(lane, fx.Int32(top_k - 1))))
    tot = fx.Float32(fx.ptr_load(k.ldsf + tmp))
    for c in range(1, top_k):
        tot = tot + fx.Float32(fx.ptr_load(k.ldsf + (tmp + c)))
    sh = fx.min(fx.max(lane - top_k, fx.Int32(0)), fx.Int32(max(nshared, 1) - 1))
    lg_sh = (
        fx.Float32(fx.ptr_load(k.ldsf + (lgw + experts + sh)))
        if nshared
        else fx.Float32(0.0)
    )
    sig = fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(lg_sh * fx.Float32(-LOG2E)))
    return mine_e, (lane < top_k).select(mine_p / tot, sig)


def _publish_route(k, r, e, w, top_k, nshared, ids, wts):
    lane = fresh(k.lane)
    _put_i32_if(k, lane < top_k, ids, r * top_k + fx.min(lane, fx.Int32(top_k - 1)), e)
    _put_f32_if(
        k,
        lane < top_k + nshared,
        wts,
        r * (top_k + nshared) + fx.min(lane, fx.Int32(top_k + nshared - 1)),
        w,
    )


@traced
def _put_i32_if(k, ok, ch, i, v):
    if ok:
        chan.put_i32(k.mb[ch], i, v, k.tag)


@traced
def _put_f32_if(k, ok, ch, i, v):
    if ok:
        chan.put_f32(k.mb[ch], i, v, k.tag)


# ------------------------------------------------------------------ composition (recipe level)


@dataclass(frozen=True)
class MoEArgs:
    """Pointer-table arg names of one MoE layer: the router weight and Atom's four expert
    tensors (``w13`` / ``w13_scale`` / ``w2`` / ``w2_scale``)."""

    router: str = "moe.router"
    w13: str = "moe.w13"
    s13: str = "moe.w13_scale"
    w2: str = "moe.w2"
    s2: str = "moe.w2_scale"


_ARGS = MoEArgs()


def moe_blocks(
    name: str,
    geo: MoEGeometry,
    x_bf16: Channel,
    x_mx: Channel,
    out: Combine,
    args: MoEArgs = _ARGS,
    knobs: dict | None = None,
    base: int = 0,
) -> list[Block]:
    """The target's MoE: the Router's GEMV (from ``x_bf16``, BF16) -> ``moe_line.LineMoE`` (the routing selected in
    every line CTA from the Router's partials; gate_up with the fused shared expert, SwiGLU, down and the weighted
    combine into ``out.hc``; experts from ``x_mx``, MXFP8, the same rows). ``knobs``: ``router`` / ``line``
    constructor knobs; the Router's and the line's CTAs from ``base``."""
    from atom.models.qwen38.mono.kernels.blocks.moe_line import LineMoE

    knobs = knobs or {}
    if "line" not in knobs:
        raise PlanError(
            f"moe_blocks {name}: the target MoE runs the line stage (knobs['line'])"
        )
    ids = Channel(f"{name}.ids", geo.top_k, Fmt.I32)
    wts = Channel(f"{name}.wts", geo.top_k + geo.shared, Fmt.F32)
    router = Router(
        f"{name}.router",
        x_bf16,
        args.router,
        geo.experts,
        geo.top_k,
        ids,
        wts,
        geo.shared,
        **{"base": base, **knobs.get("router", {})},
    )
    return [
        router,
        LineMoE(
            name,
            geo,
            x_mx,
            router.source,
            out.hc,
            args,
            **{"base": base, **knobs["line"]},
        ),
    ]
