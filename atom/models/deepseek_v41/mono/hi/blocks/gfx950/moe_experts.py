# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``MoEExperts``: routed experts of a top-k MoE (DeepSeek-V4.1: 6 of 384, MXFP4 weights x MXFP8 activations),
gathered per step, plus the combine with a shared expert's output.

Per row ``r`` with picks ``(e_rk, w_rk)``, k < K (``idx`` / ``weights``: F32 channels of width K, ids as int32
bits, ascending by id: the router's order), the reference (``inference/model.py`` ``MoE`` / ``Expert``)::

    g, u  = bf16(x W1_e^T), bf16(x W3_e^T)                    (x through MXFP8: ``x`` is an MXFP8 channel)
    m     = bf16(w_rk * silu(min(g, L)) * clamp(u, -L, L))    (fp32 math; L = ``limit``, 0: no clamp)
    y_rk  = bf16(MXFP8(m) W2_e^T)
    out_r = bf16(((0 + y_r0) + y_r1 + ... + y_r,K-1) + shared_r)    (fp32 adds, k ascending = expert id ascending)

with ``shared`` a BF16 channel (the shared expert, e.g. ``Linear`` blocks) or none.

Stages:

1. ``<name>.rt`` (``route="cta"``, the default; a task on every CTA): the step's ``rows * K`` picks (``route_from``:
   a ``MoERouter`` whose route this stage computes itself from the router's dot partials,
   instead of polling ``idx`` / ``weights``: one hop fewer, the same picks) deduplicated
   into slots, one per distinct expert, ascending by id (per-expert LDS counts and a CTA prefix sum: a thread per
   expert up to 512, two (2 t, 2 t + 1) up to 1024; a pick a thread, more picks than threads: t, t + 512, ..); a
   slot's picks (at most 16: a row picks an expert once) are an MFMA tile's N. Over 16 rows (route "cta", down
   "slots", up to ``RT_ROWS``) an expert
   with more than 16 picks takes ``ceil(picks / 16)`` consecutive slots (its picks in row order, from a per-expert
   row mask every CTA builds alike, one 32-bit word per 32 rows, in runs of 16: every CTA's slot holds the same
   picks), so a slot is
   still one tile and the slot count stays at most ``rows * K``; the folded shared expert's tasks (every row a pick)
   repeat a 16-row chunk at a time (``sh_own`` too: its CTAs take every chunk's tasks). A row's products and sums do
   not depend on its tile, so its bits are the same at any row count. The table (slot count, each slot's expert and
   picks ``row * K + k``) stays in the CTA's LDS for ug and down.
   ``route="stage"``: one ``<name>.route`` task builds it and publishes it; ug / down tasks poll their slot.
2. ``<name>.ug``: ``rows * K`` slots x ``ug_parts`` tasks (slots past the step's count do nothing): the picks' x
   rows gathered into LDS, gate / up row groups interleaved per 16 (as ``Linear``'s SwiGLU), the expert's weights
   at a 64-bit per-slot base, K split over the 8 waves; the epilogue above -> MXFP8 channel ``<name>.mid`` (width
   K * I per row: pick k's segment at k * I).
3. ``<name>.down``: slots x ``down_parts`` tasks: the picks' MXFP8 m rows gathered, W2_e -> bf16 y_rk into the BF16
   channel ``<name>.y`` (width K * N per row).
4. ``<name>.combine``: rows x ceil(N / 2048) tasks, 4 features a thread: the sum above -> ``out`` (BF16 channel).

TP (``Scenario.tp`` > 1), two layouts (``layout``; ``pack_experts`` gives a rank its share):

- ``expert`` (the reference's): rank r holds experts ``r * E / tp ..``, packed as local ids 0 ..; the route keeps
  this rank's picks only; the combine sums them (others' as 0, same order). Bit-exact against the reference's TP.
- ``intra`` (ATOM's): every rank holds every expert's slice ``r * I / tp ..`` of the expert width (gate / up rows,
  down columns), padded to a whole 128-wide K step (``MoEExperts.inter_on``): SwiGLU stays rank-local, each y_rk is
  the rank's bf16 partial of down. Balanced at any routing; D10, not bit-exact.

Either way (``reduce="push"``, the default) the combine pushes the fp32 partial straight to every rank's peer region
``<name>.part`` (a task per chunk x rank), and ``<name>.dsum`` sums the ranks' partials in rank order (fp32), adds a
``shared`` channel and writes ``out``; ``reduce`` = a ``PeerAllReduce`` proto instead goes combine -> local partial
-> ``PeerAllReduce`` -> ``finish``. Every rank's ``out`` is the same bits. ``reduce="2shot"``: each column quarter
to its owner only (``peer.push_rs``), the partials as (fp32, tag) pairs or, ``rs_fmt`` "f32p" / "f32q", packed
(the owner's sums the same bits); its readers (``<name>.own``, a sharded seam's
``PeerPartial("rs", rs_fmt=)``) take the same format.

``down="cols"`` (a knob: slower so far): one stage of column tasks, each 16
output columns x every slot (+ the folded shared expert), the picks' mid rows gathered into LDS, the combine in LDS,
the partial pushed from the epilogue.

Weights (``pack``): per expert ``w1`` / ``w3`` / ``w2`` (``.packed`` MXFP4 bytes, ``.codes``), all experts'
units concatenated (``gemv.pack_weight`` layout per expert): args ``<name>.ug.w`` / ``.ug.ws`` / ``.down.w`` /
``.down.ws``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

from ...device.gfx950 import buffer_ops as bo
from ...device.gfx950 import channels as chan
from ...device.gfx950 import gemv
from ...device.gfx950 import mailbox as mb
from ...device.gfx950.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    block_excl_scan,
    rsrc,
    traced,
    uniform,
    wave_incl_scan,
)
from ...device.gfx950.peer import (
    partial3_pairs,
    partial_pairs,
    push_partial,
    push_rs,
    read_ag,
    shot2_pairs,
    sum_partials,
)
from ...plan.program import LdsKeep, Region, Scenario, Stage
from ...plan.trace import Space, note
from ...target import MI355X
from ..base import Block, Channel, Fmt
from .peer import PeerAllReduce

WAVES, THREADS = MI355X.waves, MI355X.threads
TILE = 16  # picks a slot: an MFMA's N
SH_PREFETCH = (
    8  # the folded shared expert's MXFP8 units in flight a wave (32 B a lane each)
)
RT_ROWS = 64  # route "cta": the most step rows (an expert's row mask in two words; over 16 rows runs of 16 picks)
A16_ROWS = 8  # A16W4 (``acts="bf16"``): the most step rows (their bf16 x rows sit in LDS whole: 8 x 12 KB at 6144)
TASK_BYTES = (
    80 << 10,
    40 << 10,
)  # a ug / down task's fixed cost (B gather, K-wave reduce, epilogue) as weight bytes
# at a CTA's streaming rate (~3 / ~1.5 us at ~27 GB/s), for ``MoEExperts.groups`` (calibrated)
# intra TP row groups a ug / down part (``MoEExperts.groups``); down 24: e.g. 384 groups x 16 slots = 256 tasks at TP4,
# one round of 156 KB instead of 192 of 208 KB
UG_CANDS, DOWN_CANDS = (4, 8, 12), (16, 24, 32)
INTRA_DOWN_G = 32  # intra TP: down's row groups a task (its K is the rank's slice: 5 steps at V4.1 TP4)
SH_G, SH_DG = (
    4,
    8,
)  # its row groups a gate / up and a down task (gate / up: 72 tasks at TP1, 20 at TP4)
SH_DG_SOLO = 16  # ... a down task at TP without the down queue (``MoEExperts.cfg``)


@dataclass(frozen=True)
class _Cfg:
    name: str
    rows: int
    k: int  # picks a row (top-k)
    experts: int
    hidden: int  # model width: x's K, down's N
    inter: int  # expert width on this rank: ug's N / 2, down's K (``intra``: the rank's slice, padded)
    limit: float
    x: str
    idx: str
    wts: str
    shared: str  # "" if none
    out: str
    ug_g: int  # row groups a ug task (gate / up pairs: ug_g / 2)
    down_g: int
    prefetch: int = (
        8  # weight units a wave keeps in flight (16 put the TP4 6-row layer at 256 VGPRs)
    )
    pre_b: int = 4  # of which issued before the task's picks and B rows are polled (ug)
    down_pre_b: int = 4  # the same for down
    tp: int = 1  # TP ranks; the combine's partial is all-reduced (fp32)
    layout: str = (
        "expert"  # TP: "expert" (rank r holds experts r * e_local ..) or "intra" (every expert, a slice);
    )
    # "intra_gu": intra with the gate / up weight row groups as ATOM's w13 holds them (every gate row group, then every
    # up one: an engine's bytes read in place), the scale codes still interleaved (``pack_experts``)
    live: int = (
        0  # intra: the slice's real width (``inter`` - live is zero padding: its ug weights are not loaded)
    )
    sh_inter: int = (
        0  # the folded shared expert's width on this rank (padded to 128; 0: none folded)
    )
    sh_live: int = 0  # ... its real width (the rest zero padding, not loaded)
    sh_g: int = 8  # row groups a shared ug task
    sh_dg: int = 8  # row groups a shared down task
    down: str = (
        "cols"  # "cols": a down task owns 16 output columns x every slot; "slots": (slot, part)
    )
    sh_sep: bool = (
        False  # TP push: the folded shared expert's fp32 partial travels as its own field, so the sum can
    )
    # round it to bf16 and add it after the routed sum, as the reference (``MoE.forward``: all-reduce, then + shared)
    sh_prefetch: int = SH_PREFETCH
    sh_fmt: str = (
        "mxfp8"  # the folded shared expert's weights: "mxfp8" (V4.1's FP8) or "mxfp4"
    )
    sh_ks: int = (
        1  # the folded shared expert's gate / up: K slices a part (tasks), summed in slice order (fp32)
    )
    rs: bool = (
        False  # reduce "2shot": the combine pushes each column quarter to its owner only (peer.push_rs)
    )
    own_quads: int = (
        MI355X.threads
    )  # 2shot owner stage: element quads a task (a thread each; fewer: more tasks)
    dq: bool = (
        False  # down as a work queue (routed + folded shared items, an atomic counter; route_mode "cta")
    )
    route_mode: str = (
        "cta"  # "stage": one route task publishes the slot table; "cta": each CTA builds it
    )
    sh_own: bool = (
        False  # the folded shared expert's stages on CTAs of their own (the top ``sh_ctas``): no rt there,
    )
    # the routed rt / ug / down on the other ``n_r`` CTAs, a task a CTA looping over its (slot, part) items
    rt_picks: bool = (
        False  # rt: only the threads with a pick poll the routing (into LDS), not every thread of the CTA
    )
    rt_lean: bool = (
        False  # rt (with rt_picks): the counts zeroed before the poll, both scans on one barrier pair
    )
    # the activation and its options, always these values (w silu(g) u, the route weight on the act, the combine's
    # one fp32 rounding); kept: the stage keys hash every field
    act: str = "silu"
    alpha: float = 1.702
    bias: bool = False
    wdown: bool = False
    cround: str = "f32"
    out_arg: str = (
        ""  # the combine (TP: dsum) also stores ``out`` to this plain bf16 arg [rows, hidden]
    )
    rplan: object = (
        None  # route_from: the router's route plan (``MoERouter.route_plan``): rt computes the picks
    )
    ug_ring: int = (
        0  # ug: weight units a wave from an LDS ring, filled by the CTA's previous ug task (``_gemv``)
    )
    down_ring: int = (
        0  # down: a task's first weight units a wave in an LDS ring, copied before it polls mid
    )
    rs_fmt: str = (
        "f32"  # rs: the reduce-scatter partials' format, "f32" pairs or "f32p" / "f32q" packed (peer P16)
    )
    part_fmt: str = (
        "f32"  # TP push: the pushed partial's format (``peer.partial_pairs``): "f32" | "bf16"
    )
    acts: str = (
        "mxfp8"  # the experts' activations (x and mid): "mxfp8" (W4A8) or "bf16" (A16W4, ``MoEExperts``)
    )
    x_lds: bool = (
        False  # route "cta": the rt stage stages every row of x in LDS; ug tasks read their picks' rows there
    )
    # the routed experts' scale codes as ATOM holds them (``atom_cols``: aiter's ``shuffle_scale`` order, a 32-row
    # block contiguous over K, rows padded to its 8-group columns), read in place; False: the hi runtime's unit order
    atom_scales: bool = False

    @property
    def xl_pitch(
        self,
    ) -> tuple[int, int]:  # x_lds: a row's words (+ 4 pad) and (MXFP8) codes (+ 1)
        return (
            (self.hidden // 2 + 4, 0)
            if self.b16
            else (self.hidden // 4 + 4, self.hidden // 32 + 1)
        )

    @property
    def xl_words(self) -> int:
        pitch, spitch = self.xl_pitch
        return self.rows * (pitch + spitch) if self.x_lds else 0

    @property
    def b16(
        self,
    ) -> (
        bool
    ):  # A16W4: bf16 x / mid rows in LDS, the MXFP4 units decoded to bf16 (``gemv16``)
        return self.acts == "bf16"

    @property
    def sh_ctas(self) -> int:
        return max(self.sh_tasks, self.sh_down_items) if self.sh_own else 0

    @property
    def n_r(self) -> int:  # CTAs of the routed stages (sh_own); a task each
        return MI355X.cus - self.sh_ctas

    @property
    def sh_y(
        self,
    ) -> str:  # sh_sep: the shared expert's fp32 down partial, [rows, hidden]
        return f"{self.name}.sh_y"

    @property
    def sh(self) -> str:  # the folded shared expert's MXFP8 mid channel
        return f"{self.name}.sh_mid"

    @property
    def y_k(self) -> int:  # y segments a row: the K picks, + the folded shared expert's
        return self.k + (1 if self.sh_inter and not self.sh_sep else 0)

    @property
    def sh_kp(
        self,
    ) -> (
        str
    ):  # sh_ks > 1: every K slice's fp32 gate / up sums, [slice][row][2 sh_inter]
        return f"{self.name}.sh_kp"

    @property
    def rc(
        self,
    ) -> (
        int
    ):  # 16-row chunks of the step (the folded shared expert's tiles; more than 1: chunked slots)
        return -(-self.rows // TILE)

    @property
    def rt_pp(
        self,
    ) -> (
        int
    ):  # route "cta": picks a thread of the rt stage (thread t: picks t, t + THREADS, ..)
        return -(-self.slots // THREADS)

    @property
    def rt_mw(
        self,
    ) -> (
        int
    ):  # over 16 rows: an expert's row-mask words (32 rows each; word w at LDS w x experts)
        return -(-self.rows // 32)

    @property
    def sh_tasks(
        self,
    ) -> int:  # the folded shared gate / up: parts x K slices, a 16-row chunk at a time
        return self.sh_parts * self.sh_ks * self.rc

    @property
    def sh_down_items(self) -> int:  # its down: parts, a 16-row chunk at a time
        return self.sh_down_parts * self.rc

    @property
    def sh_parts(self) -> int:
        return 2 * self.sh_inter // 16 // self.sh_g

    @property
    def sh_down_parts(self) -> int:
        return self.hidden // 16 // self.sh_dg

    @property
    def e_route(
        self,
    ) -> int:  # the experts the route counts: this rank's (``expert`` TP) or all
        return self.e_local if self.ep else self.experts

    @property
    def e_pairs(
        self,
    ) -> (
        bool
    ):  # route "cta": two experts a thread (2 t, 2 t + 1), more experts than threads
        return self.e_route > THREADS

    @property
    def ep(
        self,
    ) -> bool:  # experts sharded across ranks: the route keeps this rank's picks only
        return self.tp > 1 and self.layout == "expert"

    @property
    def e_local(self) -> int:
        return self.experts // self.tp

    @property
    def part(self) -> str:  # TP: the rank's fp32 partial, then the reduced sum
        return f"{self.name}.part"

    @property
    def red(self) -> str:
        return f"{self.name}.red"

    @property
    def slots(self) -> int:
        return self.rows * self.k

    @property
    def route(self) -> str:
        return f"{self.name}.route"

    @property
    def mid(self) -> str:
        return f"{self.name}.mid"

    @property
    def y(self) -> str:
        return f"{self.name}.y"

    @property
    def ug_parts(self) -> int:
        return 2 * self.inter // 16 // self.ug_g

    @property
    def ug_live_parts(
        self,
    ) -> (
        int
    ):  # ug parts with weight bytes (intra: the rest are the slice's zero padding)
        if not self.live or self.live >= self.inter:
            return self.ug_parts
        return -(-(2 * self.live // 16) // self.ug_g)

    @property
    def down_parts(self) -> int:
        return self.hidden // 16 // self.down_g

    # route region (pairs): [0] slot count; [1 + s] expert; [1 + S + s] pick count (slots past the count: 0, 0);
    # [1 + 2S + 16 s + j] pick; [1 + 2S + 16 S + 16 s + j] its route weight

    # route region (pairs): [0] slot count; [1 + s] expert; [1 + S + s] pick count (slots past the count: 0, 0);
    # [1 + 2S + 16 s + j] pick; [1 + 2S + 16 S + 16 s + j] its route weight
    def r_count(self) -> int:
        return 0

    def r_expert(self, s):
        return 1 + s

    def r_npick(self, s):
        return 1 + self.slots + s

    def r_pick(self, s, j):
        return 1 + 2 * self.slots + s * TILE + j

    def r_wt(self, s, j):
        return 1 + 2 * self.slots + (self.slots + s) * TILE + j

    @property
    def gemv_lds(
        self,
    ) -> int:  # the larger of the gathered rows and the K-wave partials
        kd = max(self.inter, self.sh_inter)
        ug = TILE * (self.hidden // 4 + 4 + self.hidden // 32 + 1)
        down = TILE * (kd // 4 + 4 + kd // 32 + 1)
        gemvs = [(self.hidden, self.ug_g), (self.inter, self.down_g)]
        if self.sh_inter:
            gemvs += [(self.hidden, self.sh_g), (self.sh_inter, self.sh_dg)]
        red = (
            max(_wk(kk // 128, g) * g for kk, g in gemvs) * TILE * 16
        )  # [K wave][g][16 picks][16 features]
        return max(ug, down, red)

    def atom_cols(self, kk: int) -> int:
        """The scale columns of a routed weight of K ``kk`` as ATOM holds them: its K groups, padded to 8 (aiter's
        ``shuffle_scale``); the hi runtime's order: the K groups."""
        return -(-kk // 32 // 8) * 8 if self.atom_scales else kk // 32

    @property
    def ws_words(
        self,
    ) -> (
        int
    ):  # a task's weight scale codes staged in LDS (after the picks): 4 words a row group's 32 K
        ug = max(
            self.ug_g * self.atom_cols(self.hidden),
            self.sh_g * (self.hidden // 32) if self.sh_inter else 0,
        )
        down = max(
            self.down_g * self.atom_cols(self.inter), self.sh_dg * (self.sh_inter // 32)
        )
        return max(ug, down) * 4

    @property
    def pick_lds(self) -> int:  # 16 picks, then 16 route weights
        return self.gemv_lds

    # route="cta": the slot table in LDS after the ug / down words (``stage_lds``): [0] slot count; [1 + s] expert;
    # [1 + S + s] its first pick's place in the pick list (s <= count: [count] = the list's length; a slot's pick
    # count is the difference); [2 + 2S + i] the pick list (the slots' picks in slot order), [2 + 3S + i] their route
    # weights
    @property
    def stage_lds(
        self,
    ) -> (
        int
    ):  # ug / down: the gathered rows / partials, 16 picks + weights, the scale codes
        return self.gemv_lds + 2 * TILE + self.ws_words

    @property
    def rt(self) -> int:
        return self.stage_lds

    def rt_expert(self, s):
        return self.rt + 1 + s

    def rt_start(self, s):
        return self.rt + 1 + self.slots + s

    def rt_pick(self, i):
        return self.rt + 2 + 2 * self.slots + i

    def rt_wt(self, i):
        return self.rt + 2 + 3 * self.slots + i

    @property
    def rt_words(self) -> int:
        return 2 + 4 * self.slots if self.route_mode == "cta" else 0

    @property
    def ring_lds(
        self,
    ) -> (
        int
    ):  # ug_ring / down_ring: the ring's first word (past the dq word), WAVES x n units
        return self.stage_lds + self.rt_words + 1

    @property
    def ring_words(self) -> int:
        n = max(self.ug_ring, self.down_ring)
        return WAVES * n * 256 if n else 0

    # dq: the queue's counter (a scratch word, reset by the rt stage each launch) and its ready flag; a grabbed item
    # reaches the CTA's threads through LDS word ``dq_word`` (after the slot table)
    @property
    def dq_ctr(self) -> str:
        return f"{self.name}.dq"

    @property
    def dq_ready(self) -> str:
        return f"{self.name}.dqr"

    @property
    def dq_word(self) -> int:
        return self.stage_lds + self.rt_words


# ------------------------------------------------------------------ 1. route


def _live_picks(k, c: _Cfg):
    """The picks the route counts: every row's (``c.slots`` = rows x K), or (``Program.dyn_rows``) the live rows'
    (``k.n`` x K): a dead row's clamped poll returns the last live row's ids, which must not join their slots (a slot
    holds at most a row's worth of picks: an MFMA tile's N)."""
    return c.slots if isinstance(k.n, int) else k.n * c.k


def _route(k, t, c: _Cfg):
    """Thread p < rows * K: pick p; thread e < the expert count: expert e. Per expert an LDS count of its picks
    (a workgroup atomic add, whose old value is the pick's place in its slot: arrival order, which only permutes a
    slot's MFMA columns); the experts with picks, numbered by a CTA prefix sum, are the slots (ascending ids).
    """
    n, ne = c.slots, c.e_route
    p = fx.min(k.tid, fx.Int32(n - 1))
    e, wt = (v[0] for v in mb.poll([(k.mb[c.idx], p, 1), (k.mb[c.wts], p, 1)], k.tag))
    local = fx.Int32(1)
    if c.ep:  # this rank's experts only
        e = e - k.rank * c.e_local
        local = ((e >= 0) & (e < c.e_local)).select(fx.Int32(1), fx.Int32(0))
        e = (local == 1).select(e, fx.Int32(0))
    pick_ok = (k.tid < _live_picks(k, c)) & (local == 1)
    cnt, slot_of = k.lds, k.lds + ne
    _route_zero(k, cnt, ne)
    gpu.barrier()
    rank = _route_count(cnt + e, pick_ok)
    gpu.barrier()
    ec = fx.min(k.tid, fx.Int32(ne - 1))
    picks = fx.ptr_load(cnt + ec)
    used = ((k.tid < ne) & (picks > 0)).select(fx.Int32(1), fx.Int32(0))
    slot, count = block_excl_scan(used, k.lane, k.wave, k.lds + 2 * ne)
    _route_slot(k, slot_of, slot, ne)
    _route_put(k, c, used == 1, c.r_expert(slot), ec)
    _route_put(k, c, used == 1, c.r_npick(slot), picks)
    gpu.barrier()
    ps = fx.ptr_load(slot_of + e)
    _route_put(k, c, pick_ok, c.r_pick(ps, rank), p)
    _route_put(k, c, pick_ok, c.r_wt(ps, rank), wt)
    idle = (k.tid < n) & (
        k.tid >= count
    )  # slots past the count: expert 0, no picks (one poll for a task's slot)
    _route_put(k, c, idle, c.r_expert(k.tid), fx.Int32(0))
    _route_put(k, c, idle, c.r_npick(k.tid), fx.Int32(0))
    _route_put(k, c, k.tid == 0, c.r_count(), count)
    gpu.barrier()


@traced
def _route_zero(k, cnt, ne):
    if k.tid < ne:
        fx.ptr_store(fx.Int32(0), cnt + k.tid)


def _route_zero_e(k, c: _Cfg, cnt, ne):
    """The per-expert counts zeroed: a thread's expert, or its two (``e_pairs``); over 32 rows each mask word."""
    for w in range_constexpr(c.rt_mw if c.rc > 1 else 1):
        _route_zero(k, cnt + w * ne if w else cnt, ne)


@traced
def _route_count(addr, ok):
    rank = fx.Int32(0)
    if ok:
        rank = fx.atomic_add(addr, fx.Int32(1), syncscope=rocdl.SyncScope.Workgroup)
    return rank


@traced
def _route_mask(addr, ok, bit):
    if ok:
        fx.atomic_or(addr, bit, syncscope=rocdl.SyncScope.Workgroup)


@traced
def _route_slot(k, slot_of, slot, ne):
    if k.tid < ne:
        fx.ptr_store(slot, slot_of + k.tid)


def _route_put(k, c: _Cfg, ok, i, v):
    _put_if(k, ok, k.mb[c.route], i, v)


@traced
def _put_if(k, ok, a, i, v):
    """Pair ``i`` of region ``a`` := (v, tag) where ``ok`` (no method calls inside the dynamic if:)."""
    if ok:
        mb.put_words(a, i, [v], k.tag)


def _cta_route(k, t, c: _Cfg):
    """route="cta", stage ``<name>.rt``, a task on every CTA: ``_route``'s slot table into this CTA's
    LDS (``_Cfg.rt_*``), where ug and down read it (no route stage, no per-task route polls). The same counts, scan
    and slots (ascending ids, so every CTA numbers the slots alike; a slot's pick order is this CTA's atomic order,
    which only permutes MFMA columns; over 16 rows the row order, from the experts' row masks, so an expert's runs of
    16 hold the same picks on every CTA); the picks as one list in slot order (a second scan places each slot's run).
    The table outlives the stage: the MoE's stages up to down use only LDS words below ``rt``. The build's scratch
    (per-expert counts, slot and run start of each expert, the two scans' wave sums) is at word 0. More picks than
    threads: thread t takes picks t, t + THREADS, .. (``rt_pp``); over 32 rows an expert's mask has ``rt_mw`` words,
    word w at ``w x experts`` (past the first: the slot words, written after the masks are read).
    """
    n, ne = c.slots, c.e_route
    p = fx.min(k.tid, fx.Int32(n - 1))
    ps = [p] + [fx.min(k.tid + j * THREADS, fx.Int32(n - 1)) for j in range(1, c.rt_pp)]
    cnt, slot_of, start_of = k.lds, k.lds + ne, k.lds + 2 * ne
    if const_expr(
        c.rt_lean
    ):  # the counts zeroed while the routing is out: the poll's barrier is theirs too
        _route_zero_e(k, c, cnt, ne)
    if const_expr(
        c.rt_picks
    ):  # threads < n poll their pick into LDS (scratch words past the build's), the rest read
        at = 3 * ne + 2 * WAVES
        _rt_poll_picks(k, c, p, at)
        for j in range_constexpr(1, c.rt_pp):
            _rt_poll_more(k, c, j, at)
        gpu.barrier()
        ews = [
            (
                fx.Int32(fx.ptr_load(k.lds + (at + q))),
                fx.Int32(fx.ptr_load(k.lds + (at + n + q))),
            )
            for q in ps
        ]
    nl = _live_picks(
        k, c
    )  # the picks counted: n, or (``Program.dyn_rows``) the live rows' (a dead row's are clamped)
    es, wts, oks = [], [], []
    for j, (e, wt) in enumerate(ews):
        local = fx.Int32(1)
        es.append(e)
        wts.append(wt)
        oks.append(
            ((k.tid < nl) if j == 0 else (k.tid + j * THREADS < nl)) & (local == 1)
        )
    if const_expr(not c.rt_lean):
        _route_zero_e(k, c, cnt, ne)
        gpu.barrier()
    if const_expr(
        c.rc == 1
    ):  # a count a pick (an atomic add: its place in the expert's slot)
        ranks = [_route_count(cnt + e, ok) for e, ok in zip(es, oks, strict=True)]
        gpu.barrier()
    else:  # an expert's rows as a bit mask (an OR: the same on every CTA), its picks in row order, in runs of 16
        rows = [q // c.k for q in ps]
        for e, ok, row in zip(es, oks, rows, strict=True):
            if const_expr(c.rt_mw == 1):
                _route_mask(cnt + e, ok, fx.Int32(1) << row)
            else:
                _route_mask(cnt + (e + row // 32 * ne), ok, fx.Int32(1) << row % 32)
        gpu.barrier()
    if const_expr(c.rc == 1):
        ec = fx.min(k.tid, fx.Int32(ne - 1))
        picks = (k.tid < ne).select(fx.ptr_load(cnt + ec), fx.Int32(0))
        used = (picks > 0).select(fx.Int32(1), fx.Int32(0))
    else:
        ec = fx.min(k.tid, fx.Int32(ne - 1))
        ranks = [
            _mask_rank(c, cnt, e, row, ne) for e, row in zip(es, rows, strict=True)
        ]
        picks = (k.tid < ne).select(_mask_picks(c, cnt, ec, ne), fx.Int32(0))
        used = (picks + (TILE - 1)) // TILE
    if const_expr(
        c.rt_lean
    ):  # the same two prefixes, their wave totals through LDS together
        (slot, count), (start, total) = _block_excl_scan2(
            k, used, picks, k.lds + 3 * ne
        )
    else:
        slot, count = block_excl_scan(used, k.lane, k.wave, k.lds + 3 * ne)
        start, total = block_excl_scan(picks, k.lane, k.wave, k.lds + (3 * ne + WAVES))
    _route_slot(k, slot_of, slot, ne)
    _route_slot(k, start_of, start, ne)
    if const_expr(c.rc == 1):
        _lds_put_if(k, used == 1, c.rt_expert(slot), ec)
        _lds_put_if(k, used == 1, c.rt_start(slot), start)
    else:
        _rt_runs(k, c, used, slot, ec, start)
    _lds_put_if(k, k.tid == 0, c.rt, count)
    _lds_put_if(k, k.tid == 0, c.rt_start(count), total)
    gpu.barrier()
    for e, wt, ok, rank, q in zip(es, wts, oks, ranks, ps, strict=True):
        at = fx.ptr_load(start_of + e) + rank
        _lds_put_if(k, ok, c.rt_pick(at), q)
        _lds_put_if(k, ok, c.rt_wt(at), wt)
    gpu.barrier()
    if (
        c.dq
    ):  # CTA 0 restarts the down queue: the counter (device scope, drained), then the ready flag
        _dq_reset(k, c, (t == 0) & (k.tid == 0))


def _rt_runs(k, c: _Cfg, used, slot, e, start):
    """Over 16 rows: run j < ``used`` of expert ``e``'s picks is slot ``slot + j``, from its start + 16 j."""
    for j in range_constexpr(c.rc):
        _lds_put_if(
            k, used > j, c.rt_expert(fx.min(slot + j, fx.Int32(c.slots - 1))), e
        )
        _lds_put_if(
            k,
            used > j,
            c.rt_start(fx.min(slot + j, fx.Int32(c.slots - 1))),
            start + j * TILE,
        )


def _mask_rank(c: _Cfg, cnt, e, row, ne):
    """A pick's place among its expert's picks: the expert's mask bits below its row."""
    if c.rt_mw == 1:
        return fx.Int32(fx.ctpop(fx.ptr_load(cnt + e) & ((fx.Int32(1) << row) - 1)))
    rank = None
    for w in range(c.rt_mw):
        lo = 32 * w
        below = (row >= lo + 32).select(fx.Int32(-1), (fx.Int32(1) << fx.min(fx.max(row - lo, fx.Int32(0)),
                                                                             fx.Int32(31))) - 1)  # fmt: skip
        r = fx.Int32(
            fx.ctpop(fx.ptr_load(cnt + (e + w * ne) if w else cnt + e) & below)
        )
        rank = r if rank is None else rank + r
    return rank


def _mask_picks(c: _Cfg, cnt, e, ne):
    """Expert ``e``'s picks: its mask words' bits."""
    tot = None
    for w in range(c.rt_mw):
        r = fx.Int32(fx.ctpop(fx.ptr_load(cnt + (e + w * ne) if w else cnt + e)))
        tot = r if tot is None else tot + r
    return tot


def _block_excl_scan2(k, a, b, buf):
    """``block_excl_scan`` of ``a`` and of ``b`` (wave totals at ``buf`` and ``buf + WAVES``, as the two calls put
    them) behind one barrier: the same prefixes and totals. The words are not read before this barrier by anyone
    (the rt stage's scratch: its previous reader is a previous task, behind the stage's barriers).
    """
    ia, ib = wave_incl_scan(a, k.lane), wave_incl_scan(b, k.lane)
    _scan_tot_put(k, ia, ib, buf)
    gpu.barrier()
    oa = ob = ta = tb = fx.Int32(0)
    for w in range_constexpr(WAVES):
        ca, cb = fx.ptr_load(buf + w), fx.ptr_load(buf + (WAVES + w))
        oa = oa + (w < k.wave).select(ca, 0)
        ob = ob + (w < k.wave).select(cb, 0)
        ta, tb = ta + ca, tb + cb
    return (ia - a + oa, ta), (ib - b + ob, tb)


@traced
def _scan_tot_put(k, ia, ib, buf):
    if k.lane == 63:
        fx.ptr_store(ia, buf + k.wave)
        fx.ptr_store(ib, buf + (WAVES + k.wave))


@traced
def _rt_poll_picks(k, c: _Cfg, p, at):
    if k.tid < c.slots:
        e, wt = (
            v[0] for v in mb.poll([(k.mb[c.idx], p, 1), (k.mb[c.wts], p, 1)], k.tag)
        )
        fx.ptr_store(fx.Int32(e), k.lds + (at + p))
        fx.ptr_store(fx.Int32(wt), k.lds + (at + c.slots + p))


@traced
def _rt_poll_more(k, c: _Cfg, j, at):
    """``_rt_poll_picks`` of a thread's pick ``j`` > 0 (more picks than threads: pick tid + j x THREADS)."""
    q = k.tid + j * THREADS
    if q < c.slots:
        e, wt = (
            v[0] for v in mb.poll([(k.mb[c.idx], q, 1), (k.mb[c.wts], q, 1)], k.tag)
        )
        fx.ptr_store(fx.Int32(e), k.lds + (at + q))
        fx.ptr_store(fx.Int32(wt), k.lds + (at + c.slots + q))


@traced
def _dq_reset(k, c: _Cfg, ok):
    if ok:
        note(c.dq_ctr, Space.SCRATCH, "put")
        bo.buffer_store(
            fx.Int32(0), rsrc(k.mb[c.dq_ctr].value), 0, cache_modifier=CM_DEV
        )
        rocdl.s_waitcnt(vmcnt=0)
        mb.put(k.mb[c.dq_ready], 0, fx.Int32(1), k.tag)


@traced
def _lds_put_if(k, ok, i, v):
    if ok:
        fx.ptr_store(fx.Int32(v), k.lds + i)


def _rt_slot(k, c: _Cfg, s):
    """route="cta": ``_slot`` from this CTA's table (a slot past the count: garbage, unused)."""
    sc = fx.min(s, fx.Int32(c.slots - 1))
    count, e, a, b = (uniform(fx.ptr_load(k.lds + i)) for i in (c.rt, c.rt_expert(sc), c.rt_start(sc),
                                                                 c.rt_start(sc + 1)))  # fmt: skip
    return count, e, b - a


def _pick_lds(k, c: _Cfg, s, cta):
    """LDS words of slot ``s``'s picks and of their route weights: this CTA's table (route="cta") or the staged 16."""
    if not cta:
        return c.pick_lds, c.pick_lds + TILE
    start = uniform(fx.ptr_load(k.lds + c.rt_start(s)))
    return c.rt_pick(start), c.rt_wt(start)


# ------------------------------------------------------------------ the gathered GEMV (ug and down)


def _slot(k, c: _Cfg, s):
    """(slot count, expert, pick count) of slot ``s``: one poll."""
    a = k.mb[c.route]
    got = mb.poll(
        [(a, c.r_count(), 1), (a, c.r_expert(s), 1), (a, c.r_npick(s), 1)], k.tag
    )
    return tuple(uniform(g[0]) for g in got)


def _slot_picks(k, c: _Cfg, s, npick, pick_lds, with_weights):
    """Slot ``s``'s picks (clamped to the last) into LDS words ``pick_lds .. + 16`` and, ``with_weights``, their
    route weights after them."""
    a = k.mb[c.route]
    j = fx.min(k.lane % TILE, npick - 1)
    specs = [(a, c.r_pick(s, 0) + j, 1)] + (
        [(a, c.r_wt(s, 0) + j, 1)] if with_weights else []
    )
    _stage_picks(k, pick_lds, [v[0] for v in mb.poll(specs, k.tag)])
    gpu.barrier()


@traced
def _stage_picks(k, base, vals):
    if k.tid < TILE:
        for i in range_constexpr(len(vals)):
            fx.ptr_store(vals[i], k.lds + (base + i * TILE + k.tid))


def _rows_sh(k, c: _Cfg, pick_lds, *gather, ch=None, npick=None):
    """``stage_b`` of the folded shared expert's tasks: picks = every row (pick j = row j as ``j * K``), weight 1;
    ``ch`` (over 16 rows): the rows of 16-row chunk ``ch`` (``npick`` of them)."""

    def stage():
        if ch is None:
            _stage_picks(
                k,
                pick_lds,
                [
                    fx.min(k.tid, fx.Int32(c.rows - 1)) * c.k,
                    fx.Float32(1.0).bitcast(fx.Int32),
                ],
            )
            gpu.barrier()
            _gather_rows(k, c, pick_lds, fx.Int32(c.rows), *gather)
        else:
            row = fx.min(k.tid + ch * TILE, fx.Int32(c.rows - 1))
            _stage_picks(k, pick_lds, [row * c.k, fx.Float32(1.0).bitcast(fx.Int32)])
            gpu.barrier()
            _gather_rows(k, c, pick_lds, npick, *gather)

    return stage


def _sh_chunk(c: _Cfg, t, per: int):
    """Over 16 rows: shared task / item ``t`` -> (its 16-row chunk, the task within the chunk's ``per``, the chunk's
    rows)."""
    ch, t = t // per, t % per
    return ch, t, fx.min(fx.Int32(c.rows) - ch * TILE, fx.Int32(TILE))


def _rows_rt(k, c: _Cfg, npick, pick_lds, *gather):
    """``stage_b`` with route="cta": the picks are already in this CTA's table; their rows only."""

    def stage():
        _gather_rows(k, c, pick_lds, npick, *gather)

    return stage


GATHER_ROWS = 2  # rows a gather batch at least: a slot's picks are gathered in batches, not all 16 rows at once


def _rows_b(k, c: _Cfg, s, npick, pick_lds, with_weights, *gather):
    """``_gemv``'s ``stage_b``: slot ``s``'s picks into LDS, then their rows (``_gather_rows`` with ``gather``)."""

    def stage():
        _slot_picks(k, c, s, npick, pick_lds, with_weights)
        _gather_rows(k, c, pick_lds, npick, *gather)

    return stage


def _gather_n(kk: int) -> int:
    """Rows a gather batch of ``kk``-element rows: as many as one pass of the CTA's threads covers (a row's two-pair
    specs: kk / 8), at least ``GATHER_ROWS``. Each batch is a poll round trip, which under a saturated HBM costs
    ~1.3 us: the 6-row shared down's 640-wide B took 3 batches, ~4 us; x rows (5120) stay 2.
    """
    return max(GATHER_ROWS, min(TILE, THREADS // max(kk // 8, 1)))


def _gather_rows(k, c: _Cfg, pick_lds, npick, *row):
    """The slot's ``npick`` picks' MXFP8 rows into LDS rows 0 .. npick - 1 (``_gather_batch``), in a dynamic loop of
    ``_gather_n``-row batches (LDS rows past npick hold stale words: their MFMA columns are never read). A16W4
    (``c.b16``): their bf16 rows, every staged row in one poll."""
    gr = _gather_n(row[3])
    _gather_loop(k, c, pick_lds, uniform((npick + (gr - 1)) // gr), npick - 1, row)


@traced
def _gather_loop(k, c: _Cfg, pick_lds, batches, jmax, row):
    gr = _gather_n(row[3])
    for b in range(batches):
        _gather_batch(k, c, pick_lds, fx.Int32(b) * gr, jmax, *row)


def _gather_batch(
    k, c: _Cfg, pick_lds, j0, jmax, src, width, seg_w, kk, pitch, spitch, koff=0
):
    """Rows j0 .. j0 + _gather_n(kk) - 1 (row j = pick j of the slot; rows past ``jmax``, its last, repeat it)
    from channel ``src`` (width ``width`` a row) at segment ``k * seg_w`` of the pick's row (``seg_w`` 0: the whole
    row from 0), ``kk`` elements from element ``koff`` of it. Words at ``j * pitch``, codes at ``16 * pitch + j *
    spitch``."""
    words, codes = kk // 4, kk // 32
    gr = _gather_n(kk)
    specs, dests = [], []
    nq = gr * words // 2  # two-pair specs
    for it in range(-(-nq // THREADS)):
        i = fx.min(k.tid + it * THREADS, fx.Int32(nq - 1))
        j, w2 = fx.min(j0 + i // (words // 2), jmax), i % (words // 2)
        pick = fx.ptr_load(k.lds + (pick_lds + j))
        r, kp = pick // c.k, pick % c.k
        base = r * (width // 4) + kp * (seg_w // 4) + koff // 4
        specs.append((k.mb[f"{src}.q"], base + w2 * 2, 2))
        dests.append(j * pitch + w2 * 2)
    for it in range(-(-gr * codes // THREADS)):
        i = fx.min(k.tid + it * THREADS, fx.Int32(gr * codes - 1))
        j, g = fx.min(j0 + i // codes, jmax), i % codes
        pick = fx.ptr_load(k.lds + (pick_lds + j))
        r, kp = pick // c.k, pick % c.k
        specs.append(
            (
                k.mb[f"{src}.s"],
                r * (width // 32) + kp * (seg_w // 32) + koff // 32 + g,
                1,
            )
        )
        dests.append(TILE * pitch + j * spitch + g)
    got = mb.poll(specs, k.tag)
    for g, d in zip(got, dests, strict=True):
        for e in range(len(g)):
            fx.ptr_store(g[e], k.lds + (d + e))


def _wk(steps, g):
    """Waves along K: the most of (8, 4, 2, 1) that divides the K steps, the other waves splitting the ``g`` row
    groups (V4.1 down: 18 steps -> 2; intra TP4 down: 5 steps -> 1, where 8 left 3 waves idle). None fits: 8, with
    the waves past the last step loading zeros."""
    return next(
        (wk for wk in (8, 4, 2, 1) if steps % wk == 0 and g % (WAVES // wk) == 0), WAVES
    )


def _gemv(k, c: _Cfg, e, g, kk, n_rows, part, w_arg, s_arg, pitch, spitch, stage_b, pre_b, live_rg=0,
          fmt=gemv.MXFP4, prefetch=None, live_k=0, k0=0, nsteps=None, mk=0, ring=None, b_at=0, xs_at=None,
          brow=None, gu_split=False, cols=0):  # fmt: skip
    """The ``g`` row groups ``part * g ..`` of expert ``e``'s ``n_rows`` x ``kk`` weight, all 16 B operand columns:
    wave w takes K steps w % wk, + wk, ... of row groups (w // wk) * g / (8 / wk) .. (``_wk``); partial sums
    [K wave][g][16 picks][16 features] into LDS. ``stage_b()`` gathers the B operand rows into LDS: called after the
    first weight loads are issued, so they fly while it waits on its producers. ``live_rg`` > 0: row groups from
    it on are zero padding: their units read 0 without touching memory. ``fmt``: the weight format (the folded
    shared expert is MXFP8); ``prefetch``: units in flight a wave (None: the config's). ``live_k`` > 0: K from it on
    is zero padding (intra's padded down K): those lanes' chunks read 0 without touching memory. ``k0`` /
    ``nsteps``: only K steps k0 .. k0 + nsteps - 1 (a K slice; ``stage_b`` gathers that slice of B). ``ring``:
    the wave's first ``ring.n`` units come from its LDS ring slots (``_ring_word``; ``_ug_body`` fills
    them for the CTA's next task of the stage after its epilogue). ``cols`` > 0: the scale codes are ATOM's in place
    (``_Cfg.atom_scales``: aiter's ``shuffle_scale`` order, ``cols`` columns a row; full K, ``g`` even).
    """
    prefetch = prefetch or c.prefetch
    full = kk // 128  # the weight's K steps (its unit layout)
    steps = nsteps or full  # this task's
    sliced = steps != full
    w_bytes = fmt.weight_bytes(n_rows, kk)
    s_bytes = n_rows * (cols or kk // 32)
    if cols and (sliced or g % 2 or gu_split):
        raise ValueError(
            f"MoEExperts {c.name}: ATOM's scales need full K, even row groups and the interleaved rows"
        )
    r_w = rsrc(k.args[w_arg] + fx.Int64(e) * fx.Int64(w_bytes), w_bytes)
    r_s = rsrc(k.args[s_arg] + fx.Int64(e) * fx.Int64(s_bytes), s_bytes)
    wk = _wk(steps, g)
    gw = g // (WAVES // wk)  # row groups a wave
    kw, g0 = k.wave % wk, (k.wave // wk) * gw
    per = -(-steps // wk)
    units = [(i, jg) for i in range(per) for jg in range(gw)]

    def unit_of(i, jg):
        st = kw + wk * i
        valid = None if steps % wk == 0 else st < steps
        stc = st if valid is None else fx.min(st, fx.Int32(steps - 1))
        rg = part * g + g0 + jg
        if live_rg:
            valid = (rg < live_rg) if valid is None else valid & (rg < live_rg)
        return (
            _w_rg(rg, n_rows, gu_split) * full + (k0 + stc if sliced else stc),
            valid,
            stc,
        )

    def issue(u):
        i, jg = units[u]
        unit, valid, stc = unit_of(i, jg)
        k_left = (
            fx.Int32(live_k) - (k0 + stc if sliced else stc) * 128 if live_k else None
        )
        return gemv.load_weight(r_w, unit, k.lane, valid, fmt, k_left)

    nr = 0 if ring is None else min(ring.n, len(units))  # units from the ring
    # the task's scale codes (contiguous: g row groups x every K step, 64 B a unit) with 16 B loads, issued first and
    # staged in LDS (as ``Linear``'s): one load per lane per unit cost 2-6 us at TP4
    ws_base = c.gemv_lds + 2 * TILE
    # ATOM's: the task's g row groups are g / 2 contiguous 32-row blocks of cols x 32 B (16 B chunks: g x cols)
    chunks = g * cols if cols else g * steps * 4
    codes = []
    for r in range(-(-chunks // THREADS)):
        ch = fx.min(fx.Int32(r * THREADS) + k.tid, fx.Int32(chunks - 1))
        src = part * chunks + ch
        if sliced:  # a K slice: g runs of ``steps`` units' codes
            src = ((part * g + ch // (steps * 4)) * full + k0) * 4 + ch % (steps * 4)
        codes.append((gemv.load_scales(r_s, src), ws_base + ch * 4))
    codes_lds = fx.recast_iter(fx.Uint8, k.lds) + ws_base * 4
    q = [None] * nr + [issue(u) for u in range(nr, min(nr + pre_b, len(units)))]
    for v, off in codes:
        fx.ptr_store(v, k.lds + off)
    stage_b()
    gpu.barrier()
    q += [issue(u) for u in range(len(q), min(nr + prefetch, len(units)))]
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(gw)]
    row, j = k.lane % 16, k.lane // 16
    xs_base = (
        TILE * pitch if xs_at is None else xs_at
    )  # B rows at ``b_at`` (x_lds: the resident rows, ``brow``)
    if brow is None:
        brow = fx.min(row, fx.Int32(c.rows - 1)) if c.b16 else row
    for u in range(len(units)):
        if u >= nr and u + prefetch < len(units):
            q.append(issue(u + prefetch))
        i, jg = units[u]
        st = fx.min(kw + wk * i, fx.Int32(steps - 1))
        x, xs = gemv.b_operand(
            k.lds + b_at, k.lds + xs_base, brow, j, st, pitch, spitch
        )
        # a ring unit: this wave's copy landed (the B rows' polls and the codes' loads, issued after it, were waited)
        w = (
            q[u]
            if u >= nr
            else fx.Vector(
                fx.ptr_load(k.lds + _ring_word(c, ring, u, k), result_type=gemv._v4i())
            )
        )
        if const_expr(
            cols > 0
        ):  # aiter's order: lane l's codes of a (row-group pair, K-step pair) one dword
            lg = g0 + jg
            at = (
                (lg // 2) * (cols * 32)
                + (st // 2) * 256
                + k.lane * 4
                + (st % 2) * 2
                + lg % 2
            )
        else:
            at = ((g0 + jg) * steps + st) * 64 + k.lane
        ws = fx.Int32(fx.ptr_load(codes_lds + at))
        acc[jg] = gemv.mfma_wxa8(acc[jg], w, ws, x, xs, fmt)
    gpu.barrier()  # every wave done with the gathered rows: the partials overlay them
    for jg in range(gw):
        off = ((kw * g + g0 + jg) * TILE + row) * 16 + j * 4
        fx.ptr_store(acc[jg], k.ldsf + off)
    gpu.barrier()
    return wk


def _ring_word(c: _Cfg, ring, u, k):
    """LDS word of this lane's 16 B of ring unit ``u`` (the wave's slots: ``ring.n`` units of 1 KiB)."""
    return c.ring_lds + (k.wave * ring.n + u) * 256 + k.lane * 4


def _w_rg(rg, n_rows, gu_split):
    """The weight row group of task row group ``rg`` (gate / up interleaved: 2 i + u): itself, or (``gu_split``, layout
    "intra_gu") ATOM's w13 row group, every gate one then every up one."""
    return (rg % 2) * (n_rows // 32) + rg // 2 if gu_split else rg


def _wave_sum(k, g, gl, n, f4, wk):
    """Sum over the K waves (in order) of the partials of row group ``gl``, pick ``n``, features f4 * 4 .. + 3."""
    tot = None
    for w in range(wk):
        v = fx.Vector(fx.ptr_load(k.ldsf + (((w * g + gl) * TILE + n) * 16 + f4 * 4),
                                  result_type=fx.Vector.make_type(4, fx.Float32)))  # fmt: skip
        vals = [v[e] for e in range(4)]
        tot = vals if tot is None else [tot[e] + vals[e] for e in range(4)]
    return tot


# ------------------------------------------------------------------ 2. ug


def _ug(k, t, c: _Cfg):
    s, part = t // c.ug_parts, t % c.ug_parts
    count, e, npick = _rt_slot(k, c, s) if c.route_mode == "cta" else _slot(k, c, s)
    _ug_live(k, c, s, part, count, e, npick)


def _ug_items(k, t, c: _Cfg):
    """sh_own: CTA t's items t, t + n_r, ... below the step's count x ug_parts: first every slot's parts with weights
    (``ug_live_parts``), then the zero-padding parts (intra: no weight bytes, they only write mid's zero tail), so a
    second round, if any, is the cheap items."""
    count = uniform(fx.ptr_load(k.lds + c.rt))
    _ug_loop(k, c, t, count * c.ug_parts, count * c.ug_live_parts)


@traced
def _ug_loop(k, c: _Cfg, t, total, heavy):
    lp, pp = c.ug_live_parts, c.ug_parts - c.ug_live_parts
    for i in range(t, total, c.n_r):
        j = i - heavy
        s = (i < heavy).select(i // lp, j // max(pp, 1))
        part = (i < heavy).select(i % lp, lp + j % max(pp, 1))
        _, e, npick = _rt_slot(k, c, s)
        _ug_body(k, c, s, part, e, npick)


@traced
def _ug_live(k, c: _Cfg, s, part, count, e, npick, ring=None):
    if s < count:
        _ug_body(k, c, s, part, e, npick, ring=ring)


def _sh_ug(k, t, c: _Cfg):
    """The folded shared expert's gate / up, part ``t // sh_ks``, K slice ``t % sh_ks``: every row is a pick (weight
    1), MXFP8 weights (over 16 rows: task t of a 16-row chunk's)."""
    if const_expr(c.rc == 1):
        _ug_body(
            k,
            c,
            0,
            t // c.sh_ks,
            fx.Int32(0),
            fx.Int32(c.rows),
            sh=True,
            kh=t % c.sh_ks,
        )
    else:
        ch, t, npick = _sh_chunk(c, t, c.sh_parts * c.sh_ks)
        _ug_body(
            k, c, 0, t // c.sh_ks, fx.Int32(0), npick, sh=True, kh=t % c.sh_ks, ch=ch
        )


def _ug_body(k, c: _Cfg, s, part, e, npick, sh=False, kh=0, ch=None, ring=None):
    pitch, spitch = (
        c.hidden // 2 + 4 if c.b16 else c.hidden // 4 + 4
    ), c.hidden // 32 + 1
    ks = c.sh_ks if sh else 1
    nst = c.hidden // 128 // ks  # K steps a slice
    cta = c.route_mode == "cta" and not sh
    pick_lds, wt_lds = _pick_lds(k, c, s, cta)
    g, inter, live = (
        (c.sh_g, c.sh_inter, c.sh_live) if sh else (c.ug_g, c.inter, c.live)
    )
    pre = "sh.ug" if sh else "ug"
    gather = (c.x, c.hidden, 0, c.hidden, pitch, spitch) if ks == 1 else (c.x, c.hidden, 0, nst * 128, pitch, spitch,
                                                                           kh * nst * 128)  # fmt: skip
    stage_b = (_rows_sh(k, c, pick_lds, *gather, ch=ch, npick=npick) if sh else
               _rows_rt(k, c, npick, pick_lds, *gather) if cta else
               _rows_b(k, c, s, npick, pick_lds, True, *gather))  # fmt: skip
    res = {}
    wk = _gemv(k, c, e, g, c.hidden, 2 * inter, part, f"{c.name}.{pre}.w", f"{c.name}.{pre}.ws", pitch, spitch,
               stage_b, c.pre_b, 2 * live // 16 if live and live < inter else 0,
               _sh_wfmt(c) if sh else gemv.MXFP4, _sh_pre(c) if sh else None,
               **({} if ks == 1 else {"k0": kh * nst, "nsteps": nst}), mk=5, ring=None if sh else ring,
               gu_split=c.layout == "intra_gu" and not sh, cols=c.atom_cols(c.hidden) if c.atom_scales and not sh
               else 0, **res)  # fmt: skip
    # epilogue item = (pick n, pair p, f4): 32 features of a pick in 8 adjacent lanes (an MX group)
    pairs = g // 2
    items = TILE * pairs * 4
    for ps in range(-(-items // THREADS)):
        i = fx.min(k.tid + ps * THREADS, fx.Int32(items - 1))
        n, p, f4 = i // (pairs * 4), (i // 4) % pairs, i % 4
        ok = (k.tid + ps * THREADS < items) & (n < npick)
        gate = _wave_sum(k, g, 2 * p, n, f4, wk)
        up = _wave_sum(k, g, 2 * p + 1, n, f4, wk)
        if (
            ks > 1
        ):  # every slice publishes its sums; all of them sum the slices in order; slice 0 stores
            gate, up = _sh_slices(
                k,
                c,
                ok,
                kh,
                n if ch is None else n + ch * TILE,
                part * g + 2 * p,
                f4,
                gate,
                up,
            )
            ok = ok & (kh == 0)
        pick = fx.ptr_load(k.lds + (pick_lds + n))
        w = fx.ptr_load(k.lds + (wt_lds + n)).bitcast(fx.Float32)
        mid = []
        for q in range(4):
            gq, uq = bf16_round(gate[q]), bf16_round(up[q])
            if c.limit > 0:
                gq = fx.min(gq, fx.Float32(c.limit))
                uq = fx.max(fx.min(uq, fx.Float32(c.limit)), fx.Float32(-c.limit))
            mid.append(bf16_round(w * (gemv.silu_f32(gq) * uq)))
        amax = gemv.group_amax(mid, 8)
        code = gemv.activation_code(amax)
        word = gemv.quant_word(mid, code)
        r, kp = (
            pick // c.k,
            pick % c.k,
        )  # (after the quantization, as before A16W4: the op order is the ISA's)
        lead = (p % 2 == 0) & (f4 == 0)
        if sh:  # its own channel, a row of sh_inter
            _put_mid_ch(
                k,
                ok,
                c.sh,
                r * inter + (part * pairs + p) * 16 + f4 * 4,
                word,
                code,
                lead,
            )
        else:
            _put_mid_ch(k, ok, c.mid, r * (c.k * inter) + kp * inter + (part * pairs + p) * 16 + f4 * 4, word, code,
                        lead)  # fmt: skip
    gpu.barrier()


def _sh_slices(k, c: _Cfg, ok, kh, n, rg, f4, gate, up):
    """sh_ks > 1: this slice's gate / up sums (row n of the step, row groups rg / rg + 1, features f4 * 4 ..) into
    ``sh_kp``
    slot ``kh``; then every slice's, summed in slice order (fp32)."""
    a = k.mb[c.sh_kp]
    width = 2 * c.sh_inter

    def at(sl, n_, r):
        return (sl * c.rows + n_) * width + (rg + r) * 16 + f4 * 4

    _put_f32_if(k, ok, a, at(kh, n, 0), gate)
    _put_f32_if(k, ok, a, at(kh, n, 1), up)
    nc = fx.min(n, fx.Int32(c.rows - 1))
    specs = [
        (a, at(sl, nc, r) + h, 2)
        for sl in range(c.sh_ks)
        for r in range(2)
        for h in (0, 2)
    ]
    got = mb.poll(specs, k.tag)
    tot = [None, None]
    for sl in range(c.sh_ks):
        for r in range(2):
            v = [
                w.bitcast(fx.Float32)
                for w in got[(sl * 2 + r) * 2] + got[(sl * 2 + r) * 2 + 1]
            ]
            tot[r] = v if tot[r] is None else [tot[r][q] + v[q] for q in range(4)]
    return tot[0], tot[1]


def _put_mid_ch(k, ok, ch, e, word, code, lead):
    _put_mid_if(k, ok, k.mb[f"{ch}.q"], k.mb[f"{ch}.s"], e, word, code, lead)


@traced
def _put_mid_if(k, ok, aq, as_, e, word, code, lead):
    if ok:
        chan.put_mxfp8_word(aq, e // 4, word, k.tag)
        if lead:
            chan.put_mxfp8_scale(as_, e // 32, code, k.tag)


# ------------------------------------------------------------------ 3. down


def _down(k, t, c: _Cfg):
    s, part = t // c.down_parts, t % c.down_parts
    count, e, npick = _rt_slot(k, c, s) if c.route_mode == "cta" else _slot(k, c, s)
    _down_live(k, c, s, part, count, e, npick)


@traced
def _down_live(k, c: _Cfg, s, part, count, e, npick):
    if s < count:
        _down_body(k, c, s, part, e, npick)


def _down_q(k, t, c: _Cfg):
    """dq: a CTA takes down items from the queue until it is empty: items 0 .. count x down_parts - 1 the routed
    slots' (slot, part), then the folded shared expert's parts. The grab order hands out work as CTAs free up, so the
    stage's bytes spread over the CTAs whatever the routing."""
    mb.poll([(k.mb[c.dq_ready], 0, 1)], k.tag)
    routed = uniform(fx.ptr_load(k.lds + c.rt)) * c.down_parts
    _down_loop(
        k, c, routed, routed + (c.sh_down_items if c.sh_inter and not c.sh_own else 0)
    )


@traced
def _down_loop(k, c: _Cfg, routed, total):
    w = _dq_grab(k, c)
    while w < total:
        _down_item(k, c, w, routed)
        w = _dq_grab(k, c)


@traced
def _down_item(k, c: _Cfg, w, routed):
    if w < routed:
        s = w // c.down_parts
        _, e, npick = _rt_slot(k, c, s)
        _down_body(k, c, s, w % c.down_parts, e, npick)
    else:
        if const_expr(c.sh_inter > 0 and not c.sh_own and c.rc == 1):
            _down_body(k, c, 0, w - routed, fx.Int32(0), fx.Int32(c.rows), sh=True)
        if const_expr(c.sh_inter > 0 and not c.sh_own and c.rc > 1):
            ch, part, npick = _sh_chunk(c, w - routed, c.sh_down_parts)
            _down_body(k, c, 0, part, fx.Int32(0), npick, sh=True, ch=ch)


def _dq_grab(k, c: _Cfg):
    """The next queue item, the same for every thread of the CTA."""
    _dq_grab_t0(k, c)
    gpu.barrier()
    w = uniform(fx.ptr_load(k.lds + c.dq_word))
    gpu.barrier()
    return w


@traced
def _dq_grab_t0(k, c: _Cfg):
    if k.tid == 0:
        note(c.dq_ctr, Space.SCRATCH, "put")
        ptr = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
            k.mb[c.dq_ctr].value,
        )
        fx.ptr_store(
            fx.Int32(fx.atomic_add(ptr, fx.Int32(1), syncscope=rocdl.SyncScope.Agent)),
            k.lds + c.dq_word,
        )


def _sh_down(k, t, c: _Cfg):
    """The folded shared expert's down, part ``t``: every row is a pick; its y_r goes to segment K of y (over 16
    rows: part t of a 16-row chunk's)."""
    if const_expr(c.rc == 1):
        _down_body(k, c, 0, t, fx.Int32(0), fx.Int32(c.rows), sh=True)


def _down_body(k, c: _Cfg, s, part, e, npick, sh=False, ch=None):
    g, inter = (c.sh_dg, c.sh_inter) if sh else (c.down_g, c.inter)
    # down_ring: the task's first units a wave copied into its LDS ring at its start, so they stream while it polls mid
    ring = (
        SimpleNamespace(n=c.down_ring, own=k.tid >= 0)
        if c.down_ring and not sh
        else None
    )
    pitch, spitch = (inter // 2 + 4 if c.b16 else inter // 4 + 4), inter // 32 + 1
    cta = c.route_mode == "cta" and not sh
    pick_lds, _wt_lds = _pick_lds(k, c, s, cta)
    pre = "sh.down" if sh else "down"
    stage_b = (_rows_sh(k, c, pick_lds, c.sh, inter, 0, inter, pitch, spitch, ch=ch, npick=npick) if sh else
               _rows_rt(k, c, npick, pick_lds, c.mid, c.k * inter, inter, inter, pitch, spitch) if cta else
               _rows_b(k, c, s, npick, pick_lds, c.wdown, c.mid, c.k * inter, inter, inter, pitch, spitch))  # fmt: skip
    items = TILE * g * 4
    live = c.sh_live if sh else c.live  # intra / TP: the padded K tail is not loaded
    wk = _gemv(k, c, e, g, inter, c.hidden, part, f"{c.name}.{pre}.w", f"{c.name}.{pre}.ws", pitch, spitch,
               stage_b, c.down_pre_b, 0, _sh_wfmt(c) if sh else gemv.MXFP4, _sh_pre(c) if sh else None,
               live if live and live < inter else 0, ring=ring,
               cols=c.atom_cols(inter) if c.atom_scales and not sh else 0)  # fmt: skip
    for ps in range(-(-items // THREADS)):
        i = fx.min(k.tid + ps * THREADS, fx.Int32(items - 1))
        n, gl, f4 = i // (g * 4), (i // 4) % g, i % 4
        ok = (k.tid + ps * THREADS < items) & (n < npick)
        y = _wave_sum(k, g, gl, n, f4, wk)
        pick = fx.ptr_load(k.lds + (pick_lds + n))
        r, kp = pick // c.k, (fx.Int32(c.k) if sh else pick % c.k)
        col = kp * c.hidden + (part * g + gl) * 16 + f4 * 4
        _put_y(k, c, ok, r * (c.y_k * c.hidden) + col, y)
    gpu.barrier()


def _put_y(k, c: _Cfg, ok, e, vals):
    _put_bf16_if(k, ok, k.mb[c.y], e, [bf16_round(v) for v in vals])


@traced
def _put_bf16_if(k, ok, a, e, vals):
    if ok:
        chan.put_bf16(a, e, vals, k.tag)


# ------------------------------------------------------------------ 4. combine


COMBINE_COLS = 4 * THREADS  # features a combine task


def _combine(k, t, c: _Cfg, push=False):
    """``push`` (TP, ``reduce="push"``): task t sums chunk t % (rows x per_row) and pushes it to rank t // that only
    (every destination its own task), for ``_dsum``; no local partial, no ``PeerAllReduce``, no ``finish``.
    """
    per_row = -(-c.hidden // COMBINE_COLS)
    dst = None
    if push and not c.rs:
        dst, t = t // (c.rows * per_row), t % (c.rows * per_row)
    r, part = t // per_row, t % per_row
    col = part * COMBINE_COLS + k.tid * 4
    ok = col < c.hidden
    colc = fx.min(col, fx.Int32(c.hidden - 4))
    specs = [
        (k.mb[c.y], (r * (c.y_k * c.hidden) + kp * c.hidden + colc) // 2, 2)
        for kp in range(c.y_k)
    ]
    got = mb.poll(specs, k.tag)
    acc = [fx.Float32(0.0)] * 4
    for i, g in enumerate(got):
        vals = list(bf2_f32(g[0])) + list(bf2_f32(g[1]))
        acc = [acc[q] + vals[q] for q in range(4)]
    if (
        push and c.rs
    ):  # a wave's 256 columns lie in one owner's quarter (hidden / tp a multiple of 256)
        _push_rs_if(k, c, ok, r * c.hidden + colc, acc)
    elif push:
        _push_one_if(k, c, ok, r * c.hidden + colc, acc, dst)


@traced
def _push_rs_if(k, c: _Cfg, ok, e, vals):
    if (
        ok
    ):  # a wave's 256 columns: one owner, or (hidden / tp not a multiple of 256, e.g. 8 ranks) at most two
        push_rs(
            k,
            c.part,
            e,
            vals,
            c.hidden,
            c.rs_fmt,
            two_owners=(c.hidden // c.tp) % (4 * 64) != 0,
        )


@traced
def _push_one_if(k, c: _Cfg, ok, e, vals, dst):
    if ok:
        push_partial(k, c.part, c.part_fmt, e, vals, [dst])


@traced
def _put_f32_if(k, ok, a, e, vals):
    if ok:
        mb.put2(a, e, vals[0], vals[1], k.tag)
        mb.put2(a, e + 2, vals[2], vals[3], k.tag)


def _finish(k, t, c: _Cfg):
    """TP: out = bf16(the all-reduced partial + shared), 4 features a thread."""
    per_row = -(-c.hidden // COMBINE_COLS)
    r, part = t // per_row, t % per_row
    col = part * COMBINE_COLS + k.tid * 4
    ok = col < c.hidden
    colc = fx.min(col, fx.Int32(c.hidden - 4))
    specs = [
        (k.mb[c.red], r * c.hidden + colc, 2),
        (k.mb[c.red], r * c.hidden + colc + 2, 2),
    ]
    if c.shared:
        specs.append((k.mb[c.shared], (r * c.hidden + colc) // 2, 2))
    got = mb.poll(specs, k.tag)
    acc = [w.bitcast(fx.Float32) for w in got[0] + got[1]]
    if c.shared:
        sh = list(bf2_f32(got[2][0])) + list(bf2_f32(got[2][1]))
        acc = [acc[q] + sh[q] for q in range(4)]
    _put_out(k, c, ok, r * c.hidden + colc, acc)


def _put_out(k, c: _Cfg, ok, e, vals):
    _put_bf16_if(k, ok, k.mb[c.out], e, [bf16_round(v) for v in vals])


# ------------------------------------------------------------------ 3'. down owning columns


def _sh_wfmt(c: _Cfg):
    """The folded shared expert's weight format (``sh_fmt``)."""
    return gemv.MXFP4 if c.sh_fmt == "mxfp4" else gemv.MXFP8


def _sh_pre(c: _Cfg):
    """Its units in flight a wave: ``sh_prefetch`` (MXFP8 units), the routed experts' for MXFP4 (None)."""
    return None if c.sh_fmt == "mxfp4" else c.sh_prefetch


def _dsum(k, t, c: _Cfg):
    """TP: 4 elements a thread, the ranks' partials summed in rank order -> bf16 ``out``."""
    w = t * THREADS + k.tid
    n = c.rows * c.hidden
    ok = w * 4 < n
    e = fx.min(w, fx.Int32(n // 4 - 1)) * 4
    if c.rs:  # the owners' sums (``_own``)
        acc = read_ag(k, f"{c.part}.dag", e, c.hidden, "f32" if c.shared else "bf16")
    _put_out(k, c, ok, e, acc)


@traced
def _own(k, t, c: _Cfg):
    """reduce "2shot", the owner's half (region ``<part>.dag``: ``.ag`` is the next launch's seam's): elements
    ``4 i .. + 3`` of this rank's column quarter, the ranks' fp32
    partials summed in rank order (the one-shot's bits) and put to every rank (bf16: dsum rounds to bf16 anyway;
    f32 when a shared channel is added after)."""
    n = c.rows * c.hidden // c.tp
    chunks = -(-n // 4 // c.own_quads)
    dst, i = (
        t // chunks,
        (t % chunks) * c.own_quads + k.tid,
    )  # a destination a task: the remote puts spread
    ok = i * 4 < n
    if const_expr(c.own_quads < THREADS):
        ok = ok & (k.tid < c.own_quads)
    if ok:
        s4 = sum_partials(k, c.part, c.rs_fmt, i * 4, c.tp)
        push_partial(
            k, f"{c.part}.dag", "f32" if c.shared else "bf16", i * 4, s4, [dst]
        )


# ------------------------------------------------------------------ the block


class MoEExperts(Block):
    """Routed experts + combine (module docstring). ``x``: MXFP8 channel (width ``hidden``); ``idx`` / ``weights``:
    the router's F32 channels (width ``k``); ``shared``: a BF16 channel (width ``hidden``) or None (beside a folded
    shared expert, ``shared_inter``, TP intra: a channel added last, after the all-reduce: e.g. the residual);
    ``out``: BF16 channel (width ``hidden``). ``experts``: expert count (the weights' leading dimension); ``inter``:
    expert width; ``limit``: the SwiGLU clamp (V4.1: 10). Knobs: ``ug_g`` / ``down_g`` row groups a task (None:
    ``groups``; a dict {rows: groups}: per scenario rows, rows not listed take ``groups``' rule); ``prefetch`` weight
    units a wave keeps in flight, ``pre_b`` of them issued before a task's picks and rows are polled.
    """

    def __init__(self, name: str, x: Channel, idx: Channel, weights: Channel, out: Channel, experts: int, inter: int,
                 shared: Channel | None = None, limit: float = 0.0, ug_g: int | None = None,
                 down_g: int | None = None, prefetch: int = 8, pre_b: int = 4, down_pre_b: int = 4,
                 layout: str = "expert", reduce: str = "push", shared_inter: int = 0, down: str = "slots",
                 shared_order: str = "fold", dsum: bool = True, route: str = "cta", sh_g: int = SH_G,
                 sh_dg: int | None = None, sh_prefetch: int = SH_PREFETCH, sh_ks: int | None = None,
                 sh_fmt: str = "mxfp8",
                 down_queue: bool | None = None, sh_own: bool | None = None,
                 weight: str = "ffn.experts", rt_poll: str = "all", chain_own: bool = False,
                 own_quads: int = THREADS, rt_lean: bool = False, out_arg: str = "",
                 route_from=None, ug_ring: int = 0, rs_fmt: str = "f32", down_ring: int = 0,
                 part_fmt: str = "f32", acts: str = "mxfp8", x_lds: bool = False,
                 atom_scales: bool = False):  # fmt: skip
        if acts not in ("mxfp8", "bf16"):
            raise ValueError(f"MoEExperts: acts 'mxfp8' or 'bf16', got {acts!r}")
        xfmt = Fmt.BF16 if acts == "bf16" else Fmt.MXFP8
        if (
            x.fmt is not xfmt
            or x.flagged
            or out.fmt is not Fmt.BF16
            or out.width != x.width
        ):
            raise ValueError(f"MoEExperts: x a tagged {xfmt.name} channel (acts {acts!r}), out a BF16 channel of the "
                             "same width")  # fmt: skip
        for ch in (idx, weights):
            if ch.fmt is not Fmt.F32 or ch.flagged or ch.width != idx.width:
                raise ValueError(
                    "MoEExperts: idx / weights tagged F32 channels of width k"
                )
        if shared is not None and (
            shared.fmt is not Fmt.BF16 or shared.width != x.width
        ):
            raise ValueError("MoEExperts: shared is a BF16 channel of the model width")
        ugs = ug_g.values() if isinstance(ug_g, dict) else [ug_g]
        dgs = down_g.values() if isinstance(down_g, dict) else [down_g]
        # ug_g: whole MX groups of mid (4 row groups: 32 features), A16W4 whole gate / up pairs (2)
        ug_mult = 2 if acts == "bf16" else 4
        bad_g = any((g or 4) % ug_mult for g in ugs) or any(
            (x.width // 16) % (g or 1) for g in dgs
        )
        # sh_g likewise: the shared gate / up epilogue writes whole groups of mid from (gate, up) row-group pairs
        # (``_ug_body``): sh_g 1 wrote nothing (its down polled forever), an odd sh_g dropped its last group
        bad_g = bad_g or sh_g < ug_mult or sh_g % ug_mult
        if bad_g or inter % 128 or x.width % 128:
            raise ValueError(f"MoEExperts: ug_g / sh_g multiples of {ug_mult}; down_g dividing N / 16; inter, N "
                             f"multiples of 128 (inter {inter}, N {x.width}; pad other widths with zero tails: "
                             "NormSpec.pitch, "
                             "BLOCK_CATALOG 'K')")  # fmt: skip
        if experts > 2 * THREADS or experts > THREADS and route != "cta":
            raise ValueError(f"MoEExperts: at most {THREADS} experts (route 'stage': a thread per expert), "
                             f"{2 * THREADS} (route 'cta': two a thread)")  # fmt: skip
        if shared_inter and (
            shared_inter % 128
            or (shared is not None and not layout.startswith("intra"))
        ):
            # with a folded shared expert a ``shared`` channel is the last add (e.g. the residual), TP intra
            raise ValueError("MoEExperts: a folded shared expert (shared_inter): width % 128; a shared channel beside "
                             "it with layout 'intra' only")  # fmt: skip
        if layout not in ("expert", "intra", "intra_gu"):
            raise ValueError(f"MoEExperts: layout 'expert' or 'intra', got {layout!r}")
        self.name, self.x, self.idx, self.weights, self.out, self.shared = (
            name,
            x,
            idx,
            weights,
            out,
            shared,
        )
        self.experts, self.inter, self.limit, self.ug_g, self.down_g, self.weight = (
            experts,
            inter,
            limit,
            ug_g,
            down_g,
            weight,
        )
        if not (0 <= pre_b <= prefetch and 0 <= down_pre_b <= prefetch):
            raise ValueError("MoEExperts: 0 <= pre_b, down_pre_b <= prefetch")
        self.prefetch, self.pre_b, self.down_pre_b, self.layout = (
            prefetch,
            pre_b,
            down_pre_b,
            layout,
        )
        # TP: "push" (combine pushes the partial, dsum sums it), "2shot" (the combine pushes each column quarter to its
        # owner only, the owner sums and gathers; the folded "fold" order) or a
        # PeerAllReduce proto
        if reduce == "2shot" and shared_order != "fold":
            raise ValueError("MoEExperts: reduce '2shot' needs shared_order 'fold'")
        if rs_fmt not in ("f32", "f32q", "f32p") or (
            rs_fmt != "f32" and reduce != "2shot"
        ):
            raise ValueError(
                f"MoEExperts: rs_fmt 'f32' or (reduce '2shot') 'f32q' / 'f32p', got {rs_fmt!r}"
            )
        self.rs_fmt = rs_fmt
        self.reduce = reduce
        if rt_poll not in ("all", "picks"):
            raise ValueError(f"MoEExperts: rt_poll 'all' or 'picks', got {rt_poll!r}")
        # route "cta": the rt stage's routing poll by every thread of every CTA ("all": the threads past the picks poll
        # a clamped duplicate) or by the threads with a pick ("picks": through LDS to the rest; the same table)
        self.rt_poll = rt_poll
        self.rt_lean = rt_lean  # rt_poll "picks": the rt table on fewer barriers (``_Cfg.rt_lean``; same table)
        self.shared_inter = shared_inter  # the folded shared expert's width (0: none)
        if down not in ("cols", "slots"):
            raise ValueError(f"MoEExperts: down 'cols' or 'slots', got {down!r}")
        self.down = down
        if shared_order not in ("ref", "fold"):
            raise ValueError(
                f"MoEExperts: shared_order 'ref' or 'fold', got {shared_order!r}"
            )
        # TP, folded shared expert: "fold" (the speed default) adds its partial into the routed one (one field, D10);
        # "ref" (validation) all-reduces it apart (packed with the routed partial, 12 B an element), rounds it to bf16
        # and adds it after the routed sum: the reference's arithmetic, +0.6-0.9 / 1.6-2.5 / 4.5-5.9 us at 1 / 6 / 16
        # rows
        self.shared_order = shared_order
        # TP push: False = no dsum stage; the partial region (``<name>.part``, pushed by the combine, now the program's
        # last stage) is summed by its consumer, e.g. the next layer's seam (``HCSeam(pend=PeerPartial(...,
        # prev_launch=True))``, ``Stage.prev_reads``); ``out`` is not written. Either order: "fold" one f32 field
        # (``sum_partials``), "ref" two (``sum_partials3``: the consumer adds bf16(shared) after the routed sum, as
        # dsum); no shared channel
        if not dsum and (reduce not in ("push", "2shot") or shared is not None):
            raise ValueError(
                "MoEExperts: dsum=False needs reduce='push' and no shared channel"
            )
        self.dsum = dsum
        # reduce "2shot" without dsum: the owner stage runs in this launch (``<name>.own``, the program's last stage:
        # the ranks' fp32 column-quarter partials summed in rank order, bf16 to every rank's ``<part>.dag``), so the
        # next launch's seam reads the owners' sums (``HCSeam(pend=PeerPartial(<part>.dag, "ag", prev_launch=True))``)
        # instead of running the owner hop after its own start (False: the consumer runs it, ``PeerPartial("2shot")``)
        if chain_own and (reduce != "2shot" or dsum or shared is not None):
            raise ValueError(
                "MoEExperts: chain_own needs reduce='2shot', dsum=False and no shared channel"
            )
        self.chain_own = chain_own
        # 2shot: the owner stage's element quads a task (a thread each; THREADS: today's; fewer: more tasks, each
        # pulling less)
        if not 0 < own_quads <= THREADS or own_quads % 64:
            raise ValueError(f"MoEExperts: own_quads a multiple of 64 up to {THREADS}")
        self.own_quads = own_quads
        # "stage": one route task dedupes the picks into slots and publishes the table (every ug / down task polls its
        # slot); "cta": every CTA builds the table in LDS from the router's channels on its first
        # ug / down task (no route stage, no per-task route polls; down="slots" only)
        if route not in ("stage", "cta") or (route == "cta" and down != "slots"):
            raise ValueError(
                f"MoEExperts: route 'stage' or 'cta' (down='slots'), got {route!r}"
            )
        self.route = route
        self.sh_g, self.sh_dg = (
            sh_g,
            sh_dg,
        )  # the folded shared expert's row groups a gate / up and a down task
        self.sh_prefetch = sh_prefetch  # ... and its MXFP8 units in flight a wave
        if sh_fmt not in ("mxfp8", "mxfp4"):
            raise ValueError(f"MoEExperts: sh_fmt 'mxfp8' or 'mxfp4', got {sh_fmt!r}")
        self.sh_fmt = (
            sh_fmt  # the folded shared expert's weights (``pack_shared(fmt=)``)
        )
        # ... and its gate / up's K slices (one task each, summed in slice order: D10 against the unsliced sum, so
        # 1 for the validation order; sh_ks must divide hidden / 128). None: ``sh_slices``
        if sh_ks is not None and (sh_ks < 1 or (x.width // 128) % sh_ks):
            raise ValueError(
                f"MoEExperts: sh_ks {sh_ks} must divide {x.width // 128} K steps"
            )
        self.sh_ks = sh_ks
        # down as a work queue (route "cta", down "slots"): CTAs take routed / shared down items as they free up.
        # None: ``down_queued``
        if down_queue and (route != "cta" or down != "slots"):
            raise ValueError(
                "MoEExperts: down_queue needs route='cta' and down='slots'"
            )
        self.down_queue = down_queue
        # the folded shared expert on CTAs of its own (route "cta", down "slots"; None: off, a knob)
        if sh_own and (route != "cta" or down != "slots" or not shared_inter):
            raise ValueError(
                "MoEExperts: sh_own needs a folded shared expert, route='cta' and down='slots'"
            )
        self.sh_own = sh_own
        self.out_arg = (
            out_arg  # the output also stored to a plain arg (TP: by dsum, push reduce)
        )
        # a MoERouter whose route the rt stage computes (route "cta"): a softmax_topk / sigmoid router with the dot's
        # slice partials and wave-0 scores, or a ``consumer`` sigmoid / sqrtsoftplus router (any TP: every rank's dot
        # all n experts; a wave a row)
        if route_from is not None and (
            route != "cta" or rt_lean or not route_from.dot or not route_from.dot.part
            or not (route_from.consumer or (route_from.score_func in ("softmax_topk", "sigmoid")
                                            and route_from.scores_by == "wave0"))
        ):  # fmt: skip
            raise ValueError("MoEExperts: route_from needs route 'cta', no rt_lean and a softmax_topk / sigmoid "
                             "(wave0 scores) router with dot partials, or a consumer router")  # fmt: skip
        self.route_from = route_from
        # ug: the first weight units a wave of every task from an LDS ring the CTA's previous ug task fills after its
        # epilogue (``_gemv``; MXFP4 routed experts, route "cta", no sh_own); 0: none (build unchanged)
        if ug_ring and (route != "cta" or ug_ring < 0):
            raise ValueError("MoEExperts: ug_ring needs route='cta'")
        self.ug_ring = ug_ring
        # down: every routed down task's first weight units a wave copied into an LDS ring at its start (``_gemv``), so
        # they stream while it polls its mid rows; 0: none (build unchanged)
        if down_ring < 0:
            raise ValueError("MoEExperts: down_ring >= 0")
        self.down_ring = down_ring
        # TP push: the partial each rank pushes, "f32" or "bf16" (half the exchanged
        # bytes; each rank's partial rounded to bf16 before the rank-order fp32 sum: D10). One field (no "ref" order),
        # one shot
        if part_fmt not in ("f32", "bf16") or (
            part_fmt == "bf16" and (reduce != "push" or shared_order != "fold")
        ):
            raise ValueError(
                "MoEExperts: part_fmt 'f32' or 'bf16' (reduce 'push', shared_order 'fold')"
            )
        self.part_fmt = part_fmt
        # A16W4 (``acts="bf16"``, ATOM's MoE up to 4 tokens): x a BF16 channel and mid in bf16, every MXFP4 unit decoded
        # to exact bf16 and multiplied on the bf16 MFMA (``gemv16``): bf16 x bf16 products, fp32 sums. The step's rows
        # sit in LDS whole (at most A16_ROWS), so route "cta", down "slots", no queue / own shared CTAs / ug ring,
        # MXFP4 shared weights, no biases (the ug ring's units are raw MXFP4: any B format)
        if acts == "bf16" and (route != "cta" or down != "slots" or down_queue or sh_own
                               or (shared_inter and sh_fmt != "mxfp4")):  # fmt: skip
            raise ValueError("MoEExperts: acts 'bf16' needs route 'cta', down 'slots', no down queue / sh_own, "
                             "MXFP4 shared weights")  # fmt: skip
        self.acts = acts
        # x_lds: the rt stage (route "cta") stages every row of x in its CTA's LDS once; the routed ug tasks read
        # their picks' rows there instead of gathering them each (a task's B gather is a poll round trip of up to 12
        # KB a row); up to A16_ROWS rows, no sh_own (its CTAs skip rt)
        if x_lds and (route != "cta" or sh_own or down_queue or ug_ring):
            raise ValueError(
                "MoEExperts: x_lds needs route 'cta', no sh_own / down queue / ug_ring"
            )
        self.x_lds = x_lds
        self.atom_scales = atom_scales  # the routed experts' scale codes in ATOM's order, in place (``_Cfg``)

    def sh_on(self, sc: Scenario) -> tuple[int, int]:
        """(width, real width) of the folded shared expert on a rank: TP slices it (gate / up rows, down columns)
        whatever the layout, padded to a whole 128-wide K step (``pack_shared``)."""
        live = self.shared_inter // sc.tp
        return -(-live // 128) * 128, live

    def inter_on(self, sc: Scenario) -> int:
        """The expert width a rank's kernel runs: ``intra`` TP splits it (gate / up along N, down along K) into
        ``inter / tp`` slices, padded to a whole 128-wide K step with zero weights (V4.1 TP4: 576 -> 640).
        """
        if self.inter % (32 * sc.tp):
            raise ValueError(
                f"MoEExperts {self.name}: inter {self.inter} not a multiple of 32 x tp"
            )
        return -(-self.inter // sc.tp // 128) * 128

    def groups(self, sc: Scenario) -> tuple[int, int]:
        """(ug_g, down_g): given, else
        - intra TP: the candidate whose busiest CTA streams the least: ``ceil(slots x parts /
          CTAs)`` rounds x (a part's weight bytes + a task's fixed cost in bytes, ``TASK_BYTES``), with the most
          slots a step can have (rows x K); ties go to fewer, larger tasks. A partial last round costs a whole part:
          at 6 rows TP4 ~31-35 distinct experts x 10 ug parts of 8 groups put two 330 KB parts on ~100 CTAs, 4
          groups at most 3 x 165 KB. With the down queue (16 rows) down keeps ``INTRA_DOWN_G``.
        - else (expert layout, TP1: not measured with the rule): the largest of (4, 8, 12) / (4, 8, 16) row groups a
          task that keeps the expected live slots x parts >= 192 tasks."""
        n16 = self.x.width // 16
        ug_given, down_given = (
            v.get(sc.rows) if isinstance(v, dict) else v
            for v in (self.ug_g, self.down_g)
        )
        if self.layout.startswith("intra") and sc.tp > 1:
            inter, live = self.inter_on(sc), self.inter // sc.tp
            slots = min(sc.rows * self.idx.width, self.experts)

            def busiest(g, groups, gbytes, fixed):
                return -(-int(slots * -(-groups // g)) // MI355X.cus) * (
                    g * gbytes + fixed
                )

            def best(given, cands, groups, gbytes, fixed, steps):
                if given:
                    return given
                ok = [
                    g
                    for g in cands
                    if groups % g == 0 and g % (WAVES // _wk(steps, g)) == 0
                ]
                return min(ok, key=lambda g: (busiest(g, groups, gbytes, fixed), -g))

            ug = best(ug_given, UG_CANDS, 2 * inter // 16, 16 * self.x.width * 17 // 32, TASK_BYTES[0],
                      self.x.width // 128)  # fmt: skip
            if down_given or n16 % INTRA_DOWN_G or self.down_queued(sc, INTRA_DOWN_G):
                return (
                    ug,
                    down_given or INTRA_DOWN_G,
                )  # the queue balances its items itself
            return ug, best(
                None, DOWN_CANDS, n16, 16 * live * 17 // 32, TASK_BYTES[1], inter // 128
            )
        slots = sc.rows * self.idx.width / (sc.tp if self.layout == "expert" else 1)

        def pick(given, cands, groups):
            if given:
                return given
            fit = [g for g in cands if groups % g == 0 and slots * (groups // g) >= 192]
            return max(fit) if fit else min(g for g in cands if groups % g == 0)

        return pick(ug_given, (4, 8, 12), 2 * self.inter_on(sc) // 16), pick(
            down_given, (4, 8, 16), n16
        )

    def sh_slices(self, sc: Scenario, ug_g: int, sh_inter: int) -> int:
        """The shared gate / up's K slices: given, else the most of (4, 2) whose tasks still fit beside the expected
        routed ug tasks in 3 / 4 of the CTAs (its 327 KB parts on 20 CTAs are the 1-row long pole; at 6 rows the
        routed tasks fill the CTAs and slices would double up); 1 at TP1 / for shared_order
        "ref" (the reference's order)."""
        slots = min(sc.rows * self.idx.width, self.experts) / (
            sc.tp if self.layout == "expert" else 1
        )
        routed = slots * (2 * self.inter_on(sc) // 16 // ug_g)
        parts = 2 * sh_inter // 16 // self.sh_g
        fill = (
            MI355X.cus * 3 // 4
        )  # headroom: the routed count is an expectation (V4.1 expert TP4 6 rows: 216 + 40)
        return next(
            (
                ks
                for ks in (4, 2)
                if (self.x.width // 128) % ks == 0 and routed + parts * ks <= fill
            ),
            1,
        )

    def down_queued(self, sc: Scenario, down_g: int) -> bool:
        """Down as a work queue: given, else when the expected routed down items pass two rounds of CTAs (TP4 16 rows:
        −2.5 to −3.9 µs; at 1 and 6 rows the grab's atomic round trip before each item's loads costs 0.3-1 µs).
        """
        slots = sc.rows * self.idx.width / (sc.tp if self.layout == "expert" else 1)
        return slots * (self.x.width // 16 // down_g) > 2 * MI355X.cus

    def sh_owned(self, sc: Scenario) -> bool:
        """The folded shared expert's gate / up and down on the top CTAs, which skip the route table and every routed
        task. Off unless given: the TP4 6-row block gains 0.7 us, the V4.1 layer nothing (its shared expert runs
        ahead of the routed experts anyway), and at 16 rows its CTAs leave the down queue (+2.5 us).
        """
        return bool(self.sh_own) and bool(self.shared_inter) and sc.tp > 1

    def cfg(self, sc: Scenario) -> _Cfg:
        if (
            self.reduce == "2shot"
            and sc.tp > 1
            and ((self.x.width // sc.tp) % 4 or self.x.width // sc.tp < 4 * 64)
        ):
            raise ValueError(
                f"MoEExperts {self.name}: reduce '2shot' needs hidden / tp >= 256 (a wave's), quads"
            )
        if self.rs_fmt == "f32q" and sc.tp > 1 and (self.x.width // sc.tp) % (4 * 64):
            raise ValueError(f"MoEExperts {self.name}: rs_fmt 'f32q' takes one owner a wave (hidden / tp a multiple of "
                             "256); 'f32p' or pairs cross an owner boundary")  # fmt: skip
        if sc.rows > TILE and (
            self.route != "cta" or self.down != "slots" or sc.rows > RT_ROWS
        ):
            raise ValueError(f"MoEExperts {self.name}: over {TILE} rows (up to {RT_ROWS}) only route 'cta', down "
                             "'slots'")  # fmt: skip
        picks = sc.rows * self.idx.width  # (the step's picks: rows x K)
        if (
            self.route != "cta" and picks > THREADS
        ):  # ``_route``: thread p < rows x K holds pick p
            raise ValueError(f"MoEExperts {self.name}: route 'stage' holds a pick a thread: rows x K <= {THREADS}, got "
                             f"{picks}")  # fmt: skip
        if (
            sc.tp > 1 and self.layout == "expert" and picks > THREADS
        ):  # the B operand: thread p < slots, pick p
            raise ValueError(f"MoEExperts {self.name}: layout 'expert' at TP checks a pick a thread: rows x K <= "
                             f"{THREADS}, got {picks}")  # fmt: skip
        ug_g, down_g = self.groups(sc)
        inter = self.inter_on(sc)
        if (2 * inter // 16) % ug_g:
            raise ValueError(
                f"MoEExperts {self.name}: ug_g {ug_g} does not divide the {2 * inter // 16} gate / up groups"
            )
        sh_inter, sh_live = self.sh_on(sc)
        sh_ks = self.sh_slices(sc, ug_g, sh_inter)
        own = self.sh_owned(sc)
        dq = (
            self.down_queued(sc, down_g) and self.acts != "bf16" and not self.x_lds
        )  # (their LDS words: the dq's)
        if self.x_lds and sc.rows > A16_ROWS:
            raise ValueError(
                f"MoEExperts {self.name}: x_lds up to {A16_ROWS} rows, got {sc.rows}"
            )
        if self.acts == "bf16" and sc.rows > A16_ROWS:
            raise ValueError(f"MoEExperts {self.name}: acts 'bf16' up to {A16_ROWS} rows (the rows sit in LDS), got "
                             f"{sc.rows}")  # fmt: skip
        sh_dg = self.sh_dg or SH_DG
        if (
            self.sh_dg is None
            and sc.tp > 1
            and sh_inter
            and not dq
            and (self.x.width // 16) % SH_DG_SOLO == 0
        ):
            # half the parts, on the top CTAs clear of the routed down tasks: with 8 the 40 parts sat on CTAs 216-255
            # and CTAs 216-229 ran a routed ug item, a shared down part and a routed down item (the layer's down
            # tail, 2.7-4.6 us median -> last at 6 rows; 16: 1.4-1.7). With the down queue the parts are queue
            # items and smaller ones balance better (16 rows: +0.7-1.8 us with 16)
            sh_dg = SH_DG_SOLO
        return _Cfg(
            self.name, sc.rows, self.idx.width, self.experts, self.x.width, inter, self.limit, self.x.name,
            self.idx.name, self.weights.name, self.shared.name if self.shared else "", self.out.name, ug_g,
            down_g, self.prefetch, self.pre_b, self.down_pre_b, sc.tp, self.layout,
            self.inter // sc.tp if self.layout.startswith("intra") and sc.tp > 1 else 0,
            sh_inter, sh_live if sh_live < sh_inter else 0, self.sh_g, sh_dg, self.down,
            bool(sh_inter) and sc.tp > 1 and self.reduce == "push" and self.shared_order == "ref",
            rs=sc.tp > 1 and self.reduce == "2shot",
            sh_prefetch=self.sh_prefetch, sh_ks=sh_ks, dq=dq, sh_fmt=self.sh_fmt,
            route_mode=self.route, sh_own=own, rt_picks=self.rt_poll == "picks",
            rt_lean=self.rt_lean and self.rt_poll == "picks",
            own_quads=self.own_quads,
            out_arg=self.out_arg,
            rplan=(self.route_from.route_plan(sc) if self.route_from is not None
                   and (sc.tp == 1 or self.route_from.consumer) else None),
            ug_ring=self.ug_ring if not own else 0,
            down_ring=self.down_ring if not own else 0,
            rs_fmt=self.rs_fmt if sc.tp > 1 and self.reduce == "2shot" else "f32",
            part_fmt=self.part_fmt if sc.tp > 1 else "f32", acts=self.acts, x_lds=self.x_lds,
            atom_scales=self.atom_scales,
        )  # fmt: skip

    def stages(self, sc: Scenario) -> list[Stage]:
        """``_stages``, with route "cta": the rt stage's slot table (each CTA's own, in LDS for its ug / down tasks)
        declared ``LdsKeep``, so a stage of another block placed in between that would overwrite it is refused at
        plan time (a placement or insertion that once deadlocked: the readers' polls wait on a garbage table). In a
        ``Program.dyn_rows`` build their dead rows' work is skipped inside (the route counts the live picks only), so
        they are ``Stage.partial_rows``: their readers clamp dead rows."""
        stages = self._stages(sc)
        for st in stages:
            st.partial_rows = True
        c = self.cfg(sc)
        if c.route_mode == "cta":
            names = tuple(st.name for st in stages)
            readers = tuple(n for n in (f"{c.name}.ug", f"{c.name}.down") if n in names)
            for st in stages:
                if (
                    st.name == f"{c.name}.rt"
                ):  # (x_lds: the staged x rows past the table, read by ug, kept too)
                    hi = c.rt + c.rt_words + c.xl_words
                    st.lds_keep = (
                        LdsKeep(f"{c.name}.rt_table", c.rt, hi, readers, aware=names),
                    )
        return stages

    def _stages(self, sc: Scenario) -> list[Stage]:
        c = self.cfg(sc)
        scratch = (
            3 * c.e_route
            + 2 * WAVES
            + (2 * c.slots if c.rt_picks and c.rplan is None else 0)
        )
        if c.route_mode == "cta" and (c.e_pairs or c.rt_pp > 1) and scratch > c.rt:
            raise ValueError(f"MoEExperts {c.name}: {c.e_route} experts, {c.slots} picks need {scratch} LDS words "
                             "below the slot table")  # fmt: skip
        afmt = Fmt.BF16 if c.b16 else Fmt.MXFP8  # mid / the folded shared expert's mid
        mid = Channel(c.mid, c.k * c.inter, afmt)
        y = Channel(c.y, c.y_k * c.hidden, Fmt.BF16)
        lds = c.stage_lds + c.rt_words + c.xl_words
        down, sh_down = f"{c.name}.down", f"{c.name}.sh_down"
        y_writers = (down, sh_down) if c.sh_inter else (down,)
        stages = []
        if (
            c.sh_inter
        ):  # the shared expert's gate / up first, on the last CTAs: it needs no route (ATOM shared0)
            sh_w = Channel(c.sh, c.sh_inter, afmt).regions(
                sc.rows, (f"{c.name}.sh_ug",)
            )
            if c.sh_ks > 1:
                sh_w = (
                    *sh_w,
                    Region(
                        c.sh_kp,
                        c.sh_ks * sc.rows * 2 * c.sh_inter,
                        (f"{c.name}.sh_ug",),
                        exchange=True,
                    ),
                )
            stages.append(Stage(f"{c.name}.sh_ug", c.sh_tasks, partial(_sh_ug, c=c), lds_words=lds, writes=sh_w,
                                args=(f"{c.name}.sh.ug.w", f"{c.name}.sh.ug.ws"), base=MI355X.cus - c.sh_tasks,
                                key=("moe_sh_ug", c)))  # fmt: skip
        rt_w = ()
        if c.dq:
            rt_w = (
                Region(c.dq_ctr, 1, (f"{c.name}.rt", down)),
                Region(c.dq_ready, 1, (f"{c.name}.rt",)),
            )
        rt_args = (c.rplan.bias,) if c.rplan is not None else ()
        stages.append(Stage(f"{c.name}.rt", c.n_r, partial(_cta_route, c=c), lds_words=lds, base=0,
                            writes=rt_w, args=rt_args, key=("moe_rt", c)))  # fmt: skip
        ug_tasks, ug_emit = (
            (c.n_r, _ug_items) if c.sh_own else (c.slots * c.ug_parts, _ug)
        )
        ug_lds = lds + (
            1 + c.ring_words if c.ug_ring else 0
        )  # ug_ring: the dq word, then the ring
        stages += [
            Stage(f"{c.name}.ug", ug_tasks, partial(ug_emit, c=c), lds_words=ug_lds,
                  writes=mid.regions(sc.rows, (f"{c.name}.ug",)), args=(f"{c.name}.ug.w", f"{c.name}.ug.ws"),
                  base=0 if c.sh_own else None, key=("moe_ug", c)),
        ]  # fmt: skip
        sh_q = (
            c.dq and not c.sh_own and c.sh_inter > 0
        )  # the shared down items in the queue: no sh_down stage
        if c.sh_sep or not c.sh_inter or sh_q:
            y_writers = (down,)
        down_w = (
            ()
            if c.sh_inter and not c.sh_sep and not sh_q
            else tuple(y.regions(sc.rows, y_writers))
        )
        if c.sh_inter and not sh_q:  # its down on the last CTAs (its mid lands first)
            sh_w = (
                Channel(c.sh_y, c.hidden, Fmt.F32).regions(sc.rows, (sh_down,))
                if c.sh_sep
                else y.regions(sc.rows, y_writers)
            )
            stages.append(Stage(sh_down, c.sh_down_items, partial(_sh_down, c=c), lds_words=lds,
                                writes=sh_w, args=(f"{c.name}.sh.down.w", f"{c.name}.sh.down.ws"),
                                base=MI355X.cus - c.sh_down_items, key=("moe_sh_down", c)))  # fmt: skip
        if (
            c.dq
        ):  # one queue for the routed (and, without sh_own, the shared) down items
            args = (f"{c.name}.down.w", f"{c.name}.down.ws")
            if sh_q:
                args += (f"{c.name}.sh.down.w", f"{c.name}.sh.down.ws")
            dq_lds = (
                lds + 1 + (c.ring_words if c.down_ring else 0)
            )  # the dq word, then down_ring's ring
            stages.append(Stage(down, c.n_r, partial(_down_q, c=c), lds_words=dq_lds, writes=down_w,
                                args=args, base=0, key=("moe_down_q", c)))  # fmt: skip
        else:
            stages.append(Stage(down, c.slots * c.down_parts, partial(_down, c=c),
                                lds_words=lds + (1 + c.ring_words if c.down_ring else 0), writes=down_w,
                                args=(f"{c.name}.down.w", f"{c.name}.down.ws"), key=("moe_down", c)))  # fmt: skip
        tasks = sc.rows * -(-c.hidden // COMBINE_COLS)
        if self.reduce in (
            "push",
            "2shot",
        ):  # combine pushes the partial (a task per chunk x rank, or per chunk to
            # its owners: "2shot"), the consumer stage sums
            comb, dsum, n = f"{c.name}.combine", f"{c.name}.dsum", c.rows * c.hidden
            pairs = (
                shot2_pairs(c.rows, c.hidden, c.tp, c.rs_fmt)
                if c.rs
                else (
                    partial3_pairs(c.rows, c.hidden)
                    if c.sh_sep
                    else partial_pairs(c.part_fmt, c.rows, c.hidden)
                )
            )
            part = Region(c.part, pairs, (comb,), space=Space.PEER)
            push = Stage(comb, tasks * (1 if c.rs else c.tp), partial(_combine, c=c, push=True), writes=(part,),
                         key=("moe_combine_push", c))  # fmt: skip
            if (
                not self.dsum and not self.chain_own
            ):  # the consumer (another stage or the next launch) sums them
                return stages + [push]
            own = []
            if c.rs:
                ag = Region(f"{c.part}.dag", shot2_pairs(c.rows, c.hidden, c.tp, "f32" if c.shared else "bf16"),
                            (f"{c.name}.own",), space=Space.PEER)  # fmt: skip
                own = [Stage(f"{c.name}.own", c.tp * -(-n // c.tp // 4 // c.own_quads), partial(_own, c=c),
                             writes=(ag,), key=("moe_own", c))]  # fmt: skip
            if self.chain_own:  # the owners' sums are the next launch's (``chain_own``)
                return stages + [push, *own]
            return stages + [
                push,
                *own,
                Stage(dsum, -(-n // 4 // THREADS), partial(_dsum, c=c), writes=self.out.regions(sc.rows, (dsum,)),
                      key=("moe_dsum", c), args=(c.out_arg,) if c.out_arg else ()),
            ]  # fmt: skip
        part, red = Channel(c.part, c.hidden, Fmt.F32), Channel(
            c.red, c.hidden, Fmt.F32
        )
        return (
            stages
            + [Stage(f"{c.name}.combine", tasks, partial(_combine, c=c),
                     writes=part.regions(sc.rows, (f"{c.name}.combine",)), key=("moe_combine", c))]
            + PeerAllReduce(f"{c.name}.reduce", part, out=red, proto=self.reduce).stages(sc)
            + [Stage(f"{c.name}.finish", tasks, partial(_finish, c=c),
                     writes=self.out.regions(sc.rows, (f"{c.name}.finish",)), key=("moe_finish", c))]
        )  # fmt: skip

    def pack(self, layer) -> dict[str, torch.Tensor]:
        """``layer.experts``: per expert an object with ``w1`` / ``w3`` / ``w2`` (``.packed`` MXFP4 [N, K / 2] bytes,
        ``.codes`` [N, K / 32])."""
        return pack_experts(layer.experts, self.name)


def _rows(t, a, b, n):
    """Rows a .. b of t, zero rows appended up to n."""
    out = t.new_zeros((n,) + t.shape[1:])
    out[: b - a] = t[a:b]
    return out


def pack_shared(sh, name: str, tp: int = 1, rank: int = 0, fmt: str = "mxfp8",
                local: bool = False) -> dict[str, torch.Tensor]:  # fmt: skip
    """The folded shared expert (``MoEExperts(shared_inter=)``): ``sh.w1`` / ``w3`` / ``w2`` MXFP8 (``.packed``
    E4M3 [N, K], ``.codes``; ``fmt="mxfp4"``: E2M1 bytes [N, K / 2], ``MoEExperts(sh_fmt=)``), rank ``rank``'s slice of
    its width (``sh_on``; ``local``: ``sh`` is that slice already), gate / up interleaved per 16 rows.
    """
    f, per = (
        (gemv.MXFP4, 2) if fmt == "mxfp4" else (gemv.MXFP8, 1)
    )  # (format, K elements a byte)
    inter = sh.w1.packed.shape[0] * (tp if local else 1)
    n = inter // tp
    pad = n if tp == 1 else -(-n // 128) * 128
    a, b = (0, n) if local else (rank * n, (rank + 1) * n)
    u8 = [m.packed.view(torch.uint8) for m in (sh.w1, sh.w3)]
    w1, w3 = (_rows(m, a, b, pad) for m in u8)
    c1, c3 = (_rows(m.codes, a, b, pad) for m in (sh.w1, sh.w3))
    gu = torch.stack(
        (w1.reshape(pad // 16, 16, -1), w3.reshape(pad // 16, 16, -1)), 1
    ).reshape(2 * pad, -1)
    gc = torch.stack(
        (c1.reshape(pad // 16, 16, -1), c3.reshape(pad // 16, 16, -1)), 1
    ).reshape(2 * pad, -1)
    ug_w, ug_s = gemv.pack_weight(gu, gc, f)
    w2 = _rows(
        sh.w2.packed.view(torch.uint8).T, a // per, b // per, pad // per
    ).T.contiguous()
    c2 = _rows(sh.w2.codes.T, a // 32, b // 32, pad // 32).T.contiguous()
    dn_w, dn_s = gemv.pack_weight(w2, c2, f)
    return {
        f"{name}.sh.ug.w": ug_w,
        f"{name}.sh.ug.ws": ug_s,
        f"{name}.sh.down.w": dn_w,
        f"{name}.sh.down.ws": dn_s,
    }


def _intra(ex, rank, tp, pad):
    """Rank ``rank``'s slice of an expert (``intra``): w1 / w3 rows and w2 columns ``rank * inter / tp ..``, zero
    padded to ``pad`` (zero codes: the padded mid features are 0, their w2 columns 0).
    """
    n = ex.w1.packed.shape[0] // tp
    a, b = rank * n, (rank + 1) * n
    w1, w3 = (SimpleNamespace(packed=_rows(m.packed.view(torch.uint8), a, b, pad), codes=_rows(m.codes, a, b, pad))
              for m in (ex.w1, ex.w3))  # fmt: skip
    w2 = SimpleNamespace(packed=_rows(ex.w2.packed.view(torch.uint8).T, a // 2, b // 2, pad // 2).T.contiguous(),
                         codes=_rows(ex.w2.codes.T, a // 32, b // 32, pad // 32).T.contiguous())  # fmt: skip
    return SimpleNamespace(w1=w1, w3=w3, w2=w2)


def gu_split_units(w: torch.Tensor, n_rows: int, k: int) -> torch.Tensor:
    """One expert's ug units (``gemv.pack_weight``, gate / up row groups interleaved) -> layout "intra_gu"'s order
    (every gate row group, then every up one; ATOM's w13 bytes)."""
    return w.reshape(n_rows // 32, 2, -1).transpose(0, 1).contiguous().reshape(-1)


def pack_experts(
    experts, name: str, tp: int = 1, rank: int = 0, layout: str = "expert"
) -> dict[str, torch.Tensor]:
    """Every expert's units, concatenated. ``tp`` > 1: rank ``rank``'s share for ``layout``: ``expert``, the experts
    ``rank * E / tp ..``; ``intra``, every expert's slice (``MoEExperts.inter_on``)."""
    if tp > 1 and layout == "expert":
        el = len(experts) // tp
        experts = experts[rank * el : (rank + 1) * el]
    elif tp > 1:
        inter = experts[0].w1.packed.shape[0]
        pad = -(-inter // tp // 128) * 128
        experts = [_intra(ex, rank, tp, pad) for ex in experts]
    ug_w, ug_s, dn_w, dn_s = [], [], [], []
    for ex in experts:
        n2 = ex.w1.packed.shape[0]
        packed = torch.stack(
            (
                ex.w1.packed.view(torch.uint8).reshape(n2 // 16, 16, -1),
                ex.w3.packed.view(torch.uint8).reshape(n2 // 16, 16, -1),
            ),
            1,
        ).reshape(2 * n2, -1)
        codes = torch.stack(
            (
                ex.w1.codes.reshape(n2 // 16, 16, -1),
                ex.w3.codes.reshape(n2 // 16, 16, -1),
            ),
            1,
        )
        w, s = gemv.pack_weight(packed, codes.reshape(2 * n2, -1))
        if (
            layout == "intra_gu"
        ):  # the weights' row groups as ATOM's w13 (gate, then up); the codes stay interleaved
            w = gu_split_units(w, 2 * n2, ex.w1.packed.shape[1] * 2)
        ug_w.append(w)
        ug_s.append(s)
        w, s = gemv.pack_weight(ex.w2.packed.view(torch.uint8), ex.w2.codes)
        dn_w.append(w)
        dn_s.append(s)
    cat = torch.cat
    return {
        f"{name}.ug.w": cat(ug_w),
        f"{name}.ug.ws": cat(ug_s),
        f"{name}.down.w": cat(dn_w),
        f"{name}.down.ws": cat(dn_s),
    }
