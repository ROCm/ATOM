# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's DSpark draft on the hi path: the context-KV write (``context_blocks``), one program a step.

After every target forward the draft's three stages (``mtp.0`` .. ``mtp.2``, window-only layers 40-42) take the step's
verified rows into their window rings. Per row (the reference ``DSparkBlock.forward_embed`` and
``DSparkAttention.forward``'s ``main_kv``; ATOM ``DeepseekV41DSpark.write_context_kv``)::

    x      = cat(tap37, tap38, tap39)              bf16 [rows, 15360]: the inputs of target layers 37-39 (the mean
                                                   over the 4 streams; the engine hook writes them)
    main_x = main_norm(main_proj(MXFP8(x)))        FP8 32x32 linear 15360 -> 5120 (MXFP8 from the checkpoint's FP8:
                                                   exact), RMSNorm (rounded once)
    key_s  = QAT(rope(kv_norm_s(wkv_s(MXFP8(main_x)))))   stage s = 0, 1, 2: wkv 5120 -> 512, RMSNorm, RoPE on the last
                                                   64 dims at the row's position, MXFP8 group-32 quantize-dequantize
    ring_s[slot, p % ring_slots] = key_s           the row at position p into stage s's ring (the engine's layout)

Stages (one program, ``Scenario(1, rows, tp)``):

    cx_q        TapQuant: the three taps (plain bf16 args ``cx.tap0..2`` [rows, 5120]) -> one MXFP8 channel (15360)
    cx_proj     Linear main_proj; at TP column-sharded: this rank's 5120 / tp outputs (the same products and the same
                fp32 sum per output as one rank computing them all: the weight rows are split, not K)
    cx_ag       ColGather (TP): every rank's columns into every rank's full row (tagged peer pairs, one hop)
    cx_mnorm    NormQuant main_norm -> MXFP8 (5120)
    cx_wkv<s>   Linear wkv of stage s -> BF16 (512)
    cx_kvn<s>   NormQuant kv_norm -> BF16 (its MXFP8 output unread, as in the layer)
    cx_kvp<s>   KVPrep: RoPE + the QAT (the layer's block)
    cx_st<s>    RingStore: the row into stage s's ring at the engine's address (``RingStore``: slot, liveness)

Args: ``cx.tap0..2``, ``pos`` int32 [1] (the step's first position: row r is at pos + r), ``rope`` f32 [max_pos, 64]
(the window rope: base theta, no YaRN, as every window-only layer), the weights (``pack``), and per stage s
``cx.ring<s>`` (the stage's ring base: the engine's pool at the layer's ring start) and ``cx.geo`` int32 [3, 8]
(per stage: num_slots, slot_rows, ring_slots, ring_stride, run_rows: ``row = slot * slot_rows + (q // ring_stride) *
run_rows + q % ring_stride``, q = p % ring_slots). The slot and which rows are live: ``direct`` (the same launch as
the target step) from the step's ``cx.bids`` (int32 [rows], -1 on padding) and ``cx.slots`` (int32: the request's
slot), as the engine's own window write decides; or ``meta`` (a later launch) from ``cx.meta`` int32 [4] = (slot or
-1, live-row mask, first position, 0), a snapshot the target step took. B requests (``Scenario(B, spec, tp)``): row r
is request r // spec's row r % spec, at ``pos[r // spec] + r % spec``; direct: its slot ``cx.slots[cx.bids[r]]``
(``cx.slots`` int32 [B]; ``cx.bids[r]`` -1: padding, stores nothing); meta: ``cx.meta`` int32 [B, 4], request b's
snapshot in row b (its mask over its own rows).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr

from ..blocks.base import Block, Channel, Fmt
from ..blocks.gfx950.attention.mla_window import KVPrep
from ..blocks.gfx950.linear import BF16Out, Linear
from ..blocks.gfx950.norm import (
    WAVES,
    NormQuant,
    NormSpec,
    _map_load,
    norm_quant,
    put_row,
)
from ..device.gfx950 import buffer_ops as bo
from ..device.gfx950 import mailbox as mb
from ..device.gfx950.ops import ld_i32, rsrc, traced, uniform
from ..device.gfx950.peer import from_rank, to_rank
from ..plan.program import Program, Region, Scenario, Stage
from ..plan.trace import Space
from ..target import MI355X
from .config import G

THREADS = MI355X.threads
HIDDEN, KVD, EPS = G.hidden, G.head_dim, G.eps
TAPS = 3  # dspark_target_layer_ids: 37, 38, 39
STAGES = 3  # num_nextn_predict_layers
AUX_W = TAPS * HIDDEN  # 15360
GEO = 8  # int32 words a stage in ``cx.geo``
QSPEC = NormSpec(
    width=AUX_W,
    eps=1e-6,
    threads=64,
    wps=4,
    split=AUX_W // 512,
    sum="wave",
    identity=True,
)  # eps unused
PER_TAP = QSPEC.split // TAPS  # quantize tasks a (row, tap)


def _nspec(width, threads, wps):
    return NormSpec(
        width=width, eps=EPS, threads=threads, wps=wps, plain=True, plain_once=True
    )


# ------------------------------------------------------------------ TapQuant


@dataclass(frozen=True)
class _TPlan:
    name: str
    src: str  # the tap's plain arg [rows, HIDDEN] bf16
    tap: int
    out: str  # the MXFP8 channel (AUX_W)


def _tapq(k, t, p: _TPlan):
    if QSPEC.waves < WAVES:
        _tapq_waves(k, t, p)


@traced
def _tapq_waves(k, t, p: _TPlan):
    if k.wave < QSPEC.waves:
        _tapq_body(k, t, p)


def _tapq_body(k, t, p: _TPlan):
    """Task t: part ``t % PER_TAP`` of row ``t // PER_TAP`` of the tap, the quantize-only norm (``NormQuant`` with
    ``identity``), into its columns of the concatenated channel."""
    row, part = t // PER_TAP, t % PER_TAP
    m = QSPEC.qmap
    q = _map_load(m, k.args[p.src], row * (HIDDEN // 2) + part * m.words, k.tid)
    chunks = norm_quant(QSPEC, [(q, None)], None, k.ldsf, k.tid, k.lane, k.wave)[0]
    put_row(
        QSPEC,
        k.mb[f"{p.out}.q"],
        k.mb[f"{p.out}.s"],
        None,
        row,
        k.tid,
        k.tag,
        chunks,
        p.tap * PER_TAP + part,
    )


class TapQuant(Block):
    """The taps (plain bf16 args ``srcs``, [rows, HIDDEN] each) concatenated and quantized to MXFP8 (group 32, E8M0
    ceil scales: ``NormQuant``'s identity spec, the same codes as one quantize of the concatenated row): one stage a
    tap, ``rows * PER_TAP`` tasks each."""

    def __init__(self, name: str, srcs: tuple[str, ...], out: Channel):
        if (
            len(srcs) != TAPS
            or out.fmt is not Fmt.MXFP8
            or out.width != AUX_W
            or out.flagged
        ):
            raise ValueError(
                f"TapQuant {name}: {TAPS} taps into a tagged MXFP8 channel of width {AUX_W}"
            )
        self.name, self.srcs, self.out = name, tuple(srcs), out

    def stages(self, sc: Scenario) -> list[Stage]:
        names = tuple(f"{self.name}{i}" for i in range(TAPS))
        writes = self.out.regions(
            sc.rows, names
        )  # declared once (the first stage), every tap's stage a writer
        return [Stage(n, sc.rows * PER_TAP, partial(_tapq, p=_TPlan(n, src, i, self.out.name)),
                      writes=writes if i == 0 else (), args=(src,), key=("tapq", n, src, i, self.out.name, QSPEC))
                for i, (n, src) in enumerate(zip(names, self.srcs, strict=True))]  # fmt: skip


# ------------------------------------------------------------------ ColGather


@dataclass(frozen=True)
class _GPlan:
    name: str
    x: str  # BF16 channel: this rank's columns [rows, cw * 2]
    out: str  # BF16 channel: the full row [rows, cw * 2 * tp]
    peer: str
    words: int  # 32-bit words of this rank's columns (rows * cw)
    cw: int  # words a row of this rank's columns
    tp: int


@traced
def _gather_cols(k, t, p: _GPlan):
    """Word w of this rank's columns: to every rank's slot (its own included), then every rank's word w into its
    columns of the full row. Each task waits only for the same task's puts on the other ranks (all CTAs resident).
    """
    w = t * THREADS + k.tid
    valid = w < p.words
    wc = fx.min(
        w, fx.Int32(p.words - 1)
    )  # an idle thread polls a real word, puts nothing
    mine = mb.poll([(k.mb[p.x], wc, 1)], k.tag)[0][0]
    if valid:
        for dst in range_constexpr(p.tp):
            mb.put(to_rank(k, p.peer, dst), w, mine, k.tag)
    got = mb.poll([(from_rank(k, p.peer, src), wc, 1) for src in range(p.tp)], k.tag)
    base = (wc // p.cw) * (p.cw * p.tp) + wc % p.cw
    if valid:
        for src in range_constexpr(p.tp):
            mb.put(k.mb[p.out], base + src * p.cw, got[src][0], k.tag)


class ColGather(Block):
    """A column-sharded BF16 output gathered: rank r's ``x`` holds columns ``r * n .. r * n + n - 1`` of every row
    (n = ``x.width``); ``out`` (width n * tp) gets every rank's, the same bits on every rank (a copy, no arithmetic).
    """

    def __init__(self, name: str, x: Channel, out: Channel):
        if x.fmt is not Fmt.BF16 or out.fmt is not Fmt.BF16 or x.flagged or out.flagged:
            raise ValueError(f"ColGather {name}: tagged BF16 channels")
        self.name, self.x, self.out = name, x, out

    def stages(self, sc: Scenario) -> list[Stage]:
        if sc.tp < 2 or self.out.width != self.x.width * sc.tp:
            raise ValueError(
                f"ColGather {self.name}: TP, out = x.width x tp ({self.x.width} x {sc.tp})"
            )
        cw = self.x.width // 2
        p = _GPlan(
            self.name,
            self.x.name,
            self.out.name,
            f"{self.name}.peer",
            sc.rows * cw,
            cw,
            sc.tp,
        )
        writes = (Region(p.peer, p.words, (self.name,), space=Space.PEER, exchange=True),
                  *self.out.regions(sc.rows, (self.name,)))  # fmt: skip
        return [
            Stage(
                self.name,
                -(-p.words // THREADS),
                partial(_gather_cols, p=p),
                writes=writes,
                key=(p,),
            )
        ]


# ------------------------------------------------------------------ RingStore


@dataclass(frozen=True)
class _SPlan:
    name: str
    kvn: str  # BF16 channel (KVD): the row's key
    ring: str  # arg: the stage's ring base
    geo: str  # arg: int32 [STAGES, GEO]
    stage: int
    pos: str  # arg: int32 [1], the step's first position (direct)
    bids: str  # direct: int32 [rows]
    slots: str  # direct: int32 [batch]
    meta: str  # meta: int32 [4] (slot or -1, live-row mask, first position, 0); "" = direct
    spec: int = (
        0  # B > 1 requests (``Scenario(B, spec)``): rows a request; 0: one request (the original build)
    )


@traced
def _ring_store(k, r, p: _SPlan):
    g0 = p.stage * GEO
    geo = k.args[p.geo]
    num_slots = uniform(ld_i32(geo, g0))
    slot_rows = uniform(ld_i32(geo, g0 + 1))
    ring_slots = uniform(ld_i32(geo, g0 + 2))
    stride = uniform(ld_i32(geo, g0 + 3))
    run = uniform(ld_i32(geo, g0 + 4))
    if const_expr(
        p.spec
    ):  # B requests: row r's request (bids) and its slot, at its request's position
        br = uniform(ld_i32(k.args[p.bids], r))
        slot = uniform(ld_i32(k.args[p.slots], fx.max(br, fx.Int32(0))))
        pos = uniform(ld_i32(k.args[p.pos], r // p.spec)) + r % p.spec
        live = (br >= 0) & (slot >= 0) & (slot < num_slots)
    else:
        b0 = uniform(ld_i32(k.args[p.bids], 0))
        br = uniform(ld_i32(k.args[p.bids], r))
        slot = uniform(ld_i32(k.args[p.slots], fx.max(b0, fx.Int32(0))))
        pos = uniform(ld_i32(k.args[p.pos], 0)) + r
        live = (b0 >= 0) & (br == b0) & (slot >= 0) & (slot < num_slots)
    q = pos % ring_slots
    row = fx.max(slot, fx.Int32(0)) * slot_rows + (q // stride) * run + q % stride
    if k.tid < KVD // 2:
        w = mb.poll([(k.mb[p.kvn], (r * KVD) // 2 + k.tid, 1)], k.tag)[0][0]
        if live:
            bo.buffer_store(w, rsrc(k.args[p.ring]), row * (KVD // 2) + k.tid)


class RingStore(Block):
    """Row r of ``kvn`` (a key, bf16 [KVD]) into an engine-owned window ring (module docstring: the address, the
    slot and the liveness), one task a row. Rows that are not the step's request's (padding, a dummy step) store
    nothing; nothing else of the ring is written."""

    def __init__(self, name: str, kvn: Channel, ring: str, geo: str, stage: int, pos: str = "pos",
                 bids: str = "cx.bids", slots: str = "cx.slots", meta: str = ""):  # fmt: skip
        if kvn.fmt is not Fmt.BF16 or kvn.width != KVD or kvn.flagged:
            raise ValueError(f"RingStore {name}: a tagged BF16 channel of width {KVD}")
        self.p = _SPlan(name, kvn.name, ring, geo, stage, pos, bids if not meta else "", slots if not meta else "",
                        meta)  # fmt: skip
        self.name = name

    def stages(self, sc: Scenario) -> list[Stage]:
        p = replace(self.p, spec=sc.spec) if sc.batch > 1 else self.p
        args = (
            (p.ring, p.geo, p.meta)
            if p.meta
            else (p.ring, p.geo, p.pos, p.bids, p.slots)
        )
        return [Stage(p.name, sc.rows, partial(_ring_store, p=p), args=args, key=(p,))]


# ------------------------------------------------------------------ the program


def channels(tp: int, cols: str = "shard"):
    """The context program's channels (``cols``: "shard" = main_proj column-sharded at TP, "rep" = replicated)."""
    local = HIDDEN // tp if tp > 1 and cols == "shard" else HIDDEN
    return {
        "auxq": Channel("cx_auxq", AUX_W, Fmt.MXFP8),
        "fcol": Channel("cx_fcol", local, Fmt.BF16),
        "fco": Channel("cx_fco", HIDDEN, Fmt.BF16),
        "mnq": Channel("cx_mnq", HIDDEN, Fmt.MXFP8),
        "kv": [Channel(f"cx_kv{s}", KVD, Fmt.BF16) for s in range(STAGES)],
        "kvq": [Channel(f"cx_kvq{s}", KVD, Fmt.MXFP8) for s in range(STAGES)],
        "kvb": [Channel(f"cx_kvb{s}", KVD, Fmt.BF16) for s in range(STAGES)],
        "kvn": [Channel(f"cx_kvn{s}", KVD, Fmt.BF16) for s in range(STAGES)],
    }


# the Linears' K slices (None: the plan's choice). main_proj in 3 (240 tasks; the plan's 2: 160): 20.8 vs 22.1 µs a
# launch, 4: 21.1; wkv in 2: 23.0 (TP4, 6 rows, CUDA-graph timing; bit-exact vs the reference in every variant)
KNOBS = {"PROJ_KSPLIT": 3, "WKV_KSPLIT": None}
# by rows (``context_program``): main_proj in 6 K slices over 18 rows (3 leaves no task shape at 24 rows, 4 / 5 spill
# SGPRs)
KNOBS_ROWS = {r: {"PROJ_KSPLIT": 6} for r in range(19, 25)}


def context_blocks(
    tp: int = 1, cols: str = "shard", meta: bool = False, knobs: dict | None = None
) -> list[Block]:
    """The context write's blocks (module docstring). ``cols``: "shard" (TP: main_proj column-sharded + ``ColGather``)
    or "rep" (every rank all 5120 columns: the same bits, 4x the weight bytes); ``meta``: the slot / liveness from the
    ``cx.meta`` snapshot (a launch after the target step's) instead of the step's ``cx.bids`` / ``cx.slots``;
    ``knobs``: values over ``KNOBS``'."""
    if cols not in ("shard", "rep"):
        raise ValueError(f"cols {cols!r}: 'shard' or 'rep'")
    kn = {**KNOBS, **(knobs or {})}
    C = channels(tp, cols)
    shard = tp > 1 and cols == "shard"
    w8 = {"weight_format": "mxfp8"}
    out = BF16Out(C["fcol"] if shard else C["fco"])
    blocks = [
        TapQuant("cx_q", tuple(f"cx.tap{i}" for i in range(TAPS)), C["auxq"]),
        Linear(
            "cx_proj",
            ("main_proj",),
            C["auxq"],
            C["fcol"].width,
            out,
            k_split=kn["PROJ_KSPLIT"],
            **w8,
        ),
    ]
    if shard:
        blocks.append(ColGather("cx_ag", C["fcol"], C["fco"]))
    blocks.append(
        NormQuant(
            "cx_mnorm",
            C["fco"],
            C["mnq"],
            "cx.main_norm.g",
            spec=_nspec(HIDDEN, 320, 8),
        )
    )
    for s in range(STAGES):
        blocks.append(Linear(f"cx_wkv{s}", (f"wkv{s}",), C["mnq"], KVD, BF16Out(C["kv"][s]), k_split=kn["WKV_KSPLIT"],
                             **w8))  # fmt: skip
    for s in range(STAGES):
        blocks.append(NormQuant(f"cx_kvn{s}", C["kv"][s], C["kvq"][s], f"cx.kv_norm{s}.g", bf16_out=C["kvb"][s],
                                spec=_nspec(KVD, 64, 4)))  # fmt: skip
    for s in range(STAGES):
        blocks.append(KVPrep(f"cx_kvp{s}", C["kvb"][s], C["kvn"][s], "pos", "rope"))
    for s in range(STAGES):
        blocks.append(
            RingStore(
                f"cx_st{s}",
                C["kvn"][s],
                f"cx.ring{s}",
                "cx.geo",
                s,
                meta="cx.meta" if meta else "",
            )
        )
    return blocks


def context_program(name: str, sc: Scenario, cols: str = "shard", meta: bool = False, knobs: dict | None = None,
                    **program):  # fmt: skip
    """-> (blocks, ``Program``) of the context write at ``sc`` (``Scenario(1, rows, tp)``); ``knobs``: values over
    ``KNOBS``' (and ``KNOBS_ROWS[sc.rows]``'); ``program``: ``Program`` options. ``Scenario(B,
    rows, tp)``: B requests' rows, each at its request's position (``pos`` [B]) and ring slot.
    """
    blocks = context_blocks(
        sc.tp, cols, meta, {**KNOBS_ROWS.get(sc.rows, {}), **(knobs or {})}
    )
    return blocks, Program(
        name, sc, [st for b in blocks for st in b.stages(sc)], **program
    )


# ------------------------------------------------------------------ weights


def pack(
    raw: dict[str, torch.Tensor],
    tp: int = 1,
    rank: int = 0,
    cols: str = "shard",
    device="cuda",
) -> dict:
    """Every weight arg of ``context_blocks(tp, cols)`` for rank ``rank``: main_proj (this rank's 5120 / tp output
    rows when sharded), the stages' wkv (MXFP8 from FP8 32 x 32: exact), the norm weights (bf16).
    """
    from types import SimpleNamespace

    from ..weights.mx import fp8_block_to_mxfp8

    def mx(wt, sc):
        m = fp8_block_to_mxfp8(wt.contiguous(), sc.contiguous())
        m.packed, m.codes = m.packed.to(device), m.codes.to(device)
        return m

    wt, sc = raw["mtp.0.main_proj.weight"], raw["mtp.0.main_proj.scale"]
    if tp > 1 and cols == "shard":
        n = HIDDEN // tp
        wt, sc = wt.narrow(0, rank * n, n), sc.narrow(0, rank * n // 32, n // 32)
    linears = {"main_proj": mx(wt, sc)}
    for s in range(STAGES):
        linears[f"wkv{s}"] = mx(
            raw[f"mtp.{s}.attn.wkv.weight"], raw[f"mtp.{s}.attn.wkv.scale"]
        )
    lay = SimpleNamespace(linears=linears, retained={})
    out = {}
    for b in context_blocks(
        tp, cols
    ):  # (the weight layout does not depend on the knobs)
        if isinstance(b, Linear):
            out.update(b.pack(lay))
    bf = lambda t: t.bfloat16().contiguous().to(device)
    out["cx.main_norm.g"] = bf(raw["mtp.0.main_norm.weight"])
    for s in range(STAGES):
        out[f"cx.kv_norm{s}.g"] = bf(raw[f"mtp.{s}.attn.kv_norm.weight"])
    return out


# ------------------------------------------------------------------ torch reference (the official arithmetic)


# ------------------------------------------------------------------ the draft stages (the block backbone)
#
# The three stages are V4.1 window layers (``recipe.window_layer_blocks``) with the draft's routing (128 experts,
# top-3) and the block attention (``draft_attn.MLABlockAttn``: every row the request's window up to the anchor and the
# whole block, read from the engine's draft ring in place; no ring write). Chained as the target's layers: stage 0
# starts from the block's embedding streams (``first``: ``h_in`` / ``gin``, the engine's ``build_block_state``) and
# pushes its MoE partials, stages 1 and 2 sum the previous launch's and push theirs, the draft head's fold sums stage
# 2's. Args beyond a layer's: ``dr.ring`` (the stage's ring base), ``dr.geo`` (int32 [8]: num_slots, slot_rows,
# ring_slots, ring_stride, run_rows), ``dr.slot`` (int32 [batch]: each request's slot); ``pos`` is each request's
# first block position (anchor + 1).

DRAFT_EXPERTS, DRAFT_TOPK = (
    128,
    3,
)  # dspark_n_routed_experts / dspark_num_experts_per_tok
STAGE_ROLES = {
    0: (True, True, False),
    1: (False, True, True),
    2: (False, True, True),
}  # (first, push, prev)


def stage_blocks(
    stage: int, tp: int = 4, rows: int = 5, knobs: dict | None = None
) -> list[Block]:
    """Draft stage ``stage``'s blocks: the window layer's with the block attention in place of ``MLAWindowAttn`` and
    no ``RingWrite``."""
    from .draft_attn import MLABlockAttn
    from .recipe import window_layer_blocks

    first, push, prev = STAGE_ROLES[stage]
    # (the draft's own chain keeps its residual replicated and its rs partials as pairs: the target's ``RESIDUAL`` /
    # ``RS_PACK`` are the target layers')
    knobs = {**(knobs or {}), "RESIDUAL": "replicated", "RS_PACK": "pair"}
    blocks = window_layer_blocks(first=first, tp=tp, push_moe=push, prev_moe=prev, rows=rows, knobs=knobs,
                                 experts=DRAFT_EXPERTS, topk=DRAFT_TOPK)  # fmt: skip
    out = []
    for b in blocks:
        if b.name == "ringw":
            continue
        if (
            b.name == "attn"
        ):  # (its output's inverse RoPE as the window attention's: ATTN_OUT_F32_ROPE)
            b = MLABlockAttn("attn", b.q, b.kvn, b.out, b.pos, b.rope, b.sink, "dr.ring", "dr.geo", "dr.slot",
                             heads=b.heads, unrope_f32=b.unrope_f32)  # fmt: skip
        out.append(b)
    return out


def stage_program(
    name: str, stage: int, sc: Scenario, knobs: dict | None = None, **program
):
    """-> (blocks, ``Program``) of draft stage ``stage`` at ``sc`` (``Scenario(1, block, tp)``), its chain regions
    pinned first (``Program.peer_first``). SGPR spills into VGPR lanes allowed (never memory): stage 0 (``first``, the
    block attention's ring arithmetic) is one SGPR over the budget."""
    from . import schedules
    from .recipe import CHAIN_PEER, moe_route_early

    blocks = stage_blocks(stage, sc.tp, sc.rows, knobs)
    kn = schedules.knobs("window", sc.rows, over=knobs, tp=sc.tp)
    stages = moe_route_early([st for b in blocks for st in b.stages(sc)], kn)
    return blocks, Program(
        name, sc, stages, peer_first=CHAIN_PEER, sgpr_spill_ok=True, **program
    )


def pack_stage(stage: int, w: dict, raw: dict, tp: int = 4, rank: int = 0, device="cuda", local: bool = False,
               moe: dict | None = None) -> dict:  # fmt: skip
    """Every weight arg of ``stage_blocks(stage, tp)`` for rank ``rank`` (``layer_impl.pack`` with the draft's experts
    and prefix; ``local`` / ``moe``: its rank-local inputs)."""
    from .layer_impl import pack as pack_layer

    return pack_layer(40 + stage, w, raw, device=device, tp=tp, rank=rank, local=local, moe=moe,
                      experts=DRAFT_EXPERTS, prefix=f"mtp.{stage}.")  # fmt: skip


# ------------------------------------------------------------------ the draft head


# rows -> the MarkovHead's prefetch over the verify head's rule: B = 2 / 3 / 4 requests of 5 positions build at 32 /
# 24 / 16 units
DRAFT_PREFETCH = {10: 32, 15: 24, 20: 16}

# (rows, tp) -> x's K slices at a TP other than the rule's (compile only, MI355X): TP2 at 6 requests of 5 (two 128-id
# blocks a task, the base logits as bf16 pairs) spills 4 VGPRs to memory at 10 slices (and at 32), 16 / 20 / 40 build
DRAFT_KPARTS_TP: dict[tuple[int, int], int] = {(30, 2): 16}


def DRAFT_KPARTS(rows: int, tp: int = 4) -> int:
    """x's K slices at least (``MarkovHead(kparts=)``; 0: the fewest that fit) at a draft of ``rows`` positions: 10
    past 20 rows (5-6 requests of 5: the fewest that fit spill 18-20 VGPRs to memory, 10 build at prefetch 8 and 16;
    7-8 requests spill 10-18 at any K slices: the B requests' side-by-side Markov steps, not x; compile only, TP4).
    ``DRAFT_KPARTS_TP``'s (rows, tp) entry first.
    """
    return DRAFT_KPARTS_TP.get((rows, tp)) or (10 if rows > 20 else 0)


def head_blocks(
    tp: int = 4,
    rows: int = 5,
    res_out: str | None = None,
    prefetch: int | None = None,
    kparts: int | None = None,
) -> list[Block]:
    """The draft's head after stage 2 (chained: its fold sums stage 2's pushed MoE partials): the fold and the
    collapse (``head.head_blocks``' seam), the last stage's ``norm`` (arg ``norm.g``: ``mtp.2.norm``), then
    ``MarkovHead`` on the target's LM head (arg ``mk.w``: the verify head's packed ``head.w``, the same tensor) and
    the Markov tables (``mk.w1`` / ``mk.w2``: ``mtp.2.markov_head.{embed,head}``), x_{-1} = ``mk.anchor``, the ids
    into ``mk.ids``. ``res_out``: the fold's R (stage 2's output streams) also into that plain arg.
    """
    from ..blocks.gfx950.markov import MarkovHead
    from . import head as H

    # (stage 2 pushed in the draft's replicated layout, ``stage_blocks``: the fold sums those partials, not the
    # target's ``RESIDUAL``)
    hb = H.head_blocks(tp, chain=True, rows=rows, res_out=res_out, knobs=_draft_knobs())
    pf = (
        prefetch
        or DRAFT_PREFETCH.get(rows)
        or (H.PREFETCH if rows <= 5 else H._prefetch(None, rows))
    )
    kp = DRAFT_KPARTS(rows, tp) if kparts is None else kparts
    return hb[:2] + [
        MarkovHead(
            "mk",
            hb[2].x,
            H.VOCAB // tp,
            layout=H.LAYOUT,
            prefetch=pf,
            lead=H.LEAD,
            kparts=kp,
        )
    ]


def head_program(name: str, sc: Scenario, res_out: str | None = None, **program):
    """-> (blocks, ``Program``) of the draft head at ``sc``: after stage 2's launch, its chain regions declared and
    pinned first (``head.head_program``'s chained form)."""
    from . import head as H
    from .recipe import CHAIN_PEER
    from .schedules import KEY_TP

    blocks = head_blocks(sc.tp, sc.rows, res_out)
    stages = [st for b in blocks for st in b.stages(sc)]
    # at a tp other than the entries': SGPR spills into VGPR lanes allowed (TP8: 12), as the verify head's
    program.setdefault("sgpr_spill_ok", sc.tp != KEY_TP)
    return blocks, Program(name, sc, stages, host_regions=H.chain_regions(sc, _draft_knobs()), peer_first=CHAIN_PEER,
                           **program)  # fmt: skip


def _draft_knobs() -> dict:
    """The layer knobs of the draft's own chain: the live ones with the residual replicated and the rs partials as
    pairs (``stage_blocks``)."""
    from . import schedules

    return {**schedules.current(), "RESIDUAL": "replicated", "RS_PACK": "pair"}
