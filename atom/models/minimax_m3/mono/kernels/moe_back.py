# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``mono_moe_back``: the back half of one MiniMax-M3 sparse MoE layer for 32 / 64 token rows.

One persistent launch (``BLOCKS`` CTAs x ``THREADS`` threads) that takes the front half's ``x_moe``
([n, HIDDEN] bf16, the gemma-normalized MoE input) and returns ``ar_out`` ([n, HIDDEN] bf16, the rank-summed FFN
output)::

    R   router      task = (TPT = n / 16 tokens, 8 experts), one per CTA: bf16 gate GEMV, sigmoid(bf16 logit) + bias key,
                    tagged score pairs; the same CTA quantizes its slice of xn to MXFP8 (shared) and MXFP4 (routed)
    T   routing     every CTA, from the scores: top-4 per token, per expert a 64-bit token mask, the (expert, 16-column
                    tile) UNITS and per unit column c the mid row (token * 4 + slot) and route weight
    U   up / gate   pair = (unit, 32 mid rows); A = MXFP4 weight tiles, K split over the 8 waves, B = the unit's columns'
                    xn4 rows read straight from scratch; the shared expert's 24 pairs run on the last CTAs
    D   down        CTA td < 192 owns 32 output rows over EVERY unit; wave q takes units q, q + 4, ..; per (unit, column)
                    bf16(acc * w) scattered into the wave's own per-token LDS partials; shared expert's MXFP8 GEMV
    AR  all-reduce  reduce-scatter to the owner rank (t % W), rank-ordered fp32 sum, all-gather

Numerics are those of the K4 wide a4w4 path with ``prod_round`` (``stages/moe_wide.py``). See PLAN.md.
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import Int32, Int64

from atom.models.minimax_m3.mono.config import HIDDEN, MOE_SLOTS, N_ROUTED, TOP_K
from atom.models.minimax_m3.mono.kernels.stages.moe_back64 import (
    MID4_WORDS,
    MID8_WORDS,
    MIDSC_WORDS,
    N_DN,
    UG_PAIRS,
    XN4_WORDS,
    XN8_WORDS,
    XSC_WORDS,
    back_lds_layout,
    moe_back_defs,
    umax_of,
)
from atom.models.minimax_m3.mono.layout import SCORE_COPIES, _layout
from atom.models.minimax_m3.mono.sources import SOURCES
from atom.mono.device.ops import kernel_symbol, launcher, uniform
from atom.mono.device.stamps import stamp as stamp_point
from atom.mono.device.stamps import stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, sreg
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS
from atom.mono.runtime.abi import KernelAbi

BACK_ABI = KernelAbi(
    (
        "x_moe", "w_gate", "bias", "w13", "s13", "w2", "s2", "w13s", "s13s", "w2s", "s2s",
        "ar_out", "scratch", "sym", "peers", "rank", "layer", "batch_ids", "tl",
    )
)  # fmt: skip


@dataclass(frozen=True)
class BackBuild:
    """Every parameter of a back-half build (all of them reach the JIT cache key)."""

    npes: int = field(metadata={"sym": "tp"})
    tokens: int = field(metadata={"sym": "s"})
    route_scale: float
    shared_weight: float
    swiglu_limit: float
    dn_depth: int = field(
        metadata={"sym": "dd"}
    )  # down units whose weights are in flight
    sh_units: int = field(
        metadata={"sym": "su"}
    )  # routed pairs a shared pair displaces on its host
    debug: bool = False
    stop: str = ""
    timeline: bool = field(
        default=False, metadata={"sym": "tl"}
    )  # per-CTA stage stamps into ``tl``
    a_nt: int = field(default=1, metadata={"sym": "nt"})  # nontemporal weight loads
    nocompute: int = field(
        default=0, metadata={"sym": "nc"}
    )  # a probe: the weight tiles loaded and summed, no MFMA
    nopubwait: int = field(
        default=0, metadata={"sym": "pw"}
    )  # a timing probe: the up / gate flag without the store drain (wrong results)
    dn_split: int = field(
        default=0, metadata={"sym": "ds"}
    )  # the down stage polls the up / gate flags in two halves (see ``stage_down``)
    dn_bdepth: int = field(
        default=1, metadata={"sym": "db"}
    )  # down units whose mid rows are in flight
    steal: int = field(
        default=0, metadata={"sym": "st"}
    )  # up / gate pairs by atomic work stealing (the shared pair's host just claims later)
    ar_flag: int = field(
        default=0, metadata={"sym": "af"}
    )  # 1: the FFN all-reduce sends plain payload + a flag pair, not tagged pairs  # a timing / resource probe: leave the program after this stage (router, routing, ug, shared)


def back_scratch_layout(tokens: int) -> dict:
    """Scratch regions of an n-row build (byte offsets + ``_bytes``), 256-byte aligned. Sizes in 8-byte pairs."""
    n = tokens
    um = umax_of(n)
    return _layout(
        [
            ("scores", 2 * N_ROUTED * n * SCORE_COPIES),
            ("xn8p", n * XN8_WORDS // 2),
            ("xscp", n * XSC_WORDS // 2),
            ("xn4p", n * XN4_WORDS // 2),
            ("xs4p", n * XSC_WORDS // 2),
            ("xndone", BLOCKS),
            ("mid4p", TOP_K * n * MID4_WORDS // 2),
            ("mids4p", TOP_K * n * MIDSC_WORDS // 2),
            ("mid8p", n * MID8_WORDS // 2),
            ("midsp", n * MIDSC_WORDS // 2),
            ("ugdone", (um + 1) * UG_PAIRS),
            ("sel", 2 * MOE_SLOTS * n),
            ("claim", 64),  # the work-stealing counters, a word a layer (layer & 127)
        ]
    )


def back_sym_layout(npes: int, tokens: int) -> dict:
    """Per-rank peer buffer: every source rank's FFN partial rows, then the owners' gathered sums."""
    return _layout(
        [
            ("ffn", npes * tokens * HIDDEN // 2),
            ("ffn_ag", tokens * HIDDEN // 2),
            # ``ar_flag``: plain bf16 payloads [src][token][HIDDEN] and the owners' sums, then their flag pairs
            ("ffnb", npes * tokens * HIDDEN // 4),
            ("ffnb_ag", tokens * HIDDEN // 4),
            ("ffnb_f", npes * N_DN),
            ("ffnb_agf", N_DN),
        ]
    )


_CURRENT_STREAM = fx.Stream(None)


def build_moe_back_kernel(
    npes: int,
    route_scale: float,
    shared_weight: float,
    swiglu_limit: float,
    tokens: int = 32,
    dn_depth: int = 3,
    sh_units: int = 8,
    debug: bool = False,
    stop: str = "",
    ar_flag: int = 0,
    timeline: bool = False,
    a_nt: int = 1,
    nocompute: int = 0,
    steal: int = 0,
    nopubwait: int = 0,
    dn_split: int = 0,
    dn_bdepth: int = 1,
):
    """``@flyc.jit`` launcher of the back half for ``tokens`` (a multiple of 16, <= 64) rows on one rank of an
    ``npes``-way TP group (``BACK_ABI``)."""
    assert tokens % 16 == 0 and 16 <= tokens <= 64, tokens
    key = BackBuild(
        npes, tokens, route_scale, shared_weight, swiglu_limit, dn_depth, sh_units, debug=debug, stop=stop, timeline=timeline,
        ar_flag=ar_flag, a_nt=a_nt, nocompute=nocompute, steal=steal, nopubwait=nopubwait, dn_split=dn_split, dn_bdepth=dn_bdepth,
    )  # fmt: skip
    build_key = key_tuple(key, SOURCES)
    n = tokens
    W = npes
    G = BLOCKS
    SCR = back_scratch_layout(n)
    SY = back_sym_layout(W, n)
    LDS_WORDS = back_lds_layout(n)["end"]

    @fx.struct
    class Smem:
        x: fx.Array[fx.Int32, LDS_WORDS, 16]
        tls: fx.Array[fx.Int64, 32 if timeline else 1, 16]

    kernel_name = kernel_symbol("minimax_m3_moe_back", **symbol_params(key))

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def moe_back_kernel(
        x_moe: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        w13s: Int64,
        s13s: Int64,
        w2s: Int64,
        s2s: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        batch_ids: Int64,
        tl: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        g4 = lane // 16
        l16 = lane % 16
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        v4f = fx.Vector.make_type(4, fx.Float32)
        mbox = Mailbox(layer, None, {}, ())
        mb_put = mbox.put
        mb_put_bf = mbox.put_bf
        mb_put_words = mbox.put_words
        mb_poll = mbox.poll

        def mb(name):
            return scratch + fx.Int64(SCR[name])

        def mr(name):
            return sreg(scratch, SCR[name], name)

        def acquire():
            fx.memory_fence(
                syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
            )

        tls = lds.tls.ptr

        def stamp(k):
            stamp_point(timeline, tls, tid, k)

        stamp_begin(timeline, tls, tid, 32)
        x_base = x_moe
        W = npes
        MERGED = False
        AR_FLAG = ar_flag
        A_NT = a_nt
        NOCOMP = nocompute
        STEAL = steal
        DN_SPLIT = dn_split
        DEPB = dn_bdepth
        NOPUBWAIT = nopubwait
        DEP = dn_depth
        SU = sh_units
        ctx = dict(locals())
        ctx["n"] = n
        ctx["SY"] = SY
        ctx["stop"] = stop
        ctx["route_scale"] = route_scale
        ctx["shared_weight"] = shared_weight
        ctx["swiglu_limit"] = swiglu_limit
        ctx.update(moe_back_defs(ctx))
        stage_router = ctx["stage_router"]
        stage_routing = ctx["stage_routing"]
        stage_ug = ctx["stage_ug"]
        stage_shared_host = ctx["stage_shared_host"]
        stage_down = ctx["stage_down"]
        wscan = ctx["wscan"]

        # ------------------------------------------------------------------ the program
        stage_router()
        if const_expr(stop == "router"):
            return
        if const_expr(stop != "ug"):
            stage_shared_host()
        stage_routing()
        if const_expr(stop == "routing"):
            return
        nu = uniform(fx.ptr_load(wscan + 2))
        stage_ug(nu)
        if const_expr(stop in ("ug", "shared")):
            return
        if bid < N_DN:
            stage_down(nu)
        stamp_flush(timeline, tls, tl, tid, bid, 32)

    @launcher(kernel_name)
    def launch_moe_back(
        x_moe: Int64,
        w_gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        w13s: Int64,
        s13s: Int64,
        w2s: Int64,
        s2s: Int64,
        ar_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        layer: Int32,
        batch_ids: Int64,
        tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = build_key
        moe_back_kernel(
            x_moe, w_gate, bias, w13, s13, w2, s2, w13s, s13s, w2s, s2s,
            ar_out, scratch, sym, peers, rank, layer, batch_ids, tl,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)  # fmt: skip

    BACK_ABI.check(moe_back_kernel, launch_moe_back)
    return launch_moe_back
