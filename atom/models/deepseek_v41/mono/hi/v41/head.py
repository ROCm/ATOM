# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1's verify head as a hi program: the final fold, the final RMSNorm and the greedy LM head, per
row the argmax id over the whole vocab (TP: vocab-sharded, the exact cross-rank argmax), no logits.

The reference (``inference/model.py`` ``Transformer.forward``) after the last layer::

    h      = hc_post(ffn_out, R_F, post_f, comb_f)        # the last layer's MoE output folded into the streams
    x      = hc_pre(h, pre_f)                             # the streams collapsed with the last FFN seam's pre
    logits = head(norm(x))                                # bf16 [V, 5120] head, vocab-sharded at TP
    ids    = argmax(logits)

ATOM (``atom/models/deepseek_v41``) runs the same as ``SinglePassHCState.collapse`` (``aiter.mhc_post``, then the
collapse kernel), ``RMSNorm`` and ``ParallelHead.get_logits`` (bf16 ``tgemm.mm``, then the all-gather of the bf16
logits), and its greedy sampler / rejection take ``topk_select(logits, 1, tie="low")``: the argmax of the
bf16-rounded logits, ties to the lowest id. ``KEY`` "bf16" is that rule.

Stages (``head_blocks``):

    hz      HCSeam(mixes=False): the fold (pend: the plain ``ffn_out``, the chain tail's; or, chained, the previous
            launch's MoE exchange, ``layer.chain_pend``) and the collapse -> lin_h (BF16)
    norm    NormQuant(plain_once): the final RMSNorm -> xhb (BF16; its MXFP8 output xh is unread)
    head    ArgmaxHead -> plain ``head.ids`` int32 [rows] (and ``head.vals``: the winning bf16 logit)

Args: ``r_f`` [rows, 4, 5120] bf16 (the last layer's R_F), ``gate_f`` [rows, 24] f32 (its FFN seam's gate: pre |
post | comb), ``ffn_out`` [rows, 5120] bf16 (unless chained), ``norm.g`` [5120] bf16, ``head.w`` (``pack``).

``res_out`` (an arg name, plain / chain sources): the fold's R [rows, 4, 5120] bf16 also stored there (``HCSeam``'s
``res_out``): the settled streams after the last layer, for a caller that still collapses them itself (an engine's
fallback head on the same step; ATOM's ``collapse`` of a settled state is one launch).
"""

from __future__ import annotations

import os

import torch

from ..blocks.base import Block, Channel, Fmt
from ..blocks.gfx950.head import ArgmaxHead
from ..blocks.gfx950.head import pack as pack_head
from ..blocks.gfx950.mhc import HCSeam
from ..blocks.gfx950.norm import NormQuant, NormSpec
from .layer import EPS, HIDDEN

VOCAB = 129280
KEY = "bf16"  # the head's comparison: "bf16" (ATOM's greedy sampler: the bf16 logits) or "f32" (the reference's)
LAYOUT = "wave"  # the head weight: the hi runtime's copy in 1 KiB units, "wave" (each wave's 160 KiB contiguous: 1 / 6 rows
#                  -1.1 / -0.7 us vs "packed", A B x3), "packed", "kmajor" (+4 us) or "rows" (ATOM's tensor as it is:
#                  +21 us, the address math takes the prefetch's registers)
# weight units a wave keeps in flight (6 rows: 8 / 16 / 24 / 32 -> 67.9 / 66.4 / 64.1 / 62.0 us; 48 spills); 9-16 rows
# (B = 2 requests: 12) 16 units with x landing beside them (12 rows: 65.4 us vs 66.9 with 32 / 28 / 24 units and the
# rest of the prefetch after x lands); over 16 rows (two 16-row chunks, x in K slices) the most that build: 24 up to 20
# rows, else 16
PREFETCH = 32


def _prefetch(prefetch, rows: int | None, race: bool | str = False) -> int:
    if rows is None or rows <= 8:
        return PREFETCH
    if (
        rows > 30
    ):  # the batch band's 36-48 rows (with KPARTS_ROWS' 10 slices): 16 units spill 3-5 VGPRs, 8 build
        return 8
    if (
        race and rows > 16
    ):  # the race build: 24 units at 17-20 rows spill (18 rows: 4 VGPRs), 16 build
        return 16
    return 16 if rows <= 16 else 24 if rows <= 20 else 16


# x's K slices at least (ArgmaxHead kparts; 0: the fewest that fit the LDS), by rows: KPARTS_ROWS[r] from r rows on.
# The batch band's widths (compile only at TP4): at 28-30 rows the 2 slices that fit the LDS land ~75 VGPRs of
# x beside the prefetch (14-15 spilled to memory at any prefetch); 4 slices build at 30 rows (B5, prefetch 16, greedy
# and sampling); from 36 rows (3 16-row chunks) 10 slices with 8 units of prefetch build at 36-48 (B6-8), 16 do not
KPARTS_ROWS: dict[int, int] = {25: 4, 31: 10}


def _kparts(kparts, rows: int | None) -> int:
    return max(
        (v for r, v in KPARTS_ROWS.items() if rows is not None and rows >= r), default=0
    )


LEAD = 4  # ArgmaxHead: units before x's poll attempt
# the rest of the prefetch after x lands: "auto" from LATE_ROWS rows (over 16 rows: x's K slices land between the
# steps; 12 rows: 32 units beside x spill 65 VGPRs, 16 do not), else a bool
LATE, LATE_ROWS = "auto", 17


def _late(late, rows: int | None) -> bool:
    late = LATE if late is None else late
    return (
        bool(rows is not None and rows >= LATE_ROWS) if late == "auto" else bool(late)
    )


# The dot's row groups a task (``ArgmaxHead(spread=)``, "Spread tasks": the same bits), by TP, for a one-request head
# of at most 16 rows (``head_program``): a rank's 16-row groups ceil(V / 16 tp) in tasks of G, one a CTA. 8 (the
# head's own): TP8's 1010 row groups in 127 tasks, the dot's 165 MB a rank streamed by 127 of the 256 CUs (TP4: 253
# tasks of 8 already). TP8 4: 253 tasks of 4 (4 streaming waves a CU). With the context write merged
# (``head_program(ctx=True)``) ``SPREAD_CTX_TP``: the dot's tasks take the CUs the context write ran on, so it stays
# 8 unless asked (``CTX_AT`` "split" gives the context write's late stages the CUs left: e.g. 5 -> 202 tasks, 48
# free CTAs). Env ATOM_DSV41_MONO_HI_HEAD_SPREAD (an int 1-8, any TP; every rank the same) overrides both.
SPREAD_TP = {8: 4}
SPREAD_CTX_TP: dict[int, int] = {}
SPREAD_ENV = "ATOM_DSV41_MONO_HI_HEAD_SPREAD"


def env_int(name: str, lo: int, hi: int) -> int | None:
    """An int env override in ``lo .. hi`` (None: unset / "auto")."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("", "auto"):
        return None
    v = int(raw)
    if not lo <= v <= hi:
        raise ValueError(f"{name}={raw!r}: {lo} .. {hi}")
    return v


def dot_spread(sc, ctx: bool = False) -> int:
    """``head_program``'s dot row groups a task at ``sc`` (``SPREAD_TP`` / ``SPREAD_CTX_TP``, the env's): a one-request
    head of at most 16 rows, else 8."""
    if sc.batch != 1 or sc.rows > 16:
        return 8
    env = env_int(SPREAD_ENV, 1, 8)
    if env is not None:
        return env
    return (SPREAD_CTX_TP if ctx else SPREAD_TP).get(sc.tp, 8)


def head_blocks(tp: int = 1, chain: bool = False, key: str | None = None, layout: str | None = None,
                prefetch: int | None = None, lead: int | None = None, late: bool | None = None,
                src: str | None = None, res_out: str | None = None,
                rows: int | None = None, sample: bool = False, knobs: dict | None = None,
                race: bool | str = False, kparts: int | None = None, spread: int = 0) -> list[Block]:  # fmt: skip
    """The head program's blocks for a TP-``tp`` rank. Where the fold's inputs come from (``src``): "plain" (args
    ``r_f``, ``gate_f``, ``ffn_out``: a chain tail's), "chain" (``chain=True``: the fold sums the previous launch's
    MoE partials, the last layer launched with ``push_moe``; the program needs ``peer_first=CHAIN_PEER``) or "fused"
    (the last layer's own launch: ``window_layer_blocks(head=...)``, its channels ``rf``, ``gate_f``, ``moe_out``).
    ``res_out``: the fold's R into that plain arg too (module docstring; not with "fused"). ``rows``: the scenario's
    (chained: the previous launch's exchange depends on it, ``layer.chain_pend``: from 6 rows the owners' gathered
    bf16 sums ``moe.part.dag``, else the pushed partials ``moe.part``). ``knobs``: the layer knobs the fold's layout
    follows (``layer.residual_layout`` / ``chain_pend``; default the live ones), e.g. the draft's replicated chain.
    ``sample``: the head's sampling mode (``ArgmaxHead``); ``race``: with it, rows with T > 0 and top_p >= 1 drawn in
    the dot's epilogue (the hook's served head; "check": every row's logits kept). ``spread``: the dot's row groups
    a task (``ArgmaxHead``; 0: 8)."""
    lin, xq, xb = (
        Channel("lin_h", HIDDEN, Fmt.BF16),
        Channel("xh", HIDDEN, Fmt.MXFP8),
        Channel("xhb", HIDDEN, Fmt.BF16),
    )
    src = src or ("chain" if chain else "plain")
    if src == "chain" and tp < 2:
        raise ValueError(
            "chain: a TP chain (the previous launch's pushed MoE partials)"
        )
    spec = NormSpec(
        width=HIDDEN, eps=EPS, threads=320, wps=8, plain=True, plain_once=True
    )
    if res_out is not None and src == "fused":
        raise ValueError(
            "res_out: the plain / chain sources (a fused head's R is the layer's own r_f)"
        )
    from .layer import residual_layout

    # a sharded residual (``RESIDUAL``, ``residual_layout``): a chained / fused head's fold runs on the rank's columns
    # (R_F's, the MoE output's), its LIN gathered for the norm; the plain head reads the chain tail's whole R_F as
    # before
    shard = src != "plain" and residual_layout(tp, knobs=knobs, rows=rows) == "sharded"
    rsd = {"residual": "sharded"} if shard else {}
    from .layer import chain_pend

    pend = chain_pend(tp, rows, knobs=knobs) if src == "chain" else "ffn_out"
    out = (
        {"res_gather": "hzg", "res_full": res_out}
        if shard and res_out
        else {"res_out": res_out}
    )
    seam = HCSeam(
        "hz", "r_f", lin, "gate_f", "", "", "", pend=pend, mixes=False, **out, **rsd
    )
    from ..blocks.gfx950.mhc import GATHER_TILE

    return [
        seam,
        NormQuant(
            "norm",
            lin,
            xq,
            "norm.g",
            bf16_out=xb,
            spec=spec,
            **({"src_gathered": GATHER_TILE} if shard else {}),
        ),
        ArgmaxHead(
            "head",
            xb,
            VOCAB // tp,
            key=key or KEY,
            layout=layout or LAYOUT,
            prefetch=_prefetch(prefetch, rows, race),
            lead=LEAD if lead is None else lead,
            late=_late(late, rows),
            sample=sample,
            race=race,
            kparts=_kparts(kparts, rows),
            spread=spread,
        ),
    ]


def chain_regions(sc, knobs: dict | None = None) -> tuple:
    """The chain's cross-launch peer regions (``layer.CHAIN_PEER``: the MoE partial) as the layer programs declare
    them: a chained head program declares them as ``host_regions`` (the previous launch writes them) and pins them
    first (``Program.peer_first``), so its fold reads them at the layers' offset. ``knobs``: the layer knobs of the
    launch before (default the live ones; the draft's replicated chain)."""
    from .layer import CHAIN_PEER, window_layer_blocks

    kw = {"knobs": knobs} if knobs is not None else {}
    moe = next(b for b in window_layer_blocks(first=True, tp=sc.tp, push_moe=True, rows=sc.rows, **kw)
               if b.name == "moe")  # fmt: skip
    return tuple(r for st in moe.stages(sc) for r in st.writes if r.name in CHAIN_PEER)


PLACE = False  # head_program: the dot on CTAs 0 .., the norm and the pick on the CTAs after it (free when it has < 256
#                tasks; a CTA running a norm task starts its dot task late). A B x2: +0.6 / +0.2 us at 1 / 6 rows: off


def place(stages, cus: int = 256):
    """``PLACE``: the slice at CTA 0, the dot at 0, the norm and the pick right after the dot's tasks."""
    if not PLACE:
        return stages
    by = {st.name: st for st in stages}
    dot = by["head"]
    dot.base = 0
    for name in ("norm", "head.pick"):
        if name in by:
            by[name].base = dot.tasks % cus
    if "hz.slice" in by:
        by["hz.slice"].base = 0
    return stages


# ``head_program(ctx=True)``: the head's stages the merged context write keeps off (the dot and the pick: a context
# task queued behind one of them on its CTA would start when it ends)
CTX_AVOID = ("head", "head.pick")
# where the merged context write's stages go: "after" (every one after the head's stages, on the CTAs the dot and the
# pick leave idle: ``place_after``) or "split" (its early stages, ``CTX_EARLY``: the taps' quantize and main_proj, whose
# inputs are all there at the launch's start, in front of the dot in program order on the least loaded CTAs, where
# they run while the dot's x is still being folded and normed; the rest, from the cross-GPU gather on, after the head's
# stages on the CTAs the dot and the pick leave idle, as "after"). Env ATOM_DSV41_MONO_HI_CTX_AT overrides it (every
# rank the same)
CTX_AT = "after"
CTX_AT_ENV = "ATOM_DSV41_MONO_HI_CTX_AT"
CTX_EARLY = ("cx_q0", "cx_q1", "cx_q2", "cx_proj")
CTX_DOT_LOAD = 4  # ``place_front``: a dot task's weight in the CTAs' loads (a context task in front of it delays it)


def ctx_at() -> str:
    at = os.environ.get(CTX_AT_ENV, "").strip().lower() or CTX_AT
    if at not in ("after", "split"):
        raise ValueError(f"{CTX_AT_ENV}={at!r}: 'after' or 'split'")
    return at


def place_front(stages, front, before: str, heavy: tuple = ("head",), cus: int = 256):
    """``stages`` with ``front`` inserted before stage ``before`` in program order, each front stage on the least
    loaded cyclic run of CTAs (the load: the tasks of ``stages`` before ``before`` and of the front stages placed so
    far, a ``heavy`` stage's task ``CTX_DOT_LOAD``). Every ``stages`` base is frozen at ``plan``'s placement first, so
    the host program's own placement is unchanged; the front stages poll only each other (earlier in program order)."""
    from .recipe import plan_bases

    for st, base in zip(stages, plan_bases(stages, cus), strict=True):
        st.base = base
    at = next(i for i, st in enumerate(stages) if st.name == before)
    load = [0] * cus
    for st in stages:
        w = CTX_DOT_LOAD if st.name in heavy else 1 if stages.index(st) < at else 0
        for t in range(st.tasks):
            load[(st.base + t) % cus] += w
    for st in front:
        n = min(st.tasks, cus)
        at0 = min(range(cus), key=lambda a: (sum(load[(a + t) % cus] for t in range(n)), a))
        for t in range(st.tasks):
            load[(at0 + t) % cus] += 1
        st.base = at0
    return stages[:at] + list(front) + stages[at:]


def head_program(name: str, sc, chain: bool = False, dyn: bool = False, ctx: bool = False,
                 ctx_knobs: dict | None = None, **knobs):  # fmt: skip
    """The head's ``Program`` (``head_blocks``); ``chain``: after a ``push_moe`` layer launch, sharing its peer
    group (the chain's regions declared and pinned first). ``dyn``: a ``Program.dyn_rows`` build (a batch band's head,
    launched with the step's live rows). ``ctx``: the DSpark context write merged into the launch
    (``draft.context_blocks(taps="streams")``: its taps from the tapped layers' streams, args ``cx.ra0..2``, not
    the end-step's), after the head's stages in program order on the CTAs the dot and the pick leave idle
    (``draft.place_after``); the two share no region, so each keeps its own hand-offs. One request (its
    ``cx.bids`` / ``cx.slots`` are the step's). ``ctx_knobs``: the context write's knobs over ``draft.KNOBS``' (as
    ``draft.context_program``'s: ``KV_FMT`` "fp8", ATOM's fp8 pool, its rings' rows written as the "_kv8" CTX's)."""
    from ..plan.program import Program
    from .layer import chain_peer_first, residual_layout
    from .schedules import KEY_TP

    stages = place(
        [
            st
            for b in head_blocks(sc.tp, chain=chain, rows=sc.rows, spread=0 if dyn else dot_spread(sc, ctx),
                                 **knobs)
            for st in b.stages(sc)
        ]
    )
    if ctx:
        from .draft import context_blocks_at, place_after

        if sc.batch != 1 or dyn:
            raise ValueError(f"{name}: the context write merges into a one-request head")
        cx = [st for b in context_blocks_at(sc, "shard", False, ctx_knobs, taps="streams") for st in b.stages(sc)]
        if ctx_at() == "split":
            early = [st for st in cx if st.name in CTX_EARLY]
            late = [st for st in cx if st.name not in CTX_EARLY]
            stages = place_after(place_front(stages, early, "head"), late, CTX_AVOID)
        else:
            stages = place_after(stages, cx, CTX_AVOID)
    wide = (
        sc.tp > KEY_TP
    )  # past the entries' tp: SGPR spills into VGPR lanes allowed (TP8: up to 12)
    # a sharded fold (the rank columns of R): SGPR spills into VGPR lanes allowed (B = 4: 1, 5 with sampling)
    spill = residual_layout(sc.tp, rows=sc.rows) == "sharded" or wide
    return Program(name, sc, stages, host_regions=chain_regions(sc), peer_first=chain_peer_first(sc.tp, rows=sc.rows),
                   sgpr_spill_ok=spill, dyn_rows=dyn)  # fmt: skip


def pack(
    norm_w: torch.Tensor, head_w: torch.Tensor, layout: str | None = None, device="cuda"
) -> dict:
    """The weight args of ``head_blocks``: ``norm.g`` and ``head.w`` (this rank's rows, ``layout``)."""
    return {"norm.g": norm_w.bfloat16().contiguous().to(device),
            "head.w": pack_head(head_w.bfloat16().to(device), layout or LAYOUT)}  # fmt: skip
