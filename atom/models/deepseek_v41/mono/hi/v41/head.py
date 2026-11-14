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


def head_blocks(tp: int = 1, chain: bool = False, key: str | None = None, layout: str | None = None,
                prefetch: int | None = None, lead: int | None = None, late: bool | None = None,
                src: str | None = None, res_out: str | None = None,
                rows: int | None = None, sample: bool = False, knobs: dict | None = None,
                race: bool | str = False, kparts: int | None = None) -> list[Block]:  # fmt: skip
    """The head program's blocks for a TP-``tp`` rank. Where the fold's inputs come from (``src``): "plain" (args
    ``r_f``, ``gate_f``, ``ffn_out``: a chain tail's), "chain" (``chain=True``: the fold sums the previous launch's
    MoE partials, the last layer launched with ``push_moe``; the program needs ``peer_first=CHAIN_PEER``) or "fused"
    (the last layer's own launch: ``window_layer_blocks(head=...)``, its channels ``rf``, ``gate_f``, ``moe_out``).
    ``res_out``: the fold's R into that plain arg too (module docstring; not with "fused"). ``rows``: the scenario's
    (chained: the previous launch's exchange depends on it, ``layer.chain_pend``: from 6 rows the owners' gathered
    bf16 sums ``moe.part.dag``, else the pushed partials ``moe.part``). ``knobs``: the layer knobs the fold's layout
    follows (``layer.residual_layout`` / ``chain_pend``; default the live ones), e.g. the draft's replicated chain.
    ``sample``: the head's sampling mode (``ArgmaxHead``); ``race``: with it, rows with T > 0 and top_p >= 1 drawn in
    the dot's epilogue (the hook's served head; "check": every row's logits kept)."""
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


def head_program(name: str, sc, chain: bool = False, dyn: bool = False, **knobs):
    """The head's ``Program`` (``head_blocks``); ``chain``: after a ``push_moe`` layer launch, sharing its peer
    group (the chain's regions declared and pinned first). ``dyn``: a ``Program.dyn_rows`` build (a batch band's head,
    launched with the step's live rows)."""
    from ..plan.program import Program
    from .layer import chain_peer_first, residual_layout
    from .schedules import KEY_TP

    stages = place(
        [
            st
            for b in head_blocks(sc.tp, chain=chain, rows=sc.rows, **knobs)
            for st in b.stages(sc)
        ]
    )
    wide = (
        sc.tp != KEY_TP
    )  # at a tp other than the entries': SGPR spills into VGPR lanes allowed (TP8: up to 12; TP2 band 6: 2)
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
