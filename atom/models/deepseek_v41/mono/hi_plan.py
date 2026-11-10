# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook's layer plan, pure Python (no Triton, AITER or FlyDSL: the CPU tests import it as is).

``plan_layers``: which layers run the hi runtime's layer program, as which
program kind, reading which owner's caches. A layer runs on the hi path when every
owner it reads (its KV owner, its top-k owner, the candidate source) runs on
hi too (a consumer reads its owner's selection, which only a hi owner
writes in the same step), and,
without ``engram_on``, when it is not an Engram
layer. With V4.1's topology that is layers 0, 2-13 and 20-39 (33 of 40):
layers 1 / 14 (Engram) and 15-19 (owner 14) keep ATOM's K1 / K2. With
``engram`` all 40: an Engram layer runs the hi runtime's layer with the Engram's gate
half in its seam A (``window_layer_blocks(engram=True)``: layer 1 a non-first
window layer, layer 14 a ratio-2 KV + index source), its token-only half in a
launch before layer 0 (``engram_kv_stages``), and 15-19 follow owner 14.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerPlan:
    layer_id: int
    kind: str  # "window0" | "consumer" | "kv+index source" | "index source" | "window" (ATOM)
    ratio: int
    hi: bool
    why: str = ""  # why ATOM runs it
    kv_owner: int | None = None
    topk_owner: int | None = None
    cand: str = ""  # "source" (layer 20) | "reader" (an index source past it) | ""
    engram: bool = False  # an Engram layer on the hi path (the gate half in seam A)

    @property
    def program_key(self) -> tuple:
        eng = ("engram",) if self.engram else ()
        if self.kind == "window0":
            return ("window0",)
        if self.kind == "consumer":
            return ("consumer", self.ratio)
        if self.kind == "window":
            return ("window",) + eng
        return (self.kind, self.ratio, self.cand) + eng


def plan_layers(topology, engram_ids, engram_on: bool = False) -> list[LayerPlan]:
    """Every layer's plan from ATOM's attention topology (``LayerAttentionSpec``s)
    and the Engram layer ids; ``engram_on``: the Engram layers on the hi path too
    (module docstring)."""
    engram = set(engram_ids)
    out: dict[int, LayerPlan] = {}
    for spec in topology:
        i = spec.layer_id
        cand = ""
        if spec.ratio == 0:
            kind = "window0" if i == 0 else "window"
        elif spec.topk_owner == i and spec.kv_owner == i:
            kind = "kv+index source"
            cand = "source" if spec.produces_candidates else ""
        elif spec.topk_owner == i:
            kind = "index source"
            cand = "reader" if spec.candidate_source is not None else ""
        else:
            kind = "consumer"
        why = ""
        if i in engram and not engram_on:
            why = "Engram (engram_on False)"
        elif i in engram and i == 0:
            why = "an Engram layer 0 (its gate half needs a fold before it)"
        elif kind == "window" and i not in engram:
            why = "a window layer past layer 0 without Engram (V4.1 has none)"
        else:
            for owner, what in ((spec.kv_owner, "KV"), (spec.topk_owner, "top-k"),
                                (spec.candidate_source, "candidate")):  # fmt: skip
                if owner is not None and owner != i and not out[owner].hi:
                    why = f"its {what} owner {owner} runs on ATOM"
                    break
        eng = engram_on and i in engram
        out[i] = LayerPlan(
            i, kind, spec.ratio, not why, why, spec.kv_owner, spec.topk_owner, cand, eng
        )
    return [out[i] for i in sorted(out)]


def chain_role(plan: dict, i: int, head: bool = False) -> tuple[bool, bool]:
    """(push_moe, prev_moe) of hi layer i in a chained hook (``plan``:
    layer id -> ``LayerPlan``): it pushes its MoE partials when the next launch
    sums them (a hi layer i + 1, or, ``head``, the verify head after the
    model's last layer), and it sums the previous launch's when layer i - 1 ran
    on the hi path. A run's last layer before an ATOM layer is its tail (sums, no
    push)."""
    nxt, prv = plan.get(i + 1), plan.get(i - 1)
    push = bool(nxt and nxt.hi) or (head and nxt is None)
    return push, bool(i and prv and prv.hi)


def n_max(max_ctx: int, ratio: int) -> int:
    """The compressed positions a step can see at ``max_ctx`` (+ 16: a verify's
    rows), a multiple of 32: the programs' cache length ."""
    return -(-((max_ctx + 16) // ratio) // 32) * 32
