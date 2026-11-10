# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The DeepSeek-V4.1 layer on a TP rank: its weights packed for ``recipe.window_layer_blocks`` (``pack``)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ..blocks.gfx950.attention.dsa_indexer import Indexer
from ..blocks.gfx950.compressor import Compressor
from ..blocks.gfx950.linear import Linear
from ..blocks.gfx950.moe_experts import pack_experts, pack_shared
from ..weights.mx import fp8_block_to_mxfp8
from . import schedules
from .recipe import EXPERTS, HD, HEADS, OGROUPS, ORANK, Source, window_layer_blocks


def pack(
    layer: int,
    w: dict,
    raw: dict,
    device="cuda",
    tp: int = 1,
    rank: int = 0,
    local: bool = False,
    moe: dict | None = None,
    source: Source | None = None,
    engram: bool = False,
    experts: int | None = None,
    prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    """Every weight arg of ``window_layer_blocks(tp=tp)`` for ``layer``, rank ``rank``: from
    the checkpoint's tensors (``w``, TP1 tensors, sliced here as the official model splits them) and the raw
    FP8 wo_a (``raw``: ``layers.<i>.attn.wo_a.{weight,scale}``). ``local``: ``w`` / ``raw`` are
    already this rank's, as ATOM holds them (``hi_weights.block_tensors``: nothing is sliced, the shared expert is
    its rank slice), and ``moe`` the routed experts' packs (``hi_weights.expert_packs``). The knobs it reads
    (``MOE_LAYOUT``, ``SHARED``) are the live ones. ``experts`` (default ``EXPERTS``) / ``prefix`` (default
    ``layers.<layer>.``): the routed experts and the checkpoint prefix of ``raw``'s names, for the DSpark draft's
    stages (128, ``mtp.<s>.``)."""
    kn = schedules.current()
    experts_n = EXPERTS if experts is None else experts
    prefix = f"layers.{layer}." if prefix is None else prefix

    def part(t, dim, n, block=1):
        """Rank's 1 / tp of dim ``dim`` (``n``: its full size; ``block``: rows / cols a scale entry stands for)."""
        if local or tp == 1:
            return t
        size = n // tp // block
        return t.narrow(dim, rank * size, size)

    def mx(name, wt=None, sc=None, dim=None, n=None):
        wt = w[f"{name}.weight"] if wt is None else wt
        sc = w[f"{name}.scale"] if sc is None else sc
        if dim is not None:
            wt, sc = part(wt, dim, n), part(sc, dim, n, 32)
        m = fp8_block_to_mxfp8(wt.contiguous(), sc.contiguous())
        m.packed, m.codes = m.packed.to(device), m.codes.to(device)
        return m

    wo_a = (raw[f"{prefix}attn.wo_a.weight"], raw[f"{prefix}attn.wo_a.scale"])
    mats = {
        "wq_a": mx("attn.wq_a"), "wkv": mx("attn.wkv"), "wq_b": mx("attn.wq_b", dim=0, n=HEADS * HD),
        "wo_b": mx("attn.wo_b", dim=1, n=OGROUPS * ORANK),
        "wo_a": mx("", *wo_a, dim=0, n=OGROUPS * ORANK),
        "se_w1": mx("ffn.shared_experts.w1"), "se_w3": mx("ffn.shared_experts.w3"),
        "se_w2": mx("ffn.shared_experts.w2"),
    }  # fmt: skip
    lay = SimpleNamespace(
        linears=mats,
        retained={"q_norm": w["attn.q_norm.weight"].bfloat16().contiguous().to(device)},
    )
    args = {}
    for blk in window_layer_blocks(first=True, split=False, tp=tp, source=source):
        if isinstance(blk, Linear):
            args.update(blk.pack(lay))
        if isinstance(
            blk, (Compressor, Indexer)
        ):  # the source layer's own (replicated at TP)
            args.update({k: v.to(device) for k, v in blk.pack(w).items()})
    per_rank = experts_n // tp
    intra = tp > 1 and kn["MOE_LAYOUT"] == "intra"
    fold = kn["SHARED"] == "fold" or (kn["SHARED"] == "auto" and tp > 1)
    if (
        local
    ):  # (ATOM's experts: every expert's slice of the rank; its shared expert: the rank's slice)
        if moe is None or (tp > 1 and not (intra and fold)):
            raise ValueError("rank-local tensors pack with the routed experts' packs, at MOE_LAYOUT 'intra' with the "
                             "shared expert folded")  # fmt: skip
        args.update(moe)
    else:
        experts = [
            SimpleNamespace(**{n: SimpleNamespace(packed=w[f"ffn.experts.{e}.{n}.weight"].view(torch.uint8),
                                                  codes=w[f"ffn.experts.{e}.{n}.scale"].view(torch.uint8))
                               for n in ("w1", "w2", "w3")})
            for e in (range(experts_n) if intra else range(rank * per_rank, (rank + 1) * per_rank))
        ]  # fmt: skip
        packed = (
            pack_experts(experts, "moe", tp, rank, "intra")
            if intra
            else pack_experts(experts, "moe")
        )
        args.update({k: v.to(device) for k, v in packed.items()})
    if fold:  # rank's slice of the full shared expert
        sh = SimpleNamespace(w1=mats["se_w1"], w3=mats["se_w3"], w2=mats["se_w2"])
        args.update(
            {
                k: v.to(device)
                for k, v in pack_shared(sh, "moe", tp, rank, local=local).items()
            }
        )
    f32 = lambda t: t.float().contiguous().to(device)
    bf = lambda t: t.bfloat16().contiguous().to(device)
    args.update({
        "hca.fn": f32(w["hc_attn_fn"]), "hca.scale": f32(w["hc_attn_scale"]), "hca.base": f32(w["hc_attn_base"]),
        "hcf.fn": f32(w["hc_ffn_fn"]), "hcf.scale": f32(w["hc_ffn_scale"]), "hcf.base": f32(w["hc_ffn_base"]),
        "attn_norm.g": bf(w["attn_norm.weight"]), "ffn_norm.g": bf(w["ffn_norm.weight"]),
        "q_norm.g": bf(w["attn.q_norm.weight"]), "kv_norm.g": bf(w["attn.kv_norm.weight"]),
        "attn.sink": f32(part(w["attn.attn_sink"], 0, HEADS)), "router.w": bf(w["ffn.gate.weight"]),
        "router.bias": f32(w["ffn.gate.bias"]),
    })  # fmt: skip
    if (
        engram
    ):  # the Engram gate half's weight (``Engram.pack``'s): q_weight k_weight, f32 [4, 5120]
        args["eng.gw"] = (
            (w["engram.q_weight"].float() * w["engram.k_weight"].float())
            .contiguous()
            .to(device)
        )
    return args
