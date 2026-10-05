# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono kernels' hand-off contracts, checked on a build.

Each kernel's mailbox regions (``scratch_layout`` of its module, the peer
regions) with the stages that put them; tracing a build (compile-only, no
launch, no GPU needed with ``ARCH`` set) records its accesses
(``atom.mono.plan.trace``) and ``atom.mono.plan.check`` holds them to the table.
A build served from FlyDSL's disk cache is not traced, so the caller disables
the cache (``FLYDSL_RUNTIME_ENABLE_CACHE=0``).

K1, K2a and K2b of a layer share the scratch: their regions of one name are
told apart by the tag ranges their launches take (``runner.POST_TAG0`` /
``MOE_TAG0``), not by address.

Usage: python -m atom.models.deepseek_v41.mono.contract [S ...]
"""

import sys

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels.moe_shape import MoeBuild
from atom.mono.plan.check import RegionDecl
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import check_traced

S, P = Space.SCRATCH, Space.PEER


def k1_regions(key: k1.AttnPreBuild) -> list[RegionDecl]:
    """K1's: stages slice, gate, norm, wqkv_a, qkv, wq_b, then (a selecting
    layer's plain norm) index_w, index_quant."""
    regions = [
        RegionDecl("lin", S, "k1.slice"),
        RegionDecl("pmix", S, "k1.slice"),
        RegionDecl("x8", S, "k1.norm"),
        RegionDecl("x8s", S, "k1.norm"),
        RegionDecl("qkv", S, "k1.wqkv_a"),
        RegionDecl("qx8", S, "k1.qkv"),
        RegionDecl("qx8s", S, "k1.qkv"),
    ]
    if key.feeds_index:
        regions += [
            RegionDecl("nb", S, "k1.norm"),
            RegionDecl("iq", S, "k1.wq_b"),
            RegionDecl("iw", S, "k1.index_w"),
        ]
    return regions


def indexer_regions(bound_max: int) -> list[RegionDecl]:
    """The indexer's: the scores, then per kind (sel, blk) a stage a tree level,
    each level but the last putting its lists for the next. A token's selection
    is final at the level its bound reaches (``index_plan.final_level``), so any
    sel level may put its ready flag."""
    sel_levels = range(ip.top_level("sel", bound_max) + 1)
    regions = [
        RegionDecl("isf", S, "k2a.iscore"),
        RegionDecl("irdy", S, tuple(f"k2a.isel{lv}" for lv in sel_levels)),
    ]
    for kind in ip.KINDS:
        regions += [
            RegionDecl(f"i{kind}{lv}", S, f"k2a.i{kind}{lv}")
            for lv in range(ip.top_level(kind, bound_max))
        ]
    return regions


def k2a_regions(key: k2a.AttnPostBuild) -> list[RegionDecl]:
    """K2a's: (a selecting layer) the indexer, then attn_score, irq, wo_a,
    wo_b (pushing the all-reduce's partials), slice, gate, norm."""
    regions = indexer_regions(key.index_bound_max) if key.index_bound_max else []
    return regions + [
        RegionDecl("am", S, "k2a.attn_score"),
        RegionDecl("al", S, "k2a.attn_score"),
        RegionDecl("ap", S, "k2a.attn_score"),
        RegionDecl("xo", S, "k2a.irq"),
        RegionDecl("xos", S, "k2a.irq"),
        RegionDecl("x8b", S, "k2a.wo_a"),
        RegionDecl("x8bs", S, "k2a.wo_a"),
        RegionDecl("attn", P, "k2a.wo_b"),
        RegionDecl("lin", S, "k2a.slice"),
        RegionDecl("pmix", S, "k2a.slice"),
    ]


def k2b_regions(key: MoeBuild) -> list[RegionDecl]:
    """K2b's: stages xq (its MXFP8 rows plain words, a token's X8RDY announcing
    them), router, route, shared (an odd task polls the even task's
    intermediate: an exchange), ug, down (pushing the all-reduce's partials),
    reduce."""
    return [
        RegionDecl("x8rdy", S, "k2b.xq"),
        RegionDecl("logit", S, "k2b.router"),
        RegionDecl("route", S, "k2b.route"),
        RegionDecl("smid", S, "k2b.shared", exchange=True),
        RegionDecl("smq", S, "k2b.shared"),
        RegionDecl("smc", S, "k2b.shared"),
        RegionDecl("ugf", S, "k2b.ug"),
        RegionDecl("ffn", P, "k2b.down"),
    ]


def k2_regions(key: k2.LayerPostBuild) -> list[RegionDecl]:
    """K2a's and K2b's in one launch, and K2b's wait for a token's normed row."""
    key_a, key_b = key.halves()
    return [
        *k2a_regions(key_a),
        *k2b_regions(key_b),
        RegionDecl("xrdy", S, "k2a.norm"),
    ]


def check_k1(key: k1.AttnPreBuild) -> None:
    """Trace K1 for ``key`` and check its hand-offs; raises ``ContractError``."""
    check_traced(k1.build_attn_pre(key), k1.ABI, k1_regions(key))


def check_k2(key: k2.LayerPostBuild) -> None:
    check_traced(k2.build_layer_post(key), k2.ABI, k2_regions(key))


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [6, 12, 24, 48]
    for s in sizes:
        for feeds_index, fp4 in ((False, False), (True, False), (True, True)):
            check_k1(k1.AttnPreBuild(tokens=s, feeds_index=feeds_index, index_fp4=fp4))
        for bound, fp4 in ((0, False), (65536, False), (65536, True)):
            check_k2(k2.LayerPostBuild(tokens=s, index_bound_max=bound, index_fp4=fp4))
        print(f"S={s}: contract holds")
