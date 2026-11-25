# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The V4.1 mono kernels' hand-off contracts, checked on a build.

Each kernel's mailbox regions (``scratch_layout`` of its module, the peer
regions) with the stages that put them; tracing a build (compile-only, no
launch, no GPU needed with ``ARCH`` set) records its accesses
(``atom.mono.plan.trace``) and ``atom.mono.plan.check`` holds them to the table.
FlyDSL traces a build before it looks it up, so a build in its disk cache is
traced too.

K1, K2a and K2b of a layer share the scratch: their regions of one name are
told apart by the tag ranges their launches take (``runner.POST_TAG0`` /
``MOE_TAG0``), not by address.

Usage: python -m atom.models.deepseek_v41.mono.contract [S ...]
"""

import sys

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.config import MAX_MOE_ROWS, SUPPORTED_DPA
from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import compress as cp
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels.moe import down_by_unit
from atom.models.deepseek_v41.mono.kernels.moe_shape import MoeBuild, route_shape
from atom.mono.plan.check import RegionDecl
from atom.mono.plan.trace import Space
from atom.mono.runtime.compile import check_traced

S, P = Space.SCRATCH, Space.PEER


def seam_regions(part: str, shard: bool, gate: str) -> list[RegionDecl]:
    """Part ``part``'s seam: the slice's LIN, its squares and the mixes'
    partials; column-sharded (``attn_pre.shard_seam``), LIN and its squares
    into every rank's gathered copy, and the gate blocks' sums of the partials
    into every rank's ``gate`` region (seam A's PMIXG, seam F's PMIXGF)."""
    slice_ = f"{part}.slice"
    regions = [RegionDecl("pmix", S, slice_)]
    if shard:
        return regions + [
            RegionDecl("ling", P, slice_),
            RegionDecl("lssqg", P, slice_),
            RegionDecl(gate, P, f"{part}.gate_block"),
        ]
    return regions + [RegionDecl("lin", S, slice_), RegionDecl("lssq", S, slice_)]


def k1_regions(key: k1.AttnPreBuild) -> list[RegionDecl]:
    """K1's: stages slice, gate, norm, wqkv_a, qkv, wq_b, then (a selecting
    layer's plain norm) index_w, index_quant."""
    regions = [
        *seam_regions("k1", key.shard, "pmixg"),
        RegionDecl("x8", S, "k1.norm"),
        RegionDecl("x8s", S, "k1.norm"),
        RegionDecl("qkv", S, "k1.wqkv_a"),
        # X8 goes to wqkv_a by a flag a token (``attn_pre.put_x8``)
        RegionDecl("x8f", S, "k1.norm"),
    ]
    if key.tok_par:
        # every rank's q latent and window rows, gathered in the peer buffer
        regions += [
            RegionDecl("qg", P, "k1.qkv"),
            RegionDecl("qgs", P, "k1.qkv"),
            RegionDecl("kg", P, "k1.qkv"),
        ]
    else:
        regions += [RegionDecl("qx8", S, "k1.qkv"), RegionDecl("qx8s", S, "k1.qkv")]
    if key.feeds_index:
        regions += [RegionDecl("nb", S, "k1.norm"), RegionDecl("iq", S, "k1.wq_b")]
        if key.tok_par:
            regions += [
                RegionDecl("nbg", P, "k1.norm"),
                RegionDecl("iwg", P, "k1.index_w"),
            ]
        else:
            regions += [RegionDecl("iw", S, "k1.index_w")]
    return regions


def indexer_regions(
    bound_max: int, tok_par: bool, split: bool = False
) -> list[RegionDecl]:
    """The indexer's: the scores, then per kind (sel, blk) a stage a tree level,
    each level but the last putting its lists for the next. A token's selection
    is final at the level its bound reaches (``index_plan.final_level``), so any
    sel level may put its ready flag. A candidate producer's blocks put ICRDY,
    awaited by a compacted row's selection (``index_plan.compacts``), so its
    blk stages run before its sel ones. In a token-parallel launch a row's
    owner puts its flags on every rank, ICRDY too (also awaited by
    ``k2a.icwait``). ``split``: a split row's final list puts its answer's flag
    on every rank (IXRDY, awaited by ``k2a.imerge``), which then puts IRDY."""

    def finals(kind):
        return tuple(
            f"k2a.i{kind}{lv}" for lv in range(ip.region_levels(kind, bound_max) + 1)
        )

    regions = [RegionDecl("isf", S, "k2a.iscore")]
    if tok_par:
        regions += [
            RegionDecl("irdy", P, finals("sel")),
            RegionDecl("icrdy", P, finals("blk")),
        ]
    elif split:
        regions += [
            RegionDecl("irdy", S, (*finals("sel"), "k2a.imerge")),
            RegionDecl("ixrdy", P, finals("sel")),
            RegionDecl("icrdy", S, finals("blk")),
        ]
    else:
        regions += [
            RegionDecl("irdy", S, finals("sel")),
            RegionDecl("icrdy", S, finals("blk")),
        ]
    for kind in ip.KINDS:
        regions += [
            RegionDecl(f"i{kind}{lv}", S, f"k2a.i{kind}{lv}")
            for lv in range(ip.region_levels(kind, bound_max))
        ]
    return regions


def k2a_regions(key: k2a.AttnPostBuild, shard: bool = False) -> list[RegionDecl]:
    """K2a's: (a selecting layer) the indexer, then attn_score, irq, wo_a,
    wo_b (pushing the all-reduce's partials), slice, gate, norm; ``shard``:
    its seam column-sharded (``seam_regions``)."""
    regions = (
        indexer_regions(key.index_bound_max, key.tok_par, key.index_split)
        if key.index_bound_max
        else []
    )
    return regions + [
        RegionDecl("as", S, "k2a.attn_score"),
        RegionDecl("xo", S, "k2a.irq"),
        RegionDecl("xos", S, "k2a.irq"),
        RegionDecl("x8b", S, "k2a.wo_a"),
        RegionDecl("x8bs", S, "k2a.wo_a"),
        RegionDecl("attn", P, "k2a.wo_b"),
        *seam_regions("k2a", shard, "pmixgf"),
    ]


def k2b_regions(key: MoeBuild) -> list[RegionDecl]:
    """K2b's: stages xq (its MXFP8 rows plain words, a token's X8RDY announcing
    them), router, route, shared (an odd task polls the even task's
    intermediate: an exchange), ug, down (pushing the all-reduce's partials),
    reduce. Down by queue units (``RouteShape.down_y``): ug's MID as tagged pairs in
    place of its flags, and down's units putting DY / SDO that its combine
    polls (an exchange). Down by unit (``moe.down_by_unit``): ug flags a CTA
    done (UGC) and every down unit a CTA's (DNF), the combine waiting on all."""
    if route_shape(key).down_y:
        handoff = [
            RegionDecl("mid", S, "k2b.ug"),
            RegionDecl("mids", S, "k2b.ug"),
            RegionDecl("dy", S, "k2b.down", exchange=True),
            RegionDecl("sdo", S, "k2b.down", exchange=True),
        ]
    elif down_by_unit(route_shape(key)):
        handoff = [
            RegionDecl("ugc", S, "k2b.ug"),
            RegionDecl("dnf", S, "k2b.down", exchange=True),
        ]
    else:
        handoff = [RegionDecl("ugf", S, "k2b.ug")]
    # a rank's routes of its own rows to every rank's RT (``routes_pushed``)
    if route_shape(key).routes_pushed:
        routes = RegionDecl("rt", P, "k2b.route")
    else:
        routes = RegionDecl("route", S, "k2b.route")
    return [
        RegionDecl("x8rdy", S, "k2b.xq"),
        RegionDecl("logit", S, "k2b.router"),
        routes,
        RegionDecl("smid", S, "k2b.shared", exchange=True),
        RegionDecl("smq", S, "k2b.shared"),
        RegionDecl("smc", S, "k2b.shared"),
        *handoff,
        RegionDecl("ffn", P, "k2b.down"),
    ]


def k2_regions(key: k2.LayerPostBuild) -> list[RegionDecl]:
    """K2a's and K2b's in one launch, and K2b's wait for a token's normed row."""
    key_a, key_b = key.halves()
    if key.tok_par:
        # a token's flag here from its owner's norm or the row's gather;
        # the owner pushes its row to every other rank's XG
        gathered = [
            RegionDecl("xrdy", S, ("k2a.norm", "k2b.gather")),
            RegionDecl("xg", P, "k2a.norm"),
        ]
    elif key.dp_ranks > 1:
        # DP attention: each rank's norm pushes its own step's rows to every
        # rank's XG, the MoE's rows of all of them
        gathered = [
            RegionDecl("xrdy", S, "k2a.norm"),
            RegionDecl("xg", P, "k2a.norm"),
        ]
    else:
        gathered = [RegionDecl("xrdy", S, "k2a.norm")]
    return [*k2a_regions(key_a, key.shard), *k2b_regions(key_b), *gathered]


def compress_regions() -> list[RegionDecl]:
    """The compressor's (``kernels.compress``): the projection CTAs' flags,
    the slices' sums of squares (every slice of a row polls its row's) and
    their key shares (the row's first slice adds them)."""
    return [
        RegionDecl("cpf", S, "cp.project"),
        RegionDecl("css", S, "cp.pool", exchange=True),
        RegionDecl("ckp", S, "cp.pool"),
    ]


def check_compress(key: cp.CompressBuild) -> None:
    check_traced(cp.build_compress(key), cp.ABI, compress_regions())


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
        for bound, fp4, mfma in (
            (0, False, False),
            (65536, False, False),
            (65536, True, False),
            (65536, True, True),
        ):
            check_k2(k2.LayerPostBuild(
                tokens=s, index_bound_max=bound, index_fp4=fp4, index_mfma=mfma
            ))  # fmt: skip
        # an FP8 pool: the window row's bytes, the attention's dequant
        check_k1(k1.AttnPreBuild(tokens=s, kv_fp8=True))
        # the residual column-sharded: an indexer layer's K1 and K2
        check_k1(k1.AttnPreBuild(
            tokens=s, feeds_index=True, index_fp4=True, shard=True
        ))  # fmt: skip
        check_k2(k2.LayerPostBuild(
            tokens=s, index_bound_max=65536, index_fp4=True, shard=True
        ))  # fmt: skip
        check_k2(k2.LayerPostBuild(
            tokens=s, index_bound_max=65536, index_fp4=True, kv_fp8=True
        ))  # fmt: skip
        if s % 4 == 0 and s >= 12:
            for feeds_index in (False, True):
                check_k1(k1.AttnPreBuild(
                    tokens=s, feeds_index=feeds_index, index_fp4=feeds_index,
                    tok_par=True,
                ))  # fmt: skip
            check_k1(k1.AttnPreBuild(tokens=s, tok_par=True, kv_fp8=True))
            check_k2(k2.LayerPostBuild(
                tokens=s, index_bound_max=65536, index_fp4=True, tok_par=True
            ))  # fmt: skip
        # DP attention: each rank's attention whole (TP 1), the MoE over the
        # DP ranks' rows, expert parallel or TP
        for dp in SUPPORTED_DPA:
            if s * dp <= MAX_MOE_ROWS:
                check_k1(k1.AttnPreBuild(tokens=s, tp=1, kv_fp8=True))
                for ep, moe_tp in ((dp, 1), (1, dp)):
                    check_k2(k2.LayerPostBuild(
                        tokens=s, tp=1, ep=ep, moe_tp=moe_tp, index_bound_max=65536,
                        index_fp4=True, kv_fp8=True,
                    ))  # fmt: skip
        for ratio in (1, 2):
            for kv_fp8 in (False, True):
                check_compress(cp.CompressBuild(
                    rows=s, ratio=ratio, eps=1e-6, k_eps=1e-6, kv_fp8=kv_fp8
                ))  # fmt: skip
        print(f"S={s}: contract holds")
