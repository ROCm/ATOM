# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone indexer build keys and buffer geometry, without GPU imports."""

from dataclasses import dataclass, replace

from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import selection_plan as isp
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.abi import KernelAbi

MAX_ROWS = 64


@dataclass(frozen=True)
class IndexerBuild:
    tokens: int
    bound_max: int
    fp4: bool = True
    mfma: bool = True
    diag_off: int = -1
    world: int = 1
    peer_offset: int = 0
    peer_plain: int = 0
    full_only: bool = False
    hidden_size: int = 5120
    query_rank: int = 1280

    def __post_init__(self):
        if self.hidden_size <= 0 or self.hidden_size % 8:
            raise ValueError(
                "paged indexer hidden size must be a positive multiple of 8"
            )
        if self.query_rank <= 0 or self.query_rank % 128:
            raise ValueError(
                "paged indexer query rank must be a positive multiple of 128"
            )


ABI = KernelAbi((
    "hidden", "qr", "qr_scale", "iq_w", "iq_ws", "iw_w", "pos", "cos", "sin",
    "iq", "iqs", "iw", "plane", "pscale", "iplan", "itab", "itab_stride",
    "itab_len", "ibatch", "ishift", "ibound", "ilog", "lstride", "isel",
    "icand", "ilift", "iprod", "ibmax", "bstride", "icout", "irow",
    "kmeta", "table", "rows_per_page", "page_rows", "main_off", "scratch", "layer",
    "peers", "rank", "ix_meta",
    "step_state", "step_fence",
))  # fmt: skip


def _select_segment(key):
    if key.full_only:
        return min(8192, ip.cdiv(key.bound_max, 2048) * 2048)
    return isp.row_segment(key.bound_max) if key.fp4 else ip.SEG


def scratch_layout(key: IndexerBuild):
    segment = _select_segment(key)
    lists = {"isf": key.tokens * ip.chunks(key.bound_max), "irdy": key.tokens}
    if key.world > 1:
        lists["ix_root_ready"] = key.tokens * (1 if key.full_only else len(tp.KINDS))
        lists["ix_fenced"] = 1
        if not key.full_only:
            lists["ix_rowhist"] = tp.row_hist_words(key.tokens) // 2
            lists["ix_segment_ready"] = key.tokens * len(tp.KINDS) * BLOCKS * 2
            lists["ix_emitted_ready"] = key.tokens * len(tp.KINDS) * BLOCKS
            lists["ix_phase_ready"] = key.tokens * len(tp.KINDS) * 4 * 3
    for kind in (("sel",) if key.full_only else ip.KINDS):
        tree_segment = (
            (8192 if kind == "blk" else 4096)
            if key.world > 1
            else isp.segment_for(segment, kind)
        )
        for level in range(ip.top_level(kind, key.bound_max, seg=tree_segment)):
            slots = ip.list_slots(kind, level, key.bound_max, seg=tree_segment)
            lists[f"i{kind}{level}"] = key.tokens * slots * ip.KINDS[kind] * 2
            lists[f"i{kind}ready{level}"] = key.tokens * slots
    return pair_layout(
        {
            "iq": key.tokens * ip.HEADS * ip.DIM // 2,
            "iw": key.tokens * ip.HEADS,
            "qdone": BLOCKS,
            "qready": key.tokens * ip.HEADS * (2 if key.tokens > 32 else 4),
            "iwready": key.tokens * ip.HEADS if _split_weights(key) else 1,
            **lists,
        }.items()
    )


def _split_weights(key):
    # Three 16-row query tiles use 192 CTAs. The remaining 64 can compute
    # two head-weight tiles while the query projections are still running.
    return key.fp4 and 32 < key.tokens <= 48


def scratch_bytes(key: IndexerBuild):
    return max(o + n for o, n in scratch_layout(key).values())


def reindex_build(key: IndexerBuild):
    """A local binary for the at-most-16K compressed REINDEX key domain.

    Keep the existing 18K leaf above the 16K boundary. A separate binary
    avoids carrying the TP merge's register footprint through this local path.
    """
    return replace(
        key,
        bound_max=min(key.bound_max, 18432),
        world=1,
        peer_offset=0,
        peer_plain=0,
    )


def full_build(key: IndexerBuild, ratio: int):
    """A Top512-only binary bounded by the FULL layers' compression ratio.

    The caller uses the smallest ratio among its non-producing FULL layers.
    Keep the original TP world and peer offsets so the scorer's walk plan and
    the first layer's epoch fence agree with the normal request planner.
    """
    if not key.fp4 or ratio not in (1, 2):
        raise ValueError("FULL indexer requires FP4 and compression ratio 1 or 2")
    return replace(
        key,
        bound_max=ip.cdiv(key.bound_max, ratio * ip.CHUNK) * ip.CHUNK,
        full_only=True,
    )


def selection_keys(key):
    """Selection storage also stages all six FP4 score rows and their bounds."""
    keys = _select_segment(key)
    if key.fp4:
        row_words = ip.HEADS * ip.DIM // 8 + ip.HEADS * 2 + ip.HEADS
        keys = max(keys, ip.WALK_ROWS * (row_words + 1))
    if key.world > 1:
        keys = max(keys, key.world * (ip.TOPK if key.full_only else ip.TOPK_BLOCKS))
    return keys
