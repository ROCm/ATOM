# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A rank's share of V4.1's widths: every constant a kernel derives from the
deployment, in one place (a build key's ``tp`` and ``ep`` pick it).

TP: attention split by heads, the routed experts along their intermediate width,
over ``tp`` ranks. DP attention with expert parallel (``ep`` > 1): attention whole
on each rank (``tp`` 1), the routed experts split by expert over ``ep`` ranks at
their whole width. The shared expert, replicated there, is split along its
intermediate width over every rank of the MoE (``ranks``) either way."""

from dataclasses import dataclass

from atom.mono.plan.shard import Shard, Tiles, cdiv, padded

# the model's full widths
HIDDEN = 5120
HC = 4  # hyper-connection streams
Q_RANK, KV_DIM = 1280, 512  # the q latent, the window row
# a KV row in an FP8 pool (``kv_fp8``): E4M3 values, then an E8M0 scale a 32
KV_FP8_ROW_BYTES = KV_DIM + KV_DIM // 32
Q_HEADS = 64  # attention query heads
O_GROUPS = 8  # wo_a groups
O_RANK = 1024  # wo_a output rows a group
HEAD_DIM = 512
MOE_INTER = 2304  # a routed expert's intermediate width
SHARED_INTER = 2304  # the shared expert's
# the loader pads a routed expert's rank share to a multiple of this (``FusedMoE``
# ``pad_align``) with zero weights
MOE_PAD = 128
ROWS = 16  # GEMV rows a task
WOA_ROWS = 32  # a wo_a task's rows: one FP8 group of y
UG_PART = 32  # intermediate columns an ug task part: one FP4 group
TOPK = 6  # routed experts a token (the target's; the DSpark draft's fewer)
# CTA b polls route copy b % ROUTE_COPIES: every CTA on one small region
# serializes on its lines (K2b S=24 -0.5 us)
ROUTE_COPIES = 8
HEAD_TILE = 16  # heads an attention MFMA takes (its N)


@dataclass(frozen=True)
class Dims:
    tp: int
    # expert-parallel ranks: 1 under TP
    ep: int = 1

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(Q_HEADS, "query heads")
        shard.split(O_GROUPS, "wo_a groups")
        shard.split(MOE_INTER, "routed intermediate")
        Shard(self.ranks).split(SHARED_INTER, "shared intermediate")
        if self.ep > 1 and self.tp != 1:
            raise ValueError(f"expert parallel {self.ep} at attention TP {self.tp}")

    @property
    def ranks(self) -> int:
        """The MoE's ranks: its all-reduce's (or reduce-scatter's) group."""
        return self.tp * self.ep

    @property
    def heads(self) -> int:
        return Q_HEADS // self.tp

    @property
    def head_tiles(self) -> Tiles:
        """The rank's heads in attention MFMA tiles."""
        return Tiles(self.heads, HEAD_TILE)

    @property
    def groups(self) -> int:
        """The rank's wo_a groups."""
        return O_GROUPS // self.tp

    @property
    def group_k(self) -> int:
        """wo_a's K a group: its heads' attention outputs."""
        return self.heads // self.groups * HEAD_DIM

    @property
    def o_rows(self) -> int:
        """wo_a's output rows: the rank's groups x o_rank."""
        return self.groups * O_RANK

    @property
    def woa_tasks(self) -> int:
        return self.o_rows // WOA_ROWS

    @property
    def wqb_tasks(self) -> int:
        """K1's wq_b tasks: the rank's heads' query rows."""
        return self.heads * HEAD_DIM // ROWS

    @property
    def xo_words(self) -> int:
        """FP8 words of a token's attention output."""
        return self.heads * HEAD_DIM // 4

    @property
    def yb_words(self) -> int:
        """FP8 words of a token's wo_a output (wo_b's input)."""
        return self.o_rows // 4

    @property
    def wob_parts(self) -> int:
        """wo_b's K in parts of at most a wo_a group's K (the LDS row ``xl``
        holds): a part's partial pushed as a source of its own, summed with the
        ranks' in the all-reduce's order -- at TP 1 the two halves TP 2's ranks
        hold, summed as they are."""
        return cdiv(self.o_rows, self.group_k)

    @property
    def wob_k(self) -> int:
        """A wo_b part's K."""
        return self.o_rows // self.wob_parts

    @property
    def attn_sources(self) -> int:
        """The attention all-reduce's partials: a rank's parts each."""
        return self.tp * self.wob_parts

    @property
    def inter_real(self) -> int:
        """A routed expert's real intermediate width on this rank."""
        return MOE_INTER // self.tp

    @property
    def inter(self) -> int:
        """... padded as the loader pads it (the rest zero weights)."""
        return padded(self.inter_real, MOE_PAD)

    @property
    def sh_inter(self) -> int:
        """The shared expert's intermediate width on this rank."""
        return SHARED_INTER // self.ranks

    @property
    def shared_tasks(self) -> int:
        return self.sh_inter // ROWS

    @property
    def mid_words(self) -> int:
        """MXFP8 words of a pick's routed intermediate."""
        return self.inter // 4

    @property
    def down_scale_cols(self) -> int:
        """w2_s's e8m0 columns: inter / 32, padded to a multiple of 8
        (aiter ``shuffle_scale``)."""
        return padded(self.inter // 32, 8)
