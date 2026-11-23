# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A rank's share of V4.1's widths: every constant a kernel derives from the
deployment, in one place (a build key's ``tp`` and ``ep`` pick it).

TP: attention split by heads, the routed experts along their intermediate width,
over ``tp`` ranks. Expert parallel (``ep`` > 1): the routed experts split by
expert over ``ep`` ranks at their whole width -- the attention's TP ranks
(``tp`` = ``ep``), or DP attention's (``tp`` 1, attention whole on each rank).
The shared expert (replicated under DP attention) is split along its
intermediate width over every rank of the MoE (``ranks``) either way."""

from dataclasses import dataclass

from atom.mono.plan.shard import Shard, Tiles, cdiv, padded

# the model's full widths
HIDDEN = 5120
HC = 4  # hyper-connection streams
Q_RANK, KV_DIM = 1280, 512  # the q latent, the window row
# a KV row in an FP8 pool (``kv_fp8``): its E4M3 values in the data plane, its
# E8M0 scales (one a 32 columns) in the scale plane
KV_FP8_ROW_BYTES = KV_DIM + KV_DIM // 32


# ``kvrow``'s row kinds a step row: its window row, then (a compressing layer)
# the compressed entry it completes
KVROW_KINDS = 2


def kvrow_bytes(rows: int, kv_fp8: bool) -> int:
    """The layer kernel's ``kvrow`` for a step of ``rows`` rows, laid out as the
    pool's planes: row j < rows the step's window row j, row rows + j the
    compressed entry step row j completes (a compressing layer); BF16 rows,
    or an FP8 pool's E4M3 rows and then, from ``kvrow_scale0``, their scale
    rows."""
    return KVROW_KINDS * rows * (KV_FP8_ROW_BYTES if kv_fp8 else 2 * KV_DIM)


def kvrow_scale0(rows: int) -> int:
    """An FP8 ``kvrow``'s scale plane (``kvrow_bytes``): its byte offset."""
    return KVROW_KINDS * rows * KV_DIM


Q_HEADS = 64  # attention query heads
O_GROUPS = 8  # wo_a groups
O_RANK = 1024  # wo_a output rows a group
HEAD_DIM = 512
MOE_INTER = 2304  # a routed expert's intermediate width
SHARED_INTER = 2304  # the shared expert's
# the loader pads a routed expert's rank share to a multiple of this (``FusedMoE``
# ``pad_align``) with zero weights
MOE_PAD = 128
# the loader pads the shared expert's rank share (split over TP) to a multiple of
# this, zero rows of gate_up and columns of w2 (``linear.fp8_group32_k_pad_align``,
# gfx950's group32 FP8 GEMM: TP8's 288 -> 320). Must equal
# ``linear.GROUP32_FP8_K_ALIGN`` (this module stays free of AITER;
# tests/test_fp8_group32_k_pad_pair.py holds the two equal).
# TODO(ROCm/aiter#7967): temporary, as that constant; it goes once the GEMM takes
# a 32-wide K tail
SHARED_PAD = 64
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
    # expert-parallel ranks, the routed experts split over them (ATOM's EP
    # group: every TP and DP rank); 1 under TP
    ep: int = 1
    # DP attention with the MoE TP over its ranks (``tp`` of them): the shared
    # expert replicated, as under expert parallel at attention TP 1
    dp_attention_tp: bool = False

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(Q_HEADS, "query heads")
        shard.split(O_GROUPS, "wo_a groups")
        if self.ep > 1 and self.tp not in (1, self.ep):
            raise ValueError(f"expert parallel {self.ep} at attention TP {self.tp}")
        Shard(self.ranks // self.ep).split(MOE_INTER, "routed intermediate")
        Shard(self.ranks).split(SHARED_INTER, "shared intermediate")

    @property
    def dp_attention(self) -> bool:
        """The MoE over DP attention's ranks, each rank's own rows (else over
        the attention's TP ranks, every rank's rows the same): its TP over
        them, or expert parallel at attention TP 1."""
        return self.dp_attention_tp or (self.ep > 1 and self.tp == 1)

    @property
    def ranks(self) -> int:
        """The MoE's ranks: its all-reduce's (or reduce-scatter's) group --
        the attention's TP ranks, or DP attention's."""
        return max(self.tp, self.ep)

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
        """A routed expert's real intermediate width on this rank: whole
        under expert parallel."""
        return MOE_INTER // (self.ranks // self.ep)

    @property
    def inter(self) -> int:
        """... padded as the loader pads it (the rest zero weights)."""
        return padded(self.inter_real, MOE_PAD)

    @property
    def sh_inter(self) -> int:
        """The shared expert's intermediate width on this rank: under TP
        attention its share padded as the loader pads it (the rest zero
        weights); under DP attention its part of the replicated weight."""
        share = SHARED_INTER // self.ranks
        return share if self.dp_attention else padded(share, SHARED_PAD)

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
