# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSA build admission and buffer geometry, importable without FlyDSL."""

from dataclasses import dataclass, field

from atom.mono.dsa_indexer import selection_plan as isel
from atom.mono.dsa_indexer.plan import (
    REPLICA_INTERLEAVE,
    TILE,
    IndexerShape,
    local_len,
    unsupported,
)
from atom.mono.plan.build_key import kernel_symbol
from atom.mono.plan.execution import BLOCKS

MAX_LEGACY_TOKENS = 64
MAX_TOKENS = BLOCKS
MAX_VERIFY_TOKENS = BLOCKS
TL_PASS0 = 7
# ``IxBuild.dc1`` bits
DC1_ACTIVE_POLL = 1
DC1_PACKED_HIST = 2
DC1_FLAT_STORE = 16
DC1_XPOSE_SUM = 32
DC1_EARLY_CNT = 64
DC1_BUCKET_EXIT = 128
DC1_EARLY_W = 1024
DC1_JOINT_POLL = 2048
DC1_ALL = (
    DC1_ACTIVE_POLL
    | DC1_PACKED_HIST
    | DC1_FLAT_STORE
    | DC1_XPOSE_SUM
    | DC1_EARLY_CNT
    | DC1_BUCKET_EXIT
    | DC1_EARLY_W
    | DC1_JOINT_POLL
)
# The options every binding builds with (``bind.bind``): all of them. Chosen on
# the production DCP8 build (Kimi-K3.1, 128K, K4, 8 x MI355X), every option
# bitwise equal to the kernel without them (indices, indptr, counts and logits,
# 5..60 rows, equal-logit rows): without / with all but ``DC1_PACKED_HIST``
# (3313) / with all (3315), us a launch: 34.96 / 30.88 / 31.81 at 5 rows,
# 41.77 / 38.04 / 37.31 at 20, 62.95 / 53.79 / 50.47 at 60. Packing the
# histograms pays from ``PACKED_HIST_MIN_ROWS`` rows, which ``bind`` applies.
DC1_DEFAULT = DC1_ALL
# ``packed_hist`` from this many rows a step: with fewer, packing the pushes
# costs more than their smaller size saves
PACKED_HIST_MIN_ROWS = 20


def local_step_symbol(rows, qlen, world, compact, debug=False):
    """Name every local-reset codegen setting on CPU and in the launcher."""
    prefix, flag_stride, suffix, total = compact
    return kernel_symbol(
        "mono_local_step",
        s=rows,
        q=qlen,
        w=world,
        debug=int(debug),
        prefix=prefix,
        flag_stride=flag_stride,
        suffix=suffix,
        total=total,
    )


@dataclass(frozen=True)
class IxBuild:
    # the layer's shape: its ``tag`` is the kernel symbol's stem (``name``)
    shape: IndexerShape
    # ``sym``: the field's short name in the kernel symbol (``symbol_params``)
    tokens: int = field(metadata={"sym": "s"})  # S rows, a query token each
    world: int = field(metadata={"sym": "w"})  # the ranks splitting a row's keys
    # a row's keys at most: max_model_len
    max_ctx: int = field(metadata={"sym": "c"})
    # every rank stores every key (no DCP): the ranks score 128-key runs each
    # (``plan.REPLICA_INTERLEAVE``) and every rank writes the whole selection
    replicated: bool = field(default=False, metadata={"sym": "rep"})
    # per-CTA stamps (``timeline_points``) to plain "tl"
    timeline: bool = field(default=False, metadata={"sym": "t"})
    # query rows a request: 1 a plain decode, a speculative verify's otherwise
    # (row i of a request sees its context less the qlen - 1 - i drafts past it)
    qlen: int = field(default=1, metadata={"sym": "q"})
    # harness only (``dsa_ix_bench.py`` SHARE_K2): this rank waits
    # ``delay_ticks`` before the last pass's decide, holding open the window
    # in which a faster rank's next kernel could overwrite pairs it still reads
    delay_rank: int = field(default=-1, metadata={"sym": "dr"})
    delay_ticks: int = field(default=0, metadata={"sym": "dt"})
    # 512 requires two resident CTAs per gfx950 CU. Wide buckets split Q too.
    score_blocks: int = field(default=BLOCKS, metadata={"sym": "sb"})
    # gfx950 short verify: BF16 prepacked Q projection and local MQA scheduling.
    short_gemm: bool = field(default=False, metadata={"sym": "sq"})
    # DC1's bit-identical changes (``DC1_*``), a mask; 0 names no symbol field
    dc1: int = field(default=0, metadata={"sym": "dc"})

    @property
    def stream_scores(self) -> bool:
        return (
            (64 <= self.tokens <= 128 or self.split_q)
            and self.world > 1
            and not self.replicated
            and self.qlen == 1
        )

    @property
    def split_q(self) -> bool:
        return self.score_blocks > BLOCKS and (
            self.tokens > 128 or (self.tokens == 128 and self.shape.q_lora == 2048)
        )

    @property
    def reuse_q_scratch(self) -> bool:
        return self.score_blocks > BLOCKS and self.shape.q_lora == 2048

    @property
    def radix(self) -> isel.Radix:
        if self.split_q:
            if self.max_ctx <= 16384:
                return isel.SHORT_WIDE_RADIX
            if self.tokens == 128 and self.shape.q_lora == 2048:
                return isel.DEFAULT_RADIX
            return isel.WIDE_RADIX
        return isel.DEFAULT_RADIX

    @property
    def timeline_points(self) -> int:
        return TL_PASS0 + 3 * self.radix.passes + 4

    @property
    def interleave(self) -> int:
        """A rank's runs of global keys (``plan.key_pos``)."""
        return REPLICA_INTERLEAVE if self.replicated else 1

    @property
    def short_owned_verify(self) -> bool:
        """Allow runtime short-verify ownership and rank-local Q projection."""
        return (
            self.replicated
            and self.world > 1
            and self.qlen in (6, 8)
            and 0 < self.max_ctx <= isel.SPLIT_MIN
            and 0
            < self.tokens
            <= (MAX_VERIFY_TOKENS if self.qlen == 6 else MAX_LEGACY_TOKENS)
            and self.tokens % self.qlen == 0
            and (self.qlen == 6 or self.tokens // self.qlen <= self.world)
        )

    @property
    def own_rows(self) -> bool:
        """Replicated with at least one request a rank: each request's rows
        scored and selected whole by one rank (``select.owner``), the other
        ranks reading its winners from its slab -- no cross-rank pass (V4.1's
        rows split over ranks). A request its owner would score alone while the others idle still has its keys
        split over the ranks (``select.split_keys``, decided a step). Fewer
        requests than ranks normally split each row's keys; short verify
        requests stay owned to avoid histogram and winner exchanges."""
        return (
            self.replicated
            and self.world > 1
            and (self.tokens // self.qlen >= self.world or self.short_owned_verify)
        )

    @property
    def lmax(self) -> int:
        """A rank's share of a row's keys at most (``plan.local_len``)."""
        return local_len(self.max_ctx, 0, self.world, self.interleave)

    @property
    def row_keys(self) -> int:
        """A row's keys held here at most: the whole row's to its owner
        (``own_rows``), a rank's share otherwise. Row r's ILOG words start at r
        times this (a score task stores its keys' alone)."""
        if self.short_owned_verify and self.tokens // self.qlen < self.world:
            # Only short requests stay owned at low BS; long ones retain the
            # original split-row stride instead of reserving a whole context.
            return max(self.lmax, min(self.max_ctx, isel.SHORT_VERIFY_CTX))
        return self.max_ctx if self.own_rows else self.lmax

    @property
    def sflag_row(self) -> int:
        """A row's SFLAG pairs (a ``plan.SCORE_KEYS`` unit each), row r's at r
        times this."""
        return -(-self.row_keys // isel.SCORE_KEYS)

    @property
    def ilog_words(self) -> int:
        return self.tokens * self.row_keys

    @property
    def qtasks(self) -> int:
        """wq_b's 16-row tasks."""
        return self.shape.qrows // TILE

    @property
    def active_poll(self) -> bool:
        """Only the deciding waves poll the ranks' histograms (``select._decide``)."""
        return bool(self.dc1 & DC1_ACTIVE_POLL)

    @property
    def packed_hist(self) -> bool:
        """The ranks' histograms pushed three bins a pair (``select.PACK``)."""
        return bool(self.dc1 & DC1_PACKED_HIST)

    @property
    def flat_store(self) -> bool:
        return bool(self.dc1 & DC1_FLAT_STORE)

    @property
    def xpose_sum(self) -> bool:
        """Score head sums four (row, unit)s a reduction, a store each
        (``ops.head_sums4``; masked stores as ``flat_store``'s)."""
        return bool(self.dc1 & DC1_XPOSE_SUM)

    @property
    def early_cnt(self) -> bool:
        """A long row's CNT (``rows_before``) from its radix passes' counts of
        this rank, before its segments' winners are counted (``_choose``)."""
        return bool(self.dc1 & DC1_EARLY_CNT)

    @property
    def bucket_exit(self) -> bool:
        """A long row's radix passes stop once its threshold bin is needed
        whole (``stage_segment``): the same keys, in the same order."""
        return bool(self.dc1 & DC1_BUCKET_EXIT)

    @property
    def early_w(self) -> bool:
        """A q quant task's head weight is loaded before its QRDY poll."""
        return bool(self.dc1 & DC1_EARLY_W)

    @property
    def joint_poll(self) -> bool:
        """A long row's segment polls the earlier rows' CNT (``rows_before``)
        alongside its row's earlier segments' SGE, before one barrier (their
        CNT is early: ``early_cnt``)."""
        return bool(self.dc1 & DC1_JOINT_POLL)

    @property
    def name(self) -> str:
        from atom.mono.plan.build_key import symbol_params

        params = symbol_params(self)
        if not params["dc"]:
            del params["dc"]
        return kernel_symbol(f"mono_dsa_indexer_{self.shape.tag}", **params)


def unfit(key: IxBuild) -> str:
    """Why a build of ``key`` cannot exist, or ''."""
    if key.qlen < 1:
        return "query length must be positive"
    if key.world not in (1, 2, 4, 8):
        return f"unsupported TP world: {key.world}"
    limit = MAX_VERIFY_TOKENS if key.qlen == 6 else MAX_TOKENS
    if key.short_gemm and (
        not key.replicated
        or key.world <= 1
        or key.shape.heads != 32
        or key.shape.head_dim != 128
        or key.shape.q_lora not in (1024, 1536, 2048)
        or key.max_ctx > isel.SPLIT_MIN
    ):
        return "short GEMM requires replicated 32x128 heads and Q LoRA 1024/1536/2048"
    if not 1 <= key.tokens <= limit:
        return f"{key.tokens} rows: qlen {key.qlen} built for 1 .. {limit}"
    if key.score_blocks not in (BLOCKS, 2 * BLOCKS):
        return f"unsupported score grid: {key.score_blocks}"
    if key.score_blocks > BLOCKS and not (
        key.stream_scores
        and key.tokens in (128, 160, 192, 224, 256)
        and key.world == 8
        and dual_score_shape(key.shape)
    ):
        return (
            "512 CTAs require a tested Kimi/GLM DCP8 shape at a 128..256 step-32 bucket"
        )
    if key.tokens > MAX_LEGACY_TOKENS and (
        key.qlen not in (1, 6) or (key.world <= 1 and not key.replicated)
    ):
        return (
            f"{key.tokens} rows: above {MAX_LEGACY_TOKENS} requires qlen 1 or 6 "
            "with DCP or a replicated cache"
        )
    if key.tokens % key.qlen:
        return f"{key.tokens} rows: not whole requests of {key.qlen}"
    if key.shape.topk < key.world * key.interleave:
        # a long row must hold keys on every rank: each announces its count
        return (
            f"top-{key.shape.topk} under {key.world} ranks' {key.interleave}-key runs"
        )
    if key.row_keys >= isel.ARRIVED:
        # a row histogram word holds a bin's count below its segments
        return f"{key.row_keys} keys a row: the row histograms count < {isel.ARRIVED}"
    if key.dc1 & ~DC1_ALL:
        return f"unknown DC1 bits {key.dc1:#x}"
    if key.packed_hist and (
        key.replicated
        or key.radix != isel.DEFAULT_RADIX
        or key.row_keys >= 1 << isel.PACK_BITS
    ):
        return "packed histograms: DCP builds of the default radix, counts < 2^18"
    if (key.bucket_exit or key.early_cnt) and key.replicated:
        return "bucket exit / early CNT: DCP builds"
    if key.joint_poll and not key.early_cnt:
        return "joint CNT / SGE poll: with early CNT"
    return ""


def dual_score_shape(shape: IndexerShape) -> bool:
    """Shapes measured with a dual-resident grid; keep other models unchanged."""
    if unsupported(shape) or shape.kv_dtype != "fp8":
        return False
    return (
        shape.q_lora == 1536 and shape.block_rows == 128 and not shape.rope_dim
    ) or (
        shape.q_lora == 2048
        and shape.topk == 2048
        and shape.block_rows in (16, 128)
        and shape.rope_dim == 64
        and not shape.rope_neox
    )


def dual_score_resources_fit(resources: list[dict]) -> bool:
    """gfx950, 8 waves/CTA: two CTAs need <=128 VGPR and <=80 KiB LDS each."""
    return bool(resources) and all(
        r["vgpr"] <= 128 and r["lds"] <= 80 * 1024 and r["scratch"] == 0
        for r in resources
    )
