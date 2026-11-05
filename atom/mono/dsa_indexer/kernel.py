# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A DSA indexer at decode, its keys split over the TP ranks, one launch a
DSA layer: ``wq_b``, the index cache writer (``_indexer_nope_quant_and_cache``
without RoPE, ``indexer_qk_rope_quant_and_cache`` with) and the selection of
the original path -- under DCP ``dcp_decode_candidate_exchange_fused``'s,
to the same owned KV slots; without it (``IxBuild.replicated``, every rank's
cache whole) ``score_topk`` and the slot conversion's, every rank the whole
selection at the MLA's indptr (``plan``; its ``IndexerShape`` is the layer's
numbers, and ``plan.unsupported`` the shapes this kernel is not written for).

    qgemv   (a CTA a 16-row task: rows 16 g .. of wq_b [heads x head_dim,
            q_lora] bf16 against the token rows' q latent, 16 rows in LDS at
            a time) -> bf16 q to IQB, QRDY[g]
    k       (a token, on its owner rank only: slot >= 0) the original's
            LayerNorm -> bf16 (-> RoPE -> bf16) -> per-token power-of-two FP8
            -> the index cache row (preshuffled, ``plan.cache_byte``) and its
            scale, KDONE
    iquant  (a (token, head), wave 0): the head's row groups (QRDY) (-> RoPE
            -> bf16) -> per-(token, head) FP8 q and its head weight (w q_scale
            weights_scale) -> IQ8 / IW (the scorer's operands), QQ
    score   (a long row's unit of 128 local keys, every CTA): ``select``
    select  (a long row's segments): its distributed selection (``select``)
    row     (a row, CTA 255 - r): a short row's every key; replicated, a long
            row's every rank's winners; its slots at its CSR offset
            (``select.stage_row``)

Rounding points and formulas are the original writer's: the K scale's
``exp2(ceil(log2(amax / 448)))`` on the hardware log / exp the original
compiles to; the NoPE writer's LayerNorm mul and add unfused (``nope.py``),
the RoPE writer's one fma and its q scale a power of two too.
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.mono.device import gemv
from atom.mono.device.mx import FP8_MAX, clamp_fp8
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_log2,
    hw_rsq,
    kernel_symbol,
    ld_f32,
    load_ptr64,
    row_sum,
    rsrc,
    traced,
    wave_sum,
    xshfl,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, preg, publish, sreg
from atom.mono.dsa_indexer import fp4
from atom.mono.dsa_indexer import select as isel
from atom.mono.dsa_indexer.plan import (
    REPLICA_INTERLEAVE,
    IndexerShape,
    cache_byte,
    local_len,
    scale_byte,
    unsupported,
)
from atom.mono.plan.build_key import key_tuple, source_digest, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.layout import byte_layout, pair_layout
from atom.mono.runtime.abi import KernelAbi

SOURCES = source_digest("mono")
Q_AMAX_SEED = 1e-10
K_AMAX_SEED = 1e-4
LN_EPS = 1e-6
# timeline points: start, qgemv, iquant, score, a segment's scores all in, its
# threshold, its slots out; then a pass each: counted, pushed, every rank's in;
# then a CTA's (last) score task: entered, its flags in, scored, published
TL_PASS0 = 7
TL_SCORE0 = TL_PASS0 + 3 * isel.PASSES
TL_POINTS = TL_SCORE0 + 4
# The original planner handles one row per lane. Plain decode also serves
# up to four 64-row tiles; the q GEMV stages gemv.ROWS rows at a time.
# One resident CTA per row is required for key writes and row emission.
MAX_LEGACY_TOKENS = 64
MAX_TOKENS = BLOCKS
MAX_VERIFY_TOKENS = 256

_STREAM = fx.Stream(None)


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
    # per-CTA stamps (``TL_POINTS``) to plain "tl"
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
    score_blocks: int = BLOCKS
    # gfx950 short verify: BF16 prepacked Q projection and local MQA scheduling.
    short_gemm: bool = field(default=False, metadata={"sym": "sq"})

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
        return self.shape.qrows // gemv.ROWS

    @property
    def name(self) -> str:
        return kernel_symbol(
            f"mono_dsa_indexer_{self.shape.tag}", **symbol_params(self)
        )


ABI = KernelAbi(
    (
        "qr",
        "qr_ld",
        "w_qb",
        "w_qb_packed",
        "kp",
        "wp",
        "kw_ld",
        "kn_w",
        "kn_b",
        "slots",
        # the RoPE operands (an indexer without RoPE passes zeros): positions
        # (int64), the bf16 cos / sin tables, their row stride and rows
        "pos",
        "cos",
        "sin",
        "cs_ld",
        "maxpos",
        "cache",
        "cache_scale",
        "ctx",
        # a query token's request (``batch_id_per_q_token``; a verify's only:
        # a plain decode's token r is request r), its block table row and ctx
        "bids",
        "bt",
        "bt_ld",
        "out_idx",
        "out_ptr",
        "out_cnt",
        "scratch",
        "plain",
        # replicated: the rows' ILOG, the production indexer's logits plane
        # (``LogitsWorkspace``; under DCP a region of ``plain``, this 0)
        "ilog",
        "peers",
        "peer_off",
        "rank",
        "layer",
    )
)


def scratch_layout(key: IxBuild) -> dict:
    """name -> (byte offset, bytes): the mailbox flags, and the rows'
    histograms."""
    s = key.tokens
    return pair_layout(
        [
            # Reserve both Q halves for every wide build: size stays monotonic
            # even when the scheduler's largest width is not a dual-CTA bucket.
            ("qrdy", key.qtasks * (2 if s > 128 or key.split_q else 1)),
            ("qq", s * key.shape.heads),  # a (token, head)'s FP8 q and weight out
            ("kdone", s),  # a token's key in the cache (its owner only)
            ("sflag", s * key.sflag_row),  # a (row, score unit) scored
            ("cnt", s),  # a long row's slot count (its last segment's)
            ("sge", 2 * BLOCKS),  # a step segment's (above, equal) counts
            # replicated: a step segment's winners in this rank's slab
            ("emitted", BLOCKS if key.replicated else 0),
        ]
    )


def plain_layout(key: IxBuild) -> dict:
    """name -> (byte offset, bytes) of the plain rows, in the caller's
    ``plain`` buffer (``byte_layout``: never under another launch's flags)."""
    s, sh = key.tokens, key.shape
    return byte_layout(
        (
            ("iqb", s * sh.qrows * 2),  # bf16 q
            ("iq8", s * sh.qrows),  # fp8 q
            ("iw", s * sh.heads * 4),  # f32 head weights
            # the long rows' local scores (replicated: the ``ilog`` argument)
            ("ilog", 0 if key.replicated else key.ilog_words * 4),
            ("tl", BLOCKS * key.timeline_points * 8 if key.timeline else 0),
            # replicated: (row, rank)'s winners' KV slots, written by that rank
            # (the plain part of a peer buffer: every rank's at one offset)
            ("slab", s * key.world * sh.topk * 4 if key.replicated else 0),
        )
    )


def plain_bytes(key: IxBuild) -> int:
    return max(o + n for o, n in plain_layout(key).values())


def unfit(key: IxBuild) -> str:
    """Why a build of ``key`` cannot exist, or ''."""
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
    return ""


def scratch_bytes(key: IxBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def peer_layout(key: IxBuild) -> dict:
    """name -> (byte offset, bytes) from the indexer's own start: a launch's
    ``peer_off`` into a peer buffer it may share with other kernels (past
    their regions, so a rank already in the next kernel cannot land pushes on
    pairs this one's peers are still polling)."""
    return pair_layout(
        isel.peer_pairs(key.tokens, key.world, key.replicated, key.radix).items()
    )


def peer_bytes(key: IxBuild) -> int:
    """The indexer's peer bytes from its own start (``peer_layout``)."""
    return max(o + n for o, n in peer_layout(key).values())


# ---------------------------------------------------------------- qgemv
@traced
def stage_qgemv(c, g, ops, t0: int, sg: int):
    """wq_b rows 16 g .. of token rows t0 .. t0 + sg (in LDS): fp32 dots ->
    bf16 q -> IQB (device scope). Ends at a barrier past every read of ``xl``
    and ``red``: the next group's rows may land."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    sh = c["shape"]
    acc = gemv.mfmas(lane, wave, c["xl"], c["xrow"], sh.q_lora, sg, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < (gemv.ROWS // 2) * sg:
        rp = tid % (gemv.ROWS // 2)
        t = tid // (gemv.ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, t))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, t))
        w = fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16)
        bo.buffer_store(
            w.bitcast(fx.Int32)[0],
            rsrc(c["iqb"]),
            ((t0 + t) * sh.qrows + g * gemv.ROWS + 2 * rp) // 2,
            cache_modifier=CM_DEV,
        )
    gpu.barrier()


# ---------------------------------------------------------------- rope
def _rope_cs(c, t, pair):
    """Token t's (cos, sin) of rotary pair ``pair``: the bf16 tables at its
    position, clamped into them (a padded row's, as the original clamps)."""
    a = c["args"]
    pos = fx.Int32(bo.buffer_load(rsrc(a["pos"]), 2 * t, vec_width=1, dtype=T.i32))
    pos = fx.min(fx.max(pos, fx.Int32(0)), a["maxpos"] - 1)
    at = pos * a["cs_ld"] + pair
    return ld_f32(a["cos"], at, T.bf16), ld_f32(a["sin"], at, T.bf16)


def _rotate(v, partner, cs, lead):
    """One dim of a RoPE pair, the original's ``lead ? v c - p s : v c + p s``,
    rounded to bf16 as the original materializes it."""
    cos_v, sin_v = cs
    lo, hi = v * cos_v - partner * sin_v, v * cos_v + partner * sin_v
    if isinstance(lead, bool):
        return bf16_round(lo if lead else hi)
    return bf16_round(fx.Float32(lead.select(lo, hi)))


def _rope_k(c, t, lane, v):
    """Key dim ``lane`` (all of 0 .. 63 RoPE dims, one a lane) rotated with its
    pair's partner lane."""
    half = c["shape"].rope_dim // 2
    if const_expr(c["shape"].rope_neox):
        return _rotate(v, xshfl(v, half), _rope_cs(c, t, lane % half), lane < half)
    return _rotate(v, xshfl(v, 1), _rope_cs(c, t, lane // 2), lane % 2 == 0)


def _rope_q(c, t, lane, x0, x1):
    """A q lane's dims 2 lane, 2 lane + 1, rotated where they are RoPE dims
    (lanes 0 .. rope_dim / 2 - 1): interleaved pairs sit in the lane, NeoX
    pairs in lane ^ (half / 2)."""
    half = c["shape"].rope_dim // 2
    rope = lane < half
    if const_expr(c["shape"].rope_neox):
        lead = lane < half // 2
        y0 = _rotate(x0, xshfl(x0, half // 2), _rope_cs(c, t, (2 * lane) % half), lead)
        y1 = _rotate(
            x1, xshfl(x1, half // 2), _rope_cs(c, t, (2 * lane + 1) % half), lead
        )
    else:
        cs = _rope_cs(c, t, fx.min(lane, fx.Int32(half - 1)))
        y0 = _rotate(x0, x1, cs, True)
        y1 = _rotate(x1, x0, cs, False)
    return fx.Float32(rope.select(y0, x0)), fx.Float32(rope.select(y1, x1))


# ---------------------------------------------------------------- k
@traced
def stage_k(c, t):
    """Token t's index key, on the rank that holds it (wave 0, lane l dims l
    and 64 + l): LayerNorm (two passes) -> bf16 -> amax (floor 1e-4) -> scale
    2^ceil(log2(amax / 448)) -> FP8 -> the cache row and its scale. The NoPE
    writer's LayerNorm (``nope.py``): mul / add unfused, the affine params
    through bf16; the fused RoPE writer's: fp32 params, scale and shift one
    fma (its default contraction), then the rotation (dims 0 .. 63)."""
    lane, wave, sh = c["lane"], c["wave"], c["shape"]
    a = c["args"]
    slot = fx.Int32(bo.buffer_load(rsrc(a["slots"]), 2 * t, vec_width=1, dtype=T.i32))
    if (wave == 0) & (slot >= 0):
        dims = [lane, lane + 64]
        x = [ld_f32(a["kp"], t * a["kw_ld"] + d, T.bf16) for d in dims]
        g = [ld_f32(a["kn_w"], d) for d in dims]
        b = [ld_f32(a["kn_b"], d) for d in dims]
        inv_n = fx.Float32(1.0 / sh.head_dim)
        mean = wave_sum(x[0] + x[1]) * inv_n
        cs = [v - mean for v in x]
        var = wave_sum(cs[0] * cs[0] + cs[1] * cs[1]) * inv_n
        r = hw_rsq(var + fx.Float32(LN_EPS))
        if const_expr(sh.rope_dim):
            y = [
                bf16_round(fx.Float32(fmath.fma(cs[i] * r, g[i], b[i])))
                for i in range(2)
            ]
            y[0] = _rope_k(c, t, lane, y[0])
        else:
            y = [
                bf16_round((cs[i] * r) * bf16_round(g[i]) + bf16_round(b[i]))
                for i in range(2)
            ]
        if const_expr(sh.kv_dtype == "fp4"):
            fp4.store_key(c, slot, y)
        else:
            amax = butterfly(fx.max(abs(y[0]), abs(y[1])), (32, 16, 8, 4, 2, 1), fx.max)
            amax = fx.max(amax, fx.Float32(K_AMAX_SEED))
            ks = hw_exp2(fmath.ceil(hw_log2(amax / fx.Float32(FP8_MAX))))
            inv = fx.Float32(1.0) / ks
            q = fp8_pack4(
                clamp_fp8(y[0] * inv),
                clamp_fp8(y[1] * inv),
                fx.Float32(0.0),
                fx.Float32(0.0),
            )
            blk = slot // sh.block_rows
            row = slot % sh.block_rows
            base = rsrc(a["cache"] + fx.Int64(blk) * fx.Int64(sh.block_bytes))
            # device scope: this launch's scorer of the row reads them (KDONE),
            # from a CTA on any XCD
            for i in range_constexpr(2):
                bo.buffer_store(
                    fx.Int8((q >> (8 * i)) & 0xFF),
                    base,
                    cache_byte(sh, row, dims[i]),
                    cache_modifier=CM_DEV,
                )
            if lane == 0:
                bo.buffer_store(
                    ks, base, scale_byte(sh, row) // 4, cache_modifier=CM_DEV
                )
    # this rank's scorer of the token's row reads the key it just wrote
    publish(c["put"], c["kdone"], t, fx.Int32(1), (c["tid"] == 0) & (slot >= 0))


# ---------------------------------------------------------------- iquant
@traced
def stage_iquant(c, bid, tokens):
    """This CTA's (token, head) tasks bid, bid + BLOCKS, .., a wave each
    (``WAVES`` a round): every one is head bid % heads (BLOCKS a multiple of
    heads), so that head's row-group flags are polled once. A task is the
    original's per-(token, head) FP8 quant and the head weight folded with its
    scale; each round's flags go out together. Wide dual-CTA builds divide
    token rows into two halves, with separate QRDY groups for each half."""
    tid, lane, wave, sh = c["tid"], c["lane"], c["wave"], c["shape"]
    assert BLOCKS % sh.heads == 0
    offset = fx.Int32(0)
    ready_offset = fx.Int32(0)
    if const_expr(c["split_q"]):
        offset = bid // BLOCKS * (tokens // 2) * sh.heads
        ready_offset = bid // BLOCKS * BLOCKS
        bid = bid % BLOCKS
        tokens = tokens // 2
    elif const_expr(c["short_owned_verify"] and not c.get("local_short", False)):
        offset = isel.short_verify_step(c).select(
            c["rank"] * c["qlen"] * sh.heads, fx.Int32(0)
        )
    n = fx.Int32(tokens * sh.heads)
    if const_expr(c["short_owned_verify"] and not c.get("local_short", False)):
        n = isel.short_verify_step(c).select(fx.Int32(c["qlen"] * sh.heads), n)
    rounds = -(-(tokens * sh.heads) // (BLOCKS * WAVES))
    groups = sh.head_dim // gemv.ROWS
    if tid < groups:
        c["poll"]([(c["qrdy"], ready_offset + (bid % sh.heads) * groups + tid, 1)])
    gpu.barrier()
    for r in range_constexpr(rounds):
        task = bid + BLOCKS * (r * WAVES + wave)
        if task < n:
            iquant_task(c, task + offset, lane)
        publish(c["put"], c["qq"], task + offset, fx.Int32(1), (lane == 0) & (task < n))


@traced
def iquant_task(c, task, lane):
    """Token task // heads, head task % heads on one wave, 2 dims a lane."""
    sh = c["shape"]
    t = task // sh.heads
    h = task % sh.heads
    at = t * sh.qrows + h * sh.head_dim + 2 * lane
    qw = fx.Int32(
        bo.buffer_load(
            rsrc(c["iqb"]), at // 2, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
        )
    )
    x0, x1 = bf2_f32(qw)
    iquant_values(c, task, lane, x0, x1)


@traced
def iquant_values(c, task, lane, x0, x1):
    a, sh = c["args"], c["shape"]
    t, h = task // sh.heads, task % sh.heads
    at = t * sh.qrows + h * sh.head_dim + 2 * lane
    if const_expr(sh.rope_dim):
        x0, x1 = _rope_q(c, t, lane, x0, x1)
    scale = fx.Float32(1.0)
    if const_expr(sh.kv_dtype == "fp4"):
        fp4.store_query(c, task, lane, x0, x1)
    else:
        amax = butterfly(
            fx.max(fx.max(abs(x0), abs(x1)), fx.Float32(Q_AMAX_SEED)),
            (32, 16, 8, 4, 2, 1),
            fx.max,
        )
        scale = amax * fx.Float32(1.0 / FP8_MAX)
        if const_expr(sh.rope_dim):
            # the fused RoPE writer's scale is ue8m0 (``Indexer.scale_fmt``): a
            # power of two; the NoPE writer's is amax / 448 itself
            scale = hw_exp2(fmath.ceil(hw_log2(scale)))
        inv = fx.Float32(1.0) / scale
        q = fp8_pack4(
            clamp_fp8(x0 * inv), clamp_fp8(x1 * inv), fx.Float32(0.0), fx.Float32(0.0)
        )
        bo.buffer_store(
            fx.Int16(q & 0xFFFF), rsrc(c["iq8"]), at // 2, cache_modifier=CM_DEV
        )
    if lane == 0:
        w = ld_f32(a["wp"], t * a["kw_ld"] + h, T.bf16)
        bo.buffer_store(
            (w * scale) * fx.Float32(sh.weights_scale),
            rsrc(c["iw"]),
            t * sh.heads + h,
            cache_modifier=CM_DEV,
        )


# ---------------------------------------------------------------- kernel
@traced
def stage_q_head(c, qr, qr_ld, w_qb):
    """A CTA projects complete heads, then quantizes them without a global rendezvous."""
    sh, tid, lane, wave = c["shape"], c["tid"], c["lane"], c["wave"]
    rows = 32 if c["S"] > 128 else 16
    head = c["bid"] % sh.heads
    base = c["bid"] // sh.heads * rows
    if base < c["S"]:
        count = fx.min(fx.Int32(rows), c["S"] - base)
        src = qr + fx.Int64(base) * fx.Int64(qr_ld) * fx.Int64(2)
        # The last tile repeats its final live row on padded lanes.
        for chunk in range_constexpr(rows * sh.q_lora // (8 * THREADS)):
            at = tid + chunk * THREADS
            token, col = at // (sh.q_lora // 8), at % (sh.q_lora // 8) * 8
            value = fx.Vector(
                bo.buffer_load(
                    rsrc(src),
                    (fx.min(token, count - 1) * qr_ld + col) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=0,
                )
            )
            fx.ptr_store(value, c["xl"] + (token * c["xrow"] + col) // 2)
        gpu.barrier()
        accumulators = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(rows // 16)]

        def loads(k0):
            packed = rsrc(w_qb)
            return [
                [
                    fx.Vector(
                        bo.buffer_load(
                            packed,
                            (
                                (
                                    (
                                        head * (sh.q_lora // 32)
                                        + k0 // 32
                                        + ch * 4
                                        + step
                                    )
                                    * 8
                                    + wave
                                )
                                * 64
                                + lane
                            )
                            * 4,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=0,
                        )
                    )
                    for step in range(4)
                ]
                for ch in range(4)
            ]

        pending = loads(0)
        for slab in range_constexpr(sh.q_lora // 512):
            operands = pending
            if const_expr(slab + 1 < sh.q_lora // 512):
                pending = loads((slab + 1) * 512)
            for tile in range_constexpr(rows // 16):
                acc = accumulators[tile]
                for ch in range_constexpr(4):
                    for step in range_constexpr(4):
                        kk = slab * 512 + ch * 128 + step * 32 + lane // 16 * 8
                        xb = fx.Vector(
                            fx.ptr_load(
                                c["xl"]
                                + ((tile * 16 + lane % 16) * c["xrow"] + kk) // 2,
                                result_type=fx.Vector.make_type(4, fx.Int32),
                            )
                        )
                        acc = gemv.mfma_bf16(
                            operands[ch][step].bitcast(fx.BFloat16),
                            xb.bitcast(fx.BFloat16),
                            acc,
                        )
                accumulators[tile] = acc
        local = fx.recast_iter(fx.Int32, c["red"])
        for tile in range_constexpr(rows // 16):
            token = tile * 16 + lane % 16
            for pair in range_constexpr(2):
                value = (
                    fx.Vector.from_elements(
                        [
                            accumulators[tile][2 * pair],
                            accumulators[tile][2 * pair + 1],
                        ],
                        fx.Float32,
                    )
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                dim = wave * 16 + lane // 16 * 4 + 2 * pair
                fx.ptr_store(value, local + (token * 128 + dim) // 2)
                if token < count:
                    bo.buffer_store(
                        value,
                        rsrc(c["iqb"]),
                        ((base + token) * sh.qrows + head * 128 + dim) // 2,
                        cache_modifier=CM_DEV,
                    )
        gpu.barrier()
        for turn in range_constexpr(rows // WAVES):
            token = wave + turn * WAVES
            task = (base + token) * sh.heads + head
            if token < count:
                packed = fx.ptr_load(local + (token * 128 + 2 * lane) // 2)
                x0, x1 = bf2_f32(packed)
                iquant_values(c, task, lane, x0, x1)
        flag_task = (base + tid) * sh.heads + head
        publish(c["put"], c["qq"], flag_task, fx.Int32(1), tid < count)


_BUILDS: dict = {}


def build(key: IxBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, sh = key.tokens, key.shape
    score_blocks = key.score_blocks
    split_q, radix = key.split_q, key.radix
    reuse_q_scratch = key.reuse_q_scratch
    tl_score0, tl_points = TL_PASS0 + 3 * radix.passes, key.timeline_points
    why = unsupported(sh) or unfit(key)
    assert not why, why
    if score_blocks > BLOCKS and not dual_score_device():
        from atom.mono.runtime.consensus import MonoUnsupported

        raise MonoUnsupported(
            "512 indexer CTAs require an unmasked 256-CU gfx950 device"
        )
    assert key.qtasks == BLOCKS
    lay = scratch_layout(key)
    plain_lay = plain_layout(key)
    play = peer_layout(key)
    xrow = sh.q_lora + gemv.LDS_PAD
    short_select = (
        key.replicated
        and (s <= MAX_LEGACY_TOKENS or key.qlen == 6)
        and key.qlen in (1, 6, 8)
        and sh.topk == 2048
        and (key.own_rows or key.world == 1)
    )

    tuned_score = (
        key.short_gemm and key.short_owned_verify and key.qlen == 6 and s >= 48
    )
    tiled_q = tuned_score and s >= 96

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        # Tiled short Q uses up to 32 rows, then reuses this storage for MQA.
        xl: fx.Array[
            fx.Int32, min(s, 32 if tiled_q and s > 128 else gemv.ROWS) * xrow // 2, 16
        ]
        hist: fx.Array[fx.Int32, radix.bins, 16]
        # a row's earlier segments' (above, equal) counts (``stage_segment``)
        if not reuse_q_scratch:
            before: fx.Array[fx.Int32, 2 * BLOCKS, 16]
        # ``select.SEL_*``: a pass's decisions, then a word a row (or, a
        # replicated row's gather, a rank)
        sel: fx.Array[fx.Int32, isel.SEL_ROWS + max(s, key.world), 16]
        # a block scan's wave totals; ``select._sum_before``'s two halves
        scan: fx.Array[fx.Int32, 2 * WAVES, 16]
        # every row's kind (``select.step_table``, including all wide tiles)
        rows: fx.Array[fx.Int32, isel.ROW_WORDS * s, 16]
        # the step's score and segment tasks (``select.step_table``)
        plan: fx.Array[fx.Int32, isel.PLAN_FIRSTS + 2 * s, 16]
        # a score task's rows 1 .. rg - 1's queries and head weights
        # (``select._rows_loads``)
        if isel.piece_rows(key.qlen) > 1 and not tiled_q:
            qrows: fx.Array[
                fx.Int32, (isel.piece_rows(key.qlen) - 1) * isel.qrow_words(sh), 16
            ]
        if key.timeline:
            tls: fx.Array[fx.Int64, tl_points, 16]

    keyed = key_tuple(key, SOURCES)
    on = key.timeline

    @flyc.kernel(name=key.name, known_block_size=[THREADS, 1, 1])
    def indexer(
        qr: Int64,
        qr_ld: Int32,
        w_qb: Int64,
        w_qb_packed: Int64,
        kp: Int64,
        wp: Int64,
        kw_ld: Int32,
        kn_w: Int64,
        kn_b: Int64,
        slots: Int64,
        pos: Int64,
        cos: Int64,
        sin: Int64,
        cs_ld: Int32,
        maxpos: Int32,
        cache: Int64,
        cache_scale: Int64,
        ctx: Int64,
        bids: Int64,
        bt: Int64,
        bt_ld: Int32,
        out_idx: Int64,
        out_ptr: Int64,
        out_cnt: Int64,
        scratch: Int64,
        plain: Int64,
        ilog: Int64,
        peers: Int64,
        peer_off: Int64,
        rank: Int32,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        mb = Mailbox(layer)
        hist_at, rcnt_at = play["hist"][0], play["rcnt"][0]
        bases = peer_bases(peers, key.world)
        own = load_ptr64(peers, rank)
        slab = plain + fx.Int64(plain_lay["slab"][0])
        c = {
            "S": s,
            "score_blocks": score_blocks,
            "split_q": split_q,
            "short_owned_verify": key.short_owned_verify,
            "local_short": key.short_owned_verify,
            "fast_short": tuned_score
            or (key.short_gemm and key.qlen == 6 and s == key.qlen),
            "short_score": key.short_owned_verify and s == key.qlen,
            "max_ctx": key.max_ctx,
            "radix": radix,
            "shape": sh,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "rank": rank,
            "world": key.world,
            "replicated": key.replicated,
            "own_rows": key.own_rows,
            "row_keys": key.row_keys,
            "sflag_row": key.sflag_row,
            # a segment's chunk loads past the last row read 0 (``_chunk_keys``)
            "ilog_bytes": key.ilog_words * 4,
            "inter": key.interleave,
            "qlen": key.qlen,
            "rg": isel.piece_rows(key.qlen),
            "ppr": isel.pieces_a_request(key.qlen),
            "delay_rank": key.delay_rank,
            "delay_ticks": key.delay_ticks,
            "put": mb.put,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "spin": mb.spin,
            "red": lds.red.ptr,
            "xl": lds.xl.ptr,
            "xrow": xrow,
            "hist": lds.hist.ptr,
            # Q's final reduction barrier retires red before selection starts.
            "before": (
                fx.recast_iter(fx.Int32, lds.red.ptr)
                if reuse_q_scratch
                else lds.before.ptr
            ),
            "sel": lds.sel.ptr,
            "scan": lds.scan.ptr,
            "rows": lds.rows.ptr,
            "plan": lds.plan.ptr,
            "qrows": (
                lds.xl.ptr
                if tiled_q
                else (lds.qrows.ptr if isel.piece_rows(key.qlen) > 1 else None)
            ),
            # the indexer's regions start ``peer_off`` into every rank's buffer
            "phist": [preg(b + peer_off, hist_at, "hist") for b in bases],
            "phist_own": preg(own + peer_off, hist_at, "hist"),
            # replicated: every rank's slab (``plain`` is a peer buffer's plain
            # part, at one offset from every rank's base) and count pairs
            "prcnt": [preg(b + peer_off, rcnt_at, "rcnt") for b in bases],
            "prcnt_own": preg(own + peer_off, rcnt_at, "rcnt"),
            "prdone": [preg(b + peer_off, play["rdone"][0], "rdone") for b in bases],
            "prdone_own": preg(own + peer_off, play["rdone"][0], "rdone"),
            "pslab": [b + (slab - own) for b in bases],
            "slab": slab,
            "iqb": plain + fx.Int64(plain_lay["iqb"][0]),
            "iq8": plain + fx.Int64(plain_lay["iq8"][0]),
            "iw": plain + fx.Int64(plain_lay["iw"][0]),
            "ilog": ilog if key.replicated else plain + fx.Int64(plain_lay["ilog"][0]),
            "rowhist": preg(own + peer_off, play["rowhist"][0], "rowhist"),
            "args": {
                "kp": kp,
                "wp": wp,
                "kw_ld": kw_ld,
                "kn_w": kn_w,
                "kn_b": kn_b,
                "slots": slots,
                "pos": pos,
                "cos": cos,
                "sin": sin,
                "cs_ld": cs_ld,
                "maxpos": maxpos,
                "cache": cache,
                "cache_scale": cache_scale,
                "ctx": ctx,
                "bids": bids,
                "bt": bt,
                "bt_ld": bt_ld,
                "out_idx": out_idx,
                "out_ptr": out_ptr,
                "out_cnt": out_cnt,
            },
        }
        for region in (
            "qrdy", "qq", "kdone", "sflag", "cnt", "sge", "emitted"
        ):  # fmt: skip
            c[region] = sreg(scratch, lay[region][0], region)
        tls = lds.tls.ptr if on else None
        c["stamp"] = lambda k: stamp(on, tls, tid, k)
        c["tl_pass"] = TL_PASS0
        c["tl_score"] = tl_score0
        stamp_begin(on, tls, tid, tl_points)
        isel.step_table(c)
        total, lg = isel.score_plan(c)
        # the query feeds only the scorer: a step of short rows alone (every
        # key selected) writes its keys and slots without wq_b. Every CTA
        # reads the same rows, so every CTA takes the same branch.
        projects = total > 0
        if const_expr(score_blocks > BLOCKS and not split_q):
            projects = projects & (bid < BLOCKS)
        if projects:
            # wq_b's rows in flight first; the per-token stage runs under them
            qgroup = bid % BLOCKS if const_expr(split_q) else bid
            qbase = bid // BLOCKS * (s // 2) if const_expr(split_q) else fx.Int32(0)

            # the token rows 16 at a time against the same wq_b rows; a group's
            # chunks all issued before any is stored, the next group's issued
            # before this one's MFMAs
            def project(base, count, ops):
                groups = [
                    (t0, min(gemv.ROWS, count - t0))
                    for t0 in range(0, count, gemv.ROWS)
                ]

                def group_loads(t0, sg):
                    rows = qr + fx.Int64(base + t0) * fx.Int64(qr_ld) * fx.Int64(2)
                    return gemv.rows_loads(tid, rows, sg, sh.q_lora, qr_ld, cm=0)

                xs = group_loads(*groups[0])
                for i, (t0, sg) in enumerate(groups):
                    gemv.rows_store(tid, xs, sg, sh.q_lora, c["xl"], xrow)
                    gpu.barrier()
                    if i + 1 < len(groups):
                        xs = group_loads(*groups[i + 1])
                    stage_qgemv(c, qgroup, ops, base + t0, sg)

            if const_expr(tiled_q):
                if isel.short_verify_step(c):
                    if bid < s:
                        stage_k(c, bid)
                    stage_q_head(c, qr, qr_ld, w_qb_packed)
                    stamp(on, tls, tid, 1)
                else:
                    ops = gemv.loads(
                        c["lane"], c["wave"], w_qb, qgroup * gemv.ROWS, sh.q_lora
                    )
                    if bid < s:
                        stage_k(c, bid)
                    project(qbase, s, ops)
                    publish(c["put"], c["qrdy"], bid, fx.Int32(1), tid == 0)
                    stamp(on, tls, tid, 1)
                    stage_iquant(c, bid, s)
            else:
                ops = gemv.loads(
                    c["lane"], c["wave"], w_qb, qgroup * gemv.ROWS, sh.q_lora
                )
                if bid < s:
                    stage_k(c, bid)
                project(qbase, s // 2 if split_q else s, ops)
                publish(c["put"], c["qrdy"], bid, fx.Int32(1), tid == 0)
                stamp(on, tls, tid, 1)
                stage_iquant(c, bid, s)
        else:
            if bid < s:
                stage_k(c, bid)
        stamp(on, tls, tid, 2)
        if const_expr(key.stream_scores):
            isel.score_row_stream(c)
        else:
            if const_expr(key.short_owned_verify and s == key.qlen):
                if isel.short_verify_step(c):
                    narrow = {
                        **c,
                        "rg": 2,
                        "ppr": key.qlen // 2,
                        "wide_fp8": key.short_gemm and key.qlen == 6,
                    }
                    for task in range(bid, total, BLOCKS):
                        q, t = isel.locate_score(narrow, task)
                        isel.stage_score(narrow, q, t, fx.Int32(1), 1)
                else:
                    for task in range(bid, total, BLOCKS):
                        q, t = isel.locate_score(c, task)
                        isel.score_task(c, q, t, lg)
            elif const_expr(tuned_score):
                if isel.short_verify_step(c):
                    requests = s // key.qlen
                    per = BLOCKS // requests
                    width = -(-isel.SHORT_VERIFY_CTX // (isel.SCORE_KEYS * per))
                    p, part = bid % requests, bid // requests
                    units = isel.score_units(c, p)
                    u0 = part * width
                    if (part < per) & (u0 < units):
                        isel.stage_score(
                            {**c, "wide_fp8": True},
                            p,
                            u0,
                            fx.min(units - u0, fx.Int32(width)),
                            width,
                        )
                else:
                    for task in range(bid, total, BLOCKS):
                        q, t = isel.locate_score(c, task)
                        isel.score_task(c, q, t, lg)
            else:
                for task in range(bid, total, BLOCKS):
                    q, t = isel.locate_score(c, task)
                    isel.score_task(c, q, t, lg)
        stamp(on, tls, tid, 3)
        # a long row's segments next, one a CTA, all resident: each waits on
        # its own units' scores, then on its row's other segments every pass
        stotal, cps_lg = isel.segment_plan(c)
        if bid < stotal:
            r, g = isel.locate_segment(c, bid)
            if const_expr(short_select):
                _, n, _, split = isel.row_kind(c, r)
                if (n <= isel.LOCAL_SELECT_MAX) & (~split | (key.world == 1)):
                    if g == 0:
                        if const_expr(key.short_owned_verify):
                            if isel.short_verify_step(c):
                                isel.stage_short_selection(
                                    {**c, "direct": True}, r, bid, n
                                )
                            else:
                                isel.stage_short_selection(c, r, bid, n)
                        else:
                            isel.stage_short_selection(c, r, bid, n)
                else:
                    isel.stage_segment(c, r, g, bid, cps_lg)
            else:
                isel.stage_segment(c, r, g, bid, cps_lg)
            stamp(on, tls, tid, 6)
        # a row's other slots last: a short row waits on the earlier long rows'
        # counts only, a replicated long row on every rank's winners
        emits = bid >= BLOCKS - s
        if const_expr(score_blocks > BLOCKS):
            emits = emits & (bid < BLOCKS)
        if emits:
            isel.stage_row(c, BLOCKS - 1 - bid)
        if bid < BLOCKS:
            stamp_flush(
                on, tls, plain + fx.Int64(plain_lay["tl"][0]), tid, bid, tl_points
            )

    @flyc.jit
    def launch(
        qr: Int64,
        qr_ld: Int32,
        w_qb: Int64,
        w_qb_packed: Int64,
        kp: Int64,
        wp: Int64,
        kw_ld: Int32,
        kn_w: Int64,
        kn_b: Int64,
        slots: Int64,
        pos: Int64,
        cos: Int64,
        sin: Int64,
        cs_ld: Int32,
        maxpos: Int32,
        cache: Int64,
        cache_scale: Int64,
        ctx: Int64,
        bids: Int64,
        bt: Int64,
        bt_ld: Int32,
        out_idx: Int64,
        out_ptr: Int64,
        out_cnt: Int64,
        scratch: Int64,
        plain: Int64,
        ilog: Int64,
        peers: Int64,
        peer_off: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        indexer(
            qr,
            qr_ld,
            w_qb,
            w_qb_packed,
            kp,
            wp,
            kw_ld,
            kn_w,
            kn_b,
            slots,
            pos,
            cos,
            sin,
            cs_ld,
            maxpos,
            cache,
            cache_scale,
            ctx,
            bids,
            bt,
            bt_ld,
            out_idx,
            out_ptr,
            out_cnt,
            scratch,
            plain,
            ilog,
            peers,
            peer_off,
            rank,
            layer,
        ).launch(grid=(score_blocks,), block=(THREADS,), stream=stream)

    ABI.check(indexer, launch)
    if score_blocks > BLOCKS:
        from atom.mono.runtime.compile import compile_kernels, launcher_resources
        from atom.mono.runtime.consensus import MonoUnsupported

        compile_kernels([(launch, ABI)])
        resources = launcher_resources(launch)
        if not dual_score_resources_fit(resources):
            raise MonoUnsupported(f"512 indexer CTAs cannot all reside: {resources}")
    _BUILDS[key] = launch
    return launch


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


def dual_score_device(device=None) -> bool:
    import os

    import torch

    if not torch.cuda.is_available() or any(
        os.environ.get(name)
        for name in ("ROC_GLOBAL_CU_MASK", "HSA_CU_MASK", "HIP_CU_MASK")
    ):
        return False
    props = torch.cuda.get_device_properties(device)
    return (
        getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx950"
        and props.multi_processor_count == BLOCKS
    )
