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
# the widest step built and checked (Kimi-K3's 64-row decode); the q GEMV takes
# it 16 rows (``gemv.ROWS``) at a time; at most a wave (``select.step_table``
# gives a row a lane)
MAX_TOKENS = 64

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

    @property
    def interleave(self) -> int:
        """A rank's runs of global keys (``plan.key_pos``)."""
        return REPLICA_INTERLEAVE if self.replicated else 1

    @property
    def own_rows(self) -> bool:
        """Replicated with at least one request a rank: each request's rows
        scored and selected whole by one rank (``select.owner``), the other
        ranks reading its winners from its slab -- no cross-rank pass (V4.1's
        rows split over ranks). A request its owner would score alone while the others idle still has its keys
        split over the ranks (``select.split_keys``, decided a step). Fewer
        requests than ranks: the ranks split each row's keys."""
        return (
            self.replicated
            and self.world > 1
            and self.tokens // self.qlen >= self.world
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
            ("qrdy", key.qtasks),
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
            ("tl", BLOCKS * TL_POINTS * 8 if key.timeline else 0),
            # replicated: (row, rank)'s winners' KV slots, written by that rank
            # (the plain part of a peer buffer: every rank's at one offset)
            ("slab", s * key.world * sh.topk * 4 if key.replicated else 0),
        )
    )


def plain_bytes(key: IxBuild) -> int:
    return max(o + n for o, n in plain_layout(key).values())


def unfit(key: IxBuild) -> str:
    """Why a build of ``key`` cannot exist, or ''."""
    if not 1 <= key.tokens <= MAX_TOKENS:
        return f"{key.tokens} rows: built for 1 .. {MAX_TOKENS}"
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
    return pair_layout(isel.peer_pairs(key.tokens, key.world, key.replicated).items())


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
            bo.buffer_store(ks, base, scale_byte(sh, row) // 4, cache_modifier=CM_DEV)
    # this rank's scorer of the token's row reads the key it just wrote
    publish(c["put"], c["kdone"], t, fx.Int32(1), (c["tid"] == 0) & (slot >= 0))


# ---------------------------------------------------------------- iquant
@traced
def stage_iquant(c, bid, tokens):
    """This CTA's (token, head) tasks bid, bid + BLOCKS, .., a wave each
    (``WAVES`` a round): every one is head bid % heads (BLOCKS a multiple of
    heads), so that head's row-group flags are polled once. A task is the
    original's per-(token, head) FP8 quant and the head weight folded with its
    scale; each round's flags go out together."""
    tid, lane, wave, sh = c["tid"], c["lane"], c["wave"], c["shape"]
    assert BLOCKS % sh.heads == 0
    n = tokens * sh.heads
    rounds = -(-n // (BLOCKS * WAVES))
    groups = sh.head_dim // gemv.ROWS
    if tid < groups:
        c["poll"]([(c["qrdy"], (bid % sh.heads) * groups + tid, 1)])
    gpu.barrier()
    for r in range_constexpr(rounds):
        task = bid + BLOCKS * (r * WAVES + wave)
        if task < n:
            iquant_task(c, task, lane)
        publish(c["put"], c["qq"], task, fx.Int32(1), (lane == 0) & (task < n))


@traced
def iquant_task(c, task, lane):
    """Token task // heads, head task % heads on one wave, 2 dims a lane."""
    a, sh = c["args"], c["shape"]
    t = task // sh.heads
    h = task % sh.heads
    at = t * sh.qrows + h * sh.head_dim + 2 * lane
    qw = fx.Int32(
        bo.buffer_load(
            rsrc(c["iqb"]), at // 2, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
        )
    )
    x0, x1 = bf2_f32(qw)
    if const_expr(sh.rope_dim):
        x0, x1 = _rope_q(c, t, lane, x0, x1)
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
_BUILDS: dict = {}


def build(key: IxBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, sh = key.tokens, key.shape
    why = unsupported(sh) or unfit(key)
    assert not why, why
    assert key.qtasks == BLOCKS
    lay = scratch_layout(key)
    plain_lay = plain_layout(key)
    play = peer_layout(key)
    xrow = sh.q_lora + gemv.LDS_PAD

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        # one 16-row group of the q latent at a time (``stage_qgemv``)
        xl: fx.Array[fx.Int32, min(s, gemv.ROWS) * xrow // 2, 16]
        hist: fx.Array[fx.Int32, isel.BINS, 16]
        # a row's earlier segments' (above, equal) counts (``stage_segment``)
        before: fx.Array[fx.Int32, 2 * BLOCKS, 16]
        # ``select.SEL_*``: a pass's decisions, then a word a row (or, a
        # replicated row's gather, a rank)
        sel: fx.Array[fx.Int32, isel.SEL_ROWS + max(s, key.world), 16]
        # a block scan's wave totals; ``select._sum_before``'s two halves
        scan: fx.Array[fx.Int32, 2 * WAVES, 16]
        # every row's kind (``select.step_table``)
        rows: fx.Array[fx.Int32, isel.ROW_WORDS * s, 16]
        # the step's score and segment tasks (``select.step_table``)
        plan: fx.Array[fx.Int32, isel.PLAN_FIRSTS + 2 * s, 16]
        # a score task's rows 1 .. rg - 1's queries and head weights
        # (``select._rows_loads``)
        if isel.piece_rows(key.qlen) > 1:
            qrows: fx.Array[
                fx.Int32, (isel.piece_rows(key.qlen) - 1) * isel.qrow_words(sh), 16
            ]
        if key.timeline:
            tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, SOURCES)
    on = key.timeline

    @flyc.kernel(name=key.name, known_block_size=[THREADS, 1, 1])
    def indexer(
        qr: Int64,
        qr_ld: Int32,
        w_qb: Int64,
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
            "before": lds.before.ptr,
            "sel": lds.sel.ptr,
            "scan": lds.scan.ptr,
            "rows": lds.rows.ptr,
            "plan": lds.plan.ptr,
            "qrows": lds.qrows.ptr if isel.piece_rows(key.qlen) > 1 else None,
            # the indexer's regions start ``peer_off`` into every rank's buffer
            "phist": [preg(b + peer_off, hist_at, "hist") for b in bases],
            "phist_own": preg(own + peer_off, hist_at, "hist"),
            # replicated: every rank's slab (``plain`` is a peer buffer's plain
            # part, at one offset from every rank's base) and count pairs
            "prcnt": [preg(b + peer_off, rcnt_at, "rcnt") for b in bases],
            "prcnt_own": preg(own + peer_off, rcnt_at, "rcnt"),
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
        c["tl_score"] = TL_SCORE0
        stamp_begin(on, tls, tid, TL_POINTS)
        isel.step_table(c)
        total, lg = isel.score_plan(c)
        # the query feeds only the scorer: a step of short rows alone (every
        # key selected) writes its keys and slots without wq_b. Every CTA
        # reads the same rows, so every CTA takes the same branch.
        if total > 0:
            # wq_b's rows in flight first; the per-token stage runs under them
            ops = gemv.loads(c["lane"], c["wave"], w_qb, bid * gemv.ROWS, sh.q_lora)
            if bid < s:
                stage_k(c, bid)
            # the token rows 16 at a time against the same wq_b rows; a group's
            # chunks all issued before any is stored, the next group's issued
            # before this one's MFMAs
            groups = [(t0, min(gemv.ROWS, s - t0)) for t0 in range(0, s, gemv.ROWS)]

            def group_loads(t0, sg):
                rows = qr + fx.Int64(t0) * fx.Int64(qr_ld) * fx.Int64(2)
                return gemv.rows_loads(tid, rows, sg, sh.q_lora, qr_ld, cm=0)

            xs = group_loads(*groups[0])
            for i, (t0, sg) in enumerate(groups):
                gemv.rows_store(tid, xs, sg, sh.q_lora, c["xl"], xrow)
                gpu.barrier()
                if i + 1 < len(groups):
                    xs = group_loads(*groups[i + 1])
                stage_qgemv(c, bid, ops, t0, sg)
            publish(c["put"], c["qrdy"], bid, fx.Int32(1), tid == 0)
            stamp(on, tls, tid, 1)
            stage_iquant(c, bid, s)
        else:
            if bid < s:
                stage_k(c, bid)
        stamp(on, tls, tid, 2)
        for task in range(bid, total, BLOCKS):
            q, t = isel.locate_score(c, task)
            isel.score_task(c, q, t, lg)
        stamp(on, tls, tid, 3)
        # a long row's segments next, one a CTA, all resident: each waits on
        # its own units' scores, then on its row's other segments every pass
        stotal, cps_lg = isel.segment_plan(c)
        if bid < stotal:
            r, g = isel.locate_segment(c, bid)
            isel.stage_segment(c, r, g, bid, cps_lg)
            stamp(on, tls, tid, 6)
        # a row's other slots last: a short row waits on the earlier long rows'
        # counts only, a replicated long row on every rank's winners
        if bid >= BLOCKS - s:
            isel.stage_row(c, BLOCKS - 1 - bid)
        stamp_flush(on, tls, plain + fx.Int64(plain_lay["tl"][0]), tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        qr: Int64,
        qr_ld: Int32,
        w_qb: Int64,
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
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(indexer, launch)
    _BUILDS[key] = launch
    return launch
