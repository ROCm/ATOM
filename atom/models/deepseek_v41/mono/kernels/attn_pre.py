# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""K1 ``attn_pre``: one V4.1 layer from the attention seam to the rotated query.

One launch a layer and rank, ``BLOCKS`` CTAs x ``THREADS`` threads, S = ``tokens``
rows (a decode step). Stages, each a task table placed round the CTAs and
handed off through the tagged mailbox (device scope; across GPUs only in a
token-parallel build, below):

    slice   (160, 32 hidden columns x 4 streams each): the owed post folded into
            the residual (R_new, bf16 out), the collapsed layer input (mailbox
            LIN) and the partial mix projection / sum of squares (mailbox PMIX)
    norm    (S): attn_norm of LIN -> bf16, then quantize_fp8 of it (mailbox X8);
            none in an ordinary sharded layer (``AttnPreBuild.slice_x8``),
            whose slices quantize x for wqkv_a themselves
    wqkv_a  (112, 16 rows): FP8 GEMV -> the q latent | kv row (mailbox QKV)
    gate    (S): the 160 partials -> post / comb (Sinkhorn) and the next pre:
            the next seam's, so after the stages a CTA may hold with it
    qkv     (S): q_norm -> MXFP8 (mailbox QX8); kv_norm -> RoPE -> FP8 QAT ->
            the window ring row
    wq_b    (the rank's heads x 512 / 16, 16 rows: 512 at TP4): FP8 GEMV -> bf16
            -> RoPE on each head's tail -> q out

Token parallel (``tok_par``, S a multiple of TP; the owed post arrives
reduce-scattered): slice .. qkv run on this rank's tokens (``own_view``), qkv
pushes each token's q latent to every rank's QG / QGS (wq_b reads every
token's from there) and its window row to every other rank's KG, which the
launch's last stage writes into the ring; an indexer layer's iw runs on this
rank's tokens too (head weights to every rank's IWG) and its normed rows reach
every rank's ``normed`` through NBG, for the compressor.

Every elementwise step follows the original seam's rounding (aiter's
``mhc_fused_post_pre_delayed_rmsnorm``, then ATOM's ``quantize_fp8``); the mix
projection and the sums of squares, split-K reductions, differ only in their
split and order.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.config import MAX_ROWS
from atom.models.deepseek_v41.mono.kernels.debug import mailbox
from atom.models.deepseek_v41.mono.kernels.dims import (
    HC,
    HEAD_DIM,
    HIDDEN,
    KV_DIM,
    Q_RANK,
    Dims,
    kvrow_scale0,
)
from atom.models.deepseek_v41.mono.kernels.index_query import (
    IDX_ROPE,
    IDX_ROWS,
    IQ_IN_PLACE_MAX,
    IW_TASKS,
    iq_group_fp4,
    iq_group_weight,
    iq_in_place,
    iw_tasks,
    stage_iquant,
    stage_iquant_fp4,
    stage_iw,
    step_queried,
)
from atom.models.deepseek_v41.mono.kernels.index_query import (
    scratch_pairs as indexer_pairs,
)
from atom.models.deepseek_v41.mono.kernels.mx import AMAX_FLOOR, code_round, fp8_value
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.kernels.red import acc_sum, acc_word
from atom.models.deepseek_v41.mono.kernels.token_par import (
    EMBED_ROW_BYTES,
    TOKEN_ROW_BYTES,
    own_view,
)
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.mx import (
    FP8_MAX,
    UNIT_SCALE,
    clamp_fp8,
    code_ceil,
    mfma_scaled,
    pow2,
    rcp_pow2,
)
from atom.mono.device.numa import region_addrs
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    batched_rounds,
    bf16_pair,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    div_rn,
    first_rounds,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    kernel_symbol,
    launcher,
    ld_f32,
    ld_i32,
    load_ptr64,
    rows_to_lds_together,
    rsrc,
    traced,
    xshfl,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import POLL_MAX, Quad, preg, publish
from atom.mono.plan.arith import sel
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES, first_task
from atom.mono.plan.layout import pair_layout
from atom.mono.plan.trace import enter_stage, note
from atom.mono.runtime.abi import KernelAbi

K_MIX = HC * HIDDEN
MIX = HC * (HC + 2)  # 24 projections: pre 4 | post 4 | comb 16
QKV_ROWS = Q_RANK + KV_DIM  # 1792
ROPE = 64
HALF = ROPE // 2
# the index query rotates with the attention's RoPE table (``stage_iq``)
assert IDX_ROPE == ROPE
EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0
LOG2E = 1.0 / math.log(2.0)

COLS = 32  # hidden columns a slice task
SLICES = HIDDEN // COLS  # 160
KT = HC * COLS  # projection columns a slice task
# a token's R_new row and a mix's fn row in LDS (``rl`` / ``fl``), padded: the
# mixes' MFMA reads one column of sixteen rows an instruction, which at a
# multiple of the banks' words all land on one bank
KT_ROW = KT + 4
# a slice's mix weights are whole rounds of the CTA (``slice_fn_loads``)
assert MIX * KT % THREADS == 0
PART = MIX + 1  # a slice's partials a token: the mixes, then the sum of squares
ROWS = 16  # GEMV rows a task
TILE = 16  # an MFMA's N: the tokens a GEMV pass takes (``token_tiles``)
MAX_TOKENS = MAX_ROWS  # a step's rows at most (``config``)
WQKV_TASKS = QKV_ROWS // ROWS  # 112
# q's rows lead QKV: wq_b waits for them only (``stage_q_norm``)
Q_TASKS = Q_RANK // ROWS
X8_WORDS = HIDDEN // 4  # fp8 words of a normed row
QX8_WORDS = Q_RANK // 4

# every stage's first CTA; the norm's have no slice, so its weights go out at
# launch
GATE0 = SLICES
NORM0 = GATE0 + 8
WQKV0 = NORM0 + 8
# a ``slice_x8`` build's: no norm runs, so its gate blocks go onto slice CTAs
# from SLICE_X8_GB0 (each after its slice) and wqkv_a's tasks start at GATE0 --
# q's 96 on the CTAs no slice holds, kv's 16 on slice CTAs 0 .. 15 (their
# weights out only past the slice; kv_norm's path is not wq_b's)
SLICE_X8_GB0 = 16
# a ``slice_x8`` build's rows at most: past 12 (a 16-row tile) wqkv_a's poll of
# every row's quads at once (``load_slice_x8``, 20 a thread) takes the K1's 256
# VGPRs and spills to 28 B of scratch, which the widths refuse (18 and 24 rows
# then took the original path); a wider step keeps the seam norm
SLICE_X8_MAX_TOKENS = 12
QKV0 = (WQKV0 + WQKV_TASKS) % BLOCKS
# kv_norm's tokens on CTAs of their own, apart from q_norm's: in the
# layer kernel between the scores' CTAs and wq_b's (``WQB_LAYER0``)
KVN0 = QKV0 + 32


def norm0(own_n: int, busy: int) -> int:
    """The seam norm's first CTA (a token a CTA): its last CTA right before
    wqkv_a's first (``WQKV0``), so no norm shares a CTA with a wqkv_a task (a
    CTA's wqkv_a loads would queue behind the norm's stores, in vmcnt's issue
    order) -- but never before ``busy``, the first CTA past the slices' and
    a sharded seam's gate blocks (a norm behind its CTA's gate block waits
    for it). Where the CTAs between are fewer than the rows, the last norms
    share wqkv_a's first CTAs."""
    return max(busy, WQKV0 - own_n)


WQB0 = QKV0 + 8
# past one token tile, q's and kv's tasks apart: a q task a CTA, its tiles in
# series on its weights, on CTAs past the norm's (wrapping onto the slices'
# first); kv's (task, tile) units on slice CTAs past q_norm's and kv_norm's
WQKV_Q0 = BLOCKS - 64
WQKV_KV0 = 88
# a compressing layer past one tile (``LayerBuild.q_apart``): the
# compressor's projection on APART_PROJ_CTAS CTAs from the seam norm's first
# (past the slices' and the gate blocks'), the q path on the rest -- q's
# tasks, then kv's, each a CTA its tiles in series --, q_norm and kv_norm on
# projection CTAs (theirs done by then)
APART_PROJ0 = SLICES + 5
APART_PROJ_CTAS = 128
APART_Q0 = (APART_PROJ0 + APART_PROJ_CTAS) % BLOCKS
APART_KV0 = APART_Q0 + Q_TASKS
APART_QN0 = APART_PROJ0 + 35
APART_KVN0 = APART_QN0 + 24
# the layer kernel's wq_b CTAs, from this one: the ones before hold the
# attention's scores (``AttnPostBuild.score0`` on), free to land their keys
# before q
WQB_LAYER0 = 84


def iw0(tok_par: bool, own_n: int) -> int:
    """The head-weight tasks' first CTA: on CTAs that wait for wq_b's q latent
    with nothing to do, where they read only the normed rows. One token tile:
    past kv_norm's CTAs (a token each from ``KVN0``), every later one free up
    to the gates' (``GATE0``); else CTA 96, its 96 tasks (S 17..24) ending
    before the q GEMV's CTAs (``WQKV_Q0``) -- past 24 rows its tasks reach
    them (a slower placement, not a wrong one); a token-parallel launch keeps
    those CTAs busy (0)."""
    if tok_par:
        return 0
    return KVN0 + own_n if len(token_tiles(own_n)) == 1 else 96


IQ_TASKS = IDX_ROWS // ROWS  # 256
IQ0 = IW_TASKS
IQUANT0 = 0
# the in-place FP4 query's group tasks (``_iq_groups``): a 32-dim group (two
# GEMV tasks) a CTA, past q_norm's and kv_norm's CTAs
IQ_GROUP0 = 70
assert IQ_IN_PLACE_MAX == TILE  # one token tile
# a CTA's wq_b tasks whose weights stay in registers through every token tile
# (TP 2's 4; TP 1's 8 held, with an indexer layer's iq task, spilled)
WQB_HELD = 4
# a GEMV task's partials in ``red`` (a wave's lanes, 4 values each), and the
# wq_b tasks of a CTA one ``stage_wqb_group`` takes: ``red`` holds that many
RED_TASK = WAVES * 64 * 4
WQB_GROUP = 3

_CURRENT_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AttnPreBuild:
    tokens: int
    # the seam folds the owed post of the previous sublayer (every layer but 0 and
    # the Engram ones, whose residual arrives settled)
    fold: bool = True
    # FULL / REINDEX layers, whose normed row and q latent feed the indexer and
    # the compressor: the bf16 ``normed`` row out (and NB), the q latent's MXFP8
    # pair out, and the indexer's q / weights projections
    feeds_index: bool = False
    # the index plane is FP4: the indexer query quantized as its scorer reads
    # it (``stage_iquant_fp4``)
    index_fp4: bool = False
    # DSpark tap: the mean over the hc copies of the settled residual -> ``aux``
    aux: bool = False
    # the TP size: every width a rank holds (``Dims``)
    tp: int = 4
    timeline: bool = field(default=False, metadata={"sym": "tl"})
    # ``ATOM_MONO_DEBUG``: bounded mailbox waits recording at this scratch
    # offset (``debug``); -1 is the normal build
    diag_off: int = -1
    # token parallel: the owed post arrives reduce-scattered (each rank its own
    # tokens' rows), slice .. qkv run on this rank's tokens, and their q latent
    # (QG) and window rows (KG) are gathered from every rank
    tok_par: bool = False
    # the pool's rows are MXFP8 (FP8 values then E8M0 scales, ``row_bytes``):
    # the window row is stored so, not as its BF16 QAT
    kv_fp8: bool = False
    # the regions kept in two copies, one on each I/O die (``numa_plan``); the
    # scratch's calibration word that picks a CTA's copy is the ``numa_word``
    # argument, measured at run time
    numa: str = ""
    # the residual column-sharded (``shard_seam``): the seam folds this rank's
    # columns, writing only those of R_new
    shard: bool = False
    # layer 0: ``res_in`` is the embedding [rows, hidden], every stream of the
    # residual; the seam writes the streams to ``res_out`` (no expanded copy
    # before the launch)
    embed_in: bool = False

    @property
    def slice_x8(self) -> bool:
        """An ordinary sharded layer (it folds, feeds no indexer, takes no
        embedding): seam A's slices hand wqkv_a its x themselves, as TK's QKV
        takes it -- bf16(bf16(LIN) w) without the attention rstd (q_norm and
        kv_norm cancel the row's scale), MXFP8 over a slice's 32 columns, each
        word to its destination rank's X8G with its code in its tag
        (``slice_direct_pair``, ``load_slice_x8``). No norm stage, no LIN /
        LSSQ pushes, no X8 flag: wqkv_a waits on the words themselves.
        At most ``SLICE_X8_MAX_TOKENS`` rows: ``load_slice_x8`` polls every
        row's quads at once."""
        return (
            self.shard
            and self.fold
            and not self.feeds_index
            and not self.embed_in
            and self.tokens <= SLICE_X8_MAX_TOKENS
        )

    def __post_init__(self):
        assert not (self.shard and self.tok_par), "a seam is sharded by columns or rows"
        assert not (self.embed_in and self.fold), "the embedding owes no post"
        assert not self.shard or seam_shardable(
            self.tp
        ), f"TP{self.tp}: a rank's slices are not whole gate blocks"


# ``timeline`` stamps: kernel start, then the end of each stage's task loop
TL_STAGES = ("slice", "norm", "wqkv_a", "gate", "qkv", "wq_b", "index")


# the per-token arguments slice .. qkv read and write (``own_view``)
K1_OWN = (
    "res_in", "pend", "post_in", "comb_in", "pre_in", "res_out", "post_out",
    "comb_out", "pre_out", "pos", "ring_rows", "normed", "aux", "qr_out", "qrs_out",
)  # fmt: skip


TL_POINTS = 1 + len(TL_STAGES)


ABI = KernelAbi(
    (
        "res_in", "pend", "post_in", "comb_in", "pre_in",
        "hc_fn", "hc_scale", "hc_base", "attn_w", "wqkv", "wqkv_s", "qn_w", "kvn_w",
        "wqb", "wqb_s", "cos", "sin", "pos", "ring", "ring_scale", "ring_rows", "ring_off",
        "res_out", "post_out", "comb_out", "pre_out", "q_out", "normed", "qr_out",
        "qrs_out", "aux", "iq_w", "iq_ws", "iw_w", "iq_out", "iqs_out", "iw_out",
        "ibound", "scratch", "numa_word", "sym", "peers", "rank", "layer", "tl",
    )
)  # fmt: skip


def scratch_layout(tokens: int) -> dict[str, tuple[int, int]]:
    """Mailbox region -> (byte offset, bytes) in K1's scratch; 8 B a pair."""
    pairs = {
        "lin": tokens * HIDDEN // 2,  # bf16 pairs
        "lssq": tokens * SLICES,  # LIN's sums of squares, a slice task's columns
        "pmix": SLICES * tokens * PART,
        # plain words (``stage_norm``), each token's announced by its flag X8F
        "x8": tokens * X8_WORDS,
        "x8s": tokens * HIDDEN // 32,
        "x8f": tokens,
        "qkv": tokens * QKV_ROWS,
        "qx8": tokens * QX8_WORDS,
        "qx8s": tokens * Q_RANK // 32,
        **indexer_pairs(tokens, HIDDEN),
    }
    return pair_layout(pairs.items())


def scratch_bytes(tokens: int) -> int:
    return sum(n for _, n in scratch_layout(tokens).values())


def fma(a, b, c):
    return fx.Float32(fmath.fma(a, b, c))


def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def ld_bf8(ptr, i):
    """Elements i .. i + 7 of a bf16 buffer as f32, one 16-byte load (``i`` a
    multiple of 8)."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=4, dtype=T.i32))
    out = []
    for k in range(4):
        out += [bf_lo(w[k]), bf_hi(w[k])]
    return out


def token_tiles(s):
    """A step's tokens in GEMV passes: [(first token, tokens)]. A GEMV loads its
    weights once and runs every pass on them, an x tile in LDS at a time."""
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    """The LDS rows an x tile holds."""
    return min(s, TILE)


# a token's LDS row of MXFP8 words / E8M0 codes, 16 of which a MFMA B operand
# gathers, is padded by 16 B: at a multiple of 64 words they share a bank group
LDS_PAD = 4


def lds_row(words):
    return words + LDS_PAD


def lds_at(u, row):
    """Element u of a span of rows ``row`` wide, at the ``lds_row`` stride."""
    return u + u // row * LDS_PAD


def lds_tail(k):
    """Words a K = ``k`` row's last K step reads past its ``lds_row`` (selected
    out): the last row's buffer must hold them."""
    return max(0, 32 * -(-k // 128) - lds_row(k // 4))


def plus(t0, v):
    """Token ``v`` of the tile from ``t0`` (a first tile adds nothing; ``t0``
    traced: a tile a CTA picks at run time)."""
    return v if isinstance(t0, int) and t0 == 0 else v + t0


def tile_tasks(bid, base, ntiles, ntasks):
    """Past one token tile, a stage of ``ntasks`` tasks each run on a token
    tile: this CTA's tile (CTAs placed from ``base``) and its tasks, every
    ``per_tile``-th of those the tile's CTAs share -- a tile's x polled once a
    CTA and every task's weights in flight together, where a (task, tile) a
    unit polled a tile again for each of a CTA's units. A task from past
    ``ntasks`` is not run (the caller's ``< ntasks``)."""
    b = first_task(bid, base)
    tile = b % ntiles
    per_tile = (BLOCKS - tile + ntiles - 1) // ntiles
    most = -(-ntasks // (BLOCKS // ntiles))
    return tile, [b // ntiles + k * per_tile for k in range(most)]


def step_token(c, t):
    """Token t of ``c``'s view as a token of the step: past this rank's first
    in a token-parallel view (``own_view``)."""
    return c.get("t0", 0) + t


def pend_value(c, t, col, cc):
    """The owed post's sublayer output at (token t, column col): read from
    ``pend``, or where a caller staged it (``pend_lds``, [token][32] of this
    task's columns: a reduction it did first)."""
    if const_expr(c.get("pend_lds") is None):
        return ld_bf(c["args"]["pend"], t * HIDDEN + col)
    return fx.ptr_load(c["pend_lds"] + (t * COLS + cc))


def plain(a, kind):
    """Region ``a``'s address for plain stores / loads (a hand-off announced
    by a flag), the access noted as a ``kind`` ("put" / "poll") of the region
    for the contract."""
    assert a.copy is None, f"{a.region}: a dual region is only put and polled"
    note(a.region, a.space, kind)
    return a.value


# ---------------------------------------------------------------- slice
@traced
def put_out(c, kind, name, idx, v):
    """A seam output ``name`` (lin, lssq, pmix) put with ``c[kind]``: a
    sharded seam's task (its slice and destination rank ``c["dst"]``) into
    that rank's gathered copy (``c["fan"][name]``), its mix partials only by
    the task whose destination is this rank (local); else its own region."""
    fan = c["fan"]
    if const_expr(name in fan):
        for p in range_constexpr(len(fan[name])):
            if c["dst"] == p:
                c[kind](fan[name][p], idx, v)
    elif const_expr(c["shard_f"]):
        if c["dst"] == c["rank"]:
            c[kind](c[name], idx, v)
    else:
        c[kind](c[name], idx, v)


@traced
def slice_input(c, task, idx):
    """Thread ``idx``'s pair of the layer input (token idx / 16, columns
    32 task + 2 (idx % 16) ..), the token's LSSQ partial over the task's
    columns (``lin_rstd``) and the drafter's tap."""
    s, rl = c["S"], c["rl"]
    a = c["args"]
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        cc = 2 * (idx % (COLS // 2))
        outs = []
        for d in range_constexpr(2):
            acc = fx.Float32(0.0)
            for h in range_constexpr(HC):
                rb = fx.ptr_load(rl + (t * KT_ROW + h * COLS + cc + d))
                acc = acc + ld_f32(a["pre_in"], t * HC + h) * rb
            outs.append(acc)
        put_out(c, "put_bf", "lin", t * HIDDEN + c0 + cc, outs)
        # the squares of the bf16 values put, summed over the token's 16 lanes
        ys = [bf16_round(v) for v in outs]
        sq = butterfly(ys[0] * ys[0] + ys[1] * ys[1], (1, 2, 4, 8))
        if idx % (COLS // 2) == 0:
            put_out(c, "put", "lssq", t * SLICES + task, sq)
        if const_expr(c["aux"]):
            # the drafter's tap: torch mean over the bf16 streams (sum in order, / 4)
            for d in range_constexpr(2):
                acc = fx.Float32(0.0)
                for h in range_constexpr(HC):
                    acc = acc + fx.ptr_load(rl + (t * KT_ROW + h * COLS + cc + d))
                bo.buffer_store(
                    (acc * fx.Float32(1.0 / HC)).to(fx.BFloat16),
                    rsrc(a["aux"]),
                    t * HIDDEN + c0 + cc + d,
                )


def slice_direct(c) -> bool:
    """Whether ``stage_slice`` folds and collapses in registers (at every
    width): R_new's LDS round trip and the barrier
    between the fold and the layer input are gone."""
    return c["fold"]


def slice_split(s: int) -> int:
    """Adjacent lanes sharing a (row, 2 columns) of the direct fold, a group
    of streams a lane: as many as the CTA's threads hold, at most a stream a
    lane; past the threads one lane a (row, 2 columns), in rounds."""
    split = HC
    while split > 1 and s * (COLS // 2) * split > THREADS:
        split //= 2
    return split


def slice_direct_threads(s: int) -> int:
    """The direct fold's working (virtual) threads: past THREADS, in rounds
    of THREADS (``slice_direct_rounds``)."""
    return s * (COLS // 2) * slice_split(s)


def slice_direct_rounds(s: int) -> int:
    return -(-slice_direct_threads(s) // THREADS)


def from_lane(v, k: int):
    """v of lane ``lane ^ k`` (k < 4: within a lane's stream group)."""
    for bit in (1, 2):
        if k & bit:
            v = xshfl(v, bit)
    return v


def slice_direct_loads(c, c0, q):
    """Thread q's inputs: lane group q / split's (row, columns c0 + 2 ..),
    every stream's residual pair (a word); its own streams' comb columns,
    post and pre gates."""
    a = c["args"]
    split = slice_split(c["S"])
    g = q % split
    p = q // split
    t = p // (COLS // 2)
    col = c0 + 2 * (p % (COLS // 2))
    res = [ld_i32(a["res_in"], ((t * HC + h) * HIDDEN + col) // 2) for h in range(HC)]
    own = range(HC // split)
    o = [g * (HC // split) + j for j in own]
    cm = [
        [ld_f32(a["comb_in"], (t * HC + h) * HC + o[j]) for h in range(HC)] for j in own
    ]
    post = [ld_f32(a["post_in"], t * HC + o[j]) for j in own]
    pre = [ld_f32(a["pre_in"], t * HC + o[j]) for j in own]
    return res, cm, post, pre


@traced
def slice_direct_pair(c, task, q, got, wn0=None):
    """``slice_row`` and ``slice_input`` of thread q's streams of its group's
    (row, 2 columns), from ``slice_direct_loads``' values: its streams' R_new
    (out, and into LDS ``rl`` for the norm's squares and the mixes); on the
    group's first lane, from the others' products, the layer input, its
    squares' sum over the row's 16 groups -> LSSQ, the drafter's tap. The same
    operations in the same order. A ``slice_x8`` build quantizes the layer
    input for wqkv_a instead of putting LIN and LSSQ (its norm weights ``wn0``,
    loaded with the slice's first loads)."""
    s, a, rl = c["S"], c["args"], c["rl"]
    # read here, not in the region below: a method call on ``c`` there would
    # make the rewriter carry ``c`` as the if's state
    slice_x8 = c.get("slice_x8", False)
    split = slice_split(s)
    own = HC // split
    c0 = task * COLS
    if q < slice_direct_threads(s):
        g = q % split
        p = q // split
        t = p // (COLS // 2)
        cc = 2 * (p % (COLS // 2))
        col = c0 + cc
        res, cm, post, pre = got
        r = [[bf_lo(w), bf_hi(w)] for w in res]
        pend = [pend_value(c, t, col + d, cc + d) for d in range(2)]
        new = []
        for j in range_constexpr(own):
            o = g * own + j
            vs = []
            for d in range_constexpr(2):
                v = post[j] * pend[d] + (r[0][d] * cm[j][0])
                for h in range_constexpr(1, HC - 1):
                    v = v + cm[j][h] * r[h][d]
                v = bf16_round(fma(cm[j][HC - 1], r[HC - 1][d], v))
                bo.buffer_store(
                    v.to(fx.BFloat16),
                    rsrc(a["res_out"]),
                    (t * HC + o) * HIDDEN + col + d,
                )
                fx.ptr_store(v, rl + (t * KT_ROW + o * COLS + cc + d))
                vs.append(v)
            new.append(vs)
        # stream h of the group's first lane: lane h / own's value j = h % own
        outs = []
        for d in range_constexpr(2):
            acc = fx.Float32(0.0)
            for h in range_constexpr(HC):
                acc = acc + from_lane(pre[h % own] * new[h % own][d], h // own)
            outs.append(acc)
        ys = [bf16_round(v) for v in outs]
        sq = butterfly(
            ys[0] * ys[0] + ys[1] * ys[1], tuple(split * k for k in (1, 2, 4, 8))
        )
        taps = []
        if const_expr(c["aux"]):
            for d in range_constexpr(2):
                acc = fx.Float32(0.0)
                for h in range_constexpr(HC):
                    acc = acc + from_lane(new[h % own][d], h // own)
                taps.append(acc)
        # the next column pair's values (its group's lanes, ``split`` on): an
        # even pair's first lane puts the 4 columns, one 16-byte put (a quad:
        # an even pair's column a multiple of 4)
        nxt = [xshfl(v, split) for v in outs]
        if const_expr(slice_x8):
            # x as wqkv_a takes it: bf16(LIN w), no rstd; the group's 32
            # columns' amax (the squares' strides) -> ATOM's quantize_fp8 code
            wn = (
                wn0
                if wn0 is not None
                else [ld_bf(a["attn_w"], col + d) for d in range(2)]
            )
            yn = [bf16_round(ys[d] * wn[d]) for d in range(2)]
            am = butterfly(
                fx.max(abs(yn[0]), abs(yn[1])),
                tuple(split * k for k in (1, 2, 4, 8)), fx.max,
            )  # fmt: skip
            code = code_ceil(
                fx.max(am, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX)
            )
            inv = rcp_pow2(code)
            f8 = [clamp_fp8(yn[d] * inv) for d in range(2)]
            f8n = [xshfl(v, split) for v in f8]
            word = fp8_pack4(f8[0], f8[1], f8n[0], f8n[1])
        if (g == 0) & (p % 2 == 0):
            if const_expr(slice_x8):
                # the 4 columns' word, its group's code in its tag
                c["put_coded"](
                    c["x8_dst"], c["x8g_pair0"] + t * X8_WORDS + col // 4, word, code
                )
            else:
                put_out(c, "put_bf", "lin", Quad((t * HIDDEN + col) // 4), outs + nxt)
        if g == 0:
            for d in range_constexpr(len(taps)):
                bo.buffer_store(
                    (taps[d] * fx.Float32(1.0 / HC)).to(fx.BFloat16),
                    rsrc(a["aux"]),
                    t * HIDDEN + col + d,
                )
        if const_expr(not slice_x8):  # noqa: SIM102
            if q % (COLS // 2 * split) == 0:
                put_out(c, "put", "lssq", t * SLICES + task, sq)


def slice_at(c0, e):
    """Element e of a slice task at column c0: (token, stream, hidden column)."""
    return e // KT, (e % KT) // COLS, c0 + e % COLS


def slice_loads(c, c0, e):
    """Element e's loads of ``stage_slice`` (e clamped into the step's)."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(c.get("embed_in")):
        # the embedding: every stream's value (``slice_row`` writes the stream)
        return ld_bf(a["res_in"], t * HIDDEN + col)
    if const_expr(not c["fold"]):
        # a settled residual: read as it is, and left where it is
        return ld_bf(a["res_in"], (t * HC + o) * HIDDEN + col)
    r = [ld_bf(a["res_in"], (t * HC + h) * HIDDEN + col) for h in range(HC)]
    cm = [ld_f32(a["comb_in"], (t * HC + h) * HC + o) for h in range(HC)]
    return r, cm, ld_f32(a["post_in"], t * HC + o)


def slice_row(c, c0, e, got):
    """Element e of ``stage_slice``'s R_new from ``slice_loads``' values."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(c["fold"]):
        r, cm, post = got
        # aiter's mhc_fused_post_pre_delayed_rmsnorm: ((post x + c0 r0)
        # + c1 r1) + c2 r2, each product rounded, then fma(c3, r3, ..)
        v = post * pend_value(c, t, col, e % COLS) + (r[0] * cm[0])
        for h in range_constexpr(1, HC - 1):
            v = v + cm[h] * r[h]
        # bf16 R_new is what the collapse, the sums of squares and the
        # mix projection all read
        v = bf16_round(fma(cm[HC - 1], r[HC - 1], v))
        bo.buffer_store(
            v.to(fx.BFloat16), rsrc(a["res_out"]), (t * HC + o) * HIDDEN + col
        )
    else:
        v = got
        if const_expr(c.get("embed_in")):
            bo.buffer_store(
                v.to(fx.BFloat16), rsrc(a["res_out"]), (t * HC + o) * HIDDEN + col
            )
    fx.ptr_store(v, c["rl"] + (t * KT_ROW + o * COLS + e % COLS))


def slice_fn_loads(c, task):
    """``task``'s columns of the mix projection's weights, a thread's
    ``MIX KT / THREADS``: no wait decides them, so they go out first."""
    a = c["args"]
    c0 = task * COLS
    fns = []
    for i in range_constexpr(MIX * KT // THREADS):
        e = c["tid"] + THREADS * i
        j = e // KT
        o = (e % KT) // COLS
        fns.append(ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS))
    return fns


# a slice of at most this many rows holds its weights through K2a's wait as
# well; wider, those registers spill
SLICE_FN_EARLY_MAX = 24


def slice_issued(c, task, pre):
    """``stage_slice``'s (weights, first rows' loads) issued ahead of its rows:
    what the caller issued (``pre``; weights None: loaded after the rows), or
    else the weights now."""
    return (slice_fn_loads(c, task), None) if pre is None else pre


def slice_prefetch(c, task):
    """``stage_slice``'s first residual / gate loads for ``task`` and, up to
    ``SLICE_FN_EARLY_MAX`` rows, its weights, issued before a wait they do not
    depend on (K2a's: the attention partials)."""
    c0 = task * COLS
    if const_expr(slice_direct(c)):
        rows = slice_direct_loads(
            c, c0, fx.min(c["tid"], slice_direct_threads(c["S"]) - 1)
        )
    else:
        rows = first_rounds(c["tid"], c["S"] * KT, lambda e: slice_loads(c, c0, e))
    fns = slice_fn_loads(c, task) if c["S"] <= SLICE_FN_EARLY_MAX else None
    return fns, rows


def store_fns(c, fns):
    """The mix weights into LDS ``fl``: the original's two bf16 MFMAs take fn
    as hi = bf16(fn) and lo = bf16(fn - hi), whose sum, exact in fp32, is the
    weight it applies."""
    tid, fl = c["tid"], c["fl"]
    for i in range_constexpr(MIX * KT // THREADS):
        hi = bf16_round(fns[i])
        e = tid + THREADS * i
        fx.ptr_store(hi + bf16_round(fns[i] - hi), fl + (e // KT * KT_ROW + e % KT))


@traced
def stage_slice(c, task, pre=None):
    """R_new, the layer input and the partial projections of hidden columns
    32 task .. +32, all four streams, every token. ``pre``: its weights and
    first loads, already issued (``slice_prefetch``)."""
    s, tid = c["S"], c["tid"]
    c0 = task * COLS
    direct = slice_direct(c)
    slice_x8 = c.get("slice_x8", False)
    wns = None
    if const_expr(slice_x8 and direct):
        # the norm weights of each round's columns, which no wait decides, out
        # before the slice's own loads: at the quantization they were a load
        # on every slice's path
        last0 = slice_direct_threads(s) - 1
        split0 = slice_split(s)
        wns = []
        for r in range_constexpr(slice_direct_rounds(s)):
            q = fx.min(plus(tid, THREADS * r), last0)
            col = c0 + 2 * (q // split0 % (COLS // 2))
            wns.append([ld_bf(c["args"]["attn_w"], col + d) for d in range(2)])
    fns, rows = slice_issued(c, task, pre)
    if const_expr(direct):
        # every round's loads first, then each round's fold
        last = slice_direct_threads(s) - 1
        got = [
            rows if r == 0 and rows is not None
            else slice_direct_loads(c, c0, fx.min(plus(tid, THREADS * r), last))
            for r in range(slice_direct_rounds(s))
        ]  # fmt: skip
        for r in range_constexpr(len(got)):
            slice_direct_pair(c, task, plus(tid, THREADS * r), got[r],
                              None if wns is None else wns[r])  # fmt: skip
    else:
        batched_rounds(
            tid, s * KT, lambda e: slice_loads(c, c0, e),
            lambda e, got: slice_row(c, c0, e, got), first=rows,
        )  # fmt: skip
    store_fns(c, slice_fn_loads(c, task) if fns is None else fns)
    gpu.barrier()
    # the layer input: bf16 R_new of the four streams by the incoming pre-mix,
    # products rounded, summed in order (the Triton collapse, no contraction);
    # two columns a thread, one bf16 pair a mailbox store (direct: put above)
    for pas in range_constexpr(0, 0 if direct else s * COLS // 2, THREADS):
        slice_input(c, task, plus(pas, tid))
    if const_expr(c["dst"] is None):
        slice_tail(c, task)
    else:
        own_slice_tail(c, task)


@traced
def own_slice_tail(c, task):
    """A sharded seam's (slice, destination) task: the mixes by the task whose
    destination is this rank only; the others are done once LIN is put."""
    if c["dst"] == c["rank"]:
        slice_tail(c, task)


@traced
def slice_tail(c, task):
    """``stage_slice`` past LIN: the sums of squares and the mix partials."""
    s, lane, wave, rl = c["S"], c["lane"], c["wave"], c["rl"]
    # sum of squares: token t on wave t % WAVES, two columns a lane
    for t in range_constexpr(s):
        if wave == t % WAVES:
            x0 = fx.ptr_load(rl + (t * KT_ROW + 2 * lane))
            x1 = fx.ptr_load(rl + (t * KT_ROW + 2 * lane + 1))
            sq = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
            if lane == 0:
                put_out(c, "put", "pmix", (task * s + t) * PART + MIX, sq)
    # mixes: fp32 MFMA 16x16x4, A = fn rows (two 16-row tiles), B = R_new^T,
    # a token tile at a time
    for t0, n in token_tiles(s):
        slice_mix(c, task, t0, n)


@traced
def slice_mix(c, task, t0, n):
    """The mix projection partials of tokens t0 .. t0 + n (``stage_slice``)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl, red = c["rl"], c["fl"], c["red"]
    tok = plus(t0, fx.min(lane % 16, n - 1))
    kw = KT // WAVES
    cs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for kk in range_constexpr(kw // 4):
        colk = wave * kw + lane // 16 + 4 * kk
        b = fx.ptr_load(rl + (tok * KT_ROW + colk))
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            av = fx.ptr_load(fl + (fx.min(row, MIX - 1) * KT_ROW + colk))
            av = (row < MIX).select(av, fx.Float32(0.0))
            ops = rocdl._split_mfma_operands([av, b, cs[h], 0, 0, 0])
            cs[h] = fx.Vector(rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result)
    for h in range_constexpr(2):
        fx.ptr_store(cs[h], red + ((wave * 2 + h) * 64 + lane) * 4)
    gpu.barrier()
    if tid < MIX * n:
        tl = tid // MIX
        t = plus(t0, tl)
        j = tid % MIX
        h = j // 16
        ln = 16 * ((j % 16) // 4) + tl
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((w * 2 + h) * 64 + ln) * 4 + j % 4))
        put_out(c, "put", "pmix", (task * s + t) * PART + j, tot)
    gpu.barrier()


# ---------------------------------------------------------------- gate
GATE_BLOCK = 8  # slices a gate thread sums


def shard_gate_block(tp: int) -> int:
    """Slices a sharded seam's gate block sums (``stage_gate_block``, on their
    owner): GATE_BLOCK, or half of it where a rank's slices are not whole
    blocks of it (TP8's 20 a rank: blocks of 4)."""
    own = SLICES // tp
    return GATE_BLOCK if own % GATE_BLOCK == 0 else GATE_BLOCK // 2


def seam_shardable(tp: int) -> bool:
    """Whether a column-sharded seam serves ``tp`` ranks: a rank's slices are
    whole gate blocks (``shard_gate_block``)."""
    return SLICES % tp == 0 and (SLICES // tp) % shard_gate_block(tp) == 0


# comb[r][c] at lane COMB0 + 4 r + c: one DPP row of 16 lanes, a row's sum
# lanes xor 1, 2 and a column's xor 4, 8. Both add (e0 + e1) + (e2 + e3), the
# reference's order
COMB0 = 16


def _row_sum(v):
    return butterfly(v, (1, 2))


def _col_sum(v):
    return butterfly(v, (4, 8))


def stage_gate_block(c, g):
    """A sharded seam's gate block ``g`` (slices gb g .., this rank's, gb
    ``shard_gate_block``): every token's mix partials and squares summed over
    its slices in slice order (``stage_gate``'s first sums) into every rank's
    ``pmixg``."""
    s = c["S"]
    for pas in range_constexpr(0, s * PART, THREADS):
        gate_block_pair(c, g, plus(pas, c["tid"]))


@traced
def gate_block_pair(c, g, idx):
    """Thread ``idx``'s (token, partial) of ``stage_gate_block``."""
    s = c["S"]
    gb = shard_gate_block(c["d"].tp)
    if idx < s * PART:
        t = idx // PART
        j = idx % PART
        vals = c["poll"](
            [(c["pmix"], ((g * gb + k) * s + t) * PART + j, 1) for k in range(gb)]
        )
        acc = vals[0][0].bitcast(fx.Float32)
        for k in range_constexpr(1, gb):
            acc = acc + vals[k][0].bitcast(fx.Float32)
        for dst in c["pmixg_fan"]:
            c["put"](dst, (g * s + t) * PART + j, acc)


@traced
def gate_group_sum(c, t, gb, groups, idx):
    """``stage_gate``'s thread ``idx``: token t's group g of ``gb`` slices,
    partial j (idx = g PART + j), summed in slice order into ``red`` -- a
    sharded seam's from its owner (``stage_gate_block``)."""
    s = c["S"]
    if idx < groups * PART:
        g = idx // PART
        j = idx % PART
        if const_expr(c["shard_f"]):
            acc = c["poll"]([(c["pmixg"], (g * s + t) * PART + j, 1)])[0][0]
            acc = acc.bitcast(fx.Float32)
        else:
            vals = c["poll"](
                [(c["pmix"], ((g * gb + k) * s + t) * PART + j, 1) for k in range(gb)]
            )
            acc = vals[0][0].bitcast(fx.Float32)
            for k in range_constexpr(1, gb):
                acc = acc + vals[k][0].bitcast(fx.Float32)
        fx.ptr_store(acc, c["red"] + (g * PART + j))


@traced
def stage_gate(c, t):
    """Token t's 160 slices, in slice order, -> post / comb / next pre."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    shard = c["shard_f"]
    gb = shard_gate_block(c["d"].tp) if shard else GATE_BLOCK
    groups = SLICES // gb
    for pas in range_constexpr(0, groups * PART, THREADS):
        gate_group_sum(c, t, gb, groups, plus(pas, tid))
    gpu.barrier()
    if wave == 0:
        # lane -> its mix: pre / post on lanes 0 .. 7, comb from lane COMB0
        j = (lane < COMB0).select(
            lane % (2 * HC), fx.min(lane - COMB0 + 2 * HC, MIX - 1)
        )
        mix = fx.ptr_load(red + j)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + j))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        sc = [ld_f32(a["hc_scale"], i) for i in range(3)]
        base = ld_f32(a["hc_base"], j)
        # aiter's mhc_fused_post_pre_delayed_rmsnorm reduce: one rstd =
        # rsq(fma(sum, 1/K, eps)), then each gate fma(mix rstd, scale, base)
        v = mix * hw_rsq(fma(sq, fx.Float32(1.0 / K_MIX), fx.Float32(EPS)))
        if lane < HC:
            pre = sigmoid(fma(v, sc[0], base))
            bo.buffer_store(pre + fx.Float32(HC_EPS), rsrc(a["pre_out"]), t * HC + lane)
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(fma(v, sc[1], base))
            bo.buffer_store(
                post * fx.Float32(POST_MULT), rsrc(a["post_out"]), t * HC + lane - HC
            )
        cv = fma(v, sc[2], base)
        # Triton's exp: exp2 of (x - max) log2(e)
        m = butterfly(cv, (1, 2), fx.max)
        cv = hw_exp2((cv - m) * fx.Float32(LOG2E))
        # fast_dividef is x rcp(y); its first "+ eps" contracts into the fma
        cv = fma(cv, hw_rcp(_row_sum(cv)), fx.Float32(HC_EPS))
        cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        for _ in range_constexpr(SINKHORN - 1):
            cv = cv * hw_rcp(_row_sum(cv) + fx.Float32(HC_EPS))
            cv = cv * hw_rcp(_col_sum(cv) + fx.Float32(HC_EPS))
        if (lane >= COMB0) & (lane < COMB0 + HC * HC):
            bo.buffer_store(cv, rsrc(a["comb_out"]), t * HC * HC + lane - COMB0)
    gpu.barrier()


# ---------------------------------------------------------------- norm
# a lane's LSSQ partials of a token (``lin_rstd``)
LSSQ_PER_LANE = -(-SLICES // 64)


def lssq_specs(lssq, t, lane):
    """The poll specs of token t's LSSQ partials lane ``lane`` sums
    (``lin_rstd``): lane + 64 k, clamped past the last."""
    return [
        (lssq, t * SLICES + fx.min(lane + 64 * k, SLICES - 1), 1)
        for k in range(LSSQ_PER_LANE)
    ]


def lin_rstd(lane, got):
    """A token's rstd = rsq(fma(sum, 1/H, eps)) from its LSSQ partials polled
    by ``lssq_specs`` (every lane of a wave alike): a lane's in index order,
    then the wave's butterfly. The one order the seam's norm and the router
    (which normalizes its own K part) both take, so both get the same bits."""
    v = got[0][0].bitcast(fx.Float32)
    for k in range_constexpr(1, LSSQ_PER_LANE):
        q = got[k][0].bitcast(fx.Float32)
        if const_expr(64 * (k + 1) > SLICES):
            q = (lane + 64 * k < SLICES).select(q, fx.Float32(0.0))
        v = v + q
    tot = butterfly(v, (1, 2, 4, 8, 16, 32))
    return hw_rsq(fma(tot, fx.Float32(1.0 / HIDDEN), fx.Float32(EPS)))


@traced
def stage_norm(c, t):
    """The seam's RMSNorm (rstd ``lin_rstd``, bf16((x rstd) w); 256
    threads, 3 chunks of 8 at 8 t + 2048 c), then ATOM's quantize_fp8 of the bf16
    row (amax floored at AMAX_FLOOR, the ceil code, x / scale) -> x8 words and codes;
    the FFN seam (``ffn``, whose x8 K2b quantizes itself) stores the bf16 row as
    ``normed`` instead. A layer that feeds its indexer (``index``) stores it as
    ``normed`` too and puts it in NB. Threads 256.. mirror thread 255 and store
    nothing."""
    tid, lane = c["tid"], c["lane"]
    a = c["args"]
    # the row's store policy: device scope when this launch also reads it
    normed_cm = c.get("normed_cm", 0)
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + 2048 * ch for ch in range(3)]
    # the norm weights, which the row does not decide: loaded before its wait
    wts = [ld_bf8(a["attn_w"], fx.min(cols[ch], HIDDEN - 8)) for ch in range(3)]
    bases = [t * HIDDEN + fx.min(cols[ch], HIDDEN - 8) for ch in range(3)]
    # a chunk's 8 columns: 4 bf16-pair words, two 16-byte polls (two quads: the
    # base a multiple of 8); with the LSSQ partials one poll batch (``POLL_MAX``)
    words = c["poll"](
        [(c["lin"], Quad(bases[ch] // 4 + k), 2) for ch in range(3) for k in range(2)]
        + lssq_specs(c["lssq"], t, lane)
    )
    xs = []
    for ch in range_constexpr(3):
        v = []
        for w in words[2 * ch] + words[2 * ch + 1]:
            v += [bf_lo(w), bf_hi(w)]
        xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    r = lin_rstd(lane, words[6:])
    rows = []
    for ch in range_constexpr(3):
        col = fx.min(cols[ch], HIDDEN - 8)
        ys = [bf16_round((xs[ch][j] * r) * wts[ch][j]) for j in range(8)]
        amax = fx.Float32(0.0)
        for y in ys:
            amax = fx.max(amax, abs(y))
        amax = fx.max(butterfly(amax, (1, 2), fx.max), fx.Float32(AMAX_FLOOR))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        inv = rcp_pow2(code)
        if mine & (cols[ch] < HIDDEN):
            if const_expr(c["index"] or c["ffn"]):
                # the thread's 8 bf16 as one 16-byte store (col is 8-aligned)
                normed_words = [
                    bf16_pair(ys[2 * q], ys[2 * q + 1]).bitcast(fx.Int32)
                    for q in range(4)
                ]
                bo.buffer_store(
                    fx.Vector.from_elements(normed_words, fx.Int32),
                    rsrc(a["normed"]),
                    (t * HIDDEN + col) // 2,
                    cache_modifier=normed_cm,
                )
            if const_expr(c["index"]):
                # the indexer's weights_proj reads the bf16 row
                q = (t * HIDDEN + col) // 4
                c["put_bf"](c["nb"], Quad(q), ys[0:4])
                c["put_bf"](c["nb"], Quad(q + 1), ys[4:8])
            if const_expr(not c["ffn"]):
                w0 = fp8_pack4(*[clamp_fp8(ys[i] * inv) for i in range(4)])
                w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * inv) for i in range(4)])
                put_x8(c, t, col, tt, w0, w1, code)
        rows.append((col, ys))
    if const_expr(not c["ffn"]):
        # the token's words drained, then its flag
        publish(c["put"], c["x8f"], t, 1, tid == 0)
    else:
        gpu.barrier()
    if const_expr(c["index"] and c["tok_par"]):
        # token parallel: the row for every other rank's compressor, pushed
        # past the flag (its drain would wait for these remote stores' acks:
        # vmcnt counts them with the local ones); their reader polls the pairs
        tg = c["t0"] + t
        for ch in range_constexpr(3):
            col, ys = rows[ch]
            if mine & (cols[ch] < HIDDEN):
                for p in range_constexpr(c["d"].tp):
                    if c["rank"] != p:
                        dst = c["nb_dsts"][p]
                        c["put_bf"](dst, tg * HIDDEN + col, ys[0:4])
                        c["put_bf"](dst, tg * HIDDEN + col + 4, ys[4:8])


@traced
def put_x8(c, t, col, tt, w0, w1, code):
    """Token t's x8 words at ``col`` and (a group's first thread) its code, as
    plain words (device scope): X8 goes to wqkv_a by flag a token (X8F), a
    wqkv_a task polling a flag a token rather than every row's tagged pairs in
    POLL_MAX batches."""
    at, gat = t * X8_WORDS + col // 4, t * (HIDDEN // 32) + col // 32
    bo.buffer_store(
        fx.Vector.from_elements([w0, w1], fx.Int32), rsrc(plain(c["x8"], "put")),
        at, cache_modifier=CM_DEV,
    )  # fmt: skip
    if tt % 4 == 0:
        bo.buffer_store(
            fx.Int32(code), rsrc(plain(c["x8s"], "put")), gat,
            cache_modifier=CM_DEV,
        )  # fmt: skip


# ---------------------------------------------------------------- GEMV
def _fp8_step_load(c, r_w, r_ws, k, rg, st):
    """K step st's global operands of rows 16 rg ..: the two 64-column weight
    chunks (the upper clamped into the row) and the raw scale word. The (16, 16)
    shuffle lays a row group out in 32-column blocks of 16 rows, a chunk two of
    them (lanes 0-31, 32-63); ``k`` % 64 = 32 (a last quarter step): the lanes
    of a block past the row read its last block."""
    lane = c["lane"]
    j = lane // 16
    kg = k // 32
    if const_expr(k % 64 == 0):
        lo_at = (rg * (k // 64) + 2 * st) * 256 + lane * 4
    else:
        lo_at = rg * (k * 4) + fx.min(4 * st + lane // 32, kg - 1) * 128 + lane % 32 * 4
    lo = fx.Vector(
        bo.buffer_load(r_w, lo_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    if const_expr(k % 64 == 0):
        hi_at = (rg * (k // 64) + fx.min(2 * st + 1, k // 64 - 1)) * 256 + lane * 4
    else:
        hi_at = rg * (k * 4) + fx.min(2 * st + 1, k // 64 - 1) * 256 + lane * 4
    hi = fx.Vector(
        bo.buffer_load(r_w, hi_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    wsi = (rg // 2) * kg + fx.min(st * 4 + j, kg - 1)
    word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF


def _fp8_step_mfma(c, k, xl, xsl, st, ops, acc, live=None, rows=None):
    """``acc`` through K step st's scaled MFMA: ``_fp8_step_load``'s operands
    against every token's MXFP8 row in LDS. ``live`` false: a step past the end
    (its operands those of the last step), zero at unit scale. ``k`` % 64 = 32:
    a lane's lower block past the row is zero on both sides."""
    s, lane = c["S"] if rows is None else rows, c["lane"]
    lo, hi, sa = ops
    kg = k // 32
    x_row, s_row = lds_row(k // 4), lds_row(kg)
    j = lane // 16
    col = fx.min(lane % 16, s - 1)
    if const_expr(live is not None):
        st = fx.min(st, (k + 127) // 128 - 1)
        lo = [live.select(lo[d], fx.Int32(0)) for d in range(4)]
    past = (2 * st + 1) * 64 >= k
    if const_expr(live is not None):
        past = past | ~live
    av = fx.Vector.from_elements(
        [lo[d] for d in range(4)] + [past.select(fx.Int32(0), hi[d]) for d in range(4)],
        fx.Int32,
    )
    grp = st * 4 + j
    xlo = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xhi = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + 16 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    if const_expr(k % 64 != 0):
        over = 4 * st + j // 2 >= kg
        av = fx.Vector.from_elements(
            [over.select(fx.Int32(0), av[d]) for d in range(4)]
            + [av[4 + d] for d in range(4)],
            fx.Int32,
        )
        xlo = [over.select(fx.Int32(0), xlo[d]) for d in range(4)]
    xv = fx.Vector.from_elements(
        [xlo[d] for d in range(4)]
        + [past.select(fx.Int32(0), xhi[d]) for d in range(4)],
        fx.Int32,
    )
    sb = fx.ptr_load(xsl + (col * s_row + fx.min(grp, kg - 1)))
    unit = grp >= kg
    if const_expr(live is not None):
        unit = unit | ~live
    sa = unit.select(fx.Int32(UNIT_SCALE), sa)
    sb = unit.select(fx.Int32(UNIT_SCALE), sb)
    return mfma_scaled(av, xv, acc, sa, sb)


def gemv_fp8(c, w, ws, k, rg, xl, xsl, red, split=None):
    """Rows 16 rg .. +16 of the (16, 16)-preshuffled FP8 weight ``w`` (32x32 e8m0
    ``ws``) times every token's MXFP8 row in LDS (``xl`` words, ``xsl`` codes as
    i32) -> red[wave][lane] partial C: ``gemv_fp8_loads`` then
    ``gemv_fp8_mfmas``. A stage whose input it must wait for issues the loads
    itself first, so the weights' latency overlaps the wait."""
    ops = gemv_fp8_loads(c, w, ws, k, rg, split)
    gemv_fp8_mfmas(c, k, xl, xsl, red, ops, split)


def _gemv_steps(c, k, split, tiled=False):
    """This wave's K steps (a step past the end marked, ``split`` waves past
    split repeat the last one's; ``tiled``: wave w takes part w % split of tile
    w / split): [(step, live)]."""
    wave = c["wave"]
    steps = (k + 127) // 128
    if const_expr(split is None):
        per = (steps + WAVES - 1) // WAVES
        ragged = steps % WAVES != 0
        return [
            (wave + WAVES * i, (wave + WAVES * i < steps) if ragged else None)
            for i in range(per)
        ]
    assert split <= WAVES and steps % split == 0
    per = steps // split
    part = wave % split if tiled else fx.min(wave, split - 1)
    return [(part * per + i, None) for i in range(per)]


def _fp8_part_load(c, r_w, r_ws, k, k_row, rg, k0, st):
    """``_fp8_step_load`` of a part of the rows: step st of the ``k`` columns
    from column ``k0`` (a multiple of 32: a scale block; traced, a rank's or a
    task's part) of rows ``k_row`` long. The shuffle's 32-column blocks of a
    row group are contiguous, so the part is the whole-row layout shifted by
    its first block, and clamped within the part (not the row)."""
    lane = c["lane"]
    j = lane // 16
    kg = k // 32
    base = rg * (k_row * 4) + k0 * 4
    if const_expr(k % 64 == 0):
        lo_at = base + st * 512 + lane * 4
    else:
        lo_at = base + fx.min(4 * st + lane // 32, kg - 1) * 128 + lane % 32 * 4
    lo = fx.Vector(
        bo.buffer_load(r_w, lo_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    hi_at = base + fx.min(2 * st + 1, k // 64 - 1) * 256 + lane * 4
    hi = fx.Vector(
        bo.buffer_load(r_w, hi_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    wsi = (rg // 2) * (k_row // 32) + k0 // 32 + fx.min(st * 4 + j, kg - 1)
    word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF


def gemv_fp8_loads(c, w, ws, k, rg, split=None, tiled=False, k_row=None, k0=0):
    """The weight operands of ``gemv_fp8``'s rows for this wave, every one in
    flight at once (a step's latency each would put them in series): loads
    only. A step past the end loads the last step. ``tiled`` (split x tiles =
    WAVES): row groups rg .., one a group of ``split`` waves. ``k_row``: the
    weight's rows are this long, its columns k0 .. the ``k`` taken (a part of
    K: ``Dims.wob_parts``, a rank's share of a replicated weight)."""
    steps = (k + 127) // 128
    r_w, r_ws = rsrc(w), rsrc(ws)
    if const_expr(tiled):
        assert WAVES % split == 0
        rg = rg + c["wave"] // split
    if k_row is None:
        k_row = k
    sts = [
        st if live is None else fx.min(st, steps - 1)
        for st, live in _gemv_steps(c, k, split, tiled)
    ]
    # the whole rows: no column offset to apply (a traced k0 is a part's)
    if const_expr(k_row == k and isinstance(k0, int) and k0 == 0):
        return [_fp8_step_load(c, r_w, r_ws, k, rg, st) for st in sts]
    assert k_row % 64 == 0
    return [_fp8_part_load(c, r_w, r_ws, k, k_row, rg, k0, st) for st in sts]


def gemv_fp8_mfmas(c, k, xl, xsl, red, ops, split=None, tiled=False, rows=None):
    """``gemv_fp8_loads``' operands against the rows in LDS (``rows``: a token
    tile's, every token's by default) ->
    red[wave][lane] in ``red``'s layout (``acc_word``); the scaled 16x16x128 MFMA, chunk kc in dwords 0-3 and kc +
    1 in 4-7. ``k`` a multiple of 32: a last half step has its upper chunk zeroed
    on both sides at unit scale (a NaN byte past the row must not reach the
    MFMA), its loads clamped into the row: the last row group's would read past
    the tensor; a last quarter step its lower chunk's upper block too. A step
    past the end adds zero at unit scale.

    The K order is the original kernel's, which ``red.acc_sum`` completes:
    wave w takes steps w, w + 8, ... (``split`` None: aiter's preshuffled group32
    GEMM, P0.5 microbenchmark 1), or waves 0 .. split - 1 one contiguous K range
    each, the other waves zero (aiter's split-K bmm, whose last split sums the
    partials in split order)."""
    lane, wave = c["lane"], c["wave"]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i, (st, live) in enumerate(_gemv_steps(c, k, split, tiled)):
        acc = _fp8_step_mfma(c, k, xl, xsl, st, ops[i], acc, live, rows)
    if const_expr(split is not None and not tiled):
        acc = fx.Vector.from_elements(
            [(wave < split).select(acc[e], fx.Float32(0.0)) for e in range(4)],
            fx.Float32,
        )
    fx.ptr_store(acc, red + acc_word(wave, lane // 16, lane % 16))


@dataclass(frozen=True)
class SpanAt:
    """Where a ``poll_copy`` span's elements lie in its region: element u at
    pair ``pair(u)``; elements u and u + 1 (u even) at ``Quad`` ``quad(u)``,
    None where the span's rows do not start at a quad (a spec an element)."""

    pair: Callable
    quad: Callable | None = None


def flat_span(first, row_pairs: int) -> SpanAt:
    """Rows of ``row_pairs`` pairs back to back from row ``first`` (traced or
    not): quad-aligned where ``row_pairs`` is even."""

    def pair(u):
        return plus(first * row_pairs, u)

    def quad(u):
        return Quad(plus(first * (row_pairs // 2), u // 2))

    return SpanAt(pair, quad if row_pairs % 2 == 0 else None)


def strided_span(first, stride: int, part, unit: int, width: int) -> SpanAt:
    """Rows ``stride`` pairs apart from row ``first``, each's ``width`` pairs
    from pair ``part`` x ``unit`` (``part`` traced or not): quad-aligned
    where ``stride`` and ``unit`` are even."""

    def pair(u):
        return plus(first, u // width) * stride + part * unit + u % width

    def quad(u):
        row = plus(first, u // width) * (stride // 2)
        return Quad(row + part * (unit // 2) + u % width // 2)

    return SpanAt(pair, quad if stride % 2 == 0 and unit % 2 == 0 else None)


@traced
def poll_copy(c, idx, width, spans, live=None):
    """Mailbox spans [(region, at, n, dst, row)] into LDS: element u (``at``:
    a ``SpanAt``) to dst + u, or ``lds_at(u, row)`` for a span of rows ``row``
    words wide. A spec takes ``span_pairs`` consecutive elements (two where
    the span is quad-aligned and its rows and length even: one 16-byte load a
    spec, half the specs); thread ``idx`` of ``width`` takes spec idx, idx + width, ...,
    POLL_MAX pairs a round trip (``pair_batches``), each batch stored before
    the next is polled (holding them all spills past S = 16); a thread past a
    span's end re-reads its last spec and stores nothing.

    ``live``: a span's elements the reader needs (traced, <= n, a multiple of
    its ``span_pairs``), a span each; a batch past them is not polled and an
    element past them neither polled nor written (one past a step's rows is
    never written)."""
    idx = fresh(idx)  # its elements' indices not hoisted out of a task loop
    if const_expr(live is None):
        _poll_store_batches(c, [
            (region, at, n, (dst, row), k, k * (idx + width * i))
            for region, at, n, dst, row in spans
            for k in [span_pairs(n, row, at)]
            for i in range((n // k + width - 1) // width)
        ])  # fmt: skip
        return
    for (region, at, n, dst, row), need in zip(spans, live):
        k = span_pairs(n, row, at)
        chunks = [
            (region, at, need, (dst, row), k, k * (idx + width * i))
            for i in range((n // k + width - 1) // width)
        ]
        for batch in pair_batches(chunks):
            if batch[0][5] < need:
                _poll_store_batches(c, batch)


def span_pairs(n, row, at: SpanAt):
    """Elements a ``poll_copy`` spec takes of a span of ``n`` (rows ``row``
    words wide, None: one row) at ``at``: two where it is quad-aligned and
    both split evenly, so a spec never straddles a row or the span's end;
    else one."""
    even = n % 2 == 0 and (row is None or row % 2 == 0)
    return 2 if at.quad is not None and even else 1


def pair_batches(chunks):
    """``poll_copy``'s chunks in order, cut into batches of at most POLL_MAX
    pairs (a chunk's ``k`` each): the live words a round trip holds."""
    batches, pairs = [[]], 0
    for ch in chunks:
        if pairs + ch[4] > POLL_MAX:
            batches.append([])
            pairs = 0
        batches[-1].append(ch)
        pairs += ch[4]
    return [b for b in batches if b]


@traced
def _poll_store_batches(c, chunks):
    """``poll_copy``'s chunks [(region, at, n, (dst, row), k, u)] (spec of
    elements u .. u + k), a ``pair_batches`` batch a round trip, each stored
    before the next is polled."""

    def spec(at, u, k):
        return at.pair(u) if k == 1 else at.quad(u)

    for batch in pair_batches(chunks):
        got = c["poll"](
            [(r, spec(at, fx.min(u, n - k), k), k) for r, at, n, _, k, u in batch],
            batch=len(batch),
        )
        for (_, _, n, (dst, row), k, u), v in zip(batch, got):
            at = u if row is None else lds_at(u, row)
            if u < n:
                word = v[0] if k == 1 else fx.Vector.from_elements(v, fx.Int32)
                fx.ptr_store(word, dst + at)


@traced
def load_x8(c, words_mb, codes_mb, words, groups, xl, xsl, t0=0, n=None):
    """Tokens t0 .. t0 + n's MXFP8 rows (every token's by default) from the
    mailbox into LDS rows 0 .. (codes as i32). ``n`` traced (a tile a CTA picks
    at run time): a TILE-row copy of which only n rows are polled."""
    n = c["S"] if n is None else n
    rows = n if isinstance(n, int) else TILE
    poll_copy(
        c,
        c["tid"],
        THREADS,
        [
            (words_mb, flat_span(t0, words), rows * words, xl, words),
            (codes_mb, flat_span(t0, groups), rows * groups, xsl, groups),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
    )
    gpu.barrier()


def load_seam_x8(c, t0=0, n=None):
    """Tokens t0 .. t0 + n's x8 rows into LDS ``xl`` / ``xsl``: the seam
    norm's (``_load_seam_x8``), or a ``slice_x8`` build's slices' words
    (``load_slice_x8``)."""
    if c.get("slice_x8", False):
        n2 = c["S"] if n is None else n
        load_slice_x8(c, t0, n2, n2 if isinstance(n2, int) else TILE)
    else:
        _load_seam_x8(c, t0, n)


@traced
def _load_seam_x8(c, t0=0, n=None):
    """The seam's normed rows (X8 / X8S: ``stage_norm``) of tokens t0 .. t0 + n
    into LDS ``xl`` / ``xsl`` as ``load_x8`` lays them: their flags (X8F)
    polled, then the plain words; ``n`` traced (a tile a CTA picks at run
    time): a TILE-row copy whose rows past n read the last token's flag."""
    n = c["S"] if n is None else n
    rows = n if isinstance(n, int) else TILE
    tid, groups = c["tid"], HIDDEN // 32
    # a flag a thread, then the CTA's barrier: not every thread polling every
    # flag, which puts the CTA's polls in series batches
    if tid < rows:
        c["poll"]([(c["x8f"], fx.min(plus(t0, tid), c["S"] - 1), 1)])
    gpu.barrier()
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Acquire
    )
    x8 = plain(c["x8"], "poll") + fx.Int64(t0 * X8_WORDS * 4)
    x8s = plain(c["x8s"], "poll") + fx.Int64(t0 * groups * 4)
    # the words' and the codes' loads out together: one round trip (the codes'
    # in series after the words' stores was a second)
    rows_to_lds_together(
        tid,
        [
            (x8, rows, X8_WORDS, c["xl"], lds_row(X8_WORDS)),
            (x8s, rows, groups, c["xsl"], lds_row(groups)),
        ],
    )
    gpu.barrier()


def load_slice_x8(c, t0, n, rows):
    """A ``slice_x8`` build's x8: tokens t0 .. t0 + n's words from this rank's
    X8G (every rank's slices put them, ``slice_direct_pair``) into LDS ``xl``,
    and each 32 columns' code from its words' tags into ``xsl``: one poll of
    every thread's quads (one batch at 6 rows), no flag and no second round
    trip for the codes."""
    tid, groups = c["tid"], HIDDEN // 32
    nq = rows * X8_WORDS // 2
    qs = [tid + THREADS * i for i in range(-(-nq // THREADS))]
    specs = []
    for e in qs:
        e = fx.min(e, nq - 1)
        r, w = e // (X8_WORDS // 2), e % (X8_WORDS // 2) * 2
        tok = fx.min(plus(t0, fx.min(r, n - 1)), c["S"] - 1)
        specs.append((c["x8g"], Quad((c["x8g_pair0"] + tok * X8_WORDS + w) // 2), 2))
    got = c["poll_coded"](specs)
    for i, e in enumerate(qs):
        r, w = e // (X8_WORDS // 2), e % (X8_WORDS // 2) * 2
        vals, code = got[i]
        _store_if(e < nq, fx.Vector.from_elements(vals, fx.Int32),
                  c["xl"] + (r * lds_row(X8_WORDS) + w))  # fmt: skip
        # a group's 8 words: its first quad's thread lands the code
        _store_if((e < nq) & (e % 4 == 0), code,
                  c["xsl"] + (r * lds_row(groups) + w // 8))  # fmt: skip
    gpu.barrier()


@traced
def _store_if(live, v, at):
    if live:
        fx.ptr_store(v, at)


@traced
def stage_wqkv(c, task, ops, t0, n):
    """Rows 16 task .. of wqkv_a (weights ``ops``) against tokens t0 .. t0 + n,
    their x in LDS (``load_x8``) -> QKV, two adjacent rows' bf16 a word (the
    norms poll half the pairs)."""
    tid, red = c["tid"], c["red"]
    gemv_fp8_mfmas(c, HIDDEN, c["xl"], c["xsl"], red, ops, rows=n)
    gpu.barrier()
    r = tid % ROWS
    v = acc_sum(red, r, fx.min(tid // ROWS, n - 1))
    # the odd row's value from the next lane (lanes r, r + 1 of a token)
    vn = xshfl(v, 1)
    if (tid < ROWS * n) & (r % 2 == 0):
        t = plus(t0, tid // ROWS)
        c["put_bf"](c["qkv"], t * QKV_ROWS + task * ROWS + r, [v, vn])
    gpu.barrier()


# ---------------------------------------------------------------- q / kv norms
@traced
def stage_q_norm(c, t):
    """Token t: q_norm -> MXFP8 (the Triton dual norm, 4 warps x 8 columns),
    wq_b's input. Polls only q's columns of QKV, so wq_b does not wait for the
    kv columns (``stage_kv_norm``, a task of its own on other CTAs).

    Threads 256.. mirror thread 255's loads (so every wave reaches the barrier)
    and store nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    # every weight the token's rows do not decide, loaded before their wait
    qnw = [ld_bf(a["qn_w"], qc[j]) for j in range(8)]
    # the thread's 8 columns: 4 bf16-pair words, two 16-byte polls
    q = (t * QKV_ROWS + fx.min(tt, Q_RANK // 8 - 1) * 8) // 4
    qv = c["poll"]([(c["qkv"], Quad(q), 2), (c["qkv"], Quad(q + 1), 2)])
    vals = []
    for w in qv[0] + qv[1]:
        vals += [bf_lo(w), bf_hi(w)]
    xq = [((tt * 8 + j) < Q_RANK).select(vals[j], fx.Float32(0.0)) for j in range(8)]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    ssq = butterfly(ssq, (8, 4, 2, 1, 16, 32))
    if (lane == 63) & mine:
        fx.ptr_store(ssq, red + wave)
    gpu.barrier()
    tq = butterfly(fx.ptr_load(red + lane % 4), (2, 1))
    nq = hw_rsq(
        div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
        + fx.Float32(EPS)
    )
    # q: MXFP8 by _mxfp8_quant_op, a group of 32 = 4 threads
    ys = [(xq[j] * nq) * qnw[j] for j in range(8)]
    amax = fx.Float32(0.0)
    for y in ys:
        amax = fx.max(amax, abs(y))
    code = code_round(fx.max(butterfly(amax, (1, 2), fx.max), fx.Float32(AMAX_FLOOR)))
    mul = rcp_pow2(code)
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[ys[i] * mul for i in range(4)])
        w1 = fp8_pack4(*[ys[4 + i] * mul for i in range(4)])
        if const_expr(c["tok_par"]):
            # token parallel: to every rank, at the token's row of the step
            tg = c["t0"] + t
            for p in range_constexpr(c["d"].tp):
                qw, qs = c["q_dsts"][p]
                c["put_words"](qw, tg * QX8_WORDS + 2 * tt, [w0, w1])
                if tt % 4 == 0:
                    c["put"](qs, tg * (Q_RANK // 32) + tt // 4, code)
        else:
            c["put_words"](c["qx8"], Quad(t * (QX8_WORDS // 2) + tt), [w0, w1])
            if tt % 4 == 0:
                c["put"](c["qx8s"], t * (Q_RANK // 32) + tt // 4, code)
        if const_expr(c["index"]):
            # the (qr, qr_scale) pair the original indexer reads
            bo.buffer_store(
                fx.Vector.from_elements([w0, w1], fx.Int32),
                rsrc(a["qr_out"]),
                t * QX8_WORDS + 2 * tt,
            )
            if tt % 4 == 0:
                bo.buffer_store(
                    fx.Int8(code), rsrc(a["qrs_out"]), t * (Q_RANK // 32) + tt // 4
                )
    gpu.barrier()


@traced
def stage_kv_norm(c, t):
    """Token t: kv_norm -> GPT-J RoPE of the tail -> bf16 -> group-32 FP8 QAT
    -> the ring (and in the layer kernel ``kvrow``). Polls only kv's columns of
    QKV (``stage_q_norm`` takes q's).

    Threads 256.. mirror thread 255's loads (so every wave reaches the barrier)
    and store nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    # every weight and table value the token's rows do not decide, loaded
    # before their wait
    kvnw = [ld_bf(a["kvn_w"], tt * 2 + j) for j in range(2)]
    pos = ld_i32(a["pos"], 2 * t)
    jr = tt - (KV_DIM - ROPE) // 2
    live = jr >= 0
    jj = fx.max(jr, 0)
    cs = live.select(ld_bf(a["cos"], pos * HALF + jj), fx.Float32(1.0))
    sn = live.select(ld_bf(a["sin"], pos * HALF + jj), fx.Float32(0.0))
    rel = ld_i32(a["ring_rows"], t)
    # the thread's 2 columns: one bf16-pair word
    kvw = c["poll"]([(c["qkv"], (t * QKV_ROWS + Q_RANK + tt * 2) // 2, 1)])[0][0]
    xk = [bf_lo(kvw), bf_hi(kvw)]
    ssk = butterfly(xk[0] * xk[0] + xk[1] * xk[1], (8, 4, 2, 1, 16, 32))
    if (lane == 63) & mine:
        fx.ptr_store(ssk, red + wave)
    gpu.barrier()
    tk = butterfly(fx.ptr_load(red + lane % 4), (2, 1))
    nk = hw_rsq(
        div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
        + fx.Float32(EPS)
    )
    # this thread's pair (2 tt, 2 tt + 1), RoPE on the last 64 columns
    kn = [bf16_round((xk[j] * nk) * kvnw[j]) for j in range(2)]
    e0 = bf16_round(kn[0] * cs - kn[1] * sn)
    e1 = bf16_round(kn[0] * sn + kn[1] * cs)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8), fx.max)
    code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    step = pow2(code)
    scaled = [clamp_fp8(v * rcp_pow2(code)) for v in (e0, e1)]
    # the step's row of this token (int32, -1: none) past the layer's ring start
    row = rel + a["ring_off"]
    if const_expr(c["kv_fp8"]):
        _ring_row_fp8(c, t, tt, mine, rel, row, scaled, code)
    else:
        qat = [fp8_value(v) * step for v in scaled]
        # a [rows, 512] bf16 pool's row: the QAT values
        if mine & (rel >= 0):
            dst = rsrc(a["ring"] + fx.Int64(row) * (KV_DIM * 2))
            for j in range_constexpr(2):
                bo.buffer_store(qat[j].to(fx.BFloat16), dst, tt * 2 + j)
        if const_expr(c["fused"]):  # noqa: SIM102
            # the layer kernel: the row for this launch's attention too
            # (uncached: its readers on other XCDs), its flag KRDY
            if mine:
                for j in range_constexpr(2):
                    bo.buffer_store(
                        qat[j].to(fx.BFloat16),
                        rsrc(a["kvrow"]),
                        step_token(c, t) * KV_DIM + tt * 2 + j,
                    )
        if const_expr(c["tok_par"]):  # noqa: SIM102
            # token parallel: the row to every other rank (``ring_row_from``)
            if mine:
                tg = c["t0"] + t
                for p in range_constexpr(c["d"].tp):
                    if c["rank"] != p:
                        c["put_bf"](c["kv_dsts"][p], tg * KV_DIM + 2 * tt, qat)
    gpu.barrier()


@traced
def _ring_row_fp8(c, t, tt, mine, rel, row, scaled, code):
    """``stage_kv_norm``'s window row in an FP8 pool: thread tt's pair of E4M3
    values (``scaled``, in range) and its group's E8M0 ``code``. An even thread
    stores its pair and its neighbour's as one word, a group's first thread
    its scale; the layer kernel also into ``kvrow``'s planes
    (``dims.kvrow_bytes``); token parallel pushes the same words to every
    other rank's KG (``ring_row_from``: data words, then a group's code a
    word)."""
    a = c["args"]
    nb = [xshfl(v, 1) for v in scaled]
    word = fp8_pack4(scaled[0], scaled[1], nb[0], nb[1])
    if mine & (rel >= 0):
        dst = rsrc(a["ring"] + fx.Int64(row) * KV_DIM)
        if tt % 2 == 0:
            bo.buffer_store(word, dst, tt // 2)
        if tt % 16 == 0:
            bo.buffer_store(
                fx.Int8(code),
                rsrc(a["ring_scale"] + fx.Int64(row) * (KV_DIM // 32)),
                tt // 16,
            )
    if const_expr(c["fused"]):  # noqa: SIM102
        # the layer kernel: the row for this launch's attention too
        # (uncached: its readers on other XCDs), its flag KRDY
        if mine:
            ts = step_token(c, t)
            if tt % 2 == 0:
                bo.buffer_store(word, rsrc(a["kvrow"]), (ts * KV_DIM) // 4 + tt // 2)
            if tt % 16 == 0:
                bo.buffer_store(
                    fx.Int8(code),
                    rsrc(a["kvrow"] + kvrow_scale0(c["step_rows"])),
                    ts * (KV_DIM // 32) + tt // 16,
                )
    if const_expr(c["tok_par"]):  # noqa: SIM102
        if mine:
            tg = c["t0"] + t
            for p in range_constexpr(c["d"].tp):
                if c["rank"] != p:
                    if tt % 2 == 0:
                        c["put"](c["kv_dsts"][p], tg * (KV_DIM // 2) + tt // 2, word)
                    if tt % 16 == 0:
                        c["put"](
                            c["kv_dsts"][p],
                            tg * (KV_DIM // 2) + KV_DIM // 4 + tt // 16,
                            code,
                        )


@traced
def normed_row_from(c, nbg, t, n):
    """Token t, when another rank owns it: its normed row from NBG -> ``normed``
    (``stage_norm``'s store)."""
    tid = c["tid"]
    a = c["args"]
    if t // n != c["rank"]:
        for k in range_constexpr(-(-(HIDDEN // 4) // THREADS)):
            q = tid + THREADS * k
            if q < HIDDEN // 4:
                (w,) = c["poll"]([(nbg, t * (HIDDEN // 2) + 2 * q, 2)])
                bo.buffer_store(
                    fx.Vector.from_elements([w[0], w[1]], fx.Int32),
                    rsrc(a["normed"]),
                    t * (HIDDEN // 2) + 2 * q,
                )


@traced
def ring_row_from(c, kg, t, n):
    """Token t, when another rank owns it: its window row from KG -> the ring
    row (``stage_kv_norm``'s store): a BF16 row's words, or an FP8 row's data
    words and then its codes, a word each (``_ring_row_fp8``); in the layer
    kernel also its ``kvrow`` row and its KRDY, where this launch's attention
    reads it."""
    tid = c["tid"]
    a = c["args"]
    if t // n != c["rank"]:
        rel = ld_i32(a["ring_rows"], t)
        row = rel + a["ring_off"]
        words = KV_DIM // 4 + KV_DIM // 32 if const_expr(c["kv_fp8"]) else KV_DIM // 2
        if tid < words:
            (w,) = c["poll"]([(kg, t * (KV_DIM // 2) + tid, 1)])
            if const_expr(c["fused"]):
                kvrow_word(c, t, tid, w[0])
            if rel >= 0:
                if const_expr(c["kv_fp8"]):
                    dst = rsrc(a["ring"] + fx.Int64(row) * KV_DIM)
                    if tid < KV_DIM // 4:
                        bo.buffer_store(w[0], dst, tid)
                    else:
                        bo.buffer_store(
                            fx.Int8(w[0]),
                            rsrc(a["ring_scale"] + fx.Int64(row) * (KV_DIM // 32)),
                            tid - KV_DIM // 4,
                        )
                else:
                    bo.buffer_store(
                        w[0],
                        rsrc(a["ring"] + fx.Int64(row) * (KV_DIM * 2)),
                        tid,
                    )
        if const_expr(c["fused"]):
            publish(c["put"], c["krdy"], t, 1, tid == 0)


@traced
def kvrow_word(c, t, i, w):
    """Word ``i`` of token t's window row as KG carries it (``ring_row_from``)
    into ``kvrow``'s planes: a BF16 row's bf16 pairs, or an FP8 row's data
    words and then its codes, a byte each."""
    a = c["args"]
    if const_expr(not c["kv_fp8"]):
        bo.buffer_store(w, rsrc(a["kvrow"]), t * (KV_DIM // 2) + i)
    elif i < KV_DIM // 4:
        bo.buffer_store(w, rsrc(a["kvrow"]), t * (KV_DIM // 4) + i)
    else:
        bo.buffer_store(
            fx.Int8(w),
            rsrc(a["kvrow"] + kvrow_scale0(c["step_rows"])),
            t * (KV_DIM // 32) + i - KV_DIM // 4,
        )


# ---------------------------------------------------------------- wq_b
@traced
def rope_pair(c, t0, n, dim0, pairs=ROWS // 2):
    """This thread's (cos, sin) for its token of the tile from ``t0`` (``n``
    rows, the index clamped into them) at pair dim0 // 2 of a head's rotated
    tail -- the thread map of ``stage_wqb`` / ``stage_iq``'s epilogue (``pairs``
    threads a token); loaded before the GEMV waits for its x, used after it."""
    a = c["args"]
    tl = c["tid"] // pairs
    pos = ld_i32(a["pos"], 2 * plus(t0, fx.min(tl, n - 1)))
    jj = fx.max(dim0, 0) // 2
    return ld_bf(a["cos"], pos * HALF + jj), ld_bf(a["sin"], pos * HALF + jj)


@traced
def wqb_rope(c, task, t0, n):
    """``rope_pair`` for row task 16 + 2 (tid % 8) of wq_b."""
    d = (task * ROWS + 2 * (c["tid"] % (ROWS // 2))) % HEAD_DIM
    return rope_pair(c, t0, n, d - (HEAD_DIM - ROPE))


@traced
def stage_wqb(c, task, ops, t0, n, rope):
    """Rows 16 task .. +16 of the rank's heads x 512: FP8 GEMV -> bf16 -> the
    GPT-J RoPE of a head's last 64 dims (the Triton seam: bf16 products are exact,
    so its contraction does not show) -> q out; the token tile from ``t0`` (its
    x in LDS), ``rope`` its tables (``wqb_rope``, loaded before x's wait)."""
    n = c["S"] if n is None else n
    gemv_fp8_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], c["red"], ops, rows=n)
    gpu.barrier()
    wqb_epilogue(c, task, t0, n, rope, c["red"])
    gpu.barrier()


@traced
def stage_wqb_group(c, tasks, ops, t0, n, ropes, live):
    """A CTA's wq_b tasks (at most ``WQB_GROUP``) as one: every task's GEMV
    into its own part of ``red``, one barrier, then each task's epilogue
    (``wqb_epilogue``; ``live[i]`` False: past the stage's tasks) -- not
    three GEMVs each behind its own barriers."""
    for i in range_constexpr(len(tasks)):
        part = c["red"] + i * RED_TASK
        gemv_fp8_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], part, ops[i], rows=n)
    gpu.barrier()
    for i in range_constexpr(len(tasks)):
        part = c["red"] + i * RED_TASK
        # a compile-time test, then a traced one: not one condition
        if const_expr(live[i] is True):  # noqa: SIM114
            wqb_epilogue(c, tasks[i], t0, n, ropes[i], part)
        elif live[i]:
            wqb_epilogue(c, tasks[i], t0, n, ropes[i], part)
    gpu.barrier()


def gemv_fp8_tile_accs(c, k, xl, xsl, ops, tiles):
    """``gemv_fp8_mfmas``' chain (split None) for each token tile [(t0, n)] of
    the rows in LDS ``xl`` / ``xsl`` from row 0, a K step's operands shared:
    each tile's partial C, not stored."""
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in tiles]
    x_row, s_row = lds_row(k // 4), lds_row(k // 32)
    for i, (st, live) in enumerate(_gemv_steps(c, k, None)):
        accs = [
            _fp8_step_mfma(
                c, k, xl + t0 * x_row, xsl + t0 * s_row, st, ops[i], acc, live, n
            )
            for (t0, n), acc in zip(tiles, accs)
        ]
    return accs


@traced
def wqb_epilogue(c, task, t0, n, rope, red):
    """``stage_wqb``'s rows of ``red`` (task ``task``'s GEMV) -> bf16 -> RoPE ->
    q out."""
    tid = c["tid"]
    a = c["args"]
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        r0 = task * ROWS + 2 * rp  # global row: head r0 // 512, dim r0 % 512
        v0 = bf16_round(acc_sum(red, 2 * rp, tl))
        v1 = bf16_round(acc_sum(red, 2 * rp + 1, tl))
        d = r0 % HEAD_DIM
        live = d >= HEAD_DIM - ROPE
        cs, sn = rope
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        out = t * c["d"].heads * HEAD_DIM + r0
        if const_expr(c["fused"]):
            # the layer kernel: to its scores as tagged pairs (QP)
            c["put_bf"](c["qp"], out, [e0, e1])
        else:
            bo.buffer_store(e0.to(fx.BFloat16), rsrc(a["q_out"]), out)
            bo.buffer_store(e1.to(fx.BFloat16), rsrc(a["q_out"]), out + 1)


@traced
def iq_rope(c, task, t0, n):
    """``rope_pair`` for row task 16 + 2 (tid % 8) of the index query."""
    d = (task * ROWS + 2 * (c["tid"] % (ROWS // 2))) % ip.DIM
    return rope_pair(c, t0, n, d - (ip.DIM - IDX_ROPE))


@traced
def stage_iq(c, task, ops, t0, n, rope):
    """Rows 16 task .. of the indexer's wq_b [4096, 1280]: FP8 GEMV of the q
    latent -> bf16 -> the GPT-J RoPE of a head's dims 64..127 -> IQ; the token
    tile from ``t0`` (its x in LDS), ``rope`` its tables (``iq_rope``)."""
    tid, red = c["tid"], c["red"]
    n = c["S"] if n is None else n
    gemv_fp8_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], red, ops, rows=n)
    gpu.barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        r0 = task * ROWS + 2 * rp
        v0 = bf16_round(acc_sum(red, 2 * rp, tl))
        v1 = bf16_round(acc_sum(red, 2 * rp + 1, tl))
        d = r0 % ip.DIM
        live = d >= ip.DIM - IDX_ROPE
        cs, sn = rope
        # bf16 x bf16 products are exact: the original's contraction cannot show
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        c["put_bf"](c["iq"], t * IDX_ROWS + r0, [e0, e1])
    gpu.barrier()


@traced
def run_wqb(c, key, first, stride, per, with_iq=True):
    """This CTA's wq_b tasks ``first`` + ``stride`` i, i < ``per`` (past the
    stage's tasks: not run), and (``with_iq``) an indexer layer's wq task, two
    at a time: the first two's weights in flight before the q latent's wait;
    the weights stay in registers for the later token tiles (and the iq task
    runs in each, ahead of them: held on past them, its weights spill in
    index_w)."""
    a, dims = c["args"], c["d"]
    wqb, wqb_s = a["wqb"], a["wqb_s"]
    iq = key.feeds_index and with_iq
    tasks = [first if i == 0 else first + stride * i for i in range(per)]
    whole = stride * per == dims.wqb_tasks  # every slot a task
    clamped = tasks if whole else [fx.min(t, dims.wqb_tasks - 1) for t in tasks]
    # the layer kernel's three a CTA: all three in flight before the wait
    early = min(2, per) if whole else min(per, WQB_HELD)
    # the index query's weights issued first (loads wait in issue order)
    iq_task = first_task(c["bid"], IQ0)
    if const_expr(iq):
        iq_ops = gemv_fp8_loads(c, a["iq_w"], a["iq_ws"], Q_RANK, iq_task)
    wqb_all = [gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, clamped[i]) for i in range(early)]
    for t0, n in token_tiles(key.tokens):
        # the epilogues' RoPE tables, loaded before the q latent's wait
        ropes = [wqb_rope(c, clamped[b], t0, n) for b in range(per)]
        if const_expr(iq):
            iq_ropes = iq_rope(c, iq_task, t0, n)
        load_x8(
            c, c["qx8"], c["qx8s"], QX8_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"],
            t0, n,
        )  # fmt: skip
        # the index query before the attention's q: the scorer waits on it, the
        # attention's q only past the selection
        if const_expr(iq):
            # K1 alone: every step's (``_index_tasks``)
            if const_expr(not c["fused"]):  # noqa: SIM114
                stage_iq(c, iq_task, iq_ops, t0, n, iq_ropes)
            elif step_queried(c):
                stage_iq(c, iq_task, iq_ops, t0, n, iq_ropes)
        if const_expr(early == per and per <= WQB_GROUP):
            # every task's weights in flight already: one GEMV group
            live = [True if whole else tasks[i] < dims.wqb_tasks for i in range(per)]
            stage_wqb_group(c, tasks, wqb_all, t0, n, ropes, live)
        else:
            run_wqb_pairs(c, tasks, wqb_all, clamped, t0, n, ropes, early, whole)


@traced
def run_wqb_late(c, key, bid):
    """The layer kernel's indexer layer: the attention's wq_b tasks after K2a's
    scorer (its q is read only past the selection, so the scorer's CTAs do
    not wait for it), on the CTAs from WQB_LAYER0, its weights issued here."""
    dims = c["d"]
    enter_stage("k1.wq_b")
    wqb_ctas = BLOCKS - WQB_LAYER0
    if bid >= WQB_LAYER0:
        run_wqb(
            c, key, bid - WQB_LAYER0, wqb_ctas, -(-dims.wqb_tasks // wqb_ctas),
            with_iq=False,
        )  # fmt: skip


@traced
def run_iq(c, key):
    """An indexer layer's wq task of this CTA alone (``run_wqb``'s), on a CTA
    the layer kernel keeps off wq_b: every CTA runs its own (``IQ_TASKS`` ==
    ``BLOCKS``) wherever wq_b runs; on a step that scores a row only
    (``step_queried``)."""
    if const_expr(key.feeds_index):  # noqa: SIM102
        if step_queried(c):
            _iq_alone(c, key)


@traced
def _iq_groups(c, key):
    """The layer kernel's FP4 index query in place (``iq_in_place``, one token
    tile): group task g (CTA ``IQ_GROUP0`` + g) the GEMV tasks 2 g and 2 g + 1
    -- one 32-dim FP4 group of a head -- into two ``red`` parts behind one
    barrier, then ``iq_group_fp4`` quantizes the group: no quantization stage,
    no second hand-off. The head weight polled before the q latent's wait."""
    a = c["args"]
    group = first_task(c["bid"], IQ_GROUP0)
    if group < IQ_TASKS // 2:
        ops = [
            gemv_fp8_loads(c, a["iq_w"], a["iq_ws"], Q_RANK, 2 * group + i)
            for i in range(2)
        ]
        n = key.tokens
        d = (group * 2 * ROWS + 2 * (c["tid"] % ROWS)) % ip.DIM
        rope = rope_pair(c, 0, n, d - (ip.DIM - IDX_ROPE), pairs=ROWS)
        iw = iq_group_weight(c, group, 0, n)
        load_x8(
            c, c["qx8"], c["qx8s"], QX8_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"],
            0, n,
        )  # fmt: skip
        for i in range_constexpr(2):
            part = c["red"] + i * RED_TASK
            gemv_fp8_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], part, ops[i], rows=n)
        gpu.barrier()
        iq_group_fp4(c, group, 0, n, rope, iw, c["red"], RED_TASK)
        gpu.barrier()


@traced
def _iq_alone(c, key):
    """``run_iq``'s task: the in-place groups (``_iq_groups``) where the step
    quantizes its query there, else this CTA's 16-row GEMV task."""
    if const_expr(c["fused"] and iq_in_place(key.tokens, key.index_fp4)):
        _iq_groups(c, key)
    else:
        _iq_task(c, key)


@traced
def _iq_task(c, key):
    """``_iq_alone``'s 16-row task: its weights, then each token tile's GEMV."""
    a = c["args"]
    iq_task = first_task(c["bid"], IQ0)
    iq_ops = gemv_fp8_loads(c, a["iq_w"], a["iq_ws"], Q_RANK, iq_task)
    for t0, n in token_tiles(key.tokens):
        iq_ropes = iq_rope(c, iq_task, t0, n)
        load_x8(
            c, c["qx8"], c["qx8s"], QX8_WORDS, Q_RANK // 32, c["qxl"],
            c["qxsl"], t0, n,
        )  # fmt: skip
        stage_iq(c, iq_task, iq_ops, t0, n, iq_ropes)


@traced
def run_wqb_pairs(c, tasks, wqb_all, clamped, t0, n, ropes, early, whole):
    """``run_wqb``'s tasks two at a time (``stage_wqb`` each) where not all of
    them have their weights in flight: the later pairs' loaded here, past
    ``WQB_HELD`` again each token tile (L2 hits)."""
    a, dims = c["args"], c["d"]
    wqb, wqb_s = a["wqb"], a["wqb_s"]
    per = len(tasks)
    for b in range_constexpr(0, per, 2):
        pair = range(b, min(b + 2, per))
        if const_expr(b >= WQB_HELD):
            wqb_all[b:] = [
                gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, clamped[i]) for i in pair
            ]
        elif const_expr(b >= early and t0 == 0):
            wqb_all += [gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, clamped[i]) for i in pair]
        for i in range_constexpr(b, min(b + 2, per)):
            # a compile-time test, then a traced one: not one condition
            if const_expr(whole):  # noqa: SIM114
                stage_wqb(c, tasks[i], wqb_all[i], t0, n, ropes[i])
            elif tasks[i] < dims.wqb_tasks:
                stage_wqb(c, tasks[i], wqb_all[i], t0, n, ropes[i])


@traced
def run_seam_a_gates(c, co, bid, own_n, base=GATE0):
    """Seam A's gates (``stage_gate``) from CTA ``base``, each published
    (GDONE) in the layer kernel: only seam F reads them."""
    enter_stage("k1.gate")
    for t in range(first_task(bid, base), own_n, BLOCKS):
        stage_gate(co, t)
        if const_expr(c["fused"]):
            publish(c["put"], c["gdone"], t, 1, c["tid"] == 0)


def _index_tasks(c, n):
    """``n`` of an indexer stage's tasks, or none on a step that scores no row
    (``step_queried``, read where each stage is gated) -- in the layer kernel
    only: K1 alone sits at its register limit, where a gate spills small
    widths."""
    if const_expr(c["fused"]):
        return sel(step_queried(c), n, 0)
    return n


@traced
def _index_w(co, bid, nbl, load_group, tasks, first):
    """This CTA's head-weight tasks (``stage_iw``) of the first ``tasks``,
    placed from CTA ``first``."""
    for task in range(first_task(bid, first), tasks, BLOCKS):
        stage_iw(co, task, HIDDEN, nbl, load_group)


@traced
def _index_quant(c, key, bid, tasks):
    """This CTA's (token, head) quantization tasks of the first ``tasks``; the
    FP4 plane's a wave each (a CTA its WAVES adjacent tasks a round: one round
    where a wave a CTA took three at S=24)."""
    if const_expr(key.index_fp4):
        slots = (tasks + WAVES - 1) // WAVES
        for slot in range(first_task(bid, IQUANT0), slots, BLOCKS):
            task = slot * WAVES + c["wave"]
            stage_iquant_fp4(c, fx.min(task, tasks - 1), task < tasks)
    else:
        for task in range(first_task(bid, IQUANT0), tasks, BLOCKS):
            stage_iquant(c, task)


@traced
def run_attn_pre(
    c, co, key, tls, tl0, nbl, gates=True, after_norm=None, q_apart=False
):  # fmt: skip
    """K1's stages, in order, on context ``c`` (``co``: the own-rows view of a
    token-parallel build, else ``c``); ``timeline`` stamps from point ``tl0`` +
    1, ``nbl`` an indexer layer's normed-row LDS (``stage_iw``). ``gates``:
    seam A's gates after wqkv_a (else the caller runs ``run_seam_a_gates``
    later). ``after_norm``: the layer kernel's work on the normed rows (its
    compressor's projection), after the norm's stages, before wqkv_a.
    ``q_apart``: the q path apart from that projection (``APART_*``)."""
    s, tid, bid = key.tokens, c["tid"], c["bid"]
    a, dims, timeline = c["args"], c["d"], key.timeline
    own_n = s // key.tp if key.tok_par else s
    own_tiles, own_rows_lds = token_tiles(own_n), tile_rows(own_n)
    wqb_per_cta = dims.wqb_tasks // BLOCKS
    fused = c["fused"]
    slice_x8 = c.get("slice_x8", False)
    enter_stage("k1.slice")
    # a sharded seam (the layer kernel): this rank's quarter of the slices,
    # the same tasks its seam F runs (their SDONE local)
    shard = c["shard_f"]
    own = SLICES // dims.tp if shard else SLICES
    # a sharded seam: a task a (slice of this rank's, destination rank) --
    # each CTA's stores to one rank, the fold computed tp times
    units = own * dims.tp if shard else SLICES
    if const_expr(shard):
        # at most a task a CTA (units <= BLOCKS): its destination set outside
        # the task's region (a context entry assigned in a traced region would
        # make the context the region's state)
        assert units <= BLOCKS
        unit = first_task(bid, 0)
        c["dst"] = unit // own
        if const_expr(slice_x8):
            # the destination rank's X8G, once a task (not every rank's held
            # through the fold: SGPRs)
            c["x8_dst"] = preg(load_ptr64(c["x8_peers"], c["dst"]), 0, "x8g")
        task = c["rank"] * own + unit % own
        if unit < units:
            stage_slice(co, task)
            if const_expr(fused):
                # the layer kernel's attention seam reads this task's R_new columns
                publish(c["put"], c["sdone"], task, 1,
                        (tid == 0) & (c["dst"] == c["rank"]))  # fmt: skip
        c["dst"] = None
    else:
        for task in range(first_task(bid, 0), units, BLOCKS):
            stage_slice(co, task)
            if const_expr(fused):
                # the layer kernel's attention seam reads this task's R_new columns
                publish(c["put"], c["sdone"], task, 1, tid == 0)
    if const_expr(shard):
        # this rank's gate blocks, on the CTAs past its slices'
        enter_stage("k1.gate_block")
        blocks = own // shard_gate_block(dims.tp)
        for unit in range(
            first_task(bid, SLICE_X8_GB0 if slice_x8 else units), blocks, BLOCKS
        ):
            stage_gate_block(c, c["rank"] * blocks + unit)
    stamp(timeline, tls, tid, tl0 + 1)
    # the gate's outputs are the next seam's: it runs after the stages
    # this launch waits on that its CTAs may also hold (norm, wqkv_a)
    enter_stage("k1.norm")
    # past the slices' CTAs and a sharded seam's gate blocks (``units`` ..)
    busy = units + (own // shard_gate_block(dims.tp) if shard else 0)
    for t in range(
        first_task(bid, norm0(own_n, busy)), 0 if slice_x8 else own_n, BLOCKS
    ):
        stage_norm(co, t)
    stamp(timeline, tls, tid, tl0 + 2)
    if const_expr(after_norm is not None):
        after_norm()
    enter_stage("k1.wqkv_a")
    # a CTA's wqkv task (at most one): its weights in flight before the
    # normed row's wait
    assert WQKV_TASKS <= BLOCKS
    if const_expr(len(own_tiles) == 1):
        wqkv_task = first_task(bid, GATE0 if slice_x8 else WQKV0)
        if wqkv_task < WQKV_TASKS:
            wqkv_ops = gemv_fp8_loads(co, a["wqkv"], a["wqkv_s"], HIDDEN, wqkv_task)
            load_seam_x8(co, 0, own_rows_lds)
            stage_wqkv(co, wqkv_task, wqkv_ops, 0, own_n)
    elif const_expr(q_apart):
        # a task a CTA, its tiles in series on its weights: q's, then kv's
        for task, first in ((first_task(bid, APART_Q0), 0),
                            (first_task(bid, APART_KV0), Q_TASKS)):  # fmt: skip
            if task < (Q_TASKS if first == 0 else WQKV_TASKS - Q_TASKS):
                ops = gemv_fp8_loads(co, a["wqkv"], a["wqkv_s"], HIDDEN, first + task)
                for t0, n in own_tiles:
                    load_seam_x8(co, t0, n)
                    stage_wqkv(co, first + task, ops, t0, n)
    elif const_expr(KVN0 + own_n <= WQKV_KV0):
        ntiles = len(own_tiles)
        # q's CTAs before q_norm's, kv's past kv_norm's
        assert (WQKV_Q0 + Q_TASKS) % BLOCKS <= QKV0
        q_task = first_task(bid, WQKV_Q0)
        if q_task < Q_TASKS:
            q_ops = gemv_fp8_loads(co, a["wqkv"], a["wqkv_s"], HIDDEN, q_task)
            for t0, n in own_tiles:
                load_seam_x8(co, t0, n)
                stage_wqkv(co, q_task, q_ops, t0, n)
        kv_unit = first_task(bid, WQKV_KV0)
        if kv_unit < (WQKV_TASKS - Q_TASKS) * ntiles:
            kv_task = Q_TASKS + kv_unit // ntiles
            kv_ops = gemv_fp8_loads(co, a["wqkv"], a["wqkv_s"], HIDDEN, kv_task)
            t0 = kv_unit % ntiles * TILE
            n = fx.min(fx.Int32(TILE), own_n - t0)
            load_seam_x8(co, t0, n)
            stage_wqkv(co, kv_task, kv_ops, t0, n)
    else:
        # (tile, task) units, not a task's tiles in series on one CTA, which
        # would leave most CTAs idle (the weights read again a tile: L2 hits)
        tile, wqkv_tasks = tile_tasks(bid, WQKV0, len(own_tiles), WQKV_TASKS)
        wqkv_ops = [
            gemv_fp8_loads(
                co, a["wqkv"], a["wqkv_s"], HIDDEN, fx.min(t, WQKV_TASKS - 1)
            )
            for t in wqkv_tasks
        ]
        t0 = tile * TILE
        n = fx.min(fx.Int32(TILE), own_n - t0)
        load_seam_x8(co, t0, n)
        for k in range_constexpr(len(wqkv_tasks)):
            if wqkv_tasks[k] < WQKV_TASKS:
                stage_wqkv(co, wqkv_tasks[k], wqkv_ops[k], t0, n)
    stamp(timeline, tls, tid, tl0 + 3)
    if const_expr(gates):
        run_seam_a_gates(c, co, bid, own_n)
    stamp(timeline, tls, tid, tl0 + 4)
    enter_stage("k1.qkv")
    for t in range(first_task(bid, APART_QN0 if q_apart else QKV0), own_n, BLOCKS):
        stage_q_norm(co, t)
    for t in range(first_task(bid, APART_KVN0 if q_apart else KVN0), own_n, BLOCKS):
        stage_kv_norm(co, t)
        if const_expr(fused):
            # ... and its attention this token's window row (``kvrow``)
            publish(c["put"], c["krdy"], step_token(co, t), 1, tid == 0)
    if const_expr(key.tok_par):
        # wq_b reads every token's q latent where qkv pushed it
        pl = peer_offsets(s, dims)
        c["qx8"] = preg(c["sym"], pl["qg"], "qg")
        c["qx8s"] = preg(c["sym"], pl["qgs"], "qgs")
    if const_expr(key.feeds_index):
        # the head weights read the normed rows alone: ahead of wq_b and the
        # index query, which wait on the q latent
        enter_stage("k1.index_w")

        def load_group(g, n):
            poll_copy(
                co, tid, THREADS,
                [(co["nb"], flat_span(g, HIDDEN // 2), n * HIDDEN // 2, nbl, None)],
            )  # fmt: skip
            gpu.barrier()

        _index_w(
            co, bid, nbl, load_group, _index_tasks(c, iw_tasks(co["S"])),
            iw0(key.tok_par, own_n),
        )  # fmt: skip
    stamp(timeline, tls, tid, tl0 + 5)
    enter_stage("k1.wq_b")
    # every CTA's wq_b tasks (and an indexer layer's wq task), two at a
    # time: the first two's weights in flight before the q latent's wait;
    # the weights stay in registers for the later token tiles (and the iq
    # task runs in each: held on past them, its weights spill in index_w)
    assert IQ_TASKS == BLOCKS
    # the layer kernel's FP4 query quantized by its GEMV's group tasks
    # (``iq_in_place``): no quantization stage
    iq_quantizes = fused and key.feeds_index and iq_in_place(s, key.index_fp4)
    if const_expr(iq_quantizes and key.tok_par):
        # the group tasks read every token's head weights where iw pushed them
        c["iw"] = preg(c["sym"], peer_offsets(s, dims)["iwg"], "iwg")
    if const_expr(fused and key.feeds_index):
        # the layer kernel's indexer layer: every CTA's wq task here, the
        # attention's wq_b after K2a's scorer (``run_wqb_late``)
        run_iq(c, key)
    elif const_expr(fused):
        # the layer kernel: wq_b on the CTAs from WQB_LAYER0, every other
        # CTA free once its earlier K1 stages end (the attention's scores land
        # their keys there before q's wait)
        wqb_ctas = BLOCKS - WQB_LAYER0
        if bid >= WQB_LAYER0:
            run_wqb(c, key, bid - WQB_LAYER0, wqb_ctas, -(-dims.wqb_tasks // wqb_ctas))
        else:
            # an indexer layer's wq task of the CTAs wq_b skips
            run_iq(c, key)
    else:
        run_wqb(c, key, first_task(bid, WQB0), BLOCKS, wqb_per_cta)
    stamp(timeline, tls, tid, tl0 + 6)
    if const_expr(key.feeds_index and not iq_quantizes):
        if const_expr(key.tok_par):
            # the quantization reads every token's head weights where iw
            # pushed them
            c["iw"] = preg(c["sym"], peer_offsets(s, dims)["iwg"], "iwg")
        enter_stage("k1.index_quant")
        _index_quant(c, key, bid, _index_tasks(c, s * ip.HEADS))
    stamp(timeline, tls, tid, tl0 + len(TL_STAGES))
    if const_expr(key.tok_par):
        pl = peer_offsets(s, dims)
        kg = preg(c["sym"], pl["kg"], "kg")
        for t in range(first_task(bid, QKV0), s, BLOCKS):
            ring_row_from(c, kg, t, own_n)
        if const_expr(key.feeds_index):
            # the compressor reads every token's normed row
            nbg = preg(c["sym"], pl["nbg"], "nbg")
            for t in range(first_task(bid, NORM0), s, BLOCKS):
                normed_row_from(c, nbg, t, own_n)


def lds_struct(name, fields):
    """An ``fx.struct`` of ``fields`` (name -> array type), in that order."""
    return fx.struct(
        type(name, (), {"__annotations__": fields, "__module__": __name__})
    )


def lds_union(name, members):
    """An ``fx.union`` of ``members`` (name -> struct)."""
    return fx.union(
        type(name, (), {"__annotations__": members, "__module__": __name__})
    )


def attn_pre_smem(key: AttnPreBuild) -> dict:
    """K1's LDS, a struct a stage group: the members of a union, so only one
    group's buffers take space at a time (a barrier ends each stage). Each
    opens with what several stages read, at the same offsets -- ``red`` and
    wq_b's q latent rows -- then slice's, wqkv_a's or index_w's own."""
    s = key.tokens
    rows = tile_rows(s)
    shared = {
        "red": fx.Array[fx.Float32, max(2, WQB_GROUP) * RED_TASK, 16],
        "qxl": fx.Array[fx.Int32, rows * lds_row(QX8_WORDS), 16],
        "qxsl": fx.Array[fx.Int32, rows * lds_row(Q_RANK // 32), 16],
    }
    stages = {
        "slice": {
            "rl": fx.Array[fx.Float32, s * KT_ROW, 16],
            "fl": fx.Array[fx.Float32, MIX * KT_ROW, 16],
        },
        "wqkv": {
            "xl": fx.Array[fx.Int32, rows * lds_row(X8_WORDS), 16],
            "xsl": fx.Array[fx.Int32, rows * lds_row(HIDDEN // 32), 16],
        },
        # a group of WAVES tokens' normed rows (``stage_iw``)
        "iw": {
            "nbl": fx.Array[
                fx.Int32, min(s, WAVES) * HIDDEN // 2 if key.feeds_index else 1, 16
            ],
        },
    }
    members = {}
    for name, fields in stages.items():
        layout = dict(shared)
        layout.update(fields)
        members[f"k1_{name}"] = lds_struct(f"K1{name.title()}Lds", layout)
    return members


def attn_pre_context(key, lds, mb, layout, scratch, args, rank, sym, peers):
    """K1's stage context ``c`` and the one its seam stages run on (``co``: a
    token-parallel build's own-rows view, else ``c``); ``lds`` the
    ``attn_pre_smem`` member views by name."""
    s, dims = key.tokens, Dims(key.tp)
    tid = fx.thread_idx.x
    sl, wl = lds["k1_slice"], lds["k1_wqkv"]
    c = {
        "S": s, "step_rows": s,
        "tid": tid, "bid": fx.block_idx.x, "lane": tid % 64, "wave": tid // 64,
        "rl": sl.rl.ptr, "fl": sl.fl.ptr, "red": wl.red.ptr,
        "xl": wl.xl.ptr, "xsl": wl.xsl.ptr, "qxl": wl.qxl.ptr, "qxsl": wl.qxsl.ptr,
        "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
        "put_coded": mb.put_coded, "poll_coded": mb.poll_coded,
        "fan": {}, "shard_f": False, "dst": None,  # outputs put locally (``put_out``)
        "args": args,
        "fold": key.fold, "index": key.feeds_index, "aux": key.aux, "ffn": False,
        "embed_in": key.embed_in,
        "d": dims, "tok_par": False, "kv_fp8": key.kv_fp8,
        # the layer kernel (``layer``): its attention follows in this launch
        "fused": False,
    }  # fmt: skip
    c.update(region_addrs(scratch, layout, key.numa, args["numa_word"]))
    c["rank"], c["sym"] = rank, sym
    if const_expr(key.shard):
        peer = peer_offsets(s, dims)
        shard_seam(c, peer, peer_bases(peers, key.tp), sym, key.tp, "pmixg")
        if const_expr(key.slice_x8):
            c["slice_x8"] = True
            c["x8_peers"] = peers
            assert peer["x8g"] % 16 == 0
            # X8G a pair index from the peer buffer's base: one buffer resource
            # (SGPRs) for this rank's polls and a destination's puts
            c["x8g_pair0"] = peer["x8g"] // 8
            c["x8g"] = preg(sym, 0, "x8g")
    if const_expr(not key.tok_par):
        return c, c
    bases = peer_bases(peers, key.tp)
    c["peer_addr"] = lambda p: bases[p]
    pl = peer_offsets(s, dims)
    co = own_view(c, K1_OWN, EMBED_ROW_BYTES if key.embed_in else TOKEN_ROW_BYTES)
    co["tok_par"] = True
    # qkv pushes its token's q latent to every rank's QG / QGS and its
    # window row to every other rank's KG
    co["q_dsts"] = [
        (preg(bases[p], pl["qg"], "qg"), preg(bases[p], pl["qgs"], "qgs"))
        for p in range(key.tp)
    ]
    co["kv_dsts"] = [preg(bases[p], pl["kg"], "kg") for p in range(key.tp)]
    co["nb_dsts"] = [preg(bases[p], pl["nbg"], "nbg") for p in range(key.tp)]
    co["iw_dsts"] = [preg(bases[p], pl["iwg"], "iwg") for p in range(key.tp)]
    return c, co


def shard_seam(c, peer, bases, sym, tp, gate):
    """A seam's context ``c`` column-sharded: a rank's slices put LIN and its
    squares into every rank's gathered copy (LING / LSSQG), the readers (the
    norm, the router's norm) reading this rank's; the mixes' partials stay
    local, summed a gate block by their owner into every rank's ``gate``
    region (the gate's first sums: seam A's PMIXG, seam F's PMIXGF).

    LING / LSSQG are shared by seam A and seam F, their tags apart: a peer
    pushes the next seam's only once this rank's norm read them (the norm
    feeds the sublayer whose partials the peer waits on). The gate sums are
    not: the gate feeds no sublayer (the residual's post / comb and the next
    seam's pre), so a peer's next seam would overwrite a gate's sums before
    a late gate reads them."""
    gathered = (("lin", "ling"), ("lssq", "lssqg"))
    c["shard_f"] = True
    c["fan"] = {n: [preg(bases[p], peer[g], g) for p in range(tp)] for n, g in gathered}
    for n, g in gathered:
        c[n] = preg(sym, peer[g], g)
    c["pmixg"] = preg(sym, peer[gate], gate)
    c["pmixg_fan"] = [preg(bases[p], peer[gate], gate) for p in range(tp)]


def attn_pre_args(*values) -> dict:
    """K1's kernel arguments by name (``ABI`` order, less the shared tail)."""
    names = ABI.names[: ABI.names.index("scratch")]
    assert len(values) == len(names)
    return dict(zip(names, values))


def build_attn_pre(key: AttnPreBuild):
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s)
    keyed = key_tuple(key, SOURCES)
    dims = Dims(key.tp)
    assert dims.wqb_tasks % BLOCKS == 0
    if key.tok_par:
        assert s % key.tp == 0, f"{s} rows do not split over {key.tp} ranks"
    members = attn_pre_smem(key)
    Smem = lds_union("K1Lds", members)

    @fx.struct
    class Stamps:
        tls: fx.Array[fx.Int64, TL_POINTS if key.timeline else 1, 16]

    name = kernel_symbol(
        "v41_attn_pre",
        s=s,
        f=int(key.fold),
        ix=int(key.feeds_index),
        i4=int(key.index_fp4),
        a=int(key.aux),
        tl=int(key.timeline),
        tpar=int(key.tok_par),
        sh=int(key.shard),
        e=int(key.embed_in),
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_pre(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_scale: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64,
        iw_out: Int64, ibound: Int64, scratch: Int64, numa_word: Int64, sym: Int64,
        peers: Int64, rank: Int32,
        layer: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        tls = alloc.allocate(Stamps).peek().tls.ptr
        smem = alloc.allocate(Smem)
        lds = {name: getattr(smem, name).peek() for name in members}
        args = attn_pre_args(
            res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_w,
            wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s, cos, sin, pos, ring, ring_scale,
            ring_rows, ring_off, res_out, post_out, comb_out, pre_out, q_out, normed,
            qr_out, qrs_out, aux, iq_w, iq_ws, iw_w, iq_out, iqs_out, iw_out,
            ibound,
        )  # fmt: skip
        args["numa_word"] = numa_word
        c, co = attn_pre_context(
            key, lds, mailbox(layer, scratch, key.diag_off, layout),
            layout, scratch, args, rank, sym, peers,
        )  # fmt: skip
        stamp_begin(key.timeline, tls, tid, TL_POINTS)
        run_attn_pre(c, co, key, tls, 0, lds["k1_iw"].nbl.ptr)
        stamp_flush(key.timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @launcher(name)
    def launch(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_scale: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64,
        iw_out: Int64, ibound: Int64, scratch: Int64, numa_word: Int64, sym: Int64,
        peers: Int64, rank: Int32,
        layer: Int32, tl: Int64, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        attn_pre(
            res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_w,
            wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s, cos, sin, pos, ring, ring_scale, ring_rows, ring_off,
            res_out, post_out, comb_out, pre_out, q_out, normed, qr_out, qrs_out, aux,
            iq_w, iq_ws, iw_w, iq_out, iqs_out, iw_out, ibound, scratch, numa_word,
            sym, peers, rank, layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(attn_pre, launch)
    return launch
