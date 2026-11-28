# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""K2a ``attn_post``: one V4.1 layer from the sparse decode attention to the MoE
input.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S = ``tokens`` rows:

Task counts are TP4's (``Dims``: the rank's widths at the build's ``tp``):

    score    (S x 8 a head tile, 8 splits a task): the decode attention's scores
             and softmax statistics (``attention.py``) -> AM, AL, AP
    irq      (S x 16 a head tile, 32 columns of its heads a task): the attention's PV and
             split combine (``attention.py``), then the inverse RoPE of each
             head's tail, group-32 FP8 of the fp32 result (aiter
             ``inverse_rope_group_quant``: amax floored at AMAX_FLOOR, the ceil code,
             the hardware scaled convert) -> mailbox XO
    wo_a     (64, 32 rows: two tiles, K split 4 each): the grouped FP8 GEMV (8
             heads a group) -> bf16 -> ATOM ``quantize_fp8`` of the 32 (one
             group: the ceil code, floor AMAX_FLOOR) -> X8B, X8BS
    wo_b     (320, 16 rows): FP8 GEMV -> the rank's bf16 partial -> pushed to
             every rank's ATTN region (system scope)
    slice    (160): the TP partials of its 32 columns summed as the all-reduce
             the original path runs sums them (fp32, in rank order 0 .. TP - 1;
             bf16) -- the attention output -- folded into the residual; then
             the FFN seam as in K1
    norm, gate: ffn_norm -> ``normed`` (the MoE input), then the FFN gates

Token parallel (``tok_par``, S a multiple of TP): wo_b pushes a token's partial
to its owner alone (rank r owns rows r S/TP ..: a reduce-scatter), and the
slice, gate and norm run on this rank's tokens (``own_view``); the caller
gathers the normed rows (``layer_post``). The sums are the all-reduce's, in
the same order, so every output is the same to the bit.

A selecting layer's indexer (``index_score.py``) runs first in this launch;
before it stay the compressor and each token's window metadata
(``step_meta.write_step_meta``).
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from atom.models.deepseek_v41.mono import attention_plan as aplan
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels.attention import (
    MERGE_OUT_WORDS,
    MERGE_P_WORDS,
    MERGE_STAT_WORDS,
    MERGE_SUM_WORDS,
    MERGE_TREE_WORDS,
    MERGE_V_WORDS,
    Q_LDS_WORDS,
    SCORE_LDS_WORDS,
    SCORE_Q_WORDS,
    SLICE,
    merge_task,
    merge_tasks,
    only_live,
    score_tasks,
    stage_attn_merge,
    stage_attn_score,
    tile_head,
)
from atom.models.deepseek_v41.mono.kernels.attention import (
    scratch_pairs as attention_pairs,
)

# helpers the kernel calls are imported by name: FlyDSL keys a build by the
# source of the same-directory functions it names, never through a module
from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    MAX_TOKENS,
    TILE,
    gemv_fp8_loads,
    gemv_fp8_mfmas,
    gemv_fp8_tile_accs,
    ld_bf,
    lds_row,
    lds_struct,
    lds_union,
    plus,
    poll_copy,
    slice_prefetch,
    stage_gate,
    stage_norm,
    stage_slice,
    strided_span,
    tile_rows,
    tile_tasks,
    token_tiles,
)
from atom.models.deepseek_v41.mono.kernels.debug import mailbox
from atom.models.deepseek_v41.mono.kernels.dims import (
    HEAD_TILE,
    WOA_ROWS,
    Dims,
)
from atom.models.deepseek_v41.mono.kernels.index_score import (
    PER_THREAD as SEL_PER_THREAD,
)
from atom.models.deepseek_v41.mono.kernels.index_score import (
    SEL_BITS_WORDS,
    SEL_RADIX_WORDS,
    run_indexer,
)
from atom.models.deepseek_v41.mono.kernels.index_score import (
    scratch_pairs as index_pairs,
)
from atom.models.deepseek_v41.mono.kernels.mailbox import preg, region_addrs
from atom.models.deepseek_v41.mono.kernels.mx import AMAX_FLOOR
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_offsets
from atom.models.deepseek_v41.mono.kernels.red import acc_sum, acc_word
from atom.models.deepseek_v41.mono.kernels.token_par import own_count, own_view
from atom.models.deepseek_v41.mono.kernels.topk import lds_words
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import (
    bf16_round,
    butterfly,
    fp8_pack4,
    kernel_symbol,
    lane_gather,
    launcher,
    ld_i32,
    traced,
    xshfl,
)
from atom.mono.device.ranks import peer_bases, sum_partials
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES, first_task
from atom.mono.plan.layout import pair_layout
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi

HEAD_DIM, ROPE = k1.HEAD_DIM, k1.ROPE
HIDDEN = k1.HIDDEN
ROWS = k1.ROWS
# wo_a's K order: the first WOA_SPLIT_K waves of a 32-row task's two tiles take
# one contiguous K range each, summed in wave order. Picked for speed: the
# original's own split need not be matched
WOA_SPLIT_K = WAVES * ROWS // WOA_ROWS
_SEL = lds_words(SEL_PER_THREAD)
SEL_KEYS, SEL_HIST, SEL_BUF = _SEL["keys"], _SEL["hist"], _SEL["buf"]
WOB_TASKS = HIDDEN // ROWS  # 320
# wo_b tasks a CTA holds the weights of at once past one token tile (TP 1)
WOB_LIVE = 3
# up to this many rows past one token tile wo_b runs a pair of tasks a CTA on
# every token (``wob_pairs_tiles``)
WOB_PAIRS_MAX = 24

GATE0 = k1.SLICES
NORM0 = GATE0 + 8
_CURRENT_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AttnPostBuild:
    tokens: int
    # the indexer's largest column bound (max_model_len): its mailbox regions and
    # merge levels (``index_plan``); 0 builds no indexer
    index_bound_max: int = 0
    # the index plane is FP4: its scorer is ``index_score_fp4`` (``iqs``,
    # ``pscale`` and the FULL layers' task plan ``iplan``)
    index_fp4: bool = False
    # FP4 keys in OPUS 64-row groups; the query remains Mono's own layout.
    index_mfma: bool = False
    # the TP size: every width a rank holds (``Dims``)
    tp: int = 4
    timeline: bool = field(default=False, metadata={"sym": "tl"})
    # ``ATOM_MONO_DEBUG``: bounded mailbox waits recording at this scratch
    # offset (``debug``); -1 is the normal build
    diag_off: int = -1
    # token parallel: wo_b's partials go to each token's owner rank only (a
    # reduce-scatter), and the slice, gate and norm run on this rank's tokens
    # (``own_view``); the caller gathers the normed rows
    tok_par: bool = False
    # the pool's rows are MXFP8: QK on FP8 (q quantized by the original path's
    # rule), PV on the rows dequantized (``attention``)
    kv_fp8: bool = False
    # the regions kept in two copies, one on each I/O die (``numa_plan``); the
    # scratch's calibration word that picks a CTA's copy is the ``numa_word``
    # argument, measured at run time
    numa: str = ""
    # the attention's keys include a selection (its first ``index_plan.TOPK``
    # slots, ``attention_plan.selection_slots``); False: window keys alone,
    # every slot the window's (the DSpark draft: window and block)
    attn_sel: bool = True
    # a request's step rows form a block every row of which sees all of it
    # (the DSpark draft's, its width): a row's keys run past its own, through
    # the block's last row (``attention._new_rows``); 0: causal (the target)
    block_rows: int = 0

    @property
    def score0(self) -> int:
        """The score stage's first CTA, from which the attention chain (merge,
        wo_a, wo_b, seam F, K2b) is laid out: past the indexer's last list
        level, a task a token from CTA 0 (``index_score._run_tree``), whose
        CTAs end with the selection."""
        if self.index_bound_max == 0:
            return 0
        return self.tokens // self.tp if self.tok_par else self.tokens

    # a split row's answer staged in LDS and put after its select
    # (``index_score._put_answer``): the leaner build ``runner._lean_rows``
    # takes where putting it from the select spills; False: put as selected
    staged_answer: bool = False

    @property
    def index_split(self) -> bool:
        """The FULL layers' long rows split over the TP ranks
        (``index_plan.row_shares``; ``index_score.stage_imerge``): the FP4
        plane's walk plan carries the shares, a token-parallel launch selects
        its own rows instead."""
        return (
            self.index_fp4
            and not self.tok_par
            and self.index_bound_max > ip.SPLIT_FROM
            and ip.split_tp(self.tp)
        )


# the per-token arguments the token-parallel stages read and write
K2A_OWN = (
    "res_in", "res_out", "pre_in", "post_in", "comb_in", "pre_out", "post_out",
    "comb_out", "normed",
)  # fmt: skip


# ``timeline`` stamps: kernel start, then the end of each stage's task loop
TL_STAGES = (
    "index",
    "attn_score",
    "irq",
    "wo_a",
    "wo_b",
    "slice",
    "norm",
    "gate",
)
TL_POINTS = 1 + len(TL_STAGES)


ABI = KernelAbi(
    (
        "iact", "iq", "iw", "plane", "iqs", "pscale", "iplan", "itab", "itab_stride",
        "itab_len", "ibatch", "ishift", "ibound",
        "ilog", "lstride", "isel", "icand", "ilift", "iprod", "ibmax", "bstride",
        "icout", "irow", "ixsel", "sbound",
        "q", "pool", "pool_scale", "window", "window_scale", "sel", "topk", "kmeta", "table", "rows_per_page", "page_rows",
        "main_off", "ring_off", "ring_slots", "sink", "qk_scale", "pos", "cos", "sin",
        "woa", "woa_s", "wob", "wob_s",
        "res_in", "post_in", "comb_in", "pre_in", "hc_fn", "hc_scale", "hc_base",
        "ffn_w", "res_out", "post_out", "comb_out", "pre_out", "normed",
        "scratch", "numa_word", "sym", "peers", "rank", "layer", "tl",
    )
)  # fmt: skip


def scratch_layout(key: AttnPostBuild) -> dict[str, tuple[int, int]]:
    tokens, index_bound_max, d = key.tokens, key.index_bound_max, Dims(key.tp)
    base = k1.scratch_layout(tokens)
    pairs = {
        "xo": tokens * d.xo_words,
        "xos": tokens * d.heads * HEAD_DIM // 32,
        **attention_pairs(tokens, d),
        "x8b": tokens * d.yb_words,
        "x8bs": tokens * d.o_rows // 32,
        **(
            index_pairs(tokens, index_bound_max, key.tok_par) if index_bound_max else {}
        ),
    }
    out = {name: base[name] for name in ("lin", "lssq", "pmix", "x8", "x8s")}
    start = max(o + n for o, n in base.values())
    return out | pair_layout(pairs.items(), start=start)


def scratch_bytes(key: AttnPostBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def attn_region_pair(src, t, s, col):
    return (src * s + t) * (HIDDEN // 2) + col // 2


def merge0(s, tiles, first):
    """The merge stage's first CTA: past the score tasks (from ``first``,
    ``AttnPostBuild.score0``)."""
    return (first + score_tasks(s, tiles)[1]) % BLOCKS


def merge_task_live(task, s, d):
    """Whether merge task ``task`` holds a live head. A ragged head tile's dead
    head groups (TP8: the rank's 8 heads in a 16-head tile) make tasks that
    gather V and write nothing (``irq_slice`` puts live heads only): half of
    TP8's merge tasks and their V gathers."""
    _, tile, hg, _ = merge_task(task, s, d.head_tiles)
    return tile * HEAD_TILE + hg * aplan.merge_heads(s) < d.heads


def woa0(s, tiles, first):
    """wo_a's first CTA: past the merge tasks, so no wo_a
    task waits behind a merge task to start its weights' loads."""
    return (merge0(s, tiles, first) + merge_tasks(s, tiles)) % BLOCKS


# ---------------------------------------------------------------- irq
@traced
def stage_attn_irq(c, task):
    """Merge task ``task`` (``attention.merge_task``): the decode attention's
    softmax, PV and combine (``stage_attn_merge``), then its inverse RoPE and
    group-32 FP8 quant, a thread (head of the group, a pair of dims). The RoPE
    table loads issue before the merge: they do not wait on it."""
    s = c["S"]
    mh = aplan.merge_heads(s)
    t, tile, hg, sl = merge_task(task, s, c["d"].head_tiles)
    rope = irq_rope_loads(c, t, sl)
    stage_attn_merge(c, t, tile, hg, sl, mh)
    irq_slice(c, t, tile, hg, sl, mh, rope)


def irq_rope_loads(c, t, sl):
    """This thread's (cos, sin) of its dim pair in slice ``sl`` of token t (1, 0
    off the rotated tail): loads only."""
    a = c["args"]
    pos = ld_i32(a["pos"], 2 * t)
    d = sl * SLICE + c["tid"] % (SLICE // 2) * 2
    live = d >= HEAD_DIM - ROPE
    jj = fx.max((d - (HEAD_DIM - ROPE)) // 2, 0)
    return (
        live.select(ld_bf(a["cos"], pos * k1.HALF + jj), fx.Float32(1.0)),
        live.select(ld_bf(a["sin"], pos * k1.HALF + jj), fx.Float32(0.0)),
    )


@traced
def irq_slice(c, t, tile, hg, sl, mh, rope):
    """Token t, heads ``mh`` hg .. of a head tile, dims 64 sl .. (the merge's
    ``att["out"]`` [head][64], the dims' ``rope`` (cos, sin):
    ``irq_rope_loads``) -> inverse RoPE, group-32 FP8 quant -> XO / XOS: a
    thread (head, dim pair), a group 16 lanes."""
    tid, dims = c["tid"], c["d"]
    pairs = SLICE // 2
    mine = tid < mh * pairs
    h = fx.min(tid, mh * pairs - 1) // pairs
    head, live = tile_head(dims, tile, hg * mh + h)
    mine = only_live(mine, live)
    tt = tid % pairs
    d = sl * SLICE + tt * 2
    out = c["att"]["out"] + h * SLICE
    e = fx.ptr_load(out + tt * 2)
    o = fx.ptr_load(out + (tt * 2 + 1))
    cs, sn = rope
    # inverse rotation, kept fp32 (bf16 products are exact: any contraction agrees)
    e1 = e * cs + o * sn
    o1 = o * cs - e * sn
    amax = butterfly(fx.max(abs(e1), abs(o1)), (1, 2, 4, 8), fx.max)
    code = code_ceil(fx.max(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    inv = rcp_pow2(code)
    # this thread's two bytes and its neighbour's make a word (4 dims)
    q0, q1 = clamp_fp8(e1 * inv), clamp_fp8(o1 * inv)
    n0, n1 = xshfl(q0, 1), xshfl(q1, 1)
    w = fp8_pack4(q0, q1, n0, n1)
    if mine & (tt % 2 == 0):
        c["put_words"](c["xo"], t * dims.xo_words + (head * HEAD_DIM + d) // 4, [w])
    if mine & (tt % 16 == 0):
        c["put"](
            c["xos"],
            t * (dims.heads * HEAD_DIM // 32) + (head * HEAD_DIM + d) // 32,
            code,
        )


@traced
def load_rows(
    c, words_mb, codes_mb, stride_w, words, stride_g, groups, part, xl, xsl,
    t0=0, n=None,
):  # fmt: skip
    """Tokens t0 .. t0 + n's (every token's by default) part ``part`` (words
    ``[part words, (part + 1) words)``) of a mailbox of rows ``stride`` words
    apart (and its codes), into LDS rows at the ``lds_row`` stride. ``n``
    traced (a tile a CTA picks at run time): a TILE-row copy of which only n
    rows are polled."""
    n = c["S"] if n is None else n
    rows = n if isinstance(n, int) else TILE
    poll_copy(
        c,
        c["tid"],
        THREADS,
        [
            (
                words_mb,
                strided_span(t0, stride_w, part, words, words),
                rows * words,
                xl,
                words,
            ),
            (
                codes_mb,
                strided_span(t0, stride_g, part, groups, groups),
                rows * groups,
                xsl,
                groups,
            ),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
    )
    gpu.barrier()


@traced
def stage_woa(c, task, ops, t0=0, n=None):
    """Rows 32 task .. of wo_a (two tiles, ``woa_loads``' operands) -> bf16 ->
    ATOM ``quantize_fp8`` of the 32 (one group: amax floored AMAX_FLOOR, the ceil
    code) -> X8B / X8BS, wo_b's input; the token tile from ``t0`` (its x in
    LDS)."""
    tid = c["tid"]
    red, d = c["red"], c["d"]
    n = c["S"] if n is None else n
    split = WOA_SPLIT_K
    gemv_fp8_mfmas(c, d.group_k, c["xl"], c["xsl"], red, ops, split, tiled=True, rows=n)
    gpu.barrier()
    # thread (t, row): a token's 32 rows are 32 lanes of one wave
    mine = tid < WOA_ROWS * n
    tl = fx.min(tid // WOA_ROWS, n - 1)
    t = plus(t0, tl)
    row = tid % WOA_ROWS
    tile = row // ROWS
    y = fx.Float32(
        acc_sum(red, row % ROWS, tl, [tile * split + w for w in range(split)]).to(
            fx.BFloat16
        )
    )
    amax = fx.max(butterfly(abs(y), (1, 2, 4, 8, 16), fx.max), fx.Float32(AMAX_FLOOR))
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    q = clamp_fp8(y * rcp_pow2(code))
    # rows row .. row + 3 (row a multiple of 4) make a word
    lane = c["lane"]
    w = fp8_pack4(
        q,
        *[
            lane_gather(q.bitcast(fx.Int32), fx.min(lane + k, 63)).bitcast(fx.Float32)
            for k in (1, 2, 3)
        ],
    )
    if mine & (row % 4 == 0):
        c["put_words"](c["x8b"], t * d.yb_words + (task * WOA_ROWS + row) // 4, [w])
    if mine & (row == 0):
        c["put"](c["x8bs"], t * (d.o_rows // 32) + task, code)
    gpu.barrier()


@traced
def woa_tile(c, task, ops, t0, n):
    """wo_a task ``task`` (``woa_loads``' ``ops``) on tokens t0 .. t0 + n: its
    group's attention output rows into LDS, then ``stage_woa``."""
    d = c["d"]
    g = task // (d.woa_tasks // d.groups)
    load_rows(
        c, c["xo"], c["xos"], d.xo_words, d.group_k // 4,
        d.heads * HEAD_DIM // 32, d.group_k // 32, g, c["xl"], c["xsl"], t0, n,
    )  # fmt: skip
    stage_woa(c, task, ops, t0, n)


def woa_loads(c, task):
    a = c["args"]
    split = WOA_SPLIT_K
    return gemv_fp8_loads(
        c, a["woa"], a["woa_s"], c["d"].group_k, task * (WOA_ROWS // ROWS), split,
        tiled=True,
    )  # fmt: skip


def wob_loads(c, task, part=0):
    """wo_b task ``task``'s weight operands over K part ``part``
    (``Dims.wob_parts``)."""
    a, d = c["args"], c["d"]
    if const_expr(d.wob_parts == 1):
        return gemv_fp8_loads(c, a["wob"], a["wob_s"], d.o_rows, task)
    return gemv_fp8_loads(
        c, a["wob"], a["wob_s"], d.wob_k, task,
        k_row=d.o_rows, k0=part * d.wob_k,
    )  # fmt: skip


def load_wob_rows(c, part, t0, n):
    """wo_b's input over K part ``part`` (X8B / X8BS) of tokens t0 .. t0 + n
    into LDS ``xl`` / ``xsl``."""
    d = c["d"]
    load_rows(
        c, c["x8b"], c["x8bs"], d.yb_words, d.wob_k // 4,
        d.o_rows // 32, d.wob_k // 32, part, c["xl"], c["xsl"], t0, n,
    )  # fmt: skip


def stage_wob(c, task, ops, t0=0, n=None, part=0):
    """Rows 16 task .. of wo_b over K part ``part`` (``wob_loads``' operands)
    -> this rank's bf16 partial of the part, pushed to every rank (token
    parallel: to the token's owner) as source ``rank parts + part``; the token
    tile from ``t0`` (its x in LDS)."""
    wob_mfma(c, ops, n)
    wob_push(c, task, t0, n, part)
    gpu.barrier()


@traced
def stage_wob_adjacent_pair(c, pair):
    """wo_b tasks 2 pair, 2 pair + 1 on this CTA (none past WOB_TASKS / 2)."""
    if pair < WOB_TASKS // 2:
        pair_tasks = [2 * pair, 2 * pair + 1]
        pair_ops = [wob_loads(c, t) for t in pair_tasks]
        load_wob_rows(c, 0, 0, c["S"])
        stage_wob_pair(c, pair_tasks, pair_ops, True, 0)


def wob_to_owner(c) -> bool:
    """Whether wo_b pushes a partial to one rank only (``wob_push``): a
    column's to its owner (a sharded seam F) or a token's (token parallel)."""
    return c["shard_f"] or c["tok_par"]


def seam_a_gates0(s, d, to_owner, first) -> int:
    """The layer kernel's first CTA of seam A's gates, which run after the
    merge and only seam F reads: a CTA without a wo_a task -- seam F's first
    slice CTAs in the one-tile ring (``seam_rot``: no wo_a / wo_b task there);
    with every token's wo_b pairs, which wrap onto wo_a's CTAs, the CTAs past
    wo_a's units (wo_b's, which wait for wo_a anyway); else ``GATE0``."""
    tiles = len(token_tiles(s))
    if tiles > 1 and wob_pairs_tiles(s, d, to_owner):
        return (woa0(s, d.head_tiles.count, first) + d.woa_tasks * tiles) % BLOCKS
    rot = seam_rot(s, d, to_owner, first)
    return rot if rot > 0 else GATE0


def wob_pairs_tiles(s, d, to_owner) -> bool:
    """Past one token tile, whether wo_b runs a pair of tasks a CTA on every
    token: where a partial goes to one rank only (``to_owner``:
    ``wob_to_owner``; to every rank, the (tile, task) units), up to
    ``WOB_PAIRS_MAX`` tokens, every token's input (one K part) in the LDS rows
    ``xl`` holds (a tile's wo_a group rows)."""
    return (
        to_owner and TILE < s <= WOB_PAIRS_MAX and d.wob_parts == 1
        and s * lds_row(d.wob_k // 4) <= TILE * lds_row(d.group_k // 4)
    )  # fmt: skip


@traced
def stage_wob_adjacent_pair_tiles(c, pair, tiles):
    """``stage_wob_adjacent_pair`` past one token tile: every token's rows in LDS,
    each K step's weight operands on every tile [(t0, n)] (each tile's chain
    in ``gemv_fp8_mfmas``' order: bits equal); then a tile at a time, the two
    tasks' partials through ``red``'s halves."""
    if pair < WOB_TASKS // 2:
        lane, wave, d = c["lane"], c["wave"], c["d"]
        tasks = [2 * pair, 2 * pair + 1]
        ops = [wob_loads(c, t) for t in tasks]
        load_wob_rows(c, 0, 0, c["S"])
        accs = [
            gemv_fp8_tile_accs(c, d.wob_k, c["xl"], c["xsl"], o, tiles) for o in ops
        ]
        reds = [c["red"], c["red"] + WAVES * 64 * 4]
        for j, (t0, n) in enumerate(tiles):
            for i in range_constexpr(2):
                fx.ptr_store(
                    accs[i][j], reds[i] + acc_word(wave, lane // 16, lane % 16)
                )
            gpu.barrier()
            for i in range_constexpr(2):
                wob_push(c, tasks[i], t0, n, red=reds[i])
            gpu.barrier()


def seam_rot(s, d, to_owner, first) -> int:
    """Seam F's first CTA, and every later stage's placement turned by as much
    (the norm, the gate, K2b's): past wo_b's CTAs in the paired layout (one
    token tile, or every token's pairs: ``wob_pairs_tiles``) -- the ring wo_a,
    then wo_b, then the slices; else 0. ``first``: the score stage's first CTA
    (``AttnPostBuild.score0``)."""
    tiles = len(token_tiles(s))
    if (tiles == 1 and d.wob_parts == 1) or wob_pairs_tiles(s, d, to_owner):
        woa_units = d.woa_tasks * tiles
        woa = woa0(s, d.head_tiles.count, first)
        return (woa + woa_units + WOB_TASKS // 2) % BLOCKS
    return 0


@traced
def stage_wob_pair(c, tasks, ops, has_second, part):
    """A CTA's one or two wo_b tasks (``stage_wob``) on every token: the
    second's GEMV into ``red``'s upper half before the first's barrier, then
    both pushed -- the pair's MFMAs and row sums together, not one task's after
    the other's."""
    red2 = c["red"] + WAVES * 64 * 4
    gemv_fp8_mfmas(c, c["d"].wob_k, c["xl"], c["xsl"], c["red"], ops[0])
    if has_second:
        gemv_fp8_mfmas(c, c["d"].wob_k, c["xl"], c["xsl"], red2, ops[1])
    gpu.barrier()
    wob_push(c, tasks[0], part=part)
    if has_second:
        wob_push(c, tasks[1], part=part, red=red2)
    gpu.barrier()


def wob_mfma(c, ops, n=None):
    """``stage_wob``'s GEMV into LDS ``red``: once it is issued, the registers
    of ``ops`` are free for the next task's loads."""
    n = c["S"] if n is None else n
    gemv_fp8_mfmas(c, c["d"].wob_k, c["xl"], c["xsl"], c["red"], ops, rows=n)
    gpu.barrier()


@traced
def wob_push(c, task, t0=0, n=None, part=0, red=None):
    """``stage_wob``'s row sums of ``red`` (LDS ``red`` by default) pushed as
    source ``rank parts + part``; the caller's barrier frees ``red``."""
    s, tid, d = c["S"], c["tid"], c["d"]
    red = c["red"] if red is None else red
    n = s if n is None else n
    src = c["rank"] * d.wob_parts + part if d.wob_parts > 1 else c["rank"]
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        col = task * ROWS + 2 * rp
        v0 = acc_sum(red, 2 * rp, tl)
        v1 = acc_sum(red, 2 * rp + 1, tl)
        if const_expr(c["shard_f"]):
            # a sharded seam F: the column's owner only (a reduce-scatter)
            owner = col // (HIDDEN // d.tp)
            for p in range_constexpr(d.tp):
                if owner == p:
                    c["put_bf"](
                        preg(c["peer_addr"](p), c["peer"]["attn"], "attn"),
                        2 * attn_region_pair(src, t, s, col), [v0, v1],
                    )  # fmt: skip
        elif const_expr(c["tok_par"]):
            # to the token's owner only, at its own token index there
            own_n = own_count(s, d.tp)
            owner = t // own_n
            for p in range_constexpr(d.tp):
                if owner == p:
                    dst = preg(c["peer_addr"](p), c["peer"]["attn"], "attn")
                    pair = attn_region_pair(src, t - p * own_n, own_n, col)
                    c["put_bf"](dst, 2 * pair, [v0, v1])
        else:
            for p in range_constexpr(d.tp):
                dst = preg(c["peer_addr"](p), c["peer"]["attn"], "attn")
                c["put_bf"](dst, 2 * attn_region_pair(src, t, s, col), [v0, v1])


@traced
def stage_reduce(c, task):
    """The attention output at this slice's 32 columns: the TP ranks' partials
    in the all-reduce's order (``sum_partials``), in fp32 -> bf16 ->
    ``pend_lds``."""
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * k1.COLS // 2, THREADS):
        reduce_attn_pair(c, task, plus(pas, tid))
    gpu.barrier()


@traced
def reduce_attn_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 16 (``stage_reduce``)."""
    s = c["S"]
    c0 = task * k1.COLS
    if idx < s * k1.COLS // 2:
        t = idx // (k1.COLS // 2)
        col = c0 + 2 * (idx % (k1.COLS // 2))
        own = preg(c["sym"], c["peer"]["attn"], "attn")
        acc0, acc1 = sum_partials(
            c["poll"],
            own,
            lambda src: attn_region_pair(src, t, s, col),
            c["d"].attn_sources,
        )
        lds = c["pend_lds"] + (t * k1.COLS + col - c0)
        fx.ptr_store(bf16_round(acc0), lds)
        fx.ptr_store(bf16_round(acc1), lds + 1)


def attn_post_args(*values) -> dict:
    """K2a's kernel arguments by name (``ABI`` order, less the shared tail)."""
    names = ABI.names[: ABI.names.index("scratch")]
    assert len(values) == len(names)
    out = dict(zip(names, values))
    out["attn_w"] = out.pop("ffn_w")
    return out


def attn_post_smem(s, timeline, bound_max, d, kv_fp8=False) -> dict:
    """K2a's LDS, a struct a stage: the members of a union, so only one stage's
    buffers take space at a time (``run_attn_post`` fences each hand-over).
    Each opens with what every stage shares at the same offsets -- ``red`` and
    the stamp record (``timeline``). ``kv_fp8``: the score's MXFP8 q operands
    (``attention._q_shared``)."""
    shared = {
        # the reductions' words, or the score's padded q tile (``_q_tile``)
        "red": fx.Array[fx.Float32, max(k1.WAVES * 2 * 64 * 4, Q_LDS_WORDS), 16],
        "tls": fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16],
    }
    stages = {
        # the score's key rows, 16 a wave (``attention``)
        "score": {
            "sc_kv": fx.Array[fx.Int32, WAVES * SCORE_LDS_WORDS, 16],
            "sc_q": fx.Array[fx.Int32, SCORE_Q_WORDS if kv_fp8 else 1, 16],
        },
        "sel": {
            "sk": fx.Array[fx.Int32, SEL_KEYS if bound_max else 1, 16],
            "sh": fx.Array[fx.Int32, SEL_HIST, 16],
            "sb": fx.Array[fx.Int32, SEL_BUF, 16],
            # the radix select (``index_score._select_sel``) and the
            # winners' bits by place
            "sx": fx.Array[fx.Int32, SEL_RADIX_WORDS if bound_max else 1, 16],
            "sbits": fx.Array[fx.Int32, SEL_BITS_WORDS if bound_max else 1, 16],
        },
        # the merge's values, p, statistics and output (``attention``)
        "attn": {
            "at_vt": fx.Array[fx.Int32, MERGE_V_WORDS, 16],
            "at_p": fx.Array[fx.Int32, MERGE_P_WORDS, 16],
            "at_mx": fx.Array[fx.Float32, MERGE_STAT_WORDS, 16],
            "at_sum": fx.Array[fx.Float32, MERGE_SUM_WORDS, 16],
            "at_stat": fx.Array[fx.Float32, 2 * HEAD_TILE, 16],
            "at_tree": fx.Array[fx.Float32, MERGE_TREE_WORDS, 16],
            "at_out": fx.Array[fx.Float32, MERGE_OUT_WORDS, 16],
        },
        # wo_a's group row, and wo_b's input (``yb_words``, not longer)
        "gemv": {
            "xl": fx.Array[fx.Int32, tile_rows(s) * lds_row(d.group_k // 4), 16],
            "xsl": fx.Array[fx.Int32, tile_rows(s) * lds_row(d.group_k // 32), 16],
        },
        "slice": {
            "rl": fx.Array[fx.Float32, s * k1.KT_ROW, 16],
            "fl": fx.Array[fx.Float32, k1.MIX * k1.KT_ROW, 16],
            "pl": fx.Array[fx.Float32, s * k1.COLS, 16],
        },
    }
    return {
        f"k2a_{name}": lds_struct(f"K2a{name.title()}Lds", {**shared, **fields})
        for name, fields in stages.items()
    }


def attn_post_context(key, s, lds, mb, bases, layout, scratch, args, rank, sym):
    """K2a's stage context: LDS (``attn_post_smem``'s member views by name),
    mailbox, scratch regions, arguments."""
    tid = fx.thread_idx.x
    sl, gm, at, se, sc = (
        lds[f"k2a_{n}"] for n in ("slice", "gemv", "attn", "sel", "score")
    )
    c = {
        "S": s, "tid": tid, "bid": fx.block_idx.x, "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr, "fl": sl.fl.ptr, "red": gm.red.ptr,
        "xl": gm.xl.ptr, "xsl": gm.xsl.ptr, "pend_lds": sl.pl.ptr,
        "att": {
            "vt": at.at_vt.ptr, "p": at.at_p.ptr, "mx": at.at_mx.ptr,
            "sum": at.at_sum.ptr, "stat": at.at_stat.ptr, "tree": at.at_tree.ptr,
            "out": at.at_out.ptr, "score_kv": sc.sc_kv.ptr, "score_q": sc.sc_q.ptr,
        },
        "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words,
        "poll": mb.poll, "peer_addr": lambda p: bases[p], "rank": rank, "sym": sym,
        "peer": peer_offsets(s, Dims(key.tp)),
        "fold": True, "index": False, "aux": False, "ffn": True,
        "index_fp4": key.index_fp4, "index_mfma": key.index_mfma,
        "index_split": key.index_split, "staged_answer": key.staged_answer,
        "tok_par": key.tok_par, "kv_fp8": key.kv_fp8,
        "sel_slots": aplan.selection_slots(key.attn_sel),
        "block_rows": key.block_rows,
        "args": args, "d": Dims(key.tp),
        # the layer kernel (``layer``): K1's stages ran before, in this launch
        "fused": False,
        # a sharded seam F (``layer``): its outputs' every rank's copy (``put_out``)
        "shard_f": False, "fan": {}, "dst": None,
    }  # fmt: skip
    c.update(region_addrs(scratch, layout, key.numa, args["numa_word"]))
    if const_expr(key.tok_par and key.index_bound_max > 0):
        c["irdy"] = preg(sym, c["peer"]["irdy"], "irdy")
    c["sel_lds"] = {
        "keys": se.sk.ptr, "hist": se.sh.ptr, "buf": se.sb.ptr,
        "hi": se.sx.ptr, "bits": se.sbits.ptr,
    }  # fmt: skip
    return c


@traced
def run_attn_post(
    c, key, bid, tls, tl0, after_norm=None, gates=True, before_woa=None,
    after_index=None,
):  # fmt: skip
    """K2a's stages, in order; ``timeline`` stamps from point ``tl0`` + 1.
    ``after_norm(t)``: once token t's ``normed`` row is written (a launch that
    also reads it: ``layer_post``). ``gates``: the FFN gates last (else the
    caller runs ``run_gates`` later and stamps it). ``before_woa()``: work
    after the attention's merge, before wo_a (the layer kernel: seam A's
    gates, off K1's chain). ``after_index()``: work after the indexer, before
    the attention (the layer kernel's indexer layer: K1's wq_b)."""
    s, timeline, tid, d = key.tokens, key.timeline, c["tid"], c["d"]
    bound_max = key.index_bound_max
    tiles = d.head_tiles.count
    if const_expr(bound_max > 0):  # noqa: SIM102
        if c["args"]["iact"] != 0:
            run_indexer(c, s, bid, bound_max)
    # the stages' LDS is one union (``attn_post_smem``): a barrier hands it over
    gpu.barrier()
    if const_expr(after_index is not None):
        after_index()
        gpu.barrier()
    stamp(timeline, tls, tid, tl0 + 1)
    enter_stage("k2a.attn_score")
    per, scores = score_tasks(s, tiles)
    for task in range(first_task(bid, key.score0), scores, BLOCKS):
        stage_attn_score(c, task, per)
    stamp(timeline, tls, tid, tl0 + 2)
    enter_stage("k2a.irq")
    merges = merge_tasks(s, tiles)
    if const_expr(aplan.merge_one_test(s, merges, BLOCKS)):
        merge_one = first_task(bid, merge0(s, tiles, key.score0))
        if merge_one < merges:
            # a merge with no live head writes nothing: not run (bit-exact) --
            # not in an indexer layer, whose attention waits on the selection
            # (the skip buys nothing there; its build stays as it was)
            if const_expr(d.heads % HEAD_TILE == 0 or bound_max > 0):
                stage_attn_irq(c, merge_one)
            else:  # a compile-time test, then a traced one
                if merge_task_live(merge_one, s, d):
                    stage_attn_irq(c, merge_one)
    else:
        for task in range(
            first_task(bid, merge0(s, tiles, key.score0)), merges, BLOCKS
        ):
            stage_attn_irq(c, task)
    if const_expr(before_woa is not None):
        gpu.barrier()
        before_woa()
    woa_0 = woa0(s, tiles, key.score0)
    gpu.barrier()
    stamp(timeline, tls, tid, tl0 + 3)
    enter_stage("k2a.wo_a")
    # a CTA's wo_a task (at most one): its weights in flight before the
    # attention output's wait
    assert d.woa_tasks <= BLOCKS
    tiles = token_tiles(s)
    if const_expr(len(tiles) == 1):
        woa_task = first_task(bid, woa_0)
        if woa_task < d.woa_tasks:
            woa_ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, woa_ops, 0, s)
    elif const_expr(d.woa_tasks == BLOCKS):
        # TP 1 (DP attention): a task a CTA fills every CTA already, so a CTA
        # runs its task's token tiles in series on its weights, loaded once
        # (EP 24 / 48 rows -3.4 / -5.1 us)
        woa_task = first_task(bid, woa_0)
        woa_ops = woa_loads(c, woa_task)
        for t0, n in tiles:
            woa_tile(c, woa_task, woa_ops, t0, n)
    else:
        # a (task, token tile) a unit, a CTA each: a task's tiles in series on
        # one CTA left three quarters of the CTAs idle through wo_a (its weights
        # read again a tile: L2 hits)
        for unit in range(first_task(bid, woa_0), d.woa_tasks * len(tiles), BLOCKS):
            woa_task = unit % d.woa_tasks
            t0 = unit // d.woa_tasks * TILE
            woa_ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, woa_ops, t0, fx.min(fx.Int32(TILE), s - t0))
    stamp(timeline, tls, tid, tl0 + 4)
    enter_stage("k2a.wo_b")
    assert d.wob_k // 4 <= d.group_k // 4  # xl holds a part
    assert BLOCKS < WOB_TASKS <= 2 * BLOCKS
    if const_expr(len(tiles) == 1 and d.wob_parts == 2):
        # TP 1 (DP attention): wo_a takes every CTA, so wo_b's weights wait for
        # it; a CTA a K part of 2 or 3 tasks (half the CTAs a part), every
        # task's weights in flight at once and the part's rows loaded once: one
        # HBM round, not a part's after the other's
        unit = first_task(bid, woa_0 + d.woa_tasks)
        part, slot = unit % 2, unit // 2
        per = BLOCKS // 2
        tasks = [plus(slot, per * k) for k in range(-(-WOB_TASKS // per))]
        ops = [wob_loads(c, fx.min(t, WOB_TASKS - 1), part) for t in tasks]
        load_wob_rows(c, part, 0, s)
        for k in range_constexpr(len(tasks)):
            if tasks[k] < WOB_TASKS:
                stage_wob(c, tasks[k], ops[k], 0, s, part)
    elif const_expr(len(tiles) == 1 and d.wob_parts == 1):
        # wo_b off the wo_a CTAs, two adjacent 16-row tasks a CTA
        stage_wob_adjacent_pair(c, first_task(bid, woa_0 + d.woa_tasks))
    elif const_expr(len(tiles) == 1):
        # every CTA has one or two wo_b tasks: both tasks' weights in flight
        # before y's wait (a CTA with one loads its own again: L2 hits); a
        # later K part's after the part before it
        wob_task = first_task(bid, woa_0 + d.woa_tasks)
        wob_second = wob_task + BLOCKS
        has_second = wob_second < WOB_TASKS
        second = has_second.select(wob_second, wob_task)
        wob_ops = wob_loads(c, wob_task)
        wob_ops2 = wob_loads(c, second)
        for part in range_constexpr(d.wob_parts):
            if const_expr(part > 0):
                wob_ops = wob_loads(c, wob_task, part)
                wob_ops2 = wob_loads(c, second, part)
            load_wob_rows(c, part, 0, s)
            stage_wob_pair(
                c, [wob_task, wob_second], [wob_ops, wob_ops2], has_second, part
            )
    elif const_expr(d.wob_parts == 2):
        # TP 1 past one tile: a CTA a (tile, K part) and its tasks, as above
        vt, wob_tasks = tile_tasks(
            bid, woa_0 + d.woa_tasks * len(tiles), 2 * len(tiles), WOB_TASKS
        )
        part, tile = vt % 2, vt // 2
        t0 = tile * TILE
        n = fx.min(fx.Int32(TILE), s - t0)

        def loads(k):
            return wob_loads(c, fx.min(wob_tasks[k], WOB_TASKS - 1), part)

        # WOB_LIVE tasks' weights in flight, the next issued as a GEMV frees
        # its registers (all of a CTA's five at once spilled)
        live = [loads(k) for k in range(min(WOB_LIVE, len(wob_tasks)))]
        load_wob_rows(c, part, t0, n)
        for k in range_constexpr(len(wob_tasks)):
            ops = live[k % WOB_LIVE]
            wob_mfma(c, ops, n)
            if const_expr(k + WOB_LIVE < len(wob_tasks)):
                live[k % WOB_LIVE] = loads(k + WOB_LIVE)
            if wob_tasks[k] < WOB_TASKS:
                wob_push(c, wob_tasks[k], t0, n, part)
            gpu.barrier()
    elif const_expr(wob_pairs_tiles(s, d, wob_to_owner(c))):
        # two adjacent 16-row tasks a CTA on every token, weights read once
        stage_wob_adjacent_pair_tiles(
            c, first_task(bid, woa_0 + d.woa_tasks * len(tiles)), tiles
        )
    else:
        tile, wob_tasks = tile_tasks(
            bid, woa_0 + d.woa_tasks * len(tiles), len(tiles), WOB_TASKS
        )
        t0 = tile * TILE
        n = fx.min(fx.Int32(TILE), s - t0)
        for part in range_constexpr(d.wob_parts):
            wob_ops = [wob_loads(c, fx.min(t, WOB_TASKS - 1), part) for t in wob_tasks]
            load_wob_rows(c, part, t0, n)
            for k in range_constexpr(len(wob_tasks)):
                if wob_tasks[k] < WOB_TASKS:
                    stage_wob(c, wob_tasks[k], wob_ops[k], t0, n, part)
    stamp(timeline, tls, tid, tl0 + 5)
    # token parallel: this rank's tokens from here on (``after_norm`` gets the
    # token's index among them)
    co = own_view(c, K2A_OWN) if const_expr(key.tok_par) else c
    n = co["S"]
    enter_stage("k2a.slice")
    # a sharded seam F: this rank's quarter of the slices (its columns)
    shard = co["shard_f"]
    own = k1.SLICES // d.tp if shard else k1.SLICES
    # a sharded seam: a task a (slice of this rank's, destination rank)
    units = own * d.tp if shard else k1.SLICES
    rot = seam_rot(s, d, wob_to_owner(co), key.score0)
    if const_expr(shard):
        # at most a task a CTA: its destination set outside the task's region
        assert units <= BLOCKS
        unit = first_task(bid, rot)
        co["dst"] = unit // own
        task = co["rank"] * own + unit % own
        if unit < units:
            seam_f_task(co, task)
        co["dst"] = None
    else:
        for task in range(first_task(bid, rot), units, BLOCKS):
            seam_f_task(co, task)
    if const_expr(shard):
        # this rank's gate blocks, on the CTAs past its slices'
        enter_stage("k2a.gate_block")
        blocks = own // k1.shard_gate_block(co["d"].tp)
        for unit in range(first_task(bid, rot + units), blocks, BLOCKS):
            k1.stage_gate_block(co, co["rank"] * blocks + unit)
    stamp(timeline, tls, tid, tl0 + 6)
    # the norm before the gate: the MoE waits for it, the gate's outputs are
    # the next layer's
    enter_stage("k2a.norm")
    for t in range(first_task(bid, NORM0 + rot), n, BLOCKS):
        stage_norm(co, t)
        if const_expr(after_norm is not None):
            after_norm(t)
    stamp(timeline, tls, tid, tl0 + 7)
    if const_expr(gates):
        run_gates(c, key, bid)
        stamp(timeline, tls, tid, tl0 + 8)


def seam_f_task(c, task):
    """Seam F's slice ``task``: its inputs awaited, its residual and gate loads in
    flight through the partials' wait, the partials reduced, the slice."""
    await_seam_inputs(c, task)
    pre = slice_prefetch(c, task)
    stage_reduce(c, task)
    stage_slice(c, task, pre)


@traced
def await_seam_inputs(c, task):
    """The layer kernel: slice ``task``'s residual columns (K1's slice task
    ``task``, SDONE) and every token's gates (GDONE) written, polled at K1's
    tag."""
    if const_expr(c["fused"]):
        tid = c["tid"]
        c["k1_poll"]([(c["sdone"], task, 1), (c["gdone"], tid % c["S"], 1)])
        gpu.barrier()


@traced
def run_gates(c, key, bid, base=None):
    """The FFN seam's gates (``stage_gate``) -> the next layer's post / comb /
    pre: nothing in this launch waits on them. From CTA ``base`` (default
    past the seam's slices)."""
    co = own_view(c, K2A_OWN) if const_expr(key.tok_par) else c
    if const_expr(base is None):
        base = GATE0 + seam_rot(key.tokens, c["d"], wob_to_owner(c), key.score0)
    enter_stage("k2a.gate")
    for t in range(first_task(bid, base), co["S"], BLOCKS):
        stage_gate(co, t)


def build_attn_post(key: AttnPostBuild):
    s = key.tokens
    timeline = key.timeline
    assert 1 <= s <= MAX_TOKENS
    bound_max = key.index_bound_max
    layout = scratch_layout(key)
    keyed = key_tuple(key, SOURCES)
    members = attn_post_smem(s, timeline, bound_max, Dims(key.tp), key.kv_fp8)
    Smem = lds_union("K2aLds", members)

    name = kernel_symbol(
        "v41_attn_post",
        s=s,
        tl=int(timeline),
        ix=int(bound_max > 0),
        i4=int(key.index_fp4),
        im=int(key.index_mfma),
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_post(
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, ixsel: Int64,
        sbound: Int64,
        q: Int64, pool: Int64, pool_scale: Int64, window: Int64, window_scale: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64, scratch: Int64, numa_word: Int64, sym: Int64, peers: Int64,
        rank: Int32, layer: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Smem)
        lds = {name: getattr(smem, name).peek() for name in members}
        args = attn_post_args(
            iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
            ibatch, ishift, ibound, ilog, lstride,
            isel, icand, ilift, iprod, ibmax, bstride, icout, irow, ixsel, sbound,
            q, pool, pool_scale, window, window_scale, sel, topk, kmeta, table, rows_per_page, page_rows, main_off,
            ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob,
            wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
            res_out, post_out, comb_out, pre_out, normed,
        )  # fmt: skip
        args["numa_word"] = numa_word
        c = attn_post_context(
            key, s, lds, mailbox(layer, scratch, key.diag_off, layout),
            peer_bases(peers, key.tp), layout, scratch, args,
            rank, sym,
        )  # fmt: skip
        tls = lds["k2a_gemv"].tls.ptr
        stamp_begin(timeline, tls, tid, TL_POINTS)
        run_attn_post(c, key, bid, tls, 0)
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @launcher(name)
    def launch(
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, ixsel: Int64,
        sbound: Int64,
        q: Int64, pool: Int64, pool_scale: Int64, window: Int64, window_scale: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64, scratch: Int64, numa_word: Int64, sym: Int64, peers: Int64,
        rank: Int32, layer: Int32, tl: Int64, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        attn_post(
            iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
            ibatch, ishift, ibound, ilog, lstride,
            isel, icand, ilift, iprod, ibmax, bstride, icout, irow, ixsel, sbound,
            q, pool, pool_scale, window, window_scale, sel, topk, kmeta, table, rows_per_page, page_rows, main_off,
            ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob,
            wob_s, res_in, post_in, comb_in,
            pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out,
            pre_out, normed, scratch, numa_word, sym, peers, rank, layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(attn_post, launch)
    return launch
