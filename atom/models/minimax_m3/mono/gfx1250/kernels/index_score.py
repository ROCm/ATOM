# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 indexer block scores (``kernels.index_score`` on wave32 and WMMA).

The tasks, their placement and the mailbox they fill are the gfx950 module's.
A task's key tile is a 16x16x32 bf16 WMMA per 32 dims: the two lanes of a key
read its 128 fp8 bytes as 64 contiguous bytes each (lane half g: dims 64 g ..
64 g + 64), converted by v_cvt_scale_pk8_bf16_fp8, and index q is fed in the
same per-lane dim order, so no transpose is needed.
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    ONE_INDEX_HEAD,
    SPARSE_BLOCK,
    TP,
    IndexHeads,
)
from atom.models.minimax_m3.mono.kernels.index_score import (
    INDEX_BLOCK_BYTES,
    index_scale_log2e,
    scored_count,
    step_rows,
)
from atom.models.minimax_m3.mono.sources import SOURCES
from atom.mono.device.gfx1250 import (
    BLOCKS,
    THREADS,
    WAVE,
    WAVES,
    fp8x8_bf16,
    lds_barrier,
    wmma_bf16_k32,
    xshfl,
)
from atom.mono.device.ops import (
    kernel_symbol,
    lane_gather,
    launcher,
    readlane,
    rsrc,
    traced,
    uniform,
    unlikely,
)
from atom.mono.device.sync import Addr, Mailbox
from atom.mono.plan.arith import pick
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.trace import Space

KEY_TILES = SPARSE_BLOCK // 16  # WMMA row tiles of a block
CHUNKS = HEAD_DIM // 32  # k32 chunks of a key
CP_TILES = 1  # key tiles a context-parallel task holds at a time
TASK_TILES = 4  # and a wave task (more than BLOCKS tasks)
_CURRENT_STREAM = fx.Stream(None)


def q_dims(lane, c):
    """The first of the 16 index-q dims lane ``lane`` feeds k-chunk ``c``."""
    return 64 * (lane // 16) + 16 * c


@traced
def emit_index_scores(
    slot, lane, wave, red, mb_put, iscore, q_frag, tokens, q_len, init_blocks,
    local_blocks, scale, index_cache, block_table, seq_lens, bt_width,
    wait_new_keys=None, prepare=None, heads=ONE_INDEX_HEAD, q_frag_head=None,
    prepare_heads=None, new_keys=None,
):  # fmt: skip
    """``kernels.index_score.emit_index_scores`` on gfx1250. ``q_frag(k, c)``:
    this lane's B operand for token k, k-chunk c (16 bf16 at ``q_dims``);
    ``red``: WAVES * 16 floats. ``new_keys``: (``make(first, last)``, LDS ``at``):
    tokens first .. last's index keys as their insert writes them, token first
    + j's HEAD_DIM / 4 words at ``at`` + j HEAD_DIM / 4, made by the CTA (behind
    ``prepare``'s barrier) instead of waited for (``wait_new_keys``) where a
    CTA's block holds the step's keys."""
    assert local_blocks >= 1, "the block of the current token must be pinned"
    l16 = lane % 16
    g = lane // 16
    r_ic = rsrc(index_cache)
    rows = step_rows(seq_lens, tokens, q_len, heads)
    seq = rows.seq
    cp = heads.count > 1
    first_own = init_blocks + (heads.own - init_blocks) % TP
    pre = [fx.Int32(0)]
    pre_cp = [fx.Int32(0)]
    for k in range_constexpr(tokens):
        last = fx.Int32(k) % q_len == q_len - 1
        n_scored = scored_count(rows.n_blks[k], init_blocks, local_blocks)
        if const_expr(cp):
            long = rows.long[k]
            n_own = fx.max(init_blocks + n_scored - first_own + TP - 1, 0) // TP
            pre.append(pre[-1] + (last & ~long).select(n_scored, fx.Int32(0)))
            pre_cp.append(pre_cp[-1] + (last & long).select(n_own, fx.Int32(0)))
        else:
            pre.append(pre[-1] + last.select(n_scored, fx.Int32(0)))
    total = pre[tokens]
    total_cp = pre_cp[-1]

    r_bt = rsrc(block_table)

    def task_block(u, pre=pre, first=init_blocks, stride=1):
        tk = fx.Int32(0)
        for j in range_constexpr(1, tokens):
            tk = tk + (u >= pre[j]).select(fx.Int32(1), fx.Int32(0))
        return tk, first + stride * (u - pick(pre[:tokens], tk))

    def page_of(tk, b):
        return fx.Int32(
            bo.buffer_load(r_bt, tk * bt_width + b, vec_width=1, dtype=T.i32)
        )

    def new_keys_in(tk, b):
        return (b + 1) * SPARSE_BLOCK > pick(seq, tk) - MAX_TOKENS

    def cols_of(tk):
        col = tk - (q_len - 1) + fx.min(l16, q_len - 1)
        return col, pick(seq, col)

    def load_keys(page, tiles):
        """Per tile, this lane's key 16 i + lane % 16, bytes 64 g .. 64 g + 64:
        four 16 B loads, one a k-chunk, in flight."""
        out = []
        for i in tiles:
            row = (page * INDEX_BLOCK_BYTES + (16 * i + l16) * HEAD_DIM + 64 * g) // 4
            out.append(
                [
                    fx.Vector(
                        bo.buffer_load(r_ic, row + 4 * c, vec_width=4, dtype=T.i32)
                    )
                    for c in range(CHUNKS)
                ]
            )
        return out

    def key_operand(w4):
        lo = fp8x8_bf16([w4[0], w4[1]])
        hi = fp8x8_bf16([w4[2], w4[3]])
        return fx.Vector.from_elements(
            [lo[e] for e in range(8)] + [hi[e] for e in range(8)], fx.BFloat16
        )

    def with_new_keys(w4s, tk, b, tile, at):
        """``load_keys``' chunks of key 16 tile + lane % 16 of block b, the step's
        keys of token tk's row (the last q_len positions of its sequence) taken
        from LDS ``at`` instead."""
        j = b * SPARSE_BLOCK + 16 * tile + l16 - (pick(seq, tk) - q_len)
        new = (j >= 0) & (j < q_len)
        jc = fx.min(fx.max(j, 0), q_len - 1)
        out = []
        for c in range_constexpr(CHUNKS):
            nv = fx.Vector(
                fx.ptr_load(
                    at + (jc * (HEAD_DIM // 4) + 16 * g + 4 * c),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            out.append(
                fx.Vector.from_elements(
                    [new.select(nv[e], w4s[c][e]) for e in range(4)], fx.Int32
                )
            )
        return out

    def tiles_max(b, qb, col_seq, tiles, kw):
        """Max over key tiles ``tiles`` of block b of this lane's column's causal
        scores, the same in both lanes of the column."""
        v = fx.Float32(float("-inf"))
        for n, i in enumerate(tiles):
            c = fx.Vector.filled(8, 0.0, fx.Float32)
            for s in range_constexpr(CHUNKS):
                c = wmma_bf16_k32(key_operand(kw[n][s]), qb[s], c)
            # lane holds keys 16 i + 8 g + e of column lane % 16
            key0 = b * SPARSE_BLOCK + 16 * i + 8 * g
            for e in range_constexpr(8):
                v = fx.max(v, (key0 + e < col_seq).select(c[e], float("-inf")))
        return fx.max(v, xshfl(v, 16))

    def put(b, col, v):
        if (lane < 16) & (l16 < q_len):
            mb_put(iscore, col * MAX_INDEX_BLOCKS + b, v * scale)

    def emit_cp_tasks():
        if const_expr(prepare_heads is not None):
            prepare_heads()
        seq_lane = pick(seq, fx.min(lane, tokens - 1))
        n_cols = TP * q_len

        passes = TP * MAX_TOKENS // 16

        def score_cp(tk, b, page):
            """Every pass's (head, row) columns over the block's key tiles: a pass
            reads the keys CP_TILES tiles at a time (every tile's keys held across
            the passes would not fit wave32's registers; the re-reads hit L2)."""
            if const_expr(wait_new_keys is not None):  # noqa: SIM102
                if (b + 1) * SPARSE_BLOCK > readlane(seq_lane, tk) - MAX_TOKENS:
                    wait_new_keys()
            for p in range_constexpr(passes):
                if p * 16 < n_cols:
                    f = p * 16 + l16
                    head = fx.min(f // q_len, TP - 1)
                    col = tk - (q_len - 1) + f % q_len
                    qb = [q_frag_head(col, s, head) for s in range(CHUNKS)]
                    col_seq = lane_gather(seq_lane, col)
                    v = fx.Float32(float("-inf"))
                    for t0 in range_constexpr(0, KEY_TILES, CP_TILES):
                        group = range(t0, t0 + CP_TILES)
                        v = fx.max(
                            v, tiles_max(b, qb, col_seq, group, load_keys(page, group))
                        )
                    if (lane < 16) & (f < n_cols):
                        mb_put(
                            iscore,
                            col * MAX_INDEX_BLOCKS
                            + head * (MAX_INDEX_BLOCKS // TP)
                            + b // TP,
                            v * scale,
                        )

        step = BLOCKS * WAVES
        for c0 in range(slot + BLOCKS * wave, total_cp, step * WAVE):
            c0 = fx.Int32(c0)
            tk_j, b_j = task_block(
                fx.min(c0 + step * lane, total_cp - 1), pre_cp, first_own, TP
            )
            pages = page_of(tk_j, b_j)
            for i in range(fx.min((total_cp - c0 + step - 1) // step, WAVE)):
                i = fx.Int32(i)
                score_cp(readlane(tk_j, i), readlane(b_j, i), readlane(pages, i))

    # Up to BLOCKS tasks: one a CTA, a key tile on each of waves 0 .. KEY_TILES,
    # the columns' maxima met in LDS. More: a task a wave.
    if total <= BLOCKS:
        tw = wave < KEY_TILES
        for u in range(slot, total, BLOCKS):
            tk, b = task_block(fx.Int32(u))
            page = uniform(page_of(tk, b))
            if const_expr(new_keys is not None):
                assert prepare is not None
                if new_keys_in(tk, b):
                    new_keys[0](tk - (q_len - 1), tk)
            elif const_expr(wait_new_keys is not None):
                if new_keys_in(tk, b):
                    wait_new_keys()
            tile = fx.min(wave, KEY_TILES - 1)
            kw = load_keys(page, [tile])
            if const_expr(prepare is not None):
                prepare(tk - (q_len - 1), tk)
            if const_expr(new_keys is not None):
                kw = [with_new_keys(kw[0], tk, b, tile, new_keys[1])]
            col, col_seq = cols_of(tk)
            if tw:
                qb = [q_frag(col, s) for s in range(CHUNKS)]
                v = tiles_max(b, qb, col_seq, [wave], kw)
                if lane < 16:
                    fx.ptr_store(v, red + (wave * 16 + l16))
            lds_barrier()
            if wave == 0:
                v = fx.ptr_load(red + l16)
                for w in range_constexpr(1, KEY_TILES):
                    v = fx.max(v, fx.ptr_load(red + (w * 16 + l16)))
                put(b, col, v)
            lds_barrier()
    if total > BLOCKS:
        if const_expr(prepare is not None):
            prepare(fx.Int32(0), fx.Int32(tokens - 1))
        seq_lane = pick(seq, fx.min(lane, tokens - 1))

        def score_task(tk, b, page):
            if const_expr(wait_new_keys is not None):  # noqa: SIM102
                if (b + 1) * SPARSE_BLOCK > readlane(seq_lane, tk) - MAX_TOKENS:
                    wait_new_keys()
            col = tk - (q_len - 1) + fx.min(l16, q_len - 1)
            col_seq = lane_gather(seq_lane, col)
            qb = [q_frag(col, s) for s in range(CHUNKS)]
            # TASK_TILES tiles' keys in flight at a time: all eight are 128
            # VGPRs on wave32
            v = fx.Float32(float("-inf"))
            for t0 in range_constexpr(0, KEY_TILES, TASK_TILES):
                group = range(t0, t0 + TASK_TILES)
                v = fx.max(v, tiles_max(b, qb, col_seq, group, load_keys(page, group)))
            put(b, col, v)

        step = BLOCKS * WAVES
        for c0 in range(slot + BLOCKS * wave, total, step * WAVE):
            c0 = fx.Int32(c0)
            tk_j, b_j = task_block(fx.min(c0 + step * lane, total - 1))
            pages = page_of(tk_j, b_j)
            for i in range(fx.min((total - c0 + step - 1) // step, WAVE)):
                i = fx.Int32(i)
                score_task(readlane(tk_j, i), readlane(b_j, i), readlane(pages, i))
    if const_expr(cp):  # noqa: SIM102
        if unlikely(total_cp > 0):
            emit_cp_tasks()


def q_frag_from(iq, heads=ONE_INDEX_HEAD):
    """(``q_frag``, ``q_frag_head``) reading index q from memory: [token][head]
    [HEAD_DIM] bf16, ``heads.count`` heads a token."""

    def q_frag_head(tk, c, head):
        lane = fx.thread_idx.x % WAVE
        base = ((tk * heads.count + head) * HEAD_DIM + q_dims(lane, c)) // 2
        w = [
            fx.Vector(bo.buffer_load(rsrc(iq), base + 4 * h, vec_width=4, dtype=T.i32))
            for h in range(2)
        ]
        return fx.Vector.from_elements(
            [w[h][e] for h in range(2) for e in range(4)], fx.Int32
        ).bitcast(fx.BFloat16)

    def q_frag(tk, c):
        return q_frag_head(tk, c, heads.own)

    return q_frag, q_frag_head


@dataclass(frozen=True)
class ScoreBuild:
    tokens: int = field(metadata={"sym": "s"})
    init_blocks: int = field(metadata={"sym": "ib"})
    local_blocks: int = field(metadata={"sym": "lb"})
    index_heads: int = field(metadata={"sym": "ih"})
    index_own: int = field(metadata={"sym": "io"})
    sm_scale: float


def build_index_score_kernel(
    sm_scale: float,
    init_blocks: int,
    local_blocks: int,
    tokens: int = 1,
    heads: IndexHeads = ONE_INDEX_HEAD,
):
    """``kernels.index_score.build_index_score_kernel`` on gfx1250."""
    assert 1 <= tokens <= MAX_TOKENS
    key = ScoreBuild(
        tokens=tokens, init_blocks=init_blocks, local_blocks=local_blocks,
        index_heads=heads.count, index_own=heads.own, sm_scale=sm_scale,
    )  # fmt: skip
    build_key = key_tuple(key, SOURCES)
    scale = index_scale_log2e(sm_scale)
    ih_count, ih_own = heads.count, heads.own

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 16, 16]

    kernel_name = kernel_symbol("minimax_m3_index_score_g1250", **symbol_params(key))

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def index_score_kernel(
        iq: Int64,
        index_cache: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        layer: Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(Smem).peek()
        red = lds.red.ptr
        mbox = Mailbox(layer, gfx1250=True)
        mb_put = mbox.put
        k_heads = IndexHeads(ih_count, ih_own)
        q_frag, q_frag_head = q_frag_from(iq, k_heads)
        emit_index_scores(
            fx.block_idx.x, tid % WAVE, uniform(tid // WAVE), red, mb_put,
            Addr("iscore", Space.SCRATCH, iscore),
            q_frag, tokens, q_len, init_blocks, local_blocks, scale, index_cache,
            block_table, seq_lens, bt_width, heads=k_heads, q_frag_head=q_frag_head,
        )  # fmt: skip

    @launcher(kernel_name)
    def launch_index_score(
        iq: Int64,
        index_cache: Int64,
        block_table: Int64,
        seq_lens: Int64,
        bt_width: Int32,
        q_len: Int32,
        iscore: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        _ = build_key
        index_score_kernel(
            iq, index_cache, block_table, seq_lens, bt_width, q_len, iscore, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch_index_score
