# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2a's indexer (a layer that selects: FULL, and REINDEX within layer 20's
candidates): the scores of every visible index row, the top-512 of them and, on
the candidate-producing layer, the top-2048 blocks -- the answer of the
original ``score_topk_paged`` (logits: gluon ``deepgemm_fp8_paged_mqa_logits``;
selection: ``top_k_per_row_decode(stable=True)``; blocks: ``pick_candidate_blocks``)
at any context length. Which cells each stage writes and polls is decided per
step only in ``index_plan`` (DESIGN_v3 4.9); the CPU test there checks it.

    iscore  (a task: a run of a request's tokens x up to TASK_CHUNKS chunks
            of 128 columns, ``index_plan.score_tasks``; a wave a tile pair):
            every key load first, then each token's scores (4 chained
            16x16x32 fp8 MFMAs, relu x k-scale, gluon's head sum;
            ``harness/p31_score.py``) -> global ``ilog``; a candidate
            producer's 8-row block maxima (NaN ignored, capped at the largest
            finite, the newest block +inf) -> global ``ibmax``; then, once a
            task, the flags (ISF) of its (token, chunk)s. On the FP4 plane
            ``index_score_fp4.run_scores4`` instead (the row-group scorer's
            arithmetic), writing the same cells
    ilist   (a level's lists, per token): ``topk.lds_topk`` of a segment's
            values (level 0, after its chunks' flags) or of the lists it merges
            (level l > 0, from ISEL<l-1> / IBLK<l-1>), in index order. A list
            that is its token's last -> global (``isel``, lifted to real rows on
            a REINDEX layer; ``icout``) and, for the selection, the ready flag
            (IRDY) the attention waits on; else -> ISEL<l> / IBLK<l>.

Data too large for tagged pairs (the logits, up to max_model_len a token) goes
through global memory at device scope (SC1: past the per-XCD L2): stored,
``s_waitcnt vmcnt(0)``, then the flag: a task's drain and barrier, hence
several chunks a task.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, T

from atom.mono.device.ops import (
    CM_SYS,
    butterfly,
    head_sum,
    ld_dev,
    ld_i32,
    mfma_fp8,
    rsrc,
    st_dev,
    traced,
)
from atom.mono.device.sync import POLL_MAX, preg, publish
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import selection_plan as isp
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.dsa_indexer.paged.plan import own_rows
from atom.mono.dsa_indexer.paged.score_fp4 import run_scores4
from atom.mono.dsa_indexer.paged.topk import key_bits, lds_topk
from atom.mono.dsa_indexer.paged.topk_fast import key_offset
from atom.mono.dsa_indexer.paged.tp_work import row_work
from atom.mono.plan.arith import pick, sel
from atom.mono.plan.execution import BLOCKS, THREADS, first_task
from atom.mono.plan.trace import enter_stage

TILE = ip.BLOCK_ROWS
# a score task's chunks at most: 1 << this, loads in flight together
TASK_CHUNKS_LOG2 = 3
TASK_CHUNKS = 1 << TASK_CHUNKS_LOG2
TILE_BYTES = TILE * ip.DIM + TILE * 4  # 8 preshuffled rows, then 8 fp32 scales
PER_THREAD = ip.SEG // THREADS
NEG = float("-inf")
FLT_MAX = 3.4028234663852886e38


def scratch_pairs(tokens: int, bound_max: int, tok_par: bool = False) -> dict[str, int]:
    """The indexer's mailbox regions. The last level's list is always a token's
    final one (global), so only levels below it have a region. A token-parallel
    launch's selection flags are peer regions (``peer_layout``: its owner's)."""
    out = {"isf": tokens * ip.chunks(bound_max)}
    if not tok_par:
        out["irdy"] = tokens
    for kind, k in ip.KINDS.items():
        for level in range(ip.top_level(kind, bound_max)):
            slots = ip.list_slots(kind, level, bound_max)
            out[f"i{kind}{level}"] = tokens * slots * k * 2
    return out


def _plane_ptr(plane, tile, byte):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
        plane + fx.Int64(tile) * TILE_BYTES + fx.Int64(byte),
    )


def _chunk_keys(a, t, ch0, width, wave, lane):
    """This lane's keys of chunks [ch0, ch0 + width) through token t's tile
    table (wave: a tile pair a chunk): every table load, then every key load,
    none waited on. A slot past ``width`` re-reads the last chunk (cache hits);
    a chunk past the table reads its last tile."""
    j = lane // 16
    r8 = lane % 8
    blks = [
        (ch0 + fx.min(width - 1, m)) * (ip.CHUNK // TILE) + 2 * wave + lane % 16 // 8
        for m in range(TASK_CHUNKS)
    ]
    tiles = [
        fx.Int32(
            bo.buffer_load(
                rsrc(a["itab"]),
                t * a["itab_stride"] + fx.min(blk, a["itab_len"] - 1),
                vec_width=1,
                dtype=T.i32,
            )
        )
        for blk in blks
    ]
    keys = []
    for blk, tile in zip(blks, tiles):
        ks = fx.Int32(
            fx.ptr_load(_plane_ptr(a["plane"], tile, TILE * ip.DIM + r8 * 4))
        ).bitcast(fx.Float32)
        kv = [
            fx.Vector(
                fx.ptr_load(
                    _plane_ptr(a["plane"], tile, ((4 * p + j) * TILE + r8) * 16),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            for p in range(2)
        ]
        b64 = fx.Vector.from_elements(
            [kv[p][d] for p in range(2) for d in range(4)], fx.Int32
        ).bitcast(fx.Int64)
        keys.append((blk, ks, b64))
    return keys


def _token_query(a, t, lane):
    """Token t's index q (this lane's MFMA A operand) and head weights, per
    16-head half."""
    j = lane // 16
    query = []
    for h in range(2):
        qw = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["iq"]),
                    ((t * ip.HEADS + h * 16 + lane % 16) * ip.DIM + (4 * p + j) * 16)
                    // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for p in range(2)
        ]
        a64 = fx.Vector.from_elements(
            [qw[p][d] for p in range(2) for d in range(4)], fx.Int32
        ).bitcast(fx.Int64)
        wts = [
            fx.Float32(
                bo.buffer_load(
                    rsrc(a["iw"]),
                    t * ip.HEADS + h * 16 + 4 * j + e,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            for e in range(4)
        ]
        query.append((a64, wts))
    return query


@traced
def _score_chunk(c, t, bound, blocks, query, blk, ks, b64):
    """Token t's scores (``bound`` its columns, ``blocks`` its candidate
    blocks) of the tile holding ``blk``, and its block's best on a candidate
    producer; see the module."""
    lane = c["lane"]
    a = c["args"]
    r8 = lane % 8
    col = blk * TILE + r8
    terms = {}
    for h in range_constexpr(2):
        a64, wts = query[h]
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for x in range_constexpr(4):
            acc = mfma_fp8(a64[x], b64[x], acc)
        for e in range_constexpr(4):
            terms[h, e] = (fx.max(acc[e] * ks, fx.Float32(0.0)), wts[e])
    live = col < bound
    score = fx.Float32(live.select(head_sum(terms), fx.Float32(NEG)))
    if lane < 16:
        st_dev(score, a["ilog"], t * a["lstride"] + col)
    if a["iprod"] != 0:
        # a block's best visible score: its 8 rows are 8 lanes
        best = fx.Float32(fx.isnan(score).select(fx.Float32(NEG), score))
        best = fx.Float32(live.select(best, fx.Float32(NEG)))
        best = butterfly(best, (1, 2, 4), fx.max)
        best = fx.min(best, fx.Float32(FLT_MAX))
        newest = blk == blocks - 1
        best = fx.Float32(newest.select(fx.Float32(float("inf")), best))
        if (lane < 16) & (r8 == 0) & (blk < blocks):
            st_dev(best, a["ibmax"], t * a["bstride"] + blk)


@traced
def stage_iscore(c, lead, span, ch0, width, max_chunks):
    """Chunks [ch0, ch0 + width) of the ``span`` tokens from ``lead``, which
    read lead's tile table (``index_plan.score_tasks``): the keys loaded once,
    every (token, chunk) scored, then one drain and the chunks' flags."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    a = c["args"]
    # a thread a (token, chunk) flag; its bound loaded with the keys
    row = lead + tid // TASK_CHUNKS
    m = tid % TASK_CHUNKS
    row_bound = ld_i32(a["ibound"], fx.min(row, c["S"] - 1))
    keys = _chunk_keys(a, lead, ch0, width, wave, lane)
    for r in range(span):
        tok = lead + r
        tok_bound = ld_i32(a["ibound"], tok)
        query = _token_query(a, tok, lane)
        tok_blocks = ip.blocks(tok_bound)
        for k in range_constexpr(TASK_CHUNKS):
            if ip.task_writes(tok_bound, ch0, width, k):
                _score_chunk(c, tok, tok_bound, tok_blocks, query, *keys[k])
    who = (tid < span * TASK_CHUNKS) & ip.task_writes(row_bound, ch0, width, m)
    publish(c["put"], c["isf"], row * max_chunks + ch0 + m, 1, who)


@traced
def _wait_chunks(c, token, max_chunks, first, count, groups):
    """A block-selection segment can depend on more than one flag per thread."""
    if const_expr(groups == 1):
        if c["tid"] < count:
            c["poll"]([(c["isf"], token * max_chunks + first + c["tid"], 1)])
    else:
        for u in range(c["tid"], count, THREADS):
            c["poll"]([(c["isf"], token * max_chunks + first + u, 1)])
    gpu.barrier()


@traced
def stage_ilist(c, t, j, raw, kind, level, bound_max, max_chunks):
    """List j of ``level`` for token t (``raw`` its column bound); see the
    module. Only the lists ``index_plan.list_count`` gives a token run."""
    tid = c["tid"]
    segment = c.get("select_seg", ip.SEG)
    if const_expr(kind == "blk"):
        segment = c.get("block_select_seg", segment)
    per_thread = segment // THREADS
    a = c["args"]
    k = ip.KINDS[kind]
    bound = ip.kind_bound(kind, raw)
    fast_hist = c.get("fast_select", False)
    if fast_hist:
        segment = isp.segment_for(segment, kind)
        per_thread = segment // THREADS
    storage_segment = c.get("list_storage_segment", segment)
    if const_expr(kind == "blk"):
        storage_segment = c.get("block_list_storage_segment", segment)
    slots = ip.list_slots(kind, level, bound_max, seg=storage_segment)
    if fast_hist and level > 0:
        max_inputs = ip.list_inputs(
            ip.kind_bound(kind, bound_max), level, 0, k, seg=segment
        )
        if c.get("tight_select_width", False):
            per_thread = ip.cdiv(max_inputs, THREADS * 4) * 4
        elif max_inputs <= 2048:
            per_thread = 4
        elif max_inputs <= 8192:
            per_thread = 16
    compact_rows = c.get("compact_selected_rows", False)
    list_ready = c.get("list_ready", False)
    distributed = c.get("world", 1) > 1
    mode, key_first = fx.Int32(tp.LOCAL), fx.Int32(0)
    if const_expr(distributed):
        mode, _owner, key_first, raw = row_work(c, t)
    bound = ip.kind_bound(kind, raw)
    value_first = key_first if kind == "sel" else key_first // ip.BLOCK_ROWS
    if j < ip.list_count(bound, level, k, seg=segment):
        n = ip.list_inputs(bound, level, j, k, seg=segment)
        final = ip.is_final(bound, level, k, seg=segment)
        # the top level has one list, a token's last: it has no region above it
        to_final = (
            True if level == ip.top_level(kind, bound_max, seg=segment) else final
        )
        got = []
        if const_expr(level == 0):
            if const_expr(kind == "sel"):
                first, count = (segment // ip.CHUNK) * j, ip.seg_chunks(
                    bound, j, seg=segment
                )
            else:
                first, count = ip.block_chunks(raw, j, seg=segment)
            if const_expr(distributed):
                # A split tail <=512 still has scored keys when its global
                # row needs Top512. Poll those global chunk flags as usual.
                if const_expr(kind == "sel"):
                    count = ip.cdiv(ip.seg_len(raw, j, seg=segment), ip.CHUNK)
                else:
                    count = ip.cdiv(
                        ip.imax(
                            ip.imin(raw, (j + 1) * segment * ip.BLOCK_ROWS)
                            - j * segment * ip.BLOCK_ROWS,
                            0,
                        ),
                        ip.CHUNK,
                    )
                count = (ld_i32(a["ibound"], t) > ip.TOPK).select(count, fx.Int32(0))
                first = first + key_first // ip.CHUNK
            flag_groups = -(
                -(segment * (ip.BLOCK_ROWS if kind == "blk" else 1))
                // (ip.CHUNK * THREADS)
            )
            _wait_chunks(c, t, max_chunks, first, count, flag_groups)
        else:
            below = ip.list_slots(kind, level - 1, bound_max, seg=storage_segment)
            base = tid * per_thread
            if const_expr(list_ready):
                if tid < ip.cdiv(n, k):
                    c["poll"](
                        [
                            (
                                c[f"i{kind}ready{level-1}"],
                                t * below + j * ip.fan(k, seg=segment) + tid,
                                1,
                            )
                        ]
                    )
                gpu.barrier()
            else:
                got = c["poll"](
                    [
                        (
                            c[f"i{kind}{level - 1}"],
                            (
                                (
                                    t * below
                                    + j * ip.fan(k, seg=segment)
                                    + fx.min(base + d, n - 1) // k
                                )
                                * k
                                + fx.min(base + d, n - 1) % k
                            )
                            * 2,
                            2,
                        )
                        for d in range(per_thread)
                    ]
                )
        src, stride = (
            (a["ilog"], a["lstride"]) if kind == "sel" else (a["ibmax"], a["bstride"])
        )

        def load(q, i):
            if const_expr(level == 0):
                # a quad stays in the row: strides are whole chunks / blocks
                v = fx.Vector(
                    ld_dev(src, t * stride + value_first + j * segment + i, 4)
                )
                return [fx.Int32(v[e]) for e in range(4)]
            if const_expr(list_ready):
                return [
                    fx.Int32(
                        ld_dev(
                            c[f"i{kind}{level-1}"].value,
                            ((t * below + j * ip.fan(k, seg=segment)) * k + i + e) * 4,
                        )
                    )
                    for e in range(4)
                ]
            return [got[4 * q + e][0] for e in range(4)]

        keys = c["sel_lds"]["keys"]
        first_i = tid * per_thread

        def ident_of(d, i):
            if const_expr(level == 0):
                return value_first + j * segment + i
            if const_expr(list_ready):
                return fx.Int32(
                    ld_dev(
                        c[f"i{kind}{level-1}"].value,
                        ((t * below + j * ip.fan(k, seg=segment)) * k + i) * 4 + 2,
                    )
                )
            return got[d][1]

        # a selection's rows (``_selected_rows``), in flight during the passes
        def selected_rows():
            return [
                _selected_rows(c, t, ident_of(d, fx.min(first_i + d, n - 1)))
                for d in range(per_thread)
            ]

        def emit(pos, d, i, rows, value_bits=None):
            ident = fx.Int32(-1)
            if const_expr(d is None):
                ident, bits = fx.Int32(-1), fx.Float32(NEG).bitcast(fx.Int32)
            else:
                bits = (
                    value_bits
                    if value_bits is not None
                    else key_bits(
                        fx.ptr_load(keys + (key_offset(i) if fast_hist else i))
                    )
                )
            if to_final:
                if const_expr(compact_rows):
                    if const_expr(distributed):
                        if mode != tp.LOCAL:
                            fx.ptr_store(bits, c["sel_lds"]["hist"] + pos)
                            fx.ptr_store(i, c["sel_lds"]["hist"] + k + pos)
                        else:
                            fx.ptr_store(i, c["sel_lds"]["hist"] + pos)
                    else:
                        fx.ptr_store(i, c["sel_lds"]["hist"] + pos)
                else:
                    if const_expr(d is None):
                        picked = (fx.Int32(-1), fx.Int32(0))
                    else:
                        ident = ident_of(d, i)
                        picked = _picked_rows(c, kind, t, ident, rows, d)
                    _final(c, kind, t, pos, ident, picked)
            else:
                if const_expr(fast_hist and list_ready):
                    fx.ptr_store(bits, c["sel_lds"]["hist"] + pos)
                    fx.ptr_store(i, c["sel_lds"]["hist"] + k + pos)
                else:
                    if const_expr(d is not None):
                        ident = ident_of(d, i)
                    c["put_words"](
                        c[f"i{kind}{level}"],
                        ((t * slots + j) * k + pos) * 2,
                        [bits, ident],
                    )

        if const_expr(fast_hist and per_thread >= 16):
            medium = 8 if per_thread == 16 else 16
            if n <= 2048:
                lds_topk(
                    n,
                    k,
                    load,
                    emit,
                    c["sel_lds"],
                    tid,
                    4,
                    fast_hist=True,
                    emit_key=True,
                )
            elif n <= medium * THREADS:
                lds_topk(
                    n,
                    k,
                    load,
                    emit,
                    c["sel_lds"],
                    tid,
                    medium,
                    fast_hist=True,
                    emit_key=True,
                )
            else:
                lds_topk(
                    n,
                    k,
                    load,
                    emit,
                    c["sel_lds"],
                    tid,
                    per_thread,
                    fast_hist=True,
                    emit_key=True,
                )
        else:
            lds_topk(
                n, k, load, emit, c["sel_lds"], tid, per_thread,
                _prefetch_rows(c, kind, selected_rows),
                fast_hist=fast_hist,
                emit_key=fast_hist,
            )  # fmt: skip
        if const_expr(compact_rows):  # noqa: SIM102
            if to_final:
                if const_expr(distributed):
                    if mode != tp.LOCAL:
                        for r in range_constexpr(-(-k // THREADS)):
                            pos = tid + r * THREADS
                            if pos < k:
                                bits = fx.ptr_load(c["sel_lds"]["hist"] + pos)
                                i = fx.ptr_load(c["sel_lds"]["hist"] + k + pos)
                                ident = (i >= 0).select(
                                    ident_of(None, fx.max(i, 0)), fx.Int32(-1)
                                )
                                st_dev(
                                    fx.Vector.from_elements([bits, ident], fx.Int32),
                                    c["ix_pairs"][kind],
                                    (t * k + pos) * 2,
                                )
                    else:
                        _compact_final(c, kind, t, k, ident_of)
                else:
                    _compact_final(c, kind, t, k, ident_of)
        has_list_output = list_ready and level < ip.top_level(
            kind, bound_max, seg=segment
        )
        if const_expr(has_list_output):  # noqa: SIM102
            if ~final:
                if const_expr(fast_hist):
                    for r in range_constexpr(-(-k // THREADS)):
                        pos = tid + r * THREADS
                        if pos < k:
                            bits = fx.ptr_load(c["sel_lds"]["hist"] + pos)
                            i = fx.ptr_load(c["sel_lds"]["hist"] + k + pos)
                            ident = (i >= 0).select(
                                ident_of(None, fx.max(i, 0)), fx.Int32(-1)
                            )
                            c["put_words"](
                                c[f"i{kind}{level}"],
                                ((t * slots + j) * k + pos) * 2,
                                [bits, ident],
                            )
                publish(c["put"], c[f"i{kind}ready{level}"], t * slots + j, 1, tid == 0)
        # a selection's flag for the attention; the candidate blocks' only in a
        # token-parallel launch, which waits for every rank's (``await_candidates``)
        region = "irdy" if kind == "sel" else "icrdy" if c["tok_par"] else None
        if const_expr(region is not None):
            if final:
                if const_expr(distributed):
                    if mode != tp.LOCAL:
                        publish(
                            c["put"],
                            c["ix_root_ready"],
                            tp.KINDS.index(kind) * c["S"] + t,
                            fx.min(bound, k),
                            tid == 0,
                        )
                    else:
                        _publish_all(c, region, t)
                else:
                    _publish_all(c, region, t)
        elif const_expr(distributed):  # noqa: SIM102
            if final & (mode != tp.LOCAL):
                publish(
                    c["put"],
                    c["ix_root_ready"],
                    tp.KINDS.index(kind) * c["S"] + t,
                    fx.min(bound, k),
                    tid == 0,
                )


@traced
def _compact_final(c, kind, t, k, ident_of):
    for r in range_constexpr(-(-k // THREADS)):
        pos = c["tid"] + r * THREADS
        if pos < k:
            i = fx.ptr_load(c["sel_lds"]["hist"] + pos)
            ident = (i >= 0).select(ident_of(None, fx.max(i, 0)), fx.Int32(-1))
            if const_expr(kind == "sel"):
                _final(c, kind, t, pos, ident, _selected_rows(c, t, ident))
            else:
                _final(c, kind, t, pos, ident)


def _prefetch_rows(c, kind, fn):
    return fn if kind == "sel" and not c.get("defer_selected_rows", False) else None


def _picked_rows(c, kind, t, ident, rows, d):
    if kind != "sel":
        return None
    if c.get("defer_selected_rows", False):
        return _selected_rows(c, t, ident)
    return rows[d]


def _selected_rows(c, t, ident):
    """Column ``ident`` of token t as a selection holds it (a REINDEX layer's
    compacted column lifted back to the row it stands for) and as the
    attention reads it: its pool row, through the block table (``attention.
    _key_row``'s arithmetic; every layer sharing this selection shares its KV
    owner, so its row). Issue the loads for every element from ``after_keys``
    before the top-k passes, allowing pool-table reads to overlap the passes.
    Keep candidate loads branch-free and inside the token's candidate row;
    FULL and padding discard their values. Branching around each candidate
    load can serialize these prefetches through the in-order load counter."""
    a = c["args"]
    cand = fx.Int32(
        bo.buffer_load(
            rsrc(a["icand"]),
            ip.candidate_offset(t, ident),
            vec_width=1,
            dtype=T.i32,
        )
    )
    lifted = cand * TILE + fx.max(ident, 0) % TILE
    row = fx.Int32(((a["ilift"] != 0) & (ident >= 0)).select(lifted, ident))
    toff = fx.Int32(
        bo.buffer_load(
            rsrc(a["kmeta"]), t * ip.KEY_META_WORDS + 2, vec_width=1, dtype=T.i32
        )
    )
    rpp = a["rows_per_page"]
    live = fx.max(row, 0)
    page = fx.Int32(
        bo.buffer_load(rsrc(a["table"]), toff + live // rpp, vec_width=1, dtype=T.i32)
    )
    pool_row = page * a["page_rows"] + a["main_off"] + live % rpp
    return row, (ident >= 0).select(pool_row, fx.Int32(0))


def _final(c, kind, t, pos, ident, rows=None):
    if const_expr(kind == "sel"):
        row, pool_row = rows
        # the attention reads the pool rows; the selection only check mode
        # compares, which never runs token parallel: this rank's alone
        st_dev(row, c["args"]["isel"], t * ip.TOPK + pos)
        _st_out(c, pool_row, "irow", t * ip.TOPK + pos)
    else:
        _st_out(c, Int32(ident), "icout", t * ip.TOPK_BLOCKS + pos)


def _st_out(c, v, name, i):
    """Word i of output ``name``: this rank's, or in a token-parallel launch,
    which selects only its own rows (``_own_rows``), every rank's copy. The
    outputs live in the peer buffer's plain part (``peer_layout.plain_layout``),
    at one offset from every rank's base."""
    a = c["args"]
    if const_expr(c["tok_par"]):
        off = a[name] - c["sym"]
        for p in range_constexpr(c["d"].tp):
            bo.buffer_store(v, rsrc(c["peer_addr"](p) + off), i, cache_modifier=CM_SYS)
    else:
        st_dev(v, a[name], i)


def _publish_all(c, region, t):
    """Token t's flag in ``region`` after this CTA's outputs: this rank's
    scratch, or in a token-parallel launch every rank's peer region."""
    if const_expr(c["tok_par"]):
        flags = [
            preg(c["peer_addr"](p), c["peer"][region], region) for p in range(c["d"].tp)
        ]

        def put_each(dsts, i, v):
            for dst in dsts:
                c["put"](dst, i, v)

        publish(put_each, flags, t, 1, c["tid"] == 0)
    else:
        publish(c["put"], c[region], t, 1, c["tid"] == 0)


@traced
def await_candidates(c, s, bid):
    """A token-parallel launch of the candidate-producing layer ends only once
    every rank's rows' candidate blocks are in this rank's copy: the host's
    candidate table reads them after it (one CTA waits, at the launch's end)."""
    enter_stage("k2a.icwait")
    if (c["args"]["iact"] != 0) & (c["args"]["iprod"] != 0) & (bid == 0):
        flags = preg(c["sym"], c["peer"]["icrdy"], "icrdy")
        for t0 in range_constexpr(0, s, POLL_MAX):
            c["poll"]([(flags, t, 1) for t in range(t0, min(t0 + POLL_MAX, s))])


def _own_rows(c, s):
    """(first row, rows) this rank scores and selects: every row, or in a
    token-parallel launch its own (``token_par.own_rows``) -- a row's
    selection depends on that row alone, and ``_st_out`` hands it to every rank."""
    if const_expr(c["tok_par"]):
        return own_rows(s, c["d"].tp, c["rank"])
    return 0, s


@traced
def run_indexer(c, s, bid, bound_max):
    """Every indexer stage of a selecting layer, the tasks of each laid out from
    the step's bounds (``index_plan``): only live tasks. The bounds were written
    before this launch: plain loads."""
    a = c["args"]
    max_chunks = ip.chunks(bound_max)
    t0, n = _own_rows(c, s)
    bounds = [ld_i32(a["ibound"], t0 + t) for t in range(n)]
    if const_expr(c["index_fp4"]):
        run_scores4(c, t0, n, bid, max_chunks)
    else:
        run_scores(c, t0, n, bid, bounds, max_chunks)
    _run_tree(c, t0, n, bid, "sel", bounds, bound_max, max_chunks)
    if a["iprod"] != 0:
        _run_tree(c, t0, n, bid, "blk", bounds, bound_max, max_chunks)


@traced
def run_scores(c, t0, n, bid, bounds, max_chunks):
    """The score stage's tasks (``index_plan.score_tasks``) over rows t0 ..
    t0 + n. A request's tokens share its tile table (key metadata word 2: its
    block table offset); a REINDEX layer's are each token's own candidates."""
    assert n * TASK_CHUNKS <= THREADS, n  # a thread a (token, chunk) flag
    a = c["args"]
    reindex = a["ilift"] != 0
    owners = [
        sel(reindex, t0 + t, ld_i32(a["kmeta"], (t0 + t) * ip.KEY_META_WORDS + 2))
        for t in range(n)
    ]
    counts, spans, width_log2 = ip.score_tasks(bounds, owners, BLOCKS, TASK_CHUNKS_LOG2)
    width = Int32(1) << width_log2
    enter_stage("k2a.iscore")
    for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
        lead, i = ip.flat_task(counts, task)
        stage_iscore(
            c, t0 + lead, pick(spans, lead), i << width_log2, width, max_chunks
        )


@traced
def _run_tree(c, t0, n, bid, kind, bounds, bound_max, max_chunks):
    k = ip.KINDS[kind]
    for level in range_constexpr(ip.top_level(kind, bound_max) + 1):
        counts = [ip.list_count(ip.kind_bound(kind, b), level, k) for b in bounds]
        enter_stage(f"k2a.i{kind}{level}")
        for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
            t, j = ip.flat_task(counts, task)
            stage_ilist(
                c, t0 + t, j, pick(bounds, t), kind, level, bound_max, max_chunks
            )
