# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
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
    ilist   (a level's lists, per token): ``topk.radix_select`` of a segment's
            values (level 0, after its chunks' flags) or of the lists it merges
            (level l > 0, from ISEL<l-1> / IBLK<l-1>), in index order. A list
            that is its token's last -> global (``isel``, lifted to real rows on
            a REINDEX layer; ``icout``) and, for the selection, the ready flag
            (IRDY) the attention waits on; else -> ISEL<l> / IBLK<l>. A
            candidate producer ranks its blocks first: a row with TOPK_BLOCKS
            of them or more then has its candidates' scores compacted to
            the front of its ``ilog`` row (ICRDY), and its selection ranks
            those alone, lifted as a REINDEX row's (``index_plan.compacts``)
    imerge  (a row split over the TP ranks, ``index_plan.row_shares``): each
            rank's tree ranks its share; its last list goes to every rank's
            IXSEL (plain words, then the IXRDY flag), and every rank ranks a
            split row's tp lists into its selection.

Data too large for tagged pairs (the logits, up to max_model_len a token) goes
through global memory at device scope (SC1: past the per-XCD L2): stored,
``s_waitcnt vmcnt(0)``, then the flag: a task's drain and barrier, hence
several chunks a task.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, T

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.index_query import await_queries
from atom.models.deepseek_v41.mono.kernels.index_score_fp4 import run_scores4, sload4
from atom.models.deepseek_v41.mono.kernels.mailbox import preg
from atom.models.deepseek_v41.mono.kernels.token_par import own_rows
from atom.models.deepseek_v41.mono.kernels.topk import (
    radix_select,
    radix_slot,
    select_words,
)
from atom.models.deepseek_v41.mono.step_meta import KMETA
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
    uniform,
)
from atom.mono.device.select import key_bits, order_key
from atom.mono.device.sync import POLL_MAX, publish
from atom.mono.plan.arith import pick, sel
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES, first_task
from atom.mono.plan.trace import enter_stage

TILE = ip.BLOCK_ROWS
# a score task's chunks at most: 1 << this, loads in flight together
TASK_CHUNKS_LOG2 = 3
TASK_CHUNKS = 1 << TASK_CHUNKS_LOG2
TILE_BYTES = TILE * ip.DIM + TILE * 4  # 8 preshuffled rows, then 8 fp32 scales
PER_THREAD = ip.SEG // THREADS
# the selects' LDS (``topk.radix_select``, 16-bit places: the rows' and the
# candidate blocks') and the winners' bits by place
SEL_RADIX_WORDS = select_words(max(ip.TOPK, ip.TOPK_BLOCKS))
SEL_BITS_WORDS = ip.TOPK
NEG = float("-inf")
FLT_MAX = 3.4028234663852886e38


def scratch_pairs(tokens: int, bound_max: int, tok_par: bool = False) -> dict[str, int]:
    """The indexer's mailbox regions. A tree's last level's list is always a
    token's final one (global), so only levels below it have a region, sized
    for every tree of its kind (``index_plan.trees``). A token-parallel launch's
    selection and candidate flags are peer regions (``peer_layout``: its
    owner's)."""
    out = {"isf": tokens * ip.chunks(bound_max)}
    if not tok_par:
        out["irdy"] = tokens
        out["icrdy"] = tokens
    for kind, k in ip.KINDS.items():
        for level in range(ip.region_levels(kind, bound_max)):
            slots = ip.region_slots(kind, level, bound_max)
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
def stage_iscore(c, lead, span, span_max, ch0, width, max_chunks):
    """Chunks [ch0, ch0 + width) of the ``span`` tokens from ``lead`` (traced,
    at most the static ``span_max``), which read lead's tile table
    (``index_plan.score_tasks``): the keys loaded once, every (token, chunk)
    scored, then one drain and the chunks' flags."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    a = c["args"]
    # a thread a (token, chunk) flag; its bound loaded with the keys
    row = lead + tid // TASK_CHUNKS
    m = tid % TASK_CHUNKS
    row_bound = ld_i32(a["ibound"], fx.min(row, c["S"] - 1))
    keys = _chunk_keys(a, lead, ch0, width, wave, lane)
    # the layer kernel: the rows' queries from this launch's K1 (the key
    # loads above go out first)
    await_queries(c, lead, span, span_max)
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
def stage_ilist(
    c, t, j, share, kind, level, bound_max, max_chunks, seg, gathers_ok=True,
    compact=None,
):  # fmt: skip
    """List j of ``level`` for token t (``share``: its (column bound, columns
    on this rank, split), ``_shares``; a split build's FULL rows carry no
    bound) of a tree of ``seg`` segments; see the module. Only the lists
    ``index_plan.list_count`` gives a token run. ``compact`` (the blk tree,
    traced): whether its rows are compacted (``run_indexer``)."""
    tid = c["tid"]
    a = c["args"]
    k = ip.KINDS[kind]
    raw, cols, split = share[:3]
    # a share's first column: carried (``_sel_rows``), else the plan's
    col0 = share[3] if len(share) > 3 else (0 if split is None else _row_share(c, t)[0])
    # a split row's columns are always scored (its bound is past SPLIT_FROM)
    any_scored = ip.scored(raw) if split is None else (split != 0) | ip.scored(cols)
    slots = ip.list_slots(kind, level, bound_max, seg)
    bound = ip.kind_bound(kind, cols)
    # a candidate producer's row of TOPK_BLOCKS blocks or more: its selection
    # ranks its candidate blocks' rows, compacted to the row's front (``_compact``)
    gathers = False if raw is None else (a["iprod"] != 0) & ip.compacts(raw)
    if const_expr(not gathers_ok):
        gathers = False
    if const_expr(compact is not None and gathers is not False):
        gathers = gathers & compact
    if j < ip.list_count(bound, level, k, seg):
        n = ip.list_inputs(bound, level, j, k, seg)
        final = ip.is_final(bound, level, k, seg)
        # the top level has one list, a token's last: it has no region above it
        to_final = True if level == ip.top_level(kind, bound_max, seg) else final
        per = ip.per_thread(level, ip.kind_bound(kind, bound_max), k, THREADS, seg)
        got = []
        if const_expr(level == 0):
            if const_expr(kind == "sel"):
                first = col0 // ip.CHUNK + seg // ip.CHUNK * j
                count = ip.share_chunks(any_scored, cols, j, seg)
                if const_expr(raw is not None):
                    # a compacted row's columns: its candidates' flag instead
                    count = sel(gathers, 0, count)
                    if const_expr(gathers is not False):  # noqa: SIM102
                        if gathers & (tid == 0):
                            c["poll"]([(_icrdy(c), t, 1)])
            else:
                first, count = ip.block_chunks(raw, j)
            if tid < count:
                c["poll"]([(c["isf"], t * max_chunks + first + tid, 1)])
            gpu.barrier()
        else:
            below = ip.list_slots(kind, level - 1, bound_max, seg)
            base = tid * per
            got = c["poll"](
                [
                    (
                        c[f"i{kind}{level - 1}"],
                        (
                            (
                                t * below
                                + j * ip.fan(k, seg)
                                + fx.min(base + d, n - 1) // k
                            )
                            * k
                            + fx.min(base + d, n - 1) % k
                        )
                        * 2,
                        2,
                    )
                    for d in range(per)
                ]
            )
        src, stride = (
            (a["ilog"], a["lstride"]) if kind == "sel" else (a["ibmax"], a["bstride"])
        )

        def load(q, i):
            if const_expr(level == 0):
                # a quad stays in the row: strides are whole chunks / blocks
                v = fx.Vector(ld_dev(src, t * stride + col0 + j * seg + i, 4))
                return [fx.Int32(v[e]) for e in range(4)]
            return [got[4 * q + e][0] for e in range(4)]

        # a final selection's winners by place (``hist``: free once the passes
        # are done), lifted to their rows one a thread after the routine
        slot = c["sel_lds"]["hist"]

        # this list's k-th of the step before (``topk.radix_select``)
        anchor = _anchor(c, t, kind, seg, level, j)
        if const_expr(kind == "sel"):
            # a list above: written by place here (none at the top level)
            listed = (
                None
                if to_final is True
                else (c[f"i{kind}{level}"], ip.lists(bound, level, k, seg) > 1)
            )
            _select_sel(
                c, t, j, n, per, load, got, level, seg, slots, to_final, listed,
                col0, split, anchor,
            )  # fmt: skip
            if to_final:
                _final_rows(c, t, slot, split, gathers)
        else:
            idents = _select_blk(
                c, t, j, n, per, load, got, level, seg, slots, to_final, kind, anchor
            )
            if final & gathers:
                _compact(c, t, idents)
        # a selection's flag for the attention (a split row's answer's for
        # every rank's merge instead); the candidate blocks' for this row's
        # compacted selection and, in a token-parallel launch, every rank's
        # ``await_candidates``
        if final:
            _publish_final(c, "irdy" if kind == "sel" else "icrdy", t, split)


@traced
def _select_sel(
    c, t, j, n, per, load, got, level, seg, slots, to_final, listed, col0=0,
    split=None, anchor=None,
):  # fmt: skip
    """The top-512 of a selection list's ``n`` values (``topk.radix_select``:
    keys in registers, a pass whose digit is taken whole ends the search, the
    winners into slots in place order), then a place a thread: a final list's
    columns into the slots (``_final_rows``) -- a split row's (bits, column)
    pairs to every rank's merge instead (``_put_answer``) --, another's into
    its region ``listed`` (region, put?). A merge's columns go by place
    through ``keys`` (its inputs' own); level 0's are its segment's past the
    share's first column ``col0``. ``anchor``: the list's (``_anchor``)."""
    tid = c["tid"]
    k = ip.TOPK
    keys, slot, rlds = c["sel_lds"]["keys"], c["sel_lds"]["hist"], c["sel_lds"]["hi"]
    base = tid * per
    last = fx.max(n - 1, 0) & ~3
    us = []
    for q in range_constexpr(per // 4):
        quad = load(q, fx.min(base + 4 * q, last))
        us += [order_key(quad[e]) for e in range(4)]
    if const_expr(level > 0):
        for d in range_constexpr(per):
            fx.ptr_store(got[d][1], keys + (base + d))
    n_out = radix_select(rlds, tid, us, base, n, k, anchor)
    for p in range_constexpr((k + THREADS - 1) // THREADS):
        place = tid + THREADS * p
        if place < k:
            u, idx = radix_slot(rlds, place, n_out, k)
            won = idx >= 0
            if const_expr(level == 0):
                col = col0 + j * seg + idx
            else:
                col = fx.Int32(fx.ptr_load(keys + fx.max(idx, 0)))
            ident = won.select(col, fx.Int32(-1))
            bits = won.select(key_bits(u), fx.Float32(NEG).bitcast(fx.Int32))
            if to_final:
                fx.ptr_store(ident, slot + place)
                if const_expr(split is not None and c["staged_answer"]):
                    # a split row's answer carries its values (``_put_answer``)
                    fx.ptr_store(bits, c["sel_lds"]["bits"] + place)
                elif const_expr(split is not None):  # noqa: SIM102
                    if split != 0:
                        _put_place(c, t, place, bits, ident)
            if const_expr(listed is not None):
                region, more = listed
                if more:
                    c["put_words"](
                        region, ((t * slots + j) * k + place) * 2, [bits, ident]
                    )
    gpu.barrier()


@traced
def _select_blk(
    c, t, j, n, per, load, got, level, seg, slots, to_final, kind, anchor=None
):
    """The candidate blocks' list: the top-2048 of its ``n`` block maxima
    (``topk.radix_select``, as ``_select_sel``), a place a thread out to
    ``icout`` (a final list) or the level's region. A merge's blocks go by
    place through ``keys`` (its inputs' own); level 0's are its segment's.
    -> this thread's places' blocks (-1 past the list), in place order."""
    tid = c["tid"]
    idents = []
    k = ip.TOPK_BLOCKS
    keys, rlds = c["sel_lds"]["keys"], c["sel_lds"]["hi"]
    base = tid * per
    last = fx.max(n - 1, 0) & ~3
    us = []
    for q in range_constexpr(per // 4):
        quad = load(q, fx.min(base + 4 * q, last))
        us += [order_key(quad[e]) for e in range(4)]
    if const_expr(level > 0):
        for d in range_constexpr(per):
            fx.ptr_store(got[d][1], keys + (base + d))
    n_out = radix_select(rlds, tid, us, base, n, k, anchor)
    for p in range_constexpr(k // THREADS):
        place = tid + THREADS * p
        u, idx = radix_slot(rlds, place, n_out, k)
        won = idx >= 0
        if const_expr(level == 0):
            ident = won.select(j * seg + idx, fx.Int32(-1))
        else:
            ident = won.select(
                fx.Int32(fx.ptr_load(keys + fx.max(idx, 0))), fx.Int32(-1)
            )
        idents.append(ident)
        if to_final:
            _st_out(c, ident, "icout", t * k + place)
        else:
            bits = won.select(key_bits(u), fx.Float32(NEG).bitcast(fx.Int32))
            c["put_words"](
                c[f"i{kind}{level}"], ((t * slots + j) * k + place) * 2, [bits, ident]
            )
    gpu.barrier()
    return idents


@traced
def _compact(c, t, idents):
    """Token t's candidate blocks' scores moved to the front of its ``ilog``
    row, place p's eight at columns 8p.. (``index_plan.candidate_bound``):
    what its selection then ranks, lifted back through ``icout``
    (``_selected_rows``). The blocks ascend, so a place's columns never pass
    its block's: every thread's reads land before any write. A compacted row
    has at least as many blocks as places, so every place holds one."""
    a = c["args"]
    base = t * a["lstride"]
    tid = c["tid"]
    vals = [
        [fx.Vector(ld_dev(a["ilog"], base + b * TILE + 4 * h, 4)) for h in range(2)]
        for b in idents
    ]
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    for p, pair in enumerate(vals):
        place = tid + THREADS * p
        for h in range_constexpr(2):
            st_dev(pair[h], a["ilog"], base + place * TILE + 4 * h)


def _anchor(c, t, kind, seg, level, j=0):
    """List j of ``level`` of token t's ``kind`` / ``seg`` tree's anchor
    (kind "merge": its split row's merge), (pointer, word): the anchors in
    front of the layer's ``isel`` (``index_plan.anchor_base``, laid out for the
    build's bound, ``run_indexer``); None with ``index_plan.ANCHOR`` off."""
    if const_expr(not ip.ANCHOR):
        return None
    bound_max = c["anchor_bound"]
    row = ip.anchor_row_words(bound_max)
    ptr = c["args"]["isel"] - fx.Int64(ip.anchor_bytes(bound_max))
    return ptr, t * row + ip.anchor_base(bound_max, kind, seg, level) + j


def _icrdy(c):
    """This rank's ICRDY: token t's candidate blocks out (``icout``) and, with
    ``TOPK_BLOCKS`` blocks or more, its row compacted (``_compact``); in a
    token-parallel launch the peer region every row's owner puts on every
    rank."""
    if const_expr(c["tok_par"]):
        return preg(c["sym"], c["peer"]["icrdy"], "icrdy")
    return c["icrdy"]


@traced
def _final_rows(c, t, slot, split=None, lift=False):
    """Token t's final selection out of the winners' slots (``stage_ilist``'s
    emit, by place): thread p lifts place p's column (``_selected_rows``) and
    stores it whole -- the 512 winners' table reads alone, not every ranked
    key's before the passes, and contiguous stores. A padding place (-1) keeps
    row -1 and pool row 0. The barrier ends the slots' reads before the CTA's
    next list reuses them. A split row's goes to every rank's merge instead:
    as selected (``_put_place``), or a staged build's from the slots here
    (``_put_answer``). ``lift``: a compacted row's columns, lifted as a
    REINDEX layer's (a split row is never compacted)."""
    if const_expr(split is None):
        _final_rows_of(c, t, slot, lift)
    else:
        if split == 0:
            _final_rows_of(c, t, slot)
        elif const_expr(c["staged_answer"]):
            _put_answer(c, t, slot)


@traced
def _final_rows_of(c, t, slot, lift=False):
    tid = c["tid"]
    for p in range_constexpr((ip.TOPK + THREADS - 1) // THREADS):
        place = tid + THREADS * p
        if place < ip.TOPK:
            ident = fx.ptr_load(slot + place)
            row, pool_row = _selected_rows(c, t, ident, lift)
            # the attention reads the pool rows; the selection only check mode
            # compares, which never runs token parallel: this rank's alone
            st_dev(row, c["args"]["isel"], t * ip.TOPK + place)
            _st_out(c, (ident >= 0).select(pool_row, 0), "irow", t * ip.TOPK + place)
    gpu.barrier()


@traced
def _selected_rows(c, t, ident, lift=False):
    """Column ``ident`` of token t as a selection holds it (a REINDEX layer's
    compacted column, or a ``lift`` one -- a candidate producer's compacted
    row, ``_compact`` --, lifted back to the row it stands for) and as the
    attention reads it: its pool row, through the block table (``attention.
    _key_row``'s arithmetic; every layer sharing this selection shares its KV
    owner, so its row). Keep candidate loads inside the token's candidate
    row; FULL and padding discard their values. A ``lift`` row's candidates
    are this launch's: read at device scope (a REINDEX layer's are an
    earlier launch's, a plain load)."""
    a = c["args"]
    off = ip.candidate_offset(t, ident)
    cand = ld_i32(a["icand"], off)
    if const_expr(lift is not False):  # noqa: SIM102
        if lift:
            cand = fx.Int32(ld_dev(a["icand"], off))
    lifted = cand * TILE + fx.max(ident, 0) % TILE
    lifts = (a["ilift"] != 0) | lift
    row = fx.Int32((lifts & (ident >= 0)).select(lifted, ident))
    toff = fx.Int32(
        bo.buffer_load(rsrc(a["kmeta"]), t * KMETA + 2, vec_width=1, dtype=T.i32)
    )
    rpp = a["rows_per_page"]
    live = fx.max(row, 0)
    page = fx.Int32(
        bo.buffer_load(rsrc(a["table"]), toff + live // rpp, vec_width=1, dtype=T.i32)
    )
    return row, page * a["page_rows"] + a["main_off"] + live % rpp


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


@traced
def _publish_all(c, region, t):
    """Token t's flag in ``region`` after this CTA's outputs: this rank's
    scratch, or in a token-parallel launch every rank's peer region."""
    if const_expr(c["tok_par"]):
        _publish_peers(c, region, t)
    else:
        publish(c["put"], c[region], t, 1, c["tid"] == 0)


def _publish_peers(c, region, i):
    """Flag i of every rank's peer region ``region`` after this CTA's outputs."""
    flags = [
        preg(c["peer_addr"](p), c["peer"][region], region) for p in range(c["d"].tp)
    ]

    def put_each(dsts, i, v):
        for dst in dsts:
            c["put"](dst, i, v)

    publish(put_each, flags, i, 1, c["tid"] == 0)


@traced
def _publish_final(c, region, t, split):
    """Token t's final list published: its selection's flag, or a split row's
    answer's (IXRDY [token][source rank]) for every rank's merge."""
    if const_expr(split is None):
        _publish_all(c, region, t)
    else:
        if split != 0:
            _publish_peers(c, "ixrdy", t * c["d"].tp + c["rank"])
        else:
            _publish_all(c, region, t)


def _put_place(c, t, place, bits, ident):
    """Place ``place`` of a split row's answer on this rank, (value bits,
    column), into every rank's IXSEL [token][source rank][TOPK] (the peer
    buffer's plain part, at one offset from every rank's base) as the select
    hands it out; its flag after the CTA's drain (``_publish_final``)."""
    tp = c["d"].tp
    off = c["args"]["ixsel"] - c["sym"]
    pair = fx.Vector.from_elements([bits, ident], fx.Int32)
    w = ((t * tp + c["rank"]) * ip.TOPK + place) * 2
    for p in range_constexpr(tp):
        bo.buffer_store(pair, rsrc(c["peer_addr"](p) + off), w, cache_modifier=CM_SYS)


@traced
def _put_answer(c, t, slot):
    """A split row's answer on this rank out of the winners' slots (columns)
    and bits (values) by place, as (value bits, column) pairs into every
    rank's IXSEL [token][source rank][TOPK] (the peer buffer's plain part, at
    one offset from every rank's base; its flag after the CTA's drain,
    ``_publish_final``): a wave a rank, ``WAVES / tp`` waves
    splitting its places, a lane adjacent places as 16-byte stores (one buffer
    resource a wave). Wave w's rank is this one's + w: each rank's waves start
    on different peers. The barrier as ``_final_rows_of``'s."""
    tp = c["d"].tp
    groups = WAVES // tp
    per = ip.TOPK // (groups * 64)
    wave = uniform(c["wave"])
    dst = pick([c["peer_addr"](p) for p in range(tp)], (c["rank"] + wave) % tp)
    place = (wave // tp * 64 + c["lane"]) * per
    words = []
    for e in range_constexpr(per):
        words += [
            fx.Int32(fx.ptr_load(c["sel_lds"]["bits"] + place + e)),
            fx.Int32(fx.ptr_load(slot + place + e)),
        ]
    out = rsrc(dst + (c["args"]["ixsel"] - c["sym"]))
    w = ((t * tp + c["rank"]) * ip.TOPK + place) * 2
    for q in range_constexpr(per // 2):
        quad = fx.Vector.from_elements(words[4 * q : 4 * q + 4], fx.Int32)
        bo.buffer_store(quad, out, w + 4 * q, cache_modifier=CM_SYS)
    gpu.barrier()


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


# a candidate producer's selection tasks' first CTA when they take every column:
# past its blk tree's first-level lists (6 rows x 7 segments at 400K columns)
PROD_SEL0 = 48
# a candidate producer's widest row up to which its selection takes every
# column (32 segments: 6 rows' first level within the grid beside the blk tree;
# at 400K columns, 49 segments, it ran slower than the compacted columns)
PROD_FULL_MAX = 32 * ip.SEG
# a candidate producer's split rows' merges (``_run_merge``): past its blk
# tree's lists (CTAs 0 ..) and the attention score's (from ``score0``), before
# the late wq_b's (``attn_pre.WQB_LAYER0``): their CTAs end with the merge
PROD_MERGE0 = 66
# a split build's selection takes ``_sel_rows`` (a candidate producer's wide
# rows split over the TP ranks) up to this many rows; a wider build keeps the
# compacted candidates (its 24-row layer_post held more than its SGPRs: 36 B
# of scratch)
SEL_ROWS_MAX = 12


@traced
def run_indexer(c, s, bid, bound_max):
    """Every indexer stage of a selecting layer, the tasks of each laid out from
    the step's bounds (``index_plan``): only live tasks. The bounds were written
    before this launch: plain loads."""
    a = c["args"]
    max_chunks = ip.chunks(bound_max)
    c["anchor_bound"] = bound_max  # ``_anchor``'s layout
    t0, n = _own_rows(c, s)
    split_sel = c["index_split"] and n <= SEL_ROWS_MAX
    # the FP4 scorer reads its rows' bounds itself: the trees' are loaded past
    # it, none held through its walk
    if const_expr(c["index_fp4"]):
        run_scores4(c, t0, n, bid, max_chunks)
    else:
        run_scores(c, t0, n, bid, _bounds(c, t0, n), max_chunks)
    # a candidate producer's blocks first: its long rows' selection ranks
    # only their rows (``index_plan.compacts``), a CTA's tasks in this order
    if a["iprod"] != 0:
        # its long rows compacted (``_compact``) only where its selection then
        # ranks the compacted columns (``_wide``). Beside the full-column tree
        # a compaction would rewrite the columns that tree's level 0 is still
        # reading, and nothing would read it: a REINDEX layer scores its
        # candidates afresh (``index_plan.compacts``); ICRDY says its blocks
        # are out either way
        ((blk, bseg),) = ip.trees("blk", bound_max)
        # a split build's selection never ranks compacted columns (``_sel_rows``)
        compact = False if split_sel else _wide(c, t0, n)
        _run_tree(
            c, t0, n, bid, "blk", _whole(c, t0, n), blk, max_chunks, bseg,
            compact=compact,
        )  # fmt: skip
    # the selection's trees (``index_plan.trees``): the columns', a REINDEX
    # row's -- a candidate producer's too, over its candidates
    (full, seg), (reindex, rseg) = ip.trees("sel", bound_max)
    if a["ilift"] != 0:
        _run_tree(c, t0, n, bid, "sel", _whole(c, t0, n), reindex, max_chunks, rseg)
    elif const_expr(split_sel):
        # one tree for a FULL layer's rows and a candidate producer's
        # (``_sel_rows``): every row as a share, split or whole; a producer's
        # from PROD_SEL0 (past its blk tree's first level), beside that tree
        rows = _sel_rows(c, t0, n)
        prod = a["iprod"] != 0
        _run_tree(
            c, t0, n, bid, "sel", rows, full, max_chunks, seg,
            sel(prod, Int32(PROD_SEL0), Int32(0)), False,
        )  # fmt: skip
        _run_merge(c, t0, n, bid, rows, sel(prod, Int32(PROD_MERGE0), Int32(0)))
    elif a["iprod"] != 0:
        bounds = _bounds(c, t0, n)
        if _wide(c, t0, n):
            # past PROD_FULL_MAX columns the full tree's first level outgrows
            # the grid: its candidates' columns (``index_plan.candidate_bound``)
            # after the blk tree's compaction, apart from a REINDEX row's tree
            cols = [ip.candidate_bound(b) for b in bounds]
            _run_tree(
                c, t0, n, bid, "sel", (_whole_of, cols), reindex, max_chunks, rseg
            )
        else:
            # every column, a FULL row's tree, on CTAs from PROD_SEL0 (past the
            # blk tree's first level): the attention waits on this selection
            # alone, so it runs beside the blk tree and compaction, which only
            # the later REINDEX layers read (``index_plan.compacts``: the same
            # answer)
            _run_tree(
                c, t0, n, bid, "sel", _whole(c, t0, n), full, max_chunks, seg,
                PROD_SEL0, False,
            )  # fmt: skip
    else:
        _run_tree(c, t0, n, bid, "sel", _shares(c, t0, n), full, max_chunks, seg)
        if const_expr(c["index_split"]):
            _run_merge(c, t0, n, bid)


def _bounds(c, t0, n):
    return [ld_i32(c["args"]["ibound"], t0 + t) for t in range(n)]


def _wide(c, t0, n):
    """Is a candidate producer's widest row past PROD_FULL_MAX? Its
    selection then ranks its rows' candidate columns, compacted by its blk
    tree, else every column beside that tree (``run_indexer``)."""
    bounds = _bounds(c, t0, n)
    widest = bounds[0]
    for b in bounds[1:]:
        widest = fx.max(widest, b)
    return widest > PROD_FULL_MAX


def _whole(c, t0, n):
    """A tree's rows as (decode, a word a row): the decode gives a row's
    (column bound, columns, split) from its word. Here every row's bound, all
    its columns, unsplit (split None: a build or tree that never splits)."""
    return _whole_of, _bounds(c, t0, n)


def _shares(c, t0, n):
    """A FULL layer's rows (``_whole``'s form) in a build that splits
    (``AttnPostBuild.index_split``): a word a row from the step's plan past its
    tasks (``index_plan.share_word``; a whole row's columns are its bound,
    which the tree then needs no more of), a scalar load -- one word a row
    held, decoded at its use; the share's first column at its use
    (``_row_share``)."""
    if const_expr(not c["index_split"]):
        return _whole(c, t0, n)
    return _share_of, [_row_share(c, t0 + t)[1] for t in range(n)]


def _share_of(word):
    """A row's share from its ``share_word``: no bound (``stage_ilist``)."""
    return None, ip.word_cols(word), ip.word_split(word)


def _sel_rows(c, t0, n):
    """A split build's selection rows (``_whole``'s form, a (``share_word``,
    first column) a row): a FULL layer's from the step's plan (``_shares``); a
    candidate producer's from its bounds, a row past PROD_FULL_MAX split over
    the TP ranks by ``index_plan.row_shares``' arithmetic on its own bound
    (its shares partition its columns in rank order, each scored whole: the
    producer's walk takes every column for the blk tree), the others whole --
    where its full-column tree would outgrow the grid beside the blk tree, the
    split tree and the ranks' merge (``stage_imerge``) instead of that tree's
    compacted candidates: the same ids in the same order (a winner is in its
    share's top-512; rank order is column order; ties to the lower column)."""
    prod = c["args"]["iprod"] != 0
    tp, rank = c["d"].tp, c["rank"]
    rows = []
    for t in range(n):
        share = _row_share(c, t0 + t)
        first, word = share[0], share[1]
        # a candidate producer's own plan row is whole (``index_plan.splits``:
        # its ratio 1 never splits): its columns are its bound, ``ibound``'s
        # (both the step's visible rows at its ratio), a scalar load already
        # made, not a vector one ahead of the tree (a FULL row's tree too)
        b = ip.word_cols(word)
        split = b > PROD_FULL_MAX
        q = ip.share_len(b, tp)
        p_first = sel(split, rank * q, Int32(0))
        p_cols = sel(split, ip.imin(ip.imax(b - p_first, 0), q), b)
        p_word = ip.share_word(p_cols, sel(split, Int32(1), Int32(0)))
        rows.append((sel(prod, p_word, word), sel(prod, p_first, first)))
    return _sel_of, rows


def _sel_of(row):
    """A ``_sel_rows`` row's share: no bound, its first column carried."""
    word, first = row
    return None, ip.word_cols(word), ip.word_split(word), first


def _whole_of(bound):
    return bound, bound, None


def _row_share(c, t):
    """Row t's (first column, ``share_word``, 0, 0) on this rank
    (``index_plan.fill_shares``), a scalar load."""
    return sload4(c["args"]["iplan"] + fx.Int64(BLOCKS * 16) + fx.Int64(t) * 16)


@traced
def run_scores(c, t0, n, bid, bounds, max_chunks):
    """The score stage's tasks (``index_plan.score_tasks``) over rows t0 ..
    t0 + n. A request's tokens share its tile table (key metadata word 2: its
    block table offset); a REINDEX layer's are each token's own candidates."""
    assert n * TASK_CHUNKS <= THREADS, n  # a thread a (token, chunk) flag
    a = c["args"]
    reindex = a["ilift"] != 0
    owners = [
        sel(reindex, t0 + t, ld_i32(a["kmeta"], (t0 + t) * KMETA + 2)) for t in range(n)
    ]
    counts, spans, width_log2 = ip.score_tasks(bounds, owners, BLOCKS, TASK_CHUNKS_LOG2)
    width = Int32(1) << width_log2
    enter_stage("k2a.iscore")
    for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
        lead, i = ip.flat_task(counts, task)
        stage_iscore(
            c, t0 + lead, pick(spans, lead), n, i << width_log2, width, max_chunks
        )


@traced
def _run_tree(
    c, t0, n, bid, kind, rows, bound_max, max_chunks, seg, first=0, gathers_ok=True,
    compact=None,
):  # fmt: skip
    k = ip.KINDS[kind]
    decode, words = rows
    for level in range_constexpr(ip.top_level(kind, bound_max, seg) + 1):
        counts = [
            ip.list_count(ip.kind_bound(kind, decode(w)[1]), level, k, seg)
            for w in words
        ]
        enter_stage(f"k2a.i{kind}{level}")
        for task in range(first_task(bid, first), ip.flat_total(counts), BLOCKS):
            t, j = ip.flat_task(counts, task)
            share = decode(_pick_row(words, t))
            stage_ilist(
                c, t0 + t, j, share, kind, level, bound_max, max_chunks, seg,
                gathers_ok, compact,
            )  # fmt: skip


def _pick_row(words, t):
    """Row t of a tree's rows (a word, or a tuple of words: picked each)."""
    if isinstance(words[0], tuple):
        return tuple(pick([w[k] for w in words], t) for k in range(len(words[0])))
    return pick(words, t)


@traced
def _run_merge(c, t0, n, bid, rows=None, first=0):
    """The merges of the step's split rows (``stage_imerge``; ``rows``:
    ``_sel_rows``', None: the plan's), a row a CTA from CTA ``first``: 0 for a
    FULL layer -- the CTAs whose selection lists end first --, PROD_MERGE0 for
    a candidate producer."""
    enter_stage("k2a.imerge")
    for t in range(first_task(bid, first), n, BLOCKS):
        if const_expr(rows is None):
            split = ip.word_split(_row_share(c, t0 + t)[1])
        else:
            split = rows[0](_pick_row(rows[1], t))[2]
        if split != 0:
            stage_imerge(c, t0 + t)


@traced
def stage_imerge(c, t):
    """Split row t's selection: every rank's answer (TOPK (value bits, column)
    pairs, ``_put_answer``; rank order is column order) ranked as one list --
    the top-512 of them is the unsplit tree's (``index_plan.row_shares``) --
    into its slots, rows and IRDY as an unsplit row's last list."""
    tid = c["tid"]
    tp = c["d"].tp
    n = tp * ip.TOPK
    per = n // THREADS
    flags = preg(c["sym"], c["peer"]["ixrdy"], "ixrdy")
    if tid < tp:
        c["poll"]([(flags, t * tp + tid, 1)])
    gpu.barrier()
    # the peers' plain stores, past their flags: loads at their scope
    src = rsrc(c["args"]["ixsel"])
    base = t * n + tid * per
    words = [
        fx.Vector(
            bo.buffer_load(
                src, (base + 2 * w) * 2, vec_width=4, dtype=T.i32, cache_modifier=CM_SYS
            )
        )
        for w in range(per // 2)
    ]
    got = [
        (fx.Int32(words[d // 2][2 * (d % 2)]), fx.Int32(words[d // 2][2 * (d % 2) + 1]))
        for d in range(per)
    ]

    def load(q, i):
        return [got[4 * q + e][0] for e in range(4)]

    _select_sel(
        c, t, 0, fx.Int32(n), per, load, got, 1, ip.SEG, 0, True, None, 0, None,
        _anchor(c, t, "merge", 0, 0),
    )  # fmt: skip
    _final_rows_of(c, t, c["sel_lds"]["hist"])
    _publish_all(c, "irdy", t)
