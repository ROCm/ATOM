# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Paged index query, head weights, scoring and both selections in one launch.

Reads the same MXFP8 latent and BF16 attention norm as the native indexer.
The stages are shared with the layer monokernels, including FP4 query
rounding, the candidate-block tree and lifting REINDEX columns to real rows.
Each TP rank holds the replicated indexer and writes its own selection.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.mono.device.ops import (
    CM_DEV,
    kernel_symbol,
    launcher,
    ld_i32,
    read_lane_i32,
    rsrc,
    traced,
    uniform,
    wave_incl_scan,
    wave_max,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.sync import Mailbox, publish, sreg
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged import selection_plan as isp
from atom.mono.dsa_indexer.paged import tp_plan as tp
from atom.mono.dsa_indexer.paged.build import (
    ABI,
    MAX_ROWS,
    IndexerBuild,
    _select_segment,
    _split_weights,
    scratch_layout,
    selection_keys,
)
from atom.mono.dsa_indexer.paged.projection import (
    head_weight,
    lds_row,
    project_group,
    tile_rows,
    token_tiles,
)
from atom.mono.dsa_indexer.paged.score import (
    stage_ilist,
)
from atom.mono.dsa_indexer.paged.score_fp4 import run_scores4
from atom.mono.dsa_indexer.paged.sources import SOURCES
from atom.mono.dsa_indexer.paged.topk import lds_words
from atom.mono.dsa_indexer.paged.tp_radix import (
    run_splits,
)
from atom.mono.dsa_indexer.paged.tp_select import finish_row
from atom.mono.dsa_indexer.paged.tp_work import (
    initialize as initialize_tp,
)
from atom.mono.dsa_indexer.paged.tp_work import (
    initialize_scores,
    row_work,
)
from atom.mono.plan.arith import pick
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES, first_task
from atom.mono.runtime.debug import region_ids
from atom.mono.runtime.step_begin import EPOCH, FENCE_BYTES, _fence, _ld64

_CURRENT_STREAM = fx.Stream(None)


def _tree(c, s, bid, kind, bound_max, base=0):
    if c.get("world", 1) > 1:
        return _tree_tp(c, s, bid, kind, bound_max, base)
    return _tree_local(c, s, bid, kind, bound_max, base)


@traced
def _tree_local(c, s, bid, kind, bound_max, base=0):
    """Dispatch against this step's largest bound; keep mailbox strides fixed."""
    segment = isp.segment_for(c["select_seg"], kind)
    raw_max = uniform(wave_max(ld_i32(c["args"]["ibound"], fx.min(c["lane"], s - 1))))
    for level in range_constexpr(ip.top_level(kind, bound_max, seg=segment) + 1):
        slots = ip.list_count(
            ip.kind_bound(kind, raw_max), level, ip.KINDS[kind], seg=segment
        )
        for task in range(first_task(bid, base), s * slots, BLOCKS):
            # Distribute each segment across the rows before its successor;
            # any second task on a CTA then comes from the final short segment.
            t, j = task % s, task // s
            raw = ld_i32(c["args"]["ibound"], t)
            stage_ilist(c, t, j, raw, kind, level, bound_max, ip.chunks(bound_max))
        base = base + s * slots
    return base


@traced
def _tree_tp(c, s, bid, kind, bound_max, base=0):
    segment = c["block_select_seg"] if kind == "blk" else c["select_seg"]
    lane = c["lane"]
    mode, owner, _first, raw = row_work(c, fx.min(lane, s - 1))
    bound = ip.kind_bound(kind, raw)
    eligible = (lane < s) & ((mode != tp.OWNED) | (owner == c["rank"]))
    if const_expr("select_split" in c):
        eligible = eligible & (
            (mode == tp.SPLIT) if c["select_split"] else (mode != tp.SPLIT)
        )
    for level in range_constexpr(ip.top_level(kind, bound_max, seg=segment) + 1):
        count = eligible.select(
            ip.list_count(bound, level, ip.KINDS[kind], seg=segment), fx.Int32(0)
        )
        end = wave_incl_scan(count, lane)
        start = end - count
        total = uniform(read_lane_i32(end, s - 1))
        for task in range(first_task(bid, base), total, BLOCKS):
            hit = (task >= start) & (task < end) & eligible
            row = uniform(wave_max(hit.select(lane, fx.Int32(-1))))
            index = task - uniform(read_lane_i32(start, row))
            stage_ilist(
                c,
                row,
                index,
                ld_i32(c["args"]["ibound"], row),
                kind,
                level,
                bound_max,
                ip.chunks(bound_max),
            )
        base = base + total
    return base


@traced
def _run_fp4_local(c, s, bid, bound_max):
    run_scores4(c, 0, s, bid, ip.chunks(bound_max))
    next_base = _tree(c, s, bid, "sel", bound_max)
    if const_expr(not c.get("full_only", False)):  # noqa: SIM102
        if c["args"]["iprod"] != 0:
            _tree(c, s, bid, "blk", bound_max, next_base)


@traced
def _run_fp4_tp(c, key, peers, rank, ix_meta, step_fence):
    s = key.tokens
    if step_fence != 0:
        c["poll"]([(c["ix_fenced"], 0, 1)])
    _run_tp_scores(c, key.world, rank, ix_meta)
    bases = peer_bases(peers, key.world)
    co = {
        **c,
        "world": key.world,
        "rank": rank,
        "tp_direct": False,
        "select_seg": 8192 if key.full_only else 4096,
        "block_select_seg": 8192,
        "list_storage_segment": 4096,
        "block_list_storage_segment": 8192,
        "peer_addr": lambda p: pick(bases, p),
        "peer_plain": key.peer_plain,
        "ix_peer_layout": {
            n: (off + key.peer_offset, size)
            for n, (off, size) in tp.peer_layout(s, key.world).items()
        },
        "ix_plain_layout": tp.plain_layout(s),
        "ix_pairs": {
            kind: pick(bases, rank)
            + key.peer_plain
            + tp.plain_layout(s)[f"ix_{kind}_pairs"][0]
            for kind in tp.KINDS
        },
        "args": {**c["args"], "ix_meta": ix_meta},
        "load_bound": lambda row: ld_i32(c["args"]["ibound"], row),
    }
    initialize_tp(co)
    # A split row has fewer leaves. An 8K leaf keeps its Top512 root in
    # two levels at the common TP2/TP4 160K sizes; owners keep 4K leaves
    # to distribute many independent rows across the whole CTA grid.
    next_base = _tree(
        {**co, "select_seg": 8192, "select_split": True},
        s,
        co["bid"],
        "sel",
        key.bound_max,
    )
    next_base = _tree(
        {**co, "select_split": False}, s, co["bid"], "sel", key.bound_max, next_base
    )
    if const_expr(not key.full_only):  # noqa: SIM102
        if co["args"]["iprod"] != 0:
            next_base = run_splits(co, "blk", next_base)
            _tree(
                {**co, "select_split": False},
                s,
                co["bid"],
                "blk",
                key.bound_max,
                next_base,
            )
    for t in range(co["bid"], s, BLOCKS):
        mode, owner, _first, _count = row_work(co, t)
        finish_row(co, t, "sel", mode, owner)
        if const_expr(not key.full_only):  # noqa: SIM102
            if co["args"]["iprod"] != 0:
                finish_row({**co, "tp_direct": True}, t, "blk", mode, owner)


@traced
def _run_tp_scores(c, world, rank, meta):
    if c["args"]["ilift"] != 0:
        _run_tp_reindex_scores(c, world, rank, meta)
    else:
        run_scores4(c, 0, c["S"], c["bid"], ip.chunks(c["bound_max"]))


@traced
def _run_tp_reindex_scores(c, world, rank, meta):
    co = {**c, "world": world, "rank": rank, "args": {**c["args"], "ix_meta": meta}}
    initialize_scores(co)
    run_scores4(co, 0, co["S"], co["bid"], ip.chunks(co["bound_max"]))


@traced
def _load_latent(tid, qr, qr_scale, qxl, qxsl, t0, n, query_rank):
    qx8_words = query_rank // 4
    for u in range(tid, n * qx8_words, THREADS):
        t, col = u // qx8_words, u % qx8_words
        v = bo.buffer_load(
            rsrc(qr),
            (t0 + t) * qx8_words + col,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
        fx.ptr_store(v, qxl + t * lds_row(qx8_words) + col)
    for u in range(tid, n * (query_rank // 32), THREADS):
        t, col = u // (query_rank // 32), u % (query_rank // 32)
        at = (t0 + t) * (query_rank // 32) + col
        v = bo.buffer_load(
            rsrc(qr_scale), at // 4, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
        )
        fx.ptr_store(
            (fx.Int32(v) >> ((at % 4) * 8)) & 255,
            qxsl + t * lds_row(query_rank // 32) + col,
        )
    gpu.barrier()


def build_indexer(key: IndexerBuild):
    s = key.tokens
    if not 1 <= s <= MAX_ROWS or key.bound_max <= 0:
        raise ValueError(f"unsupported V4.1 indexer build: {key}")
    if not key.fp4:
        raise ValueError("standalone V4.1 indexer requires FP4")
    if key.world not in (1, 2, 4, 8):
        raise ValueError("cross-TP V4.1 indexer requires FP4 and TP1/2/4/8")
    query_rank, qx8_words = key.query_rank, key.query_rank // 4
    layout = scratch_layout(key)
    keyed = key_tuple(key, SOURCES)
    rows = tile_rows(s)
    tiles = token_tiles(s)
    parts = 2 if s > 32 else 4
    split_weights = _split_weights(key)
    select_segment = _select_segment(key)
    sel = lds_words(select_segment // THREADS, fast_hist=True)
    sel_hist, sel_buf = sel["hist"], sel["buf"]
    peer_topk = ip.TOPK if key.full_only else ip.TOPK_BLOCKS
    sel_keys = selection_keys(key)

    @fx.struct
    class ProjectionLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        qxl: fx.Array[fx.Int32, rows * lds_row(qx8_words), 16]
        qxsl: fx.Array[fx.Int32, rows * lds_row(query_rank // 32), 16]
        nbl: fx.Array[fx.Int32, 1, 16]
        qbf: fx.Array[fx.BFloat16, rows * ip.DIM, 16]

    @fx.struct
    class SelectionLds:
        sk: fx.Array[fx.Int32, sel_keys, 16]
        sh: fx.Array[fx.Int32, sel_hist, 16]
        sb: fx.Array[fx.Int32, sel_buf, 16]
        si: fx.Array[fx.Int32, key.world * peer_topk if key.world > 1 else 1, 16]
        rs: fx.Array[fx.Int32, 4, 16]
        rc: fx.Array[fx.Int32, WAVES * 2, 16]
        rb: fx.Array[
            fx.Int32,
            (BLOCKS * 2 if key.world > 1 and not key.full_only else 1),
            16,
        ]

    @fx.union
    class StageLds:
        project: ProjectionLds
        select: SelectionLds

    name = kernel_symbol(
        "mono_paged_indexer",
        s=s,
        c=key.bound_max,
        fp4=int(key.fp4),
        mfma=int(key.mfma),
        diag=key.diag_off,
        tp=key.world,
        po=key.peer_offset,
        pp=key.peer_plain,
        full=int(key.full_only),
        hidden=key.hidden_size,
        qr=query_rank,
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def indexer(
        hidden: Int64, qr: Int64, qr_scale: Int64, iq_w: Int64, iq_ws: Int64,
        iw_w: Int64, pos: Int64, cos: Int64, sin: Int64, iq: Int64, iqs: Int64,
        iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64, lstride: Int32, isel: Int64, icand: Int64,
        ilift: Int32, iprod: Int32, ibmax: Int64, bstride: Int32, icout: Int64,
        irow: Int64, kmeta: Int64, table: Int64, rows_per_page: Int32,
        page_rows: Int32, main_off: Int32, scratch: Int64, layer: Int32,
        peers: Int64, rank: Int32, ix_meta: Int64,
        step_state: Int64, step_fence: Int32,
    ):  # fmt: skip
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator()
        storage = alloc.allocate(StageLds)
        pr, se = storage.project.peek(), storage.select.peek()
        mb = (
            Mailbox(layer)
            if key.diag_off < 0
            else Mailbox(layer, scratch + fx.Int64(key.diag_off), region_ids(layout))
        )
        if const_expr(key.full_only):
            ilift = fx.Int32(0)
            iprod = fx.Int32(0)
        c = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64,
            "bound_max":key.bound_max,
            "hidden_size": key.hidden_size, "query_rank": query_rank,
            "red": pr.red.ptr, "qxl": pr.qxl.ptr, "qxsl": pr.qxsl.ptr,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words,
            "poll": mb.poll, "tok_par": False,
            "spin": mb.spin,
            "radix_sel": se.rs.ptr,"radix_scan": se.rc.ptr,"radix_before": se.rb.ptr,
            "defer_selected_rows": True,
            "score_depth": 1,
            "single_row_depth": 2,
            "wide_single_row_score": True,
            "fast_select": True,
            "compact_selected_rows": True,
            "list_ready": True,
            "full_only": key.full_only,
            "tight_select_width": key.full_only,
            "select_seg": select_segment,
            "query_ready": True,
            "query_parts": parts,
            "split_weights": split_weights,
            "single_row_full": True,
            "index_fp4": True, "index_mfma": key.mfma,
            "args": {
                "iact": 1, "iq_w": iq_w, "iq_ws": iq_ws, "iw_w": iw_w,
                "pos": pos, "cos": cos, "sin": sin,
                "iq_out": iq, "iqs_out": iqs, "iw_out": iw,
                "iq": iq, "iqs": iqs, "iw": iw, "plane": plane, "pscale": pscale,
                "iplan": iplan, "itab": itab, "itab_stride": itab_stride,
                "itab_len": itab_len, "ibatch": ibatch, "ishift": ishift,
                "ibound": ibound, "ilog": ilog, "lstride": lstride, "isel": isel,
                "icand": icand, "ilift": ilift, "iprod": iprod,
                "ibmax": ibmax, "bstride": bstride, "icout": icout, "irow": irow,
                "kmeta": kmeta, "table": table, "rows_per_page": rows_per_page,
                "page_rows": page_rows, "main_off": main_off,
            },
        }  # fmt: skip
        for region, (off, _) in layout.items():
            c[region] = sreg(scratch, off, region)
        short_step = fx.Boolean(True)
        if const_expr(key.world > 1):
            short_step = (ld_i32(ix_meta, 3) & 65536) != 0
            if (step_fence != 0) & ~short_step & (ilift == 0) & (bid == BLOCKS - 1):
                if tid < 64:
                    epoch = _ld64(rsrc(step_state), EPOCH) + 1
                    _fence(
                        peers,
                        fx.Int64(key.peer_plain - FENCE_BYTES),
                        rank,
                        step_state,
                        epoch,
                        key.world,
                        False,
                    )
                publish(c["put"], c["ix_fenced"], 0, 1, tid == 0)
        for rep in range_constexpr(-(-(len(tiles) * ip.HEADS * parts) // BLOCKS)):
            task = bid + rep * BLOCKS
            for i, (t0, n) in enumerate(tiles):
                if (task >= i * ip.HEADS * parts) & (task < (i + 1) * ip.HEADS * parts):
                    _load_latent(
                        tid, qr, qr_scale, pr.qxl.ptr, pr.qxsl.ptr, t0, n, query_rank
                    )
                    project_group(
                        c,
                        hidden,
                        t0,
                        n,
                        (task % (ip.HEADS * parts)) // parts,
                        task % parts,
                        parts,
                        with_weight=not split_weights,
                    )
        if const_expr(split_weights):
            first = (s + 1) // 2
            for i, (t0, n) in enumerate(((0, first), (first, s - first))):
                start = len(tiles) * ip.HEADS * parts + i * ip.HEADS
                if (bid >= start) & (bid < start + ip.HEADS):
                    head = bid - start
                    head_weight(c, hidden, t0, n, head)
                    publish(
                        c["put"],
                        c["iwready"],
                        (t0 + tid) * ip.HEADS + head,
                        1,
                        tid < n,
                    )
        c["sel_lds"] = {
            "keys": se.sk.ptr,
            "hist": se.sh.ptr,
            "buf": se.sb.ptr,
            "identities": se.si.ptr,
        }
        if const_expr(key.world > 1):
            # REINDEX has at most 2048 eight-row candidate blocks.
            # Its compressed key set is cheaper to select locally than
            # to split, merge and publish over the peer interconnect.
            if short_step | (ilift != 0):
                _run_fp4_local(c, s, bid, key.bound_max)
            else:
                _run_fp4_tp(c, key, peers, rank, ix_meta, step_fence)
        else:
            _run_fp4_local(c, s, bid, key.bound_max)
        _ = keyed

    @launcher(name)
    def launch(
        hidden: Int64, qr: Int64, qr_scale: Int64, iq_w: Int64, iq_ws: Int64,
        iw_w: Int64, pos: Int64, cos: Int64, sin: Int64, iq: Int64, iqs: Int64,
        iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64, lstride: Int32, isel: Int64, icand: Int64,
        ilift: Int32, iprod: Int32, ibmax: Int64, bstride: Int32, icout: Int64,
        irow: Int64, kmeta: Int64, table: Int64, rows_per_page: Int32,
        page_rows: Int32, main_off: Int32, scratch: Int64, layer: Int32,
        peers: Int64, rank: Int32, ix_meta: Int64,
        step_state: Int64, step_fence: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        indexer(
            hidden,
            qr,
            qr_scale,
            iq_w,
            iq_ws,
            iw_w,
            pos,
            cos,
            sin,
            iq,
            iqs,
            iw,
            plane,
            pscale,
            iplan,
            itab,
            itab_stride,
            itab_len,
            ibatch,
            ishift,
            ibound,
            ilog,
            lstride,
            isel,
            icand,
            ilift,
            iprod,
            ibmax,
            bstride,
            icout,
            irow,
            kmeta,
            table,
            rows_per_page,
            page_rows,
            main_off,
            scratch,
            layer,
            peers,
            rank,
            ix_meta,
            step_state,
            step_fence,
        ).launch(grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    ABI.check(indexer, launch)
    return launch
