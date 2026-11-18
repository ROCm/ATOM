# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4.1's index query, head weights, scoring and both selections in one launch.

Reads the same MXFP8 latent and BF16 attention norm as the native indexer.
The stages are shared with the layer monokernels, including FP4 query
rounding, the candidate-block tree and lifting REINDEX columns to real rows.
Each TP rank holds the replicated indexer and writes its own selection.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    HIDDEN,
    Q_RANK,
    QX8_WORDS,
    gemv_fp8_loads,
    iq_rope,
    lds_row,
    stage_iq,
    tile_rows,
    token_tiles,
)
from atom.models.deepseek_v41.mono.kernels.debug import mailbox
from atom.models.deepseek_v41.mono.kernels.index_query import (
    IDX_ROWS,
    IW_TASKS,
    stage_iquant,
    stage_iquant_fp4,
    stage_iw,
)
from atom.models.deepseek_v41.mono.kernels.index_score import (
    PER_THREAD,
    run_indexer,
    scratch_pairs,
    stage_ilist,
)
from atom.models.deepseek_v41.mono.kernels.index_score_fp4 import run_scores4
from atom.models.deepseek_v41.mono.kernels.topk import lds_words
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import CM_DEV, kernel_symbol, launcher, ld_i32, rsrc, traced
from atom.mono.device.sync import publish, sreg
from atom.mono.plan.build_key import key_tuple
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES, first_task
from atom.mono.plan.layout import pair_layout
from atom.mono.runtime.abi import KernelAbi

MAX_ROWS = 64
_CURRENT_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class IndexerBuild:
    tokens: int
    bound_max: int
    fp4: bool = True
    mfma: bool = True
    diag_off: int = -1


ABI = KernelAbi((
    "hidden", "qr", "qr_scale", "iq_w", "iq_ws", "iw_w", "pos", "cos", "sin",
    "iq", "iqs", "iw", "plane", "pscale", "iplan", "itab", "itab_stride",
    "itab_len", "ibatch", "ishift", "ibound", "ilog", "lstride", "isel",
    "icand", "ilift", "iprod", "ibmax", "bstride", "icout", "irow",
    "kmeta", "table", "rows_per_page", "page_rows", "main_off", "scratch", "layer",
))  # fmt: skip


def scratch_layout(key: IndexerBuild):
    return pair_layout(
        {
            "iq": key.tokens * IDX_ROWS // 2,
            "iw": key.tokens * ip.HEADS,
            "qdone": BLOCKS,
            **scratch_pairs(key.tokens, key.bound_max),
        }.items()
    )


def scratch_bytes(key: IndexerBuild):
    return max(o + n for o, n in scratch_layout(key).values())


@traced
def _tree(c, s, bid, kind, bound_max):
    """Fixed slots avoid keeping every query's dynamic tree counts live."""
    for level in range_constexpr(ip.top_level(kind, bound_max) + 1):
        slots = ip.list_slots(kind, level, bound_max)
        for task in range(first_task(bid, 0), s * slots, BLOCKS):
            t, j = task // slots, task % slots
            raw = ld_i32(c["args"]["ibound"], t)
            stage_ilist(c, t, j, raw, kind, level, bound_max, ip.chunks(bound_max))


@traced
def _run_fp4(c, s, bid, bound_max):
    run_scores4(c, 0, s, bid, ip.chunks(bound_max))
    _tree(c, s, bid, "sel", bound_max)
    if c["args"]["iprod"] != 0:
        _tree(c, s, bid, "blk", bound_max)


@traced
def _load_latent(tid, qr, qr_scale, qxl, qxsl, t0, n):
    for u in range(tid, n * QX8_WORDS, THREADS):
        t, col = u // QX8_WORDS, u % QX8_WORDS
        v = bo.buffer_load(
            rsrc(qr),
            (t0 + t) * QX8_WORDS + col,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
        fx.ptr_store(v, qxl + t * lds_row(QX8_WORDS) + col)
    for u in range(tid, n * (Q_RANK // 32), THREADS):
        t, col = u // (Q_RANK // 32), u % (Q_RANK // 32)
        at = (t0 + t) * (Q_RANK // 32) + col
        v = bo.buffer_load(
            rsrc(qr_scale), at // 4, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
        )
        fx.ptr_store(
            (fx.Int32(v) >> ((at % 4) * 8)) & 255,
            qxsl + t * lds_row(Q_RANK // 32) + col,
        )
    gpu.barrier()


def build_indexer(key: IndexerBuild):
    s = key.tokens
    if not 1 <= s <= MAX_ROWS or key.bound_max <= 0:
        raise ValueError(f"unsupported V4.1 indexer build: {key}")
    if key.mfma and not key.fp4:
        raise ValueError("MFMA layout is an FP4 index plane")
    layout = scratch_layout(key)
    keyed = key_tuple(key, SOURCES)
    rows = tile_rows(s)
    tiles = token_tiles(s)
    sel = lds_words(PER_THREAD)
    sel_keys, sel_hist, sel_buf = sel["keys"], sel["hist"], sel["buf"]

    @fx.struct
    class ProjectionLds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        qxl: fx.Array[fx.Int32, rows * lds_row(QX8_WORDS), 16]
        qxsl: fx.Array[fx.Int32, rows * lds_row(Q_RANK // 32), 16]
        nbl: fx.Array[fx.Int32, min(s, WAVES) * HIDDEN // 2, 16]

    @fx.struct
    class SelectionLds:
        sk: fx.Array[fx.Int32, sel_keys, 16]
        sh: fx.Array[fx.Int32, sel_hist, 16]
        sb: fx.Array[fx.Int32, sel_buf, 16]

    @fx.union
    class StageLds:
        project: ProjectionLds
        select: SelectionLds

    name = kernel_symbol(
        "mono_v41_indexer",
        s=s,
        c=key.bound_max,
        fp4=int(key.fp4),
        mfma=int(key.mfma),
        diag=key.diag_off,
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
    ):  # fmt: skip
        tid, bid = fx.thread_idx.x, fx.block_idx.x
        alloc = fx.SharedAllocator()
        storage = alloc.allocate(StageLds)
        pr, se = storage.project.peek(), storage.select.peek()
        mb = mailbox(layer, scratch, key.diag_off, layout)
        c = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64,
            "red": pr.red.ptr, "qxl": pr.qxl.ptr, "qxsl": pr.qxsl.ptr,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words,
            "poll": mb.poll, "tok_par": False,
            "defer_selected_rows": True,
            "score_depth": 1,
            "single_row_full": True,
            "index_fp4": key.fp4, "index_mfma": key.mfma,
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
        task = first_task(bid, 0)
        ops = gemv_fp8_loads(c, iq_w, iq_ws, Q_RANK, task)
        for t0, n in tiles:
            rope = iq_rope(c, task, t0, n)
            _load_latent(tid, qr, qr_scale, pr.qxl.ptr, pr.qxsl.ptr, t0, n)
            stage_iq(c, task, ops, t0, n, rope)

        @traced
        def load_group(g, n):
            for u in range(tid, n * HIDDEN // 2, THREADS):
                v = bo.buffer_load(
                    rsrc(hidden),
                    g * (HIDDEN // 2) + u,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
                fx.ptr_store(v, pr.nbl.ptr + u)
            gpu.barrier()

        for task in range(first_task(bid, 0), IW_TASKS, BLOCKS):
            stage_iw(c, task, HIDDEN, pr.nbl.ptr, load_group)
        for task in range(first_task(bid, IW_TASKS), s * ip.HEADS, BLOCKS):
            if const_expr(key.fp4):
                stage_iquant_fp4(c, task)
            else:
                stage_iquant(c, task)
        # Quantized Q and IW are plain buffers. Drain them before announcing
        # this CTA; all resident CTAs finish projection before any scores read Q.
        publish(c["put"], c["qdone"], bid, 1, tid == 0)
        if tid < BLOCKS:
            c["poll"]([(c["qdone"], tid, 1)])
        gpu.barrier()
        c["sel_lds"] = {"keys": se.sk.ptr, "hist": se.sh.ptr, "buf": se.sb.ptr}
        if const_expr(key.fp4):
            _run_fp4(c, s, bid, key.bound_max)
        else:
            run_indexer(c, s, bid, key.bound_max)
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
        ).launch(grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    ABI.check(indexer, launch)
    return launch
