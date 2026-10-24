# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 merged K2, following gfx950 layer_post's two-half schedule.

Score/PV/IRQ/WO -> tagged attention TP reduce -> delayed seam/norm ->
router/shared/routed MoE -> tagged FFN TP reduce. The two halves retain
separate global scratch and share one LDS arena. Native index/compressor
metadata is prepared before this launch. StepMailboxes owns clear/fence/tag
lifetime; PeerBuffer owns uncached IPC allocation and peer mappings.
"""

from functools import lru_cache
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.kernels.gfx1250_attention import (
    ATTENTION_LDS_BYTES,
    attention_tasks,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_attn_post import GROUP_K, HIDDEN
from atom.models.deepseek_v41.mono.kernels.gfx1250_ffn import (
    LDS_BYTES,
    ffn_gfx1250,
    ffn_scratch_specs,
    ffn_state_layout,
    run_ffn,
    TRANSITIVE_FFN_WAITS,
    C1_ROUTE_UNION,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_gemv import gemv_lds_bytes
from atom.models.deepseek_v41.mono.kernels.gfx1250_k1 import (
    _await,
    _CooperativeKernel,
    _i32_pointer,
    _publish,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_layer_post import (
    ATTENTION_STATE_SIZE,
    WORKERS,
    SYNC_DEBUG,
    _dispatch,
    _tensor_spec,
    attention_projection_gfx1250,
    attention_scratch_specs,
    run_attention_projection,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_mhc import _typed
from atom.models.deepseek_v41.mono.kernels.gfx1250_moe import GU_PROJECTIONS, GU_WAVE_GROUPS, DOWN_PARALLEL, TOPK, moe_widths
from atom.models.deepseek_v41.mono.kernels.gfx1250_indexer import index_lds_bytes, run_index
from atom.models.deepseek_v41.mono.kernels import gfx1250_timeline as timeline_ops
from atom.models.deepseek_v41.mono.kernels.gfx1250_attention import PRUNE_EMPTY
from atom.models.deepseek_v41.mono.kernels.gfx1250_ffn import SEAM_OWNER, FFN_PROGRESS_WORDS
from atom.models.deepseek_v41.mono.kernels.gfx1250_layer_post import WO_PREFETCH
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_bytes, peer_offsets
from atom.mono.device.ops import CM_DEV, bf16_pair, load_ptr64, rsrc, traced
from atom.mono.device.ranks import peer_bases, sum_partials
from atom.mono.device.sync import Mailbox, preg
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.step_begin import FENCE_SLOTS, MISSING, STATE_WORDS

POST_TAG0, MOE_TAG0 = 64, 128
PEER_DIAG_WORDS = WORKERS * 8
_CURRENT_STREAM = fx.Stream(None)
C1_ATTENTION_PLACEMENT = os.environ.get("ATOM_DSV41_GFX1250_C1_ATTENTION_PLACEMENT", "0") == "1"


def merged_state_layout(rows, tp, index_bound_max=0):
    f = ffn_state_layout(rows, tp)
    fbase = ATTENTION_STATE_SIZE - 1
    reduce = fbase + f["diagnostic"][0]
    index = reduce + WORKERS
    return {"ffn": fbase, "reduce": reduce, "index": index,
            "diagnostic": index + (rows * ((index_bound_max + 127) // 128 + 1) if index_bound_max else 0)}


def merged_scratch_specs(rows, tp, index_bound_max=0):
    a = attention_scratch_specs(rows, 64 // tp)
    f = ffn_scratch_specs(rows, tp)
    layout = merged_state_layout(rows, tp, index_bound_max)
    return {
        **{"a_" + n: spec for n, spec in a.items() if n not in ("state", "debug_out")},
        **{"f_" + n: spec for n, spec in f.items() if n != "state"},
        "state": ((layout["diagnostic"] + 1,), torch.int32),
        "peer_diag": ((PEER_DIAG_WORDS,), torch.int32),
        "i_logits": ((rows, index_bound_max), torch.float32),
        "score_progress": ((WORKERS, 3), torch.int32),
        "ffn_progress": ((WORKERS, FFN_PROGRESS_WORDS), torch.int32),
        "sync_detail": ((16,), torch.int32),
        "timeline": ((WORKERS, timeline_ops.POINTS) if timeline_ops.ENABLED else (0,), torch.int64),
    }


@traced
def publish_reduce(
    source, dest, region, put_words, poll, bases, rank, tid, bid, rows, tp, ready,
    reduce=True,
):
    """gfx950's paired BF16 payload/tag put, then the rank-ordered FP32 sum."""
    own_base = bases[0]
    for p in range_constexpr(1, tp):
        own_base = (rank == p).select(bases[p], own_base)
    if tid < 8:
        for row in range_constexpr(rows):
            at = row * (HIDDEN // 2) + bid * 8 + tid
            # Each CTA owns 16 output columns. A failed producer still sends
            # fresh zero pairs, so a healthy peer never waits for a skipped put.
            word = fx.Int32(0)
            if ready:
                word = fx.Int32(
                    bo.buffer_load(
                        rsrc(fx.Int64(fx.ptrtoint(source))),
                        at,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
            pair = rank * rows * (HIDDEN // 2) + at
            for p in range_constexpr(tp):
                put_words(
                    preg(bases[p], region, "attn" if region == 0 else "ffn"),
                    pair,
                    [word],
                )
            if reduce:
                own = preg(own_base, region, "attn" if region == 0 else "ffn")
                lo, hi = sum_partials(
                    poll,
                    own,
                    lambda src, at=at: src * rows * (HIDDEN // 2) + at,
                    tp,
                )
                fx.ptr_store(
                    bf16_pair(lo, hi).bitcast(fx.Int32),
                    _typed(dest, fx.Int32) + at,
                )
    fx.gpu.barrier()


def merged_name(
    rows,
    tp,
    table_rows,
    pool_rows,
    ring_rows,
    prefix_capacity,
    extend_capacity,
    softmax_scale,
    interleaved,
    index_bound_max=0,
):
    from atom.models.deepseek_v41.mono.kernels.gfx1250_layer_post import _attention_name

    base = _attention_name(
        rows,
        64 // tp,
        table_rows,
        pool_rows,
        ring_rows,
        prefix_capacity,
        extend_capacity,
        softmax_scale,
        False,
    )
    return base.replace("k2_attention", "k2_merged") + f"_tp{tp}_i{int(interleaved)}_idx{index_bound_max}_syncd{int(SYNC_DEBUG)}_tl{int(timeline_ops.ENABLED)}_so{int(SEAM_OWNER)}_ap{int(PRUNE_EMPTY)}_pf{WO_PREFETCH}_gu{GU_PROJECTIONS}_dp{DOWN_PARALLEL}_wg{int(GU_WAVE_GROUPS)}_tw{int(TRANSITIVE_FFN_WAITS)}_ru{int(C1_ROUTE_UNION)}_pl{int(C1_ATTENTION_PLACEMENT and rows == 1)}"


ARGUMENTS = KernelAbi(
    (
        "query",
        "pool",
        "ring",
        "prefix",
        "prefix_indptr",
        "extend",
        "extend_indptr",
        "sink",
        "positions",
        "cos",
        "sin",
        "woa",
        "woa_s",
        "wob",
        "wob_s",
        "residual",
        "pre",
        "post",
        "comb",
        "fn",
        "scale",
        "bias",
        "nw",
        "gate_w",
        "gate_bias",
        "w13",
        "w13_s",
        "w2",
        "w2_s",
        "sgu",
        "sgu_s",
        "sw2",
        "sw2_s",
        "peers",
        "metadata",
        "step_state",
        "codes",
        "sticky",
        "i_query", "i_weights", "i_plane", "i_visible", "i_table",
        "i_keymeta", "i_config", "i_selection", "i_candidates",
        "attention",
        "residual_out",
        "pre_out",
        "post_out",
        "comb_out",
        "normed",
        "out",
        "a_maxima",
        "a_sums",
        "a_probabilities",
        "a_attention_q",
        "a_attention_s",
        "a_mid",
        "a_mid_q",
        "a_mid_s",
        "f_collapse",
        "f_partial",
        "f_shared_q",
        "f_shared_s",
        "f_routed_q",
        "f_routed_s",
        "f_logits",
        "f_route_ids",
        "f_route_weights",
        "f_union_ids",
        "f_union_count",
        "f_pick_slots",
        "f_shared_gup",
        "f_shared_mid",
        "f_shared_mid_q",
        "f_shared_mid_s",
        "f_shared_out",
        "f_routed_gup",
        "f_routed_mid",
        "f_routed_mid_q",
        "f_routed_mid_s",
        "f_contributions",
        "state",
        "peer_diag",
        "i_logits",
        "score_progress",
        "ffn_progress",
        "sync_detail",
        "timeline",
    )
)
ABI = KernelAbi(("arguments",))
READONLY = 47


def merged_lds_bytes(rows, index_bound_max=0):
    workspace = max(ATTENTION_LDS_BYTES, gemv_lds_bytes(rows, GROUP_K), LDS_BYTES,
                    index_lds_bytes(index_bound_max) if index_bound_max else 0)
    return workspace + 32 + (timeline_ops.LDS_BYTES if timeline_ops.ENABLED else 0)


@lru_cache(maxsize=64)
def _merged_launcher(
    rows,
    tp,
    table_rows,
    pool_rows,
    ring_rows,
    prefix_capacity,
    extend_capacity,
    softmax_scale,
    interleaved=True,
    index_bound_max=0,
):
    heads = 64 // tp
    real, padded = moe_widths(tp)
    workspace = max(ATTENTION_LDS_BYTES, gemv_lds_bytes(rows, GROUP_K), LDS_BYTES,
                    index_lds_bytes(index_bound_max) if index_bound_max else 0)
    score_tasks, pv_tasks = attention_tasks(rows, heads)
    layout = merged_state_layout(rows, tp, index_bound_max)
    foff = {key: off for key, (off, _) in ffn_state_layout(rows, tp).items()}
    poff = peer_offsets(rows, tp)
    name = merged_name(
        rows,
        tp,
        table_rows,
        pool_rows,
        ring_rows,
        prefix_capacity,
        extend_capacity,
        softmax_scale,
        interleaved,
        index_bound_max,
    )

    @flyc.jit
    def launch(arguments: fx.Pointer, stream: fx.Stream = _CURRENT_STREAM):
        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def kernel(arguments: fx.Pointer):
            def argument(name):
                table = fx.Int64(
                    llvm.inline_asm(
                        T.i64,
                        [fx.Int64(fx.ptrtoint(arguments)).ir_value()],
                        "s_mov_b64 $0, $1",
                        "=s,s",
                        has_side_effects=True,
                    )
                )
                value = load_ptr64(table, ARGUMENTS.names.index(name))
                return fx.inttoptr(fx.PointerType.get(fx.Uint8.ir_type, 1, 16), value)

            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(merged_lds_bytes(rows, index_bound_max))
            local = _i32_pointer(arena.base_ptr)
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            status = local + workspace // 4
            tls = (_typed(arena.base_ptr, fx.Int64) + (workspace + 32) // 8
                   if timeline_ops.ENABLED else None)
            flags = _i32_pointer(argument("state"))
            fflags = flags + layout["ffn"]
            diag = flags + layout["diagnostic"]
            failure_detail = _i32_pointer(argument("sync_detail"))
            if tid == 0:
                meta = _typed(argument("metadata"), fx.Int32)
                layer = fx.ptr_load(meta + 1)
                # All 40 layers reuse this scratch; StepMailboxes clears it
                # before each step. Use one immutable layer generation for
                # every CTA, independent of a CTA's previous completion load.
                fx.ptr_store(layer + 1, local)
                failed = fx.max(
                    fx.ptr_load(_typed(argument("codes"), fx.Int32) + layer),
                    fx.ptr_load(_typed(argument("sticky"), fx.Int32)),
                )
                if failed != 0:
                    fx.atomic_cas(
                        diag,
                        fx.Int32(0),
                        failed,
                        success_ordering=fx.AtomicOrdering.Release,
                        failure_ordering=fx.AtomicOrdering.Monotonic,
                        syncscope="agent",
                    )
                for p in range_constexpr(FENCE_SLOTS):
                    if (
                        fx.ptr_load(
                            _typed(argument("step_state"), fx.Int64) + MISSING + p
                        )
                        != 0
                    ):
                        fx.atomic_cas(
                            diag,
                            fx.Int32(0),
                            fx.Int32(31_000_000) + p,
                            success_ordering=fx.AtomicOrdering.Release,
                            failure_ordering=fx.AtomicOrdering.Monotonic,
                            syncscope="agent",
                        )
            fx.gpu.barrier()
            epoch = fx.ptr_load(local)
            fx.gpu.barrier()
            timeline_ops.begin(tls, tid, epoch)

            def after_projection(ready, task):
                timeline_ops.mark(tls, tid, "attn_tp_put")
                metadata = _typed(argument("metadata"), fx.Int32)
                rank, layer = fx.ptr_load(metadata), fx.ptr_load(metadata + 1)
                bases = peer_bases(fx.Int64(fx.ptrtoint(argument("peers"))), tp)
                ma = Mailbox(
                    layer + POST_TAG0,
                    fx.Int64(fx.ptrtoint(argument("peer_diag"))),
                    {"attn": 0, "ffn": 1},
                    steady=True,
                    failure=diag,
                )
                publish_reduce(
                    attention,
                    attention,
                    poff["attn"],
                    ma.put_words,
                    ma.poll,
                    bases,
                    rank,
                    tid,
                    task,
                    rows,
                    tp,
                    ready,
                    reduce=False,
                )
                _publish(flags, layout["reduce"] + task, epoch, synchronized=True,
                         joined=tls is not None or (C1_ATTENTION_PLACEMENT and rows == 1))
                timeline_ops.mark(tls, tid, "attn_tp_put", True)

            def before_slice():
                timeline_ops.mark(tls, tid, "reduce_local_wait")
                acquired = _await(
                    flags,
                    layout["reduce"] + 2 * (bid % 160),
                    2,
                    epoch,
                    diag,
                    local,
                    tid,
                    WORKERS,
                    32,
                    status=status,
                    failure_detail=failure_detail,
                )
                timeline_ops.mark(tls, tid, "reduce_local_wait", True)
                timeline_ops.mark(tls, tid, "attn_tp_sum")
                # As in gfx950 stage_reduce -> stage_slice, the seam CTA
                # owns its 32 columns' rank sum. WO CTAs only publish partials.
                meta = _typed(argument("metadata"), fx.Int32)
                rank, layer = fx.ptr_load(meta), fx.ptr_load(meta + 1)
                bases = peer_bases(fx.Int64(fx.ptrtoint(argument("peers"))), tp)
                own = bases[0]
                for p in range_constexpr(1, tp):
                    own = (rank == p).select(bases[p], own)
                ma = Mailbox(
                    layer + POST_TAG0,
                    fx.Int64(fx.ptrtoint(argument("peer_diag"))),
                    {"attn": 0, "ffn": 1}, steady=True, failure=diag,
                )
                if tid < 16:
                    for row in range_constexpr(rows):
                        at = row * (HIDDEN // 2) + (bid % 160) * 16 + tid
                        lo, hi = sum_partials(
                            ma.poll, preg(own, poff["attn"], "attn"),
                            lambda src: src * rows * (HIDDEN // 2) + at, tp,
                        )
                        fx.ptr_store(
                            bf16_pair(lo, hi).bitcast(fx.Int32),
                            _typed(attention, fx.Int32) + at,
                        )
                fx.gpu.barrier()
                timeline_ops.mark(tls, tid, "attn_tp_sum", True)
                return acquired

            def after_combine(ready):
                timeline_ops.mark(tls, tid, "ffn_tp_sum")
                metadata = _typed(argument("metadata"), fx.Int32)
                rank, layer = fx.ptr_load(metadata), fx.ptr_load(metadata + 1)
                bases = peer_bases(fx.Int64(fx.ptrtoint(argument("peers"))), tp)
                mf = Mailbox(
                    layer + MOE_TAG0,
                    fx.Int64(fx.ptrtoint(argument("peer_diag"))),
                    {"attn": 0, "ffn": 1},
                    steady=True,
                    failure=diag,
                )
                publish_reduce(
                    out,
                    out,
                    poff["ffn"],
                    mf.put_words,
                    mf.poll,
                    bases,
                    rank,
                    tid,
                    bid,
                    rows,
                    tp,
                    ready,
                )
                timeline_ops.mark(tls, tid, "ffn_tp_sum", True)

            attention = argument("attention")
            if index_bound_max:
                timeline_ops.mark(tls, tid, "index")
                run_index(argument, {
                    "rows": rows, "tid": tid, "bid": bid, "flags": flags,
                    "diag": diag, "epoch": epoch, "local": local,
                    "status": status, "arena": arena, "pool_rows": pool_rows,
                    "failure_detail": failure_detail,
                    "ring_rows": ring_rows, "window_size": extend_capacity // rows,
                }, index_bound_max, layout["index"])
                timeline_ops.mark(tls, tid, "index", True)

            def before_score(row):
                return _await(
                    flags, layout["index"] + rows * ((index_bound_max + 127) // 128) + row,
                    1, epoch, diag, local, tid, WORKERS, 39, status=status,
                    failure_detail=failure_detail,
                )

            def attention_arg(name):
                return argument(
                    {
                        "query": "query",
                        "pool": "pool",
                        "ring": "ring",
                        "prefix": "prefix",
                        "prefix_indptr": "prefix_indptr",
                        "extend": "extend",
                        "extend_indptr": "extend_indptr",
                        "sink": "sink",
                        "positions": "positions",
                        "cos": "cos",
                        "sin": "sin",
                        "woa": "woa",
                        "woa_s": "woa_s",
                        "wob": "wob",
                        "wob_s": "wob_s",
                        "out": "attention",
                        "maxima": "a_maxima",
                        "sums": "a_sums",
                        "probabilities": "a_probabilities",
                        "attention_q": "a_attention_q",
                        "attention_s": "a_attention_s",
                        "mid": "a_mid",
                        "mid_q": "a_mid_q",
                        "mid_s": "a_mid_s",
                        "debug_out": "query",
                    }[name]
                )

            run_attention_projection(
                attention_arg,
                {
                    "rows": rows,
                    "heads": heads,
                    "table_rows": table_rows,
                    "pool_rows": pool_rows,
                    "ring_rows": ring_rows,
                    "prefix_capacity": prefix_capacity,
                    "extend_capacity": extend_capacity,
                    "softmax_scale": softmax_scale,
                    "score_tasks": score_tasks,
                    "pv_tasks": pv_tasks,
                    "arena": arena,
                    "local": local,
                    "lds": lds,
                    "status": status,
                    "flags": flags,
                    "diag": diag,
                    "epoch": epoch,
                    "tid": tid,
                    "bid": bid,
                    "debug": False,
                    "placed_attention": C1_ATTENTION_PLACEMENT,
                    "progress": _i32_pointer(argument("score_progress")) if SYNC_DEBUG else None,
                    "failure_detail": failure_detail,
                    "timeline": tls,
                },
                after_projection=after_projection,
                before_score=before_score if index_bound_max else None,
            )
            out = argument("out")

            def ffn_arg(name):
                return argument(
                    {
                        "residual": "residual",
                        "attention": "attention",
                        "pre": "pre",
                        "post": "post",
                        "comb": "comb",
                        "fn": "fn",
                        "scale": "scale",
                        "bias": "bias",
                        "nw": "nw",
                        "gate_w": "gate_w",
                        "gate_bias": "gate_bias",
                        "w13": "w13",
                        "w13_s": "w13_s",
                        "w2": "w2",
                        "w2_s": "w2_s",
                        "sgu": "sgu",
                        "sgu_s": "sgu_s",
                        "sw2": "sw2",
                        "sw2_s": "sw2_s",
                        "residual_out": "residual_out",
                        "pre_out": "pre_out",
                        "post_out": "post_out",
                        "comb_out": "comb_out",
                        "normed": "normed",
                        "out": "out",
                        "collapse": "f_collapse",
                        "partial": "f_partial",
                        "shared_q": "f_shared_q",
                        "shared_s": "f_shared_s",
                        "routed_q": "f_routed_q",
                        "routed_s": "f_routed_s",
                        "logits": "f_logits",
                        "route_ids": "f_route_ids",
                        "route_weights": "f_route_weights",
                        "union_ids": "f_union_ids",
                        "union_count": "f_union_count",
                        "pick_slots": "f_pick_slots",
                        "shared_gup": "f_shared_gup",
                        "shared_mid": "f_shared_mid",
                        "shared_mid_q": "f_shared_mid_q",
                        "shared_mid_s": "f_shared_mid_s",
                        "shared_out": "f_shared_out",
                        "routed_gup": "f_routed_gup",
                        "routed_mid": "f_routed_mid",
                        "routed_mid_q": "f_routed_mid_q",
                        "routed_mid_s": "f_routed_mid_s",
                        "contributions": "f_contributions",
                    }[name]
                )

            run_ffn(
                ffn_arg,
                {
                    "rows": rows,
                    "tp": tp,
                    "interleaved": interleaved,
                    "real": real,
                    "padded": padded,
                    "arena": arena,
                    "local": local,
                    "lds": lds,
                    "status": status,
                    "diag": diag,
                    "epoch": epoch,
                    "tid": tid,
                    "bid": bid,
                    "max_union": rows * TOPK,
                    "ug_parts": padded // 32,
                    "flags_off": foff,
                    "flags": fflags,
                    "progress": _i32_pointer(argument("ffn_progress")) if SYNC_DEBUG else None,
                    "failure_detail": failure_detail,
                    "timeline": tls,
                },
                before_slice=before_slice,
                after_combine=after_combine,
            )
            if timeline_ops.ENABLED:
                timeline_ops.flush(tls, _typed(argument("timeline"), fx.Int64), tid, bid)

        kernel(arguments).launch(grid=(WORKERS, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


_MERGED_KERNELS = {}


class PreparedK2:
    """Validated pointer table and module; no parameter work at dispatch."""

    def __init__(self, runtime, table, tensors, output):
        self.runtime, self.table = runtime, table
        self.tensors, self.output = tuple(tensors), output

    def __call__(self):
        self.runtime((self.table,), torch.cuda.current_stream().cuda_stream)
        return self.output


def layer_post_gfx1250(
    a,
    f,
    scratch,
    peers,
    metadata,
    step_state,
    codes,
    sticky,
    *,
    tp,
    softmax_scale,
    interleaved=True,
    argument_tables,
    _prepare_only=False,
    index=None,
    index_bound_max=0,
    timeline=None,
):
    """Merged K2. Arguments use the standalone attention/FFN ABI names.

    Native parameter preparation and layout validation happen before binding;
    all mutable buffers are one ordered stream's, cleared by StepMailboxes.
    ``metadata`` is [rank, layer], with gfx950's POST/MOE tag offsets. It must
    stay fixed for the lifetime of captured graphs. Runtime use remains opt-in
    until GPU numerical, progress, residency and TP replay validation passes.
    """
    import math

    rows, heads, dim = a["query"].shape
    if index_bound_max:
        if not 0 < index_bound_max <= 8192 or index_bound_max % 128 or index is None:
            raise ValueError("[gfx1250 merged K2] invalid in-kernel index build")
        for name, shape, dtype in (
            ("i_query", (rows, 32, 128), torch.float8_e4m3fn),
            ("i_weights", (rows, 32), torch.float32),
            ("i_visible", (rows,), torch.int32),
            ("i_keymeta", (rows, 4), torch.int32),
            ("i_config", (11,), torch.int32),
            ("i_selection", (rows, 512), torch.int32),
            ("i_candidates", (rows, 2048), torch.int32),
        ):
            _tensor_spec(index[name], shape, dtype)
        if index["i_table"].dtype != torch.int32 or not index["i_table"].is_contiguous():
            raise ValueError("[gfx1250 merged K2] page table must be contiguous int32")
    specs = merged_scratch_specs(rows, tp, index_bound_max)
    if heads != 64 // tp or dim != 512:
        raise ValueError("[gfx1250 merged K2] query must match TP width")
    if (
        max(
            a["pool"].shape[0],
            a["ring"].shape[0],
            a["prefix"].numel(),
            a["extend"].numel(),
        )
        >= 2**31
    ):
        raise ValueError("[gfx1250 merged K2] row IDs/capacities must fit int32")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("[gfx1250 merged K2] invalid attention scale")
    table_rows = a["cos"].shape[0]
    if table_rows <= 0:
        raise ValueError("[gfx1250 merged K2] empty RoPE table")
    for n, spec in specs.items():
        _tensor_spec(scratch[n], *spec)
    _tensor_spec(peers.addresses, (tp,), torch.int64)
    if peers.bytes.nbytes < peer_bytes(rows, tp):
        raise ValueError("[gfx1250 merged K2] insufficient peer data capacity")
    _tensor_spec(metadata, (2,), torch.int32)
    _tensor_spec(step_state, (STATE_WORDS,), torch.int64)
    _tensor_spec(codes, (41,), torch.int32)
    _tensor_spec(sticky, (1,), torch.int32)
    a_specs = attention_scratch_specs(rows, heads)
    a_scratch = {
        n: scratch["a_" + n] for n in a_specs if n not in ("state", "debug_out")
    }
    a_scratch["state"] = scratch["state"][:ATTENTION_STATE_SIZE]
    a_scratch["debug_out"] = a["query"].reshape(-1)[:0]
    attention_projection_gfx1250(
        **a, scratch=a_scratch, softmax_scale=softmax_scale, _validate_only=True
    )
    f_specs = ffn_scratch_specs(rows, tp)
    f_scratch = {n: scratch["f_" + n] for n in f_specs if n != "state"}
    fbase = merged_state_layout(rows, tp)["ffn"]
    f_scratch["state"] = scratch["state"][fbase : fbase + f_specs["state"][0][0]]
    ffn_gfx1250(
        **f, scratch=f_scratch, tp=tp, interleaved=interleaved, _validate_only=True
    )
    # Standalone validation reuses the same shape/scale contract without
    # dispatch: assemble the complete merged ABI by name, then check aliases.
    values = {
        "query": a["query"],
        "pool": a["pool"],
        "ring": a["ring"],
        "prefix": a["prefix"],
        "prefix_indptr": a["prefix_indptr"],
        "extend": a["extend"],
        "extend_indptr": a["extend_indptr"],
        "sink": a["sink"],
        "positions": a["positions"],
        "cos": a["cos"],
        "sin": a["sin"],
        "woa": a["woa"],
        "woa_s": a["woa_s"],
        "wob": a["wob"],
        "wob_s": a["wob_s"],
        "residual": f["residual"],
        "pre": f["pre"],
        "post": f["post"],
        "comb": f["comb"],
        "fn": f["fn"],
        "scale": f["scale"],
        "bias": f["bias"],
        "nw": f["nw"],
        "gate_w": f["gate_w"],
        "gate_bias": f["gate_bias"],
        "w13": f["w13"],
        "w13_s": f["w13_s"],
        "w2": f["w2"],
        "w2_s": f["w2_s"],
        "sgu": f["sgu"],
        "sgu_s": f["sgu_s"],
        "sw2": f["sw2"],
        "sw2_s": f["sw2_s"],
        "peers": peers.addresses,
        "metadata": metadata,
        "step_state": step_state,
        "codes": codes,
        "sticky": sticky,
        **({n: a["query"] for n in (
            "i_query", "i_weights", "i_plane", "i_visible", "i_table",
            "i_keymeta", "i_config", "i_selection", "i_candidates",
        )} if index is None else index),
        "attention": a["out"],
        "residual_out": f["residual_out"],
        "pre_out": f["pre_out"],
        "post_out": f["post_out"],
        "comb_out": f["comb_out"],
        "normed": f["normed"],
        "out": f["out"],
        **scratch,
    }
    if timeline is not None:
        _tensor_spec(timeline, *specs["timeline"])
        values["timeline"] = timeline
    key = (
        rows,
        tp,
        table_rows,
        a["pool"].shape[0],
        a["ring"].shape[0],
        a["prefix"].numel(),
        a["extend"].numel(),
        float(softmax_scale),
        bool(interleaved),
        index_bound_max,
    )
    tensors = ARGUMENTS.pack(values)
    table = _bind_argument_table(tensors, argument_tables)
    if _prepare_only:
        return PreparedK2(prepare_merged_kernel(key, table), table, tensors, f["out"])
    _dispatch(
        _merged_launcher(*key),
        merged_name(*key),
        [table],
        1,
        rows,
        WORKERS,
        merged_lds_bytes(rows, index_bound_max),
        _MERGED_KERNELS,
        key,
    )
    return f["out"]


def _bind_argument_table(tensors, argument_tables):
    device = tensors[0].device
    for index in range(READONLY, len(tensors)):
        item = tensors[index]
        if item.numel() == 0:
            continue
        lo, hi = item.data_ptr(), item.data_ptr() + item.nbytes
        if any(
            lo < other.data_ptr() + other.nbytes and other.data_ptr() < hi
            for other in tensors[:index]
            if other.numel()
        ):
            raise ValueError("[gfx1250 merged K2] written buffers must not overlap")
    if not tensors[0].is_cuda or any(t.device != device for t in tensors):
        raise ValueError("[gfx1250 merged K2] all buffers must share one GPU")
    address_key = (device.index, *[t.data_ptr() for t in tensors])
    if address_key not in argument_tables:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 merged K2] bind address table before capture")
        table = torch.tensor(
            [t.data_ptr() for t in tensors], dtype=torch.int64, device=device
        )
        argument_tables[address_key] = (table, tuple(tensors))
    table = argument_tables[address_key][0]
    return table


def prepare_merged_kernel(key, table):
    """Compile/load and query cooperative residency; never submit a kernel."""
    device = table.device
    cache_key = (device.index, *key)
    if cache_key not in _MERGED_KERNELS:
        rows = key[0]
        stream = torch.cuda.current_stream(device).cuda_stream
        _MERGED_KERNELS[cache_key] = _CooperativeKernel(
            _merged_launcher(*key),
            (ptr_arg(table), stream),
            merged_name(*key),
            rows,
            WORKERS,
            device.index,
            smem=merged_lds_bytes(rows, key[9] if len(key) > 9 else 0),
            label="gfx1250 merged K2",
        )
    return _MERGED_KERNELS[cache_key]
