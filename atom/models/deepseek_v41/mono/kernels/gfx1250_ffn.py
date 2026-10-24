# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Local K2b: delayed attention seam -> norm/router/shared/routed -> TP partial.

The attention TP reduction must finish before this cooperative launch.
FFN TP reduction follows it, then its pending output/gates feed the next K1.
These external reduction boundaries are real; this is not yet complete K2.
"""

import os
from functools import lru_cache
from atom.models.deepseek_v41.mono.kernels.gfx1250_timeline import (
    mark as timeline_mark, union_count as timeline_union_count,
)

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels.tensor_shim import check_e8m0
from flydsl.expr import const_expr, range_constexpr

from atom.models.deepseek_v41.mono.kernels.gfx1250_k1 import (
    _await,
    _i32_pointer,
    _publish,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_layer_post import (
    _dispatch,
    _tensor_spec,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_mhc import (
    _typed,
    gate_device,
    slice_device,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_moe import (
    EXPERTS,
    GU_PROJECTIONS,
    GU_WAVE_GROUPS,
    DOWN_PARALLEL,
    HIDDEN,
    TOPK,
    combine_device,
    compact_union_c1,
    moe_widths,
    project_device,
    quantize_row_device,
    route_device,
    routed_down_device,
    routed_down_parallel_device,
    routed_up_device,
    router_device,
    shared_up_device,
    union_device,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_producers import hidden_norm_device
from atom.models.deepseek_v41.mono.sources import SOURCES
from atom.mono.device.ops import traced

SEAM_OWNER = os.environ.get("ATOM_DSV41_GFX1250_SEAM_OWNER", "0") == "1"

TRANSITIVE_FFN_WAITS = os.environ.get("ATOM_DSV41_GFX1250_FFN_TRANSITIVE_WAITS", "0") == "1"

C1_ROUTE_UNION = os.environ.get("ATOM_DSV41_GFX1250_C1_ROUTE_UNION", "0") == "1"

WORKERS, LDS_BYTES = HIDDEN // 16, 4096
FFN_PROGRESS_WORDS = 11  # epoch/stage/publication, then eight producer-wait waves
ROUTER0, ROUTE0, UNION0, SHARED0 = 164, 188, 190, 192


@traced
def _ffn_progress(progress, tid, bid, epoch, stage, published=False):
    """Optional last-observed producer boundary for a bounded wait failure.

    The merged debug build owns a separate per-CTA progress row. These words
    are diagnostics, never ready flags; eight wave words track producer waits.
    No additional workgroup join or payload workspace is introduced.
    """
    if const_expr(progress is not None):
        if tid == 0:
            fx.generic_store(progress + bid * FFN_PROGRESS_WORDS, epoch,
                memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent")
            fx.generic_store(progress + bid * FFN_PROGRESS_WORDS + 1, fx.Int32(stage),
                memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent")
            if const_expr(stage == 100):
                fx.generic_store(progress + bid * FFN_PROGRESS_WORDS + 2, fx.Int32(0),
                    memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent")
            if const_expr(published):
                fx.generic_store(progress + bid * FFN_PROGRESS_WORDS + 2, fx.Int32(stage),
                    memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent")


def ffn_state_layout(rows, tp):
    real, padded = moe_widths(tp)
    if rows not in (1, 2):
        raise ValueError("[gfx1250 K2b] requires rows1/2")
    counts = {
        "slice": 160,
        "norm": rows,
        "router": EXPERTS // 16,
        "route": rows,
        "union": 1,
        "shared": real // 32,
        "gates": rows,
        "ug": rows * TOPK * (padded // 32),
        "completion": WORKERS,
        "diagnostic": 1,
    }
    offset, layout = 0, {}
    for name, count in counts.items():
        layout[name] = (offset, count)
        offset += count
    return layout


def ffn_scratch_specs(rows, tp):
    real, padded = moe_widths(tp)
    state = ffn_state_layout(rows, tp)
    union = rows * TOPK
    return {
        "collapse": ((rows, HIDDEN), torch.bfloat16),
        "partial": ((rows, 160, 26), torch.float32),
        "shared_q": ((rows, HIDDEN), torch.float8_e4m3fn),
        "shared_s": ((rows, HIDDEN // 32), torch.uint8),
        "routed_q": ((rows, HIDDEN), torch.float8_e4m3fn),
        "routed_s": ((rows, HIDDEN // 32), torch.uint8),
        "logits": ((rows, EXPERTS), torch.bfloat16),
        "route_ids": ((rows, TOPK), torch.int32),
        "route_weights": ((rows, TOPK), torch.float32),
        "union_ids": ((union,), torch.int32),
        "union_count": ((1,), torch.int32),
        "pick_slots": ((rows, TOPK), torch.int32),
        "shared_gup": ((rows, 2 * real), torch.bfloat16),
        "shared_mid": ((rows, real), torch.bfloat16),
        "shared_mid_q": ((rows, real), torch.float8_e4m3fn),
        "shared_mid_s": ((rows, real // 32), torch.uint8),
        "shared_out": ((rows, HIDDEN), torch.bfloat16),
        "routed_gup": ((union, rows, 2 * padded), torch.bfloat16),
        "routed_mid": ((union, rows, padded), torch.bfloat16),
        "routed_mid_q": ((union, rows, padded), torch.float8_e4m3fn),
        "routed_mid_s": ((union, rows, padded // 32), torch.uint8),
        "contributions": ((rows, TOPK, HIDDEN), torch.bfloat16),
        "state": ((state["diagnostic"][0] + 1,), torch.int32),
    }


def _ffn_name(rows, tp, interleaved):
    return f"atom_gfx1250_k2_ffn_r{rows}_tp{tp}_itlv{int(interleaved)}_src{SOURCES}_gu{GU_PROJECTIONS}_dp{DOWN_PARALLEL}_wg{int(GU_WAVE_GROUPS)}_tw{int(TRANSITIVE_FFN_WAITS)}_ru{int(C1_ROUTE_UNION)}"


@lru_cache(maxsize=8)
def _ffn_launcher(rows, tp, interleaved=True):
    real, padded = moe_widths(tp)
    name = _ffn_name(rows, tp, interleaved)
    offsets = ffn_state_layout(rows, tp)
    flags_off = {key: off for key, (off, _) in offsets.items()}
    max_union, ug_parts = rows * TOPK, padded // 32

    @flyc.jit
    def launch(
        residual: fx.Pointer,
        attention: fx.Pointer,
        pre: fx.Pointer,
        post: fx.Pointer,
        comb: fx.Pointer,
        fn: fx.Pointer,
        scale: fx.Pointer,
        bias: fx.Pointer,
        nw: fx.Pointer,
        gate_w: fx.Pointer,
        gate_bias: fx.Pointer,
        w13: fx.Pointer,
        w13_s: fx.Pointer,
        w2: fx.Pointer,
        w2_s: fx.Pointer,
        sgu: fx.Pointer,
        sgu_s: fx.Pointer,
        sw2: fx.Pointer,
        sw2_s: fx.Pointer,
        residual_out: fx.Pointer,
        pre_out: fx.Pointer,
        post_out: fx.Pointer,
        comb_out: fx.Pointer,
        normed: fx.Pointer,
        out: fx.Pointer,
        collapse: fx.Pointer,
        partial: fx.Pointer,
        shared_q: fx.Pointer,
        shared_s: fx.Pointer,
        routed_q: fx.Pointer,
        routed_s: fx.Pointer,
        logits: fx.Pointer,
        route_ids: fx.Pointer,
        route_weights: fx.Pointer,
        union_ids: fx.Pointer,
        union_count: fx.Pointer,
        pick_slots: fx.Pointer,
        shared_gup: fx.Pointer,
        shared_mid: fx.Pointer,
        shared_mid_q: fx.Pointer,
        shared_mid_s: fx.Pointer,
        shared_out: fx.Pointer,
        routed_gup: fx.Pointer,
        routed_mid: fx.Pointer,
        routed_mid_q: fx.Pointer,
        routed_mid_s: fx.Pointer,
        contributions: fx.Pointer,
        state: fx.Pointer,
        stream: fx.Stream,
    ):
        @flyc.kernel(name=name, known_block_size=[256, 1, 1])
        def kernel(
            residual: fx.Pointer,
            attention: fx.Pointer,
            pre: fx.Pointer,
            post: fx.Pointer,
            comb: fx.Pointer,
            fn: fx.Pointer,
            scale: fx.Pointer,
            bias: fx.Pointer,
            nw: fx.Pointer,
            gate_w: fx.Pointer,
            gate_bias: fx.Pointer,
            w13: fx.Pointer,
            w13_s: fx.Pointer,
            w2: fx.Pointer,
            w2_s: fx.Pointer,
            sgu: fx.Pointer,
            sgu_s: fx.Pointer,
            sw2: fx.Pointer,
            sw2_s: fx.Pointer,
            residual_out: fx.Pointer,
            pre_out: fx.Pointer,
            post_out: fx.Pointer,
            comb_out: fx.Pointer,
            normed: fx.Pointer,
            out: fx.Pointer,
            collapse: fx.Pointer,
            partial: fx.Pointer,
            shared_q: fx.Pointer,
            shared_s: fx.Pointer,
            routed_q: fx.Pointer,
            routed_s: fx.Pointer,
            logits: fx.Pointer,
            route_ids: fx.Pointer,
            route_weights: fx.Pointer,
            union_ids: fx.Pointer,
            union_count: fx.Pointer,
            pick_slots: fx.Pointer,
            shared_gup: fx.Pointer,
            shared_mid: fx.Pointer,
            shared_mid_q: fx.Pointer,
            shared_mid_s: fx.Pointer,
            shared_out: fx.Pointer,
            routed_gup: fx.Pointer,
            routed_mid: fx.Pointer,
            routed_mid_q: fx.Pointer,
            routed_mid_s: fx.Pointer,
            contributions: fx.Pointer,
            state: fx.Pointer,
        ):
            tid, bid = fx.Int32(fx.thread_idx.x), fx.Int32(fx.block_idx.x)
            arena = fx.SharedAllocator(static=False)
            arena.allocate(LDS_BYTES + 32)
            local = _i32_pointer(arena.base_ptr)
            lds = fx.Int64(fx.ptrtoint(arena.base_ptr))
            status = local + LDS_BYTES // 4
            flags = _i32_pointer(state)
            diag = flags + flags_off["diagnostic"]
            if tid == 0:
                previous = fx.generic_load(
                    flags + flags_off["completion"] + bid,
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="agent",
                )
                fx.ptr_store(previous + 1, local)
            fx.gpu.barrier()
            epoch = fx.ptr_load(local)
            fx.gpu.barrier()

            run_ffn(
                {
                    "residual": residual,
                    "attention": attention,
                    "pre": pre,
                    "post": post,
                    "comb": comb,
                    "fn": fn,
                    "scale": scale,
                    "bias": bias,
                    "nw": nw,
                    "gate_w": gate_w,
                    "gate_bias": gate_bias,
                    "w13": w13,
                    "w13_s": w13_s,
                    "w2": w2,
                    "w2_s": w2_s,
                    "sgu": sgu,
                    "sgu_s": sgu_s,
                    "sw2": sw2,
                    "sw2_s": sw2_s,
                    "residual_out": residual_out,
                    "pre_out": pre_out,
                    "post_out": post_out,
                    "comb_out": comb_out,
                    "normed": normed,
                    "out": out,
                    "collapse": collapse,
                    "partial": partial,
                    "shared_q": shared_q,
                    "shared_s": shared_s,
                    "routed_q": routed_q,
                    "routed_s": routed_s,
                    "logits": logits,
                    "route_ids": route_ids,
                    "route_weights": route_weights,
                    "union_ids": union_ids,
                    "union_count": union_count,
                    "pick_slots": pick_slots,
                    "shared_gup": shared_gup,
                    "shared_mid": shared_mid,
                    "shared_mid_q": shared_mid_q,
                    "shared_mid_s": shared_mid_s,
                    "shared_out": shared_out,
                    "routed_gup": routed_gup,
                    "routed_mid": routed_mid,
                    "routed_mid_q": routed_mid_q,
                    "routed_mid_s": routed_mid_s,
                    "contributions": contributions,
                }.__getitem__,
                {
                    "rows": rows,
                    "tp": tp,
                    "interleaved": interleaved,
                    "real": real,
                    "padded": padded,
                    "max_union": max_union,
                    "ug_parts": ug_parts,
                    "flags_off": flags_off,
                    "arena": arena,
                    "local": local,
                    "lds": lds,
                    "status": status,
                    "flags": flags,
                    "diag": diag,
                    "epoch": epoch,
                    "tid": tid,
                    "bid": bid,
                },
            )

        kernel(
            residual,
            attention,
            pre,
            post,
            comb,
            fn,
            scale,
            bias,
            nw,
            gate_w,
            gate_bias,
            w13,
            w13_s,
            w2,
            w2_s,
            sgu,
            sgu_s,
            sw2,
            sw2_s,
            residual_out,
            pre_out,
            post_out,
            comb_out,
            normed,
            out,
            collapse,
            partial,
            shared_q,
            shared_s,
            routed_q,
            routed_s,
            logits,
            route_ids,
            route_weights,
            union_ids,
            union_count,
            pick_slots,
            shared_gup,
            shared_mid,
            shared_mid_q,
            shared_mid_s,
            shared_out,
            routed_gup,
            routed_mid,
            routed_mid_q,
            routed_mid_s,
            contributions,
            state,
        ).launch(grid=(WORKERS, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


_FFN_KERNELS = {}


def ffn_gfx1250(
    residual,
    attention,
    pre,
    post,
    comb,
    fn,
    scale,
    bias,
    nw,
    gate_w,
    gate_bias,
    w13,
    w13_s,
    w2,
    w2_s,
    sgu,
    sgu_s,
    sw2,
    sw2_s,
    residual_out,
    pre_out,
    post_out,
    comb_out,
    normed,
    out,
    scratch,
    *,
    tp,
    interleaved=True,
    _validate_only=False,
):
    """Single local K2b launch, returning the FFN rank partial and delayed state.

    Scratch/state is exclusive to one stream and initially zero. Routed
    weights must use gfx1250's native row order and n32k4 scale shuffle.
    Caller checks routing text mode, finite inputs and loaded zero padding;
    image routing, shared-expert fusion and alternative scale layouts refuse
    at binding time. Every nonzero diagnostic invalidates outputs and reuse.
    """
    rows = residual.shape[0] if residual.ndim == 3 else 0
    real, padded = moe_widths(tp)
    specs = ffn_scratch_specs(rows, tp)
    for tensor, shape, dtype in [
        (residual, (rows, 4, HIDDEN), torch.bfloat16),
        (attention, (rows, HIDDEN), torch.bfloat16),
        (pre, (rows, 4), torch.float32),
        (post, (rows, 4), torch.float32),
        (comb, (rows, 4, 4), torch.float32),
        (fn, (24, 4 * HIDDEN), torch.float32),
        (scale, (3,), torch.float32),
        (bias, (24,), torch.float32),
        (nw, (HIDDEN,), torch.bfloat16),
        (gate_w, (EXPERTS, HIDDEN), torch.bfloat16),
        (gate_bias, (EXPERTS,), torch.float32),
        (w13, (EXPERTS, 2 * padded, HIDDEN // 2), torch.float4_e2m1fn_x2),
        (w2, (EXPERTS, HIDDEN, padded // 2), torch.float4_e2m1fn_x2),
        (sgu, (2 * real, HIDDEN), torch.float8_e4m3fn),
        (sw2, (HIDDEN, real), torch.float8_e4m3fn),
        (residual_out, (rows, 4, HIDDEN), torch.bfloat16),
        (pre_out, (rows, 4), torch.float32),
        (post_out, (rows, 4), torch.float32),
        (comb_out, (rows, 4, 4), torch.float32),
        (normed, (rows, HIDDEN), torch.bfloat16),
        (out, (rows, HIDDEN), torch.bfloat16),
        *[
            (scratch[name], *spec)
            for name, spec in specs.items()
            if not name.endswith("_s")
        ],
    ]:
        _tensor_spec(tensor, shape, dtype)
    for tensor, shape, name in [
        (w13_s, (EXPERTS, 2 * padded // 32, HIDDEN), "w13_s"),
        (w2_s, (EXPERTS, HIDDEN // 32, padded), "w2_s"),
        (sgu_s, (2 * real // 32, HIDDEN // 32), "sgu_s"),
        (sw2_s, (HIDDEN // 32, real // 32), "sw2_s"),
        *[
            (scratch[name], spec[0], name)
            for name, spec in specs.items()
            if name.endswith("_s")
        ],
    ]:
        check_e8m0(tensor, name, "gfx1250 K2b", shape)
    tensors = [
        residual,
        attention,
        pre,
        post,
        comb,
        fn,
        scale,
        bias,
        nw,
        gate_w,
        gate_bias,
        w13,
        w13_s,
        w2,
        w2_s,
        sgu,
        sgu_s,
        sw2,
        sw2_s,
        residual_out,
        pre_out,
        post_out,
        comb_out,
        normed,
        out,
    ] + [scratch[name] for name in specs]
    key = (rows, tp, bool(interleaved))
    if _validate_only:
        return key, tensors
    _dispatch(
        _ffn_launcher(*key),
        _ffn_name(*key),
        tensors,
        19,
        rows,
        WORKERS,
        LDS_BYTES + 32,
        _FFN_KERNELS,
        key,
    )
    return out, residual_out, pre_out, post_out, comb_out


@traced
def run_ffn(a, c, before_slice=None, after_combine=None):
    rows = c["rows"]
    interleaved = c["interleaved"]
    real = c["real"]
    padded = c["padded"]
    max_union = c["max_union"]
    ug_parts = c["ug_parts"]
    flags_off = c["flags_off"]
    arena = c["arena"]
    local = c["local"]
    lds = c["lds"]
    status = c["status"]
    flags = c["flags"]
    diag = c["diag"]
    epoch = c["epoch"]
    tid = c["tid"]
    bid = c["bid"]
    failure_detail = c.get("failure_detail")
    tls = c.get("timeline")
    progress = c.get("progress")

    def mark_progress(stage, published=False):
        _ffn_progress(progress, tid, bid, epoch, stage, published)

    def wait(name, count, phase, first=0, per_wave=False, active_count=None):
        stamp = {20:"norm_wait",21:"router_wait",22:"route_wait",23:"union_wait",
                 24:"shared_up_wait",25:"gates_wait",26:"ug_wait",27:"shared_down_wait",
                 28:"routed_down_wait",29:"combine"}[phase]
        timeline_mark(tls, tid, stamp)
        producer_bias = {
            "slice": 0,
            "norm": 0,
            "router": ROUTER0 - flags_off["router"],
            "route": ROUTE0 - flags_off["route"],
            "union": (ROUTE0 if C1_ROUTE_UNION and rows == 1 else UNION0) - flags_off["union"],
            "shared": SHARED0 - flags_off["shared"],
        }.get(name, 0)
        acquired = _await(
            flags,
            flags_off[name] + first,
            count,
            epoch,
            diag,
            local,
            tid,
            WORKERS,
            phase,
            status=status,
            failure_detail=failure_detail,
            # Translate the local flag slot to its actual producer CTA;
            # router/route/union slots do not share that CTA's numeric ID.
            progress=progress if name in ("slice", "norm", "router", "route", "union", "shared") else None,
            progress_stride=FFN_PROGRESS_WORDS,
            progress_bias=producer_bias,
            monitor_waves=(name == "slice" and count == 160) or phase in (21, 22, 23, 24, 25, 27),
            poll_threads=32 if per_wave else 256,
            active_count=active_count,
        )
        timeline_mark(tls, tid, stamp, True)
        return acquired

    mark_progress(100)
    ready = wait("slice", 0, 20)
    if const_expr(before_slice is not None):
        if (bid < 160) if const_expr(SEAM_OWNER) else fx.Boolean(True):
            # Only these CTAs consume the reduced attention slice. Other
            # FFN tasks acquire slice/norm publications below.
            mark_progress(110)
            ready = ready & before_slice()
            mark_progress(111)
    if (bid < 160) & ready:
        mark_progress(120)
        timeline_mark(tls, tid, "slice")
        slice_device(
            a("residual"),
            a("attention"),
            a("pre"),
            a("post"),
            a("comb"),
            a("fn"),
            a("residual_out"),
            a("collapse"),
            a("partial"),
            arena.base_ptr,
            tid,
            bid,
            rows,
            True,
        )
        _publish(flags, flags_off["slice"] + bid, epoch, synchronized=True, joined=tls is not None)
        mark_progress(121, True)
        timeline_mark(tls, tid, "slice", True)
    if (bid >= 160) & (bid < 160 + rows):
        mark_progress(200)
        ready = wait("slice", 160, 20)
        mark_progress(201)
        if ready:
            mark_progress(210)
            timeline_mark(tls, tid, "norm")
            hidden_norm_device(
                a("collapse"),
                a("nw"),
                a("normed"),
                a("shared_q"),
                a("shared_s"),
                lds,
                tid,
                bid - 160,
                rows,
                True,
                a("partial"),
            )
            mark_progress(211)
            mark_progress(220)
            quantize_row_device(
                a("normed"), a("routed_q"), a("routed_s"), tid, bid - 160, HIDDEN, 1e-10
            )
            mark_progress(221)
            _publish(flags, flags_off["norm"] + bid - 160, epoch, synchronized=True, joined=tls is not None)
            mark_progress(230, True)
            timeline_mark(tls, tid, "norm", True)
    if (bid >= ROUTER0) & (bid < ROUTER0 + EXPERTS // 16):
        mark_progress(300)
        ready = wait("norm", rows, 21)
        mark_progress(301)
        if ready:
            mark_progress(310)
            timeline_mark(tls, tid, "router")
            router_device(
                a("normed"),
                a("gate_w"),
                a("logits"),
                arena.base_ptr,
                tid,
                bid - ROUTER0,
                rows,
            )
            mark_progress(311)
            _publish(
                flags,
                flags_off["router"] + bid - ROUTER0,
                epoch,
                synchronized=True,
            )
            mark_progress(320, True)
            timeline_mark(tls, tid, "router", True)
    if (bid >= ROUTE0) & (bid < ROUTE0 + rows):
        mark_progress(400)
        ready = wait("router", EXPERTS // 16, 22)
        mark_progress(401)
        if ready:
            mark_progress(410)
            timeline_mark(tls, tid, "route")
            route_device(
                a("logits"),
                a("gate_bias"),
                a("route_ids"),
                a("route_weights"),
                tid,
                bid - ROUTE0,
            )
            if const_expr(C1_ROUTE_UNION and rows == 1):
                timeline_mark(tls, tid, "union")
                compact_union_c1(a("route_ids"), a("union_ids"), a("union_count"),
                    a("pick_slots"), diag, tid)
            mark_progress(411)
            _publish(
                flags,
                flags_off["route"] + bid - ROUTE0,
                epoch,
                synchronized=True,
            )
            mark_progress(420, True)
            if const_expr(C1_ROUTE_UNION and rows == 1):
                _publish(flags, flags_off["union"], epoch, synchronized=True,
                    joined=tls is not None)
                timeline_mark(tls, tid, "union", True)
            timeline_mark(tls, tid, "route", True)
    if const_expr(not C1_ROUTE_UNION or rows != 1):
        if bid == UNION0:
            mark_progress(500)
            ready = wait("route", rows, 23)
            mark_progress(501)
            if ready:
                mark_progress(510)
                timeline_mark(tls, tid, "union")
                union_device(
                    a("route_ids"),
                    a("union_ids"),
                    a("union_count"),
                    a("pick_slots"),
                    arena.base_ptr,
                    diag,
                    tid,
                    rows,
                )
                mark_progress(511)
                _publish(flags, flags_off["union"], epoch, synchronized=True, joined=tls is not None)
                mark_progress(520, True)
                timeline_mark(tls, tid, "union", True)
    if (bid >= SHARED0) & (bid < SHARED0 + real // 32):
        mark_progress(600)
        ready = wait("norm", rows, 24)
        mark_progress(601)
        if ready:
            mark_progress(610)
            timeline_mark(tls, tid, "shared_up")
            shared_up_device(
                a("shared_q"),
                a("shared_s"),
                a("sgu"),
                a("sgu_s"),
                a("shared_gup"),
                a("shared_mid"),
                a("shared_mid_q"),
                a("shared_mid_s"),
                arena.base_ptr,
                tid,
                bid - SHARED0,
                rows,
                real,
            )
            mark_progress(611)
            _publish(
                flags,
                flags_off["shared"] + bid - SHARED0,
                epoch,
                synchronized=True,
            )
            mark_progress(620, True)
            timeline_mark(tls, tid, "shared_up", True)
    # Gates are separate next-layer producers, after the shared stage.
    # Routed UG does not acquire these gates or wait for a grid barrier.
    if (bid >= 162) & (bid < 162 + rows):
        mark_progress(700)
        ready = wait("shared", real // 32, 25)
        mark_progress(701)
        if ready:
            mark_progress(710)
            timeline_mark(tls, tid, "gates")
            gate_device(
                a("partial"),
                a("scale"),
                a("bias"),
                a("pre_out"),
                a("post_out"),
                a("comb_out"),
                arena.base_ptr,
                tid,
                bid - 162,
                rows,
            )
            mark_progress(711)
            _publish(flags, flags_off["gates"] + bid - 162, epoch, synchronized=True, joined=tls is not None)
            mark_progress(720, True)
            timeline_mark(tls, tid, "gates", True)
    ready = wait("union", 1, 26)
    # norm -> every router -> route -> union -> this acquire is transitive.
    # All original producer barriers, releases, and retirement fences remain.
    if const_expr(not TRANSITIVE_FFN_WAITS):
        ready = ready & wait("norm", rows, 26)
    if ready:
        timeline_mark(tls, tid, "ug")
        n = fx.ptr_load(_typed(a("union_count"), fx.Int32))
        timeline_union_count(tls, tid, n)
        for round in range_constexpr((max_union * ug_parts + WORKERS - 1) // WORKERS):
            task = bid + round * WORKERS
            if (task < max_union * ug_parts) & (task // ug_parts < n):
                routed_up_device(
                    a("routed_q"),
                    a("routed_s"),
                    a("w13"),
                    a("w13_s"),
                    a("routed_gup"),
                    a("routed_mid"),
                    a("routed_mid_q"),
                    a("routed_mid_s"),
                    a("union_ids"),
                    arena.base_ptr,
                    tid,
                    task,
                    rows,
                    real,
                    padded,
                    interleaved,
                )
                _publish(flags, flags_off["ug"] + task, epoch, synchronized=True, joined=tls is not None)
        timeline_mark(tls, tid, "ug", True)
    if const_expr(TRANSITIVE_FFN_WAITS):
        # Carry the acquired union status across UG. Shared is still acquired
        # here, and every UG expert is still acquired before routed down.
        ready = ready & wait("shared", real // 32, 27)
    else:
        ready = wait("shared", real // 32, 27)
        ready = ready & wait("union", 1, 27)
    if ready:
        timeline_mark(tls, tid, "shared_down")
        project_device(
            a("shared_mid_q"),
            a("shared_mid_s"),
            a("sw2"),
            a("sw2_s"),
            a("shared_out"),
            arena.base_ptr,
            tid,
            bid,
            rows,
            real,
            HIDDEN,
        )
        timeline_mark(tls, tid, "shared_down", True)
        n = fx.ptr_load(_typed(a("union_count"), fx.Int32))
        if const_expr(DOWN_PARALLEL):
            def acquire_expert(slot, live):
                # Each wave polls every N32 group of its own expert. The
                # original CTA joins and one fail-closed result are retained.
                return wait("ug", ug_parts, 28, slot * ug_parts,
                    per_wave=True, active_count=live.select(fx.Int32(ug_parts), fx.Int32(0)))
            def mark_down(ending):
                timeline_mark(tls, tid, "routed_down", ending)
            routed_down_parallel_device(
                a("routed_mid_q"), a("routed_mid_s"), a("w2"), a("w2_s"),
                a("contributions"), a("union_ids"), a("pick_slots"), a("route_weights"),
                arena.base_ptr, tid, bid, n, rows, padded,
                before_wait=acquire_expert, prefetch=DOWN_PARALLEL == 2, mark_body=mark_down,
            )
        else:
            for slot in range(n):
                acquired = wait("ug", ug_parts, 28, slot * ug_parts)
                if acquired:
                    timeline_mark(tls, tid, "routed_down")
                    routed_down_device(
                        a("routed_mid_q"), a("routed_mid_s"), a("w2"), a("w2_s"),
                        a("contributions"), a("union_ids"), a("pick_slots"), a("route_weights"),
                        arena.base_ptr, tid, bid, slot, rows, padded,
                    )
                    timeline_mark(tls, tid, "routed_down", True)
        ready = wait("union", 0, 29)
        if ready:
            timeline_mark(tls, tid, "combine")
            combine_device(
                a("contributions"), a("shared_out"), a("out"), tid, bid, rows
            )
            timeline_mark(tls, tid, "combine", True)
    timeline_mark(tls, tid, "ffn_publish")
    _publish(flags, flags_off["completion"] + bid, epoch, joined=tls is not None)
    timeline_mark(tls, tid, "ffn_publish", True)
    if const_expr(after_combine is not None):
        after_combine(ready)
