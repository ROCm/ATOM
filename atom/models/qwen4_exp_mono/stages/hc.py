# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen hyper-connection seam emitted inside the persistent K2 kernel.

The stage preserves the BF16 boundaries of ``hc_rows`` while keeping the
whole dependency graph in one launch:

    deferred combine + grouped Gemma RMSNorm
    -> concatenated down/injection projection
    -> scaled SiLU
    -> up projection + gated stream mean

HC projections use BF16 MFMA with explicit reduction and rounding boundaries.
Independent weight loads overlap activation readiness waits in the accepted
schedules. Tagged device-scope mailboxes connect the resident CTA stages.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.layout import (
    HC_DOWN_COLS,
    HC_DOWN_TASKS,
    HC_DOWN_TASKS_PER_STREAM,
    HC_NOUT,
    HC_OUT_COLS,
    HC_OUT_TASKS,
)
from atom.models.qwen4_exp_mono.shapes import HC, HC_RANK, HIDDEN
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    butterfly,
    hw_exp2,
    hw_rsq,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import WAVES, first_task

LOG2E = 1.4426950408889634
EPS = 1e-6

# Reuse the same low-numbered resident CTAs across stages. This keeps the 84
# down CTAs busy until the gate hand-off and lets them continue directly into
# the first 84 output tiles instead of making a disjoint 160-CTA set spin early.
NORM0 = DOWN0 = GATE0 = OUT0 = 0
NORM_COLS_PER_PASS = 512
NORM_PASSES = HIDDEN // NORM_COLS_PER_PASS
NORM_EARLY_PASSES = 3
HC_UP_GROUPS = 2
HC_UP_GROUP_TILES = HC_OUT_TASKS // HC_UP_GROUPS
HC_UP_GROUP_COLS = HIDDEN // HC_UP_GROUPS
HC_UP_GROUP_K_PASSES = HC_UP_GROUP_COLS // (32 * WAVES)
UP_GROUP0 = HC_OUT_TASKS

assert HIDDEN % NORM_COLS_PER_PASS == 0
assert 0 < NORM_EARLY_PASSES < NORM_PASSES
assert HC_OUT_TASKS % HC_UP_GROUPS == 0
assert HIDDEN % HC_UP_GROUPS == 0


def _ld_bf(ptr, i, cache_modifier=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(
                rsrc(ptr),
                i,
                vec_width=1,
                dtype=T.bf16,
                cache_modifier=cache_modifier,
            )
        )
    )


def _st_bf(ptr, i, value, cache_modifier=0):
    bo.buffer_store(
        fx.Float32(value).to(fx.BFloat16),
        rsrc(ptr),
        i,
        cache_modifier=cache_modifier,
    )


def _ld_f32(ptr, i, cache_modifier=0):
    return fx.Float32(
        bo.buffer_load(
            rsrc(ptr),
            i,
            vec_width=1,
            dtype=T.f32,
            cache_modifier=cache_modifier,
        )
    )


def _st_f32(ptr, i, value, cache_modifier=0):
    bo.buffer_store(fx.Float32(value), rsrc(ptr), i, cache_modifier=cache_modifier)


def _ld_bf8(ptr, bf16_index, cache_modifier=0):
    """Eight contiguous BF16 values in the operand order of gfx950 MFMA."""

    return fx.Vector(
        bo.buffer_load(
            rsrc(ptr),
            bf16_index // 2,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=cache_modifier,
        )
    ).bitcast(fx.BFloat16)


def _ld_lds_bf8(ptr, bf16_index):
    return fx.Vector(
        fx.ptr_load(ptr + bf16_index // 2, result_type=T.vec(4, T.i32))
    ).bitcast(fx.BFloat16)


def _fma(a, b, c):
    return fx.Float32(fmath.fma(a, b, c))


def _sigmoid(x):
    # Matches Triton's tl.sigmoid lowering used by hc_rows.
    return fx.Float32(1.0) / (
        fx.Float32(1.0) + hw_exp2(fx.Float32(x) * fx.Float32(-LOG2E))
    )


@traced
def combine_gdn_output_pair(c, tile, row0, block0, block1):
    """Inject a resident GDN output pair into the HC residual streams."""
    combine_gdn_output_pair_raw(
        c["args"]["streams"],
        c["args"]["attn_raw"],
        c["args"]["streams_out"],
        tile,
        row0,
        block0,
        block1,
    )


@traced
def combine_gdn_output_pair_raw(
    streams, attn_raw, streams_out, tile, row0, block0, block1
):
    """HC hand-off from a resident output pair using three seam pointers."""

    col0 = tile * HC_OUT_COLS + row0
    for stream in range_constexpr(HC):
        raw = _ld_bf(attn_raw, stream)
        scaled_raw = bf16_round(raw * fx.Float32(1.0 / HC))
        inject = fx.Float32(2.0) * _sigmoid(scaled_raw)
        x0 = _ld_bf(streams, stream * HIDDEN + col0)
        x1 = _ld_bf(streams, stream * HIDDEN + col0 + 1)
        _st_bf(
            streams_out,
            stream * HIDDEN + col0,
            bf16_round(_fma(block0, inject, x0)),
            CM_DEV,
        )
        _st_bf(
            streams_out,
            stream * HIDDEN + col0 + 1,
            bf16_round(_fma(block1, inject, x1)),
            CM_DEV,
        )


@traced
def _finish_norm(
    c, stream, ss, cache_modifier, combined_values=None, norm_weights=None
):
    """One RMS reduction and affine writeout after either HC input schedule."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    a, red = c["args"], c["red"]
    ss = butterfly(ss, (32, 16, 8, 4, 2, 1))
    if lane == 0:
        fx.ptr_store(ss, red + wave)
    gpu.barrier()
    total = fx.ptr_load(red + lane % 8)
    total = butterfly(total, (4, 2, 1))
    rstd = hw_rsq(_fma(total, fx.Float32(1.0 / HIDDEN), fx.Float32(EPS)))

    for k_pass in range_constexpr(NORM_PASSES):
        col = tid + 512 * k_pass
        combined = (
            combined_values[k_pass]
            if combined_values is not None
            else _ld_bf(a["streams_out"], stream * HIDDEN + col, cache_modifier)
        )
        gamma = (
            norm_weights[k_pass]
            if norm_weights is not None
            else _ld_bf(a["hc_norm"], stream * HIDDEN + col) + fx.Float32(1.0)
        )
        normed = bf16_round((combined * rstd) * gamma)
        _st_bf(c["normed"], stream * HIDDEN + col, normed, CM_DEV)
    publish(c["put"], c["norm_ready"], stream, 1, tid == 0)


@traced
def _stage_norm(c, stream):
    """One CTA combines and normalizes one of the four residual streams."""

    tid = c["tid"]
    a = c["args"]
    cached = c.get("norm_cache", False)
    values = [] if cached else None
    gamma = (
        [
            _ld_bf(a["hc_norm"], stream * HIDDEN + tid + 512 * k_pass) + fx.Float32(1.0)
            for k_pass in range(NORM_PASSES)
        ]
        if cached
        else None
    )
    attn_ready = c.get("attn_ready")
    if const_expr(attn_ready is not None):
        if tid == 0:
            c["poll"]([(attn_ready, 0, 1)])
        gpu.barrier()
    attn_cm = c.get("attn_cm", 0)
    raw = _ld_bf(a["attn_raw"], stream)
    scaled_raw = bf16_round(raw * fx.Float32(1.0 / HC))
    inject = fx.Float32(2.0) * _sigmoid(scaled_raw)

    ss = fx.Float32(0.0)
    for k_pass in range_constexpr(HIDDEN // 512):
        col = tid + 512 * k_pass
        x = _ld_bf(a["streams"], stream * HIDDEN + col)
        block = _ld_bf(a["attn_out"], col, attn_cm)
        combined = bf16_round(_fma(block, inject, x))
        if cached:
            values.append(combined)
        ss = _fma(combined, combined, ss)
        _st_bf(a["streams_out"], stream * HIDDEN + col, combined)

    _finish_norm(c, stream, ss, 0, values, gamma)


@traced
def _stage_precombined_norm(c, stream):
    """Overlap early HC squares with the final GDN output group."""

    tid = c["tid"]
    a = c["args"]
    cached = c.get("norm_cache", False)
    values = [] if cached else None
    gamma = (
        [
            _ld_bf(a["hc_norm"], stream * HIDDEN + tid + NORM_COLS_PER_PASS * k_pass)
            + fx.Float32(1.0)
            for k_pass in range(NORM_PASSES)
        ]
        if cached
        else None
    )
    if tid == 0:
        c["poll"]([(c["attn_group_ready"], 0, 1)])
    gpu.barrier()

    ss = fx.Float32(0.0)
    for k_pass in range_constexpr(NORM_EARLY_PASSES):
        col = tid + NORM_COLS_PER_PASS * k_pass
        combined = _ld_bf(a["streams_out"], stream * HIDDEN + col, CM_DEV)
        if cached:
            values.append(combined)
        ss = _fma(combined, combined, ss)

    if tid == 0:
        c["poll"]([(c["attn_group_ready"], 1, 1)])
    gpu.barrier()
    for k_pass in range_constexpr(NORM_EARLY_PASSES, NORM_PASSES):
        col = tid + NORM_COLS_PER_PASS * k_pass
        combined = _ld_bf(a["streams_out"], stream * HIDDEN + col, CM_DEV)
        if cached:
            values.append(combined)
        ss = _fma(combined, combined, ss)

    _finish_norm(c, stream, ss, CM_DEV, values, gamma)


@traced
def _stage_down(c, task):
    """One CTA computes an N16 tile of one stream's down/injection GEMV."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    stream = task // HC_DOWN_TASKS_PER_STREAM
    tile = task % HC_DOWN_TASKS_PER_STREAM
    row = fx.min(tile * HC_DOWN_COLS + lane % 16, HC_NOUT - 1)
    weights = (
        [
            _ld_bf8(
                c["args"]["hc_down_inject"],
                row * (HC * HIDDEN)
                + stream * HIDDEN
                + (wave + 8 * k_pass) * 32
                + 8 * (lane // 16),
                CM_NT,
            )
            for k_pass in range(5)
        ]
        if c.get("gdn_down_prefetch", False)
        else None
    )
    c["poll"]([(c["norm_ready"], stream, 1)])

    # MFMA's N16 mapping repeats a one-token A operand in sixteen columns.
    # Stage it once so those repeats are LDS reads instead of VMEM reads.
    if tid < HIDDEN // 8:
        at = tid * 8
        x = fx.Vector(
            bo.buffer_load(
                rsrc(c["normed"]),
                (stream * HIDDEN + at) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(x, c["x"] + at // 2)
    gpu.barrier()

    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    # K=2560: eighty K32 steps, ten per wave. Each operand load is 16 B.
    for k_pass in range_constexpr(HIDDEN // (32 * 8)):
        k0 = (wave + 8 * k_pass) * 32 + 8 * (lane // 16)
        w = (
            weights[k_pass]
            if weights is not None and k_pass < 5
            else _ld_bf8(
                c["args"]["hc_down_inject"],
                row * (HC * HIDDEN) + stream * HIDDEN + k0,
                CM_NT,
            )
        )
        x = _ld_lds_bf8(c["x"], k0)
        acc = mfma_bf16(w, x, acc)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < HC_DOWN_COLS:
        out_row = tile * HC_DOWN_COLS + tid
        if out_row < HC_NOUT:
            _st_f32(
                c["part"],
                stream * HC_NOUT + out_row,
                row_sum(c["red"], tid, 0),
                CM_DEV,
            )
    publish(c["put"], c["down_ready"], task, 1, tid == 0)


@traced
def poll_ready_tiles(c, ready, start, count):
    """Eight wave leaders cover disjoint tiles; a CTA barrier joins them."""
    if c.get("parallel_ready", False) or c.get("expert_owned", False):
        if c["lane"] == 0:
            for owner in range_constexpr(WAVES):
                if c["wave"] == owner:
                    c["poll"](
                        [
                            (ready, start + tile, 1)
                            for tile in range(owner, count, WAVES)
                        ]
                    )
        gpu.barrier()
    else:
        c["poll"]([(ready, start + tile, 1) for tile in range(count)])


@traced
def _stage_gate(c):
    """Sum stream partials, apply scaled SiLU, and expose injection logits."""

    tid = c["tid"]
    poll_ready_tiles(c, c["down_ready"], 0, HC_DOWN_TASKS)
    if tid < HC_RANK:
        value = _ld_f32(c["part"], tid, CM_DEV)
        for stream in range_constexpr(1, HC):
            value = value + _ld_f32(c["part"], stream * HC_NOUT + tid, CM_DEV)
        value = bf16_round(value)
        value = bf16_round(value * fx.Float32(1.0 / HC))
        gate = bf16_round(value * _sigmoid(value))
        _st_bf(c["gate"], tid, gate, CM_DEV)
    if tid < HC:
        raw = _ld_f32(c["part"], HC_RANK + tid, CM_DEV)
        for stream in range_constexpr(1, HC):
            raw = raw + _ld_f32(c["part"], stream * HC_NOUT + HC_RANK + tid, CM_DEV)
        _st_bf(c["args"]["pending_raw"], tid, bf16_round(raw))
    publish(c["put"], c["gate_ready"], 0, 1, tid == 0)


@traced
def _stage_up(c, task):
    """One CTA forms an N16 mixed tile from the four stream projections."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    k0 = wave * 32 + 8 * (lane // 16)
    weights = (
        [
            _ld_bf8(
                c["args"]["hc_up"],
                (stream * HIDDEN + task * HC_OUT_COLS + lane % 16) * HC_RANK + k0,
                CM_NT,
            )
            for stream in range(HC)
        ]
        if c.get("expert_owned", False) or c.get("gdn_up_prefetch", False)
        else None
    )
    c["poll"]([(c["gate_ready"], 0, 1)])

    if tid < HC_RANK // 8:
        at = tid * 8
        gate = fx.Vector(
            bo.buffer_load(
                rsrc(c["gate"]),
                at // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(gate, c["x"] + at // 2)
    gpu.barrier()

    # K=320: ten K32 steps. Waves 0/1 take a second step; the others take one.
    for stream in range_constexpr(HC):
        row = stream * HIDDEN + task * HC_OUT_COLS + lane % 16
        k0 = wave * 32 + 8 * (lane // 16)
        acc = mfma_bf16(
            (
                weights[stream]
                if weights is not None
                else _ld_bf8(c["args"]["hc_up"], row * HC_RANK + k0, CM_NT)
            ),
            _ld_lds_bf8(c["x"], k0),
            fx.Vector.filled(4, 0.0, fx.Float32),
        )
        if wave < 2:
            k1 = (8 + wave) * 32 + 8 * (lane // 16)
            acc = mfma_bf16(
                _ld_bf8(c["args"]["hc_up"], row * HC_RANK + k1, CM_NT),
                _ld_lds_bf8(c["x"], k1),
                acc,
            )
        fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < HC_OUT_COLS:
            fx.ptr_store(
                row_sum(c["red"], tid, 0), c["up"] + stream * HC_OUT_COLS + tid
            )
        gpu.barrier()

    if tid < HC_OUT_COLS:
        col = task * HC_OUT_COLS + tid
        total = fx.Float32(0.0)
        for stream in range_constexpr(HC):
            u = bf16_round(fx.ptr_load(c["up"] + stream * HC_OUT_COLS + tid))
            x = _ld_bf(c["normed"], stream * HIDDEN + col, CM_DEV)
            total = total + _sigmoid(u) * x
        _st_bf(
            c["args"]["pending_out"],
            col,
            bf16_round(total * fx.Float32(1.0 / HC)),
            CM_DEV,
        )
    publish(c["put"], c["up_ready"], task, 1, tid == 0)


@traced
def _stage_up_compact_join(c, task):
    """Keep eight K workers per stream; join four compact partial sets once."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    k0 = wave * 32 + 8 * (lane // 16)
    weights = (
        [
            _ld_bf8(
                c["args"]["hc_up"],
                (stream * HIDDEN + task * HC_OUT_COLS + lane % 16) * HC_RANK + k0,
                CM_NT,
            )
            for stream in range(HC)
        ]
        if c.get("expert_owned", False) or c.get("gdn_up_prefetch", False)
        else None
    )
    c["poll"]([(c["gate_ready"], 0, 1)])
    if tid < HC_RANK // 8:
        at = tid * 8
        gate = fx.Vector(
            bo.buffer_load(
                rsrc(c["gate"]),
                at // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(gate, c["x"] + at // 2)
    gpu.barrier()
    for stream in range_constexpr(HC):
        row = stream * HIDDEN + task * HC_OUT_COLS + lane % 16
        acc = mfma_bf16(
            (
                weights[stream]
                if weights is not None
                else _ld_bf8(c["args"]["hc_up"], row * HC_RANK + k0, CM_NT)
            ),
            _ld_lds_bf8(c["x"], k0),
            fx.Vector.filled(4, 0.0, fx.Float32),
        )
        if wave < 2:
            k1 = (8 + wave) * 32 + 8 * (lane // 16)
            acc = mfma_bf16(
                _ld_bf8(c["args"]["hc_up"], row * HC_RANK + k1, CM_NT),
                _ld_lds_bf8(c["x"], k1),
                acc,
            )
        if lane % 16 == 0:
            fx.ptr_store(
                acc, c["red"] + (stream * WAVES + wave) * 16 + (lane // 16) * 4
            )
    gpu.barrier()
    if (wave < HC) & (lane < HC_OUT_COLS):
        projected = fx.Float32(0.0)
        for owner in range_constexpr(WAVES):
            projected = projected + fx.ptr_load(
                c["red"] + (wave * WAVES + owner) * 16 + lane
            )
        col = task * HC_OUT_COLS + lane
        u = bf16_round(projected)
        x = _ld_bf(c["normed"], wave * HIDDEN + col, CM_DEV)
        fx.ptr_store(_sigmoid(u) * x, c["up"] + wave * HC_OUT_COLS + lane)
    gpu.barrier()
    if tid < HC_OUT_COLS:
        col = task * HC_OUT_COLS + tid
        total = fx.Float32(0.0)
        for stream in range_constexpr(HC):
            total = total + fx.ptr_load(c["up"] + stream * HC_OUT_COLS + tid)
        _st_bf(
            c["args"]["pending_out"],
            col,
            bf16_round(total * fx.Float32(1.0 / HC)),
            CM_DEV,
        )
    publish(c["put"], c["up_ready"], task, 1, tid == 0)


@traced
def _stage_up_group(c, group):
    """Collapse one K1280 half of HC output for front-MoE consumers."""

    if c.get("parallel_ready", False) or c.get("expert_owned", False):
        poll_ready_tiles(c, c["up_ready"], group * HC_UP_GROUP_TILES, HC_UP_GROUP_TILES)
    else:
        if c["tid"] == 0:
            c["poll"](
                [
                    (c["up_ready"], group * HC_UP_GROUP_TILES + tile, 1)
                    for tile in range_constexpr(HC_UP_GROUP_TILES)
                ]
            )
        gpu.barrier()
    publish(c["put"], c["up_group_ready"], group, 1, c["tid"] == 0)


@traced
def _stage_pending_group(c, group):
    """Stage one ready K1280 half of pending_out in this CTA's LDS."""

    if c["tid"] == 0:
        c["poll"]([(c["up_group_ready"], group, 1)])
    gpu.barrier()
    if c["tid"] < HC_UP_GROUP_COLS // 8:
        at = group * HC_UP_GROUP_COLS + c["tid"] * 8
        words = fx.Vector(
            bo.buffer_load(
                rsrc(c["args"]["pending_out"]),
                at // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(words, c["x"] + at // 2)
    gpu.barrier()


@traced
def run_hc_finish(c, bid):
    """Emit the four HC stages into the caller's persistent kernel."""

    task = first_task(bid, NORM0)
    if const_expr(c.get("attn_precombined", False)):
        if task < HC:
            _stage_precombined_norm(c, task)
    else:
        if task < HC:
            _stage_norm(c, task)

    task = first_task(bid, DOWN0)
    if task < HC_DOWN_TASKS:
        _stage_down(c, task)

    task = first_task(bid, GATE0)
    if task < 1:
        _stage_gate(c)

    up_stage = _stage_up_compact_join if c.get("hc_compact_join", False) else _stage_up
    task = first_task(bid, OUT0)
    if task < HC_OUT_TASKS:
        up_stage(c, task)

    task = first_task(bid, UP_GROUP0)
    if task < HC_UP_GROUPS:
        _stage_up_group(c, task)
