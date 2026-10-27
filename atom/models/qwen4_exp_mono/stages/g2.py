# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen S=1 shared/routed G2 and direct top-k merge in persistent K2.

One CTA owns one N16 output tile.  It forms the BF16 shared projection, then
consumes the ten routed slots in their exact top-k order.  Every routed
contribution and every running add is rounded to BF16, matching the stage-2
buffer contract without materializing ``[topk, H]`` global partials.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.layout import (
    G2_COLS,
    G2_TASKS,
    MIDQ_GROUPS,
)
from atom.models.qwen4_exp_mono.shapes import HIDDEN, INTERMEDIATE, TOPK
from atom.models.qwen4_exp_mono.stages.collective import publish_pair
from atom.models.qwen4_exp_mono.stages.hc import (
    _ld_bf,
    _ld_f32,
    _ld_lds_bf8,
    _st_bf,
)
from atom.models.qwen4_exp_mono.stages.memory import shuffled_fp8_tile
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    mfma_bf16,
    mfma_fp8,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from atom.mono.plan.execution import WAVES, first_task

G2_START = 0
K_BLOCK = 128
K_BLOCKS = INTERMEDIATE // K_BLOCK
W2_SCALE_ROWS = HIDDEN // K_BLOCK


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _ld_bf8_masked(ptr, bf16_index, valid, nbytes, cache_modifier=0):
    return fx.Vector(
        bo.buffer_load(
            rsrc(ptr, nbytes=nbytes),
            bf16_index // 2,
            vec_width=4,
            dtype=T.i32,
            mask=valid,
            cache_modifier=cache_modifier,
        )
    ).bitcast(fx.BFloat16)


def _scratch_word(c, name):
    return c["layout"][name][0] // 4


def _scratch_f32(c, name, index):
    return fx.Float32(
        bo.buffer_load(
            rsrc(c["scratch"]),
            _scratch_word(c, name) + index,
            vec_width=1,
            dtype=T.f32,
            cache_modifier=CM_DEV,
        )
    )


def _load_midq(c, slot, kb, sub):
    half = sub // 2
    pair = sub % 2
    words = fx.Vector(
        bo.buffer_load(
            rsrc(c["scratch"]),
            _scratch_word(c, "midq")
            + slot * (INTERMEDIATE // 4)
            + kb * (K_BLOCK // 4)
            + half * 16
            + (c["lane"] // 16) * 4
            + pair * 2,
            vec_width=2,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    return words.bitcast(fx.Int64)[0]


def _load_w2(c, w2, expert, row_group, kb):
    return shuffled_fp8_tile(
        w2,
        expert,
        row_group,
        kb,
        c["lane"],
        HIDDEN,
        INTERMEDIATE,
        cache_modifier=0 if c.get("owned_sparse_g2", False) else CM_NT,
    )


@traced
def _shared_g2(c, part):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    shared_intermediate = c.get("shared_intermediate", INTERMEDIATE)
    shared_vecs = _ceil_div(shared_intermediate, 8)
    shared_lds_vecs = _ceil_div(shared_intermediate, 32) * 4
    c["poll"]([(c["shared_mid_ready"], 0, 1)])
    if tid < shared_lds_vecs:
        words = fx.Vector(
            bo.buffer_load(
                rsrc(c["scratch"]),
                _scratch_word(c, "shared_mid") + tid * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        words = (tid < shared_vecs).select(
            words,
            fx.Vector.filled(4, 0, fx.Int32),
        )
        fx.ptr_store(words, c["x"] + tid * 4)
    gpu.barrier()

    row = part * G2_COLS + lane % 16
    shared_w2 = c["args"]["shared_w2"]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    shared_steps = _ceil_div(shared_intermediate, 32)
    # TP8 owns I=80: emit only the rank-local K turns (TP1 still uses three).
    shared_turns = _ceil_div(shared_steps, WAVES)
    for turn in range_constexpr(shared_turns):
        step = wave + WAVES * turn
        live = step < shared_steps
        use_step = fx.min(step, shared_steps - 1)
        k0 = use_step * 32 + 8 * (lane // 16)
        block = mfma_bf16(
            _ld_bf8_masked(
                shared_w2,
                row * shared_intermediate + k0,
                k0 < shared_intermediate,
                HIDDEN * shared_intermediate * 2,
                CM_NT,
            ),
            (k0 < shared_intermediate).select(
                _ld_lds_bf8(
                    c["x"],
                    k0,
                ),
                fx.Vector.filled(8, 0.0, fx.BFloat16),
            ),
            fx.Vector.filled(4, 0.0, fx.Float32),
        )
        scale = live.select(fx.Float32(1.0), fx.Float32(0.0))
        acc = acc + block * scale
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < G2_COLS:
        fx.ptr_store(bf16_round(row_sum(c["red"], tid, 0)), c["up"] + G2_COLS + tid)
        fx.ptr_store(fx.Float32(0.0), c["up"] + tid)
    gpu.barrier()


@traced
def _routed_slot(c, part, slot):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    route_weight = uniform(
        _scratch_f32(c, "route_weights", slot).bitcast(fx.Int32)
    ).bitcast(fx.Float32)
    active = route_weight > fx.Float32(0.0)
    owned_sparse = c.get("owned_sparse_g2", False)
    weight_cache = 0 if owned_sparse else CM_NT
    if active:
        c["poll"](
            [
                (c["midq_ready"], slot * MIDQ_GROUPS + group, 1)
                for group in range(MIDQ_GROUPS)
            ]
        )
        expert = fx.Int32(
            bo.buffer_load(
                rsrc(c["scratch"]),
                _scratch_word(c, "route_ids") + slot,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        use_kb = fx.min(wave, K_BLOCKS - 1)
        routed_w2 = c["args"]["routed_w2"]
        routed_w2_scale = c["args"]["routed_w2_scale"]
        partial = fx.Vector.filled(4, 0.0, fx.Float32)
        if not owned_sparse or wave < K_BLOCKS:
            weights = _load_w2(c, routed_w2, expert, part, use_kb)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for sub in range_constexpr(4):
                acc = mfma_fp8(weights[sub], _load_midq(c, slot, use_kb, sub), acc)
            a_scale = _scratch_f32(c, "midq_scale", slot * MIDQ_GROUPS + use_kb)
            w_scale = _ld_f32(
                routed_w2_scale,
                (expert * W2_SCALE_ROWS + part // 8) * K_BLOCKS + use_kb,
                weight_cache,
            )
            scale = (wave < K_BLOCKS).select(a_scale * w_scale, fx.Float32(0.0))
            partial = acc * scale
        fx.ptr_store(partial, c["red"] + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < G2_COLS:
            raw = row_sum(c["red"], tid, 0)
            weighted = bf16_round(raw * route_weight)
            previous = fx.ptr_load(c["up"] + tid)
            fx.ptr_store(bf16_round(previous + weighted), c["up"] + tid)
        gpu.barrier()


@traced
def _stage_g2(c, part):
    _shared_g2(c, part)
    for slot in range(0, TOPK, 1):
        _routed_slot(c, part, slot)
    tid = c["tid"]
    if tid < G2_COLS:
        routed = fx.ptr_load(c["up"] + tid)
        shared = fx.ptr_load(c["up"] + G2_COLS + tid)
        gate = bf16_round(_scratch_f32(c, "shared_gate", 0))
        gated_shared = bf16_round(shared * gate)
        _st_bf(
            c["args"]["moe_out"],
            part * G2_COLS + tid,
            bf16_round(routed + gated_shared),
        )
    gpu.barrier()
    if c.get("publish_partial", False):  # noqa: SIM102 - compile-time DSL branch
        if tid < G2_COLS // 2:
            col = part * G2_COLS + 2 * tid
            publish_pair(
                c["mailbox"],
                c["peer_sym"],
                c["peer_region_off"],
                "moe",
                c["layer"],
                1,
                0,
                col,
                _ld_bf(c["args"]["moe_out"], col, CM_DEV).to(fx.BFloat16),
                _ld_bf(c["args"]["moe_out"], col + 1, CM_DEV).to(fx.BFloat16),
            )


@traced
def run_g2(c, bid):
    task = first_task(bid, G2_START)
    if task < G2_TASKS:
        _stage_g2(c, task)


@traced
def wave_routed_g2(c, part):
    """Two eight-wave expert joins, preserving scaled K and BF16 top-k order."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    for turn in range_constexpr((TOPK + WAVES - 1) // WAVES):
        slot = turn * WAVES + wave
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        if slot < TOPK:
            route_weight = _scratch_f32(c, "route_weights", slot)
            if route_weight > fx.Float32(0.0):
                c["poll"](
                    [
                        (c["midq_ready"], slot * MIDQ_GROUPS + group, 1)
                        for group in range(MIDQ_GROUPS)
                    ]
                )
                expert = fx.Int32(
                    bo.buffer_load(
                        rsrc(c["scratch"]),
                        _scratch_word(c, "route_ids") + slot,
                        vec_width=1,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                first_weights = _load_w2(c, c["args"]["routed_w2"], expert, part, 0)
                first_scale = _ld_f32(
                    c["args"]["routed_w2_scale"],
                    (expert * W2_SCALE_ROWS + part // 8) * K_BLOCKS,
                    CM_NT,
                )
                weights, w_scale = first_weights, first_scale
                next_weights, next_scale = weights, w_scale
                for kb in range_constexpr(K_BLOCKS):
                    if kb + 1 < K_BLOCKS:
                        next_weights = _load_w2(
                            c, c["args"]["routed_w2"], expert, part, kb + 1
                        )
                        next_scale = _ld_f32(
                            c["args"]["routed_w2_scale"],
                            (expert * W2_SCALE_ROWS + part // 8) * K_BLOCKS + kb + 1,
                            CM_NT,
                        )
                    block = fx.Vector.filled(4, 0.0, fx.Float32)
                    for sub in range_constexpr(4):
                        block = mfma_fp8(
                            weights[sub], _load_midq(c, slot, kb, sub), block
                        )
                    a_scale = _scratch_f32(c, "midq_scale", slot * MIDQ_GROUPS + kb)
                    acc = acc + block * (a_scale * w_scale)
                    weights, w_scale = next_weights, next_scale
                for _ in range_constexpr(WAVES - K_BLOCKS):
                    acc = acc + fx.Vector.filled(4, 0.0, fx.Float32)
        fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < G2_COLS:
            routed = fx.ptr_load(c["up"] + tid)
            for slot_wave in range_constexpr(min(WAVES, TOPK - turn * WAVES)):
                raw = row_sum(c["red"], tid, 0, waves=(slot_wave,))
                route_weight = _scratch_f32(
                    c, "route_weights", turn * WAVES + slot_wave
                )
                weighted = bf16_round(raw * route_weight)
                routed = bf16_round(routed + weighted)
            fx.ptr_store(routed, c["up"] + tid)
        gpu.barrier()


@traced
def run_wave_expert_g2(c, bid):
    """One expert per wave, retaining the K-block and top-k BF16 sum order.

    Each wave computes the original five independently scaled K128 partials
    and adds them in wave/K-block order. Eight slots then share one CTA join;
    the final two use a second join. No cross-CTA contribution buffer is added.
    """
    if bid < G2_TASKS:
        _shared_g2(c, bid)
        tid = c["tid"]
        wave_routed_g2(c, bid)
        if tid < G2_COLS:
            routed = fx.ptr_load(c["up"] + tid)
            shared = fx.ptr_load(c["up"] + G2_COLS + tid)
            gate = bf16_round(_scratch_f32(c, "shared_gate", 0))
            gated_shared = bf16_round(shared * gate)
            _st_bf(
                c["args"]["moe_out"],
                bid * G2_COLS + tid,
                bf16_round(routed + gated_shared),
            )
        gpu.barrier()
