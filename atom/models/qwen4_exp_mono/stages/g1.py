# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen routed and shared G1 inside the persistent layer-post kernel.

The first 256 tasks are statically spread over the resident CTAs.  CTAs then
claim the remaining tasks through an agent-scope counter, so CTAs delayed by
HC, quant or router work naturally take less G1 work.  This mirrors the DSV4.1
MonoKernel scheduler while preserving Qwen's own weight contracts:

* routed experts: shuffled FP8 gate/up halves with FP32 128x128 block scales;
* shared expert: ordinary BF16 gate/up halves.

Each task produces one N16 intermediate tile.  Routed tasks accumulate four
K32 MFMAs inside each K128 scale block before applying activation and weight
scales.  Shared tasks retain the BF16 projection boundaries of the original
two linear layers before SiLU x up.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import buf_scalar_load, ptr_buf_tensor
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.layout import (
    G1_COLS,
    G1_PARTS,
    G1_ROUTED_TASKS,
    G1_SHARED_TASKS,
    G1_TASKS,
    MIDQ_GROUP,
    MIDQ_GROUPS,
    ROUTER_TASKS,
    XQ_GROUPS,
    XQ_WORDS,
)
from atom.models.qwen4_exp_mono.shapes import HIDDEN, INTERMEDIATE
from atom.models.qwen4_exp_mono.stages.hc import (
    HC_UP_GROUP_K_PASSES,
    HC_UP_GROUPS,
    _ld_bf8,
    _ld_f32,
    _ld_lds_bf8,
    _sigmoid,
    _st_bf,
    _st_f32,
    _stage_pending_group,
)
from atom.models.qwen4_exp_mono.stages.memory import shuffled_fp8_tile
from atom.models.qwen4_exp_mono.stages.quant import quantize_values
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    bf_hi,
    bf_lo,
    mfma_bf16,
    mfma_fp8,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import BLOCKS, WAVES

G1_START = ROUTER_TASKS
G1_FIXED = BLOCKS
K_BLOCK = 128
K_BLOCKS = HIDDEN // K_BLOCK
W13_ROWS = 2 * INTERMEDIATE
W13_SCALE_ROWS = W13_ROWS // K_BLOCK
RED_WORDS = WAVES * 64 * 4
MIDQ_THREADS = MIDQ_GROUP // 4


def _load_fp8_x(c, kb, sub, staged=False):
    # A K128 block is two K64 chunks.  Within one chunk a lane owns four
    # dwords, whose low/high i64 operands are its two K32 MFMAs.  Keep the
    # lane-group stride inside each K64 chunk; treating sub as four globally
    # contiguous K32 spans would mix the four lane groups.
    half = sub // 2
    pair = sub % 2
    at = kb * (K_BLOCK // 4) + half * 16 + (c["lane"] // 16) * 4 + pair * 2
    if const_expr(staged):
        words = fx.Vector(fx.ptr_load(c["x"] + at, result_type=T.vec(2, T.i32)))
    else:
        words = fx.Vector(
            bo.buffer_load(
                rsrc(c["xq"]),
                at,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
    return words.bitcast(fx.Int64)[0]


def _load_x_scale(c, kb, staged=False):
    if const_expr(staged):
        return fx.ptr_load(c["x"] + XQ_WORDS + kb).bitcast(fx.Float32)
    return _ld_f32(c["xq_scale"], kb, CM_DEV)


@traced
def _stage_routed_xq(c):
    """Cache this CTA's token activation and scales once for static G1 work."""

    tid = c["tid"]
    c["poll"]([(c["route_ready"], 0, 1)])
    gpu.barrier()
    if tid < XQ_WORDS // 4:
        words = fx.Vector(
            bo.buffer_load(
                rsrc(c["xq"]),
                tid * 4,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        fx.ptr_store(words, c["x"] + tid * 4)
    if tid < XQ_GROUPS:
        scale = _ld_f32(c["xq_scale"], tid, CM_DEV)
        fx.ptr_store(scale.bitcast(fx.Int32), c["x"] + XQ_WORDS + tid)
    gpu.barrier()


def _load_fp8_weight(c, expert, row_group, kb):
    return shuffled_fp8_tile(
        c["args"]["routed_w13"], expert, row_group, kb, c["lane"], W13_ROWS, HIDDEN
    )


def _load_w13_scale(c, expert, row_block, kb, scalar=False):
    at = (expert * W13_SCALE_ROWS + row_block) * K_BLOCKS + kb
    if const_expr(scalar):
        bits = fx.Int32(
            buf_scalar_load(
                ptr_buf_tensor(c["args"]["routed_w13_scale"]),
                at,
            )
        )
        return bits.bitcast(fx.Float32)
    return _ld_f32(
        c["args"]["routed_w13_scale"],
        at,
        CM_NT,
    )


@traced
def _quant_routed_mid(c, slot, group):
    """One routed Mid group of 128 BF16 values -> FP8 + continuous scale."""

    tid = c["tid"]
    c["poll"](
        [
            (
                c["g1_ready"],
                G1_SHARED_TASKS + slot * G1_PARTS + group * 8 + part,
                1,
            )
            for part in range(8)
        ]
    )
    words = fx.Vector.filled(2, 0, fx.Int32)
    if tid < MIDQ_THREADS:
        words = fx.Vector(
            bo.buffer_load(
                rsrc(c["routed_mid"]),
                (slot * INTERMEDIATE + group * MIDQ_GROUP) // 2 + tid * 2,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
    values = [bf_lo(words[0]), bf_hi(words[0]), bf_lo(words[1]), bf_hi(words[1])]
    packed, scale = quantize_values(values)
    if tid < MIDQ_THREADS:
        bo.buffer_store(
            packed,
            rsrc(c["midq"]),
            slot * (INTERMEDIATE // 4) + group * MIDQ_THREADS + tid,
            cache_modifier=CM_DEV,
        )
    if tid == 0:
        _st_f32(c["midq_scale"], slot * MIDQ_GROUPS + group, scale, CM_DEV)
    publish(
        c["put"],
        c["midq_ready"],
        slot * MIDQ_GROUPS + group,
        1,
        tid == 0,
    )


@traced
def _stage_routed_g1(
    c, routed_task, ready_task, staged=False, quantize=True, prefetch=None
):
    prefetch = not staged and (quantize if prefetch is None else prefetch)
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    slot = routed_task // G1_PARTS
    part = routed_task % G1_PARTS
    if const_expr(not staged):
        c["poll"]([(c["route_ready"], 0, 1)])
    active = _ld_f32(c["route_weights"], slot, CM_DEV) > fx.Float32(0.0)
    if active:
        expert = fx.Int32(
            bo.buffer_load(
                rsrc(c["route_ids"]),
                slot,
                vec_width=1,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        gate = fx.Vector.filled(4, 0.0, fx.Float32)
        up = fx.Vector.filled(4, 0.0, fx.Float32)

        def load_turn(turn):
            use_kb = fx.min(wave + WAVES * turn, K_BLOCKS - 1)
            return (
                _load_x_scale(c, use_kb, staged),
                _load_w13_scale(c, expert, part // 8, use_kb, staged),
                _load_w13_scale(c, expert, 5 + part // 8, use_kb, staged),
                _load_fp8_weight(c, expert, part, use_kb),
                _load_fp8_weight(c, expert, G1_PARTS + part, use_kb),
                [_load_fp8_x(c, use_kb, sub, staged) for sub in range(4)],
            )

        # Keep the caller's task schedule and one future K128 bundle live,
        # consuming all scale blocks in the original order.
        operands = load_turn(0) if prefetch else None
        next_operands = operands
        for turn in range_constexpr(3):
            kb = wave + WAVES * turn
            live = kb < K_BLOCKS
            use_kb = fx.min(kb, K_BLOCKS - 1)
            if const_expr(prefetch) and turn + 1 < 3:
                next_operands = load_turn(turn + 1)
            if const_expr(not prefetch):
                x_scale = _load_x_scale(c, use_kb, staged)
                gate_scale = _load_w13_scale(c, expert, part // 8, use_kb, staged)
                up_scale = _load_w13_scale(c, expert, 5 + part // 8, use_kb, staged)
                gw = _load_fp8_weight(c, expert, part, use_kb)
                uw = _load_fp8_weight(c, expert, G1_PARTS + part, use_kb)
            else:
                x_scale, gate_scale, up_scale, gw, uw, xs = operands
            gate_block = fx.Vector.filled(4, 0.0, fx.Float32)
            up_block = fx.Vector.filled(4, 0.0, fx.Float32)
            for sub in range_constexpr(4):
                x = xs[sub] if prefetch else _load_fp8_x(c, use_kb, sub, staged)
                gate_block = mfma_fp8(gw[sub], x, gate_block)
                up_block = mfma_fp8(uw[sub], x, up_block)
            gs = live.select(x_scale * gate_scale, fx.Float32(0.0))
            us = live.select(x_scale * up_scale, fx.Float32(0.0))
            gate = gate + gate_block * gs
            up = up + up_block * us
            operands = next_operands

        fx.ptr_store(gate, c["red"] + (wave * 64 + lane) * 4)
        fx.ptr_store(up, c["red"] + RED_WORDS + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < G1_COLS:
            g = row_sum(c["red"], tid, 0)
            u = row_sum(c["red"] + RED_WORDS, tid, 0)
            mid = bf16_round((g * _sigmoid(g)) * u)
            _st_bf(
                c["routed_mid"],
                slot * INTERMEDIATE + part * G1_COLS + tid,
                mid,
                CM_DEV,
            )
    publish(c["put"], c["g1_ready"], ready_task, 1, tid == 0)
    if const_expr(quantize) and active & (part % 8 == 7):
        _quant_routed_mid(c, slot, part // 8)


@traced
def _stage_shared_g1(c, part, ready_task):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    shared_intermediate = c.get("shared_intermediate", INTERMEDIATE)
    shared_parts = shared_intermediate // G1_COLS
    gate_row = part * G1_COLS + lane % 16
    up_row = shared_intermediate + gate_row
    gate = fx.Vector.filled(4, 0.0, fx.Float32)
    up = fx.Vector.filled(4, 0.0, fx.Float32)
    for group in range_constexpr(HC_UP_GROUPS):
        _stage_pending_group(c, group)
        for local in range_constexpr(HC_UP_GROUP_K_PASSES):
            k_pass = group * HC_UP_GROUP_K_PASSES + local
            k0 = (wave + WAVES * k_pass) * 32 + 8 * (lane // 16)
            x = _ld_lds_bf8(c["x"], k0)
            gate = mfma_bf16(
                _ld_bf8(c["args"]["shared_w13"], gate_row * HIDDEN + k0, CM_NT),
                x,
                gate,
            )
            up = mfma_bf16(
                _ld_bf8(c["args"]["shared_w13"], up_row * HIDDEN + k0, CM_NT),
                x,
                up,
            )
    fx.ptr_store(gate, c["red"] + (wave * 64 + lane) * 4)
    fx.ptr_store(up, c["red"] + RED_WORDS + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < G1_COLS:
        g = bf16_round(row_sum(c["red"], tid, 0))
        u = bf16_round(row_sum(c["red"] + RED_WORDS, tid, 0))
        _st_bf(
            c["shared_mid"],
            part * G1_COLS + tid,
            bf16_round((g * _sigmoid(g)) * u),
            CM_DEV,
        )
    publish(c["put"], c["g1_ready"], ready_task, 1, tid == 0)
    if part == shared_parts - 1:
        c["poll"]([(c["g1_ready"], task, 1) for task in range(shared_parts)])
        publish(c["put"], c["shared_mid_ready"], 0, 1, tid == 0)


@traced
def _take_g1_task(c):
    if c["tid"] == 0:
        at = c["g1_queue"] + fx.Int64(c["tag"] * 4)
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), at)
        task = G1_FIXED + fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        fx.ptr_store(task, c["queue"])
    gpu.barrier()
    return fx.ptr_load(c["queue"])


@traced
def run_g1(c, bid):
    """Emit shared and routed G1 through one persistent dynamic task loop."""

    # Rotate the fixed round past the 33 router CTAs.  Tasks 0..39 are shared,
    # so they begin on CTAs 33..72 while the router owners finish their work.
    task = (bid - G1_START + BLOCKS) % BLOCKS
    while task < G1_TASKS:
        if task < G1_SHARED_TASKS:
            _stage_shared_g1(c, task, task)
        else:
            routed = task - G1_SHARED_TASKS
            if routed < G1_ROUTED_TASKS:
                _stage_routed_g1(c, routed, task)
        task = _take_g1_task(c)


@traced
def run_g1_tp8(c, bid):
    """Emit the rank-local shared shard and a static two-round routed schedule."""

    shared_parts = c["shared_intermediate"] // G1_COLS
    owner = (bid - G1_START + BLOCKS) % BLOCKS
    if owner < shared_parts:
        _stage_shared_g1(c, owner, owner)

    # Every CTA caches the token once. One fixed routed task covers the first
    # 256 tiles; the 144-task tail goes to owners 5..148, which excludes the
    # five shared producers and the physical router CTAs 0..32.
    routed1 = BLOCKS + owner - shared_parts
    has_second = (owner >= shared_parts) & (
        owner < shared_parts + G1_ROUTED_TASKS - BLOCKS
    )
    _stage_routed_xq(c)
    if owner < G1_ROUTED_TASKS:
        _stage_routed_g1(c, owner, G1_SHARED_TASKS + owner, staged=True)
    if has_second:
        _stage_routed_g1(c, routed1, G1_SHARED_TASKS + routed1, staged=True)
