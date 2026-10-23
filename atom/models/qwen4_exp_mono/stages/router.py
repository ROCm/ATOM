# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen S=1 router and exact route selection inside the persistent K2.

The 513-row BF16 projection consumes the HC mixed vector.  Thirty-three CTAs
form N16 tiles with the same eight-wave K split used by the HC projections.
One resident CTA then ranks the 512 routed logits.  Its signed 64-bit key is
ordered by logit and then by the reverse expert id, matching AITER's lower-id
tie rule.  The routed weights are the stable softmax over the selected ten
logits; the tail logit is an independent FP32 sigmoid shared-expert gate.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.layout import (
    ROUTER_COLS,
    ROUTER_TASKS,
)
from atom.models.qwen4_exp_mono.shapes import EXPERTS, HIDDEN, TOPK
from atom.models.qwen4_exp_mono.stages.hc import (
    HC_UP_GROUP_K_PASSES,
    HC_UP_GROUPS,
    LOG2E,
    _ld_bf,
    _ld_bf8,
    _ld_f32,
    _ld_lds_bf8,
    _sigmoid,
    _st_bf,
    _st_f32,
    _stage_pending_group,
    poll_ready_tiles,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    hw_exp2,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from atom.mono.device.select import (
    KEY_LEAST,
    compact,
    count_beaten,
    group_floor,
    put_key,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import first_task

ROUTER0 = ROUTE0 = 0
ROUTE_FLOOR_K = 16
ROUTE_CAND_CAP = 128
ROUTE_KEY_WORDS = 2 * EXPERTS
ROUTE_CAND_AT = ROUTE_KEY_WORDS
ROUTE_COUNTER_AT = ROUTE_CAND_AT + 2 * ROUTE_CAND_CAP
ROUTE_FLOOR_AT = ROUTE_COUNTER_AT + 1


def _st_i32(ptr, i, value, cache_modifier=0):
    bo.buffer_store(fx.Int32(value), rsrc(ptr), i, cache_modifier=cache_modifier)


def _float_key(value):
    """Signed Int32 whose order matches finite FP32 numerical order."""

    finite = fx.isnan(value).select(fx.Float32(float("-inf")), value)
    bits = finite.bitcast(fx.Int32)
    return bits ^ ((bits >> 31) & fx.Int32(0x7FFFFFFF)), finite


@traced
def _stage_router(c, task):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    row = fx.min(task * ROUTER_COLS + lane % 16, EXPERTS)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)

    def load_group(group):
        return [
            _ld_bf8(
                c["args"]["router"],
                row * HIDDEN
                + (wave + 8 * (group * HC_UP_GROUP_K_PASSES + local)) * 32
                + 8 * (lane // 16),
                0,
            )
            for local in range(HC_UP_GROUP_K_PASSES)
        ]

    weights = (
        load_group(0)
        if c.get("router_prefetch", c.get("expert_owned", False))
        else None
    )
    next_weights = weights
    for group in range_constexpr(HC_UP_GROUPS):
        _stage_pending_group(c, group)
        if (
            c.get("router_prefetch", c.get("expert_owned", False))
            and group + 1 < HC_UP_GROUPS
        ):
            next_weights = load_group(group + 1)
        for local in range_constexpr(HC_UP_GROUP_K_PASSES):
            k_pass = group * HC_UP_GROUP_K_PASSES + local
            k0 = (wave + 8 * k_pass) * 32 + 8 * (lane // 16)
            weight = (
                weights[local]
                if weights is not None
                else _ld_bf8(c["args"]["router"], row * HIDDEN + k0, CM_NT)
            )
            acc = mfma_bf16(weight, _ld_lds_bf8(c["x"], k0), acc)
        weights = next_weights
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROUTER_COLS:
        out_row = task * ROUTER_COLS + tid
        if out_row <= EXPERTS:
            _st_bf(
                c["router_logits"],
                out_row,
                bf16_round(row_sum(c["red"], tid, 0)),
                CM_DEV,
            )
    publish(c["put"], c["router_tile_ready"], task, 1, tid == 0)


@traced
def _stage_route(c):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    poll_ready_tiles(c, c["router_tile_ready"], 0, ROUTER_TASKS)

    logit = _ld_bf(c["router_logits"], tid, CM_DEV)
    key, logit = _float_key(logit)
    # Descending Int64 comparison sees the logit key first.  Reverse the low
    # word so a smaller expert id beats a larger id when logits are equal.
    reverse_id = fx.Int32(0x7FFFFFFF) - tid
    keys = fx.recast_iter(fx.Int32, c["red"])
    cand = keys + ROUTE_CAND_AT
    counter = keys + ROUTE_COUNTER_AT
    floor_red = keys + ROUTE_FLOOR_AT
    mine = put_key(keys, tid, key, reverse_id, fx.Boolean(True))
    if tid < ROUTE_CAND_CAP:
        fx.ptr_store(fx.Int32(0), cand + 2 * tid)
        fx.ptr_store(fx.Int32(KEY_LEAST), cand + (2 * tid + 1))
    if tid == 0:
        fx.ptr_store(fx.Int32(0), counter)

    # Keep exact top-k semantics while avoiding the 512x512 comparison.  A
    # key-only floor retains every tie, then the full 64-bit key (including
    # reverse expert id) ranks the compact superset.  Pathological ties that
    # exceed the bounded table fall back to all 512 entries.
    floor = group_floor([key], ROUTE_FLOOR_K, floor_red, lane, wave)
    compact(
        cand,
        counter,
        ROUTE_CAND_CAP,
        [(key, reverse_id, fx.Boolean(True))],
        floor,
    )
    gpu.barrier()
    n_cand = fx.ptr_load(counter)
    every = n_cand > ROUTE_CAND_CAP
    base = keys + every.select(fx.Int32(0), fx.Int32(ROUTE_CAND_AT))
    count = every.select(fx.Int32(EXPERTS), n_cand)
    rank = count_beaten(base, [mine], count)[0]
    if rank < TOPK:
        _st_i32(c["route_ids"], rank, tid, CM_DEV)
        fx.ptr_store(logit, c["up"] + rank)
    gpu.barrier()

    # AITER accumulates the ten exp numerators in slot order, then multiplies
    # each by one reciprocal.  Keep that serial FP32 order for the route ABI.
    if tid == 0:
        top = fx.ptr_load(c["up"])
        total = fx.Float32(0.0)
        for slot in range_constexpr(TOPK):
            value = fx.ptr_load(c["up"] + slot)
            numer = hw_exp2((value - top) * fx.Float32(LOG2E))
            fx.ptr_store(numer, c["up"] + TOPK + slot)
            total = total + numer
        inv = fx.Float32(1.0) / total
        for slot in range_constexpr(TOPK):
            _st_f32(
                c["route_weights"],
                slot,
                fx.ptr_load(c["up"] + TOPK + slot) * inv,
                CM_DEV,
            )
        shared = _sigmoid(_ld_bf(c["router_logits"], EXPERTS, CM_DEV))
        _st_f32(c["shared_gate"], 0, shared, CM_DEV)
    if c.get("expert_owned", False):
        # Every rank selects the same global top-k.  Before publishing the
        # route, translate it to this rank's physical expert ids and zero the
        # weights of non-local slots.  G1/G2 can then use the existing compact
        # route ABI without ever indexing a non-local expert buffer.
        gpu.barrier()
        if tid < TOPK:
            global_expert = fx.Int32(
                bo.buffer_load(
                    rsrc(c["route_ids"]),
                    tid,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            local_expert = fx.Int32(
                bo.buffer_load(
                    rsrc(c["args"]["expert_map"]),
                    global_expert,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            local = local_expert >= 0
            _st_i32(
                c["route_ids"],
                tid,
                local.select(local_expert, fx.Int32(0)),
                CM_DEV,
            )
            route_weight = _ld_f32(c["route_weights"], tid, CM_DEV)
            _st_f32(
                c["route_weights"],
                tid,
                local.select(route_weight, fx.Float32(0.0)),
                CM_DEV,
            )
        gpu.barrier()
    publish(c["put"], c["route_ready"], 0, 1, tid == 0)


@traced
def run_router(c, bid):
    """Emit the router projection and route selection into the caller's K2."""

    task = first_task(bid, ROUTER0)
    if task < ROUTER_TASKS:
        _stage_router(c, task)

    task = first_task(bid, ROUTE0)
    if task < 1:
        _stage_route(c)
