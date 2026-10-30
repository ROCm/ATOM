"""FlyDSL compact M16 task emitter for fixed Native top-k IDs/weights.

One complete wave sorts 64 integer keys (44 picks plus invalid lane slots).
Each expert's first lane owns all 16 output entries of that expert's tile,
including padding. There are no overlapping padding/pick writes, LDS barriers,
or cross-CTA waits. The emitter may run on wave0 of a larger HC CTA after its
caller establishes route readiness. Standalone G1 consumes after stream order.
"""

import hashlib
from functools import cache
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.chain.moe_front_device_plan import SORT_STEPS
from atom.mono.device.ops import (
    ballot,
    kernel_symbol,
    lane_gather,
    lanes_below,
    launcher,
    popcount,
    rsrc,
    traced,
)
from atom.mono.runtime.abi import KernelAbi

SOURCE = hashlib.sha256(
    Path(__file__).read_bytes()
    + Path(__file__).with_name("moe_front_device_plan.py").read_bytes()
).hexdigest()
ABI = KernelAbi(
    (
        "ids",
        "weights",
        "sorted_ids",
        "sorted_weights",
        "experts",
        "valid",
        "m_indices",
        "reverse",
        "out",
    )
)
_STREAM = fx.Stream(None)


@traced
def emit_tasks(
    ids,
    weights,
    sorted_ids,
    sorted_weights,
    experts,
    valid,
    m_indices,
    reverse,
    lane,
    input_cm=0,
):
    """Caller activates every lane of exactly one wave; no other wave writes."""
    token, route = lane // 16, lane % 16
    eid = fx.Int32(
        bo.buffer_load(
            rsrc(ids),
            token * 11 + fx.min(route, 10),
            vec_width=1,
            dtype=T.i32,
            cache_modifier=input_cm,
        )
    )
    eid = (route < 11).select(eid, fx.Int32(1023))
    key = eid * 64 + lane
    for step in range_constexpr(len(SORT_STEPS)):
        width, distance = SORT_STEPS[step]
        peer = lane_gather(key, lane ^ distance)
        take_min = ((lane & distance) == 0) == ((lane & width) == 0)
        key = take_min.select(fx.min(key, peer), fx.max(key, peer))
    eid = key // 64
    previous = lane_gather(key, fx.max(lane - 1, 0)) // 64
    head = (eid < 513) & ((lane == 0) | (eid != previous))
    heads = ballot(head)
    tile = lanes_below(heads, lane)
    if head:
        bo.buffer_store(eid, rsrc(experts), tile)
    # All lanes execute each gather; only the tile owner stores. Top-k has
    # no repeated expert in one row, so each expert has at most four picks.
    for j in range_constexpr(4):
        peer = lane_gather(key, fx.min(lane + j, 63))
        pick = peer % 64
        token, route = pick // 16, pick % 16
        valid_pick = (peer // 64) == eid
        weight = fx.Float32(
            bo.buffer_load(
                rsrc(weights),
                token * 11 + fx.min(route, 10),
                vec_width=1,
                dtype=T.f32,
                cache_modifier=input_cm,
            )
        )
        if head:
            at = tile * 16 + j
            bo.buffer_store(
                valid_pick.select((route << 24) | token, fx.Int32(4)),
                rsrc(sorted_ids),
                at,
            )
            bo.buffer_store(
                valid_pick.select(weight, fx.Float32(0)), rsrc(sorted_weights), at
            )
            bo.buffer_store(valid_pick.select(token, fx.Int32(4)), rsrc(m_indices), at)
            if valid_pick:
                bo.buffer_store(at, rsrc(reverse), token * 11 + route)
    if head:
        for j in range_constexpr(4, 16):
            at = tile * 16 + j
            bo.buffer_store(fx.Int32(4), rsrc(sorted_ids), at)
            bo.buffer_store(fx.Float32(0), rsrc(sorted_weights), at)
            bo.buffer_store(fx.Int32(4), rsrc(m_indices), at)
    if lane == 0:
        bo.buffer_store(popcount(heads) * 16, rsrc(valid), 0)
        bo.buffer_store(fx.Int32(4), rsrc(valid), 1)


@cache
def build():
    name = kernel_symbol("qwen_mtp_compact_m16_tasks", rows=4, version=1)

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(
        ids: Int64,
        weights: Int64,
        sorted_ids: Int64,
        sorted_weights: Int64,
        experts: Int64,
        valid: Int64,
        m_indices: Int64,
        reverse: Int64,
        out: Int64,
    ):
        tid, cta = fx.thread_idx.x, fx.block_idx.x
        if cta == 0:
            emit_tasks(
                ids,
                weights,
                sorted_ids,
                sorted_weights,
                experts,
                valid,
                m_indices,
                reverse,
                tid,
            )
        else:
            zero = fx.Vector.filled(4, 0, fx.Int32)
            for j in range_constexpr(2):
                # Ten CTAs, each clears 512 i32 words = 1024 BF16 elements.
                bo.buffer_store(zero, rsrc(out), (cta - 1) * 512 + j * 256 + tid * 4)

    @launcher(name)
    def launch(
        ids: Int64,
        weights: Int64,
        sorted_ids: Int64,
        sorted_weights: Int64,
        experts: Int64,
        valid: Int64,
        m_indices: Int64,
        reverse: Int64,
        out: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCE
        kernel(
            ids,
            weights,
            sorted_ids,
            sorted_weights,
            experts,
            valid,
            m_indices,
            reverse,
            out,
        ).launch(grid=(11,), block=(64,), stream=stream)

    ABI.check(kernel, launch)
    return launch
