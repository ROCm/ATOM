# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen S=1 shared/routed G2 and direct top-k merge in persistent K2.

One CTA owns one N16 output tile.  It forms the BF16 shared projection, then
consumes the ten routed slots in their exact top-k order.  Every routed
contribution and every running add is rounded to BF16, matching the stage-2
buffer contract without materializing ``[topk, H]`` global partials.
"""

import flydsl.expr as fx
from atom.mono.device.sync import Addr
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.config import HIDDEN, INTERMEDIATE, TOPK
from atom.models.qwen4_exp_mono.kernels.collective import publish_pair
from atom.models.qwen4_exp_mono.kernels.finish_shape import (
    G2_COLS,
    G2_TASKS,
    MIDQ_GROUPS,
)
from atom.models.qwen4_exp_mono.kernels.hc import (
    _ld_bf,
    _ld_f32,
    _ld_lds_bf8,
    _st_bf,
)
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
from atom.mono.plan.execution import BLOCKS, WAVES, first_task

G2_START = 0
G2_S2_TASKS = 2 * G2_TASKS
G2_S4_PAIR_BLOCKS = BLOCKS // 2
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


def _i64_pair(words, pair):
    return fx.Vector.from_elements(
        [words[2 * pair], words[2 * pair + 1]], fx.Int32
    ).bitcast(fx.Int64)[0]


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
    lane = c["lane"]
    base = expert * (HIDDEN * INTERMEDIATE // 4)
    tile = row_group * (INTERMEDIATE // 64) + 2 * kb
    lo = fx.Vector(
        bo.buffer_load(
            rsrc(w2),
            base + tile * 256 + lane * 4,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=c.get("weight_cache", CM_NT),
        )
    )
    hi = fx.Vector(
        bo.buffer_load(
            rsrc(w2),
            base + (tile + 1) * 256 + lane * 4,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=c.get("weight_cache", CM_NT),
        )
    )
    return [
        _i64_pair(lo, 0),
        _i64_pair(lo, 1),
        _i64_pair(hi, 0),
        _i64_pair(hi, 1),
    ]


@traced
def _shared_g2(c, part):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    shared_intermediate = c.get("shared_intermediate", INTERMEDIATE)
    paired_shared = c.get("paired_shared_g2", False)
    if c.get("preloaded_shared_g2", False):
        if tid < G2_COLS:
            fx.ptr_store(
                _ld_bf(c["args"]["moe_out"], part * G2_COLS + tid, CM_DEV),
                c["up"] + G2_COLS + tid,
            )
            fx.ptr_store(fx.Float32(0.0), c["up"] + tid)
        gpu.barrier()
        return
    shared_vecs = _ceil_div(shared_intermediate, 8)
    shared_lds_vecs = _ceil_div(shared_intermediate, 32) * 4
    c["poll"]([(c["shared_mid_ready"], 0, 1)])
    if paired_shared:
        c["poll"]([(c["paired_shared_ready"], 0, 1)])
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
        if paired_shared:
            other = fx.Vector(
                bo.buffer_load(
                    rsrc(c["paired_shared_scratch"]),
                    _scratch_word(c, "shared_mid") + tid * 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            other = (tid < shared_vecs).select(other, fx.Vector.filled(4, 0, fx.Int32))
            fx.ptr_store(other, c["x"] + shared_lds_vecs * 4 + tid * 4)
    gpu.barrier()

    row = part * G2_COLS + lane % 16
    shared_w2 = c["args"]["shared_w2"]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    shared_steps = _ceil_div(shared_intermediate, 32)
    # The paired probe uses two MFMA B columns; leave all existing emitters intact.
    shared_turns = _ceil_div(shared_steps, WAVES) if paired_shared else 3
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
                c.get("weight_cache", CM_NT),
            ),
            (k0 < shared_intermediate).select(
                _ld_lds_bf8(
                    c["x"],
                    k0
                    + (
                        fx.min(lane % 16, 1) * shared_lds_vecs * 8
                        if paired_shared
                        else 0
                    ),
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
        if paired_shared:
            _st_bf(
                c["paired_shared_output"],
                part * G2_COLS + tid,
                bf16_round(row_sum(c["red"], tid, 1)),
                CM_DEV,
            )
    gpu.barrier()


@traced
def _routed_slot(c, part, slot):
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    route_weight = uniform(
        _scratch_f32(c, "route_weights", slot).bitcast(fx.Int32)
    ).bitcast(fx.Float32)
    active = route_weight > fx.Float32(0.0)
    weight_cache = c.get("weight_cache", CM_NT)
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
        fx.ptr_store(acc * scale, c["red"] + (wave * 64 + lane) * 4)
        gpu.barrier()
        if tid < G2_COLS:
            raw = row_sum(c["red"], tid, 0)
            weighted = bf16_round(raw * route_weight)
            previous = fx.ptr_load(c["up"] + tid)
            fx.ptr_store(bf16_round(previous + weighted), c["up"] + tid)
        gpu.barrier()


@traced
def _stage_g2(c, part):
    tokens, token = c.get("tokens", 1), c.get("token", 0)
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
                tokens,
                token,
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
def _take_g2_rank_s2(c):
    """Arrival rank used to place the 64 S=2 output tiles past one CTA round."""

    if c["tid"] == 0:
        # S=2 routed G1 uses token 0's queue. Token 1's word remains zero and
        # becomes the independent G2 arrival counter without growing scratch.
        at = c["g1_queue"] + fx.Int64(c["tag"] * 4)
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), at)
        rank = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        fx.ptr_store(rank, c["queue"])
    gpu.barrier()
    return fx.ptr_load(c["queue"])


@traced
def _stage_g2_s2(c0, c1, task):
    if const_expr(c0.get("expert_owned", False)):
        if task % 2 == 0:
            _stage_g2(c0, task // 2)
        else:
            _stage_g2(c1, task // 2)
        return
    first = task % 2 == 0
    selected = dict(c0)
    for field in (
        "scratch",
        "midq",
        "midq_scale",
        "route_ids",
        "route_weights",
        "shared_gate",
        "shared_mid",
    ):
        selected[field] = first.select(c0[field], c1[field])
    for field in ("shared_mid_ready", "midq_ready"):
        a, b = c0[field], c1[field]
        selected[field] = Addr(a.region, a.space, first.select(a.value, b.value))
    selected["args"] = dict(c0["args"])
    selected["args"]["moe_out"] = first.select(
        c0["args"]["moe_out"], c1["args"]["moe_out"]
    )
    _stage_g2(selected, task // 2)


@traced
def run_g2_s2(c0, c1, bid):
    """Cover 320 token/output tiles with one round plus 64 arrival tasks."""

    rank = _take_g2_rank_s2(c1)
    _stage_g2_s2(c0, c1, bid)
    if rank < G2_S2_TASKS - BLOCKS:
        _stage_g2_s2(c0, c1, BLOCKS + rank)


@traced
def run_g2_s2_partition(c0, c1, bid):
    """Run one S=2 G2 pair on 128 CTAs without a cross-pair tail."""

    rank = _take_g2_rank_s2(c1)
    _stage_g2_s2(c0, c1, bid)
    _stage_g2_s2(c0, c1, G2_S4_PAIR_BLOCKS + rank)
    if rank < G2_S2_TASKS - 2 * G2_S4_PAIR_BLOCKS:
        _stage_g2_s2(c0, c1, 2 * G2_S4_PAIR_BLOCKS + rank)
