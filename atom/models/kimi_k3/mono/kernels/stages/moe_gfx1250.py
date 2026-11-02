# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 wave32 Kimi K2 router/union/latent-quant callables.

Expert matrix and post-expert local compute live in separate architecture
modules. Persistent orchestration and XGMI publication remain refused.
"""

import math
from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.kernels.stages.moe_layout import (
    EXPERTS,
    LATENT,
    MX_PACKED_WORDS,
    MX_WORDS,
    TOPK,
    LatentQuantContract,
    RouterContract,
    RouteUnionContract,
)
from atom.mono.device.ops import (
    hw_exp2,
    hw_rcp,
    rsrc,
    traced,
    uniform,
)
from atom.mono.plan.architecture import GFX1250

LOG2E = 1.0 / math.log(2.0)


def _ballot32(predicate):
    """gfx1250 ballot ABI used by AITER MegaMoE and FMHA."""

    return fx.Uint32(rocdl.ballot(T.i32, predicate))


def _popcount32(mask):
    return fx.Int32(fx.ctpop(mask))


def _lanes_below32(mask, lane):
    below = (fx.Uint32(1) << fx.Uint32(lane)) - fx.Uint32(1)
    return _popcount32(mask & below)


def _wave_inclusive_scan32(value, lane):
    """Inclusive lane-order scan using gfx1250-evidenced shuffle_xor."""

    block = value
    inclusive = value
    for offset in (1, 2, 4, 8, 16):
        peer = block.shuffle_xor(offset, GFX1250.wave_size)
        inclusive = inclusive + ((lane & offset) != 0).select(peer, fx.Int32(0))
        block = block + peer
    return inclusive


@traced
def _block_exclusive_scan32(value, lane, wave, scratch, waves):
    inclusive = _wave_inclusive_scan32(value, lane)
    gpu.barrier()
    if lane == GFX1250.wave_size - 1:
        fx.ptr_store(inclusive, scratch + wave)
    gpu.barrier()
    wave_offset = fx.Int32(0)
    total = fx.Int32(0)
    for other_wave in range_constexpr(waves):
        count = fx.ptr_load(scratch + other_wave)
        wave_offset = wave_offset + (other_wave < wave).select(count, fx.Int32(0))
        total = total + count
    return inclusive - value + wave_offset, total


def _f32_of(word):
    return word.bitcast(fx.Float32)


def _load_f32(pointer, index):
    return fx.Float32(bo.buffer_load(rsrc(pointer), index, vec_width=1, dtype=T.f32))


def _ordered_key(value):
    bits = value.bitcast(fx.Int32)
    negative = bits < fx.Int32(0)
    return fx.Uint32(negative.select(~bits, bits ^ fx.Int32(-2147483648)))


@traced
def stage_route(c, token, front_payload_words=0):
    """Wave32 router/top-16 with the production key, tie, and sum order."""

    lane, wave, red = c["lane"], c["wave"], c["red"]
    if wave == 0:
        per_lane = EXPERTS // GFX1250.wave_size
        scores = []
        sigmoid = []
        for index in range_constexpr(per_lane):
            expert = lane + GFX1250.wave_size * index
            source = expert // (EXPERTS // 8)
            location = (source * c["S"] + token) * (EXPERTS // 8) + expert % (
                EXPERTS // 8
            )
            if const_expr(front_payload_words != 0):
                location = (
                    source * front_payload_words
                    + token * (EXPERTS // 8)
                    + expert % (EXPERTS // 8)
                )
            got = c["poll"](
                [
                    (
                        c["peer"]["rlog_own"],
                        location,
                        1,
                    )
                ]
            )[
                0
            ][0]
            logit = _f32_of(got)
            sig = hw_rcp(fx.Float32(1.0) + hw_exp2(logit * fx.Float32(-LOG2E)))
            score = sig + _load_f32(c["args"]["bias"], expert)
            live = score == score  # noqa: PLR0124 - NaN ranks last.
            scores.append(_ordered_key(live.select(score, fx.Float32(-3.0e38))))
            sigmoid.append(
                (sig == sig).select(sig, fx.Float32(0.0))  # noqa: PLR0124 - NaN guard.
            )

        def count_at_least(threshold):
            count = fx.Int32(0)
            for index in range_constexpr(per_lane):
                count = count + _popcount32(_ballot32(scores[index] >= threshold))
            return count

        threshold = fx.Uint32(0)
        for bit in range_constexpr(31, -1, -1):
            candidate = threshold | fx.Uint32(1 << bit)
            threshold = (count_at_least(candidate) >= TOPK).select(candidate, threshold)

        greater = fx.Int32(0)
        for index in range_constexpr(per_lane):
            greater = greater + _popcount32(_ballot32(scores[index] > threshold))
        needed = fx.Int32(TOPK) - greater
        equal_before = fx.Int32(0)
        selected_before = fx.Int32(0)
        for index in range_constexpr(per_lane):
            equal_ballot = _ballot32(scores[index] == threshold)
            equal_rank = equal_before + _lanes_below32(equal_ballot, lane)
            selected = (scores[index] > threshold) | (
                (scores[index] == threshold) & (equal_rank < needed)
            )
            selected_ballot = _ballot32(selected)
            position = selected_before + _lanes_below32(selected_ballot, lane)
            if selected:
                expert = lane + GFX1250.wave_size * index
                fx.ptr_store(
                    scores[index].bitcast(fx.Float32),
                    red + position * 3,
                )
                fx.ptr_store(
                    fx.Int32(expert).bitcast(fx.Float32),
                    red + position * 3 + 1,
                )
                fx.ptr_store(sigmoid[index], red + position * 3 + 2)
            equal_before = equal_before + _popcount32(equal_ballot)
            selected_before = selected_before + _popcount32(selected_ballot)

        item = fx.min(lane, TOPK - 1)
        my_key = fx.Uint32(fx.ptr_load(red + item * 3).bitcast(fx.Int32))
        my_expert = fx.ptr_load(red + item * 3 + 1).bitcast(fx.Int32)
        my_sigmoid = fx.ptr_load(red + item * 3 + 2)
        rank = fx.Int32(0)
        for other in range_constexpr(TOPK):
            other_key = fx.Uint32(fx.ptr_load(red + other * 3).bitcast(fx.Int32))
            other_expert = fx.ptr_load(red + other * 3 + 1).bitcast(fx.Int32)
            before = (other_key > my_key) | (
                (other_key == my_key) & (other_expert < my_expert)
            )
            rank = rank + before.select(fx.Int32(1), fx.Int32(0))
        if lane < TOPK:
            fx.ptr_store(my_sigmoid, red + 64 + rank)
        total = fx.Float32(0.0)
        for item in range_constexpr(TOPK):
            total = total + fx.ptr_load(red + 64 + item)
        inverse = fx.Float32(1.0) / total
        if lane < TOPK:
            c["put"](
                c["route"],
                token * 2 * TOPK + rank,
                fx.min(fx.max(my_expert, 0), fx.Int32(EXPERTS - 1)),
            )
            c["put"](
                c["route"],
                token * 2 * TOPK + TOPK + rank,
                my_sigmoid * inverse,
            )
    gpu.barrier()


@traced
def load_route(c, threads, waves):
    """Build an ascending expert union with a wave32 CTA-wide scan."""

    tokens, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    route = c["rt"]
    route_words = tokens * 2 * TOPK
    for round_index in range_constexpr((route_words + threads - 1) // threads):
        index = tid + threads * round_index
        if index < route_words:
            value = c["poll"]([(c["route"], index, 1)])[0][0]
            fx.ptr_store(value, route["route"] + index)

    for round_index in range_constexpr((EXPERTS + threads - 1) // threads):
        expert = tid + threads * round_index
        if expert < EXPERTS:
            fx.ptr_store(fx.Int32(0), route["flag"] + expert)
    union_bound = min(tokens * TOPK, EXPERTS)
    kof_words = union_bound * tokens
    for round_index in range_constexpr((kof_words + threads - 1) // threads):
        index = tid + threads * round_index
        if index < kof_words:
            fx.ptr_store(fx.Int32(-1), route["kof"] + index)
    gpu.barrier()

    picks = tokens * TOPK
    for round_index in range_constexpr((picks + threads - 1) // threads):
        pick = tid + threads * round_index
        if pick < picks:
            token = pick // TOPK
            topk = pick % TOPK
            expert = fx.max(
                fx.min(
                    fx.ptr_load(route["route"] + token * 2 * TOPK + topk),
                    EXPERTS - 1,
                ),
                0,
            )
            fx.ptr_store(fx.Int32(1), route["flag"] + expert)
    gpu.barrier()

    chunk = (EXPERTS + threads - 1) // threads
    first = tid * chunk
    local_count = fx.Int32(0)
    flags = []
    for offset in range_constexpr(chunk):
        expert = fx.min(first + offset, EXPERTS - 1)
        live = first + offset < EXPERTS
        flag = live.select(fx.ptr_load(route["flag"] + expert), fx.Int32(0))
        flags.append(flag)
        local_count = local_count + flag
    union_offset, union_size = _block_exclusive_scan32(
        local_count,
        lane,
        wave,
        c["scan"],
        waves,
    )
    local_offset = fx.Int32(0)
    for offset in range_constexpr(chunk):
        expert = first + offset
        if flags[offset] == 1:
            slot = union_offset + local_offset
            fx.ptr_store(slot, route["flag"] + expert)
            fx.ptr_store(fx.Int32(expert), route["expert"] + slot)
            local_offset = local_offset + 1
    gpu.barrier()

    for round_index in range_constexpr((picks + threads - 1) // threads):
        pick = tid + threads * round_index
        if pick < picks:
            token = pick // TOPK
            topk = pick % TOPK
            expert = fx.max(
                fx.min(
                    fx.ptr_load(route["route"] + token * 2 * TOPK + topk),
                    EXPERTS - 1,
                ),
                0,
            )
            slot = fx.ptr_load(route["flag"] + expert)
            fx.ptr_store(fx.Int32(topk), route["kof"] + slot * tokens + token)
    gpu.barrier()
    return uniform(union_size)


def _e8m0_roundup(amax):
    """Preserve Kimi's ceil(amax/6) exponent code, capped at 254."""

    bits = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    code = fx.Int32(fx.Uint32(bits + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)) & 0xFF
    return fx.min(code, fx.Int32(254))


def _emit_f32_to_e2m1(value):
    """RNE E2M1 conversion using current FlyDSL Numeric bitcast types."""

    denorm_bias = fx.Int32(0x4A800000)
    bits = fx.Float32(value).bitcast(fx.Int32)
    sign = bits & fx.Int32(-2147483648)
    absolute = bits & fx.Int32(0x7FFFFFFF)
    absolute_u = fx.Uint32(absolute)
    denormal = absolute_u < fx.Uint32(0x3F800000)
    normal = (absolute_u < fx.Uint32(0x40C00000)) & (
        absolute_u >= fx.Uint32(0x3F800000)
    )
    denormal_f32 = absolute.bitcast(fx.Float32) + denorm_bias.bitcast(fx.Float32)
    denormal_x = denormal_f32.bitcast(fx.Int32) - denorm_bias
    mantissa_odd = fx.Int32((absolute_u >> fx.Uint32(22)) & fx.Uint32(1))
    normal_x = fx.Uint32(absolute + fx.Int32(-1054867457) + mantissa_odd) >> fx.Uint32(
        22
    )
    e2m1 = normal.select(normal_x, fx.Uint32(7))
    e2m1 = denormal.select(fx.Uint32(denormal_x), e2m1)
    return fx.Int32((fx.Uint32(sign) >> fx.Uint32(28)) | fx.Uint32(e2m1))


def _pack_bf16_words(words, reciprocal_scale):
    packed = fx.Int32(0)
    for word_index in range_constexpr(4):
        pair = fx.Vector.from_elements([fx.Int32(words[word_index])], fx.Int32).bitcast(
            fx.BFloat16
        )
        low = fx.Int32(_emit_f32_to_e2m1(fx.Float32(pair[0]) * reciprocal_scale))
        high = fx.Int32(_emit_f32_to_e2m1(fx.Float32(pair[1]) * reciprocal_scale))
        packed = packed | (low << (8 * word_index))
        packed = packed | (high << (8 * word_index + 4))
    return packed


@traced
def load_latq(c, threads):
    """Quantize latent rows in 32-element groups using evidenced gfx1250 E2M1."""

    tokens, tid = c["S"], c["tid"]
    blocks = tokens * (LATENT // 32)
    for round_index in range_constexpr((blocks + threads - 1) // threads):
        block = fx.min(tid + threads * round_index, blocks - 1)
        token = block // (LATENT // 32)
        group = block % (LATENT // 32)
        base = (token * LATENT + group * 32) // 2
        polled = c["poll"](
            [(c["peer"]["lat_own"], base + index, 1) for index in range(MX_WORDS)]
        )
        words = [polled[index][0] for index in range(MX_WORDS)]
        maximum_bits = fx.Int32(0)
        for word in words:
            maximum_bits = fx.max(maximum_bits, word & 0x7FFF)
            maximum_bits = fx.max(maximum_bits, (word >> 16) & 0x7FFF)
        maximum = (maximum_bits << 16).bitcast(fx.Float32)
        code = _e8m0_roundup(maximum)
        reciprocal_scale = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        if tid + threads * round_index < blocks:
            for packed_word in range_constexpr(MX_PACKED_WORDS):
                value = _pack_bf16_words(
                    words[4 * packed_word : 4 * packed_word + 4],
                    reciprocal_scale,
                )
                fx.ptr_store(
                    value,
                    c["latq"]
                    + token * (LATENT // 8)
                    + group * MX_PACKED_WORDS
                    + packed_word,
                )
            fx.ptr_store(
                code,
                c["latsc"] + token * (LATENT // 32) + group,
            )
    gpu.barrier()


@dataclass(frozen=True)
class StageBackend:
    router: RouterContract
    route_union: RouteUnionContract
    latent_quant: LatentQuantContract

    def __post_init__(self):
        for architecture in (
            self.router.architecture,
            self.route_union.architecture,
            self.latent_quant.architecture,
        ):
            if architecture is not GFX1250:
                raise ValueError("gfx1250 K2 backend requires GFX1250 contracts")
        if self.route_union.threads != self.latent_quant.threads:
            raise ValueError("K2 route and quant thread counts differ")
        if self.route_union.tokens != self.latent_quant.tokens:
            raise ValueError("K2 route and quant token counts differ")

    @staticmethod
    def stage_route(c, token):
        stage_route(c, token)

    @property
    def token_tiles(self):
        return self.route_union.token_tiles

    @staticmethod
    def _require_tile_context(c):
        if c["S"] > 16:
            raise ValueError(
                "gfx1250 K2 front device callables require an M16 tile "
                "context for S > 16"
            )

    def load_route(self, c):
        self._require_tile_context(c)
        return load_route(c, self.route_union.threads, self.route_union.waves)

    def load_latq(self, c):
        self._require_tile_context(c)
        load_latq(c, self.latent_quant.threads)
