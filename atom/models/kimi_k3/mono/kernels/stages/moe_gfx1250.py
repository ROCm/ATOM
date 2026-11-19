# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 wave32 Kimi K2 router/union/latent-quant callables.

Expert matrix and post-expert local compute live in separate architecture
modules. Persistent orchestration and XGMI publication remain refused.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.models.kimi_k3.mono.config import MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages.moe_layout import (
    EXPERTS,
    LATENT,
    MX_PACKED_WORDS,
    MX_WORDS,
    TOPK,
)
from atom.mono.device.ops import (
    CM_DEV,
    butterfly,
    hw_exp2,
    hw_rcp,
    lane_gather,
    readlane,
    rsrc,
    traced,
    uniform,
    xshfl,
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


WAVE = GFX1250.wave_size
# A survivor's aux is its expert id times SLOTS plus its LDS slot.
SLOTS = 2 * WAVE
# ``stage_route``'s ``red``: three words a survivor slot.
ROUTE_RED_WORDS = 3 * SLOTS


def _before(key, aux, other_key, other_aux):
    return (key > other_key) | ((key == other_key) & (aux < other_aux))


def _bitonic_descending(keys, auxes, lane):
    """Bitonic sort of ``len(keys)`` (key, aux) pairs a lane, element
    ``r * WAVE + lane`` in register ``r``: element ``i`` ends as the i-th by
    descending key, then ascending aux."""
    keys, auxes = list(keys), list(auxes)
    count = len(keys) * WAVE
    size = 2
    while size <= count:
        stride = size // 2
        while stride:
            if stride >= WAVE:
                step = stride // WAVE
                for low in (r for r in range(len(keys)) if not r // step % 2):
                    high = low + step
                    swap = _before(keys[high], auxes[high], keys[low], auxes[low])
                    if low * WAVE // size % 2:
                        swap = _before(keys[low], auxes[low], keys[high], auxes[high])
                    keys[low], keys[high] = (
                        swap.select(keys[high], keys[low]),
                        swap.select(keys[low], keys[high]),
                    )
                    auxes[low], auxes[high] = (
                        swap.select(auxes[high], auxes[low]),
                        swap.select(auxes[low], auxes[high]),
                    )
            else:
                for r in range(len(keys)):
                    other_key = fx.Uint32(xshfl(fx.Int32(keys[r]), stride, GFX1250))
                    other_aux = xshfl(auxes[r], stride, GFX1250)
                    mine_first = _before(keys[r], auxes[r], other_key, other_aux)
                    # 0 where this lane keeps the pair's earlier element.
                    later_here = ((lane // stride) ^ ((lane + r * WAVE) // size)) & 1
                    keep = mine_first.select(fx.Int32(0), fx.Int32(1)) == later_here
                    keys[r] = keep.select(keys[r], other_key)
                    auxes[r] = keep.select(auxes[r], other_aux)
            stride //= 2
        size *= 2
    return keys, auxes


@traced
def _ranked_survivors(scores, sigmoid, bound, kept, survivors, red, lane, registers):
    """(expert, sigmoid) of rank ``lane`` among the ``survivors`` scores at or
    above ``bound``, ``kept`` of them this lane's: compacted into LDS slots,
    then one bitonic sort of ``registers`` slots a lane."""
    slot = kept
    for step in range_constexpr(5):
        shift = 1 << step
        below = lane_gather(slot, fx.max(lane - shift, 0))
        slot = slot + (lane >= shift).select(below, fx.Int32(0))
    slot = slot - kept
    for index in range_constexpr(len(scores)):
        if scores[index] >= bound:
            expert = lane + WAVE * index
            fx.ptr_store(scores[index].bitcast(fx.Float32), red + slot * 3)
            fx.ptr_store(fx.Int32(expert).bitcast(fx.Float32), red + slot * 3 + 1)
            fx.ptr_store(sigmoid[index], red + slot * 3 + 2)
            slot = slot + 1
    keys, auxes = [], []
    for r in range_constexpr(registers):
        entry = lane + r * WAVE
        live = entry < survivors
        keys.append(
            live.select(
                fx.Uint32(fx.ptr_load(red + entry * 3).bitcast(fx.Int32)), fx.Uint32(0)
            )
        )
        # The padding sorts after every expert.
        auxes.append(
            live.select(
                fx.ptr_load(red + entry * 3 + 1).bitcast(fx.Int32), fx.Int32(EXPERTS)
            )
            * SLOTS
            + entry
        )
    keys, auxes = _bitonic_descending(keys, auxes, lane)
    return auxes[0] // SLOTS, fx.ptr_load(red + (auxes[0] % SLOTS) * 3 + 2)


def _umax_tree(values):
    while len(values) > 1:
        paired = [
            (values[i] > values[i + 1]).select(values[i], values[i + 1])
            for i in range(0, len(values) - 1, 2)
        ]
        values = paired + values[len(paired) * 2 :]
    return values[0]


def _count_tree(predicates):
    counts = [p.select(fx.Int32(1), fx.Int32(0)) for p in predicates]
    while len(counts) > 1:
        paired = [counts[i] + counts[i + 1] for i in range(0, len(counts) - 1, 2)]
        counts = paired + counts[len(paired) * 2 :]
    return counts[0]


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
    contiguous = "contiguous_logits" in c and c["contiguous_logits"]
    # HERD routing keeps TOPK + 1 candidates with their raw sigmoid scores;
    # the MoE drops one and renormalizes.
    width = TOPK + 1 if "herd" in c and c["herd"] else TOPK
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
            if const_expr(contiguous):
                location = token * EXPERTS + expert
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

        # Exact prefilter: at least ``width`` scores reach the width-th largest
        # lane maximum, so the top ``width`` are among those scores -- about 22
        # of the 896, more than a wave about once in a hundred tokens. One
        # bitonic sort of them ranks the top ``width`` by key and id; a tie
        # wider than two waves takes the radix select.
        bound = _bitonic_descending([_umax_tree(scores)], [fx.Int32(0)], lane)[0][0]
        bound = fx.Uint32(readlane(fx.Int32(bound), width - 1))
        kept = _count_tree([score >= bound for score in scores])
        # A scalar: as a DPP sum the count looks divergent to the compiler, and
        # branching on it under exec masks spills SGPRs in the lite finish.
        survivors = readlane(butterfly(kept, (16, 8, 4, 2, 1), architecture=GFX1250), 0)
        rank = fx.Int32(0)
        my_expert = fx.Int32(0)
        my_sigmoid = fx.Float32(0.0)
        if survivors <= SLOTS:
            if survivors <= WAVE:
                one = _ranked_survivors(
                    scores, sigmoid, bound, kept, survivors, red, lane, 1
                )
                my_expert = one[0]
                my_sigmoid = one[1]
            else:
                two = _ranked_survivors(
                    scores, sigmoid, bound, kept, survivors, red, lane, 2
                )
                my_expert = two[0]
                my_sigmoid = two[1]
            rank = lane
        else:

            def count_at_least(threshold):
                # Per-lane VALU counts and one wave sum: a ballot per score
                # serializes on its VALU -> SALU mask hand-off.
                count = fx.Int32(0)
                for index in range_constexpr(per_lane):
                    count = count + (scores[index] >= threshold).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                return butterfly(count, (16, 8, 4, 2, 1), architecture=GFX1250)

            # A dynamic loop: unrolled, the 32 steps are ~23 KB of straight-line
            # code that runs from a cold instruction cache after other kernels.
            threshold = fx.Uint32(0)
            for step in range(32):
                candidate = threshold | (fx.Uint32(1 << 31) >> fx.Uint32(step))
                threshold = (count_at_least(candidate) >= width).select(
                    candidate, threshold
                )

            greater = fx.Int32(0)
            for index in range_constexpr(per_lane):
                greater = greater + (scores[index] > threshold).select(
                    fx.Int32(1), fx.Int32(0)
                )
            needed = fx.Int32(width) - readlane(
                butterfly(greater, (16, 8, 4, 2, 1), architecture=GFX1250), 0
            )
            # The first ``needed`` keys at the threshold in index-major order:
            # the ranking below orders the kept set by key and id.
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

            item = fx.min(lane, width - 1)
            my_key = fx.Uint32(fx.ptr_load(red + item * 3).bitcast(fx.Int32))
            my_expert = fx.ptr_load(red + item * 3 + 1).bitcast(fx.Int32)
            my_sigmoid = fx.ptr_load(red + item * 3 + 2)
            rank = fx.Int32(0)
            for other in range_constexpr(width):
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
        if lane < width:
            expert = fx.min(fx.max(my_expert, 0), fx.Int32(EXPERTS - 1))
            if const_expr(width != TOPK):
                c["put"](c["route"], token * width + rank, expert)
                c["put"](c["route_weights"], token * width + rank, my_sigmoid)
                if const_expr("herd_candidates" in c):
                    pairs, tag = c["herd_candidates"]
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [my_sigmoid.bitcast(fx.Int32), tag | expert], fx.Int32
                        ),
                        rsrc(pairs),
                        (token * width + rank) * 2,
                        cache_modifier=CM_DEV,
                    )
            elif const_expr("route_weights" in c):
                c["put"](c["route"], token * TOPK + rank, expert)
                c["put"](c["route_weights"], token * TOPK + rank, my_sigmoid * inverse)
            else:
                c["put"](c["route"], token * 2 * TOPK + rank, expert)
                c["put"](
                    c["route"],
                    token * 2 * TOPK + TOPK + rank,
                    my_sigmoid * inverse,
                )
    gpu.barrier()


def route_cta_words(threads: int) -> dict[str, int]:
    """LDS i32 words ``stage_route_cta`` needs with ``threads`` threads."""
    return {
        "maxima": threads,
        "count": 1,
        "ahead": EXPERTS,
        "keys": EXPERTS,
        "experts": EXPERTS,
        "sigmoids": EXPERTS,
        "top_experts": WAVE,
        "top_sigmoids": WAVE,
    }


@traced
def _cta_largest32(value, nth, lane, wave, waves, greater):
    """The ``nth`` largest of a wave's 32 ``value`` (Uint32, every wave the
    same), the CTA's waves splitting the compares: each counts, for every
    lane, the lanes above its value among ``WAVE // waves`` of them (LDS
    permutes) into ``greater`` (zeroed before), and after a barrier the lanes
    with fewer than ``nth`` above are the ones at or above the nth largest,
    so it is their minimum."""
    assert WAVE % waves == 0, waves
    above = fx.Int32(0)
    for step in range_constexpr(WAVE // waves):
        peer = fx.Uint32(
            lane_gather(value, (lane + wave * (WAVE // waves) + step) % WAVE)
        )
        above = above + (peer > value).select(fx.Int32(1), fx.Int32(0))
    if above > 0:
        fx.atomic_add(greater + lane, above, syncscope="workgroup")
    gpu.barrier()
    top = fx.Int32(
        (fx.ptr_load(greater + lane) < nth).select(value, fx.Uint32(0xFFFFFFFF))
    )
    for offset in (1, 2, 4, 8, 16):
        other = xshfl(top, offset, GFX1250)
        top = fx.Int32((fx.Uint32(other) < fx.Uint32(top)).select(other, top))
    return fx.Uint32(uniform(top))


@traced
def stage_route_cta(c, token, front_payload_words=0):
    """``stage_route``'s routes, bit for bit, by the whole CTA in runtime loops:
    a cold launch fetches a few hundred instructions instead of its unrolled
    28-expert lane code and its sorts.

    A thread scores experts ``tid + threads * j``, so wave ``w`` lane ``l``
    holds some of ``stage_route`` lane ``l``'s experts and the waves' maxima
    give its lane maxima and the same bound. Each score at or above the bound
    takes an LDS slot (in any order); a survivor's rank is the count of
    survivors before it by key, then expert id (ids are distinct, so the ranks
    are the sorted order), and the top ``width`` land at their ranks."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    threads = c["threads"]
    waves = threads // WAVE
    lds = c["route_lds"]
    split_bound = c.get("split_bound", False)
    contiguous = "contiguous_logits" in c and c["contiguous_logits"]
    width = TOPK + 1 if "herd" in c and c["herd"] else TOPK
    per_thread = (EXPERTS + threads - 1) // threads
    scores = []
    sigmoid = []
    for index in range_constexpr(per_thread):
        expert = fx.min(tid + threads * index, EXPERTS - 1)
        source = expert // (EXPERTS // 8)
        location = (source * c["S"] + token) * (EXPERTS // 8) + expert % (EXPERTS // 8)
        if const_expr(front_payload_words != 0):
            location = (
                source * front_payload_words
                + token * (EXPERTS // 8)
                + expert % (EXPERTS // 8)
            )
        if const_expr(contiguous):
            location = token * EXPERTS + expert
        got = c["poll"]([(c["peer"]["rlog_own"], location, 1)])[0][0]
        logit = _f32_of(got)
        sig = hw_rcp(fx.Float32(1.0) + hw_exp2(logit * fx.Float32(-LOG2E)))
        score = sig + _load_f32(c["args"]["bias"], expert)
        live = score == score  # noqa: PLR0124 - NaN ranks last.
        key = _ordered_key(live.select(score, fx.Float32(-3.0e38)))
        if const_expr(threads * (index + 1) > EXPERTS):
            # Below every score's key (the lowest is -3e38's), so it never
            # survives.
            key = (tid + threads * index < EXPERTS).select(key, fx.Uint32(0))
        scores.append(key)
        sigmoid.append((sig == sig).select(sig, fx.Float32(0.0)))  # noqa: PLR0124
    fx.ptr_store(fx.Int32(_umax_tree(scores)), lds["maxima"] + tid)
    if tid == 0:
        fx.ptr_store(fx.Int32(0), lds["count"])
    if const_expr(split_bound):  # noqa: SIM102
        # The bound's counts (``top_experts`` is first written by the ranks).
        if tid < WAVE:
            fx.ptr_store(fx.Int32(0), lds["top_experts"] + tid)
    gpu.barrier()
    # Every wave takes the bound itself: ``stage_route``'s lane maxima, sorted.
    lane_max = fx.Uint32(fx.ptr_load(lds["maxima"] + lane))
    for other_wave in range_constexpr(1, waves):
        wave_max = fx.Uint32(fx.ptr_load(lds["maxima"] + other_wave * WAVE + lane))
        lane_max = (wave_max > lane_max).select(wave_max, lane_max)
    if const_expr(split_bound):
        bound = _cta_largest32(lane_max, width, lane, wave, waves, lds["top_experts"])
    else:
        # The lane maxima's width-th largest: the one with width - 1 ahead of it
        # by value, then lane.
        max_rank = fx.Int32(0)
        for peer_lane in range(fx.Int32(0), fx.Int32(WAVE), unroll=4):
            peer_max = fx.Uint32(readlane(fx.Int32(lane_max), peer_lane))
            max_before = (peer_max > lane_max) | (
                (peer_max == lane_max) & (peer_lane < lane)
            )
            max_rank = max_rank + max_before.select(fx.Int32(1), fx.Int32(0))
        at_bound = _ballot32(max_rank == width - 1)
        bound_lane = _popcount32((at_bound & (~at_bound + fx.Uint32(1))) - fx.Uint32(1))
        bound = fx.Uint32(readlane(fx.Int32(lane_max), bound_lane))
    # A wave's survivors take consecutive slots from one LDS atomic.
    kept = _count_tree([score >= bound for score in scores])
    inclusive = _wave_inclusive_scan32(kept, lane)
    wave_kept = readlane(inclusive, WAVE - 1)
    wave_first = fx.Int32(0)
    if (wave_kept > 0) & (lane == 0):
        wave_first = fx.atomic_add(lds["count"], wave_kept, syncscope="workgroup")
    slot = readlane(wave_first, 0) + inclusive - kept
    for index in range_constexpr(per_thread):
        if scores[index] >= bound:
            fx.ptr_store(fx.Int32(scores[index]), lds["keys"] + slot)
            fx.ptr_store(tid + threads * index, lds["experts"] + slot)
            fx.ptr_store(sigmoid[index], lds["sigmoids"] + slot)
            fx.ptr_store(fx.Int32(0), lds["ahead"] + slot)
        slot = slot + (scores[index] >= bound).select(fx.Int32(1), fx.Int32(0))
    gpu.barrier()
    survivors = uniform(fx.ptr_load(lds["count"]))
    # A survivor's rank, the count of survivors before it, summed over the
    # waves: each compares every survivor with every waves-th one.
    for first in range(fx.Int32(0), survivors, fx.Int32(WAVE), unroll=1):
        entry = fx.min(first + lane, survivors - 1)
        entry_key = fx.Uint32(fx.ptr_load(lds["keys"] + entry))
        entry_expert = fx.ptr_load(lds["experts"] + entry)
        ahead = fx.Int32(0)
        for chunk in range(fx.Int32(0), survivors, fx.Int32(WAVE), unroll=1):
            chunk_at = fx.min(chunk + lane, survivors - 1)
            chunk_keys = fx.ptr_load(lds["keys"] + chunk_at)
            chunk_experts = fx.ptr_load(lds["experts"] + chunk_at)
            for peer in range(
                wave, fx.min(survivors - chunk, WAVE), fx.Int32(waves), unroll=1
            ):
                peer_key = fx.Uint32(readlane(chunk_keys, peer))
                peer_expert = readlane(chunk_experts, peer)
                entry_before = (peer_key > entry_key) | (
                    (peer_key == entry_key) & (peer_expert < entry_expert)
                )
                ahead = ahead + entry_before.select(fx.Int32(1), fx.Int32(0))
        if (first + lane < survivors) & (ahead > 0):
            fx.atomic_add(lds["ahead"] + entry, ahead, syncscope="workgroup")
    gpu.barrier()
    if wave == 0:
        # The top ``width`` to their ranks: a wave's LDS accesses are in order.
        for first in range(fx.Int32(0), survivors, fx.Int32(WAVE), unroll=1):
            entry = fx.min(first + lane, survivors - 1)
            entry_rank = fx.ptr_load(lds["ahead"] + entry)
            if (first + lane < survivors) & (entry_rank < width):
                fx.ptr_store(
                    fx.ptr_load(lds["experts"] + entry), lds["top_experts"] + entry_rank
                )
                fx.ptr_store(
                    fx.ptr_load(lds["sigmoids"] + entry),
                    lds["top_sigmoids"] + entry_rank,
                )
        rank = lane
        top = fx.min(lane, width - 1)
        my_expert = fx.ptr_load(lds["top_experts"] + top)
        my_sigmoid = fx.ptr_load(lds["top_sigmoids"] + top)
        total = fx.Float32(0.0)
        for item in range_constexpr(TOPK):
            total = total + fx.ptr_load(lds["top_sigmoids"] + item)
        inverse = fx.Float32(1.0) / total
        if lane < width:
            expert = fx.min(fx.max(my_expert, 0), fx.Int32(EXPERTS - 1))
            if const_expr(width != TOPK):
                c["put"](c["route"], token * width + rank, expert)
                c["put"](c["route_weights"], token * width + rank, my_sigmoid)
                if const_expr("herd_candidates" in c):
                    pairs, tag = c["herd_candidates"]
                    bo.buffer_store(
                        fx.Vector.from_elements(
                            [my_sigmoid.bitcast(fx.Int32), tag | expert], fx.Int32
                        ),
                        rsrc(pairs),
                        (token * width + rank) * 2,
                        cache_modifier=CM_DEV,
                    )
            elif const_expr("route_weights" in c):
                c["put"](c["route"], token * TOPK + rank, expert)
                c["put"](c["route_weights"], token * TOPK + rank, my_sigmoid * inverse)
            else:
                c["put"](c["route"], token * 2 * TOPK + rank, expert)
                c["put"](
                    c["route"],
                    token * 2 * TOPK + TOPK + rank,
                    my_sigmoid * inverse,
                )
    gpu.barrier()


# A HERD candidate's tag word: the launch's tag over the expert id's bits.
_HERD_EXPERT_BITS = 10
assert EXPERTS <= 1 << _HERD_EXPERT_BITS
# ``herd_finalize_words``' tail: the staged kept routes and their expert slots.
_HERD_ROUTE_SLOTS = 48 * TOPK
# A sort key's candidate index bits, below its expert id.
_HERD_KEY_SHIFT = 5


def _sort_network(n: int) -> list[tuple[int, int]]:
    """Batcher's odd-even merge sort over the next power of two, without the
    comparators past ``n`` (those positions would hold +inf and never swap)."""
    size = 1 << (n - 1).bit_length()
    pairs = []
    p = 1
    while p < size:
        k = p
        while k:
            for j in range(k % p, size - k, 2 * k):
                for i in range(min(k, size - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return [(a, b) for a, b in pairs if b < n]


def herd_tag(launch):
    """Launch ``launch``'s candidate tag (the counter zeroed, then one up a
    launch): never the last launch's, nor zeroed scratch's."""
    return (launch + 1) << _HERD_EXPERT_BITS


def herd_finalize_words(rows: int) -> dict[str, int]:
    """LDS i32 words ``stage_herd_finalize`` needs for ``rows`` tokens: the
    kept routes' at least 48 rows' (one layout up to 48)."""
    if rows > MAX_ROWS:
        raise ValueError(f"HERD finalize stages at most {MAX_ROWS} rows, got {rows}")
    slots = max(_HERD_ROUTE_SLOTS, rows * TOPK)
    return {
        "popularity": EXPERTS,
        "counts": EXPERTS,
        "starts": EXPERTS,
        "scan": 8,
        "ids": rows * (TOPK + 1),
        "values": rows * (TOPK + 1),
        "kept_ids": slots,
        "kept_weights": slots,
        "slots": slots,
    }


@traced
def _herd_drop_scan(c, my_id, my_value, my_pop, lane, live, kp1):
    """``stage_herd_token``'s drop, kept sum and places on wave 0, by scans."""
    # Runtime loops, in candidate order: a cold launch fetches one body.
    drop = fx.Int32(0)
    drop_id = readlane(my_id, 0)
    drop_value = readlane(my_value, 0)
    drop_pop = readlane(my_pop, 0)
    for j in range(fx.Int32(1), fx.Int32(kp1), unroll=1):
        j_id, j_value, j_pop = readlane(my_id, j), readlane(my_value, j), readlane(my_pop, j)
        take = (j_pop < drop_pop) | (
            (j_pop == drop_pop)
            & ((j_value < drop_value) | ((j_value == drop_value) & (j_id < drop_id)))
        )
        drop = take.select(j, drop)
        drop_id = take.select(j_id, drop_id)
        drop_value = take.select(j_value, drop_value)
        drop_pop = take.select(j_pop, drop_pop)
    kept_sum = fx.Float32(0.0)
    place = fx.Int32(0)
    for k in range(fx.Int32(0), fx.Int32(kp1), unroll=1):
        kept = drop != k
        kept_sum = kept_sum + kept.select(readlane(my_value, k), fx.Float32(0.0))
        below = kept & (readlane(my_id, k) < my_id)
        place = place + below.select(fx.Int32(1), fx.Int32(0))
    kept = live & (lane != drop)
    return kept_sum, place, kept


def _herd_drop_wave(c, my_id, my_value, my_pop, lane, live, kp1):
    """``stage_herd_token``'s drop, kept sum and places on wave 0, bit for bit,
    without its serial scans: the drop the arg-min of (popularity, value, id)
    over the live lanes by a butterfly (the ids are distinct, so the order is
    total and any comparison order finds the scan's candidate); the kept sum
    the scan's adds in candidate order; a kept candidate's place the kept ones
    of lower id, ranked by ballots of the ids' bits, most significant first."""
    key_pop = live.select(my_pop, fx.Int32(0x7FFFFFFF))
    key_value = live.select(my_value, fx.Float32(math.inf))
    key_id = live.select(my_id, fx.Int32(0x7FFFFFFF))
    for offset in (1, 2, 4, 8, 16):
        o_pop = xshfl(key_pop, offset, GFX1250)
        o_value = xshfl(key_value, offset, GFX1250)
        o_id = xshfl(key_id, offset, GFX1250)
        take = (o_pop < key_pop) | (
            (o_pop == key_pop)
            & ((o_value < key_value) | ((o_value == key_value) & (o_id < key_id)))
        )
        key_pop = take.select(o_pop, key_pop)
        key_value = take.select(o_value, key_value)
        key_id = take.select(o_id, key_id)
    drop_id = uniform(key_id)
    kept_sum = fx.Float32(0.0)
    for k in range(kp1):
        kept_sum = kept_sum + (readlane(my_id, k) != drop_id).select(
            readlane(my_value, k), fx.Float32(0.0)
        )
    kept = live & (my_id != drop_id)
    same = _ballot32(kept)
    place = fx.Int32(0)
    for bit in range(_HERD_EXPERT_BITS - 1, -1, -1):
        one = ((my_id >> bit) & 1) != 0
        zeros = _ballot32(((my_id >> bit) & 1) == 0)
        place = place + one.select(_popcount32(same & zeros), fx.Int32(0))
        same = one.select(same & (zeros ^ fx.Uint32(0xFFFFFFFF)), same & zeros)
    return kept_sum, place, kept


@traced
def stage_herd_token(c, rows, token):
    """``stage_herd_finalize``'s ``topk_ids`` and ``topk_weights`` of token
    ``token`` alone, bit for bit, by its routing CTA: it polls every token's
    candidates as the HERD CTA does, counts the batch popularity in its own
    LDS and drops and orders its own token's, a candidate a lane of wave 0.
    ``masked_m``, ``psum`` and ``topids_to_rows`` are left alone (only
    AITER's grouped MoE reads them).
    ``c`` as ``stage_herd_finalize``'s, its LDS ``popularity``, ``ids`` and
    ``values`` used; ``c["poll_limit"]`` (optional) bounds the polls, past it
    the stale candidates are taken as they are."""

    tid, threads = c["tid"], c["threads"]
    wave_drop = c.get("wave_drop", False)
    lds = c["herd_lds"]
    popularity, ids_s, values_s = lds["popularity"], lds["ids"], lds["values"]
    kp1 = TOPK + 1
    candidates = rows * kp1

    for index in range_constexpr((EXPERTS + threads - 1) // threads):
        expert = tid + index * threads
        if expert < EXPERTS:
            fx.ptr_store(fx.Int32(0), popularity + expert)
    pairs, tag = c["candidates"]
    rounds = (candidates + threads - 1) // threads

    def load_pairs():
        words = []
        for index in range_constexpr(rounds):
            at = fx.min(tid + index * threads, candidates - 1)
            pair = fx.Vector(
                bo.buffer_load(
                    rsrc(pairs), at * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV
                )
            )
            words += [pair[0], pair[1]]
        return fx.Vector.from_elements(words, fx.Int32)

    tag_bits = fx.Int32(-(1 << _HERD_EXPERT_BITS))

    def stale(words):
        late = (fx.Int32(words[1]) & tag_bits) != tag
        for index in range_constexpr(1, rounds):
            late = late | ((fx.Int32(words[2 * index + 1]) & tag_bits) != tag)
        return late

    limit = c.get("poll_limit")
    words = load_pairs()
    if const_expr(limit is None):
        while stale(words):
            rocdl.s_sleep(1)
            words = load_pairs()
    else:
        polls = fx.Int32(0)
        while stale(words) & (polls < limit):
            rocdl.s_sleep(1)
            polls = polls + 1
            words = load_pairs()
    if const_expr("stamp" in c):
        c["stamp"]("herd_in")
    gpu.barrier()
    for index in range_constexpr(rounds):
        at = tid + index * threads
        if at < candidates:
            expert = fx.Int32(words[2 * index + 1]) & ((1 << _HERD_EXPERT_BITS) - 1)
            fx.ptr_store(expert, ids_s + at)
            fx.ptr_store(fx.Int32(words[2 * index]).bitcast(fx.Float32), values_s + at)
            fx.atomic_add(popularity + expert, fx.Int32(1), syncscope="workgroup")
    gpu.barrier()
    store_cm = c.get("store_cm", 0)
    drop_order = _herd_drop_wave if wave_drop else _herd_drop_scan
    if c["wave"] == 0:
        # Lane j holds candidate j; the drop, the kept sum and the scale are
        # ``stage_herd_finalize``'s scan on the wave-uniform copies, in its
        # order, and a kept candidate's place is the count of kept ones of
        # lower id (the ids are distinct, so that is the sorted keys' order).
        lane = c["lane"]
        base = token * kp1
        mine = fx.min(lane, kp1 - 1)
        my_id = fx.ptr_load(ids_s + base + mine)
        my_value = fx.ptr_load(values_s + base + mine)
        my_pop = fx.ptr_load(popularity + my_id)
        live = lane < kp1
        kept_sum, place, kept = drop_order(c, my_id, my_value, my_pop, lane, live, kp1)
        scale = c["routed_scaling_factor"] / fx.max(kept_sum, fx.Float32(1e-20))
        if kept:
            bo.buffer_store(
                my_id, rsrc(c["topk_ids"]), token * TOPK + place, cache_modifier=store_cm
            )
            bo.buffer_store(
                my_value * scale,
                rsrc(c["topk_weights"]),
                token * TOPK + place,
                cache_modifier=store_cm,
            )


@traced
def stage_herd_finalize(c, rows):
    """AITER's ``herd_route_psum`` for this launch's Top-(K+1) candidates, by
    one CTA polling the routing CTAs' tagged (sigmoid, tag | expert) pairs.

    As AITER drops it, each token loses its candidate of the lowest batch
    popularity (then value, then expert id); the K kept are ordered by expert
    id and weighted by their sigmoid over the kept sum taken in candidate order,
    so ids and weights are bit-identical. ``psum`` and ``masked_m`` are the
    ``tile_m``-aligned grouped layout, and ``topids_to_rows`` a route's row in
    it; a route's slot within its expert is taken by an LDS atomic, as AITER's
    is, so only the order inside an expert may differ, which no GEMM row reads
    across."""

    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    threads, waves = c["threads"], c["waves"]
    lds = c["herd_lds"]
    popularity, counts, starts = lds["popularity"], lds["counts"], lds["starts"]
    ids_s, values_s = lds["ids"], lds["values"]
    kept_ids, kept_weights, slots = lds["kept_ids"], lds["kept_weights"], lds["slots"]
    kp1 = TOPK + 1
    candidates = rows * kp1
    routes = rows * TOPK

    for index in range_constexpr((EXPERTS + threads - 1) // threads):
        expert = tid + index * threads
        if expert < EXPERTS:
            fx.ptr_store(fx.Int32(0), popularity + expert)
            fx.ptr_store(fx.Int32(0), counts + expert)
    pairs, tag = c["candidates"]
    rounds = (candidates + threads - 1) // threads

    def load_pairs():
        words = []
        for index in range_constexpr(rounds):
            at = fx.min(tid + index * threads, candidates - 1)
            pair = fx.Vector(
                bo.buffer_load(
                    rsrc(pairs), at * 2, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV
                )
            )
            words += [pair[0], pair[1]]
        return fx.Vector.from_elements(words, fx.Int32)

    tag_bits = fx.Int32(-(1 << _HERD_EXPERT_BITS))

    def stale(words):
        late = (fx.Int32(words[1]) & tag_bits) != tag
        for index in range_constexpr(1, rounds):
            late = late | ((fx.Int32(words[2 * index + 1]) & tag_bits) != tag)
        return late

    words = load_pairs()
    while stale(words):
        rocdl.s_sleep(1)
        words = load_pairs()
    if const_expr("stamp" in c):
        c["stamp"]("herd_in")
    loaded = [
        (
            fx.Int32(words[2 * index + 1]) & ((1 << _HERD_EXPERT_BITS) - 1),
            fx.Int32(words[2 * index]).bitcast(fx.Float32),
        )
        for index in range(rounds)
    ]
    gpu.barrier()
    for index in range_constexpr(len(loaded)):
        at = tid + index * threads
        if at < candidates:
            expert, value = loaded[index]
            fx.ptr_store(expert, ids_s + at)
            fx.ptr_store(value, values_s + at)
            fx.atomic_add(popularity + expert, fx.Int32(1), syncscope="workgroup")
    gpu.barrier()

    if tid < rows:
        base = tid * kp1
        ids = [fx.ptr_load(ids_s + base + j) for j in range_constexpr(kp1)]
        values = [fx.ptr_load(values_s + base + j) for j in range_constexpr(kp1)]
        pops = [fx.ptr_load(popularity + ids[j]) for j in range_constexpr(kp1)]
        drop = fx.Int32(0)
        drop_id, drop_value, drop_pop = ids[0], values[0], pops[0]
        for j in range_constexpr(1, kp1):
            take = (pops[j] < drop_pop) | (
                (pops[j] == drop_pop)
                & (
                    (values[j] < drop_value)
                    | ((values[j] == drop_value) & (ids[j] < drop_id))
                )
            )
            drop = take.select(fx.Int32(j), drop)
            drop_id = take.select(ids[j], drop_id)
            drop_value = take.select(values[j], drop_value)
            drop_pop = take.select(pops[j], drop_pop)
        kept_sum = fx.Float32(0.0)
        for j in range_constexpr(kp1):
            kept_sum = kept_sum + (drop != fx.Int32(j)).select(
                values[j], fx.Float32(0.0)
            )
        scale = c["routed_scaling_factor"] / fx.max(kept_sum, fx.Float32(1e-20))
        # The ids are distinct, so (id, candidate) keys sort to one order; the
        # K kept are the sorted keys with the dropped one skipped.
        keys = [(ids[j] << _HERD_KEY_SHIFT) | fx.Int32(j) for j in range_constexpr(kp1)]
        for a, b in _sort_network(kp1):
            keys[a], keys[b] = fx.min(keys[a], keys[b]), fx.max(keys[a], keys[b])
        drop_key = (drop_id << _HERD_KEY_SHIFT) | drop
        kept = [
            (keys[p] < drop_key).select(keys[p], keys[p + 1])
            for p in range_constexpr(TOPK)
        ]
        candidate = fx.Int32((1 << _HERD_KEY_SHIFT) - 1)
        weights = [
            fx.ptr_load(values_s + base + (kept[p] & candidate)) * scale
            for p in range_constexpr(TOPK)
        ]
        experts = [kept[p] >> _HERD_KEY_SHIFT for p in range_constexpr(TOPK)]
        # Every slot is taken before any is stored, so the LDS atomics are in
        # flight together.
        taken = [
            fx.atomic_add(counts + experts[p], fx.Int32(1), syncscope="workgroup")
            for p in range_constexpr(TOPK)
        ]
        for p in range_constexpr(TOPK):
            at = tid * TOPK + p
            fx.ptr_store(experts[p], kept_ids + at)
            fx.ptr_store(weights[p], kept_weights + at)
            fx.ptr_store(taken[p], slots + at)
    gpu.barrier()

    # read here: a method called in a branch makes ``c`` its traced state
    store_cm = c.get("store_cm", 0)
    for index in range_constexpr((routes + threads - 1) // threads):
        at = tid + index * threads
        if at < routes:
            bo.buffer_store(
                fx.ptr_load(kept_ids + at),
                rsrc(c["topk_ids"]),
                at,
                cache_modifier=store_cm,
            )
            bo.buffer_store(
                fx.ptr_load(kept_weights + at),
                rsrc(c["topk_weights"]),
                at,
                cache_modifier=store_cm,
            )
    # Tile-aligned exclusive scan of the experts' route counts, a run of
    # experts a thread.
    per = (EXPERTS + threads - 1) // threads
    first = tid * per
    counted = []
    aligned_total = fx.Int32(0)
    tile = c["tile_m"]
    for index in range_constexpr(per):
        expert = first + index
        count = (expert < EXPERTS).select(
            fx.ptr_load(counts + fx.min(expert, EXPERTS - 1)), fx.Int32(0)
        )
        counted.append(count)
        aligned_total = aligned_total + (count + tile - 1) // tile * tile
    start, _ = _block_exclusive_scan32(aligned_total, lane, wave, lds["scan"], waves)
    for index in range_constexpr(per):
        expert = first + index
        if expert < EXPERTS:
            fx.ptr_store(start, starts + expert)
            bo.buffer_store(counted[index], rsrc(c["masked_m"]), expert)
            bo.buffer_store(start + counted[index], rsrc(c["psum"]), expert)
        start = start + (counted[index] + tile - 1) // tile * tile
    gpu.barrier()
    for index in range_constexpr((routes + threads - 1) // threads):
        at = tid + index * threads
        if at < routes:
            expert = fx.ptr_load(kept_ids + at)
            bo.buffer_store(
                fx.ptr_load(starts + expert) + fx.ptr_load(slots + at),
                rsrc(c["topids_to_rows"]),
                at,
            )


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
