# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 routed experts of a whole decode step as one weight stream: the
gate/up (``build_expert_up``), the down with its gather-reduce
(``build_expert_down``), and both in one resident launch (``build_moe``),
each bit-identical to AITER's grouped MoE.

The gate/up: the step's expert union is built in each CTA's LDS from the routes (every
token's TOPK ids), with each expert's picks -- its (token, k) routes -- listed
contiguously, so an expert's weights are read once whatever the tokens. A task
is 256 gate|up rows (AITER's GUGU interleave) of one expert: wave w owns 32
rows and walks the whole K, its WMMA accumulating K in order. The rows stream
through the Tensor Data Mover a 512-column K tile at a time -- one 2-D copy of
the task's sixteen 16-row groups, each contiguous in AITER's preshuffle -- into
a ring of LDS stages that runs on across the CTA's tasks, so the next task's
first tiles are in flight under this one's last. Every token's MXFP4 latent
sits in LDS for the whole launch; a WMMA's tokens are its expert's picks, at
most three 16-pick tiles against the same LDS weights.

The epilogue is AITER gemm1's fused one (``stage1_act=3``, ``stage1_quant_out``):
its SiTUv2, each 32-intermediate block's E8M0 from its amax and the scaled
BF16 -> FP4 conversion -- a 64-row pair of waves a block -- stored a pick:
``inter[pick][48 words]``, ``isc[pick][12]`` (E8M0 bytes).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import batched_situv2, situv2_consts
from aiter.ops.flydsl.kernels.quant_utils import (
    emit_amax_e8m0_native_scale,
    emit_cvt_scalef32_pk8_fp4_bf16,
)
from aiter.utility.mx_types import MxDtypeInt
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.config import SITU_BETA, SITU_LINEAR_BETA
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages.moe_expert_gfx1250 import (
    _activation_fragment,
    _weight_scale_word,
    _wmma,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_gfx1250 import _block_exclusive_scan32
from atom.models.kimi_k3.mono.kernels.stages.moe_layout import (
    EXPERTS,
    LATENT,
    TOPK,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device import queue as task_queue
from atom.mono.device.ops import (
    CM_DEV,
    ld_dev,
    readlane,
    rsrc,
    st_dev,
    traced,
    uniform,
)
from atom.mono.plan.architecture import GFX1250

THREADS = 256
WAVES = THREADS // GFX1250.wave_size
TASK_ROWS = WAVES * 32
GROUPS = TASK_ROWS // 16
K_TILE = 512
K_STEP = 128
STEPS = K_TILE // K_STEP
GROUP_TILE_BYTES = 16 * K_TILE // 2
STAGE_BYTES = GROUPS * GROUP_TILE_BYTES
STAGES = 3
# A pick's gate/up output where a down on another CTA reads it: 48 FP4 words,
# then its 12 E8M0 codes, padded to a 256-byte row of its own cache lines.
INTER_ROW_WORDS = 64
# LDS row strides of the fused MoE's activations: 4 words past a multiple of
# the 64 banks, so a WMMA operand's 16 rows land on distinct banks (at a
# multiple, every read was a 16-way conflict).
LATENT_LDS_WORDS = LATENT // 8 + 4
INTER_LDS_WORDS = INTER_ROW_WORDS + 4
LATENT_WORDS = LATENT // 8
LATENT_SCALES = LATENT // 32
INTER_GROUPS = 12  # MXFP4 blocks of an expert's 384 intermediates
INTER_WORDS = 48


@dataclass(frozen=True)
class ExpertUpKernel:
    name: str
    launcher: object
    grid: tuple[int]


@traced
def _union(ids, rows, tid, lane, wave, lds, threads=THREADS):
    """The step's ascending expert union and its picks, in LDS: ``slot[e]``
    an expert's union index, ``expert[u]``, ``start[u]``/``count[u]`` its run
    in ``picks`` (token * TOPK + k). Returns the union size."""
    routes = rows * TOPK
    count, slot, start = lds["count"], lds["slot"], lds["start"]
    for index in range_constexpr((EXPERTS + threads - 1) // threads):
        expert = tid + index * threads
        if expert < EXPERTS:
            fx.ptr_store(fx.Int32(0), count + expert)
    gpu.barrier()
    rank = []
    for index in range_constexpr((routes + threads - 1) // threads):
        at = fx.min(tid + index * threads, routes - 1)
        expert = fx.Int32(bo.buffer_load(rsrc(ids), at, vec_width=1, dtype=T.i32))
        taken = fx.atomic_add(
            count + expert,
            (tid + index * threads < routes).select(fx.Int32(1), fx.Int32(0)),
            syncscope="workgroup",
        )
        rank.append((expert, taken))
    gpu.barrier()
    per = (EXPERTS + threads - 1) // threads
    first = tid * per
    present, picks = fx.Int32(0), fx.Int32(0)
    counted = []
    for index in range_constexpr(per):
        expert = first + index
        n = (expert < EXPERTS).select(
            fx.ptr_load(count + fx.min(expert, EXPERTS - 1)), fx.Int32(0)
        )
        counted.append(n)
        present = present + (n > 0).select(fx.Int32(1), fx.Int32(0))
        picks = picks + n
    union_base, union = _block_exclusive_scan32(
        present, lane, wave, lds["scan"], threads // GFX1250.wave_size
    )
    gpu.barrier()
    pick_base, _ = _block_exclusive_scan32(
        picks, lane, wave, lds["scan"], threads // GFX1250.wave_size
    )
    for index in range_constexpr(per):
        expert = first + index
        if (expert < EXPERTS) & (counted[index] > 0):
            fx.ptr_store(union_base, slot + expert)
            fx.ptr_store(expert, lds["expert"] + union_base)
            fx.ptr_store(pick_base, start + union_base)
            fx.ptr_store(counted[index], lds["picks_n"] + union_base)
        union_base = union_base + (counted[index] > 0).select(fx.Int32(1), fx.Int32(0))
        pick_base = pick_base + counted[index]
    gpu.barrier()
    for index in range_constexpr(len(rank)):
        at = tid + index * threads
        if at < routes:
            expert, taken = rank[index]
            where = fx.ptr_load(start + fx.ptr_load(slot + expert)) + taken
            fx.ptr_store(fx.Int32(at), lds["picks"] + where)
    gpu.barrier()
    return union


@traced
def _stage_latent(latent, scales, rows, tid, lds, stride=LATENT_WORDS):
    """Every token's MXFP4 latent words (rows ``stride`` words apart) and E8M0
    scale bytes (four a word, a K step's) into LDS."""
    words = rows * LATENT_WORDS
    for index in range_constexpr((words + THREADS - 1) // THREADS):
        at = tid + index * THREADS
        if at < words:
            value = bo.buffer_load(rsrc(latent), at, vec_width=1, dtype=T.i32)
            fx.ptr_store(
                fx.Int32(value),
                lds["latq"] + at // LATENT_WORDS * stride + at % LATENT_WORDS,
            )
    scale_words = rows * LATENT_SCALES // 4
    for index in range_constexpr((scale_words + THREADS - 1) // THREADS):
        at = tid + index * THREADS
        if at < scale_words:
            word = bo.buffer_load(rsrc(scales), at, vec_width=1, dtype=T.i32)
            fx.ptr_store(fx.Int32(word), lds["latsc"] + at)
    gpu.barrier()


def _tile_source(w13, expert, block, tile, group_bytes=GROUP_TILE_BYTES):
    """Global address of task (``expert``, ``block``)'s K tile ``tile``: its
    first 16-row group's 4 KB, the next groups ``LATENT * 8`` bytes on."""
    group0 = expert * (2 * 384 // 16) + block * GROUPS
    return w13 + fx.Int64(group0) * (LATENT * 8) + fx.Int64(tile * group_bytes)


def _weight_operand(stage, wave, lane, step, group_bytes=GROUP_TILE_BYTES):
    """This wave's v16 weight operand of K step ``step`` of an LDS stage, laid
    out as ``moe_expert_gfx1250._weight_fragment`` reads AITER's preshuffle."""
    lane16 = lane % 16
    k_parity = lane // 16
    parts = []
    for output_half in range_constexpr(2):
        group = 2 * wave + output_half
        for k64_half in range_constexpr(2):
            word = (
                group * (group_bytes // 4)
                + step * 256
                + k64_half * 128
                + k_parity * 64
                + lane16 * 4
            )
            value = fx.Vector(
                fx.ptr_load(stage + word, result_type=fx.Vector.make_type(4, fx.Int32))
            )
            parts.extend([value[i] for i in range_constexpr(4)])
    return fx.Vector.from_elements(parts, fx.Int32)


def _situ(accumulator):
    """AITER's SiTUv2 of a wave's 32x16 gate|up accumulator: its lane's 8
    intermediates (each a GUGU pair of the 16)."""
    consts = situv2_consts(fx.Float32(SITU_BETA), fx.Float32(SITU_LINEAR_BETA))
    return batched_situv2(
        [(accumulator[2 * e], accumulator[2 * e + 1]) for e in range(8)],
        consts=consts,
        range_constexpr=range_constexpr,
    )


@traced
def _epilogue(
    lds,
    inter,
    isc,
    picks,
    count,
    block,
    pair,
    tile,
    lane,
    in_row=False,
    pairs=TASK_ROWS // 64,
    activated=False,
):
    """One MXFP4 block (waves 2 ``pair``, + 1: 64 GUGU rows) of M tile
    ``tile``: AITER's SiTUv2, the block's E8M0 scale, its FP4 words at each
    pick. A lane holds rows 8 (lane / 16) .. + 8 and 16 + those of a 32-row
    tile, its token's: the lanes of a pick hold 16 intermediates each.
    ``in_row``: a pick's row ``INTER_ROW_WORDS`` wide with its codes past the
    FP4 words, stored at device scope for another CTA's down. ``activated``:
    the waves left their ``_situ`` in ``red`` (8 a lane), not accumulators."""
    red = lds["red"]
    logical = []
    for half in range_constexpr(2):
        slot = (2 * pair + half) * GFX1250.wave_size + lane
        if const_expr(activated):
            logical.append([fx.ptr_load(red + slot * 8 + e) for e in range(8)])
        else:
            physical = [fx.ptr_load(red + slot * 16 + e) for e in range(16)]
            logical.append(_situ(physical))
    peer = [[v.shuffle_xor(16, GFX1250.wave_size) for v in half] for half in logical]
    lower = logical[0][:4] + peer[0][:4] + logical[1][:4] + peer[1][:4]
    upper = peer[0][4:] + logical[0][4:] + peer[1][4:] + logical[1][4:]
    values = [(lane < 16).select(lower[e], upper[e]) for e in range_constexpr(16)]
    scale, code = emit_amax_e8m0_native_scale(
        values, wave_size=GFX1250.wave_size, dtype=MxDtypeInt.FP4_E2M1
    )
    index = tile * 16 + lane % 16
    if index < count:
        pick = fx.ptr_load(picks + index)
        group = block * pairs + pair
        parity = lane // 16
        for half in range_constexpr(2):
            packed = emit_cvt_scalef32_pk8_fp4_bf16(
                fx.Vector.from_elements(values[half * 8 : half * 8 + 8], fx.Float32)
                .to(fx.BFloat16)
                .ir_value(),
                scale,
                i32_ty=T.i32,
            )
            bo.buffer_store(
                fx.Int32(packed),
                rsrc(inter),
                pick * (INTER_ROW_WORDS if in_row else INTER_WORDS)
                + group * 4
                + parity
                + 2 * half,
                cache_modifier=CM_DEV if in_row else 0,
            )
        if lane < 16:
            if const_expr(in_row):
                bo.buffer_store(
                    fx.Int8(fx.Uint8(code)),
                    rsrc(inter),
                    (pick * INTER_ROW_WORDS + INTER_WORDS) * 4 + group,
                    cache_modifier=CM_DEV,
                )
            else:
                bo.buffer_store(
                    fx.Int8(fx.Uint8(code)), rsrc(isc), pick * INTER_GROUPS + group
                )


def build_expert_up(
    rows: int, stages: int = STAGES, k_tile: int = K_TILE
) -> ExpertUpKernel:
    """The gate/up of ``rows`` tokens' routes on a resident 256-CTA grid: a
    CTA's tasks are ``bid``, ``bid + 256``, .. of the union's 3 a expert."""
    if not 1 <= rows <= 48:
        raise ValueError(f"the streamed gate/up serves 1..48 rows, got {rows}")
    tiles = (rows + 15) // 16
    blocks = 2 * 384 // TASK_ROWS
    k_tiles = LATENT // k_tile
    steps = k_tile // K_STEP
    stage_bytes = GROUPS * 16 * k_tile // 2
    ring_words = stages * stage_bytes // 4
    routes = rows * TOPK

    @fx.struct
    class Lds:
        stages: fx.Array[fx.Int32, ring_words, 16]
        latq: fx.Array[fx.Int32, rows * LATENT_WORDS, 16]
        latsc: fx.Array[fx.Int32, rows * LATENT_SCALES // 4, 16]
        red: fx.Array[fx.Float32, WAVES * GFX1250.wave_size * 16, 16]
        count: fx.Array[fx.Int32, EXPERTS, 16]
        slot: fx.Array[fx.Int32, EXPERTS, 16]
        expert: fx.Array[fx.Int32, routes, 16]
        start: fx.Array[fx.Int32, routes, 16]
        picks_n: fx.Array[fx.Int32, routes, 16]
        picks: fx.Array[fx.Int32, routes, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    name = f"k3_gfx1250_expert_up_tdm_s{rows}_st{stages}_k{k_tile}"
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        ids: Int64,
        latent: Int64,
        scales: Int64,
        w13: Int64,
        w13s: Int64,
        inter: Int64,
        isc: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        smem = fx.SharedAllocator().allocate(Lds).peek()
        lds = {
            name: getattr(smem, name).ptr
            for name in (
                "latq",
                "latsc",
                "red",
                "count",
                "slot",
                "expert",
                "start",
                "picks_n",
                "picks",
                "scan",
            )
        }
        union = uniform(_union(ids, rows, tid, lane, wave, lds))
        _stage_latent(latent, scales, rows, tid, lds)
        tasks = union * blocks
        mine = (tasks - bid + 255) // 256

        def task_of(n):
            task = bid + n * 256
            return task // blocks, task % blocks

        def issue(n, tile, stage):
            slot, block = task_of(n)
            expert = uniform(fx.ptr_load(lds["expert"] + fx.min(slot, union - 1)))
            if wave == 0:
                gemv_tdm.load_rows(
                    _tile_source(w13, expert, block, tile, stage_bytes // GROUPS),
                    LATENT * 4,
                    smem.stages.ptr + stage * (stage_bytes // 4),
                    GROUPS,
                    GROUPS,
                    padded=False,
                    cache_policy=1,
                    whole=True,
                    width=stage_bytes // GROUPS // 2,
                )

        flat = mine * k_tiles
        for ahead in range_constexpr(stages - 1):
            if ahead < flat:
                issue(ahead // k_tiles, ahead % k_tiles, ahead)
        n = fx.Int32(0)
        while n < mine:
            slot, block = task_of(n)
            expert = uniform(fx.ptr_load(lds["expert"] + slot))
            start = fx.ptr_load(lds["start"] + slot)
            count = uniform(fx.ptr_load(lds["picks_n"] + slot))
            picks = lds["picks"] + start
            row = block * TASK_ROWS + wave * 32
            tokens = [
                fx.ptr_load(picks + fx.min(m * 16 + lane % 16, count - 1)) // TOPK
                for m in range(tiles)
            ]
            accumulators = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(tiles)]
            weight_scales = [
                _weight_scale_word(w13s, expert, 2 * 384, LATENT, row + lane, k_step)
                for k_step in range(k_tiles * steps)
            ]
            for tile_k in range_constexpr(k_tiles):
                flat_at = n * k_tiles + tile_k
                if flat_at + stages - 2 < flat:
                    rocdl.s_wait_tensorcnt(stages - 2)
                else:
                    rocdl.s_wait_tensorcnt(0)
                gpu.barrier()
                ahead = flat_at + stages - 1
                if ahead < flat:
                    issue(ahead // k_tiles, ahead % k_tiles, ahead % stages)
                stage = smem.stages.ptr + (flat_at % stages) * (stage_bytes // 4)
                for step in range_constexpr(steps):
                    k_step = tile_k * steps + step
                    weight = _weight_operand(
                        stage, wave, lane, step, stage_bytes // GROUPS
                    )
                    weight_scale = weight_scales[k_step]
                    for m in range_constexpr(tiles):
                        if m == 0 or m * 16 < count:
                            activation = _activation_fragment(
                                lds["latq"], LATENT_WORDS, tokens[m], k_step, lane
                            )
                            activation_scale = fx.ptr_load(
                                lds["latsc"] + tokens[m] * (LATENT_SCALES // 4) + k_step
                            )
                            accumulators[m] = _wmma(
                                weight,
                                activation,
                                accumulators[m],
                                weight_scale,
                                activation_scale,
                            )
            for m in range_constexpr(tiles):
                if m == 0 or m * 16 < count:
                    fx.ptr_store(
                        accumulators[m],
                        lds["red"] + (wave * GFX1250.wave_size + lane) * 16,
                    )
                    gpu.barrier()
                    if wave < TASK_ROWS // 64:
                        _epilogue(lds, inter, isc, picks, count, block, wave, m, lane)
                    gpu.barrier()
            n = n + 1

    @flyc.jit
    def launch(
        ids: Int64,
        latent: Int64,
        scales: Int64,
        w13: Int64,
        w13s: Int64,
        inter: Int64,
        isc: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(ids, latent, scales, w13, w13s, inter, isc).launch(
            grid=(256,), block=(THREADS,), stream=stream
        )

    return ExpertUpKernel(name, launch, (256,))


DOWN_ROWS = 16
DOWN_TASKS = LATENT // DOWN_ROWS
DOWN_STEPS = 384 // K_STEP
GROUP_DOWN_BYTES = 16 * 384 // 2


@dataclass(frozen=True)
class ExpertDownKernel:
    name: str
    launcher: object
    grid: tuple[int]


def _wmma16(weight, activation, accumulator, weight_scale, activation_scale):
    """AITER gemm2's FP4 x FP4 16x16x128 scaled WMMA: 16 weight rows a lane's
    eight (8 (lane / 16) ..) against its pick (lane % 16)."""
    return fx.Vector(
        rocdl.wmma_scale_f32_16x16x128_f8f6f4(
            T.vec(8, T.f32),
            weight.ir_value(),
            activation.ir_value(),
            accumulator.ir_value(),
            weight_scale,
            activation_scale,
        )
    )


def _load_v8(resource, first, second):
    a = fx.Vector(bo.buffer_load(resource, first, vec_width=4, dtype=T.i32))
    b = fx.Vector(bo.buffer_load(resource, second, vec_width=4, dtype=T.i32))
    return fx.Vector.from_elements(
        [a[i] for i in range(4)] + [b[i] for i in range(4)], fx.Int32
    )


def _down_loads(c, slot, block, lane):
    """Expert ``slot``'s down weights and scales for rows 16 ``block`` .., and
    its first 16 picks' FP4 intermediates and E8M0 words: what the wave issues
    one expert ahead of its WMMAs."""
    lds = c["lds"]
    expert = uniform(fx.ptr_load(lds["expert"] + slot))
    start = fx.ptr_load(lds["start"] + slot)
    count = uniform(fx.ptr_load(lds["picks_n"] + slot))
    kgrp = lane // 16
    group = rsrc(c["w2"] + fx.Int64(expert * DOWN_TASKS + block) * GROUP_DOWN_BYTES)
    weights = [
        _load_v8(
            group,
            (step * 1024 + kgrp * 256 + (lane % 16) * 16) // 4,
            (step * 1024 + kgrp * 256 + (lane % 16) * 16 + 512) // 4,
        )
        for step in range(DOWN_STEPS)
    ]
    scales = [
        _weight_scale_word(c["w2s"], expert, LATENT, 384, block * 16 + lane % 16, step)
        for step in range(DOWN_STEPS)
    ]
    pick = fx.ptr_load(lds["picks"] + start + fx.min(lane % 16, count - 1))
    return [expert, start, count, pick, weights, scales, _pick_loads(c, pick, lane)]


def _pick_loads(c, pick, lane):
    """A pick's FP4 intermediate operands and E8M0 words, a K step each."""
    kgrp = lane // 16
    inter = rsrc(c["inter"])
    activations = [
        _load_v8(
            inter,
            pick * INTER_WORDS + step * 16 + kgrp * 4,
            pick * INTER_WORDS + step * 16 + 8 + kgrp * 4,
        )
        for step in range(DOWN_STEPS)
    ]
    codes = [
        fx.Int32(
            bo.buffer_load(
                rsrc(c["isc"]),
                pick * (INTER_GROUPS // 4) + step,
                vec_width=1,
                dtype=T.i32,
            )
        )
        for step in range(DOWN_STEPS)
    ]
    return activations, codes


@traced
def _down_expert(c, loaded, out, lane, valid):
    """``_down_loads``' expert against each of its picks, rounded to BF16
    (gemm2's output) into ``out[pick][16 rows]``; picks past 16 load here."""
    lds = c["lds"]
    _, start, count, pick, weights, scales, first = loaded
    kgrp = lane // 16
    for tile in range_constexpr(c["tiles"]):
        if tile == 0 or tile * 16 < count:
            index = tile * 16 + lane % 16
            activations, codes = first
            if const_expr(tile != 0):
                pick = fx.ptr_load(lds["picks"] + start + fx.min(index, count - 1))
                activations, codes = _pick_loads(c, pick, lane)
            accumulator = fx.Vector.filled(8, 0.0, fx.Float32)
            for step in range_constexpr(DOWN_STEPS):
                accumulator = _wmma16(
                    weights[step],
                    activations[step],
                    accumulator,
                    scales[step],
                    codes[step],
                )
            if (index < count) & valid:
                rounded = accumulator.to(fx.BFloat16).to(fx.Float32)
                for e in range_constexpr(8):
                    fx.ptr_store(rounded[e], out + pick * DOWN_ROWS + kgrp * 8 + e)


def build_expert_down(rows: int, threads: int = THREADS) -> ExpertDownKernel:
    """The routed down of ``rows`` tokens and its gather-reduce: CTA ``b`` <
    224 takes latent rows 16 ``b`` .., every union expert's (a wave every
    eighth), and sums each token's 16 picks in route order (AITER's
    ``moe_gather_reduce``) into ``routed[rows, LATENT]`` BF16."""
    if not 1 <= rows <= 48:
        raise ValueError(f"the streamed down serves 1..48 rows, got {rows}")
    tiles = (rows + 15) // 16
    waves = threads // GFX1250.wave_size
    routes = rows * TOPK

    @fx.struct
    class Lds:
        out: fx.Array[fx.Float32, routes * DOWN_ROWS, 16]
        count: fx.Array[fx.Int32, EXPERTS, 16]
        slot: fx.Array[fx.Int32, EXPERTS, 16]
        expert: fx.Array[fx.Int32, routes, 16]
        start: fx.Array[fx.Int32, routes, 16]
        picks_n: fx.Array[fx.Int32, routes, 16]
        picks: fx.Array[fx.Int32, routes, 16]
        scan: fx.Array[fx.Int32, waves, 16]

    name = f"k3_gfx1250_expert_down_s{rows}_t{threads}"
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def kernel(
        ids: Int64,
        route_weights: Int64,
        inter: Int64,
        isc: Int64,
        w2: Int64,
        w2s: Int64,
        routed: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        smem = fx.SharedAllocator().allocate(Lds).peek()
        lds = {
            name: getattr(smem, name).ptr
            for name in ("count", "slot", "expert", "start", "picks_n", "picks", "scan")
        }
        if bid < DOWN_TASKS:
            union = uniform(_union(ids, rows, tid, lane, wave, lds, threads))
            context = {
                "lds": lds,
                "w2": w2,
                "w2s": w2s,
                "inter": inter,
                "isc": isc,
                "tiles": tiles,
            }
            # One expert's loads in flight under the previous one's WMMAs.
            # Two register sets in turn: one expert's loads in flight under
            # the other's WMMAs, with no loop-carried copy to wait on.
            if wave < union:
                ahead = _down_loads(context, wave, bid, lane)
                slot = wave
                while slot < union:
                    behind = _down_loads(
                        context, fx.min(slot + waves, union - 1), bid, lane
                    )
                    _down_expert(context, ahead, smem.out.ptr, lane, True)
                    ahead = _down_loads(
                        context, fx.min(slot + 2 * waves, union - 1), bid, lane
                    )
                    _down_expert(
                        context, behind, smem.out.ptr, lane, slot + waves < union
                    )
                    slot = slot + 2 * waves
            gpu.barrier()
            pairs = rows * DOWN_ROWS // 2
            for index in range_constexpr((pairs + threads - 1) // threads):
                at = tid + index * threads
                if at < pairs:
                    token = at // (DOWN_ROWS // 2)
                    row = at % (DOWN_ROWS // 2) * 2
                    sums = [fx.Float32(0.0), fx.Float32(0.0)]
                    for k in range_constexpr(TOPK):
                        weight = fx.Float32(
                            bo.buffer_load(
                                rsrc(route_weights),
                                token * TOPK + k,
                                vec_width=1,
                                dtype=T.f32,
                            )
                        )
                        for e in range_constexpr(2):
                            value = fx.ptr_load(
                                smem.out.ptr + (token * TOPK + k) * DOWN_ROWS + row + e
                            )
                            sums[e] = sums[e] + weight * (fx.Float32(0.0) + value)
                    word = (
                        fx.Vector.from_elements(sums, fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32)[0]
                    )
                    bo.buffer_store(
                        word,
                        rsrc(routed),
                        (token * LATENT + bid * DOWN_ROWS + row) // 2,
                    )

    @flyc.jit
    def launch(
        ids: Int64,
        route_weights: Int64,
        inter: Int64,
        isc: Int64,
        w2: Int64,
        w2s: Int64,
        routed: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(ids, route_weights, inter, isc, w2, w2s, routed).launch(
            grid=(256,), block=(threads,), stream=stream
        )

    return ExpertDownKernel(name, launch, (256,))


DOWN_TILES = LATENT // TASK_ROWS
# The fused MoE streams 48 KB items through a ring of four: a gate/up K tile
# of 12 groups (192 rows, three MXFP4 blocks: four tasks an expert) or a down
# tile of 16 groups (256 rows). Three in flight keep a CU fed past the HBM
# latency, where two 64 KB stages did not.
UP_GROUPS = 12
UP_ROWS = UP_GROUPS * 16
UP_BLOCKS = 2 * 384 // UP_ROWS
UP_WAVES = UP_ROWS // 32
SLOT_BYTES = UP_GROUPS * GROUP_TILE_BYTES
RING = 4
assert GROUPS * GROUP_DOWN_BYTES == SLOT_BYTES
# An item's weight scales ride in with it (wave 1's TDM copy into a ring of
# their own): a gate/up K tile's 6 row groups x 4 K steps x 32 rows, a down
# tile's 8 x 3 x 32 -- 3 KB either way, read where a lane's row is.
SCALE_WORDS = 768
# The TDM copies' waves: a gate/up K tile's rows are waves 0..5's, so its two
# copies leave their path (a down tile's rows are every wave's).
WEIGHT_ISSUER = WAVES - 2
SCALE_ISSUER = WAVES - 1
assert UP_WAVES * STEPS * 32 == DOWN_STEPS * 8 * 32 == SCALE_WORDS
FLAG_WORDS = UP_BLOCKS
# A gate/up task's and a down tile's weight bytes, in 16 KB units: what the
# fused MoE balances CTAs by.
UP_UNITS = UP_GROUPS * LATENT * 8 // 16384
DOWN_UNITS = GROUPS * GROUP_DOWN_BYTES // 16384
# The fused MoE's state words: unused, the CTAs done, the epoch, and a
# timed-out wait's record (0: none).
MOE_STATE_WORDS = 4
# Polls before a flag wait gives up and records itself: never a hung GPU.
POLL_LIMIT = 1 << 16


@dataclass(frozen=True)
class MoeKernel:
    name: str
    launcher: object
    grid: tuple[int]


def _realtime():
    """The 100 MHz device clock (gfx1250 has no s_memrealtime)."""
    from flydsl._mlir.dialects import llvm

    return fx.Int64(
        llvm.inline_asm(
            T.i64,
            [],
            "s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)\n s_wait_kmcnt 0",
            "=s",
            has_side_effects=True,
        )
    )


def _cycles():
    """The core-clock cycle counter's low word (``llvm.readcyclecounter``)."""
    from flydsl._mlir.dialects import llvm

    return fx.Int32(
        fx.Int64(llvm.call_intrinsic(T.i64, "llvm.readcyclecounter", [], [], []))
    )


def _lds_barrier():
    """A workgroup barrier for LDS alone: this wave's LDS traffic drained,
    then signal and wait. ``gpu.barrier``'s workgroup fence also waits out
    every global store first (s_wait_storecnt 0) -- an epilogue's device-scope
    stores, a down tile's output -- which nothing past these barriers reads;
    the one release that needs them (a gate/up flag) drains them itself. As
    side-effecting asm it keeps the compiler's memory ops on their side."""
    from flydsl._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [],
        "s_wait_dscnt 0x0\n s_barrier_signal -1\n s_barrier_wait -1",
        "",
        has_side_effects=True,
    )


@traced
def _bounded_wait(flags, slot, tag, state, bid):
    """Until expert ``slot``'s four gate/up flags carry ``tag``; past
    ``POLL_LIMIT`` polls the wait records (CTA, slot, a flag) in ``state[3]``
    and goes on."""
    polls = fx.Int32(0)
    v = fx.Vector(ld_dev(flags, slot * FLAG_WORDS, 4))
    while ((v[0] != tag) | (v[1] != tag) | (v[2] != tag) | (v[3] != tag)) & (
        polls < POLL_LIMIT
    ):
        rocdl.s_sleep(1)
        polls = polls + 1
        v = fx.Vector(ld_dev(flags, slot * FLAG_WORDS, 4))
    if polls >= POLL_LIMIT:
        record = (bid << 20) | (slot << 8) | (fx.Int32(v[0]) & 0xFF) | (1 << 30)
        st_dev(record, state, 3)


def _down_share(union, bid, ups):
    """CTA ``bid``'s run of the step's down tiles (slot-major, ``DOWN_TILES``
    an expert): (first, count). With ``ups`` gate/up tasks a CTA, those with
    one more take fewer, so every CTA streams about the same weight bytes; the
    CTAs with fewer gate/up tasks start first and take the lowest slots, whose
    gate/up ran in the first rounds."""
    tasks = union * UP_BLOCKS
    rest = tasks % 256
    tiles = union * DOWN_TILES
    total = tasks * UP_UNITS + tiles * DOWN_UNITS
    more = fx.max(total - 256 * (ups + 1) * UP_UNITS, 0)
    fewer = fx.max(total - 256 * ups * UP_UNITS, 0)
    first_class = 256 - rest

    def before(j):
        return fx.Int64(
            (j < first_class).select(
                j * fewer, first_class * fewer + (j - first_class) * more
            )
        )

    order = (bid >= rest).select(bid - rest, first_class + bid)
    whole = before(fx.Int32(256))
    first = fx.Int32(before(order) * fx.Int64(tiles) // whole)
    last = fx.Int32(before(order + 1) * fx.Int64(tiles) // whole)
    return first, last - first


def build_moe(rows: int, profile: bool = False, publish_at: int = 1) -> MoeKernel:
    """The routed experts of ``rows`` tokens in one resident 256-CTA launch,
    with no grid barrier and no queue: every CTA works out the same static,
    byte-balanced plan -- its gate/up tasks (an expert's 192 rows, each pick's
    row ``INTER_ROW_WORDS`` wide at device scope) then a run of down tiles
    (``_down_share``) -- and streams all of its weights through one TDM ring.
    A down tile is an expert's 256 output rows against each of its picks, its
    BF16 rows (gemm2's) into ``grouped[route][LATENT]``; at a new expert the
    CTA waits on its three gate/up flags and stages its picks' rows.

    ``state`` (``MOE_STATE_WORDS`` i32, zeroed once): the CTA done last moves
    the epoch the flags are tagged with on, so a replay needs no reset.
    ``profile``: each CTA's phase stamps past it (5 words a CTA: start, plan,
    gate/up done, down done, down tiles) and its flag waits (2 words a CTA:
    100 MHz ticks, expert switches) past those."""
    routes = rows * TOPK
    lds_bytes = 4 * (
        RING * (SLOT_BYTES // 4 + SCALE_WORDS)
        + rows * (LATENT_LDS_WORDS + LATENT_SCALES // 4)
        + UP_WAVES * GFX1250.wave_size * 8
        + 2 * EXPERTS
        + 4 * routes
        + WAVES
    )
    if rows < 1 or lds_bytes > GFX1250.max_workgroup_lds_bytes:
        raise ValueError(f"the fused MoE's {rows}-row LDS ({lds_bytes} B) does not fit")
    tiles = (rows + 15) // 16
    blocks = UP_BLOCKS
    k_tiles = LATENT // K_TILE
    ring = RING
    ring_words = ring * SLOT_BYTES // 4

    @fx.struct
    class Lds:
        stages: fx.Array[fx.Int32, ring_words, 16]
        item_scales: fx.Array[fx.Int32, ring * SCALE_WORDS, 16]
        latq: fx.Array[fx.Int32, rows * LATENT_LDS_WORDS, 16]
        latsc: fx.Array[fx.Int32, rows * LATENT_SCALES // 4, 16]
        red: fx.Array[fx.Float32, UP_WAVES * GFX1250.wave_size * 8, 16]
        count: fx.Array[fx.Int32, EXPERTS, 16]
        slot: fx.Array[fx.Int32, EXPERTS, 16]
        expert: fx.Array[fx.Int32, routes, 16]
        start: fx.Array[fx.Int32, routes, 16]
        picks_n: fx.Array[fx.Int32, routes, 16]
        picks: fx.Array[fx.Int32, routes, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]

    name = f"k3_gfx1250_moe_s{rows}_p{publish_at}" + ("_profile" if profile else "")
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        ids: Int64,
        latent: Int64,
        scales: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        inter: Int64,
        grouped: Int64,
        flags: Int64,
        state: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        smem = fx.SharedAllocator().allocate(Lds).peek()
        lds = {
            name: getattr(smem, name).ptr
            for name in (
                "latq",
                "latsc",
                "red",
                "count",
                "slot",
                "expert",
                "start",
                "picks_n",
                "picks",
                "scan",
            )
        }
        epoch = uniform(fx.Int32(ld_dev(state, 2)))
        tag = epoch + 1

        def stamp(i, value=None):
            if const_expr(profile):  # noqa: SIM102
                if tid == 0:
                    v = fx.Int32(_realtime()) if value is None else value
                    st_dev(v, state, MOE_STATE_WORDS + bid * 5 + i)

        stamp(0)

        def publish(word):
            st_dev(tag, flags, word)

        union = uniform(_union(ids, rows, tid, lane, wave, lds))
        tasks = union * blocks
        ups = tasks // 256
        mine = (tasks - bid + 255) // 256
        first_down, downs = _down_share(union, bid, ups)
        up_items = mine * k_tiles
        flat = up_items + downs
        stamp(1)

        def down_tile(i):
            g = first_down + i
            return g // DOWN_TILES, g % DOWN_TILES

        # The experts the CTA streams, a lane each, read where an item issues
        # (no LDS round trip there): lane n < 8 its n-th gate/up task's,
        # lane 8 + j the j-th slot of its down run's.
        first_slot = first_down // DOWN_TILES
        experts = fx.ptr_load(
            lds["expert"]
            + fx.min(
                (lane < 8).select((bid + lane * 256) // blocks, first_slot + lane - 8),
                union - 1,
            )
        )

        def issue(item, stage):
            buffer = smem.stages.ptr + stage * (SLOT_BYTES // 4)
            scale_buffer = smem.item_scales.ptr + stage * SCALE_WORDS
            if item < up_items:
                n = item // k_tiles
                task = bid + n * 256
                expert = readlane(experts, n)
                group0 = expert * (2 * 384 // 16) + task % blocks * UP_GROUPS
                if wave == WEIGHT_ISSUER:
                    gemv_tdm.load_rows(
                        w13
                        + fx.Int64(group0) * (LATENT * 8)
                        + fx.Int64(item % k_tiles * GROUP_TILE_BYTES),
                        LATENT * 4,
                        buffer,
                        UP_GROUPS,
                        UP_GROUPS,
                        padded=False,
                        cache_policy=1,
                        whole=True,
                        width=GROUP_TILE_BYTES // 2,
                    )
                # n32k4: (expert, 32-row group, K step, row) words; a task's
                # row groups are 28 K steps (3584 bytes) apart.
                row_group0 = expert * (2 * 384 // 32) + task % blocks * UP_WAVES
                if wave == SCALE_ISSUER:
                    gemv_tdm.load_rows(
                        w13s
                        + fx.Int64(
                            row_group0 * (LATENT // 128) + item % k_tiles * STEPS
                        )
                        * 128,
                        LATENT // 128 * 64,
                        scale_buffer,
                        UP_WAVES,
                        UP_WAVES,
                        padded=False,
                        cache_policy=1,
                        whole=True,
                        width=STEPS * 64,
                    )
            else:
                slot, tile = down_tile(item - up_items)
                expert = readlane(experts, 8 + slot - first_slot)
                if wave == WEIGHT_ISSUER:
                    gemv_tdm.load_rows(
                        w2
                        + fx.Int64(expert * DOWN_TASKS + tile * GROUPS)
                        * GROUP_DOWN_BYTES,
                        GROUP_DOWN_BYTES // 2,
                        buffer,
                        GROUPS,
                        GROUPS,
                        padded=False,
                        cache_policy=1,
                        whole=True,
                        width=GROUP_DOWN_BYTES // 2,
                    )
                if wave == SCALE_ISSUER:
                    gemv_tdm.load_rows(
                        w2s
                        + fx.Int64((expert * (LATENT // 32) + tile * 8) * DOWN_STEPS)
                        * 128,
                        DOWN_STEPS * 64,
                        scale_buffer,
                        8,
                        8,
                        padded=False,
                        cache_policy=1,
                        whole=True,
                        width=DOWN_STEPS * 64,
                    )

        def wait_for(item):
            if item + ring - 2 < flat:
                rocdl.s_wait_tensorcnt(ring - 2)
            else:
                rocdl.s_wait_tensorcnt(0)

        for ahead in range_constexpr(ring - 1):
            if ahead < flat:
                issue(fx.Int32(ahead), ahead)
        # The weights stream in while the latent goes to LDS.
        _stage_latent(latent, scales, rows, tid, lds, LATENT_LDS_WORDS)
        # A task's flag goes out a K tile into the next, its stores drained.
        pending = fx.Int32(-1)
        up_wait = fx.Int32(0)
        up_epilogue = fx.Int32(0)

        def clock():
            return _cycles() if profile else fx.Int32(0)

        def since(before):
            return clock() - before

        up_start = clock()
        n = fx.Int32(0)
        while n < mine:
            task = bid + n * 256
            slot = task // blocks
            block = task % blocks
            start = fx.ptr_load(lds["start"] + slot)
            count = uniform(fx.ptr_load(lds["picks_n"] + slot))
            picks = lds["picks"] + start
            tokens = [
                fx.ptr_load(picks + fx.min(m * 16 + lane % 16, count - 1)) // TOPK
                for m in range(tiles)
            ]
            accumulators = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(tiles)]
            for tile_k in range_constexpr(k_tiles):
                item = n * k_tiles + tile_k
                before = clock()
                wait_for(item)
                if const_expr(tile_k == publish_at):
                    rocdl.s_wait_storecnt(0)
                _lds_barrier()
                up_wait = up_wait + since(before)
                if const_expr(tile_k == publish_at):  # noqa: SIM102
                    if (tid == 0) & (pending >= 0):
                        publish(pending)
                if item + ring - 1 < flat:
                    issue(item + ring - 1, (item + ring - 1) % ring)
                stage = smem.stages.ptr + (item % ring) * (SLOT_BYTES // 4)
                # One branch a wave (6 and 7 hold no rows), then every K step's
                # weights and scales loaded ahead of the first tile's WMMA chain:
                # a tile's own branch held the loads behind each WMMA.
                if wave < UP_WAVES:
                    item_scale = (
                        smem.item_scales.ptr
                        + (item % ring) * SCALE_WORDS
                        + wave * STEPS * 32
                        + lane
                    )
                    weights = [
                        _weight_operand(stage, wave, lane, step)
                        for step in range(STEPS)
                    ]
                    weight_scales = [
                        fx.ptr_load(item_scale + step * 32) for step in range(STEPS)
                    ]
                    for m in range_constexpr(tiles):
                        if m == 0 or m * 16 < count:
                            for step in range_constexpr(STEPS):
                                k_step = tile_k * STEPS + step
                                accumulators[m] = _wmma(
                                    weights[step],
                                    _activation_fragment(
                                        lds["latq"],
                                        LATENT_LDS_WORDS,
                                        tokens[m],
                                        k_step,
                                        lane,
                                    ),
                                    accumulators[m],
                                    weight_scales[step],
                                    fx.ptr_load(
                                        lds["latsc"]
                                        + tokens[m] * (LATENT_SCALES // 4)
                                        + k_step
                                    ),
                                )
            before = clock()
            for m in range_constexpr(tiles):
                if m == 0 or m * 16 < count:
                    # Each wave its own SiTU: 6 waves' share, not 3's.
                    if wave < UP_WAVES:
                        fx.ptr_store(
                            fx.Vector.from_elements(_situ(accumulators[m]), fx.Float32),
                            lds["red"] + (wave * GFX1250.wave_size + lane) * 8,
                        )
                    _lds_barrier()
                    if wave < UP_ROWS // 64:
                        _epilogue(
                            lds,
                            inter,
                            0,
                            picks,
                            count,
                            block,
                            wave,
                            m,
                            lane,
                            in_row=True,
                            pairs=UP_ROWS // 64,
                            activated=True,
                        )
                    _lds_barrier()
            up_epilogue = up_epilogue + since(before)
            pending = slot * FLAG_WORDS + block
            n = n + 1
        rocdl.s_wait_storecnt(0)
        _lds_barrier()
        if (tid == 0) & (pending >= 0):
            publish(pending)
        up_cycles = since(up_start)
        stamp(2)

        rows_lds = lds["latq"]
        current = fx.Int32(-1)
        start = fx.Int32(0)
        count = fx.Int32(0)
        first_pick = fx.Int32(0)
        waited = fx.Int32(0)
        switches = fx.Int32(0)
        down_wait = fx.Int32(0)
        i = fx.Int32(0)
        while i < downs:
            slot, tile = down_tile(i)
            # An expert's picks change only at a switch: its run's tiles carry
            # them, off each tile's path.
            if slot != current:
                start = fx.ptr_load(lds["start"] + slot)
                count = uniform(fx.ptr_load(lds["picks_n"] + slot))
                first_pick = fx.ptr_load(
                    lds["picks"] + start + fx.min(lane % 16, count - 1)
                )
                before = fx.Int32(_realtime()) if profile else fx.Int32(0)
                if tid == 0:
                    _bounded_wait(flags, slot, tag, state, bid)
                _lds_barrier()
                if const_expr(profile):
                    waited = waited + (fx.Int32(_realtime()) - before)
                    switches = switches + 1
                for index in range_constexpr(
                    (rows * INTER_ROW_WORDS + THREADS - 1) // THREADS
                ):
                    at = tid + index * THREADS
                    if at < count * INTER_ROW_WORDS:
                        pick = fx.ptr_load(lds["picks"] + start + at // INTER_ROW_WORDS)
                        value = ld_dev(
                            inter, pick * INTER_ROW_WORDS + at % INTER_ROW_WORDS
                        )
                        fx.ptr_store(
                            fx.Int32(value),
                            rows_lds
                            + at // INTER_ROW_WORDS * INTER_LDS_WORDS
                            + at % INTER_ROW_WORDS,
                        )
            current = slot
            item = up_items + i
            before = clock()
            wait_for(item)
            _lds_barrier()
            down_wait = down_wait + since(before)
            if item + ring - 1 < flat:
                issue(item + ring - 1, (item + ring - 1) % ring)
            stage = smem.stages.ptr + (item % ring) * (SLOT_BYTES // 4)
            item_scale = (
                smem.item_scales.ptr
                + (item % ring) * SCALE_WORDS
                + wave * DOWN_STEPS * 32
                + lane
            )
            out = tile * TASK_ROWS + wave * 32 + (lane // 16) * 8
            for m in range_constexpr(tiles):
                if m == 0 or m * 16 < count:
                    index = m * 16 + lane % 16
                    local = fx.min(index, count - 1)
                    accumulator = fx.Vector.filled(16, 0.0, fx.Float32)
                    for step in range_constexpr(DOWN_STEPS):
                        accumulator = _wmma(
                            _weight_operand(stage, wave, lane, step, GROUP_DOWN_BYTES),
                            _activation_fragment(
                                rows_lds, INTER_LDS_WORDS, local, step, lane
                            ),
                            accumulator,
                            fx.ptr_load(item_scale + step * 32),
                            fx.ptr_load(
                                rows_lds + local * INTER_LDS_WORDS + INTER_WORDS + step
                            ),
                        )
                    if index < count:
                        pick = (
                            first_pick
                            if m == 0
                            else fx.ptr_load(lds["picks"] + start + index)
                        )
                        for half in range_constexpr(2):
                            packed = (
                                fx.Vector.from_elements(
                                    [accumulator[half * 8 + e] for e in range(8)],
                                    fx.Float32,
                                )
                                .to(fx.BFloat16)
                                .bitcast(fx.Int32)
                            )
                            bo.buffer_store(
                                packed,
                                rsrc(grouped),
                                (pick * LATENT + out + half * 16) // 2,
                            )
            i = i + 1
        stamp(3)
        stamp(4, downs)
        if const_expr(profile):  # noqa: SIM102
            if tid == 0:
                st_dev(waited, state, MOE_STATE_WORDS + 256 * 5 + bid * 2)
                st_dev(switches, state, MOE_STATE_WORDS + 256 * 5 + bid * 2 + 1)
                st_dev(up_wait, state, MOE_STATE_WORDS + 256 * 7 + bid * 3)
                st_dev(up_epilogue, state, MOE_STATE_WORDS + 256 * 7 + bid * 3 + 1)
                st_dev(down_wait, state, MOE_STATE_WORDS + 256 * 7 + bid * 3 + 2)
                st_dev(up_cycles, state, MOE_STATE_WORDS + 256 * 10 + bid)
        if tid == 0:  # noqa: SIM102
            if task_queue.take(task_queue.word(state + 4)) == 255:
                st_dev(fx.Int32(0), state, 1)
                st_dev(epoch + 1, state, 2)

    @flyc.jit
    def launch(
        ids: Int64,
        latent: Int64,
        scales: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        inter: Int64,
        grouped: Int64,
        flags: Int64,
        state: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            ids, latent, scales, w13, w13s, w2, w2s, inter, grouped, flags, state
        ).launch(grid=(256,), block=(THREADS,), stream=stream)

    return MoeKernel(name, launch, (256,))
