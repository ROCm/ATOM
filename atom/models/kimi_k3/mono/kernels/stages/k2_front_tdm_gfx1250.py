# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replicated gfx1250 K2 front: no all-gather.

Every rank projects the whole router and latent, like the production MoE, plus
its shared gate/up shard, through ``gemv_tdm_gfx1250``. CTA ``p`` owns two
16-row tasks of one weight: in the latent they are one 32-column E8M0 group,
and in the shared gate/up they are the gate and the up rows of the same 16
columns, so a CTA quantizes its latent columns or activates its shared columns
without waiting on any other CTA.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT, SUPPORTED_TP
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import (
    gemv_gfx1250,
    moe_expert_gfx1250,
    moe_gfx1250,
    tdm_timeline,
)
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    TDM_FP8_PAIR_DEPTH,
    TDM_SPLIT_TILE_WORDS,
    tdm_lds_words,
    tdm_split_lds_words,
    tdm_split_out_word,
    tdm_split_post_words,
    tdm_split_resident_fits,
    tdm_split_scratch_bytes,
    tdm_split_stages,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import bf_hi, bf_lo, butterfly, lds_bytes, rsrc, traced
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi
from atom.utils import envs

THREADS = 256
WAVES = 8
ROWS = 16
SHARED_ROWS = 2 * 2 * 3072 // SUPPORTED_TP[0]
SHARED_COLUMNS = SHARED_ROWS // 2
ROUTER_TASKS = EXPERTS // ROWS
LATENT_TASKS = LATENT // ROWS
SHARED_TASKS = SHARED_ROWS // ROWS
LATENT_PAIR0 = ROUTER_TASKS // 2
SHARED_PAIR0 = (ROUTER_TASKS + LATENT_TASKS) // 2
PAIRS = (ROUTER_TASKS + LATENT_TASKS + SHARED_TASKS) // 2
ACCUMULATOR_WORDS = WAVES * GFX1250.wave_size * gemv_gfx1250.ACCUMULATOR_ELEMENTS
# The projection epilogue reduces both tasks of a tile in one round.
assert THREADS == 2 * ROWS * 8
# Split-K (``gemv_tdm.project_split``): a router cluster owns four tasks (a
# task a CTA), a latent one eight (a 32-column group a CTA) and a shared one
# the gate and up rows of 64 columns (16 a CTA).
SPLIT_ROUTER_CLUSTERS = ROUTER_TASKS // 4
SPLIT_LATENT_CLUSTERS = LATENT_TASKS // 8
SPLIT_SHARED_CLUSTERS = SHARED_TASKS // 8
SPLIT_CTAS = 4 * (SPLIT_ROUTER_CLUSTERS + SPLIT_LATENT_CLUSTERS + SPLIT_SHARED_CLUSTERS)

REPLICATED_FRONT_ABI = KernelAbi(
    (
        "x",
        "router_weight",
        "latent_weight",
        "shared_weight",
        "shared_scale",
        "router_logits",
        "latent_fp4",
        "latent_scales",
        "shared_hidden",
    )
)


# Split-K with the latent weight MXFP4: its E8M0 [LATENT, HIDDEN / 32] plane.
REPLICATED_FRONT_FP4_ABI = KernelAbi((*REPLICATED_FRONT_ABI.names, "latent_scale"))
_SPLIT_KERNEL_ABI = KernelAbi((*REPLICATED_FRONT_FP4_ABI.names, "scratch"))


@dataclass(frozen=True)
class K2ReplicatedFrontKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    abi: KernelAbi = REPLICATED_FRONT_ABI
    launch_count: int = 1
    # split-K: the launch's f32 partials, kept as long as the launcher
    scratch: object = None


def _stage_accumulators(accumulators, red, tiles, wave, lane):
    """Both tasks' accumulators of every tile, as words at ``red``."""
    for tile in range_constexpr(tiles):
        for task in range_constexpr(2):
            fx.ptr_store(
                accumulators[task][tile].bitcast(fx.Int32),
                red
                + (2 * tile + task) * ACCUMULATOR_WORDS
                + gemv_gfx1250.accumulator_offset(wave, lane),
            )


def _row_sum(red, row, token, waves=WAVES):
    """``gemv_gfx1250.row_sum`` over accumulators kept as words."""
    lane = token + ROWS * (row // gemv_gfx1250.ACCUMULATOR_ELEMENTS)
    element = row % gemv_gfx1250.ACCUMULATOR_ELEMENTS
    if waves == 1:
        word = fx.ptr_load(red + gemv_gfx1250.accumulator_offset(0, lane) + element)
        return fx.Int32(word).bitcast(fx.Float32)
    total = fx.Float32(0.0)
    for wave in range(waves):
        word = fx.ptr_load(red + gemv_gfx1250.accumulator_offset(wave, lane) + element)
        total = total + fx.Int32(word).bitcast(fx.Float32)
    return total


@traced
def _front_tasks(
    lds,
    x,
    weights,
    outputs,
    pair,
    tid,
    lane,
    wave,
    rows,
    tiles,
    fp8_shared,
    prologue=None,
    x_cache_policy=0,
    wide=False,
    depth=gemv_tdm.DEPTH,
    paired=False,
    pipe=None,
    fp8_depth=None,
):
    """CTA ``pair``'s two projection tasks of ``x`` and their epilogue. The
    accumulators are reduced in the weight slabs' LDS, consumed by then (not
    the token slabs', which a cluster peer's multicast may still be filling).
    ``paired``: the projections walk K two slabs a step, the FP8 shared one
    ``fp8_depth`` ahead (None: ``TDM_FP8_PAIR_DEPTH``, unpaired
    ``FP8_DEPTH``). ``pipe``: ``front_pipe``'s, ``gemv_tdm.project``'s late,
    token_steps, early_scales and unroll and the epilogue's hw_fp4 (None:
    their defaults)."""
    if fp8_depth is None:
        fp8_depth = TDM_FP8_PAIR_DEPTH if paired else gemv_tdm.FP8_DEPTH
    pipe = pipe or {}
    gemv_pipe = {
        k: v
        for k, v in pipe.items()
        if k in ("late", "token_steps", "early_scales", "unroll")
    }
    red = (
        lds.gemv.ptr
        + tdm_lds_words(
            rows, 2, group=2 if paired else 1, token_steps=pipe.get("token_steps", 2)
        )[0]
    )
    router_weight, latent_weight, shared_weight, shared_scale = weights
    router = pair < LATENT_PAIR0
    latent = (pair >= LATENT_PAIR0) & (pair < SHARED_PAIR0)
    weight = router.select(router_weight, latent.select(latent_weight, shared_weight))
    weight_rows = router.select(
        fx.Int32(EXPERTS),
        latent.select(fx.Int32(LATENT), fx.Int32(SHARED_ROWS)),
    )
    first_pair = router.select(
        fx.Int32(0),
        latent.select(fx.Int32(LATENT_PAIR0), fx.Int32(SHARED_PAIR0)),
    )
    row0 = (pair - first_pair) * (2 * ROWS)
    shared_column0 = (pair - first_pair) * ROWS
    second = (router | latent).select(row0 + ROWS, shared_column0 + SHARED_COLUMNS)
    first = (router | latent).select(row0, shared_column0)
    if const_expr(fp8_shared):
        if pair >= SHARED_PAIR0:
            _stage_accumulators(
                gemv_tdm.project(
                    lds.gemv.ptr,
                    x,
                    HIDDEN,
                    rows,
                    shared_weight,
                    SHARED_ROWS,
                    HIDDEN,
                    [first, second],
                    lane,
                    wave,
                    w_scale=shared_scale,
                    depth=fp8_depth,
                    prologue=prologue,
                    x_cache_policy=x_cache_policy,
                    wide=wide,
                    paired=paired,
                    **gemv_pipe,
                ),
                red,
                tiles,
                wave,
                lane,
            )
        else:
            _stage_accumulators(
                gemv_tdm.project(
                    lds.gemv.ptr,
                    x,
                    HIDDEN,
                    rows,
                    weight,
                    weight_rows,
                    HIDDEN,
                    [first, second],
                    lane,
                    wave,
                    prologue=prologue,
                    x_cache_policy=x_cache_policy,
                    wide=wide,
                    depth=depth,
                    paired=paired,
                    **gemv_pipe,
                ),
                red,
                tiles,
                wave,
                lane,
            )
    else:
        _stage_accumulators(
            gemv_tdm.project(
                lds.gemv.ptr,
                x,
                HIDDEN,
                rows,
                weight,
                weight_rows,
                HIDDEN,
                [first, second],
                lane,
                wave,
                prologue=prologue,
                x_cache_policy=x_cache_policy,
                wide=wide,
                depth=depth,
                paired=paired,
                **gemv_pipe,
            ),
            red,
            tiles,
            wave,
            lane,
        )
    gpu.barrier()
    tdm_timeline.stamp(tdm_timeline.STAGED)
    _front_epilogue(
        lds,
        outputs,
        router,
        latent,
        pair >= SHARED_PAIR0,
        row0,
        shared_column0,
        lambda task, tile: red + (2 * tile + task) * ACCUMULATOR_WORDS,
        WAVES,
        None,
        tid,
        rows,
        tiles,
        pipe.get("hw_fp4", False),
    )
    tdm_timeline.stamp(tdm_timeline.STORES_ISSUED)
    tdm_timeline.stores_done()


@traced
def _front_epilogue(
    lds,
    outputs,
    router,
    latent,
    shared,
    row0,
    shared_column0,
    slot_of,
    waves,
    owned,
    tid,
    rows,
    tiles,
    hw_fp4=False,
):
    """The two projected tasks' epilogue: router logits, the latent's FP4
    group or the shared gate/up's activation. ``slot_of(task, tile)``: the
    tile's accumulators of ``waves`` waves; ``owned``: how many of the two
    tasks the CTA projected (both if None); ``hw_fp4``: the latent's FP4 by
    ``_fp4_word``."""
    router_logits, latent_fp4, latent_scales, shared_hidden = outputs
    # A round reduces one tile: threads [0, 128) the first task, the
    # rest the second.
    task = tid // (ROWS * 8)
    for tile in range_constexpr(tiles):
        tile_rows = min(ROWS, rows - tile * ROWS)
        slot = slot_of(task, tile)
        local_token = tid % (ROWS * 8) // 8
        row_pair = tid % 8
        token = tile * ROWS + local_token
        low = _row_sum(slot, 2 * row_pair, local_token, waves)
        high = _row_sum(slot, 2 * row_pair + 1, local_token, waves)
        row = row0 + task * ROWS + 2 * row_pair
        packed = (
            fx.Vector.from_elements([low, high], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        live = local_token < tile_rows
        if const_expr(owned is not None):
            live = live & (task < owned)
        if live:
            if router:
                bo.buffer_store(low, rsrc(router_logits), token * EXPERTS + row)
                bo.buffer_store(high, rsrc(router_logits), token * EXPERTS + row + 1)
            else:
                fx.ptr_store(packed, lds.group.ptr + token * 16 + task * 8 + row_pair)
    tdm_timeline.stamp(tdm_timeline.EPI_SUMS)
    gpu.barrier()
    tdm_timeline.stamp(tdm_timeline.EPI_BARRIER)
    if shared:
        for round_index in range_constexpr((rows * 8 + THREADS - 1) // THREADS):
            work = tid + THREADS * round_index
            if work < rows * 8:
                token = work // 8
                column_pair = work % 8
                gate = fx.ptr_load(lds.group.ptr + token * 16 + column_pair)
                up = fx.ptr_load(lds.group.ptr + token * 16 + 8 + column_pair)
                low = moe_expert_gfx1250._situ(bf_lo(gate), bf_lo(up))
                high = moe_expert_gfx1250._situ(bf_hi(gate), bf_hi(up))
                bo.buffer_store(
                    fx.Vector.from_elements([low, high], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0],
                    rsrc(shared_hidden),
                    (token * SHARED_COLUMNS + shared_column0) // 2 + column_pair,
                )
    if latent & (tid < rows * 4):
        token = tid // 4
        packed_word = tid % 4
        words = [
            fx.ptr_load(lds.group.ptr + token * 16 + packed_word * 4 + index)
            for index in range_constexpr(4)
        ]
        maximum_bits = fx.Int32(0)
        for word in words:
            maximum_bits = fx.max(maximum_bits, word & 0x7FFF)
            maximum_bits = fx.max(maximum_bits, (word >> 16) & 0x7FFF)
        maximum = (maximum_bits << 16).bitcast(fx.Float32)
        maximum = butterfly(maximum, (2, 1), fx.max, architecture=GFX1250)
        code = moe_gfx1250._e8m0_roundup(maximum)
        reciprocal_scale = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        group = row0 // 32
        bo.buffer_store(
            (
                _fp4_word(words, code, reciprocal_scale)
                if const_expr(hw_fp4)
                else moe_gfx1250._pack_bf16_words(words, reciprocal_scale)
            ),
            rsrc(latent_fp4),
            token * (LATENT // 8) + group * 4 + packed_word,
        )
        if packed_word == 0:
            bo.buffer_store(
                fx.Int8(code),
                rsrc(latent_scales),
                token * (LATENT // 32) + group,
            )


@traced
def _fp4_word(words, code, reciprocal_scale):
    """``moe_gfx1250._pack_bf16_words(words, reciprocal_scale)`` of up to four
    BF16 pair words (absent ones zero) with E8M0 ``code``, as one
    ``v_cvt_scalef32_pk8_fp4_bf16`` by ``2^(code - 127)``: the same nibbles at
    every code but 254, an infinite amax's
    (``tools/kimi_k3_mono_fp4_quant_probe.py``), where the software
    conversion stays."""
    words = list(words) + [fx.Int32(0)] * (4 - len(words))
    packed = fx.Int32(
        rocdl.cvt_scalef32_pk8_fp4_bf16(
            T.i32,
            as_ir_value(fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)),
            as_ir_value((code << 23).bitcast(fx.Float32)),
        )
    )
    if code == 254:
        packed = moe_gfx1250._pack_bf16_words(words, reciprocal_scale)
    return packed


def front_split_plan(rows, fp8_shared, fp4_latent, lds_words, resident=False):
    """``{kind: (format, stages, resident)}`` of the split-K front's three
    projections in ``lds_words`` LDS words, or None where one does not fit.
    ``resident``: each projection whose whole weight share fits is resident."""
    kinds = {
        "router": ("bf16", 1),
        "latent": ("fp4" if fp4_latent else "bf16", 2),
        "shared": ("fp8" if fp8_shared else "bf16", 2),
    }
    plan = {}
    for kind, (fmt, groups) in kinds.items():
        if tdm_split_post_words(rows) > lds_words:
            return None
        whole = resident and tdm_split_resident_fits(
            rows, HIDDEN, fmt, lds_words, groups
        )
        stages = tdm_split_stages(rows, HIDDEN, fmt, lds_words, groups)
        if not stages and not whole:
            return None
        plan[kind] = (fmt, stages, whole)
    return plan


def front_split_lds_words(rows, plan):
    return max(
        tdm_split_lds_words(
            rows,
            HIDDEN,
            fmt,
            gemv_tdm.tdm_split_slabs(HIDDEN) if whole else stages,
            1 if kind == "router" else 2,
        )
        for kind, (fmt, stages, whole) in plan.items()
    )


@traced
def front_split_tasks(
    lds,
    x,
    weights,
    outputs,
    tid,
    lane,
    wave,
    rows,
    tiles,
    scratch,
    plan,
    prologue=None,
    x_cache_policy=0,
):
    """The split-K front's projections and epilogue, CTA ``block_idx.x <
    SPLIT_CTAS``: cluster c owns router tasks [4c, 4c + 4), then latent tasks
    [8c', 8c' + 8), then the shared gate and up rows of 64 columns.
    ``weights``: router, latent, latent scale (MXFP4), shared, shared scale.
    ``plan``: ``front_split_plan``; ``lds.gemv`` holds ``front_split_lds_words``.
    ``prologue``/``x_cache_policy``: ``gemv_tdm.project_split``'s."""
    router_weight, latent_weight, latent_scale, shared_weight, shared_scale = weights
    rank = fx.Int32(rocdl.cluster_workgroup_id_x())
    cluster = (fx.Int32(fx.block_idx.x) - rank) // 4
    router = cluster < SPLIT_ROUTER_CLUSTERS
    latent_cluster = cluster - SPLIT_ROUTER_CLUSTERS
    shared_cluster = latent_cluster - SPLIT_LATENT_CLUSTERS
    latent = (latent_cluster >= 0) & (shared_cluster < 0)
    shared = shared_cluster >= 0
    gemv = lds.gemv.ptr
    out = gemv + tdm_split_out_word(rows)

    def project(weight, weight_rows, groups, owners, kind, scale):
        fmt, stages, whole = plan[kind]
        gemv_tdm.project_split(
            gemv,
            x,
            HIDDEN,
            rows,
            weight,
            weight_rows,
            HIDDEN,
            groups,
            owners,
            lane,
            wave,
            scratch,
            out,
            fmt=fmt,
            w_scale=scale if fmt != "bf16" else None,
            stages=stages,
            resident=whole,
            prologue=prologue,
            x_cache_policy=x_cache_policy,
        )

    if router:
        project(
            router_weight,
            EXPERTS,
            [cluster * 64],
            [lambda r: r],
            "router",
            None,
        )
    if latent:
        project(
            latent_weight,
            LATENT,
            [latent_cluster * 128, latent_cluster * 128 + 64],
            [lambda r: 2 * r, lambda r: 2 * r + 1],
            "latent",
            latent_scale,
        )
    if shared:
        project(
            shared_weight,
            SHARED_ROWS,
            [shared_cluster * 64, SHARED_COLUMNS + shared_cluster * 64],
            [lambda r: r, lambda r: r + 4],
            "shared",
            shared_scale,
        )
    _front_epilogue(
        lds,
        outputs,
        router,
        latent,
        shared,
        router.select(cluster * 64 + rank * ROWS, latent_cluster * 128 + rank * 32),
        shared_cluster * 64 + rank * ROWS,
        lambda task, tile: out + (task * tiles + tile) * TDM_SPLIT_TILE_WORDS,
        1,
        router.select(fx.Int32(1), fx.Int32(2)),
        tid,
        rows,
        tiles,
    )


def _front_words(rows: int, depth: int, paired: bool, token_steps: int = 2) -> int:
    """The replicated front's GEMV LDS words: the BF16 plan, which the FP8
    shared one fits in."""
    return sum(
        tdm_lds_words(
            rows, 2, depth=depth, group=2 if paired else 1, token_steps=token_steps
        )
    )


# Late copies (``gemv_tdm.project(late)``), three steps of tokens: (paired,
# BF16 depth, FP8 depth), the first whose LDS fits the width. A late weight
# copy is issued a barrier later, so it runs a slab (unpaired) or two
# (paired) deeper for the same copies in flight; a wave still keeps three.
# Shallower plans starve their copies: paired four ahead (10-20 rows), and
# unpaired where the default pipeline pairs (16-30), or three or two ahead
# (past 50 rows), ran whole layers slower than the default pipeline.
_LATE_PLANS = (
    (True, 6, 6),
    (False, 5, 6),
    (False, 4, 6),
)
# The narrowest width an unpaired late plan won whole sustained layers at
# (40 and 48 rows did; 32 ran slower). The paired one (to 9 rows) won none.
LATE_MIN_ROWS = 33


def front_pipe(
    rows: int, pairable: bool, lds_bytes: int, sites=None, paired_default=False
):
    """The front's pipeline for ``ATOM_KIMI_MONO_FRONT_PIPE``'s ``sites``
    (None: the env's) at ``rows``. "late": ``{paired, depth, fp8_depth, late,
    token_steps, early_scales}`` from ``_LATE_PLANS``, unpaired ones from
    ``LATE_MIN_ROWS`` rows where the default pipeline (``paired_default``) is
    unpaired ("ua": at any width), paired ones with "lp" if ``pairable``;
    none where no plan fits ``lds_bytes`` beside the group rows. "early":
    ``early_scales`` at any width; "hw4": the latent's FP4 by ``_fp4_word``
    (``hw_fp4``); "u2" (with a late plan): the step loop unrolled twice
    (``unroll``). The same bits."""
    sites = envs.ATOM_KIMI_MONO_FRONT_PIPE if sites is None else sites
    pipe = {}
    if "late" in sites:
        for paired, depth, fp8_depth in _LATE_PLANS:
            if paired and not (pairable and "lp" in sites):
                continue
            if (
                not paired
                and (paired_default or rows < LATE_MIN_ROWS)
                and "ua" not in sites
            ):
                continue
            if _front_words(rows, depth, paired, 3) + rows * 16 <= lds_bytes // 4:
                pipe = {
                    "paired": paired,
                    "depth": depth,
                    "fp8_depth": fp8_depth,
                    "late": True,
                    "token_steps": 3,
                    "early_scales": True,
                }
                break
    if "early" in sites:
        pipe["early_scales"] = True
    if "hw4" in sites:
        pipe["hw_fp4"] = True
    if pipe.get("late") and "u2" in sites:
        pipe["unroll"] = 2
    return pipe


def _pipe_suffix(pipe) -> str:
    return (
        ("_late" if pipe.get("late") else "")
        + ("_es" if pipe.get("early_scales") else "")
        + ("_hw4" if pipe.get("hw_fp4") else "")
        + (f"_u{pipe['unroll']}" if pipe.get("unroll") else "")
    )


def build_replicated_front(
    plan: Gfx1250KimiBuildPlan,
    fp8_shared: bool = False,
    wide: bool = True,
    depth: int = 4,
    fp4_latent: bool = False,
    split: bool | None = None,
    pipe_sites=None,
) -> K2ReplicatedFrontKernel:
    """``fp8_shared``: the shared gate/up is ATOM's online per_1x128 FP8 with
    its E8M0 block scales in ``shared_scale``. ``split`` (default: "front" in
    ATOM_KIMI_MONO_SPLITK): split K over each cluster where the LDS holds it.
    ``fp4_latent`` (split only): the latent weight is MXFP4, its E8M0 plane
    the launcher's ``latent_scale``. ``pipe_sites``: ``front_pipe``'s (None:
    ATOM_KIMI_MONO_FRONT_PIPE)."""
    if split is None:
        split = "front" in envs.ATOM_KIMI_MONO_SPLITK
    if split or fp4_latent:
        built = _build_split_front(plan, fp8_shared, fp4_latent)
        if built is not None:
            return built
        if fp4_latent:
            raise ValueError(f"split-K front with an MXFP4 latent: {plan.tokens} rows")
    if plan.architecture is not GFX1250:
        raise ValueError("replicated K2 front requires gfx1250")
    if plan.threads != THREADS or plan.waves != WAVES:
        raise ValueError("replicated K2 front requires 256 threads/eight waves")
    if PAIRS > plan.blocks:
        raise ValueError(f"replicated K2 front needs {PAIRS} CTAs")
    rows = plan.tokens
    tiles = (rows + ROWS - 1) // ROWS
    pairable = wide and depth % 2 == 0 and "front" in gemv_tdm.paired_sites()
    paired = pairable
    # Unpaired where four slabs ahead do not fit: two ahead is slower (32-40).
    if paired and _front_words(rows, depth, True) + rows * 16 > plan.lds_bytes // 4:
        paired = False
    pipe = front_pipe(rows, pairable, plan.lds_bytes, pipe_sites, paired)
    paired = pipe.pop("paired", paired)
    depth = pipe.pop("depth", depth)
    fp8_depth = pipe.pop(
        "fp8_depth", TDM_FP8_PAIR_DEPTH if paired else gemv_tdm.FP8_DEPTH
    )
    token_steps = pipe.get("token_steps", 2)
    staged = _front_words(rows, depth, paired, token_steps)
    group = 2 if paired else 1
    assert (
        staged - tdm_lds_words(rows, 2, group=group, token_steps=token_steps)[0]
        >= 2 * tiles * ACCUMULATOR_WORDS
    )
    # The FP8 shared slabs reuse the LDS planned for the BF16 ones, and the
    # shared CTAs start a cluster, so no cluster mixes the two pipelines.
    assert (
        sum(tdm_lds_words(rows, 2, True, fp8_depth, group, token_steps)) <= staged
    )
    assert SHARED_PAIR0 % gemv_tdm.CLUSTER_LAUNCH["cluster"][0] == 0

    @fx.struct
    class Lds:
        gemv: fx.Array[fx.Int32, staged, 16]
        # Per token, 32 projected latent columns or 16 shared gate then 16 up
        # columns, as 16 BF16 pairs.
        group: fx.Array[fx.Int32, rows * 16, 16]

    if lds_bytes(Lds) > plan.lds_bytes:
        raise ValueError(f"replicated K2 front needs {lds_bytes(Lds)} B of LDS")
    name = (
        f"k3_k2_gfx1250_replicated_front{'_w' if wide else ''}_d{depth}"
        f"{'_ps' if paired else ''}{_pipe_suffix(pipe)}"
        f"{f'_f{fp8_depth}' if pipe.get('late') else ''}"
        f"_s{rows}{'_f8' if fp8_shared else ''}{gemv_tdm.cold_suffix()}"
        f"{tdm_timeline.suffix()}"
    )

    sources = SOURCES
    timeline = tdm_timeline.key()

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
    ):
        _ = sources, timeline
        tid = fx.thread_idx.x
        pair = fx.block_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        gemv_tdm.prefetch_code(wave, 10 << 10)
        gemv_tdm.prefetch_tokens(x, rows, HIDDEN * 2, pair, tid, PAIRS)
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        tdm_timeline.open_cta(pair, wave, timeline, allocator)
        if pair < PAIRS:
            _front_tasks(
                lds,
                x,
                (router_weight, latent_weight, shared_weight, shared_scale),
                (router_logits, latent_fp4, latent_scales, shared_hidden),
                pair,
                tid,
                lane,
                wave,
                rows,
                tiles,
                fp8_shared,
                wide=wide,
                depth=depth,
                paired=paired,
                pipe=pipe,
                fp8_depth=fp8_depth,
            )
        tdm_timeline.close_cta()

    @flyc.jit
    def launch(
        x: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources, timeline
        kernel(
            x,
            router_weight,
            latent_weight,
            shared_weight,
            shared_scale,
            router_logits,
            latent_fp4,
            latent_scales,
            shared_hidden,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(PAIRS,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    REPLICATED_FRONT_ABI.check(kernel, launch)
    gemv_tdm.apply_preload(launch)
    return K2ReplicatedFrontKernel(name, launch, (PAIRS,), (THREADS,))


class _TailedLauncher:
    """``launch`` with trailing arguments the caller does not pass (the split
    front's scratch, the BF16 latent's absent scale); its ``_mem_cache`` is
    ``launch``'s, for ``launcher_resources``."""

    def __init__(self, launch, tail):
        self.launch = launch
        self.tail = tail

    def __call__(self, *args, **kwargs):
        return self.launch(*args, *self.tail, **kwargs)

    @property
    def _mem_cache(self):
        return self.launch._mem_cache


def _split_front_plan(plan, fp8_shared, fp4_latent):
    if plan.architecture is not GFX1250 or SPLIT_CTAS > plan.blocks:
        return None
    return front_split_plan(
        plan.tokens,
        fp8_shared,
        fp4_latent,
        (plan.lds_bytes - 1024) // 4 - plan.tokens * 16,
    )


def split_front_fits(plan, fp8_shared: bool = False, fp4_latent: bool = False) -> bool:
    """Whether ``build_replicated_front`` splits K at ``plan``'s width (an
    MXFP4 latent needs it)."""
    return _split_front_plan(plan, fp8_shared, fp4_latent) is not None


def _build_split_front(plan, fp8_shared, fp4_latent):
    import torch

    split = _split_front_plan(plan, fp8_shared, fp4_latent)
    if split is None:
        return None
    rows = plan.tokens
    tiles = (rows + ROWS - 1) // ROWS
    group_words = rows * 16
    words = front_split_lds_words(rows, split)

    @fx.struct
    class Lds:
        gemv: fx.Array[fx.Int32, words, 16]
        group: fx.Array[fx.Int32, group_words, 16]

    if lds_bytes(Lds) > plan.lds_bytes:
        return None
    stages = "".join(str(stages) for _, stages, _ in split.values())
    name = (
        f"k3_k2_gfx1250_replicated_front_sk{stages}_s{rows}"
        f"{'_f8' if fp8_shared else ''}{'_f4' if fp4_latent else ''}"
    )
    scratch = torch.empty(
        tdm_split_scratch_bytes(SPLIT_CTAS, rows),
        dtype=torch.uint8,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    sources = SOURCES

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
        latent_scale: Int64,
        scratch: Int64,
    ):
        _ = sources
        tid = fx.thread_idx.x
        lane = tid % GFX1250.wave_size
        wave = tid // GFX1250.wave_size
        lds = fx.SharedAllocator().allocate(Lds).peek()
        front_split_tasks(
            lds,
            x,
            (router_weight, latent_weight, latent_scale, shared_weight, shared_scale),
            (router_logits, latent_fp4, latent_scales, shared_hidden),
            tid,
            lane,
            wave,
            rows,
            tiles,
            scratch,
            split,
        )

    @flyc.jit
    def launch(
        x: Int64,
        router_weight: Int64,
        latent_weight: Int64,
        shared_weight: Int64,
        shared_scale: Int64,
        router_logits: Int64,
        latent_fp4: Int64,
        latent_scales: Int64,
        shared_hidden: Int64,
        latent_scale: Int64,
        scratch: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            x,
            router_weight,
            latent_weight,
            shared_weight,
            shared_scale,
            router_logits,
            latent_fp4,
            latent_scales,
            shared_hidden,
            latent_scale,
            scratch,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(SPLIT_CTAS,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    _SPLIT_KERNEL_ABI.check(kernel, launch)
    tail = (scratch.data_ptr(),) if fp4_latent else (0, scratch.data_ptr())
    launcher = _TailedLauncher(launch, tail)

    return K2ReplicatedFrontKernel(
        name,
        launcher,
        (SPLIT_CTAS,),
        (THREADS,),
        REPLICATED_FRONT_FP4_ABI if fp4_latent else REPLICATED_FRONT_ABI,
        scratch=scratch,
    )
