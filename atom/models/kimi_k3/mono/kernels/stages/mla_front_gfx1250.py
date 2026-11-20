# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 MLA front in one resident launch: a full-attention layer's
projections of its normed input up to what ``KimiFullAttention._gated_core``
hands the attention, in place of fused_qkv_a_proj's GEMM and split-K reduce,
the q_a / kv_a RMSNorms, q_b_proj's activation quant and g_proj's GEMM and
reduce.

Every GEMV CTA projects one 16-row task of ``[fused_qkv_a_proj | g_proj]``
(``mla_front_layout``) through ``gemv_tdm_gfx1250``'s split-K TDM GEMV
(``project_split_resident_fp8`` where the CTA's K quarter fits LDS whole,
``project_split`` past 57 rows): BF16 token rows times the FP8 weights
dequantized exactly by their E8M0 block scales, f32 WMMA accumulation, K split
over the TDM cluster and the f32 partials summed in CTA order. A task's sums
are rounded to BF16 and stored, a BF16 pair a word: the gate and rope rows to
their outputs, the q_a and kv_a rows unnormalized to ``qkv_c`` at device
scope. The gate CTAs, which nothing waits for, hold their copies back
(``gate_delay``) so the q_a / kv_a weights stream first.

CTAs of their own (the CUs the GEMV leaves; ``norm_ctas``) normalize the q and
kv rows, a wave a row. ``qkv_c`` holds ``UNWRITTEN`` words wherever no
producer has stored yet (a producer never stores that pair of all-ones NaNs),
so a wave polls its row at device scope until none is left, with no flag
between, and puts them back once it has read the row, for the next launch. A
norm runs AITER ``fused_qk_rmsnorm``'s arithmetic at q_lora 1536: its 256
threads' eight consecutive elements squared and summed in order (a lane holds
eight threads' runs, summed pairwise as the threads' butterfly does), the
lanes' xor butterfly, the mean an IEEE division by the width, eps added in
f32, ``v_rsq_f32``, then ``bf16((x * rstd) * w)`` rounded to f32 at each
product. A q row is quantized for q_b_proj as AITER's
per-1x128 E8M0 ``dynamic_per_group_scaled_quant`` does it
(``per_1x128_e8m0_quant``): each 128-element group's amax over the BF16 values
floored at ``kAmaxFloor``, the E8M0 code ``ceil_pow2(amax * f32(1 / 448))``
(at most 254), the values divided by it with ``v_cvt_scalef32_pk8_fp8_bf16``;
the codes in its transposed layout, group ``g`` of row ``t`` at byte
``g * rows + t``.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int64, T, as_ir_value

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import tdm_timeline
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    ROWS,
    TDM_CLUSTER,
    TDM_SPLIT_TILE_WORDS,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    hw_rsq,
    ld_dev,
    rsrc,
    traced,
    uniform,
    xshfl,
)
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = lay.THREADS
WAVES = lay.WAVES
LANES = GFX1250.wave_size
ACCUMULATOR_ELEMENTS = 8
# a lane's 64 elements: eight 16-byte runs, one AITER thread's each
RUNS = lay.NORM_THREADS // LANES
RUN_WORDS = lay.NORM_RUN // 2
LANE_WORDS = lay.LANE_ELEMENTS // 2
UNWRITTEN = lay.UNWRITTEN - (1 << 32)
QUIET_NAN_PAIR = lay.QUIET_NAN_PAIR

MLA_FRONT_ABI = KernelAbi(
    (
        "x",
        "w_a",
        "s_a",
        "w_g",
        "s_g",
        "q_norm",
        "kv_norm",
        "qkv_c",
        "q",
        "q_fp8",
        "q_scale",
        "kv",
        "k_rope",
        "gate",
        "partials",
        "qkv_seen",
    )
)


@dataclass(frozen=True)
class MlaFrontKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    plan: lay.FrontPlan
    gate_rows: int
    abi: KernelAbi = MLA_FRONT_ABI
    launch_count: int = 1


def _pack(low, high):
    return (
        fx.Vector.from_elements([low, high], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def _task_sum(out, row, token):
    """``project_split``'s sum of task row ``row`` (0..15) for tile token
    ``token``, in wave 0's accumulator layout."""
    lane = token + ROWS * (row // ACCUMULATOR_ELEMENTS)
    word = fx.ptr_load(out + lane * ACCUMULATOR_ELEMENTS + row % ACCUMULATOR_ELEMENTS)
    return fx.Int32(word).bitcast(fx.Float32)


@traced
def _store_task(out, tid, rows, tiles, dest, stride, column):
    """The CTA's task sums rounded to BF16 into ``dest[token, column + r]``
    (rows ``stride`` elements apart), a row pair a thread, at device scope. A
    pair of all-ones NaNs (a NaN input row's, at most) is stored as quiet
    NaNs instead: as ``UNWRITTEN`` its norm wave would wait for it for ever."""
    local_token = tid % (ROWS * 8) // 8
    row_pair = tid % 8
    for tile in range_constexpr(tiles):
        slot = out + tile * TDM_SPLIT_TILE_WORDS
        token = tile * ROWS + local_token
        if (tid < ROWS * 8) & (local_token < min(ROWS, rows - tile * ROWS)):
            word = _pack(
                _task_sum(slot, 2 * row_pair, local_token),
                _task_sum(slot, 2 * row_pair + 1, local_token),
            )
            bo.buffer_store(
                (word == UNWRITTEN).select(fx.Int32(QUIET_NAN_PAIR), word),
                rsrc(dest),
                (token * stride + column) // 2 + row_pair,
                cache_modifier=CM_DEV,
            )


def _row_words(qkv_c, at):
    """This lane's 32 words of a ``qkv_c`` row from word ``at``, at device
    scope."""
    words = []
    for run in range_constexpr(RUNS):
        loaded = fx.Vector(ld_dev(qkv_c, at + run * RUN_WORDS, RUN_WORDS))
        words.extend(fx.Int32(loaded[w]) for w in range_constexpr(RUN_WORDS))
    return fx.Vector.from_elements(words, fx.Int32)


def _unwritten(words, live):
    left = fx.Int32(words[0]) == UNWRITTEN
    for w in range_constexpr(1, LANE_WORDS):
        left = left | (fx.Int32(words[w]) == UNWRITTEN)
    return live & left


@traced
def _poll_row(qkv_c, at, live):
    """The lane's words once its producers stored every one of them. (A
    second poll in flight while one is checked, volatile, saw the stores
    microseconds later, not sooner.)"""
    words = _row_words(qkv_c, at)
    while _unwritten(words, live):
        rocdl.s_sleep(1)
        words = _row_words(qkv_c, at)
    return words


def _scaled(x, rstd, weight):
    """``(x * rstd) * weight``, each product rounded to f32, as AITER's
    v_pk_mul_f32 pair: LLVM would fold the second product into the BF16
    conversion (``v_fma_mixlo_bf16``), one rounding instead of two."""
    return fx.Float32(
        llvm.inline_asm(
            T.f32,
            [fx.Float32(v).ir_value() for v in (x, rstd, weight)],
            "v_mul_f32 $0, $1, $2\n v_mul_f32 $0, $0, $3",
            "=&v,v,v,v",
            has_side_effects=False,
        )
    )


def _run_square_sum(words):
    """One AITER thread's eight elements squared and summed in order (each
    square of a BF16 value is exact in f32)."""
    values = []
    for word in words:
        values.extend((bf_lo(word), bf_hi(word)))
    total = values[0] * values[0]
    for value in values[1:]:
        total = total + value * value
    return total


def _pairwise(values):
    while len(values) > 1:
        values = [values[2 * i] + values[2 * i + 1] for i in range(len(values) // 2)]
    return values[0]


def _store_runs(words, dest, at, cache_modifier=0):
    for run in range_constexpr(RUNS):
        bo.buffer_store(
            fx.Vector.from_elements(
                words[run * RUN_WORDS : (run + 1) * RUN_WORDS], fx.Int32
            ),
            rsrc(dest),
            at + run * RUN_WORDS,
            cache_modifier=cache_modifier,
        )


@traced
def _norm_row(c, job):
    """Job ``job``'s row (a q row below ``rows``, then the kv rows):
    normalized, and a q row quantized for q_b_proj."""
    rows, lane = c["rows"], c["lane"]
    is_q = job < rows
    token = is_q.select(job, job - rows)
    lanes = is_q.select(fx.Int32(lay.Q_LANES), fx.Int32(lay.KV_LANES))
    live = lane < lanes
    at = fx.min(lane, lanes - 1) * LANE_WORDS
    weight_ptr = is_q.select(c["q_norm"], c["kv_norm"])
    gains = []
    for run in range_constexpr(RUNS):
        loaded = fx.Vector(
            bo.buffer_load(
                rsrc(weight_ptr), at + run * RUN_WORDS, vec_width=RUN_WORDS, dtype=T.i32
            )
        )
        gains.extend(fx.Int32(loaded[w]) for w in range_constexpr(RUN_WORDS))
    source = (token * lay.QKV_C + is_q.select(fx.Int32(0), fx.Int32(lay.Q_LORA))) // 2
    landed = _poll_row(c["qkv_c"], source + at, live)
    tdm_timeline.stamp(tdm_timeline.NORM_POLLED)
    words = [
        live.select(fx.Int32(landed[w]), fx.Int32(0))
        for w in range_constexpr(LANE_WORDS)
    ]
    if live:
        _store_runs([fx.Int32(UNWRITTEN)] * LANE_WORDS, c["qkv_c"], source + at, CM_DEV)
        if c["qkv_seen"] != 0:
            _store_runs(words, c["qkv_seen"], source + at)
    runs = [
        _run_square_sum(words[run * RUN_WORDS : (run + 1) * RUN_WORDS])
        for run in range_constexpr(RUNS)
    ]
    total = butterfly(_pairwise(runs), (1, 2, 4, 8, 16), architecture=GFX1250)
    width = is_q.select(fx.Float32(float(lay.Q_LORA)), fx.Float32(float(lay.KV_LORA)))
    eps = is_q.select(fx.Float32(c["q_eps"]), fx.Float32(c["kv_eps"]))
    rstd = hw_rsq(total / width + eps)
    tdm_timeline.stamp(tdm_timeline.NORM_REDUCED)
    normed = [
        _pack(
            _scaled(bf_lo(word), rstd, bf_lo(gain)),
            _scaled(bf_hi(word), rstd, bf_hi(gain)),
        )
        for word, gain in zip(words, gains)
    ]
    if (job >= rows) & live:
        _store_runs(normed, c["kv"], token * (lay.KV_LORA // 2) + lane * LANE_WORDS)
    if is_q & live & (c["q"] != 0):
        _store_runs(normed, c["q"], token * (lay.Q_LORA // 2) + lane * LANE_WORDS)
    if is_q & (c["q_fp8"] != 0):
        _quantize_q(c, normed, token, live)
    tdm_timeline.stamp(tdm_timeline.NORM_STORED)


@traced
def _quantize_q(c, normed, token, live):
    """A normed q row's 64 elements of this lane as FP8 with its group's
    E8M0 code (a group is a lane pair)."""
    lane, rows = c["lane"], c["rows"]
    magnitude = fx.Int32(0)
    for word in normed:
        magnitude = fx.max(magnitude, word & 0x7FFF)
        magnitude = fx.max(magnitude, (word >> 16) & 0x7FFF)
    magnitude = fx.max(magnitude, xshfl(magnitude, 1, GFX1250))
    amax = fx.max(
        fx.Int32(lay.AMAX_FLOOR_BITS).bitcast(fx.Float32),
        (magnitude << 16).bitcast(fx.Float32),
    )
    bits = (amax * fx.Int32(lay.INV_FP8_MAX_BITS).bitcast(fx.Float32)).bitcast(fx.Int32)
    code = fx.Int32(
        fx.Uint32((bits & 0x7FFFFFFF) + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)
    )
    code = fx.min(code, fx.Int32(254))
    scale = (code << 23).bitcast(fx.Float32)
    packed = []
    for run in range_constexpr(RUNS):
        eight = fx.Vector.from_elements(
            normed[run * RUN_WORDS : (run + 1) * RUN_WORDS], fx.Int32
        ).bitcast(fx.BFloat16)
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk8_fp8_bf16(
                T.vec(2, T.i32), as_ir_value(eight), as_ir_value(scale)
            )
        )
        packed.extend(fx.Int32(pair[w]) for w in range_constexpr(2))
    if live:
        for quad in range_constexpr(len(packed) // 4):
            bo.buffer_store(
                fx.Vector.from_elements(packed[quad * 4 : quad * 4 + 4], fx.Int32),
                rsrc(c["q_fp8"]),
                token * (lay.Q_LORA // 4) + lane * (lay.LANE_ELEMENTS // 4) + quad * 4,
            )
        if lane % 2 == 0:
            bo.buffer_store(
                fx.Int8(code), rsrc(c["q_scale"]), (lane // 2) * rows + token
            )


@traced
def _project_task(lds, x, weights, outputs, partials, p, tid, bid):
    """This CTA's task of ``[fused_qkv_a_proj | g_proj]`` projected and
    stored: the q_a / kv_a rows to ``qkv_c``, the rope and gate rows to
    their outputs."""
    w_a, s_a, w_g, s_g = weights
    qkv_c, k_rope, gate = outputs
    rows, tiles, plan, gate_rows, gate_delay = p
    lane = tid % LANES
    wave = tid // LANES
    cluster = uniform(bid // TDM_CLUSTER)
    rank = uniform(bid % TDM_CLUSTER)
    on_a = cluster < lay.A_CLUSTERS
    group_row = on_a.select(
        cluster * lay.GROUP_ROWS, (cluster - lay.A_CLUSTERS) * lay.GROUP_ROWS
    )
    sleeps = [127] * (gate_delay // 127) + [gate_delay % 127] * (gate_delay % 127 > 0)
    if const_expr(sleeps):  # noqa: SIM102
        if cluster >= lay.A_CLUSTERS:
            for i in range_constexpr(len(sleeps)):
                rocdl.s_sleep(sleeps[i])
    if const_expr(plan.resident):
        gemv_tdm.project_split_resident_fp8(
            lds,
            x,
            HIDDEN,
            rows,
            on_a.select(w_a, w_g),
            HIDDEN,
            group_row,
            lane,
            wave,
            partials,
            lds,
            on_a.select(s_a, s_g),
        )
    else:
        gemv_tdm.project_split(
            lds,
            x,
            HIDDEN,
            rows,
            on_a.select(w_a, w_g),
            on_a.select(fx.Int32(lay.QKV_A_PADDED), fx.Int32(gate_rows)),
            HIDDEN,
            [group_row],
            [lambda r: r],
            lane,
            wave,
            partials,
            lds,
            fmt="fp8",
            w_scale=on_a.select(s_a, s_g),
            stages=plan.stages,
        )
    tdm_timeline.stamp(tdm_timeline.PROJECT_DONE)
    row0 = group_row + rank * ROWS
    produces = cluster < lay.ROPE_CLUSTER
    rope = cluster == lay.ROPE_CLUSTER
    dest = produces.select(qkv_c, rope.select(k_rope, gate))
    stride = produces.select(
        fx.Int32(lay.QKV_C),
        rope.select(fx.Int32(lay.ROPE), fx.Int32(gate_rows)),
    )
    column = rope.select(row0 - lay.Q_LORA - lay.KV_LORA, row0)
    _store_task(lds, tid, rows, tiles, dest, stride, column)
    tdm_timeline.stamp(tdm_timeline.STORES_ISSUED)
    tdm_timeline.stores_done()


def build_mla_front(
    rows: int,
    gate_rows: int,
    q_eps: float = 1e-6,
    kv_eps: float = 1e-6,
    norm_ctas: int | None = None,
    gate_delay: int = lay.GATE_DELAY,
) -> MlaFrontKernel:
    """The front at ``rows`` token rows and this rank's ``gate_rows`` g_proj
    rows (64-row groups). ``x`` [rows, HIDDEN] BF16; ``w_a``/``s_a``
    fused_qkv_a_proj's preshuffled FP8 [2176, HIDDEN] and its [17, 56] E8M0
    block scales, ``w_g``/``s_g`` g_proj's [gate_rows, HIDDEN] and
    [gate_rows / 128, 56]; ``q_norm``/``kv_norm`` the BF16 norm weights.
    ``qkv_c``: [MAX_ROWS, 2048] i32 words, every one ``UNWRITTEN`` before the
    first launch (each launch leaves them so); ``partials``
    ``mla_front_layout.partial_bytes``. Out: ``q`` [rows, 1536] BF16 (0: not
    written), ``q_fp8`` [rows, 1536] E4M3 and ``q_scale`` its [12, rows] E8M0
    codes (0: not written), ``kv`` [rows, 512], ``k_rope`` [rows, 64],
    ``gate`` [rows, gate_rows], BF16. ``qkv_seen`` (checks only; 0: not
    written): the unnormalized q_a | kv_a rows the norms read, laid out as
    ``qkv_c``. ``norm_ctas``: the norm waves on CTAs of their own past the
    GEMV's (whole clusters; None: the CUs the GEMV leaves, 0: on the gate
    CTAs); ``gate_delay``: the gate CTAs sleep that many ``s_sleep`` units
    before their first copies, so the q_a / kv_a weights the norms wait for
    stream first (the gate has the slack)."""
    if norm_ctas is None:
        norm_ctas = lay.spare_ctas(gate_rows)
    plan = lay.front_plan(rows)
    grid = lay.grid(gate_rows, norm_ctas)
    gemv_ctas = lay.gemv_ctas(gate_rows)
    first_norm, norm_count = lay.norm_placement(gate_rows, norm_ctas)
    if lay.norm_jobs(rows) > norm_count * WAVES:
        raise ValueError(f"{rows} rows: more norm rows than norm waves")
    tiles = plan.tiles
    words = plan.words
    name = (
        f"k3_mla_front_gfx1250_s{rows}_g{gate_rows}"
        + ("_r" if plan.resident else f"_st{plan.stages}")
        + (f"_n{norm_ctas}" if norm_ctas else "")
        + (f"_d{gate_delay}" if gate_delay else "")
        + tdm_timeline.suffix()
    )
    sources = SOURCES
    timeline = tdm_timeline.key()

    @fx.struct
    class Lds:
        gemv: fx.Array[fx.Int32, words, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        w_a: Int64,
        s_a: Int64,
        w_g: Int64,
        s_g: Int64,
        q_norm: Int64,
        kv_norm: Int64,
        qkv_c: Int64,
        q: Int64,
        q_fp8: Int64,
        q_scale: Int64,
        kv: Int64,
        k_rope: Int64,
        gate: Int64,
        partials: Int64,
        qkv_seen: Int64,
    ):
        _ = sources, timeline
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % LANES
        wave = tid // LANES
        gemv_tdm.prefetch_tokens(x, rows, HIDDEN * 2, bid, tid, gemv_ctas)
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        tdm_timeline.open_cta(bid, wave, timeline, allocator)
        weights = (w_a, s_a, w_g, s_g)
        outputs = (qkv_c, k_rope, gate)
        p = (rows, tiles, plan, gate_rows, gate_delay)
        if const_expr(norm_ctas):
            if bid < gemv_ctas:
                _project_task(lds.gemv.ptr, x, weights, outputs, partials, p, tid, bid)
        else:
            _project_task(lds.gemv.ptr, x, weights, outputs, partials, p, tid, bid)
        c = {
            "rows": rows,
            "lane": lane,
            "qkv_c": qkv_c,
            "q_norm": q_norm,
            "kv_norm": kv_norm,
            "q": q,
            "q_fp8": q_fp8,
            "q_scale": q_scale,
            "kv": kv,
            "qkv_seen": qkv_seen,
            "q_eps": q_eps,
            "kv_eps": kv_eps,
        }
        job = bid - first_norm + norm_count * uniform(wave)
        if (bid >= first_norm) & (job < lay.norm_jobs(rows)):
            _norm_row(c, job)
        tdm_timeline.close_cta()

    @flyc.jit
    def launch(
        x: Int64,
        w_a: Int64,
        s_a: Int64,
        w_g: Int64,
        s_g: Int64,
        q_norm: Int64,
        kv_norm: Int64,
        qkv_c: Int64,
        q: Int64,
        q_fp8: Int64,
        q_scale: Int64,
        kv: Int64,
        k_rope: Int64,
        gate: Int64,
        partials: Int64,
        qkv_seen: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources, timeline
        kernel(
            x,
            w_a,
            s_a,
            w_g,
            s_g,
            q_norm,
            kv_norm,
            qkv_c,
            q,
            q_fp8,
            q_scale,
            kv,
            k_rope,
            gate,
            partials,
            qkv_seen,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    MLA_FRONT_ABI.check(kernel, launch)
    return MlaFrontKernel(name, launch, (grid,), (THREADS,), plan, gate_rows)
