# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 DSpark draft MLA front in one resident launch: what
``K3DSparkMLAAttention.forward`` computes from its input norm's FP8 row
before the attention, in place of ``fused_qkv_a_proj``'s GEMM (and its
split-K reduce) and ``fused_qk_rmsnorm_group_quant``.

Every GEMV CTA (``mla_draft_front_layout``; a cluster of four owns a 64-row
group) copies its four 28 KB runs of the preshuffled FP8 weight to LDS by the
TDM, a wave half a run, and chains the scaled FP8 WMMA (16x16x128, E8M0 codes
on the weights and the tokens) over them a seven-group piece at a time, the
token rows of a piece -- the FP8 row with its transposed per-1x128 codes --
staged in LDS in the WMMA operand layout by all threads, double-buffered (the
next piece's loads in flight while one is multiplied). That is production's
FlyDSL ``gemm_a8w8_mxfp8_128`` at the split-K the
dispatch picks for the width: at ``ksplit`` 4 a chain a task a token tile
over the CTA's K quarter, the owner of each task summing the four quarters in
order through the split-K scratch past one cluster barrier (the split-K
reduce's sum), at ``ksplit`` 1 a chain over all of K. The sums are rounded to
BF16: the rope rows to ``k_rope``, the q_a / kv_a rows to ``qkv_c`` at device
scope.

The norm CTAs past the GEMV's normalize a row a wave, polling ``qkv_c`` for
the ``UNWRITTEN`` words its producers replace (``mla_front_gfx1250``'s
protocol) and putting them back once read. A row runs AITER
``fused_qk_rmsnorm_group_quant``'s arithmetic (128 threads of 16 elements):
each thread's squares summed in order, the threads' xor butterfly a wave,
the waves' sums pairwise, the mean an IEEE division by the width, eps added,
``v_rsq_f32``, then ``(x * rstd) * w`` rounded to f32 at each product. A kv
row is stored as BF16; a q row is quantized as the kernel does it from those
f32 values: each 128-element group's amax floored at 1e-10, the fp32 scale
``amax * f32(1 / 448)`` (stored, transposed: group ``g`` of row ``t`` at
``g * rows + t``), each value times ``v_rcp_f32`` of it, clamped to +-448 by
``v_med3_f32`` and converted by ``v_cvt_pk_fp8_f32``.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as lay
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.models.kimi_k3.mono.kernels.stages.mla_absorb_gfx1250 import (
    _byte,
    _fp8_words,
    _mma_fp8,
    _saturated,
    _scalar_byte,
    _splat,
    _store_words,
    _words,
)
from atom.models.kimi_k3.mono.kernels.stages.mla_front_gfx1250 import _pack
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import (
    CM_DEV,
    bf_hi,
    bf_lo,
    butterfly,
    hw_rcp,
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
LANES = lay.LANES
ACC = 8
SCOPE_SE = gemv_tdm.SCOPE_SE
UNWRITTEN = lay.UNWRITTEN - (1 << 32)
GROUP_WORDS = lay.GROUP_BYTES // 4
# a norm wave lane's run: one AITER thread's 16 elements, eight words
RUN_WORDS = lay.NORM_THREAD_ELEMENTS // 2
WAVE_WORDS = lay.NORM_WAVE_ELEMENTS // 2

MLA_DRAFT_FRONT_ABI = KernelAbi(
    (
        "x",
        "x_scale",
        "w_a",
        "s_a",
        "q_norm",
        "kv_norm",
        "qkv_c",
        "q_fp8",
        "q_scale",
        "kv",
        "k_rope",
        "partials",
        "qkv_seen",
    )
)


@dataclass(frozen=True)
class MlaDraftFrontKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    plan: lay.DraftFrontPlan
    abi: KernelAbi = MLA_DRAFT_FRONT_ABI
    launch_count: int = 1


def _weight_codes(c, block):
    """The E8M0 codes of the chain's K groups in the weight's 128-row block
    ``block``: 14 scalar byte loads at ksplit 4 (a quarter), 14 scalar words
    of four at ksplit 1 (all of K: a row of codes is word-aligned)."""
    if const_expr(c["ksplit"] == 4):
        first = block * lay.GROUPS + c["rank"] * lay.RUN_GROUPS
        return [_scalar_byte(c["s_a"], first + j) for j in range(lay.RUN_GROUPS)]
    words = [
        fx.Int32(
            bo.buffer_load(
                rsrc(c["s_a"]),
                block * (lay.GROUPS // 4) + i,
                vec_width=1,
                dtype=T.i32,
                is_scalar=True,
            )
        )
        for i in range(lay.GROUPS // 4)
    ]
    return [(words[j // 4] >> (j % 4 * 8)) & 0xFF for j in range(lay.GROUPS)]


def _weight_operand(lds, group, lane):
    """The 16 weight rows of one K group in AITER's preshuffle as they lie in
    LDS (``[4 chunks][2 halves][16 rows][16 B]``): a lane's 16 bytes of each
    32-wide chunk, its half ``lane / 16``."""
    words = []
    for k in range_constexpr(4):
        byte = ((k * 2 + lane // ROWS) * ROWS + lane % ROWS) * 16
        raw = fx.Vector(
            fx.ptr_load(
                lds + group * GROUP_WORDS + byte // 4,
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
        )
        words.extend(fx.Int32(raw[i]) for i in range_constexpr(4))
    return fx.Vector.from_elements(words, fx.Int32)


def _piece_column(c, piece):
    """First byte (of a token row) of K piece ``piece`` of the CTA's chains."""
    if const_expr(c["ksplit"] == 4):
        return c["rank"] * (lay.RUN_GROUPS * lay.K_GROUP) + piece * lay.PIECE_BYTES
    return fx.Int32(piece * lay.PIECE_BYTES)


def _load_piece(c, piece, tid):
    """Every thread's share of K piece ``piece``'s token rows (16-byte units,
    consecutive threads consecutive bytes) and of their E8M0 codes (a code a
    word), loaded to registers: ``(units, codes)``, each a list of (value,
    LDS word, live)."""
    rows = c["rows"]
    units = rows * lay.PIECE_UNITS
    column = _piece_column(c, piece)
    loaded = []
    for i in range_constexpr(-(-units // THREADS)):
        u = fx.min(tid + THREADS * i, units - 1)
        token, rem = u // lay.PIECE_UNITS, u % lay.PIECE_UNITS
        words = _words(c["x"], (token * HIDDEN + column + rem * 16) // 4, 4)
        unit = ((token // ROWS * lay.PIECE_GROUPS + rem // 8) * 8 + rem % 8) * ROWS
        loaded.append((words, (unit + token % ROWS) * 4, tid + THREADS * i < units))
    codes = []
    count = lay.PIECE_GROUPS * rows
    first = (column // lay.K_GROUP) * rows
    for i in range_constexpr(-(-count // THREADS)):
        n = fx.min(tid + THREADS * i, count - 1)
        codes.append((_byte(c["x_scale"], first + n), n, tid + THREADS * i < count))
    return loaded, codes


@traced
def _store_piece(loaded, tokens, codes_lds):
    """A piece's token units into LDS in the WMMA operand layout
    (``[tile][group][4 chunks][2 halves][16 tokens][16 B]``, a lane's read
    of 16 consecutive tokens conflict-free), its codes a word each."""
    units, codes = loaded
    for words, at, live in units:
        if live:
            fx.ptr_store(fx.Vector.from_elements(words, fx.Int32), tokens + at)
    for code, at, live in codes:
        if live:
            fx.ptr_store(fx.Int32(code), codes_lds + at)


def _piece_body(c, weights, tokens, codes_lds, lane, task, tile, piece, acc):
    rows, ksplit = c["rows"], c["ksplit"]
    token = fx.min(fx.Int32(tile * ROWS) + lane % ROWS, fx.Int32(rows - 1))
    if const_expr(ksplit == 4):
        group0 = task * lay.RUN_GROUPS + piece * lay.PIECE_GROUPS
    else:
        group0 = fx.Int32(piece * lay.PIECE_GROUPS)
    for j in range_constexpr(lay.PIECE_GROUPS):
        a = _weight_operand(weights, group0 + j, lane)
        words = []
        for k in range_constexpr(4):
            at = (((tile * lay.PIECE_GROUPS + j) * 8 + k * 2 + lane // ROWS) * ROWS
                  + lane % ROWS) * 4  # fmt: skip
            raw = fx.Vector(
                fx.ptr_load(tokens + at, result_type=fx.Vector.make_type(4, fx.Int32))
            )
            words.extend(fx.Int32(raw[i]) for i in range_constexpr(4))
        b = fx.Vector.from_elements(words, fx.Int32)
        x_code = fx.Int32(fx.ptr_load(codes_lds + j * rows + token))
        w_code = c["w_codes"][piece * lay.PIECE_GROUPS + j]
        acc = _mma_fp8(a, b, acc, _splat(w_code), _splat(x_code))
    return acc


@traced
def _piece(c, weights, tokens, codes_lds, lane, task, tile, piece, acc, live):
    """A chain's scaled WMMAs over K piece ``piece``, chained on ``acc``:
    the groups in K order."""
    if live:
        acc = _piece_body(c, weights, tokens, codes_lds, lane, task, tile, piece, acc)
    return acc


@traced
def _wait_weights(c, wave, piece):
    """The weight copies piece ``piece`` reads landed: their waves wait (one
    copy a wave), the CTA barrier follows."""
    if const_expr(c["ksplit"] == 4):
        needed = wave // lay.RUNS == piece
    else:
        needed = wave == piece // 2 + lay.RUNS * (piece % 2)
    if needed:
        rocdl.s_wait_tensorcnt(0)


def _bf16_words(acc):
    return [_pack(acc[2 * i], acc[2 * i + 1]) for i in range_constexpr(ACC // 2)]


@traced
def _store_sum(c, words, tile, task, lane):
    """A task's 16 rows of a token tile, BF16, a lane its token's eight rows:
    the rope rows to ``k_rope``, the q_a / kv_a rows to ``qkv_c`` at device
    scope (a pair of all-ones NaNs as quiet NaNs: as ``UNWRITTEN`` its norm
    wave would wait for it for ever)."""
    rows = c["rows"]
    token = tile * ROWS + lane % ROWS
    row0 = task * ROWS + (lane // ROWS) * ACC
    if token < rows:
        if c["cluster"] == lay.ROPE_CLUSTER:
            _store_words(words, c["k_rope"], (token * lay.ROPE + row0) // 2)
        else:
            safe = [
                (w == UNWRITTEN).select(fx.Int32(lay.QUIET_NAN_PAIR), w) for w in words
            ]
            column = c["cluster"] * lay.GROUP_ROWS + row0
            _store_words(safe, c["qkv_c"], (token * lay.QKV_C + column) // 2, CM_DEV)


@traced
def _project(c, lds, lane, wave):
    """A GEMV CTA: its weight runs copied, its chains, then (ksplit 4) the
    quarters' sums exchanged and the owned task's summed in order."""
    rank, cluster_id, ksplit = c["rank"], c["cluster"], c["ksplit"]
    run, half = wave % lay.RUNS, wave // lay.RUNS
    if const_expr(ksplit == 4):
        task, group = run, rank * lay.RUN_GROUPS
    else:
        task, group = rank, run * lay.RUN_GROUPS
    group = group + half * lay.PIECE_GROUPS
    source = fx.Int64(cluster_id * lay.GROUP_ROWS + task * ROWS) * fx.Int64(
        HIDDEN
    ) + fx.Int64(group) * fx.Int64(lay.GROUP_BYTES)
    gemv_tdm.load_rows(
        c["w_a"] + source,
        lay.GROUP_BYTES // 2,
        lds + (run * lay.RUN_GROUPS + half * lay.PIECE_GROUPS) * GROUP_WORDS,
        lay.PIECE_GROUPS,
        lay.PIECE_GROUPS,
        padded=False,
        cache_policy=gemv_tdm._WEIGHT_CACHE_POLICY,
        whole=True,
        width=lay.GROUP_BYTES // 2,
        uniform=True,
    )
    # the weights' E8M0 codes of the chain's groups: the group's 128-row block
    c["w_codes"] = _weight_codes(c, cluster_id * lay.GROUP_ROWS // 128)
    plan = c["plan"]
    pieces = lay.pieces(ksplit)
    tokens = [lds + lay.token_offset(b, plan.tiles) for b in range(2)]
    codes_lds = [lds + lay.code_offset(b, plan.tiles) for b in range(2)]
    tid = wave * LANES + lane
    _store_piece(_load_piece(c, 0, tid), tokens[0], codes_lds[0])
    ahead = _load_piece(c, 1, tid) if const_expr(pieces > 1) else None
    rounds = -(-plan.chain_count // WAVES)
    chains = []
    for r in range_constexpr(rounds):
        index = wave + WAVES * r
        if const_expr(ksplit == 4):
            chains.append(
                (index % lay.TASKS, index // lay.TASKS, index < plan.chain_count)
            )
        else:
            chains.append((fx.Int32(0), index, index < plan.chain_count))
    accs = [fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(rounds)]
    for p in range_constexpr(pieces):
        _wait_weights(c, wave, p)
        gpu.barrier()
        for r in range_constexpr(rounds):
            task, tile, live = chains[r]
            accs[r] = _piece(c, lds, tokens[p % 2], codes_lds[p % 2], lane, task, tile,
                             p, accs[r], live)  # fmt: skip
        if const_expr(p + 1 < pieces):
            # the buffer piece p - 1 read, past the barrier every wave crossed
            # once done with it
            _store_piece(ahead, tokens[(p + 1) % 2], codes_lds[(p + 1) % 2])
            if const_expr(p + 2 < pieces):
                ahead = _load_piece(c, p + 2, tid)
    results = [(accs[r], *chains[r]) for r in range(rounds)]
    if const_expr(ksplit == 1):
        for acc, _, tile, live in results:
            if live:
                _store_sum(c, _bf16_words(acc), tile, rank, lane)
        return
    # ksplit 4: every chain's quarter sum to its task's owner, slot
    # [owner CTA][quarter][tile]
    tiles = plan.tiles
    cta = c["bid"]
    for acc, chain_task, tile, live in results:
        if live:
            owner = cta - rank + chain_task
            at = (((owner * lay.TASKS + rank) * tiles + tile) * LANES + lane) * ACC
            for h in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(
                        [acc[h * 4 + i] for i in range(4)], fx.Float32
                    ),
                    rsrc(c["partials"]),
                    at + h * 4,
                    cache_modifier=SCOPE_SE,
                )
    rocdl.s_wait_storecnt(0)
    cluster.cluster_barrier()
    _sum_quarters(c, lane, wave)


@traced
def _sum_quarters(c, lane, wave):
    """The owned task's four quarter sums, in quarter order (the split-K
    reduce: ``((p0 + p1) + p2) + p3``), a wave a tile."""
    tiles = c["plan"].tiles
    cta, rank = c["bid"], c["rank"]
    for r in range_constexpr(-(-tiles // WAVES)):
        tile = wave + WAVES * r
        if tile < tiles:
            total = None
            for q in range_constexpr(lay.TASKS):
                at = (((cta * lay.TASKS + q) * tiles + tile) * LANES + lane) * ACC
                halves = [
                    fx.Vector(
                        bo.buffer_load(
                            rsrc(c["partials"]),
                            at + h * 4,
                            vec_width=4,
                            dtype=T.f32,
                            cache_modifier=SCOPE_SE,
                        )
                    )
                    for h in range(2)
                ]
                part = [fx.Float32(halves[i // 4][i % 4]) for i in range(ACC)]
                total = part if total is None else [a + b for a, b in zip(total, part)]
            _store_sum(c, [_pack(total[2 * i], total[2 * i + 1]) for i in range(4)],
                       tile, rank, lane)  # fmt: skip


def _scaled_pair(word, rstd, gain):
    """``(x * rstd) * gain`` of a BF16 pair word's two elements by a gain pair
    word, AITER's two ``v_pk_mul_f32``, each product rounded to f32: the pair
    as f32."""
    out = fx.Vector(
        llvm.inline_asm(
            T.vec(2, T.f32),
            [
                fx.Vector.from_elements(pair, fx.Float32).ir_value()
                for pair in (
                    [bf_lo(word), bf_hi(word)],
                    [rstd, rstd],
                    [bf_lo(gain), bf_hi(gain)],
                )
            ],
            "v_pk_mul_f32 $0, $1, $2\n v_pk_mul_f32 $0, $0, $3",
            "=&v,v,v,v",
            has_side_effects=False,
        )
    )
    return fx.Float32(out[0]), fx.Float32(out[1])


def _square_sum(words):
    """One AITER thread's 16 elements squared and summed in order (each
    square of a BF16 value is exact in f32)."""
    values = []
    for word in words:
        values.extend((bf_lo(word), bf_hi(word)))
    total = values[0] * values[0]
    for value in values[1:]:
        total = total + value * value
    return total


def _scalar_rcp(width: int):
    """``v_s_rcp_f32`` of the width: the kernel (built ``-ffast-math``) takes
    the mean as ``fma(rcp(n), sum, eps)``, its reciprocal the scalar
    transcendental's, not an IEEE ``1 / n`` (they differ at 1536)."""
    return fx.Float32(
        llvm.inline_asm(
            T.f32,
            [fx.Float32(float(width)).ir_value()],
            "v_s_rcp_f32 $0, $1",
            "=s,s",
            has_side_effects=False,
        )
    )


def _magnitude(value):
    return fx.Float32(value).bitcast(fx.Int32) & 0x7FFFFFFF


def _load_runs(qkv_c, at):
    """The lane's three runs of a row from words ``at``, at device scope, as
    one vector (run-major)."""
    words = []
    for r in range_constexpr(lay.Q_RUNS):
        for h in range_constexpr(2):
            loaded = fx.Vector(ld_dev(qkv_c, at[r] + 4 * h, 4))
            words.extend(fx.Int32(loaded[w]) for w in range_constexpr(4))
    return fx.Vector.from_elements(words, fx.Int32)


def _unwritten(words, live):
    left = fx.Boolean(False)
    for r in range_constexpr(lay.Q_RUNS):
        for w in range_constexpr(RUN_WORDS):
            left = left | (live[r] & (fx.Int32(words[r * RUN_WORDS + w]) == UNWRITTEN))
    return left


@traced
def _poll_runs(qkv_c, at, live):
    """The lane's runs once their producers stored every word; a run not
    ``live`` is not waited for."""
    words = _load_runs(qkv_c, at)
    while _unwritten(words, live):
        rocdl.s_sleep(1)
        words = _load_runs(qkv_c, at)
    return words


@traced
def _norm_row(c, job):
    """Job ``job``'s row (a q row below ``rows``, then the kv rows):
    normalized; a kv row stored, a q row quantized for q_b_proj."""
    rows, lane = c["rows"], c["lane"]
    is_q = job < rows
    token = is_q.select(job, job - rows)
    base = token * (lay.QKV_C // 2) + is_q.select(
        fx.Int32(0), fx.Int32(lay.Q_LORA // 2)
    )
    live = [fx.Boolean(True)] + [is_q] * (lay.Q_RUNS - 1)
    # a kv row's runs past the first read the first again, and are not used
    offsets = [lane * RUN_WORDS] + [
        is_q.select(fx.Int32(r * WAVE_WORDS), fx.Int32(0)) + lane * RUN_WORDS
        for r in range(1, lay.Q_RUNS)
    ]
    polled = _poll_runs(c["qkv_c"], [base + o for o in offsets], live)
    runs = [
        [fx.Int32(polled[r * RUN_WORDS + w]) for w in range(RUN_WORDS)]
        for r in range(lay.Q_RUNS)
    ]
    for r in range_constexpr(lay.Q_RUNS):
        if live[r]:
            for h in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements([fx.Int32(UNWRITTEN)] * 4, fx.Int32),
                    rsrc(c["qkv_c"]),
                    base + offsets[r] + 4 * h,
                    cache_modifier=CM_DEV,
                )
                if c["qkv_seen"] != 0:
                    _store_words(runs[r][4 * h : 4 * h + 4], c["qkv_seen"],
                                 base + offsets[r] + 4 * h)  # fmt: skip
    weight = is_q.select(c["q_norm"], c["kv_norm"])
    gains = []
    for r in range_constexpr(lay.Q_RUNS):
        gains.append(_words(weight, offsets[r], 4) + _words(weight, offsets[r] + 4, 4))
    zero = fx.Float32(0.0)
    sums = [_square_sum(runs[0])] + [
        is_q.select(_square_sum(runs[r]), zero) for r in range(1, lay.Q_RUNS)
    ]
    waves = [butterfly(s, (1, 2, 4, 8, 16), architecture=GFX1250) for s in sums]
    total = (waves[0] + waves[1]) + (waves[2] + zero)
    inverse = is_q.select(_scalar_rcp(lay.Q_LORA), _scalar_rcp(lay.KV_LORA))
    eps = is_q.select(fx.Float32(c["q_eps"]), fx.Float32(c["kv_eps"]))
    rstd = hw_rsq(fx.Float32(fmath.fma(inverse, total, eps)))
    normed = [
        [_scaled_pair(word, rstd, gain) for word, gain in zip(runs[r], gains[r])]
        for r in range(lay.Q_RUNS)
    ]
    if is_q:
        _quantize_q(c, normed, token)
    else:
        _store_words(
            [_pack(a, b) for a, b in normed[0][:4]],
            c["kv"],
            token * (lay.KV_LORA // 2) + lane * RUN_WORDS,
        )
        _store_words(
            [_pack(a, b) for a, b in normed[0][4:]],
            c["kv"],
            token * (lay.KV_LORA // 2) + lane * RUN_WORDS + 4,
        )


@traced
def _quantize_q(c, normed, token):
    """A q row's three runs of this lane as FP8 with their groups' fp32
    scales (a group is eight lanes of a run)."""
    lane, rows = c["lane"], c["rows"]
    floor = fx.Float32(lay.AMAX_FLOOR).bitcast(fx.Int32)
    for r in range_constexpr(lay.Q_RUNS):
        magnitude = floor
        for a, b in normed[r]:
            magnitude = fx.max(magnitude, fx.max(_magnitude(a), _magnitude(b)))
        for off in (1, 2, 4):
            magnitude = fx.max(magnitude, xshfl(magnitude, off, GFX1250))
        amax = magnitude.bitcast(fx.Float32)
        scale = amax * fx.Int32(lay.INV_FP8_MAX_BITS).bitcast(fx.Float32)
        if lane % lay.GROUP_LANES == 0:
            group = (
                r * (lay.NORM_WAVE_ELEMENTS // lay.QUANT_GROUP)
                + lane // lay.GROUP_LANES
            )
            bo.buffer_store(scale, rsrc(c["q_scale"]), group * rows + token)
        inverse = hw_rcp(scale)
        values = []
        for a, b in normed[r]:
            values.extend((_saturated(a * inverse), _saturated(b * inverse)))
        packed = _fp8_words(values)
        _store_words(
            packed,
            c["q_fp8"],
            (token * lay.Q_LORA + r * lay.NORM_WAVE_ELEMENTS) // 4 + lane * 4,
        )


def build_mla_draft_front(
    rows: int, ksplit: int, q_eps: float = 1e-5, kv_eps: float = 1e-5
) -> MlaDraftFrontKernel:
    """The draft front at ``rows`` token rows, ``fused_qkv_a_proj``'s sum
    split in ``ksplit`` (production's split_k at the width: 1 or 4). ``x``
    [rows, HIDDEN] E4M3 and ``x_scale`` its [56, rows] E8M0 codes (the input
    norm's transposed per-1x128 layout); ``w_a``/``s_a`` fused_qkv_a_proj's
    preshuffled FP8 [2176, HIDDEN] and [17, 56] E8M0 block scales;
    ``q_norm``/``kv_norm`` the BF16 norm weights. ``qkv_c``: [MAX_ROWS,
    2048] BF16 as i32 words, every one ``UNWRITTEN`` before the first launch
    (each launch leaves them so); ``partials``
    ``mla_draft_front_layout.partial_bytes``. Out: ``q_fp8`` [rows, 1536]
    E4M3 and ``q_scale`` its [12, rows] fp32 scales, ``kv`` [rows, 512] and
    ``k_rope`` [rows, 64] BF16. ``qkv_seen`` (checks only; 0: not written):
    the unnormalized q_a | kv_a rows the norms read, laid out as ``qkv_c``."""
    plan = lay.DraftFrontPlan(rows, ksplit)
    grid = lay.grid(rows)
    gemv_ctas = lay.gemv_ctas()
    norm_ctas = lay.norm_ctas(rows)
    name = f"k3_mla_draft_front_gfx1250_s{rows}_k{ksplit}"
    sources = SOURCES
    lds_count = lay.lds_words(rows)

    @fx.struct
    class Lds:
        words: fx.Array[fx.Int32, lds_count, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        x: Int64,
        x_scale: Int64,
        w_a: Int64,
        s_a: Int64,
        q_norm: Int64,
        kv_norm: Int64,
        qkv_c: Int64,
        q_fp8: Int64,
        q_scale: Int64,
        kv: Int64,
        k_rope: Int64,
        partials: Int64,
        qkv_seen: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % LANES
        wave = uniform(tid // LANES)
        c = {
            "rows": rows,
            "ksplit": ksplit,
            "plan": plan,
            "lane": lane,
            "bid": uniform(bid),
            "rank": uniform(bid % TDM_CLUSTER),
            "cluster": uniform(bid // TDM_CLUSTER),
            "x": x,
            "x_scale": x_scale,
            "w_a": w_a,
            "s_a": s_a,
            "q_norm": q_norm,
            "kv_norm": kv_norm,
            "qkv_c": qkv_c,
            "q_fp8": q_fp8,
            "q_scale": q_scale,
            "kv": kv,
            "k_rope": k_rope,
            "partials": partials,
            "qkv_seen": qkv_seen,
            "q_eps": q_eps,
            "kv_eps": kv_eps,
        }
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        if bid < gemv_ctas:
            _project(c, lds.words.ptr, lane, wave)
        else:
            job = bid - gemv_ctas + norm_ctas * wave
            if job < lay.norm_jobs(rows):
                _norm_row(c, job)

    @flyc.jit
    def launch(
        x: Int64,
        x_scale: Int64,
        w_a: Int64,
        s_a: Int64,
        q_norm: Int64,
        kv_norm: Int64,
        qkv_c: Int64,
        q_fp8: Int64,
        q_scale: Int64,
        kv: Int64,
        k_rope: Int64,
        partials: Int64,
        qkv_seen: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            x,
            x_scale,
            w_a,
            s_a,
            q_norm,
            kv_norm,
            qkv_c,
            q_fp8,
            q_scale,
            kv,
            k_rope,
            partials,
            qkv_seen,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    MLA_DRAFT_FRONT_ABI.check(kernel, launch)
    return MlaDraftFrontKernel(name, launch, (grid,), (THREADS,), plan)
