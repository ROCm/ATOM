# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 MLA absorb in one resident launch: the decode branch of
``MLAAttention.forward_impl`` from q_b_proj's FP8 input to the attention's
``q_out`` and the KV cache, in place of q_b_proj's GEMM, the W_UK batched
GEMM and ``fused_qk_rope_concat_and_cache_mla``.

Each head CTA (``mla_absorb_layout``; a cluster of four owns two heads)
projects six 16-row tasks of q_b_proj, three a head: a wave takes one head's
three tasks over one K quarter, its FP8 weight quarters copied to LDS by the
TDM (online per_1x128 FP8, AITER's gfx1250 preshuffle, E8M0 128x128 block
scales) and its token rows -- the FP8 q with its per-token E8M0 group codes
(``per_1x128_e8m0_quant``'s transposed layout) -- loaded straight to
registers; both dequantized exactly to BF16, f32 WMMA. The four quarters'
sums are added in order in LDS and rounded to BF16: ``q``, swapped through
``q`` (``[rows, heads, 192]``, production's q_b output) at SE scope past one
cluster barrier.

Then each wave absorbs two 16-row tasks of its head's ``W_UK`` (FP8, one
per-tensor scale, loaded to registers at the start) over the head's 128 nope
values as the gluon ``batched_gemm_a16w8`` does it: the FP8 weights widened
to BF16, four K32 WMMAs chained from zero, ``bf16(acc * scale)``. q_out's
nope part is that saturated to E4M3 (``fused_qk_rope_concat_and_cache_mla``:
``v_med3_f32`` to +-448, ``v_cvt_pk_fp8_f32``), its rope part the q rope rows
through the NoPE rotary embedding (``cos`` 1, ``sin`` 0, NeoX pairs: ``x - y *
0`` / ``x + y * 0``) converted without the clamp. The CTAs past the heads'
write a token's KV-cache entry each (``[kv | k_rope]`` the same way) when its
slot is not negative, and zero q_out's padding heads. Scales ``q_scale`` /
``k_scale`` are 1 (the binding checks).
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay
from atom.models.kimi_k3.mono.kernels.stages.gemv_gfx1250 import fp8x8_to_bf16
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import bf_hi, bf_lo, mma_bf16, rsrc, traced, uniform
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = lay.THREADS
WAVES = lay.WAVES
LANES = GFX1250.wave_size
ACC = 8
SCOPE_SE = gemv_tdm.SCOPE_SE
# a K quarter of a 16-row FP8 task in AITER's preshuffle: 12 K32 chunks of
# [2][16 rows][16 B]
QUARTER_STEPS = lay.K_QUARTER // 32
CHUNK_BYTES = 2 * ROWS * 16
QUARTER_BYTES = QUARTER_STEPS * CHUNK_BYTES
QUARTER_WORDS = QUARTER_BYTES // 4
UK_STEPS = lay.NOPE // 32
E8M0_ONE = 127

MLA_ABSORB_ABI = KernelAbi(
    (
        "q_fp8",
        "q_scale",
        "w_qb",
        "s_qb",
        "w_uk",
        "s_uk",
        "kv",
        "k_rope",
        "slots",
        "kv_cache",
        "kv_stride",
        "q_out",
        "q",
        "ql_nope",
    )
)


@dataclass(frozen=True)
class MlaAbsorbKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    plan: lay.AbsorbPlan
    abi: KernelAbi = MLA_ABSORB_ABI
    launch_count: int = 1


def _pack(low, high):
    return (
        fx.Vector.from_elements([low, high], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )


def _byte(ptr, index):
    """Byte ``index`` of a buffer, zero-extended."""
    word = fx.Int32(bo.buffer_load(rsrc(ptr), index >> 2, vec_width=1, dtype=T.i32))
    return (word >> ((index & 3) * 8)) & 0xFF


def _scalar_byte(ptr, index):
    """``_byte`` of a wave-uniform index, through the scalar cache."""
    word = fx.Int32(
        bo.buffer_load(rsrc(ptr), index >> 2, vec_width=1, dtype=T.i32, is_scalar=True)
    )
    return (word >> ((index & 3) * 8)) & 0xFF


def _words(ptr, index, count, cache_modifier=0):
    loaded = fx.Vector(
        bo.buffer_load(
            rsrc(ptr), index, vec_width=count, dtype=T.i32, cache_modifier=cache_modifier
        )
    )
    return [fx.Int32(loaded[i]) for i in range_constexpr(count)]


def _fp8_operand(pieces, exponent):
    """A v16 BF16 WMMA operand from two 8-byte FP8 pieces (two words each)
    times ``exponent``'s power of two (an E8M0 code in every byte)."""
    elements = []
    for piece in pieces:
        converted = fp8x8_to_bf16(fx.Vector.from_elements(piece, fx.Int32), exponent)
        elements.extend(converted[i] for i in range_constexpr(8))
    return fx.Vector.from_elements(elements, fx.BFloat16)


def _bf16_operand(pieces):
    """A v16 BF16 WMMA operand from two 16-byte BF16 pieces."""
    words = [w for piece in pieces for w in piece]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.BFloat16)


def _splat(code):
    return fx.Int32(code) * 0x01010101


def _saturated(value):
    """``value`` clamped to E4M3's +-448 by ``v_med3_f32``, as
    ``fp32_to_fp8_scaled_x2`` does (its NaN handling included). The
    intrinsic, not inline asm: LLVM's hazard recognizer does not look into
    inline asm, and a VALU write into a WMMA's source registers right behind
    it corrupts the WMMA's operands."""
    return fx.Float32(
        rocdl_dialect.fmed3(
            T.f32,
            fx.Float32(value).ir_value(),
            fx.Float32(-lay.FP8_MAX).ir_value(),
            fx.Float32(lay.FP8_MAX).ir_value(),
        )
    )


def _fp8_words(values):
    """Eight f32 as two words of E4M3, ``v_cvt_pk_fp8_f32`` (RNE, no clamp)."""
    words = []
    for quad in range_constexpr(len(values) // 4):
        a, b, c, d = values[4 * quad : 4 * quad + 4]
        low = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
        words.append(fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, low, True)))
    return words


def _bf16_values(words):
    values = []
    for word in words:
        values.extend((bf_lo(word), bf_hi(word)))
    return values


def _nope_fp8(words):
    """q_out's / the cache's nope bytes of BF16 pair words: saturated."""
    return _fp8_words([_saturated(v) for v in _bf16_values(words)])


def _rope_fp8(words, partner, low):
    """The NoPE rotary embedding of eight elements (``low``: the first half,
    ``x * 1 - y * 0``; else ``x * 1 + y * 0``) and their E4M3 bytes, no
    clamp."""
    xs, ys = _bf16_values(words), _bf16_values(partner)
    zero = fx.Float32(0.0)
    rotated = [low.select(x - y * zero, x + y * zero) for x, y in zip(xs, ys)]
    return _fp8_words(rotated)


def _store_words(words, ptr, index, cache_modifier=0):
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(ptr),
        index,
        cache_modifier=cache_modifier,
    )


def _slot(slots, token):
    """``slot_mapping[token]`` (int64): (its low word, its high word)."""
    lo, hi = _words(slots, token * 2, 2)
    return lo, hi


@traced
def _project(c, lds, lane, wave):
    """Phase 1: this wave's K quarter of its head's three q_b tasks, summed
    over the quarters in LDS; then this CTA's q rows to ``q`` at SE scope."""
    rows, heads, tiles = c["rows"], c["heads"], c["tiles"]
    rank, head0 = c["rank"], c["head0"]
    job, quarter = wave // lay.K_QUARTERS, wave % lay.K_QUARTERS
    head = head0 + job
    task_rows = [
        head * lay.HEAD + (lay.NOPE_TASKS * rank + i) * ROWS
        for i in range(lay.NOPE_TASKS)
    ] + [head * lay.HEAD + lay.NOPE + rank * ROWS]
    weights = [
        lds + ((job * lay.HEAD_TASKS + t) * lay.K_QUARTERS + quarter) * QUARTER_WORDS
        for t in range(lay.HEAD_TASKS)
    ]
    for t in range_constexpr(lay.HEAD_TASKS):
        gemv_tdm.load_rows(
            c["w_qb"]
            + fx.Int64(task_rows[t]) * fx.Int64(lay.Q_LORA)
            + fx.Int64(quarter * QUARTER_BYTES),
            QUARTER_BYTES // 2,
            weights[t],
            1,
            1,
            padded=False,
            cache_policy=gemv_tdm._WEIGHT_CACHE_POLICY,
            whole=True,
            width=QUARTER_BYTES // 2,
            uniform=True,
        )
    # the weights' E8M0 codes: a task's 128-row block, the quarter's groups
    w_codes = [
        [
            _splat(
                _scalar_byte(
                    c["s_qb"],
                    task_rows[t] // 128 * lay.GROUPS + quarter * lay.QUARTER_GROUPS + g,
                )
            )
            for g in range(lay.QUARTER_GROUPS)
        ]
        for t in range(lay.HEAD_TASKS)
    ]
    lane_group = lane // ROWS
    pieces = []
    x_codes = []
    for tile in range_constexpr(tiles):
        token = fx.min(fx.Int32(tile * ROWS) + lane % ROWS, fx.Int32(rows - 1))
        row = token * (lay.Q_LORA // 4) + (quarter * lay.K_QUARTER + lane_group * 8) // 4
        pieces.append(
            [
                [_words(c["q_fp8"], row + (32 * s + 16 * half) // 4, 2) for half in range(2)]
                for s in range(QUARTER_STEPS)
            ]
        )
        x_codes.append(
            [
                _splat(
                    _byte(
                        c["q_scale"],
                        (quarter * lay.QUARTER_GROUPS + g) * rows + token,
                    )
                )
                for g in range(lay.QUARTER_GROUPS)
            ]
        )
    rocdl.s_wait_tensorcnt(0)
    accs = [
        [fx.Vector.filled(ACC, 0.0, fx.Float32) for _ in range(tiles)]
        for _ in range(lay.HEAD_TASKS)
    ]
    weight_row = lane % ROWS
    for s in range_constexpr(QUARTER_STEPS):
        group = s // 4
        a = []
        for t in range_constexpr(lay.HEAD_TASKS):
            chunk = []
            for half in range_constexpr(2):
                byte = ((s * 2 + half) * ROWS + weight_row) * 16 + lane_group * 8
                raw = fx.Vector(
                    fx.ptr_load(
                        weights[t] + byte // 4,
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                )
                chunk.append([fx.Int32(raw[0]), fx.Int32(raw[1])])
            a.append(_fp8_operand(chunk, w_codes[t][group]))
        for tile in range_constexpr(tiles):
            b = _fp8_operand(pieces[tile][s], x_codes[tile][group])
            for t in range_constexpr(lay.HEAD_TASKS):
                accs[t][tile] = mma_bf16(a[t], b, accs[t][tile], GFX1250)
    sums = lds + lay.HEADS_PER_CLUSTER * lay.HEAD_TASKS * lay.K_QUARTERS * QUARTER_WORDS
    for t in range_constexpr(lay.HEAD_TASKS):
        for tile in range_constexpr(tiles):
            slot = ((job * lay.HEAD_TASKS + t) * tiles + tile) * lay.K_QUARTERS + quarter
            fx.ptr_store(
                accs[t][tile].bitcast(fx.Int32),
                sums + (slot * LANES + lane) * ACC,
            )
    gpu.barrier()
    tid = wave * LANES + lane
    items = lay.HEADS_PER_CLUSTER * lay.HEAD_TASKS * tiles * LANES
    for round_index in range_constexpr(-(-items // THREADS)):
        item = tid + THREADS * round_index
        sum_lane = item % LANES
        tile = item // LANES % tiles
        task = item // (LANES * tiles) % lay.HEAD_TASKS
        pair = item // (LANES * tiles * lay.HEAD_TASKS)
        token = tile * ROWS + sum_lane % ROWS
        if (item < items) & (token < rows):
            base = sums + (((pair * lay.HEAD_TASKS + task) * tiles + tile) * lay.K_QUARTERS * LANES + sum_lane) * ACC
            total = None
            for q in range_constexpr(lay.K_QUARTERS):
                part = fx.Vector(
                    fx.ptr_load(
                        base + q * LANES * ACC,
                        result_type=fx.Vector.make_type(ACC, fx.Int32),
                    )
                ).bitcast(fx.Float32)
                total = (
                    part
                    if total is None
                    else fx.Vector.from_elements(
                        [total[i] + part[i] for i in range_constexpr(ACC)], fx.Float32
                    )
                )
            in_head = (task < lay.NOPE_TASKS).select(
                (lay.NOPE_TASKS * rank + task) * ROWS, lay.NOPE + rank * ROWS
            ) + sum_lane // ROWS * ACC
            words = [
                _pack(total[2 * i], total[2 * i + 1]) for i in range_constexpr(ACC // 2)
            ]
            _store_words(
                words,
                c["q"],
                ((token * heads + head0 + pair) * lay.HEAD + in_head) // 2,
                SCOPE_SE,
            )
    rocdl.s_wait_storecnt(0)
    cluster.cluster_barrier()


@traced
def _absorb(c, uk, lane, wave):
    """Phase 2: this wave's two W_UK tasks of its head over the head's q nope
    rows (from ``q``), q_out's nope bytes (and the ``ql_nope`` tap)."""
    rows, heads, tiles = c["rows"], c["heads"], c["tiles"]
    rank, head0 = c["rank"], c["head0"]
    job, quarter = wave // lay.K_QUARTERS, wave % lay.K_QUARTERS
    head = head0 + job
    lane_group = lane // ROWS
    scale = fx.Int32(
        bo.buffer_load(rsrc(c["s_uk"]), 0, vec_width=1, dtype=T.i32, is_scalar=True)
    ).bitcast(fx.Float32)
    one = _splat(fx.Int32(E8M0_ONE))
    weights = [
        [_fp8_operand(uk[u][s], one) for s in range(UK_STEPS)] for u in range(lay.UK_TASKS)
    ]
    for tile in range_constexpr(tiles):
        token = fx.min(fx.Int32(tile * ROWS) + lane % ROWS, fx.Int32(rows - 1))
        row = (token * heads + head) * lay.HEAD // 2
        b = [
            _bf16_operand(
                [
                    _words(c["q"], row + (32 * s + 16 * half + lane_group * 8) // 2, 4, SCOPE_SE)
                    for half in range(2)
                ]
            )
            for s in range(UK_STEPS)
        ]
        out_token = tile * ROWS + lane % ROWS
        live = out_token < rows
        high = fx.Int32(0)
        if const_expr(not c["compute_all"]):
            _, high = _slot(c["slots"], fx.min(out_token, fx.Int32(rows - 1)))
        zero = high < 0
        for u in range_constexpr(lay.UK_TASKS):
            acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
            for s in range_constexpr(UK_STEPS):
                acc = mma_bf16(weights[u][s], b[s], acc, GFX1250)
            n0 = rank * lay.UK_ROWS + quarter * lay.UK_WAVE_ROWS + u * ROWS + lane_group * ACC
            words = [
                _pack(acc[2 * i] * scale, acc[2 * i + 1] * scale)
                for i in range_constexpr(ACC // 2)
            ]
            if live & (c["ql_nope"] != 0):
                _store_words(
                    words, c["ql_nope"], ((out_token * heads + head) * lay.KV_LORA + n0) // 2
                )
            nope = _nope_fp8(words)
            nope = [zero.select(fx.Int32(0), w) for w in nope]
            if live:
                _store_words(
                    nope,
                    c["q_out"],
                    (out_token * c["q_out_row"] + head * lay.ENTRY + n0) // 4,
                )


@traced
def _rope(c, tid):
    """Phase 3: q_out's rope bytes of this CTA's tokens (every fourth), both
    heads, a thread an 8-element chunk."""
    rows, heads = c["rows"], c["heads"]
    rank, head0 = c["rank"], c["head0"]
    chunk = tid % lay.VEC
    pair = tid // lay.VEC % lay.HEADS_PER_CLUSTER
    token = rank + TDM_CLUSTER * (tid // (lay.VEC * lay.HEADS_PER_CLUSTER))
    if token < rows:
        head = head0 + pair
        row = ((token * heads + head) * lay.HEAD + lay.NOPE) // 2
        x = _words(c["q"], row + chunk * 4, 4, SCOPE_SE)
        y = _words(c["q"], row + (chunk ^ 4) * 4, 4, SCOPE_SE)
        out = _rope_fp8(x, y, chunk < 4)
        if const_expr(not c["compute_all"]):
            _, high = _slot(c["slots"], token)
            out = [(high < 0).select(fx.Int32(0), w) for w in out]
        _store_words(
            out,
            c["q_out"],
            (token * c["q_out_row"] + head * lay.ENTRY + lay.KV_LORA) // 4 + chunk * 2,
        )


@traced
def _uk_weights(c, lane, wave):
    """This wave's two W_UK tasks as FP8 operand pieces, loaded at the
    start: ``[task][step][half]`` two words."""
    rank, head0 = c["rank"], c["head0"]
    job, quarter = wave // lay.K_QUARTERS, wave % lay.K_QUARTERS
    head = head0 + job
    lane_group = lane // ROWS
    out = []
    for u in range_constexpr(lay.UK_TASKS):
        n = rank * lay.UK_ROWS + quarter * lay.UK_WAVE_ROWS + u * ROWS + lane % ROWS
        row = (head * lay.KV_LORA + n) * (lay.NOPE // 4)
        out.append(
            [
                [
                    _words(c["w_uk"], row + (32 * s + 16 * half + lane_group * 8) // 4, 2)
                    for half in range(2)
                ]
                for s in range(UK_STEPS)
            ]
        )
    return out


@traced
def _cache(c, token, tid):
    """A KV-cache CTA: token ``token``'s ``[kv | k_rope]`` entry at its slot
    (none when negative), and q_out's padding heads zero."""
    rows = c["rows"]
    if token < rows:
        lo, hi = _slot(c["slots"], token)
        if hi >= 0:
            slot = (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))
            entry = c["kv_cache"] + slot * c["kv_stride"]
            if tid < lay.KV_LORA // lay.VEC:
                words = _words(c["kv"], token * (lay.KV_LORA // 2) + tid * 4, 4)
                _store_words(_nope_fp8(words), entry, tid * 2)
            rope = tid - lay.KV_LORA // lay.VEC
            if (rope >= 0) & (rope < lay.ROPE // lay.VEC):
                base = token * (lay.ROPE // 2)
                x = _words(c["k_rope"], base + rope * 4, 4)
                y = _words(c["k_rope"], base + (rope ^ 4) * 4, 4)
                _store_words(_rope_fp8(x, y, rope < 4), entry, lay.KV_LORA // 4 + rope * 2)
        pad_words = (c["out_heads"] - c["heads"]) * lay.ENTRY // 16
        if const_expr(pad_words > 0):
            first = (token * c["q_out_row"] + c["heads"] * lay.ENTRY) // 4
            for round_index in range_constexpr(-(-pad_words // THREADS)):
                item = tid + THREADS * round_index
                if item < pad_words:
                    _store_words([fx.Int32(0)] * 4, c["q_out"], first + item * 4)


def build_mla_absorb(
    rows: int, heads: int, out_heads: int | None = None, compute_all: bool = True
) -> MlaAbsorbKernel:
    """The absorb at ``rows`` token rows for ``heads`` query heads
    (``mla_absorb_layout.AbsorbPlan``). ``q_fp8`` [rows, 1536] E4M3 and
    ``q_scale`` its [12, rows] E8M0 codes; ``w_qb``/``s_qb`` q_b_proj's
    preshuffled FP8 [heads * 192, 1536] and [heads * 192 / 128, 12] E8M0
    scales; ``w_uk`` W_UK [heads, 512, 128] E4M3 and ``s_uk`` its f32 scale;
    ``kv`` [rows, 512] / ``k_rope`` [rows, 64] BF16; ``slots`` [rows] int64;
    ``kv_cache`` the E4M3 cache and ``kv_stride`` its bytes a slot. Out:
    ``q_out`` [rows, out_heads, 576] E4M3, ``q`` [rows, heads, 192] BF16 (the
    exchange; production's q), ``ql_nope`` [rows, heads, 512] BF16 (checks
    only; 0: not written)."""
    plan = lay.AbsorbPlan(rows, heads, out_heads or heads, compute_all)
    grid = lay.grid(heads)
    head_ctas = lay.head_ctas(heads)
    tiles = plan.tiles
    lds_count = lay.lds_words(rows)
    name = (
        f"k3_mla_absorb_gfx1250_s{rows}_h{heads}"
        + (f"_o{plan.out_heads}" if plan.out_heads != heads else "")
        + ("" if compute_all else "_slot")
    )
    sources = SOURCES

    @fx.struct
    class Lds:
        words: fx.Array[fx.Int32, lds_count, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        q_fp8: Int64,
        q_scale: Int64,
        w_qb: Int64,
        s_qb: Int64,
        w_uk: Int64,
        s_uk: Int64,
        kv: Int64,
        k_rope: Int64,
        slots: Int64,
        kv_cache: Int64,
        kv_stride: Int64,
        q_out: Int64,
        q: Int64,
        ql_nope: Int64,
    ):
        _ = sources
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % LANES
        wave = uniform(tid // LANES)
        c = {
            "rows": rows,
            "heads": heads,
            "out_heads": plan.out_heads,
            "q_out_row": plan.q_out_row,
            "tiles": tiles,
            "compute_all": compute_all,
            "rank": uniform(bid % TDM_CLUSTER),
            "head0": uniform(bid // TDM_CLUSTER) * lay.HEADS_PER_CLUSTER,
            "q_fp8": q_fp8,
            "q_scale": q_scale,
            "w_qb": w_qb,
            "s_qb": s_qb,
            "w_uk": w_uk,
            "s_uk": s_uk,
            "kv": kv,
            "k_rope": k_rope,
            "slots": slots,
            "kv_cache": kv_cache,
            "kv_stride": kv_stride,
            "q_out": q_out,
            "q": q,
            "ql_nope": ql_nope,
        }
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        if bid < head_ctas:
            uk = _uk_weights(c, lane, wave)
            _project(c, lds.words.ptr, lane, wave)
            _absorb(c, uk, lane, wave)
            _rope(c, tid)
        else:
            _cache(c, bid - head_ctas, tid)

    @flyc.jit
    def launch(
        q_fp8: Int64,
        q_scale: Int64,
        w_qb: Int64,
        s_qb: Int64,
        w_uk: Int64,
        s_uk: Int64,
        kv: Int64,
        k_rope: Int64,
        slots: Int64,
        kv_cache: Int64,
        kv_stride: Int64,
        q_out: Int64,
        q: Int64,
        ql_nope: Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = sources
        kernel(
            q_fp8,
            q_scale,
            w_qb,
            s_qb,
            w_uk,
            s_uk,
            kv,
            k_rope,
            slots,
            kv_cache,
            kv_stride,
            q_out,
            q,
            ql_nope,
            value_attrs=gemv_tdm.CLUSTER_LAUNCH["value_attrs"],
        ).launch(
            grid=(grid,),
            block=(THREADS,),
            stream=stream,
            cluster=gemv_tdm.CLUSTER_LAUNCH["cluster"],
        )

    MLA_ABSORB_ABI.check(kernel, launch)
    return MlaAbsorbKernel(name, launch, (grid,), (THREADS,), plan)
