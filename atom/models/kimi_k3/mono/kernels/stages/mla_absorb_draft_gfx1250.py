# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 MLA absorb of the DSpark draft's layers in one resident launch:
``mla_absorb_gfx1250`` for a layer whose q_b_proj takes fp32 activation
scales and whose rotary embedding is a real one. From the draft front's FP8 q
(per-1x128 fp32 scales, transposed) to the attention's ``q_out`` and the KV
cache, in place of q_b_proj's GEMM (and its split-K reduce), the E8M0 weight
scales' fp32 copy, the W_UK batched GEMM and
``fused_qk_rope_concat_and_cache_mla``.

The geometry is the absorb's (``mla_absorb_layout``: a cluster of four CTAs
owns two heads, a CTA three 16-row q_b tasks a head). q_b_proj runs as
production's gluon ``gemm_a8w8_blockscale_preshuffle`` does it with fp32
scales: per 128-wide K group an FP8 WMMA (16x16x128) from zero, its result
times ``x_scale * w_scale`` added with one ``fma`` -- ``w_scale`` the E8M0
code's power of two, exactly -- the groups of a split in order, the splits
(``qb_split_groups`` groups each, production's NUM_KSPLIT at the width)
summed as the split-K reduce's ``tl.sum`` does, pairwise over a power-of-two
count padded with zeros; a chain a task a token tile, over its K in LDS (the
weights copied by the TDM as the absorb does, the token rows and their
scales staged by every thread in the WMMA operand layout). Then the
absorb's exchange, W_UK and nope bytes unchanged.

The rope part rotates GPT-J pairs (``x[2i], x[2i + 1]``) by the BF16
``cos`` / ``sin`` row of the token's position (clamped into the cache, as
the kernel's per-entry layout does): ``x c - y s`` and ``y c + x s`` (each
product of two BF16 values exact in f32, so one rounding whatever the
contraction), converted to E4M3 without the clamp -- q_out's and the cached
k_rope's alike.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int64, T

from atom.models.kimi_k3.mono.kernels.stages import gemv_tdm_gfx1250 as gemv_tdm
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.models.kimi_k3.mono.kernels.stages.mla_absorb_gfx1250 import (
    QUARTER_BYTES,
    QUARTER_WORDS,
    SCOPE_SE,
    _absorb,
    _fp8_words,
    _nope_fp8,
    _pack,
    _scalar_byte,
    _slot,
    _stage_nope,
    _store_words,
    _uk_weights,
    _words,
)
from atom.models.kimi_k3.mono.sources import SOURCES
from atom.mono.device.ops import bf_hi, bf_lo, rsrc, traced, uniform
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.abi import KernelAbi

THREADS = lay.THREADS
WAVES = lay.WAVES
LANES = GFX1250.wave_size
ACC = 8
TASKS = lay.HEADS_PER_CLUSTER * lay.HEAD_TASKS
GROUP_WORDS = ROWS * lay.QUANT_GROUP // 4
UNITS_PER_ROW = lay.Q_LORA // 16
HALF = lay.ROPE // 2

MLA_ABSORB_DRAFT_ABI = KernelAbi(
    (
        "q_fp8",
        "q_scale",
        "w_qb",
        "s_qb",
        "w_uk",
        "s_uk",
        "kv",
        "k_rope",
        "k_rope_stride",
        "slots",
        "positions",
        "cos",
        "sin",
        "kv_cache",
        "kv_stride",
        "q_out",
        "q",
        "ql_nope",
    )
)


@dataclass(frozen=True)
class MlaAbsorbDraftKernel:
    name: str
    launcher: object
    grid: tuple[int]
    block: tuple[int]
    plan: lay.AbsorbPlan
    qb_split_groups: int
    abi: KernelAbi = MLA_ABSORB_DRAFT_ABI
    launch_count: int = 1


def token_words(rows: int) -> int:
    """The staged FP8 q rows: every tile's 16 rows of every K group in the
    WMMA operand layout (``[tile][group][4 chunks][2 halves][16 rows][16 B]``)."""
    return -(-rows // ROWS) * lay.GROUPS * GROUP_WORDS


def weight_words() -> int:
    return TASKS * lay.K_QUARTERS * QUARTER_WORDS


def lds_words(rows: int) -> int:
    """The q_b weights (the absorb's), the staged q rows, their fp32 scales
    (``[group][rows]``)."""
    return weight_words() + token_words(rows) + lay.GROUPS * rows


def _word(ptr, index):
    return fx.Int32(bo.buffer_load(rsrc(ptr), index, vec_width=1, dtype=T.i32))


def _stage_tokens(c, lds, tid):
    """Every thread's share of the FP8 q rows (16-byte units, consecutive
    threads consecutive bytes) and their fp32 scales into LDS, in batches of
    rounds whose loads go out together."""
    rows = c["rows"]
    tokens = lds + weight_words()
    scales = tokens + token_words(rows)
    units = rows * UNITS_PER_ROW
    rounds = -(-units // THREADS)
    batch = 12
    for first in range_constexpr(0, rounds, batch):
        loaded = []
        for i in range_constexpr(first, min(first + batch, rounds)):
            u = fx.min(tid + THREADS * i, units - 1)
            token, rem = u // UNITS_PER_ROW, u % UNITS_PER_ROW
            words = _words(c["q_fp8"], (token * lay.Q_LORA + rem * 16) // 4, 4)
            unit = ((token // ROWS * lay.GROUPS + rem // 8) * 8 + rem % 8) * ROWS
            loaded.append((words, (unit + token % ROWS) * 4, tid + THREADS * i < units))
        _store_units(loaded, tokens)
    count = lay.GROUPS * rows
    loaded = []
    for i in range_constexpr(-(-count // THREADS)):
        n = fx.min(tid + THREADS * i, count - 1)
        loaded.append(([_word(c["q_scale"], n)], n, tid + THREADS * i < count))
    _store_units(loaded, scales)


@traced
def _store_units(loaded, base):
    for words, at, live in loaded:
        if live:
            if const_expr(len(words) == 1):
                fx.ptr_store(fx.Int32(words[0]), base + at)
            else:
                fx.ptr_store(fx.Vector.from_elements(words, fx.Int32), base + at)


def _gluon_operand(base, lane):
    """A v16 FP8 WMMA operand of one 128-wide K group from LDS in the
    preshuffle layout (``[4 chunks][2 halves][16 rows][16 B]``), its K in
    the registers as the gluon kernel's dot operand (``k_width`` 8) holds
    it: eight-byte runs, a lane half ``h`` K ``16 j + 8 h .. + 8`` in run
    ``j``. (The hardware sums a WMMA's products by register slot, so the
    run order is the sum order.)"""
    words = []
    for j in range_constexpr(8):
        byte = (((j // 2) * 2 + j % 2) * ROWS + lane % ROWS) * 16 + (lane // ROWS) * 8
        raw = fx.Vector(
            fx.ptr_load(base + byte // 4, result_type=fx.Vector.make_type(2, fx.Int32))
        )
        words.extend(fx.Int32(raw[i]) for i in range_constexpr(2))
    return fx.Vector.from_elements(words, fx.Int32)


def _task_operand(weights, task, group, lane):
    """Task ``task``'s 16 q_b rows of K group ``group`` from the absorb's
    weight copies (its four quarters one after another)."""
    return _gluon_operand(
        weights + task * lay.K_QUARTERS * QUARTER_WORDS + group * GROUP_WORDS, lane
    )


def _token_operand(tokens, tile, group, lane):
    return _gluon_operand(tokens + (tile * lay.GROUPS + group) * GROUP_WORDS, lane)


def _wmma_fp8(a, b):
    """An unscaled FP8 WMMA (16x16x128) from zero."""
    zero = fx.Vector.filled(ACC, 0.0, fx.Float32)
    return fx.Vector(
        rocdl.wmma_f32_16x16x128_fp8_fp8(
            T.vec(ACC, T.f32), a.ir_value(), b.ir_value(), zero.ir_value()
        )
    )


def _w_scale(code):
    """An E8M0 code's power of two as f32 (the binding refuses codes 0 and
    255, the denormal and the NaN)."""
    return (fx.Int32(code) << 23).bitcast(fx.Float32)


def _tree(values):
    """``tl.sum`` over a power-of-two axis: adjacent pairs, level by level."""
    while len(values) > 1:
        values = [
            [a + b for a, b in zip(values[2 * i], values[2 * i + 1])]
            for i in range(len(values) // 2)
        ]
    return values[0]


def _chain_body(c, lds, lane, task, tile):
    rows, split = c["rows"], c["qb_split_groups"]
    weights = lds
    tokens = lds + weight_words()
    scales = tokens + token_words(rows)
    token = fx.min(fx.Int32(tile * ROWS) + lane % ROWS, fx.Int32(rows - 1))
    splits = []
    for s in range_constexpr(-(-lay.GROUPS // split)):
        acc = [fx.Float32(0.0)] * ACC
        for g in range_constexpr(s * split, min((s + 1) * split, lay.GROUPS)):
            res = _wmma_fp8(
                _task_operand(weights, task, g, lane),
                _token_operand(tokens, tile, g, lane),
            )
            x_scale = fx.Int32(fx.ptr_load(scales + g * rows + token)).bitcast(
                fx.Float32
            )
            scale = x_scale * c["w_scales"][g]
            acc = [
                fx.Float32(fmath.fma(scale, fx.Float32(res[i]), acc[i]))
                for i in range(ACC)
            ]
        splits.append(acc)
    if const_expr(len(splits) > 1):
        pad = 1 << (len(splits) - 1).bit_length()
        splits = splits + [[fx.Float32(0.0)] * ACC] * (pad - len(splits))
    total = _tree(splits)
    return fx.Vector.from_elements(total, fx.Float32)


@traced
def _chain(c, lds, lane, task, tile, live):
    acc = fx.Vector.filled(ACC, 0.0, fx.Float32)
    if live:
        acc = _chain_body(c, lds, lane, task, tile)
    return acc


@traced
def _store_q(c, acc, task, tile, lane, live):
    """A chain's 16 q rows of its tile, BF16, to ``q`` at SE scope (the
    absorb's exchange)."""
    rows, heads, rank, head0 = c["rows"], c["heads"], c["rank"], c["head0"]
    token = tile * ROWS + lane % ROWS
    pair, t = task // lay.HEAD_TASKS, task % lay.HEAD_TASKS
    in_head = (t < lay.NOPE_TASKS).select(
        (lay.NOPE_TASKS * rank + t) * ROWS, lay.NOPE + rank * ROWS
    ) + lane // ROWS * ACC
    if live & (token < rows):
        words = [_pack(acc[2 * i], acc[2 * i + 1]) for i in range_constexpr(ACC // 2)]
        _store_words(
            words, c["q"], ((token * heads + head0 + pair) * lay.HEAD + in_head) // 2,
            SCOPE_SE,
        )  # fmt: skip


@traced
def _project(c, lds, lane, wave):
    """Phase 1: the q_b weights copied (the absorb's copies), the q rows and
    scales staged, a chain a (task, tile); then the CTA's q rows to ``q`` and
    the cluster barrier. Returns the W_UK weights, loaded behind the
    staging."""
    rank, head0 = c["rank"], c["head0"]
    job, quarter = wave // lay.K_QUARTERS, wave % lay.K_QUARTERS
    head = head0 + job
    task_rows = [
        head * lay.HEAD + (lay.NOPE_TASKS * rank + i) * ROWS
        for i in range(lay.NOPE_TASKS)
    ] + [head * lay.HEAD + lay.NOPE + rank * ROWS]
    for t in range_constexpr(lay.HEAD_TASKS):
        gemv_tdm.load_rows(
            c["w_qb"]
            + fx.Int64(task_rows[t]) * fx.Int64(lay.Q_LORA)
            + fx.Int64(quarter * QUARTER_BYTES),
            QUARTER_BYTES // 2,
            lds
            + ((job * lay.HEAD_TASKS + t) * lay.K_QUARTERS + quarter) * QUARTER_WORDS,
            1,
            1,
            padded=False,
            cache_policy=gemv_tdm._WEIGHT_CACHE_POLICY,
            whole=True,
            width=QUARTER_BYTES // 2,
            uniform=True,
        )
    tid = wave * LANES + lane
    _stage_tokens(c, lds, tid)
    uk = _uk_weights(c, lane, wave)
    rocdl.s_wait_tensorcnt(0)
    gpu.barrier()
    plan = c["plan"]
    chains = TASKS * plan.tiles
    for r in range_constexpr(-(-chains // WAVES)):
        index = wave + WAVES * r
        live = index < chains
        task, tile = index % TASKS, index // TASKS
        # the task's 128-row block of the weight scales
        pair, t = task // lay.HEAD_TASKS, task % lay.HEAD_TASKS
        row = (head0 + pair) * lay.HEAD + (t < lay.NOPE_TASKS).select(
            (lay.NOPE_TASKS * rank + t) * ROWS, lay.NOPE + rank * ROWS
        )
        block = uniform(row // 128)
        c["w_scales"] = [
            _w_scale(_scalar_byte(c["s_qb"], block * lay.GROUPS + g))
            for g in range(lay.GROUPS)
        ]
        acc = _chain(c, lds, lane, task, tile, live)
        _store_q(c, acc, task, tile, lane, live)
    rocdl.s_wait_storecnt(0)
    cluster.cluster_barrier()
    return uk


def _rotate(words, cos, sin):
    """GPT-J pairs of eight BF16 elements rotated by four cos / sin values:
    ``x c - y s``, ``y c + x s``, as f32."""
    xs = []
    for w in words:
        xs.extend((bf_lo(w), bf_hi(w)))
    cs = []
    for w in cos:
        cs.extend((bf_lo(w), bf_hi(w)))
    ss = []
    for w in sin:
        ss.extend((bf_lo(w), bf_hi(w)))
    out = []
    for i in range_constexpr(4):
        x, y, cv, sv = xs[2 * i], xs[2 * i + 1], cs[i], ss[i]
        out.extend((x * cv - y * sv, y * cv + x * sv))
    return out


def _position(c, token):
    """The token's position clamped into the cos / sin cache."""
    pos = _word(c["positions"], token * 2)
    return fx.min(fx.max(pos, fx.Int32(0)), fx.Int32(c["max_pos"] - 1))


def _cos_sin(c, pos, chunk):
    """The four cos / sin values of an eight-element chunk's pairs (two
    words each)."""
    at = (pos * HALF + chunk * 4) // 2
    return _words(c["cos"], at, 2), _words(c["sin"], at, 2)


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
        cos, sin = _cos_sin(c, _position(c, token), chunk)
        out = _fp8_words(_rotate(x, cos, sin))
        if const_expr(not c["compute_all"]):
            _, high = _slot(c["slots"], token)
            out = [(high < 0).select(fx.Int32(0), w) for w in out]
        _store_words(
            out,
            c["q_out"],
            (token * c["q_out_row"] + head * lay.ENTRY + lay.KV_LORA) // 4 + chunk * 2,
        )


@traced
def _cache(c, token, tid):
    """A KV-cache CTA: token ``token``'s ``[kv | k_rope]`` entry at its slot
    (none when negative), k_rope rotated; q_out's padding heads zero."""
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
                x = _words(c["k_rope"], token * c["k_rope_stride"] // 2 + rope * 4, 4)
                cos, sin = _cos_sin(c, _position(c, token), rope)
                _store_words(_fp8_words(_rotate(x, cos, sin)), entry,
                             lay.KV_LORA // 4 + rope * 2)  # fmt: skip
        pad_words = (c["out_heads"] - c["heads"]) * lay.ENTRY // 16
        if const_expr(pad_words > 0):
            first = (token * c["q_out_row"] + c["heads"] * lay.ENTRY) // 4
            for round_index in range_constexpr(-(-pad_words // THREADS)):
                item = tid + THREADS * round_index
                if item < pad_words:
                    _store_words([fx.Int32(0)] * 4, c["q_out"], first + item * 4)


def build_mla_absorb_draft(
    rows: int,
    heads: int,
    out_heads: int | None,
    compute_all: bool,
    qb_split_groups: int,
    max_pos: int,
) -> MlaAbsorbDraftKernel:
    """The draft absorb at ``rows`` token rows for ``heads`` query heads
    (``mla_absorb_layout.AbsorbPlan``), q_b_proj's K summed in splits of
    ``qb_split_groups`` 128-wide groups (production's SPLITK_BLOCK_SIZE /
    128 at the width; 12: one split) and the rope cache ``max_pos`` rows.
    ``q_fp8`` [rows, 1536] E4M3 and ``q_scale`` its [12, rows] fp32 scales
    (the draft front's, transposed); ``w_qb``/``s_qb`` q_b_proj's
    preshuffled FP8 and [heads * 192 / 128, 12] E8M0 scales; ``w_uk`` /
    ``s_uk`` W_UK [heads, 512, 128] E4M3 and its f32 scale; ``kv`` [rows,
    512] / ``k_rope`` [rows, 64] BF16 (rows ``k_rope_stride`` elements apart,
    even: production's rope rows are a view of fused_qkv_a_proj's output);
    ``slots`` / ``positions`` [rows]
    int64; ``cos`` / ``sin`` [max_pos, 32] BF16; ``kv_cache`` the E4M3
    cache and ``kv_stride`` its bytes a slot. Out as ``build_mla_absorb``."""
    if not 1 <= qb_split_groups <= lay.GROUPS:
        raise ValueError(f"q_b_proj splits of {qb_split_groups} groups")
    if max_pos < 1 or max_pos * HALF >= 1 << 31:
        raise ValueError(f"a rope cache of {max_pos} rows")
    plan = lay.AbsorbPlan(rows, heads, out_heads or heads, compute_all)
    grid = lay.grid(heads)
    head_ctas = lay.head_ctas(heads)
    lds_count = lds_words(rows)
    name = (
        f"k3_mla_absorb_draft_gfx1250_s{rows}_h{heads}"
        + (f"_o{plan.out_heads}" if plan.out_heads != heads else "")
        + ("" if compute_all else "_slot")
        + f"_k{qb_split_groups}"
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
        k_rope_stride: Int64,
        slots: Int64,
        positions: Int64,
        cos: Int64,
        sin: Int64,
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
            "tiles": plan.tiles,
            "plan": plan,
            "compute_all": compute_all,
            "qb_split_groups": qb_split_groups,
            "max_pos": max_pos,
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
            "k_rope_stride": fx.Int32(k_rope_stride),
            "slots": slots,
            "positions": positions,
            "cos": cos,
            "sin": sin,
            "kv_cache": kv_cache,
            "kv_stride": kv_stride,
            "q_out": q_out,
            "q": q,
            "ql_nope": ql_nope,
        }
        allocator = fx.SharedAllocator()
        lds = allocator.allocate(Lds).peek()
        if bid < head_ctas:
            uk = _project(c, lds.words.ptr, lane, wave)
            _stage_nope(c, lds.words.ptr, tid)
            _absorb(c, uk, lds.words.ptr, lane, wave)
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
        k_rope_stride: Int64,
        slots: Int64,
        positions: Int64,
        cos: Int64,
        sin: Int64,
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
            k_rope_stride,
            slots,
            positions,
            cos,
            sin,
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

    MLA_ABSORB_DRAFT_ABI.check(kernel, launch)
    return MlaAbsorbDraftKernel(
        name, launch, (grid,), (THREADS,), plan, qb_split_groups
    )
