# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 sparse BF16 attention stages for the gfx1250 local K2a chain.

As in gfx950, score publishes independent 16-key splits and BF16 P, then
PV combines the splits and adds the denominator-only sink. Wave32 WMMA
replaces MFMA; no wave64 reduction order or bitwise output contract is used.
Native prefix/pool and extend/ring index lists remain separate. PV emits
inverse-RoPE group32 FP8 directly to the grouped output projection.
"""

import os

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.kernels.gfx1250_attn_post import (
    HEAD_DIM,
    IRQ_FLOOR,
    inverse_rope_pair,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_mhc import _typed
from atom.mono.device.mx import FP8_MAX, clamp_fp8, code_ceil, rcp_pow2
from atom.mono.device.ops import bf16_round, fp8_pack4, hw_exp2, lane_gather, traced

PRUNE_EMPTY = os.environ.get("ATOM_DSV41_GFX1250_ATTENTION_PRUNE", "0") == "1"

HEAD_TILE, SPLITS, KEYS_PER_SPLIT = 16, 64, 16
MAX_KEYS = SPLITS * KEYS_PER_SPLIT
SCORE_PER_TILE, PV_PER_TILE = 8, HEAD_DIM // 32
LOG2E, NEG = 1.4426950408889634, -3.4028234663852886e38
PV_PARTIAL_BYTES = 8 * HEAD_TILE * 32 * 4
ATTENTION_LDS_BYTES = PV_PARTIAL_BYTES + 256


def attention_tasks(rows, heads):
    tiles = heads // HEAD_TILE
    return rows * tiles * SCORE_PER_TILE, rows * tiles * PV_PER_TILE


def span_valid(ps, pe, es, ee, prefix_capacity, extend_capacity):
    """Shared host/device shape predicate, independent of payload values."""
    return (
        (ps >= 0)
        & (pe >= ps)
        & (pe <= prefix_capacity)
        & (es >= 0)
        & (ee >= es)
        & (ee <= extend_capacity)
        & (pe - ps <= MAX_KEYS)
        & (ee - es <= MAX_KEYS)
        & (pe - ps + ee - es <= MAX_KEYS)
    )


def _spans(prefix_indptr, extend_indptr, row):
    pp, ep = _typed(prefix_indptr, fx.Int32), _typed(extend_indptr, fx.Int32)
    return (
        fx.ptr_load(pp + row),
        fx.ptr_load(pp + row + 1),
        fx.ptr_load(ep + row),
        fx.ptr_load(ep + row + 1),
    )


@traced
def validate_attention_spans(
    prefix_indptr, extend_indptr, row, prefix_capacity, extend_capacity, diag, tid
):
    ps, pe, es, ee = _spans(prefix_indptr, extend_indptr, row)
    valid = span_valid(ps, pe, es, ee, prefix_capacity, extend_capacity)
    if (~valid) & (tid == 0):
        fx.atomic_cas(
            diag,
            fx.Int32(0),
            fx.Int32(14_000_000) + fx.Int32(fx.block_idx.x),
            success_ordering=fx.AtomicOrdering.Release,
            failure_ordering=fx.AtomicOrdering.Monotonic,
            syncscope="agent",
        )


@traced
def _key_address(pool, ring, prefix, extend, spans, key, pool_rows, ring_rows, diag):
    """A checked key's 64-bit row address; -1 skips, other illegal IDs fail.

    Span validation precedes this function. Inactive regions never load an
    index, so empty inputs may have null pointers. No row is dereferenced
    until both the index-list bounds and the source row bounds have passed.
    """
    ps, pe, es, ee = spans
    nprefix, nextend = pe - ps, ee - es
    picked = key < nprefix
    live = (key >= 0) & (key < nprefix + nextend)
    slot = fx.Int32(-1)
    if live:
        if picked:
            slot = fx.ptr_load(_typed(prefix, fx.Int32) + ps + key)
        else:
            slot = fx.ptr_load(_typed(extend, fx.Int32) + es + key - nprefix)
    limit = picked.select(fx.Int32(pool_rows), fx.Int32(ring_rows))
    invalid = live & ((slot < -1) | (slot >= limit))
    if invalid:
        fx.atomic_cas(
            diag,
            fx.Int32(0),
            fx.Int32(15_000_000) + fx.Int32(fx.block_idx.x),
            success_ordering=fx.AtomicOrdering.Release,
            failure_ordering=fx.AtomicOrdering.Monotonic,
            syncscope="agent",
        )
    base = picked.select(fx.Int64(fx.ptrtoint(pool)), fx.Int64(fx.ptrtoint(ring)))
    return base + fx.Int64(slot) * (HEAD_DIM * 2), live & (slot >= 0) & (slot < limit)


def _load_bf16x16(address):
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
        address,
    )
    return fx.Vector(
        fx.ptr_load(ptr, result_type=fx.Vector.make_type(8, fx.Int32))
    ).bitcast(fx.BFloat16)


@traced
def _load_key_bf16x16(address, live, column):
    values = fx.Vector.filled(16, 0.0, fx.BFloat16)
    if live:
        values = _load_bf16x16(address + fx.Int64(column) * 2)
    return values


def _wmma(accum, a_values, b_values):
    atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.BFloat16, fx.Float32))
    a, b = fx.make_rmem_tensor(16, fx.BFloat16), fx.make_rmem_tensor(16, fx.BFloat16)
    a.store(a_values)
    b.store(b_values)
    fx.gemm(atom, accum, b, a, accum)


@traced
def attention_score_device(
    query,
    pool,
    ring,
    prefix,
    prefix_indptr,
    extend,
    extend_indptr,
    maxima,
    sums,
    probabilities,
    diag,
    tid,
    task,
    rows,
    heads,
    pool_rows,
    ring_rows,
    softmax_scale,
):
    """Eight waves each produce one 16-key split for a 16-head tile."""
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    tiles = heads // HEAD_TILE
    owner, group = task // SCORE_PER_TILE, task % SCORE_PER_TILE
    row, tile = owner // tiles, owner % tiles
    split = group * 8 + wave
    spans = _spans(prefix_indptr, extend_indptr, row)
    address, live = _key_address(
        pool,
        ring,
        prefix,
        extend,
        spans,
        split * KEYS_PER_SPLIT + lane16,
        pool_rows,
        ring_rows,
        diag,
    )
    q_address = (
        fx.Int64(fx.ptrtoint(query))
        + fx.Int64((row * heads + tile * HEAD_TILE + lane16) * HEAD_DIM) * 2
    )
    accum = fx.make_rmem_tensor(8, fx.Float32)
    accum.store(fx.full(8, 0.0, fx.Float32))
    nkeys = spans[1] - spans[0] + spans[3] - spans[2]
    if (split * KEYS_PER_SPLIT < nkeys) if const_expr(PRUNE_EMPTY) else fx.Boolean(True):
        for chunk in range_constexpr(HEAD_DIM // 32):
            column = chunk * 32 + half * 16
            qv = _load_bf16x16(q_address + fx.Int64(column) * 2)
            kv = _load_key_bf16x16(address, live, column)
            _wmma(accum, qv, kv)
    values = accum.load()
    scores, keep = [], []
    for j in range_constexpr(8):
        valid = lane_gather(live.to(fx.Int32), half * 8 + j) != 0
        keep.append(valid)
        scores.append(
            valid.select(values[j] * fx.Float32(softmax_scale * LOG2E), fx.Float32(NEG))
        )
    maximum = fx.Float32(NEG)
    for score in scores:
        maximum = fx.max(maximum, score)
    maximum = fx.max(maximum, maximum.shuffle_xor(16, 32))
    ps = [
        keep[j].select(hw_exp2(scores[j] - maximum), fx.Float32(0.0)) for j in range(8)
    ]
    total = fx.Float32(0.0)
    for value in ps:
        total = total + value
    total = total + total.shuffle_xor(16, 32)
    stat = (row * SPLITS + split) * heads + tile * HEAD_TILE + lane16
    if half == 0:
        fx.ptr_store(maximum, _typed(maxima, fx.Float32) + stat)
        fx.ptr_store(total, _typed(sums, fx.Float32) + stat)
    at = (
        ((row * tiles + tile) * SPLITS + split) * HEAD_TILE * KEYS_PER_SPLIT
        + lane16 * KEYS_PER_SPLIT
        + half * 8
    )
    fx.ptr_store(
        fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16),
        _typed(probabilities, fx.BFloat16) + at,
    )
    fx.gpu.barrier()


@traced
def attention_pv_quant_device(
    pool,
    ring,
    prefix,
    prefix_indptr,
    extend,
    extend_indptr,
    sink,
    positions,
    cos,
    sin,
    maxima,
    sums,
    probabilities,
    attention_q,
    attention_s,
    debug_out,
    local,
    diag,
    tid,
    task,
    rows,
    heads,
    pool_rows,
    ring_rows,
    debug=False,
):
    """Combine all split PVs, add sink, round BF16, rotate and quantize.

    KV tiles and wave partials share one LDS union. Every wave passes a CTA
    barrier after each tile's WMMA consumption before another wave can reuse
    that storage. Per-head stats remain beyond the union until epilogue ends.
    """
    lane, wave = tid % 32, tid // 32
    lane16, half = lane % 16, lane // 16
    tiles = heads // HEAD_TILE
    owner, group = task // PV_PER_TILE, task % PV_PER_TILE
    row, tile = owner // tiles, owner % tiles
    red = _typed(local, fx.Float32)
    stat_m, stat_l = (
        red + PV_PARTIAL_BYTES // 4,
        red + PV_PARTIAL_BYTES // 4 + HEAD_TILE,
    )
    am, al = _typed(maxima, fx.Float32), _typed(sums, fx.Float32)
    for k in range_constexpr(2):
        head = tile * HEAD_TILE + wave * 2 + k
        at0 = (row * SPLITS + lane) * heads + head
        at1 = (row * SPLITS + lane + 32) * heads + head
        m0, m1 = fx.ptr_load(am + at0), fx.ptr_load(am + at1)
        maximum = fx.max(m0, m1)
        for off in (16, 8, 4, 2, 1):
            maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
        total = fx.ptr_load(al + at0) * hw_exp2(m0 - maximum) + fx.ptr_load(
            al + at1
        ) * hw_exp2(m1 - maximum)
        for off in (16, 8, 4, 2, 1):
            total = total + total.shuffle_xor(off, 32)
        if lane == 0:
            fx.ptr_store(maximum, stat_m + wave * 2 + k)
            fx.ptr_store(total, stat_l + wave * 2 + k)
    fx.gpu.barrier()
    maximum = fx.ptr_load(stat_m + lane16)
    kv_lds = _typed(local, fx.BFloat16)
    spans = _spans(prefix_indptr, extend_indptr, row)
    totals = [fx.make_rmem_tensor(8, fx.Float32) for _ in range(2)]
    for total in totals:
        total.store(fx.full(8, 0.0, fx.Float32))
    nkeys = spans[1] - spans[0] + spans[3] - spans[2]
    for iteration in range_constexpr(SPLITS // 8):
        # This predicate is CTA-uniform. All waves enter/skip the same LDS
        # barriers; the final partial tile retains its existing live masks.
        if (iteration * 8 * KEYS_PER_SPLIT < nkeys) if const_expr(PRUNE_EMPTY) else fx.Boolean(True):
            split = wave + iteration * 8
            address, live = _key_address(
                pool,
                ring,
                prefix,
                extend,
                spans,
                split * KEYS_PER_SPLIT + lane // 2,
                pool_rows,
                ring_rows,
                diag,
            )
            kv = _load_key_bf16x16(address, live, group * 32 + (lane % 2) * 16)
            fx.ptr_store(
                kv, kv_lds + (wave * KEYS_PER_SPLIT + lane // 2) * 32 + (lane % 2) * 16
            )
            fx.gpu.barrier()
            p_at = (
                (row * tiles + tile) * SPLITS + split
            ) * HEAD_TILE * KEYS_PER_SPLIT + lane16 * KEYS_PER_SPLIT
            pv = fx.Vector(
                fx.ptr_load(
                    _typed(probabilities, fx.BFloat16) + p_at,
                    result_type=fx.Vector.make_type(16, fx.BFloat16),
                )
            )
            p_values = fx.Vector.from_elements(
                [(half == 0).select(pv[j], fx.BFloat16(0.0)) for j in range(16)],
                fx.BFloat16,
            )
            alpha = hw_exp2(
                fx.ptr_load(am + (row * SPLITS + split) * heads + tile * HEAD_TILE + lane16)
                - maximum
            )
            for part in range_constexpr(2):
                b_values = fx.Vector.from_elements(
                    [
                        (half == 0).select(
                            fx.ptr_load(
                                kv_lds
                                + (wave * KEYS_PER_SPLIT + j) * 32
                                + part * 16
                                + lane16
                            ),
                            fx.BFloat16(0.0),
                        )
                        for j in range(16)
                    ],
                    fx.BFloat16,
                )
                accum = fx.make_rmem_tensor(8, fx.Float32)
                accum.store(fx.full(8, 0.0, fx.Float32))
                _wmma(accum, p_values, b_values)
                got, old = accum.load(), totals[part].load()
                totals[part].store(
                    fx.Vector.from_elements(
                        [fx.fma(got[j], alpha, old[j]) for j in range(8)], fx.Float32
                    )
                )
            fx.gpu.barrier()
        # The final iteration's barrier protects the LDS KV->partial union.
    for part in range_constexpr(2):
        fx.ptr_store(
            totals[part].load(),
            red + (wave * HEAD_TILE + lane16) * 32 + part * 16 + half * 8,
        )
    fx.gpu.barrier()
    head, pair = tid // 16, tid % 16
    sums2 = [fx.Float32(0.0), fx.Float32(0.0)]
    for w in range_constexpr(8):
        for j in range_constexpr(2):
            sums2[j] = sums2[j] + fx.ptr_load(
                red + (w * HEAD_TILE + head) * 32 + pair * 2 + j
            )
    sink_value = fx.ptr_load(
        _typed(sink, fx.Float32) + tile * HEAD_TILE + head
    ) * fx.Float32(LOG2E)
    m = fx.ptr_load(stat_m + head)
    final_m = fx.max(m, sink_value)
    alpha_kv = hw_exp2(m - final_m)
    denom = fx.fma(fx.ptr_load(stat_l + head), alpha_kv, hw_exp2(sink_value - final_m))
    denom = fx.max(denom, fx.Float32(1e-30))
    values = [bf16_round(sums2[j] * alpha_kv / denom) for j in range(2)]
    column = group * 32 + pair * 2
    at = (row * heads + tile * HEAD_TILE + head) * HEAD_DIM + column
    if const_expr(debug):
        for j in range_constexpr(2):
            fx.ptr_store(
                values[j].to(fx.BFloat16), _typed(debug_out, fx.BFloat16) + at + j
            )
    position = fx.ptr_load(_typed(positions, fx.Int64) + row)
    even, odd = inverse_rope_pair(values[0], values[1], column, position, cos, sin)
    maximum = fx.max(fx.max(abs(even), abs(odd)), fx.Float32(IRQ_FLOOR))
    for off in (1, 2, 4, 8):
        maximum = fx.max(maximum, maximum.shuffle_xor(off, 32))
    code = code_ceil(maximum * fx.Float32(1.0 / FP8_MAX))
    multiplier = rcp_pow2(code)
    packed = [clamp_fp8(even * multiplier), clamp_fp8(odd * multiplier)]
    next_values = [v.shuffle_xor(1, 32) for v in packed]
    if pair % 2 == 0:
        fx.ptr_store(
            fp8_pack4(*packed, *next_values), _typed(attention_q, fx.Int32) + at // 4
        )
    if pair == 0:
        fx.ptr_store(code.to(fx.Uint8), _typed(attention_s, fx.Uint8) + at // 32)
    fx.gpu.barrier()
