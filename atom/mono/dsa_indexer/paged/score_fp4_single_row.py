# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-row FP4 scoring with all 32 heads in one MFMA matrix."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from atom.mono.device.mx import FP4
from atom.mono.device.ops import CM_DEV, butterfly, global_load, ld_i32, rsrc, traced
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged.score_fp4 import (
    BOUNDS_AT,
    FLT_MAX,
    NEG,
    Q_WORDS,
    QS_WORDS,
    _stage_args,
    _swap_add,
    _vop,
)


def query_row(c):
    lane, lds = c["lane"], c["sel_lds"]["keys"]
    head, group = lane % 32, lane // 32
    v4 = fx.Vector.make_type(4, fx.Int32)
    q, scale = [], fx.Int32(0)
    for step in range(2):
        kg = step * 2 + group
        q.append(fx.Vector(fx.ptr_load(lds + head * 16 + kg * 4, result_type=v4)))
        code = fx.Int32(fx.ptr_load(lds + Q_WORDS + kg * 16 + head % 16))
        scale = scale | (((code >> ((head // 16) * 8)) & 255) << (step * 8))
    weights = []
    for part in range(2):
        values = []
        for mi in range(2):
            first = group * 4 + part * 8 + mi * 16
            word = fx.Vector(
                fx.ptr_load(lds + Q_WORDS + QS_WORDS + first, result_type=v4)
            ).bitcast(fx.Float32)
            values.extend([word[e] * fx.Float32(0.5) for e in range(4)])
        weights.append(values)
    return [(fx.Int32(fx.ptr_load(lds + BOUNDS_AT)), q, scale, weights)]


def key_page(c, trow, st):
    a, l = _stage_args(c), c["lane"] % 32
    shift = a["ishift"]
    blk = fx.min(st * (ip.STEP // ip.BLOCK_ROWS) + l // 4, (a["itab_len"] << shift) - 1)
    entry = ld_i32(a["itab"], trow * a["itab_stride"] + (blk >> shift))
    return (entry << shift) | (blk & ((fx.Int32(1) << shift) - 1))


def load_keys(c, page, nt):
    a, lane = _stage_args(c), c["lane"]
    l, g = lane % 32, lane // 32
    page = fx.Int64(page)
    values, scales = [], fx.Int32(0)
    for tile in range(2):
        row = (l % 4) * 2 + tile
        for step in range(2):
            kg = step * 2 + g
            if c["index_mfma"]:
                physical_row = (page % 8) * 8 + fx.Int64(row)
                base = (
                    a["plane"]
                    + (page // 8) * 4096
                    + fx.Int64(kg * 1024)
                    + physical_row * 16
                )
                scale_base = (
                    a["pscale"]
                    + (page // 8) * 256
                    + fx.Int64((kg % 2) * 128)
                    + (physical_row % 32) * 4
                )
                word = fx.Int32(global_load(scale_base))
                code = (
                    word >> ((fx.Int32(kg // 2 * 2) + fx.Int32(physical_row // 32)) * 8)
                ) & 255
            else:
                base = (
                    a["plane"]
                    + page * 512
                    + fx.Int64(kg * 32 + (row // 4) * 16 + (row % 4) * 128)
                )
                scale_base = a["pscale"] + page * 32 + fx.Int64(kg * 8 + row)
                word = fx.Int32(global_load(scale_base & fx.Int64(-4)))
                code = (word >> (fx.Int32(scale_base & fx.Int64(3)) * 8)) & 255
            values.append(fx.Vector(global_load(base, 4, nt)))
            scales = scales | (code << ((tile * 2 + step) * 8))
    return values, scales


@traced
def score_step(c, lead, rows, outs, kv, kvs, st, after_mfmas):
    a, lane = _stage_args(c), c["lane"]
    bound, query, qscale, weights = rows[0]
    totals = []
    for tile in range_constexpr(2):
        rocdl.s_setprio(1)
        acc = fx.Vector.filled(16, 0.0, fx.Float32)
        for step in range_constexpr(2):
            acc = fx.Vector(
                rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                    T.vec(16, T.f32),
                    [
                        query[step],
                        kv[tile * 2 + step],
                        acc,
                        FP4,
                        FP4,
                        step,
                        qscale,
                        tile * 2 + step,
                        kvs,
                    ],
                )
            )
        rocdl.s_setprio(0)
        if tile == 1:
            rocdl.sched_barrier(0)
            after_mfmas()
            rocdl.sched_barrier(0)
        partials = []
        for part in range_constexpr(2):
            total = None
            for mi in range_constexpr(2):
                for e in range_constexpr(4):
                    x = fx.Float32(acc[part * 4 + mi * 8 + e])
                    relu2 = x + fx.Float32(fmath.absf(x))
                    w = weights[part][mi * 4 + e]
                    total = (
                        _vop("v_mul_f32 $0, $1, $2", relu2, w)
                        if total is None
                        else _vop("v_fma_f32 $0, $1, $2, $3", relu2, w, total)
                    )
            partials.append(total)
        totals.append(_vop("v_add_f32 $0, $1, $2", partials[0], partials[1]))
    total = _swap_add(32, totals[0], totals[1])
    key = st * ip.STEP + (lane % 32) * 2 + lane // 32
    bo.buffer_store(total, outs[0], key, cache_modifier=CM_DEV)
    if a["iprod"] != 0:
        best = fx.Float32(fx.isnan(total).select(fx.Float32(NEG), total))
        best = (key < bound).select(best, fx.Float32(NEG))
        best = fx.min(butterfly(best, (32, 1, 2), fx.max), fx.Float32(FLT_MAX))
        block = st * (ip.STEP // ip.BLOCK_ROWS) + (lane % 32) // 4
        blocks = ip.blocks(bound)
        best = (block == blocks - 1).select(fx.Float32(float("inf")), best)
        if (lane // 32 == 0) & (lane % 4 == 0) & (block < blocks):
            bo.buffer_store(
                best,
                rsrc(a["ibmax"]),
                lead * a["bstride"] + block,
                cache_modifier=CM_DEV,
            )
