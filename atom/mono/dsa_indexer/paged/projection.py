# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Project one head part over a query tile of at most 16 rows."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.mx import UNIT_SCALE, mfma_scaled
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    ld_i32,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.paged.query import (
    IDX_WEIGHTS_SCALE,
    head_weight_sum,
    quantize_fp4_pair,
)
from atom.mono.plan.execution import WAVES

TILE = 16
LDS_PAD = 4


def token_tiles(s):
    """A step's tokens in GEMV passes: [(first token, tokens)]. A GEMV loads its
    weights once and runs every pass on them, an x tile in LDS at a time."""
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    """The LDS rows an x tile holds."""
    return min(s, TILE)


def lds_row(words):
    return words + LDS_PAD


def _fp8_step_load(c, r_w, r_ws, k, rg, st):
    """K step st's global operands of rows 16 rg ..: the two 64-column weight
    chunks (the upper clamped into the row) and the raw scale word. The (16, 16)
    shuffle lays a row group out in 32-column blocks of 16 rows, a chunk two of
    them (lanes 0-31, 32-63); ``k`` % 64 = 32 (a last quarter step): the lanes
    of a block past the row read its last block."""
    lane = c["lane"]
    j = lane // 16
    kg = k // 32
    if const_expr(k % 64 == 0):
        lo_at = (rg * (k // 64) + 2 * st) * 256 + lane * 4
    else:
        lo_at = rg * (k * 4) + fx.min(4 * st + lane // 32, kg - 1) * 128 + lane % 32 * 4
    lo = fx.Vector(
        bo.buffer_load(r_w, lo_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    if const_expr(k % 64 == 0):
        hi_at = (rg * (k // 64) + fx.min(2 * st + 1, k // 64 - 1)) * 256 + lane * 4
    else:
        hi_at = rg * (k * 4) + fx.min(2 * st + 1, k // 64 - 1) * 256 + lane * 4
    hi = fx.Vector(
        bo.buffer_load(r_w, hi_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    wsi = (rg // 2) * kg + fx.min(st * 4 + j, kg - 1)
    word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF


def _fp8_step_mfma(c, k, xl, xsl, st, ops, acc, live=None, rows=None):
    """``acc`` through K step st's scaled MFMA: ``_fp8_step_load``'s operands
    against every token's MXFP8 row in LDS. ``live`` false: a step past the end
    (its operands those of the last step), zero at unit scale. ``k`` % 64 = 32:
    a lane's lower block past the row is zero on both sides."""
    s, lane = c["S"] if rows is None else rows, c["lane"]
    lo, hi, sa = ops
    kg = k // 32
    x_row, s_row = lds_row(k // 4), lds_row(kg)
    j = lane // 16
    col = fx.min(lane % 16, s - 1)
    if const_expr(live is not None):
        st = fx.min(st, (k + 127) // 128 - 1)
        lo = [live.select(lo[d], fx.Int32(0)) for d in range(4)]
    past = (2 * st + 1) * 64 >= k
    if const_expr(live is not None):
        past = past | ~live
    av = fx.Vector.from_elements(
        [lo[d] for d in range(4)] + [past.select(fx.Int32(0), hi[d]) for d in range(4)],
        fx.Int32,
    )
    grp = st * 4 + j
    xlo = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xhi = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + 16 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    if const_expr(k % 64 != 0):
        over = 4 * st + j // 2 >= kg
        av = fx.Vector.from_elements(
            [over.select(fx.Int32(0), av[d]) for d in range(4)]
            + [av[4 + d] for d in range(4)],
            fx.Int32,
        )
        xlo = [over.select(fx.Int32(0), xlo[d]) for d in range(4)]
    xv = fx.Vector.from_elements(
        [xlo[d] for d in range(4)]
        + [past.select(fx.Int32(0), xhi[d]) for d in range(4)],
        fx.Int32,
    )
    sb = fx.ptr_load(xsl + (col * s_row + fx.min(grp, kg - 1)))
    unit = grp >= kg
    if const_expr(live is not None):
        unit = unit | ~live
    sa = unit.select(fx.Int32(UNIT_SCALE), sa)
    sb = unit.select(fx.Int32(UNIT_SCALE), sb)
    return mfma_scaled(av, xv, acc, sa, sb)


@traced
def head_weight(c, hidden, t0, n, h):
    a, lane, wave = c["args"], c["lane"], c["wave"]
    hidden_size = c["hidden_size"]
    ks = [
        k1 + k2 * 512
        for k1 in range(0, hidden_size, 2048)
        for k2 in range(4)
        if k1 + k2 * 512 < hidden_size
    ]
    wv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["iw_w"]),
                (h * hidden_size + fx.min(k0 + lane * 8, hidden_size - 8)) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_NT,
            )
        )
        for k0 in ks
    ]
    for g in range_constexpr(0, n, WAVES):
        nn = min(WAVES, n - g)
        if wave < nn:
            tok = t0 + g + wave
            xv = [
                fx.Vector(
                    bo.buffer_load(
                        rsrc(hidden),
                        (tok * hidden_size + fx.min(k0 + lane * 8, hidden_size - 8))
                        // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                for k0 in ks
            ]
            weight = bf16_round(
                head_weight_sum(xv, wv, ks, hidden_size, lane)
            ) * fx.Float32(IDX_WEIGHTS_SCALE)
            if lane == 0:
                bo.buffer_store(weight, rsrc(a["iw_out"]), tok * ip.HEADS + h)


@traced
def project_group(c, hidden, t0, n, h, group, parts, with_weight=True):
    a, tid, lane, wave = c["args"], c["tid"], c["lane"], c["wave"]
    query_rank = c["query_rank"]
    k_steps = -(-query_rank // 128)
    width = ip.DIM // parts
    pairs = width // 2
    rg = h * 8 + group * (width // 16) + wave // parts
    steps = [wave % parts + i * parts for i in range(-(-k_steps // parts))]
    ops = [
        _fp8_step_load(
            c,
            rsrc(a["iq_w"]),
            rsrc(a["iq_ws"]),
            query_rank,
            rg,
            fx.min(st, k_steps - 1),
        )
        for st in steps
    ]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(-(-k_steps // parts)):
        acc = _fp8_step_mfma(
            c,
            query_rank,
            c["qxl"],
            c["qxsl"],
            steps[i],
            ops[i],
            acc,
            live=steps[i] < k_steps,
            rows=n,
        )
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < n * pairs:
        tr, rp = tid // pairs, tid % pairs
        values = []
        for e in range_constexpr(2):
            dim = rp * 2 + e
            rowgroup = dim // 16
            mlane = tr + (dim % 16 // 4) * 16
            v = fx.Float32(0.0)
            for part in range_constexpr(parts):
                v = v + fx.ptr_load(
                    c["red"] + ((rowgroup * parts + part) * 64 + mlane) * 4 + dim % 4
                )
            values.append(bf16_round(v))
        tok = t0 + tr
        pos = ld_i32(a["pos"], tok * 2)
        ri = fx.max(group * pairs + rp - 32, 0)
        cs = fx.Float32(
            bo.buffer_load(rsrc(a["cos"]), pos * 32 + ri, vec_width=1, dtype=T.bf16)
        )
        sn = fx.Float32(
            bo.buffer_load(rsrc(a["sin"]), pos * 32 + ri, vec_width=1, dtype=T.bf16)
        )
        v0, v1 = values
        x0 = bf16_round(
            (group * (width // 32) + rp // 16 >= 2).select(v0 * cs - v1 * sn, v0)
        )
        x1 = bf16_round(
            (group * (width // 32) + rp // 16 >= 2).select(v0 * sn + v1 * cs, v1)
        )
        byte, code = quantize_fp4_pair(x0, x1)
        bo.buffer_store(
            fx.Int8(byte),
            rsrc(a["iq_out"]),
            (tok * ip.HEADS + h) * (ip.DIM // 2) + group * pairs + rp,
        )
        if rp % 16 == 0:
            bo.buffer_store(
                fx.Int8(code),
                rsrc(a["iqs_out"]),
                tok * ip.QS_BYTES
                + (group * (width // 32) + rp // 16) * 16 * ip.QS_M_TILES
                + (h % 16) * ip.QS_M_TILES
                + h // 16,
            )
    if const_expr(with_weight):  # noqa: SIM102
        if group == 0:
            head_weight(c, hidden, t0, n, h)
    publish(
        c["put"], c["qready"], ((t0 + tid) * ip.HEADS + h) * parts + group, 1, tid < n
    )
