# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Group-32 MXFP4 arithmetic; cache addresses are OPUS gfx950's, not
the older FlyDSL DSA preshuffle. The even dimension is the low nibble.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from atom.mono.device.mx import FP4, mfma_scaled
from atom.mono.device.ops import CM_DEV, butterfly, rsrc, traced, xshfl
from atom.mono.dsa_indexer.plan import fp4_cache_byte, fp4_scale_byte

MIN_AMAX = 6.0 * 2.0**-126


def e2m1(x):
    """Saturating nearest-even conversion, including signed zero."""
    m = abs(x)
    code = fx.Int32(0)
    for edge, value, inclusive in (
        (0.25, 1, False),
        (0.75, 2, True),
        (1.25, 3, False),
        (1.75, 4, True),
        (2.5, 5, False),
        (3.5, 6, True),
        (5.0, 7, False),
    ):
        past = m >= fx.Float32(edge) if inclusive else m > fx.Float32(edge)
        code = fx.Int32(past.select(fx.Int32(value), code))
    return code | ((x.bitcast(fx.Int32) >> 28) & 8)


def scale(amax):
    """AITER scaling mode 1: ceil(log2(max(amax, MIN_AMAX) * f32(1/6)))."""
    bits = (fx.max(amax, fx.Float32(MIN_AMAX)) * fx.Float32(1.0 / 6.0)).bitcast(
        fx.Int32
    )
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
    inv = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
    return code, inv


@traced
def store_key(c, slot, y):
    """A wave's dim lane and lane+64 after LayerNorm/RoPE, two group pairs."""
    lane, sh, a = c["lane"], c["shape"], c["args"]
    page, row = slot // sh.block_rows, slot % sh.block_rows
    data = rsrc(a["cache"] + fx.Int64(page) * fx.Int64(sh.block_bytes))
    scales = rsrc(a["cache_scale"] + fx.Int64(page) * fx.Int64(sh.block_scale_bytes))
    for i in range_constexpr(2):
        amax = butterfly(abs(y[i]), (1, 2, 4, 8, 16), fx.max)
        code, inv = scale(amax)
        nibble = e2m1(y[i] * inv)
        partner = xshfl(nibble, 1)
        if lane % 2 == 0:
            bo.buffer_store(
                fx.Int8(nibble | (partner << 4)),
                data,
                fp4_cache_byte(row, lane // 2 + i * 32),
                cache_modifier=CM_DEV,
            )
        if lane % 32 == 0:
            bo.buffer_store(
                fx.Int8(code),
                scales,
                fp4_scale_byte(row, lane // 32 + i * 2),
                cache_modifier=CM_DEV,
            )


@traced
def store_query(c, task, lane, x0, x1):
    """Internal query scratch keeps its 128-byte head stride: 64 packed bytes
    followed by four natural-order scale bytes. This is not an external ABI."""
    sh = c["shape"]
    amax = butterfly(fx.max(abs(x0), abs(x1)), (1, 2, 4, 8), fx.max)
    code, inv = scale(amax)
    out = rsrc(c["iq8"])
    bo.buffer_store(
        fx.Int8(e2m1(x0 * inv) | (e2m1(x1 * inv) << 4)),
        out,
        task * sh.head_dim + lane,
        cache_modifier=CM_DEV,
    )
    if lane % 16 == 0:
        bo.buffer_store(
            fx.Int8(code),
            out,
            task * sh.head_dim + 64 + lane // 16,
            cache_modifier=CM_DEV,
        )


def keys(c, block, key, chunk, page):
    """One 16-key MFMA tile, 32 dimensions and its E8M0 scale per lane."""
    sh = c["shape"]
    data = fx.Vector(
        bo.buffer_load(
            block,
            fp4_cache_byte(key, chunk * 16) // 4,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    scales = rsrc(
        c["args"]["cache_scale"] + fx.Int64(page) * fx.Int64(sh.block_scale_bytes)
    )
    byte = fp4_scale_byte(key, chunk)
    word = fx.Int32(
        bo.buffer_load(
            scales,
            byte // 4,
            vec_width=1,
            dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )
    return data, (word >> ((byte % 4) * 8)) & 255


def query(sh, chunk, lane, q_at, w_at):
    qs, ws = [], []
    for hh in range_constexpr(sh.heads // 16):
        head = hh * 16 + lane % 16
        data = q_at(head, chunk * 4)
        word = q_at(head, 16)[0]
        qs.append((data, (fx.Int32(word) >> (chunk * 8)) & 255))
        w = w_at(hh * 16 + chunk * 4)
        ws.append([fx.Int32(w[i]).bitcast(fx.Float32) for i in range(4)])
    return qs, ws


def terms(sh, qs, ws, data, scales):
    result = {}
    for hh in range_constexpr(sh.heads // 16):
        acc = mfma_scaled(
            qs[hh][0],
            data,
            fx.Vector.filled(4, 0.0, fx.Float32),
            qs[hh][1],
            scales,
            FP4,
            FP4,
        )
        for e in range_constexpr(4):
            result[hh, e] = (fx.max(acc[e], fx.Float32(0.0)), ws[hh][e])
    return result
