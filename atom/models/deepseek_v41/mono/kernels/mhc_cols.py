# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The mHC seam's per-column math on 8 columns of a row: the owed post settled
into the residual streams and the streams collapsed, each step rounded as the
original (aiter ``mhc_post``: fma(x, post, r0 c0), then fma(r_h, c_h, ..),
RNE to bf16; ``collapse_streams``: 0 + p0 s0 + .., no fma, bf16). Every column
is its own: a rank settles and collapses the columns it owns alone."""

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr

from atom.models.deepseek_v41.mono.kernels.dims import HC
from atom.mono.device.ops import bf2_f32, bf16_round, words4


@dataclass
class Seam:
    """A row's mixes of a seam that owes its post: ``post`` [HC], ``comb``
    [HC][HC] (stream h into stream o) and the next ``pre`` [HC], fp32."""

    post: object
    comb: list
    pre: object


def seam_mixes(post, comb, pre, t) -> Seam:
    """Row t's mixes from the [rows, HC] / [rows, HC, HC] / [rows, HC] fp32
    buffers at ``post`` / ``comb`` / ``pre``."""
    return Seam(
        words4(post, t * HC).bitcast(fx.Float32),
        [words4(comb, (t * HC + h) * HC).bitcast(fx.Float32) for h in range(HC)],
        words4(pre, t * HC).bitcast(fx.Float32),
    )


def settled8(res, sub, hidden, t, col, mix):
    """Row t's streams at columns col .. + 7 with the owed post (``sub``, the
    sublayer's output) settled: [HC][8] f32, each bf16-rounded. ``res``
    [rows, HC, hidden] and ``sub`` [rows, hidden] bf16."""
    rw = [words4(res, ((t * HC + h) * hidden + col) // 2) for h in range(HC)]
    xw = words4(sub, (t * hidden + col) // 2)
    out = [[None] * 8 for _ in range(HC)]
    for k in range_constexpr(8):
        x = bf2_f32(xw[k // 2])[k % 2]
        r = [bf2_f32(rw[h][k // 2])[k % 2] for h in range(HC)]
        for o in range_constexpr(HC):
            v = fx.Float32(fmath.fma(x, mix.post[o], r[0] * mix.comb[0][o]))
            for h in range_constexpr(1, HC):
                v = fx.Float32(fmath.fma(r[h], mix.comb[h][o], v))
            out[o][k] = bf16_round(v)
    return out


def collapsed8(streams, mix):
    """The settled streams (``settled8``) collapsed by the next pre: 8 f32,
    bf16-rounded."""
    out = []
    for k in range_constexpr(8):
        acc = fx.Float32(0.0)
        for o in range_constexpr(HC):
            acc = acc + mix.pre[o] * streams[o][k]
        out.append(bf16_round(acc))
    return out
