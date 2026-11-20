# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Project one head part over a query tile of at most 16 rows."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.kernels.attn_pre import (
    HIDDEN,
    Q_RANK,
    _fp8_step_load,
    _fp8_step_mfma,
)
from atom.models.deepseek_v41.mono.kernels.index_query import (
    IDX_WEIGHTS_SCALE,
    head_weight_sum,
    quantize_fp4_pair,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    ld_i32,
    rsrc,
    traced,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import WAVES


@traced
def head_weight(c, hidden, t0, n, h):
    a, lane, wave = c["args"], c["lane"], c["wave"]
    ks = [
        k1 + k2 * 512
        for k1 in range(0, HIDDEN, 2048)
        for k2 in range(4)
        if k1 + k2 * 512 < HIDDEN
    ]
    wv = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["iw_w"]),
                (h * HIDDEN + fx.min(k0 + lane * 8, HIDDEN - 8)) // 2,
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
                        (tok * HIDDEN + fx.min(k0 + lane * 8, HIDDEN - 8)) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                for k0 in ks
            ]
            weight = bf16_round(head_weight_sum(xv, wv, ks, HIDDEN, lane)) * fx.Float32(
                IDX_WEIGHTS_SCALE
            )
            if lane == 0:
                bo.buffer_store(weight, rsrc(a["iw_out"]), tok * ip.HEADS + h)


@traced
def project_group(c, hidden, t0, n, h, group, parts, with_weight=True):
    a, tid, lane, wave = c["args"], c["tid"], c["lane"], c["wave"]
    width = ip.DIM // parts
    pairs = width // 2
    rg = h * 8 + group * (width // 16) + wave // parts
    steps = [wave % parts + i * parts for i in range(-(-10 // parts))]
    ops = [
        _fp8_step_load(c, rsrc(a["iq_w"]), rsrc(a["iq_ws"]), Q_RANK, rg, fx.min(st, 9))
        for st in steps
    ]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(-(-10 // parts)):
        acc = _fp8_step_mfma(
            c,
            Q_RANK,
            c["qxl"],
            c["qxsl"],
            steps[i],
            ops[i],
            acc,
            live=steps[i] < 10,
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
