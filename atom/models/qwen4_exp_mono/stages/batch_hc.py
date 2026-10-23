# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small-batch HC gate, MFMA up projection and ordered four-stream mixing."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.qwen4_exp_mono.stages.hc import _ld_bf, _ld_f32, _sigmoid, _st_bf
from atom.mono.device.ops import bf16_pair, bf16_round, mfma_bf16, rsrc, traced


@fx.struct
class Smem:
    gate: fx.Array[fx.Int32, 4 * 160, 16]
    partial: fx.Array[fx.Float32, 4 * 8 * 4 * 16, 16]


@traced
def emit_up(part, normed, weight, raw, stride, sm, tokens, has_inject, store):
    nout = 324 if has_inject else 320
    tid = fx.thread_idx.x
    lane = tid % 64
    wave = tid // 64
    task = fx.block_idx.x
    row = task * 16 + lane % 16
    k0 = wave * 32 + 8 * (lane // 16)
    weights = [
        fx.Vector(
            bo.buffer_load(
                rsrc(weight, nbytes=10240 * 320 * 2),
                (stream * 2560 + row) * 160 + k0 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=2,
            )
        ).bitcast(fx.BFloat16)
        for stream in range(4)
    ]
    k1 = fx.min(8 + wave, 9) * 32 + 8 * (lane // 16)
    tail_weights = [
        fx.Vector(
            bo.buffer_load(
                rsrc(weight, nbytes=10240 * 320 * 2),
                (stream * 2560 + row) * 160 + k1 // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=2,
                mask=wave < 2,
            )
        ).bitcast(fx.BFloat16)
        for stream in range(4)
    ]
    if tid < 160:
        for token in range_constexpr(tokens):
            gates = []
            for half in range_constexpr(2):
                at = tid * 2 + half
                d = _ld_f32(part, token * 4 * nout + at)
                for stream in range_constexpr(1, 4):
                    d = d + _ld_f32(part, token * 4 * nout + stream * nout + at)
                d = bf16_round(d)
                d = bf16_round(d * fx.Float32(0.25))
                gates.append(bf16_round(d * _sigmoid(d)))
            fx.ptr_store(
                bf16_pair(gates[0], gates[1]).bitcast(fx.Int32),
                sm.gate.ptr + token * 160 + tid,
            )
    if has_inject and task == 0:  # noqa: SIM102 -- compile-time injection guard
        if tid < tokens * 4:
            token = tid // 4
            col = 320 + tid % 4
            value = _ld_f32(part, token * 4 * nout + col)
            for stream in range_constexpr(1, 4):
                value = value + _ld_f32(part, token * 4 * nout + stream * nout + col)
            _st_bf(raw, tid, bf16_round(value))
    gpu.barrier()
    token = fx.min(lane % 16, tokens - 1)
    row = task * 16 + lane % 16
    k0 = wave * 32 + 8 * (lane // 16)
    x = fx.Vector(
        fx.ptr_load(sm.gate.ptr + token * 160 + k0 // 2, result_type=T.vec(4, T.i32))
    ).bitcast(fx.BFloat16)
    for stream in range_constexpr(4):
        w = weights[stream]
        acc = mfma_bf16(w, x, fx.Vector.filled(4, 0.0, fx.Float32))
        if wave < 2:
            k1 = (8 + wave) * 32 + 8 * (lane // 16)
            w1 = tail_weights[stream]
            x1 = fx.Vector(
                fx.ptr_load(
                    sm.gate.ptr + token * 160 + k1 // 2, result_type=T.vec(4, T.i32)
                )
            ).bitcast(fx.BFloat16)
            acc = mfma_bf16(w1, x1, acc)
        if lane % 16 < tokens:
            fx.ptr_store(
                acc,
                sm.partial.ptr
                + (stream * 8 + wave) * tokens * 16
                + (lane % 16) * 16
                + (lane // 16) * 4,
            )
    gpu.barrier()
    if tid < 16 * tokens:
        token = tid // 16
        col = task * 16 + tid % 16
        total = fx.Float32(0.0)
        for stream in range_constexpr(4):
            v = fx.Float32(0.0)
            for worker in range_constexpr(8):
                v = v + fx.ptr_load(
                    sm.partial.ptr
                    + (stream * 8 + worker) * tokens * 16
                    + token * 16
                    + tid % 16
                )
            u = bf16_round(v)
            total = total + _sigmoid(u) * _ld_bf(
                normed, token * stride + stream * 2560 + col
            )
        store(token * 2560 + col, bf16_round(total * fx.Float32(0.25)))
