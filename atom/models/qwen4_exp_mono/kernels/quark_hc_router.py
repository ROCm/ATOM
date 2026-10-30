# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Quark HC up/mix and BF16 router, sharing the existing tagged HC hand-off."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.kernels.batch_hc_front import (
    Smem,
    load_payload,
    store_payload,
)
from atom.models.qwen4_exp_mono.shapes import HC_ROWS
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_hc import emit_up, smem_type
from atom.models.qwen4_exp_mono.stages.batch_projection import _router_projection
from atom.models.qwen4_exp_mono.stages.hc import _st_bf
from atom.mono.device.ops import CM_DEV, kernel_symbol, launcher, rsrc
from atom.mono.runtime.abi import KernelAbi

ABI = KernelAbi(
    (
        "part",
        "normed",
        "weight",
        "raw",
        "stride",
        "router",
        "logits",
        "mixed",
        "payload",
    )
)
_STREAM = fx.Stream(None)


@cache
def _smem_type(tokens):
    if tokens <= 4:
        return Smem
    hc_type = smem_type(tokens)

    @fx.struct
    class RouterSmem:
        activation: fx.Array[fx.Int32, tokens * 1280, 16]
        reduction: fx.Array[fx.Float32, 8 * 64 * 4, 16]

    @fx.union
    class VerifySmem:
        hc: hc_type
        front: RouterSmem

    return VerifySmem


@cache
def build(tokens, split_k=1, *, native_down=False):
    assert tokens in HC_ROWS
    assert native_down == (tokens > 4)
    assert split_k in ((1,) if native_down else (1, 2))
    name = kernel_symbol(
        "qwen_quark_hc_router", b=tokens, split_k=split_k, down=1 if native_down else 4
    )
    smem = _smem_type(tokens)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        router: Int64,
        logits: Int64,
        mixed: Int64,
        payload: Int64,
    ):
        sm = fx.SharedAllocator().allocate(smem)
        task = fx.block_idx.x
        epoch = (
            fx.Int32(
                bo.buffer_load(
                    rsrc(payload),
                    task * 16 + 1,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            + 1
        )

        def store(col, value):
            _st_bf(mixed, col, value)
            store_payload(payload, col, value, epoch)

        emit_up(
            part,
            normed,
            weight,
            raw,
            stride,
            sm.hc.peek(),
            tokens,
            True,
            store,
            down_parts=1 if native_down else 4,
        )
        gpu.barrier()
        front_sm = sm.front.peek()
        if task < 33 * split_k:
            _router_projection(
                payload,
                router,
                logits,
                task,
                tokens,
                front_sm,
                load=lambda pair, width: load_payload(payload, pair, epoch, width),
                prefetch=True,
                split_k=split_k,
            )

    @launcher(name)
    def launch(
        part: Int64,
        normed: Int64,
        weight: Int64,
        raw: Int64,
        stride: Int64,
        router: Int64,
        logits: Int64,
        mixed: Int64,
        payload: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(
            part, normed, weight, raw, stride, router, logits, mixed, payload
        ).launch(
            grid=(160,),
            block=(512,),
            stream=stream,
        )

    return launch
