# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Independent routed G1 tiles, per-request quantization, then routed G2."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64, T

from atom.models.qwen4_exp_mono.layout import (
    finish_scratch_bytes,
    finish_scratch_layout,
)
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.finish import FinishSmem
from atom.models.qwen4_exp_mono.stages.g1 import (
    _stage_routed_g1,
)
from atom.models.qwen4_exp_mono.stages.g2 import _routed_slot, wave_routed_g2
from atom.mono.device.ops import (
    bf16_round,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
)

_STRIDE = finish_scratch_bytes()
_LAYOUT = finish_scratch_layout()


def context(sm, scratch, tid, args):
    c = {
        "tid": tid,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": sm.red.ptr,
        "up": sm.up.ptr,
        "x": sm.x.ptr,
        "scratch": scratch,
        "layout": _LAYOUT,
        "args": args,
        "poll": lambda specs: None,
        "put": lambda *args: None,
    }
    for name, (offset, _) in _LAYOUT.items():
        c[name] = scratch + fx.Int64(offset)
    return c


_STREAM = fx.Stream(None)


def build_g1(tokens, *, prefetch=False):
    name = kernel_symbol("qwen_batched_parallel_routed_g1", b=tokens)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(scratch: Int64, weight: Int64, scale: Int64):
        token = fx.Int64(fx.block_idx.x // 400)
        task = fx.block_idx.x % 400
        sm = fx.SharedAllocator().allocate(FinishSmem).peek()
        c = context(
            sm,
            scratch + token * _STRIDE,
            fx.thread_idx.x,
            {"routed_w13": weight, "routed_w13_scale": scale},
        )
        _stage_routed_g1(c, task, 40 + task, quantize=False, prefetch=prefetch)

    @launcher(name)
    def launch(
        scratch: Int64, weight: Int64, scale: Int64, stream: fx.Stream = _STREAM
    ):
        _ = SOURCES
        kernel(scratch, weight, scale).launch(
            grid=(tokens * 400,), block=(512,), stream=stream
        )

    return launch


def build_g2_partials(tokens):
    """Independent expert contributions; the merge keeps top-k BF16 order."""
    return _build_g2(tokens, parallel_experts=True)


@traced
def _g2_tile(
    scratch,
    weight,
    scale,
    out,
    *,
    parallel_experts,
    shared=None,
    wave_experts=False,
    owned_sparse=False,
):
    tiles_per_token = 1600 if parallel_experts else 160
    task = fx.block_idx.x
    token = fx.Int64(task // tiles_per_token)
    slot = task // 160 % 10
    tile = task % 160
    tid = fx.thread_idx.x
    sm = fx.SharedAllocator().allocate(FinishSmem).peek()
    c = context(
        sm,
        scratch + token * _STRIDE,
        tid,
        {"routed_w2": weight, "routed_w2_scale": scale},
    )
    c["owned_sparse_g2"] = owned_sparse
    if tid < 16:
        fx.ptr_store(fx.Float32(0.0), sm.up.ptr + tid)
    fx.gpu.barrier()
    if parallel_experts:
        _routed_slot(c, tile, slot)
    elif wave_experts:
        wave_routed_g2(c, tile)
    else:
        for slot in range_constexpr(10):
            _routed_slot(c, tile, slot)
    if tid < 16:
        value = fx.ptr_load(sm.up.ptr + tid)
        if parallel_experts:
            bo.buffer_store(
                value.to(fx.BFloat16),
                rsrc(out),
                (token * 10 + slot) * 2560 + tile * 16 + tid,
            )
        else:
            value = bf16_round(value)
            if shared is not None:
                shared_value = fx.Float32(
                    bo.buffer_load(
                        rsrc(shared),
                        token * 2560 + tile * 16 + tid,
                        vec_width=1,
                        dtype=T.bf16,
                    )
                )
                value = value + shared_value
            bo.buffer_store(
                value.to(fx.BFloat16),
                rsrc(out),
                token * 2560 + tile * 16 + tid,
            )


def _build_g2(tokens, *, parallel_experts):
    tiles_per_token = 1600 if parallel_experts else 160
    name = kernel_symbol(
        (
            "qwen_routed_g2_parallel_experts"
            if parallel_experts
            else "qwen_batched_routed_g2_seam"
        ),
        b=tokens,
    )

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(scratch: Int64, weight: Int64, scale: Int64, out: Int64):
        _g2_tile(scratch, weight, scale, out, parallel_experts=parallel_experts)

    @launcher(name)
    def launch(
        scratch: Int64,
        weight: Int64,
        scale: Int64,
        out: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(scratch, weight, scale, out).launch(
            grid=(tokens * tiles_per_token,), block=(512,), stream=stream
        )

    return launch


def build_g2_shared(tokens, *, wave_experts=False, owned_sparse=False):
    """Keep routed BF16 reduction order and add shared at its final endpoint."""
    name = kernel_symbol("qwen_batched_routed_g2_shared", b=tokens)

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(scratch: Int64, weight: Int64, scale: Int64, shared: Int64, out: Int64):
        _g2_tile(
            scratch,
            weight,
            scale,
            out,
            parallel_experts=False,
            shared=shared,
            wave_experts=wave_experts,
            owned_sparse=owned_sparse,
        )

    @launcher(name)
    def launch(
        scratch: Int64,
        weight: Int64,
        scale: Int64,
        shared: Int64,
        out: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(scratch, weight, scale, shared, out).launch(
            grid=(tokens * 160,), block=(512,), stream=stream
        )

    return launch


def build_g2_merge(tokens):
    name = kernel_symbol("qwen_routed_g2_shared_merge", b=tokens)

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(partials: Int64, shared: Int64, out: Int64):
        col = fx.block_idx.x * 256 + fx.thread_idx.x
        token = fx.Int64(fx.block_idx.y)
        acc = fx.Float32(0.0)
        for slot in range_constexpr(10):
            value = fx.Float32(
                bo.buffer_load(
                    rsrc(partials),
                    (token * 10 + slot) * 2560 + col,
                    vec_width=1,
                    dtype=T.bf16,
                )
            )
            acc = bf16_round(acc + value)
        shared_value = fx.Float32(
            bo.buffer_load(rsrc(shared), token * 2560 + col, vec_width=1, dtype=T.bf16)
        )
        bo.buffer_store(
            (acc + shared_value).to(fx.BFloat16), rsrc(out), token * 2560 + col
        )

    @launcher(name)
    def launch(partials: Int64, shared: Int64, out: Int64, stream: fx.Stream = _STREAM):
        _ = SOURCES
        kernel(partials, shared, out).launch(
            grid=(10, tokens), block=(256,), stream=stream
        )

    return launch
