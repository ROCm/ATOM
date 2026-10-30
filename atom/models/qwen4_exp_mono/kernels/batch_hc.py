# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small-row HC up shares BF16 weights through MFMA after one gate preparation."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
from flydsl.expr.typing import Int64

from atom.model_ops.qwen4_exp.ops.hc_fused import _hc_pre_kernel
from atom.models.qwen4_exp_mono.shapes import DECODE_ROWS
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_hc import emit_up, smem_type
from atom.models.qwen4_exp_mono.stages.hc import _st_bf
from atom.mono.device.ops import (
    kernel_symbol,
    launcher,
)

_STREAM = fx.Stream(None)


@cache
def build(tokens, has_inject=True, *, native_down=False):
    assert (tokens == 16 and has_inject) if native_down else tokens in DECODE_ROWS
    Smem = smem_type(tokens)
    name = kernel_symbol(
        "qwen_batch_hc_up", b=tokens, inject=has_inject, native_down=native_down
    )

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        part: Int64, normed: Int64, weight: Int64, out: Int64, raw: Int64, stride: Int64
    ):
        sm = fx.SharedAllocator().allocate(Smem).peek()
        emit_up(
            part,
            normed,
            weight,
            raw,
            stride,
            sm,
            tokens,
            has_inject,
            lambda col, value: _st_bf(out, col, value),
            down_parts=1 if native_down else 4,
        )

    @launcher(name)
    def launch(
        part: Int64,
        normed: Int64,
        weight: Int64,
        out: Int64,
        raw: Int64,
        stride: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = SOURCES
        kernel(part, normed, weight, out, raw, stride).launch(
            grid=(160,), block=(512,), stream=stream
        )

    return launch


def prepare(
    h, norm_weight, w_cat, w_up, eps, hc, has_inject, block=None, raw=None, *, pre=None
):
    """Prepare native HC streams, normalization and down projection."""
    rows, width = h.shape
    assert rows in DECODE_ROWS and width == 10240 and hc == 4
    nout = 324 if has_inject else 320
    assert w_cat.shape == (nout, 10240) and w_up.shape == (10240, 320)
    combined = block is not None
    normed = torch.empty_like(h)
    streams = torch.empty_like(h) if combined else h
    part = torch.empty((rows, 4, nout), device=h.device, dtype=torch.float32)
    kernel = pre if pre is not None else _hc_pre_kernel
    bn, warps = (2, 1) if pre is not None and rows == 2 else (8, 4)
    grid = (triton.cdiv(nout, bn), 4, rows)
    placement = {}
    if pre is not None and rows == 4:
        # Quark's dot2 pre supports grouping rows within sixteen N tiles.
        grid = (triton.cdiv(nout, 16 * bn) * 16 * rows, 4, 1)
        placement = {"INTERLEAVE_ROWS": rows}
    kernel[grid](
        h,
        block if combined else h,
        raw if combined else h,
        norm_weight,
        w_cat,
        streams,
        normed,
        part,
        eps,
        h.stride(0),
        block.stride(0) if combined else 0,
        raw.stride(0) if combined else 0,
        H=2560,
        HC=4,
        NOUT=nout,
        COMBINE=combined,
        BN=bn,
        BK=512,
        num_warps=warps,
        **placement,
    )
    return streams, normed, part


def run(
    h,
    norm_weight,
    w_cat,
    w_up,
    eps,
    hc,
    has_inject,
    block=None,
    raw=None,
    *,
    kernel,
    pre=None,
):
    """Retain native combine/norm/down and replace only batch up/mix."""
    streams, normed, part = prepare(
        h, norm_weight, w_cat, w_up, eps, hc, has_inject, block, raw, pre=pre
    )
    rows = h.shape[0]
    mixed = torch.empty((rows, 2560), device=h.device, dtype=h.dtype)
    inject = (
        torch.empty((rows, 4), device=h.device, dtype=h.dtype) if has_inject else None
    )
    kernel(
        part.data_ptr(),
        normed.data_ptr(),
        w_up.data_ptr(),
        mixed.data_ptr(),
        inject.data_ptr() if has_inject else mixed.data_ptr(),
        normed.stride(0),
        stream=torch.cuda.current_stream(),
    )
    return streams, mixed, inject
