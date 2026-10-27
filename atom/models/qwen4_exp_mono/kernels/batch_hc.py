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
from atom.models.qwen4_exp_mono.sources import SOURCES
from atom.models.qwen4_exp_mono.stages.batch_hc import Smem, emit_up
from atom.models.qwen4_exp_mono.stages.hc import _st_bf
from atom.mono.device.ops import (
    kernel_symbol,
    launcher,
)

_STREAM = fx.Stream(None)


@cache
def build(tokens, has_inject=True):
    name = kernel_symbol("qwen_batch_hc_up", b=tokens, inject=has_inject)

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


def prepare(h, norm_weight, w_cat, w_up, eps, hc, has_inject, block=None, raw=None):
    """Prepare native HC streams, normalization and down projection."""
    rows, width = h.shape
    assert rows in (1, 2, 4) and width == 10240 and hc == 4
    nout = 324 if has_inject else 320
    assert w_cat.shape == (nout, 10240) and w_up.shape == (10240, 320)
    combined = block is not None
    normed = torch.empty_like(h)
    streams = torch.empty_like(h) if combined else h
    part = torch.empty((rows, 4, nout), device=h.device, dtype=torch.float32)
    _hc_pre_kernel[(triton.cdiv(nout, 8), 4, rows)](
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
        BN=8,
        BK=512,
        num_warps=4,
    )
    return streams, normed, part


def run(
    h, norm_weight, w_cat, w_up, eps, hc, has_inject, block=None, raw=None, *, kernel
):
    """Retain native combine/norm/down and replace only batch up/mix."""
    streams, normed, part = prepare(
        h, norm_weight, w_cat, w_up, eps, hc, has_inject, block, raw
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
