# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL emitters and launchers for Qwen's deferred TP8 hand-off."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from atom.models.qwen4_exp_mono.config import HIDDEN
from atom.models.qwen4_exp_mono.peer_layout import PART_COLS, PARTS, SLOTS, peer_offsets
from atom.mono.device.ops import (
    CM_DEV,
    bf2_f32,
    kernel_symbol,
    launcher,
    rsrc,
    traced,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.sync import Mailbox, preg
from atom.mono.plan.execution import BLOCKS, THREADS
from atom.mono.runtime.abi import KernelAbi

_CURRENT_STREAM = fx.Stream(None)

PULL_ABI = KernelAbi(("peers", "output", "layer"))


def _pair(layer, tokens, token, col):
    return (((layer & (SLOTS - 1)) * tokens + token) * HIDDEN + col) // 2


def _ld_bf(ptr, index):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), index, vec_width=1, dtype=T.bf16))
    )


@traced
def publish_pair(mailbox, sym, region_off, region, layer, tokens, token, col, v0, v1):
    """Publish one BF16 column pair into this source rank's symmetric buffer."""

    dst = preg(sym, region_off, region)
    mailbox.put_bf(dst, 2 * _pair(layer, tokens, token, col), [v0, v1])


@traced
def pull_sum_pair(mailbox, peers, region_off, region, layer, tokens, token, col, tp):
    """Poll every source and return its fixed rank-order FP32 pair sum."""

    bases = peer_bases(peers, tp)
    pair = _pair(layer, tokens, token, col)
    words = mailbox.poll(
        [(preg(bases[src], region_off, region), pair, 1) for src in range(tp)]
    )
    acc0, acc1 = bf2_f32(words[0][0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(words[src][0])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1


@traced
def publish_bf16_part(
    mailbox,
    partial,
    sym,
    region_off,
    region,
    layer,
    tokens,
    token,
    part,
):
    """One CTA publishes an N32 part from a row-major BF16 partial."""

    tid = fx.thread_idx.x
    if tid < PART_COLS // 2:
        col = part * PART_COLS + 2 * tid
        row = token * HIDDEN
        publish_pair(
            mailbox,
            sym,
            region_off,
            region,
            layer,
            tokens,
            token,
            col,
            _ld_bf(partial, row + col).to(fx.BFloat16),
            _ld_bf(partial, row + col + 1).to(fx.BFloat16),
        )


@traced
def pull_part_to_bf16(
    mailbox,
    peers,
    output,
    region_off,
    region,
    layer,
    tokens,
    token,
    part,
    tp,
):
    """One CTA consumes an N32 part with one BF16 endpoint rounding."""

    tid = fx.thread_idx.x
    if tid < PART_COLS // 2:
        col = part * PART_COLS + 2 * tid
        acc0, acc1 = pull_sum_pair(
            mailbox,
            peers,
            region_off,
            region,
            layer,
            tokens,
            token,
            col,
            tp,
        )
        row = token * HIDDEN
        bo.buffer_store(
            acc0.to(fx.BFloat16),
            rsrc(output),
            row + col,
            cache_modifier=CM_DEV,
        )
        bo.buffer_store(
            acc1.to(fx.BFloat16),
            rsrc(output),
            row + col + 1,
            cache_modifier=CM_DEV,
        )


@functools.cache
def build_pull(tokens: int, tp: int):
    """Build the S-width K1-side peer consumer."""

    if tokens not in (1, 2, 4, 8) or tp != 8:
        raise ValueError("Qwen deferred pull supports S=1/S=2/S=4/S=8 TP8")
    region_off = peer_offsets(tokens, tp)["moe"]
    name = kernel_symbol("qwen_tp_pull", s=tokens, tp=tp)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(peers: Int64, output: Int64, layer: Int32):
        task = fx.block_idx.x
        token = task // PARTS
        part = task % PARTS
        if task < tokens * PARTS:
            pull_part_to_bf16(
                Mailbox(layer),
                peers,
                output,
                region_off,
                "moe",
                layer,
                tokens,
                token,
                part,
                tp,
            )

    @launcher(name)
    def launch(
        peers: Int64,
        output: Int64,
        layer: Int32,
        stream: fx.Stream = _CURRENT_STREAM,
    ):
        kernel(peers, output, layer).launch(
            grid=(max(BLOCKS, PARTS * tokens),), block=(THREADS,), stream=stream
        )

    PULL_ABI.check(kernel, launch)
    return launch
