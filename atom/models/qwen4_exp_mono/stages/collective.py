# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Model-specific peer publication and ordered pair reduction primitives."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr

from atom.models.qwen4_exp_mono.peer_layout import PART_COLS, SLOTS
from atom.models.qwen4_exp_mono.shapes import HIDDEN
from atom.mono.device.ops import CM_DEV, bf2_f32, rsrc, traced
from atom.mono.device.ranks import peer_bases
from atom.mono.device.sync import preg


def _pair(layer, tokens, token, col):
    return (((layer & (SLOTS - 1)) * tokens + token) * HIDDEN + col) // 2


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
