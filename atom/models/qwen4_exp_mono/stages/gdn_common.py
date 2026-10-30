# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared GDN convolution and BF16 boundaries for TP1 and rank-local TP8.

The width-four state update is identical for both geometries. The launch
supplies its rank-local channel stride and keeps projection publication,
recurrence scheduling and output reduction outside this numerical stage.
"""

import flydsl.expr as fx

from atom.models.qwen4_exp_mono.layout import GDR_NORM_HEADS_PER_TASK
from atom.models.qwen4_exp_mono.stages.hc import _ld_bf, _sigmoid, _st_bf
from atom.models.qwen4_exp_mono.stages.memory import raw_tensor
from atom.mono.device.ops import (
    CM_DEV,
    bf16_round,
    hw_exp2,
    hw_rsq,
    row_sum,
    traced,
    wave_sum,
)

LOG2E = 1.4426950408889634
CONV_WIDTH = 4


def _silu(x):
    return fx.Float32(x) / (
        fx.Float32(1.0) + hw_exp2(fx.Float32(x) * fx.Float32(-LOG2E))
    )


def _bf16_round_strict(x):
    """RNE BF16 boundary that remains visible under kernel-wide fast math."""

    bits = fx.Float32(x).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    value = (rounded >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
    return fx.Float32(value)


@traced
def conv_projection_value(
    projected,
    channel,
    conv_weight,
    conv_state,
    read_slot,
    write_slot,
    conv_dim,
):
    """Apply the width-4 depthwise conv and update its three-value state."""

    state_stride = (CONV_WIDTH - 1) * conv_dim
    read_base = read_slot * state_stride + channel
    write_base = write_slot * state_stride + channel
    old0 = _ld_bf(conv_state, read_base)
    old1 = _ld_bf(conv_state, read_base + conv_dim)
    old2 = _ld_bf(conv_state, read_base + 2 * conv_dim)
    # Native causal_conv1d_update rounds each BF16 product before FP32 addition.
    acc = _bf16_round_strict(old0 * _ld_bf(conv_weight, channel * CONV_WIDTH))
    acc = acc + _bf16_round_strict(old1 * _ld_bf(conv_weight, channel * CONV_WIDTH + 1))
    acc = acc + _bf16_round_strict(old2 * _ld_bf(conv_weight, channel * CONV_WIDTH + 2))
    acc = acc + _bf16_round_strict(
        projected * _ld_bf(conv_weight, channel * CONV_WIDTH + 3)
    )
    _st_bf(conv_state, write_base, old1)
    _st_bf(conv_state, write_base + conv_dim, old2)
    _st_bf(conv_state, write_base + 2 * conv_dim, projected)
    return bf16_round(_silu(acc))


def recurrent_views(
    projected, dt_bias, a_log, read_indices, write_indices, state, out, dims
):
    """Zero-copy Q/K/V/A/B and state views for the shared recurrence emitter."""
    bf16_bytes = fx.Int64(2)
    b_row = dims.gdn_conv_dim + dims.gdn_value_dim
    a_row = b_row + dims.gdn_value_heads
    return (
        raw_tensor(projected, fx.BFloat16),
        raw_tensor(projected + bf16_bytes * fx.Int64(dims.gdn_key_dim), fx.BFloat16),
        raw_tensor(
            projected + bf16_bytes * fx.Int64(2 * dims.gdn_key_dim), fx.BFloat16
        ),
        raw_tensor(projected + bf16_bytes * fx.Int64(a_row), fx.BFloat16),
        raw_tensor(projected + bf16_bytes * fx.Int64(b_row), fx.BFloat16),
        raw_tensor(dt_bias, fx.BFloat16),
        raw_tensor(a_log, fx.BFloat16),
        raw_tensor(read_indices, fx.Int32, 4),
        raw_tensor(write_indices, fx.Int32, 4),
        raw_tensor(state, fx.Float32),
        raw_tensor(out, fx.BFloat16),
    )


def norm_head_values(x0, x1, weight, lane):
    """Preserve the head reduction and BF16 normalization/affine boundaries."""
    total = wave_sum(x0 * x0 + x1 * x1)
    rrms = hw_rsq(total * fx.Float32(1.0 / 128) + 1e-6)
    normalized0 = _bf16_round_strict(x0 * rrms)
    normalized1 = _bf16_round_strict(x1 * rrms)
    affine0 = _bf16_round_strict(normalized0 * _ld_bf(weight, lane))
    affine1 = _bf16_round_strict(normalized1 * _ld_bf(weight, lane + 64))
    return affine0, affine1


@traced
def norm_head_pair(core, weight, head, lane):
    """Normalize two lanes of one GDN head with native BF16 boundaries."""
    index0 = head * 128 + lane
    index1 = index0 + 64
    x0 = _ld_bf(core, index0)
    x1 = _ld_bf(core, index1)
    affine0, affine1 = norm_head_values(x0, x1, weight, lane)
    return index0, index1, affine0, affine1


@traced
def projection_conv_store(
    reduction,
    task,
    tid,
    out,
    conv_weight,
    conv_state,
    read_slot,
    write_slot,
    conv_dim,
):
    """Shared convolutional projection prefix; the caller owns tail publication."""
    if tid < 16:
        channel = task * 16 + tid
        projected = bf16_round(row_sum(reduction, tid, 0))
        # Graph padding uses negative state slots and must not touch the pool.
        value = fx.Float32(0.0)
        if (read_slot >= 0) & (write_slot >= 0):
            value = conv_projection_value(
                projected,
                channel,
                conv_weight,
                conv_state,
                read_slot,
                write_slot,
                conv_dim,
            )
        _st_bf(out, channel, value, CM_DEV)


@traced
def sigmoid_norm_task(core, gate, weight, normed, task, lane, wave):
    head = task * GDR_NORM_HEADS_PER_TASK + wave
    index0, index1, affine0, affine1 = norm_head_pair(core, weight, head, lane)
    result0 = affine0 * _sigmoid(_ld_bf(gate, index0))
    result1 = affine1 * _sigmoid(_ld_bf(gate, index1))
    _st_bf(normed, index0, result0, CM_DEV)
    _st_bf(normed, index1, result1, CM_DEV)
