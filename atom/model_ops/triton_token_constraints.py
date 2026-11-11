# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared token-ID masking primitives for dense sets and GPU state machines."""

import triton
import triton.language as tl


@triton.jit
def mask_token_columns(row_ptr, columns, forbidden, valid, stride=1):
    """Deny selected columns; allow/force masks supply their complement."""
    tl.store(row_ptr + columns * stride, -float("inf"), mask=valid & forbidden)


@triton.jit
def _mask_logits(
    logits,
    masks,
    rows,
    logits_stride,
    token_stride,
    mask_stride,
    vocab_size,
    BLOCK: tl.constexpr,
):
    mask_row = tl.program_id(0).to(tl.int64)
    row = tl.load(rows + mask_row).to(tl.int64)
    token = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(
        masks + mask_row * mask_stride + token // 32,
        mask=token < vocab_size,
        other=-1,
    )
    forbidden = ((word >> (token % 32)) & 1) == 0
    mask_token_columns(
        logits + row * logits_stride,
        token,
        forbidden,
        token < vocab_size,
        token_stride,
    )


def apply_packed_bitmask(logits, masks, rows):
    if rows.numel() == 0:
        return
    block = 1024
    _mask_logits[(rows.numel(), triton.cdiv(logits.shape[1], block))](
        logits,
        masks,
        rows,
        logits.stride(0),
        logits.stride(1),
        masks.stride(0),
        logits.shape[1],
        BLOCK=block,
    )
