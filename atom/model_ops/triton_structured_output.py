# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Apply packed grammar masks without materializing a vocabulary-sized mask."""

import triton
import triton.language as tl


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
    mask_row = tl.program_id(0)
    row = tl.load(rows + mask_row)
    token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(
        masks + mask_row * mask_stride + token // 32,
        mask=token < vocab_size,
        other=-1,
    )
    forbidden = ((word >> (token % 32)) & 1) == 0
    tl.store(
        logits + row * logits_stride + token * token_stride,
        -float("inf"),
        mask=(token < vocab_size) & forbidden,
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
