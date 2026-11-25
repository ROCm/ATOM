# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels behind ``gather.gather_units``; only GPU callers import them."""

import triton
import triton.language as tl


@triton.jit
def gather_strided_units(
    src,
    indices,
    dst,
    SRC_WORDS: tl.constexpr,
    DST_WORDS: tl.constexpr,
    FIRST: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    unit = tl.program_id(0)
    word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    block_id = tl.load(indices + unit).to(tl.int64)
    src_offset = block_id * SRC_WORDS + FIRST + word // WIDTH * STRIDE + word % WIDTH
    value = tl.load(src + src_offset, mask=word < DST_WORDS)
    tl.store(dst + unit * DST_WORDS + word, value, mask=word < DST_WORDS)
