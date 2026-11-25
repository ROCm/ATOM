# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels behind ``gather.gather_units``; only GPU callers import them."""

import triton
import triton.language as tl


@triton.jit
def gather_unit_words(
    src,
    indices,
    dst,
    src_offsets,
    src_words,
    dst_words,
    BLOCK: tl.constexpr,
):
    """Program ``(u, b)`` fills block ``b`` of ``BLOCK`` words of output unit ``u``,
    word ``w`` from word ``src_offsets[w]`` of source unit ``indices[u]``."""
    unit = tl.program_id(0)
    word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = word < dst_words
    block_id = tl.load(indices + unit).to(tl.int64)
    offset = tl.load(src_offsets + word, mask=mask)
    value = tl.load(src + block_id * src_words + offset, mask=mask)
    tl.store(dst + unit * dst_words + word, value, mask=mask)
