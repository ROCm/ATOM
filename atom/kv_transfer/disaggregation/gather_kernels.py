# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernels behind ``gather.gather_units``; only GPU callers import them."""

import triton
import triton.language as tl


@triton.jit
def gather_unit_pieces(
    src,
    indices,
    dst,
    pieces,
    src_words,
    dst_words,
    BLOCK: tl.constexpr,
):
    """Program ``(u, p)`` copies piece ``p`` of unit ``indices[u]``: ``pieces[p]``
    is ``(src word, dst word, words)``, at most ``BLOCK`` words."""
    unit = tl.program_id(0)
    piece = pieces + tl.program_id(1) * 3
    block_id = tl.load(indices + unit).to(tl.int64)
    src_word = tl.load(piece)
    dst_word = tl.load(piece + 1)
    word = tl.arange(0, BLOCK)
    mask = word < tl.load(piece + 2)
    value = tl.load(src + block_id * src_words + src_word + word, mask=mask)
    tl.store(dst + unit * dst_words + dst_word + word, value, mask=mask)
