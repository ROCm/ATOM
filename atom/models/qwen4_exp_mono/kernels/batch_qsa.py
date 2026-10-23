# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Emit causal QSA indices when every visible group fits the index budget."""

import torch
import triton
import triton.language as tl


@triton.jit
def _all_visible_indices(
    positions,
    requests,
    lengths,
    out,
    num_requests: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + token)
    valid = (request >= 0) & (request < num_requests)
    length = tl.load(lengths + request, valid, 0)
    position = tl.load(positions + token)
    visible = tl.where(valid & (position >= 0) & (position < length), position + 1, 0)
    tl.store(
        out + token * width + col,
        tl.where(col < visible, col, -1).to(tl.int32),
        col < width,
    )


def select_all(metadata, rows, budget, ratio):
    reachable = (metadata.max_seq_len + ratio - 1) // ratio
    bucket = 1 << (max(reachable, 1) - 1).bit_length()
    # Match the native selection's graph-constant width and incomplete tail.
    width = min(bucket, budget // ratio) * ratio + ratio - 1
    selected = torch.empty(
        (rows, width), device=metadata.seq_lens.device, dtype=torch.int32
    )
    _all_visible_indices[(rows, triton.cdiv(width, 256))](
        metadata.logical_positions,
        metadata.token_to_req,
        metadata.seq_lens,
        selected,
        metadata.seq_lens.numel(),
        width,
        256,
        num_warps=4,
    )
    return selected
