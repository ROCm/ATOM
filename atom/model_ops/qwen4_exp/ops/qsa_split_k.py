# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Split-K partials of QSA sparse GQA and their reduce.

The portable kernel (`qsa.py`) and the gfx1250 one (`qsa_gfx1250.py`) write
the same partial layout as aiter's MLA decode: per (token, head, split) a
base-2 running max and sum, an unnormalized FP32 output row, and one valid
width per token. Both hand it to `_mla_decode_fwd_reduce_kernel`.
"""

import torch
from aiter.ops.triton._triton_kernels.attention.mla import (
    _mla_decode_fwd_reduce_kernel,
)


def split_k_buffers(out: torch.Tensor, kv_splits: int):
    """`(target, partial_max, partial_sum, widths)` for a launch into `out`.

    One split writes the normalized result straight into `out`, so the
    partial arguments alias it and `widths` is None.
    """
    if kv_splits == 1:
        return out, out, out, None
    tokens, heads, head_dim = out.shape
    shape = (tokens, heads, kv_splits)
    partial_max = torch.empty(shape, device=out.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    target = torch.empty((*shape, head_dim), device=out.device, dtype=torch.float32)
    widths = torch.empty(tokens, dtype=torch.int32, device=out.device)
    return target, partial_max, partial_sum, widths


def reduce_split_k(
    out: torch.Tensor,
    target: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    widths: torch.Tensor | None,
    kv_splits: int,
    tile_size: int,
) -> None:
    """Merge the partials of `split_k_buffers` into `out`; nothing for one split."""
    if kv_splits == 1:
        return
    tokens, heads, head_dim = out.shape
    _mla_decode_fwd_reduce_kernel[(tokens, heads)](
        out,
        target,
        partial_max,
        partial_sum,
        widths,
        None,
        tokens,
        heads,
        out.stride(0),
        out.stride(1),
        1,
        1,
        tokens,
        TILE_SIZE=tile_size,
        KV_LORA_RANK=head_dim,
        query_start_len_ptr=None,
        BLOCK_Q=1,
        NUM_SEGMENTS_PER_SEQ=kv_splits,
        ALL_DECODE=True,
    )
