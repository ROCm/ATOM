# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The top-16 block ids of decode score rows in one launch, query and window lengths at runtime."""

import functools

import torch
from amd_kernels.warm_up import launcher

from .prefill_kernel import BLK, TOPK, compile_topk_prefill
from .top16_indices_p128_wave4 import compile_top16_indices_wave4
from ..host_stream import current_stream as _current_stream


@functools.lru_cache(maxsize=3)
@launcher
def _get_decode_indices_kernel(rank_owned_order: bool = False, large_rows: bool = False):
    if rank_owned_order and not large_rows:
        return compile_top16_indices_wave4()
    return compile_topk_prefill(26, decode=True, rank_owned_order=rank_owned_order)


def index_topk_decode_indices(
    score: torch.Tensor,
    topk_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    max_seq_len: int,
    query_len: int,
    block_size: int = 128,
    rank_owned_order: bool = False,
) -> None:
    """The top-16 block ids of each decode score row in one launch; ``rank_owned_order``: ``score`` rows in the
    rank-owned scorer's block order."""
    from .host import MAX_SLOTS, _pow2_ceil, _run_compiled

    heads, rows, width = score.shape
    assert score.dtype == torch.float32 and score.stride(2) == 1
    assert topk_idx.dtype == torch.int32 and topk_idx.stride(2) == 1
    assert topk_idx.shape == (heads, rows, TOPK)
    assert block_size == BLK and query_len > 0 and rows % query_len == 0
    assert seq_lens.dtype == torch.int32 and seq_lens.is_contiguous()
    assert seq_lens.numel() >= rows // query_len
    slots = _pow2_ceil((max_seq_len + BLK * 64 - 1) // (BLK * 64))
    assert slots <= MAX_SLOTS and width >= slots * 64
    if rows == 0:
        return
    _run_compiled(
        _get_decode_indices_kernel(rank_owned_order, rank_owned_order and rows > 512),
        score.data_ptr(),
        score.stride(0),
        score.stride(1),
        slots * 64,
        seq_lens.data_ptr(),
        query_len,
        topk_idx.data_ptr(),
        topk_idx.stride(0),
        topk_idx.stride(1),
        rows,
        heads,
        _current_stream(),
    )
