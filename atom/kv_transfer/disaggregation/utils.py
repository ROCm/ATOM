# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Shared utilities for KV cache disaggregation backends (MoRIIO, Mooncake, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

MAX_RDMA_CHUNK_BYTES = 2 * 1024 * 1024 * 1024 - 64 * 1024  # just under 2 GiB


def chunk_tensor_for_rdma(
    tensor: torch.Tensor, block_size_in_dim0: int = 1
) -> tuple[list[tuple[int, int]], int]:
    """Split a tensor into <2 GiB RDMA-registrable chunks along dim 0.

    Args:
        tensor: contiguous torch.Tensor whose dim-0 is the block (or
            token) axis.
        block_size_in_dim0: elements per logical block in dim 0.
            Non-MLA: 1 (dim 0 = num_blocks).
            MLA: block_size (dim 0 = num_blocks * block_size).

    Returns:
        ``(chunks, blocks_per_chunk)`` where *chunks* is a list of
        ``(data_ptr, size_bytes)`` pairs and *blocks_per_chunk* is
        the number of logical blocks in each full chunk.
    """
    elem_sz = tensor.element_size()
    per_block_bytes = block_size_in_dim0 * tensor.stride(0) * elem_sz
    total_blocks = tensor.shape[0] // block_size_in_dim0
    bpc = max(1, MAX_RDMA_CHUNK_BYTES // per_block_bytes)
    chunks: list[tuple[int, int]] = []
    base = tensor.data_ptr()
    for start in range(0, total_blocks, bpc):
        end = min(start + bpc, total_blocks)
        chunks.append((base + start * per_block_bytes, (end - start) * per_block_bytes))
    return chunks, bpc
