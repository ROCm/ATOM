# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Shared utilities for KV cache disaggregation backends (MoRIIO, Mooncake, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import triton
import triton.language as tl

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("atom")

MAX_RDMA_CHUNK_BYTES = 2 * 1024 * 1024 * 1024 - 64 * 1024  # just under 2 GiB


# ---------------------------------------------------------------------------
# GPU memory fence for RDMA-written KV cache blocks
#
# Not needed when producer and consumer are in the same network partition
# (same-partition RDMA completes before decode starts).
# Enable for cross-partition deployments where higher RDMA latency
# can cause stale GPU reads (improves accuracy from ~0.85 to ~0.90).
# ---------------------------------------------------------------------------


@triton.jit
def _gpu_fence_kernel(
    kv_ptr,
    block_ids_ptr,
    total_blocks: tl.int64,
    elems_per_block: tl.int64,
    NUM_CL: tl.constexpr,
):
    blk_idx = tl.program_id(0)
    layer_idx = tl.program_id(1).to(tl.int64)
    blk = tl.load(block_ids_ptr + blk_idx).to(tl.int64)
    base = (layer_idx * total_blocks + blk) * elems_per_block
    cl_offsets = tl.arange(0, NUM_CL).to(tl.int64) * 32
    mask = cl_offsets < elems_per_block
    tl.atomic_or(
        kv_ptr + base + cl_offsets, tl.zeros([NUM_CL], dtype=tl.int32), mask=mask
    )


def gpu_memory_fence(
    kv_cache: "torch.Tensor",
    block_ids: list[int],
    use_mla: bool,
) -> None:
    """Force GPU memory coherence for RDMA-written KV cache blocks.

    Single Triton kernel launch -- one atomic_or(0) per cache line (128 bytes)
    across all (layer, block) pairs.

    Args:
        kv_cache: The full KV cache tensor.
        block_ids: Physical block IDs that received RDMA writes.
        use_mla: Whether the model uses Multi-head Latent Attention.
    """
    import torch

    block_ids_t = torch.tensor(block_ids, dtype=torch.int32, device=kv_cache.device)
    kv_flat = kv_cache.view(torch.int32)
    if use_mla:
        num_layers = kv_cache.shape[0]
        total_blocks = kv_cache.shape[1]
    else:
        num_layers = 2 * kv_cache.shape[1]
        total_blocks = kv_cache.shape[2]
    elems_per_block = kv_flat.numel() // (num_layers * total_blocks)
    num_cl = triton.cdiv(elems_per_block, 32)
    NUM_CL = triton.next_power_of_2(num_cl)
    _gpu_fence_kernel[(len(block_ids), num_layers)](
        kv_flat,
        block_ids_t,
        total_blocks,
        elems_per_block,
        NUM_CL=NUM_CL,
    )


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
