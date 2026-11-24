# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Slicing replacement for FSDP2's fused reduce-scatter copy-in.

FSDP2 packs the unsharded gradients of one module group into the reduce-scatter
input with a single ``torch._chunk_cat`` launch (``foreach_reduce_scatter_copy_in``
is its only caller). On ROCm that kernel has been observed to abort the HSA queue
with ``HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION``:

    Kernel Name: at::native::detail::chunk_cat_cuda_kernel<float, c10::BFloat16>
    grid=[19447936, 8, 1], workgroup=[128, 1, 1]

Twice on Qwen3-30B-A3B FSDP2 + vLLM, on different ranks (3, then 7) and different
configurations, always in the backward reduce-scatter copy-in. It is rare rather
than deterministic: both failures landed around 10^5 calls into the run, and the
same kernel at the same shape is bit-exact when driven in isolation, with and
without the caching allocator.

This module sidesteps it. Each gradient is copied chunk by chunk with ordinary
slice assignments -- same bytes, same layout, no fused kernel. It costs
``len(grads) * world_size`` copies per group instead of one launch (88 vs 1 for a
Qwen3-MoE layer at world_size 8), which is a few tens of milliseconds per step
against a ~9 minute step, but each copy is large and contiguous so the bandwidth
is unchanged.

Enable with ``ATOMRL_FSDP_CHUNK_CAT_FALLBACK=1``. Off by default: it is a
workaround for an unexplained fault, not a fix, and if the crash survives it then
ROCm's attribution of the abort to this kernel was wrong and the real culprit is
elsewhere.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_INSTALLED = False


def fallback_enabled() -> bool:
    return os.environ.get("ATOMRL_FSDP_CHUNK_CAT_FALLBACK", "0") == "1"


def reduce_scatter_copy_in(
    unsharded_grads: list[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    """Drop-in for ``foreach_reduce_scatter_copy_in``.

    Reproduces ``chunk_cat(grads, dim=0, num_chunks=world_size, out=...)``: every
    gradient is padded on dim 0 up to a multiple of ``world_size`` with zeros,
    split into ``world_size`` row-blocks, and block ``r`` is appended to row ``r``
    of the output. Verified bit-exact against ``torch._chunk_cat`` on the real
    Qwen3-MoE layer shapes.
    """
    out = reduce_scatter_input.view(world_size, -1)
    offset = 0
    for grad in unsharded_grads:
        rows = grad.shape[0]
        trailing = grad.shape[1:]
        row_numel = grad.numel() // rows if rows else 0
        rows_per_chunk = -(-rows // world_size)  # ceil, i.e. the padded split
        span = rows_per_chunk * row_numel

        for rank in range(world_size):
            start = rank * rows_per_chunk
            end = min(start + rows_per_chunk, rows)
            dst = out[rank].narrow(0, offset, span)
            if end <= start:
                dst.zero_()  # this chunk is entirely padding
                continue
            filled = (end - start) * row_numel
            dst.narrow(0, 0, filled).view(end - start, *trailing).copy_(grad[start:end])
            if filled < span:
                dst.narrow(0, filled, span - filled).zero_()
        offset += span


def install_chunk_cat_fallback() -> bool:
    """Swap FSDP2's copy-in for the slicing version. Idempotent; returns whether active."""
    global _INSTALLED
    if _INSTALLED or not fallback_enabled():
        return _INSTALLED

    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives
    except ImportError as exc:  # pragma: no cover - depends on the torch build
        logger.warning("FSDP2 chunk_cat fallback unavailable: %s", exc)
        return False

    # foreach_reduce looks this up as a module global at call time, so replacing
    # the attribute is enough.
    _fsdp_collectives.foreach_reduce_scatter_copy_in = reduce_scatter_copy_in
    _INSTALLED = True
    print(
        "[atomrl] FSDP2 reduce-scatter copy-in: slicing fallback installed", flush=True
    )
    return True
