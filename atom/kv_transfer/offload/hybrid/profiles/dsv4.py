# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 PAGE+SLOT profile."""

from __future__ import annotations

from math import lcm

from atom.kv_transfer.offload.hybrid.profiles.base import HybridProfile


def build_dsv4_profile(config, *, chunk_size: int) -> HybridProfile:
    """Resolve DSV4 geometry from config without consulting worker tensors."""

    block_size = int(config.kv_cache_block_size)
    dcp_size = int(getattr(config, "decode_context_parallel_size", 1) or 1)
    chunk_size = int(chunk_size)
    if block_size <= 0 or dcp_size <= 0 or chunk_size <= 0:
        raise ValueError("DSV4 block, DCP, and LMCache chunk sizes must be positive")

    hash_block_size = block_size * dcp_size
    if chunk_size % hash_block_size:
        raise ValueError(
            "DSV4 LMCache chunk size must be divisible by the virtual DCP "
            f"block size: chunk={chunk_size}, virtual_block={hash_block_size}"
        )
    resume_alignment = lcm(chunk_size, hash_block_size)

    checkpoint_interval = max(
        0,
        int(getattr(config, "state_checkpoint_interval_tokens", 0) or 0),
    )
    checkpoint_interval -= checkpoint_interval % hash_block_size
    sidecar_interval = (
        lcm(checkpoint_interval, resume_alignment) if checkpoint_interval else 0
    )

    hf_config = getattr(config, "hf_config", None)
    return HybridProfile(
        name="deepseek-v4-page-slot",
        block_size=block_size,
        dcp_size=dcp_size,
        hash_block_size=hash_block_size,
        chunk_size=chunk_size,
        resume_alignment=resume_alignment,
        checkpoint_interval=checkpoint_interval,
        sidecar_interval=sidecar_interval,
        kv_head_dim=int(getattr(hf_config, "kv_head_dim", 512) or 512),
        index_head_dim=int(getattr(hf_config, "index_head_dim", 128) or 128),
    )
