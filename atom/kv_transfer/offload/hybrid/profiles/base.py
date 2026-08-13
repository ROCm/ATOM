# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-neutral geometry used by hybrid offload schedulers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HybridProfile:
    """Resolved token grids and cache dimensions for one hybrid layout."""

    name: str
    block_size: int
    dcp_size: int
    hash_block_size: int
    chunk_size: int
    resume_alignment: int
    checkpoint_interval: int
    sidecar_interval: int
    kv_head_dim: int
    index_head_dim: int
