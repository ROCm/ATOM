# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch


def cyclic_row_indices(
    total_rows: int,
    world_size: int,
    rank: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the global rows owned by ``rank`` under cyclic TP sharding."""
    if rank >= total_rows:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.arange(rank, total_rows, world_size, dtype=torch.int64, device=device)


def restore_cyclic_row_order(
    gathered: torch.Tensor,
    world_size: int,
    shard_rows: int,
    total_rows: int,
) -> torch.Tensor:
    """Convert rank-major cyclic row shards back to the original row order."""
    return (
        gathered.view(world_size, shard_rows, gathered.shape[-1])
        .transpose(0, 1)
        .reshape(world_size * shard_rows, gathered.shape[-1])[:total_rows]
    )
