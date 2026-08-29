# SPDX-License-Identifier: MIT

import torch


def restore_cyclic_row_order(
    gathered: torch.Tensor,
    world_size: int,
    shard_rows: int,
    total_rows: int,
) -> torch.Tensor:
    """Convert rank-major cyclic row shards back to the original row order."""
    return (
        gathered.view(world_size, shard_rows, -1)
        .transpose(0, 1)
        .reshape(world_size * shard_rows, -1)[:total_rows]
    )
