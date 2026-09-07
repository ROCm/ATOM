# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from atom.model_ops.v4_indexer_utils import (
    cyclic_row_indices,
    restore_cyclic_row_order,
)


def test_restore_cyclic_row_order_trims_tail_padding():
    # Four rank-major shards for global rows [0..9]. The last two slots are
    # collective padding and must disappear after cyclic interleaving.
    gathered = torch.tensor(
        [
            [0, 100],
            [4, 104],
            [8, 108],
            [1, 101],
            [5, 105],
            [9, 109],
            [2, 102],
            [6, 106],
            [-1, -1],
            [3, 103],
            [7, 107],
            [-1, -1],
        ],
        dtype=torch.int32,
    )

    restored = restore_cyclic_row_order(
        gathered, world_size=4, shard_rows=3, total_rows=10
    )

    assert restored.tolist() == [[i, i + 100] for i in range(10)]


@pytest.mark.parametrize(
    ("total_rows", "world_size"),
    [
        (0, 4),
        (1, 4),
        (3, 4),
        (4, 4),
        (10, 4),
        (11, 3),
        (12, 3),
    ],
)
def test_cyclic_row_sharding_round_trip(total_rows, world_size):
    source = torch.arange(total_rows * 2, dtype=torch.int32).reshape(total_rows, 2)
    shard_rows = (total_rows + world_size - 1) // world_size
    shards = []

    for rank in range(world_size):
        row_indices = cyclic_row_indices(total_rows, world_size, rank)
        shard = source[row_indices]
        padded = torch.full((shard_rows, 2), -1, dtype=source.dtype)
        padded[: shard.shape[0]].copy_(shard)
        shards.append(padded)

    gathered = torch.cat(shards, dim=0)
    restored = restore_cyclic_row_order(
        gathered,
        world_size=world_size,
        shard_rows=shard_rows,
        total_rows=total_rows,
    )

    assert torch.equal(restored, source)
