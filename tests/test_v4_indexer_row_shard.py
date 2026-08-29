# SPDX-License-Identifier: MIT

import torch

from atom.models.deepseek_v4 import _restore_cyclic_row_order


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

    restored = _restore_cyclic_row_order(
        gathered, world_size=4, shard_rows=3, total_rows=10
    )

    assert restored.tolist() == [[i, i + 100] for i in range(10)]
