# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RCCL integration test for V4 cyclic row-shard gather and restoration."""

import os

import pytest
import torch
import torch.distributed as dist

from atom.model_ops.v4_indexer_utils import (
    cyclic_row_indices,
    restore_cyclic_row_order,
)


@pytest.mark.skipif(
    "LOCAL_RANK" not in os.environ,
    reason="run with torchrun on at least four ROCm GPUs",
)
def test_cyclic_row_shard_rccl_tp4_round_trip():
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 4
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    try:
        for total_rows in (1, 3, 4, 5, 10, 11, 12):
            source = torch.arange(
                total_rows * 2, dtype=torch.int32, device=device
            ).reshape(total_rows, 2)
            row_indices = cyclic_row_indices(
                total_rows, world_size, local_rank, device=device
            )
            local = source[row_indices]
            shard_rows = (total_rows + world_size - 1) // world_size
            padded = torch.full((shard_rows, 2), -1, dtype=torch.int32, device=device)
            padded[: local.shape[0]].copy_(local)

            gathered = torch.empty(
                (world_size * shard_rows, 2), dtype=torch.int32, device=device
            )
            dist.all_gather_into_tensor(gathered, padded.contiguous())
            restored = restore_cyclic_row_order(
                gathered,
                world_size=world_size,
                shard_rows=shard_rows,
                total_rows=total_rows,
            )
            assert torch.equal(restored, source)
    finally:
        dist.destroy_process_group()
