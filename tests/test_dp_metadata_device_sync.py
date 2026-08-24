# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from atom.utils.forward_context import ForwardMode
from atom.utils.tbo.ubatching import DP_METADATA_MAX_FIELDS, sync_dp_metadata


def _device_sync_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        gathered_buffer = torch.empty(
            world_size * DP_METADATA_MAX_FIELDS,
            dtype=torch.int32,
            device=torch.device("cuda", rank),
        )
        buffer_ptr = gathered_buffer.data_ptr()
        result = sync_dp_metadata(
            dp_group=dist.group.WORLD,
            dp_size=world_size,
            device=torch.device("cuda", rank),
            gathered_buffer=gathered_buffer,
            scheduled_tokens=11 + 6 * rank,
            scheduled_bs=2 + rank,
            is_prefill=True,
            tbo_on=True,
            local_meets_min_tokens=rank == 1,
            local_can_split=True,
            local_ub_tokens=(5 + 3 * rank, 6 + 3 * rank),
            max_seqlen_q=4 + rank,
        )

        assert result.num_tokens_across_dp.device.type == "cpu"
        assert result.num_tokens_across_dp.tolist() == [11, 17]
        assert result.max_bs_across_dp == 3
        assert result.any_rank_has_prefill is True
        assert result.tbo_collective_active is True
        assert result.ub_max_tokens_across_dp == (8, 9)
        assert result.max_seqlen_q_across_dp == 5

        assert gathered_buffer.data_ptr() == buffer_ptr
        assert gathered_buffer.view(
            world_size, DP_METADATA_MAX_FIELDS
        ).cpu().tolist() == [
            [11, 2, 1, 0, 1, 5, 6, 4],
            [17, 3, 1, 1, 1, 8, 9, 5],
        ]

        # The same max-sized allocation also backs the smaller no-TBO/no-DSpark
        # wire format. Exercise the production ForwardMode forwarding path too.
        small_mode = ForwardMode.decide(
            batch=SimpleNamespace(
                total_tokens_num_prefill=0,
                total_tokens_num=4 + rank,
                total_seqs_num=4 + rank,
            ),
            dp_group=dist.group.WORLD,
            dp_size=world_size,
            dp_sync_device=torch.device("cuda", rank),
            dp_sync_buffer=gathered_buffer,
            enforce_eager=False,
            capture_sizes=[1, 2, 4, 8],
            captured_tokens=None,
            is_block_drafter=False,
            tbo_on=False,
            local_tbo=(False, False, 0, 0),
            max_seqlen_q=1,
        )
        small = small_mode.sync
        assert small is not None
        assert gathered_buffer.data_ptr() == buffer_ptr
        assert gathered_buffer[: world_size * 3].cpu().tolist() == [
            4,
            4,
            0,
            5,
            5,
            0,
        ]
        assert small.num_tokens_across_dp.tolist() == [4, 5]
        assert small.max_bs_across_dp == 5
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two GPUs and NCCL/RCCL",
)
def test_device_sync_gathers_rank_metadata(tmp_path):
    mp.spawn(
        _device_sync_worker,
        args=(2, str(tmp_path / "device-sync-store")),
        nprocs=2,
        join=True,
    )
