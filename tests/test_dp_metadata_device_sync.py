# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from atom.utils.forward_context import ForwardMode
from atom.utils.tbo.ubatching import (
    DP_METADATA_MAX_FIELDS,
    DPMetadataBuffers,
    begin_sync_dp_metadata,
    finish_sync_dp_metadata,
    sync_dp_metadata,
)


def test_device_sync_submission_defers_host_wait(monkeypatch):
    """The device path must not synchronize until its result is consumed."""

    class FakeStream:
        def __init__(self):
            self.synchronize_calls = 0

        def synchronize(self):
            self.synchronize_calls += 1

    stream = FakeStream()
    local_cpu = torch.empty(DP_METADATA_MAX_FIELDS, dtype=torch.int32)
    gathered_cpu = torch.empty(2 * DP_METADATA_MAX_FIELDS, dtype=torch.int32)
    buffers = SimpleNamespace(
        dp_size=2,
        local_cpu=local_cpu,
        local_device=torch.empty_like(local_cpu),
        gathered_device=torch.empty_like(gathered_cpu),
        gathered_cpu=gathered_cpu,
        local_numpy=local_cpu.numpy(),
        stream=stream,
    )

    monkeypatch.setattr(torch.cuda, "stream", lambda unused: nullcontext())

    def fake_all_gather_into_tensor(output, local, group=None):
        del group
        peer = local.clone()
        peer[:] = torch.tensor([17, 3, 1, 1, 1, 8, 9, 5])
        output.copy_(torch.cat((local, peer)))

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", fake_all_gather_into_tensor
    )

    pending = begin_sync_dp_metadata(
        dp_group=object(),
        dp_size=2,
        buffers=buffers,
        scheduled_tokens=11,
        scheduled_bs=2,
        is_prefill=True,
        tbo_on=True,
        local_meets_min_tokens=False,
        local_can_split=True,
        local_ub_tokens=(5, 6),
        max_seqlen_q=4,
    )

    assert stream.synchronize_calls == 0
    assert buffers.local_device.tolist() == [11, 2, 1, 0, 1, 5, 6, 4]

    result = finish_sync_dp_metadata(pending)
    assert stream.synchronize_calls == 1
    assert result.num_tokens_across_dp.tolist() == [11, 17]
    assert result.max_bs_across_dp == 3
    assert result.any_rank_has_prefill
    assert result.tbo_collective_active
    assert result.ub_max_tokens_across_dp == (8, 9)
    assert result.max_seqlen_q_across_dp == 5

    # Finishing twice is harmless and never adds a second host wait.
    assert finish_sync_dp_metadata(pending) is result
    assert stream.synchronize_calls == 1


def _device_sync_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        buffers = DPMetadataBuffers.allocate(world_size, torch.device("cuda", rank))
        buffer_ptrs = {
            "local_cpu": buffers.local_cpu.data_ptr(),
            "local_device": buffers.local_device.data_ptr(),
            "gathered_device": buffers.gathered_device.data_ptr(),
            "gathered_cpu": buffers.gathered_cpu.data_ptr(),
        }
        result = sync_dp_metadata(
            dp_group=dist.group.WORLD,
            dp_size=world_size,
            buffers=buffers,
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

        assert buffers.local_cpu.is_pinned()
        assert buffers.gathered_cpu.is_pinned()
        assert buffers.stream != torch.cuda.current_stream(rank)
        assert buffers.local_device[:DP_METADATA_MAX_FIELDS].cpu().tolist() == [
            11 + 6 * rank,
            2 + rank,
            1,
            1 if rank == 1 else 0,
            1,
            5 + 3 * rank,
            6 + 3 * rank,
            4 + rank,
        ]
        assert buffers.gathered_device.view(
            world_size, DP_METADATA_MAX_FIELDS
        ).cpu().tolist() == [
            [11, 2, 1, 0, 1, 5, 6, 4],
            [17, 3, 1, 1, 1, 8, 9, 5],
        ]

        # The same max-sized allocations also back the smaller no-TBO/no-DSpark
        # wire format. Exercise the production ForwardMode forwarding path too.
        small_mode = ForwardMode.decide(
            batch=SimpleNamespace(
                total_tokens_num_prefill=0,
                total_tokens_num=4 + rank,
                total_seqs_num=4 + rank,
            ),
            dp_group=dist.group.WORLD,
            dp_size=world_size,
            dp_sync_buffers=buffers,
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
        assert {
            "local_cpu": buffers.local_cpu.data_ptr(),
            "local_device": buffers.local_device.data_ptr(),
            "gathered_device": buffers.gathered_device.data_ptr(),
            "gathered_cpu": buffers.gathered_cpu.data_ptr(),
        } == buffer_ptrs
        assert buffers.gathered_device[: world_size * 3].cpu().tolist() == [
            4,
            4,
            0,
            5,
            5,
            0,
        ]
        assert small.num_tokens_across_dp.tolist() == [4, 5]
        assert small.max_bs_across_dp == 5
        # The returned rank vector owns its storage; reusing the pinned D2H
        # landing buffer for this second call must not mutate the first result.
        assert result.num_tokens_across_dp.tolist() == [11, 17]
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
