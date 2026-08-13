# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.atom_page_region_codec import (
    ATOMPageRegionCodec,
    PageCopy,
    build_copy_tiles,
)


def _region(
    base_addr: int,
    unit_bytes: int,
    num_blocks: int,
    *,
    reverse_indexed: bool = False,
) -> KVTransferRegion:
    return KVTransferRegion(
        base_addr=base_addr,
        total_bytes=unit_bytes * num_blocks,
        unit_bytes=unit_bytes,
        reverse_indexed=reverse_indexed,
    )


def _codec(
    region_units: list[int],
    *,
    num_blocks: int = 3,
    device: torch.device | str = "cpu",
) -> ATOMPageRegionCodec:
    regions = [
        _region(1_000 * (index + 1), unit_bytes, num_blocks)
        for index, unit_bytes in enumerate(region_units)
    ]
    return ATOMPageRegionCodec(regions, num_blocks=num_blocks, device=device)


def test_bytes_per_block_is_region_sum():
    codec = _codec([512, 128, 64], num_blocks=8)

    assert codec.bytes_per_block == 704
    assert codec.num_blocks == 8
    assert codec.device == torch.device("cpu")
    assert codec.has_fused_chunk_major_staging is False


def test_unindexed_cuda_device_matches_its_indexed_buffer_device():
    codec = _codec([8])
    codec.device = torch.device("cuda")

    assert codec._matches_device(torch.device("cuda:0"))


def test_page_major_order_is_block_then_region_for_reordered_ids():
    codec = _codec([4, 2], num_blocks=3)

    assert codec.copy_plan([2, 0]) == [
        PageCopy(
            block_id=2,
            region_index=0,
            src_addr=1_008,
            dst_offset=0,
            nbytes=4,
        ),
        PageCopy(
            block_id=2,
            region_index=1,
            src_addr=2_004,
            dst_offset=4,
            nbytes=2,
        ),
        PageCopy(
            block_id=0,
            region_index=0,
            src_addr=1_000,
            dst_offset=6,
            nbytes=4,
        ),
        PageCopy(
            block_id=0,
            region_index=1,
            src_addr=2_000,
            dst_offset=10,
            nbytes=2,
        ),
    ]


def test_connector_gather_call_flattens_groups_and_forwards_stream():
    codec = _codec([4, 2], num_blocks=3)
    calls = []

    class _FakeStaging:
        @staticmethod
        def gather_copy_plan(plan, dst, *, stream=None):
            calls.append((plan, dst, stream))

    codec._fused_kv_staging = _FakeStaging()
    device_buf = torch.empty(3 * codec.bytes_per_block, dtype=torch.uint8)
    stream = object()

    codec.gpu_to_chunk_major_device_buffer(
        device_buf,
        [[2, 0], [1]],
        stream=stream,
    )

    assert calls == [(codec.copy_plan([2, 0, 1]), device_buf, stream)]


@pytest.mark.parametrize("block_id", [-1, 3])
def test_copy_plan_rejects_block_ids_outside_pool(block_id: int):
    codec = _codec([8], num_blocks=3)

    with pytest.raises(ValueError, match="block id"):
        codec.copy_plan([block_id])


def test_codec_rejects_reverse_indexed_page_regions():
    region = _region(1_000, 8, 3, reverse_indexed=True)

    with pytest.raises(ValueError, match="reverse_indexed"):
        ATOMPageRegionCodec([region], num_blocks=3, device="cpu")


def test_codec_rejects_empty_regions():
    with pytest.raises(ValueError, match="at least one"):
        ATOMPageRegionCodec([], num_blocks=3, device="cpu")


@pytest.mark.parametrize("base_addr", [0, -1])
def test_codec_rejects_nonpositive_region_base_addr(base_addr: int):
    region = KVTransferRegion(
        base_addr=base_addr,
        total_bytes=24,
        unit_bytes=8,
    )

    with pytest.raises(ValueError, match="base_addr must be > 0"):
        ATOMPageRegionCodec([region], num_blocks=3, device="cpu")


@pytest.mark.parametrize("total_bytes", [0, -1])
def test_codec_rejects_nonpositive_region_total_bytes(total_bytes: int):
    region = KVTransferRegion(
        base_addr=1_000,
        total_bytes=total_bytes,
        unit_bytes=8,
    )

    with pytest.raises(ValueError, match="total_bytes must be > 0"):
        ATOMPageRegionCodec([region], num_blocks=3, device="cpu")


def test_codec_rejects_region_too_small_for_num_blocks():
    region = KVTransferRegion(
        base_addr=1_000,
        total_bytes=23,
        unit_bytes=8,
    )

    with pytest.raises(ValueError, match="total_bytes.*need 24"):
        ATOMPageRegionCodec([region], num_blocks=3, device="cpu")


def test_codec_snapshots_validated_region_metadata():
    region = _region(1_000, 8, 3)
    codec = ATOMPageRegionCodec([region], num_blocks=3, device="cpu")

    region.base_addr = 9_000
    region.total_bytes = 300
    region.unit_bytes = 100

    assert codec.bytes_per_block == 8
    assert codec.copy_plan([2]) == [
        PageCopy(
            block_id=2,
            region_index=0,
            src_addr=1_016,
            dst_offset=0,
            nbytes=8,
        )
    ]


@pytest.mark.parametrize("num_blocks", [0, -1])
def test_codec_rejects_nonpositive_num_blocks(num_blocks: int):
    with pytest.raises(ValueError, match="num_blocks must be > 0"):
        ATOMPageRegionCodec(
            [_region(1_000, 8, 3)],
            num_blocks=num_blocks,
            device="cpu",
        )


@pytest.mark.parametrize("unit_bytes", [0, -1])
def test_codec_rejects_nonpositive_region_unit_bytes(unit_bytes: int):
    with pytest.raises(ValueError, match="unit_bytes must be > 0"):
        ATOMPageRegionCodec(
            [_region(1_000, unit_bytes, 3)],
            num_blocks=3,
            device="cpu",
        )


def _move(
    codec: ATOMPageRegionCodec,
    direction: str,
    buffer: torch.Tensor,
) -> None:
    if direction == "gather":
        codec.gpu_to_chunk_major_device_buffer([0], buffer)
    else:
        codec.chunk_major_device_buffer_to_gpu(buffer, [0])


@pytest.mark.parametrize("direction", ["gather", "scatter"])
def test_codec_rejects_non_uint8_staging_buffers(direction: str):
    codec = _codec([8], num_blocks=3)
    buffer = torch.empty(codec.bytes_per_block, dtype=torch.float32)

    with pytest.raises(TypeError, match="uint8"):
        _move(codec, direction, buffer)


@pytest.mark.parametrize("direction", ["gather", "scatter"])
def test_codec_rejects_noncontiguous_uint8_staging_buffers(direction: str):
    codec = _codec([8], num_blocks=3)
    buffer = torch.empty(
        (codec.bytes_per_block, 2),
        dtype=torch.uint8,
    )[:, 0]
    assert not buffer.is_contiguous()

    with pytest.raises(ValueError, match="contiguous"):
        _move(codec, direction, buffer)


@pytest.mark.parametrize("direction", ["gather", "scatter"])
def test_codec_rejects_staging_buffers_on_another_device(direction: str):
    codec = _codec([8], num_blocks=3, device="meta")
    buffer = torch.empty(codec.bytes_per_block, dtype=torch.uint8)

    with pytest.raises(TypeError, match="cache device"):
        _move(codec, direction, buffer)


@pytest.mark.parametrize("direction", ["gather", "scatter"])
def test_codec_rejects_undersized_staging_buffers(direction: str):
    codec = _codec([8, 4], num_blocks=3)
    buffer = torch.empty(codec.bytes_per_block - 1, dtype=torch.uint8)

    with pytest.raises(ValueError, match="too small"):
        _move(codec, direction, buffer)


@pytest.mark.parametrize(
    ("direction", "block_ids"),
    [
        ("gather", [1, 1]),
        ("gather", [[1], [1]]),
        ("scatter", [1, 1]),
        ("scatter", [[1], [1]]),
    ],
)
def test_codec_rejects_duplicate_block_ids(direction: str, block_ids):
    codec = _codec([8], num_blocks=3)
    buffer = torch.empty(2 * codec.bytes_per_block, dtype=torch.uint8)

    with pytest.raises(ValueError, match="duplicate block ids"):
        if direction == "gather" and isinstance(block_ids[0], list):
            codec.gpu_to_chunk_major_device_buffer(buffer, block_ids)
        elif direction == "gather":
            codec.gpu_to_chunk_major_device_buffer(block_ids, buffer)
        else:
            codec.chunk_major_device_buffer_to_gpu(buffer, block_ids)


def test_copy_tile_job_count_scales_with_each_copy_size():
    plan = [
        PageCopy(0, 0, 1_000, 0, 1),
        PageCopy(0, 1, 2_000, 1, 1_025),
        PageCopy(1, 0, 4_000, 1_026, 2_048),
    ]

    tiles = build_copy_tiles(plan, tile_bytes=1_024)

    assert len(tiles) == sum((copy.nbytes + 1_023) // 1_024 for copy in plan) == 5
    assert [(tile.src_addr, tile.dst_offset, tile.nbytes) for tile in tiles] == [
        (1_000, 0, 1),
        (2_000, 1, 1_024),
        (3_024, 1_025, 1),
        (4_000, 1_026, 1_024),
        (5_024, 2_050, 1_024),
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="page-region gather/scatter kernels require CUDA/HIP",
)
def test_gpu_heterogeneous_multitile_round_trip_on_nondefault_stream():
    device = torch.device("cuda", torch.cuda.current_device())
    num_blocks = 3
    region0_unit = 1_537
    region1_unit = 257
    region0 = (
        torch.arange(num_blocks * region0_unit, device=device)
        .remainder(251)
        .to(torch.uint8)
        .reshape(num_blocks, region0_unit)
    )
    region1 = (
        torch.arange(num_blocks * region1_unit, device=device)
        .add(101)
        .remainder(251)
        .to(torch.uint8)
        .reshape(num_blocks, region1_unit)
    )
    original0 = region0.clone()
    original1 = region1.clone()
    codec = ATOMPageRegionCodec(
        [
            _region(region0.data_ptr(), region0_unit, num_blocks),
            _region(region1.data_ptr(), region1_unit, num_blocks),
        ],
        num_blocks=num_blocks,
        device=device,
    )
    block_ids = [2, 0]
    staging = torch.empty(
        len(block_ids) * codec.bytes_per_block,
        dtype=torch.uint8,
        device=device,
    )
    stream = torch.cuda.Stream(device=device)
    assert stream != torch.cuda.default_stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))

    codec.gpu_to_chunk_major_device_buffer(block_ids, staging, stream=stream)
    stream.synchronize()

    expected = torch.cat(
        [
            original0[2],
            original1[2],
            original0[0],
            original1[0],
        ]
    )
    assert torch.equal(staging, expected)

    region0.zero_()
    region1.zero_()
    stream.wait_stream(torch.cuda.current_stream(device))
    codec.chunk_major_device_buffer_to_gpu(staging, block_ids, stream=stream)
    stream.synchronize()

    for block_id in block_ids:
        assert torch.equal(region0[block_id], original0[block_id])
        assert torch.equal(region1[block_id], original1[block_id])
    assert torch.count_nonzero(region0[1]) == 0
    assert torch.count_nonzero(region1[1]) == 0
