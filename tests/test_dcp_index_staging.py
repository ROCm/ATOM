# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import replace

import pytest
import torch

from atom.kv_transfer.disaggregation.index_staging import (
    gather_dcp_mla_pages,
    gather_dcp_preshuffled_index_pages,
    prepare_dcp_index_gather_indices,
)
from atom.kv_transfer.disaggregation.sharded_transfer import build_dcp_shard_plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
@pytest.mark.parametrize("width", [48, 576])
def test_mla_gather_preserves_sharded_bytes_and_partial_page(dcp_size, dtype, width):
    pytest.importorskip("triton", reason="MLA gather is a Triton kernel")
    device = torch.device("cuda")
    block_size = 16
    # Permuted source blocks and a partial final page exercise physical block
    # addressing, every DCP rank, and padding without depending on the planner.
    src_ids = [4, 0, 6, 2, 5]
    raw = torch.arange(7 * block_size * width, dtype=torch.int64).remainder(251)
    source_bytes = raw.to(torch.uint8).reshape(7, block_size, width)
    source = source_bytes.to(device).view(dtype)
    for rank in range(dcp_size):
        plan = build_dcp_shard_plan(
            src_ids, block_size=block_size, dcp_size=dcp_size, dcp_rank=rank
        )
        indices = prepare_dcp_index_gather_indices(plan, torch.device(device))
        indices = replace(
            indices,
            src_block_id_per_token=indices.src_block_id_per_token.to(index_dtype),
            src_token=indices.src_token.to(index_dtype),
        )
        staging = torch.full(
            (plan.dst_pages + 1, block_size * width),
            253,
            dtype=torch.uint8,
            device=device,
        )
        pages = gather_dcp_mla_pages(source, staging, indices, block_size)
        assert pages == plan.dst_pages
        expected = torch.zeros(plan.dst_pages * block_size, width, dtype=torch.uint8)
        for local_token in range(expected.shape[0]):
            global_token = local_token * dcp_size + rank
            block, token = divmod(global_token, block_size)
            if block < len(src_ids):
                expected[local_token] = source_bytes[src_ids[block], token]
        torch.testing.assert_close(
            staging[: plan.dst_pages].cpu(), expected.reshape(plan.dst_pages, -1)
        )
        assert torch.all(
            staging[-1] == 253
        ), "gather overwrote a neighboring pool region"


@pytest.fixture
def mla_gather_inputs():
    block_size, width = 16, 48
    plan = build_dcp_shard_plan(
        [3, 0, 2], block_size=block_size, dcp_size=2, dcp_rank=1
    )
    indices = prepare_dcp_index_gather_indices(plan, torch.device("cpu"))
    source = torch.zeros(4, block_size, width, dtype=torch.uint8)
    staging = torch.full(
        (plan.dst_pages + 1, block_size * width), 253, dtype=torch.uint8
    )
    return source, staging, indices, block_size


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
def test_mla_gather_rejects_non_byte_staging_before_write(mla_gather_inputs, dtype):
    source, staging, indices, block_size = mla_gather_inputs
    staging = staging.to(dtype)
    before = staging.clone()
    with pytest.raises(TypeError, match="staging must have dtype torch.uint8"):
        gather_dcp_mla_pages(source, staging, indices, block_size)
    torch.testing.assert_close(staging, before)


@pytest.mark.parametrize(
    "field", ["source", "staging", "src_block_id_per_token", "src_token", "valid"]
)
def test_mla_gather_rejects_mixed_devices(mla_gather_inputs, field):
    source, staging, indices, block_size = mla_gather_inputs
    # A metadata-only device exercises this contract without requiring a GPU.
    if field == "source":
        source = source.to("meta")
    elif field == "staging":
        staging = staging.to("meta")
    else:
        indices = replace(indices, **{field: getattr(indices, field).to("meta")})
    with pytest.raises(ValueError, match="same device as source"):
        gather_dcp_mla_pages(source, staging, indices, block_size)


@pytest.mark.parametrize("field", ["src_block_id_per_token", "src_token", "valid"])
@pytest.mark.parametrize("shape", ["broadcast", "short", "2d"])
def test_mla_gather_rejects_malformed_index_shapes(mla_gather_inputs, field, shape):
    source, staging, indices, block_size = mla_gather_inputs
    tensor = getattr(indices, field)
    if shape == "broadcast":
        tensor = tensor[:1]
    elif shape == "short":
        tensor = tensor[:-1]
    else:
        tensor = tensor.reshape(1, -1)
    indices = replace(indices, **{field: tensor})
    with pytest.raises(ValueError, match=f"{field} must be a 1-D tensor"):
        gather_dcp_mla_pages(source, staging, indices, block_size)
    assert torch.all(staging == 253)


@pytest.mark.parametrize(
    "field,dtype",
    [
        ("src_block_id_per_token", torch.bool),
        ("src_token", torch.float32),
        ("valid", torch.uint8),
    ],
)
def test_mla_gather_rejects_invalid_index_dtypes(mla_gather_inputs, field, dtype):
    source, staging, indices, block_size = mla_gather_inputs
    indices = replace(indices, **{field: getattr(indices, field).to(dtype)})
    with pytest.raises(TypeError, match=f"{field} must have dtype"):
        gather_dcp_mla_pages(source, staging, indices, block_size)
    assert torch.all(staging == 253)


def test_mla_gather_rejects_negative_page_count(mla_gather_inputs):
    source, staging, indices, block_size = mla_gather_inputs
    with pytest.raises(ValueError, match="nonnegative destination page count"):
        gather_dcp_mla_pages(
            source, staging, replace(indices, dst_pages=-1), block_size
        )


@pytest.mark.parametrize("shape", [(), (4,), (0, 16, 48), (4, 16, 0)])
def test_mla_gather_rejects_invalid_source_pages(mla_gather_inputs, shape):
    _, staging, indices, block_size = mla_gather_inputs
    source = torch.empty(shape, dtype=torch.uint8)
    with pytest.raises(ValueError, match="nonempty source pages"):
        gather_dcp_mla_pages(source, staging, indices, block_size)


def test_mla_gather_empty_plan_preserves_staging_and_checks_dtype(mla_gather_inputs):
    source, staging, _, block_size = mla_gather_inputs
    plan = build_dcp_shard_plan([], block_size=block_size, dcp_size=2, dcp_rank=0)
    indices = prepare_dcp_index_gather_indices(plan, torch.device("cpu"))
    assert gather_dcp_mla_pages(source, staging, indices, block_size) == 0
    assert torch.all(staging == 253)
    with pytest.raises(TypeError, match="staging must have dtype torch.uint8"):
        gather_dcp_mla_pages(source, staging.float(), indices, block_size)


@pytest.mark.parametrize("index_head_dim", [128, 256])
def test_preshuffled_index_gather_reorganizes_pages_and_zeros_tail(index_head_dim):
    scheduler_block_size = 64
    source_page_size = 4
    block_ratio = scheduler_block_size // source_page_size
    token_tiles = scheduler_block_size // 16
    aligned_index_dim = ((index_head_dim + 4 + 15) // 16) * 16
    num_source_blocks = 5
    page_bytes = scheduler_block_size * aligned_index_dim
    staging_pages = 256

    source_pages = torch.zeros(num_source_blocks, page_bytes, dtype=torch.uint8)
    source_keys = source_pages[:, : scheduler_block_size * index_head_dim].view(
        num_source_blocks, token_tiles, index_head_dim // 16, 16, 16
    )
    source_scales = source_pages[
        :,
        scheduler_block_size
        * index_head_dim : scheduler_block_size
        * (index_head_dim + 4),
    ].view(torch.float32)
    for block_id in range(num_source_blocks):
        for token in range(scheduler_block_size):
            token_tile = token // 16
            token_in_tile = token % 16
            for dim in range(index_head_dim):
                source_keys[
                    block_id, token_tile, dim // 16, token_in_tile, dim % 16
                ] = (block_id * 37 + token * 11 + dim) % 251
            source_scales[block_id, token] = block_id * 1000 + token
    source = source_pages.view(
        num_source_blocks * block_ratio,
        source_page_size,
        aligned_index_dim,
    )

    src_block_ids = [2, 0, 4, 1, 3]
    dcp_size, dcp_rank = 4, 3
    plan = build_dcp_shard_plan(
        src_block_ids,
        block_size=scheduler_block_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )
    indices = prepare_dcp_index_gather_indices(plan, torch.device("cpu"))
    staging = torch.full(
        (staging_pages, page_bytes),
        0xFF,
        dtype=torch.uint8,
    )

    pages = gather_dcp_preshuffled_index_pages(
        source,
        staging,
        indices,
        index_head_dim,
        scheduler_block_size,
        block_ratio,
    )

    output_keys = staging[
        : plan.dst_pages, : scheduler_block_size * index_head_dim
    ].view(plan.dst_pages, token_tiles, index_head_dim // 16, 16, 16)
    output_scales = staging[
        : plan.dst_pages,
        scheduler_block_size
        * index_head_dim : scheduler_block_size
        * (index_head_dim + 4),
    ].view(torch.float32)
    for local_token in range(plan.dst_pages * scheduler_block_size):
        dst_page, dst_token = divmod(local_token, scheduler_block_size)
        global_token = local_token * dcp_size + dcp_rank
        src_ordinal, src_token = divmod(global_token, scheduler_block_size)
        valid = src_ordinal < len(src_block_ids)
        dst_tile = dst_token // 16
        dst_in_tile = dst_token % 16
        for dim in range(index_head_dim):
            actual = output_keys[
                dst_page, dst_tile, dim // 16, dst_in_tile, dim % 16
            ].item()
            expected = (
                (src_block_ids[src_ordinal] * 37 + src_token * 11 + dim) % 251
                if valid
                else 0
            )
            assert actual == expected
        expected_scale = src_block_ids[src_ordinal] * 1000 + src_token if valid else 0
        assert output_scales[dst_page, dst_token].item() == expected_scale

    payload_bytes = scheduler_block_size * (index_head_dim + 4)
    assert pages == plan.dst_pages
    assert not staging[: plan.dst_pages, payload_bytes:].any()
    assert (staging[plan.dst_pages :] == 0xFF).all()


def test_preshuffled_index_gather_rejects_narrow_staging_slot():
    plan = build_dcp_shard_plan(
        [0, 1, 2, 3],
        block_size=16,
        dcp_size=2,
        dcp_rank=0,
    )
    indices = prepare_dcp_index_gather_indices(plan, torch.device("cpu"))
    source = torch.zeros(16, 1, 144, dtype=torch.uint8)
    staging = torch.zeros(plan.dst_pages, 16 * 128, dtype=torch.uint8)
    with pytest.raises(ValueError, match="bytes wide"):
        gather_dcp_preshuffled_index_pages(source, staging, indices, 128, 16, 16)
