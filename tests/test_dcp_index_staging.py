# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from atom.kv_transfer.disaggregation.index_staging import (
    gather_dcp_preshuffled_index_pages,
    prepare_dcp_index_gather_indices,
)
from atom.kv_transfer.disaggregation.sharded_transfer import build_dcp_shard_plan


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
