# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`v4_pool_fields`: what a DeepSeek-V4 KV row and indexer block hold.

Runs on CI, which is the reason this arithmetic sits in `pool_layout/` at all:
the backend that consumes it imports aiter, so nothing in that file is reachable
on a plain runner. Everything here is bytes and shapes -- no GPU, no kernels.

The scans over head dims are the point rather than decoration. Both layouts
happen to land on `entry_arena`'s 256 B field alignment at the dims this repo
ships, so a test at one point cannot tell a packed layout from an aligned one,
and those are the two answers that differ by a silently-misplaced scale region.
"""

from __future__ import annotations

import pytest
import torch

from atom.model_ops.attentions.pool_layout.entry_arena import entry_bytes_for
from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    CSA_INDEXER_DATA,
    CSA_INDEXER_SCALE,
    MAIN_KV_NOPE,
    MAIN_KV_ROPE,
    fp4_indexer_block_fields,
    fp8_indexer_block_fields,
    indexer_block_regions,
    main_kv_plane_fields,
)

# What the shipped V4 configs use, so at least one case is the real one.
_SHIPPED_ROWS = 64
_SHIPPED_INDEX_HEAD_DIM = 128


class TestMainKvPlaneFields:
    def test_a_bf16_build_declares_one_plane(self):
        fields = main_kv_plane_fields(512, torch.bfloat16)

        assert [f.name for f in fields] == [MAIN_KV_NOPE]
        assert [f.bytes_per_entry for f in fields] == [1024]

    def test_an_fp8_build_declares_the_rope_plane_at_its_own_width(self):
        fields = main_kv_plane_fields(512, torch.uint8, (64, torch.bfloat16))

        assert [f.name for f in fields] == [MAIN_KV_NOPE, MAIN_KV_ROPE]
        assert [f.bytes_per_entry for f in fields] == [512, 128]

    def test_a_row_is_one_layer_and_one_position(self):
        """`bytes_per_entry` is the row width only while `layers` is one, and
        several callers price a row that way."""
        for field in main_kv_plane_fields(512, torch.uint8, (64, torch.bfloat16)):
            assert field.layers == 1
            assert len(field.shape) == 1

    def test_plane_names_are_the_pd_wire_format(self):
        """Spelled out because a consumer matches its producer's regions on
        them, and the two may be running different builds."""
        assert (MAIN_KV_NOPE, MAIN_KV_ROPE) == (
            "dsv4.main_kv.nope",
            "dsv4.main_kv.rope",
        )


class TestIndexerBlockRegions:
    @pytest.mark.parametrize("index_head_dim", [16, 64, 128, 132, 256])
    @pytest.mark.parametrize("rows", [3, 32, 64])
    def test_fp8_block_is_the_data_region_then_the_scale_region(
        self, rows, index_head_dim
    ):
        fields = fp8_indexer_block_fields(rows, index_head_dim, torch.uint8)
        regions, block_bytes = indexer_block_regions(fields)

        assert regions[CSA_INDEXER_DATA] == 0
        assert regions[CSA_INDEXER_SCALE] == rows * index_head_dim
        assert block_bytes == rows * (index_head_dim + 4)

    @pytest.mark.parametrize("index_head_dim", [128, 256, 384])
    @pytest.mark.parametrize("rows", [3, 32, 64])
    def test_fp4_block_is_16_packed_bytes_and_one_scale_per_group(
        self, rows, index_head_dim
    ):
        groups = (index_head_dim // 128) * 4 * rows
        fields = fp4_indexer_block_fields(rows, index_head_dim)
        regions, block_bytes = indexer_block_regions(fields)

        assert regions[CSA_INDEXER_DATA] == 0
        assert regions[CSA_INDEXER_SCALE] == groups * 16
        assert block_bytes == groups * 17

    def test_fp4_pools_are_one_region_each(self):
        """FP4's two regions are two tensors, so a region's shape is a pool's
        shape after the layer and block axes."""
        data, scale = fp4_indexer_block_fields(64, 256)

        assert data.shape == (2, 4, 64, 16)
        assert scale.shape == (2, 4, 64)
        assert data.dtype is scale.dtype is torch.uint8

    def test_regions_are_a_prefix_sum_with_nothing_between_them(self):
        fields = fp8_indexer_block_fields(64, 128, torch.uint8)
        regions, block_bytes = indexer_block_regions(fields)

        running = 0
        for field in fields:
            assert regions[field.name] == running
            running += field.bytes_per_entry
        assert running == block_bytes

    @pytest.mark.parametrize(
        "fields",
        [
            fp8_indexer_block_fields(3, 5, torch.uint8),
            fp4_indexer_block_fields(3, 128),
        ],
        ids=["fp8", "fp4"],
    )
    def test_a_block_is_packed_where_an_entry_would_be_aligned(self, fields):
        """The one case that tells the two layouts apart.

        `entry_bytes_for` rounds each field to 256 B, which every dim this repo
        ships already satisfies -- so only a block whose regions are NOT a
        multiple of 256 shows that these regions are packed. If this ever
        starts passing by equality, the layout has quietly borrowed the aligned
        walk and every consumer's `rows * head_dim` scale offset is wrong.
        """
        _, block_bytes = indexer_block_regions(fields)

        assert block_bytes < entry_bytes_for(fields)

    def test_the_shipped_fp8_block_divides_into_its_rows(self):
        """The pool is one `[layers, blocks, rows, X]` tensor -- the kernel
        reads the block size off axis 1 -- so `X` has to be a whole number."""
        _, block_bytes = indexer_block_regions(
            fp8_indexer_block_fields(
                _SHIPPED_ROWS, _SHIPPED_INDEX_HEAD_DIM, torch.uint8
            )
        )

        assert block_bytes % _SHIPPED_ROWS == 0
        assert block_bytes // _SHIPPED_ROWS == _SHIPPED_INDEX_HEAD_DIM + 4

    def test_the_shipped_fp8_block_stride_takes_dwordx4_loads(self):
        """What the backend asserts at startup; pinned here so the dims that
        satisfy it are on the record rather than only the assertion."""
        _, block_bytes = indexer_block_regions(
            fp8_indexer_block_fields(
                _SHIPPED_ROWS, _SHIPPED_INDEX_HEAD_DIM, torch.uint8
            )
        )

        assert block_bytes == 8448
        assert block_bytes % 16 == 0
