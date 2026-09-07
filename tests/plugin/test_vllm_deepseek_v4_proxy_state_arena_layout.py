# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from atom.model_ops.attentions.pool_layout.state_arena import (
    StateField,
    plan_field_planes,
    plan_regions,
)
from atom.model_ops.attentions.pool_layout.v4_pool_geometry import UnifiedPoolGeometry
from atom.plugin.vllm.deepseek_v4_bridge import (
    ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT,
    _proxy_region_byte_sizes,
    slice_deepseek_v4_proxy_cache_views,
)

FLASH_RATIOS = [4] * 21 + [128] * 20
_ALIGN = ATOM_DEEPSEEK_V4_PROXY_ALIGNMENT


def _flash_state_layout(kv_fp8: bool, *, ring_extra: int = 1):
    head_dim = 512
    index_head_dim = 128
    n_csa = sum(1 for r in FLASH_RATIOS if r == 4)
    n_hca = sum(1 for r in FLASH_RATIOS if r == 128)
    fields = [
        StateField("csa_main_kv", n_csa, (8 + ring_extra, 2 * head_dim), torch.float32),
        StateField(
            "csa_main_score",
            n_csa,
            (8 + ring_extra, 2 * head_dim),
            torch.float32,
            float("-inf"),
        ),
        StateField(
            "csa_idx_kv",
            n_csa,
            (8 + ring_extra, 2 * index_head_dim),
            torch.float32,
        ),
        StateField(
            "csa_idx_score",
            n_csa,
            (8 + ring_extra, 2 * index_head_dim),
            torch.float32,
            float("-inf"),
        ),
        # Mirrors the bridge, `in_checkpoint` included: this list is that
        # list's oracle, so a difference here is a difference nobody catches.
        StateField(
            "hca_main_kv",
            n_hca,
            (128 + ring_extra, head_dim),
            torch.float32,
            in_checkpoint=False,
        ),
        StateField(
            "hca_main_score",
            n_hca,
            (128 + ring_extra, head_dim),
            torch.float32,
            float("-inf"),
            in_checkpoint=False,
        ),
    ]
    row_widths = [head_dim * (1 if kv_fp8 else 2)]
    if kv_fp8:
        row_widths.append(64 * 2)
    arena_planes, arena_rows = plan_field_planes(fields, row_widths)
    return arena_planes, arena_rows, row_widths


def _proxy_total_bytes(
    *,
    num_blocks: int,
    num_slots: int,
    kv_fp8: bool,
    arena_rows: int,
) -> int:
    geometry = UnifiedPoolGeometry(
        FLASH_RATIOS,
        num_blocks=num_blocks,
        num_slots=num_slots,
        ring_slots=129,
        block_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        arena_rows=arena_rows,
    )
    regions = _proxy_region_byte_sizes(
        geometry=geometry,
        csa_layers=21,
        num_blocks=num_blocks,
        head_dim=512,
        rope_head_dim=64,
        index_head_dim=128,
        kv_fp8=kv_fp8,
    )
    _, total = plan_regions(regions)
    return total


def _make_proxy_kv_cache(
    total_bytes: int, num_blocks: int, storage_offset: int
) -> torch.Tensor:
    block_bytes = 2 * ATOM_DEEPSEEK_V4_BLOCK_SIZE
    page_bytes = (total_bytes + _ALIGN - 1 + num_blocks - 1) // num_blocks
    head_size = (page_bytes + block_bytes - 1) // block_bytes
    alloc = num_blocks * block_bytes * head_size
    raw = torch.zeros(storage_offset + alloc, dtype=torch.uint8)
    physical = raw[storage_offset:].view(
        num_blocks, 2, ATOM_DEEPSEEK_V4_BLOCK_SIZE, 1, head_size
    )
    # vLLM logical shape; slice permutes back to the contiguous block-major view.
    return physical.permute(1, 0, 2, 3, 4)


class TestProxyRegionLayout:

    def test_regions_place_rope_before_indexers(self):
        geometry = UnifiedPoolGeometry(
            FLASH_RATIOS,
            num_blocks=513,
            num_slots=8,
            ring_slots=129,
            block_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
            arena_rows=100,
        )
        regions = _proxy_region_byte_sizes(
            geometry=geometry,
            csa_layers=21,
            num_blocks=513,
            head_dim=512,
            rope_head_dim=64,
            index_head_dim=128,
            kv_fp8=True,
        )
        offsets, _ = plan_regions(regions)
        assert regions[0:2] == [
            geometry.plane_bytes(512),
            geometry.plane_bytes(128),
        ]
        assert offsets[1] == regions[0]
        assert offsets[1] % _ALIGN == 0
        assert offsets[2] >= offsets[1] + regions[1]

    @pytest.mark.parametrize("num_blocks", [513, 1024, 1025])
    @pytest.mark.parametrize("kv_fp8", [False, True])
    @pytest.mark.parametrize("storage_offset", [0, 128])
    def test_slice_state_arena_bufs_are_256b_aligned(
        self, num_blocks, kv_fp8, storage_offset
    ):
        arena_planes, arena_rows, row_widths = _flash_state_layout(kv_fp8)
        total = _proxy_total_bytes(
            num_blocks=num_blocks,
            num_slots=8,
            kv_fp8=kv_fp8,
            arena_rows=arena_rows,
        )
        proxy = _make_proxy_kv_cache(total, num_blocks, storage_offset)
        views = slice_deepseek_v4_proxy_cache_views(
            proxy,
            compress_ratios=FLASH_RATIOS,
            num_slots=8,
            window_size=129,
            head_dim=512,
            index_head_dim=128,
            kv_fp8=kv_fp8,
            rope_head_dim=64,
            arena_planes=arena_planes,
            arena_rows=arena_rows,
            row_widths=row_widths,
        )
        arena = views["state_arena"]
        assert arena is not None
        for plane_arena in arena.arenas:
            assert plane_arena.buf.storage_offset() % _ALIGN == 0
