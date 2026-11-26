# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A paged pool a KV connector provides (`umbp_pool`): one allocation for the
whole arena, its PAGE regions named for the connector, whole-block copies.
Without a provider the cache allocates exactly as before."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("aiter", reason="the paged cache and V4 kernels reach AITER")

from atom.model_ops.attentions.deepseek_v41.backend import (
    DeepseekV41MetadataBuilder,
)
from atom.model_ops.attentions.deepseek_v41.cache import (
    PagedAttentionCache,
    cut_slab,
)
from tests.attentions.deepseek_v41.helpers import geometry

DTYPES = ["bf16", "fp4", "fp8"]


def _cache(small_config, dtype, **kwargs):
    geo = replace(geometry(small_config), window_size=32, kv_dtype=dtype)
    return PagedAttentionCache(geo, 8, 4, "cpu", **kwargs)


def _builder(cache):
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.cache = cache
    builder.device = "cpu"
    return builder


def _storages(cache):
    return {buf.untyped_storage().data_ptr() for buf in cache.allocations}


def test_cut_slab_places_each_view_at_its_alignment():
    slab, views = cut_slab(lambda n: torch.zeros(n, dtype=torch.uint8), [5, 3, 9])
    assert slab.numel() == 17  # unaligned: back to back
    assert [v.data_ptr() - slab.data_ptr() for v in views] == [0, 5, 8]
    slab, views = cut_slab(lambda n: torch.zeros(n, dtype=torch.uint8), [5, 3, 9], 8)
    assert slab.numel() == 32
    assert [v.data_ptr() - slab.data_ptr() for v in views] == [0, 8, 16]
    assert [v.numel() for v in views] == [5, 3, 9]


@pytest.mark.parametrize("dtype", DTYPES)
def test_without_a_provider_the_cache_allocates_on_its_own(small_config, dtype):
    cache = _cache(small_config, dtype)
    assert cache.arena is None
    assert _builder(cache).owned_pool_layout() is None
    if dtype != "bf16":
        # One buffer per source, each its own allocation.
        assert len(_storages(cache)) == len(cache.allocations)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_provider_backs_the_whole_pool_with_one_allocation(small_config, dtype):
    asked = []

    def allocate(nbytes, device):
        asked.append(nbytes)
        return torch.zeros(nbytes, dtype=torch.uint8, device=device)

    cache = _cache(small_config, dtype, allocate_backing=allocate)
    assert len(asked) == 1 and cache.arena.numel() == asked[0]
    lo = cache.arena.data_ptr()
    hi = lo + cache.arena.numel()
    if dtype != "bf16":
        assert _storages(cache) == {cache.arena.untyped_storage().data_ptr()}
        for buf in cache.allocations:
            assert (buf.data_ptr() - lo) % PagedAttentionCache._BUFFER_ALIGN == 0
    arena, regions = _builder(cache).owned_pool_layout()
    assert arena is cache.arena
    units = cache.unit_regions()
    assert len(regions) == len(units)
    for region, (base, unit) in zip(regions, units):
        assert (region.base_addr, region.unit_bytes) == (base, unit)
        assert lo <= base and base + cache.num_pages * unit <= hi


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_block_copy_moves_every_region_of_its_page(small_config, dtype):
    cache = _cache(small_config, dtype)
    for view in cache.unit_views(2):
        view.copy_(torch.randint(0, 256, view.shape, dtype=torch.uint8))
    before = [view.clone() for unit in (2, 3) for view in cache.unit_views(unit)]
    _builder(cache).copy_paged_blocks([(2, 5)])
    for src, dst in zip(cache.unit_views(2), cache.unit_views(5)):
        assert torch.equal(src, dst)
    after = [view for unit in (2, 3) for view in cache.unit_views(unit)]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
