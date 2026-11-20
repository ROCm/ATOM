# SPDX-License-Identifier: MIT
"""Standalone indexer ABI, cache isolation and TP resource admission."""

import dataclasses

import pytest

from atom.models.deepseek_v41.mono import indexer_build as kernel

BASE = kernel.IndexerBuild(8, 8320, fp4=True, mfma=False)
OTHER = {
    "tokens": 16,
    "bound_max": 163968,
    "fp4": False,
    "mfma": True,
    "diag_off": 1 << 24,
    "world": 2,
    "peer_offset": 4096,
    "peer_plain": 1 << 24,
    "full_only": True,
}


def test_every_indexer_field_is_listed():
    assert set(OTHER) == {f.name for f in dataclasses.fields(BASE)}


def test_adaptive_local_widths_fit_the_shared_tp_arena():
    for world in (2, 4, 8):
        for capacity in (8320, 32768, 163968, 262144):
            wide = dataclasses.replace(
                BASE, tokens=kernel.MAX_ROWS, bound_max=capacity, world=world
            )
            arena = kernel.scratch_bytes(wide)
            for rows in (1, 6, 8):
                local = dataclasses.replace(wide, tokens=rows, world=1)
                assert kernel.scratch_bytes(local) <= arena


@pytest.mark.parametrize("bound", [8320, 16384, 163968, 524288])
def test_reindex_build_keeps_projection_and_bounds_the_compressed_domain(bound):
    key = dataclasses.replace(
        BASE,
        bound_max=bound,
        world=4,
        peer_offset=4096,
        peer_plain=1 << 24,
    )
    expected = dataclasses.replace(
        key, bound_max=min(bound, 18432), world=1, peer_offset=0, peer_plain=0
    )
    assert kernel.reindex_build(key) == expected


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("bound", [8320, 163968, 262144, 524288])
def test_full_build_bounds_keys_and_preserves_peer_fence_geometry(ratio, bound):
    key = dataclasses.replace(
        BASE, bound_max=bound, world=4, peer_offset=4096, peer_plain=1 << 24
    )
    full = kernel.full_build(key, ratio)
    assert full == dataclasses.replace(
        key,
        bound_max=((bound + ratio * 128 - 1) // (ratio * 128)) * 128,
        full_only=True,
    )
    layout = kernel.scratch_layout(full)
    assert not any(name.startswith("iblk") for name in layout)
    assert "ix_rowhist" not in layout and "ix_segment_ready" not in layout
    assert kernel.scratch_bytes(full) < kernel.scratch_bytes(key)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("bound", [128, 2048, 4096, 8320, 163968])
@pytest.mark.parametrize("ratio", [1, 2])
def test_full_selection_storage_also_fits_score_rows(world, bound, ratio):
    key = kernel.full_build(
        dataclasses.replace(BASE, bound_max=bound, world=world), ratio
    )
    assert kernel.selection_keys(key) >= 3654
    assert kernel.selection_keys(key) >= world * 512
