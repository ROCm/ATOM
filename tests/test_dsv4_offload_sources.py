# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for DSV4 offload source-region mapping + admission gate."""

from __future__ import annotations

import pytest

from atom.kv_transfer.disaggregation.types import KVTransferRegion, KVTransferTensors
from atom.kv_transfer.offload.dsv4.sources import DSV4OffloadSources


def _tt(with_staging: bool = True) -> KVTransferTensors:
    # 2 compressed layers (bpb 512) + 1 CSA indexer (bpb 256).
    block_regions = [
        KVTransferRegion(0x10000, 512 * 64, 512),
        KVTransferRegion(0x20000, 512 * 64, 512),
        KVTransferRegion(0x30000, 256 * 64, 256),
    ]
    swa_block_regions = [
        KVTransferRegion(0x40000, 384 * 8, 384),
        KVTransferRegion(0x50000, 384 * 8, 384),
    ]
    staging = KVTransferRegion(0x90000, 1024 * 4, 1024) if with_staging else None
    return KVTransferTensors(
        block_regions=block_regions,
        slot_regions=[],
        num_blocks=64,
        num_slots=0,
        swa_block_regions=swa_block_regions,
        staging_region=staging,
        staging_pool_size=4 if with_staging else 0,
        gather_slot=(lambda s, p: None) if with_staging else None,
        scatter_slot=(lambda s, p: None) if with_staging else None,
    )


def _sources(**kw) -> DSV4OffloadSources:
    return DSV4OffloadSources(_tt(**kw), block_size=256, window_size=128)


def test_admission_gate():
    s = _sources()
    assert s.is_checkpoint_boundary(16384)
    assert s.is_checkpoint_boundary(128)
    assert not s.is_checkpoint_boundary(16130)  # not 128-aligned (16128 + 2)
    assert not s.is_checkpoint_boundary(64)  # below one window
    assert not s.is_checkpoint_boundary(0)


def test_num_compressed_blocks_ceil():
    s = _sources()
    assert s.num_compressed_blocks(16384) == 64
    assert s.num_compressed_blocks(256) == 1
    assert s.num_compressed_blocks(257) == 2


def test_compressed_sources_share_block_ids():
    s = _sources()
    block_table = list(range(100))
    regions = s.compressed_sources(block_table, B=16384)
    assert len(regions) == 3
    # All compressed regions use the SAME first-64 block ids.
    for r in regions:
        assert list(r.physical_ids) == list(range(64))
    assert regions[0].base_addr == 0x10000 and regions[0].unit_bytes == 512
    assert regions[2].base_addr == 0x30000 and regions[2].unit_bytes == 256


def test_compressed_block_table_too_short_raises():
    s = _sources()
    with pytest.raises(ValueError, match="too short"):
        s.compressed_sources(list(range(10)), B=16384)


def test_swa_sources_skip_window_freed():
    s = _sources()
    swa_bt = [-1, -1, 7, 3]  # only the live tail survives window-freeing
    regions = s.swa_sources(swa_bt)
    assert len(regions) == 2
    for r in regions:
        assert list(r.physical_ids) == [7, 3]


def test_csa_state_sources():
    s = _sources()
    regions = s.csa_state_sources(2)
    assert len(regions) == 1
    assert regions[0].base_addr == 0x90000
    assert regions[0].unit_bytes == 1024
    assert list(regions[0].physical_ids) == [2]
    with pytest.raises(ValueError, match="out of range"):
        s.csa_state_sources(4)


def test_csa_state_absent_when_no_staging():
    s = _sources(with_staging=False)
    assert s.csa_state_sources(0) == []


def test_build_save_sources_requires_aligned_boundary():
    s = _sources()
    with pytest.raises(ValueError, match="checkpoint boundary"):
        s.build_save_sources(
            block_table=list(range(64)),
            swa_block_table=[1, 2],
            state_pool_idx=0,
            B=16130,
        )


def test_build_save_sources_shapes():
    s = _sources()
    comp, swa, state = s.build_save_sources(
        block_table=list(range(64)),
        swa_block_table=[-1, 9, 4],
        state_pool_idx=1,
        B=16384,
    )
    assert len(comp) == 3 and len(swa) == 2 and len(state) == 1
    assert list(comp[0].physical_ids) == list(range(64))
    assert list(swa[0].physical_ids) == [9, 4]
    assert list(state[0].physical_ids) == [1]


def test_component_and_unit_bytes():
    s = _sources()
    block_table = list(range(64))
    swa_bt = [-1, 9, 4]  # 2 live blocks
    c, sw, st = s.component_bytes(block_table=block_table, swa_block_table=swa_bt, B=16384)
    # compressed = (512 + 512 + 256) * 64 ; swa = (384 + 384) * 2 ; state = 1024
    assert c == (512 + 512 + 256) * 64
    assert sw == (384 + 384) * 2
    assert st == 1024
    total = s.unit_bytes(block_table=block_table, swa_block_table=swa_bt, B=16384)
    assert total >= c + sw + st  # includes header + alignment padding
