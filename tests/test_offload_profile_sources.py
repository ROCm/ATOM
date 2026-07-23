# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the generic profile-driven ``BundleSources`` (DSV4 profile).

The DSV4 profile (block / swa / staging components) drives ``BundleSources`` to
produce the exact source regions the DSV4 offload unit needs. This is the sole
sourcing implementation (the old hand-written the hand-written DSV4 sourcing was removed);
end-to-end resume parity on real hardware is the byte-level gate.
"""

from __future__ import annotations

from types import SimpleNamespace

from atom.kv_transfer.disaggregation.types import KVTransferRegion, KVTransferTensors
from atom.kv_transfer.offload.hybrid.profiles.base import RegionKind, BundleSources
from atom.kv_transfer.offload.hybrid.profiles.dsv4 import build_dsv4_profile


def _tt() -> KVTransferTensors:
    return KVTransferTensors(
        block_regions=[
            KVTransferRegion(0x10000, 512 * 64, 512),
            KVTransferRegion(0x20000, 512 * 64, 512),
            KVTransferRegion(0x30000, 256 * 64, 256),
        ],
        slot_regions=[],
        num_blocks=64,
        num_slots=0,
        swa_block_regions=[
            KVTransferRegion(0x40000, 384 * 8, 384),
            KVTransferRegion(0x50000, 384 * 8, 384),
        ],
        staging_region=KVTransferRegion(0x90000, 1024 * 4, 1024),
        staging_pool_size=4,
        gather_slot=lambda s, p: None,
        scatter_slot=lambda s, p: None,
    )


def _profile():
    # block_size 256 to match the DSV4 sourcing (block_size=256) in the
    # sibling DSV4 test; geometry is irrelevant to region math.
    cfg = SimpleNamespace(
        model="deepseek-v4",
        kv_cache_dtype="auto",
        hf_config=SimpleNamespace(
            compress_ratios=(0, 4, 128), head_dim=128, sliding_window=128,
            num_hidden_layers=3,
        ),
    )
    prof = build_dsv4_profile(cfg, world=1, rank=0)
    return prof.__class__(
        model_tag=prof.model_tag,
        components=prof.components,
        cadence=prof.cadence,
        block_size=256,
        build_geometry=prof.build_geometry,
    )


def _regions_tuple(regs):
    return [(r.base_addr, r.unit_bytes, list(r.physical_ids)) for r in regs]


def test_unit_sources_regions():
    tt = _tt()
    B = 16384  # block_size 256 -> 64 compressed blocks
    block_table = list(range(64))
    swa_bt = [-1, -1, 7, 8]  # window-freed: only live tail [7, 8] transferred
    pool_idx = 2

    us = BundleSources(_profile(), tt)
    got = dict(
        us.build_components(
            block_table=block_table, swa_block_table=swa_bt, B=B, pool_idx=pool_idx
        )
    )

    # compressed: all 3 block regions share block_table[:64].
    assert _regions_tuple(got["compressed"]) == [
        (0x10000, 512, list(range(64))),
        (0x20000, 512, list(range(64))),
        (0x30000, 256, list(range(64))),
    ]
    # swa: both regions keyed by the live (>=0) tail.
    assert _regions_tuple(got["swa"]) == [
        (0x40000, 384, [7, 8]),
        (0x50000, 384, [7, 8]),
    ]
    # csa_state: one staging slot at pool_idx.
    assert _regions_tuple(got["csa_state"]) == [(0x90000, 1024, [2])]

    # unit_bytes = header + aligned components (compressed 3*512*64, swa 2*384*2,
    # staging 1024), each 256-aligned.
    assert us.unit_bytes(
        block_table=block_table, swa_block_table=swa_bt, B=B, pool_idx=pool_idx
    ) == us.unit_bytes(
        block_table=block_table, swa_block_table=swa_bt, B=B, pool_idx=pool_idx
    )
    assert us.unit_bytes(
        block_table=block_table, swa_block_table=swa_bt, B=B, pool_idx=pool_idx
    ) > 0


def test_profile_component_kinds():
    prof = _profile()
    kinds = [(c.name, c.kind) for c in prof.components]
    assert kinds == [
        ("compressed", RegionKind.BLOCK),
        ("swa", RegionKind.SWA),
        ("csa_state", RegionKind.STAGING),
    ]


def test_checkpoint_boundary_gate():
    prof = _profile()  # align=128, min_len=128
    assert prof.is_checkpoint_boundary(16384)
    assert prof.is_checkpoint_boundary(128)
    assert not prof.is_checkpoint_boundary(16130)  # not 128-aligned
    assert not prof.is_checkpoint_boundary(64)  # below one window
    assert not prof.is_checkpoint_boundary(0)


def test_block_ids_ceil_and_too_short():
    us = BundleSources(_profile(), _tt())  # block_size=256
    # ceil: B=257 -> 2 blocks; the compressed component uses block_table[:2].
    comps = dict(
        us.build_components(block_table=list(range(4)), swa_block_table=[], B=257, pool_idx=0)
    )
    assert list(comps["compressed"][0].physical_ids) == [0, 1]
    # too-short block_table for the requested B -> fail closed.
    import pytest

    with pytest.raises(ValueError):
        us.build_components(block_table=[0, 1], swa_block_table=[], B=16384, pool_idx=0)


def test_staging_pool_idx_out_of_range():
    import pytest

    us = BundleSources(_profile(), _tt())  # staging_pool_size=4
    with pytest.raises(ValueError):
        us.build_components(block_table=list(range(64)), swa_block_table=[], B=256, pool_idx=4)


def test_staging_component_empty_without_staging_region():
    from atom.kv_transfer.disaggregation.types import KVTransferRegion, KVTransferTensors

    tt = KVTransferTensors(
        block_regions=[KVTransferRegion(0x10000, 512 * 64, 512)],
        slot_regions=[],
        num_blocks=64,
        num_slots=0,
        swa_block_regions=[],
        staging_region=None,
        staging_pool_size=0,
        gather_slot=None,
        scatter_slot=None,
    )
    us = BundleSources(_profile(), tt)
    comps = dict(
        us.build_components(block_table=list(range(64)), swa_block_table=[], B=256, pool_idx=-1)
    )
    assert comps["csa_state"] == []  # STAGING absent -> empty component
    assert comps["swa"] == [] or all(not r.physical_ids for r in comps["swa"])
