# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Offload-side scaffold for Qwen3-Next (GDN hybrid KV): dispatch + profile +
SLOT-component sourcing. Exercises the ``SLOT`` region kind (GDN recurrent state)
through the profile path with mock transfer tensors — the model-side
``get_kv_transfer_tensors`` (block_regions + slot_regions) is not yet wired.
"""

from __future__ import annotations

from types import SimpleNamespace

from atom.kv_transfer.disaggregation.types import KVTransferRegion, KVTransferTensors
from atom.kv_transfer.offload.connector import select_variant
from atom.kv_transfer.offload.hybrid.profiles import select_profile
from atom.kv_transfer.offload.hybrid.profiles.base import RegionKind, BundleSources
from atom.kv_transfer.offload.hybrid.profiles.qwen3_next import build_qwen3_next_profile


def _qwen_cfg(block_size=16, linear_key_heads=16):
    hf = SimpleNamespace(
        num_hidden_layers=48,
        head_dim=128,
        linear_num_key_heads=linear_key_heads,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention" if (i + 1) % 4 else "full_attention"
                     for i in range(48)],
    )
    return SimpleNamespace(
        model="qwen3-next",
        kv_cache_dtype="auto",
        kv_cache_block_size=block_size,
        tensor_parallel_size=1,
        hf_config=hf,
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": "offload"},
    )


def _tt():
    # full-attn KV (block-indexed) + GDN state (per-req slot).
    return KVTransferTensors(
        block_regions=[KVTransferRegion(0x10000, 512 * 32, 512)],
        slot_regions=[KVTransferRegion(0x70000, 4096 * 8, 4096)],
        num_blocks=32,
        num_slots=8,
        swa_block_regions=[],
        staging_region=None,
        staging_pool_size=0,
        gather_slot=None,
        scatter_slot=None,
    )


def test_dispatch_routes_qwen_to_hybrid():
    # No compress_ratios, but linear_num_key_heads present => hybrid.
    assert select_variant(_qwen_cfg()) == "hybrid"


def test_select_profile_picks_qwen():
    prof = select_profile(_qwen_cfg(), world=1, rank=0)
    assert prof.model_tag == "qwen3-next"
    kinds = [(c.name, c.kind) for c in prof.components]
    assert kinds == [
        ("full_kv", RegionKind.BLOCK),
        ("gdn_state", RegionKind.SLOT),
    ]
    assert prof.cadence.align == 16  # = block_size


def test_qwen_unit_sources_slot_component():
    prof = build_qwen3_next_profile(_qwen_cfg(block_size=16), world=1, rank=0)
    us = BundleSources(prof, _tt())
    comps = dict(
        us.build_components(
            block_table=list(range(32)), swa_block_table=[], B=256, slot_id=5
        )
    )
    # full_kv: BLOCK, keyed by block_table[:ceil(256/16)] = first 16 blocks.
    fk = comps["full_kv"]
    assert len(fk) == 1
    assert fk[0].base_addr == 0x10000 and fk[0].unit_bytes == 512
    assert list(fk[0].physical_ids) == list(range(16))
    # gdn_state: SLOT, one per-req slot id.
    gs = comps["gdn_state"]
    assert len(gs) == 1
    assert gs[0].base_addr == 0x70000 and gs[0].unit_bytes == 4096
    assert list(gs[0].physical_ids) == [5]


def test_qwen_slot_requires_slot_id():
    import pytest

    us = BundleSources(build_qwen3_next_profile(_qwen_cfg(), world=1, rank=0), _tt())
    with pytest.raises(ValueError):
        # SLOT component with no slot_id (default -1) must fail closed.
        us.build_components(block_table=list(range(32)), swa_block_table=[], B=256)


def test_qwen_geometry_fingerprint_distinguishes_config():
    p_a = build_qwen3_next_profile(_qwen_cfg(linear_key_heads=16), world=1, rank=0)
    p_b = build_qwen3_next_profile(_qwen_cfg(linear_key_heads=8), world=1, rank=0)
    # A change in GDN state layout MUST flip the fingerprint (load fails closed).
    assert p_a.build_geometry().fingerprint() != p_b.build_geometry().fingerprint()
    # Same config => stable fingerprint.
    assert (
        build_qwen3_next_profile(_qwen_cfg(), world=1, rank=0).build_geometry().fingerprint()
        == build_qwen3_next_profile(_qwen_cfg(), world=1, rank=0).build_geometry().fingerprint()
    )


def test_scheduler_shell_selects_qwen_hybrid():
    from atom.kv_transfer.offload.connector import LMCacheOffloadConnectorScheduler
    from atom.kv_transfer.offload.hybrid.connector import HybridOffloadScheduler

    sch = LMCacheOffloadConnectorScheduler(_qwen_cfg(block_size=16))
    assert isinstance(sch._impl, HybridOffloadScheduler)
    # cadence align comes from the Qwen profile (block_size), not DSV4's 128.
    assert sch._impl._align == 16
