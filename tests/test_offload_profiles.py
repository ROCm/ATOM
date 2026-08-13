# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.hybrid.profiles import build_dsv4_profile
from atom.kv_transfer.offload.hybrid.policy import (
    select_pending_sidecar_boundary,
    sidecar_boundary_tokens,
)


def _config(**overrides):
    values = {
        "kv_cache_block_size": 256,
        "decode_context_parallel_size": 2,
        "state_checkpoint_interval_tokens": 9000,
        "hf_config": SimpleNamespace(kv_head_dim=576, index_head_dim=160),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dsv4_profile_resolves_virtual_grid_and_cadence():
    profile = build_dsv4_profile(_config(), chunk_size=8192)

    assert profile.name == "deepseek-v4-page-slot"
    assert profile.block_size == 256
    assert profile.dcp_size == 2
    assert profile.hash_block_size == 512
    assert profile.resume_alignment == 8192
    assert profile.checkpoint_interval == 8704
    assert profile.sidecar_interval == 139264
    assert profile.kv_head_dim == 576
    assert profile.index_head_dim == 160


def test_dsv4_profile_rejects_chunk_that_splits_virtual_dcp_block():
    with pytest.raises(ValueError, match="virtual DCP block"):
        build_dsv4_profile(_config(), chunk_size=768)


def test_sidecar_policy_includes_regular_and_terminal_boundaries():
    assert sidecar_boundary_tokens(
        num_prompt_tokens=20,
        resume_alignment=4,
        sidecar_interval=8,
    ) == (8, 16, 20)


def test_pending_policy_does_not_cross_later_boundary_while_one_is_inflight():
    assert (
        select_pending_sidecar_boundary(
            [(8, 101), (16, 202)],
            start=0,
            end=16,
            committed_hashes=set(),
            inflight=(object(), 8, 101),
            failed=set(),
        )
        is None
    )
