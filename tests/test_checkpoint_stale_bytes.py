# SPDX-License-Identifier: MIT

"""No stale bytes from a PAGE checkpoint reach a KV read.

A released checkpoint's units go back to the KV pool and are zeroed before any
forward writes them as KV. The attention tail tile reads a block's unwritten
positions with zero weight -- and `0 * Inf` is NaN.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.state_runtime import StateMaintenanceOps


def test_release_units_records_the_released_ids():
    pool = BlockPool(num_blocks=6)
    owner = ("state-checkpoint", 0)
    units = pool.reserve_units(3, owner)
    assert pool.released_raw_units == []
    pool.release_units(units, owner)
    assert pool.released_raw_units == units


def test_zero_blocks_alone_is_maintenance():
    assert StateMaintenanceOps().empty
    assert not StateMaintenanceOps(zero_blocks=(4,)).empty


def test_zero_kv_blocks_clears_every_field_of_every_layer():
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    from atom.model_ops.attentions.backends import AttentionMetadataBuilder

    k = torch.full((3, 5, 2, 4), float("inf"))  # [layers, blocks, ...]
    v = torch.full((3, 5, 8), float("nan"))
    builder = SimpleNamespace(
        kv_pools={0: SimpleNamespace(_views={"k": k, "v": v})},
        device=torch.device("cpu"),
    )
    AttentionMetadataBuilder.zero_kv_blocks(builder, [1, 3])
    for t in (k, v):
        assert (t[:, [1, 3]] == 0).all()
        assert not torch.isfinite(t[:, [0, 2, 4]]).any()  # untouched

