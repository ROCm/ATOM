# SPDX-License-Identifier: MIT

"""No stale bytes from a PAGE checkpoint reach a KV read.

A released checkpoint's units go back to the KV pool and are zeroed before any
forward writes them as KV, and a GDN prefill zeroes the conv window's dead
speculative tail so no checkpoint image carries it. Both guard the attention
tail tile, which reads unwritten positions with zero weight -- and `0 * Inf`
is NaN.
"""

from types import SimpleNamespace

import numpy as np
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


def test_gdn_prefill_zeroes_only_the_dead_conv_tail():
    pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

    # [layers, slots, kernel - 1 + num_spec, channels]: kernel 4, 3 drafts
    conv = torch.full((2, 4, 6, 3), float("nan"))
    conv[:, :, :3] = 1.0
    runner = SimpleNamespace(
        mamba_k_cache=conv,
        config=SimpleNamespace(hf_config=SimpleNamespace(linear_conv_kernel_dim=4)),
    )
    builder = SimpleNamespace(
        model_runner=runner,
        # prefill rows' write slots; a negative entry is padding
        non_spec_state_indices_tensor=SimpleNamespace(np=np.array([2, -1, 0, 3])),
        _staged_i64=lambda name, values: torch.as_tensor(values, dtype=torch.int64),
    )
    builder._conv_state_len = lambda: GDNAttentionMetadataBuilder._conv_state_len(
        builder
    )
    GDNAttentionMetadataBuilder._zero_dead_conv_columns(builder, 3)
    assert (conv[:, [0, 2], 3:] == 0).all()  # dead tail of the written slots
    assert (conv[:, :, :3] == 1.0).all()  # history kept everywhere
    assert torch.isnan(conv[:, [1, 3], 3:]).all()  # slots not in this prefill
