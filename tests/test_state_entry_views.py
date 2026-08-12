# SPDX-License-Identifier: MIT
# state_entry_views must cover a group's whole state, contiguously.
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
)
from atom.model_ops.attentions.gdn_attn import GDNStateMixin
from atom.model_ops.attentions.v4_pool_geometry import UnifiedPoolGeometry

LAYERS, GROUPS = 3, 4
SHAPE_K, SHAPE_V = (2, 5), (2, 3, 4)


def build(num_spec: int):
    span = 1 + num_spec
    slots = GROUPS * span
    k = torch.arange(LAYERS * slots * 10, dtype=torch.float32)[
        : LAYERS * slots * SHAPE_K[0] * SHAPE_K[1]
    ].reshape((LAYERS, slots) + SHAPE_K)
    v = torch.zeros((LAYERS, slots) + SHAPE_V)
    stub = SimpleNamespace(
        num_spec=num_spec,
        model_runner=SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v),
    )
    return stub, k, v, span


@pytest.mark.parametrize("num_spec", [0, 2])
def test_every_view_is_contiguous(num_spec):
    """_build_meta rejects a strided segment (triton_kv_staging.py:135)."""
    stub, _, _, _ = build(num_spec)
    views = GDNStateMixin.state_entry_views(stub, 1)
    assert views
    assert all(v.is_contiguous() for v in views)


def test_views_cover_the_whole_group_and_nothing_else():
    stub, k, v, span = build(num_spec=2)
    views = GDNStateMixin.state_entry_views(stub, 1)
    total = sum(int(x.numel()) for x in views)
    expected = LAYERS * span * (k[0, 0].numel() + v[0, 0].numel())
    assert total == expected


def test_writing_through_the_views_writes_the_cache():
    """Views must alias, not copy — the packer reads them in place."""
    stub, k, _, span = build(num_spec=1)
    for view in GDNStateMixin.state_entry_views(stub, 2):
        view.fill_(7.0)
    assert torch.all(k[:, 2 * span : 3 * span] == 7.0)
    assert not torch.all(k[:, 0:span] == 7.0)


# --- DeepSeek-V4 ---------------------------------------------------------
#
# The V4 side goes through the real `_slot_views` / `UnifiedPoolGeometry`
# addressing: only the two plane tensors and `pool_plan.entries` are stubbed,
# so the geometry does its own arithmetic and `physical_slot` its own bound
# check. No GPU allocation is involved — the planes are ordinary CPU tensors.

V4_RATIOS = [0, 4, 128]  # one dense, one CSA, one HCA layer
V4_BLOCK_SIZE = 256
V4_RING_SLOTS = 8
V4_ARENA_ROWS = 4
V4_NOPE_WIDTH, V4_ROPE_WIDTH = 8, 4
V4_GROUPS, V4_STAGING = 4, 2  # admission groups, plus the staging ring


def build_v4(num_groups: int = V4_GROUPS, staging: int = V4_STAGING):
    """A V4 builder whose pool was *allocated* for `num_groups + staging`.

    That sum is what `PoolPlan.entries[STATE_SLOT_CLASS]` holds — the
    allocation count, not the admission count the BlockManager leases from.
    """
    num_slots = num_groups + staging
    geo = UnifiedPoolGeometry(
        V4_RATIOS,
        num_blocks=2,
        num_slots=num_slots,
        ring_slots=V4_RING_SLOTS,
        block_size=V4_BLOCK_SIZE,
        arena_rows=V4_ARENA_ROWS,
    )
    nope = torch.arange(geo.plane_rows * V4_NOPE_WIDTH, dtype=torch.float32).reshape(
        geo.plane_rows, V4_NOPE_WIDTH
    )
    rope = torch.zeros(geo.plane_rows, V4_ROPE_WIDTH)

    builder = object.__new__(DeepseekV4AttentionMetadataBuilder)
    builder.pool_geometry = geo
    builder._slot_view_cache = None
    builder.head_dim = V4_NOPE_WIDTH
    builder.rope_head_dim = V4_ROPE_WIDTH
    builder._classical_dtype = torch.float32
    builder._rope_dtype = torch.float32
    builder._kv_fp8 = True  # two planes, which is the harder shape
    builder.model_runner = SimpleNamespace(
        pool_plan=SimpleNamespace(entries={STATE_SLOT_CLASS: num_slots}),
        v4_kv_plane=nope,
        v4_kv_plane_rope=rope,
    )
    return builder, nope, rope, geo


def test_v4_every_view_is_contiguous():
    """One view per plane, each a whole slot, each contiguous."""
    builder, _, _, _ = build_v4()
    views = builder.state_entry_views(1)
    assert len(views) == 2  # NoPE and RoPE
    assert all(v.is_contiguous() for v in views)


def test_v4_views_are_the_whole_slot_and_alias_the_planes():
    builder, nope, rope, geo = build_v4()
    views = builder.state_entry_views(1)
    assert [tuple(v.shape) for v in views] == [
        (geo.slot_rows, V4_NOPE_WIDTH),
        (geo.slot_rows, V4_ROPE_WIDTH),
    ]
    for view in views:
        view.fill_(7.0)
    start, stop = geo.slot_span(geo.physical_slot(1))
    assert torch.all(nope[start:stop] == 7.0)
    assert torch.all(rope[start:stop] == 7.0)
    assert not torch.all(nope[:start] == 7.0)


def test_v4_a_staging_group_past_the_admission_count_still_resolves():
    """Group `V4_GROUPS` is in the staging ring, past what admission leases.

    It resolves because `num_state_slots` is the *allocation* count
    (`entries[STATE_SLOT_CLASS]`, staging groups included) — the same entry
    `allocate_per_req_cache` hands `with_capacity`, so `_slot_views`' span and
    `physical_slot`'s bound cover the staging groups too. Make
    `num_state_slots` the admission count instead and this raises IndexError
    here rather than crashing the offload path at runtime.
    """
    builder, _, _, _ = build_v4()
    in_pool = builder.state_entry_views(V4_GROUPS - 1)
    staging = builder.state_entry_views(V4_GROUPS)
    assert all(v.is_contiguous() for v in staging)
    assert [tuple(v.shape) for v in staging] == [tuple(v.shape) for v in in_pool]
    assert [v.numel() for v in staging] == [v.numel() for v in in_pool]
    # A distinct slot, not an alias of the last admitted one.
    assert staging[0].data_ptr() != in_pool[0].data_ptr()
    # The whole ring is addressable, right up to the last staging group.
    assert builder.state_entry_views(V4_GROUPS + V4_STAGING - 1)
