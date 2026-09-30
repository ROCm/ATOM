# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for the vLLM plugin's PAGE layout published to LMCache MP.

The plugin owns no allocation: vLLM does. What it publishes is therefore an
*alias* of vLLM's KV tensors, and every property LMCache MP relies on -- block
count, unit bytes, stride, zero-copy identity -- is a property of that alias
rather than of anything the plugin allocated. These tests pin the alias.
"""

from __future__ import annotations

import pytest
import torch

from atom.config import KVCacheTensor
from atom.kv_transfer.offload.mp.page_views import (
    _build_cache_views,
    validate_page_views,
)
from atom.plugin.vllm.kv_transfer.kda_state import KdaPageViews
from atom.plugin.vllm.kv_transfer.mp_page_layout import (
    _NON_PAGE_ROLES,
    _PAGE_ROLES,
    build_mp_recurrent_groups,
    build_mp_transfer_tensors,
    summarize_layout,
)

_NUM_BLOCKS = 64
_BLOCK_SIZE = 16
# GLM-5.2 shapes: one fused MLA latent plane plus an FP8 index plane per layer.
_LATENT_WIDTH = 576
_INDEX_WIDTH = 132


def _layer(layer_num: int, *, rows: int = _NUM_BLOCKS * _BLOCK_SIZE) -> KVCacheTensor:
    return KVCacheTensor(
        layer_num=layer_num,
        k_cache=torch.zeros((rows, _LATENT_WIDTH), dtype=torch.bfloat16),
        # MLA fuses K and V into the latent; vLLM publishes an empty V plane.
        v_cache=torch.tensor([]),
        index_cache=torch.zeros((rows, _INDEX_WIDTH), dtype=torch.uint8),
    )


def _layers(count: int = 4) -> list[KVCacheTensor]:
    return [_layer(i) for i in range(count)]


def test_roles_cover_every_kv_cache_tensor_field():
    """A new plane on KVCacheTensor must not be silently left behind.

    Publishing a partial layout is not a transport error: LMCache stores and
    restores exactly the planes it was given, so an unlisted plane produces a
    prefix that reads back with stale bytes under a valid hash.
    """
    declared = set(KVCacheTensor.__dataclass_fields__) - {"layer_num"}
    accounted = (set(_PAGE_ROLES) | set(_NON_PAGE_ROLES)) - {"layer_num"}
    assert declared - accounted == set()


def test_published_layout_passes_lmcache_mp_validation():
    tensors = build_mp_transfer_tensors(_layers(), num_blocks=_NUM_BLOCKS)
    views = validate_page_views(tensors, num_blocks=_NUM_BLOCKS, block_size=_BLOCK_SIZE)
    assert len(views) == 8  # 4 layers x (latent, index)
    cache_views = _build_cache_views(tensors, num_blocks=_NUM_BLOCKS)
    # Planes group by (dtype, trailing shape), so the bf16 latents form one
    # copy-kernel group and the uint8 index planes the other.
    assert sorted(len(g) for g in cache_views.layer_groups) == [4, 4]
    assert cache_views.bytes_per_block == 4 * (
        _BLOCK_SIZE * _LATENT_WIDTH * 2 + _BLOCK_SIZE * _INDEX_WIDTH
    )


def test_published_views_alias_the_vllm_tensors():
    """Zero-copy, or LMCache would restore into a buffer nobody reads."""
    layers = _layers(1)
    tensors = build_mp_transfer_tensors(layers, num_blocks=_NUM_BLOCKS)
    view = validate_page_views(tensors, num_blocks=_NUM_BLOCKS)[0].view
    view[3].fill_(0xA5)
    source = layers[0].k_cache
    touched = source[3 * _BLOCK_SIZE : 4 * _BLOCK_SIZE]
    assert not torch.equal(touched, torch.zeros_like(touched))
    neighbour = source[4 * _BLOCK_SIZE : 5 * _BLOCK_SIZE]
    assert torch.equal(neighbour, torch.zeros_like(neighbour))


def test_mla_leading_dimension_skew_is_absorbed_by_unit_bytes():
    """ATOM's MLA backend asks vLLM for kernel block size 1.

    vLLM then allocates one row per token, so ``shape[0]`` is ``block_size``
    times the block count. Sizing units from the byte total rather than from
    the leading dimension is what keeps the published geometry correct.
    """
    tensors = build_mp_transfer_tensors(_layers(1), num_blocks=_NUM_BLOCKS)
    region = tensors.block_regions[0]
    assert region.unit_bytes == _BLOCK_SIZE * _LATENT_WIDTH * 2
    assert region.total_bytes == _NUM_BLOCKS * region.unit_bytes


def test_ragged_leading_dimension_is_refused():
    """Byte divisibility alone does not catch a wrong leading dimension.

    1025 rows of 1152 B over 64 blocks still divides into a whole 18450 B
    unit, and every later check -- ``set_block_count``, ``validate_page_views``
    -- is expressed in those same bytes and would agree. The stride would just
    be wrong by less than one row per block.
    """
    ragged = [_layer(0, rows=_NUM_BLOCKS * _BLOCK_SIZE + 1)]
    with pytest.raises(ValueError, match="leading dimension"):
        build_mp_transfer_tensors(ragged, num_blocks=_NUM_BLOCKS)


def test_per_request_state_is_refused_rather_than_skipped():
    """Recurrent state is slot-addressed; no block-addressed mover may take it.

    Skipping the entry would publish that layer's KV as absent, and a restore
    would then resume a recurrent layer from whatever its slot last held.
    """
    layers = _layers(1)
    layers[0].per_request_state = True
    with pytest.raises(ValueError, match="per-request recurrent state"):
        build_mp_transfer_tensors(layers, num_blocks=_NUM_BLOCKS)


def test_summary_reports_what_was_published():
    tensors = build_mp_transfer_tensors(_layers(), num_blocks=_NUM_BLOCKS)
    summary = summarize_layout(tensors)
    assert "8" in summary and str(_NUM_BLOCKS) in summary


# --- recurrent groups -------------------------------------------------------

_STATE_BLOCKS = 5
_STATE_WIDTH = 24
_TOKENS_PER_BLOCK = 1536


def _state_views(*, groups: int = 3, layers: int = 2, blocks: int = _STATE_BLOCKS):
    """A KdaPageViews over `groups` mamba groups, K3's shape in miniature."""
    return KdaPageViews(
        [
            [
                torch.zeros((blocks, _STATE_WIDTH), dtype=torch.bfloat16)
                for _ in range(layers)
            ]
            for _ in range(groups)
        ],
        layout_id="test",
    )


def test_one_engine_group_per_mamba_group_in_gather_order():
    """Group order is the key space, so it has to be the codec's own order."""
    views = _state_views()
    recurrent = build_mp_recurrent_groups(views, tokens_per_block=_TOKENS_PER_BLOCK)
    assert len(recurrent) == views.num_groups
    for group, tensors in zip(recurrent, views.groups):
        assert len(group.pages) == len(tensors)
        assert group.num_blocks == _STATE_BLOCKS
        assert group.tokens_per_block == _TOKENS_PER_BLOCK


def test_pages_alias_the_state_tensors_without_copying():
    """The server writes through these addresses; a copy would restore nothing."""
    views = _state_views(groups=1, layers=1)
    (group,) = build_mp_recurrent_groups(views, tokens_per_block=_TOKENS_PER_BLOCK)
    (page,) = group.pages
    plane = views.groups[0][0]
    assert page.view.data_ptr() == plane.data_ptr()
    assert page.region.unit_bytes == plane[0].numel() * plane.element_size()


def test_a_recurrent_block_is_one_snapshot_not_one_token():
    """Sized off its own leading axis, never off the attention block count.

    The whole reason recurrent groups are published apart is that they count
    in a different block id space; sizing a unit from the PAGE block count
    would move a fraction of each snapshot and pass every later check.
    """
    views = _state_views(groups=1, layers=1, blocks=_STATE_BLOCKS)
    (group,) = build_mp_recurrent_groups(views, tokens_per_block=_TOKENS_PER_BLOCK)
    assert group.num_blocks == _STATE_BLOCKS != _NUM_BLOCKS


def test_a_non_block_major_state_plane_is_refused():
    views = _state_views(groups=1, layers=1)
    views.groups[0][0] = torch.zeros((_STATE_BLOCKS + 1, _STATE_WIDTH))
    with pytest.raises(ValueError, match="not the block axis"):
        build_mp_recurrent_groups(views, tokens_per_block=_TOKENS_PER_BLOCK)


@pytest.mark.parametrize("bad", [0, -1, 1.0, None])
def test_a_non_positive_token_span_is_refused(bad):
    """A snapshot's row count says nothing about how many tokens it covers."""
    with pytest.raises(ValueError, match="tokens_per_block"):
        build_mp_recurrent_groups(_state_views(), tokens_per_block=bad)


def test_summary_names_the_recurrent_groups_when_there_are_any():
    tensors = build_mp_transfer_tensors(_layers(), num_blocks=_NUM_BLOCKS)
    assert "recurrent" not in summarize_layout(tensors)
    object.__setattr__(
        tensors,
        "recurrent_page_groups",
        build_mp_recurrent_groups(_state_views(), tokens_per_block=_TOKENS_PER_BLOCK),
    )
    summary = summarize_layout(tensors)
    assert "3 recurrent group(s)" in summary
    assert str(_TOKENS_PER_BLOCK) in summary
