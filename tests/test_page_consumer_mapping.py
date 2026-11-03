# SPDX-License-Identifier: MIT
"""A PP stage's appended draft starts after the full consumer target layout."""

import pytest

from atom.kv_transfer.disaggregation.types import (
    KVTransferRegion,
    KVTransferTensors,
    PageRegion,
)


def _tensors(n, indices=None, count=None):
    return KVTransferTensors(
        pages=[PageRegion(KVTransferRegion(0, 128, 64)) for _ in range(n)],
        block_region_consumer_indices=indices,
        block_region_consumer_count=count,
    )


def test_draft_starts_after_all_target_regions_not_local_stage_regions():
    target = _tensors(2, [3, 7], 12)
    target.merge_pages(_tensors(4))
    assert target.block_region_consumer_indices == [3, 7, 12, 13, 14, 15]
    assert target.block_region_consumer_count == 16
    assert len(target.pages) == 6
    target.merge_pages(_tensors(1))
    assert target.block_region_consumer_indices[-1] == 16
    assert target.block_region_consumer_count == 17


@pytest.mark.parametrize("mapped_target", [False, True])
def test_mapped_contributor_keeps_its_own_global_offsets(mapped_target):
    target = _tensors(2, [3, 7], 12) if mapped_target else _tensors(2)
    target.merge_pages(_tensors(2, [1, 3], 5))
    expected = [3, 7, 13, 15] if mapped_target else [0, 1, 3, 5]
    assert target.block_region_consumer_indices == expected
    assert target.block_region_consumer_count == (17 if mapped_target else 7)


@pytest.mark.parametrize("missing_count_on_target", [False, True])
def test_incomplete_map_is_rejected_before_either_contributor_changes(
    missing_count_on_target,
):
    target = _tensors(2, [3, 7], None if missing_count_on_target else 12)
    draft = _tensors(1) if missing_count_on_target else _tensors(1, [2])
    with pytest.raises(ValueError, match="full consumer count"):
        target.merge_pages(draft)
    assert len(target.pages) == 2
    assert target.block_region_consumer_indices == [3, 7]
    assert len(draft.pages) == 1


def test_unmapped_and_empty_merges_keep_existing_contract():
    target = _tensors(2)
    target.merge_pages(_tensors(1))
    assert len(target.pages) == 3
    assert target.block_region_consumer_indices is None
    mapped = _tensors(2, [3, 7], 12)
    mapped.merge_pages(_tensors(0))
    assert mapped.block_region_consumer_indices == [3, 7]
    assert mapped.block_region_consumer_count == 12
