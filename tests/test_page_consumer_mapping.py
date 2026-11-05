# SPDX-License-Identifier: MIT
"""Regions appended after a mapped contributor (a PP stage's draft) are
numbered from the consumer's end, where the consumer appends them too."""

import pytest

from atom.kv_transfer.disaggregation.types import (
    KVTransferRegion,
    KVTransferTensors,
    PageRegion,
)


def _tensors(n, indices=None):
    return KVTransferTensors(
        pages=[PageRegion(KVTransferRegion(0, 128, 64)) for _ in range(n)],
        block_region_consumer_indices=indices,
    )


def test_appended_regions_count_from_the_consumer_end():
    target = _tensors(2, [3, 7])
    target.merge_pages(_tensors(4))
    assert target.block_region_consumer_indices == [3, 7, -4, -3, -2, -1]
    assert len(target.pages) == 6
    # A second append is the consumer's last now; the first moves further in.
    target.merge_pages(_tensors(1))
    assert target.block_region_consumer_indices == [3, 7, -5, -4, -3, -2, -1]


def test_only_the_first_contributor_may_map_its_regions():
    target = _tensors(2, [3, 7])
    draft = _tensors(1, [2])
    with pytest.raises(ValueError, match="first PAGE contributor"):
        target.merge_pages(draft)
    assert len(target.pages) == 2
    assert target.block_region_consumer_indices == [3, 7]


def test_unmapped_and_empty_merges_keep_existing_contract():
    target = _tensors(2)
    target.merge_pages(_tensors(1))
    assert len(target.pages) == 3
    assert target.block_region_consumer_indices is None
    mapped = _tensors(2, [3, 7])
    mapped.merge_pages(_tensors(0))
    assert mapped.block_region_consumer_indices == [3, 7]
