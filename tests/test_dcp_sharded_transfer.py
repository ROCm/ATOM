# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import pytest

from atom.kv_transfer.disaggregation.sharded_transfer import (
    build_dcp_shard_plan,
    coalesce_contiguous,
)


def _expand_runs(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, start + length) for start, length in zip(starts, lengths)]
    )


@pytest.mark.parametrize(
    ("dcp_size", "dcp_rank"),
    [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)],
)
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_shard_plan_matches_dcp_token_ownership(dcp_size, dcp_rank, interleave):
    block_size = 16
    src_block_ids = [7, 3, 11, 5, 13]
    dst_pages = (len(src_block_ids) + dcp_size - 1) // dcp_size
    dst_block_ids = list(range(20, 20 + dst_pages))
    plan = build_dcp_shard_plan(
        src_block_ids,
        block_size=block_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        interleave_size=interleave,
    )
    assert plan.src_token.size == dst_pages * block_size // interleave
    assert (plan.run_length == interleave).all()

    src_starts, dst_starts, lengths = plan.token_runs(dst_block_ids)
    actual_src = _expand_runs(src_starts, lengths)
    actual_dst = _expand_runs(dst_starts, lengths)

    expected_src = []
    expected_dst = []
    for local_token in range(dst_pages * block_size):
        global_token = (
            (local_token // interleave) * dcp_size + dcp_rank
        ) * interleave + local_token % interleave
        src_ordinal, src_token = divmod(global_token, block_size)
        if src_ordinal >= len(src_block_ids):
            continue
        dst_page, dst_token = divmod(local_token, block_size)
        expected_src.append(src_block_ids[src_ordinal] * block_size + src_token)
        expected_dst.append(dst_block_ids[dst_page] * block_size + dst_token)

    np.testing.assert_array_equal(actual_src, expected_src)
    np.testing.assert_array_equal(actual_dst, expected_dst)


def test_shard_plan_page_slice_preserves_source_mapping_and_rebases_destination():
    plan = build_dcp_shard_plan(
        [9, 4, 12, 7, 15, 2, 18, 5, 21],
        block_size=16,
        dcp_size=4,
        dcp_rank=3,
    )

    sliced = plan.slice_pages(1, 3)
    row_start = plan.block_size
    row_stop = 3 * plan.block_size

    assert sliced.dst_pages == 2
    np.testing.assert_array_equal(
        sliced.src_block_id_per_run,
        plan.src_block_id_per_run[row_start:row_stop],
    )
    np.testing.assert_array_equal(sliced.src_token, plan.src_token[row_start:row_stop])
    np.testing.assert_array_equal(
        sliced.dst_page, plan.dst_page[row_start:row_stop] - 1
    )
    np.testing.assert_array_equal(
        sliced.run_length, plan.run_length[row_start:row_stop]
    )
    np.testing.assert_array_equal(sliced.valid, plan.valid[row_start:row_stop])


def test_shard_plan_keeps_only_valid_part_of_trailing_virtual_page():
    plan = build_dcp_shard_plan(
        [0, 1, 2, 3, 4],
        block_size=16,
        dcp_size=4,
        dcp_rank=3,
    )

    assert plan.dst_pages == 2
    assert plan.valid[:16].all()
    assert plan.valid[16:20].all()
    assert not plan.valid[20:].any()


def test_shard_plan_validates_interleave_geometry():
    with pytest.raises(ValueError, match="must divide"):
        build_dcp_shard_plan(
            [0],
            block_size=16,
            dcp_size=4,
            dcp_rank=0,
            interleave_size=3,
        )


def test_coalesce_contiguous_preserves_address_gaps():
    src, dst, length = coalesce_contiguous(
        np.array([100, 104, 200], dtype=np.int64),
        np.array([300, 304, 500], dtype=np.int64),
        np.array([4, 4, 8], dtype=np.int64),
    )

    np.testing.assert_array_equal(src, [100, 200])
    np.testing.assert_array_equal(dst, [300, 500])
    np.testing.assert_array_equal(length, [8, 8])
