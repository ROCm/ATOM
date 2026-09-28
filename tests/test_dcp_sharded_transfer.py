# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import pytest

from atom.distributed.dcp_layout import (
    dcp_global_pos,
    dcp_local_index,
    dcp_owner_rank,
)
from atom.kv_transfer.disaggregation.sharded_transfer import (
    build_dcp_shard_plan,
    coalesce_contiguous,
)


def _expand_runs(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, start + length) for start, length in zip(starts, lengths)]
    )


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_dcp_layout_helpers_round_trip(dcp_size, interleave):
    for pos in range(dcp_size * interleave * 8):
        rank = dcp_owner_rank(pos, dcp_size, interleave)
        local = dcp_local_index(pos, dcp_size, interleave)
        assert dcp_global_pos(local, rank, dcp_size, interleave) == pos


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
        global_token = dcp_global_pos(local_token, dcp_rank, dcp_size, interleave)
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


@pytest.mark.parametrize("page_bytes", [16 * 576, 64 * 576, 64 * 1152])
@pytest.mark.parametrize("boundary", ["source", "destination", "both"])
def test_coalesce_contiguous_splits_registered_mr_boundaries(page_bytes, boundary):
    max_chunk = 2 * 1024**3 - 64 * 1024
    mr_bytes = max_chunk - max_chunk % page_bytes
    src_base, dst_base = 2**40 + 128, 2**42 + 256
    src_start = src_base + (mr_bytes - page_bytes if boundary != "destination" else 0)
    dst_start = dst_base + (mr_bytes - page_bytes if boundary != "source" else 0)
    src, dst, length = coalesce_contiguous(
        src_start + np.arange(3, dtype=np.int64) * page_bytes,
        dst_start + np.arange(3, dtype=np.int64) * page_bytes,
        np.full(3, page_bytes, dtype=np.int64),
        src_mr=(src_base, mr_bytes),
        dst_mr=(dst_base, mr_bytes),
    )
    np.testing.assert_array_equal(src, [src_start, src_start + page_bytes])
    np.testing.assert_array_equal(dst, [dst_start, dst_start + page_bytes])
    np.testing.assert_array_equal(length, [page_bytes, 2 * page_bytes])
    for addresses, base in ((src, src_base), (dst, dst_base)):
        np.testing.assert_array_equal(
            (addresses - base) // mr_bytes,
            (addresses + length - 1 - base) // mr_bytes,
        )


def test_coalesce_splits_single_run_at_different_source_and_destination_boundaries():
    src, dst, length = coalesce_contiguous(
        np.array([103]),
        np.array([208]),
        np.array([23]),
        src_mr=(100, 10),
        dst_mr=(200, 12),
    )
    # Splits inside the original run must preserve every byte in order.
    np.testing.assert_array_equal(_expand_runs(src, length), np.arange(103, 126))
    np.testing.assert_array_equal(_expand_runs(dst, length), np.arange(208, 231))
    for addresses, base, chunk in ((src, 100, 10), (dst, 200, 12)):
        np.testing.assert_array_equal(
            (addresses - base) // chunk, (addresses + length - 1 - base) // chunk
        )


def test_coalesce_with_mr_boundaries_preserves_empty_input_and_address_gaps():
    src, dst, length = coalesce_contiguous(
        np.array([100, 104, 120]),
        np.array([200, 204, 220]),
        np.array([4, 4, 4]),
        src_mr=(100, 64),
        dst_mr=(200, 64),
    )
    np.testing.assert_array_equal(src, [100, 120])
    np.testing.assert_array_equal(dst, [200, 220])
    np.testing.assert_array_equal(length, [8, 4])
    empty = np.empty(0, dtype=np.int64)
    assert all(
        a.size == 0
        for a in coalesce_contiguous(
            empty, empty, empty, src_mr=(100, 64), dst_mr=(200, 64)
        )
    )


@pytest.mark.parametrize("boundary", ["source", "destination", "both"])
@pytest.mark.parametrize("seed", range(8))
def test_coalesce_mr_splits_preserve_randomized_bytes_and_are_maximal(boundary, seed):
    rng = np.random.default_rng(seed)
    # Keep addresses beyond float64's exact-integer range: address arithmetic
    # must stay integral even when expanding thousands of MR fragments.
    src_base, dst_base = 2**54 + 37, 2**55 + 91
    count = 48
    lengths = rng.integers(0, 65, count, dtype=np.int64)
    lengths[0] = 4096 if seed % 2 else 0
    src_gaps = rng.choice([0, 0, 0, 7], count - 1)
    dst_gaps = rng.choice([0, 0, 0, 11], count - 1)
    src = src_base + 3 + np.concatenate(([0], np.cumsum(lengths[:-1] + src_gaps)))
    dst = dst_base + 5 + np.concatenate(([0], np.cumsum(lengths[:-1] + dst_gaps)))
    if seed % 2:
        # Destination descriptors need not be ordered by physical address.
        dst = dst[rng.permutation(count)]
    src_mr = (src_base, [1, 7, 32, 127][seed % 4])
    dst_mr = (dst_base, [1, 11, 32, 65][seed % 4])
    if boundary == "source":
        dst_mr = None
    elif boundary == "destination":
        src_mr = None
    actual_src, actual_dst, actual_length = coalesce_contiguous(
        src, dst, lengths, src_mr=src_mr, dst_mr=dst_mr
    )
    assert (actual_length > 0).all()
    np.testing.assert_array_equal(
        _expand_runs(actual_src, actual_length), _expand_runs(src, lengths)
    )
    np.testing.assert_array_equal(
        _expand_runs(actual_dst, actual_length), _expand_runs(dst, lengths)
    )
    # No output descriptor may cross either MR, and adjacent outputs should
    # only remain separate because of an address gap or an MR boundary.
    can_merge = (actual_src[1:] == actual_src[:-1] + actual_length[:-1]) & (
        actual_dst[1:] == actual_dst[:-1] + actual_length[:-1]
    )
    for addresses, mr in ((actual_src, src_mr), (actual_dst, dst_mr)):
        if mr is None:
            continue
        base, chunk = mr
        start_mr = (addresses - base) // chunk
        end_mr = (addresses + actual_length - 1 - base) // chunk
        np.testing.assert_array_equal(start_mr, end_mr)
        can_merge &= start_mr[:-1] == end_mr[1:]
    assert not can_merge.any()


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
@pytest.mark.parametrize("lengths", [[0], [0, 0, 0], [0, 8, 0], [8, 0, 8], [16, 8, 24]])
def test_coalesce_mr_split_handles_zero_lengths_and_exact_boundaries(dtype, lengths):
    lengths = np.array(lengths, dtype=dtype)
    # Strided inputs also exercise the public ndarray contract.
    src = np.arange(100, 100 + 16 * lengths.size, 8, dtype=dtype)[::2]
    dst = np.arange(200, 200 + 16 * lengths.size, 8, dtype=dtype)[::2]
    actual = coalesce_contiguous(src, dst, lengths, src_mr=(100, 8), dst_mr=(200, 16))
    assert all(a.dtype == np.int64 for a in actual)
    if not lengths.any():
        assert all(a.size == 0 for a in actual)
        return
    actual_src, actual_dst, actual_length = actual
    assert (actual_length == 8).all()
    np.testing.assert_array_equal(
        _expand_runs(actual_src, actual_length), _expand_runs(src, lengths)
    )
    np.testing.assert_array_equal(
        _expand_runs(actual_dst, actual_length), _expand_runs(dst, lengths)
    )


@pytest.mark.parametrize("side", ["src_mr", "dst_mr"])
@pytest.mark.parametrize("chunk", [0, -1])
def test_coalesce_rejects_invalid_mr_chunks_even_for_empty_input(side, chunk):
    empty = np.empty(0, dtype=np.int64)
    with pytest.raises(ValueError, match="MR chunk bytes must be positive"):
        coalesce_contiguous(empty, empty, empty, **{side: (0, chunk)})
