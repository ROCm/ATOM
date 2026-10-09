# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Row chunking for the dense fp8_mqa_logits indexer buffer.

The chunk size is not just a memory-pressure knob: aiter's fp8_mqa_logits
switches to a USE_BUFFER_STORE=False specialization once the logits tensor
reaches 2 GiB, and that specialization does not compile on gfx950 -- every TP
rank abort()s inside the Triton AMDGCN backend. So the cap is a hard one and the
bound is exclusive.
"""

import pytest

from atom.model_ops.sparse_indexer_chunk import (
    _BUFFER_DESCRIPTOR_LIMIT_BYTES,
    sparse_indexer_row_chunk,
)

_DEFAULT_BUDGET_MB = 2048


def _logits_bytes(rows: int, row_width: int) -> int:
    return rows * row_width * 4


@pytest.mark.parametrize(
    "total_rows,row_width",
    [
        # The live GLM-5.2 crash: 16384 rows x 32768 committed tokens x 4 B is
        # 2 GiB to the byte, which the old `budget_bytes // (width * 4) < rows`
        # gate let through un-chunked.
        (16384, 32768),
        (16384, 32513),  # just under the 256-column alignment step
        (16384, 32769),  # just over
        (8192, 65536),
        (16384, 16384),  # comfortably inside
        (128, 4 * 1024 * 1024),  # below one 128-row tile
    ],
)
def test_chunk_keeps_the_buffer_strictly_under_the_2gib_cap(total_rows, row_width):
    rows = sparse_indexer_row_chunk(total_rows, row_width, _DEFAULT_BUDGET_MB)
    assert 0 < rows <= total_rows
    assert _logits_bytes(rows, row_width) < _BUFFER_DESCRIPTOR_LIMIT_BYTES


def test_a_single_shot_that_fits_is_left_alone():
    assert sparse_indexer_row_chunk(4096, 8192, _DEFAULT_BUDGET_MB) == 4096


def test_the_hard_cap_survives_a_disabled_soft_budget():
    # 0 turns off the user's byte budget, not the buffer-descriptor limit.
    rows = sparse_indexer_row_chunk(16384, 32768, 0)
    assert rows < 16384
    assert _logits_bytes(rows, 32768) < _BUFFER_DESCRIPTOR_LIMIT_BYTES


def test_the_hard_cap_survives_an_oversized_soft_budget():
    rows = sparse_indexer_row_chunk(16384, 32768, 64 * 1024)
    assert _logits_bytes(rows, 32768) < _BUFFER_DESCRIPTOR_LIMIT_BYTES


def test_a_tighter_soft_budget_still_wins():
    rows = sparse_indexer_row_chunk(16384, 32768, 256)
    assert _logits_bytes(rows, 32768) <= 256 * 1024 * 1024


def test_chunks_land_on_the_kernels_128_row_tile():
    rows = sparse_indexer_row_chunk(16384, 32768, _DEFAULT_BUDGET_MB)
    assert rows % 128 == 0


def test_below_one_tile_it_degrades_by_powers_of_two_not_to_one():
    # A row so wide that the budget affords only 63 of them (8 MiB x 4 B each).
    rows = sparse_indexer_row_chunk(4096, 8 * 1024 * 1024, _DEFAULT_BUDGET_MB)
    assert rows == 32
    assert _logits_bytes(rows, 8 * 1024 * 1024) < _BUFFER_DESCRIPTOR_LIMIT_BYTES


def test_degenerate_shapes_do_not_produce_a_zero_stride():
    # range(0, n, 0) would raise; the loop must always make progress.
    assert sparse_indexer_row_chunk(0, 1024, _DEFAULT_BUDGET_MB) >= 1
    assert sparse_indexer_row_chunk(1024, 0, _DEFAULT_BUDGET_MB) >= 1
