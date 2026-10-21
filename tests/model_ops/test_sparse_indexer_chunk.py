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
import torch

from atom.model_ops.sparse_indexer_chunk import (
    _BUFFER_DESCRIPTOR_LIMIT_BYTES,
    capped_logits_rows,
    decode_band_rows,
    flat_view,
    plane_rows,
    sparse_indexer_row_chunk,
)
from atom.utils import envs

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


# --- Decode bands: whole requests of `next_n` rows into one shared workspace ---

_GLM52_WIDTH = 1 << 20  # max_model_len


@pytest.fixture
def budget(monkeypatch):
    def set_mb(mb):
        monkeypatch.setattr(envs, "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", mb)

    set_mb(_DEFAULT_BUDGET_MB)
    return set_mb


@pytest.mark.parametrize("next_n", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("bs", [1, 7, 96, 129, 512])
@pytest.mark.parametrize("width", [8192, 131072, _GLM52_WIDTH, 3 << 20])
@pytest.mark.parametrize("budget_mb", [0, 64, 2048])
def test_a_decode_band_is_whole_requests_within_the_budget(
    budget, next_n, bs, width, budget_mb
):
    # A band that split a request would slice its query, lengths and block
    # table at the wrong request and silently corrupt its top-k.
    budget(budget_mb)
    rows = bs * next_n
    band = decode_band_rows(rows, width, next_n)
    assert band % next_n == 0
    assert next_n <= band <= rows
    if band > next_n:  # past one request, the plane respects both caps
        assert _logits_bytes(band, width) < _BUFFER_DESCRIPTOR_LIMIT_BYTES
        assert band <= plane_rows(width)
        if budget_mb:
            assert _logits_bytes(band, width) <= budget_mb << 20


def test_a_power_of_two_band_still_rounds_to_whole_requests(budget):
    budget(64)  # 64 MiB / 4 MiB rows -> 15 rows -> power-of-2 floor 8
    assert decode_band_rows(512 * 3, _GLM52_WIDTH, 3) == 6


def test_one_request_past_the_budget_is_one_band(budget):
    budget(1)  # one 4 MiB row is already past 1 MiB
    assert decode_band_rows(64, _GLM52_WIDTH, 4) == 4


def test_a_batch_that_fits_is_one_band(budget):
    assert decode_band_rows(32 * 4, _GLM52_WIDTH, 4) == 128


def test_glm52_mtp3_at_bs512_is_six_bands_of_96_requests(budget):
    assert decode_band_rows(512 * 4, _GLM52_WIDTH, 4) == 384


def test_the_workspace_capacity_clamps_the_band(budget):
    assert decode_band_rows(2048, _GLM52_WIDTH, 4, capacity=101) == 100
    assert decode_band_rows(2048, _GLM52_WIDTH, 3, capacity=101) == 99


@pytest.mark.parametrize("next_n", [1, 4])
def test_no_decode_rows_is_an_empty_loop_not_a_zero_step(budget, next_n):
    band = decode_band_rows(0, _GLM52_WIDTH, next_n)
    assert band == next_n
    assert list(range(0, 0, band)) == []


def test_capped_logits_rows_is_bounded_by_the_workspace(budget):
    assert capped_logits_rows(2048, _GLM52_WIDTH, 100) == 100
    assert capped_logits_rows(64, _GLM52_WIDTH, 512) == 64
    with pytest.raises(ValueError):
        capped_logits_rows(2048, _GLM52_WIDTH, 0)
    with pytest.raises(ValueError):
        capped_logits_rows(2048, _GLM52_WIDTH, 3, min_rows=4)


def test_a_workspace_below_one_request_fails(budget):
    with pytest.raises(ValueError):
        decode_band_rows(2048, _GLM52_WIDTH, 4, capacity=3)


@pytest.mark.parametrize("next_n", [1, 3, 4])
def test_the_max_batch_band_covers_every_smaller_batch(budget, next_n):
    # The builder sizes the workspace from the largest batch; every band a
    # smaller batch scores at runtime must fit in it.
    max_rows = 512 * next_n
    held = decode_band_rows(max_rows, _GLM52_WIDTH, next_n)
    for bs in range(1, 513):
        assert decode_band_rows(bs * next_n, _GLM52_WIDTH, next_n, held) <= held


def test_flat_view_shares_the_workspace_and_refuses_to_grow():
    flat = torch.zeros(10)
    view = flat_view(flat, 2, 4)
    assert view.shape == (2, 4) and view.data_ptr() == flat.data_ptr()
    view.fill_(1)
    assert flat[:8].eq(1).all() and flat[8:].eq(0).all()
    with pytest.raises(ValueError):
        flat_view(flat, 3, 4)
