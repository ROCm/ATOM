# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Row chunking for the dense fp8_mqa_logits indexer buffer.

The chunk size is not just a memory-pressure knob: aiter's fp8_mqa_logits
switches to a USE_BUFFER_STORE=False specialization once the logits tensor
reaches 2 GiB, and that specialization does not compile on gfx950 -- every TP
rank abort()s inside the Triton AMDGCN backend. So the cap is a hard one and the
bound is exclusive.
"""

import numpy as np
import pytest
import torch

from atom.model_ops.sparse_indexer_chunk import (
    _BUFFER_DESCRIPTOR_LIMIT_BYTES,
    ConcatKeyWorkspace,
    LogitsWorkspace,
    capped_logits_rows,
    concat_plane_width,
    concat_window_bands,
    decode_band_rows,
    flat_view,
    indexer_qsplit_world,
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


def test_concat_plane_width_is_aiter_s_row_pitch():
    """A concat band's rows sit `CONCAT_PLANE_ROW_ALIGN` columns apart, the
    pitch aiter pads its own plane to."""
    assert concat_plane_width(1) == 256
    assert concat_plane_width(256) == 256
    assert concat_plane_width(257) == 512
    assert concat_plane_width(359_768) == 359_936


def _check_window_bands(first, end, row_cu, key_cu, plane, bands, windows):
    """The bands tile `[first, end)`; each fits the plane at its window's pitch,
    and its window holds every key its rows' requests have."""
    assert bands[0][0] == first and bands[-1][1] == end
    assert all(a < b == c for (a, b), (c, _) in zip(bands, bands[1:] + ((end, 0),)))
    for (a, b), (lo, hi) in zip(bands, windows):
        assert (b - a) * concat_plane_width(hi - lo) <= plane.capacity
        for row in range(a, b):
            request = np.searchsorted(row_cu, row, side="right") - 1
            assert lo <= key_cu[request] and key_cu[request + 1] <= hi


@pytest.mark.parametrize(
    "rows, keys, capacity, expected",
    [
        # Everything fits one row's plane: one band against every key.
        ([3, 2], [100, 50], 4096, [((0, 5), (0, 150))]),
        # Two requests together outgrow the plane: one band each.
        ([4, 4], [300, 300], 3072, [((0, 4), (0, 300)), ((4, 8), (300, 600))]),
        # One request's rows outgrow the plane: cut, each piece its window.
        ([5], [200], 512, [((0, 2), (0, 200)), ((2, 4), (0, 200)), ((4, 5), (0, 200))]),
        # An empty request (a PCP rank owning none of its rows) widens a window.
        ([2, 0, 2], [100, 100, 100], 4096, [((0, 4), (0, 300))]),
    ],
)
def test_concat_window_bands(rows, keys, capacity, expected):
    from itertools import accumulate

    row_cu = np.array(list(accumulate(rows, initial=0)))
    key_cu = np.array(list(accumulate(keys, initial=0)))
    plane = LogitsWorkspace(capacity, "cpu")
    bands, windows = concat_window_bands(0, row_cu[-1], row_cu, key_cu, plane)
    assert list(zip(bands, windows)) == expected
    _check_window_bands(0, row_cu[-1], row_cu, key_cu, plane, bands, windows)


def test_concat_window_bands_fit_a_one_row_plane():
    """1M context, 8 MiB budget: the plane holds one 1M-key row, two requests'
    keys together do not; each row is scored against its own request's."""
    width = concat_plane_width(1 << 20)
    plane = LogitsWorkspace(width, "cpu")
    row_cu, key_cu = np.array([0, 3, 6]), np.array([0, 525_312, 1_050_624])
    bands, windows = concat_window_bands(0, 6, row_cu, key_cu, plane)
    assert len(bands) == 6
    _check_window_bands(0, 6, row_cu, key_cu, plane, bands, windows)


@pytest.mark.parametrize("rank", range(8))
def test_concat_window_bands_cover_a_dcp_rank_of_many_cached_requests(rank):
    """DCP8, 32K context, 256 batched tokens: 64 requests each adding 4 rows
    to a 30,720-key prefix fill 1.97M columns, the 32-row plane 1M."""
    plane = LogitsWorkspace(32 * concat_plane_width(32_768), "cpu")
    row_cu = np.arange(65) * 4
    key_cu = np.arange(65) * (30_720 + 4)
    first, end = rank * 32, rank * 32 + 32
    bands, windows = concat_window_bands(first, end, row_cu, key_cu, plane)
    _check_window_bands(first, end, row_cu, key_cu, plane, bands, windows)


def test_concat_window_bands_refuse_a_request_wider_than_the_plane():
    plane = LogitsWorkspace(512, "cpu")
    with pytest.raises(ValueError, match="no row of request 1"):
        concat_window_bands(0, 2, np.array([0, 1, 2]), np.array([0, 10, 600]), plane)


def test_concat_key_workspace_hands_out_views_of_held_buffers():
    """The gathered keys come back as views of buffers held across steps: a
    step that fits takes no allocation, one past them doubles them once."""
    ws = ConcatKeyWorkspace(64, 2, torch.uint8, "cpu", gathered_capacity=96)
    values, scales = ws.keys(40)
    assert values.shape == (40, 2) and scales.shape == (40, 1)
    held = values.data_ptr()
    # A long prefill's chunks: more keys every step, never a new buffer.
    for total in range(8, 65, 8):
        assert ws.keys(total)[0].data_ptr() == held
    grown, grown_scales = ws.keys(65)
    assert grown.shape == (65, 2) and grown_scales.shape == (65, 1)
    assert ws.keys(128)[0].data_ptr() == grown.data_ptr()
    shards, shard_scales = ws.gathered(96)
    assert shards.shape == (96, 2) and shard_scales.dtype == torch.float32
    assert ws.nbytes == 128 * 2 + 128 * 4 + 96 * 2 + 96 * 4


# ---- DeepSeek-V4 prefill indexer split by query row across plain TP ---------


_GATE = {
    "physical_tp": 8,
    "enabled": True,
    "gfx1250_opus": False,
    "dp_attention": False,
    "tbo": False,
    "pcp": 1,
    "dcp": 1,
    "kv_consumer": False,
}


def test_qsplit_on_for_plain_tp():
    assert indexer_qsplit_world(8, **_GATE) == 8


@pytest.mark.parametrize(
    "tp, override",
    [
        (1, {"physical_tp": 1}),
        (8, {"physical_tp": 4}),  # simulated TP: the device group is narrower
        (8, {"dp_attention": True}),
        (8, {"pcp": 2}),
        (8, {"dcp": 2}),
    ],
)
def test_qsplit_off_where_ranks_do_not_share_the_rows(tp, override):
    assert indexer_qsplit_world(tp, **{**_GATE, **override}) == 1


@pytest.mark.parametrize("override", [{"tbo": True}, {"gfx1250_opus": True}])
def test_qsplit_not_yet_under_tbo_or_gfx1250_opus(override):
    """Known limitation: the ranks share these rows, but a TBO ubatch's
    rows and gfx1250 OPUS's plans do not take the split yet."""
    assert indexer_qsplit_world(8, **{**_GATE, **override}) == 1


def test_qsplit_off_on_a_pd_decode_instance():
    """A trade-off: a decode instance only recomputes prefills, so it keeps
    them whole rather than pay for the split's startup gather."""
    assert indexer_qsplit_world(8, **{**_GATE, "kv_consumer": True}) == 1


def test_qsplit_off_with_the_flag():
    assert indexer_qsplit_world(8, **{**_GATE, "enabled": False}) == 1
