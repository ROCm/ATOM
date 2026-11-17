# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only layout proofs for Kimi's gfx950 MFMA and gfx1250 WMMA GEMVs."""

import pytest

from atom.models.kimi_k3.mono.kernels.stages import gemv_layout as layout
from atom.mono.plan.architecture import (
    GFX950,
    GFX1250,
    Architecture,
)


@pytest.mark.parametrize("gemv", [layout.GFX950_GEMV, layout.GFX1250_GEMV])
@pytest.mark.parametrize(
    "coordinates_for", ["a_operand_coordinates", "b_operand_coordinates"]
)
def test_row_major_operand_fragments_cover_one_16x32_tile_exactly(
    gemv, coordinates_for
):
    coordinates_for = getattr(gemv, coordinates_for)
    coordinates = [
        coordinate
        for lane in range(gemv.architecture.wave_size)
        for coordinate in coordinates_for(lane, 0)
    ]
    assert len(coordinates) == layout.ROWS * layout.K_PER_MMA
    assert set(coordinates) == {
        (row, k) for row in range(layout.ROWS) for k in range(layout.K_PER_MMA)
    }


@pytest.mark.parametrize("gemv", [layout.GFX950_GEMV, layout.GFX1250_GEMV])
def test_accumulator_fragments_cover_one_16x16_tile_exactly(gemv):
    coordinates = [
        coordinate
        for lane in range(gemv.architecture.wave_size)
        for coordinate in gemv.accumulator_coordinates(lane)
    ]
    assert len(coordinates) == layout.ROWS * layout.ROWS
    assert set(coordinates) == {
        (row, token) for row in range(layout.ROWS) for token in range(layout.ROWS)
    }
    for lane in range(gemv.architecture.wave_size):
        for element, coordinate in enumerate(gemv.accumulator_coordinates(lane)):
            assert gemv.accumulator_location(*coordinate) == (lane, element)


def test_gfx1250_v16_operand_uses_alternating_k_packs():
    gemv = layout.GFX1250_GEMV
    assert gemv.operand_elements_per_lane == 16
    assert gemv.a_operand_coordinates(0, 0) == tuple(
        (0, k) for k in (*range(8), *range(16, 24))
    )
    assert gemv.a_operand_coordinates(16, 0) == tuple(
        (0, k) for k in (*range(8, 16), *range(24, 32))
    )
    assert gemv.a_operand_coordinates(31, 3) == tuple(
        (15, k) for k in (*range(104, 112), *range(120, 128))
    )


def test_gfx1250_row_major_a_and_b_use_the_same_lane_mapping():
    """AITER uses one operand loader for pa_decode's A and B row-major tiles.

    This does not claim all WMMA A/B layouts are interchangeable: FMHA's
    transpose-loaded V-as-A path is deliberately outside this contract.
    """

    gemv = layout.GFX1250_GEMV
    for lane in range(GFX1250.wave_size):
        for step in range(gemv.mma_steps_per_chunk):
            assert gemv.a_operand_coordinates(lane, step) == gemv.b_operand_coordinates(
                lane, step
            )


def test_gfx1250_v8_accumulator_lane_mapping():
    gemv = layout.GFX1250_GEMV
    assert gemv.accumulator_elements_per_lane == 8
    assert gemv.accumulator_coordinates(0) == tuple((row, 0) for row in range(8))
    assert gemv.accumulator_coordinates(16) == tuple((row, 0) for row in range(8, 16))
    assert gemv.accumulator_coordinates(31) == tuple((row, 15) for row in range(8, 16))
    assert gemv.accumulator_offset(0, 0) == 0
    assert gemv.accumulator_offset(1, 0) == 32 * 8
    assert gemv.accumulator_offset(1, 31) == 63 * 8


def test_wave_ownership_is_strided_and_complete_without_fixing_cta_size():
    gemv = layout.GFX1250_GEMV
    k, waves = 12 * layout.K_PER_CHUNK, 5
    owned = [gemv.chunk_ids(k, wave, waves) for wave in range(waves)]
    assert owned == [
        (0, 5, 10),
        (1, 6, 11),
        (2, 7),
        (3, 8),
        (4, 9),
    ]
    assert sorted(chunk for chunks in owned for chunk in chunks) == list(range(12))
    assert gemv.chunks_per_wave(k, waves) == (3, 12)


def test_layout_selection_is_explicit():
    assert layout.layout_for(GFX950) is layout.GFX950_GEMV
    assert layout.layout_for(GFX1250) is layout.GFX1250_GEMV


@pytest.mark.parametrize("rows", [1, 10, 16, 17, 40, 48])
def test_split_sums_land_past_partials_inside_the_reduction(rows):
    tiles = (rows + layout.ROWS - 1) // layout.ROWS
    accumulators = tiles * layout.TDM_SPLIT_MAX_TASKS * 8 * layout.TDM_SPLIT_TILE_WORDS
    partials = tiles * 2 * layout.TDM_SPLIT_TILE_WORDS
    sums = tiles * 2 * layout.TDM_SPLIT_TILE_WORDS
    out = layout.tdm_split_out_word(rows)
    assert out == accumulators + partials
    assert out + sums <= layout.tdm_split_post_words(rows)


def test_split_fp4_rows_spread_over_distinct_banks():
    pitch = layout.TDM_SPLIT_FP4_PITCH
    assert pitch - layout.TDM_SPLIT_FP4_ROW_WORDS == 1
    assert layout.TDM_SPLIT_FP4_ROW_WORDS * 8 == layout.TDM_SPLIT_SLAB
    for word in range(layout.TDM_SPLIT_FP4_ROW_WORDS):
        banks = {(row * pitch + word) % 64 for row in range(layout.ROWS)}
        assert len(banks) == layout.ROWS
    assert layout.tdm_split_task_words("fp4") == layout.ROWS * pitch


@pytest.mark.parametrize("fmt", layout.SPLIT_FORMATS)
def test_split_stages_are_the_most_that_fit(fmt):
    k, rows = 7168, 40
    words = {s: layout.tdm_split_lds_words(rows, k, fmt, s) for s in (2, 3)}
    for budget in (words[2] - 1, words[2], words[3] - 1, words[3]):
        fitting = [s for s in (3, 2) if words[s] <= budget]
        assert layout.tdm_split_stages(rows, k, fmt, budget) == max(fitting, default=0)


def test_split_k_quarters_whole_slabs():
    assert layout.tdm_split_slabs(7168) == 7
    with pytest.raises(ValueError, match="divisible"):
        layout.tdm_split_slabs(7168 + 256)
    assert layout.tdm_split_scratch_bytes(200, 40) == 200 * 8 * 3 * 256 * 4


def test_unknown_architecture_has_no_inferred_backend():
    future = Architecture(
        gfx="gfx9999",
        wave_size=32,
        bf16_mma=GFX1250.bf16_mma,
        native_permlane_exchange=False,
    )
    with pytest.raises(ValueError, match="no Kimi GEMV layout"):
        layout.layout_for(future)
