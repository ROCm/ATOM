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


def test_layout_and_backend_selection_is_explicit(monkeypatch):
    assert layout.layout_for(GFX950) is layout.GFX950_GEMV
    assert layout.layout_for(GFX1250) is layout.GFX1250_GEMV

    imported = []
    monkeypatch.setattr(
        layout,
        "import_module",
        lambda name: imported.append(name) or name,
    )
    assert layout.backend_for(GFX950).endswith(".gemv")
    assert layout.backend_for(GFX1250).endswith(".gemv_gfx1250")
    assert imported == [
        "atom.models.kimi_k3.mono.kernels.stages.gemv",
        "atom.models.kimi_k3.mono.kernels.stages.gemv_gfx1250",
    ]


def test_stage_selection_requires_explicit_gfx1250_wave_count(monkeypatch):
    gfx950_backend = object()

    class Gfx1250Module:
        @staticmethod
        def StageBackend(waves):
            return "gfx1250", waves

    def imported(name):
        return Gfx1250Module if name.endswith("gemv_gfx1250") else gfx950_backend

    monkeypatch.setattr(layout, "import_module", imported)
    assert layout.stage_backend_for(GFX950) is gfx950_backend
    assert layout.stage_backend_for(GFX950, waves=8) is gfx950_backend
    with pytest.raises(ValueError, match="fixed to 8 waves"):
        layout.stage_backend_for(GFX950, waves=4)
    with pytest.raises(ValueError, match="explicit waves count"):
        layout.stage_backend_for(GFX1250)
    assert layout.stage_backend_for(GFX1250, waves=11) == ("gfx1250", 11)


def test_unknown_architecture_has_no_inferred_backend():
    future = Architecture(
        gfx="gfx9999",
        wave_size=32,
        bf16_mma=GFX1250.bf16_mma,
        native_permlane_exchange=False,
    )
    with pytest.raises(ValueError, match="no Kimi GEMV layout"):
        layout.layout_for(future)
    with pytest.raises(ValueError, match="no Kimi GEMV backend"):
        layout.backend_for(future)
