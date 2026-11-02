# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU/static proofs for gfx1250 Kimi K2 scaled-WMMA expert compute."""

from pathlib import Path

import pytest

from atom.models.kimi_k3.mono.kernels.stages import (
    moe_expert_layout as ex,
)
from atom.mono.plan.architecture import GFX950, GFX1250


def test_architecture_contracts_are_distinct_and_dimensionally_complete():
    assert ex.expert_contract_for(GFX950) is ex.GFX950_EXPERT
    assert ex.expert_contract_for(GFX1250) is ex.GFX1250_EXPERT
    assert (
        ex.GFX950_EXPERT.weight_dwords,
        ex.GFX950_EXPERT.activation_dwords,
        ex.GFX950_EXPERT.accumulator_elements,
    ) == (4, 4, 4)
    assert (
        ex.GFX1250_EXPERT.weight_dwords,
        ex.GFX1250_EXPERT.activation_dwords,
        ex.GFX1250_EXPERT.accumulator_elements,
    ) == (16, 8, 16)
    assert ex.GFX1250_EXPERT.weight_scale_bytes == 4
    assert ex.GFX1250_EXPERT.activation_scale_bytes == 4


def test_weight_fragment_coordinates_cover_n32_k128_once():
    coordinates = [
        ex.GFX1250_FRAGMENT.weight_coordinate(lane, dword, nibble)
        for lane in range(32)
        for dword in range(16)
        for nibble in range(8)
    ]
    assert len(coordinates) == 32 * 128
    assert len(coordinates) == len(set(coordinates))
    assert set(coordinates) == {(row, k) for row in range(32) for k in range(128)}


def test_activation_fragment_coordinates_cover_m16_k128_once():
    coordinates = [
        ex.GFX1250_FRAGMENT.activation_coordinate(lane, dword, nibble)
        for lane in range(32)
        for dword in range(8)
        for nibble in range(8)
    ]
    assert len(coordinates) == 16 * 128
    assert len(coordinates) == len(set(coordinates))
    assert set(coordinates) == {(token, k) for token in range(16) for k in range(128)}
    for lane in range(32):
        parity = lane // 16
        for dword in range(8):
            loader_dword = parity * 4 + (dword // 4) * 8 + dword % 4
            for nibble in range(8):
                _, k = ex.GFX1250_FRAGMENT.activation_coordinate(lane, dword, nibble)
                assert k // 8 == loader_dword
                assert k % 8 == nibble


def test_accumulator_coordinates_cover_m16_n32_once():
    coordinates = [
        ex.GFX1250_FRAGMENT.accumulator_coordinate(lane, element)
        for lane in range(32)
        for element in range(16)
    ]
    assert len(coordinates) == 16 * 32
    assert len(coordinates) == len(set(coordinates))
    assert set(coordinates) == {
        (token, column) for token in range(16) for column in range(32)
    }
    for token in range(16):
        lower = set(ex.GFX1250_FRAGMENT.output_columns_for_lane(token))
        upper = set(ex.GFX1250_FRAGMENT.output_columns_for_lane(token + 16))
        assert not lower & upper
        assert lower | upper == set(range(32))


def test_scale_word_ownership_matches_n32_and_m16_rows():
    for step in range(ex.LATENT // ex.K_STEP):
        weight = [
            coordinate
            for lane in range(32)
            for coordinate in ex.GFX1250_FRAGMENT.weight_scale_coordinates(lane, step)
        ]
        assert len(weight) == len(set(weight)) == 32 * 4
        assert set(weight) == {
            (row, step * 4 + byte) for row in range(32) for byte in range(4)
        }
        activation = [
            ex.GFX1250_FRAGMENT.activation_scale_coordinates(lane, step)
            for lane in range(32)
        ]
        for token in range(16):
            assert activation[token] == activation[token + 16]
            assert activation[token] == tuple(
                (token, step * 4 + byte) for byte in range(4)
            )


@pytest.mark.parametrize(
    ("rows", "k"),
    [
        (32, 128),
        (2 * ex.ROUTED_INTERMEDIATE, ex.LATENT),
        (ex.LATENT, ex.ROUTED_INTERMEDIATE),
    ],
)
def test_payload_shuffle_index_round_trip(rows, k):
    packing = ex.ExpertWeightPacking(rows, k)
    for row in range(rows):
        for kk in range(k):
            index, nibble = packing.payload_index(row, kk)
            assert packing.payload_coordinate(index, nibble) == (row, kk)


@pytest.mark.parametrize(
    ("rows", "k", "row0", "step"),
    [
        (2 * ex.ROUTED_INTERMEDIATE, ex.LATENT, 0, 0),
        (2 * ex.ROUTED_INTERMEDIATE, ex.LATENT, 736, 27),
        (ex.LATENT, ex.ROUTED_INTERMEDIATE, 0, 0),
        (ex.LATENT, ex.ROUTED_INTERMEDIATE, 3552, 2),
    ],
)
def test_fragment_coordinates_match_shuffle_weight_byte_indices(rows, k, row0, step):
    packing = ex.ExpertWeightPacking(rows, k)
    for lane in range(32):
        lane16, parity = lane % 16, lane // 16
        for dword in range(16):
            output_half = dword // 8
            within_half = dword % 8
            k64_half, word = divmod(within_half, 4)
            loader_dword = (
                ((row0 // 16) + output_half) * (2 * k)
                + step * 256
                + k64_half * 128
                + parity * 64
                + lane16 * 4
                + word
            )
            for nibble in range(8):
                relative_row, relative_k = ex.GFX1250_FRAGMENT.weight_coordinate(
                    lane, dword, nibble
                )
                index, packed_nibble = packing.payload_index(
                    row0 + relative_row,
                    step * 128 + relative_k,
                )
                assert index == loader_dword * 4 + nibble // 2
                assert packed_nibble == nibble % 2


@pytest.mark.parametrize(
    ("rows", "k"),
    [
        (2 * ex.ROUTED_INTERMEDIATE, ex.LATENT),
        (ex.LATENT, ex.ROUTED_INTERMEDIATE),
    ],
)
def test_scale_repack_map_is_complete_unique_and_round_trips(rows, k):
    packing = ex.ExpertWeightPacking(rows, k)
    mapping = packing.scale_repack_map()
    sources = [source for source, _ in mapping]
    destinations = [destination for _, destination in mapping]
    assert len(mapping) == rows * packing.scale_groups
    assert len(sources) == len(set(sources))
    assert sorted(destinations) == list(range(len(destinations)))
    for row in range(rows):
        for group in range(packing.scale_groups):
            destination = packing.gfx1250_scale_index(row, group)
            assert packing.gfx1250_scale_coordinate(destination) == (
                row,
                group,
            )


def test_scale_repacking_materializes_the_declared_index_map():
    for packing in (
        ex.ExpertWeightPacking(2 * ex.ROUTED_INTERMEDIATE, ex.LATENT),
        ex.ExpertWeightPacking(ex.LATENT, ex.ROUTED_INTERMEDIATE),
    ):
        source = bytes(index % 251 for index in range(packing.gfx950_scale_bytes))
        result = packing.repack_scales(source)
        assert len(result) == packing.gfx1250_scale_bytes
        assert all(
            result[destination] == source[old]
            for old, destination in packing.scale_repack_map()
        )
        with pytest.raises(ValueError, match="expected"):
            packing.repack_scales(source[:-1])


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_runtime_scale_specs_are_architecture_and_tp_explicit(tp):
    inter = 3072 // tp
    for rank in range(tp):
        for name, rows, k in (
            ("w13_scale", 2 * inter, ex.LATENT),
            ("w2_scale", ex.LATENT, inter),
        ):
            spec = ex.ExpertScaleSpec(name, 256, rows, k, rank, tp)
            assert spec.storage_shape(ex.ExpertScaleLayout.GFX950_SHUFFLE) == (
                ((spec.rows + 255) // 256) * 256,
                ((spec.groups + 7) // 8) * 8,
            )
            assert spec.storage_shape(ex.ExpertScaleLayout.GFX1250_N32K4) == (
                256,
                rows // 32,
                spec.groups * 32,
            )
            assert spec.packing.gfx1250_scale_bytes == (256 * rows * spec.groups)
    assert (
        ex.ExpertScaleSpec("w13_scale", 256, 2 * inter, ex.LATENT, 0, tp)
        != ex.ExpertScaleSpec("w13_scale", 256, 2 * inter, ex.LATENT, tp - 1, tp)
        or tp == 1
    )


def test_runtime_binding_is_scale_only_and_gfx950_default_is_unchanged():
    root = Path(__file__).parents[2]
    materializer = (root / "atom/models/kimi_k3/mono/expert_scales.py").read_text()
    weights = (root / "atom/models/kimi_k3/mono/weights.py").read_text()
    deployment = (root / "atom/mono/runtime/deployment.py").read_text()
    assert "index_select" in materializer
    assert "source_for_destination" in materializer
    assert "w13_weight" not in materializer
    assert "w2_weight" not in materializer
    assert "return w13.data, s13.data, w2.data, s2.data" in weights
    assert 'supported_gfx: tuple[str, ...] = ("gfx950",)' in deployment


def test_eight_wave_gate_up_preserves_half_accumulation_order():
    tasks = ex.ExpertTaskContract(8)
    for role in ("gate", "up"):
        steps = [
            step
            for wave in tasks.ug_reduction_waves(role)
            for step in tasks.ug_k_steps(wave)
        ]
        assert len(steps) == len(set(steps))
        assert sorted(steps) == list(range(ex.LATENT // ex.K_STEP))
    assert tasks.ug_k_steps(0) == tuple(range(14))
    assert tasks.ug_k_steps(4) == tuple(range(14, 28))
    assert tasks.ug_k_steps(2) == tuple(range(14))
    assert tasks.ug_k_steps(6) == tuple(range(14, 28))
    assert tasks.ug_reduction_waves("gate") == (0, 4)
    assert tasks.ug_reduction_waves("up") == (2, 6)
    assert [tasks.ug_role(wave) for wave in range(8)] == [
        "gate",
        None,
        "up",
        None,
        "gate",
        None,
        "up",
        None,
    ]


@pytest.mark.parametrize("waves", [1, 2, 4, 8])
def test_down_slot_schedules_are_complete(waves):
    tasks = ex.ExpertTaskContract(waves)
    for union_size in range(257):
        slots = [
            slot for wave in range(waves) for slot in tasks.down_slots(union_size, wave)
        ]
        assert len(slots) == len(set(slots))
        assert sorted(slots) == list(range(union_size))
    assert tasks.down_tasks == ex.LATENT // 32
    assert tasks.down_contribution_elements == 16 * 16 * 32
    if waves == 8:
        assert tasks.ug_accumulator_elements == 4 * 32 * 16


def test_device_source_uses_exact_evidenced_scaled_wmma_boundary():
    root = Path(__file__).parents[2]
    source = (
        root / "atom/models/kimi_k3/mono/kernels/stages/moe_expert_gfx1250.py"
    ).read_text()
    assert "rocdl.wmma_scale_f32_32x16x128_f4(" in source
    assert "T.vec(16, T.f32)" in source
    assert "scaleAType=0" in source
    assert "scaleBType=0" in source
    assert "mfma_scaled" not in source
    assert "def stage_ug(" in source and "def stage_down(" in source


def test_scale_repacking_evidence_is_explicit():
    assert {item.primitive for item in ex.GFX1250_EXPERT_EVIDENCE} == {
        "scaled FP4 WMMA",
        "operand lane loads",
        "accumulator output mapping",
        "weight payload shuffle",
        "weight n32k4 scales",
    }
