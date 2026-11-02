# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only Kimi attention projection/gate layout and schedule proofs."""

from itertools import pairwise
from pathlib import Path

import pytest

from atom.models.kimi_k3.mono.kernels import execution_plan
from atom.models.kimi_k3.mono.kernels.attention import layout
from atom.mono.plan.architecture import GFX950, GFX1250


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_attention_operands_cover_row_major_tile_exactly(architecture):
    contract = layout.kda_contract_for(architecture, waves=8)
    for operand in (
        contract.gemv.a_operand_coordinates,
        contract.gemv.b_operand_coordinates,
    ):
        coordinates = [
            coordinate
            for lane in range(architecture.wave_size)
            for coordinate in operand(lane, 0)
        ]
        assert len(coordinates) == 16 * 32
        assert set(coordinates) == {(row, k) for row in range(16) for k in range(32)}
    for lane in range(architecture.wave_size):
        assert contract.gemv.a_operand_coordinates(
            lane, 0
        ) == contract.gemv.b_operand_coordinates(lane, 0)


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_kda_projection_chunks_are_complete_without_overlap(architecture):
    contract = layout.kda_contract_for(architecture, waves=8)
    owned = [contract.projection_chunk_ids(wave) for wave in range(contract.waves)]
    flattened = [chunk for chunks in owned for chunk in chunks]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(layout.HIDDEN // 128))
    assert contract.group_columns == 1024
    assert contract.projection_groups == 7


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_kda_gate_accumulators_cover_all_rows_and_tokens(architecture):
    contract = layout.kda_contract_for(architecture, waves=8)
    coordinates = [
        coordinate
        for wave in range(contract.waves)
        for lane in range(architecture.wave_size)
        for coordinate in contract.gate_output_coordinates(wave, lane)
    ]
    assert len(coordinates) == 128 * layout.ROWS
    assert len(coordinates) == len(set(coordinates))
    assert set(coordinates) == {
        (row, token) for row in range(128) for token in range(layout.ROWS)
    }


def test_gfx1250_gate_uses_v8_row_halves():
    contract = layout.kda_contract_for(GFX1250, waves=8)
    assert contract.gate_output_coordinates(0, 0) == tuple((row, 0) for row in range(8))
    assert contract.gate_output_coordinates(0, 16) == tuple(
        (row, 0) for row in range(8, 16)
    )
    assert contract.gate_output_coordinates(7, 31) == tuple(
        (row, 15) for row in range(120, 128)
    )


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_recurrence_state_mapping_covers_16x128_exactly(architecture):
    contract = layout.kda_contract_for(architecture, waves=8)
    coordinates = [
        coordinate
        for wave in range(contract.waves)
        for lane in range(architecture.wave_size)
        for coordinate in contract.recurrence_coordinates(wave, lane)
    ]
    assert len(coordinates) == 16 * 128
    assert len(coordinates) == len(set(coordinates))
    assert set(coordinates) == {
        (row, column) for row in range(16) for column in range(128)
    }
    assert contract.recurrence_reduction_offsets == (16, 8, 4, 2, 1)


def test_wave32_recurrence_serializes_two_rows_per_wave():
    gfx950 = layout.kda_contract_for(GFX950, waves=8)
    gfx1250 = layout.kda_contract_for(GFX1250, waves=8)
    assert gfx950.recurrence_rows(0, 0) == (0,)
    assert gfx950.recurrence_rows(0, 1) == (1,)
    assert gfx1250.recurrence_rows(0, 0) == (0, 8)
    assert gfx1250.recurrence_rows(7, 0) == (7, 15)


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_qk_norm_lane_mapping_covers_128_columns(architecture):
    contract = layout.kda_contract_for(architecture, waves=8)
    columns = [
        column
        for lane in range(architecture.wave_size)
        for column in contract.qk_columns(lane)
    ]
    assert len(columns) == len(set(columns)) == 128
    assert sorted(columns) == list(range(128))


def test_wave32_qk_reduction_preserves_wave64_add_tree():
    gfx950 = layout.kda_contract_for(GFX950, waves=8)
    gfx1250 = layout.kda_contract_for(GFX1250, waves=8)
    assert gfx950.qk_reduction_offsets == (32, 16, 8, 4, 2, 1)
    assert gfx1250.qk_reduction_offsets == (16, 8, 4, 2, 1)
    assert gfx1250.qk_reduction_tree() == gfx950.qk_reduction_tree()


def test_accumulator_lds_size_is_layout_derived():
    gfx950 = layout.kda_contract_for(GFX950, waves=8)
    gfx1250 = layout.kda_contract_for(GFX1250, waves=8)
    assert gfx950.accumulator_lds_elements == 8 * 64 * 4
    assert gfx1250.accumulator_lds_elements == 8 * 32 * 8
    assert gfx1250.accumulator_lds_elements == gfx950.accumulator_lds_elements


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_attention_slice_work_covers_every_value_once(architecture):
    contract = layout.kda_contract_for(architecture, waves=8)
    tokens, sources = 16, 7
    threads = contract.waves * architecture.wave_size
    work = [
        item
        for thread in range(threads)
        for item in contract.slice_work(tokens, sources, thread)
    ]
    expected = {
        (token, source, column_group)
        for token in range(tokens)
        for source in range(sources)
        for column_group in range(32)
    }
    assert len(work) == len(set(work)) == len(expected)
    assert set(work) == expected


@pytest.mark.parametrize("architecture", [GFX950, GFX1250])
def test_attention_mix_token_ownership_is_complete(architecture):
    contract = layout.kda_contract_for(architecture, waves=8)
    owned = [
        token
        for wave in range(contract.waves)
        for subgroup in range(contract.recurrence_subgroups_per_wave)
        for token in contract.mix_tokens(16, wave, subgroup)
    ]
    assert sorted(owned) == list(range(16))
    assert len(owned) == len(set(owned))


@pytest.mark.parametrize("sources", [1, 7, 16, 32])
@pytest.mark.parametrize("identity", ["max_identity", "sum_identity"])
def test_wave32_softmax_tree_matches_wave64_after_identities(sources, identity):
    gfx950 = layout.kda_contract_for(GFX950, waves=8)
    gfx1250 = layout.kda_contract_for(GFX1250, waves=8)
    assert gfx1250.softmax_reduction_tree(
        sources, identity
    ) == gfx950.softmax_reduction_tree(sources, identity)
    assert [gfx1250.gate_source(lane, sources) for lane in range(32)] == [
        *range(sources),
        *([None] * (32 - sources)),
    ]


def test_mix_value_indices_cover_vals_without_overlap():
    contract = layout.kda_contract_for(GFX1250, waves=8)
    tokens, sources = 16, 7
    indices = [
        index
        for token in range(tokens)
        for source in range(sources)
        for column_group in range(32)
        for index in contract.mix_indices(token, source, sources, column_group)
    ]
    assert len(indices) == len(set(indices)) == tokens * sources * 128
    assert sorted(indices) == list(range(tokens * sources * 128))


def test_attention_seam_lds_is_bounded_by_candidate():
    plan = execution_plan.Gfx1250KimiBuildPlan(
        threads=256,
        blocks=37,
        lds_bytes=160 * 1024,
        tokens=16,
    )
    assert plan.attention_seam_lds_bytes(7) <= plan.lds_bytes
    with pytest.raises(ValueError, match="attention seam needs"):
        plan.attention_seam_lds_bytes(32)

    maximum = execution_plan.Gfx1250KimiBuildPlan(
        threads=256,
        blocks=37,
        lds_bytes=327680,
        tokens=16,
    )
    assert maximum.attention_seam_lds_bytes(32) <= maximum.lds_bytes


@pytest.mark.parametrize("rows", [1, 8, 16])
@pytest.mark.parametrize("lds_bytes", [160 * 1024, 327680])
def test_kda_slabs_cover_hidden_in_order_and_fit_lds(rows, lds_bytes):
    contract = layout.kda_contract_for(GFX1250, waves=8)
    fixed = 8192
    slabs = contract.slab_plan(
        rows, lds_capacity_bytes=lds_bytes, fixed_lds_bytes=fixed
    )
    assert slabs[0][0] == 0
    assert sum(groups for _, groups in slabs) == contract.projection_groups
    assert all(
        next_first == first + groups
        for (first, groups), (next_first, _) in pairwise(slabs)
    )
    assert all(
        rows * (groups * contract.group_columns + layout.LDS_PAD) * 2 + fixed
        <= lds_bytes
        for _, groups in slabs
    )


def test_builder_plan_supplies_kda_schedule_without_inferred_topology():
    plan = execution_plan.Gfx1250KimiBuildPlan(
        threads=256,
        blocks=37,
        lds_bytes=160 * 1024,
        tokens=16,
    )
    assert plan.waves == 8
    assert plan.kda_matrix_contract().architecture is GFX1250
    assert plan.kda_slab_plan(16, fixed_lds_bytes=8192) == (
        (0, 4),
        (4, 3),
    )
    assert plan.kda_accumulator_lds_elements == 8 * 32 * 8

    unsupported = execution_plan.Gfx1250KimiBuildPlan(
        threads=128,
        blocks=37,
        lds_bytes=327680,
        tokens=16,
        nblocks=6,
    )
    with pytest.raises(ValueError, match="requires 8 explicit waves"):
        unsupported.kda_backend()
    with pytest.raises(ValueError, match="explicit nblocks"):
        plan.kda_backend()


def test_attention_backend_selection_is_architecture_explicit(monkeypatch):
    imported = []

    class Gfx1250Module:
        @staticmethod
        def StageBackend(contract):
            return "gfx1250", contract

    monkeypatch.setattr(
        layout,
        "import_module",
        lambda name: imported.append(name) or Gfx1250Module,
    )
    monkeypatch.setattr(
        layout,
        "stage_backend_for",
        lambda architecture, *, waves: ("back", architecture, waves),
    )
    assert layout.back_backend_for(GFX1250, waves=8) == (
        "back",
        GFX1250,
        8,
    )
    selected, contract = layout.kda_backend_for(GFX1250, waves=8)
    assert selected == "gfx1250"
    assert contract.architecture is GFX1250 and contract.waves == 8
    assert imported == ["atom.models.kimi_k3.mono.kernels.attention.kda_gfx1250"]


def test_common_deployment_default_is_gfx950_but_k3_opts_into_gfx1250():
    deployment = (
        Path(__file__).parents[2] / "atom" / "mono" / "runtime" / "deployment.py"
    ).read_text()
    assert 'supported_gfx: tuple[str, ...] = ("gfx950",)' in deployment
    config = (
        Path(__file__).parents[2] / "atom/models/kimi_k3/mono/config.py"
    ).read_text()
    assert 'supported_gfx=("gfx950", "gfx1250")' in config
