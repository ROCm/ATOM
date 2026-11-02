# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU/static proofs for the gfx1250 Kimi K2 local-compute tail."""

from itertools import pairwise
from pathlib import Path

import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages import (
    moe_tail_layout as tail,
)
from atom.mono.plan.architecture import GFX950, GFX1250


def _plan(tokens=16):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=tokens,
    )


def test_gfx950_and_gfx1250_reduction_contracts_are_distinct():
    assert tail.GFX950_TAIL_REDUCTION.architecture is GFX950
    assert tail.GFX1250_TAIL_REDUCTION.architecture is GFX1250
    assert tail.GFX950_TAIL_REDUCTION.numerical_order != (
        tail.GFX1250_TAIL_REDUCTION.numerical_order
    )
    levels = tail.GFX950_TAIL_REDUCTION.wave64_pair_tree()
    assert len(levels) == 6
    assert levels[0][0] == (0, 32)
    assert levels[-1][0] == (0, 1)
    ownership = tail.tail_contract_for(GFX1250, threads=256, waves=8, tokens=16)
    assert not ownership.matrix_bitwise_equivalence_proven
    assert "not claimed bitwise identical" in ownership.matrix_numerical_order


def test_wave32_norm_tree_preserves_wave64_f32_order():
    values = tuple(
        tail.f32(((-1.0) ** index) * (index + 1) / 17.0) for index in range(64)
    )
    old = tail.GFX950_TAIL_REDUCTION.reduce_norm_pairs(values)
    new = tail.GFX1250_TAIL_REDUCTION.reduce_norm_pairs_wave32(values)
    assert old == new


def test_rank_reductions_are_explicitly_serial_zero_through_seven():
    values = tuple(
        tail.f32((rank + 1) * (1.0 if rank % 2 else -0.75)) for rank in range(tail.TP)
    )
    expected = tail.f32(0.0)
    for value in values:
        expected = tail.f32(expected + value)
    assert tail.GFX1250_TAIL_REDUCTION.rank_order(values) == expected


@pytest.mark.parametrize("tokens", range(1, 17))
def test_thread_pair_ownership_is_complete_without_overlap(tokens):
    contract = tail.tail_contract_for(
        GFX1250,
        threads=256,
        waves=8,
        tokens=tokens,
    )
    for pairs in (8, 64):
        work = [
            coordinate
            for tid in range(contract.threads)
            for coordinate in contract.thread_pair_work(pairs, tid)
        ]
        expected = [(token, pair) for token in range(tokens) for pair in range(pairs)]
        assert len(work) == len(set(work))
        assert sorted(work) == expected
    wave_tokens = [
        token
        for wave in range(contract.waves)
        for token in contract.norm_wave_tokens(wave)
    ]
    assert sorted(wave_tokens) == list(range(tokens))
    assert len(wave_tokens) == len(set(wave_tokens))


def test_task_rows_and_columns_cover_every_output_once():
    shared = [
        row
        for task in range(tail.SHARED_DOWN_TASKS)
        for row in tail.TailOwnershipContract.shared_down_rows(task)
    ]
    latent = [
        column
        for task in range(tail.LNORM_TASKS)
        for column in tail.TailOwnershipContract.lnorm_columns(task)
    ]
    up = [
        row
        for rank in range(tail.TP)
        for task in range(tail.UP_TASKS)
        for row in tail.TailOwnershipContract.up_rows(rank, task)
    ]
    final = [
        column
        for task in range(tail.FINAL_TASKS)
        for column in tail.TailOwnershipContract.final_columns(task)
    ]
    for coordinates, width in (
        (shared, tail.HIDDEN),
        (latent, tail.LATENT),
        (up, tail.HIDDEN),
        (final, tail.HIDDEN),
    ):
        assert len(coordinates) == len(set(coordinates)) == width
        assert sorted(coordinates) == list(range(width))


@pytest.mark.parametrize("tokens", range(1, 17))
def test_tail_lds_offsets_are_aligned_nonoverlapping_and_bounded(tokens):
    layout = tail.tail_lds_plan(tokens, waves=8)
    for regions in (layout.gemv_regions, layout.norm_regions):
        for region in regions:
            assert region.offset % tail.ALIGNMENT == 0
        for left, right in pairwise(regions):
            assert left.end <= right.offset
    plan = _plan(tokens)
    assert layout == plan.moe_tail_lds_plan()
    assert layout.required_bytes <= plan.lds_plan().stage_budget_bytes


@pytest.mark.parametrize("tokens", range(1, 17))
def test_transport_staging_shapes_are_rank_major(tokens):
    transport = tail.TailTransportContract()
    assert transport.required_order == "rank-major 0..7"
    assert transport.latent_elements(tokens) == (tail.TP * tokens * tail.LATENT)
    assert transport.final_elements(tokens) == (tail.TP * tokens * tail.HIDDEN)
    assert "unsupported" in transport.capability


def test_device_source_uses_wmma_backend_without_xgmi_publication():
    root = Path(__file__).parents[2]
    source = (
        root / "atom/models/kimi_k3/mono/kernels/stages/moe_tail_gfx1250.py"
    ).read_text()
    for callable_name in (
        "stage_latr_reduce",
        "stage_lnorm_stats",
        "stage_lnorm_rstd",
        "stage_lnorm_apply",
        "stage_sdown",
        "stage_up",
        "stage_final_reduce",
    ):
        assert f"def {callable_name}(" in source
    assert "stage_backend_for(" in source
    assert "wave_sum(value, GFX1250)" in source
    assert "sum_partials" not in source
    assert "publish(" not in source
    assert 'c["poll"]' not in source
    assert 'c["put"]' not in source
    assert {item.primitive for item in tail.GFX1250_TAIL_EVIDENCE} == {
        "BF16 WMMA GEMV",
        "wave32 XOR reduction",
        "rank-order fold",
    }
    assert all(item.source and item.contract for item in tail.GFX1250_TAIL_EVIDENCE)
