# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU proofs for gfx1250 Kimi M16 token tiling through S=48."""

import pytest

from atom.models.kimi_k3.mono.kernels import execution_plan as ep
from atom.models.kimi_k3.mono.kernels.attention import layout as attention
from atom.models.kimi_k3.mono.kernels.attention.kda_builder_gfx1250 import (
    declare_local,
)
from atom.models.kimi_k3.mono.kernels.attention.protocol import K1Protocol
from atom.models.kimi_k3.mono.kernels.stages import (
    gemv_layout,
    moe_layout,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import (
    Gfx1250TokenTiling,
)
from atom.mono.plan.architecture import GFX1250

BOUNDARIES = (16, 17, 32, 33, 48)


def _plan(tokens):
    return ep.Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=tokens,
        nblocks=6,
    )


@pytest.mark.parametrize("tokens", BOUNDARIES)
def test_token_tiles_cover_every_row_once(tokens):
    tiling = Gfx1250TokenTiling(tokens)
    rows = [row for tile in tiling.tiles for row in tile.row_range]
    assert rows == list(range(tokens))
    assert len(rows) == len(set(rows))
    assert all(1 <= tile.rows <= 16 for tile in tiling.tiles)
    for row in rows:
        tile, local = tiling.owner(row)
        assert tiling.tile(tile).global_row(local) == row


def test_token_tiling_rejects_row_49():
    with pytest.raises(ValueError, match=r"\[1, 48\]"):
        Gfx1250TokenTiling(49)
    with pytest.raises(ValueError, match=r"\[1, 48\]"):
        _plan(49)


@pytest.mark.parametrize("tokens", BOUNDARIES)
def test_gemv_tiles_cover_all_output_coordinates(tokens):
    tiling = gemv_layout.GFX1250_GEMV.token_tiling(tokens)
    coordinates = []
    for tile in tiling.tiles:
        for lane in range(32):
            for (
                output_row,
                local_token,
            ) in gemv_layout.GFX1250_GEMV.accumulator_coordinates(lane):
                if local_token < tile.rows:
                    coordinates.append((output_row, tile.global_row(local_token)))
    expected = {
        (output_row, token) for output_row in range(16) for token in range(tokens)
    }
    assert len(coordinates) == len(set(coordinates)) == len(expected)
    assert set(coordinates) == expected


def _routes(tokens):
    return tuple(
        tuple(
            (token * moe_layout.TOPK + topk) % moe_layout.EXPERTS
            for topk in range(moe_layout.TOPK)
        )
        for token in range(tokens)
    )


@pytest.mark.parametrize("tokens", BOUNDARIES)
def test_route_kof_tiles_and_quant_groups_cover_all_tokens(tokens):
    route = moe_layout.RouteUnionContract(GFX1250, 256, tokens)
    union = route.build(_routes(tokens))
    reconstructed = [[] for _ in union.kof]
    for tile in route.token_tiles:
        chunk = route.kof_tile(union, tile.index)
        assert all(len(row) == tile.rows for row in chunk)
        for slot, row in enumerate(chunk):
            reconstructed[slot].extend(row)
    assert tuple(tuple(row) for row in reconstructed) == union.kof

    quant = moe_layout.LatentQuantContract(GFX1250, 256, tokens)
    blocks = [
        block for tid in range(quant.threads) for block in quant.blocks_for_thread(tid)
    ]
    assert len(blocks) == len(set(blocks)) == quant.blocks
    assert sorted(blocks) == list(range(quant.blocks))


@pytest.mark.parametrize("tokens", BOUNDARIES)
def test_attention_and_protocol_cover_all_rows(tokens):
    contract = attention.kda_contract_for(GFX1250, waves=8)
    mixed = [
        token
        for wave in range(contract.waves)
        for subgroup in range(contract.recurrence_subgroups_per_wave)
        for token in contract.mix_tokens(tokens, wave, subgroup)
    ]
    assert sorted(mixed) == list(range(tokens))
    assert len(mixed) == len(set(mixed))
    assert sum(tile.rows for tile in contract.token_tiles(tokens)) == tokens

    protocol = K1Protocol(
        blocks=256,
        threads=256,
        tokens=tokens,
        qlen=1,
        sources=7,
    )
    gates = [
        assignment.gate_token
        for assignment in protocol.assignments
        if assignment.gate_token is not None
    ]
    assert gates == list(range(tokens))
    tasks = [
        task for assignment in protocol.assignments for task in assignment.kda_tasks
    ]
    assert len(tasks) == len(set(tasks)) == protocol.kda_task_count
    assert sorted(tasks) == list(range(protocol.kda_task_count))


@pytest.mark.parametrize("tokens", BOUNDARIES)
def test_s48_lds_unions_are_tile_local_aligned_and_bounded(tokens):
    plan = _plan(tokens)
    arena = plan.lds_plan()
    assert arena.used_bytes <= arena.capacity_bytes <= 320 * 1024
    assert all(region.offset % 16 == 0 for region in arena.regions)
    for left, right in zip(arena.regions, arena.regions[1:]):
        assert left.end <= right.offset

    tail = plan.moe_tail_lds_plan()
    assert tail.lds_rows <= 16
    assert tail.required_bytes <= arena.stage_budget_bytes
    assert sum(tile.rows for tile in tail.token_tiles) == tokens

    k1 = declare_local(plan, qlen=1)
    assert sum(tile.rows for tile in k1.token_tiles) == tokens
    assert k1.lds.arena.used_bytes <= 320 * 1024
    assert k1.lds.variant("projection").fields[0].bytes == (
        min(tokens, 16) * k1.xrow * 2
    )
