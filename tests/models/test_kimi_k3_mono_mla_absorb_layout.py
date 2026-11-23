# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 MLA absorb: every q_b_proj row in one
CTA task inside one scale block, every W_UK row in one wave, every token's
rope part and KV-cache entry written once, the LDS within the budget."""

import pytest

from atom.models.kimi_k3.mono.config import MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay
from atom.mono.plan.architecture import GFX1250

QREP_HEADS = 96  # the DCP8 group's heads at TP8 (QREP)
LOCAL_HEADS = 12  # one rank's at TP8


def test_widths_are_kimi_k3s_mla():
    assert (lay.Q_LORA, lay.NOPE, lay.ROPE, lay.KV_LORA) == (1536, 128, 64, 512)
    assert lay.HEAD == 192 and lay.ENTRY == 576
    assert lay.K_QUARTERS * lay.K_QUARTER == lay.Q_LORA
    assert lay.K_QUARTER % lay.QUANT_GROUP == 0


@pytest.mark.parametrize("heads", [QREP_HEADS, LOCAL_HEADS])
def test_every_q_b_row_is_one_ctas_task_inside_one_scale_block(heads):
    seen = []
    for cta in range(lay.head_ctas(heads)):
        first, second, rank = lay.cta_heads(cta, heads)
        for head in (first, second):
            for row in lay.q_b_tasks(head, rank):
                assert row % 16 == 0 and row // 128 == (row + 15) // 128
                seen.extend(range(row, row + 16))
    assert sorted(seen) == list(range(heads * lay.HEAD))


@pytest.mark.parametrize("heads", [QREP_HEADS, LOCAL_HEADS])
def test_every_w_uk_row_is_one_waves(heads):
    seen = {}
    for cta in range(lay.head_ctas(heads)):
        first, _, rank = lay.cta_heads(cta, heads)
        for wave in range(lay.WAVES):
            job, quarter = lay.wave_job(wave)
            for row in lay.uk_rows(first + job, rank, quarter):
                key = (first + job, row)
                assert key not in seen
                seen[key] = (cta, wave)
    assert len(seen) == heads * lay.KV_LORA


def test_a_wave_projects_its_absorbed_heads_q_over_one_k_quarter():
    assert [lay.wave_job(w) for w in range(lay.WAVES)] == [
        (0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),
    ]  # fmt: skip
    assert lay.UK_TASKS == 2 and lay.UK_WAVE_ROWS == 32


@pytest.mark.parametrize("rows", [1, 3, 16, 17, 63, 64])
def test_every_tokens_rope_part_is_one_ctas(rows):
    tokens = sorted(t for rank in range(4) for t in lay.rope_tokens(rank, rows))
    assert tokens == list(range(rows))
    # a thread an 8-element chunk of one head of the pair: one round
    assert -(-rows // 4) * lay.HEADS_PER_CLUSTER * (lay.ROPE // lay.VEC) <= lay.THREADS


def test_the_grid_is_resident_at_tp8_with_a_kv_cta_a_token():
    assert lay.grid(QREP_HEADS) == 256 == lay.head_ctas(QREP_HEADS) + MAX_ROWS
    assert lay.grid(LOCAL_HEADS) == 24 + MAX_ROWS
    with pytest.raises(ValueError, match="pairs"):
        lay.grid(13)


@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1, 7))
def test_the_lds_fits(rows):
    assert 4 * lay.lds_words(rows) <= GFX1250.max_workgroup_lds_bytes


def test_the_plan_checks_its_shape():
    plan = lay.AbsorbPlan(17, LOCAL_HEADS, 16, False)
    assert plan.tiles == 2 and plan.q_rows == 2304 and plan.q_out_row == 16 * 576
    with pytest.raises(ValueError, match="rows"):
        lay.AbsorbPlan(MAX_ROWS + 1, QREP_HEADS, QREP_HEADS, True)
    with pytest.raises(ValueError, match="q_out heads"):
        lay.AbsorbPlan(8, QREP_HEADS, 64, True)


def test_the_q_exchange_holds_productions_q():
    assert lay.q_words(64, QREP_HEADS) * 4 == 64 * 18432 * 2
