# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 DSpark draft MLA front
(``mla_draft_front_layout``) and the draft absorb's LDS."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as lay
from atom.mono.plan.architecture import GFX1250


def test_the_groups_cover_fused_qkv_a_proj_once():
    kinds = [lay.cluster_rows(c) for c in range(lay.A_CLUSTERS)]
    rows = [row for _, row in kinds]
    assert rows == list(range(0, lay.QKV_A_ROWS, lay.GROUP_ROWS))
    assert [k for k, _ in kinds].count("q") * lay.GROUP_ROWS == lay.Q_LORA
    assert [k for k, _ in kinds].count("kv") * lay.GROUP_ROWS == lay.KV_LORA
    assert kinds[-1] == ("rope", lay.Q_LORA + lay.KV_LORA)
    with pytest.raises(ValueError):
        lay.cluster_rows(lay.A_CLUSTERS)


@pytest.mark.parametrize("ksplit", lay.KSPLITS)
def test_a_ctas_runs_are_its_chains_weights(ksplit):
    """Every (task, K group) of a group is copied by exactly one CTA of its
    cluster, and a CTA's runs are four 28 KB extents of the preshuffled
    weight (a 16-row task's K groups one after another)."""
    seen = set()
    for rank in range(4):
        for run in range(lay.RUNS):
            task, group = lay.run_source(ksplit, 0, rank, run)
            for g in range(group, group + lay.RUN_GROUPS):
                assert (task, g) not in seen
                seen.add((task, g))
            offset = lay.run_offset(ksplit, 3, rank, run)
            assert offset == (3 * 64 + task * 16) * HIDDEN + group * lay.GROUP_BYTES
    assert seen == {(t, g) for t in range(lay.TASKS) for g in range(lay.GROUPS)}


def test_a_wave_copies_half_a_run():
    pieces = {lay.wave_piece(w) for w in range(lay.WAVES)}
    assert pieces == {(r, h) for r in range(lay.RUNS) for h in range(2)}
    assert lay.PIECE_GROUPS * 2 == lay.RUN_GROUPS


@pytest.mark.parametrize("rows", (1, 7, 16, 17, 28, 32, 33, 64))
@pytest.mark.parametrize("ksplit", lay.KSPLITS)
def test_chains_cover_every_task_and_tile(rows, ksplit):
    tiles = -(-rows // 16)
    chains = lay.chains(ksplit, rows)
    if ksplit == 4:
        assert sorted(chains) == sorted(
            (t, m) for t in range(lay.TASKS) for m in range(tiles)
        )
        assert lay.pieces(ksplit) == 2
    else:
        assert chains == [(0, m) for m in range(tiles)]
        assert lay.pieces(ksplit) == 8
    assert lay.pieces(ksplit) * lay.PIECE_GROUPS == lay.chain_groups(ksplit)


@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_every_width_fits_the_workgroup_lds_and_one_wave_a_norm_row(rows):
    assert lay.lds_words(rows) * 4 <= GFX1250.max_workgroup_lds_bytes
    jobs = {
        lay.norm_job(cta, wave, rows)
        for cta in range(lay.gemv_ctas(), lay.grid(rows))
        for wave in range(lay.WAVES)
    } - {None}
    assert jobs == set(range(lay.norm_jobs(rows)))
    assert lay.grid(rows) <= 256 and lay.norm_ctas(rows) % 4 == 0


def test_the_plan_takes_one_to_max_rows_and_the_mirrored_splits():
    for bad in (0, MAX_ROWS + 1):
        with pytest.raises(ValueError):
            lay.DraftFrontPlan(bad, 4)
    with pytest.raises(ValueError):
        lay.DraftFrontPlan(8, 2)
    assert lay.DraftFrontPlan(17, 4).tiles == 2


def test_the_split_k_of_a_flydsl_kernel_name():
    name = "flydsl_mxfp8_128_bpreshuffle_wmma_t32x32x256_mw2_nw2_nb4_sk4_cm1_cn1"
    assert lay.qkv_a_split_k(name) == 4
    assert lay.qkv_a_split_k(name.replace("sk4", "sk1")) == 1
    assert lay.qkv_a_split_k("") is None and lay.qkv_a_split_k(None) is None


def test_the_q_quant_constants_are_the_kernels():
    import struct

    assert struct.unpack("<f", struct.pack("<I", lay.INV_FP8_MAX_BITS))[0] == (
        struct.unpack("<f", struct.pack("<f", 1.0 / 448.0))[0]
    )
    assert lay.Q_GROUPS == 12 and lay.GROUP_LANES == 8


def test_the_draft_absorbs_lds_fits_every_width():
    pytest.importorskip("flydsl", reason="the draft absorb module imports FlyDSL")
    from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_draft_gfx1250 as ma

    for rows in range(1, MAX_ROWS + 1):
        assert ma.lds_words(rows) * 4 <= GFX1250.max_workgroup_lds_bytes
