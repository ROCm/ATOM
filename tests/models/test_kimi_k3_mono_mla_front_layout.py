# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 MLA front (``mla_front_layout``)."""

import struct

import pytest

from atom.models.kimi_k3.mono.config import GFX1250_BLOCKS, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as lay
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    TDM_CLUSTER,
    tdm_split_slabs,
)
from atom.mono.plan.architecture import GFX1250

GATE_ROWS = 96 * 128 // 8  # num_attention_heads * v_head_dim at TP8


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def test_widths_are_kimi_k3s_mla():
    assert (lay.Q_LORA, lay.KV_LORA, lay.ROPE) == (1536, 512, 64)
    assert lay.QKV_A_ROWS == 2112 and lay.QKV_A_PADDED == 2176
    assert lay.QKV_C == 2048 and lay.GROUP_ROWS == 64


def test_every_weight_row_is_one_ctas_task_inside_one_scale_block():
    seen = {"a": [], "gate": []}
    clusters = lay.gemv_ctas(GATE_ROWS) // TDM_CLUSTER
    for cluster in range(clusters):
        kind, row = lay.cluster_rows(cluster, GATE_ROWS)
        weight = "gate" if kind == "gate" else "a"
        for rank in range(TDM_CLUSTER):
            first = row + 16 * rank
            seen[weight].extend(range(first, first + 16))
        # a group's four tasks share their 128-row E8M0 block
        assert row // 128 == (row + lay.GROUP_ROWS - 1) // 128
        if kind == "q":
            assert row + lay.GROUP_ROWS <= lay.Q_LORA
        elif kind == "kv":
            assert lay.Q_LORA <= row < lay.QKV_C
        elif kind == "rope":
            assert row == lay.QKV_C and cluster == lay.ROPE_CLUSTER
    assert seen["a"] == list(range(lay.QKV_A_ROWS))
    assert seen["gate"] == list(range(GATE_ROWS))
    with pytest.raises(ValueError):
        lay.cluster_rows(clusters, GATE_ROWS)


def test_the_gemv_and_the_norms_fill_the_gpu_at_tp8():
    assert lay.gemv_ctas(GATE_ROWS) == 228
    assert lay.spare_ctas(GATE_ROWS) == 28
    assert lay.grid(GATE_ROWS, lay.spare_ctas(GATE_ROWS)) == GFX1250_BLOCKS
    assert lay.grid(GATE_ROWS) == 228
    with pytest.raises(ValueError, match="whole clusters"):
        lay.grid(GATE_ROWS, 6)
    with pytest.raises(ValueError, match="64-row groups"):
        lay.gate_clusters(1560)


@pytest.mark.parametrize("norm_ctas", (0, 28))
@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_every_norm_row_is_one_waves(rows, norm_ctas):
    first, count = lay.norm_placement(GATE_ROWS, norm_ctas)
    jobs = [
        job
        for cta in range(lay.grid(GATE_ROWS, norm_ctas))
        for wave in range(lay.WAVES)
        if (job := lay.norm_job(cta, wave, GATE_ROWS, norm_ctas)) is not None
        and job < lay.norm_jobs(rows)
    ]
    assert sorted(jobs) == list(range(2 * rows))
    if norm_ctas:
        # the norms' own CTAs are the GEMV's spare ones, past it
        assert first == lay.gemv_ctas(GATE_ROWS) and count == norm_ctas
    else:
        # the gate CTAs, which publish nothing the norms wait for
        assert first == lay.A_CLUSTERS * TDM_CLUSTER
    assert lay.norm_jobs(rows) <= count * lay.WAVES


def test_a_lane_holds_eight_aiter_threads_and_a_lane_pair_one_quant_group():
    # AITER fused_qk_rmsnorm at max(1536, 512): 256 threads of 8 elements
    assert lay.NORM_THREADS * lay.NORM_RUN >= lay.Q_LORA
    assert lay.LANE_ELEMENTS == 64 == 8 * lay.NORM_RUN
    assert (lay.Q_LANES, lay.KV_LANES) == (24, 8)
    assert lay.QUANT_GROUP == 2 * lay.LANE_ELEMENTS and lay.Q_GROUPS == 12
    # a lane's 64 elements are one cluster's group of four tasks
    assert lay.LANE_ELEMENTS == lay.GROUP_ROWS


def test_quant_constants_are_aiters():
    # aiter kAmaxFloor (1e-10f) and MxDtypeConfig<FP8_E4M3>::inv_max_pos
    assert lay.AMAX_FLOOR_BITS == _f32_bits(1e-10)
    assert lay.INV_FP8_MAX_BITS == 0x3B124925 == _f32_bits(1.0 / 448.0)


def test_unwritten_word_is_a_pair_of_all_ones_nans_and_never_stored():
    assert lay.UNWRITTEN == 0xFFFFFFFF
    assert lay.QUIET_NAN_PAIR != lay.UNWRITTEN
    for half in (lay.QUIET_NAN_PAIR & 0xFFFF, lay.QUIET_NAN_PAIR >> 16):
        assert half & 0x7F80 == 0x7F80 and half & 0x7F  # a BF16 NaN


@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_the_k_quarter_is_resident_up_to_57_rows_and_staged_past(rows):
    plan = lay.front_plan(rows)
    assert plan.words <= GFX1250.max_workgroup_lds_bytes // 4
    assert plan.tiles == -(-rows // 16)
    if rows <= 57:
        assert plan.resident and plan.stages == tdm_split_slabs(7168) == 7
    else:
        assert not plan.resident and plan.stages == 3


@pytest.mark.parametrize("rows", (0, MAX_ROWS + 1))
def test_the_front_refuses_widths_past_the_mono_bound(rows):
    with pytest.raises(ValueError, match="rows"):
        lay.front_plan(rows)


def test_one_hand_off_buffer_and_scratch_serve_every_width():
    assert lay.qkv_words() == MAX_ROWS * 2048 // 2
    widest = lay.partial_bytes(MAX_ROWS, GATE_ROWS)
    assert all(
        lay.partial_bytes(rows, GATE_ROWS) <= widest for rows in range(1, MAX_ROWS + 1)
    )
    # tdm_split_scratch_bytes: a CTA, eight tasks, a tile of 32 lanes x 8 f32
    assert lay.partial_bytes(16, GATE_ROWS) == 228 * 8 * 1 * 256 * 4
