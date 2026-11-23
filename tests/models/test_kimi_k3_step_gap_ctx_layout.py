# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft context KV projection's geometry (``ctx_kv_layout``), on a CPU:
every CTA's rows, every stored column, the output's and the scratch's bounds."""

import pytest

from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as front
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import ROWS, TDM_CLUSTER
from atom.models.kimi_k3.step_gap import ctx_kv_layout as lay


def test_a_layer_is_the_fronts_kv_and_rope_clusters():
    assert lay.FIRST_CLUSTER == 24 and lay.CLUSTERS == 9 and lay.LAYER_CTAS == 36
    assert lay.ENTRY == 576 == front.KV_LORA + front.ROPE
    kinds = [front.cluster_rows(lay.FIRST_CLUSTER + c)[0] for c in range(lay.CLUSTERS)]
    assert kinds == ["kv"] * 8 + ["rope"]
    # the weight rows a layer reads: 1536..2111 of the merged projection
    first = front.cluster_rows(lay.FIRST_CLUSTER)[1]
    last = (
        front.cluster_rows(lay.FIRST_CLUSTER + lay.CLUSTERS - 1)[1] + front.GROUP_ROWS
    )
    assert (first, last) == (front.Q_LORA, front.QKV_A_ROWS)


@pytest.mark.parametrize("layers", (1, 5, lay.MAX_LAYERS))
def test_every_cta_has_one_job_and_every_job_one_cta(layers):
    assert lay.grid(layers) == layers * 36 and lay.grid(layers) % TDM_CLUSTER == 0
    jobs = [lay.cta_job(cta, layers) for cta in range(lay.grid(layers))]
    assert len(set(jobs)) == len(jobs)
    assert {j[0] for j in jobs} == set(range(layers))
    assert {j[1] for j in jobs} == set(range(24, 33))
    assert {j[2] for j in jobs} == set(range(TDM_CLUSTER))
    # a TDM cluster of four CTAs is one (layer, cluster)
    for cta in range(0, lay.grid(layers), TDM_CLUSTER):
        assert len({lay.cta_job(cta + r, layers)[:2] for r in range(TDM_CLUSTER)}) == 1
    with pytest.raises(ValueError):
        lay.cta_job(lay.grid(layers), layers)


def test_the_stored_columns_tile_a_context_row_once():
    cols = sorted(
        lay.column(cluster, task, half) + e
        for cluster in range(lay.FIRST_CLUSTER, lay.FIRST_CLUSTER + lay.CLUSTERS)
        for task in range(front.TASKS)
        for half in range(2)
        for e in range(8)
    )
    assert cols == list(range(lay.ENTRY))
    # kv_a first, then the rope columns: what the cache write splits at 512
    assert lay.column(lay.FIRST_CLUSTER + 8, 0, 0) == front.KV_LORA
    assert front.TASKS * ROWS == front.GROUP_ROWS


@pytest.mark.parametrize(
    ("layers", "rows"), [(5, 1), (5, 8), (5, 35), (5, 64), (8, 64)]
)
def test_every_store_lands_inside_the_output(layers, rows):
    words = lay.out_bytes(layers, rows) // 4
    seen = set()
    for layer in range(layers):
        for token in range(rows):
            for cluster in range(lay.FIRST_CLUSTER, lay.FIRST_CLUSTER + lay.CLUSTERS):
                for task in range(front.TASKS):
                    for half in range(2):
                        # a lane stores four words: its eight BF16 columns
                        first = lay.out_word(
                            layer, rows, token, lay.column(cluster, task, half)
                        )
                        for w in range(first, first + 4):
                            assert 0 <= w < words
                            assert w not in seen
                            seen.add(w)
    assert len(seen) == words


@pytest.mark.parametrize("layers", (1, 5, lay.MAX_LAYERS))
def test_the_split_k_scratch_holds_every_ctas_slots(layers):
    tiles = -(-front.MAX_ROWS // ROWS)
    slots = lay.grid(layers) * front.TASKS * tiles * front.LANES
    # the front's slot index: [owner CTA][quarter][tile][lane], a v8 of f32
    last = (
        (((lay.grid(layers) - 1) * front.TASKS + front.TASKS - 1) * tiles + tiles - 1)
        * front.LANES
        + front.LANES
        - 1
    )
    assert last == slots - 1
    assert lay.partial_bytes(layers) == slots * 8 * 4


def test_refusals():
    assert lay.refusal(8, 4, 5) == "" and lay.refusal(64, 1, 5) == ""
    assert "rows" in lay.refusal(0, 4, 5) and "rows" in lay.refusal(65, 1, 5)
    assert "split-K" in lay.refusal(8, 2, 5)
    assert "layers" in lay.refusal(8, 4, 0) and "layers" in lay.refusal(8, 4, 9)
