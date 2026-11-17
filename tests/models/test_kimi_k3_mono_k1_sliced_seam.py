# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only plan of the column-sliced AttnRes seam inside the resident K1."""

import pytest

from atom.models.kimi_k3.mono.kernels.attention.k1_sliced_seam_layout import (
    HIDDEN,
    K1SeamPlan,
)
from atom.models.kimi_k3.mono.kernels.stages import attn_res_sliced_layout as sliced
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import TDM_CLUSTER

# K1 launches a CTA a CU
BLOCKS = sliced.MAX_CTAS
PLANS = [K1SeamPlan(BLOCKS, cluster) for cluster in sliced.CLUSTERS]


def test_k1_launches_a_cta_a_cu():
    pytest.importorskip("torch", reason="the K3 mono config imports torch")
    from atom.models.kimi_k3.mono.config import GFX1250_BLOCKS

    assert GFX1250_BLOCKS == BLOCKS


def test_k1_clusters_are_the_seam_homes():
    plan = K1SeamPlan(BLOCKS)
    assert plan.cluster == TDM_CLUSTER
    assert plan.homes == BLOCKS // TDM_CLUSTER
    assert plan.max_tokens() >= 64
    assert plan.polls() == ((0, 0), (3, 1), (7, 2), (10, 3))


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: f"c{p.cluster}")
def test_every_slab_polls_its_ranks_first(plan):
    polled = {}
    for slab, rank in plan.polls():
        assert rank not in polled
        polled[rank] = slab
    assert sorted(polled) == list(range(plan.cluster))
    assert [slab for slab, _ in plan.polls()] == sorted(polled.values())
    for slab in range(plan.slabs):
        ranks = plan.slab_ranks(slab)
        assert 1 <= len(ranks) <= 2
        for rank in ranks:
            assert polled[rank] <= slab
    covered = set()
    for slab in range(plan.slabs):
        first = slab * plan.slab
        for rank in plan.slab_ranks(slab):
            low = max(first, rank * plan.geometry.columns)
            high = min(first + plan.slab, (rank + 1) * plan.geometry.columns)
            assert low < high
            covered.update(range(low, high))
    assert covered == set(range(HIDDEN))


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: f"c{p.cluster}")
def test_ready_pairs_are_distinct_and_in_bounds(plan):
    indices = [
        plan.ready_index(rank, home)
        for rank in range(plan.cluster)
        for home in range(plan.homes)
    ]
    assert sorted(indices) == list(range(plan.ready_pairs))


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: f"c{p.cluster}")
@pytest.mark.parametrize("gram", [False, True])
def test_scratch_and_lds_fit(plan, gram):
    for count in (1, 4, 6, sliced.MAX_CANDIDATES - 1):
        records = sliced.scratch_bytes(count, 64, plan.cluster, gram)
        size = plan.scratch_bytes(count, 64, gram)
        assert size >= records + plan.ready_pairs * 8
        assert (size - plan.ready_pairs * 8) % 16 == 0
        # K1's smem.red: 8 waves x 32 lanes x 8 FP32 accumulator elements
        assert plan.lds_words(count + 1, gram) <= 8 * 32 * 8


def test_refuses_partial_clusters():
    with pytest.raises(ValueError):
        K1SeamPlan(BLOCKS + 2)
