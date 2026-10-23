# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deployment capacity covers every score, candidate and graph bucket."""

import random
from itertools import pairwise

import pytest

from atom.models.qwen4_exp_mono.layout import qsa_post_scratch_layout, scratch_extent
from atom.models.qwen4_exp_mono.qsa_plan import (
    INDEX_GROUPS_PER_TASK,
    INDEX_SEGMENT,
    SELECT_FAN,
    SELECTED_GROUPS,
    QsaPlan,
    context_buckets,
)


@pytest.mark.parametrize(
    "capacity", [1, 255, 256, 257, 8191, 8192, 8193, 16384, 32768, 262144]
)
def test_capacity_covers_paged_scores_and_selection(capacity):
    plan = QsaPlan(capacity)
    assert plan.columns * 4 >= capacity
    assert plan.columns % INDEX_GROUPS_PER_TASK == 0
    assert plan.score_tasks * INDEX_GROUPS_PER_TASK == plan.columns
    assert plan.lists[0] * INDEX_SEGMENT >= plan.columns
    assert plan.lists[-1] == 1
    assert SELECT_FAN * SELECTED_GROUPS == INDEX_SEGMENT
    for children, parents in zip(plan.lists, plan.lists[1:]):
        assert (parents - 1) * SELECT_FAN < children <= parents * SELECT_FAN
    buckets = context_buckets(capacity)
    assert buckets[-1] == capacity
    assert len(buckets) == len(set(buckets))
    assert buckets[0] == min(256, capacity)


def test_every_candidate_list_fits_disjoint_workspace():
    rng = random.Random(0)
    for capacity in [8192, 8193, 32768, 262144] + [
        rng.randrange(1, 262145) for _ in range(100)
    ]:
        plan = QsaPlan(capacity)
        layout = qsa_post_scratch_layout(max_model_len=capacity)
        regions = sorted(layout.values())
        assert all(
            off + size <= next_off for (off, size), (next_off, _) in pairwise(regions)
        )
        assert scratch_extent(layout) >= max(off + size for off, size in regions)
        assert layout["qsa_scores"][1] == plan.columns * 4
        assert layout["qsa_score_ready"][1] == plan.score_tasks * 8
        for level, count in enumerate(plan.lists[:-1]):
            assert layout[f"qsa_select_ready_{level}"][1] == count * 8
            assert layout[f"qsa_candidates_{level}"][1] == count * SELECTED_GROUPS * 8


def test_8k_keeps_one_selection_list_and_no_tree_storage():
    assert QsaPlan(8192).lists == (1,)
    assert not any("candidates" in name for name in qsa_post_scratch_layout())


@pytest.mark.parametrize("capacity", [0, -1])
def test_nonpositive_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="positive"):
        QsaPlan(capacity)


@pytest.mark.parametrize("capacity", [1, 256, 257, 2048, 2049, 8192, 32768])
def test_all_visible_graph_bucket_is_preserved(capacity):
    assert context_buckets(capacity) == tuple(
        sorted({min(256, capacity), min(2048, capacity), capacity})
    )
