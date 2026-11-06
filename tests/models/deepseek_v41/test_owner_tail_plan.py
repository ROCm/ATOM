# SPDX-License-Identifier: MIT
"""A replay frontier survives ragged TP ownership and empty owners."""

import numpy as np
import pytest

from atom.model_ops.deepseek_v41.owner_tail_plan import plan_tail_ownership


@pytest.mark.parametrize("rows,tp", [(1, 4), (5, 4), (4097, 4), (16381, 4), (8192, 2)])
def test_frontier_reconstructs_original_rows(rows, tp):
    rng = np.random.default_rng(128)
    candidates = [
        np.array([], dtype=np.int64),
        np.arange(rows),
        np.array([rows - 1]),
        rng.integers(0, rows, size=133),
    ]
    for selected in candidates:
        plan = plan_tail_ownership(selected, rows, tp)
        payloads = []
        for rank in range(tp):
            owned = np.arange(
                rank * plan.owner_capacity, (rank + 1) * plan.owner_capacity
            )
            packet = np.full(plan.packet_rows, -999)
            valid = plan.local_rows[rank] >= 0
            packet[valid] = owned[plan.local_rows[rank, valid]]
            payloads.append(packet)
        restored = np.concatenate(payloads)[plan.gathered_rows]
        np.testing.assert_array_equal(restored, selected)
        assert plan.packet_rows <= len(selected)


def test_tail_crosses_owner_and_omits_unfinished_request():
    selected = np.r_[np.arange(1000, 1029), np.arange(4070, 4097)]
    plan = plan_tail_ownership(selected, 4097, 4)
    restored = []
    for index in plan.gathered_rows:
        rank, slot = divmod(int(index), plan.packet_rows)
        restored.append(rank * plan.owner_capacity + plan.local_rows[rank, slot])
    np.testing.assert_array_equal(restored, selected)
    assert np.all(plan.local_rows[2] == -1)


@pytest.mark.parametrize("selected", [[-1], [4097], [[1, 2]]])
def test_refuses_padding_or_invalid_frontier(selected):
    with pytest.raises(ValueError):
        plan_tail_ownership(selected, 4097, 4)
