# SPDX-License-Identifier: MIT
"""``tests/fake_tp_group.py``: a rank that fails ends the run with its own error,
and the ranks waiting on it leave the barrier holding the launch lock their
thread releases (no second error in a thread)."""

import pytest
import torch
from fake_tp_group import run_ranks


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_failing_rank_raises_its_own_error():
    def fn(rank, group):
        if rank == 0:
            raise ValueError("rank 0 failed")
        return group.all_gather(torch.ones(1))

    with pytest.raises(ValueError, match="rank 0 failed"):
        run_ranks(2, fn)


def test_all_gather_concatenates_the_ranks_in_order():
    outs = run_ranks(3, lambda rank, group: group.all_gather(torch.full((1,), rank)))
    for out in outs:
        assert out.tolist() == [0, 1, 2]
