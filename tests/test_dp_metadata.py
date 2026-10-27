# SPDX-License-Identifier: MIT
"""How `DPMetadata` comes by every DP rank's running tokens.

The table sizes every DP collective in the forward, so one that does not
describe what the ranks will actually run shows up as a fixed-size collective
posted at mismatched heights -- a hang on eight ranks, nothing here.

`DPMetadata` takes the table whole; `gather_running_tokens` is the one way to a
table for a caller that knows only its own count. No aiter and no DP group:
the all_reduce is replaced, which is also what makes "did it ask the group"
observable at all.
"""

import sys
import types

import numpy as np
import pytest
import torch

from atom.utils.forward_context import (
    Context,
    DPMetadata,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
)


def _cfg(dp_size=4, dp_rank=1):
    return types.SimpleNamespace(data_parallel_size=dp_size, data_parallel_rank=dp_rank)


def test_the_sizes_are_read_off_the_table_it_is_given():
    md = DPMetadata(np.array([10, 336, 7, 1], dtype=np.int32))

    assert md.get_sizes_across_dp() == [10, 336, 7, 1]
    assert md.max_tokens_across_dp == 336


def test_a_group_of_one_gathers_without_a_collective(monkeypatch):
    """Make the ask itself fail, rather than count calls."""

    def _refuse(*a, **k):
        raise AssertionError("asked a group of one")

    monkeypatch.setattr(torch.distributed, "all_reduce", _refuse)

    table = DPMetadata.gather_running_tokens(_cfg(dp_size=1, dp_rank=0), 336)

    assert table.tolist() == [336]


def test_a_rank_contributes_its_own_entry_and_takes_the_reduction(monkeypatch):
    """Its answer is honoured whole -- the reduction, not this rank's own
    count, is the table."""
    peers = torch.tensor([10, 0, 7, 1], dtype=torch.int32)
    sent = {}

    def _all_reduce(table, group):
        sent["table"] = table.clone()
        table.add_(peers)

    group = types.SimpleNamespace(cpu_group="dp-cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        types.SimpleNamespace(get_dp_group=lambda: group),
    )

    table = DPMetadata.gather_running_tokens(_cfg(dp_size=4, dp_rank=1), 336)

    assert sent["table"].tolist() == [0, 336, 0, 0]
    assert table.tolist() == [10, 336, 7, 1]


@pytest.mark.parametrize("dp_size", [1, 4])
def test_set_forward_context_mirrors_the_table_onto_the_context(dp_size):
    """The table a forward is set with is the one its MoE all2all bounds its
    receive buffer by, and a DP group of one builds no `DPMetadata`."""
    config = types.SimpleNamespace(
        parallel_config=_cfg(dp_size=dp_size, dp_rank=0),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
    )
    context = Context(positions=torch.zeros(3, dtype=torch.long))
    table = np.arange(1, dp_size + 1, dtype=np.int32)
    try:
        set_forward_context(
            attn_metadata=None,
            atom_config=config,
            context=context,
            running_tokens_per_dp_rank=table,
        )
        assert context.running_tokens_across_dp == tuple(range(1, dp_size + 1))
        dp_metadata = get_forward_context().dp_metadata
        assert (dp_metadata is None) == (dp_size == 1)
    finally:
        reset_forward_context()
