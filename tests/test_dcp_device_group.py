# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""``dcp_utils.get_dcp_group``: the TP group's coordinator itself when the DCP
group is the TP group's ranks in the same order (every DCP collective then on
the TP communicator); the DCP group's own when the order differs or it is a
subset -- a collective splits by rank in group."""

import sys
import types

import pytest

from atom.distributed import dcp_utils


def _group(ranks, name):
    return types.SimpleNamespace(ranks=ranks, name=name)


@pytest.mark.parametrize(
    "dcp_ranks, expected",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], "tp"),  # DCP8 at TP8
        ([1, 0, 2, 3, 4, 5, 6, 7], "dcp"),  # the same ranks, another order
        ([0, 2, 4, 6], "dcp"),  # DCP4 inside TP8
    ],
)
def test_the_tp_coordinator_only_for_the_same_ranks_in_order(
    monkeypatch, dcp_ranks, expected
):
    # aiter's parallel state, stubbed (CI has no aiter): the TP group of 8
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: _group(list(range(8)), "tp")
    parallel_state.get_dcp_group = lambda: _group(dcp_ranks, "dcp")
    for name, module in (
        ("aiter", types.ModuleType("aiter")),
        ("aiter.dist", types.ModuleType("aiter.dist")),
        ("aiter.dist.parallel_state", parallel_state),
    ):
        module.__path__ = getattr(module, "__path__", [])
        monkeypatch.setitem(sys.modules, name, module)
    dcp_utils.get_dcp_group.cache_clear()
    try:
        assert dcp_utils.get_dcp_group().name == expected
    finally:
        dcp_utils.get_dcp_group.cache_clear()
