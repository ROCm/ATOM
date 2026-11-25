# SPDX-License-Identifier: MIT
"""`DPEngineCoreProc._sync_dp_state`: the engine loop's three DP flags, each
MAX-reduced in its own slot of one all_reduce."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def sync_dp_state(monkeypatch):
    # The engine loop alone, not the aiter-backed worker RPC.
    async_proc = ModuleType("atom.model_engine.async_proc")
    async_proc.AsyncIOProcManager = object
    monkeypatch.setitem(sys.modules, "atom.model_engine.async_proc", async_proc)
    from atom.model_engine.engine_core import DPEngineCoreProc

    return DPEngineCoreProc._sync_dp_state


def _core():
    return SimpleNamespace(_shutting_down=False, dp_group=object(), label="dp0")


@pytest.mark.parametrize(
    "reduced",
    [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0, 0, 0)],
)
def test_each_flag_is_read_from_its_own_slot(monkeypatch, sync_dp_state, reduced):
    sent = []

    def all_reduce(tensor, op=None, group=None):
        sent.append(tensor.tolist())
        tensor.copy_(torch.tensor(reduced, dtype=tensor.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    flags = sync_dp_state(_core(), True, False, True)

    assert sent == [[1, 0, 1]]
    assert flags == tuple(flag == 1 for flag in reduced)
    assert all(type(flag) is bool for flag in flags)


def test_a_failed_all_reduce_shuts_the_loop_down(monkeypatch, sync_dp_state):
    def all_reduce(tensor, op=None, group=None):
        raise RuntimeError("peer lost")

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    core = _core()

    assert sync_dp_state(core, True, False, False) == (
        True,
        True,
        False,
    )
    assert core._shutting_down
