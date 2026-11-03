# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fixtures every LMCache MP worker test file shares.

Imported into a test module, the autouse fixtures apply to its tests: producer
events work on CPU torch, and every worker a test builds has its transfer
thread closed in teardown, so none outlives its test and reads another test's
clock.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.mp import transfer_thread
from atom.kv_transfer.offload.mp import worker as mp_worker


class FakeCudaEvent:
    """An interprocess event stand-in: counts its records."""

    def __init__(self, interprocess: bool = False) -> None:
        self.interprocess = interprocess
        self.records = 0

    def record(self, stream=None) -> None:
        self.records += 1

    def query(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def cpu_step_events(monkeypatch):
    """Producer events on CPU torch, which has no interprocess events."""
    cuda = SimpleNamespace(Event=FakeCudaEvent, current_stream=lambda: None)
    for module in (mp_worker, transfer_thread):
        monkeypatch.setattr(module, "torch", SimpleNamespace(cuda=cuda))


@pytest.fixture(autouse=True)
def close_workers(monkeypatch):
    """Close the transfer thread of every worker built during the test."""
    built = []
    init = mp_worker.LMCacheMPConnector.__init__

    def tracking_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(mp_worker.LMCacheMPConnector, "__init__", tracking_init)
    yield
    for worker in built:
        transfer, worker._transfer = worker._transfer, None
        if transfer is not None:
            transfer.close(timeout=5.0)


def step_event(worker) -> FakeCudaEvent:
    """The step's producer event, as `start_load_kv` hands to a submit."""
    return FakeCudaEvent(interprocess=True)


def run_off_step(worker, rank: int = 0) -> None:
    """The vLLM plugin's worker: submits and polls on its transfer thread."""
    worker._rank = rank
    worker.run_transfers_off_step()


def settle(worker) -> None:
    """Wait for a polling transfer thread's round that started after this call.

    Every submission queued before it has gone out by then. A worker that polls
    on the forward thread runs nothing between steps.
    """
    transfer = worker._transfer
    if transfer is not None and transfer.polls:
        assert transfer._wait_round(5.0), "no transfer-thread round finished"
