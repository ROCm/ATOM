# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fixtures every LMCache MP worker test file shares.

Imported into a test module, the autouse fixtures apply to its tests: step
events work on CPU torch, and every worker a test builds has its transfers
closed in teardown, so no drain thread outlives its test and reads another
test's clock.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.mp import step_events as mp_step_events
from atom.kv_transfer.offload.mp import worker as mp_worker
from atom.kv_transfer.offload.mp.drain import TransferDrain


class FakeCudaEvent:
    """An interprocess event stand-in: counts its records."""

    def __init__(self, interprocess: bool = False) -> None:
        self.interprocess = interprocess
        self.records = 0

    def record(self, stream=None) -> None:
        self.records += 1


@pytest.fixture(autouse=True)
def cpu_step_events(monkeypatch):
    """Step events on CPU torch, which has no interprocess events."""
    monkeypatch.setattr(
        mp_step_events,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(Event=FakeCudaEvent, current_stream=lambda: None)
        ),
    )


@pytest.fixture(autouse=True)
def close_workers(monkeypatch):
    """Close the transfers of every worker built during the test."""
    built = []
    init = mp_worker.LMCacheMPConnector.__init__

    def tracking_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(mp_worker.LMCacheMPConnector, "__init__", tracking_init)
    yield
    for worker in built:
        worker._transfers.close(timeout=5.0)


def step_event(worker) -> mp_step_events.PendingStepEvent:
    """The step's event, as `start_load_kv` hands to a submit."""
    return mp_step_events.PendingStepEvent(worker._step_events)


def settle(worker) -> None:
    """Wait for a drain round that started after this call, if one runs.

    A transfer that settles without the server (an invalid descriptor, a
    collapsed-TP non-writer's save) starts no drain thread, and an inline
    worker (native ATOM's) runs nothing between steps.
    """
    if isinstance(worker._transfers, TransferDrain) and worker._transfers.alive:
        finished = worker._transfers.wait_round(5.0)
        # A round that failed (a deadline, say) ends the thread; the step's
        # next `get_finished` raises what it recorded.
        assert (
            finished or worker._transfers.error is not None
        ), "no drain round finished"
