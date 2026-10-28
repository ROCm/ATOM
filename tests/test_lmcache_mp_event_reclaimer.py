# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import gc
import importlib.util
import sys
from pathlib import Path

# Exercise ownership without importing GPU libraries on CPU-only runners.
_path = Path(__file__).parents[1] / "atom/kv_transfer/offload/mp/event_reclaimer.py"
_spec = importlib.util.spec_from_file_location("event_reclaimer", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
IPCEventReclaimer = _module.IPCEventReclaimer


class Owner:
    pass


class Event:
    def __init__(self, destroyed, name):
        self.destroyed = destroyed
        self.name = name

    def __del__(self):
        self.destroyed.append(self.name)


def test_future_release_does_not_destroy_ipc_events_inline():
    destroyed = []
    reclaimer = IPCEventReclaimer()
    owner = Owner()
    for name in ["producer", "completion"]:
        reclaimer.retain_until_released(owner, Event(destroyed, name))
    reclaimer.release()
    assert destroyed == []  # The transfer is still live.
    del owner
    gc.collect()
    assert destroyed == []  # Transfer bookkeeping is not a safe point.
    reclaimer.release()
    assert sorted(destroyed) == ["completion", "producer"]


def test_shared_batch_event_waits_for_every_future():
    destroyed = []
    reclaimer = IPCEventReclaimer()
    first, second = Owner(), Owner()
    event = Event(destroyed, "shared")
    reclaimer.defer(event)
    reclaimer.retain_until_released(first, event)
    reclaimer.retain_until_released(second, event)
    del event, first
    reclaimer.release()
    assert destroyed == []
    del second
    assert destroyed == []
    reclaimer.release()
    assert destroyed == ["shared"]


def test_pre_submit_rejection_is_reclaimed_at_next_safe_point():
    destroyed = []
    reclaimer = IPCEventReclaimer()
    reclaimer.defer(Event(destroyed, "rejected"))
    assert destroyed == []
    reclaimer.release()
    assert destroyed == ["rejected"]


def test_reclamation_does_not_accumulate_completed_events():
    destroyed = []
    reclaimer = IPCEventReclaimer()
    for index in range(1000):
        owner = Owner()
        reclaimer.retain_until_released(owner, Event(destroyed, index))
        del owner
        reclaimer.release()
    assert destroyed == list(range(1000))
    assert not reclaimer._retired


def test_ready_event_is_not_rerecorded_before_all_dma_completes():
    class Future:
        ready = False

        def query(self):
            return self.ready

    reclaimer = IPCEventReclaimer()
    first, second = Future(), Future()
    original = reclaimer.acquire_ready_event(object)
    reclaimer.track_ready_event(original, first)
    reclaimer.track_ready_event(original, second)
    first.ready = True
    other = reclaimer.acquire_ready_event(object)
    assert other is not original
    second.ready = True
    assert reclaimer.acquire_ready_event(object) is original
    assert len(reclaimer._ready_events) == 2


def test_unprovable_submission_keeps_its_ready_event():
    class Future:
        def query(self):
            raise RuntimeError("cannot prove DMA completion")

    reclaimer = IPCEventReclaimer()
    owner = Future()
    original = reclaimer.acquire_ready_event(object)
    reclaimer.track_ready_event(original, owner)
    assert reclaimer.acquire_ready_event(object) is not original


def test_pre_submit_rejection_keeps_other_event_owners():
    class Future:
        ready = False

        def query(self):
            return self.ready

    reclaimer = IPCEventReclaimer()
    owner = Future()
    event = reclaimer.acquire_ready_event(object)
    reclaimer.track_ready_event(event, owner)
    reclaimer.track_ready_event(event, None)
    assert reclaimer.acquire_ready_event(object) is not event
    owner.ready = True
    assert reclaimer.acquire_ready_event(object) is event
