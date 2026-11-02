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


def test_ready_event_is_never_rerecorded():
    """An exported event is never re-recorded: a peer that imported it would
    see its previous completion until the device runs the new record. Once
    every transfer that used it completed, it is retired for destruction."""

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
    third = reclaimer.acquire_ready_event(object)
    assert third is not original and third is not other
    assert original in reclaimer._retired
    assert all(e.event is not original for e in reclaimer._ready_events)


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
    reclaimer.acquire_ready_event(object)
    assert event not in reclaimer._retired  # its submitted owner still runs
    owner.ready = True
    reclaimer.acquire_ready_event(object)
    assert event in reclaimer._retired


class PendingEvent(Event):
    def __init__(self, destroyed, name):
        super().__init__(destroyed, name)
        self.done = False

    def query(self):
        return self.done


def test_unfinished_events_wait_for_a_later_safe_point():
    """Destroying an unfinished IPC event synchronizes on it, and its work can
    depend on a DP peer waiting in the collective this rank has not launched:
    such an event is kept until it completes."""
    destroyed = []
    reclaimer = IPCEventReclaimer()
    pending = PendingEvent(destroyed, "pending")
    reclaimer.defer(pending)
    reclaimer.defer(Event(destroyed, "plain"))
    reclaimer.release()
    assert destroyed == ["plain"]
    reclaimer.release()
    assert destroyed == ["plain"]  # Still running: never destroyed inline.
    pending.done = True
    del pending
    reclaimer.release()
    assert destroyed == ["plain", "pending"]


def test_background_reaper_destroys_off_the_releasing_thread():
    """A destroy that blocks (an imported IPC event whose work waits on this
    rank's next collective) must not block the runner's forward."""
    import threading

    threads = []

    class Blocking:
        def __init__(self, gate):
            self.gate = gate

        def query(self):
            return True  # HIP can report complete and still block on destroy

        def __del__(self):
            threads.append(threading.current_thread().name)
            self.gate.wait(5)

    gate = threading.Event()
    reclaimer = IPCEventReclaimer(background=True)
    reclaimer.defer(Blocking(gate))
    reclaimer.release()  # returns although the destroy blocks
    gate.set()
    assert reclaimer._reaper.flush(5)
    assert threads == ["atom-ipc-event-reaper"]
