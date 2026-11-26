# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""An abort that overtakes its Submit is held until the request arrives or
`UNMATCHED_ABORT_TTL_S` passes; a miss costs O(1), not a scan of every hold."""

import random
import threading
from collections import OrderedDict
from types import SimpleNamespace

from atom.model_engine import scheduler as sched


def _scanning_reference(aborts: dict, req_id, now) -> None:
    """The hold as it was: expire by scanning every entry on each miss."""
    for i in [i for i, t in aborts.items() if now - t > sched.UNMATCHED_ABORT_TTL_S]:
        del aborts[i]
    aborts[req_id] = now


def test_the_hold_matches_the_scan_it_replaces(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(sched, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    held = SimpleNamespace(_unmatched_aborts=OrderedDict())
    reference: dict = {}
    rng = random.Random(1)
    for op in range(20_000):
        clock[0] += rng.expovariate(2000 / sched.UNMATCHED_ABORT_TTL_S)
        req_id = rng.randrange(2_000)
        if rng.random() < 0.7:  # an abort no queue holds
            sched.Scheduler._remember_unmatched_abort(held, req_id)
            _scanning_reference(reference, req_id, clock[0])
            assert dict(held._unmatched_aborts) == reference, op
        else:  # admission consumes a held abort
            assert (held._unmatched_aborts.pop(req_id, None) is None) == (
                reference.pop(req_id, None) is None
            )


def test_expiring_stops_at_the_first_live_hold(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(sched, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    held = SimpleNamespace(_unmatched_aborts=OrderedDict())
    for req_id in range(3):
        sched.Scheduler._remember_unmatched_abort(held, req_id)
    clock[0] = sched.UNMATCHED_ABORT_TTL_S / 2
    sched.Scheduler._remember_unmatched_abort(held, 3)
    clock[0] = sched.UNMATCHED_ABORT_TTL_S + 1
    sched.Scheduler._remember_unmatched_abort(held, 4)
    assert list(held._unmatched_aborts) == [3, 4]


def test_the_decode_scheduler_holds_an_abort_it_cannot_match():
    decode = sched.DecodeScheduler.__new__(sched.DecodeScheduler)
    decode._prefill_lock = threading.Lock()
    decode.running, decode.waiting = [], []
    decode._unmatched_aborts = OrderedDict()
    assert decode.abort_request(5) is False
    assert list(decode._unmatched_aborts) == [5]
