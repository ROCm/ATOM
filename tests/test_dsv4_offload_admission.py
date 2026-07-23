# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for DSV4 checkpoint admission + SWA IO-pin bookkeeping."""

from __future__ import annotations

import pytest

from atom.kv_transfer.offload.dsv4.admission import (
    DSV4CheckpointAdmission,
    DSV4SwaIoPins,
)


def test_admission_hands_out_distinct_slots():
    a = DSV4CheckpointAdmission(state_pool_size=3, max_inflight_saves=8)
    slots = {a.try_admit() for _ in range(3)}
    assert slots == {0, 1, 2}
    assert a.free_slots == 0 and a.inflight == 3


def test_admission_skips_when_pool_exhausted():
    a = DSV4CheckpointAdmission(state_pool_size=2, max_inflight_saves=8)
    assert a.try_admit() is not None
    assert a.try_admit() is not None
    assert a.try_admit() is None  # skip, do not block
    assert a.inflight == 2


def test_admission_skips_when_inflight_credit_exhausted():
    a = DSV4CheckpointAdmission(state_pool_size=8, max_inflight_saves=1)
    idx = a.try_admit()
    assert idx is not None
    assert a.try_admit() is None  # credit gate, even though slots remain
    assert a.free_slots == 7


def test_admission_complete_recycles():
    a = DSV4CheckpointAdmission(state_pool_size=1, max_inflight_saves=1)
    idx = a.try_admit()
    assert idx == 0
    assert a.try_admit() is None
    a.complete(idx)
    assert a.inflight == 0 and a.free_slots == 1
    assert a.try_admit() == 0  # reusable after completion


def test_admission_double_release_rejected():
    a = DSV4CheckpointAdmission(state_pool_size=2, max_inflight_saves=2)
    idx = a.try_admit()
    a.complete(idx)
    with pytest.raises(ValueError, match="double-released"):
        a.complete(idx)


def test_swa_pins_refcount_and_reuse_guard():
    pins = DSV4SwaIoPins()
    pins.pin([5, 7, 5])  # page 5 pinned twice
    assert pins.is_pinned(5) and pins.is_pinned(7)
    assert not pins.is_pinned(9)
    pins.unpin([5])
    assert pins.is_pinned(5)  # still one ref
    pins.unpin([5, 7])
    assert not pins.is_pinned(5) and not pins.is_pinned(7)
    assert pins.pinned_pages() == set()


def test_swa_pins_skip_negative():
    pins = DSV4SwaIoPins()
    pins.pin([-1, 3, -1])
    assert pins.pinned_pages() == {3}
    pins.unpin([-1, 3])
    assert pins.pinned_pages() == set()


def test_swa_pins_underflow_rejected():
    pins = DSV4SwaIoPins()
    with pytest.raises(ValueError, match="below zero"):
        pins.unpin([1])
