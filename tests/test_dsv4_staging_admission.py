# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from atom.kv_transfer.offload.hybrid.dsv4.policy import DSV4StagingAdmission


def _invalid_integer_scalars():
    np = pytest.importorskip("numpy")
    return [
        True,
        np.bool_(True),
        torch.tensor(True),
        torch.tensor(1),
    ]


def test_admission_acquires_refuses_releases_and_reuses_smallest_id():
    admission = DSV4StagingAdmission(3)

    assert admission.capacity == 3
    assert admission.num_free == 3
    assert [admission.try_acquire() for _ in range(4)] == [0, 1, 2, None]
    assert admission.num_free == 0

    admission.release(1)
    admission.release(0)

    assert admission.num_free == 2
    assert admission.try_acquire() == 0
    assert admission.try_acquire() == 1
    assert admission.try_acquire() is None


def test_admission_quarantine_permanently_removes_acquired_id():
    admission = DSV4StagingAdmission(1)
    slot_id = admission.try_acquire()

    admission.quarantine(slot_id)

    assert admission.num_free == 0
    assert admission.try_acquire() is None
    with pytest.raises(ValueError, match="quarantined"):
        admission.release(slot_id)


@pytest.mark.parametrize("slot_id", [-1, 2, 1.5, True])
def test_admission_rejects_invalid_release(slot_id):
    admission = DSV4StagingAdmission(2)

    with pytest.raises(ValueError, match="slot"):
        admission.release(slot_id)


def test_admission_constructor_rejects_boolean_and_tensor_scalars():
    for value in _invalid_integer_scalars():
        with pytest.raises(ValueError, match="num_slots must be an integer"):
            DSV4StagingAdmission(value)


def test_admission_release_rejects_boolean_and_tensor_scalars():
    admission = DSV4StagingAdmission(2)
    assert admission.try_acquire() == 0

    for value in _invalid_integer_scalars():
        with pytest.raises(ValueError, match="slot id must be an integer"):
            admission.release(value)

    assert admission.num_free == 1
    admission.release(0)


def test_admission_accepts_numpy_integer_scalars():
    np = pytest.importorskip("numpy")
    admission = DSV4StagingAdmission(np.int64(2))

    assert admission.try_acquire() == 0
    admission.release(np.int64(0))
    assert admission.num_free == 2


def test_admission_rejects_double_release():
    admission = DSV4StagingAdmission(1)
    assert admission.try_acquire() == 0
    admission.release(0)

    with pytest.raises(ValueError, match="not acquired"):
        admission.release(0)


@pytest.mark.parametrize("num_slots", [0, -1, 1.5, True])
def test_admission_rejects_invalid_capacity(num_slots):
    with pytest.raises(ValueError, match="num_slots"):
        DSV4StagingAdmission(num_slots)


def test_admission_is_thread_safe_under_basic_contention():
    capacity = 3
    competitors = 12
    admission = DSV4StagingAdmission(capacity)
    rendezvous = threading.Barrier(competitors)

    def compete(_):
        slot_id = admission.try_acquire()
        rendezvous.wait()
        if slot_id is not None:
            admission.release(slot_id)
        return slot_id

    with ThreadPoolExecutor(max_workers=competitors) as executor:
        results = list(executor.map(compete, range(competitors)))

    acquired = [slot_id for slot_id in results if slot_id is not None]
    assert sorted(acquired) == list(range(capacity))
    assert len(acquired) == len(set(acquired))
    assert results.count(None) == competitors - capacity
    assert admission.num_free == capacity


def test_admission_keeps_id_owned_until_caller_synchronizes_and_releases():
    admission = DSV4StagingAdmission(1)
    slot_id = admission.try_acquire()

    class _FakeTransfer:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    transfer = _FakeTransfer()

    assert slot_id == 0
    assert admission.try_acquire() is None

    transfer.synchronize()

    assert transfer.synchronized
    assert admission.try_acquire() is None
    admission.release(slot_id)
    assert admission.try_acquire() == slot_id
