# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import SaveOperationId
from atom.kv_transfer.offload.mp.save_admission import MPSaveAdmission


def test_mp_save_admission_is_disabled_when_not_configured():
    admission = MPSaveAdmission.from_extra_config({})
    tracker = {
        "first": [SimpleNamespace(id=1), 0],
        "second": [SimpleNamespace(id=2), 0],
    }

    assert admission.enabled is False
    assert admission.ordered_sids(tracker, lambda _seq: 8) == ["first", "second"]
    assert (
        admission.reserve(
            SaveOperationId(1, 0),
            [1, 2],
            saved=0,
            aligned=8,
            block_size=4,
        )
        == "admitted"
    )
    assert admission.reserved_blocks == 0


@pytest.mark.parametrize("value", [True, 0, -1, "bad"])
def test_mp_save_admission_rejects_invalid_budget(value):
    with pytest.raises((TypeError, ValueError), match="max_pinned_save_blocks"):
        MPSaveAdmission.from_extra_config({MPSaveAdmission.CONFIG_KEY: value})


def test_mp_save_admission_prioritizes_finished_block_holders():
    admission = MPSaveAdmission(8, clock=lambda: 10.0)
    running = SimpleNamespace(id=1, frontier=32)
    finished_small = SimpleNamespace(
        id=2,
        frontier=16,
        _offload_finished_block_ids=[10, 11],
    )
    finished_large = SimpleNamespace(
        id=3,
        frontier=24,
        _offload_finished_block_ids=[20, 21, 22],
    )
    tracker = {
        "running": [running, 0],
        "finished-small": [finished_small, 0],
        "finished-large": [finished_large, 0],
    }

    assert admission.ordered_sids(tracker, lambda seq: seq.frontier) == [
        "finished-large",
        "finished-small",
        "running",
    ]


def test_mp_save_admission_counts_shared_blocks_once_and_releases_incrementally():
    admission = MPSaveAdmission(3)
    first = SaveOperationId(1, 0)
    second = SaveOperationId(2, 0)
    third = SaveOperationId(3, 0)

    assert (
        admission.reserve(first, [1, 2], saved=0, aligned=8, block_size=4) == "admitted"
    )
    assert (
        admission.reserve(second, [2, 3], saved=0, aligned=8, block_size=4)
        == "admitted"
    )
    assert admission.reserved_blocks == 3
    assert admission.reserve(third, [4], saved=0, aligned=4, block_size=4) == "busy"

    admission.source_safe(first, {2})
    assert admission.reserved_blocks == 2
    assert admission.reserve(third, [4], saved=0, aligned=4, block_size=4) == "admitted"


def test_mp_save_admission_serializes_one_oversized_save_for_progress():
    admission = MPSaveAdmission(1)
    oversized = SaveOperationId(1, 0)
    waiting = SaveOperationId(2, 0)

    assert (
        admission.reserve(
            oversized,
            [1, 2],
            saved=0,
            aligned=8,
            block_size=4,
        )
        == "admitted"
    )
    assert admission.oversized == 1
    assert admission.reserve(waiting, [3], saved=0, aligned=4, block_size=4) == "busy"
    admission.release(oversized)
    assert (
        admission.reserve(waiting, [3], saved=0, aligned=4, block_size=4) == "admitted"
    )
