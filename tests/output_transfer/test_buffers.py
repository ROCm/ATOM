# SPDX-License-Identifier: MIT
"""Storage capacity, completion and ownership without model dependencies."""

import numpy as np
import pytest
import torch

from atom.model_engine.output_transfer import D2HQueue
from atom.spec_decode.output import MTPStatusQueue


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_request_count_changes_reuse_full_capacity_and_event(fake_cuda, dtype):
    pool = D2HQueue(128 * 5, fake_cuda.stream)
    saved = []
    for i, bs in enumerate((64, 63, 64, 1, 128, 0, 17)):
        source = torch.full((bs, 5), i, dtype=dtype)
        pool.send(source)
        saved.append((pool.receive(), i, bs))
    assert len(fake_cuda.allocations) == 1
    assert fake_cuda.allocations[0].numel() == 128 * 5
    assert len({id(event) for call, event in fake_cuda.calls if call == "record"}) == 1
    assert [shape for _, shape in fake_cuda.copies] == [(bs, 5) for _, _, bs in saved]
    for array, i, bs in saved:
        assert array.flags.owndata and array.base is None
        np.testing.assert_array_equal(array, np.full((bs, 5), i))


def test_pending_outputs_keep_separate_slots_until_consumed(fake_cuda):
    pool = D2HQueue(20, fake_cuda.stream)
    for i in range(3):
        pool.send(torch.full((3, 5), i))
    assert len({pointer for pointer, _ in fake_cuda.copies}) == 3
    first = pool.receive()
    pool.send(torch.full((3, 5), 9))
    assert len(fake_cuda.allocations) == 3
    for i in (1, 2, 9):
        np.testing.assert_array_equal(pool.receive(), np.full((3, 5), i))
    np.testing.assert_array_equal(first, np.zeros((3, 5)))
    assert pool.receive() is None


@pytest.mark.parametrize("rows", [False, True])
def test_waits_before_reading_and_returns_independent_storage(fake_cuda, rows):
    pool = (
        MTPStatusQueue(8, fake_cuda.stream) if rows else D2HQueue(16, fake_cuda.stream)
    )
    pool.send(torch.arange(8).reshape(2, 4), ready=fake_cuda.ready)
    slot = pool._pending[0]

    class GuardedArray(np.ndarray):
        def copy(self):
            assert slot.done.synchronized, "host read preceded D2H completion"
            assert self.ndim == (1 if rows else 2), "unnecessary whole-array copy"
            return np.asarray(self).copy()

    slot.output = slot.output.view(GuardedArray)
    result = pool.receive() if rows else (pool.receive(),)
    assert all(array.flags.owndata and array.base is None for array in result)
    assert [call for call, _ in fake_cuda.calls] == [
        "wait_event",
        "copy",
        "record",
        "synchronize",
    ]
    np.testing.assert_array_equal(
        np.asarray(result),
        np.arange(8).reshape(2, 4)[::-1] if rows else np.arange(8).reshape(1, 2, 4),
    )


def test_dtype_change_replaces_storage_without_changing_received_outputs(fake_cuda):
    pool = D2HQueue(8, fake_cuda.stream)
    pool.send(torch.arange(4, dtype=torch.int32))
    first = pool.receive()
    pool.send(torch.arange(4, dtype=torch.int64))
    second = pool.receive()
    assert len(fake_cuda.allocations) == 2
    assert first.dtype == np.int32 and second.dtype == np.int64
    np.testing.assert_array_equal(first, second)


def test_over_capacity_fails_before_borrowing_or_allocating(fake_cuda):
    pool = D2HQueue(8, fake_cuda.stream)
    pool.send(torch.arange(8))
    pool.receive()
    with pytest.raises(ValueError, match="exceeds capacity"):
        pool.send(torch.arange(9))
    pool.send(torch.arange(3))
    pool.receive()
    assert len(fake_cuda.allocations) == 1


def test_strided_sources_and_scalar_outputs(fake_cuda):
    pool = D2HQueue(12, fake_cuda.stream)
    for source in (torch.arange(12).reshape(3, 4).T, torch.tensor(5), torch.empty(0)):
        pool.send(source)
        result = pool.receive()
        np.testing.assert_array_equal(result, source.numpy())
        assert result.flags.owndata


def test_real_async_copies_preserve_pending_and_received_outputs(gpu_stream):
    pool = D2HQueue(128 * 5, gpu_stream)
    producer = torch.cuda.Stream()
    sources = []
    for i, bs in enumerate((1, 17, 4, 128)):
        with torch.cuda.stream(producer):
            source = torch.full((bs, 5), i, dtype=torch.int32, device="cuda")
            ready = torch.cuda.Event()
            ready.record(producer)
        sources.append(source)
        pool.send(source, ready=ready)
    saved = [pool.receive() for _ in sources]
    for i, (source, array) in enumerate(zip(sources, saved)):
        source.add_(10)
        pool.send(source)
        np.testing.assert_array_equal(pool.receive(), np.full(source.shape, i + 10))
        np.testing.assert_array_equal(array, np.full(source.shape, i))
