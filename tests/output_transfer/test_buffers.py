# SPDX-License-Identifier: MIT
"""Storage capacity, completion and ownership without model dependencies."""

import numpy as np
import pytest
import torch

from atom.model_engine.output_transfer import D2HBufferPool


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_request_count_changes_reuse_full_capacity_and_event(fake_cuda, dtype):
    pool = D2HBufferPool(128 * 5, fake_cuda.stream)
    saved = []
    for i, bs in enumerate((64, 63, 64, 1, 128, 0, 17)):
        source = torch.full((bs, 5), i, dtype=dtype)
        saved.append((pool.copy(source).result(), i, bs))
    assert len(fake_cuda.allocations) == 1
    assert fake_cuda.allocations[0].numel() == 128 * 5
    assert len({id(event) for call, event in fake_cuda.calls if call == "record"}) == 1
    assert [shape for _, shape in fake_cuda.copies] == [(bs, 5) for _, _, bs in saved]
    for array, i, bs in saved:
        assert array.flags.owndata and array.base is None
        np.testing.assert_array_equal(array, np.full((bs, 5), i))


def test_pending_outputs_keep_separate_slots_until_consumed(fake_cuda):
    pool = D2HBufferPool(20, fake_cuda.stream)
    pending = [pool.copy(torch.full((3, 5), i)) for i in range(3)]
    assert len({pointer for pointer, _ in fake_cuda.copies}) == 3
    first = pending[0].result()
    again = pool.copy(torch.full((3, 5), 9))
    assert len(fake_cuda.allocations) == 3
    np.testing.assert_array_equal(pending[1].result(), np.full((3, 5), 1))
    np.testing.assert_array_equal(pending[2].result(), np.full((3, 5), 2))
    np.testing.assert_array_equal(again.result(), np.full((3, 5), 9))
    np.testing.assert_array_equal(first, np.zeros((3, 5)))
    with pytest.raises(RuntimeError, match="already been consumed"):
        pending[0].result()


@pytest.mark.parametrize("rows", [False, True])
def test_waits_before_reading_and_returns_independent_storage(fake_cuda, rows):
    pool = D2HBufferPool(16, fake_cuda.stream)
    pending = pool.copy(torch.arange(8).reshape(2, 4), ready=fake_cuda.ready)
    slot = pending._slot

    class GuardedArray(np.ndarray):
        def copy(self):
            assert slot.done.synchronized, "host read preceded D2H completion"
            assert self.ndim == (1 if rows else 2), "unnecessary whole-array copy"
            return np.asarray(self).copy()

    slot.array_view = slot.array_view.view(GuardedArray)
    result = pending.result_rows() if rows else (pending.result(),)
    assert all(array.flags.owndata and array.base is None for array in result)
    assert [call for call, _ in fake_cuda.calls] == [
        "wait_event",
        "copy",
        "record",
        "synchronize",
    ]
    np.testing.assert_array_equal(
        np.asarray(result), np.arange(8).reshape((2, 4) if rows else (1, 2, 4))
    )


def test_dtype_change_replaces_storage_without_changing_received_outputs(fake_cuda):
    pool = D2HBufferPool(8, fake_cuda.stream)
    first = pool.copy(torch.arange(4, dtype=torch.int32)).result()
    second = pool.copy(torch.arange(4, dtype=torch.int64)).result()
    assert len(fake_cuda.allocations) == 2
    assert first.dtype == np.int32 and second.dtype == np.int64
    np.testing.assert_array_equal(first, second)


def test_over_capacity_fails_before_borrowing_or_allocating(fake_cuda):
    pool = D2HBufferPool(8, fake_cuda.stream)
    pool.copy(torch.arange(8)).result()
    with pytest.raises(ValueError, match="exceeds capacity"):
        pool.copy(torch.arange(9))
    pool.copy(torch.arange(3)).result()
    assert len(fake_cuda.allocations) == 1


@pytest.mark.parametrize("kind", ["empty", "shape", "dtype"])
def test_row_sources_must_have_one_layout(fake_cuda, kind):
    pool = D2HBufferPool(8, fake_cuda.stream)
    sources = {
        "empty": (),
        "shape": (torch.zeros(3), torch.zeros(4)),
        "dtype": (torch.zeros(3, dtype=torch.int32), torch.zeros(3, dtype=torch.int64)),
    }[kind]
    with pytest.raises(ValueError):
        pool.copy_rows(sources, ready=fake_cuda.ready)
    assert not fake_cuda.allocations and not fake_cuda.copies


def test_strided_sources_and_scalar_outputs(fake_cuda):
    pool = D2HBufferPool(12, fake_cuda.stream)
    for source in (torch.arange(12).reshape(3, 4).T, torch.tensor(5), torch.empty(0)):
        result = pool.copy(source).result()
        np.testing.assert_array_equal(result, source.numpy())
        assert result.flags.owndata


def test_real_async_copies_preserve_pending_and_received_outputs(gpu_stream):
    pool = D2HBufferPool(128 * 5, gpu_stream)
    producer = torch.cuda.Stream()
    pending, sources = [], []
    for i, bs in enumerate((1, 17, 4, 128)):
        with torch.cuda.stream(producer):
            source = torch.full((bs, 5), i, dtype=torch.int32, device="cuda")
            ready = torch.cuda.Event()
            ready.record(producer)
        sources.append(source)
        pending.append(pool.copy(source, ready=ready))
    saved = [item.result() for item in pending]
    for i, (source, array) in enumerate(zip(sources, saved)):
        source.add_(10)
        np.testing.assert_array_equal(
            pool.copy(source).result(), np.full(source.shape, i + 10)
        )
        np.testing.assert_array_equal(array, np.full(source.shape, i))
