# SPDX-License-Identifier: MIT
"""MTP row order, completion, packing and ownership without the model runner."""

import numpy as np
import pytest
import torch

from atom.spec_decode.output import MTPStatusQueue


@pytest.mark.parametrize("bs", [0, 1, 3, 128])
def test_layout_copy_count_and_fifo(fake_cuda, bs):
    queue = MTPStatusQueue(128, fake_cuda.stream)
    counts = torch.arange(2 * bs, dtype=torch.int32).reshape(2, bs)
    queue.send(counts, ready=fake_cuda.ready)
    rejected, bonus = queue.receive()
    np.testing.assert_array_equal(rejected, np.arange(bs, 2 * bs))
    np.testing.assert_array_equal(bonus, np.arange(bs))
    assert rejected.dtype == bonus.dtype == np.int32
    assert rejected.flags.owndata and bonus.flags.owndata
    assert len(fake_cuda.allocations) == 1
    assert fake_cuda.allocations[0].numel() == 256
    assert [shape for _, shape in fake_cuda.copies] == [(2, bs)]
    assert [call for call, _ in fake_cuda.calls] == [
        "wait_event",
        "copy",
        "record",
        "synchronize",
    ]
    assert queue.receive() is None


def test_prefill_and_decode_share_capacity_across_request_counts(fake_cuda):
    queue = MTPStatusQueue(128, fake_cuda.stream)
    zeros = torch.zeros(256, dtype=torch.int32)
    saved = []
    for verified, bs in ((True, 64), (False, 63), (True, 64), (False, 1), (True, 128)):
        counts = (
            torch.arange(2 * bs, dtype=torch.int32).reshape(2, bs)
            if verified
            else zeros[: 2 * bs].view(2, bs)
        )
        queue.send(counts, ready=fake_cuda.ready)
        saved.append((queue.receive(), counts.numpy().copy()))
    assert len(fake_cuda.allocations) == 1
    assert len(fake_cuda.copies) == len(saved)
    for (rejected, bonus), expected in saved:
        np.testing.assert_array_equal(rejected, expected[1])
        np.testing.assert_array_equal(bonus, expected[0])


def test_failed_submission_does_not_publish_an_output(fake_cuda):
    queue = MTPStatusQueue(4, fake_cuda.stream)
    fake_cuda.fail_copy = 1
    with pytest.raises(RuntimeError, match="copy submission failed"):
        queue.send(torch.zeros((2, 4), dtype=torch.int32), ready=fake_cuda.ready)
    assert queue.receive() is None
    assert not any(call == "record" for call, _ in fake_cuda.calls)


def test_failed_snapshot_does_not_release_or_drop_pending_storage(fake_cuda):
    queue = MTPStatusQueue(4, fake_cuda.stream)
    counts = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    queue.send(counts, ready=fake_cuda.ready)
    slot = queue._pending[0]

    class FailedSnapshot(np.ndarray):
        def copy(self):
            raise MemoryError("snapshot allocation failed")

    output = slot.output
    slot.output = output.view(FailedSnapshot)
    with pytest.raises(MemoryError, match="snapshot allocation failed"):
        queue.receive()
    assert list(queue._pending) == [slot] and not queue._free
    slot.output = output
    rejected, bonus = queue.receive()
    np.testing.assert_array_equal(rejected, [4, 5, 6, 7])
    np.testing.assert_array_equal(bonus, [0, 1, 2, 3])


def test_gpu_copy_count_and_no_reallocation_across_batch_sizes(gpu_stream):
    queue = MTPStatusQueue(128, gpu_stream)
    zeros = torch.zeros(256, dtype=torch.int32, device="cuda")
    sources = [
        torch.arange(3 * bs, dtype=torch.int32, device="cuda").reshape(3, bs)
        for bs in (64, 63, 64, 1, 128)
    ]
    ready = torch.cuda.Event()
    ready.record()
    queue.send(sources[0][:2], ready=ready)
    queue.receive()
    for verified in (False, True):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            for verdict in sources:
                bs = verdict.shape[1]
                counts = verdict[:2] if verified else zeros[: 2 * bs].view(2, bs)
                queue.send(counts, ready=ready)
                rejected, bonus = queue.receive()
                np.testing.assert_array_equal(
                    rejected, np.arange(bs, 2 * bs) if verified else np.zeros(bs)
                )
                np.testing.assert_array_equal(
                    bonus, np.arange(bs) if verified else np.zeros(bs)
                )
        counts = {event.key: event.count for event in profile.key_averages()}
        assert counts.get("aten::copy_", 0) == len(sources)
        assert counts.get("aten::empty", 0) == counts.get("aten::empty_like", 0) == 0


def test_gpu_pending_status_and_transport_storage_survive_reuse(gpu_stream):
    queue = MTPStatusQueue(4, gpu_stream)
    sources = [
        torch.arange(8, dtype=torch.int32, device="cuda").reshape(2, 4) + 20 * i
        for i in range(3)
    ]
    ready = torch.cuda.Event()
    ready.record()
    for source in sources:
        queue.send(source, ready=ready)
    first = queue.receive()
    queue.send(sources[2], ready=ready)
    saved = [first] + [queue.receive() for _ in range(3)]
    del queue
    for (rejected, bonus), i in zip(saved, (0, 1, 2, 2)):
        assert rejected.base is bonus.base is None
        np.testing.assert_array_equal(rejected, np.arange(4, 8) + 20 * i)
        np.testing.assert_array_equal(bonus, np.arange(4) + 20 * i)
