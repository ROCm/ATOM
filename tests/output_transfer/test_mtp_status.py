# SPDX-License-Identifier: MIT
"""MTP field ordering, FIFO and packing, independently of the model runner."""

from unittest import mock

import numpy as np
import pytest
import torch

from atom.spec_decode.output import MTPStatusQueue


@pytest.mark.parametrize("bs", [0, 1, 3, 128])
@pytest.mark.parametrize("packed", [False, True])
def test_layout_copy_count_and_fifo(fake_cuda, bs, packed):
    queue = MTPStatusQueue(128, fake_cuda.stream)
    verdict = torch.arange(3 * bs, dtype=torch.int32).reshape(3, bs)
    if packed:
        queue.send_verdict(verdict, fake_cuda.ready)
    else:
        queue.send_counts(verdict[1], verdict[0], fake_cuda.ready)
    future = mock.Mock()
    queue._pending.append(future)
    status = queue.receive()
    np.testing.assert_array_equal(status.rejected, np.arange(bs, 2 * bs))
    np.testing.assert_array_equal(status.bonus, np.arange(bs))
    assert status.rejected.dtype == status.bonus.dtype == np.int32
    assert status.rejected.flags.owndata and status.bonus.flags.owndata
    assert len(fake_cuda.allocations) == 1
    assert fake_cuda.allocations[0].numel() == 256
    assert [shape for _, shape in fake_cuda.copies] == (
        [(2, bs)] if packed else [(bs,), (bs,)]
    )
    assert [call for call, _ in fake_cuda.calls] == (
        ["wait_event"] + ["copy"] * (1 if packed else 2) + ["record", "synchronize"]
    )
    future.result_rows.assert_not_called()
    assert list(queue._pending) == [future]


def test_prefill_and_decode_share_capacity_across_request_counts(fake_cuda):
    queue = MTPStatusQueue(128, fake_cuda.stream)
    saved = []
    for packed, bs in ((True, 64), (False, 63), (True, 64), (False, 1), (True, 128)):
        verdict = torch.arange(3 * bs, dtype=torch.int32).reshape(3, bs)
        if packed:
            queue.send_verdict(verdict, fake_cuda.ready)
        else:
            queue.send_counts(verdict[1], verdict[0], fake_cuda.ready)
        saved.append((queue.receive(), bs))
    assert len(fake_cuda.allocations) == 1
    for status, bs in saved:
        np.testing.assert_array_equal(status.rejected, np.arange(bs, 2 * bs))
        np.testing.assert_array_equal(status.bonus, np.arange(bs))
    assert queue.receive() == (None, None)


@pytest.mark.parametrize("packed,fail_at", [(True, 1), (False, 1), (False, 2)])
def test_failed_submission_does_not_publish_an_output(fake_cuda, packed, fail_at):
    queue = MTPStatusQueue(4, fake_cuda.stream)
    fake_cuda.fail_copy = fail_at
    verdict = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    with pytest.raises(RuntimeError, match="copy submission failed"):
        if packed:
            queue.send_verdict(verdict, fake_cuda.ready)
        else:
            queue.send_counts(verdict[1], verdict[0], fake_cuda.ready)
    assert queue.receive() == (None, None)
    assert not any(call == "record" for call, _ in fake_cuda.calls)


def test_gpu_copy_count_and_no_reallocation_across_batch_sizes(gpu_stream):
    queue = MTPStatusQueue(128, gpu_stream)
    sources = [
        torch.arange(3 * bs, dtype=torch.int32, device="cuda").reshape(3, bs)
        for bs in (64, 63, 64, 1, 128)
    ]
    ready = torch.cuda.Event()
    ready.record()
    queue.send_verdict(sources[0], ready)
    queue.receive()
    for packed, expected_copies in ((False, 2), (True, 1)):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            for verdict in sources:
                if packed:
                    queue.send_verdict(verdict, ready)
                else:
                    queue.send_counts(verdict[1], verdict[0], ready)
                status = queue.receive()
                bs = verdict.shape[1]
                np.testing.assert_array_equal(status.rejected, np.arange(bs, 2 * bs))
                np.testing.assert_array_equal(status.bonus, np.arange(bs))
        counts = {event.key: event.count for event in profile.key_averages()}
        assert counts.get("aten::copy_", 0) == expected_copies * len(sources)
        assert counts.get("aten::empty", 0) == counts.get("aten::empty_like", 0) == 0


def test_gpu_pending_status_and_transport_storage_survive_reuse(gpu_stream):
    queue = MTPStatusQueue(4, gpu_stream)
    sources = [
        torch.arange(12, dtype=torch.int32, device="cuda").reshape(3, 4) + 20 * i
        for i in range(3)
    ]
    ready = torch.cuda.Event()
    ready.record()
    for source in sources:
        queue.send_verdict(source, ready)
    first = queue.receive()
    queue.send_counts(sources[2][1], sources[2][0], ready)
    saved = [first] + [queue.receive() for _ in range(3)]
    # Dropping the entire queue/pool cannot change any received array.
    del queue
    for status, i in zip(saved, (0, 1, 2, 2)):
        assert status.rejected.base is status.bonus.base is None
        np.testing.assert_array_equal(status.rejected, np.arange(4, 8) + 20 * i)
        np.testing.assert_array_equal(status.bonus, np.arange(4) + 20 * i)
