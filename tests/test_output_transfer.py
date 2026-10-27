# SPDX-License-Identifier: MIT
"""D2H reuse, FIFO lifetime and owned outputs, independent of model imports."""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from atom.model_engine.output_transfer import D2HQueue
from atom.spec_decode.status import SpecStatusQueue


@pytest.fixture
def fake_cuda(monkeypatch):
    state = SimpleNamespace(
        stream=mock.Mock(), ready=mock.Mock(), allocations=[], events=[]
    )
    original_empty = torch.empty

    def empty(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        tensor = original_empty(*args, **kwargs)
        if pinned:
            state.allocations.append(tensor)
        return tensor

    def event():
        result = mock.Mock()
        state.events.append(result)
        return result

    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "Event", event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: state.stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    return state


def assert_owned_output(output, expected, *, status):
    arrays = output if status else (output,)
    rows = (expected[1], expected[0]) if status else (expected,)
    for array, row in zip(arrays, rows, strict=True):
        assert array.flags.owndata and array.base is None
        assert array.dtype == row.dtype
        np.testing.assert_array_equal(array, row)


@pytest.mark.parametrize("status", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_batch_changes_reuse_capacity_and_completion_event(fake_cuda, status, dtype):
    queue = (
        SpecStatusQueue(128, fake_cuda.stream)
        if status
        else D2HQueue(640, fake_cuda.stream)
    )
    saved = []
    for bs in (64, 63, 64, 1, 128, 0):
        shape = (2, bs) if status else (bs, 5)
        source = torch.arange(np.prod(shape), dtype=dtype).reshape(shape)
        queue.send(source, ready=fake_cuda.ready)
        saved.append((queue.receive(), source.numpy()))
    assert len(fake_cuda.allocations) == len(fake_cuda.events) == 1
    assert fake_cuda.allocations[0].numel() == queue.capacity
    assert fake_cuda.events[0].synchronize.call_count == len(saved)
    with pytest.raises(ValueError, match="exceeds capacity"):
        queue.send(torch.empty(queue.capacity + 1, dtype=dtype))
    for output, expected in saved:
        assert_owned_output(output, expected, status=status)


@pytest.mark.parametrize("status", [False, True])
def test_pending_and_received_outputs_survive_slot_reuse(fake_cuda, status):
    queue = (
        SpecStatusQueue(4, fake_cuda.stream)
        if status
        else D2HQueue(8, fake_cuda.stream)
    )
    sources = [torch.arange(8, dtype=torch.int32).view(2, 4) + 20 * i for i in range(3)]
    for source in sources:
        queue.send(source)
    first = queue.receive()
    queue.send(sources[2])
    assert len(fake_cuda.allocations) == 3
    outputs = [first] + [queue.receive() for _ in range(3)]
    assert queue.receive() is None
    del queue
    for output, source in zip(outputs, [*sources, sources[2]], strict=True):
        assert_owned_output(output, source.numpy(), status=status)


@pytest.mark.parametrize("status", [False, True])
def test_gpu_async_copy_count_and_reuse(status):
    if os.environ.get("RUN_D2H_GPU_TESTS") != "1":
        pytest.skip("opt-in real asynchronous D2H")
    stream, producer = torch.cuda.Stream(), torch.cuda.Stream()
    queue = SpecStatusQueue(128, stream) if status else D2HQueue(640, stream)
    with torch.cuda.stream(producer):
        zeros = torch.zeros(256, dtype=torch.int32, device="cuda")
        sources = []
        for i, bs in enumerate((64, 63, 64, 1, 128)):
            if status:
                source = (
                    zeros[: 2 * bs].view(2, bs)
                    if i % 2
                    else torch.arange(3 * bs, dtype=torch.int32, device="cuda").view(
                        3, bs
                    )[:2]
                )
            else:
                source = torch.arange(bs * 5, dtype=torch.int32, device="cuda").view(
                    bs, 5
                )
            sources.append(source)
        ready = torch.cuda.Event()
        ready.record(producer)
    # Warm the peak outstanding slots, then change every slot's active shape.
    for source in sources:
        queue.send(source, ready=ready)
    for _ in sources:
        queue.receive()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        for source in sources:
            queue.send(source, ready=ready)
        outputs = [queue.receive() for _ in sources]
    op_counts = {event.key: event.count for event in profile.key_averages()}
    assert op_counts.get("aten::copy_", 0) == len(sources)
    assert op_counts.get("aten::empty", 0) == op_counts.get("aten::empty_like", 0) == 0
    for output, source in zip(outputs, sources, strict=True):
        assert_owned_output(output, source.cpu().numpy(), status=status)
