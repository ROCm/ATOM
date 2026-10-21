# SPDX-License-Identifier: MIT
"""CPU transfer doubles and opt-in real GPU fixtures; no model-runner import."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def fake_cuda(monkeypatch):
    trace = SimpleNamespace(calls=[], allocations=[], copies=[], fail_copy=None)

    class Stream:
        def wait_stream(self, producer):
            trace.calls.append(("wait_stream", producer))

    class Event:
        synchronized = False

        def wait(self, *, stream):
            trace.calls.append(("wait_event", self))

        def record(self, stream):
            self.synchronized = False
            trace.calls.append(("record", self))

        def synchronize(self):
            self.synchronized = True
            trace.calls.append(("synchronize", self))

    original_empty = torch.empty
    original_copy = torch.Tensor.copy_

    def empty(*args, **kwargs):
        pinned = kwargs.pop("pin_memory", False)
        result = original_empty(*args, **kwargs)
        if pinned:
            assert kwargs["device"] == "cpu"
            trace.allocations.append(result)
        return result

    def copy(destination, source, *, non_blocking=False):
        assert non_blocking
        trace.copies.append((destination.data_ptr(), source.shape))
        trace.calls.append(("copy", source.shape))
        if trace.fail_copy == len(trace.copies):
            raise RuntimeError("copy submission failed")
        return original_copy(destination, source)

    trace.stream = Stream()
    trace.producer = Stream()
    trace.ready = Event()
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: trace.producer)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    return trace


@pytest.fixture
def gpu_stream():
    if os.environ.get("RUN_D2H_GPU_TESTS") != "1":
        pytest.skip("opt-in real asynchronous D2H")
    assert torch.cuda.is_available(), "RUN_D2H_GPU_TESTS requires a visible GPU"
    stream = torch.cuda.Stream()
    yield stream
    stream.synchronize()
