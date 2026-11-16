# SPDX-License-Identifier: MIT
"""The Engram side stream on a CU-masked hardware queue of its own, and the
plain stream it falls back to."""

import ctypes

import pytest
import torch

if not torch.cuda.is_available():
    # Before the import, which reaches Triton through the staging kernels.
    pytest.skip("needs a GPU stream", allow_module_level=True)

from atom.model_ops.engram.device import staging


def test_own_queue_stream_orders_work_like_any_stream():
    stream = staging._own_queue_stream(torch.device("cuda", 0))
    assert isinstance(stream, torch.cuda.Stream)
    x = torch.arange(1 << 20, device="cuda", dtype=torch.float32)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = x * 2 + 1
        done = torch.cuda.Event()
        done.record(stream)
    torch.cuda.current_stream().wait_event(done)
    assert torch.equal(y, x * 2 + 1)
    graph = torch.cuda.CUDAGraph()
    out = torch.empty_like(x)
    with torch.cuda.graph(graph):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out.copy_(x + 3)
        torch.cuda.current_stream().wait_stream(stream)
    for shift in range(3):
        x.add_(shift)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, x + 3)


def test_falls_back_to_a_plain_stream(monkeypatch):
    def missing(name):
        raise OSError(name)

    monkeypatch.setattr(ctypes, "CDLL", missing)
    stream = staging._own_queue_stream(torch.device("cuda", 0))
    assert type(stream) is torch.cuda.Stream
