# SPDX-License-Identifier: MIT
"""The M3 indexer's prefill block scores live in one buffer sized at the batch
and context bounds, so the warmup that sizes the KV cache allocates them at
their serving size and a longer prompt never adds an allocator segment."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="index_topk defines @triton.jit kernels")

from atom.model_ops.minimax_m3 import index_topk as m

CPU = torch.device("cpu")


@pytest.fixture
def stream(monkeypatch):
    """A fake current stream (CPU torch has none) the test can switch."""
    current = SimpleNamespace(cuda_stream=1)
    monkeypatch.setattr(m.torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(m, "_PREFILL_SCORE_WORKSPACE", {})
    return current


def test_first_call_allocates_the_bound_and_longer_contexts_reuse_it(stream):
    heads, rows, blocks = 4, 256, 64  # 32768 x 8192 scaled down
    warmup = m.prefill_score_buffer(heads, rows, 2, CPU, (rows, blocks))
    assert warmup.shape == (heads, rows, 2) and warmup.is_contiguous()
    base = warmup.untyped_storage().data_ptr()
    assert warmup.untyped_storage().nbytes() == heads * rows * blocks * 4
    for width in (5, 37, blocks):  # ever longer contexts: no new allocation
        score = m.prefill_score_buffer(heads, rows, width, CPU, (rows, blocks))
        assert score.shape == (heads, rows, width) and score.is_contiguous()
        assert score.untyped_storage().data_ptr() == base


def test_past_the_bound_it_grows_and_keeps_the_larger_buffer(stream):
    heads = 4
    m.prefill_score_buffer(heads, 8, 4, CPU, (8, 4))
    big = m.prefill_score_buffer(heads, 8, 9, CPU, (8, 4))
    assert big.shape == (heads, 8, 9)
    again = m.prefill_score_buffer(heads, 8, 4, CPU, (8, 4))
    assert again.untyped_storage().data_ptr() == big.untyped_storage().data_ptr()


def test_each_stream_has_its_own_buffer(stream):
    a = m.prefill_score_buffer(4, 8, 4, CPU, (8, 4))
    stream.cuda_stream = 2
    b = m.prefill_score_buffer(4, 8, 4, CPU, (8, 4))
    assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()


def test_no_reserve_or_capture_keeps_a_tensor_per_call(stream, monkeypatch):
    a = m.prefill_score_buffer(4, 8, 4, CPU)
    b = m.prefill_score_buffer(4, 8, 4, CPU)
    assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    assert not m._PREFILL_SCORE_WORKSPACE
    monkeypatch.setattr(m.torch.cuda, "is_current_stream_capturing", lambda: True)
    m.prefill_score_buffer(4, 8, 4, CPU, (8, 4))
    assert not m._PREFILL_SCORE_WORKSPACE
