# SPDX-License-Identifier: MIT
"""Ensure causal probes actually run when target tensor stages replay graphs."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.engram_layer import EngramOp
from atom.models.deepseek_v41.execution import DenseGraphExecutor
from atom.utils import forward_context

from .dspark_projection_probe import serial_verify_operations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_serial_engram_intervenes_during_graph_replay_only(monkeypatch):
    torch.manual_seed(2169)
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(step=SimpleNamespace(tentative=False)),
        context=SimpleNamespace(is_draft=False),
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    # Register restoration before the diagnostic installs its wrappers.
    original_engram = EngramOp.forward
    monkeypatch.setattr(EngramOp, "forward", original_engram)
    monkeypatch.setattr(DenseGraphExecutor, "run", DenseGraphExecutor.run)
    stats = serial_verify_operations(["engram"])

    class Layer:
        def __init__(self):
            self.engram = EngramOp(1, hidden_size=32, engram_hidden_size=32, hc_mult=4)
            self.engram = self.engram.to(device="cuda", dtype=torch.bfloat16)

        def prepare_attention(self, hidden, embeddings):
            return (self.engram(hidden, embeddings),)

    layer = Layer()
    hidden = torch.randn(1, 3, 4, 32, device="cuda", dtype=torch.bfloat16)
    embeddings = torch.randn(1, 3, 32, device="cuda", dtype=torch.bfloat16)
    executor = DenseGraphExecutor()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        executor.run(
            layer.prepare_attention, hidden, embeddings, bucket=4, capture=True
        )
        executor.run(layer.prepare_attention, hidden, embeddings, bucket=4)
        assert stats["engram"]["calls"] == 0
        ordinary_replays = executor.replays
        context.attn_metadata.step.tentative = True
        (actual,) = executor.run(layer.prepare_attention, hidden, embeddings, bucket=4)
        expected = torch.cat(
            [
                original_engram(
                    layer.engram, hidden[:, i : i + 1], embeddings[:, i : i + 1]
                )
                for i in range(3)
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == hidden.shape
        # Verify uses the captured four-row bucket, including its zero padding.
        assert stats["engram"] == {"calls": 1, "rows": 4, "max_rows": 4}
        assert stats["engram_graph_stage_bypasses"] == 1
        assert executor.replays == ordinary_replays
        context.context.is_draft = True
        executor.run(layer.prepare_attention, hidden, embeddings, bucket=4)
        assert stats["engram"]["calls"] == 1
        assert executor.replays == ordinary_replays + 1
    stream.synchronize()
