# SPDX-License-Identifier: MIT
"""The causal probes fire during target verification and nowhere else."""

from types import SimpleNamespace

import pytest
import torch
from atom.model_ops.engram_layer import EngramOp

from atom.utils import forward_context

from .dspark_projection_probe import serial_verify_operations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_serial_engram_replaces_the_batched_call_during_verification_only(
    monkeypatch, single_rank
):
    """A diagnostic that cannot be switched off is not a diagnostic.

    The probe answers "does batching Engram across the drafted rows change
    the result", so it has to change exactly one thing: the verify forward.
    A prefill or a draft pass reaching the serial path would make the
    comparison a statement about two different models.
    """
    torch.manual_seed(2169)
    context = SimpleNamespace(
        attn_metadata=SimpleNamespace(step=SimpleNamespace(tentative=False)),
        context=SimpleNamespace(is_draft=False),
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    # Register restoration before the diagnostic installs its wrapper.
    original_engram = EngramOp.forward
    monkeypatch.setattr(EngramOp, "forward", original_engram)
    stats = serial_verify_operations(["engram"])

    engram = EngramOp(1, hidden_size=32, engram_hidden_size=32, hc_mult=4).to(
        device="cuda", dtype=torch.bfloat16
    )
    # ATOM layers allocate uninitialized -- weights arrive from a checkpoint.
    engram.wkv.weight.data.normal_(std=0.1)
    engram.process_weights_after_loading()
    hidden = torch.randn(1, 3, 4, 32, device="cuda", dtype=torch.bfloat16)
    embeddings = torch.randn(1, 3, 32, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        # Not verifying: the probe stands aside, on both of its two reasons.
        engram(hidden, embeddings)
        context.context.is_draft = True
        context.attn_metadata.step.tentative = True
        engram(hidden, embeddings)
        assert stats["engram"]["calls"] == 0

        context.context.is_draft = False
        actual = engram(hidden, embeddings)
        expected = torch.cat(
            [
                original_engram(engram, hidden[:, i : i + 1], embeddings[:, i : i + 1])
                for i in range(3)
            ],
            dim=1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == hidden.shape
        assert stats["engram"] == {"calls": 1, "rows": 3, "max_rows": 3}
