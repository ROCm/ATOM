# SPDX-License-Identifier: MIT
"""The warmup forward is a prefill, read off the batch's own sequences."""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter")

from atom.model_engine import model_runner as runner_module


@pytest.mark.parametrize("max_model_len,warmup_tokens", [(4096, 16384), (8192, 4096)])
def test_warmup_batch_is_a_prefill(monkeypatch, max_model_len, warmup_tokens):
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.config = SimpleNamespace(
        max_model_len=max_model_len,
        max_num_seqs=256,
        enable_dp_attention=True,
    )
    runner.block_size = 16
    runner.label = "test"
    runner.tokenID_processor = SimpleNamespace(clean=lambda: None)
    runner.warmup_max_tokens = lambda: warmup_tokens
    seen = []
    runner.forward = seen.append
    monkeypatch.setattr(
        runner_module, "get_dp_group", lambda: SimpleNamespace(world_size=2)
    )
    monkeypatch.setattr(runner_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        runner_module.torch.cuda, "reset_peak_memory_stats", lambda: None
    )

    runner.warmup_model()

    (batch,) = seen
    assert batch.has_prefill
    assert batch.total_tokens_num == sum(batch.num_scheduled_tokens)
