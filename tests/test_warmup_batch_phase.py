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
        speculative_config=None,
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


def test_every_stage_warms_up_on_the_rows_the_last_stage_does(monkeypatch):
    """A block drafter warms up on >= 2 sequences. Read off the drafter, a
    stage without one sent 1 x 5000 rows to a last stage expecting 2 x 4096."""
    monkeypatch.setattr(runner_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        runner_module.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    monkeypatch.setattr(
        runner_module, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    rows = []
    for drafter in (None, SimpleNamespace(is_block_drafter=True)):
        runner = object.__new__(runner_module.ModelRunner)
        runner.config = SimpleNamespace(
            max_model_len=5000,
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            enable_dp_attention=False,
            prefill_context_parallel_size=1,
            speculative_config=SimpleNamespace(use_dspark=lambda: True),
        )
        if drafter is not None:
            runner.drafter = drafter
        runner.block_size, runner.label = 16, "stage"
        runner.tokenID_processor = SimpleNamespace(clean=lambda: None)
        runner.forward = lambda batch: rows.append(
            (batch.total_seqs_num, batch.total_tokens_num)
        )
        runner.warmup_model()
    assert rows == [(2, 8192), (2, 8192)]
