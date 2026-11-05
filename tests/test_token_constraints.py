# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hard token sets and sampling-boundary invariants, independent of a model."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

boundary = pytest.importorskip(
    "atom.model_engine.token_constraints", exc_type=ImportError
)


def test_unconstrained_step_does_not_touch_backend_or_logits():
    runner = SimpleNamespace(structured_output=Mock())
    batch = SimpleNamespace(required_channel=None)
    assert boundary.TokenConstraints.apply(runner, batch, None, None, True) is None
    boundary.prepare_token_constraints(runner, batch, None)
    assert runner.structured_output.mock_calls == []


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_masks_precede_sampling_and_commit_uses_final_ids(
    monkeypatch, speculative, deferred
):
    events = []
    grammar = Mock(vocab_size=8)
    prefixes = {0: object()}

    def grammar_mask(logits, batch, metadata):
        events.append("grammar")
        logits[:, 7] = -float("inf")
        return prefixes

    grammar.apply.side_effect = grammar_mask
    grammar.commit.side_effect = lambda *args: events.append("commit")
    runner = SimpleNamespace(structured_output=grammar)
    batch = SimpleNamespace(
        required_channel=object(), structured_requests={0: object()}, total_seqs_num=2
    )
    metadata = (
        SimpleNamespace(
            cu_num_draft_tokens=torch.tensor([1, 2]),
            bonus_logits_indices=torch.tensor([1, 3]),
            target_logits_indices=torch.tensor([0, 2]),
        )
        if speculative
        else None
    )
    logits = torch.arange(32, dtype=torch.float32).view(4, 8)
    raw = logits.clone()
    channel_state = (object(), object())

    def channel_mask(runner, batch, bonus, metadata, target):
        events.append("channel")
        assert torch.isneginf(bonus[:, 7]).all()
        bonus[:, 6] = -float("inf")
        if target is not None:
            assert torch.isneginf(target[:, 7]).all()
            target[:, 6] = -float("inf")
        return channel_state

    monkeypatch.setattr(boundary, "apply_required_channel", channel_mask)
    states = torch.tensor([3, 5])
    advance = Mock(return_value=states)
    monkeypatch.setattr(boundary, "advance_states", advance)
    step = boundary.TokenConstraints.apply(runner, batch, logits, metadata, True)
    assert events == ["grammar", "channel"]
    if speculative:
        assert step.premask_lse is None
        # Channel constraints must not change the source used by verification logprobs.
        torch.testing.assert_close(logits[:, 6], raw[:, 6])
        assert torch.isneginf(step.target[:, 6:]).all()
    else:
        torch.testing.assert_close(step.premask_lse, torch.logsumexp(raw[:2], dim=-1))
        assert step.logits.shape == (2, 8)
        torch.testing.assert_close(logits[2:, 6], raw[2:, 6])
    final_ids = torch.tensor([[1, 2], [3, -1]])
    result = step.commit(final_ids, batch, deferred)
    assert result is states if deferred else result is None
    grammar.commit.assert_called_once()
    committed, committed_batch, committed_prefixes = grammar.commit.call_args.args
    torch.testing.assert_close(committed, final_ids)
    assert committed_batch is batch
    assert committed_prefixes is prefixes
    if deferred:
        advance.assert_called_once_with(*channel_state, final_ids)
    else:
        advance.assert_not_called()


def test_prepare_preserves_deferred_metadata_identity():
    runner = SimpleNamespace(structured_output=Mock())
    batch = SimpleNamespace(structured_requests={0: object()})
    metadata = object()
    boundary.prepare_token_constraints(runner, batch, metadata)
    runner.structured_output.prepare.assert_called_once_with(batch, metadata)


@pytest.mark.parametrize("carry", [False, True])
def test_channel_backend_keeps_gpu_carry_and_draft_metadata(monkeypatch, carry):
    channel = object()
    device_channel = object()
    previous = torch.tensor([4, 5])
    source_rows = torch.tensor([1, 0])
    runner = SimpleNamespace(
        _device_channels={channel: device_channel},
        tokenID_processor=SimpleNamespace(
            carries_over=carry,
            prev_channel_states=previous,
            decode_src=SimpleNamespace(gpu=source_rows),
        ),
        forward_vars={
            "required_channel_states": SimpleNamespace(gpu=torch.tensor([0, 0]))
        },
    )
    batch = SimpleNamespace(required_channel=channel, total_seqs_num=2)
    metadata = SimpleNamespace(
        cu_num_draft_tokens=object(), input_ids=object(), target_logits_indices=object()
    )
    kernel = Mock(return_value=object())
    monkeypatch.setattr(boundary, "constrain_logits", kernel)
    bonus, target = object(), object()
    result = boundary.apply_required_channel(runner, batch, bonus, metadata, target)
    assert result == (device_channel, kernel.return_value)
    kwargs = kernel.call_args.kwargs
    assert kwargs["carried_states"] is (previous if carry else None)
    if carry:
        assert kwargs["carried_from"].data_ptr() == source_rows.data_ptr()
    else:
        assert kwargs["carried_from"] is None
    assert kwargs["input_ids"] is metadata.input_ids
    assert kwargs["cu_num_draft_tokens"] is metadata.cu_num_draft_tokens
    assert kwargs["target_logits_indices"] is metadata.target_logits_indices
    assert kwargs["target_logits"] is target
