# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hard token sets and sampling-boundary invariants, independent of a model."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.model_engine import token_constraints as boundary


def test_unconstrained_step_does_not_touch_backend_or_logits():
    runner = SimpleNamespace(structured_output=Mock())
    batch = SimpleNamespace(required_channel=None, structured_requests=None)
    assert boundary.TokenConstraints.apply(runner, batch, None, None) is None
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
    channel_state = (object(), torch.tensor([0, -1]))

    def channel_mask(runner, batch, bonus, metadata, target, preserve_masks):
        torch.testing.assert_close(preserve_masks, torch.tensor([True, False]))
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
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.triton_required_channel",
        SimpleNamespace(advance_states=advance),
    )
    step = boundary.TokenConstraints.apply(runner, batch, logits, metadata)
    assert events == ["grammar", "channel"]
    if speculative:
        # Channel constraints must not change the source used by verification logprobs.
        torch.testing.assert_close(logits[:, 6], raw[:, 6])
        assert torch.isneginf(step.target[:, 6:]).all()
    else:
        assert step.logits.shape == (2, 8)
        torch.testing.assert_close(logits[2:, 6], raw[2:, 6])
    final_ids = torch.tensor([[1, 2], [3, -1]])
    result = step.commit(final_ids, batch, deferred, step.failed(final_ids))
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
    batch = SimpleNamespace(structured_requests={0: object()}, required_channel=None)
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
    batch = SimpleNamespace(
        required_channel=channel, structured_requests=None, total_seqs_num=2
    )
    metadata = SimpleNamespace(
        cu_num_draft_tokens=object(), input_ids=object(), target_logits_indices=object()
    )
    kernel = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.triton_required_channel",
        SimpleNamespace(constrain_logits=kernel),
    )
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [2, 4097])
@pytest.mark.parametrize("token_stride", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_intersection_guard_respects_token_stride(dtype, vocab, token_stride, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a GPU for the real intersection kernel")
    storage = torch.full((6, vocab * token_stride), 7.0, dtype=dtype, device=device)
    logits = storage[::2, ::token_stride]
    logits.fill_(-float("inf"))
    storage[2].fill_(-float("inf"))
    logits[1, -1] = 7.0
    before = storage.clone()
    invalid = boundary._guard_empty_rows(
        logits, torch.tensor([True, True, False], device=device)
    )
    assert invalid.tolist() == [True, False, False]
    before[0, 0] = 0.0
    torch.testing.assert_close(storage, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("select_tail", [False, True])
def test_gpu_guard_row_address_exceeds_int32(dtype, select_tail):
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU for the real intersection kernel")
    stride = 2**30 + 17
    storage_bytes = (2 * stride + 1) * torch.empty((), dtype=dtype).element_size()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < storage_bytes + 64 * 1024**2:
        pytest.skip(f"Requires {storage_bytes} bytes of free GPU storage")
    logits = torch.empty_strided((3, 1), (stride, 1), dtype=dtype, device="cuda")
    logits[:, 0] = torch.tensor([3.0, -1.0, -float("inf")], dtype=dtype, device="cuda")
    selected = torch.tensor([True, True, select_tail], device="cuda")
    assert 2 * logits.stride(0) > 2**31 - 1
    invalid = boundary._guard_empty_rows(logits, selected)
    assert invalid.tolist() == [False, False, select_tail]
    expected_tail = 0.0 if select_tail else -float("inf")
    assert logits[:, 0].tolist() == [3.0, -1.0, expected_tail]
