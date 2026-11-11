# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.token_constraints import (
    TokenRuleBuffer,
    apply_token_rules_reference,
)
from atom.sampling_params import SamplingParams, TokenConstraint


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("spec", [False, True])
@pytest.mark.parametrize("until", [0, 1, 2, 4, None])
def test_rules_follow_each_actual_token_position(device, spec, until):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    # Row 0 is a long prompt with 3 generated inputs; row 1 has a shorter
    # prompt and no drafts. GPU positions include accepted deferred output.
    positions = torch.tensor([9, 10, 11, 12, 4], dtype=torch.int64)
    cu = torch.tensor([0, 4, 5], dtype=torch.int32)
    md = (
        SimpleNamespace(
            cu_num_draft_tokens=torch.tensor([3, 3], dtype=torch.int32),
            target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            bonus_logits_indices=torch.tensor([3, 4], dtype=torch.int64),
        )
        if spec
        else None
    )
    rules = [
        (0, 10, TokenConstraint.deny([0, 8], until=until)),
        (1, 5, TokenConstraint.allow([0, 2, 8])),
    ]
    last = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    target = torch.arange(27, dtype=torch.float32).reshape(3, 9) if spec else None
    expected = last.clone()
    target_expected = target.clone() if spec else None
    apply_token_rules_reference(rules, expected, target_expected, positions, cu, md)
    gpu_md = (
        SimpleNamespace(**{k: v.to(device) for k, v in vars(md).items()})
        if spec
        else None
    )
    actual = last.to(device)
    target_actual = target.to(device) if spec else None
    TokenRuleBuffer().apply(
        rules, actual, target_actual, positions.to(device), cu.to(device), gpu_md
    )
    torch.testing.assert_close(actual.cpu(), expected)
    if spec:
        torch.testing.assert_close(target_actual.cpu(), target_expected)
    # First target predicts generated token 0, so EOS remains banned until min.
    if spec and until is not None:
        assert torch.isneginf(target_expected[0, 0]).item() == (until > 0)


@pytest.mark.parametrize("value", [-1, True, 1.5, 9])
def test_min_tokens_validated(value):
    with pytest.raises(ValueError):
        SamplingParams(min_tokens=value, max_tokens=8)


def test_min_zero_preserves_old_empty_budget():
    SamplingParams(min_tokens=0, max_tokens=0)


def test_empty_intersection_reports_error_not_a_relaxed_rule():
    TokenConstraints = pytest.importorskip(
        "atom.model_engine.token_constraints", exc_type=ImportError
    ).TokenConstraints
    rules = [(0, 2, TokenConstraint.force(1)), (0, 2, TokenConstraint.deny([1]))]
    batch = SimpleNamespace(
        required_channel=None, token_constraints=rules, total_seqs_num=1
    )
    runner = SimpleNamespace(
        forward_vars={
            "positions": SimpleNamespace(gpu=torch.tensor([0, 1])),
            "cu_seqlens_q": SimpleNamespace(gpu=torch.tensor([0, 2])),
        }
    )
    logits = torch.ones(1, 4)
    step = TokenConstraints.apply(runner, batch, logits, None)
    sampled = logits.argmax(-1)
    assert step.failed(sampled).tolist() == [True]
    assert sampled.tolist() == [0]  # safe GPU carry, never an out-of-vocab sentinel


def test_rejected_future_draft_does_not_fail_current_output():
    TokenConstraints = pytest.importorskip(
        "atom.model_engine.token_constraints", exc_type=ImportError
    ).TokenConstraints
    step = TokenConstraints()
    step.logits = torch.zeros(1, 4)
    step.invalid = torch.tensor([True])
    step.target_invalid = torch.tensor([False, True])
    step.metadata = SimpleNamespace(cu_num_draft_tokens=torch.tensor([2]))
    assert step.failed(torch.tensor([[3, -1, -1]])).tolist() == [False]
    assert step.failed(torch.tensor([[3, 2, -1]])).tolist() == [True]


def test_public_min_tokens_are_not_silently_ignored():
    from atom.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )

    assert (
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hi"}], min_tokens=7
        ).min_tokens
        == 7
    )
    assert CompletionRequest(prompt="Hi", min_tokens=7).min_tokens == 7
    with pytest.raises(ValueError):
        CompletionRequest(prompt="Hi", min_tokens=True)


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("spec", [False, True])
def test_failure_uses_output_copy_without_poisoning_gpu_carry(deferred, spec):
    tokenIDProcessor = pytest.importorskip(
        "atom.model_engine.model_runner", exc_type=ImportError
    ).tokenIDProcessor
    p = object.__new__(tokenIDProcessor)
    p.is_deferred_out = deferred
    p.prev_batch = None
    p.token_ids_cpu = []
    p.recv_async_output = lambda _: p.pending if p.prev_batch else None
    p.recv_logprobs = lambda: None

    def send(ids, *args, **kwargs):
        p.pending = ids.clone()

    p.send_to_cpu_async = send
    batch = SimpleNamespace(req_ids=[71, 42], return_logprobs=[False, False])
    tokens = torch.tensor([[3, -1], [2, 1]]) if spec else torch.tensor([3, 2])
    original = tokens.clone()
    output, _ = p.prepare_sampled_ids(
        batch, tokens, None, constraint_failed=torch.tensor([True, False])
    )
    if deferred:
        assert output == {}
        assert p.prev_token_ids is tokens
        output, _ = p.prepare_sampled_ids(
            SimpleNamespace(req_ids=[99], return_logprobs=[False]), tokens[:1], None
        )
    assert output[71][0] == -2
    assert output[42] == ((2, 1) if spec else (2,))
    torch.testing.assert_close(tokens, original)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_channel_cannot_restore_schema_forbidden_token(device):
    pytest.importorskip("triton")
    from atom.model_ops.required_channel import (
        DeviceChannel,
        constrain_logits,
        constrain_logits_reference,
    )
    from atom.sampling_params import RequiredChannel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires GPU")
    channel = RequiredChannel(after=(1,), opening=(2,), opens_at=1, end_ids=(3,))
    logits = torch.ones(1, 8, device=device)
    logits[0, 2] = -float("inf")
    states = torch.tensor([1], dtype=torch.int32, device=device)
    if device == "cuda":
        constrain_logits(
            DeviceChannel.of(channel, torch.device(device)),
            states,
            logits,
            preserve_masks=True,
        )
    else:
        constrain_logits_reference(channel, states, logits, preserve_masks=True)
    assert torch.isneginf(logits).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rule_buffer_reuse_across_streams_and_noncontiguous_logits(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    buffer = TokenRuleBuffer()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for i in range(12):
        with torch.cuda.stream(streams[i % 2]):
            x = torch.ones(3, 34, device="cuda", dtype=dtype)[:, ::2]
            rules = [
                (0, 1, TokenConstraint.deny([i % 17])),
                (2, 1, TokenConstraint.allow([2, 4])),
            ]
            buffer.apply(
                rules,
                x,
                None,
                torch.arange(3, device="cuda"),
                torch.arange(4, device="cuda"),
                None,
            )
            outputs.append((x, i))
    torch.cuda.synchronize()
    for x, i in outputs:
        assert torch.isneginf(x[0, i % 17])
        assert torch.isfinite(x[1]).all()
        assert torch.isfinite(x[2]).sum() == 2


@pytest.mark.parametrize("minimum,stopped", [(0, True), (3, True), (4, False)])
def test_stop_string_obeys_minimum_in_verified_window(seq_factory, minimum, stopped):
    from unittest.mock import Mock

    from test_scheduler_mtp_max_tokens import TestMTPMaxTokens

    helper = TestMTPMaxTokens()
    sched = helper._scheduler()
    seq = helper._prefill(
        sched,
        seq_factory(
            [1, 2, 3, 4],
            stop_token_sequences=[[10, 11]],
            sampling_params=SamplingParams(
                max_tokens=16, min_tokens=minimum, ignore_eos=True
            ),
        ),
    )
    finished = helper._accept_all(sched, seq, [10, 11, 12, 13], Mock())
    assert bool(finished) == stopped
    if stopped:
        # The helper retains one already-accepted token before this window.
        assert finished[0].num_completion_tokens == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_transformed_error_ids_are_ready_before_async_copy():
    tokenIDProcessor = pytest.importorskip(
        "atom.model_engine.model_runner", exc_type=ImportError
    ).tokenIDProcessor
    p = object.__new__(tokenIDProcessor)
    p.is_deferred_out = True
    p.prev_batch = None
    p.token_ids_cpu = []
    p.logprobs_cpu = []
    p.async_copy_stream = torch.cuda.Stream()
    stream = torch.cuda.Stream()
    ready = torch.cuda.Event()
    with torch.cuda.stream(stream):
        tokens = torch.full((128, 6), 7, device="cuda", dtype=torch.int32)
        failed = torch.arange(128, device="cuda") % 2 == 0
        ready.record()
        for _ in range(5):
            # Leave work after the caller's ready event, before output encoding.
            torch.mm(
                torch.ones(512, 512, device="cuda"), torch.ones(512, 512, device="cuda")
            )
        p.prepare_sampled_ids(
            SimpleNamespace(req_ids=list(range(128))),
            tokens,
            ready,
            constraint_failed=failed,
        )
    out = p.recv_async_output(p.token_ids_cpu)
    assert (out[::2] == -2).all()
    assert (out[1::2] == 7).all()
    assert (p.prev_token_ids == 7).all()


@pytest.mark.parametrize("relaxed,synthetic", [(True, None), (False, [0.5])])
def test_unverified_acceptance_rejects_hard_rules_before_admission(
    monkeypatch, relaxed, synthetic
):
    from atom.model_engine import llm_engine

    monkeypatch.setattr(llm_engine.envs, "ATOM_ENABLE_RELAXED_MTP", relaxed)
    processor = object.__new__(llm_engine.InputOutputProcessor)
    processor.config = SimpleNamespace(
        speculative_config=SimpleNamespace(synthetic_acceptance_rates=synthetic)
    )
    with pytest.raises(ValueError, match="exact speculative verification"):
        processor.preprocess_fanout([1, 2], SamplingParams(min_tokens=2))


def test_new_sampling_fields_preserve_existing_positional_arguments():
    params = SamplingParams(0.0, -1, 1.0, 32, True)
    assert params.ignore_eos is True
    assert params.min_tokens == 0


def test_sparse_deny_retains_original_output_transfer_path():
    boundary = pytest.importorskip(
        "atom.model_engine.token_constraints", exc_type=ImportError
    )
    batch = SimpleNamespace(
        required_channel=None,
        token_constraints=[(0, 1, TokenConstraint.deny([2], until=3))],
        total_seqs_num=1,
    )
    runner = SimpleNamespace(
        forward_vars={
            "positions": SimpleNamespace(gpu=torch.tensor([0])),
            "cu_seqlens_q": SimpleNamespace(gpu=torch.tensor([0, 1])),
        }
    )
    logits = torch.ones(1, 8)
    step = boundary.TokenConstraints.apply(runner, batch, logits, None)
    assert step.invalid is None
    assert step.failed(torch.tensor([1])) is None
    assert torch.isneginf(logits[0, 2])
    assert torch.isfinite(logits[0, 1])


def test_rapidserve_rejects_explicit_rules_before_enqueue():
    from atom.model_engine import llm_engine

    processor = object.__new__(llm_engine.InputOutputProcessor)
    processor.config = SimpleNamespace(enable_rapidserve=True)
    with pytest.raises(ValueError, match="not supported by RapidServe"):
        processor.preprocess_fanout(
            [1, 2], SamplingParams(token_constraints=(TokenConstraint.force(2),))
        )


@pytest.mark.parametrize("nested", [False, True])
def test_added_token_without_output_head_row_rejected_before_enqueue(
    monkeypatch, nested
):
    from atom.model_engine import llm_engine

    monkeypatch.setattr(llm_engine.envs, "ATOM_ENABLE_RELAXED_MTP", False)
    processor = object.__new__(llm_engine.InputOutputProcessor)
    text_config = SimpleNamespace(vocab_size=8)
    processor.config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=text_config) if nested else text_config
    )
    processor.tokenizer = list(range(10))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        processor.preprocess_fanout(
            [1, 2], SamplingParams(token_constraints=(TokenConstraint.force(9),))
        )
