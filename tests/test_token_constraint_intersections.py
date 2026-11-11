# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch


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
