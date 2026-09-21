# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Verification must use the request distribution at every emitted position."""

import pytest
import torch

sampler_module = pytest.importorskip("atom.model_ops.sampler", exc_type=ImportError)
rejection_module = pytest.importorskip(
    "atom.model_ops.rejection_sampler", exc_type=ImportError
)


@pytest.mark.parametrize("uniform_filters", [False, True])
def test_ragged_verification_uses_each_requests_parameters(
    monkeypatch, uniform_filters
):
    sampler = sampler_module.Sampler()
    captured = {}

    def sample(logits, temperatures, top_ks, top_ps, **kwargs):
        captured.update(
            temperatures=temperatures, top_ks=top_ks, top_ps=top_ps, **kwargs
        )
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    monkeypatch.setattr(sampler, "forward", sample)
    temperatures = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    top_ks = torch.tensor([7] if uniform_filters else [1, 2, 3, 4, 5])
    top_ps = torch.tensor([0.9] if uniform_filters else [0.5, 0.6, 0.7, 0.8, 0.9])
    sampler.sample_verification_tokens(
        torch.zeros(6, 8),
        torch.tensor([0, 2, 2, 3, 6]),
        temperatures,
        top_ks,
        top_ps,
    )
    owners = torch.tensor([1, 1, 3, 4, 4, 4])
    torch.testing.assert_close(captured["temperatures"], temperatures[owners])
    torch.testing.assert_close(
        captured["top_ks"], top_ks if uniform_filters else top_ks[owners]
    )
    torch.testing.assert_close(
        captured["top_ps"], top_ps if uniform_filters else top_ps[owners]
    )
    assert captured["needs_independent_noise"] is True


def test_single_request_temperature_is_expanded(monkeypatch):
    sampler = sampler_module.Sampler()

    def sample(logits, temperatures, top_ks, top_ps, **kwargs):
        torch.testing.assert_close(temperatures, torch.tensor([0.7, 0.7, 0.7]))
        assert top_ks is top_ps is None
        return torch.zeros(3, dtype=torch.int32)

    monkeypatch.setattr(sampler, "forward", sample)
    sampler.sample_verification_tokens(
        torch.zeros(3, 8), torch.tensor([3]), torch.tensor([0.7]), None, None
    )


def test_no_verification_rows_does_not_invoke_sampler(monkeypatch):
    sampler = sampler_module.Sampler()

    def unexpected(*args, **kwargs):
        pytest.fail("sampling an empty set of verification rows")

    monkeypatch.setattr(sampler, "forward", unexpected)
    result = sampler.sample_verification_tokens(
        torch.zeros(0, 8), torch.tensor([0, 0]), torch.ones(2), None, None
    )
    assert result.shape == (0,)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU sampling kernels"
)
def test_verification_emits_samples_at_mismatch_and_preserves_ragged_tails():
    def tensor(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    output, accepted = rejection_module.rejection_sample(
        tensor([2, 3, 4, 5, 6, 7]),
        3,
        tensor([0, 2, 3, 6]),
        None,
        torch.zeros(6, 16, device="cuda"),
        tensor([10, 11, 12, 13]),
        target_token_ids=tensor([2, 9, 4, 5, 6, 7]),
    )
    assert output.tolist() == [
        [10, -1, -1, -1],
        [2, 9, -1, -1],
        [4, 12, -1, -1],
        [5, 6, 7, 13],
    ]
    assert accepted.tolist() == [0, 1, 1, 3]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GPU sampling kernels"
)
def test_stochastic_verification_preserves_conditional_token_distribution():
    # Draft always proposes token 0. Both target positions independently have
    # P(0)=0.7, P(1)=0.3. Greedy verification makes P(0)=1; sharing the noise
    # across draft positions makes P(second=0 | first accepted)=1.
    torch.manual_seed(1847)
    requests = 8192
    # AITER's fused sampler uses 1024 threads x 4 entries; fill a whole tile.
    logits = torch.full((requests * 2, 4096), -100.0, device="cuda")
    logits[:, :2] = torch.tensor([0.7, 0.3], device="cuda").log()
    cumulative = torch.arange(1, requests + 1, device="cuda", dtype=torch.int32) * 2
    target = sampler_module.Sampler().sample_verification_tokens(
        logits, cumulative, torch.ones(requests, device="cuda"), None, None
    )
    output, _ = rejection_module.rejection_sample(
        torch.zeros(requests * 2, device="cuda", dtype=torch.int32),
        2,
        cumulative,
        None,
        logits,
        torch.zeros(requests, device="cuda", dtype=torch.int32),
        target_token_ids=target,
    )
    emitted = output.cpu()
    assert abs((emitted[:, 0] == 0).float().mean().item() - 0.7) < 0.025
    continued = emitted[emitted[:, 0] == 0, 1]
    assert abs((continued == 0).float().mean().item() - 0.7) < 0.025
