# SPDX-License-Identifier: MIT
"""Target sampling law and ragged prefix selection for speculative decoding."""

import pytest
import torch

pytest.importorskip("aiter", reason="the sampler and rejection sampler are AITER ops")

from atom.model_ops.rejection_sampler import rejection_sample
from atom.model_ops.sampler import Sampler


def test_verification_parameters_follow_ragged_request_rows(monkeypatch):
    observed = {}

    def capture(self, logits, temperatures, top_ks, top_ps, **kwargs):
        observed.update(
            temperatures=temperatures, top_ks=top_ks, top_ps=top_ps, **kwargs
        )
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    monkeypatch.setattr(Sampler, "forward", capture)
    Sampler().sample_verification_tokens(
        torch.zeros(5, 8),
        torch.tensor([0, 2, 2, 5]),
        torch.tensor([0.5, 1.0, 1.5, 2.0]),
        torch.tensor([3]),
        torch.tensor([0.6, 0.7, 0.8, 0.9]),
    )
    assert observed["temperatures"].tolist() == [1, 1, 2, 2, 2]
    torch.testing.assert_close(
        observed["top_ps"], torch.tensor([0.7, 0.7, 0.9, 0.9, 0.9])
    )
    assert observed["top_ks"].tolist() == [3]
    assert observed["needs_independent_noise"] and not observed["all_greedy"]


def test_no_draft_rows_need_no_sampling(monkeypatch):
    monkeypatch.setattr(Sampler, "forward", lambda *a, **k: pytest.fail("empty sample"))
    sampled = Sampler().sample_verification_tokens(
        torch.empty(0, 8), torch.zeros(3, dtype=torch.int32), torch.ones(3), None, None
    )
    assert sampled.shape == (0,) and sampled.dtype == torch.int32


def test_single_request_temperature_expands_to_each_verification_row(monkeypatch):
    def capture(self, logits, temperatures, *args, **kwargs):
        assert temperatures.tolist() == [0.5] * 5
        return torch.zeros(5, dtype=torch.int32)

    monkeypatch.setattr(Sampler, "forward", capture)
    Sampler().sample_verification_tokens(
        torch.zeros(5, 8), torch.tensor([5]), torch.tensor([0.5]), None, None
    )


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")


@gpu
def test_stochastic_verification_preserves_conditional_target_distribution():
    torch.manual_seed(9348)
    n = 30000
    probabilities = torch.tensor(
        [[0.65, 0.25, 0.10], [0.20, 0.50, 0.30], [0.35, 0.05, 0.60]], device="cuda"
    )
    # The pinned AITER sampler launches 1024 four-element lanes; exercise its
    # supported vector layout while keeping only three tokens probabilistic.
    log_probs = torch.nn.functional.pad(probabilities.log(), (0, 4093), value=-1000)
    target_logits = log_probs[:2].repeat(n, 1).contiguous()
    cu = torch.arange(2, n * 2 + 1, 2, device="cuda", dtype=torch.int32)
    sampler = Sampler()
    temperatures = torch.ones(n, device="cuda")
    targets = sampler.sample_verification_tokens(
        target_logits, cu, temperatures, None, None
    )
    bonus = sampler(
        log_probs[2].expand(n, -1).contiguous(),
        temperatures,
        needs_independent_noise=True,
    )
    output, accepted = rejection_sample(
        torch.zeros(n * 2, device="cuda", dtype=torch.int32),
        2,
        cu,
        None,
        target_logits,
        bonus,
        target_token_ids=targets,
    )
    # A draft of [0, 0] exposes position j only when the preceding target draws
    # were zero. Each exposed row must retain its conditional target law.
    for j in range(3):
        visible = (output[:, :j] == 0).all(-1)
        values = output[visible, j].long()
        assert (values >= 0).all()
        measured = torch.bincount(values, minlength=3).float() / values.numel()
        torch.testing.assert_close(measured, probabilities[j], rtol=0, atol=0.025)
    expected_accepts = torch.where(
        output[:, 0] != 0, 0, torch.where(output[:, 1] != 0, 1, 2)
    )
    assert torch.equal(accepted, expected_accepts)


@gpu
@pytest.mark.parametrize("top_k,top_p", [(2, 1.0), (-1, 0.8)])
def test_stochastic_verification_applies_target_filters(top_k, top_p):
    torch.manual_seed(3982)
    n = 20000
    probabilities = torch.tensor([0.5, 0.3, 0.15, 0.05], device="cuda")
    logits = probabilities.log().expand(n, -1).contiguous()
    ids = Sampler().sample_verification_tokens(
        logits,
        torch.arange(1, n + 1, device="cuda", dtype=torch.int32),
        torch.ones(n, device="cuda"),
        torch.tensor([top_k], device="cuda", dtype=torch.int32) if top_k > 0 else None,
        torch.tensor([top_p], device="cuda") if top_p < 1 else None,
    )
    # Both configurations retain the first two probabilities, renormalized.
    measured = torch.bincount(ids.long(), minlength=4).float() / n
    torch.testing.assert_close(
        measured, torch.tensor([0.625, 0.375, 0, 0], device="cuda"), rtol=0, atol=0.015
    )


@gpu
@pytest.mark.parametrize("stochastic", [False, True])
def test_target_prefixes_cover_every_ragged_acceptance_length(stochastic):
    widths, accepts = [], []
    for width in range(6):
        for accepted in range(width + 1):
            widths.append(width)
            accepts.append(accepted)
    cu = torch.tensor(widths, device="cuda", dtype=torch.int32).cumsum(0).int()
    target_ids = []
    for width, accepted in zip(widths, accepts):
        target_ids.extend([0] * accepted + [1] * (width - accepted))
    targets = torch.tensor(target_ids, device="cuda", dtype=torch.int32)
    # Deliberately give every logit row argmax=2. Stochastic verification must
    # consume the sampled target IDs, including the correction at rejection.
    logits = (
        torch.tensor([0.0, 0.0, 1.0], device="cuda")
        .expand(len(target_ids), -1)
        .contiguous()
    )
    bonus = torch.full((len(widths),), 3, device="cuda", dtype=torch.int32)
    if not stochastic:
        logits.scatter_(1, targets.long().unsqueeze(-1), 2)
    output, counts = rejection_sample(
        torch.zeros_like(targets),
        5,
        cu,
        None,
        logits,
        bonus,
        target_token_ids=targets if stochastic else None,
    )
    for i, (width, accepted) in enumerate(zip(widths, accepts)):
        expected = [0] * accepted + [3 if accepted == width else 1]
        expected += [-1] * (6 - len(expected))
        assert output[i].tolist() == expected
    assert counts.tolist() == accepts
