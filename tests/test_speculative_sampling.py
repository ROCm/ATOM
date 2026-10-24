# SPDX-License-Identifier: MIT
"""Target sampling law and ragged prefix selection for speculative decoding."""

import pytest
import torch

pytest.importorskip("aiter", reason="the sampler and rejection sampler are AITER ops")

from atom.model_ops.rejection_sampler import rejection_sample
from atom.model_ops.sampler import Sampler

ANCHOR = 7  # never a target below: a draft read from an anchor's slot is rejected


def verify_stream(draft_ids, widths):
    """The step's input ids for requests of ``widths`` drafts each -- an anchor,
    then the request's drafts -- and the logits row verifying each draft (the
    row before it)."""
    ids, rows, d = [], [], 0
    for width in widths:
        start = len(ids)
        ids.append(ANCHOR)
        ids.extend(draft_ids[d : d + width])
        rows.extend(range(start, start + width))
        d += width
    return (
        torch.tensor(ids, device="cuda", dtype=torch.int32),
        torch.tensor(rows, device="cuda", dtype=torch.int64),
    )


def bonus_rows(widths):
    """Each request's last row in `verify_stream`'s layout: anchor + drafts."""
    ends = torch.tensor(widths, device="cuda", dtype=torch.int64).add(1).cumsum(0)
    return ends - 1


def draw(probabilities, rows):
    """`rows` ids drawn from the two-token distribution ``probabilities``."""
    return (torch.rand(rows, device="cuda") >= probabilities[0]).to(torch.int32)


def two_token_batch(probabilities, drafts, width):
    """`rejection_sample` arguments for equal-width requests whose every
    verification row scores ``probabilities``, drafting ``drafts``.

    The accept rule is the only thing left varying between the two arms below,
    so both take this same batch and add their own `draft_probs`.

    ``target_logits`` is raw LOGITS here -- `rejection_sample` builds
    `p = softmax(logits / T)` itself, because ATOM's unfiltered verify draw
    goes through a fused sampler that never forms probabilities to borrow.
    """
    rows = drafts.numel()
    batch = rows // width
    ids, logits_rows = verify_stream(drafts.tolist(), [width] * batch)
    return {
        "input_ids": ids,
        "target_logits_indices": logits_rows,
        "num_spec_steps": width,
        "cu_num_draft_tokens": torch.arange(
            width, rows + 1, width, device="cuda", dtype=torch.int32
        ),
        "target_logits": probabilities.log().expand(rows, -1).contiguous(),
        "bonus_token_ids": torch.zeros(batch, device="cuda", dtype=torch.int32),
        "bonus_logits_indices": bonus_rows([width] * batch),
        # The ratio path's q rows, flat: request b's start at b * width.
        "draft_probs_row_base": torch.arange(batch, device="cuda", dtype=torch.int32)
        * width,
    }


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
    output, (accepted, _, _) = rejection_sample(
        *verify_stream([0] * (n * 2), [2] * n),
        2,
        cu,
        None,
        target_logits,
        bonus,
        bonus_rows([2] * n),
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
    output, (counts, rejected, anchors) = rejection_sample(
        *verify_stream([0] * len(target_ids), widths),
        5,
        cu,
        None,
        logits,
        bonus,
        bonus_rows(widths),
        target_token_ids=targets if stochastic else None,
    )
    for i, (width, accepted) in enumerate(zip(widths, accepts)):
        expected = [0] * accepted + [3 if accepted == width else 1]
        expected += [-1] * (6 - len(expected))
        assert output[i].tolist() == expected
    assert counts.tolist() == accepts
    assert rejected.tolist() == [5 - a for a in accepts]
    # The anchor is the row the last emitted token was sampled at.
    starts = [sum(w + 1 for w in widths[:i]) for i in range(len(widths))]
    assert anchors.tolist() == [s + a for s, a in zip(starts, accepts)]


@gpu
def test_ratio_acceptance_emits_the_target_law_from_a_draft_that_drew_another():
    """q deliberately != p: the emitted token still carries the TARGET's law.

    This is the losslessness claim, and it is only true of a draft token drawn
    from the q the kernel is handed. Pinning the draft to one id instead
    measures that id's conditional accept rate, `min(1, p0/q0) = 0.778`, and
    the test then reads as a kernel failure for a reason that is in the
    harness -- so the drafts are DRAWN from q, and armed below to say so.

    Both halves are asserted because they fail separately: the marginal at
    position 0 catches a residual that does not correct the accept bias, and
    the law at position 1 among the rows that got there catches a chain whose
    later positions are conditioned on the accept decision before them.
    """
    torch.manual_seed(4471)
    n = 20000
    p = torch.tensor([0.7, 0.3], device="cuda")
    q = torch.tensor([0.9, 0.1], device="cuda")
    drafts = draw(q, n * 2)
    # Armed: a proposal that happened to be p would make the ratio rule accept
    # everything and prove nothing about the residual.
    torch.testing.assert_close(
        torch.bincount(drafts.long(), minlength=2).float() / drafts.numel(),
        q,
        rtol=0,
        atol=0.015,
    )
    output, (accepted, _, _) = rejection_sample(
        **two_token_batch(p, drafts, 2),
        draft_probs=q.expand(n * 2, -1).contiguous(),
        temperatures=torch.ones(n, device="cuda"),
    )
    marginal = torch.bincount(output[:, 0].long(), minlength=2).float() / n
    torch.testing.assert_close(marginal, p, rtol=0, atol=0.015)
    # Position 1 exists for every request that got past position 0, whether it
    # holds the accepted draft or the residual correction.
    reached = output[accepted >= 1, 1].long()
    assert (reached >= 0).all()
    conditional = torch.bincount(reached, minlength=2).float() / reached.numel()
    torch.testing.assert_close(conditional, p, rtol=0, atol=0.015)


@gpu
@pytest.mark.parametrize("q0", (0.9, 0.7, 0.3))
def test_ratio_acceptance_accepts_at_least_as_often_as_exact_match(q0):
    """The PR's motivation, made testable: same drafts, same target, two rules.

    Exact match accepts a draft only when the target's own draw repeats it,
    `sum_x q(x) p(x)`; the ratio rule accepts `sum_x min(p(x), q(x))`, which is
    never smaller for any q. Both arms get the SAME sampled drafts, so this
    compares the accept rules alone -- it does not compare against the greedy
    proposal that ships today, which is a different and much stronger baseline.
    """
    torch.manual_seed(1783)
    n = 30000
    p = torch.tensor([0.7, 0.3], device="cuda")
    q = torch.tensor([q0, 1 - q0], device="cuda")
    drafts = draw(q, n * 2)
    batch = two_token_batch(p, drafts, 2)
    _, (ratio_accepted, _, _) = rejection_sample(
        **batch,
        draft_probs=q.expand(n * 2, -1).contiguous(),
        temperatures=torch.ones(n, device="cuda"),
    )
    # The same drafts against the target's own draw, which is what the default
    # path verifies by equality.
    _, (exact_accepted, _, _) = rejection_sample(
        **batch, draft_probs=None, target_token_ids=draw(p, n * 2)
    )
    ratio_rate = torch.minimum(p, q).sum().item()
    exact_rate = (p * q).sum().item()
    # Width 2 and both positions independent, so the mean accepted count is
    # `a + a**2` for a per-position acceptance of `a`.
    measured = torch.tensor(
        [ratio_accepted.float().mean().item(), exact_accepted.float().mean().item()]
    )
    torch.testing.assert_close(
        measured,
        torch.tensor([ratio_rate + ratio_rate**2, exact_rate + exact_rate**2]),
        rtol=0,
        atol=0.025,
    )
    assert measured[0] >= measured[1]
