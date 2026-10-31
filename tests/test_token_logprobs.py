"""Reference probability and ragged speculative-row alignment tests."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_engine.token_logprobs import (
    capture_token_logits,
    gather_token_logprobs,
    unpack_token_logprobs,
)


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("k", [0, 20])
def test_grammar_masking_preserves_raw_target_probabilities(speculative, k):
    import numpy as np

    logits = torch.arange(6 * 32).reshape(6, 32).float().sin()
    original = logits.clone()
    batch = SimpleNamespace(return_logprobs=[False, True], top_logprobs=[0, k])
    metadata = (
        SimpleNamespace(
            num_draft_tokens_np=np.array([3, 1]),
            cu_num_draft_tokens=torch.tensor([3, 4]),
            target_logits_indices=torch.tensor([0, 1, 2, 4]),
            bonus_logits_indices=torch.tensor([3, 5]),
        )
        if speculative
        else None
    )
    ids = (
        torch.tensor([[1, 2, -1, -1], [5, 6, -1, -1]])
        if speculative
        else torch.tensor([1, 5])
    )
    snapshot = capture_token_logits(logits, batch, metadata)
    assert snapshot[0].shape == (2 if speculative else 1, 32)
    logits[:, :20] = float("-inf")
    actual = gather_token_logprobs(logits, ids, batch, metadata, snapshot=snapshot)
    expected = gather_token_logprobs(original, ids, batch, metadata)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)
    assert torch.isfinite(actual[0]).all()


@pytest.mark.parametrize("k", [0, 1, 5, 20])
def test_raw_target_distribution_and_top_candidates(k):
    logits = torch.tensor(
        [[2.0, -1.0, 4.0, 0.0, 3.0, -5.0], [-2.0, 3.0, 0.0, 1.0, 2.0, -4.0]]
    )
    ids = torch.tensor([0, 4])
    batch = SimpleNamespace(return_logprobs=[True, True], top_logprobs=[k, k])
    values, candidates, ranks = gather_token_logprobs(logits, ids, batch)
    reference = logits.double().log_softmax(-1)
    torch.testing.assert_close(
        values[:, 0, 0].double(), reference[torch.arange(2), ids], atol=1e-6, rtol=1e-6
    )
    for row in range(2):
        expected, expected_ids = reference[row].topk(min(k, 6))
        assert candidates[row, 0].tolist() == expected_ids.tolist()
        torch.testing.assert_close(
            values[row, 0, 1:].double(), expected, atol=1e-6, rtol=1e-6
        )
    assert ranks[:, 0].tolist() == [3, 2]


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_ragged_acceptance_correction_and_bonus_rows(accepted):
    # Request 0: target rows 0,1,2 + bonus row 3. Request 1: row 4 + bonus 5.
    logits = torch.arange(6 * 7, dtype=torch.float64).reshape(6, 7).sin().float()
    metadata = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([3, 4]),
        target_logits_indices=torch.tensor([0, 1, 2, 4]),
        bonus_logits_indices=torch.tensor([3, 5]),
    )
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, -1, -1]])
    ids[0, accepted + 1 :] = -1
    batch = SimpleNamespace(return_logprobs=[True, True], top_logprobs=[2, 2])
    payload = gather_token_logprobs(logits, ids, batch, metadata)
    tokens = {12: tuple(ids[0, : accepted + 1].tolist()), 7: (5, 6)}
    records = unpack_token_logprobs(tuple(t.tolist() for t in payload), [12, 7], tokens)
    reference = logits.double().log_softmax(-1)
    for i, token in enumerate(tokens[12]):
        assert records[12][i].logprob == pytest.approx(
            reference[i, token].item(), abs=1e-6
        )
    assert records[7][0].logprob == pytest.approx(reference[4, 5].item(), abs=1e-6)
    assert records[7][1].logprob == pytest.approx(reference[5, 6].item(), abs=1e-6)
    assert len(records[12]) == accepted + 1


def test_mixed_requests_do_not_score_disabled_rows():
    logits = torch.tensor([[float("nan")] * 4, [1.0, 2.0, 3.0, 4.0]])
    batch = SimpleNamespace(return_logprobs=[False, True], top_logprobs=[0, 2])
    values, _candidates, ranks = gather_token_logprobs(
        logits, torch.tensor([1, 2]), batch
    )
    assert torch.isfinite(values).all()
    assert values[0].count_nonzero() == 0
    assert ranks[1, 0] == 2


def test_zero_draft_step_uses_bonus_row():
    metadata = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([0]),
        target_logits_indices=torch.tensor([], dtype=torch.int64),
        bonus_logits_indices=torch.tensor([0]),
    )
    batch = SimpleNamespace(return_logprobs=[True], top_logprobs=[0])
    values, _, _ = gather_token_logprobs(
        torch.tensor([[2.0, 0.0]]), torch.tensor([[0, -1, -1]]), batch, metadata
    )
    assert values[0, 0, 0].item() == pytest.approx(-0.126928, abs=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU probability reference check"
)
def test_gpu_ragged_probabilities_match_cpu_reference():
    logits = torch.arange(6 * 257, dtype=torch.float32).reshape(6, 257).sin()
    ids = torch.tensor([[1, 2, -1, -1], [5, 6, -1, -1]])
    batch = SimpleNamespace(return_logprobs=[True, True], top_logprobs=[20, 20])

    def metadata(device):
        return SimpleNamespace(
            cu_num_draft_tokens=torch.tensor([3, 4], device=device),
            target_logits_indices=torch.tensor([0, 1, 2, 4], device=device),
            bonus_logits_indices=torch.tensor([3, 5], device=device),
        )

    expected = gather_token_logprobs(logits, ids, batch, metadata("cpu"))
    actual = gather_token_logprobs(logits.cuda(), ids.cuda(), batch, metadata("cuda"))
    for reference, result in zip(expected, actual):
        torch.testing.assert_close(result.cpu(), reference, atol=1e-5, rtol=1e-5)
