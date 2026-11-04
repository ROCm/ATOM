"""Reference probability and ragged speculative-row alignment tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_engine.token_logprobs import (
    capture_token_logits,
    gather_token_logprobs,
    unpack_token_logprobs,
)


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
    assert values.shape == (1, 1, 3)
    assert ranks[0, 0] == 2
    records = unpack_token_logprobs(
        (values.tolist(), _candidates.tolist(), ranks.tolist()),
        [12, 7],
        {12: (1,), 7: (2,)},
        [False, True],
    )
    assert set(records) == {7}


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU scoring")
def test_scoring_batch_transitions_and_snapshot():
    # Exercise the scorer across the shapes a live mixed workload
    # changes between: prefill, zero-draft, ragged decode, and masked snapshots.
    for bs, drafts, k, snapshot in [
        (1, None, 0, False),
        (4, None, 5, False),
        (1, 0, 0, False),
        (4, 3, 20, False),
        (1, 5, 1, False),
        (4, None, 20, True),
        (1, None, 0, True),
        (4, 2, 5, True),
    ]:
        width = 1 if drafts is None else drafts + 1
        logits = torch.randn(bs * width, 257)
        # Exact ties must keep the established strictly-greater rank semantics.
        logits[:, :2] = 3.0
        ids = torch.zeros((bs, width), dtype=torch.int32)
        batch = SimpleNamespace(
            return_logprobs=[i % 2 == 0 for i in range(bs)],
            top_logprobs=[k] * bs,
        )
        if drafts is None:
            md = None
        else:
            md = SimpleNamespace(
                cu_num_draft_tokens=torch.arange(1, bs + 1) * drafts,
                num_draft_tokens_np=np.full(bs, drafts),
                target_logits_indices=torch.tensor(
                    [i * width + j for i in range(bs) for j in range(drafts)],
                    dtype=torch.long,
                ),
                bonus_logits_indices=torch.arange(bs) * width + drafts,
            )
            ids[:, 1:] = -1
        expected = gather_token_logprobs(logits, ids, batch, md)
        gpu_md = (
            None
            if md is None
            else SimpleNamespace(
                **{
                    name: value.cuda() if isinstance(value, torch.Tensor) else value
                    for name, value in vars(md).items()
                }
            )
        )
        gpu_logits = logits.cuda()
        saved = capture_token_logits(gpu_logits, batch, gpu_md) if snapshot else None
        if snapshot:
            gpu_logits.fill_(-100)  # Scores must use pre-mask rows.
        actual = gather_token_logprobs(
            gpu_logits, ids.cuda(), batch, gpu_md, snapshot=saved
        )
        # CPU and GPU topk may order exact ties differently. Scores and ranks
        # must agree; each returned ID must belong to that same tied score.
        for index in (0, 2):
            torch.testing.assert_close(
                actual[index].cpu(), expected[index], atol=2e-5, rtol=1e-5
            )
        for compact, seq in enumerate(i for i in range(bs) if batch.return_logprobs[i]):
            for position in range(width):
                row = seq * width + position
                reference = logits[row].double().log_softmax(-1)
                candidates = actual[1][compact, position].cpu().long()
                assert len(set(candidates.tolist())) == k
                torch.testing.assert_close(
                    actual[0][compact, position, 1:].cpu().double(),
                    reference[candidates],
                    atol=2e-5,
                    rtol=1e-5,
                )
