"""Reference probability and ragged speculative-row alignment tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_engine.token_logprobs import (
    _logprob_rows,
    capture_token_logits,
    gather_token_logprobs,
    pack_token_logprobs,
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
        num_draft_tokens_np=np.array([3, 1]),
        cu_num_draft_tokens=torch.tensor([3, 4]),
        target_logits_indices=torch.tensor([0, 1, 2, 4]),
        bonus_logits_indices=torch.tensor([3, 5]),
    )
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, -1, -1]])
    ids[0, accepted + 1 :] = -1
    batch = SimpleNamespace(return_logprobs=[True, True], top_logprobs=[2, 2])
    payload = gather_token_logprobs(logits, ids, batch, metadata)
    tokens = {12: tuple(ids[0, : accepted + 1].tolist()), 7: (5, 6)}
    records = unpack_token_logprobs(pack_token_logprobs(*payload), [12, 7], tokens)
    reference = logits.double().log_softmax(-1)
    for i, token in enumerate(tokens[12]):
        assert records[12][i][0] == pytest.approx(reference[i, token].item(), abs=1e-6)
    assert records[7][0][0] == pytest.approx(reference[4, 5].item(), abs=1e-6)
    assert records[7][1][0] == pytest.approx(reference[5, 6].item(), abs=1e-6)
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
        pack_token_logprobs(values, _candidates, ranks),
        [12, 7],
        {12: (1,), 7: (2,)},
        [False, True],
    )
    assert set(records) == {7}


def test_zero_draft_step_uses_bonus_row():
    metadata = SimpleNamespace(
        num_draft_tokens_np=np.array([0]),
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
            num_draft_tokens_np=np.array([3, 1]),
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


@pytest.mark.parametrize("k", [0, 1, 20])
def test_packed_payload_preserves_float_and_int32_bits(k):
    values = torch.randn(2, 3, k + 1)
    candidates = torch.full((2, 3, k), 2**24 + 17, dtype=torch.int32)
    ranks = torch.full((2, 3), 2**30 + 3, dtype=torch.int32)
    packed = pack_token_logprobs(values, candidates, ranks)
    assert packed.is_contiguous() and packed.dtype == torch.float32
    result = unpack_token_logprobs(packed, [4, 9], {4: (1, 2, 3), 9: (4,)})
    assert len(result[4]) == 3 and len(result[9]) == 1
    assert result[4][1][0] == values[0, 1, 0].item()
    assert result[4][1][1] == 2**30 + 3
    assert tuple(zip(result[4][1][2::2], result[4][1][3::2])) == tuple(
        (2**24 + 17, v) for v in values[0, 1, 1:].tolist()
    )


@pytest.mark.parametrize("counts", [[0, 0, 0], [0, 3, 1], [5, 0, 2]])
@pytest.mark.parametrize(
    "mask", [[True, True, True], [True, False, True], [False, True, False]]
)
def test_one_row_plan_preserves_compact_snapshot_and_padding(counts, mask):
    widths = np.array(counts) + 1
    starts = np.concatenate(([0], np.cumsum(widths)[:-1]))
    metadata = SimpleNamespace(
        num_draft_tokens_np=np.array(counts),
        cu_num_draft_tokens=torch.tensor(np.cumsum(counts)),
        target_logits_indices=torch.tensor(
            [int(starts[i] + j) for i, n in enumerate(counts) for j in range(n)],
            dtype=torch.long,
        ),
        bonus_logits_indices=torch.tensor(starts + counts),
    )
    batch = SimpleNamespace(return_logprobs=mask, top_logprobs=[3] * 3)
    logits = torch.randn(int(sum(widths)), 11)
    ids = torch.full((3, 6), 2)
    plan = _logprob_rows(batch, metadata, logits.device, 6)
    snapshot = capture_token_logits(logits, batch, metadata, rows=plan)
    assert snapshot.shape[0] == sum(widths[i] for i in range(3) if mask[i])
    original = logits.clone()
    logits.fill_(-100)
    values, _, _ = gather_token_logprobs(
        logits, ids, batch, metadata, snapshot=snapshot, rows=plan
    )
    for compact, req in enumerate(i for i in range(3) if mask[i]):
        for pos in range(6):
            row = starts[req] + min(pos, counts[req])
            expected = original[row].double().log_softmax(-1)[2]
            assert values[compact, pos, 0].item() == pytest.approx(
                expected.item(), abs=1e-6
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU row planning")
def test_staged_row_plan_needs_no_pageable_h2d_or_readback(monkeypatch):
    batch = SimpleNamespace(
        return_logprobs=[False, True],
        logprob_request_indices=torch.tensor([1], device="cuda"),
    )
    metadata = SimpleNamespace(
        num_draft_tokens_np=np.array([0, 2]),
        cu_num_draft_tokens=torch.tensor([0, 2], device="cuda"),
        target_logits_indices=torch.tensor([1, 2], device="cuda"),
        bonus_logits_indices=torch.tensor([0, 3], device="cuda"),
    )
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("row plan performed a host/device conversion")

        patch.setattr(torch, "tensor", forbidden)
        for method in ("cpu", "tolist", "item"):
            patch.setattr(torch.Tensor, method, forbidden)
        plan = _logprob_rows(batch, metadata, torch.device("cuda"), 4)
    assert plan.model_rows.tolist() == [1, 2, 3]
    assert plan.sample_rows.tolist() == [[0, 1, 2, 2]]


@pytest.mark.parametrize("lengths", [[4, 1, 3], [1, 1, 1], [2, 4, 2]])
@pytest.mark.parametrize("requested", [(0, 2), (1,), (0, 1, 2)])
def test_host_row_plan_matches_target_distribution(lengths, requested):
    from atom.model_engine.token_logprobs import LogprobRows, host_logprob_rows

    width = 4
    host = host_logprob_rows(requested, np.asarray(lengths), width)
    rows = LogprobRows(*(torch.from_numpy(part) for part in host))
    logits = torch.randn(sum(lengths), 11)
    ids = torch.arange(3 * width).reshape(3, width) % 11
    batch = SimpleNamespace(
        return_logprobs=[i in requested for i in range(3)], top_logprobs=[3] * 3
    )
    actual = gather_token_logprobs(logits, ids, batch, rows=rows)[0]
    starts = np.cumsum([0] + lengths[:-1])
    for compact, request in enumerate(requested):
        for slot in range(width):
            row = starts[request] + min(slot, lengths[request] - 1)
            expected = torch.log_softmax(logits[row], dim=-1)[ids[request, slot]]
            torch.testing.assert_close(actual[compact, slot, 0], expected)
