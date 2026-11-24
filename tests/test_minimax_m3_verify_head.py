# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 in-graph verify head's kernels (`verify_head.M3VerifyHead.verify`)
on one GPU (one vocab shard): greedy rows against torch's argmax and ATOM's
exact-match verdict, sampled rows against softmax(l / T), forced acceptance
against its schedule."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

V = 50016  # one TP4 rank's shard of M3's 200064
RQ = 8


def _head(max_bs=16, rank=0, world=1):
    from atom.models.minimax_m3.verify_head import M3VerifyHead

    lm_head = SimpleNamespace(
        vocab_start_idx=rank * V,
        weight=torch.empty(V, 1, device="cuda"),
        num_embeddings=V * world,
        tp_size=world,
    )
    lm = SimpleNamespace(lm_head=lm_head, _step_input_ids=None)
    runner = SimpleNamespace(config=SimpleNamespace(max_num_seqs=max_bs))
    return M3VerifyHead(lm, runner, SimpleNamespace(mtp_k=RQ - 1))


def _reference(logits, ids, nb):
    """ATOM's greedy rejection arithmetic on torch's argmax."""
    target = logits.float().argmax(-1).view(nb, RQ).int()
    drafts = ids[1 : nb * RQ + 1].view(nb, RQ)[:, : RQ - 1]
    out = torch.full((nb, RQ), -1, dtype=torch.int32, device=logits.device)
    accepted = torch.zeros(nb, dtype=torch.int32, device=logits.device)
    for b in range(nb):
        k = 0
        while k < RQ - 1 and int(drafts[b, k]) == int(target[b, k]):
            out[b, k] = target[b, k]
            k += 1
        out[b, k] = target[b, k]
        accepted[b] = k
    return out, accepted


@pytest.mark.parametrize("nb", [1, 2, 5, 16])
def test_greedy_matches_argmax_and_exact_match(nb):
    torch.manual_seed(nb)
    head = _head()
    rows = nb * RQ
    logits = torch.randn(rows, V, device="cuda").to(torch.bfloat16)
    # a tie on row 0: the lowest id must win
    logits[0, 7] = logits[0, 9] = 100.0
    target = logits.float().argmax(-1).int()
    # drafts: request b agrees with the target for its first b % 8 positions
    ids = torch.randint(0, 200064, (rows + 1,), device="cuda", dtype=torch.int32)
    for b in range(nb):
        for j in range(RQ - 1):
            if j < b % RQ:
                ids[b * RQ + j + 1] = target[b * RQ + j]
            elif ids[b * RQ + j + 1] == target[b * RQ + j]:
                ids[b * RQ + j + 1] += 1
    temps = torch.full((16,), 1e-10, device="cuda")
    out, verdict = head.verify(logits, ids, temps, None)
    ref_out, ref_acc = _reference(logits, ids, nb)
    assert int(target[0]) == 7
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(verdict[0], ref_acc)
    torch.testing.assert_close(verdict[1], RQ - 1 - ref_acc)
    anchors = torch.arange(nb, device="cuda", dtype=torch.int32) * RQ + ref_acc
    torch.testing.assert_close(verdict[2], anchors)
    torch.testing.assert_close(verdict[3], out.gather(1, ref_acc.long()[:, None]).view(nb))


def test_sampled_rows_follow_softmax():
    head = _head()
    nb = 2
    rows = nb * RQ
    support = torch.tensor([3, 1000, 25000, 50015], device="cuda")
    base = torch.tensor([1.0, 0.0, -1.0, 0.5], device="cuda")
    temp = 0.7
    logits = torch.full((rows, V), -1e4, device="cuda")
    logits[:, support] = base
    logits = logits.to(torch.bfloat16)
    expected = torch.softmax(logits[0, support].float() / temp, -1)
    ids = torch.full((rows + 1,), 2, dtype=torch.int32, device="cuda")  # never drawn
    temps = torch.full((16,), temp, device="cuda")
    counts = torch.zeros(4, device="cuda")
    n_steps = 1000
    for _ in range(n_steps):
        out, verdict = head.verify(logits, ids, temps, None)
        head.step.add_(1)
        # every draft rejected at position 0: out[:, 0] is that row's draw
        assert int(verdict[0].max()) == 0
        drawn = out[:, 0]
        for i, v in enumerate(support):
            counts[i] += (drawn == v).sum()
    freq = counts / counts.sum()
    assert counts.sum() == n_steps * nb
    torch.testing.assert_close(freq, expected, atol=0.05, rtol=0)


def _four_shards(full_logits, ids, temps, heads):
    """The TP4 exchange on one GPU: each head packs its quarter of the vocab,
    the pairs are stacked rank-major as the all-gather lays them out."""
    packed = [
        h.shard_pack(full_logits[:, r * V : (r + 1) * V].contiguous(), temps).clone()
        for r, h in enumerate(heads)
    ]
    gathered = torch.cat(packed, dim=0)
    out, verdict = heads[0].verdict_from(gathered, ids, temps, None, full_logits.shape[0])
    for h in heads:
        h.step.add_(1)
        # each rank's verdict resets its own packed bests; here one ran for all
        h.packed64.fill_(-(2**63))
    return out.clone(), verdict.clone()


def test_four_shards_greedy_and_ties():
    torch.manual_seed(7)
    nb = 4
    rows = nb * RQ
    heads = [_head(rank=r, world=4) for r in range(4)]
    logits = torch.randn(rows, 4 * V, device="cuda").to(torch.bfloat16)
    # row 0: a tie across two shards (ids V + 5 and 3 * V + 2): the lower id wins
    logits[0, V + 5] = logits[0, 3 * V + 2] = 50.0
    # row 1: the max in the last shard only
    logits[1, 4 * V - 1] = 60.0
    ids = torch.randint(0, 4 * V, (rows + 1,), device="cuda", dtype=torch.int32)
    temps = torch.full((16,), 1e-10, device="cuda")
    out, verdict = _four_shards(logits, ids, temps, heads)
    ref_out, ref_acc = _reference(logits, ids, nb)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(verdict[0], ref_acc)
    # position 0's emitted id is its target whether accepted or the correction
    assert int(out[0, 0]) == V + 5


def test_four_shards_sampled_across_boundaries():
    nb = 2
    rows = nb * RQ
    heads = [_head(rank=r, world=4) for r in range(4)]
    support = torch.tensor([3, V + 1000, 2 * V + 7, 4 * V - 1], device="cuda")
    base = torch.tensor([0.2, 0.0, -0.5, 0.7], device="cuda")
    logits = torch.full((rows, 4 * V), -1e4, device="cuda")
    logits[:, support] = base
    logits = logits.to(torch.bfloat16)
    temp = 1.0
    expected = torch.softmax(logits[0, support].float() / temp, -1)
    ids = torch.full((rows + 1,), 2, dtype=torch.int32, device="cuda")
    temps = torch.full((16,), temp, device="cuda")
    counts = torch.zeros(4, device="cuda")
    for _ in range(1000):
        out, verdict = _four_shards(logits, ids, temps, heads)
        drawn = out[:, 0]
        for i, v in enumerate(support):
            counts[i] += (drawn == v).sum()
    assert counts.sum() == 1000 * nb
    torch.testing.assert_close(counts / counts.sum(), expected, atol=0.05, rtol=0)


def test_forced_acceptance_schedule():
    from atom.model_ops.rejection_sampler import (
        _get_synthetic_cond_rates,
        acceptance_length_to_rates,
    )

    head = _head(64)
    nb = 64
    rows = nb * RQ
    logits = torch.randn(rows, V, device="cuda").to(torch.bfloat16)
    ids = torch.randint(0, 200064, (rows + 1,), device="cuda", dtype=torch.int32)
    temps = torch.full((64,), 1.0, device="cuda")
    cond = _get_synthetic_cond_rates(
        tuple(acceptance_length_to_rates(4.04, RQ - 1)), torch.device("cuda")
    )
    total = 0.0
    steps = 50
    for _ in range(steps):
        out, verdict = head.verify(logits, ids, temps, cond)
        head.step.add_(1)
        acc = verdict[0]
        assert int(acc.min()) >= 3 and int(acc.max()) <= 4
        # accepted drafts are emitted as the drafts themselves
        drafts = ids[1 : rows + 1].view(nb, RQ)
        assert torch.equal(out[:, :3], drafts[:, :3])
        assert torch.equal(verdict[3], out.gather(1, acc.long()[:, None]).view(nb))
        total += float(acc.float().mean())
    mean_len = 1 + total / steps
    assert abs(mean_len - 4.04) < 0.02
