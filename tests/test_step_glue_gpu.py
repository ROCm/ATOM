"""Step glue (atom/model_ops/step_glue.py) against the eager path it replaces, on one GPU.

shard_topk_packed: the per-shard top-k with the presence penalty applied on load, against
PresencePenaltyStep.apply + torch.topk (the same value set; ties go to the lower ids).
StepSampler.sample: the verify's candidate sampler, rejection, presence marks and next
anchors in one kernel, against sample_topk_candidates + rejection_sample + mark (greedy:
exact up to equal-logit ties; sampled: chi-square of the draws against the exact top-k /
top-p distribution; draws keyed by seed and step counter).
"""

import math
import types

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.presence_penalty import (
    PresencePenaltyState,
    PresencePenaltyStep,
)
from atom.model_ops.rejection_sampler import rejection_sample
from atom.model_ops.sampler import SAMPLER_EPS, sample_topk_candidates
from atom.model_ops.step_glue import StepSampler, shard_topk_packed

DEV = torch.device("cuda", 0)
TP, K, TOPK, VL = (
    4,
    20,
    20,
    62080,
)  # ranks, candidates a shard, sampler top-k, shard vocab
VOCAB = TP * VL
S = 4  # drafts a request


def make_logits(n, ties=True):
    x = torch.randn(n, VL, device=DEV) * 3.0
    if ties:
        x = (x * 4).round() / 4  # many equal bf16 values
    return x.to(torch.bfloat16)


def make_step(bs, q=S + 1, pen=1.5):
    state = PresencePenaltyState(VOCAB, 64, S, DEV)
    slots = torch.arange(bs, dtype=torch.int32, device=DEV) + 3
    state.mask.zero_()
    state.mask[slots.long()] = (torch.rand(bs, VOCAB, device=DEV) < 0.002).to(
        torch.uint8
    )
    ids = torch.randint(0, VOCAB, (bs * q,), device=DEV, dtype=torch.int32)
    seq_start = torch.arange(bs, dtype=torch.int32, device=DEV) * q
    row_seq = torch.arange(bs * q, dtype=torch.int32, device=DEV) // q
    row_pos = torch.arange(bs * q, dtype=torch.int32, device=DEV) % q
    pens = torch.full((bs,), pen, dtype=torch.float32, device=DEV)
    return PresencePenaltyStep(state, slots, pens, seq_start, row_seq, row_pos, ids)


def sorted_pairs(vals, ids):
    order = np.lexsort((ids, -vals))
    return vals[order], ids[order]


def candidates_from_full(full):
    """[N, VOCAB] fp32 logits -> the gathered [TP * N, 2K] candidates."""
    packs = []
    for r in range(TP):
        shard = full[:, r * VL : (r + 1) * VL].to(torch.bfloat16)
        packs.append(shard_topk_packed(shard, K, r * VL))
    return torch.cat(packs, 0)


def eager_candidates(gathered, n):
    g = gathered.view(TP, n, 2, K).permute(1, 2, 0, 3).reshape(n, 2, TP * K)
    return g[:, 0], g[:, 1].long()


def spec_meta(bs, nd=S):
    q = nd + 1
    tgt = torch.cat([torch.arange(r * q, r * q + nd) for r in range(bs)]).to(
        DEV, torch.int32
    )
    bonus = torch.tensor([r * q + nd for r in range(bs)], device=DEV, dtype=torch.int32)
    cu = torch.arange(1, bs + 1, device=DEV, dtype=torch.int32) * nd
    return types.SimpleNamespace(
        target_logits_indices=tgt,
        bonus_logits_indices=bonus,
        cu_num_draft_tokens=cu,
        num_spec_steps=nd,
    )


@pytest.mark.parametrize("n", [5, 85, 128])
@pytest.mark.parametrize("penalty", [False, True])
def test_shard_topk_matches_torch_topk(penalty, n):
    torch.manual_seed(n)
    if penalty and n % (S + 1) != 0:
        pytest.skip("the penalty step needs whole requests")
    logits = make_logits(n)
    vocab_start = VL * 2
    step = make_step(n // (S + 1), pen=1.5) if penalty else None
    ref_logits = logits.clone()
    if step is not None:
        mask_before = step.state.mask.clone()
        step.apply(ref_logits, vocab_start)  # marks the anchors, then the penalty
        step.state.mask.copy_(mask_before)
        step._anchors_marked = False
    rv, ri = torch.topk(ref_logits, K, dim=-1)
    got = shard_topk_packed(logits, K, vocab_start, step)
    gv, gi = got[:, :K].cpu().numpy(), got[:, K:].cpu().numpy().astype(np.int64)
    rv_, ri_ = rv.float().cpu().numpy(), (ri + vocab_start).cpu().numpy()
    for r in range(n):
        a_v, a_i = sorted_pairs(gv[r], gi[r])
        b_v, b_i = sorted_pairs(rv_[r], ri_[r])
        assert np.array_equal(a_v, b_v), f"row {r}: values differ"
        thr = a_v[-1]
        assert set(a_i[a_v > thr]) == set(b_i[b_v > thr]), f"row {r}: ids differ"
        # at the threshold value: the lowest ids that hold it
        row = ref_logits[r].float().cpu().numpy()
        eq_ids = np.nonzero(row == thr)[0] + vocab_start
        want = np.sort(eq_ids)[: int((a_v == thr).sum())]
        assert np.array_equal(np.sort(a_i[a_v == thr]), want), f"row {r}: tie ids"


@pytest.mark.parametrize("bs", [1, 17, 25])
def test_step_sampler_greedy_matches_eager(bs):
    torch.manual_seed(bs)
    n = bs * (S + 1)
    full = (torch.randn(n, VOCAB, device=DEV) * 3).to(torch.bfloat16).float()
    gathered = candidates_from_full(full)
    meta = spec_meta(bs)
    step = make_step(bs)
    # drafts: the greedy pick for a random prefix of each request, then a miss
    vals, ids = eager_candidates(gathered, n)
    top = torch.gather(ids, 1, vals.argmax(1, keepdim=True)).view(-1).to(torch.int32)
    inp = step.input_ids.clone()
    accepted = torch.randint(0, S + 1, (bs,), device=DEV)
    for r in range(bs):
        for j in range(S):
            inp[r * (S + 1) + j + 1] = top[r * (S + 1) + j] if j < accepted[r] else -7
    step.input_ids = inp
    temps = torch.full((bs,), 1e-12, device=DEV)
    # eager reference
    mask0 = step.state.mask.clone()
    rows = torch.cat([meta.target_logits_indices, meta.bonus_logits_indices]).long()
    drafts = torch.arange(rows.numel() - bs, device=DEV, dtype=torch.int32)
    per_row = torch.cat(
        [
            torch.searchsorted(meta.cu_num_draft_tokens, drafts, right=True),
            torch.arange(bs, device=DEV),
        ]
    ).long()
    tok = sample_topk_candidates(
        torch.index_select(vals, 0, rows),
        torch.index_select(ids, 0, rows),
        temps[per_row],
        TOPK,
        0.95,
    )
    t_tgt, t_bonus = tok[: rows.numel() - bs], tok[rows.numel() - bs :]
    ref_out, ref_ver = rejection_sample(
        inp,
        meta.target_logits_indices,
        S,
        meta.cu_num_draft_tokens,
        None,
        None,
        t_bonus.view(-1, 1).contiguous(),
        meta.bonus_logits_indices,
        target_token_ids=t_tgt,
    )
    ref_next = torch.gather(ref_out, 1, ref_ver[0].view(-1, 1).long()).view(-1)
    step._anchors_marked = False
    step.mark(ref_out)
    ref_mask = step.state.mask.clone()
    # fused
    step.state.mask.copy_(mask0)
    step._anchors_marked = False
    out, ver, nxt = StepSampler(DEV).sample(
        gathered, n, TP, K, TOPK, meta, inp, temps, 0.95, step, SAMPLER_EPS
    )
    # a request may differ only by an exact tie at a greedy pick
    diff = (out != ref_out).any(1).nonzero().view(-1).tolist()
    for r in diff:
        j = int((out[r] != ref_out[r]).nonzero()[0])
        row = r * (S + 1) + j
        v = vals[row]
        va = v[ids[row] == int(out[r, j])]
        vb = v[ids[row] == int(ref_out[r, j])]
        assert va.numel() and vb.numel(), f"request {r}: token outside the candidates"
        assert float(va[0]) == float(vb[0]) == float(v.max()), f"request {r}: not a tie"
    same = [r for r in range(bs) if r not in diff]
    assert torch.equal(ver[:, same], ref_ver[:, same])
    assert torch.equal(nxt[same], ref_next.to(torch.int32)[same])
    if not diff:
        assert torch.equal(step.state.mask, ref_mask)


def test_step_sampler_sampled_distribution():
    torch.manual_seed(0)
    bs, temp, top_p, steps = 64, 0.8, 0.95, 1500
    n = bs * (S + 1)
    base = (torch.randn(1, VOCAB, device=DEV) * 1.2).to(torch.bfloat16).float()
    gathered = candidates_from_full(base.expand(n, VOCAB).contiguous())
    meta = spec_meta(bs)
    inp = torch.full((n,), -7, dtype=torch.int32, device=DEV)  # every draft misses
    temps = torch.full((bs,), temp, device=DEV)
    sampler = StepSampler(DEV, seed=1234)
    counts = {}
    for _ in range(steps):
        nxt = sampler.sample(
            gathered, n, TP, K, TOPK, meta, inp, temps, top_p, None, SAMPLER_EPS
        )[2]
        for t in nxt.cpu().numpy().tolist():
            counts[t] = counts.get(t, 0) + 1
    tv, ti = torch.topk(base[0], TOPK)
    pr = torch.softmax(tv / temp, 0)
    excl = pr.cumsum(0) - pr
    pr = torch.where(excl >= top_p, torch.zeros_like(pr), pr)
    pr = (pr / pr.sum()).cpu().numpy()
    total = bs * steps
    chi, dof = 0.0, 0
    top_ids = ti.cpu().numpy().tolist()
    for tid, p in zip(top_ids, pr):
        if p <= 0:
            assert counts.get(tid, 0) == 0, f"token {tid} outside top-p was drawn"
            continue
        e = p * total
        chi += (counts.get(tid, 0) - e) ** 2 / e
        dof += 1
    assert sum(c for t, c in counts.items() if t not in set(top_ids)) == 0
    dof -= 1
    z = (chi - dof) / math.sqrt(2 * dof)
    assert abs(z) < 4, f"chi2 {chi:.1f} dof {dof} z {z:.2f}"


def test_step_sampler_draws_follow_seed_and_counter():
    torch.manual_seed(1)
    bs = 64
    n = bs * (S + 1)
    base = (torch.randn(1, VOCAB, device=DEV) * 1.2).to(torch.bfloat16).float()
    gathered = candidates_from_full(base.expand(n, VOCAB).contiguous())
    meta = spec_meta(bs)
    inp = torch.full((n,), -7, dtype=torch.int32, device=DEV)
    temps = torch.full((bs,), 0.8, device=DEV)
    a, b = StepSampler(DEV, seed=99), StepSampler(DEV, seed=99)
    args = (gathered, n, TP, K, TOPK, meta, inp, temps, 0.95, None, SAMPLER_EPS)
    first_a, first_b = a.sample(*args)[2], b.sample(*args)[2]
    second_a = a.sample(*args)[2]
    assert torch.equal(first_a, first_b), "same seed and counter, different draws"
    assert not torch.equal(first_a, second_a), "the step counter did not advance"
