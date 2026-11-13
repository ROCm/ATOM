# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_REJECTION_SAMPLING: `SpecRejectionSampling` against the official sampler.

Many identical requests are drafted and verified in one launch, so empirical
distributions can be compared:

* a draft pass's draws follow the distribution its record describes;
* the first token a request emits follows the official sampler's own draws of
  that verify row, however far the draft distribution q is from p (lossless);
* the acceptance rate is sum(min(p, q)) (the rule's acceptance);
* without a record the rule falls back to the previous one (acceptance p(d));
* greedy rows take the argmax everywhere.

Rows are model-like (a few strong columns over a flat background) so that the
sampler's top-p cut binds, and Gaussian (where it keeps all top-k columns)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from types import SimpleNamespace

from atom.model_ops.sampler import fused_topk_topp_sample
from atom.model_ops.spec_rejection import SpecRejectionSampling, _kb

DEV = "cuda"
V, K, S = 32768, 20, 3
N, REPS = 2048, 4  # requests a launch, launches
F32 = {"dtype": torch.float32, "device": DEV}


@pytest.fixture(autouse=True)
def _bf16_default():
    # As in the runner, whose default dtype is the model's.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(old)


def _dist(rec):
    """[rows, 3KB+2] sampler records -> dense [rows, V] distributions (the select
    kernel's rule: keep while the mass strictly above is below its threshold)."""
    kb = _kb(K)
    cols = rec[:, :K].long()
    p = rec[:, kb : kb + K].view(torch.float32)
    thr = rec[:, 3 * kb].view(torch.float32)[:, None]
    gt = (p[:, None, :] * (p[:, None, :] > p[:, :, None])).sum(-1)
    w = torch.where(gt < thr, p, 0.0)
    w = w / w.sum(-1, keepdim=True)
    return torch.zeros(rec.shape[0], V, **F32).scatter_add_(1, cols, w)


def _freq(tokens):
    return torch.bincount(tokens.long(), minlength=V).float() / tokens.numel()


def _tv(a, b):
    return 0.5 * (a - b).abs().sum().item()


def _row(gen, peaked):
    x = torch.randn(V, generator=gen, **F32)
    if peaked:
        hot = torch.randperm(V, generator=gen, device=DEV)[:8]
        x[hot] += 15.0 * torch.linspace(1.0, 0.55, 8, **F32)
    else:
        x = x * 2.0
    return x


def _setup(seed, peaked):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rs = SpecRejectionSampling(
        max_rows=N, mtp_k=S, max_num_seqs=(REPS + 2) * N, device=DEV, draft_k=K
    )
    rs.warm(V)
    pl = [_row(gen, peaked) for _ in range(S + 1)]
    ql = [
        (pl[i] + 0.6 * torch.randn(V, generator=gen, **F32)).to(torch.bfloat16)
        for i in range(S)
    ]
    b = torch.arange(N, device=DEV)
    meta = SimpleNamespace(
        cu_num_draft_tokens=((b + 1) * S).to(torch.int32),
        target_logits_indices=(
            b[:, None] * (S + 1) + torch.arange(S, device=DEV)
        ).reshape(-1),
        bonus_logits_indices=b * (S + 1) + S,
        num_spec_steps=S,
    )
    logits = torch.stack([pl[i % (S + 1)] for i in range(N * (S + 1))]).to(
        torch.bfloat16
    )
    return rs, ql, meta, logits


def _draft(rs, ql, rep, temps, top_p):
    rs.stage_step(list(range(rep * N, (rep + 1) * N)), temps, top_p)
    ids = torch.zeros(N, S + 1, dtype=torch.int64, device=DEV)
    out = torch.empty(N, dtype=torch.int32, device=DEV)
    q0 = None
    for i in range(S):
        rs.sample_draft(ql[i].expand(N, V).contiguous(), out)
        ids[:, 1 + i] = out
        rs.store_step(i, N)
        if i == 0:
            q0 = _dist(rs.dbg_scratch[:1])[0]
    return ids, q0


def test_draft_draws_follow_their_record():
    rs, ql, _, _ = _setup(1, peaked=False)
    rs.stage_step(list(range(N)), torch.ones(N, **F32), 0.95)
    out = torch.empty(N, dtype=torch.int32, device=DEV)
    rows = ql[0].expand(N, V).contiguous()
    draws = []
    for _ in range(REPS):
        rs.sample_draft(rows, out)
        draws.append(out.clone())
        assert torch.equal(rs.tok_scratch[:N], out)
    assert _tv(_freq(torch.cat(draws)), _dist(rs.dbg_scratch[:1])[0]) < 0.04


@pytest.mark.parametrize("peaked", [True, False])
def test_verify_emits_the_official_distribution(peaked):
    rs, ql, meta, logits = _setup(2 + peaked, peaked)
    temps = torch.ones(N, **F32)
    kb = _kb(K)
    dbg = torch.empty(N * (S + 1), 3 * kb + 2, dtype=torch.int32, device=DEV)
    official = []
    for _ in range(REPS):
        tok = fused_topk_topp_sample(
            logits,
            temps,
            K,
            0.95,
            verify_rows=meta.target_logits_indices,
            bonus_rows=meta.bonus_logits_indices,
            cu_num_draft_tokens=meta.cu_num_draft_tokens,
            debug=dbg,
        )
        official.append(tok[: N * S].view(N, S)[:, 0].clone())
    p0 = _dist(dbg[:1])[0]
    first, accepted, bonus = [], [], []
    for rep in range(REPS):
        ids, q0 = _draft(rs, ql, rep, temps, 0.95)
        meta.input_ids = ids.reshape(-1)
        out_v, out_b = rs.verify(logits, meta, temps, K, 0.95)
        t0 = out_v.view(N, S)[:, 0]
        first.append(t0.clone())
        accepted.append(t0 == ids[:, 1].int())
        bonus.append(out_b.clone())
    assert _tv(_freq(torch.cat(first)), _freq(torch.cat(official))) < 0.05
    acc = torch.cat(accepted).float().mean().item()
    assert abs(acc - torch.minimum(p0, q0).sum().item()) < 0.02
    assert _tv(_freq(torch.cat(bonus)), _dist(dbg[N * S : N * S + 1])[0]) < 0.05


def test_no_record_falls_back_to_the_previous_rule():
    rs, ql, meta, logits = _setup(4, peaked=True)
    temps = torch.ones(N, **F32)
    ids, _ = _draft(rs, ql, 0, temps, 0.95)
    meta.input_ids = ids.reshape(-1)
    rs.record_tok.fill_(-1)  # no draft has a record
    kb = _kb(K)
    dbg = torch.empty(N * (S + 1), 3 * kb + 2, dtype=torch.int32, device=DEV)
    fused_topk_topp_sample(
        logits,
        temps,
        K,
        0.95,
        verify_rows=meta.target_logits_indices,
        bonus_rows=meta.bonus_logits_indices,
        cu_num_draft_tokens=meta.cu_num_draft_tokens,
        debug=dbg,
    )
    p0 = _dist(dbg[:1])[0]
    out_v, _ = rs.verify(logits, meta, temps, K, 0.95)
    acc = (out_v.view(N, S)[:, 0] == ids[:, 1].int()).float().mean().item()
    assert abs(acc - p0[ids[:, 1]].mean().item()) < 0.03


def test_greedy_rows_take_the_argmax():
    rs, ql, meta, logits = _setup(5, peaked=False)
    temps = torch.full((N,), 1e-10, **F32)
    ids, _ = _draft(rs, ql, 0, temps, 0.95)
    assert torch.all(ids[:, 1] == int(ql[0].float().argmax()))
    meta.input_ids = ids.reshape(-1)
    out_v, out_b = rs.verify(logits, meta, temps, K, 0.95)
    rows = logits.float()
    assert torch.all(out_v.view(N, S)[:, 0] == int(rows[0].argmax()))
    assert torch.all(out_b == int(rows[S].argmax()))
