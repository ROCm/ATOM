# SPDX-License-Identifier: MIT
"""The probability-ratio verify (`rejection_sample` with ``draft_probs``)
against its law: the emitted tokens follow the target's p whatever q the drafts
came from, and a request without q falls back to p.

Statistical checks use fixed seeds and tolerances several standard errors
wide, so they are deterministic on a given platform and fail only on a real
bias. The properties that hold for every coin (residual support, row-base
indirection, rank-consistent draws) are asserted exactly.
"""

import pytest
import torch

pytest.importorskip("aiter", reason="the samplers are AITER/Triton ops")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from atom.model_ops.rejection_sampler import (
    _RATIO_BLOCK_V,
    rejection_sample,
)

DEV = "cuda"
ANCHOR = 0


# ---- verify: the probability-ratio rule ---------------------------------------


def verify_batch(drafts, width, logits, temperatures):
    """`rejection_sample` arguments for equal-width requests whose row r
    scores ``logits[r]`` and drafts ``drafts[r]``."""
    rows = drafts.numel()
    batch = rows // width
    ids, logit_rows = [], []
    for b in range(batch):
        start = len(ids)
        ids.append(ANCHOR)
        ids.extend(drafts[b * width : (b + 1) * width].tolist())
        logit_rows.extend(range(start, start + width))
    return {
        "input_ids": torch.tensor(ids, device=DEV, dtype=torch.int32),
        "target_logits_indices": torch.tensor(logit_rows, device=DEV),
        "num_spec_steps": width,
        "cu_num_draft_tokens": torch.arange(
            width, rows + 1, width, device=DEV, dtype=torch.int32
        ),
        "target_probs": logits.contiguous(),
        "bonus_token_ids": torch.zeros(batch, device=DEV, dtype=torch.int32),
        "bonus_logits_indices": torch.arange(
            width, (width + 1) * batch, width + 1, device=DEV
        ),
        "temperatures": temperatures,
    }


def sparse(vocab, support):
    """A distribution over ``vocab`` ids with ``support = {id: mass}``."""
    p = torch.zeros(vocab, device=DEV)
    for i, m in support.items():
        p[i] = m
    return p


def draw(dist, n):
    return torch.multinomial(dist, n, replacement=True).to(torch.int32)


# Supports spread over three vocab tiles, the last one ragged.
VOCAB = 2 * _RATIO_BLOCK_V + 1000
P = {5: 0.5, _RATIO_BLOCK_V + 7: 0.3, VOCAB - 1: 0.2}
Q = {5: 0.2, _RATIO_BLOCK_V + 7: 0.1, VOCAB - 2: 0.7}


def test_the_emitted_token_follows_p_across_vocab_tiles():
    torch.manual_seed(91)
    n = 40000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(q, n)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        draft_probs=q.expand(n, -1).contiguous(),
        ratio_step=5,
    )
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)
    # Accepted at the rate sum min(p, q).
    rate = torch.minimum(p, q).sum().item()
    assert abs(accepted.float().mean().item() - rate) < 0.01


def test_a_residual_draw_never_lands_on_an_id_without_residual_mass():
    """Holds for every coin: a rejection's replacement has p > q."""
    torch.manual_seed(17)
    n = 20000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(q, n)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        draft_probs=q.expand(n, -1).contiguous(),
        ratio_step=6,
    )
    replaced = out[accepted == 0, 0].long()
    assert replaced.numel() > 1000
    assert ((p - q)[replaced] > 0).all()


def test_a_request_without_q_rejects_and_draws_from_p():
    torch.manual_seed(23)
    n = 30000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(p, n * 2)  # even drafts p itself would accept, given q
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 2, p.log().expand(n * 2, -1), torch.ones(n, device=DEV)),
        draft_probs=q.expand(8, -1).contiguous(),
        draft_probs_row_base=torch.full((n,), -1, dtype=torch.int32, device=DEV),
        ratio_step=7,
    )
    assert (accepted == 0).all()
    assert (out[:, 1:] == -1).all()
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)


def test_q_is_read_through_the_row_base_not_the_batch_position():
    """Scattered blocks give exactly the flat layout's answer: same coins
    (keyed by verify row and step), same q values."""
    torch.manual_seed(31)
    batch_n, width, vocab = 64, 3, 500
    logits = torch.randn(batch_n * width, vocab, device=DEV)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    temps = torch.full((batch_n,), 0.9, device=DEV)
    args = verify_batch(drafts, width, logits, temps)
    flat_out, flat_verdict = rejection_sample(**args, draft_probs=q_rows, ratio_step=9)
    slots = torch.randperm(2 * batch_n, device=DEV)[:batch_n]
    store = torch.zeros(2 * batch_n * width, vocab, device=DEV)
    base = (slots * width).to(torch.int32)
    for k in range(width):
        store[base.long() + k] = q_rows.view(batch_n, width, vocab)[:, k]
    out, verdict = rejection_sample(
        **args, draft_probs=store, draft_probs_row_base=base, ratio_step=9
    )
    assert torch.equal(out, flat_out) and torch.equal(verdict, flat_verdict)


def test_the_same_key_gives_the_same_verdict():
    """What keeps TP ranks in agreement without exchanging coins."""
    torch.manual_seed(37)
    batch_n, width, vocab = 128, 4, 777
    logits = torch.randn(batch_n * width, vocab, device=DEV)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    args = verify_batch(drafts, width, logits, torch.ones(batch_n, device=DEV))
    a = rejection_sample(**args, draft_probs=q_rows, ratio_step=40)
    b = rejection_sample(**args, draft_probs=q_rows, ratio_step=40)
    c = rejection_sample(**args, draft_probs=q_rows, ratio_step=41)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])
