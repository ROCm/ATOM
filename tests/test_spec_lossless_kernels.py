# SPDX-License-Identifier: MIT
"""The two kernels lossless speculative sampling adds, against their laws.

* `dspark_markov_sample`: each row draws ``x ~ q = softmax((base + bias) / T)``
  and writes q into the persistent store at its own row block, nowhere else.
* The probability-ratio verify (`rejection_sample` with ``draft_probs``): the
  emitted tokens follow the target's p whatever q the drafts came from; greedy
  rows are exact match; a request without q falls back to p.

Statistical checks use fixed seeds and tolerances several standard errors
wide, so they are deterministic on a given platform and fail only on a real
bias. The properties that hold for every coin (destination rows, residual
support, greedy outputs, rank-consistent draws) are asserted exactly.
"""

import pytest
import torch

# The CPU unit job has neither aiter nor triton; both imports stay below these
# guards, or the job fails to collect this file and aborts the whole run.
pytest.importorskip("aiter", reason="the samplers are AITER/Triton ops")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

import triton
import triton.language as tl

from atom.model_ops.dspark_markov_sample import (
    dspark_markov_argmax,
    dspark_markov_sample,
    dspark_markov_sample_reference,
    dspark_markov_sample_scratch,
)
from atom.model_ops.rejection_sampler import (
    _RATIO_BLOCK_V,
    _request_of_row,
    ratio_verify_scratch,
    rejection_sample,
)
from atom.model_ops.sampler import SAMPLER_EPS

DEV = "cuda"
ANCHOR = 0


def tables(rows, vocab, rank, seed=1201):
    torch.manual_seed(seed)
    kw = {"dtype": torch.bfloat16, "device": DEV}
    return (
        torch.randn(rows, vocab, **kw),
        torch.randint(0, vocab, (rows,), dtype=torch.int64, device=DEV),
        torch.randn(vocab, rank, **kw) * 0.1,
        torch.randn(vocab, rank, **kw) * 0.1,
    )


def draft_args(rows, temperature=1.0, greedy=False, step=0):
    return (
        torch.full((rows,), temperature, device=DEV),
        torch.full((rows,), int(greedy), dtype=torch.int32, device=DEV),
        torch.tensor([step], dtype=torch.int32, device=DEV),
    )


# ---- draft: dspark_markov_sample --------------------------------------------


def test_each_row_writes_its_q_at_its_own_block_and_nowhere_else():
    rows, vocab, rank, t = 6, 3000, 64, 4
    base, prev, w1, w2 = tables(rows, vocab, rank)
    store = torch.full((10 * t, vocab), -7.0, device=DEV)
    # Two rows without a destination, the others scattered and out of order.
    row_base = torch.tensor([8, -1, 0, 4, -1, 6], dtype=torch.int32, device=DEV) * t
    row_base = torch.where(row_base < 0, -1, row_base).to(torch.int32)
    position = 2
    temps, greedy, seed = draft_args(rows, temperature=0.8)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_sample(
        base, prev, w1, w2, out, store, row_base, position, temps, greedy, seed, 0
    )
    _, ref_q = dspark_markov_sample_reference(
        base, prev, w1, w2, temps, greedy, torch.ones(rows, vocab, device=DEV)
    )
    written = set()
    for r, b in enumerate(row_base.tolist()):
        if b >= 0:
            torch.testing.assert_close(
                store[b + position], ref_q[r], rtol=1e-3, atol=1e-6
            )
            written.add(b + position)
    untouched = [i for i in range(store.shape[0]) if i not in written]
    assert (store[untouched] == -7.0).all(), "a row was written outside its block"
    assert ((out >= 0) & (out < vocab)).all()


@pytest.mark.parametrize("rows", [1, 32])
def test_at_temperature_zero_it_drafts_what_the_argmax_kernel_drafts(rows):
    """The flag-on draft at T=0 is the flag-off draft, chained position by
    position, and the q it publishes is the delta on that id."""
    vocab, rank, steps = 2048, 128, 4
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows, temperature=SAMPLER_EPS, greedy=True)
    store = torch.zeros(rows * steps, vocab, device=DEV)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV) * steps
    drawn, ref = prev, prev
    for k in range(steps):
        ref_ids = torch.empty(rows, dtype=torch.int64, device=DEV)
        dspark_markov_argmax(base, ref, w1, w2, ref_ids)
        ids = torch.empty(rows, dtype=torch.int64, device=DEV)
        dspark_markov_sample(
            base, drawn, w1, w2, ids, store, row_base, k, temps, greedy, seed, 0
        )
        assert torch.equal(ids, ref_ids), k
        q = store[row_base.long() + k]
        expected = torch.zeros_like(q).scatter_(1, ids.view(-1, 1), 1.0)
        torch.testing.assert_close(q, expected, rtol=0, atol=0)
        drawn, ref = ids, ref_ids


def test_a_greedy_row_on_a_tie_publishes_the_delta_its_id_came_from():
    """softmax(s / EPS) on an exact tie is 1/2 per tied id, but the id is the
    lowest of them, not a draw; the accept rule must see the delta."""
    rows, vocab, rank, low, high = 2, 3000, 32, 40, 2900
    base = torch.randn(rows, vocab, device=DEV).to(torch.bfloat16)
    base[:, low] = 20.0
    base[:, high] = 20.0
    prev = torch.zeros(rows, dtype=torch.int64, device=DEV)
    w = torch.zeros(vocab, rank, dtype=torch.bfloat16, device=DEV)
    temps, greedy, seed = draft_args(rows, temperature=SAMPLER_EPS, greedy=True)
    store = torch.full((rows, vocab), -1.0, device=DEV)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    dspark_markov_sample(
        base, prev, w, w, out, store, row_base, 0, temps, greedy, seed, 0
    )
    assert out.tolist() == [low, low]
    expected = torch.zeros(rows, vocab, device=DEV)
    expected[:, low] = 1.0
    torch.testing.assert_close(store, expected, rtol=0, atol=0)


def test_a_greedy_row_without_a_destination_still_drafts_the_argmax():
    """Greedy requests claim no q slot; their draft must be unchanged."""
    rows, vocab, rank = 16, 2048, 64
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows, temperature=SAMPLER_EPS, greedy=True)
    store = torch.full((4, vocab), -7.0, device=DEV)
    row_base = torch.full((rows,), -1, dtype=torch.int32, device=DEV)
    ref = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_argmax(base, prev, w1, w2, ref)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_sample(
        base, prev, w1, w2, out, store, row_base, 0, temps, greedy, seed, 0
    )
    assert torch.equal(out, ref)
    assert (store == -7.0).all()


def test_a_sampled_row_without_a_destination_drafts_the_argmax():
    """Pad, dummy and align-only rows write no q; they draft the argmax
    rather than id 0, so a misclassified step would still propose sensibly."""
    rows, vocab, rank = 16, 2048, 64
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows, temperature=1.0, step=3)
    store = torch.full((4, vocab), -7.0, device=DEV)
    row_base = torch.full((rows,), -1, dtype=torch.int32, device=DEV)
    ref = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_argmax(base, prev, w1, w2, ref)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_sample(
        base, prev, w1, w2, out, store, row_base, 0, temps, greedy, seed, 0
    )
    assert torch.equal(out, ref)
    assert (store == -7.0).all()


def test_the_preallocated_scratch_drafts_exactly_what_allocation_does():
    rows, vocab, rank = 24, 3000, 64
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows, temperature=0.9, step=5)
    greedy[::3] = 1
    scratch = dspark_markov_sample_scratch(64, vocab, DEV)  # sized above rows
    results = []
    for f32, i32 in ((None, None), (scratch.f32, scratch.i32)):
        store = torch.zeros(rows, vocab, device=DEV)
        out = torch.empty(rows, dtype=torch.int64, device=DEV)
        row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
        dspark_markov_sample(
            base,
            prev,
            w1,
            w2,
            out,
            store,
            row_base,
            0,
            temps,
            greedy,
            seed,
            0,
            f32,
            i32,
        )
        results.append((out, store))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_sampled_ids_follow_the_q_they_publish():
    rows, vocab, rank = 8192, 48, 16
    base, prev, w1, w2 = tables(1, vocab, rank, seed=5)
    base, prev = base.expand(rows, -1).contiguous(), prev.expand(rows).contiguous()
    temps, greedy, seed = draft_args(rows, temperature=1.3, step=11)
    store = torch.empty(rows, vocab, device=DEV)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_sample(
        base, prev, w1, w2, out, store, row_base, 0, temps, greedy, seed, 0
    )
    q = store[0]
    assert torch.allclose(store, q.expand_as(store))
    freq = torch.bincount(out, minlength=vocab).float() / rows
    # ~4 standard errors of the largest cell.
    torch.testing.assert_close(freq, q, rtol=0, atol=0.025)


def test_the_draws_depend_on_the_key_alone():
    """TP ranks run this independently and must draft the same ids."""
    rows, vocab, rank = 64, 1000, 32
    base, prev, w1, w2 = tables(rows, vocab, rank)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    results = []
    for step in (3, 3, 4):
        temps, greedy, seed = draft_args(rows, step=step)
        out = torch.empty(rows, dtype=torch.int64, device=DEV)
        store = torch.empty(rows, vocab, device=DEV)
        dspark_markov_sample(
            base, prev, w1, w2, out, store, row_base, 0, temps, greedy, seed, 0
        )
        results.append(out)
    assert torch.equal(results[0], results[1])
    assert not torch.equal(results[0], results[2])


def test_rows_past_two_gib_of_elements_are_addressed_correctly():
    """A `[2 * 512 * 16, 163840]` store passes 2**31 elements; so does the base
    logits' row stride at large batch. Both must be int64 in the kernel."""
    vocab, rank = 163840, 64
    big = 2**31 // vocab + 4  # first row past 2**31 elements, plus margin
    store = torch.zeros(big + 1, vocab, device=DEV)
    torch.manual_seed(3)
    stride = 2**31 + 64
    flat = torch.randn(stride + vocab, dtype=torch.bfloat16, device=DEV)
    base = flat.as_strided((2, vocab), (stride, 1))
    prev = torch.tensor([1, 2], dtype=torch.int64, device=DEV)
    w1 = torch.randn(vocab, rank, dtype=torch.bfloat16, device=DEV) * 0.1
    w2 = torch.randn(vocab, rank, dtype=torch.bfloat16, device=DEV) * 0.1
    temps, greedy, seed = draft_args(2)
    row_base = torch.tensor([big, 0], dtype=torch.int32, device=DEV)
    out = torch.empty(2, dtype=torch.int64, device=DEV)
    dspark_markov_sample(
        base, prev, w1, w2, out, store, row_base, 0, temps, greedy, seed, 0
    )
    _, ref_q = dspark_markov_sample_reference(
        base.contiguous(), prev, w1, w2, temps, greedy, torch.ones(2, vocab, device=DEV)
    )
    torch.testing.assert_close(store[big], ref_q[0], rtol=1e-3, atol=1e-7)
    torch.testing.assert_close(store[0], ref_q[1], rtol=1e-3, atol=1e-7)


def test_unsupported_weights_raise_rather_than_drafting_off_the_global_rng():
    rows, vocab, rank = 4, 64, 16
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows)
    with pytest.raises(ValueError, match="cannot run"):
        dspark_markov_sample(
            base,
            prev,
            w1.float(),
            w2.float(),
            torch.empty(rows, dtype=torch.int64, device=DEV),
            torch.empty(rows, vocab, device=DEV),
            torch.arange(rows, dtype=torch.int32, device=DEV),
            0,
            temps,
            greedy,
            seed,
            0,
        )


# ---- verify: the probability-ratio rule ---------------------------------------


def verify_batch(drafts, width, logits, temperatures, row_base=None):
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
        "target_logits": logits.contiguous(),
        "bonus_token_ids": torch.zeros(batch, device=DEV, dtype=torch.int32),
        "bonus_logits_indices": torch.arange(
            width, (width + 1) * batch, width + 1, device=DEV
        ),
        "temperatures": temperatures,
        # Flat ragged q unless told otherwise: request b's rows start at b * width.
        "draft_probs_row_base": (
            torch.arange(batch, device=DEV, dtype=torch.int32) * width
            if row_base is None
            else row_base
        ),
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


@triton.jit
def _rows_to_requests(cu_ptr, out_ptr, batch_size, steps):
    row = tl.program_id(0)
    tl.store(out_ptr + row, _request_of_row(cu_ptr, row, batch_size, steps))


@pytest.mark.parametrize(
    "widths", [[3], [2, 0, 3, 1, 0, 4], [1] * 37, [0, 5, 0], [7, 7, 7, 7, 7]]
)
def test_each_draft_row_finds_its_request(widths):
    """The stats kernel's row -> request lookup, against searchsorted."""
    cu = torch.tensor(widths, device=DEV).cumsum(0).to(torch.int32)
    rows = int(cu[-1])
    out = torch.empty(rows, dtype=torch.int32, device=DEV)
    _rows_to_requests[(rows,)](cu, out, len(widths), len(widths).bit_length() + 1)
    expected = torch.searchsorted(cu, torch.arange(rows, device=DEV), right=True)
    assert torch.equal(out.long(), expected)


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


def test_a_residual_without_mass_falls_back_to_p():
    """p == q, and the draft is an id with no mass in either: it is rejected
    (q(x) = 0 is not the row x came from), relu(p - q) is 0 everywhere, and
    the replacement must still follow p rather than land on a fixed id."""
    torch.manual_seed(29)
    n = 30000
    p = sparse(VOCAB, P)
    drafts = torch.full((n,), 7, dtype=torch.int32, device=DEV)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        draft_probs=p.expand(n, -1).contiguous(),
        ratio_step=8,
    )
    assert (accepted == 0).all()
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)


def test_a_request_without_q_rejects_and_draws_from_p():
    torch.manual_seed(23)
    n = 30000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(p, n * 2)  # even drafts p itself would accept, given q
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(
            drafts,
            2,
            p.log().expand(n * 2, -1),
            torch.ones(n, device=DEV),
            row_base=torch.full((n,), -1, dtype=torch.int32, device=DEV),
        ),
        draft_probs=q.expand(8, -1).contiguous(),
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
    args["draft_probs_row_base"] = base
    out, verdict = rejection_sample(**args, draft_probs=store, ratio_step=9)
    assert torch.equal(out, flat_out) and torch.equal(verdict, flat_verdict)


@triton.jit
def _uniforms(out_ptr, seed, base, WIDE: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, N)
    if WIDE:
        u = tl.rand(seed, base + offs.to(tl.int64))
    else:
        u = tl.rand(seed, offs)
    tl.store(out_ptr + offs, u)


@pytest.mark.parametrize("wide", [False, True], ids=["int32-offset", "int64-offset"])
def test_the_key_purpose_word_separates_streams_at_any_offset_width(wide):
    """Draft noise, accept coins and residual race differ only in the key's
    high word (0, 1, 2); that word must reach Philox whether the offset is
    32-bit (the coins) or 64-bit (the draft noise and the residual race)."""
    from atom.model_ops.rejection_sampler import (
        _RATIO_ACCEPT_KEY,
        _RATIO_RESIDUAL_KEY,
    )

    step, n = 12345, 1024
    base = 5 * 2**31 if wide else 0
    streams = []
    for purpose in (0, _RATIO_ACCEPT_KEY, _RATIO_RESIDUAL_KEY):
        out = torch.empty(n, device=DEV)
        _uniforms[(1,)](out, purpose | step, base, WIDE=wide, N=n)
        streams.append(out)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(streams[i], streams[j]), (i, j)


def test_the_preallocated_scratch_verifies_exactly_what_allocation_does():
    torch.manual_seed(43)
    batch_n, width, vocab = 48, 5, 3000
    logits = torch.randn(batch_n * width, vocab, device=DEV)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    temps = torch.ones(batch_n, device=DEV)
    temps[::4] = SAMPLER_EPS
    args = verify_batch(drafts, width, logits, temps)
    plain = rejection_sample(**args, draft_probs=q_rows, ratio_step=11)
    scratch = ratio_verify_scratch(64, width, vocab, DEV)  # sized above the batch
    reused = rejection_sample(
        **args, draft_probs=q_rows, ratio_step=11, ratio_scratch=scratch
    )
    assert torch.equal(plain[0], reused[0]) and torch.equal(plain[1], reused[1])


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


def tied_logits(rows, vocab, low, high):
    """bf16 logits whose max is an exact tie between ids ``low < high``."""
    logits = torch.randn(rows, vocab, device=DEV).to(torch.bfloat16)
    logits[:, low] = 30.0
    logits[:, high] = 30.0
    return logits


@pytest.mark.parametrize("draft_first", ["tie-low", "tie-high", "other"])
def test_a_greedy_row_is_exact_match_whatever_its_batch_mates(draft_first):
    """On a bf16 tie the greedy row's p is 0.5/0.5, not a delta, so the ratio
    rule would accept the high tie half the time. Exact match never does, and
    must not care who else is in the batch."""
    torch.manual_seed(41)
    width, vocab, low, high = 3, 1024, 3, 700
    first = {"tie-low": low, "tie-high": high, "other": 11}[draft_first]
    greedy_drafts = torch.tensor([first, low, low], dtype=torch.int32, device=DEV)
    greedy_logits = tied_logits(width, vocab, low, high)

    # Reference: the flag-off kernel on the greedy request alone.
    ref_out, ref_verdict = rejection_sample(
        **verify_batch(
            greedy_drafts,
            width,
            greedy_logits,
            torch.full((1,), SAMPLER_EPS, device=DEV),
        ),
        draft_probs=None,
    )
    expected_accept = {"tie-low": 3, "tie-high": 0, "other": 0}[draft_first]
    assert ref_verdict[0, 0].item() == expected_accept
    if expected_accept == 0:
        assert ref_out[0, 0].item() == low

    # The same request among sampled ones, at several positions and keys.
    sampled_n = 7
    for slot in (0, 3, sampled_n):
        for step in range(5):
            others = torch.randint(0, vocab, (sampled_n * width,), device=DEV)
            drafts = torch.cat(
                (others[: slot * width], greedy_drafts, others[slot * width :])
            ).to(torch.int32)
            logits = torch.cat(
                (
                    torch.randn(slot * width, vocab, device=DEV).to(torch.bfloat16),
                    greedy_logits,
                    torch.randn((sampled_n - slot) * width, vocab, device=DEV).to(
                        torch.bfloat16
                    ),
                )
            )
            temps = torch.ones(sampled_n + 1, device=DEV)
            temps[slot] = SAMPLER_EPS
            # q = p for the greedy row too, the case the ratio rule gets wrong.
            q = torch.softmax(logits.float(), -1)
            out, verdict = rejection_sample(
                **verify_batch(drafts, width, logits, temps),
                draft_probs=q,
                ratio_step=100 + step,
            )
            assert torch.equal(out[slot, :width], ref_out[0, :width]), (slot, step)
            assert torch.equal(verdict[:2, slot], ref_verdict[:2, 0]), (slot, step)


def test_the_target_law_holds_at_the_real_vocab():
    torch.manual_seed(53)
    vocab, n = 163840, 6000
    support_p = {7: 0.4, 81920: 0.35, vocab - 1: 0.25}
    support_q = {7: 0.6, 81920: 0.1, vocab - 3: 0.3}
    p, q = sparse(vocab, support_p), sparse(vocab, support_q)
    drafts = draw(q, n)
    logits = p.log().to(torch.bfloat16).expand(n, -1)
    out, _ = rejection_sample(
        **verify_batch(
            drafts,
            1,
            logits,
            torch.ones(n, device=DEV),
            row_base=torch.zeros(n, dtype=torch.int32, device=DEV),
        ),
        draft_probs=q.expand(1, -1).contiguous(),
        ratio_step=3,
    )
    p_bf16 = torch.softmax(p.log().to(torch.bfloat16).float(), -1)
    freq = torch.bincount(out[:, 0].long(), minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_bf16, rtol=0, atol=0.025)
