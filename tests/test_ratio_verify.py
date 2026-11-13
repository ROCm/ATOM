# SPDX-License-Identifier: MIT
"""The probability-ratio verify (`rejection_sample` with `RatioVerifyInputs`):
the emitted tokens follow the target's p whatever q the drafts came from,
read from the sparse store exactly as a dense q would be read; greedy rows
are exact match; a request without q falls back to p; a request with top-k
/ top-p is served `Sampler`'s filtered p.

Statistical checks use fixed seeds and tolerances several standard errors
wide, so they are deterministic on a given platform and fail only on a real
bias. The properties that hold for every coin are asserted exactly.
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
from spec_lossless_reference import (
    DEV,
    VOCAB,
    P,
    Q,
    draft_args,
    draft_q,
    draw,
    empty_store,
    filtered_p,
    first_token_law,
    philox,
    ratio_verify_reference,
    sample,
    sampler_kept,
    sparse,
    tables,
    tied_logits,
    top_of,
    triton_rand,
    verify,
    verify_batch,
)

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.ratio_target_p import (
    _RATIO_BLOCK_V,
    _ratio_stats_kernel,
)
from atom.model_ops.spec_sampling.ratio_verify import (
    _RATIO_ACCEPT_KEY,
    _RATIO_RESIDUAL_KEY,
    _ratio_accept_kernel,
    _ratio_residual_pick_kernel,
    _ratio_residual_race_kernel,
    ratio_verify_scratch,
    warmup_ratio_verify,
)
from atom.model_ops.spec_sampling.sampling_filter_count import (
    _ratio_filter_count_kernel,
)
from atom.model_ops.spec_sampling.sampling_filter_descent import (
    _ratio_filter_cut_kernel,
    _ratio_filter_floor_kernel,
    _ratio_filter_pass_kernel,
)
from atom.model_ops.spec_sampling.softmax_tiles import softmax_combine_kernel
from atom.model_ops.spec_sampling.sparse_q_store import SparseQStore


@pytest.mark.parametrize(
    "kernel",
    [
        _ratio_stats_kernel,
        _ratio_filter_pass_kernel,
        _ratio_filter_floor_kernel,
        _ratio_filter_cut_kernel,
        _ratio_filter_count_kernel,
        _ratio_residual_race_kernel,
        _ratio_residual_pick_kernel,
        softmax_combine_kernel,
    ],
    ids=lambda k: k.fn.__name__,
)
def test_no_verify_kernel_specializes_on_the_batch(kernel):
    """The verify launches eagerly at whatever batch arrives; a row or request
    count Triton specialized on (== 1, % 16) would compile a new variant the
    first time such a batch arrived, mid-serve."""
    counts = {"num_tokens", "batch_size", "num_rows"}
    params = [p for p in kernel.params if p.name in counts]
    assert params
    assert all(p.do_not_specialize for p in params), [p.name for p in params]


def test_the_emitted_token_follows_p_across_vocab_tiles():
    """q != p, the drafts drawn from q: the first token follows p, and so does
    the second among the requests that reached it, whether it is the accepted
    draft or the residual's -- a chain whose later positions leaned on the
    accept before them would not."""
    torch.manual_seed(91)
    n = 40000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(q, 2 * n)
    out, (accepted, _, _) = verify(
        **verify_batch(drafts, 2, p.log().expand(2 * n, -1), torch.ones(n, device=DEV)),
        q=draft_q(q, drafts),
        step=5,
    )
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)
    reached = out[accepted >= 1, 1].long()
    assert (reached >= 0).all()
    freq = torch.bincount(reached, minlength=VOCAB).float() / reached.numel()
    torch.testing.assert_close(freq, p, rtol=0, atol=0.015)
    # Accepted at the rate sum min(p, q).
    rate = torch.minimum(p, q).sum().item()
    assert abs((accepted >= 1).float().mean().item() - rate) < 0.01


def test_a_residual_draw_never_lands_on_an_id_without_residual_mass():
    """Holds for every coin: a rejection's replacement has p > q."""
    torch.manual_seed(17)
    n = 20000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(q, n)
    out, (accepted, _, _) = verify(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        q=draft_q(q, drafts),
        step=6,
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
    out, (accepted, _, _) = verify(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        q=draft_q(p, drafts),
        step=8,
    )
    assert (accepted == 0).all()
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)


def test_a_row_without_a_finite_z_emits_its_argmax_not_id_0():
    """One +inf logit, or one NaN logit, makes Z NaN and every p 0: the
    draft is rejected, the residual and the fallback race have no positive
    key, and the emitted id must be the row's raw argmax (the +inf id, or the
    finite max beside the NaN), never id 0, which carries no mass."""
    n = 64
    hot = _RATIO_BLOCK_V + 7
    torch.manual_seed(37)
    logits = torch.randn(2, VOCAB, device=DEV)
    logits[:, 0] = -30.0
    logits[0, hot] = float("inf")
    logits[1, hot] = 10.0
    logits[1, 20] = float("nan")
    q = sparse(VOCAB, {3: 0.5, 11: 0.5})
    drafts = draw(q, 2 * n)
    out, (accepted, _, _) = verify(
        **verify_batch(drafts, 1, logits.repeat(n, 1), torch.ones(2 * n, device=DEV)),
        q=draft_q(q, drafts),
        step=9,
    )
    assert (accepted == 0).all()
    assert (out[:, 0] == hot).all()


def test_a_request_without_q_rejects_and_draws_from_p():
    torch.manual_seed(23)
    n = 30000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(p, n * 2)  # even drafts p itself would accept, given q
    out, (accepted, _, _) = verify(
        **verify_batch(
            drafts,
            2,
            p.log().expand(n * 2, -1),
            torch.ones(n, device=DEV),
            row_base=torch.full((n,), -1, dtype=torch.int32, device=DEV),
        ),
        q=draft_q(q),
        step=7,
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
    flat = draft_q(q_rows, drafts)
    flat_out, flat_verdict = verify(**args, q=flat, step=9)
    slots = torch.randperm(2 * batch_n, device=DEV)[:batch_n]
    store = empty_store(2 * batch_n * width, flat.ids.shape[1])
    base = (slots * width).to(torch.int32)
    for k in range(width):
        for dst, src in zip(store, flat):
            dst[base.long() + k] = src.view(batch_n, width, -1)[:, k]
    args["q_row_base"] = base
    out, verdict = verify(**args, q=store, step=9)
    assert torch.equal(out, flat_out) and torch.equal(verdict, flat_verdict)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("filters", ["none", "per-row", "scalar"])
def test_a_reused_scratch_verifies_exactly_what_a_fresh_one_does(dtype, filters):
    """Production reuses one scratch, holding whatever the last step left in
    it; no buffer of it, the filter's included, may be read before written."""
    torch.manual_seed(43)
    batch_n, width, vocab = 48, 5, 3000
    logits = torch.randn(batch_n * width, vocab, device=DEV).to(dtype)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    temps = torch.ones(batch_n, device=DEV)
    temps[::4] = SAMPLER_EPS
    top_ks = torch.randint(1, 300, (batch_n,), device=DEV, dtype=torch.int32)
    top_ks[1::5] = -1
    top_ps = torch.rand(batch_n, device=DEV) * 0.9 + 0.05
    top_ps[2::5] = 1.0
    ks, ps = {
        "none": (None, None),
        "per-row": (top_ks, top_ps),
        "scalar": (int(top_ks[0]), float(top_ps[0])),
    }[filters]
    args = verify_batch(drafts, width, logits, temps)
    q = draft_q(q_rows, drafts)
    plain = verify(**args, q=q, step=11, top_ks=ks, top_ps=ps)
    scratch = ratio_verify_scratch(64, width, vocab, DEV)  # sized above the batch
    for garbage in (float("nan"), -1.0):
        scratch.view(torch.float32).fill_(garbage)
        scratch[::7].random_(0, 256)
        reused = verify(
            **{**args, "ratio_scratch": scratch},
            q=q,
            step=11,
            top_ks=ks,
            top_ps=ps,
        )
        assert torch.equal(plain[0], reused[0]) and torch.equal(plain[1], reused[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_startup_warmup_compiles_every_variant_a_serving_verify_launches(
    dtype,
):
    """No warmup or capture forward verifies, so `warmup_ratio_verify` is what
    keeps the first sampled steps from compiling mid-serve: after it, a
    serving-shaped verify in each filter form `prepare_sample` hands over, at
    another batch, store contents and row bases, compiles nothing."""
    kernels = (
        _ratio_stats_kernel,
        softmax_combine_kernel,
        _ratio_filter_count_kernel,
        _ratio_filter_pass_kernel,
        _ratio_filter_floor_kernel,
        _ratio_filter_cut_kernel,
        _ratio_accept_kernel,
        _ratio_residual_race_kernel,
        _ratio_residual_pick_kernel,
    )
    max_bs, batch_n, width, q_width, vocab = 16, 5, 4, 8, 3000
    scratch = ratio_verify_scratch(max_bs, width, vocab, DEV)
    caches = [k.device_caches[torch.cuda.current_device()][0] for k in kernels]
    saved = [dict(cache) for cache in caches]
    for cache in caches:
        cache.clear()
    try:
        warmup_ratio_verify(width, vocab, q_width, dtype, scratch)
        torch.cuda.synchronize()
        warmed = [len(cache) for cache in caches]
        assert all(warmed), [k.fn.__name__ for k, n in zip(kernels, warmed) if not n]

        torch.manual_seed(53)
        rows = batch_n * width
        q_rows = torch.softmax(torch.randn(rows, vocab, device=DEV), -1)
        drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
        # Sliced out of the per-row buffers, as `prepare_sample` hands them.
        top_ks = torch.randint(1, 300, (max_bs,), device=DEV, dtype=torch.int32)
        top_ps = torch.rand(max_bs, device=DEV) * 0.9 + 0.05
        # Every kind of top-k beside every kind of top-p.
        forms = [
            (k, p)
            for k in (None, 20, 300, top_ks[:batch_n])
            for p in (None, 0.8, 1.0, top_ps[:batch_n])
        ]
        store = SparseQStore.allocate(max_bs * width, q_width, DEV)
        for t in store:
            t.zero_()
        for top_ks_form, top_ps_form in forms:
            # A step's target rows are gathered out of its whole logits.
            logits = torch.randn(rows + batch_n, vocab, device=DEV).to(dtype)
            args = verify_batch(
                drafts,
                width,
                logits[:rows],
                torch.ones(max_bs, device=DEV)[:batch_n],
                row_base=torch.arange(batch_n, device=DEV, dtype=torch.int32) * width,
                scratch=scratch,
            )
            verify(
                **args,
                q=store,
                step=3,
                top_ks=top_ks_form,
                top_ps=top_ps_form,
            )
        torch.cuda.synchronize()
        grew = [
            k.fn.__name__
            for k, cache, n in zip(kernels, caches, warmed)
            if len(cache) != n
        ]
        assert not grew, f"compiled while serving: {grew}"
    finally:
        for cache, entries in zip(caches, saved):
            cache.clear()
            cache.update(entries)


def test_the_same_key_gives_the_same_verdict():
    """What keeps TP ranks in agreement without exchanging coins."""
    torch.manual_seed(37)
    batch_n, width, vocab = 128, 4, 777
    logits = torch.randn(batch_n * width, vocab, device=DEV)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    args = verify_batch(drafts, width, logits, torch.ones(batch_n, device=DEV))
    q = draft_q(q_rows, drafts)
    a = verify(**args, q=q, step=40)
    b = verify(**args, q=q, step=40)
    c = verify(**args, q=q, step=41)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


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
    ref_out, ref_verdict = verify(
        **verify_batch(
            greedy_drafts,
            width,
            greedy_logits,
            torch.full((1,), SAMPLER_EPS, device=DEV),
        ),
        q=None,
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
            out, verdict = verify(
                **verify_batch(drafts, width, logits, temps),
                q=draft_q(q, drafts),
                step=100 + step,
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
    out, _ = verify(
        **verify_batch(
            drafts,
            1,
            logits,
            torch.ones(n, device=DEV),
            row_base=torch.zeros(n, dtype=torch.int32, device=DEV),
        ),
        q=draft_q(q),
        step=3,
    )
    p_bf16 = torch.softmax(p.log().to(torch.bfloat16).float(), -1)
    freq = torch.bincount(out[:, 0].long(), minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_bf16, rtol=0, atol=0.025)


def test_the_reference_philox_reproduces_random123():
    """`philox` against Random123's known-answer vectors for Philox4x32-10."""
    known = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF,) * 2,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ]
    for counter, key, words in known:
        out = philox(
            tuple(torch.tensor([c]) for c in counter),
            tuple(torch.tensor([k]) for k in key),
        )
        assert [int(w) for w in out] == list(words), counter


@triton.jit
def _tl_rand_kernel(out_ptr, offsets_ptr, seed, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    offsets = tl.load(offsets_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, tl.rand(seed, offsets), mask=mask)


@pytest.mark.parametrize("seed", [7, _RATIO_ACCEPT_KEY | 5, _RATIO_RESIDUAL_KEY | 9])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_the_reference_coins_are_tl_rand_bit_for_bit(seed, dtype):
    """`triton_rand` is the compiled `tl.rand`, at an int32 offset (the
    accept's draft row) and an int64 one (the race's ``row * V + id``, past
    2**32), under a key with and without a high word."""
    torch.manual_seed(97)
    n, block = 1 << 16, 1024
    high = 1 << 31 if dtype == torch.int32 else 1 << 40
    offsets = torch.randint(0, high, (n,), dtype=torch.int64, device=DEV).to(dtype)
    offsets[:block] = torch.arange(block, device=DEV)
    out = torch.empty(n, device=DEV)
    _tl_rand_kernel[(triton.cdiv(n, block),)](out, offsets, seed, n, BLOCK=block)
    assert torch.equal(out, triton_rand(seed, offsets))


@pytest.mark.parametrize("filtered", [False, True], ids=["p", "p_f"])
def test_the_verify_decides_as_the_fp64_reference_on_the_same_coins(filtered):
    """The verify against `ratio_verify_reference`, which shares none of its
    code: on the same Philox coins every request emits the reference's tokens
    and verdict, save one whose decision lies within rounding of its threshold
    (at most one a step). Greedy rows, requests without q, an emitted slot
    that does not hold the draft, a draft the store does not hold, stored
    entries that are not probabilities and, filtered, per-request top-k and
    top-p cuts are all included."""
    torch.manual_seed(61)
    batch_n, width, vocab = 96, 4, 2 * _RATIO_BLOCK_V + 500
    rows = batch_n * width
    q_logits = torch.randn(rows, vocab, device=DEV) * 2
    q_rows = top_of(q_logits, 40)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    drafts[5::23] = vocab - 1  # not stored: q(x) = 0 whatever xslot says
    logits = (q_logits + torch.randn(rows, vocab, device=DEV)).to(torch.bfloat16)
    temps = torch.tensor([1.0, 0.6, SAMPLER_EPS, 1.3], device=DEV).repeat(batch_n // 4)
    args = verify_batch(drafts, width, logits, temps)
    args["q_row_base"][7::11] = -1
    q = draft_q(q_rows, drafts)
    n, xslot = q.meta.unbind(1)
    # Slots that name another stored id: the walk must find x's true q.
    xslot[1::9] = (xslot[1::9] + 1) % n[1::9]
    # Entries that are not probabilities, at the emitted slot: q is 0 there.
    for start, value in ((3, 1.5), (6, float("nan"))):
        q.probs[start::13].scatter_(1, xslot[start::13, None].long(), value)
    top_ks = top_ps = None
    if filtered:
        top_ks = torch.tensor([-1, 30, 300, 5, -1], dtype=torch.int32, device=DEV)
        top_ps = torch.tensor([1.0, 0.9, 1.0, 0.6, 0.95, 0.8, 1.0], device=DEV)
        top_ks = top_ks.repeat(batch_n)[:batch_n]
        top_ps = top_ps.repeat(batch_n)[:batch_n]
    for step in (0, 1, 2):
        out, verdict = verify(**args, q=q, step=step, top_ks=top_ks, top_ps=top_ps)
        ref_out, ref_verdict, undecided = ratio_verify_reference(
            args, q, step, top_ks, top_ps
        )
        assert undecided.sum() <= 1, step
        decided = ~undecided
        assert torch.equal(out[decided], ref_out[decided]), step
        assert torch.equal(verdict[:, decided], ref_verdict[:, decided]), step
    sampled = temps > SAMPLER_EPS
    accepted = verdict[0][sampled]
    assert (accepted > 0).any() and (accepted < width).any()


@pytest.mark.parametrize("others", [1, 1500], ids=["first-chunk", "past-a-chunk"])
def test_a_slot_that_does_not_hold_the_draft_still_reads_its_true_q(others):
    """The walk reads q(x) at the emitted slot; where that slot holds another
    id it must look x up -- the rate is min(1, p(x) / q(x)) for x's own q, not
    the slot's, and a draft the store does not hold has q 0 and never passes.
    The lookup walks the stored entries a chunk at a time, so x also sits
    past the first chunk."""
    torch.manual_seed(67)
    n, vocab = 20000, 3000
    x, outside = 2999, 5
    p = sparse(vocab, {x: 0.3, outside: 0.7})
    store = empty_store(1, 2048)
    # `others` ids of 0.01 between them, then x at slot `others`.
    store.ids[0, : others + 1] = torch.tensor(
        [*range(1000, 1000 + others), x], device=DEV
    )
    store.probs[0, :others] = 0.01 / others
    store.probs[0, others] = 0.99
    store.meta[0] = torch.tensor([others + 1, 0], device=DEV)  # names id 1000
    for draft, rate in ((x, 0.3 / 0.99), (outside, 0.0)):
        drafts = torch.full((n,), draft, dtype=torch.int32, device=DEV)
        args = verify_batch(
            drafts,
            1,
            p.log().expand(n, -1),
            torch.ones(n, device=DEV),
            row_base=torch.zeros(n, dtype=torch.int32, device=DEV),
        )
        out, (accepted, _, _) = verify(**args, q=store, step=13)
        assert abs(accepted.float().mean().item() - rate) < 0.015, draft
        # relu(p - q) is all on `outside`: x's 0.3 is below its q of 0.99.
        assert (out[accepted == 0, 0] == outside).all(), draft


@pytest.mark.parametrize("temperature", [0.3, 1.0])
@pytest.mark.parametrize("width", [1, 7, 1024, 4096, 6000])
def test_the_first_token_follows_p_at_any_store_width(width, temperature):
    """End to end -- the sampled draft writes the store, the verify reads it --
    the first emitted token follows the target's p, whatever K cut q to, and
    never lands on an id p gives no mass."""
    torch.manual_seed(71)
    n, vocab, rank = 40000, 6000, 32
    target = torch.randn(vocab, device=DEV) * 3
    target[torch.rand(vocab, device=DEV) < 0.3] = float("-inf")
    base = (target.nan_to_num(neginf=-20.0) + torch.randn(vocab, device=DEV)).to(
        torch.bfloat16
    )
    _, prev, w1, w2 = tables(1, vocab, rank, seed=73)
    temps, greedy, seed = draft_args(n, temperature=temperature, step=17)
    store = empty_store(n, width)
    row_base = torch.arange(n, dtype=torch.int32, device=DEV)
    drafts, _ = sample(
        base.expand(n, -1).contiguous(),
        prev.expand(n).contiguous(),
        w1,
        w2,
        store,
        row_base,
        0,
        temps,
        greedy,
        seed,
    )
    out, (accepted, _, _) = verify(
        **verify_batch(drafts.to(torch.int32), 1, target.expand(n, -1), temps),
        q=store,
        step=19,
    )
    p = torch.softmax(target / temperature, -1)
    first = out[:, 0].long()
    assert (p[first] > 0).all(), "emitted an id the target gives no mass"
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)
    assert 0 < accepted.float().mean().item() < 1


@pytest.mark.parametrize("width", [7, 50])
def test_a_chained_block_follows_the_autoregressive_target(width):
    """Three chained positions, each drafted from the last and scored by a
    target conditioned on it: every emitted token follows p(. | the token
    emitted before it) -- position 0 given the anchor, later positions among
    the requests that reached them -- and none lands where p is 0."""
    torch.manual_seed(79)
    n, vocab, rank, steps, anchor = 60000, 50, 16, 3, 3
    base, _, w1, w2 = tables(1, vocab, rank, seed=83)
    # The target near the draft (the draft's own logits given each previous
    # id, perturbed), so enough requests reach the last position.
    _, draft_logits = sample(
        base.expand(vocab, -1).contiguous(),
        torch.arange(vocab, device=DEV),
        w1,
        w2,
        empty_store(vocab, 1),
        torch.arange(vocab, dtype=torch.int32, device=DEV),
        0,
        *draft_args(vocab),
    )
    table = draft_logits + torch.randn(vocab, vocab, device=DEV) * 0.5
    table[torch.rand(vocab, vocab, device=DEV) < 0.2] = float("-inf")
    table[:, 0] = 1.0  # every row keeps some mass
    temps, greedy, seed = draft_args(n, step=23)
    store = empty_store(n * steps, width)
    row_base = torch.arange(0, n * steps, steps, dtype=torch.int32, device=DEV)
    base = base.expand(n, -1).contiguous()
    prev = torch.full((n,), anchor, dtype=torch.int64, device=DEV)
    drafts, scored = [], []
    for k in range(steps):
        scored.append(table[prev])
        out, _ = sample(
            base,
            prev,
            w1,
            w2,
            store,
            row_base,
            k,
            temps,
            greedy,
            seed,
            noise_offset=k * n * vocab,
        )
        drafts.append(out)
        prev = out
    drafts = torch.stack(drafts, 1).view(-1).to(torch.int32)
    logits = torch.stack(scored, 1).view(n * steps, vocab)
    out, (accepted, _, _) = verify(
        **verify_batch(drafts, steps, logits, temps),
        q=store,
        step=29,
    )
    cond = torch.softmax(table, -1)
    given = torch.full((n,), anchor, dtype=torch.int64, device=DEV)
    for k in range(steps):
        reached = accepted >= k
        y = out[reached, k].long()
        prior = given[reached] if k == 0 else out[reached, k - 1].long()
        assert (y >= 0).all()
        assert (cond[prior, y] > 0).all(), f"position {k} emitted a zero-mass id"
        assert y.numel() > 2000, f"too few requests reached position {k}"
        # Chi-square against the mixture of the rows' conditionals.
        expected = cond[prior].double().sum(0)
        counts = torch.bincount(y, minlength=vocab).double()
        live = expected > 0
        chi2 = ((counts - expected)[live] ** 2 / expected[live]).sum().item()
        dof = int(live.sum()) - 1
        assert chi2 < dof + 6 * (2 * dof) ** 0.5, (k, chi2, dof)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "top_k, top_p", [(20, 1.0), (-1, 0.8), (50, 0.7)], ids=["top_k", "top_p", "both"]
)
def test_the_first_token_follows_samplers_filtered_p(top_k, top_p, dtype):
    """End to end, draft and verify each filtering per row: the first token
    follows p_f, never lands on a filtered-out id, and some drafts pass. fp32
    logits take the longer, 32-bit descent."""
    torch.manual_seed(139)
    n, vocab = 40000, 6000
    target = (torch.randn(vocab, device=DEV) * 3).to(dtype)
    temps = torch.full((n,), 0.7, device=DEV)
    ks = torch.full((n,), top_k, dtype=torch.int32, device=DEV)
    ps = torch.full((n,), top_p, device=DEV)
    first, accepted = first_token_law(target, temps, ks, ps)
    kept, p_f = sampler_kept(target[None], temps[:1], ks[:1], ps[:1])
    assert kept[0, first].all(), "emitted an id the filter removes"
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)
    assert 0 < accepted.float().mean().item() < 1


def test_a_mixed_batch_serves_each_request_its_own_p():
    """Greedy, unfiltered, top-k, top-p and both, interleaved in one batch of
    per-request filters: greedy requests get the argmax, the rest each their
    own p_f."""
    torch.manual_seed(149)
    kinds = [(SAMPLER_EPS, -1, 1.0), (0.9, -1, 1.0), (0.9, 15, 1.0)]
    kinds += [(0.9, -1, 0.75), (0.9, 40, 0.6)]
    n, vocab = 60000, 3000
    target = (torch.randn(vocab, device=DEV) * 3).to(torch.bfloat16)
    temps, ks, ps = (
        torch.tensor(column, device=DEV).repeat(n // len(kinds))
        for column in zip(*kinds)
    )
    first, _ = first_token_law(target, temps, ks.to(torch.int32), ps)
    for i, (temp, k, p) in enumerate(kinds):
        got = first[i :: len(kinds)]
        if temp <= SAMPLER_EPS:
            assert (got == target.float().argmax()).all()
            continue
        kept, p_f = sampler_kept(
            target[None],
            temps[i : i + 1],
            ks[i : i + 1].to(torch.int32),
            ps[i : i + 1],
        )
        assert kept[0, got].all(), (temp, k, p)
        freq = torch.bincount(got, minlength=vocab).float() / got.numel()
        torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.015)


def test_each_position_of_a_filtered_block_is_served_its_own_p_f():
    """Three draft positions per request, each scoring its own logits, under
    interleaved per-request filters (one kind unfiltered): every position a
    request reaches emits an id its own row keeps, at that row's p_f."""
    torch.manual_seed(179)
    kinds = [(-1, 1.0), (15, 1.0), (-1, 0.75), (40, 0.6)]
    n, width, vocab, support = 40000, 3, 3000, 64
    target = (torch.randn(width, vocab, device=DEV) * 3).to(torch.bfloat16)
    temps = torch.full((n,), 0.9, device=DEV)
    ks, ps = (
        torch.tensor(column, device=DEV).repeat(n // len(kinds))
        for column in zip(*kinds)
    )
    ks = ks.to(torch.int32)
    # Position k's q: the top of a noisy copy of its target, renormalized.
    noise = torch.randn(width, vocab, device=DEV) * 0.3
    noisy = torch.softmax((target.float() + noise) / 0.9, -1)
    q_probs, q_ids = noisy.topk(support, -1)
    q_probs = (q_probs / q_probs.sum(-1, keepdim=True)).repeat(n, 1)
    q_ids = q_ids.repeat(n, 1).to(torch.int32)
    slots = torch.multinomial(q_probs, 1)
    drafts = q_ids.gather(1, slots).view(-1)
    meta = torch.cat((torch.full_like(slots, support), slots), 1)
    store = SparseQStore(q_ids, q_probs.contiguous(), meta.to(torch.int32))
    out, (accepted, _, _) = verify(
        **verify_batch(drafts, width, target.repeat(n, 1), temps),
        q=store,
        step=53,
        top_ks=ks,
        top_ps=ps,
    )
    for i, kind in enumerate(kinds):
        kept, p_f = sampler_kept(target, temps[:width], ks[i].expand(width), ps[i])
        for k in range(width):
            reached = accepted[i :: len(kinds)] >= k
            got = out[i :: len(kinds), k][reached].long()
            assert got.numel() > 2000, (kind, k)
            assert kept[k, got].all(), (kind, k)
            freq = torch.bincount(got, minlength=vocab).float() / got.numel()
            torch.testing.assert_close(freq, p_f[k], rtol=0, atol=0.02)


def test_a_filtered_out_draft_is_always_rejected():
    """q all on an id top-k removes: p_f there is 0, so no coin accepts it,
    and the replacement follows p_f (relu(p_f - q) is p_f)."""
    torch.manual_seed(151)
    n, vocab = 30000, 3000
    target = (torch.randn(vocab, device=DEV) * 2).to(torch.bfloat16)
    temps = torch.ones(n, device=DEV)
    ks = torch.full((n,), 10, dtype=torch.int32, device=DEV)
    kept, p_f = sampler_kept(target[None], temps[:1], ks[:1])
    outside = int((~kept[0]).nonzero()[0])
    store = empty_store(1, 4)
    store.ids[0, 0], store.probs[0, 0] = outside, 1.0
    store.meta[0] = torch.tensor([1, 0], device=DEV)
    drafts = torch.full((n,), outside, dtype=torch.int32, device=DEV)
    out, (accepted, _, _) = verify(
        **verify_batch(
            drafts,
            1,
            target.expand(n, -1),
            temps,
            row_base=torch.zeros(n, dtype=torch.int32, device=DEV),
        ),
        q=store,
        step=43,
        top_ks=ks,
    )
    assert (accepted == 0).all()
    first = out[:, 0].long()
    assert kept[0, first].all()
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)


def test_a_filtered_residual_without_mass_falls_back_to_p_f():
    """q a hair above p_f on every id top-k keeps (so no rounding of the
    kernel's p_f leaves a residual; q need not sum to 1 for that), and the
    draft on an id it removes: rejected, relu(p_f - q) is 0 everywhere, and
    the fallback must still follow p_f, never landing where the filter cuts."""
    torch.manual_seed(181)
    n, vocab = 30000, 3000
    target = (torch.randn(vocab, device=DEV) * 2).to(torch.bfloat16)
    temps = torch.ones(n, device=DEV)
    ks = torch.full((n,), 10, dtype=torch.int32, device=DEV)
    kept, p_f = sampler_kept(target[None], temps[:1], ks[:1])
    outside = int((~kept[0]).nonzero()[0])
    q = torch.where(kept[0], p_f[0] * 1.001, 0.0)
    q[outside] = 0.01
    drafts = torch.full((n,), outside, dtype=torch.int32, device=DEV)
    out, (accepted, _, _) = verify(
        **verify_batch(drafts, 1, target.expand(n, -1), temps),
        q=draft_q(q, drafts),
        step=59,
        top_ks=ks,
    )
    assert (accepted == 0).all()
    first = out[:, 0].long()
    assert kept[0, first].all(), "the fallback emitted an id the filter removes"
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_masked_and_nan_target_rows_filter_and_emit_inside_the_vocab(dtype):
    """Filtered target rows with masked (-inf) ids, and an all-NaN one: the
    masked row keeps `Sampler`'s set and its first token follows p_f, never
    landing on an id p_f gives no mass; the NaN row emits an id inside the
    vocab."""
    torch.manual_seed(167)
    n, vocab = 30000, 3000
    target = torch.randn(2, vocab, device=DEV) * 3
    target[0, torch.rand(vocab, device=DEV) < 0.3] = float("-inf")
    target[1] = float("nan")
    target = target.to(dtype)
    temps = torch.full((2,), 0.8, device=DEV)
    ks = torch.tensor([20, 20], dtype=torch.int32, device=DEV)
    ps = torch.tensor([0.8, 0.8], device=DEV)
    kept, p_f = filtered_p(target, temps, ks, ps)
    ref, ref_p = sampler_kept(target[:1], temps[:1], ks[:1], ps[:1])
    assert torch.equal(kept[0], ref[0])
    torch.testing.assert_close(p_f[0], ref_p[0], rtol=1e-4, atol=1e-7)
    q = sparse(vocab, {i: 1 / 50 for i in range(50)})
    drafts = draw(q, 2 * n)
    out, _ = verify(
        **verify_batch(drafts, 1, target.repeat(n, 1), temps.repeat(n)),
        q=draft_q(q, drafts),
        step=47,
        top_ks=ks.repeat(n),
        top_ps=ps.repeat(n),
    )
    first = out[:, 0].long()
    assert ((first >= 0) & (first < vocab)).all()
    masked = first[0::2]
    assert (p_f[0, masked] > 0).all(), "emitted an id p_f gives no mass"
    freq = torch.bincount(masked, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)


def test_filters_that_keep_everything_leave_the_verify_bit_for_bit():
    """top_k >= V or -1 and top_p 1.0, as one value or per request: the same
    tokens and verdicts as no filter at all, coin for coin."""
    torch.manual_seed(157)
    batch_n, width, vocab = 64, 4, 2 * _RATIO_BLOCK_V + 77
    rows = batch_n * width
    q_logits = torch.randn(rows, vocab, device=DEV) * 2
    q_rows = top_of(q_logits, 30)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    logits = (q_logits + torch.randn(rows, vocab, device=DEV)).to(torch.bfloat16)
    temps = torch.tensor([1.0, 0.6, SAMPLER_EPS, 1.3], device=DEV).repeat(batch_n // 4)
    args = verify_batch(drafts, width, logits, temps)
    q = draft_q(q_rows, drafts)
    ref = verify(**args, q=q, step=3)
    every_k = torch.tensor([-1, vocab, vocab + 5, -1], device=DEV, dtype=torch.int32)
    for ks, ps in (
        (every_k.repeat(batch_n // 4), torch.ones(batch_n, device=DEV)),
        (vocab, None),
        (None, 1.0),
    ):
        got = verify(**args, q=q, step=3, top_ks=ks, top_ps=ps)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), (ks, ps)
