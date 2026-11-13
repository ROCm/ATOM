# SPDX-License-Identifier: MIT
"""The ratio verify's top-k / top-p floor (`sampling_filter`, its count and
descent): the kept set is `Sampler`'s, bit for bit, and each draft row finds
its request.

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
    filtered_p,
    sampler_kept,
    tied_rows,
)

from atom.model_ops.spec_sampling.ratio_target_p import (
    _RATIO_BLOCK_V,
)
from atom.model_ops.spec_sampling.sampling_filter import _request_of_row


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_kept_set_is_samplers_bit_for_bit(dtype):
    """Top-k alone is a count, so the set is exact on every row (`tied_rows`
    keeps clear of 0; see the test after next): cuts inside a tie group keep
    the group whole, top_k = 1 keeps the max's group, top_k >=
    V and -1 keep everything. Top-p is compared on the rows whose cut is not
    within rounding of top_p; top-p 1.0 keeps everything. fp32 logits take
    the 32-bit descent."""
    rows, vocab = 96, 3 * _RATIO_BLOCK_V + 123
    logits = tied_rows(rows, vocab, 113, dtype)
    temps = torch.rand(rows, device=DEV) + 0.3
    top_ks = torch.randint(1, 400, (rows,), device=DEV, dtype=torch.int32)
    top_ks[::8], top_ks[1::8], top_ks[2::8], top_ks[3::8] = 1, vocab, vocab + 9, -1
    top_ps = torch.rand(rows, device=DEV) * 0.98 + 0.01
    top_ps[4::8] = 1.0
    nudge = torch.full_like(top_ps, 1e-5)
    for ks, ps in (
        (top_ks, None),
        (None, top_ps),
        (top_ks, top_ps),
        (int(top_ks[5]), None),
        (None, float(top_ps[5])),
        # `prepare_sample` picks scalar or per-row for each independently.
        (top_ks, float(top_ps[5])),
        (int(top_ks[5]), top_ps),
    ):
        kept, p_f = filtered_p(logits, temps, ks, ps)
        k_rows = ks if ks is None or isinstance(ks, torch.Tensor) else top_ks * 0 + ks
        p_rows = ps if ps is None or isinstance(ps, torch.Tensor) else top_ps * 0 + ps
        ref, ref_p = sampler_kept(logits, temps, k_rows, p_rows)
        robust = torch.ones(rows, dtype=torch.bool, device=DEV)
        if p_rows is not None:
            low = sampler_kept(logits, temps, k_rows, p_rows - nudge)[0]
            high = sampler_kept(logits, temps, k_rows, p_rows + nudge)[0]
            robust = (low == high).all(1)
            assert robust.float().mean() > 0.8
        assert torch.equal(kept[robust], ref[robust]), (ks, ps)
        torch.testing.assert_close(p_f[robust], ref_p[robust], rtol=1e-4, atol=1e-7)
        torch.testing.assert_close(
            p_f.double().sum(-1),
            torch.ones(rows, device=DEV).double(),
            rtol=0,
            atol=1e-5,
        )
    # `Sampler`'s own top-k kernel keeps that set too, tie for tie.
    kept, _ = filtered_p(logits, temps, top_ks, None)
    on = top_ks > 0
    probs = (logits / temps[:, None]).softmax(dim=-1, dtype=torch.float32)
    renorm = torch.ops.aiter.top_k_renorm_probs(
        probs[on].contiguous(), top_ks[on].contiguous(), 0
    )
    assert torch.equal(renorm > 0, kept[on])


def test_ragged_requests_each_keep_their_own_set_at_the_real_vocab():
    """Requests of 0 to 7 draft rows, each with its own temperature, top-k and
    top-p, at V=163840: every row keeps its own request's set. 64 rows split
    the last count into programs of three tiles each, the last one ragged."""
    vocab = 163840
    widths = torch.tensor([3, 0, 7, 1, 5, 0, 2, 7, 4, 6, 0, 7, 7, 6, 2, 7])
    rows, batch_n = int(widths.sum()), widths.numel()
    logits = tied_rows(rows, vocab, 173)
    temps = torch.rand(batch_n, device=DEV) + 0.3
    top_ks = torch.randint(1, 400, (batch_n,), device=DEV, dtype=torch.int32)
    top_ks[::5], top_ks[1::7] = -1, vocab
    top_ps = torch.rand(batch_n, device=DEV) * 0.98 + 0.01
    top_ps[2::5] = 1.0
    nudge = torch.full((rows,), 1e-5, device=DEV)

    def per_row(x):
        return None if x is None else x.repeat_interleave(widths.to(DEV))

    for ks, ps in ((top_ks, None), (None, top_ps), (top_ks, top_ps)):
        kept, p_f = filtered_p(logits, temps, ks, ps, widths)
        k_rows, p_rows = per_row(ks), per_row(ps)
        ref, ref_p = sampler_kept(logits, per_row(temps), k_rows, p_rows)
        robust = torch.ones(rows, dtype=torch.bool, device=DEV)
        if p_rows is not None:
            low = sampler_kept(logits, per_row(temps), k_rows, p_rows - nudge)[0]
            high = sampler_kept(logits, per_row(temps), k_rows, p_rows + nudge)[0]
            # A request's rows stand or fall together, and the real vocab's
            # tie groups are heavy: fewer cuts clear the nudge than above.
            robust = (low == high).all(1)
            assert robust.float().mean() > 0.5
        assert torch.equal(kept[robust], ref[robust]), (ks, ps)
        torch.testing.assert_close(p_f[robust], ref_p[robust], rtol=1e-4, atol=1e-7)


def test_a_cut_inside_a_tie_group_keeps_the_whole_group():
    """Five ids tie for second: top-k 2 keeps all six, and top-p just past
    the max's mass keeps them too (their mass above is the max's alone)."""
    logits = torch.full((1, 64), -4.0, device=DEV)
    logits[0, 9] = 3.0
    logits[0, [2, 17, 33, 40, 63]] = 1.0
    logits = logits.to(torch.bfloat16)
    temps = torch.ones(1, device=DEV)
    top = torch.softmax(logits.float(), -1)[0, 9].item()
    for ks, ps in ((2, None), (None, top + 1e-3)):
        kept, _ = filtered_p(logits, temps, ks, ps)
        assert kept[0].nonzero().view(-1).tolist() == [2, 9, 17, 33, 40, 63]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_counted_top_k_keeps_what_the_descent_keeps(monkeypatch, dtype):
    """A top-k of at most 256 alone is counted from the stats pass's run
    maxima, not descended: the same floor bit for bit, and p_f to rounding,
    over tied rows, top_k 1 to 256 (cuts past one gather of runs), masked,
    NaN, +-inf and all--inf rows, an allow-list of 10 ids in -inf and
    plateaus that tie thousands of runs at the k-th largest max, scalar and
    per request, at the real vocab and at one whose last tile is ragged. A
    top_p that is off (1.0, scalar or per request) changes nothing, the
    +inf row included: it gets its top-k floor whatever its batch passes."""
    import atom.model_ops.spec_sampling.ratio_target_p as rtp
    import atom.model_ops.spec_sampling.sampling_filter as sf

    for vocab in (163840, 3 * _RATIO_BLOCK_V + 123):
        rows = 40
        logits = tied_rows(rows, vocab, 191, torch.float32)
        logits[0, torch.rand(vocab, device=DEV) < 0.3] = float("-inf")
        logits[1, :7] = float("nan")
        logits[2] = float("nan")
        logits[3] = float("-inf")
        logits[4, 11] = float("inf")
        logits[5, 1::2] = 0.0
        logits[5, ::2] = -0.0
        logits[6, : vocab // 2] = logits[6].max()
        logits[7] = float("-inf")
        logits[7, torch.randperm(vocab, device=DEV)[:10]] = torch.randn(10, device=DEV)
        logits[8] = 0.0
        logits[8, 5] = 10.0
        logits[9] = -1.0
        logits[9, torch.randperm(vocab, device=DEV)[:30]] = torch.arange(
            30, device=DEV, dtype=torch.float32
        )
        logits = logits.to(dtype)
        temps = torch.rand(rows, device=DEV) + 0.3
        top_ks = torch.randint(1, 257, (rows,), device=DEV, dtype=torch.int32)
        top_ks[::7], top_ks[1::9], top_ks[2::11] = 1, 256, -1
        for ks in (top_ks, 1, 20, 50, 256):
            counted = filtered_p(logits, temps, ks)
            with monkeypatch.context() as m:
                for module in (rtp, sf):
                    m.setattr(module, "_FILTER_COUNT_MAX_K", 0)
                descended = filtered_p(logits, temps, ks)
            assert torch.equal(counted[0], descended[0]), (vocab, ks)
            torch.testing.assert_close(
                counted[1], descended[1], rtol=1e-5, atol=1e-9, equal_nan=True
            )
            for ps in (1.0, torch.ones(rows, device=DEV)):
                p_off = filtered_p(logits, temps, ks, ps)
                assert torch.equal(p_off[0], counted[0]), (vocab, ks, ps)
                assert torch.equal(p_off[1].isnan(), counted[1].isnan())


def test_the_count_rule_splits_only_what_samplers_p_rounds_together():
    """The descent ranks ids by logit, aiter by `Sampler`'s fp32 p, whose
    ``logits / T - max`` rounds the distinct logits 2e-8 and 1e-8 onto one
    value: aiter's top-k 3 keeps both as a tie, the floor only the higher.
    Pinned so the gap stays the one the notes allow: every id the two
    disagree on has the p of the lowest id the floor keeps."""
    logits = torch.full((1, 4096), -20.0, device=DEV)
    logits[0, :4] = torch.tensor([1.0, 3e-8, 2e-8, 1e-8], device=DEV)
    logits = logits.to(torch.bfloat16)
    temps = torch.ones(1, device=DEV)
    ks = torch.tensor([3], dtype=torch.int32, device=DEV)
    kept, _ = filtered_p(logits, temps, ks)
    probs = (logits / temps[:, None]).softmax(dim=-1, dtype=torch.float32)
    aiter_kept = torch.ops.aiter.top_k_renorm_probs(probs, ks, 0) > 0
    assert kept[0].nonzero().view(-1).tolist() == [0, 1, 2]
    assert aiter_kept[0].nonzero().view(-1).tolist() == [0, 1, 2, 3]
    assert not (kept & ~aiter_kept).any()
    assert (probs[aiter_kept & ~kept] == probs[kept].min()).all()
