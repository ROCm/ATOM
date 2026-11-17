# SPDX-License-Identifier: MIT
"""`Sampler` armed for lossless sampling (`Sampler.keep_ties`): its draws stay
in the verify's kept set and follow p_f, a tie group at the cut kept whole.

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


from spec_lossless_reference import (
    DEV,
    assert_draws_are,
    filtered_p,
    tied_rows,
)

from atom.model_ops.sampler import Sampler


@pytest.mark.parametrize(
    "top_k, top_p", [(30, None), (None, 0.7), (100, 0.6), (1000, 0.95)]
)
def test_samplers_draws_stay_in_the_kept_set_and_follow_p_f(top_k, top_p):
    """Against `Sampler` itself (its aiter kernels, not a re-derivation), on
    tied and untied rows: Sampler's draws follow p_f. aiter has no top-p mask
    to compare sets with, so this is the top-p check against aiter itself."""
    torch.manual_seed(127)
    rows, n, vocab = 4, 40000, 4000
    logits = tied_rows(rows, vocab, 131).repeat_interleave(n, 0)
    temps = torch.full((rows * n,), 0.8, device=DEV)
    ks = None if top_k is None else torch.full((rows * n,), top_k, device=DEV)
    ps = None if top_p is None else torch.full((rows * n,), top_p, device=DEV)
    ks = None if ks is None else ks.to(torch.int32)
    draws = Sampler()(logits, temps, ks, ps).long().view(rows, n)
    kept, p_f = filtered_p(
        logits[::n],
        temps[:rows],
        None if ks is None else ks[:rows],
        None if ps is None else ps[:rows],
    )
    for row in range(rows):
        assert_draws_are(draws[row], kept[row], p_f[row])


def test_the_bonus_draw_keeps_a_top_k_tie_group_whole_at_the_real_vocab():
    """For a scalar top-k (small k, large vocab) with a top-p, aiter's joint
    op takes a top-k-first path that keeps exactly k ids, splitting the tie
    group at the cut. The ratio verify's p_f keeps the group whole, and so
    does every draw an armed `Sampler` makes (``keep_ties``): from the
    scalar the runner hands a uniform batch, and from a per-row k."""
    torch.manual_seed(163)
    n, vocab = 10000, 163840
    row = ((torch.randn(vocab, device=DEV) * 12).round() / 4).to(torch.bfloat16)
    logits = row.expand(n, -1).contiguous()
    temps = torch.full((n,), 0.8, device=DEV)
    kept, p_f = filtered_p(logits[:1], temps[:1], 50, 0.95)
    assert kept[0].sum() > 50, "the cut is not inside a tie group"
    sampler = Sampler()
    per_row = torch.full((n,), 50, dtype=torch.int32, device=DEV)
    sampler.keep_ties(per_row)
    for top_k in (50, per_row):
        draws = sampler(logits, temps, top_k, 0.95).long()
        assert_draws_are(draws, kept[0], p_f[0])
