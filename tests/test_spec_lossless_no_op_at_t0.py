# SPDX-License-Identifier: MIT
"""ATOM_SPEC_LOSSLESS_SAMPLING changes NOTHING on an all-greedy batch.

Most deployments serve at temperature 0, so the cheapest evidence that this
feature cannot regress them is that it is a bit-for-bit no-op there. It is, and
for two separate reasons that have to hold together -- the flag arms BOTH the
proposal and the accept rule, so either half drifting would be a change:

* **The draft.** Flag off, a block position is
  `argmax(base_logits + W1[x_{k-1}] @ W2^T)`
  (`DSparkMarkovHead.sample_next` -> `dspark_markov_argmax`). Flag on it is a
  draw from `softmax(that / T)` (`sample_next_q` -> `dspark_markov_sample`). At
  T=0 the second IS the first: `prepare_sample` clamps the temperature to
  `SAMPLER_EPS`, `stage_draft_sampling` marks the row in `greedy_mask`, and the
  sampler's exponential race degenerates to a plain argmax over a distribution
  that is itself a delta. Same ids, same tie-breaking, position after position
  down the chain -- and the chain matters, because position k's id is position
  k+1's `prev_ids`, so one divergence would not stay local.
* **The accept rule.** `ModelRunner.postprocess` passes `draft_probs=None` for
  an all-greedy batch, so `rejection_sample` takes the same exact-match kernel
  with the same arguments it takes with the flag off. Not an optimisation: it
  is what makes the output bit-identical rather than merely equal in
  distribution. `coin * q < p` against a delta q accepts with probability
  `p(x*)` -- the same RATE the exact-match rule has, and the same emitted
  distribution -- but it is a coin flip rather than an equality test, so it
  would not reproduce the same stream.

The delta-q fact above is also what `tests/test_spec_lossless_filtered_batch.py`
relies on: a filtered batch is forced to T=`SAMPLER_EPS` precisely so its
published q is the delta its argmax drew from.

CPU-only, over the shipped torch references lifted with `ast` --
`atom.model_ops.dspark_markov_sample` imports aiter and triton, so an imported
version of this file would sit behind `importorskip("aiter")` and get ZERO
coverage in CI (`.github/scripts/run_unit_tests.sh`). The fused Triton kernels'
agreement with these references is the subject of
`tests/test_dspark_markov_sample.py`, which is aiter-gated by necessity.
"""

import pathlib

import pytest
import torch
from test_spec_lossless_filtered_batch import SAMPLER_EPS, _lift, lossless_verify_inputs

MARKOV_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "atom/model_ops/dspark_markov_sample.py"
)

torch_sample = _lift(MARKOV_SOURCE, "_torch_dspark_markov_sample", {"torch": torch})
torch_argmax = _lift(MARKOV_SOURCE, "_torch_dspark_markov_argmax", {"torch": torch})

VOCAB = 64
RANK = 8
ROWS = 32
BLOCK = 4


def _tables(rows=ROWS, vocab=VOCAB, rank=RANK):
    """A draft block's inputs: one `[vocab]` base-logit row per block position,
    plus the Markov head's two factor tables."""
    f = {"dtype": torch.bfloat16}
    return (
        [torch.randn(rows, vocab, **f) for _ in range(BLOCK)],
        torch.randint(0, vocab, (rows,), dtype=torch.int64),
        torch.randn(vocab, rank, **f),
        torch.randn(vocab, rank, **f),
    )


def _greedy_block(base, anchors, w1, w2):
    """`DSparkDraftModel._sample_block`, flag off: argmax, chained."""
    ids = [anchors]
    for k in range(BLOCK):
        out = torch.empty(base[k].shape[0], dtype=torch.int64)
        torch_argmax(base[k], ids[-1], w1, w2, out)
        ids.append(out)
    return torch.stack(ids[1:], dim=1)


def _lossless_block(base, anchors, w1, w2, temperatures, greedy_mask):
    """`DSparkDraftModel.sample_block_lossless`, flag on: draw and publish q."""
    rows, vocab = base[0].shape
    ids = [anchors]
    q = torch.empty(BLOCK, rows, vocab, dtype=torch.float32)
    for k in range(BLOCK):
        out = torch.empty(rows, dtype=torch.int64)
        torch_sample(
            base[k], ids[-1], w1, w2, out, q[k], temperatures, greedy_mask, None, 0
        )
        ids.append(out)
    return torch.stack(ids[1:], dim=1), q


def test_a_greedy_block_drafts_the_same_ids_with_the_flag_on():
    """The claim, end to end over a chained block: identical ids, not merely
    identically distributed ones."""
    torch.manual_seed(20261013)
    base, anchors, w1, w2 = _tables()
    greedy = _greedy_block(base, anchors, w1, w2)
    drawn, _ = _lossless_block(
        base,
        anchors,
        w1,
        w2,
        torch.full((ROWS,), SAMPLER_EPS),
        torch.ones(ROWS, dtype=torch.int32),
    )
    assert torch.equal(drawn, greedy)


def test_a_greedy_row_publishes_the_delta_its_argmax_drew_from():
    """q has to BE the distribution the token came from, or the ratio rule is
    testing the wrong row. At T=`SAMPLER_EPS` it is a one-hot on the id drawn,
    which is what makes `sum_x min(p, q) = p(x*)` an identity rather than an
    approximation."""
    torch.manual_seed(20261013)
    base, anchors, w1, w2 = _tables()
    drawn, q = _lossless_block(
        base,
        anchors,
        w1,
        w2,
        torch.full((ROWS,), SAMPLER_EPS),
        torch.ones(ROWS, dtype=torch.int32),
    )
    expected = torch.zeros(BLOCK, ROWS, VOCAB)
    expected.scatter_(2, drawn.t().unsqueeze(-1), 1.0)
    torch.testing.assert_close(q, expected, rtol=0, atol=0)


def test_a_greedy_row_inside_a_stochastic_batch_still_drafts_greedily():
    """`greedy_mask` is per row, so a T=0 request batched beside sampling ones
    keeps the greedy draft -- the mixed batch is the common serving shape."""
    torch.manual_seed(20261013)
    base, anchors, w1, w2 = _tables()
    greedy_mask = torch.zeros(ROWS, dtype=torch.int32)
    greedy_mask[::2] = 1
    temperatures = torch.ones(ROWS)
    temperatures[::2] = SAMPLER_EPS
    greedy = _greedy_block(base, anchors, w1, w2)
    drawn, _ = _lossless_block(base, anchors, w1, w2, temperatures, greedy_mask)
    assert torch.equal(drawn[::2], greedy[::2])


def test_the_sampled_rows_are_not_trivially_the_argmax_too():
    """The control. Without it every test above would pass against a block
    sampler that had quietly stopped sampling at all."""
    torch.manual_seed(20261013)
    base, anchors, w1, w2 = _tables()
    greedy = _greedy_block(base, anchors, w1, w2)
    drawn, _ = _lossless_block(
        base,
        anchors,
        w1,
        w2,
        torch.ones(ROWS),
        torch.zeros(ROWS, dtype=torch.int32),
    )
    assert not torch.equal(drawn, greedy)


@pytest.mark.parametrize("top_ks, top_ps", [(None, None), (50, None), (None, 0.9)])
def test_an_all_greedy_batch_verifies_with_the_flag_off_arguments(top_ks, top_ps):
    """The other half: `rejection_sample` is handed `draft_probs=None`, i.e.
    the exact-match kernel on the arguments the flag-off path gives it, however
    the rest of the batch is parameterised."""

    class Armed:
        draft_probs = torch.zeros(4, 8)

        def draft_probs_row_base(self, req_ids):
            raise AssertionError("an all-greedy batch consulted the q row map")

    assert lossless_verify_inputs(
        Armed(), ["a", "b"], all_greedy=True, top_ks=top_ks, top_ps=top_ps
    ) == (None, None)
