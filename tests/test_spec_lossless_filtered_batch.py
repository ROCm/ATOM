# SPDX-License-Identifier: MIT
"""A top-k/top-p batch runs the flag-OFF pair, not half of each.

The probability-ratio accept rule cannot verify a filtered row:
`rejection_sampler._ratio_target_probs` builds `p` as a plain temperature
softmax and never filters it, so such a row would accept against -- and draw
its correction token from -- the UNFILTERED target. So the verify side drops the
batch to exact-match acceptance, and batch-wide at that: the ratio kernel has
one accept rule per launch and no exact-match branch to route a single row into.

That drop used to be a REGRESSION rather than a fallback. The drafts had still
been sampled, and a sampled proposal under exact-match acceptance matches the
target's own draw strictly less often than the greedy proposal the flag-off path
makes -- so one `top_p < 1` request made every row in the batch accept less often
than with the flag off entirely. The fix is on the draft side, a step earlier:
such a batch drafts GREEDILY, which puts it back on exactly the shipped pair.

Two halves, one test file:

* `_lossless_verify_inputs` -- the verify side never hands the ratio kernel a
  filtered batch.
* `stage_draft_sampling` -- the draft side answers that by clamping the DRAFT's
  temperature to `SAMPLER_EPS` and setting every `greedy_mask` row, so the token
  is an argmax AND the published `q` is the delta that argmax drew from. The
  mask alone would not do: it only replaces the exponential race, leaving `q` at
  `softmax(logits/T)`, which for `T = 1` is nothing like a delta. That matters
  because propose and verify are a step apart -- this batch's drafts are tested
  against the NEXT batch's filters -- and a delta `q` is sound under either rule.

CPU-only, and lifted out of the shipped sources with `ast` for the reason
`tests/test_dspark_q_slots.py` states at length: `atom.model_engine.model_runner`
and `atom.spec_decode.dspark_proposer` both need an AITER build to import, so an
imported version of this file would sit behind `importorskip("aiter")` and get
ZERO coverage in CI (`.github/scripts/run_unit_tests.sh`).
"""

import ast
import pathlib

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER_SOURCE = REPO / "atom/model_engine/model_runner.py"
PROPOSER_SOURCE = REPO / "atom/spec_decode/dspark_proposer.py"
SAMPLER_EPS = 1e-10


def _lift(path, name, namespace, cls=None):
    """Compile one shipped function out of `path`, without importing it.

    Not a transcription: the source read here is the one the engine runs, so a
    change to either function arrives in this test by itself rather than leaving
    it asserting on a stale copy. `namespace` carries exactly the module-level
    names the function reaches for; anything added later raises `NameError` here.
    """
    tree = ast.parse(path.read_text(), str(path))
    body = tree.body
    if cls is not None:
        found = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        assert len(found) == 1, f"{cls} moved or is defined twice in {path.name}"
        body = found[0].body
    picked = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(picked) == 1, f"{path.name} no longer defines exactly one {name}"
    exec(  # noqa: S102 - the source is this repository's own, read above
        compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


lossless_verify_inputs = _lift(RUNNER_SOURCE, "_lossless_verify_inputs", {})
stage_draft_sampling = _lift(
    PROPOSER_SOURCE,
    "stage_draft_sampling",
    {"torch": torch, "SAMPLER_EPS": SAMPLER_EPS},
    cls="DSparkProposer",
)


class FakeDrafter:
    """A drafter with `q` published: the only case the gate has to decide."""

    def __init__(self):
        self.draft_probs = torch.zeros(4, 8)
        self.asked = None

    def draft_probs_row_base(self, req_ids):
        self.asked = list(req_ids)
        return torch.arange(len(req_ids), dtype=torch.int32)


@pytest.mark.parametrize(
    "top_ks, top_ps",
    [
        (50, None),  # uniform top-k, kept on the host as a scalar
        (None, 0.9),  # uniform top-p
        (torch.tensor([50, -1], dtype=torch.int32), None),  # one row of two
        (None, torch.tensor([0.9, 1.0])),
        (20, 0.9),
    ],
)
def test_a_filtered_batch_never_reaches_the_ratio_rule(top_ks, top_ps):
    """Any filter on any row, in either of the two shapes `prepare_sample`
    emits (host scalar for a uniform filter, `[bs]` tensor for a mixed one)."""
    drafter = FakeDrafter()
    assert lossless_verify_inputs(
        drafter, ["a", "b"], all_greedy=False, top_ks=top_ks, top_ps=top_ps
    ) == (None, None)
    assert drafter.asked is None, "the row map was consulted for a dropped batch"


def test_an_unfiltered_stochastic_batch_does_reach_it():
    """The positive control: without it the test above passes vacuously."""
    drafter = FakeDrafter()
    probs, row_base = lossless_verify_inputs(
        drafter, ["a", "b"], all_greedy=False, top_ks=None, top_ps=None
    )
    assert probs is drafter.draft_probs
    assert drafter.asked == ["a", "b"]
    torch.testing.assert_close(
        row_base, torch.tensor([0, 1], dtype=torch.int32), rtol=0, atol=0
    )


def test_no_drafter_at_all_stays_on_exact_match():
    """Every drafter but an armed DSpark one: `Drafter`'s no-op hooks."""
    assert lossless_verify_inputs(
        None, ["a"], all_greedy=False, top_ks=None, top_ps=None
    ) == (None, None)


def test_a_drafter_with_no_q_published_stays_on_exact_match():
    """The default, since `ATOM_SPEC_LOSSLESS_SAMPLING` defaults off."""
    drafter = FakeDrafter()
    drafter.draft_probs = None
    assert lossless_verify_inputs(
        drafter, ["a"], all_greedy=False, top_ks=None, top_ps=None
    ) == (None, None)


def test_an_all_greedy_batch_stays_on_exact_match():
    """At temperature 0 the two rules coincide, so the ratio kernel would buy
    nothing and could only move the answer by an ulp. This is the gate that
    makes the flag a bit-for-bit no-op on a greedy batch."""
    drafter = FakeDrafter()
    assert lossless_verify_inputs(
        drafter, ["a"], all_greedy=True, top_ks=None, top_ps=None
    ) == (None, None)
    assert drafter.asked is None


class FakeProposer:
    """`stage_draft_sampling`'s world: the runner's temperatures, the draft's
    own copy of them, and the greedy mask the block head reads."""

    stage_draft_sampling = stage_draft_sampling

    def __init__(self, temperatures, has_filter, max_bs=8):
        bs = len(temperatures)
        self.draft_sampling = type(
            "DraftSampling",
            (),
            {
                "temperatures": torch.ones(max_bs),
                "greedy_mask": torch.zeros(max_bs, dtype=torch.int32),
                "seed": torch.zeros(1, dtype=torch.int32),
            },
        )()
        gpu = torch.ones(max_bs)
        gpu[:bs] = torch.tensor(temperatures, dtype=torch.float32)
        self.runner = type(
            "Runner",
            (),
            {
                "forward_vars": {"temperatures": type("B", (), {"gpu": gpu})()},
                "sampling_has_row_filter": has_filter,
            },
        )()
        self._draft_sample_step = 0
        self.bs = bs


# Mixed temperatures, so "the whole batch went greedy" is distinguishable from
# "the rows that were already greedy stayed greedy".
TEMPS = [1.0, SAMPLER_EPS, 0.7]


def test_a_filtered_batch_drafts_greedily_and_publishes_a_delta_q():
    p = FakeProposer(TEMPS, has_filter=True)
    p.stage_draft_sampling(p.bs)
    # Every row, not only the filtered ones: the verify-side drop is batch-wide,
    # so a row left sampling would be the sampled-draft/exact-match pair that is
    # worse than flag-off.
    torch.testing.assert_close(
        p.draft_sampling.greedy_mask[: p.bs],
        torch.ones(p.bs, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    # The temperature too, which is what makes the published q a delta rather
    # than merely making the id an argmax.
    torch.testing.assert_close(
        p.draft_sampling.temperatures[: p.bs],
        torch.full((p.bs,), SAMPLER_EPS),
        rtol=0,
        atol=0,
    )


def test_an_unfiltered_batch_keeps_each_requests_own_temperature():
    """The control, and the contract the ratio rule rests on: the draft divides
    by the number the target's verify softmax divides by, row for row."""
    p = FakeProposer(TEMPS, has_filter=False)
    p.stage_draft_sampling(p.bs)
    torch.testing.assert_close(
        p.draft_sampling.temperatures[: p.bs],
        torch.tensor(TEMPS, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        p.draft_sampling.greedy_mask[: p.bs],
        torch.tensor([0, 1, 0], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_the_tail_rows_past_the_scheduled_batch_are_left_alone():
    """`stage_draft_sampling` is sized by the SCHEDULED batch; the slots above
    it are fabricated padding and a 0 temperature there divides by zero inside
    the block head."""
    p = FakeProposer(TEMPS, has_filter=True)
    p.stage_draft_sampling(p.bs)
    torch.testing.assert_close(
        p.draft_sampling.temperatures[p.bs :],
        torch.ones(p.draft_sampling.temperatures.numel() - p.bs),
        rtol=0,
        atol=0,
    )
