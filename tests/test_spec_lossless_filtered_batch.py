# SPDX-License-Identifier: MIT
"""A top-k/top-p batch runs the flag-off pair, not half of each.

The probability-ratio accept rule cannot verify a filtered row:
`rejection_sampler._ratio_target_probs` builds `p` as a plain temperature
softmax and never filters it, so such a row would accept against -- and draw
its correction token from -- the unfiltered target. So the verify side drops the
batch to exact-match acceptance, and batch-wide at that: the ratio kernel has
one accept rule per launch and no exact-match branch to route a single row into.

The draft side has to move with it. A sampled proposal under exact-match
acceptance matches the target's own draw strictly less often than a greedy one,
so a filtered batch left sampling would be worse than flag-off; drafting it
greedily puts it back on exactly the shipped pair.

Two halves, one test file:

* `_lossless_verify_inputs` -- the verify side never hands the ratio kernel a
  filtered batch.
* `stage_draft_sampling_host` -- the draft side answers by clamping the draft's
  temperature to `SAMPLER_EPS` and setting every `greedy_mask` row, so the token
  is an argmax and the published `q` is the delta that argmax drew from. The
  mask alone would not do: it only replaces the exponential race, leaving `q` at
  `softmax(logits/T)`, a distribution the token was not drawn from. That matters
  because propose and verify are a step apart -- this batch's drafts are tested
  against the next batch's filters -- and a delta `q` is sound under either rule.
  `tests/test_spec_lossless_no_op_at_t0.py` is where that delta is pinned.

CPU-only, and lifted out of the shipped sources with `ast` for the reason
`tests/test_dspark_q_slots.py` gives: `atom.model_engine.model_runner` and
`atom.spec_decode.dspark_proposer` both need an AITER build to import, so an
imported version of this file would sit behind `importorskip("aiter")` and get
no coverage in CI.
"""

import ast
import pathlib

import numpy as np
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
needs_verification_sample = _lift(RUNNER_SOURCE, "_needs_verification_sample", {})
stage_draft_sampling_host = _lift(
    PROPOSER_SOURCE,
    "stage_draft_sampling_host",
    {"np": np, "torch": torch, "SAMPLER_EPS": SAMPLER_EPS},
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


def test_the_ratio_path_skips_the_target_verification_draw():
    """The saving: on the ratio path `rejection_sample` never dereferences
    `target_token_ids`, so drawing it was a full filtered `[num_draft, V]`
    sample (plus a TP broadcast) thrown away every step."""
    assert not needs_verification_sample(False, torch.zeros(4, 8))


def test_a_stochastic_batch_without_q_still_draws_its_own_tokens():
    """The trap the skip above must not fall into.

    `_lossless_verify_inputs` returns None for plenty of non-greedy batches --
    the flag off entirely, a top-k/top-p batch, a model that does not declare
    `supports_lossless_draft`, a request expired out of the q slot map. Those
    all land on `rejection_sample`'s EXACT-MATCH kernel, which with
    `target_token_ids=None` falls through to `topk_select(target_probs, 1)` and
    matches the draft against the target's ARGMAX instead of against the
    target's own draw. That is a lossy rule, and it would apply to requests
    that are not using this feature at all.

    Gating on `not all_greedy` instead of on `draft_probs is not None` is
    exactly how that happens, so pin it here rather than in a comment.
    """
    assert needs_verification_sample(False, None)
    # And the whole gate agrees: every batch the verify side drops to exact
    # match is a batch that must still be handed the target's own draw.
    drafter = FakeDrafter()
    probs, _ = lossless_verify_inputs(
        drafter, ["a", "b"], all_greedy=False, top_ks=None, top_ps=0.9
    )
    assert needs_verification_sample(False, probs)


def test_an_all_greedy_batch_never_draws_one():
    """Unchanged, and for the original reason: a greedy batch wants the argmax
    the exact-match kernel computes for itself."""
    assert not needs_verification_sample(True, None)
    assert not needs_verification_sample(True, torch.zeros(4, 8))


class FakeBuffer:
    """A `CpuGpuBuffer` as `stage_draft_sampling_host` uses it: a pinned host
    array it writes, which the runner's batched `sampling` H2D later uploads."""

    def __init__(self, n, dtype, fill):
        self.np = np.full(n, fill, dtype=dtype)


class FakeGroup:
    """A `PublicationGroup`: named slots whose counts say how many rows of each
    member this step's single upload carries."""

    def __init__(self, names):
        self.indices = {name: i for i, name in enumerate(names)}
        self.counts = [None] * len(names)


MEMBERS = (
    "temperatures",
    "top_ks",
    "top_ps",
    "draft_temperatures",
    "draft_greedy_mask",
    "draft_sample_seed",
)


class FakeProposer:
    """`stage_draft_sampling_host`'s world: the scheduler's raw per-request
    temperatures on the host, the draft's own pinned copy of them, the greedy
    mask the block head reads, and the publication they all ride."""

    stage_draft_sampling_host = stage_draft_sampling_host

    def __init__(self, temperatures, has_filter, max_bs=8):
        bs = len(temperatures)
        self.draft_sampling = object()  # merely "armed"; the buffers are below
        # Seeded exactly as `_init_lossless_sampling` seeds them: 1.0 and 1, so
        # the fabricated tail rows never divide by zero in the block head.
        self._draft_temps_buf = FakeBuffer(max_bs, np.float32, 1.0)
        self._draft_greedy_buf = FakeBuffer(max_bs, np.int32, 1)
        self._draft_seed_buf = FakeBuffer(1, np.int32, 0)
        self.runner = type("Runner", (), {"sampling_has_row_filter": has_filter})()
        self._draft_sample_step = 0
        self._staged_draft_bs = -1
        self.bs = bs
        self.group = FakeGroup(MEMBERS)
        # The RAW temperatures, pre-clamp -- which is what `prepare_sample`
        # holds and what a greedy request actually carries.
        self.batch = type(
            "Batch", (), {"temperatures": np.asarray(temperatures, dtype=np.float32)}
        )()

    def stage(self):
        self.stage_draft_sampling_host(self.batch, self.bs, self.group)

    @property
    def temps(self):
        return torch.from_numpy(self._draft_temps_buf.np)

    @property
    def mask(self):
        return torch.from_numpy(self._draft_greedy_buf.np)


# Mixed temperatures, so "the whole batch went greedy" is distinguishable from
# "the rows that were already greedy stayed greedy". 0.0 is what a greedy
# request really carries before `prepare_sample` clamps it.
TEMPS = [1.0, 0.0, 0.7]


def test_a_filtered_batch_drafts_greedily_and_publishes_a_delta_q():
    p = FakeProposer(TEMPS, has_filter=True)
    p.stage()
    # Every row, not only the filtered ones: the verify-side drop is batch-wide,
    # so a row left sampling would be the sampled-draft/exact-match pair that is
    # worse than flag-off.
    torch.testing.assert_close(
        p.mask[: p.bs], torch.ones(p.bs, dtype=torch.int32), rtol=0, atol=0
    )
    # The temperature too, which is what makes the published q a delta rather
    # than merely making the id an argmax.
    torch.testing.assert_close(
        p.temps[: p.bs], torch.full((p.bs,), SAMPLER_EPS), rtol=0, atol=0
    )


def test_an_unfiltered_batch_keeps_each_requests_own_temperature():
    """The control, and the contract the ratio rule rests on: the draft divides
    by the number the target's verify softmax divides by, row for row -- which
    means the SAME `max(t, SAMPLER_EPS)` clamp the target's copy gets."""
    p = FakeProposer(TEMPS, has_filter=False)
    p.stage()
    torch.testing.assert_close(
        p.temps[: p.bs],
        torch.tensor([1.0, SAMPLER_EPS, 0.7], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        p.mask[: p.bs], torch.tensor([0, 1, 0], dtype=torch.int32), rtol=0, atol=0
    )


def test_the_greedy_predicate_is_the_samplers_own():
    """`<= SAMPLER_EPS`, not `== 0`: `Sampler` tests the clamped value against
    the same bound, so a row cannot be greedy to the target and stochastic to
    the draft. Pre- or post-clamp is the same set -- `max(t, EPS) <= EPS` iff
    `t <= EPS` -- and a temperature just above it must stay stochastic."""
    p = FakeProposer([SAMPLER_EPS, SAMPLER_EPS * 2, 0.0], has_filter=False)
    p.stage()
    torch.testing.assert_close(
        p.mask[: p.bs], torch.tensor([1, 0, 1], dtype=torch.int32), rtol=0, atol=0
    )


def test_the_whole_step_rides_one_publication():
    """The saving: three more members in the batched `sampling` H2D instead of
    four device launches (`copy_`, compare, cast, `copy_`, `fill_`) on `[bs]`
    tensors. Counts are what put a member in that upload, so an unclaimed one
    would silently ship nothing."""
    p = FakeProposer(TEMPS, has_filter=False)
    p.stage()
    assert p.group.counts[p.group.indices["draft_temperatures"]] == p.bs
    assert p.group.counts[p.group.indices["draft_greedy_mask"]] == p.bs
    assert p.group.counts[p.group.indices["draft_sample_seed"]] == 1
    # And the step's own sampling members are not disturbed by it.
    assert p.group.counts[p.group.indices["temperatures"]] is None


def test_the_philox_key_advances_once_per_step():
    """A stale seed would make two consecutive steps draft from the SAME
    uniform stream. It is a plain SPMD counter so every TP rank reaches the
    same number without exchanging it."""
    p = FakeProposer(TEMPS, has_filter=False)
    seen = []
    for _ in range(3):
        p.stage()
        seen.append(int(p._draft_seed_buf.np[0]))
    assert seen == [0, 1, 2]


def test_an_empty_batch_stages_nothing():
    """A 0-row batch has nothing to draft; claiming a zero-length member would
    put an empty source in the upload."""
    p = FakeProposer([], has_filter=False)
    p.stage()
    assert p.group.counts[p.group.indices["draft_temperatures"]] is None
    assert p.group.counts[p.group.indices["draft_greedy_mask"]] is None
    # The key still advances: the counter is per STEP, not per drafted row.
    assert p.group.counts[p.group.indices["draft_sample_seed"]] == 1


def test_the_tail_rows_past_the_scheduled_batch_are_left_alone():
    """`stage_draft_sampling_host` is sized by the SCHEDULED batch; the slots
    above it are fabricated padding and a 0 temperature there divides by zero
    inside the block head."""
    p = FakeProposer(TEMPS, has_filter=True)
    p.stage()
    torch.testing.assert_close(
        p.temps[p.bs :], torch.ones(p.temps.numel() - p.bs), rtol=0, atol=0
    )
    torch.testing.assert_close(
        p.mask[p.bs :],
        torch.ones(p.mask.numel() - p.bs, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
