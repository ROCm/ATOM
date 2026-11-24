# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""``VerifyHead.samples_step``: the steps whose rows a verify head's token ids
stand for, in place of the logits, the sampler and the rejection sampler.

Pure Python against stand-ins: no Triton kernel runs."""

from types import SimpleNamespace

import numpy as np
import pytest
from aiter_stub import atom_without_aiter

ROWS = 12  # the steps' forward rows


@pytest.fixture(scope="module")
def verify_head():
    with atom_without_aiter() as load:
        yield load("atom.model_ops.verify_head")


@pytest.fixture
def exact(monkeypatch, verify_head):
    """Exact acceptance, no prefill context parallelism."""
    monkeypatch.setattr(verify_head, "RELAXED_TOP_N", 1)
    monkeypatch.setattr(verify_head, "get_pcp_world_size", lambda: 1)
    return verify_head


def _head(verify_head, samples=False, top_p_rows=0):
    """A head drawing temperature rows (``samples``), within top-p at up to
    ``top_p_rows`` forward rows."""

    class Head(verify_head.VerifyHead):
        def draws_top_p(self, rows):
            return rows <= top_p_rows

    head = Head()
    head.samples = samples
    return head


def _spec(*drafts, steps=5):
    return SimpleNamespace(num_spec_steps=steps, num_draft_tokens_np=np.asarray(drafts))


def _batch(temps=(0.0,), **over):
    n = len(temps)
    b = {
        "total_seqs_num": n,
        "temperatures": np.asarray(temps, dtype=np.float32),
        "top_ks": np.full(n, -1, dtype=np.int32),
        "top_ps": np.ones(n, dtype=np.float32),
        "any_logprobs": False,
        "structured_requests": {},
        "required_channel": None,
        "presence_penalties": None,
        "needs_independent_noise": np.zeros(n, dtype=bool),
    }
    b.update(over)
    return SimpleNamespace(**b)


def test_a_greedy_verify(exact):
    assert _head(exact).samples_step(_batch(), _spec(5), ROWS)


@pytest.mark.parametrize(
    "over",
    [
        {"any_logprobs": True},
        {"structured_requests": {0: object()}},
        {"required_channel": object()},
    ],
)
def test_a_step_that_reads_the_logits(exact, over):
    assert not _head(exact).samples_step(_batch(**over), _spec(5), ROWS)


def test_not_a_verify(exact):
    assert not _head(exact).samples_step(_batch(), None, ROWS)


@pytest.mark.parametrize("why", ["relaxed", "pcp"])
def test_a_verify_that_is_not_exact_match_on_one_rank(exact, monkeypatch, why):
    if why == "relaxed":
        monkeypatch.setattr(exact, "RELAXED_TOP_N", 10)
    else:
        monkeypatch.setattr(exact, "get_pcp_world_size", lambda: 2)
    assert not _head(exact).samples_step(_batch(), _spec(5), ROWS)


def test_every_request_verifies_the_full_span(exact):
    two = _batch((0.0, 0.0))
    assert _head(exact).samples_step(two, _spec(5, 5), ROWS)
    assert not _head(exact).samples_step(two, _spec(5, 3), ROWS)


def test_a_sampled_step_its_head_draws(exact):
    """Temperature alone on a head that samples; within top-p too where the
    head draws so at the step's width."""
    temps = (0.7, 0.5)
    spec = _spec(5, 5)
    assert _head(exact, samples=True).samples_step(_batch(temps), spec, ROWS)
    assert not _head(exact).samples_step(_batch(temps), spec, ROWS)
    top_p = _batch(temps, top_ps=np.asarray([0.9, 1.0], dtype=np.float32))
    assert not _head(exact, samples=True).samples_step(top_p, spec, ROWS)
    head = _head(exact, samples=True, top_p_rows=ROWS)
    assert head.samples_step(top_p, spec, ROWS)
    # a wider step than it draws top-p at: the logits
    assert not head.samples_step(top_p, spec, ROWS + 1)


@pytest.mark.parametrize(
    "over",
    [
        {"top_ks": np.full(1, 20, dtype=np.int32)},
        {"needs_independent_noise": np.ones(1, dtype=bool)},
        {"temperatures": np.asarray([1e-12], dtype=np.float32)},
    ],
)
def test_a_sampled_step_no_head_draws(exact, over):
    head = _head(exact, samples=True, top_p_rows=ROWS)
    assert not head.samples_step(_batch((0.7,), **over), _spec(5), ROWS)


def test_a_step_of_greedy_and_sampled_requests(exact):
    head = _head(exact, samples=True, top_p_rows=ROWS)
    assert not head.samples_step(_batch((0.0, 0.7)), _spec(5, 5), ROWS)


@pytest.mark.parametrize("steps, ok", [(5, True), (4, False)])
def test_the_verify_reads_drafts_at_the_steps_stride(
    monkeypatch, verify_head, steps, ok
):
    """``IdVerifiers``' request stride is the speculative config's; a step of
    another span would read its drafts at the wrong stride."""
    launched = []
    verifiers = SimpleNamespace(
        rq=6, of=lambda rows: SimpleNamespace(launch=lambda *a: launched.append(a))
    )
    picks = verify_head.ShardedPicks.__new__(verify_head.ShardedPicks)
    picks.verifiers, picks.input_ids = verifiers, lambda: "ids"
    picks.row_ids = np.zeros(ROWS)
    spec = SimpleNamespace(num_spec_steps=steps)
    monkeypatch.setattr(
        verify_head,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    if ok:
        picks.verify(ROWS)
        assert len(launched) == 1
    else:
        with pytest.raises(AssertionError):
            picks.verify(ROWS)
