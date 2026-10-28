# SPDX-License-Identifier: MIT
"""The runner's DSpark entry point: what each step hook does with the online
components, and which of them it leaves alone when they are off."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from atom.spec_decode.dspark_controller import DPPlan, DSparkController


class STS:
    def __init__(self):
        self.calls = []

    def observe(self, *args):
        self.calls.append(args)


def _controller(**parts):
    verify = NS(runtime=None, sts=None, dp=None, calibration_profile=None)
    for k, v in parts.items():
        setattr(verify, k, v)
    return DSparkController(NS(), verify)


def _prev(n=2, dummy=False, prefill=0):
    return NS(
        is_dummy_run=dummy,
        total_tokens_num_prefill=prefill,
        total_seqs_num=n,
        req_ids=[10, 11, 12][:n],
        num_scheduled_tokens=np.array([4, 2, 6][:n]),
    )


def test_no_runner_drafter_means_no_controller():
    assert DSparkController.of(NS()) is None
    assert DSparkController.of(NS(drafter=NS(verify_scheduler=None))) is None


def test_hooks_are_inert_when_every_online_part_is_off():
    ctl = _controller()
    assert ctl.dp_plan(NS()) is None
    ctl.forward_begin(NS(), NS())
    ctl.target_done()
    ctl.forward_end()
    assert ctl.observe_verdict(_prev(), np.array([1, 0])) is False
    ctl.observe_no_verdict()


def test_a_decode_verdict_labels_what_was_actually_verified():
    sts = STS()
    ctl = _controller(sts=sts)
    assert ctl.observe_verdict(_prev(), np.array([3, 1])) is True
    ids, accepted, verified = sts.calls[0]
    assert ids == [10, 11] and accepted.tolist() == [3, 1]
    assert verified.tolist() == [3, 1]  # scheduled rows minus the anchor


@pytest.mark.parametrize(
    "prev", [None, _prev(dummy=True), _prev(prefill=5), _prev(n=1)]
)
def test_a_forward_without_a_usable_verdict_still_counts(prev):
    sts = STS()
    ctl = _controller(sts=sts)
    assert ctl.observe_verdict(prev, np.array([1, 0])) is True
    assert sts.calls == [()]
    ctl.observe_no_verdict()
    assert sts.calls == [(), ()]


def test_timing_brackets_the_forward():
    events = []
    runtime = NS(
        begin=lambda batch, mode: events.append("begin") or "t",
        target_done=lambda t: events.append(("target", t)),
        finish=lambda t: events.append(("end", t)),
    )
    ctl = _controller(runtime=runtime)
    ctl.forward_begin(NS(), NS())
    ctl.target_done()
    ctl.forward_end()
    assert events == ["begin", ("target", "t"), ("end", "t")]
    assert ctl._timing is None


def _plan_batch():
    return NS(
        scheduled_tokens="sched",
        num_scheduled_tokens=np.array([6, 6]),
        total_tokens_num=12,
        total_tokens_num_decode=12,
        total_seqs_num_decode=2,
        num_spec_step=5,
    )


def test_a_plan_rewrites_from_the_scheduled_layout():
    batch = _plan_batch()
    planner = NS(
        prepare=lambda b: np.array([1.0]),
        resolve=lambda gathered, fallback: (3, np.array([5, 9]), {1: 2}),
        comm=NS(dp_rank=0),
    )
    rewrites = []

    def ragged(b, bs, full_q, lengths):
        rewrites.append((b.total_tokens_num, bs, full_q, lengths))
        b.total_tokens_num = 5

    plan = DPPlan(NS(_dspark_apply_ragged=ragged), planner, batch)
    assert plan.payload.tolist() == [1.0]
    batch.total_tokens_num = 8  # the per-rank fallback shrank it meanwhile
    sync = NS(scheduled_tokens_per_dp_rank=np.array([8, 9]), max_seqlen_q_across_dp=6)
    q, tokens = plan.apply(np.zeros((2, 1)), sync)
    assert (q, tokens.tolist()) == (3, [5, 9])
    assert rewrites == [(12, 2, 6, {1: 2})]  # restored before the rewrite


def test_a_plan_that_disagrees_with_its_layout_is_fatal():
    planner = NS(
        prepare=lambda b: np.zeros(1),
        resolve=lambda gathered, fallback: (3, np.array([7, 9]), {}),
        comm=NS(dp_rank=0),
    )
    plan = DPPlan(NS(_dspark_apply_ragged=lambda *a: None), planner, _plan_batch())
    sync = NS(scheduled_tokens_per_dp_rank=np.array([12, 9]), max_seqlen_q_across_dp=6)
    with pytest.raises(RuntimeError, match="expects 7 tokens"):
        plan.apply(np.zeros((2, 1)), sync)


def test_a_declined_plan_leaves_the_batch_alone():
    batch = _plan_batch()
    planner = NS(
        prepare=lambda b: np.zeros(1),
        resolve=lambda gathered, fallback: None,
        comm=NS(dp_rank=0),
    )
    plan = DPPlan(NS(), planner, batch)
    batch.total_tokens_num = 8
    sync = NS(scheduled_tokens_per_dp_rank=np.array([8]), max_seqlen_q_across_dp=None)
    assert plan.apply(np.zeros((1, 1)), sync) is None
    assert batch.total_tokens_num == 8
