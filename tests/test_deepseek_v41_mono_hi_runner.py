# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ModelRunner's model-side verify head (DeepSeek-V4.1's hi hook): which verify steps take the head's ids
instead of logits (``_verify_head_ids``), the sampling rows it is told (``_verify_head_sampling``), and
``postprocess`` on a head-served step (the model's verdict, no TP broadcast of the ids). A model without the hook
takes the logits path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


class _HeadModel:
    """The hook's surface as ``HiDecodeModel`` offers it."""

    samples_verify = True
    verify_head_forced = "sample"
    verify_ids_bcast = "skip"

    def __init__(self, ids):
        self.ids, self.sampling = ids, []

    @property
    def verify_head_ids(self):
        return lambda hidden_states=None: self.ids

    def set_verify_sampling(self, temps=None, top_ps=None):
        self.sampling.append(
            None if temps is None else (temps.tolist(), top_ps.tolist())
        )
        return True


def _runner(runner_module, model, synthetic=None):
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.model = model
    runner.rejection_sampler = SimpleNamespace(synthetic_acceptance_rates=synthetic)
    runner._head_sampled = False
    return runner


def _batch(temps=(0.0,), **over):
    n = len(temps)
    b = dict(
        temperatures=np.asarray(temps, dtype=np.float32),
        top_ks=np.full(n, -1, dtype=np.int32),
        top_ps=np.ones(n, dtype=np.float32),
        num_scheduled_tokens=np.full(n, 6, dtype=np.int32),
        any_logprobs=False,
        structured_requests={},
        required_channel=None,
        needs_independent_noise=np.zeros(n, dtype=bool),
    )
    b.update(over)
    return SimpleNamespace(**b)


@pytest.fixture
def spec_step(monkeypatch, runner_module):
    spec = SimpleNamespace(num_spec_steps=5)
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: 1)
    return spec


def test_a_model_without_the_hook_takes_the_logits_path(spec_step, runner_module):
    runner = _runner(runner_module, SimpleNamespace())
    runner._verify_head_sampling(_batch())
    assert runner._verify_head_ids(_batch(), torch.zeros(6, 4)) is None


def test_greedy_verify_takes_the_head_ids(spec_step, runner_module):
    ids = torch.arange(8, dtype=torch.int32)
    runner = _runner(runner_module, _HeadModel(ids))
    batch = _batch()
    runner._verify_head_sampling(batch)
    got = runner._verify_head_ids(batch, torch.zeros(6, 4))
    assert torch.equal(got, ids[:6])
    for over in (
        dict(any_logprobs=True),
        dict(structured_requests={0: object()}),
        dict(required_channel=object()),
    ):
        assert runner._verify_head_ids(_batch(**over), torch.zeros(6, 4)) is None


def test_a_replay_reads_the_captured_buffer(spec_step, runner_module):
    ids = torch.arange(12, dtype=torch.int32)
    runner = _runner(runner_module, _HeadModel(None))
    runner.graph_head_ids = {(1, 6): ids}
    got = runner._verify_head_ids(_batch(), torch.zeros(6, 4), (1, 6))
    assert torch.equal(got, ids[:6])
    assert runner._verify_head_ids(_batch(), torch.zeros(6, 4), (2, 6)) is None


def test_sampled_verify(spec_step, runner_module):
    model = _HeadModel(torch.zeros(6, dtype=torch.int32))
    runner = _runner(runner_module, model)
    batch = _batch((0.7,), top_ps=np.full(1, 0.9, dtype=np.float32))
    runner._verify_head_sampling(batch)
    assert runner._head_sampled
    (temps, top_ps), *_ = model.sampling
    assert temps == pytest.approx([0.7] * 6) and top_ps == pytest.approx([0.9] * 6)
    assert runner._verify_head_ids(batch, torch.zeros(6, 4)) is not None
    # top-k: not the head's draw (ATOM's sampler), and the head's rows go back to greedy
    topk = _batch((0.7,), top_ks=np.full(1, 20, dtype=np.int32))
    runner._verify_head_sampling(topk)
    assert not runner._head_sampled and model.sampling[-1] is None
    assert runner._verify_head_ids(topk, torch.zeros(6, 4)) is None


def test_forced_acceptance(spec_step, runner_module):
    """Under ``--spec-decode-acceptance-length`` the head serves greedy steps and the sampled steps it drew."""
    model = _HeadModel(torch.zeros(6, dtype=torch.int32))
    runner = _runner(runner_module, model, synthetic=(1.0, 1.0, 0.51, 0.0, 0.0))
    assert runner._head_forced_mode() == "sample"
    assert runner._verify_head_ids(_batch(), torch.zeros(6, 4)) is not None
    sampled = _batch((1.0,))
    runner._verify_head_sampling(sampled)
    assert runner._head_sampled
    assert runner._verify_head_ids(sampled, torch.zeros(6, 4)) is not None
    model.verify_head_forced = ""
    runner = _runner(runner_module, model, synthetic=(1.0,))
    assert runner._verify_head_ids(_batch(), torch.zeros(6, 4)) is None


def test_postprocess_takes_the_models_verdict_and_skips_the_broadcast(
    monkeypatch, runner_module
):
    bs = 2
    spec = SimpleNamespace(
        bonus_logits_indices=torch.tensor([5, 11]),
        target_logits_indices=torch.arange(10),
    )
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    broadcasts = []
    monkeypatch.setattr(
        runner_module,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=4, broadcast=lambda t, src: broadcasts.append(t) or t
        ),
    )
    monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(runner_module, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(runner_module, "sp_is_enabled", lambda: False)
    out = torch.zeros(bs, 6, dtype=torch.int32)
    verdict = torch.tensor([[3, 5], [2, 0], [3, 11]], dtype=torch.int32)
    nxt = torch.tensor([17, 19], dtype=torch.int32)
    calls = []

    class Stop(Exception):
        pass

    model = _HeadModel(None)
    model.verify_verdict = lambda s, ids, sampler: calls.append(ids) or (
        out,
        verdict,
        nxt,
    )

    def propose(batch, input_ids, hidden, next_token_ids, num_reject, anchors):
        assert next_token_ids is nxt and torch.equal(anchors, verdict[2])
        raise Stop

    runner = _runner(runner_module, model)
    runner.verify_token_ids = torch.arange(12, dtype=torch.int32)
    runner.tokenID_processor = SimpleNamespace(
        is_deferred_out=True,
        prev_batch=None,
        prev_rejected_num=None,
        prev_bonus_num=None,
        prepare_sampled_ids=lambda *a, **kw: ({7: (3,), 8: (5,), -1: 1}, None),
        spec_status_queue=SimpleNamespace(send=lambda *a, **kw: None),
        input_ids=SimpleNamespace(gpu=torch.zeros(16, dtype=torch.int32)),
    )
    runner.forward_done_event = SimpleNamespace(record=lambda: None)
    runner.drafter = object()
    runner.structured_output = None
    runner.propose_draft_token_ids = propose
    batch = SimpleNamespace(
        total_seqs_num=bs,
        total_tokens_num=12,
        any_logprobs=False,
        structured_requests={},
        required_channel=None,
    )
    with pytest.raises(Stop):
        runner_module.ModelRunner.postprocess(
            runner, batch, None, None, None, None, True, hidden_states=None
        )
    assert len(calls) == 1 and broadcasts == []


def test_the_runner_calls_a_models_decode_round(runner_module, monkeypatch):
    """The model's ``decode_round`` (the hi hook's ``HiRound``) after the target's launch (``run_model``) and before
    the staging mark; a model without one runs none of it."""
    calls = []

    class Round:
        def after_forward(self, processor):
            calls.append(("after", processor))

        def before_staging_mark(self):
            calls.append("mark")

    monkeypatch.setattr(
        runner_module.ModelRunner, "_run_model", lambda self, ids, b=None: ("l", "h")
    )
    for model, want in (
        (SimpleNamespace(decode_round=Round()), [("after", "tp"), "mark", "finish"]),
        (SimpleNamespace(), ["finish"]),
    ):
        calls.clear()
        runner = _runner(runner_module, model)
        runner.tokenID_processor = "tp"
        runner.h2d_owner = SimpleNamespace(finish=lambda: calls.append("finish"))
        assert runner.run_model("ids") == ("l", "h")
        runner._mark_staging_h2d_enqueued()
        assert calls == want
    assert (
        runner_module.ModelRunner.__new__(runner_module.ModelRunner)._decode_round()
        is None
    )
