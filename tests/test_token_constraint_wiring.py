# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The ModelRunner call sites of the token-constraint boundary.

The runner is imported against AITER/Triton stand-ins on a CPU job (the same
way as the other runner wiring tests), so none of this needs a GPU. Each test
pins one place where the runner hands a step to the boundary: preparation,
sampler-bypass eligibility, error delivery, release, and the RapidServe prefill
path.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.model_engine import token_constraints as boundary

mod = None  # the runner module, bound by `_runner_module` for the whole file


@pytest.fixture(scope="module", autouse=True)
def _runner_module():
    global mod
    with atom_without_aiter() as load:
        mod = load("atom.model_engine.model_runner")
        yield mod


@pytest.fixture(autouse=True)
def isolated_topology(monkeypatch):
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mod, "get_tp_group", lambda: SimpleNamespace(world_size=1))


def _batch(**rule):
    """A batch of two requests; `structured_requests` / `required_channel` set
    the rule it carries."""
    fields = {
        "total_seqs_num": 2,
        "total_tokens_num": 2,
        "req_ids": [10, 11],
        "return_logprobs": [False, False],
        "any_logprobs": False,
        "num_spec_step": 0,
        "structured_requests": None,
        "required_channel": None,
    }
    return SimpleNamespace(**{**fields, **rule})


GRAMMAR = {"structured_requests": {0: object()}}
CHANNEL = {"required_channel": object()}


# -- release -----------------------------------------------------------------


def test_release_reaches_the_grammar_through_the_runner():
    grammar = mock.Mock()
    runner = object.__new__(mod.ModelRunner)
    runner.structured_output = grammar
    mod.ModelRunner.release_constraints(runner, [3, 4])
    grammar.release.assert_called_once_with([3, 4])


# -- runner call sites -------------------------------------------------------


class _Reached(Exception):
    pass


def _forward_runner():
    runner = mock.MagicMock()
    runner.profiler = None
    runner.prepare_model.return_value = (torch.zeros(1), None, None, None, True, False)
    return runner


def test_forward_prepares_constraints_after_staging_and_before_the_model_runs():
    events = []
    batch = _batch(**GRAMMAR)
    metadata = object()
    runner = _forward_runner()
    runner._mark_staging_h2d_enqueued.side_effect = lambda: events.append("staged")
    runner.structured_output.prepare.side_effect = lambda *a: events.append("prepare")

    def run_model(input_ids, batch):
        events.append("run_model")
        raise _Reached

    runner.run_model.side_effect = run_model
    with (
        mock.patch.object(
            mod,
            "get_forward_context",
            lambda: SimpleNamespace(spec_decode_metadata=metadata),
        ),
        pytest.raises(_Reached),
    ):
        mod.ModelRunner.forward(runner, batch)
    assert events == ["staged", "prepare", "run_model"]
    runner.structured_output.prepare.assert_called_once_with(batch, metadata)


@pytest.mark.parametrize("rule", [{}, CHANNEL])
def test_forward_without_a_grammar_prepares_nothing(rule):
    runner = _forward_runner()
    runner.run_model.side_effect = _Reached
    with (
        mock.patch.object(
            mod,
            "get_forward_context",
            lambda: SimpleNamespace(spec_decode_metadata=None),
        ),
        pytest.raises(_Reached),
    ):
        mod.ModelRunner.forward(runner, _batch(**rule))
    runner.structured_output.prepare.assert_not_called()


def _bypass_runner(monkeypatch):
    runner = object.__new__(mod.ModelRunner)
    runner.rejection_sampler = object()
    spec = SimpleNamespace(num_spec_steps=1, num_draft_tokens_np=np.array([1, 1]))
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(spec_decode_metadata=spec)
    )
    monkeypatch.setattr(mod, "RELAXED_TOP_N", 1)
    tokens = torch.zeros(2, 2, dtype=torch.int32)
    verdict = torch.zeros(3, 2, dtype=torch.int32)
    return runner, (tokens, verdict)


@pytest.mark.parametrize(
    "rule,served", [({}, True), (GRAMMAR, False), (CHANNEL, False)]
)
def test_head_ids_serve_only_unconstrained_verifies(monkeypatch, rule, served):
    runner, head = _bypass_runner(monkeypatch)
    batch = _batch(temperatures=torch.zeros(2), top_ks=torch.full((2,), -1), **rule)
    assert runner._head_ids_serve(head, batch) is served


@pytest.mark.parametrize(
    "rule,greedy,logprobs,served",
    [
        ({}, True, False, True),
        (GRAMMAR, True, False, False),
        (CHANNEL, True, False, False),
        ({}, False, False, False),
        ({}, True, True, False),
    ],
)
def test_dp_argmax_ids_serve_only_unconstrained_greedy_steps(
    rule, greedy, logprobs, served
):
    runner = object.__new__(mod.ModelRunner)
    logits = torch.eye(4)[:2]
    local_logits = mock.Mock(return_value=logits)
    runner.dp_argmax_head = SimpleNamespace(local_logits=local_logits)
    batch = _batch(
        temperatures=np.full(2, 0.0 if greedy else 0.7), any_logprobs=logprobs, **rule
    )
    ids = torch.tensor([3.0, 1.0])
    hidden = torch.zeros(3, 4)
    output = runner._step_head(ids, hidden, batch)
    if served:
        assert output.dtype == torch.int32
        torch.testing.assert_close(output, ids.to(torch.int32))
        local_logits.assert_not_called()
    else:
        assert output is logits
        local_logits.assert_called_once()
        torch.testing.assert_close(local_logits.call_args.args[0], hidden[:2])


# -- error delivery ----------------------------------------------------------


def _postprocess_runner(ids, grammar):
    runner = object.__new__(mod.ModelRunner)
    runner._served = None
    runner.structured_output = grammar
    runner.sampler = lambda logits, *a, **k: logits.argmax(-1)
    runner.forward_done_event = SimpleNamespace(record=lambda stream=None: None)

    def prepare_sampled_ids(batch, sampled, event, payload, **kwargs):
        return dict(ids), None

    runner.tokenID_processor = SimpleNamespace(
        is_deferred_out=False,
        prev_batch=None,
        prev_rejected_num=None,
        prev_bonus_num=None,
        zero_spec_counts=torch.zeros(8, dtype=torch.int32),
        prepare_sampled_ids=prepare_sampled_ids,
    )
    return runner


def _postprocess(runner, batch):
    mod_fc = SimpleNamespace(spec_decode_metadata=None)
    with mock.patch.object(mod, "get_forward_context", lambda: mod_fc):
        return runner.postprocess(
            batch, torch.eye(4)[:2], torch.ones(2), None, None, True, None
        )


def test_dp_argmax_postprocess_skips_sampler_without_constraints(monkeypatch):
    runner = _postprocess_runner({10: (3,), 11: (1,)}, None)
    runner.dp_argmax_head = object()
    runner.sampler = mock.Mock(side_effect=AssertionError("ids must skip the sampler"))
    prepare = mock.Mock(return_value=({10: (3,), 11: (1,)}, None))
    runner.tokenID_processor.prepare_sampled_ids = prepare
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(spec_decode_metadata=None)
    )
    output = runner.postprocess(
        _batch(),
        torch.tensor([3, 1, 2], dtype=torch.int32),
        torch.zeros(2),
        None,
        None,
        True,
        None,
    )
    runner.sampler.assert_not_called()
    torch.testing.assert_close(
        prepare.call_args.args[1], torch.tensor([3, 1], dtype=torch.int32)
    )
    assert output.token_ids == [(3,), (1,)]


@pytest.mark.parametrize("rule", [GRAMMAR, CHANNEL])
def test_dp_argmax_fallback_applies_constraints_once_before_sampling(monkeypatch, rule):
    masks = mock.Mock(side_effect=lambda logits: logits[:, 3].fill_(-float("inf")))

    def grammar_mask(logits, batch, metadata):
        masks(logits)
        return {}

    def channel_mask(runner, batch, logits, metadata, target):
        masks(logits)

    grammar = SimpleNamespace(vocab_size=4, errors={}, apply=grammar_mask)
    runner = _postprocess_runner({10: (2,), 11: (2,)}, grammar)
    logits = torch.tensor([[0.0, 1.0, 2.0, 9.0], [0.0, 1.0, 2.0, 9.0]])
    runner.dp_argmax_head = SimpleNamespace(local_logits=mock.Mock(return_value=logits))
    batch = _batch(temperatures=np.zeros(2), **rule)
    head = runner._step_head(torch.tensor([3.0, 3.0]), torch.zeros(2, 4), batch)
    monkeypatch.setattr(boundary, "apply_required_channel", channel_mask)
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(spec_decode_metadata=None)
    )

    def sample(masked, *args, **kwargs):
        masks.assert_called_once()
        assert torch.isneginf(masked[:, 3]).all()
        return masked.argmax(-1)

    runner.sampler = mock.Mock(side_effect=sample)
    output = runner.postprocess(batch, head, torch.zeros(2), None, None, True, None)
    runner.dp_argmax_head.local_logits.assert_called_once()
    runner.sampler.assert_called_once()
    masks.assert_called_once()
    assert output.token_ids == [(2,), (2,)]


def test_grammar_errors_are_delivered_with_the_step_output():
    grammar = SimpleNamespace(errors={10: "grammar said no"})
    runner = _postprocess_runner({10: (4,), 11: (5,)}, grammar)
    out = _postprocess(runner, _batch())
    assert out.constraint_errors == {10: "grammar said no"}
    assert out.token_ids == [(4,), (5,)]


def test_no_grammar_means_no_errors():
    assert boundary.constraint_errors(SimpleNamespace(structured_output=None)) is None
    runner = _postprocess_runner({10: (4,), 11: (5,)}, None)
    assert not _postprocess(runner, _batch()).constraint_errors


# -- RapidServe prefill ------------------------------------------------------


def _rapid_runner(monkeypatch, logits):
    runner = object.__new__(mod.RapidServeModelRunner)
    stream = SimpleNamespace(synchronize=lambda: None)
    runner._prefill_streams = {0: stream}
    runner._advance_forward_vars = lambda: None
    runner._gate_staging_reuse = lambda: None
    runner._mark_staging_h2d_enqueued = lambda: None
    runner._record_forward_vars_event = lambda: None
    runner._record_kv_cache_ready = lambda batch: None
    runner.prepare_model = lambda batch: (
        torch.zeros(1),
        torch.ones(logits.shape[0]),
        None,
        None,
        True,
        False,
    )
    runner.run_model = lambda ids, batch: (logits, None)
    runner.sampler = lambda lg, temperatures, top_ks, top_ps, greedy: lg.argmax(-1)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(mod, "reset_forward_context", lambda: None)
    return runner


def _rapid_batch(**rule):
    return _batch(cu_stream_fraction=0, **rule)


def test_rapidserve_prefill_applies_the_channel_before_sampling(monkeypatch):
    def forbid_top(runner, batch, logits):
        logits[:, 3] = -float("inf")

    monkeypatch.setattr(mod, "apply_required_channel", forbid_top)
    logits = torch.tensor([[0.0, 1.0, 2.0, 9.0], [9.0, 1.0, 2.0, 3.0]])
    runner = _rapid_runner(monkeypatch, logits)
    assert runner.prefill_forward(_rapid_batch(**CHANNEL)) == [2, 0]


def test_rapidserve_prefill_without_a_channel_samples_raw_logits(monkeypatch):
    logits = torch.tensor([[0.0, 1.0, 2.0, 9.0], [9.0, 1.0, 2.0, 3.0]])
    runner = _rapid_runner(monkeypatch, logits)
    assert runner.prefill_forward(_rapid_batch()) == [3, 0]


# -- boundary ----------------------------------------------------------------


def test_grammar_apply_requires_the_logits_vocabulary():
    runner = SimpleNamespace(structured_output=SimpleNamespace(vocab_size=7))
    with pytest.raises(ValueError, match="do not match the prepared"):
        boundary.TokenConstraints.apply(
            runner, _batch(**GRAMMAR), torch.zeros(2, 8), None
        )


def _core_manager(prepare=None):
    from atom.model_engine.engine_core_mgr import CoreManager

    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "test"
    mgr.structured_request_prepare = prepare
    mgr.broadcast_utility_command = mock.Mock()
    mgr._seq_id_to_callback = {}
    return mgr


def test_failed_admission_releases_structured_requests():
    mgr = _core_manager(prepare=lambda seqs: None)
    mgr._add_prepared_requests = mock.Mock(side_effect=RuntimeError("admission"))
    structured = SimpleNamespace(id=1, json_schema="{}", structured_tool_format=None)
    plain = SimpleNamespace(id=2, json_schema=None, structured_tool_format=None)
    with pytest.raises(RuntimeError, match="admission"):
        mgr.add_request([structured, plain])
    mgr.broadcast_utility_command.assert_called_once_with(
        "release_constraints", request_ids=[1]
    )


def test_unstructured_requests_never_send_a_release():
    mgr = _core_manager()
    mgr._add_prepared_requests = mock.Mock(side_effect=RuntimeError("admission"))
    seq = SimpleNamespace(id=1, json_schema=None, structured_tool_format=None)
    with pytest.raises(RuntimeError):
        mgr.add_request([seq])
    mgr.broadcast_utility_command.assert_not_called()


def test_has_constraints_tolerates_a_missing_batch():
    assert boundary.has_constraints(None) is False
