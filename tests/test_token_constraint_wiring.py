# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Constraint registry guards and the ModelRunner call sites that use them.

The runner is imported against AITER/Triton stand-ins on a CPU job (the same
way as the other runner wiring tests), so none of this needs a GPU. Adapters
here are small registered fakes; the executor, registry and runner methods are
the real ones. Each test pins one place where the runner hands a step to the
registry: preparation, sampler-bypass eligibility, error delivery, release, and
the RapidServe prefill path.
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
def isolated_registry(monkeypatch):
    """Each test sees an empty adapter registry."""
    monkeypatch.setattr(boundary, "_PROVIDERS", [])
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mod, "get_tp_group", lambda: SimpleNamespace(world_size=1))


class FakeAdapter:
    """A registered adapter; `batch.active` names the backends it applies to."""

    mask_stage = 1
    commit_order = 5

    def __init__(self, backend, stage=1, order=5, on_apply=None):
        self.backend = backend
        self.mask_stage = stage
        self.commit_order = order
        self.on_apply = on_apply
        self.calls = []
        self.commit_result = None

    def enabled(self, batch):
        return self.backend in getattr(batch, "active", ())

    def masks_source(self, metadata):
        return False

    def prepare(self, runner, batch, metadata):
        self.calls.append(("prepare", runner, batch, metadata))

    def errors(self, runner):
        return None

    def release(self, runner, request_ids):
        self.calls.append(("release", list(request_ids)))

    def apply(self, step):
        self.calls.append(("apply",))
        if self.on_apply is not None:
            self.on_apply(step)

    def commit(self, step, tokens, batch, deferred):
        self.calls.append(("commit", deferred))
        return self.commit_result


def register(adapter):
    boundary.register_constraint_provider(adapter)
    return adapter


def active(*backends, **extra):
    return SimpleNamespace(
        active=backends,
        total_seqs_num=2,
        total_tokens_num=2,
        total_seqs_num_prefill=0,
        total_seqs_num_decode=2,
        num_spec_step=0,
        structured_requests=None,
        required_channel=None,
        **extra,
    )


# -- registry guards ---------------------------------------------------------


def test_duplicate_backend_registration_is_rejected():
    register(FakeAdapter("dup"))
    with pytest.raises(ValueError, match="Duplicate constraint backend: dup"):
        register(FakeAdapter("dup"))
    assert [p.backend for p in boundary._PROVIDERS] == ["dup"]


def test_masks_run_by_stage_and_commits_by_commit_order():
    events = []
    first = register(FakeAdapter("late_mask", stage=2, order=1))
    second = register(FakeAdapter("early_mask", stage=0, order=9))
    for adapter in (first, second):
        adapter.on_apply = lambda step, a=adapter: events.append(("apply", a.backend))
        adapter.commit = lambda *args, a=adapter: events.append(("commit", a.backend))
    batch = active("late_mask", "early_mask")
    step = boundary.TokenConstraints.apply(None, batch, torch.zeros(2, 4), None)
    step.commit(torch.tensor([1, 2]), batch, False)
    assert events == [
        ("apply", "early_mask"),
        ("apply", "late_mask"),
        ("commit", "late_mask"),
        ("commit", "early_mask"),
    ]


def test_builtin_channel_advance_commits_before_grammar():
    assert boundary.ChannelProvider.commit_order < boundary.GrammarProvider.commit_order


def test_only_one_adapter_may_return_gpu_carry():
    one = register(FakeAdapter("one", order=1))
    two = register(FakeAdapter("two", order=2))
    one.commit_result = torch.tensor([1])
    batch = active("one", "two")
    step = boundary.TokenConstraints.apply(None, batch, torch.zeros(2, 4), None)
    carry = step.commit(torch.tensor([1, 2]), batch, True)
    assert carry is one.commit_result
    two.commit_result = torch.tensor([2])
    with pytest.raises(ValueError, match="Multiple constraint adapters"):
        step.commit(torch.tensor([1, 2]), batch, True)


# -- release -----------------------------------------------------------------


def test_release_reaches_the_grammar_through_the_runner():
    grammar = mock.Mock()
    boundary.register_constraint_provider(boundary.GrammarProvider)
    runner = object.__new__(mod.ModelRunner)
    runner.structured_output = grammar
    mod.ModelRunner.release_constraints(runner, [3, 4])
    grammar.release.assert_called_once_with([3, 4])


def test_release_continues_after_one_adapter_fails():
    grammar = mock.Mock()
    runner = SimpleNamespace(structured_output=grammar)
    third = register(FakeAdapter("third", stage=-1))

    def fail(runner, request_ids):
        raise RuntimeError("third adapter release failed")

    third.release = fail
    after = register(FakeAdapter("after", stage=5))
    boundary.register_constraint_provider(boundary.GrammarProvider)
    with pytest.raises(RuntimeError, match="third adapter release failed"):
        boundary.release_token_constraints(runner, [7])
    grammar.release.assert_called_once_with([7])
    assert ("release", [7]) in after.calls


def test_release_reports_the_first_failure_and_still_releases_everything():
    first = register(FakeAdapter("first", stage=0))
    second = register(FakeAdapter("second", stage=1))
    last = register(FakeAdapter("last", stage=2))

    def failing(name):
        def release(runner, request_ids):
            raise RuntimeError(name)

        return release

    first.release, second.release = failing("first"), failing("second")
    with pytest.raises(RuntimeError, match="first"):
        boundary.release_token_constraints(None, [1])
    assert ("release", [1]) in last.calls


# -- runner call sites -------------------------------------------------------


class _Reached(Exception):
    pass


def test_forward_prepares_constraints_after_staging_and_before_the_model_runs():
    adapter = register(FakeAdapter("prep"))
    events = []
    batch = active("prep")
    metadata = object()
    runner = mock.MagicMock()
    runner.profiler = None
    runner.prepare_model.return_value = (torch.zeros(1), None, None, None, True, False)
    runner._mark_staging_h2d_enqueued.side_effect = lambda: events.append("staged")
    adapter.prepare = lambda *args: events.append("prepare") or adapter.calls.append(
        args
    )

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
    (call,) = adapter.calls
    assert call[:2] == (runner, batch) and call[2] is metadata
    assert len(call) == 3


def test_forward_without_constraints_prepares_nothing():
    adapter = register(FakeAdapter("idle"))
    runner = mock.MagicMock()
    runner.profiler = None
    runner.prepare_model.return_value = (torch.zeros(1), None, None, None, True, False)
    runner.run_model.side_effect = _Reached
    with (
        mock.patch.object(
            mod,
            "get_forward_context",
            lambda: SimpleNamespace(spec_decode_metadata=None),
        ),
        pytest.raises(_Reached),
    ):
        mod.ModelRunner.forward(runner, active())
    assert adapter.calls == []


def _bypass_runner(monkeypatch, batch):
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


def _plain_batch(*backends):
    return SimpleNamespace(
        active=backends,
        total_seqs_num=2,
        any_logprobs=False,
        temperatures=torch.zeros(2),
        top_ks=torch.full((2,), -1),
    )


def test_head_ids_serve_only_unconstrained_verifies(monkeypatch):
    register(FakeAdapter("c"))
    runner, head = _bypass_runner(monkeypatch, None)
    assert runner._head_ids_serve(head, _plain_batch()) is True
    assert runner._head_ids_serve(head, _plain_batch("c")) is False


# -- error delivery ----------------------------------------------------------


def _postprocess_runner(ids, deferred):
    runner = object.__new__(mod.ModelRunner)
    runner._served = None
    runner.sampler = lambda logits, *a, **k: logits.argmax(-1)
    runner.forward_done_event = SimpleNamespace(record=lambda stream=None: None)

    def prepare_sampled_ids(batch, sampled, event, payload, **kwargs):
        return dict(ids), None

    runner.tokenID_processor = SimpleNamespace(
        is_deferred_out=deferred,
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


def _quiet_batch(*backends):
    return SimpleNamespace(
        active=backends,
        total_seqs_num=2,
        total_tokens_num=2,
        req_ids=[10, 11],
        return_logprobs=[False, False],
        any_logprobs=False,
        required_channel=None,
        structured_requests=None,
        num_spec_step=0,
    )


def test_adapter_errors_are_delivered_with_the_step_output():
    adapter = register(FakeAdapter("c"))
    adapter.errors = lambda runner: {10: "adapter said no"}
    runner = _postprocess_runner({10: (4,), 11: (5,)}, deferred=False)
    out = _postprocess(runner, _quiet_batch("c"))
    assert out.constraint_errors == {10: "adapter said no"}
    assert out.token_ids == [(4,), (5,)]


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


def _rapid_batch(*backends):
    return SimpleNamespace(
        active=backends,
        total_seqs_num=2,
        cu_stream_fraction=0,
        required_channel=None,
        structured_requests=None,
    )


def test_rapidserve_prefill_applies_registered_masks_before_sampling(monkeypatch):
    def forbid_top(step):
        step.gather()
        step.logits[:, 3] = -float("inf")

    register(FakeAdapter("stateless", on_apply=forbid_top))
    logits = torch.tensor([[0.0, 1.0, 2.0, 9.0], [9.0, 1.0, 2.0, 3.0]])
    runner = _rapid_runner(monkeypatch, logits)
    assert runner.prefill_forward(_rapid_batch("stateless")) == [2, 0]


def test_rapidserve_prefill_without_constraints_samples_raw_logits(monkeypatch):
    register(FakeAdapter("stateless", on_apply=lambda s, b: pytest.fail("applied")))
    logits = torch.tensor([[0.0, 1.0, 2.0, 9.0], [9.0, 1.0, 2.0, 3.0]])
    runner = _rapid_runner(monkeypatch, logits)
    assert runner.prefill_forward(_rapid_batch()) == [3, 0]


# -- commit and delivery wiring -------------------------------------------------


def test_adapter_errors_from_every_adapter_are_merged():
    first = register(FakeAdapter("first"))
    second = register(FakeAdapter("second"))
    first.errors = lambda runner: {1: "one"}
    second.errors = lambda runner: {2: "two"}
    register(FakeAdapter("silent"))
    assert boundary.constraint_errors(None) == {1: "one", 2: "two"}


def test_no_adapter_errors_is_none():
    register(FakeAdapter("silent"))
    assert boundary.constraint_errors(None) is None


def test_grammar_apply_requires_the_logits_vocabulary():
    step = SimpleNamespace(
        runner=SimpleNamespace(structured_output=SimpleNamespace(vocab_size=7)),
        source=torch.zeros(1, 8),
        batch=None,
        metadata=None,
    )
    with pytest.raises(ValueError, match="do not match the prepared"):
        boundary.GrammarProvider.apply(step)


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
    register(FakeAdapter("silent"))
    mgr = _core_manager()
    mgr._add_prepared_requests = mock.Mock(side_effect=RuntimeError("admission"))
    seq = SimpleNamespace(id=1, json_schema=None, structured_tool_format=None)
    with pytest.raises(RuntimeError):
        mgr.add_request([seq])
    mgr.broadcast_utility_command.assert_not_called()


def test_has_constraints_tolerates_a_missing_batch():
    adapter = register(FakeAdapter("strict"))

    def enabled(batch):
        return "strict" in batch.active  # a third-party adapter may not guard None

    adapter.enabled = enabled
    assert boundary.has_constraints(None) is False
    assert boundary.has_constraints(active("strict")) is True
