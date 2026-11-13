# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Constraint registry guards and the ModelRunner call sites that use them.

The runner is imported against AITER/Triton stand-ins on a CPU job (the same
way as the other runner wiring tests), so none of this needs a GPU. Adapters
here are small registered fakes; the executor, registry and runner methods are
the real ones. Each test pins one place where the runner hands a step to the
registry: preparation, step-override decline, sampler-bypass eligibility,
error delivery, release, and the RapidServe prefill path.
"""

import contextlib
import random
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

    def __init__(self, backend, hard=(), stage=1, order=5, on_apply=None):
        self.backend = backend
        self.mask_stage = stage
        self.commit_order = order
        self.hard = list(hard)
        self.on_apply = on_apply
        self.calls = []
        self.commit_result = None

    def enabled(self, batch):
        return self.backend in getattr(batch, "active", ())

    def masks_source(self, metadata):
        return False

    def hard_requests(self, batch):
        return list(self.hard)

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

    def commit(self, step, tokens, batch, deferred, failure):
        self.calls.append(("commit", failure, deferred))
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
    step.commit(torch.tensor([1, 2]), batch, False, None)
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
    carry = step.commit(torch.tensor([1, 2]), batch, True, None)
    assert carry is one.commit_result
    two.commit_result = torch.tensor([2])
    with pytest.raises(ValueError, match="Multiple constraint adapters"):
        step.commit(torch.tensor([1, 2]), batch, True, None)


def test_grammar_commit_receives_error_sentinels_for_failed_rows():
    grammar = mock.Mock()
    step = SimpleNamespace(prefixes={0: object()}, grammar=grammar)
    tokens = torch.tensor([5, 6, 7])
    failure = torch.tensor([False, True, False])
    boundary.GrammarProvider.commit(step, tokens, "batch", False, failure)
    sent = grammar.commit.call_args.args[0]
    assert sent.tolist() == [5, -2, 7]
    boundary.GrammarProvider.commit(step, tokens, "batch", False, None)
    assert grammar.commit.call_args.args[0] is tokens


# -- intersection guard ------------------------------------------------------


def _blank_rows(rows):
    def blank(step):
        step.gather()
        step.logits[list(rows)] = -float("inf")

    return blank


def test_empty_rows_fail_only_with_two_hard_rules():
    # request 0 has two hard rules and no token left: it must fail. Request 1
    # is empty under one rule only: that adapter owns that case, not the guard.
    register(FakeAdapter("a", hard=[0, 1], on_apply=_blank_rows([0, 1])))
    register(FakeAdapter("b", hard=[0]))
    logits = torch.zeros(2, 4)
    step = boundary.TokenConstraints.apply(None, active("a", "b"), logits, None)
    assert step.invalid.tolist() == [True, False]
    assert step.failed(torch.tensor([1, 2])).tolist() == [True, False]


def test_speculative_target_rows_follow_their_request_owner():
    # request 0 (1 draft) has one rule, request 1 (2 drafts) has two. Only
    # request 1's target rows may be flagged, in draft-count order.
    def blank_all(step):
        step.gather()
        step.logits[:] = -float("inf")
        step.target[:] = -float("inf")

    register(FakeAdapter("a", hard=[1], on_apply=blank_all))
    register(FakeAdapter("b", hard=[0, 1]))
    metadata = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([1, 3]),
        bonus_logits_indices=torch.tensor([1, 4]),
        target_logits_indices=torch.tensor([0, 2, 3]),
    )
    step = boundary.TokenConstraints.apply(
        None, active("a", "b"), torch.zeros(5, 4), metadata
    )
    assert step.target_invalid.tolist() == [False, True, True]
    assert step.invalid.tolist() == [False, True]


# -- failure positions -------------------------------------------------------


def _step(invalid, target_invalid, cu):
    step = boundary.TokenConstraints()
    step.logits = torch.zeros(len(cu), 8)
    step.invalid = None if invalid is None else torch.tensor(invalid)
    step.target_invalid = (
        None if target_invalid is None else torch.tensor(target_invalid)
    )
    step.metadata = SimpleNamespace(cu_num_draft_tokens=torch.tensor(cu))
    return step


@pytest.mark.parametrize(
    "sampled,expected",
    [
        # draft rejected, no bonus emitted: the bonus-row failure is unused
        ([[3, -1]], False),
        # draft accepted and the invalid bonus row's token emitted
        ([[3, 5]], True),
    ],
)
def test_bonus_only_failure_applies_to_the_bonus_position(sampled, expected):
    step = _step([True], None, [1])
    assert step.failed(torch.tensor(sampled)).tolist() == [expected]


@pytest.mark.parametrize(
    "sampled,expected",
    [
        ([[3, -1, -1]], False),
        ([[3, 4, -1]], False),  # second draft rejected: bonus slot unused
        ([[3, 4, 5]], True),
    ],
)
def test_bonus_only_failure_with_two_drafts(sampled, expected):
    step = _step([True], None, [2])
    assert step.failed(torch.tensor(sampled)).tolist() == [expected]


@pytest.mark.parametrize(
    "sampled,expected",
    [
        ([[7, -1, -1]], False),  # the invalid target row's token was rejected
        ([[7, 3, -1]], True),
        ([[7, 3, 5]], True),
    ],
)
def test_target_only_failure_applies_to_its_draft_position(sampled, expected):
    step = _step(None, [False, True], [2])
    assert step.failed(torch.tensor(sampled)).tolist() == [expected]


def test_target_and_bonus_failures_compose_per_position():
    # draft 0 fine, draft 1's target row invalid, bonus row invalid
    step = _step([True], [False, True], [2])
    assert step.failed(torch.tensor([[7, -1, -1]])).tolist() == [False]
    assert step.failed(torch.tensor([[7, 3, -1]])).tolist() == [True]
    assert step.failed(torch.tensor([[7, 4, 5]])).tolist() == [True]
    # bonus row alone invalid, drafts fine
    step = _step([True], [False, False], [2])
    assert step.failed(torch.tensor([[7, 4, -1]])).tolist() == [False]
    assert step.failed(torch.tensor([[7, 4, 5]])).tolist() == [True]


def test_bonus_only_adapter_through_the_real_executor():
    def fail_bonus(step):
        step.gather()
        step.invalid = torch.tensor([True])

    register(FakeAdapter("bonus_only", hard=[0], stage=0, on_apply=fail_bonus))
    metadata = SimpleNamespace(
        cu_num_draft_tokens=torch.tensor([1]),
        bonus_logits_indices=torch.tensor([1]),
        target_logits_indices=torch.tensor([0]),
    )
    step = boundary.TokenConstraints.apply(
        None, active("bonus_only"), torch.randn(2, 8), metadata
    )
    assert step.failed(torch.tensor([[3, -1]])).tolist() == [False]
    assert step.failed(torch.tensor([[3, 5]])).tolist() == [True]


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


def _decode_runner(monkeypatch, override):
    """A ModelRunner whose run_model takes the graph-decode branch."""
    runner = object.__new__(mod.ModelRunner)
    runner.sp_mega_moe = None
    runner.config = SimpleNamespace(prefill_context_parallel_size=1)
    runner._detailed_label_suffix = lambda batch: ""
    runner._tr_hybrid_active = lambda: False
    runner._step_override = override
    runner.model = mock.Mock()
    graph_logits = torch.zeros(2, 4)
    runner.req_graph = SimpleNamespace(replay=mock.Mock(return_value=graph_logits))
    runner.forward_vars = {"outputs": torch.zeros(2, 4)}
    runner._step_head = lambda head, hidden, batch: head
    runner.use_mrope = False
    sync = SimpleNamespace(prefill_across_dp=[False], scheduled_tokens_per_dp_rank=None)
    mode = SimpleNamespace(
        use_cudagraph=True,
        running_tokens=2,
        scheduled_tokens=2,
        pcp_groups=None,
        sync=sync,
        assert_shape_contract=lambda ids, meta: None,
    )
    context = SimpleNamespace(
        scheduled_bs=2,
        running_bs=2,
        scheduled_tokens=2,
        is_prefill=False,
        positions=None,
        forward_mode=mode,
    )
    fc = SimpleNamespace(
        context=context,
        attn_metadata=SimpleNamespace(max_seqlen_q=1),
        ubatch_slices=None,
    )
    monkeypatch.setattr(mod, "get_forward_context", lambda: fc)
    monkeypatch.setattr(mod, "request_bucket", lambda bs, q: (bs, q))
    return runner, graph_logits


def test_constrained_decode_declines_the_step_override(monkeypatch):
    register(FakeAdapter("c"))
    override = mock.Mock(return_value=object())
    runner, graph_logits = _decode_runner(monkeypatch, override)
    batch = active("c")
    logits, _ = runner.run_model(torch.zeros(2), batch)
    runner.model.decline_step.assert_called_once_with(batch)
    override.assert_not_called()
    assert runner._served is None
    assert logits.data_ptr() == graph_logits.data_ptr()
    runner.req_graph.replay.assert_called_once()


def test_unconstrained_decode_serves_the_step_override(monkeypatch):
    register(FakeAdapter("c"))
    served = object()
    override = mock.Mock(return_value=served)
    runner, _ = _decode_runner(monkeypatch, override)
    batch = active()
    assert runner.run_model(torch.zeros(2), batch) == (None, None)
    override.assert_called_once()
    assert runner._served is served
    runner.model.decline_step.assert_not_called()
    runner.req_graph.replay.assert_not_called()


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


def _postprocess_runner(ids, deferred, prev_batch=None):
    runner = object.__new__(mod.ModelRunner)
    runner._served = None
    runner.sampler = lambda logits, *a, **k: logits.argmax(-1)
    runner.forward_done_event = SimpleNamespace(record=lambda stream=None: None)
    seen = {}

    def prepare_sampled_ids(batch, sampled, event, payload, **kwargs):
        seen["failed"] = kwargs["constraint_failed"]
        seen["channel_states"] = kwargs["channel_states"]
        seen["sampled"] = sampled.clone()
        return dict(ids), None

    runner.tokenID_processor = SimpleNamespace(
        is_deferred_out=deferred,
        prev_batch=prev_batch,
        prev_rejected_num=None,
        prev_bonus_num=None,
        zero_spec_counts=torch.zeros(8, dtype=torch.int32),
        prepare_sampled_ids=prepare_sampled_ids,
    )
    return runner, seen


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


def test_failed_rows_are_delivered_as_errors_not_tokens():
    def fail_row_one(step):
        step.invalid = torch.tensor([False, True])

    register(FakeAdapter("c", hard=[0, 1], on_apply=fail_row_one))
    runner, seen = _postprocess_runner({10: (4,), 11: (-2,)}, deferred=False)
    batch = _quiet_batch("c")
    out = _postprocess(runner, batch)
    assert seen["failed"].tolist() == [False, True]
    assert out.token_ids == [(4,), ()]
    assert out.constraint_errors == {
        11: "Token constraints leave no valid continuation"
    }


def test_adapter_errors_are_delivered_with_the_step_output():
    adapter = register(FakeAdapter("c"))
    adapter.errors = lambda runner: {10: "adapter said no"}
    runner, _ = _postprocess_runner({10: (4,), 11: (5,)}, deferred=False)
    out = _postprocess(runner, _quiet_batch("c"))
    assert out.constraint_errors == {10: "adapter said no"}
    assert out.token_ids == [(4,), (5,)]


def test_deferred_error_delivery_follows_the_previous_batch():
    # The ids delivered by a deferred step are the previous step's output, so
    # the sentinel check keys off the previous batch, not the current one.
    register(FakeAdapter("c"))
    constrained, plain = _quiet_batch("c"), _quiet_batch()
    runner, _ = _postprocess_runner({10: (-2,), 11: (6,)}, True, prev_batch=constrained)
    out = _postprocess(runner, plain)
    assert out.token_ids == [(), (6,)]
    assert out.constraint_errors == {
        10: "Token constraints leave no valid continuation"
    }
    # A constrained current step must not turn an unconstrained previous
    # step's output into errors.
    runner, _ = _postprocess_runner({10: (-2,), 11: (6,)}, True, prev_batch=plain)
    out = _postprocess(runner, constrained)
    assert out.token_ids == [(-2,), (6,)]
    assert not out.constraint_errors


def test_unconstrained_step_never_rewrites_output():
    register(FakeAdapter("c"))
    runner, seen = _postprocess_runner({10: (-2,), 11: (6,)}, deferred=False)
    out = _postprocess(runner, _quiet_batch())
    assert seen["failed"] is None
    assert out.token_ids == [(-2,), (6,)]
    assert not out.constraint_errors


class _CudaIds(torch.Tensor):
    """CPU ids that look like a device tensor to the D2H bookkeeping."""

    is_cuda = True


@pytest.mark.parametrize("failed", [False, True])
def test_deferred_failure_masking_is_ordered_before_the_d2h_copy(failed):
    processor = object.__new__(mod.tokenIDProcessor)
    processor.is_deferred_out = True
    processor.prev_batch = None
    processor.token_ids_cpu = []
    events = []
    processor.recv_async_output = lambda queue: []
    processor.recv_logprobs = lambda: None
    processor.send_to_cpu_async = lambda ids, *a, **k: events.append(
        ("copy", ids.tolist())
    )
    sampled = torch.tensor([4, 5]).as_subclass(_CudaIds)
    event = SimpleNamespace(record=lambda: events.append(("record",)))
    processor.prepare_sampled_ids(
        SimpleNamespace(req_ids=[0, 1]),
        sampled,
        event,
        constraint_failed=torch.tensor([False, True]) if failed else None,
    )
    if failed:
        assert events == [("record",), ("copy", [4, -2])]
    else:
        assert events == [("copy", [4, 5])]


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


# -- failure positions over ragged drafts -------------------------------------


def _reference_failed(drafts, invalid, target_invalid, tokens):
    """Independent per-position oracle: draft position j of a request fails on
    its own target row, the bonus position (j == drafts) on its bonus row, and
    only emitted (non-negative) tokens count."""
    out, start = [], 0
    for i, count in enumerate(drafts):
        bad = False
        for j, token in enumerate(tokens[i]):
            if token < 0:
                continue
            if j < count:
                bad |= bool(target_invalid is not None and target_invalid[start + j])
            elif j == count:
                bad |= bool(invalid is not None and invalid[i])
        out.append(bad)
        start += count
    return out


def test_failed_matches_a_per_position_reference_for_ragged_drafts():
    rnd = random.Random(2943)
    checked = 0
    for _ in range(600):
        bs = rnd.randint(1, 5)
        drafts = [rnd.randint(0, 3) for _ in range(bs)]
        width = max(drafts) + 1
        mode = rnd.choice(["target", "bonus", "both"])
        invalid = None if mode == "target" else [rnd.random() < 0.3 for _ in range(bs)]
        target_invalid = (
            None
            if mode == "bonus" or not sum(drafts)
            else [rnd.random() < 0.3 for _ in range(sum(drafts))]
        )
        if invalid is None and target_invalid is None:
            continue
        tokens = []
        for count in drafts:
            accepted = rnd.randint(0, count)  # emitted positions 0..accepted
            row = [rnd.randint(0, 50) for _ in range(accepted + 1)]
            tokens.append(row + [-1] * (width - accepted - 1))
        cu = torch.tensor([sum(drafts[: i + 1]) for i in range(bs)])
        step = boundary.TokenConstraints()
        step.logits = torch.zeros(bs, 8)
        step.invalid = None if invalid is None else torch.tensor(invalid)
        step.target_invalid = (
            None if target_invalid is None else torch.tensor(target_invalid)
        )
        step.metadata = SimpleNamespace(cu_num_draft_tokens=cu)
        got = step.failed(torch.tensor(tokens)).tolist()
        assert got == _reference_failed(drafts, invalid, target_invalid, tokens), (
            drafts,
            invalid,
            target_invalid,
            tokens,
        )
        checked += 1
    assert checked > 400


def test_ragged_failures_stay_on_their_own_request():
    # Request 0 has 2 drafts, request 1 none, request 2 one. Only request 0's
    # second target row and request 1's bonus row are invalid.
    step = _step([False, True, False], [False, True, False], [2, 2, 3])
    tokens = torch.tensor([[7, 8, 9], [5, -1, -1], [6, 4, -1]])
    assert step.failed(tokens).tolist() == [True, True, False]
    # Request 0 stopped after its first draft, so the invalid row's token was
    # never emitted; request 1 emitted nothing.
    tokens = torch.tensor([[7, -1, -1], [-1, -1, -1], [6, 4, -1]])
    assert step.failed(tokens).tolist() == [False, False, False]


# -- commit and delivery wiring -------------------------------------------------


@pytest.mark.parametrize("deferred", [False, True])
def test_postprocess_commits_with_the_failure_and_forwards_gpu_carry(deferred):
    carry = torch.tensor([3, 4])

    def fail_row_one(step):
        step.invalid = torch.tensor([False, True])

    adapter = register(FakeAdapter("c", hard=[0, 1], on_apply=fail_row_one))
    adapter.commit_result = carry
    runner, seen = _postprocess_runner(
        {10: (4,), 11: (-2,)}, deferred, prev_batch=_quiet_batch("c")
    )
    _postprocess(runner, _quiet_batch("c"))
    (commit,) = [call for call in adapter.calls if call[0] == "commit"]
    assert commit[1].tolist() == [False, True]
    assert commit[2] is deferred
    assert seen["failed"].tolist() == [False, True]
    assert seen["channel_states"] is carry


def test_error_tokens_mark_failed_rows_in_both_layouts():
    failure = torch.tensor([False, True])
    assert boundary.error_tokens(torch.tensor([5, 6]), failure).tolist() == [5, -2]
    wide = torch.tensor([[5, 6, 7], [8, 9, 1]])
    assert boundary.error_tokens(wide, failure).tolist() == [[5, 6, 7], [-2, -2, -2]]


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


@pytest.mark.parametrize("layout", ["1d", "2d"])
def test_immediate_output_masks_failed_rows_in_both_layouts(layout):
    processor = object.__new__(mod.tokenIDProcessor)
    processor.is_deferred_out = False
    batch = SimpleNamespace(req_ids=[10, 11], return_logprobs=[False, False])
    failed = torch.tensor([False, True])
    if layout == "1d":
        tokens = torch.tensor([7, 8])
    else:
        tokens = torch.tensor([[7, 8, 9], [5, 6, -1]])
    ids, _ = processor.prepare_sampled_ids(
        batch, tokens, None, constraint_failed=failed
    )
    assert -2 not in ids[10]
    assert ids[11][0] == -2
    # The caller's tensor, which feeds GPU carry, is never rewritten.
    assert tokens.tolist() == ([7, 8] if layout == "1d" else [[7, 8, 9], [5, 6, -1]])


def test_has_constraints_tolerates_a_missing_batch():
    adapter = register(FakeAdapter("strict"))

    def enabled(batch):
        return "strict" in batch.active  # a third-party adapter may not guard None

    adapter.enabled = enabled
    assert boundary.has_constraints(None) is False
    assert boundary.has_constraints(active("strict")) is True
