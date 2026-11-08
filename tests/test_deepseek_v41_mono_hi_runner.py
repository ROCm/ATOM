# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ModelRunner's model hooks (``model_engine.model_hooks``) and DeepSeek-V4.1's hi verify head behind them.

The runner's hooks are declared (``VerifyHead``, ``DecodeRound``) and resolved once at install: a model without them
takes the logits path and runs no decode round; with them the runner asks the head for a verify's ids and its
verdict (the ids broadcast over TP, the next step's ids taken from the broadcast ones), and calls the decode round
where its work sits. Which steps the hi head serves is its own policy (``round.verify_head.HiVerifyHead``), tested
here against a stub hook: pure Python, no Triton, AITER or FlyDSL."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.models.deepseek_v41.mono.round import verify_head as vh


@pytest.fixture(scope="module")
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


def _runner(runner_module, model):
    """A runner with ``model`` installed: its hooks resolved as ``ModelRunner.__init__`` resolves them."""
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.model = model
    runner.verify_head, runner.decode_round = runner_module.model_hooks.resolve(model)
    runner.verify_token_ids = None
    runner.rejection_sampler = SimpleNamespace(synthetic_acceptance_rates=None)
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


# ---- the runner's declared hooks


def _stub_head(runner_module, ids=None):
    class StubHead(runner_module.model_hooks.VerifyHead):
        """A verify head as the runner sees one."""

        def __init__(self):
            self.ids_out, self.calls = ids, []

        def before_forward(self, runner, batch):
            self.calls.append(("before", batch))

        def ids(self, runner, batch, hidden_states, graph_key=None):
            self.calls.append(("ids", graph_key))
            return self.ids_out

    return StubHead()


def test_a_model_without_the_hooks_takes_the_logits_path(runner_module):
    runner = _runner(runner_module, SimpleNamespace())
    assert runner.verify_head is None and runner.decode_round is None
    assert runner._verify_head_ids(_batch(), torch.zeros(6, 4)) is None


def test_hooks_are_declared_types(runner_module):
    """``resolve``: a model attribute of that name that is not the declared hook type is an error, not a hook."""
    with pytest.raises(TypeError, match="verify_head"):
        runner_module.model_hooks.resolve(SimpleNamespace(verify_head=object()))
    rnd = runner_module.model_hooks.DecodeRound()
    assert runner_module.model_hooks.resolve(SimpleNamespace(decode_round=rnd)) == (
        None,
        rnd,
    )
    # every member defaults to "nothing taken over"
    assert rnd.spec_status() is None and rnd.decode_ids_filler() is None
    assert rnd.host_arena_reader("token_inputs") is None
    assert not rnd.committed(None)


def test_the_runner_asks_the_models_head(runner_module):
    ids = torch.arange(6, dtype=torch.int32)
    head = _stub_head(runner_module, ids)
    runner = _runner(runner_module, SimpleNamespace(verify_head=head))
    assert runner.verify_head is head
    assert runner._verify_head_ids(_batch(), torch.zeros(6, 4)) is ids
    assert runner._verify_head_ids(_batch(), torch.zeros(6, 4), (1, 6)) is ids
    assert head.calls == [("ids", None), ("ids", (1, 6))]


def test_postprocess_takes_the_heads_verdict_and_the_broadcast_ids(
    monkeypatch, runner_module
):
    """A head-served step: the head's verdict in place of the rejection sampler's; the ids are broadcast over TP and
    every id that feeds the next step (the next forward's ids, the drafter's anchors) is taken from the broadcast
    ids, at the head verdict's accepted counts, as on every other step."""
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
    # rank 0's ids, as a rank that is not rank 0 receives them
    rank0 = torch.arange(bs * 6, dtype=torch.int32).view(bs, 6) + 100
    broadcasts = []
    monkeypatch.setattr(
        runner_module,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=4, broadcast=lambda t, src: broadcasts.append(t) or rank0
        ),
    )
    monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(runner_module, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(runner_module, "sp_is_enabled", lambda: False)
    out = torch.zeros(bs, 6, dtype=torch.int32)  # this rank's own picks
    verdict = torch.tensor([[3, 5], [2, 0], [3, 11]], dtype=torch.int32)
    calls, proposed = [], []

    class Stop(Exception):
        pass

    head = _stub_head(runner_module)
    head.verdict = lambda s, ids, sampler: calls.append(ids) or (out, verdict)

    def propose(batch, input_ids, hidden, next_token_ids, num_reject, anchors):
        proposed.append(next_token_ids)
        assert torch.equal(anchors, verdict[2])
        raise Stop

    runner = _runner(runner_module, SimpleNamespace(verify_head=head))
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
    assert len(calls) == 1 and len(broadcasts) == 1 and broadcasts[0] is out
    # rank 0's ids at each request's accepted count (3 and 5), not this rank's picks
    assert proposed[0].tolist() == [103, 111]
    assert runner.tokenID_processor.prev_token_ids is proposed[0]


def test_the_staging_mark_calls_the_decode_round(runner_module):
    calls = []

    class Round(runner_module.model_hooks.DecodeRound):
        def before_staging_mark(self):
            calls.append("mark")

    for model, want in (
        (SimpleNamespace(decode_round=Round()), ["mark", "finish"]),
        (SimpleNamespace(), ["finish"]),
    ):
        calls.clear()
        runner = _runner(runner_module, model)
        runner.h2d_owner = SimpleNamespace(finish=lambda: calls.append("finish"))
        runner._mark_staging_h2d_enqueued()
        assert calls == want


@pytest.mark.parametrize("cls", ["ModelRunner", "RapidServeModelRunner"])
def test_after_forward_follows_every_target_launch(runner_module, cls):
    """Each ``forward`` that runs a decode step calls the decode round's ``after_forward`` right after
    ``run_model`` (the target's launch), at that one call site."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(getattr(runner_module, cls).forward))
    )
    stmts = [
        b
        for node in ast.walk(tree)
        for field in ("body", "orelse")
        for b in [getattr(node, field, None)]
        if isinstance(b, list)
    ]
    found = 0
    for body in stmts:
        for i, s in enumerate(body):
            if not isinstance(s, (ast.Assign, ast.Expr)):
                continue
            if "self.run_model(" in ast.unparse(s):
                assert "self.decode_round.after_forward" in ast.unparse(body[i + 1])
                found += 1
    assert found == 1


# ---- the hi verify head's policy (``HiVerifyHead``)


class _Hook:
    """The hi hook's surface ``HiVerifyHead`` uses."""

    def __init__(self, ids=None):
        self.ids, self.sampling = ids, []

    def set_head_sampling(self, temps=None, top_ps=None, rows=None):
        self.sampling.append(
            None
            if temps is None
            else (temps.tolist(), top_ps.tolist(), [int(n) for n in rows])
        )
        return True

    def head_out(self):
        return self.ids


@pytest.fixture
def spec_step(monkeypatch):
    """A speculative verify step (``vh._spec``), exact acceptance, no context parallelism."""
    spec = SimpleNamespace(num_spec_steps=5)
    monkeypatch.setattr(vh, "_spec", lambda: spec)
    monkeypatch.setattr(vh, "_limits", lambda: (1, 1e-10))
    monkeypatch.setattr(vh, "_pcp_world_size", lambda: 1)
    return spec


_RUNNER = SimpleNamespace(rejection_sampler=object())


def test_greedy_verify_takes_the_head_ids(spec_step):
    ids = torch.arange(8, dtype=torch.int32)
    hook = _Hook(ids)
    head = vh.HiVerifyHead(hook)
    batch = _batch()
    head.before_forward(_RUNNER, batch)
    assert not head.sampled and hook.sampling == []  # greedy: nothing staged
    assert torch.equal(head.ids(_RUNNER, batch, torch.zeros(6, 4)), ids[:6])
    for over in (
        dict(any_logprobs=True),
        dict(structured_requests={0: object()}),
        dict(required_channel=object()),
    ):
        assert head.ids(_RUNNER, _batch(**over), torch.zeros(6, 4)) is None
    assert (
        head.ids(SimpleNamespace(rejection_sampler=None), batch, torch.zeros(6, 4))
        is None
    )
    assert head.ids(_RUNNER, None, torch.zeros(6, 4)) is None
    # fewer ids than rows: the logits path
    assert head.ids(_RUNNER, batch, torch.zeros(9, 4)) is None


@pytest.mark.parametrize("why", ["not_a_verify", "relaxed", "pcp"])
def test_the_logits_path(spec_step, monkeypatch, why):
    head = vh.HiVerifyHead(_Hook(torch.arange(8, dtype=torch.int32)))
    if why == "not_a_verify":
        monkeypatch.setattr(vh, "_spec", lambda: None)
    elif why == "relaxed":
        monkeypatch.setattr(vh, "_limits", lambda: (10, 1e-10))
    else:
        monkeypatch.setattr(vh, "_pcp_world_size", lambda: 2)
    assert head.ids(_RUNNER, _batch(), torch.zeros(6, 4)) is None


def test_a_replay_reads_the_captured_buffer(spec_step):
    ids = torch.arange(12, dtype=torch.int32)
    hook = _Hook(ids)
    head = vh.HiVerifyHead(hook)
    head.captured((1, 6))
    hook.ids = None
    head.captured((2, 6))  # the head did not run in that graph
    assert torch.equal(head.ids(_RUNNER, _batch(), torch.zeros(6, 4), (1, 6)), ids[:6])
    assert head.ids(_RUNNER, _batch(), torch.zeros(6, 4), (2, 6)) is None
    assert head.ids(_RUNNER, _batch(), torch.zeros(6, 4), (4, 6)) is None


def test_sampled_verify(spec_step):
    hook = _Hook(torch.zeros(12, dtype=torch.int32))
    head = vh.HiVerifyHead(hook)
    batch = _batch(
        (0.7, 0.5),
        top_ps=np.asarray([0.9, 0.8], dtype=np.float32),
        num_scheduled_tokens=np.asarray([6, 6], dtype=np.int32),
    )
    head.before_forward(_RUNNER, batch)
    assert head.sampled and head.rows_sampling
    (temps, top_ps, rows), *_ = hook.sampling
    assert temps == pytest.approx([0.7, 0.5]) and top_ps == pytest.approx([0.9, 0.8])
    assert rows == [6, 6]
    assert head.ids(_RUNNER, batch, torch.zeros(12, 4)) is not None
    # top-k: not the head's draw (ATOM's sampler), and the head's rows go back to greedy
    topk = _batch((0.7,), top_ks=np.full(1, 20, dtype=np.int32))
    head.before_forward(_RUNNER, topk)
    assert not head.sampled and not head.rows_sampling and hook.sampling[-1] is None
    assert head.ids(_RUNNER, topk, torch.zeros(6, 4)) is None
    # a greedy step after it: rows greedy already, nothing staged
    head.before_forward(_RUNNER, _batch())
    assert len(hook.sampling) == 2


@pytest.mark.parametrize(
    "over",
    [
        dict(any_logprobs=True),
        dict(needs_independent_noise=np.ones(1, dtype=bool)),
        dict(temperatures=np.asarray([1e-12], dtype=np.float32)),
    ],
)
def test_a_sampled_step_the_head_does_not_draw(spec_step, over):
    hook = _Hook(torch.zeros(6, dtype=torch.int32))
    head = vh.HiVerifyHead(hook)
    head.before_forward(_RUNNER, _batch((0.7,), **over))
    assert not head.sampled and hook.sampling == []


def test_a_mixed_step_is_not_served(spec_step):
    """Some requests greedy, some sampled: ATOM's sampler (the head draws only all-sampled steps)."""
    head = vh.HiVerifyHead(_Hook(torch.zeros(12, dtype=torch.int32)))
    batch = _batch((0.0, 0.7), num_scheduled_tokens=np.asarray([6, 6], dtype=np.int32))
    head.before_forward(_RUNNER, batch)
    assert not head.sampled
    assert head.ids(_RUNNER, batch, torch.zeros(12, 4)) is None


def test_forced_acceptance_serves_as_without_it(spec_step):
    """Under ``--spec-decode-acceptance-length`` the head serves greedy steps and the sampled steps it drew (the
    synthetic schedule decides only acceptance)."""
    runner = SimpleNamespace(
        rejection_sampler=SimpleNamespace(synthetic_acceptance_rates=(1.0, 0.5, 0.0))
    )
    head = vh.HiVerifyHead(_Hook(torch.zeros(6, dtype=torch.int32)))
    assert head.ids(runner, _batch(), torch.zeros(6, 4)) is not None
    sampled = _batch((1.0,))
    head.before_forward(runner, sampled)
    assert head.sampled
    assert head.ids(runner, sampled, torch.zeros(6, 4)) is not None


def test_a_capture_stages_nothing(spec_step):
    """A graph capture's dummy step: ``set_head_sampling`` stages nothing (False), so the head's ids are not served
    as samples."""
    hook = _Hook(torch.zeros(6, dtype=torch.int32))
    hook.set_head_sampling = lambda *a: False
    head = vh.HiVerifyHead(hook)
    sampled = _batch((1.0,))
    head.before_forward(_RUNNER, sampled)
    assert not head.sampled and not head.rows_sampling
    assert head.ids(_RUNNER, sampled, torch.zeros(6, 4)) is None


# ---- the head's sampling rows (``HiLayers.set_head_sampling``)


class _Event:
    def synchronize(self):
        pass

    def record(self):
        pass


def _sampling_hook():
    """``HiLayers`` with only ``set_head_sampling``'s state, as ``_head_tables`` lays it out: two launches of 6 rows
    (a 2-request split) and one of 12 (a wide head), each block its rows' (T, top_p) then the seed's two words.
    """
    with (
        atom_without_aiter() as load
    ):  # (Triton stubbed with AITER where it is not installed)
        hr = load("atom.models.deepseek_v41.mono.hi_runner")
    m = object.__new__(hr.HiLayers)
    m.par_blocks, off = {}, 0
    for key, (r0, n) in {(12, 0): (0, 6), (12, 1): (6, 6), (24, 0): (0, 12)}.items():
        m.par_blocks[key] = (off, r0, n)
        off += 2 * n + 2
    m.par_last = np.zeros(off, dtype=np.int32)
    for o, _, n in m.par_blocks.values():
        m.par_last[o + 2 * n : o + 2 * n + 2] = (7, 9)  # (the seed words)
    m.par_rows = 12
    m.head_par = torch.from_numpy(m.par_last.copy())
    m.par_ring = [torch.zeros(off, dtype=torch.int32) for _ in range(4)]
    m.par_events = [_Event() for _ in range(4)]
    m.par_next = 0
    m.par_row_buf = np.zeros((m.par_rows, 2), dtype=np.float32)
    m.par_work = m.par_last.copy()
    m.par_work_f = m.par_work.view(np.float32)
    m.par_ring_np = [t.numpy() for t in m.par_ring]
    return m


def test_set_head_sampling_fills_persistent_buffers_in_place(monkeypatch):
    """Request i's (T, top_p) on its rows, into the same host buffers every step (nothing allocated); one staged
    copy when they change, none when they do not; greedy (None) zeroes the rows and keeps the seed words.
    """
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    m = _sampling_hook()
    bufs = (m.par_row_buf, m.par_work, m.par_last, *m.par_ring_np)
    temps = np.asarray([0.7, 0.5], dtype=np.float32)
    top_ps = np.asarray([0.9, 1.0], dtype=np.float32)
    assert m.set_head_sampling(temps, top_ps, np.asarray([6, 6], dtype=np.int32))
    par = m.head_par.numpy().view(np.float32)
    for o, r0, n in m.par_blocks.values():
        rows = par[o : o + 2 * n].reshape(n, 2)
        want_t = [0.7 if r < 6 else 0.5 for r in range(r0, r0 + n)]
        want_p = [0.9 if r < 6 else 1.0 for r in range(r0, r0 + n)]
        assert rows[:, 0].tolist() == pytest.approx(want_t)
        assert rows[:, 1].tolist() == pytest.approx(want_p)
        assert m.head_par[o + 2 * n : o + 2 * n + 2].tolist() == [7, 9]
    assert m.par_next == 1
    # unchanged: nothing staged
    assert m.set_head_sampling(temps, top_ps, np.asarray([6, 6], dtype=np.int32))
    assert m.par_next == 1
    # one request of 3 rows: the rows past it greedy
    assert m.set_head_sampling(temps[:1], top_ps[:1], np.asarray([3], dtype=np.int32))
    o, r0, n = m.par_blocks[24, 0]
    rows = m.head_par.numpy().view(np.float32)[o : o + 2 * n].reshape(n, 2)
    assert rows[:, 0].tolist() == pytest.approx([0.7] * 3 + [0.0] * 9)
    assert m.par_next == 2
    # greedy
    assert m.set_head_sampling(None, None, None)
    for o, _, n in m.par_blocks.values():
        assert not m.head_par[o : o + 2 * n].any()
        assert m.head_par[o + 2 * n : o + 2 * n + 2].tolist() == [7, 9]
    assert all(
        a is b
        for a, b in zip(bufs, (m.par_row_buf, m.par_work, m.par_last, *m.par_ring_np))
    )
