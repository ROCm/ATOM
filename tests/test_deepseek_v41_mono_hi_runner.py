# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ModelRunner's two generic switches DeepSeek-V4.1's hi path uses, and the hi verify head's sampling rows.

- The in-graph head output (``ModelRunner._head_output``): a model's ``compute_head`` gives its forward's logits, or a
  verify's output ids and verdict (recorded with a captured forward, as the logits are at TP 1); a model without one
  gives its logits. The runner takes the ids and verdict only where they stand for the step
  (``_head_ids_serve``), broadcasts the ids over TP and takes the next step's ids from the broadcast ones.
- The deferrable status wait: an attention builder that advances the step from the device-resident verdict
  (``consumes_device_verdict``) has the runner build it from zero counts and take the status after the target's launch.

Pure Python against stand-ins: no Triton, AITER or FlyDSL."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


class _Model:
    """A model as the runner's head sees it: its logits, and (``head``) its own head output."""

    def __init__(self, head=None):
        self.head, self.logits_calls = head, 0
        if head is not None:
            self.compute_head = lambda hidden_states: self.head

    def compute_logits(self, hidden_states):
        self.logits_calls += 1
        return hidden_states + 1


def _runner(runner_module, model):
    """A runner with ``model`` installed, its head output resolved as ``ModelRunner.__init__`` resolves it."""
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.model = model
    runner.compute_head = getattr(model, "compute_head", None)
    runner.rejection_sampler = SimpleNamespace(synthetic_acceptance_rates=None)
    return runner


def _batch(temps=(0.0,), **over):
    n = len(temps)
    b = {
        "total_seqs_num": n,
        "temperatures": np.asarray(temps, dtype=np.float32),
        "top_ks": np.full(n, -1, dtype=np.int32),
        "top_ps": np.ones(n, dtype=np.float32),
        "num_scheduled_tokens": np.full(n, 6, dtype=np.int32),
        "any_logprobs": False,
        "structured_requests": {},
        "required_channel": None,
        "needs_independent_noise": np.zeros(n, dtype=bool),
    }
    b.update(over)
    return SimpleNamespace(**b)


def _head(requests=1, rows=6):
    """A verify's head output: output ids [requests, rows], verdict [3, requests]."""
    return (
        torch.zeros(requests, rows, dtype=torch.int32),
        torch.zeros(3, requests, dtype=torch.int32),
    )


@pytest.fixture
def spec_step(monkeypatch, runner_module):
    """A speculative verify step of one request (5 drafts), exact acceptance, no context parallelism."""
    spec = SimpleNamespace(num_spec_steps=5, num_draft_tokens_np=np.full(1, 5))
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    monkeypatch.setattr(runner_module, "RELAXED_TOP_N", 1)
    monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: 1)
    return spec


# ---- the head output


def test_a_model_without_its_own_head_output_gives_its_logits(runner_module):
    model = _Model()
    runner = _runner(runner_module, model)
    assert runner.compute_head is None
    hidden = torch.zeros(6, 4)
    assert torch.equal(runner._head_output(hidden), hidden + 1)
    # a capture at TP > 1 records no head output for it
    assert runner._head_output(hidden, logits=False) is None
    assert model.logits_calls == 1


def test_the_models_head_output_is_recorded_as_the_logits_are(runner_module):
    """``compute_head`` gives the head output (logits or a verify's ids and verdict) at any TP; None leaves it to the
    logits (recorded only where they are: ``logits_in_graph``)."""
    head = _head()
    model = _Model(head)
    runner = _runner(runner_module, model)
    assert runner._head_output(torch.zeros(6, 4), logits=False) is head
    model.head = None
    assert runner._head_output(torch.zeros(6, 4), logits=False) is None
    assert torch.equal(runner._head_output(torch.zeros(6, 4)), torch.ones(6, 4))


@pytest.mark.parametrize("phase", ["warmup", "record"])
def test_the_recorded_forward_ends_in_the_head_output(runner_module, phase):
    """``capture_cudagraph``'s recorded forward returns ``_head_output`` (its warmup runs it too): the head is part
    of the recorded forward, as the logits are at TP 1."""
    src = textwrap.dedent(
        inspect.getsource(runner_module.ModelRunner.capture_cudagraph)
    )
    if phase == "warmup":
        assert "self._head_output(outputs[:running_tokens], logits_in_graph)" in src
    else:
        tree = ast.parse(src)
        (full,) = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "full_forward"
        ]
        assert "return self._head_output(outputs[:n], logits_in_graph)" in (
            ast.unparse(full)
        )
    assert "compute_logits" not in src


def test_a_step_takes_the_logits_where_the_ids_do_not_stand(runner_module, spec_step):
    model = _Model(_head())
    runner = _runner(runner_module, model)
    hidden = torch.zeros(6, 4)
    head = model.head
    assert runner._step_head(head, hidden, _batch()) is head
    logits = runner._step_head(head, hidden, _batch(any_logprobs=True))
    assert torch.equal(logits, hidden + 1)
    # a replay with no recorded head output, and a recorded tensor (the logits) as they are
    assert torch.equal(runner._step_head(None, hidden, _batch()), hidden + 1)
    recorded = torch.ones(6, 4)
    assert runner._step_head(recorded, hidden, _batch()) is recorded


def test_greedy_verify_takes_the_head_ids(runner_module, spec_step):
    runner = _runner(runner_module, _Model())
    assert runner._head_ids_serve(_head(), _batch())
    for over in (
        {"any_logprobs": True},
        {"structured_requests": {0: object()}},
        {"required_channel": object()},
    ):
        assert not runner._head_ids_serve(_head(), _batch(**over))
    assert not runner._head_ids_serve(_head(), None)
    runner.rejection_sampler = None
    assert not runner._head_ids_serve(_head(), _batch())


@pytest.mark.parametrize("why", ["not_a_verify", "relaxed", "pcp"])
def test_the_logits_path(runner_module, spec_step, monkeypatch, why):
    if why == "not_a_verify":
        monkeypatch.setattr(
            runner_module,
            "get_forward_context",
            lambda: SimpleNamespace(spec_decode_metadata=None),
        )
    elif why == "relaxed":
        monkeypatch.setattr(runner_module, "RELAXED_TOP_N", 10)
    else:
        monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: 2)
    assert not _runner(runner_module, _Model())._head_ids_serve(_head(), _batch())


def test_the_verdict_stands_for_its_layout_only(runner_module, spec_step):
    """The verdict is laid out for the forward's requests each verifying the full span: a step with a shorter span,
    or whose span the output ids do not hold, reads the logits; a padded bucket's first requests are the step's.
    """
    runner = _runner(runner_module, _Model())
    spec_step.num_draft_tokens_np = np.asarray([5, 3])
    two = _batch((0.0, 0.0))
    assert not runner._head_ids_serve(_head(2), two)
    spec_step.num_draft_tokens_np = np.asarray([5, 5])
    assert runner._head_ids_serve(_head(2), two)
    assert not runner._head_ids_serve(_head(2, rows=4), two)
    # fewer requests than the step's: no
    assert not runner._head_ids_serve(_head(1), two)
    # a bs-4 bucket replayed for 2 requests: the first 2 stand
    assert runner._head_ids_serve(_head(4), two)


def test_sampled_verify(runner_module, spec_step):
    """Every request sampled from temperature and top-p alone (ATOM's sample-and-match verify): the head's draws."""
    runner = _runner(runner_module, _Model())
    spec_step.num_draft_tokens_np = np.asarray([5, 5])
    batch = _batch((0.7, 0.5), top_ps=np.asarray([0.9, 0.8], dtype=np.float32))
    assert runner._head_ids_serve(_head(2), batch)


@pytest.mark.parametrize(
    "over",
    [
        {"top_ks": np.full(1, 20, dtype=np.int32)},
        {"needs_independent_noise": np.ones(1, dtype=bool)},
        {"temperatures": np.asarray([1e-12], dtype=np.float32)},
    ],
)
def test_a_sampled_step_the_head_does_not_draw(runner_module, spec_step, over):
    runner = _runner(runner_module, _Model())
    assert not runner._head_ids_serve(_head(), _batch((0.7,), **over))


def test_a_mixed_step_is_not_served(runner_module, spec_step):
    """Some requests greedy, some sampled: ATOM's sampler on the logits."""
    runner = _runner(runner_module, _Model())
    spec_step.num_draft_tokens_np = np.asarray([5, 5])
    assert not runner._head_ids_serve(_head(2), _batch((0.0, 0.7)))


def test_forced_acceptance_serves_as_without_it(runner_module, spec_step):
    """Under ``--spec-decode-acceptance-length`` the head's ids serve greedy and sampled steps alike (the synthetic
    schedule decides only acceptance)."""
    runner = _runner(runner_module, _Model())
    runner.rejection_sampler = SimpleNamespace(
        synthetic_acceptance_rates=(1.0, 0.5, 0.0)
    )
    assert runner._head_ids_serve(_head(), _batch())
    assert runner._head_ids_serve(_head(), _batch((1.0,)))


@pytest.mark.parametrize("bs, nb", [(2, 2), (2, 4)])
def test_postprocess_takes_the_heads_verdict_and_the_broadcast_ids(
    monkeypatch, runner_module, bs, nb
):
    """A step the head output stands for: its output ids and verdict (the step's requests: the first of a padded
    bucket's) in place of the samplers'; the ids are broadcast over TP and every id that feeds the next step (the
    next forward's ids, the drafter's anchors) is taken from the broadcast ids, at the verdict's accepted counts, as
    on every other step."""
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
    out = torch.zeros(nb, 6, dtype=torch.int32)  # this rank's own picks
    verdict = torch.zeros(3, nb, dtype=torch.int32)
    verdict[:, :bs] = torch.tensor([[3, 5], [2, 0], [3, 11]], dtype=torch.int32)
    sent, proposed = [], []

    class Stop(Exception):
        pass

    def propose(batch, input_ids, hidden, next_token_ids, num_reject, anchors):
        proposed.append(next_token_ids)
        # the verdict's own anchor row (``commit_speculative_state`` sees the head output's buffer)
        assert anchors.data_ptr() == verdict[2].data_ptr()
        assert anchors.tolist() == [3, 11]
        raise Stop

    runner = _runner(runner_module, _Model())
    runner.tokenID_processor = SimpleNamespace(
        is_deferred_out=True,
        prev_batch=None,
        prev_rejected_num=None,
        prev_bonus_num=None,
        prepare_sampled_ids=lambda *a, **kw: ({7: (3,), 8: (5,), -1: 1}, None),
        spec_status_queue=SimpleNamespace(send=lambda t, **kw: sent.append(t)),
        input_ids=SimpleNamespace(gpu=torch.zeros(16, dtype=torch.int32)),
    )
    runner.forward_done_event = SimpleNamespace(record=lambda stream=None: None)
    runner._served = None  # (no step override served this step)
    runner.drafter = object()
    runner.structured_output = None
    runner.propose_draft_token_ids = propose
    batch = SimpleNamespace(
        total_seqs_num=bs,
        total_tokens_num=12,
        num_spec_step=5,
        any_logprobs=False,
        structured_requests={},
        required_channel=None,
    )
    with pytest.raises(Stop):
        runner_module.ModelRunner.postprocess(
            runner,
            batch,
            (out, verdict),
            None,
            None,
            None,
            True,
            hidden_states=torch.zeros(12, 4),
        )
    assert len(broadcasts) == 1 and broadcasts[0].shape == (bs, 6)
    assert broadcasts[0].data_ptr() == out.data_ptr()
    assert sent[0].tolist() == [[3, 5], [2, 0]]
    # rank 0's ids at each request's accepted count (3 and 5), not this rank's picks
    assert proposed[0].tolist() == [103, 111]
    assert runner.tokenID_processor.prev_token_ids is proposed[0]


# ---- the deferrable status wait


def test_no_builder_consumes_the_device_verdict_by_default():
    with atom_without_aiter() as load:
        backends = load("atom.model_ops.attentions.backends")
    builder = backends.AttentionMetadataBuilder
    assert not builder.consumes_device_verdict(None, None, 6)


def _processor(runner_module):
    """A token processor as ``prepare_input_ids`` sees it, a prefill batch's (``_stage_input_ids`` stubbed)."""
    processor = object.__new__(runner_module.tokenIDProcessor)
    processor.runner = SimpleNamespace(h2d_groups=None)  # (no publication registry)
    processor.batch_info = SimpleNamespace(np=np.zeros(3, dtype=np.int32))
    processor.draft_tokens_per_seq = 0
    processor._stage_input_ids = lambda batch, q, group: ("ids", None)
    processor.recv_spec_status = mock.Mock(
        return_value=(np.array([2, 1], np.int32), np.array([1, 0], np.int32))
    )
    processor.prev_batch = SimpleNamespace(total_seqs_num=2)
    processor.spec_status_owed = False
    return processor


def _output_batch():
    return SimpleNamespace(
        total_seqs_num=2, total_tokens_num=12, produces_output=lambda: True
    )


def test_a_step_that_consumes_the_device_verdict_waits_after_the_launch(runner_module):
    """Built from zero counts (one a request of the forward the status is for), the wait owed; taken once, after."""
    processor = _processor(runner_module)
    cls = runner_module.tokenIDProcessor
    cls.prepare_input_ids(processor, _output_batch(), 6, None, defer_spec_status=True)
    processor.recv_spec_status.assert_not_called()
    assert processor.prev_rejected_num.tolist() == [0, 0] and processor.spec_status_owed
    assert processor.prev_bonus_num.tolist() == [0, 0]
    assert processor.prev_bonus_num is not processor.prev_rejected_num
    cls.take_deferred_spec_status(processor)
    processor.recv_spec_status.assert_called_once()
    assert (
        processor.prev_rejected_num.tolist() == [2, 1]
        and not processor.spec_status_owed
    )
    cls.take_deferred_spec_status(processor)
    processor.recv_spec_status.assert_called_once()


def test_any_other_step_waits_where_it_did(runner_module):
    processor = _processor(runner_module)
    runner_module.tokenIDProcessor.prepare_input_ids(
        processor, _output_batch(), 6, None
    )
    processor.recv_spec_status.assert_called_once()
    assert processor.prev_rejected_num.tolist() == [2, 1]
    assert not processor.spec_status_owed


def test_the_builder_decides_the_deferral_before_the_input_ids(runner_module):
    """``prepare_model`` asks the attention builder once a step and hands its answer to ``prepare_input_ids``."""
    src = textwrap.dedent(inspect.getsource(runner_module.ModelRunner.prepare_model))
    ask = src.index("self.attn_metadata_builder.consumes_device_verdict(")
    assert ask < src.index("self.tokenID_processor.prepare_input_ids(")
    assert "defer_spec_status=self.spec_status_deferred" in src


@pytest.mark.parametrize("cls", ["ModelRunner", "RapidServeModelRunner"])
def test_the_deferred_wait_follows_every_target_launch(runner_module, cls):
    """Each ``forward`` that runs a step takes the deferred status right after ``run_model`` (the target's launch),
    at that one call site."""
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(getattr(runner_module, cls).forward))
    )
    found = 0
    for node in ast.walk(tree):
        for field in ("body", "orelse"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for i, s in enumerate(body):
                if isinstance(s, (ast.Assign, ast.Expr)) and "self.run_model(" in (
                    ast.unparse(s)
                ):
                    nxt = ast.unparse(body[i + 1])
                    assert nxt.startswith("if self.spec_status_deferred:")
                    assert "self.tokenID_processor.take_deferred_spec_status()" in nxt
                    found += 1
    assert found == 1


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
