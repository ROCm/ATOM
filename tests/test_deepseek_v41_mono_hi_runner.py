# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ModelRunner's two generic switches DeepSeek-V4.1's hi path uses, and the hi verify head's sampling rows.

- The verify head (``VerifyHead``): a decode graph records the forward alone; after it, ``run_lm_head`` takes the
  verify head's output ids and spec status where they stand for the step (``VerifyHead.samples_step``), else the
  logits. ``postprocess`` broadcasts the ids over TP unless the head's agree on every rank, and takes the next
  step's ids from the broadcast ones.
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


class _Logits:
    """A model whose head is its logits."""

    def compute_logits(self, hidden_states):
        return hidden_states + 1


class _Head:
    """A verify head standing for every step, giving ``verified``; its ids not
    the same on every rank (``ids_agree``)."""

    ids_agree = False

    def __init__(self, verified=None):
        self.verified = verified

    def samples_step(self, batch, spec, rows):
        return True

    def sample_token_ids(self, hidden):
        return self.verified


def _runner(runner_module, model):
    """A runner with ``model`` installed, its verify head bound as ``ModelRunner.__init__`` binds it."""
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.model = model
    runner.rejection_sampler = SimpleNamespace(synthetic_acceptance_rates=None)
    runner._bind_verify_head()
    return runner


# ---- the verify head


def test_a_model_without_a_verify_head_binds_none(runner_module):
    assert _runner(runner_module, _Logits()).verify_head is None


def test_the_models_own_verify_head_is_bound(runner_module):
    model = _Logits()
    model.verify_head = _Head()
    assert _runner(runner_module, model).verify_head is model.verify_head


def test_the_recorded_forward_is_the_forward_alone(runner_module):
    """``capture_cudagraph`` records the forward without its head (the head runs eagerly after a replay,
    ``run_lm_head``); its warmup runs the verify head once, so its kernels build before serving.
    """
    src = textwrap.dedent(
        inspect.getsource(runner_module.ModelRunner.capture_cudagraph)
    )
    (full,) = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "full_forward"
    ]
    body = ast.unparse(full)
    assert "compute_logits" not in body and "sample_token_ids" not in body
    assert "self.verify_head.sample_token_ids(outputs[:running_tokens])" in src


@pytest.mark.parametrize("bs, nb", [(2, 2), (2, 4)])
def test_postprocess_takes_the_verify_and_the_broadcast_ids(
    monkeypatch, runner_module, bs, nb
):
    """A step the verify head stands for: its output ids and spec status (the step's requests: the first of a padded
    bucket's) in place of the samplers'; the ids are broadcast over TP and every id that feeds the next step (the
    next forward's ids, the drafter's anchors) is taken from the broadcast ids, at the spec status's accepted counts,
    as on every other step."""
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
    spec_status = torch.zeros(3, nb, dtype=torch.int32)
    spec_status[:, :bs] = torch.tensor([[3, 5], [2, 0], [3, 11]], dtype=torch.int32)
    sent, proposed = [], []

    class Stop(Exception):
        pass

    def propose(batch, input_ids, hidden, next_token_ids, num_reject, anchors):
        proposed.append(next_token_ids)
        # the spec status's own anchor row (``commit_speculative_state`` sees the verify's buffer)
        assert anchors.data_ptr() == spec_status[2].data_ptr()
        assert anchors.tolist() == [3, 11]
        raise Stop

    model = _Logits()
    model.verify_head = _Head(SimpleNamespace(token_ids=out, spec_status=spec_status))
    runner = _runner(runner_module, model)
    runner.tokenID_processor = SimpleNamespace(
        is_deferred_out=True,
        is_pd_producer=False,
        prev_batch=None,
        prev_rejected_num=None,
        prev_bonus_num=None,
        prepare_sampled_ids=lambda *a, **kw: ({7: (3,), 8: (5,), -1: 1}, None),
        spec_status_queue=SimpleNamespace(send=lambda t, **kw: sent.append(t)),
        input_ids=SimpleNamespace(gpu=torch.zeros(16, dtype=torch.int32)),
    )
    runner.forward_done_event = SimpleNamespace(record=lambda stream=None: None)
    runner._served = None  # (no step override served this step)
    runner.drafter = SimpleNamespace(context_kv_only=False)
    runner.structured_output = None
    runner.propose_draft_token_ids = propose
    batch = SimpleNamespace(
        total_seqs_num=bs,
        total_tokens_num=12,
        num_spec_step=5,
        any_logprobs=False,
        structured_requests={},
        presence_penalties=None,
        required_channel=None,
    )
    hidden = torch.zeros(12, 4)
    with pytest.raises(Stop):
        runner.postprocess(
            batch,
            None,
            None,
            None,
            True,
            runner.run_lm_head(batch, hidden, True),
            hidden_states=hidden,
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
