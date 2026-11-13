# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook's host-side glue (``ATOM_DSV41_MONO_HI``), on stand-ins for ATOM's runner, token processor,
sampler, metadata builder, cache and proposer: no GPU, no hi, no aiter. The device-side advance runs in the
Triton interpreter; the parts that load a module with Triton kernels (the device round, the post-verify) skip without
Triton, the rest (the routing decision, the lean host prep, the gate's import guard) run without it.

- the step's routing decision (``HiRound.service``: ``AttentionMetaData.served_by_round``), a served step's sampling
  rows for the verify head, and the lean host prep's predicate (``lean_host``);
- the device round (``device_round``): the advance kernel, its predicates and the builder's points (``hi_round``) in
  forward order around ATOM's token processor (its deferred status wait), the first step after a prefill;
- the post-verify as the forward's head output and the commit it makes (``post_verify``), and ATOM's Engram staging
  on served steps (``EngramStep.served_by_round``);
- the flag's guard: no ``hi`` import without it.
"""

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.utils.forward_context import RoundService

MONO = Path(__file__).resolve().parents[1] / "atom/models/deepseek_v41/mono"


def _src(name: str) -> Path:
    """``mono/<name>.py`` or, the round's host work, ``mono/round/<name>.py``."""
    return (
        MONO / f"{name}.py"
        if (MONO / f"{name}.py").exists()
        else MONO / "round" / f"{name}.py"
    )


def _load(name: str):
    """``mono/<name>.py`` loaded by path, its Triton kernels in the interpreter (no GPU)."""
    old = os.environ.get("TRITON_INTERPRET")
    os.environ["TRITON_INTERPRET"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(f"_hi_host_{name}", _src(name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("TRITON_INTERPRET")
        else:
            os.environ["TRITON_INTERPRET"] = old
    return module


@pytest.fixture(scope="module")
def lh():
    return _load("lean_host")


@pytest.fixture(scope="module")
def dr():
    pytest.importorskip("triton")
    return _load("device_round")


# ---------------------------------------------------------------- lean host prep


def _lean_world():
    span = SimpleNamespace(request_id=7, length=6)
    hook = SimpleNamespace(
        runs_every_layer=True,
        served_widths={6},
        lean_host=False,
        lean_steps=0,
        plan={},
    )
    batch = SimpleNamespace(is_dummy_run=False, multimodal_data={})
    base = {
        "hook": hook,
        "batch": batch,
        "spans": (span,),
        "running_bs": 1,
        "running_tokens": 6,
        "max_q_len": 6,
        "tentative": True,
        "is_prefill": False,
        "resumes": None,
    }
    return span, hook, base


def test_lean_predicate_takes_a_served_one_request_verify_only(lh):
    span, hook, base = _lean_world()
    assert lh.lean_step(**base)
    cases = {
        "no hook": {"hook": None},
        "not every layer": {
            "hook": SimpleNamespace(**{**vars(hook), "runs_every_layer": False})
        },
        "width never served": {
            "hook": SimpleNamespace(**{**vars(hook), "served_widths": {4}})
        },
        "dummy": {"batch": SimpleNamespace(is_dummy_run=True, multimodal_data={})},
        "prefill": {"is_prefill": True},
        "not a verify": {"tentative": False},
        "recomputed rows": {"resumes": [3]},
        "two requests": {
            "running_bs": 2,
            "running_tokens": 12,
            "spans": (span, SimpleNamespace(request_id=8, length=6)),
        },
        "padded bs": {"running_bs": 2},
        "no q bucket": {"max_q_len": None},
        "width != bucket": {"running_tokens": 8},
        "other bucket": {"max_q_len": 3, "running_tokens": 3},
        "span longer than bucket": {
            "spans": (SimpleNamespace(request_id=7, length=7),)
        },
        "image rows": {
            "batch": SimpleNamespace(
                is_dummy_run=False, multimodal_data={7: {"embedding_spans": [(0, 4)]}}
            )
        },
    }
    assert [k for k, over in cases.items() if lh.lean_step(**{**base, **over})] == []
    # a shorter verify of the same request in the served bucket stays lean (the bucket is the graph that replays)
    assert lh.lean_step(**{**base, "spans": (SimpleNamespace(request_id=7, length=3),)})


def test_lean_predicate_at_batch_widths_and_the_ragged_guard(lh):
    sp = [SimpleNamespace(request_id=i, length=6) for i in range(4)]
    hook = SimpleNamespace(
        runs_every_layer=True,
        served_widths={6, 12, 24},
        lean_host=False,
        lean_steps=0,
        plan={},
        split={6: (1, 6), 12: (2, 6), 24: (4, 6)},
        bmax=4,
    )
    batch = SimpleNamespace(is_dummy_run=False, multimodal_data={})
    base = {
        "hook": hook,
        "batch": batch,
        "spans": tuple(sp[:2]),
        "running_bs": 2,
        "running_tokens": 12,
        "max_q_len": 6,
        "tentative": True,
        "is_prefill": False,
        "resumes": None,
    }
    assert lh.lean_step(**base)
    assert lh.lean_step(
        **{**base, "spans": tuple(sp), "running_bs": 4, "running_tokens": 24}
    )
    assert lh.lean_step(
        **{**base, "spans": sp[:1], "running_bs": 1, "running_tokens": 6}
    )
    cases = {
        "three requests (width not served)": {
            "spans": tuple(sp[:3]),
            "running_bs": 3,
            "running_tokens": 18,
        },
        "ragged": {"spans": (sp[0], SimpleNamespace(request_id=1, length=3))},
        "padded batch": {"spans": tuple(sp[:1])},
        "other bucket": {"max_q_len": 3, "running_tokens": 6},
        "batch width never served": {
            "hook": SimpleNamespace(**{**vars(hook), "served_widths": {6}})
        },
        "prefill": {"is_prefill": True},
        "image rows": {
            "batch": SimpleNamespace(is_dummy_run=False, multimodal_data={1: {"x": 1}})
        },
    }
    assert [k for k, over in cases.items() if lh.lean_step(**{**base, **over})] == []
    wrong = []
    for spans, width, q, want in (
        ((6, 6), 12, 6, False),
        ((6, 3), 12, 6, False),
        ((3, 6), 12, 6, True),
        ((6, 7), 12, 6, True),
        ((3,), 6, 6, False),
        ((6, 6), 12, 12, False),
    ):
        try:
            lh.check_batch_layout(
                hook, [SimpleNamespace(length=x) for x in spans], width, q
            )
            raised = False
        except RuntimeError:
            raised = True
        if raised != want:
            wrong.append((spans, width, q))
    assert wrong == []


class _ServingHook(SimpleNamespace):
    """The hook's surface ``HiRound.service`` reads (``HiLayers.serves_width``: a width of ``split`` at its bucket)."""

    def serves_width(self, rows, max_q_len):
        split = self.split.get(rows)
        return split is not None and int(max_q_len or 0) == split[1]


def test_service_is_the_steps_one_routing_decision():
    """``HiRound.service``: the step's routing decision (``AttentionMetaData.served_by_round``), which the builder,
    its cache, Engram staging, the DSpark proposer and the hook's forward all read. A decode at a served width and
    bucket is served (``FORWARD``), lean when the lean predicate holds (``LEAN``), ``DEVICE`` on a device-round step;
    a prefill, an image row or another shape is not. Pure Python (no Triton)."""
    from atom.models.deepseek_v41.mono.round import HiRound, lean_host
    from atom.utils.forward_context import RoundService as S

    span = SimpleNamespace(request_id=7, length=6)
    hook = _ServingHook(runs_every_layer=True, lean_host=False, served_widths={6}, lean_steps=0, plan={},
                        split={6: (1, 6), 12: (2, 6)})  # fmt: skip
    assert lean_host.install(hook) and hook.lean_host
    rnd = HiRound(hook)
    batch = SimpleNamespace(is_dummy_run=False, multimodal_data={})
    base = {"batch": batch, "spans": (span,), "running_bs": 1, "running_tokens": 6, "max_q_len": 6, "decode": True,
                "tentative": True, "is_prefill": False, "resumes": None, "image_rows": False}  # fmt: skip

    def service(**over):
        a = {**base, **over}
        return rnd.service(
            a.pop("batch"),
            a.pop("spans"),
            a.pop("running_bs"),
            a.pop("running_tokens"),
            **a,
        )

    assert service() is S.LEAN
    cases = {
        # the DSpark draft's context write: a prefill of the served shape (6 new tokens) is ATOM's
        "6-token prefill": (
            {"is_prefill": True, "tentative": False, "decode": False},
            S.NONE,
        ),
        # Engram staging and the forward: an image row keeps both on ATOM's path
        "image rows": ({"image_rows": True}, S.NONE),
        "not a decode": ({"decode": False}, S.NONE),
        "other bucket": ({"max_q_len": 3}, S.NONE),
        "other width": ({"running_tokens": 8, "max_q_len": 8}, S.NONE),
        # a capture's dummy step, and a width no served forward recorded yet: served, laid out in full
        "dummy (capture)": (
            {"batch": SimpleNamespace(is_dummy_run=True, multimodal_data={})},
            S.FORWARD,
        ),
        "width not recorded": (
            {"running_tokens": 12, "running_bs": 2, "spans": (span, span)},
            S.FORWARD,
        ),
    }
    assert {k: service(**over) for k, (over, _) in cases.items()} == {
        k: want for k, (_, want) in cases.items()
    }
    off = _ServingHook(**{**vars(hook), "runs_every_layer": False, "lean_host": False})
    assert not lean_host.install(off) and HiRound(off).service(
        batch, (span,), 1, 6, max_q_len=6, decode=True, tentative=True, is_prefill=False, resumes=None,
        image_rows=False) is S.FORWARD  # fmt: skip
    # a device-round step: DEVICE; one whose builder would not lay it out lean raises
    rnd.device = SimpleNamespace(
        state=SimpleNamespace(on=True), laid_out=lambda md: None
    )
    assert service() is S.DEVICE
    with pytest.raises(RuntimeError, match="device-round"):
        service(max_q_len=3)
    md = SimpleNamespace(served_by_round=S.DEVICE)
    rnd.laid_out(md)
    assert hook.lean_steps == 1


def test_a_served_step_publishes_its_sampling_rows():
    """``HiRound.service`` publishes a served step's rows' (T, top_p), request by request on its rows, for the verify
    head (``HiLayers.set_head_sampling``, with the step's metadata); a greedy step after a sampled one sets the rows
    back to greedy once; a greedy step, a capture's (no sampling rows) and a step the hook does not serve stage
    nothing. Pure Python (no Triton)."""
    from atom.models.deepseek_v41.mono.round import HiRound

    staged = []

    def set_head_sampling(temps=None, top_ps=None, rows=None):
        staged.append(
            None if temps is None else (temps.tolist(), top_ps.tolist(), rows.tolist())
        )
        return True

    hook = _ServingHook(runs_every_layer=True, lean_host=False, served_widths={6}, lean_steps=0, plan={},
                        split={6: (1, 6), 12: (2, 6)}, set_head_sampling=set_head_sampling)  # fmt: skip
    rnd = HiRound(hook)
    span = SimpleNamespace(request_id=7, length=6)

    def step(temps=None, prefill=False, bs=1):
        batch = SimpleNamespace(is_dummy_run=False, multimodal_data={})
        if temps is not None:
            batch.temperatures = np.asarray(temps, dtype=np.float32)
            batch.top_ps = np.full(len(temps), 0.9, dtype=np.float32)
            batch.num_scheduled_tokens = np.full(len(temps), 6, dtype=np.int32)
        return rnd.service(batch, (span,) * bs, bs, 6 * bs, max_q_len=6, decode=not prefill, tentative=not prefill,
                           is_prefill=prefill, resumes=None, image_rows=False)  # fmt: skip

    assert step([0.0]) and staged == []
    assert step() and staged == []  # (a capture's step: no sampling rows)
    assert step([0.7, 0.5], bs=2)
    ((temps, top_ps, rows),) = staged
    assert temps == pytest.approx([0.7, 0.5]) and top_ps == pytest.approx([0.9, 0.9])
    assert rows == [6, 6]
    assert not step([0.7], prefill=True) and len(staged) == 1
    assert step([0.0]) and staged[-1] is None
    assert step([0.0]) and len(staged) == 2


def test_lean_attributes_keep_one_meaning_each():
    """The hook's lean attributes through the code that writes them after ``HiLayers.__init__``: ``lean_host.install``
    turns ``lean_host`` on, ``HiRound.laid_out`` counts a lean or device-round step in ``lean_steps``; neither touches
    the other's nor the served widths (a name shared with another counter would turn the width set into an int).
    Pure Python (no Triton)."""
    from atom.models.deepseek_v41.mono.round import HiRound, lean_host
    from atom.utils.forward_context import RoundService as S

    hook = _ServingHook(runs_every_layer=True, lean_host=False, served_widths={6, 12}, lean_steps=0, plan={},
                        split={6: (1, 6), 12: (2, 6)})  # fmt: skip
    assert lean_host.install(hook)
    rnd = HiRound(hook)
    for service in (S.NONE, S.FORWARD, S.LEAN, S.DEVICE, S.LEAN):
        rnd.laid_out(SimpleNamespace(served_by_round=service))
    assert hook.lean_host is True and hook.lean_steps == 3
    assert hook.served_widths == {6, 12} and hook.runs_every_layer is True


# ---------------------------------------------------------------- the device round


@pytest.mark.parametrize("dt", [torch.int64, torch.int32])
def test_advance_kernel(dr, dt):
    """start + 1 + accepted + i for every accepted count 0..5, nothing past the rows, accepted read off the verdict's
    anchor row (request b's: b x rows + accepted); at B requests each request from its own start and anchor.
    """
    for acc in range(6):
        pos = torch.full((8,), -7, dtype=dt)
        start = torch.tensor([1234], dtype=torch.int32)
        verdict = torch.tensor([[acc], [5 - acc], [acc]], dtype=torch.int32)
        dr.advance(pos, start, verdict[2], 6)
        assert pos.tolist() == [1235 + acc + i for i in range(6)] + [-7, -7]
    for nb in (2, 3, 4):
        pos = torch.full((nb * 6 + 2,), -7, dtype=dt)
        start = torch.tensor([100 * (b + 1) for b in range(4)], dtype=torch.int32)
        acc = [(b * 2 + 1) % 6 for b in range(nb)]
        anchors = [b * 6 + a for b, a in enumerate(acc)]
        verdict = torch.tensor([acc, [5 - a for a in acc], anchors], dtype=torch.int32)
        dr.advance(pos, start, verdict[2], 6, nb)
        want = [100 * (b + 1) + 1 + acc[b] + i for b in range(nb) for i in range(6)]
        assert pos.tolist() == want + [-7, -7]


def _batch(req=7, **over):
    b = {
        "is_dummy_run": False,
        "total_seqs_num_prefill": 0,
        "total_seqs_num": 1,
        "total_tokens_num": 6,
        "num_spec_step": 5,
        "req_ids": [req],
        "multimodal_data": {},
        "replaying": {},
        "context_lens": [112],
        "num_rejected": [0],
        "out": True,
    }
    b.update(over)
    b = SimpleNamespace(**b)
    b.produces_output = lambda: b.out
    return b


def test_device_round_predicate(dr):
    hook = SimpleNamespace(
        lean_host=True, runs_every_layer=True, served_widths={6}, split={6: (1, 6)}
    )
    st = SimpleNamespace(
        ready=(7,)
    )  # the requests (in order) whose served step left the verdict
    assert dr.device_round_step(st, _batch(), 6, hook)
    cases = {
        "no verdict": (SimpleNamespace(ready=None), _batch(), 6, hook),
        "other request": (st, _batch(req=8), 6, hook),
        "two requests": (st, _batch(total_seqs_num=2, req_ids=[7, 8]), 6, hook),
        "prefill in it": (st, _batch(total_seqs_num_prefill=1), 6, hook),
        "dummy": (st, _batch(is_dummy_run=True), 6, hook),
        "shrunk bucket": (st, _batch(num_spec_step=2), 3, hook),
        "width not served": (
            st,
            _batch(),
            6,
            SimpleNamespace(**{**vars(hook), "served_widths": {4}}),
        ),
        "lean off": (
            st,
            _batch(),
            6,
            SimpleNamespace(**{**vars(hook), "lean_host": False}),
        ),
        "image rows": (st, _batch(multimodal_data={7: {}}), 6, hook),
        "no hook": (st, _batch(), 6, None),
    }
    assert [k for k, args in cases.items() if dr.device_round_step(*args)] == []
    # B requests: the same requests, in the same order, after their served verdict
    hook = SimpleNamespace(
        lean_host=True,
        runs_every_layer=True,
        served_widths={6, 12, 24},
        split={6: (1, 6), 12: (2, 6), 24: (4, 6)},
    )

    def batch(reqs=(7, 8), **over):
        return _batch(**{"total_seqs_num": len(reqs), "req_ids": list(reqs), **over})

    st = SimpleNamespace(ready=(7, 8))
    assert dr.device_round_step(st, batch(), 6, hook)
    assert dr.device_round_step(
        SimpleNamespace(ready=(7, 8, 9, 10)), batch((7, 8, 9, 10)), 6, hook
    )
    cases = {
        "reordered": (st, batch((8, 7)), 6, hook),
        "one left": (st, batch((7,)), 6, hook),
        "one joined": (st, batch((7, 8, 9)), 6, hook),
        "other requests": (st, batch((7, 9)), 6, hook),
        "width not served": (
            st,
            batch(),
            6,
            SimpleNamespace(**{**vars(hook), "served_widths": {6}}),
        ),
        "not the batch split": (
            st,
            batch(),
            6,
            SimpleNamespace(**{**vars(hook), "split": {12: (1, 12)}}),
        ),
        "shrunk bucket": (st, batch(num_spec_step=2), 3, hook),
    }
    assert [k for k, args in cases.items() if dr.device_round_step(*args)] == []


def _round_world(dr, monkeypatch):
    """ATOM's token processor (``tokenIDProcessor.prepare_input_ids`` / ``take_deferred_spec_status``, its staging
    stubbed) and stand-ins for the sampler, builder and cache around the decode round (``hi_round.HiRound``) at the
    points ATOM's call it, and ``forward``: one forward as ``ModelRunner.forward`` runs them (prepare_model: the
    builder's ``consumes_device_verdict`` -> prepare_input_ids -> the builder's layout (+ the model inputs' probe) ->
    run_model -> the deferred status wait -> postprocess's verdict -> the builder's commit at its anchors).
    """
    from atom.models.deepseek_v41.mono.round import HiRound

    with atom_without_aiter() as load:
        tip = load("atom.model_engine.model_runner").tokenIDProcessor
    log, queue, served_now = [], [], [True]
    hook = _ServingHook(
        lean_host=True,
        runs_every_layer=True,
        served_widths={6, 12},
        split={6: (1, 6), 12: (2, 6)},
        bmax=2,
        pos=torch.tensor([100], dtype=torch.int32),
        lean_steps=0,
    )

    rnd = HiRound(hook, device=dr.DeviceRound(hook))
    processor = object.__new__(tip)
    processor.runner = SimpleNamespace(h2d_groups=None)
    processor.batch_info = SimpleNamespace(np=np.zeros(3, dtype=np.int32))
    processor.draft_tokens_per_seq = 0
    processor._stage_input_ids = lambda b, q, group: ("ids", None)
    processor.prev_batch, processor.spec_status_owed = None, False

    def recv_spec_status():
        log.append("recv")
        return queue.pop(0)

    processor.recv_spec_status = recv_spec_status

    class Cache:  # ``PagedAttentionCache``: a device-round step takes no probe
        cursor = pending = None

        def prepare_state(self, step, *, histories=True):
            log.append("probe")

        def prepare_device_round_state(self, step, *, histories=True):
            pass

    cache = Cache()

    def build(b, running_bs, running_tokens):
        """``_prepare_step``: the routing decision, the layout, then ``laid_out``."""
        span = SimpleNamespace(request_id=b.req_ids[0], length=6)
        service = (
            rnd.service(b, (span,), running_bs, running_tokens, max_q_len=6, decode=True, tentative=True,
                        is_prefill=False, resumes=None, image_rows=False)
            if served_now[0]
            else RoundService.NONE
        )  # fmt: skip
        step = SimpleNamespace(
            positions=torch.zeros(6, dtype=torch.int64),
            tentative=True,
            slots=torch.zeros(1, dtype=torch.int32),
            running_tokens=6,
        )
        log.append(("build", running_bs, running_tokens))
        md = SimpleNamespace(step=step, cache=cache, served_by_round=service)
        rnd.laid_out(md)
        return md

    def forward(b, served=True, accepted=3):
        log.clear()
        served_now[0] = served and not b.total_seqs_num_prefill
        # ``prepare_model``: the builder's answer, then the input ids
        defer = rnd.consumes_device_verdict(b, 6)
        tip.prepare_input_ids(processor, b, 6, None, defer_spec_status=defer)
        r = processor.prev_rejected_num
        zeros = isinstance(r, np.ndarray) and not r.any()
        log.append(("ids", f"zeros{len(r)}" if zeros else r))
        md = build(b, 1, 6)
        if md.served_by_round is RoundService.DEVICE:
            cache.prepare_device_round_state(md.step, histories=False)
        else:
            cache.prepare_state(md.step, histories=False)
        log.append("replay")
        if defer:
            tip.take_deferred_spec_status(processor)
        verdict = torch.tensor(
            [[accepted], [5 - accepted], [accepted]], dtype=torch.int32
        )
        log.append("verdict")
        # ``commit_speculative_state`` at the verdict's anchors, on a served step
        if md.served_by_round:
            rnd.committed(md, verdict[2])
        if b.produces_output():
            processor.prev_batch = b
        return list(log), md

    return forward, queue, processor, hook


def test_device_round_wrappers_in_forward_order(dr, monkeypatch):
    """A device-round step skips the verdict wait (its host status zeros: built as if no draft had been rejected),
    advances, skips the probe and takes the owed wait after the launch, in queue order; a stale verdict never carries
    across a forward that left none; every other step runs as without the hooks."""
    forward, queue, processor, hook = _round_world(dr, monkeypatch)
    queue[:] = [("r0", "b0"), ("r1", "b1"), ("r2", "b2"), ("r3", "b3"), ("r4", "b4")]
    # the first decode: no verdict yet -> host path
    l1, s1 = forward(_batch())
    assert l1 == ["recv", ("ids", "r0"), ("build", 1, 6), "probe", "replay", "verdict"]
    assert s1.served_by_round is not RoundService.DEVICE
    hook.pos.fill_(100)
    # after a served verdict (accepted 3 in forward 1)
    l2, s2 = forward(_batch(), accepted=3)
    assert l2 == [("ids", "zeros1"), ("build", 1, 6), "replay", "recv", "verdict"]
    assert s2.served_by_round is RoundService.DEVICE
    assert s2.step.positions.tolist() == [104 + i for i in range(6)]
    assert processor.prev_rejected_num == "r1"
    # another request: host path, and the verdict of 7 is gone
    l3, s3 = forward(_batch(req=8))
    assert l3[:2] == ["recv", ("ids", "r2")] and "probe" in l3
    assert s3.served_by_round is not RoundService.DEVICE
    # 7 again, but forward 3 was request 8's: host path; not served -> no verdict kept
    l4, _ = forward(_batch(), served=False)
    # after an unserved forward of 7: host path
    l5, s5 = forward(_batch())
    assert l4[:2] == ["recv", ("ids", "r3")] and l5[:2] == ["recv", ("ids", "r4")]
    assert s5.served_by_round is not RoundService.DEVICE and not queue


def test_first_step_after_a_prefill(dr, monkeypatch):
    """The decode step right after its request's output-producing prefill builds from zero rejected / bonus without
    waiting and takes the owed wait after the launch (the queue stays in order); its positions are the host's; the
    step after it is a device-round step; a batch with one prefilled request gets the prefill's zeros; after a middle
    prefill chunk or a prefill of a request not in the step, the first step waits as before.
    """
    z = np.zeros(1, dtype=np.int32)
    forward, queue, processor, _ = _round_world(dr, monkeypatch)
    queue[:] = [("p", "p"), (z, z), ("r1", "b1")]
    forward(
        _batch(total_seqs_num_prefill=1)
    )  # the prefill: host path (its own status wait)
    l1, s1 = forward(_batch())  # the first step: no wait, the owed one after the launch
    assert l1 == [
        ("ids", "zeros1"),
        ("build", 1, 6),
        "probe",
        "replay",
        "recv",
        "verdict",
    ]
    assert s1.served_by_round is not RoundService.DEVICE
    assert processor.prev_rejected_num is z
    l2, s2 = forward(_batch())  # then a device-round step, as before
    assert l2 == [("ids", "zeros1"), ("build", 1, 6), "replay", "recv", "verdict"]
    assert s2.served_by_round is RoundService.DEVICE and not queue

    # a batch: request 8 prefilled alone, then a decode of 7 and 8 at the batch width
    forward, queue, processor, _ = _round_world(dr, monkeypatch)
    queue[:] = [("p", "p"), (z, z)]
    forward(_batch(req=8, total_seqs_num_prefill=1))
    lm, sm = forward(_batch(total_seqs_num=2, req_ids=[7, 8]))
    assert lm[0] == ("ids", "zeros1") and lm[-2:] == ["recv", "verdict"]
    assert sm.served_by_round is not RoundService.DEVICE and not queue

    for steps in (
        [_batch(total_seqs_num_prefill=1), _batch(total_seqs_num_prefill=1, out=False)],
        [_batch(req=8, total_seqs_num_prefill=1)],
        [
            _batch(req=8, total_seqs_num_prefill=1),
            _batch(req=9, total_seqs_num_prefill=1),
        ],
    ):
        forward, queue, _, _ = _round_world(dr, monkeypatch)
        queue[:] = [("p", "p")] * 4 + [(z, z)]
        for b in steps:
            forward(b)
        lf, _ = forward(_batch())
        assert lf[0] == "recv"  # waits as before


# ---------------------------------------------------------------- the post-verify: head output and commit


@pytest.fixture
def no_aiter_sampler(monkeypatch):
    """The rejection sampler's module (it imports aiter) as the constants ``PostVerify`` reads from it (the
    post-verify's module has Triton kernels: skipped without Triton)."""
    pytest.importorskip("triton")
    if "aiter" not in sys.modules:
        try:
            import aiter  # noqa: F401
        except ImportError:
            fake = SimpleNamespace(_SYNTHETIC_RNG_BASE_SEED=0x5EED)
            monkeypatch.setitem(sys.modules, "atom.model_ops.rejection_sampler", fake)


def test_the_post_verify_is_the_forwards_head_output(no_aiter_sampler):
    """A width's fixed outputs, laid out as the served step lays its rows out (B requests of ``rq`` rows: ``rq - 1``
    drafts and a bonus row each), are the forward's head output (``HiLayers.head_output``: output ids [B, rq],
    verdict [3, B]) exactly when this forward ran the post-verify at that width; else None (the logits).
    """
    from atom.models.deepseek_v41.mono import hi_runner
    from atom.models.deepseek_v41.mono.round import post_verify

    split = {6: (1, 6), 12: (2, 6)}
    pv = post_verify.PostVerify((6, 12), split, torch.device("cpu"))
    assert pv.bufs[12]["target"].tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert pv.bufs[12]["bonus"].tolist() == [5, 11]
    assert pv.bufs[12]["cu"].tolist() == [5, 10]
    out, verdict = pv.output(12)
    assert out is pv.bufs[12]["out"] and out.shape == (2, 6)
    assert verdict is pv.bufs[12]["verdict"] and verdict.shape == (3, 2)
    hook = object.__new__(hi_runner.HiLayers)
    hook.pv, hook.verified = pv, 0
    assert hook.head_output(12) is None  # (no post-verify this forward)
    hook.verified = 12
    assert hook.head_output(12) == (out, verdict) and hook.head_output(6) is None
    hook.pv = None
    assert hook.head_output(12) is None


def test_post_verify_skips_the_commit_at_its_own_anchors(no_aiter_sampler):
    """The builder's commit skips its launch (``committed``: the bookkeeping kept) exactly when the anchors it commits
    are a launched width's in-graph verdict's (the runner took the head output, a padded bucket's first requests
    included); at any other anchors (the logits path's verdict), or with nothing staged on the device, it runs as
    before."""
    from atom.models.deepseek_v41.mono.round import post_verify

    pv = post_verify.PostVerify((6, 12), {6: (1, 6), 12: (2, 6)}, torch.device("cpu"))
    step = object()

    def md():
        pending = SimpleNamespace(staged_on_device=True, step=step)
        return SimpleNamespace(cache=SimpleNamespace(pending=pending), step=step)

    verdict = pv.bufs[12]["verdict"]
    m = md()
    # a width the forward never ran the kernel at
    assert not pv.committed(m, verdict[2]) and m.cache.pending is not None
    pv.launched.add(12)
    assert pv.committed(m, verdict[2]) and m.cache.pending is None
    m = md()
    assert pv.committed(m, verdict[:, :1][2])  # (one request in the bs-2 bucket)
    for anchors in (verdict[2].clone(), torch.zeros(2, dtype=torch.int32), None):
        m = md()
        assert not pv.committed(m, anchors) and m.cache.pending is not None
    m = md()
    m.cache.pending.staged_on_device = False
    assert not pv.committed(m, verdict[2]) and m.cache.pending is not None
    m = md()
    m.cache.pending.step = object()
    with pytest.raises(RuntimeError, match="not prepared for this step"):
        pv.committed(m, verdict[2])


# ---------------------------------------------------------------- ATOM's Engram staging on served steps


def test_engram_staging_skips_its_lookup_on_a_served_step():
    """ATOM's ``EngramStaging`` on a step its builder routed to the hook (``EngramStep.served_by_round``, from
    ``AttentionMetaData.served_by_round``): it only advances the cursor (no snapshot, no side stream) and its join waits
    on nothing; on any other step (an image row's, at a served width) the full lookup, as the forward's ATOM path
    reads it."""
    with (
        atom_without_aiter() as load
    ):  # (its hashing kernels' Triton stubbed with AITER where it is not installed)
        EngramStaging = load("atom.model_ops.engram.device.staging").EngramStaging

    calls = []

    class Lookup(Exception):
        """The full lookup started (its side stream needs a GPU: stopped here)."""

    def snapshot(w):
        calls.append(("snapshot", w))
        raise Lookup

    staging = object.__new__(EngramStaging)
    staging.host = SimpleNamespace()
    staging._snapshot = snapshot
    staging._advance_cursor = lambda w: calls.append(("cursor", w))
    staging.step = SimpleNamespace(served_by_round=True)
    staging.start(6)
    staging.join()
    assert calls == [("cursor", 6)]
    calls.clear()
    staging.step = SimpleNamespace(served_by_round=False)
    with pytest.raises(Lookup):
        staging.start(6)
    assert calls == [("snapshot", 6)] and not staging._served


# ---------------------------------------------------------------- install and the flag's guard


def test_the_gate_decides_before_any_hi_import():
    """``dispatch`` imports the gate (``hi_gate``: configuration, environment and device properties only) and reaches
    the hook only through the gate's yes (a lazy import of ``hi_dispatch``); the hook's modules import nothing of the
    hi runtime (``mono.hi``) at module level, and importing them loads none of it."""
    hi = re.compile(r"(^|[.:])hi(_|\.|$)")
    tree = ast.parse((MONO / "dispatch.py").read_text())
    top = {
        (n.module or "") + ":" + a.name
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    }
    assert [t for t in top if hi.search(t)] == ["atom.models.deepseek_v41.mono:hi_gate"]
    lazy = [
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.col_offset and hi.search(n.module or "")
    ]
    assert lazy == ["atom.models.deepseek_v41.mono.hi_dispatch"]
    gate = ast.parse((MONO / "hi_gate.py").read_text())
    mods = {
        (n.module if isinstance(n, ast.ImportFrom) else a.name)
        for n in gate.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    assert mods <= {
        "__future__",
        "importlib.util",
        "logging",
        "torch",
        "atom.utils",
    }, mods
    for name in (
        "hi_runner",
        "hi_bridge",
        "hi_dispatch",
        "device_round",
        "post_verify",
        "lean_host",
        "hi_weights",
        "round/__init__",
        "hi_plan",
        "hi_compose",
        "hi_bands",
    ):
        tree = ast.parse(_src(name).read_text())
        for n in tree.body:
            if isinstance(n, ast.ImportFrom):
                assert ".mono.hi." not in (n.module or "") + ".", (name, n.module)
    before = set(sys.modules)
    from atom.models.deepseek_v41.mono import hi_compose, hi_gate  # noqa: F401
    from atom.models.deepseek_v41.mono.round import HiRound  # noqa: F401

    assert not [m for m in set(sys.modules) - before if ".mono.hi." in m + "."]
    pytest.importorskip("triton")
    from atom.models.deepseek_v41.mono import hi_bridge, hi_runner  # noqa: F401

    assert not [m for m in set(sys.modules) - before if ".mono.hi." in m + "."]
