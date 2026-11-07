# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook's host-side glue (``ATOM_DSV41_MONO_HI``), on stand-ins for ATOM's runner, token processor,
sampler, metadata builder, cache and proposer: no GPU, no hi, no aiter. The device-side advance runs in the
Triton interpreter.

- lean host prep (``lean_host``): the served-step predicate and the builder's hooks;
- the device round (``device_round``): the advance kernel, its predicates and its hooks (``hi_round``) in forward
  order, the first step after a prefill;
- the post-verify's routing (``post_verify``) and ATOM's Engram staging on served steps (its ``lookup_served``);
- the install's refusals (``hi_dispatch``) and the flag's guard: no ``hi`` import without it.
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

pytest.importorskip("triton")

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
    base = dict(
        hook=hook,
        batch=batch,
        spans=(span,),
        running_bs=1,
        running_tokens=6,
        max_q_len=6,
        tentative=True,
        is_prefill=False,
        resumes=None,
    )
    return span, hook, base


def test_lean_predicate_takes_a_served_one_request_verify_only(lh):
    span, hook, base = _lean_world()
    assert lh.lean_step(**base)
    cases = {
        "no hook": dict(hook=None),
        "not every layer": dict(
            hook=SimpleNamespace(**{**vars(hook), "runs_every_layer": False})
        ),
        "width never served": dict(
            hook=SimpleNamespace(**{**vars(hook), "served_widths": {4}})
        ),
        "dummy": dict(batch=SimpleNamespace(is_dummy_run=True, multimodal_data={})),
        "prefill": dict(is_prefill=True),
        "not a verify": dict(tentative=False),
        "recomputed rows": dict(resumes=[3]),
        "two requests": dict(
            running_bs=2,
            running_tokens=12,
            spans=(span, SimpleNamespace(request_id=8, length=6)),
        ),
        "padded bs": dict(running_bs=2),
        "no q bucket": dict(max_q_len=None),
        "width != bucket": dict(running_tokens=8),
        "other bucket": dict(max_q_len=3, running_tokens=3),
        "span longer than bucket": dict(
            spans=(SimpleNamespace(request_id=7, length=7),)
        ),
        "image rows": dict(
            batch=SimpleNamespace(
                is_dummy_run=False, multimodal_data={7: {"embedding_spans": [(0, 4)]}}
            )
        ),
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
    base = dict(
        hook=hook,
        batch=batch,
        spans=tuple(sp[:2]),
        running_bs=2,
        running_tokens=12,
        max_q_len=6,
        tentative=True,
        is_prefill=False,
        resumes=None,
    )
    assert lh.lean_step(**base)
    assert lh.lean_step(
        **{**base, "spans": tuple(sp), "running_bs": 4, "running_tokens": 24}
    )
    assert lh.lean_step(
        **{**base, "spans": sp[:1], "running_bs": 1, "running_tokens": 6}
    )
    cases = {
        "three requests (width not served)": dict(
            spans=tuple(sp[:3]), running_bs=3, running_tokens=18
        ),
        "ragged": dict(spans=(sp[0], SimpleNamespace(request_id=1, length=3))),
        "padded batch": dict(spans=tuple(sp[:1])),
        "other bucket": dict(max_q_len=3, running_tokens=6),
        "batch width never served": dict(
            hook=SimpleNamespace(**{**vars(hook), "served_widths": {6}})
        ),
        "prefill": dict(is_prefill=True),
        "image rows": dict(
            batch=SimpleNamespace(is_dummy_run=False, multimodal_data={1: {"x": 1}})
        ),
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


def test_lean_host_marks_a_served_step_and_refuses_without_every_layer(lh):
    """``LeanHost`` at the builder's decode-round hooks: a served step lays out lean (the builder then skips the
    compress plans and the step planners) and is marked after its layout; any other step is neither; no lean host
    prep when a layer runs on ATOM."""
    _, hook, base = _lean_world()
    off = SimpleNamespace(**{**vars(hook), "runs_every_layer": False})
    assert lh.install(off) is None and not off.lean_host
    lean = lh.install(hook)
    assert hook.lean_host

    def layout(**over):
        a = {**base, **over}
        return lean.lean_layout(a["batch"], a["spans"], a["running_bs"], a["running_tokens"], max_q_len=a["max_q_len"],
                                tentative=a["tentative"], is_prefill=a["is_prefill"], resumes=a["resumes"])  # fmt: skip

    md = SimpleNamespace(step=SimpleNamespace())
    assert layout()
    lean.laid_out(md, True)
    assert md.step.lean_host and hook.lean_steps == 1
    md2 = SimpleNamespace(step=SimpleNamespace())
    assert not layout(max_q_len=3, running_tokens=3) and not layout(is_prefill=True)
    lean.laid_out(md2, False)
    assert not hasattr(md2.step, "lean_host") and hook.lean_steps == 1


def test_lean_attributes_are_set_once_in_init():
    """The hook's lean attributes are its own: each set once in ``HiLayers.__init__``, never reassigned (a name
    shared with another counter would turn the width set into an int)."""
    tree = ast.parse((MONO / "hi_runner.py").read_text())
    names = ("served_widths", "lean_host", "lean_steps", "runs_every_layer")
    sets = {n: [] for n in names}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for t in targets:
                for e in t.elts if isinstance(t, ast.Tuple) else [t]:
                    if (
                        isinstance(e, ast.Attribute)
                        and isinstance(e.value, ast.Name)
                        and e.attr in sets
                    ):
                        sets[e.attr].append(fn.name)
    assert all(v == ["__init__"] for v in sets.values()), sets


# ---------------------------------------------------------------- the device round


@pytest.mark.parametrize("dt", [torch.int64, torch.int32])
def test_advance_kernel(dr, dt):
    """start + 1 + accepted + i for every accepted count 0..5, nothing past the rows; at B requests each request from
    its own start and verdict column."""
    for acc in range(6):
        pos = torch.full((8,), -7, dtype=dt)
        start = torch.tensor([1234], dtype=torch.int32)
        verdict = torch.tensor([[acc], [5 - acc], [acc]], dtype=torch.int32)
        dr.advance(pos, start, verdict, 6)
        assert pos.tolist() == [1235 + acc + i for i in range(6)] + [-7, -7]
    for nb in (2, 3, 4):
        pos = torch.full((nb * 6 + 2,), -7, dtype=dt)
        start = torch.tensor([100 * (b + 1) for b in range(4)], dtype=torch.int32)
        acc = [(b * 2 + 1) % 6 for b in range(nb)]
        verdict = torch.tensor([acc, [5 - a for a in acc], acc], dtype=torch.int32)
        dr.advance(pos, start, verdict, 6, nb)
        want = [100 * (b + 1) + 1 + acc[b] + i for b in range(nb) for i in range(6)]
        assert pos.tolist() == want + [-7, -7]


def _batch(req=7, **over):
    b = dict(
        is_dummy_run=False,
        total_seqs_num_prefill=0,
        total_seqs_num=1,
        num_spec_step=5,
        req_ids=[req],
        multimodal_data={},
        replaying={},
        context_lens=[112],
        num_rejected=[0],
        out=True,
    )
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


class _Recorder:
    """``prologue.Prologue``'s stand-in: never armed past ``arm``, so the plain advance runs."""

    armed = False

    def __init__(self):
        self.arms = []

    def arm(self, on):
        self.arms.append(on)


def _round_world(dr, monkeypatch):
    """Stand-ins for the runner / token processor / sampler / builder / cache calling the decode round's hooks
    (``hi_round.HiRound``) where ATOM's do, and ``forward``: one forward as ``ModelRunner.forward`` runs them
    (prepare_input_ids -> builder (+ the model inputs' probe) -> run_model -> postprocess's sampler).
    """
    from atom.models.deepseek_v41.mono.round import HiRound

    log, queue, served_now, sampler_verdict = [], [], [True], [None]
    hook = SimpleNamespace(
        lean_host=True,
        runs_every_layer=True,
        served_widths={6, 12},
        split={6: (1, 6), 12: (2, 6)},
        bmax=2,
        pos=torch.tensor([100], dtype=torch.int32),
    )
    rnd = HiRound(device=dr.DeviceRound(hook, _Recorder()))

    class TP:  # ``tokenIDProcessor.prepare_input_ids``
        prev_rejected_num = prev_bonus_num = None

        def recv_spec_status(self):
            log.append("recv")
            return queue.pop(0)

        def prepare_input_ids(self, b, max_q, publication_group=None):
            rnd.before_input_ids(b, max_q)
            if b.produces_output():
                status = rnd.spec_status()
                self.prev_rejected_num, self.prev_bonus_num = (
                    self.recv_spec_status() if status is None else status
                )
            r = self.prev_rejected_num
            zeros = isinstance(r, np.ndarray) and not r.any()
            log.append(("ids", f"zeros{len(r)}" if zeros else r))
            return "ids"

    class Sampler:
        def forward(self, *a, **kw):
            log.append("verdict")
            return "tokens", sampler_verdict[0]

    class Cache:  # ``PagedAttentionCache.prepare_state``: a device-round step takes no probe
        cursor = pending = None

        def prepare_state(self, step, *, histories=True):
            if not getattr(step, "device_round", False):
                log.append("probe")

    cache = Cache()

    class Builder:  # ``_prepare_step``: the lean mark stands in for ``LeanHost``'s
        engram = None

        def _prepare_step(
            self, b, spans, rows, cache_, running_bs, running_tokens, **kw
        ):
            step = SimpleNamespace(
                positions=torch.zeros(6, dtype=torch.int64),
                tentative=True,
                lean_host=served_now[0],
                slots=torch.zeros(1, dtype=torch.int32),
            )
            log.append(("build", running_bs, running_tokens))
            md = SimpleNamespace(step=step, cache=cache_)
            rnd.laid_out(b, spans, rows, running_bs, running_tokens, md, served_now[0])
            return md, step.positions

    class Runner:
        def __init__(self):
            self.tokenID_processor, self.rejection_sampler = TP(), Sampler()

        def run_model(self, ids, b=None):
            log.append("replay")
            rnd.after_forward(self.tokenID_processor)
            return "logits", "hidden"

    runner, builder = Runner(), Builder()

    def forward(b, served=True, accepted=3):
        log.clear()
        served_now[0] = served and not b.total_seqs_num_prefill
        runner.tokenID_processor.prepare_input_ids(b, 6)
        md, _ = builder._prepare_step(b, (), [], cache, 1, 6)
        cache.prepare_state(md.step, histories=False)
        runner.run_model("ids", b)
        sampler_verdict[0] = torch.tensor(
            [[accepted], [5 - accepted], [accepted]], dtype=torch.int32
        )
        rnd.note_verdict(runner.rejection_sampler.forward())
        return list(log), md.step

    return forward, queue, runner, hook


def test_device_round_wrappers_in_forward_order(dr, monkeypatch):
    """A device-round step skips the verdict wait, advances, skips the probe and takes the owed wait after the
    launch, in queue order; a stale verdict never carries across a forward that left none; every other step runs as
    without the hooks."""
    forward, queue, runner, hook = _round_world(dr, monkeypatch)
    queue[:] = [("r0", "b0"), ("r1", "b1"), ("r2", "b2"), ("r3", "b3"), ("r4", "b4")]
    # the first decode: no verdict yet -> host path
    l1, s1 = forward(_batch())
    assert l1 == ["recv", ("ids", "r0"), ("build", 1, 6), "probe", "replay", "verdict"]
    assert not hasattr(s1, "device_round")
    hook.pos.fill_(100)
    # after a served verdict (accepted 3 in forward 1)
    l2, s2 = forward(_batch(), accepted=3)
    assert l2 == [("ids", None), ("build", 1, 6), "replay", "recv", "verdict"]
    assert getattr(s2, "device_round", False)
    assert s2.positions.tolist() == [104 + i for i in range(6)]
    assert runner.tokenID_processor.prev_rejected_num == "r1"
    # another request: host path, and the verdict of 7 is gone
    l3, s3 = forward(_batch(req=8))
    assert l3[:2] == ["recv", ("ids", "r2")] and "probe" in l3
    assert not hasattr(s3, "device_round")
    # 7 again, but forward 3 was request 8's: host path; not served -> no verdict kept
    l4, _ = forward(_batch(), served=False)
    # after an unserved forward of 7: host path
    l5, s5 = forward(_batch())
    assert l4[:2] == ["recv", ("ids", "r3")] and l5[:2] == ["recv", ("ids", "r4")]
    assert not hasattr(s5, "device_round") and not queue


def test_first_step_after_a_prefill(dr, monkeypatch):
    """The decode step right after its request's output-producing prefill builds from zero rejected / bonus without
    waiting and takes the owed wait after the launch (the queue stays in order); its positions are the host's; the
    step after it is a device-round step; a batch with one prefilled request gets the prefill's zeros; after a middle
    prefill chunk or a prefill of a request not in the step, the first step waits as before.
    """
    z = np.zeros(1, dtype=np.int32)
    forward, queue, runner, _ = _round_world(dr, monkeypatch)
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
    assert not hasattr(s1, "device_round")
    assert runner.tokenID_processor.prev_rejected_num is z
    l2, s2 = forward(_batch())  # then a device-round step, as before
    assert l2 == [("ids", None), ("build", 1, 6), "replay", "recv", "verdict"]
    assert getattr(s2, "device_round", False) and not queue

    # a batch: request 8 prefilled alone, then a decode of 7 and 8 at the batch width
    forward, queue, runner, _ = _round_world(dr, monkeypatch)
    queue[:] = [("p", "p"), (z, z)]
    forward(_batch(req=8, total_seqs_num_prefill=1))
    lm, sm = forward(_batch(total_seqs_num=2, req_ids=[7, 8]))
    assert lm[0] == ("ids", "zeros1") and lm[-2:] == ["recv", "verdict"]
    assert not hasattr(sm, "device_round") and not queue

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


# ---------------------------------------------------------------- the post-verify's routing


@pytest.fixture
def no_aiter_sampler(monkeypatch):
    """The rejection sampler's module (it imports aiter) as the constants ``PostVerify`` reads from it."""
    if "aiter" not in sys.modules:
        try:
            import aiter  # noqa: F401
        except ImportError:
            fake = SimpleNamespace(_SYNTHETIC_RNG_BASE_SEED=0x5EED)
            monkeypatch.setitem(sys.modules, "atom.model_ops.rejection_sampler", fake)


def test_post_verify_takes_the_in_graph_results_only_where_it_ran(
    monkeypatch, no_aiter_sampler
):
    """``verdict``: the in-graph results on a step laid out as the kernel assumed (its width ran the kernel, every
    request ``rq - 1`` drafts); the one-launch verdict otherwise; either reported to the decode round. ``take``
    advances the forced-acceptance step."""
    from atom.models.deepseek_v41.mono.round import head_verdict, post_verify

    split = {6: (1, 6), 12: (2, 6)}
    pv = post_verify.PostVerify((6, 12), split, torch.device("cpu"))
    assert pv.bufs[12]["target"].tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert pv.bufs[12]["bonus"].tolist() == [5, 11] and pv.bufs[12]["cu"].tolist() == [
        5,
        10,
    ]
    noted = []
    hook = SimpleNamespace(pv=pv, round=SimpleNamespace(note_verdict=noted.append))
    sampler = SimpleNamespace(synthetic_acceptance_rates=(1.0, 0.5), _synthetic_step=3)
    eager = ("eager",)
    monkeypatch.setattr(head_verdict, "verdict", lambda spec, ids, s: eager)

    def spec(drafts, steps=5):
        return SimpleNamespace(
            num_draft_tokens_np=np.asarray(drafts), num_spec_steps=steps
        )

    ids12 = torch.zeros(12, dtype=torch.int32)
    # the forward did not run the kernel at 12 rows: the eager verdict
    assert post_verify.verdict(hook, spec([5, 5]), ids12, sampler) is eager
    pv.launched.add(12)
    got = post_verify.verdict(hook, spec([5, 5]), ids12, sampler)
    assert got == (pv.bufs[12]["out"], pv.bufs[12]["verdict"], pv.bufs[12]["next"])
    assert pv.used == 12 and sampler._synthetic_step == 4 and noted == [eager, got]
    # another layout: the eager verdict, and the next step's skip wrappers see none taken
    assert post_verify.verdict(hook, spec([5, 3]), ids12, sampler) is eager
    assert post_verify.verdict(hook, spec([5, 5], steps=3), ids12, sampler) is eager
    assert pv.used is None


def test_post_verify_skips_the_commit_and_the_anchor_staging_where_it_ran(
    no_aiter_sampler,
):
    """On a step that took the in-graph results the builder's commit and the proposer's anchor staging skip their
    launches (``committed`` / ``anchors_staged``: the bookkeeping kept); on any other step they run as before.
    """
    from atom.models.deepseek_v41.mono.round import post_verify

    pv = post_verify.PostVerify((6,), {6: (1, 6)}, torch.device("cpu"))
    step = object()
    pending = SimpleNamespace(staged_on_device=True, step=step)
    md = SimpleNamespace(cache=SimpleNamespace(pending=pending), step=step)
    pv.used = 6
    assert pv.committed(md) and md.cache.pending is None
    assert (
        not pv.anchors_staged(2) and pv.used is None
    )  # another batch: staged by the proposer
    pv.used = 6
    assert pv.anchors_staged(1) and pv.used is None
    md.cache.pending = pending
    assert not pv.committed(md) and md.cache.pending is pending
    assert not pv.anchors_staged(1)


# ---------------------------------------------------------------- ATOM's Engram staging on served steps


def test_engram_staging_skips_its_lookup_on_a_served_step():
    """ATOM's ``EngramStaging`` with the hook's ``lookup_served``: on a step the hook serves it only advances the cursor
    (no snapshot, no side stream) and its join waits on nothing."""
    from atom.model_ops.engram.device.staging import EngramStaging

    calls = []
    staging = object.__new__(EngramStaging)
    staging._snapshot = lambda w: calls.append(("snapshot", w))
    staging._advance_cursor = lambda w: calls.append(("cursor", w))
    staging.lookup_served = lambda w: w in {6}
    staging.start(6)
    staging.join()
    assert calls == [("cursor", 6)]


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
        "prologue",
        "head_verdict",
        "lean_host",
        "embed_full",
        "hi_weights",
        "round/__init__",
    ):
        tree = ast.parse(_src(name).read_text())
        for n in tree.body:
            if isinstance(n, ast.ImportFrom):
                assert ".mono.hi." not in (n.module or "") + ".", (name, n.module)
    before = set(sys.modules)
    from atom.models.deepseek_v41.mono import hi_bridge, hi_gate  # noqa: F401
    from atom.models.deepseek_v41.mono import hi_runner  # noqa: F401

    new = set(sys.modules) - before
    assert not [m for m in new if ".mono.hi." in m + "."]
