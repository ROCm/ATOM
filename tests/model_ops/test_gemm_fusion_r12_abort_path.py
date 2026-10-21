# SPDX-License-Identifier: MIT
# Torch-free tests of the R12 abort path: the runner's verdict handling and the scheduler's
# reaction. The real methods are extracted from source with ast and run against stand-ins (the
# scheduler tests in tests/test_scheduler.py need torch and run in CI only).
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _class_node(rel, cls_name):
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    return next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )


def _extract(rel, cls_name, methods, namespace):
    cls = _class_node(rel, cls_name)
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    assert {n.name for n in cls.body} == set(methods), "a method moved or was renamed"
    cls.bases, cls.decorator_list = [], []
    module = ast.Module(
        [ast.parse("from __future__ import annotations").body[0], cls], []
    )
    ns = dict(namespace)
    exec(  # noqa: S102 - real source, extracted by ast, run against stand-ins
        compile(ast.fix_missing_locations(module), rel, "exec"), ns
    )
    return ns[cls_name]


# ----------------------------------------------------------------------------------- runner side
RUNNER = "atom/model_engine/model_runner.py"
RUNNER_METHODS = {"_gemm_fusion_check_step", "_apply_gemm_fusion_verdict"}


def _runner(enabled=True):
    runner = _extract(RUNNER, "ModelRunner", RUNNER_METHODS, {})()
    runner.config = SimpleNamespace(gemm_fusion=SimpleNamespace(enabled=enabled))
    runner.tokenID_processor = SimpleNamespace(cleaned=0)
    runner.tokenID_processor.clean = lambda: setattr(
        runner.tokenID_processor, "cleaned", runner.tokenID_processor.cleaned + 1
    )
    return runner


class _CheckStep:
    """Stands in for ``gemm_fusion.check_step``: records its argument, returns ``result``."""

    def __init__(self):
        self.calls, self.result = [], None

    def __call__(self, ids=()):
        self.calls.append(ids)
        return self.result


@pytest.fixture
def check_step(monkeypatch):
    """Make ``from atom.model_ops import gemm_fusion`` (inside the runner method) resolve to a fake
    package, so the test needs neither torch nor the real package."""
    import sys
    import types

    fake = _CheckStep()
    fake_pkg = types.ModuleType("atom.model_ops")
    fake_pkg.gemm_fusion = SimpleNamespace(check_step=fake)
    monkeypatch.setitem(sys.modules, "atom.model_ops", fake_pkg)
    return fake


def test_the_check_wrapper_is_a_noop_when_fusion_is_off(check_step):
    batch = SimpleNamespace(req_ids=[1, 2], is_dummy_run=False)
    assert _runner(enabled=False)._gemm_fusion_check_step(batch) is None
    assert check_step.calls == []


def test_the_check_wrapper_passes_the_requests_of_the_step_and_returns_the_verdict(
    check_step,
):
    runner = _runner()
    batch = SimpleNamespace(req_ids=[4, 5, 6], is_dummy_run=False)
    assert runner._gemm_fusion_check_step(batch) is None
    check_step.result = [4, 5]
    assert runner._gemm_fusion_check_step(batch) == [4, 5]
    assert check_step.calls == [(4, 5, 6), (4, 5, 6)]


def test_a_dummy_batch_contributes_no_request_ids(check_step):
    _runner()._gemm_fusion_check_step(SimpleNamespace(req_ids=[1], is_dummy_run=True))
    assert check_step.calls == [()]


def test_no_verdict_leaves_the_output_and_the_deferred_state_alone():
    runner = _runner()
    out = SimpleNamespace(aborted_req_ids=None)
    assert runner._apply_gemm_fusion_verdict(out, None) is out
    assert out.aborted_req_ids is None and runner.tokenID_processor.cleaned == 0


@pytest.mark.parametrize("ids", [[3, 4], []])
def test_a_verdict_is_attached_and_drops_the_deferred_output_state(ids):
    """An empty verdict (nothing in the window) is still a verdict: the reset ran."""
    runner = _runner()
    out = SimpleNamespace(aborted_req_ids=None)
    assert runner._apply_gemm_fusion_verdict(out, ids) is out
    assert out.aborted_req_ids == ids
    assert runner.tokenID_processor.cleaned == 1


# -------------------------------------------------------------------------------- scheduler side
SCHED = "atom/model_engine/scheduler.py"
SCHED_METHODS = {"postprocess", "_abort_for_device_error"}


def _scheduler(events, live=()):
    ns = {"logger": logging.getLogger("atom.test_scheduler")}
    sched = _extract(SCHED, "Scheduler", SCHED_METHODS, ns)()

    def abort_request(rid):
        events.append(("abort", rid))
        return rid in live

    sched.abort_request = abort_request
    sched._postprocess_impl = lambda seqs, out, stream, batch: (
        events.append(("body", getattr(out, "aborted_req_ids", None))) or ["finished"]
    )
    sched.block_manager = SimpleNamespace(
        clear_cache=lambda: events.append(("clear_cache",))
    )
    return sched


def _out(aborted):
    return SimpleNamespace(aborted_req_ids=aborted)


def test_without_a_verdict_postprocess_is_the_body_and_nothing_else():
    events = []
    assert _scheduler(events).postprocess([], _out(None)) == ["finished"]
    assert events == [("body", None)]


def test_postprocess_tolerates_an_output_object_without_the_field():
    """Other producers of an output (tests, rollout) predate the field; and PP / dummy callers."""
    events = []
    sched = _scheduler(events)
    assert sched.postprocess([], SimpleNamespace()) == ["finished"]
    assert sched.postprocess([], None) == ["finished"]
    assert [e[0] for e in events] == ["body", "body"]


def test_a_verdict_aborts_before_the_body_and_clears_the_prefix_cache_after_it():
    events = []
    sched = _scheduler(events, live={11, 12})
    assert sched.postprocess([], _out([11, 12, 13])) == ["finished"]
    assert events == [
        ("abort", 11),
        ("abort", 12),
        ("abort", 13),  # already finished: abort_request says not found
        ("body", [11, 12, 13]),
        ("clear_cache",),
    ]


def test_an_empty_verdict_still_drops_the_prefix_cache():
    """The reset ran, KV of hashed blocks from the unchecked steps may be corrupt: clear."""
    events = []
    _scheduler(events).postprocess([], _out([]))
    assert events == [("body", []), ("clear_cache",)]


def test_the_scheduler_reports_how_many_requests_it_could_still_abort(caplog):
    events = []
    sched = _scheduler(events, live={1})
    with caplog.at_level(logging.ERROR, logger="atom.test_scheduler"):
        sched.postprocess([], _out([1, 2]))
    assert "aborted 1 of 2 request(s)" in caplog.text


# ------------------------------------------------------------------------- the output object
def test_the_output_object_carries_the_verdict_and_defaults_to_none():
    cls = _class_node(SCHED, "ScheduledBatchOutput")
    module = ast.Module(
        [ast.parse("from __future__ import annotations").body[0], cls], []
    )
    ns = {}
    exec(compile(ast.fix_missing_locations(module), SCHED, "exec"), ns)  # noqa: S102
    out_cls = ns["ScheduledBatchOutput"]
    plain = out_cls([1], [(5,)], None, None, None)
    assert plain.aborted_req_ids is None
    verdict = out_cls([1], [], None, None, None, aborted_req_ids=[1])
    assert verdict.aborted_req_ids == [1]
