"""Execute the real forward control flow with CPU stage doubles.

Only the method is compiled: importing ModelRunner requires the GPU kernel stack.
"""

import __future__
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _forward(namespace):
    path = Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
    module = ast.parse(path.read_text())
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    method.decorator_list = []
    code = compile(
        ast.Module(body=[method], type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)
    return namespace["forward"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "failure", [None, "prepare_model", "gpu_forward", "postprocess"]
)
def test_forward_trace_preserves_stage_execution_and_exceptions(enabled, failure):
    events = []
    output = object()
    error = RuntimeError("stage failed")
    prepared = tuple(object() for _ in range(6))
    logits, hidden = object(), object()
    batch = SimpleNamespace(is_dummy_run=False)

    @contextmanager
    def span(name):
        assert enabled, "disabled trace must not enter any context manager"
        events.append((name, "enter"))
        try:
            yield
        finally:
            events.append((name, "exit"))

    def stage(name, result):
        events.append(name)
        if failure == name:
            raise error
        return result

    def run_model(input_ids, actual_batch):
        assert input_ids is prepared[0] and actual_batch is batch
        return stage("gpu_forward", (logits, hidden))

    def postprocess(*args, **kwargs):
        assert args == (batch, logits, *prepared[1:5], hidden)
        assert kwargs == {"needs_independent_noise": prepared[5]}
        return stage("postprocess", output)

    runner = SimpleNamespace(
        _advance_forward_vars=lambda: events.append("advance"),
        _gate_staging_reuse=lambda: events.append("gate"),
        prepare_model=lambda actual_batch: stage("prepare_model", prepared),
        _mark_staging_h2d_enqueued=lambda: events.append("h2d"),
        run_model=run_model,
        _dp_draft_lockstep_active=lambda: False,
        _is_pure_middle_chunk=lambda actual_batch: False,
        postprocess=postprocess,
        _record_forward_vars_event=lambda: events.append("record"),
        _record_kv_cache_ready=lambda actual_batch: events.append("ready"),
    )
    forward = _forward(
        {
            "TTFT_TRACE_ENABLED": enabled,
            "ttft_trace_span": span,
            "get_pp_group": lambda: SimpleNamespace(world_size=1),
            "reset_forward_context": lambda: events.append("reset"),
        }
    )
    if failure is None:
        assert forward(runner, batch) is output
    else:
        with pytest.raises(RuntimeError) as caught:
            forward(runner, batch)
        assert caught.value is error

    expected = ["advance", "gate"]
    for name in ("prepare_model", "gpu_forward", "postprocess"):
        if enabled:
            expected.append((f"ttft[{name}]", "enter"))
        expected.append(name)
        if enabled:
            expected.append((f"ttft[{name}]", "exit"))
        if failure == name:
            break
        if name == "prepare_model":
            expected.append("h2d")
    if failure is None:
        expected.extend(["reset", "record", "ready"])
    assert events == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_trace_helper_uses_startup_flag(monkeypatch, enabled):
    import torch.profiler

    from atom.model_engine import run_labels

    # Runtime environment changes must not change the selected startup mode.
    monkeypatch.setattr(run_labels, "TTFT_TRACE_ENABLED", enabled)
    monkeypatch.setenv("ATOM_TTFT_TRACE", str(int(not enabled)))
    events = []

    @contextmanager
    def record_function(name):
        events.append((name, "enter"))
        try:
            yield
        finally:
            events.append((name, "exit"))

    monkeypatch.setattr(torch.profiler, "record_function", record_function)

    def forbidden():
        raise AssertionError("trace must not reread the environment")

    monkeypatch.setitem(
        run_labels.envs.environment_variables, "ATOM_TTFT_TRACE", forbidden
    )
    error = RuntimeError("work failed")
    with pytest.raises(RuntimeError) as caught:
        with run_labels.ttft_trace_span("stage"):
            raise error
    assert caught.value is error
    assert events == ([("stage", "enter"), ("stage", "exit")] if enabled else [])
