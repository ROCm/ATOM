# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A decode-shaped pass must run before `capture_cudagraph` records anything.

`warmup_model()` runs before `allocate_kv_cache()`, so its dummy batch is
prefill-only -- a decode shape would have no pool to attend. Whatever a
decode-only path initialises lazily therefore waits for the first call that
reaches it, and with nothing decode-shaped ahead of the capture loop, that
call is a recording.

Under capture that is fatal rather than slow. A host-device readback
(`.item()`, `.cpu()`, `synchronize()`) raises AND leaves the CUDA context
unusable for the rest of the process, so the failure surfaces later in
unrelated code naming neither capture nor its cause. A group's first
collective hangs with no timeout -- `warmup_exchange` covers one instance of
exactly that.

These tests read the source because the property is an ORDER inside a function
that needs a GPU, a loaded model and a bound KV pool to run. That the pass is
decode-shaped is not re-derived here: `dummy_execution(prefill=False)` builds
`total_tokens_num_prefill=0`, which
`test_spec_deferred_status_queue.py::test_a_dp_dummy_runs_at_the_schedulers_query_width`
pins by executing it.
"""

import ast
import pathlib

RUNNER = (
    pathlib.Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
)

# Substrings rather than exact names: a recording renamed or a new graph kind
# added should still be covered, since the invariant is about the phase, not
# about today's spelling of it.
RECORDS = ("record", "_build_capture")


def _capture_cudagraph() -> ast.FunctionDef:
    tree = ast.parse(RUNNER.read_text())
    fns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "capture_cudagraph"
    ]
    assert len(fns) == 1, f"expected one capture_cudagraph, found {len(fns)}"
    return fns[0]


def _calls(fn: ast.FunctionDef) -> list[tuple[int, str]]:
    return sorted(
        (n.lineno, n.func.attr)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    )


def test_capture_runs_a_decode_pass():
    assert any(name == "dummy_execution" for _, name in _calls(_capture_cudagraph()))


def test_the_decode_pass_asks_for_decode_not_prefill():
    """`prefill=False` by keyword. The default already is False, but a bare
    `dummy_execution()` here would read as "some dummy pass" and survive a
    later change of default -- the shape is the entire point of the call."""
    fn = _capture_cudagraph()
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "dummy_execution"
    ]
    assert calls, "no decode pass to check"
    for call in calls:
        kwargs = {k.arg: k.value for k in call.keywords}
        assert "prefill" in kwargs, "pass prefill= explicitly"
        assert isinstance(kwargs["prefill"], ast.Constant)
        assert kwargs["prefill"].value is False
        assert not call.args, "prefill positionally reads as a bare flag"


def test_the_decode_pass_precedes_every_recording():
    """The ordering is the whole fix: a pass after the capture loop warms
    nothing that the loop has not already recorded cold."""
    calls = _calls(_capture_cudagraph())
    warms = [ln for ln, name in calls if name == "dummy_execution"]
    records = [ln for ln, name in calls if any(r in name for r in RECORDS)]
    assert warms, "no decode pass"
    assert records, (
        "no recording call found -- RECORDS has gone stale and this test "
        "would pass vacuously"
    )
    assert min(warms) < min(records), (
        f"decode pass at line {min(warms)} runs after the first recording "
        f"at line {min(records)}"
    )
