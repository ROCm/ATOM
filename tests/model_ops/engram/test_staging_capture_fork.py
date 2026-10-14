# SPDX-License-Identifier: MIT
"""The vLLM staging keeps core's fork and moves only where it is ordered.

`EngramStaging.start` opens a side stream whose joins are per layer and lazy
-- `consume` waits on a layer's event where that layer needs its rows -- so
the fork spans the whole forward. vLLM's breakable CUDA graph ends a capture
segment at every attention break, and refuses both halves of that:

    hipErrorStreamCaptureUnjoined     a segment ended with work still forked
    hipErrorStreamCaptureIsolation    a captured stream waited on an event
                                      recorded off-capture

The answer is not to stop forking. The overlap between a largely-UVA gather
and the layers consuming it is the whole point, and upstream's own V4.1 keeps
its side stream and puts the orderings in eager breaks instead.

Core carries none of this: `atom/model_ops/engram/device/staging.py` is
untouched by this PR, and the plugin replaces six methods on the class.
These tests pin the two properties that make that replacement correct --
the fork survives, and `start` does not fold the snapshot or the cursor
advance into the broken-out region, where a replay would re-run them and
advance the cursor twice.
"""

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")

ROOT = pathlib.Path(__file__).resolve().parents[3]
PLUGIN = ROOT / "atom/plugin/vllm/v41_core_overrides.py"


def _calls_in(func_name: str, source: str) -> list[str]:
    fn = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, (ast.FunctionDef,)) and n.name == func_name
    )
    return [
        n.func.attr
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]


def test_core_staging_is_untouched():
    """The whole point of the duplication: no seam in the shared file."""
    import atom.model_ops.engram.device.staging as staging_mod

    source = pathlib.Path(staging_mod.__file__).read_text()
    for absent in ("_forks", "set_engram_staging_cls", "_await_layer", "_prefetch"):
        assert absent not in source, f"core staging still carries {absent}"


def test_the_plugin_start_keeps_state_writes_out_of_the_break():
    """`_snapshot` and `_advance_cursor` must be called from `start` itself.

    Folded into `_prefetch` they would sit inside the eager break, which a
    replay re-runs -- and re-running `_advance_cursor` advances the cursor a
    second time for the same tokens. That is silent: no error, the request is
    simply one step ahead of itself from then on.
    """
    calls = _calls_in("start", PLUGIN.read_text())
    assert "_snapshot" in calls
    assert "_advance_cursor" in calls
    assert "_prefetch" in calls
    assert calls.index("_advance_cursor") < calls.index("_prefetch")


def test_the_plugin_prefetch_still_forks():
    """A version that issued the gather on the compute stream would pass every
    other assertion here while giving up the overlap, with nothing to say so."""
    source = PLUGIN.read_text()
    calls = _calls_in("_prefetch", source)
    assert "wait_stream" in calls, "the side stream no longer waits on its parent"
    assert "stream" in calls, "the gather is no longer issued under a stream guard"


def test_the_three_boundaries_are_wrapped():
    from atom.plugin.vllm import v41_core_overrides as mod

    for name in ("_prefetch", "_await_layer", "_await_fork"):
        assert hasattr(mod.VllmEngramStaging, name)


def test_installing_is_idempotent_and_replaces_the_class_methods(monkeypatch):
    from atom.model_ops.engram.device.staging import EngramStaging
    from atom.plugin.vllm import v41_core_overrides as mod

    names = ("start", "consume", "join", "_prefetch", "_await_layer", "_await_fork")
    original = dict(vars(EngramStaging))
    target = type("Staging", (), {name: object() for name in names[:3]})
    monkeypatch.setattr(mod, "EngramStaging", target)
    mod.install_vllm_engram_staging()
    wrapped = [getattr(target, name) for name in names]
    mod.install_vllm_engram_staging()
    assert [getattr(target, name) for name in names] == wrapped
    assert wrapped == [getattr(mod.VllmEngramStaging, name) for name in names]
    assert getattr(target, mod._STAGING_MARKER)
    assert dict(vars(EngramStaging)) == original


def test_join_is_once_per_stage_and_consume_does_not_repeat_external_wait():
    from atom.plugin.vllm.v41_core_overrides import VllmEngramStaging

    staging = VllmEngramStaging.__new__(VllmEngramStaging)
    staging.host = SimpleNamespace(_tp_group=None)
    staging.done = {0: object()}
    for name in (
        "_snapshot",
        "_advance_cursor",
        "_prefetch",
        "_await_layer",
        "_await_fork",
    ):
        setattr(staging, name, Mock())
    staging.start(4)
    staging.consume(0, 4)
    staging._await_layer.assert_called_once_with(staging.done[0])
    staging.join()
    staging.join()
    staging.consume(0, 4)
    staging._await_fork.assert_called_once_with()
    staging._await_layer.assert_called_once()
    staging.start(2)
    staging.consume(0, 2)
    assert staging._await_layer.call_count == 2
    assert staging._snapshot.call_count == staging._advance_cursor.call_count == 2
    assert staging._prefetch.call_count == 2
