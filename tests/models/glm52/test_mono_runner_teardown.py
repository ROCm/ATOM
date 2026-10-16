# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dropping a GLM mono runner releases this rank's peer IPC mappings.

`Glm52MonoRunner.close()` has no caller anywhere in ATOM -- nothing tears a
runner down explicitly -- so the only thing that ever closed an op was
`_build`'s rollback path. A runner that is simply dropped, which is what a
model reload does, left every `SymmetricPeerBuffer` its ops own with its
remote mappings still open. The K3 and V4.1 runners do not have this problem
because `owned_peer_buffer` gives them a `weakref.finalize`; GLM cannot use
that helper (its peer buffers belong to the per-layer ops, not the runner), so
it installs the equivalent finalizer itself.

The subtle part, and the reason these are behavioural tests rather than a
reading of the constructor: a `weakref.finalize` whose callback referenced the
runner would hold a strong reference to it, so the runner would never be
collected and the finalizer would never fire -- a fix that looks right and
does nothing. Dropping the runner and asserting the ops closed is the only
thing that tells those two apart.
"""

import gc
from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl", reason="the GLM mono runner is FlyDSL source")
pytest.importorskip("aiter", reason="the GLM mono runner imports aiter.dist")

import atom.models.glm52.mono.runner as runner_mod
from atom.models.glm52.mono.runner import Glm52MonoRunner, _close_ops


class _Op:
    def __init__(self, log, name):
        self._log, self._name = log, name

    def close(self):
        self._log.append(self._name)


def _runner(monkeypatch):
    """A real `Glm52MonoRunner.__init__`, with only the deployment lookups
    stubbed -- the finalizer under test is installed there."""
    monkeypatch.setattr(runner_mod, "native_config", lambda **kw: object())
    monkeypatch.setattr(runner_mod, "fused_indexer_refusal", lambda cfg: "off")
    monkeypatch.setattr(runner_mod, "indexer_cp_refusal", lambda cfg: "off")
    return Glm52MonoRunner(
        SimpleNamespace(model=None),
        SimpleNamespace(
            speculative_config=None,
            tensor_parallel_size=4,
            kv_cache_dtype="fp8",
        ),
    )


def test_dropping_the_runner_closes_every_build(monkeypatch):
    log = []
    runner = _runner(monkeypatch)
    runner._ops[(0, 4)] = _Op(log, "base")
    runner._fused_ops[(0, 4)] = _Op(log, "fused")
    del runner
    gc.collect()
    assert sorted(log) == ["base", "fused"]


def test_the_finalizer_does_not_keep_the_runner_alive(monkeypatch):
    # The failure this guards is a finalizer that holds the runner through its
    # own callback: everything looks installed, nothing is ever released.
    import weakref

    runner = _runner(monkeypatch)
    ref = weakref.ref(runner)
    del runner
    gc.collect()
    assert ref() is None, "the runner outlived its own finalizer"


def test_a_build_added_after_construction_is_still_closed(monkeypatch):
    # The finalizer captures the dicts, not a snapshot of their contents, so
    # ops built later -- which is every op, since `_build` runs long after
    # `__init__` -- must still be closed.
    log = []
    runner = _runner(monkeypatch)
    for width in (4, 8):
        runner._ops[(0, width)] = _Op(log, f"w{width}")
    del runner
    gc.collect()
    assert sorted(log) == ["w4", "w8"]


def test_close_is_idempotent_so_the_finalizer_after_it_is_a_no_op(monkeypatch):
    # `close()` and the finalizer run the same code, and a runner that was
    # closed explicitly is still dropped afterwards. The second run must not
    # close an op twice.
    log = []
    runner = _runner(monkeypatch)
    runner._ops[(0, 4)] = _Op(log, "base")
    runner.close()
    del runner
    gc.collect()
    assert log == ["base"]


def test_close_ops_empties_both_dicts():
    log = []
    ops = {(0, 4): _Op(log, "base")}
    fused = {(0, 4): _Op(log, "fused")}
    _close_ops(ops, fused)
    assert sorted(log) == ["base", "fused"]
    assert ops == {} and fused == {}
