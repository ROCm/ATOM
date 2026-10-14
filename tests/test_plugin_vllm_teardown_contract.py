# SPDX-License-Identifier: MIT
"""Teardown of the collectives the vLLM plugin creates, without importing vLLM.

The defect these cover is not that a resource leaks -- the process is exiting
anyway -- but *when* it is freed. aiter frees its IPC buffers through
``torch.ops.aiter``, and by the time GC reaches a communicator's ``__del__`` at
interpreter finalization that operator namespace has already been deregistered,
so every worker ends a run with an AttributeError traceback. Closing while
torch is still alive is the whole fix, which makes "did it run, and early
enough" the thing worth asserting.

Executed by AST extraction so the real production functions are the ones under
test on a machine with no GPU, no vLLM and no aiter.
"""

import ast
import functools
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/tp_group_reuse.py"
_FUNCTIONS = ("shutdown_aiter_dist_from_vllm", "apply_vllm_worker_shutdown_patch")


def _load():
    """Execute the real teardown functions with nothing but stdlib in scope."""
    tree = ast.parse(SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS
    ]
    assert {node.name for node in functions} == set(_FUNCTIONS)
    scope = {
        "functools": functools,
        "logger": logging.getLogger("atom.test.teardown"),
        "_ATOM_OWNED_TP": None,
        "_ATOM_INSTALLED_GROUPS": {},
    }
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
        scope,
    )
    return scope


def _install(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def _fake_aiter(monkeypatch):
    """A stand-in for ``aiter.dist.parallel_state``; the real one needs a GPU."""
    parallel_state = ModuleType("aiter.dist.parallel_state")
    parallel_state._TP = None
    parallel_state._PP = None
    parallel_state._DP = None
    parallel_state._EP = None
    dist = ModuleType("aiter.dist")
    dist.parallel_state = parallel_state
    aiter = ModuleType("aiter")
    aiter.dist = dist
    _install(monkeypatch, "aiter", aiter)
    _install(monkeypatch, "aiter.dist", dist)
    _install(monkeypatch, "aiter.dist.parallel_state", parallel_state)
    return parallel_state


def _borrowed():
    """Objects owned by vLLM. Every attribute is a Mock so any call is visible."""
    return SimpleNamespace(
        pynccl=Mock(name="pynccl"),
        cpu_group=Mock(name="cpu_group"),
        device_group=Mock(name="device_group"),
        pp=Mock(name="pp"),
        dp=Mock(name="dp"),
        ep=Mock(name="ep"),
    )


def _adapter(borrowed, ca=True, qr=True):
    communicator = SimpleNamespace(
        ca_comm=Mock(name="ca_comm") if ca else None,
        qr_comm=Mock(name="qr_comm") if qr else None,
        pynccl_comm=borrowed.pynccl,
        cpu_group=borrowed.cpu_group,
        device_group=borrowed.device_group,
    )
    return SimpleNamespace(
        device_communicator=communicator,
        cpu_group=borrowed.cpu_group,
        device_group=borrowed.device_group,
    )


def _initialized(monkeypatch, **kwargs):
    """The state `init_aiter_dist_from_vllm` leaves behind on its success path."""
    aiter_ps = _fake_aiter(monkeypatch)
    scope = _load()
    borrowed = _borrowed()
    adapter = _adapter(borrowed, **kwargs)
    aiter_ps._TP = adapter
    aiter_ps._PP = borrowed.pp
    aiter_ps._DP = borrowed.dp
    aiter_ps._EP = borrowed.ep
    scope["_ATOM_OWNED_TP"] = adapter
    scope["_ATOM_INSTALLED_GROUPS"] = {
        "_TP": adapter,
        "_PP": borrowed.pp,
        "_DP": borrowed.dp,
        "_EP": borrowed.ep,
    }
    return SimpleNamespace(
        scope=scope,
        teardown=scope["shutdown_aiter_dist_from_vllm"],
        install=scope["apply_vllm_worker_shutdown_patch"],
        aiter_ps=aiter_ps,
        adapter=adapter,
        borrowed=borrowed,
        communicator=adapter.device_communicator,
    )


def _assert_borrowed_untouched(borrowed):
    for name, obj in vars(borrowed).items():
        for method in ("destroy", "close", "shutdown", "abort"):
            assert not getattr(obj, method).called, f"{name}.{method} was called"


def test_uninitialized_teardown_touches_nothing(monkeypatch):
    """TP=1 and the DP+EP fallback never build an adapter; aiter owns its own."""
    aiter_ps = _fake_aiter(monkeypatch)
    scope = _load()
    sentinel = Mock(name="someone elses TP")
    aiter_ps._TP = sentinel

    assert scope["shutdown_aiter_dist_from_vllm"]() is False
    assert aiter_ps._TP is sentinel
    _assert_borrowed_untouched(SimpleNamespace(sentinel=sentinel))


def test_teardown_closes_owned_collectives_once(monkeypatch):
    live = _initialized(monkeypatch)
    communicator = live.communicator
    ca, qr = communicator.ca_comm, communicator.qr_comm

    assert live.teardown() is True
    ca.close.assert_called_once_with()
    qr.close.assert_called_once_with()

    # A second call must not reach a freed pointer a second time.
    assert live.teardown() is False
    ca.close.assert_called_once_with()
    qr.close.assert_called_once_with()


def test_teardown_releases_references_but_not_borrowed_resources(monkeypatch):
    live = _initialized(monkeypatch)
    communicator = live.communicator

    assert live.teardown() is True

    # Dropping the last reference lets the closed object be collected now,
    # rather than at finalization where torch.ops.aiter is already gone.
    assert communicator.ca_comm is None
    assert communicator.qr_comm is None
    assert communicator.pynccl_comm is None
    assert live.adapter.device_communicator is None
    # Every aiter global this module installed is withdrawn.
    assert (
        live.aiter_ps._TP,
        live.aiter_ps._PP,
        live.aiter_ps._DP,
        live.aiter_ps._EP,
    ) == (
        None,
        None,
        None,
        None,
    )
    # vLLM owns the process groups and the PyNccl communicator; it destroys
    # them itself a few lines later in destroy_model_parallel().
    _assert_borrowed_untouched(live.borrowed)


def test_teardown_leaves_globals_a_later_owner_replaced(monkeypatch):
    live = _initialized(monkeypatch)
    successor = Mock(name="successor TP")
    live.aiter_ps._TP = successor

    assert live.teardown() is True

    assert live.aiter_ps._TP is successor
    assert live.aiter_ps._PP is None
    _assert_borrowed_untouched(live.borrowed)


@pytest.mark.parametrize("failing", ["ca_comm", "qr_comm"])
def test_one_failing_close_does_not_strand_the_others(monkeypatch, failing):
    live = _initialized(monkeypatch)
    communicator = live.communicator
    getattr(communicator, failing).close.side_effect = RuntimeError("dispose failed")
    other = "qr_comm" if failing == "ca_comm" else "ca_comm"
    survivor = getattr(communicator, other)

    assert live.teardown() is True

    survivor.close.assert_called_once_with()
    assert live.aiter_ps._TP is None
    assert live.teardown() is False


@pytest.mark.parametrize("ca,qr", [(True, False), (False, True), (False, False)])
def test_teardown_handles_absent_collectives(monkeypatch, ca, qr):
    """ca_comm is None when custom all-reduce is off; qr_comm when world_size==1."""
    live = _initialized(monkeypatch, ca=ca, qr=qr)

    assert live.teardown() is True
    assert live.aiter_ps._TP is None


def _worker_module(monkeypatch, shutdown):
    class Worker:
        pass

    Worker.shutdown = shutdown
    module = ModuleType("vllm.v1.worker.gpu_worker")
    module.Worker = Worker
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        _install(monkeypatch, name, ModuleType(name))
    _install(monkeypatch, "vllm.v1.worker.gpu_worker", module)
    return Worker


def test_patch_runs_teardown_after_vllms_own_shutdown(monkeypatch):
    """Order matters: the model runner clears the CUDA graphs that read these."""
    live = _initialized(monkeypatch)
    order = []

    def shutdown(self):
        order.append("vllm")
        return "returned"

    worker_cls = _worker_module(monkeypatch, shutdown)
    assert live.install() == 1

    ca = live.adapter.device_communicator.ca_comm
    ca.close.side_effect = lambda: order.append("atom")
    assert worker_cls().shutdown() == "returned"

    assert order == ["vllm", "atom"]
    assert live.aiter_ps._TP is None


def test_patch_tears_down_even_when_vllms_shutdown_raises(monkeypatch):
    live = _initialized(monkeypatch)
    # Held before the call: teardown drops the communicator off the adapter,
    # so afterwards there is no path from it to what was closed.
    ca = live.adapter.device_communicator.ca_comm

    def shutdown(self):
        raise RuntimeError("vLLM shutdown failed")

    worker_cls = _worker_module(monkeypatch, shutdown)
    assert live.install() == 1

    with pytest.raises(RuntimeError, match="vLLM shutdown failed"):
        worker_cls().shutdown()

    ca.close.assert_called_once_with()
    assert live.adapter.device_communicator is None
    assert live.aiter_ps._TP is None


def test_patch_never_converts_a_clean_shutdown_into_a_crash(monkeypatch):
    live = _initialized(monkeypatch)
    worker_cls = _worker_module(monkeypatch, lambda self: "returned")
    assert live.install() == 1
    monkeypatch.delitem(sys.modules, "aiter.dist")

    assert worker_cls().shutdown() == "returned"


def test_patch_is_installed_once_and_reports_its_coverage(monkeypatch):
    scope = _load()
    worker_cls = _worker_module(monkeypatch, lambda self: None)

    assert scope["apply_vllm_worker_shutdown_patch"]() == 1
    patched = worker_cls.shutdown
    # A second registration must not wrap the wrapper.
    assert scope["apply_vllm_worker_shutdown_patch"]() == 0
    assert worker_cls.shutdown is patched


def test_patch_reports_zero_when_vllm_is_absent(monkeypatch):
    scope = _load()
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_worker"):
        monkeypatch.setitem(sys.modules, name, None)

    assert scope["apply_vllm_worker_shutdown_patch"]() == 0
