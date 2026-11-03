# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The weight loader's threads take one OpenMP thread each, and only they do:
a thread started while they exist (the online-quant pool, the checkpoint
prefetch) keeps the process's intra-op default."""

import concurrent.futures
import logging
import threading

import pytest
import torch

from atom.model_loader import loading_core
from atom.model_loader.loading_core import _one_omp_thread


# ---------------------------------------------------------------------------
# Fixture: any test that touches the main thread's OMP count must restore it.
# ---------------------------------------------------------------------------
@pytest.fixture()
def main_thread_omp_guard():
    """Save and unconditionally restore the main thread's OpenMP count.

    ``_one_omp_thread`` is designed for worker threads.  Tests that call it
    on the main thread (the monkeypatch / unit tests) would otherwise leave
    the main thread at 1 OMP thread, causing every later integration test
    to ``pytest.skip`` ("one intra-op thread already").
    """
    default = torch.get_num_threads()
    # The tests patch loading_core._omp_set_num_threads, so keep the real one.
    real_set = loading_core._omp_set_num_threads
    yield default
    if real_set is not None:
        real_set(default)


# ---------------------------------------------------------------------------
# Unit tests (mock / monkeypatch, run on the main thread)
# ---------------------------------------------------------------------------
def test_omp_resolved_from_torch_extension():
    """The module-level symbol comes from the same library torch uses."""
    assert loading_core._omp_set_num_threads is not None
    assert callable(loading_core._omp_set_num_threads)


@pytest.mark.parametrize("error", [AttributeError, OSError, TypeError])
def test_resolve_returns_none_when_the_symbol_is_unavailable(monkeypatch, error):
    """A torch without the symbol must not break the import of loading_core."""

    def _fail(path):
        raise error("unavailable")

    monkeypatch.setattr(loading_core.ctypes, "CDLL", _fail)
    assert loading_core._resolve_omp_set_num_threads() is None


def test_one_omp_thread_succeeds_on_first_attempt(monkeypatch, main_thread_omp_guard):
    """Happy path: the setter works immediately, loop exits after one call."""
    calls = []
    real_set = loading_core._omp_set_num_threads

    def _spy(n):
        calls.append(n)
        real_set(n)

    monkeypatch.setattr(loading_core, "_omp_set_num_threads", _spy)
    _one_omp_thread()
    assert calls == [1]


def test_one_omp_thread_retries_when_first_set_is_overwritten(
    monkeypatch, main_thread_omp_guard
):
    """If the first set is clobbered (simulating lazy init), the retry fixes it."""
    default = main_thread_omp_guard
    if default < 2:
        pytest.skip("one intra-op thread already: nothing to tell apart")
    calls = []
    real_set = loading_core._omp_set_num_threads

    def _clobber_once(n):
        calls.append(n)
        real_set(n)
        if len(calls) == 1:
            real_set(default)

    monkeypatch.setattr(loading_core, "_omp_set_num_threads", _clobber_once)
    _one_omp_thread()
    assert calls == [1, 1]
    assert torch.get_num_threads() == 1


def test_one_omp_thread_warns_when_both_attempts_fail(
    monkeypatch, main_thread_omp_guard, caplog
):
    """If neither attempt sticks, a warning is logged (not an exception)."""
    default = main_thread_omp_guard
    if default < 2:
        pytest.skip("one intra-op thread already: nothing to tell apart")

    monkeypatch.setattr(
        loading_core, "_omp_set_num_threads", lambda n: None  # no-op: never sticks
    )
    with caplog.at_level(logging.WARNING, logger="atom"):
        _one_omp_thread()
    assert any("wanted 1" in r.message for r in caplog.records)


def test_one_omp_thread_survives_unresolved_symbol(monkeypatch, caplog):
    """If the symbol couldn't be resolved, the function warns and returns."""
    monkeypatch.setattr(loading_core, "_omp_set_num_threads", None)
    with caplog.at_level(logging.WARNING, logger="atom"):
        _one_omp_thread()
    assert any("could not be resolved" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration tests (real OMP, worker threads — main thread untouched)
# ---------------------------------------------------------------------------
def test_a_loader_thread_runs_one_omp_thread_and_no_other_thread_does():
    default = torch.get_num_threads()
    if default < 2:
        pytest.skip("one intra-op thread already: nothing to tell apart")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, initializer=_one_omp_thread
    ) as pool:
        assert pool.submit(torch.get_num_threads).result() == 1
        seen = {}
        other = threading.Thread(target=lambda: seen.update(n=torch.get_num_threads()))
        other.start()
        other.join()
        assert seen["n"] == default
        assert torch.get_num_threads() == default


def test_omp_setting_survives_tensor_work():
    """After ``_one_omp_thread``, real ATen work must not reset the count.

    Tensor operations trigger ATen paths that can interact with the OpenMP
    runtime.  The setting must survive a matmul and a dtype round-trip —
    exactly the kind of work a loader thread performs.
    """
    default = torch.get_num_threads()
    if default < 2:
        pytest.skip("one intra-op thread already: nothing to tell apart")

    def _work_then_report():
        a = torch.randn(512, 512)
        b = torch.randn(512, 512)
        _ = a @ b
        _ = a.to(torch.float16).to(torch.float32)
        return torch.get_num_threads()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, initializer=_one_omp_thread
    ) as pool:
        assert pool.submit(_work_then_report).result() == 1
        assert pool.submit(_work_then_report).result() == 1

    assert torch.get_num_threads() == default
