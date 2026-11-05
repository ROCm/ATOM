# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The weight loader's threads take one OpenMP thread each, and only they do:
a thread started while they exist (the online-quant pool, the checkpoint
prefetch) keeps the process's intra-op default."""

import concurrent.futures
import threading

import pytest
import torch

from atom.model_loader import loading_core
from atom.model_loader.loading_core import _one_omp_thread


def test_loader_thread_resolves_omp_from_torch_extension(monkeypatch):
    opened = []
    calls = []

    class Setter:
        argtypes = None
        restype = object()

        def __call__(self, value):
            calls.append(value)

    class Runtime:
        omp_set_num_threads = Setter()

    def open_library(path):
        opened.append(path)
        return Runtime()

    monkeypatch.setattr(loading_core.ctypes, "CDLL", open_library)

    _one_omp_thread()

    assert opened == [torch._C.__file__]
    assert calls == [1]


def _work_then_report():
    a = torch.randn(512, 512)
    _ = a @ a
    _ = a.to(torch.float16).to(torch.float32)
    return torch.get_num_threads()


def test_a_loader_thread_runs_one_omp_thread_and_no_other_thread_does():
    default = torch.get_num_threads()
    if default < 2:
        pytest.skip("one intra-op thread already: nothing to tell apart")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, initializer=_one_omp_thread
    ) as pool:
        assert pool.submit(torch.get_num_threads).result() == 1
        assert pool.submit(_work_then_report).result() == 1
        seen = {}
        other = threading.Thread(target=lambda: seen.update(n=torch.get_num_threads()))
        other.start()
        other.join()
        assert seen["n"] == default
        assert torch.get_num_threads() == default
