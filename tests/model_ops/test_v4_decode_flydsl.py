# SPDX-License-Identifier: MIT
"""ATOM_V4_DECODE_FLYDSL: V4 BF16 decode through plugin_poc's FlyDSL kernel.

CPU only. plugin_poc is a third-party package, so its runtime is a stand-in;
the module under test is ATOM's own.
"""

import logging
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import atom.model_ops.v4_decode_flydsl as flydsl


class _Runtime:
    def __init__(self, covers=True):
        self.covers = covers
        self.calls = []

    def covers_v4_decode(self, q, unified_kv, kv_indices, kv_indptr, rope, scales):
        return self.covers

    def sparse_attn_v4_paged_decode(
        self, q, unified_kv, kv_indices, kv_indptr, sink, scale
    ):
        self.calls.append(q.shape)
        return "flydsl"


def _install(monkeypatch, runtime):
    package = types.ModuleType("ops.sparse_mla")
    package.runtime = runtime
    monkeypatch.setitem(sys.modules, "ops", types.ModuleType("ops"))
    monkeypatch.setitem(sys.modules, "ops.sparse_mla", package)
    monkeypatch.setitem(sys.modules, "ops.sparse_mla.runtime", runtime)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(flydsl, "_RUNTIME", None)
    monkeypatch.setattr(flydsl, "_WARM", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _call(rows=4, nnz=64):
    q = SimpleNamespace(device=SimpleNamespace(index=0), shape=(rows, 16, 512))
    unified_kv = SimpleNamespace(dtype=torch.bfloat16, shape=(1024, 512))
    kv_indices = torch.zeros(nnz, dtype=torch.int32)
    kv_indptr = torch.zeros(rows + 1, dtype=torch.int32)
    sink = torch.zeros(16)
    return q, unified_kv, kv_indices, kv_indptr, sink, 512**-0.5


def test_a_covered_call_runs_flydsl(monkeypatch):
    runtime = _Runtime()
    _install(monkeypatch, runtime)
    assert flydsl.flydsl_paged_decode(*_call()) == "flydsl"
    assert runtime.calls == [(4, 16, 512)]


def test_an_uncovered_call_keeps_atoms_path(monkeypatch):
    runtime = _Runtime(covers=False)
    _install(monkeypatch, runtime)
    assert flydsl.flydsl_paged_decode(*_call()) is None
    assert runtime.calls == []


def test_a_missing_package_warns_once_and_keeps_atoms_path(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "ops", None)  # import ops -> ImportError
    with caplog.at_level(logging.WARNING, logger=flydsl.logger.name):
        assert flydsl.flydsl_paged_decode(*_call()) is None
        assert flydsl.flydsl_paged_decode(*_call()) is None
    assert sum("not importable" in r.getMessage() for r in caplog.records) == 1


def test_capture_serves_only_shapes_that_ran_eagerly(monkeypatch):
    runtime = _Runtime()
    _install(monkeypatch, runtime)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert flydsl.flydsl_paged_decode(*_call()) is None  # cold: no compile in capture
    assert runtime.calls == []

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert flydsl.flydsl_paged_decode(*_call()) == "flydsl"  # eager: compiles, warms

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert flydsl.flydsl_paged_decode(*_call()) == "flydsl"
    assert flydsl.flydsl_paged_decode(*_call(nnz=128)) is None  # another shape is cold
