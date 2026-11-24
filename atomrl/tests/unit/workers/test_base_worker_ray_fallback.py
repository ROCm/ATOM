# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Regression tests for Ray metadata fallbacks in worker startup."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

BASE_WORKER = (
    Path(__file__).resolve().parents[3] / "atomrl" / "workers" / "base_worker.py"
)


def _load_base_worker(monkeypatch, fake_ray):
    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)
    spec = importlib.util.spec_from_file_location("_base_worker_ray_test", BASE_WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ray_gpu_ids_fall_back_on_runtime_context_attribute_error(monkeypatch):
    fake_ray = SimpleNamespace(
        exceptions=SimpleNamespace(RayError=RuntimeError),
        get_runtime_context=lambda: (_ for _ in ()).throw(AttributeError("not ready")),
    )
    module = _load_base_worker(monkeypatch, fake_ray)

    assert module._get_ray_gpu_ids(fake_ray) == []


def test_ray_gpu_ids_fall_back_on_ray_error(monkeypatch):
    class FakeRayError(Exception):
        pass

    fake_ray = SimpleNamespace(
        exceptions=SimpleNamespace(RayError=FakeRayError),
        get_runtime_context=lambda: (_ for _ in ()).throw(FakeRayError("ray down")),
    )
    module = _load_base_worker(monkeypatch, fake_ray)

    assert module._get_ray_gpu_ids(fake_ray) == []


def test_ray_gpu_ids_do_not_auto_initialize_ray(monkeypatch):
    def unexpected_runtime_context():
        raise AssertionError("runtime context must not be queried before ray.init")

    fake_ray = SimpleNamespace(
        exceptions=SimpleNamespace(RayError=RuntimeError),
        is_initialized=lambda: False,
        get_runtime_context=unexpected_runtime_context,
    )
    module = _load_base_worker(monkeypatch, fake_ray)

    assert module._get_ray_gpu_ids(fake_ray) == []
