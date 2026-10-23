# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every Qwen3.8-Flash-Next mono build parameter, and the kernels' sources,
reach FlyDSL's JIT cache key: a build that differs in any one is compiled
anew, not served another build's binary. Compile-only for gfx950, no GPU
launch; short stage runs keep each build about a second."""

import inspect
import pathlib

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.qwen4_exp.mono import sources
from atom.models.qwen4_exp.mono.kernels import kernel
from atom.mono.plan.build_key import source_digest
from atom.mono.runtime.compile import compile_only

# Q7 alone (a QSA run may start anywhere after A2): every parameter applies to
# it
BASE = {"stages": ("q7",), "rows": 1}
# per field: another value, and the base it is changed from
OTHER = {
    "stages": ({"stages": ("q6", "q7")}, BASE),
    "experts": ({"experts": 16}, BASE),
    "rows": ({"rows": 2}, BASE),
    # the verify build of a two-row launch
    "seq": ({"seq": True}, {**BASE, "rows": 2}),
    # the grouped verify build of two requests of two rows
    "group": ({"group": 2}, {**BASE, "rows": 4}),
}


def _compile(**kw) -> None:
    kw = dict(kw)
    stages = kw.pop("stages")
    with compile_only():
        # (step counter, layer, scratch, *LAYER_ARGS[1:]): addresses, all 0
        kernel.build_layer(stages, **kw)(0, 0, *([0] * len(kernel.LAYER_ARGS)))


def _entries(cache: pathlib.Path) -> set:
    return {p.relative_to(cache) for p in cache.rglob("*.pkl")}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    # a launcher built earlier keeps its binaries in memory and would not
    # write them to this cache
    kernel.build_layer.cache_clear()
    yield tmp_path
    kernel.build_layer.cache_clear()


def test_every_field_is_listed():
    assert set(OTHER) == set(inspect.signature(kernel.build_layer).parameters)


def test_the_sources_are_the_kernels_package():
    assert sources.SOURCES == source_digest("models/qwen4_exp/mono/kernels")
    assert kernel.SOURCES == sources.SOURCES


def test_a_rebuild_hits_the_cache(cache):
    _compile(**BASE)
    kernel.build_layer.cache_clear()
    _compile(**BASE)
    assert len(_entries(cache)) == 1


@pytest.mark.parametrize("name", sorted(OTHER))
def test_one_field_changed_is_a_new_binary(cache, name):
    change, base = OTHER[name]
    _compile(**base)
    _compile(**{**base, **change})
    assert len(_entries(cache)) == 2


def test_a_source_edit_is_a_new_binary(cache, monkeypatch):
    _compile(**BASE)
    # the digest of edited kernels; the build is otherwise the same
    monkeypatch.setattr(kernel, "SOURCES", "0" * 16)
    kernel.build_layer.cache_clear()
    _compile(**BASE)
    assert len(_entries(cache)) == 2
