# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every V4.1 mono build parameter reaches FlyDSL's JIT cache key: a build that
differs in any one field is compiled anew, not served another build's binary.
Compile-only for gfx950, no GPU launch."""

import dataclasses
import pathlib

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.deepseek_v41.mono.kernels import attn_post as k2a
from atom.models.deepseek_v41.mono.kernels import attn_pre as k1
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.kernels.moe_shape import MoeBuild
from atom.mono.runtime.compile import compile_only

# a diag offset past every build's regions
DIAG = 1 << 24
# per build: its builder, ABI, base key and one other value per field
BUILDS = {
    "k1": (
        k1.build_attn_pre, k1.ABI, k1.AttnPreBuild(tokens=6),
        {"tokens": 12, "fold": False, "feeds_index": True, "index_fp4": True,
         "aux": True, "tp": 2, "timeline": True, "diag_off": DIAG},
    ),
    "k2a": (
        k2a.build_attn_post, k2a.ABI, k2a.AttnPostBuild(tokens=6),
        {"tokens": 12, "index_bound_max": 65536, "index_fp4": True, "tp": 2,
         "timeline": True, "diag_off": DIAG},
    ),
    "k2b": (
        k2b.build_moe, k2b.ABI, MoeBuild(tokens=6),
        {"tokens": 12, "experts": 128, "topk": 3, "tp": 2, "timeline": True,
         "diag_off": DIAG, "ug_groups": 1},
    ),
    "k2": (
        k2.build_layer_post, k2.ABI, k2.LayerPostBuild(tokens=6),
        {"tokens": 12, "index_bound_max": 65536, "index_fp4": True, "experts": 128,
         "topk": 3, "tp": 2, "timeline": True, "diag_off": DIAG, "ug_groups": 1},
    ),
}  # fmt: skip
CASES = [(b, f) for b, (_, _, _, other) in BUILDS.items() for f in sorted(other)]


def _compile(build: str, key) -> None:
    builder, abi = BUILDS[build][:2]
    with compile_only():
        builder(key)(*abi.zeros())


def _entries(cache: pathlib.Path) -> set:
    return {p.relative_to(cache) for p in cache.rglob("*.pkl")}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("build", sorted(BUILDS))
def test_every_field_is_listed(build):
    _, _, base, other = BUILDS[build]
    assert set(other) == {f.name for f in dataclasses.fields(base)}


@pytest.mark.parametrize("build", sorted(BUILDS))
def test_a_rebuild_hits_the_cache(cache, build):
    base = BUILDS[build][2]
    _compile(build, base)
    _compile(build, base)
    assert len(_entries(cache)) == 1


@pytest.mark.parametrize(("build", "name"), CASES)
def test_one_field_changed_is_a_new_binary(cache, build, name):
    _, _, base, other = BUILDS[build]
    _compile(build, base)
    _compile(build, dataclasses.replace(base, **{name: other[name]}))
    assert len(_entries(cache)) == 2
