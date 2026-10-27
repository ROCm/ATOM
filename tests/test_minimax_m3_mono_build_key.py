# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every K4 build parameter reaches FlyDSL's JIT cache key: a build that differs
in any one field is compiled anew, not served another build's binary (the four TP
ranks once all ran rank 3's kernel). Compile-only for gfx950, no GPU launch."""

import dataclasses
import pathlib

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.minimax_m3.mono.config import IndexHeads
from atom.models.minimax_m3.mono.kernels.dense_post import (
    DENSE_POST_ABI,
    DensePostBuild,
    build_dense_post_kernel,
)
from atom.models.minimax_m3.mono.kernels.post_attn import (
    K4_ABI,
    K4Build,
    build_post_attn_kernel,
)
from atom.mono.runtime.compile import compile_only

BASE = K4Build(
    npes=4, tokens=1, init_blocks=1, local_blocks=1, index_heads=1, index_own=0,
    timeline=False, index_topk=True, fuse_k1=True, sm_scale=0.088, eps=1e-6,
    route_scale=1.0, shared_weight=1.0, swiglu_limit=7.0, debug=False,
)  # fmt: skip
# one other value per field (index_heads with its own: a CP build of rank 0)
OTHER = {
    "npes": 2, "tokens": 2, "init_blocks": 2, "local_blocks": 2, "index_heads": 4,
    "index_own": 1, "timeline": True, "index_topk": False, "fuse_k1": False,
    "sm_scale": 0.09, "eps": 1e-5, "route_scale": 2.0, "shared_weight": 0.5,
    "swiglu_limit": 8.0, "debug": True,
}  # fmt: skip


def _compile(key: K4Build) -> None:
    heads = IndexHeads(key.index_heads, key.index_own)
    launch = build_post_attn_kernel(
        key.npes, key.sm_scale, key.eps, key.route_scale, key.shared_weight,
        key.swiglu_limit, key.init_blocks, key.local_blocks, key.tokens,
        timeline=key.timeline, fuse_k1=key.fuse_k1, heads=heads, debug=key.debug,
        index_topk=key.index_topk,
    )  # fmt: skip
    with compile_only():
        launch(*K4_ABI.zeros())


def _entries(cache: pathlib.Path) -> set:
    return {p.relative_to(cache) for p in cache.rglob("*.pkl")}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_every_field_is_listed():
    assert set(OTHER) == {f.name for f in dataclasses.fields(K4Build)}


def test_a_rebuild_hits_the_cache(cache):
    _compile(BASE)
    _compile(BASE)
    assert len(_entries(cache)) == 1


@pytest.mark.parametrize("name", sorted(OTHER))
def test_one_field_changed_is_a_new_binary(cache, name):
    other = dataclasses.replace(BASE, **{name: OTHER[name]})
    if name == "index_own":
        other = dataclasses.replace(other, index_heads=4)
        base = dataclasses.replace(BASE, index_heads=4)
    else:
        base = BASE
    _compile(base)
    _compile(other)
    assert len(_entries(cache)) == 2


DENSE_BASE = DensePostBuild(
    npes=4, tokens=1, index_heads=1, timeline=False, eps=1e-6, swiglu_alpha=1.702,
    swiglu_beta=1.0, swiglu_limit=7.0, debug=False,
)  # fmt: skip
DENSE_OTHER = {
    "npes": 2, "tokens": 2, "index_heads": 4, "timeline": True, "eps": 1e-5,
    "swiglu_alpha": 1.5, "swiglu_beta": 0.5, "swiglu_limit": 8.0, "debug": True,
}  # fmt: skip


def _compile_dense(key: DensePostBuild) -> None:
    launch = build_dense_post_kernel(
        key.npes, key.eps, key.swiglu_alpha, key.swiglu_beta, key.swiglu_limit,
        tokens=key.tokens, debug=key.debug, index_heads=key.index_heads,
        timeline=key.timeline,
    )  # fmt: skip
    with compile_only():
        launch(*DENSE_POST_ABI.zeros())


def test_every_dense_field_is_listed():
    assert set(DENSE_OTHER) == {f.name for f in dataclasses.fields(DensePostBuild)}


@pytest.mark.parametrize("name", sorted(DENSE_OTHER))
def test_one_dense_field_changed_is_a_new_binary(cache, name):
    _compile_dense(DENSE_BASE)
    _compile_dense(dataclasses.replace(DENSE_BASE, **{name: DENSE_OTHER[name]}))
    assert len(_entries(cache)) == 2
