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
    "swiglu_limit": 8.0, "debug": True, "xn_flag": 1,
    "shared_mxfp8": True, "mid_flag": 1, "kv_nvfp4": True, "index_nvfp4": True,
    "probe": "loop", "sh_first": 2, "attn_mxfp8": True, "moe_a16w4": False,
    "sp_f32": False, "prod_round": False, "moe_a4w4": True, "front": True,
    "moe_back": True, "back_dn_depth": 2, "back_sh_units": 4, "back_steal": 1,
    "back_dn_split": 1, "back_dn_bdepth": 2, "n_split": 4, "pipe": True,
    "score_sleep": 2, "copy_batch": 4,
}  # fmt: skip
# a field whose other value is a valid build only beside other settings: both builds take them
BESIDE = {
    "moe_a4w4": {
        "shared_mxfp8": True
    },  # the a4w4 experts need the unfused MXFP8 shared expert
    "index_own": {"index_heads": 4},
    "moe_a16w4": {"shared_mxfp8": True},  # compiled out beside the fused shared expert
    "sh_first": {"shared_mxfp8": True, "tokens": 8},  # the wide MoE's (S >= 5) option
}
# More than 16 rows is a front-half or a whole-layer build; the front half is the base of
# these fields (the whole layer for the back half's), and the whole layer ``front``'s.
WIDE = {
    "tokens": 32, "fuse_k1": True, "kv_nvfp4": True, "index_nvfp4": True,
    "shared_mxfp8": True, "attn_mxfp8": True, "moe_a4w4": True,
}  # fmt: skip
WIDE_FIELDS = (
    "front", "moe_back", "back_dn_depth", "back_sh_units", "back_steal",
    "back_dn_split", "back_dn_bdepth", "n_split", "pipe", "score_sleep", "copy_batch",
)  # fmt: skip


def _pair(name: str) -> tuple[K4Build, K4Build]:
    """The two builds that differ in field ``name``."""
    if name in WIDE_FIELDS:
        front = dataclasses.replace(BASE, **WIDE, front=True)
        whole = dataclasses.replace(BASE, **WIDE, moe_back=True)
        if name == "front":  # a front-half build against the whole layer's
            return whole, dataclasses.replace(whole, moe_back=False, front=True)
        if name == "moe_back":
            return front, whole
        base = whole if name.startswith("back_") else front  # the back half's knobs
        return base, dataclasses.replace(base, **{name: OTHER[name]})
    base = dataclasses.replace(BASE, **BESIDE.get(name, {}))
    return base, dataclasses.replace(base, **{name: OTHER[name]})


def _compile(key: K4Build) -> None:
    heads = IndexHeads(key.index_heads, key.index_own)
    launch = build_post_attn_kernel(
        key.npes, key.sm_scale, key.eps, key.route_scale, key.shared_weight,
        key.swiglu_limit, key.init_blocks, key.local_blocks, key.tokens,
        timeline=key.timeline, fuse_k1=key.fuse_k1, heads=heads, debug=key.debug,
        index_topk=key.index_topk, xn_flag=key.xn_flag, shared_mxfp8=key.shared_mxfp8,
        mid_flag=key.mid_flag, kv_nvfp4=key.kv_nvfp4, index_nvfp4=key.index_nvfp4,
        probe=key.probe, sh_first=key.sh_first, attn_mxfp8=key.attn_mxfp8,
        moe_a16w4=key.moe_a16w4, sp_f32=key.sp_f32, prod_round=key.prod_round,
        moe_a4w4=key.moe_a4w4, front=key.front, moe_back=key.moe_back,
        back_dn_depth=key.back_dn_depth, back_sh_units=key.back_sh_units,
        back_steal=key.back_steal, back_dn_split=key.back_dn_split,
        back_dn_bdepth=key.back_dn_bdepth, n_split=key.n_split, pipe=key.pipe,
        score_sleep=key.score_sleep, copy_batch=key.copy_batch,
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
    base, other = _pair(name)
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


def test_k1_attn_mxfp8_is_a_new_binary(cache):
    from atom.models.minimax_m3.mono.kernels.pre_attn import (
        K1_ABI,
        build_pre_attn_kernel,
    )

    for flag in (False, True):
        launch = build_pre_attn_kernel(1e-6, 0.088, 1, 1, 4, attn_mxfp8=flag)
        with compile_only():
            launch(*K1_ABI.zeros())
    assert len(_entries(cache)) == 2
