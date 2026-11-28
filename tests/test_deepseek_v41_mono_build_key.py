# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
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
from atom.models.deepseek_v41.mono.kernels import col_gather as cg
from atom.models.deepseek_v41.mono.kernels import compress as cp
from atom.models.deepseek_v41.mono.kernels import draft_ctx as dc
from atom.models.deepseek_v41.mono.kernels import draft_head as dh
from atom.models.deepseek_v41.mono.kernels import engram as eg
from atom.models.deepseek_v41.mono.kernels import layer_post as k2
from atom.models.deepseek_v41.mono.kernels import moe as k2b
from atom.models.deepseek_v41.mono.kernels.moe_shape import MoeBuild
from atom.mono.runtime.compile import compile_only

# a diag offset past every build's regions
DIAG = 1 << 24
# one region of a build in two copies, a dual area past every region (``numa_plan``)
NUMA = "%s:" + str(1 << 25) + ":0"
# per build: its builder, ABI, base key and one other value per field
BUILDS = {
    "k1": (
        k1.build_attn_pre, k1.ABI, k1.AttnPreBuild(tokens=12),
        {"tokens": 24, "fold": False, "feeds_index": True, "index_fp4": True,
         "aux": True, "tp": 2, "timeline": True, "diag_off": DIAG,
         "tok_par": True, "kv_fp8": True, "numa": NUMA % "lin",
         "shard": True, "embed_in": True},
    ),
    "k2a": (
        k2a.build_attn_post, k2a.ABI, k2a.AttnPostBuild(tokens=12),
        {"tokens": 24, "index_bound_max": 65536, "index_fp4": True, "index_mfma": True, "tp": 2,
         "timeline": True, "diag_off": DIAG, "tok_par": True, "kv_fp8": True,
         "numa": NUMA % "as", "attn_sel": False, "block_rows": 5, "staged_answer": True},
    ),
    "k2b": (
        k2b.build_moe, k2b.ABI, MoeBuild(tokens=6, tp=1),
        {"tokens": 12, "experts": 128, "topk": 3, "tp": 2, "timeline": True,
         "diag_off": DIAG, "ug_groups": 1, "ep": 2, "dp_attention": True,
         "numa": NUMA % "logit", "after_seam": True,
         "own_routes": True},
    ),
    "k2": (
        k2.build_layer_post, k2.ABI, k2.LayerPostBuild(tokens=12, tp=1),
        {"tokens": 24, "index_bound_max": 65536, "index_fp4": True, "index_mfma": True, "experts": 128,
         "topk": 3, "tp": 2, "timeline": True, "diag_off": DIAG, "ug_groups": 1,
         "tok_par": True, "ep": 2, "moe_tp": 2, "kv_fp8": True,
         "numa": NUMA % "as", "shard": True, "attn_sel": False,
         "block_rows": 5, "staged_answer": True},
    ),
    "cg": (
        # taps a settle's pushes fill whole waves with (``ColGatherBuild``)
        cg.build_col_gather, cg.ABI, cg.ColGatherBuild(rows=6, taps=4),
        {"rows": 12, "taps": 2, "state": cg.State.SETTLE, "tp": 2},
    ),
    "eg": (
        eg.build_engram, eg.ABI, eg.EngramBuild(rows=6, tp=4, eps=1e-20),
        {"rows": 12, "tp": 2, "eps": 1e-6, "split": False, "mode": 2, "ctas": 128},
    ),
    "cp": (
        cp.build_compress, cp.ABI,
        cp.CompressBuild(rows=6, ratio=2, eps=1e-6, k_eps=1e-6),
        {"rows": 12, "ratio": 1, "eps": 1e-5, "k_eps": 1e-5, "kv_fp8": True,
         "timeline": True, "quarters": 2},
    ),
    "dh": (
        dh.build_draft_head, dh.ABI,
        dh.DraftHeadBuild(requests=1, positions=5, vocab=2048, hidden=4096, rank_r=256),
        {"requests": 2, "positions": 4, "vocab": 1024, "hidden": 2048,
         "rank_r": 128, "tp": 2, "prefetch": 8, "from_state": True, "eps": 1e-5,
         "confidence": True, "timeline": True},
    ),
    "dc": (
        dc.build_draft_ctx, dc.ABI,
        dc.DraftCtxBuild(rows=6, tp=8, mnorm_eps=1e-20, kv_eps=1e-20),
        {"rows": 12, "tp": 4, "mnorm_eps": 1e-6, "kv_eps": 1e-6},
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
    if (build, name) in (("dh", "eps"), ("dh", "confidence")):
        # the epsilon is code only in a build that makes its rows, and the
        # confidence head reads a row its CTA made
        base = dataclasses.replace(base, from_state=True)
    if name == "staged_answer":
        # code only where the FULL layers' long rows split over the ranks
        base = dataclasses.replace(
            base, index_fp4=True, index_mfma=True, index_bound_max=65536, tp=4
        )
        assert base.index_split
    if (build, name) == ("k1", "embed_in"):
        # layer 0 reads the embedding: it owes no post to fold
        base = dataclasses.replace(base, fold=False)
    if name == "block_rows":
        pytest.skip(
            "a draft block's rows are code only in the layer kernel's attention "
            "(``kernels.layer``), whose builds this test does not compile"
        )
    _compile(build, base)
    _compile(build, dataclasses.replace(base, **{name: other[name]}))
    assert len(_entries(cache)) == 2


@pytest.mark.parametrize("build", ["k1", "k2a", "k2"])
def test_fp8_attention_build_compiles(cache, build):
    """Exercise lane-dependent loads from separate FP8 data/scale planes."""
    key = dataclasses.replace(BUILDS[build][2], tokens=6, tp=4, kv_fp8=True)
    _compile(build, key)
    assert _entries(cache)
