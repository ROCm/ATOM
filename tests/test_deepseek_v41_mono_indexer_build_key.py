# SPDX-License-Identifier: MIT
"""Standalone indexer ABI, cache isolation and TP resource admission."""

import dataclasses

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.deepseek_v41.mono.kernels import indexer as kernel
from atom.mono.runtime.compile import compile_only, launcher_resources

BASE = kernel.IndexerBuild(8, 8320, fp4=True, mfma=False)
OTHER = {
    "tokens": 16,
    "bound_max": 163968,
    "fp4": False,
    "mfma": True,
    "diag_off": 1 << 24,
    "world": 2,
    "peer_offset": 4096,
    "peer_plain": 1 << 24,
    "timing": True,
    "radix_blocks": False,
    "split_segment": 16384,
}


def _compile(key):
    fn = kernel.build_indexer(key)
    with compile_only():
        fn(*kernel.ABI.zeros())
    return fn


def test_every_indexer_field_is_listed():
    assert set(OTHER) == {f.name for f in dataclasses.fields(BASE)}


def test_adaptive_local_widths_fit_the_shared_tp_arena():
    for world in (2, 4, 8):
        for capacity in (8320, 32768, 163968, 262144):
            wide = dataclasses.replace(
                BASE, tokens=kernel.MAX_ROWS, bound_max=capacity, world=world
            )
            arena = kernel.scratch_bytes(wide)
            for rows in (1, 6, 8):
                local = dataclasses.replace(wide, tokens=rows, world=1)
                assert kernel.scratch_bytes(local) <= arena


@pytest.mark.parametrize("bound", [8320, 16384, 163968, 524288])
def test_reindex_build_keeps_projection_and_bounds_the_compressed_domain(bound):
    key = dataclasses.replace(
        BASE,
        bound_max=bound,
        world=4,
        peer_offset=4096,
        peer_plain=1 << 24,
        timing=True,
    )
    expected = dataclasses.replace(
        key, bound_max=min(bound, 18432), world=1, peer_offset=0, peer_plain=0
    )
    assert kernel.reindex_build(key) == expected


@pytest.mark.parametrize("name", sorted(OTHER))
def test_one_indexer_field_changed_is_a_new_binary(tmp_path, monkeypatch, name):
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    _compile(BASE)
    before = set(tmp_path.rglob("*.pkl"))
    _compile(BASE)
    assert set(tmp_path.rglob("*.pkl")) == before
    _compile(dataclasses.replace(BASE, **{name: OTHER[name]}))
    assert len(set(tmp_path.rglob("*.pkl"))) == len(before) + 1


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 48])
def test_long_fp4_tp_build_has_no_private_scratch(monkeypatch, world, rows):
    monkeypatch.setenv("ARCH", "gfx950")
    key = dataclasses.replace(
        BASE, tokens=rows, bound_max=163968, mfma=True, world=world
    )
    resources = launcher_resources(_compile(key))
    assert resources and all(
        r["scratch"] == 0 and r["lds"] <= 163840 for r in resources
    )
