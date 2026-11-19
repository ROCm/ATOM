# SPDX-License-Identifier: MIT
"""GPU compiler resource and cache identity checks; geometry is tested on CPU."""

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
    "full_only": True,
}


def _compile(key):
    fn = kernel.build_indexer(key)
    with compile_only():
        fn(*kernel.ABI.zeros())
    return fn


@pytest.mark.parametrize(
    "world,rows", [(2, 1), (2, 48), (4, 12), (4, 16), (4, 48), (8, 1), (8, 48)]
)
def test_full_tp_binary_has_no_private_scratch(monkeypatch, world, rows):
    monkeypatch.setenv("ARCH", "gfx950")
    key = kernel.full_build(
        dataclasses.replace(
            BASE, tokens=rows, bound_max=163968, mfma=True, world=world
        ),
        2,
    )
    resources = launcher_resources(_compile(key))
    assert resources and all(
        r["scratch"] == 0 and r["lds"] <= 163840 for r in resources
    )


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
