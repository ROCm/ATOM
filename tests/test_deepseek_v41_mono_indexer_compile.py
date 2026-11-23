# SPDX-License-Identifier: MIT
"""Resource admission for FULL and candidate TP indexer kernels."""

import dataclasses

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
from atom.mono.dsa_indexer.paged import build, kernel
from atom.mono.runtime.compile import compile_only, launcher_resources

BASE = build.IndexerBuild(8, 8320, fp4=True, mfma=False)


def _compile(key):
    fn = kernel.build_indexer(key)
    with compile_only():
        fn(*build.ABI.zeros())
    return fn


@pytest.mark.parametrize("world,rows", [(2, 1), (2, 48), (4, 12), (4, 48), (8, 48)])
def test_full_tp_binary_has_no_private_scratch(monkeypatch, world, rows):
    monkeypatch.setenv("ARCH", "gfx950")
    key = build.full_build(
        dataclasses.replace(
            BASE, tokens=rows, bound_max=163968, mfma=True, world=world
        ),
        2,
    )
    resources = launcher_resources(_compile(key))
    assert resources and all(
        r["scratch"] == 0 and r["lds"] <= 163840 for r in resources
    )


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [48])
def test_long_fp4_tp_build_has_no_private_scratch(monkeypatch, world, rows):
    monkeypatch.setenv("ARCH", "gfx950")
    key = dataclasses.replace(
        BASE, tokens=rows, bound_max=163968, mfma=True, world=world
    )
    resources = launcher_resources(_compile(key))
    assert resources and all(
        r["scratch"] == 0 and r["lds"] <= 163840 for r in resources
    )
