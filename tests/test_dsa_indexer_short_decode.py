from dataclasses import replace

import pytest

from atom.model_ops.indexer.layout import MFMA
from atom.mono.dsa_indexer.build import IxBuild, unfit
from atom.mono.dsa_indexer.plan import IndexerShape


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 16, 24, 32, 40, 64])
def test_short_decode_layout_and_cache_identity(rows):
    shape = IndexerShape(32, 128, 2048, 1536, "bf16", "fp4", 64, 0, cache_layout=MFMA)
    old = IxBuild(shape, rows, 8, 163840, replicated=True, short_gemm=True)
    new = replace(old, short_decode=True)
    assert not unfit(new)
    assert new.short_owned_verify and new.own_rows
    assert new.name != old.name
    assert new.row_keys >= 8192
    assert not replace(new, replicated=False).short_owned_verify
    assert not replace(new, world=1).short_owned_verify
    assert not replace(new, max_ctx=262145).short_owned_verify
    assert not replace(new, tokens=65).short_owned_verify


def test_production_short_decode_requires_explicit_switch(monkeypatch):
    from atom.mono.dsa_indexer.build import short_decode_enabled

    shape = IndexerShape(32, 128, 2048, 1536, "bf16", "fp4", 64, 0, cache_layout=MFMA)
    args = (shape, 8, 8, 163840, True, 1)
    monkeypatch.delenv("ATOM_MONO_K3_SHORT_DECODE", raising=False)
    assert not short_decode_enabled(*args)
    monkeypatch.setenv("ATOM_MONO_K3_SHORT_DECODE", "1")
    assert short_decode_enabled(*args)
    assert not short_decode_enabled(shape, 8, 8, 163840, True, 6)
    assert not short_decode_enabled(shape, 8, 8, 163840, False, 1)
    monkeypatch.setenv("ATOM_MONO_K3_SHORT_DECODE", "0")
    assert not short_decode_enabled(*args)
