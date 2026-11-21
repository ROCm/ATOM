# SPDX-License-Identifier: MIT
from atom.model_ops.indexer.layout import MFMA
from atom.mono.dsa_indexer.plan import IndexerShape


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
