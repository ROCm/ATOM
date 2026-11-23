# SPDX-License-Identifier: MIT
"""The reader guard preserves new-key and general-mode calls before graph capture."""
import struct
from types import SimpleNamespace

import pytest

from atom.models.deepseek_v41.mono.gfx950.tk import host


def blob(ratio=1, mode=2, score_latent=0, select_latent=0):
    words = [0] * 50
    words[7] = (mode << 32) | ratio
    words[10], words[24] = score_latent, select_latent
    return struct.pack("<50Q", *words)


@pytest.mark.parametrize("symbol", host.INDEXER_MODE2_SYMBOLS)
def test_reader_selection_and_general_fallback(symbol):
    assert host.indexer_mode2_symbol(symbol, blob()) == host.INDEXER_MODE2_SYMBOLS[symbol]
    for args in ({"mode": 0}, {"mode": 1}, {"ratio": 4},
                 {"score_latent": 0x1000}, {"select_latent": 0x2000}):
        assert host.indexer_mode2_symbol(symbol, blob(**args)) == symbol
    assert host.indexer_mode2_symbol(symbol, blob()[:-8]) == symbol
    assert host.indexer_mode2_symbol(symbol, blob() + bytes(8)) == symbol
    multirow = symbol.replace("ELi1EEEv", "ELi6EEEv")
    assert host.indexer_mode2_symbol(multirow, blob()) == multirow


@pytest.mark.parametrize("available", [False, True])
def test_prepared_selects_loaded_reader_and_keeps_immutable_arguments(available):
    symbols = list(host.INDEXER_MODE2_SYMBOLS)
    raw = [blob(), blob(score_latent=0x1000)]
    rank = object.__new__(host.TkRank)
    rank._blobs = {}
    rank.launches = [{"kernel": s, "grid": (256,), "block": (1024,), "shm": 0} for s in symbols]
    rank.blobs = lambda parity: raw
    loaded = {s: None for s in symbols}
    if available:
        loaded.update({s: None for s in host.INDEXER_MODE2_SYMBOLS.values()})
    rank.launcher = SimpleNamespace(functions=loaded, arg_bytes=lambda sym: 400,
                                    params=lambda sym, buf: (sym, bytes(buf)))
    prepared = rank._prepared(0)
    assert prepared[0][0] == (host.INDEXER_MODE2_SYMBOLS[symbols[0]] if available else symbols[0])
    assert prepared[1][0] == symbols[1]
    assert [bytes(row[-1]) for row in prepared] == raw
    assert rank._prepared(0) is prepared
