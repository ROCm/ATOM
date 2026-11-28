# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Mono DSA indexer's kernel options (``build.DC1_*``): every
binding builds with ``DC1_DEFAULT``, option 2 only from PACKED_HIST_MIN_ROWS
rows a step, and a build the mask does not fit runs with 0."""

from dataclasses import replace

import pytest

pytest.importorskip("flydsl", reason="the indexer's build keys import FlyDSL")
pytest.importorskip("aiter", reason="the binding imports aiter.dist")

from atom.mono.dsa_indexer import kernel as kix
from atom.mono.dsa_indexer.bind import IndexerBinding
from atom.mono.dsa_indexer.plan import IndexerShape

SHAPE = IndexerShape(32, 128, 2048, 1536, "bf16", "fp4", 64, 0)  # Kimi-K3's
MAX_CTX = 1048576
MASK = kix.DC1_DEFAULT


def test_the_default_is_every_option():
    assert MASK == kix.DC1_ALL == 3315


def _binding(dc1, replicated=False):
    return IndexerBinding({}, SHAPE, 8, MAX_CTX, replicated=replicated, dc1=dc1)


@pytest.mark.parametrize("rows", [5, 10, 15, 20, 25, 40, 60])
def test_option_2_only_from_packed_hist_min_rows(rows):
    key = _binding(MASK).key(rows, 5)
    packed = rows >= kix.PACKED_HIST_MIN_ROWS
    assert key.packed_hist == packed
    assert key.dc1 == (MASK if packed else MASK & ~kix.DC1_PACKED_HIST)
    assert not kix.unfit(key)


def test_a_mask_the_build_does_not_fit_runs_with_0():
    key = _binding(MASK, replicated=True).key(20, 5)
    assert not kix.unfit(key)
    # 64 and 128 need DCP
    assert kix.unfit(replace(key, dc1=MASK))
    assert key.dc1 == 0


def test_dropping_the_options_is_logged_once_naming_them(caplog):
    from atom.mono.dsa_indexer import bind

    bind._log_dc1_dropped.cache_clear()
    binding = _binding(MASK, replicated=True)
    with caplog.at_level("INFO", logger="atom"):
        for _ in range(3):
            assert binding.key(20, 5).dc1 == 0
    dropped = [r.getMessage() for r in caplog.records if "kernel options" in r.message]
    assert len(dropped) == 1
    for name in kix.dc1_names(MASK):
        assert name in dropped[0]
    # A build every option fits logs nothing.
    caplog.clear()
    with caplog.at_level("INFO", logger="atom"):
        assert _binding(MASK).key(20, 5).dc1 == MASK
    assert not [r for r in caplog.records if "kernel options" in r.message]


def test_every_option_has_a_name():
    assert len(kix.dc1_names(kix.DC1_ALL)) == kix.DC1_ALL.bit_count()
