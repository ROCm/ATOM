# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook's batch bands (``hi_compose.BANDS``): the served widths and bands, the width split past
``MAX_BATCH`` requests, and a band's widths sharing the band's programs (``pwidth`` / ``_key``); without bands every
width keys alone. Pure Python (``hi_bands``, ``hi_compose``): no Triton, AITER or FlyDSL.
"""

import types

import pytest

from atom.models.deepseek_v41.mono import hi_compose

WIDTHS = (6, 12, 18, 24, 30, 36, 42, 48)


def test_served_widths_and_bands():
    """1-4 requests of 6 rows on their own programs, 5-8 on the bands 6 and 8, each band serving widths."""
    assert hi_compose.ROWS == WIDTHS
    assert hi_compose.BANDS == (6, 8)
    assert all(b > hi_compose.MAX_BATCH for b in hi_compose.BANDS)
    plan = hi_compose.plan_widths(
        hi_compose.ROWS,
        hi_compose.REQ_ROWS,
        hi_compose.MAX_BATCH,
        hi_compose.BANDS,
        one_request_max=16,
    )
    band_of = {w: x.band for w, x in plan.items() if x.band is not None}
    assert band_of == {30: 6, 36: 6, 42: 8, 48: 8}
    assert set(band_of.values()) == set(hi_compose.BANDS)


def _split(widths, bands):
    plan = hi_compose.plan_widths(
        widths, hi_compose.REQ_ROWS, hi_compose.MAX_BATCH, bands, one_request_max=16
    )
    return {w: (x.requests, x.req_rows) for w, x in plan.items()}


def test_split_past_max_batch():
    with pytest.raises(ValueError):
        _split(WIDTHS, ())  # (no bands: 5+ requests refused)
    assert _split(WIDTHS, (6, 8)) == {w: (w // 6, 6) for w in WIDTHS}
    assert _split((6, 12, 24), ()) == _split((6, 12, 24), (8,))


def _layers(bands):
    m = object.__new__(hi_compose.HiPrograms)  # (only what ``_key`` / ``pwidth`` read)
    m.L, m.tp, m.chain, m.aux_ids, m.req_rows = types.SimpleNamespace(), 4, False, (), 6
    m.split = _split(WIDTHS, bands) if bands else {w: (w // 6, 6) for w in WIDTHS[:4]}
    m.band_of = {
        s: min(b for b in bands if b >= nb)
        for s, (nb, _) in m.split.items()
        if nb > hi_compose.MAX_BATCH
    }
    m.plan = {i: types.SimpleNamespace(program_key=("consumer", 128)) for i in range(3)}
    return m


def test_band_widths_share_programs():
    m = _layers((6, 8))
    assert {s: m.pwidth(s) for s in WIDTHS} == {
        6: 6,
        12: 12,
        18: 18,
        24: 24,
        30: 36,
        36: 36,
        42: 48,
        48: 48,
    }
    keys = {s: m._key(0, s) for s in WIDTHS}
    assert keys[30] == keys[36] != keys[42] == keys[48]
    # the static widths: one program a width
    assert len({keys[s] for s in WIDTHS[:4]}) == 4


def test_no_bands_keys():
    m = _layers(())
    assert {s: m._key(1, s) for s in WIDTHS[:4]} == {
        s: (("consumer", 128), s, False, False) for s in WIDTHS[:4]
    }
