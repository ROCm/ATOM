# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 hi composition's served widths after a width's programs are refused (``HiPrograms._drop_widths``), without
FlyDSL: the refused widths leave the step's routing (``split``) and the batch bands, the programs only they used are
not kept, and the caches' batch slots follow the widths still served."""

import pytest

from atom.models.deepseek_v41.mono.hi_compose import HiPrograms
from atom.mono.runtime.consensus import MonoUnsupported


def _composition(widths, bands):
    """A composition of ``widths`` (B x 6 rows; ``bands``: width -> its band's requests) with one program a width's
    rows and one verify head a width, as ``HiPrograms`` holds them after its builds."""
    progs = HiPrograms.__new__(HiPrograms)
    progs.req_rows = 6
    progs.band_of = dict(bands)
    progs.split = {s: (s // 6, 6) for s in widths}
    progs.width_plan = dict(progs.split)
    progs.widths = sorted(widths)
    progs.programs = {("layer", progs.pwidth(s)): object() for s in widths}
    progs.head_programs = {s: object() for s in widths}
    progs.head_mode = {s: "wide" for s in widths}
    progs._heads = {(s,): k for s, k in progs.head_programs.items()}
    progs.bmax = progs._most_requests()
    return progs


def test_a_dropped_width_leaves_routing_and_its_programs():
    progs = _composition([6, 12, 18, 30, 36], {30: 6, 36: 6})
    progs._drop_widths([18])
    assert progs.widths == [6, 12, 30, 36]
    assert 18 not in progs.split and 18 not in progs.width_plan
    assert {key[1] for key in progs.programs} == {6, 12, 36}
    assert set(progs.head_programs) == set(progs.head_mode) == {6, 12, 30, 36}
    assert {s for (s,) in progs._heads} == {6, 12, 30, 36}
    assert progs.bmax == 6


def test_a_band_dropped_with_all_its_widths_frees_its_programs_and_slots():
    """Every width of a band shares its programs (the band's rows): they go only when the band's last width does."""
    progs = _composition([6, 12, 30, 36], {30: 6, 36: 6})
    progs._drop_widths([30])
    assert {key[1] for key in progs.programs} == {6, 12, 36}
    progs._drop_widths([36])
    assert {key[1] for key in progs.programs} == {6, 12}
    assert progs.band_of == {}
    assert progs.bmax == 2


def test_no_width_left_is_unsupported():
    with pytest.raises(MonoUnsupported):
        _composition([6], {})._drop_widths([6])
