# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which path a decode step of n tokens takes (no GPU, no AITER)."""

import pytest

from atom.models.minimax_m3.mono.policy import (
    HYBRID,
    MONO,
    PRODUCTION,
    WidthPolicy,
    parse_spec,
)


def test_default_is_mono_to_16_production_above():
    p = WidthPolicy()
    assert all(p.kind(n) == MONO for n in range(1, 17))
    assert all(p.kind(n) == PRODUCTION for n in range(17, 70))


def test_hybrid_widths_come_from_the_spec_when_the_table_has_none():
    p = WidthPolicy(hybrid=True)  # the measured table has no hybrid width: it never won
    assert p.kind(16) == MONO and all(p.kind(n) == PRODUCTION for n in range(17, 66))
    p = WidthPolicy(hybrid=True, spec="17-64:hybrid")
    assert p.kind(17) == HYBRID and p.kind(64) == HYBRID
    assert p.kind(65) == PRODUCTION


def test_hybrid_entries_need_the_switch_and_the_build_reach():
    spec = "24-64:hybrid"
    assert WidthPolicy(spec=spec).kind(24) == PRODUCTION
    assert (
        WidthPolicy(hybrid=True, spec=spec, hybrid_max_rows=32).kind(33) == PRODUCTION
    )


def test_spec_overrides_and_later_items_win():
    p = WidthPolicy(hybrid=True, spec="40-48:production,16:production,8:mono")
    assert p.kind(40) == p.kind(48) == PRODUCTION
    assert p.kind(39) == HYBRID and p.kind(49) == HYBRID
    assert p.kind(16) == PRODUCTION and p.kind(8) == MONO


def test_measured_wide_table_is_the_32_and_64_row_builds():
    p = WidthPolicy(wide=True)
    assert p.kind(32) == MONO and p.kind(64) == MONO
    assert (
        p.kind(24) == PRODUCTION
        and p.kind(48) == PRODUCTION
        and p.kind(65) == PRODUCTION
    )
    assert (
        WidthPolicy(wide=True, spec="64:production").kind(64) == PRODUCTION
    )  # a run's override
    assert WidthPolicy().kind(32) == PRODUCTION  # the switch is off by default


def test_wide_mono_needs_a_build_and_the_switch():
    spec = "32:mono,64:mono,33:mono"
    assert WidthPolicy(spec=spec, wide_builds=(32, 64)).kind(32) == PRODUCTION
    p = WidthPolicy(wide=True, spec=spec, wide_builds=(32, 64))
    assert p.kind(32) == p.kind(64) == MONO
    assert p.kind(33) == PRODUCTION  # no 33-row build


@pytest.mark.parametrize("bad", ["17:fast", "0:mono", "65:production", "9-3:mono"])
def test_bad_spec_is_refused(bad):
    with pytest.raises(ValueError):
        parse_spec(bad)


def test_describe_runs():
    assert WidthPolicy(hybrid=True, spec="17-64:hybrid").describe() == (
        "1-16:mono, 17-64:hybrid, 65-up:production"
    )
    assert WidthPolicy(wide=True).describe() == (
        "1-16:mono, 17-31:production, 32:mono, 33-63:production, 64:mono, 65-up:production"
    )
    assert WidthPolicy().describe() == "1-16:mono, 17-64:production, 65-up:production"
