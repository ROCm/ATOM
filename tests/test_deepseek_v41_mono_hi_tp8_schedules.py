# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP8's schedule entries (``schedules.knobs(tp=8)``) and the ``ATOM_DSV41_MONO_HI_KNOBS`` tuning aid."""

import json

import pytest

from atom.models.deepseek_v41.mono.hi.v41 import schedules as s

ENV = "ATOM_DSV41_MONO_HI_KNOBS"
WKV2 = (*s.CONSUMERS, s.IDX1)


@pytest.fixture(autouse=True)
def no_env_knobs(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


@pytest.mark.parametrize("rows", [1, 5, 6, 12, 48])
@pytest.mark.parametrize("kind", s.KINDS)
def test_tp4_keeps_the_kind_rows_entries(kind, rows):
    assert s.knobs(kind, rows, tp=4) == s.knobs(kind, rows)


@pytest.mark.parametrize("kind", s.KINDS)
def test_tp8_one_request_widths(kind):
    """The routed MoE's ug at 4 row groups a task and down at 16 at 6 rows and the draft's 5; wkv in 2 K slices on
    the consumers and the index reader at 6 rows."""
    for rows in (5, 6):
        kn = s.knobs(kind, rows, tp=8)
        assert (kn["MOE_UG_G"], kn["MOE_DOWN_G"]) == (4, 16)
        want = 2 if rows == 6 and kind in WKV2 else s.knobs(kind, rows)["WKV_KSPLIT"]
        assert kn["WKV_KSPLIT"] == want
    assert s.knobs(kind, 12, tp=8) == s.knobs(kind, 12)


def test_a_split_kind_takes_its_base_kinds_tp8_entry():
    kn = s.knobs(s.C1 + s.SPLIT, 6, tp=8)
    assert (kn["MOE_UG_G"], kn["MOE_DOWN_G"], kn["WKV_KSPLIT"]) == (4, 16, 2)


def test_env_knobs_by_kind_most_specific_last(monkeypatch):
    monkeypatch.setenv(
        ENV,
        json.dumps(
            {
                "*": {"MOE_DOWN_G": 8},
                s.WINDOW: {"MOE_UG_G": 2},
                f"{s.WINDOW}@6": {"MOE_UG_G": 1},
            }
        ),
    )
    assert (
        s.knobs(s.WINDOW, 6, tp=8)["MOE_UG_G"],
        s.knobs(s.WINDOW, 6)["MOE_DOWN_G"],
    ) == (1, 8)
    assert s.knobs(s.WINDOW, 5, tp=8)["MOE_UG_G"] == 2
    assert (
        s.knobs(s.C1, 6, tp=8)["MOE_UG_G"],
        s.knobs(s.C1, 6, tp=8)["MOE_DOWN_G"],
    ) == (4, 8)


def test_env_knobs_override_long_ones(monkeypatch):
    """An index source past ``IDX_LONG`` takes its kind's ``LONG`` knobs; the tuning aid's go over them."""
    assert s.LONG[s.KV2]["IDX_SUBTILES"] == 2
    assert s.knobs(s.KV2, 6, tokens=1 << 30)["IDX_SUBTILES"] == 2
    monkeypatch.setenv(ENV, json.dumps({s.KV2: {"IDX_SUBTILES": 1}}))
    assert s.knobs(s.KV2, 6, tokens=1 << 30)["IDX_SUBTILES"] == 1


def test_env_knobs_refuse_an_unknown_knob(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({"*": {"NO_SUCH_KNOB": 1}}))
    with pytest.raises(KeyError, match="NO_SUCH_KNOB"):
        s.knobs(s.WINDOW, 6)
