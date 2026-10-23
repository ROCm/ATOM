# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mixed prefill+decode holds ATOM_MIXED_KV_RESERVE_FRACTION back from the KV cache."""

from types import SimpleNamespace

import pytest

# `model_runner` imports aiter at module scope; same convention as
# test_mixed_spec_decode.py.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine.model_runner import ModelRunner, RapidServeModelRunner

_GiB = 1 << 30
_TOTAL = 288 * _GiB


def _runner(mixed):
    return SimpleNamespace(config=SimpleNamespace(enable_mixed_prefill_decode=mixed))


def test_mixed_reserves_five_percent_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_MIXED_KV_RESERVE_FRACTION", raising=False)
    reserve = ModelRunner._kv_budget_extra_reserve(_runner(True), _TOTAL)
    assert reserve == int(_TOTAL * 0.05)  # 14.4 GiB, CI's validated 0.9 -> 0.85


def test_no_mixed_reserves_nothing(monkeypatch):
    monkeypatch.delenv("ATOM_MIXED_KV_RESERVE_FRACTION", raising=False)
    assert ModelRunner._kv_budget_extra_reserve(_runner(False), _TOTAL) == 0


@pytest.mark.parametrize("fraction,expected", [("0", 0), ("0.07", int(_TOTAL * 0.07))])
def test_the_fraction_is_overridable(monkeypatch, fraction, expected):
    monkeypatch.setenv("ATOM_MIXED_KV_RESERVE_FRACTION", fraction)
    assert ModelRunner._kv_budget_extra_reserve(_runner(True), _TOTAL) == expected


def test_disagg_adds_its_reserve_to_the_mixed_one(monkeypatch):
    """The override composes with the base reserve instead of replacing it."""
    monkeypatch.delenv("ATOM_MIXED_KV_RESERVE_FRACTION", raising=False)
    runner = RapidServeModelRunner.__new__(RapidServeModelRunner)
    runner.config = SimpleNamespace(enable_mixed_prefill_decode=True)
    disagg = 4 * int(_TOTAL * 0.02)
    assert runner._kv_budget_extra_reserve(_TOTAL) == int(_TOTAL * 0.05) + disagg
    runner.config.enable_mixed_prefill_decode = False
    assert runner._kv_budget_extra_reserve(_TOTAL) == disagg
