# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mixed prefill+decode holds 5% of the device back from the KV cache."""

from types import SimpleNamespace

import pytest

# `model_runner` imports aiter at module scope; same convention as
# test_mixed_spec_decode.py.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine.model_runner import (
    MIXED_KV_RESERVE_FRACTION,
    ModelRunner,
    RapidServeModelRunner,
)

_GiB = 1 << 30
_TOTAL = 288 * _GiB


def _runner(mixed):
    return SimpleNamespace(config=SimpleNamespace(enable_mixed_prefill_decode=mixed))


def test_mixed_reserves_five_percent():
    reserve = ModelRunner._kv_budget_extra_reserve(_runner(True), _TOTAL)
    assert MIXED_KV_RESERVE_FRACTION == 0.05
    assert reserve == int(_TOTAL * 0.05)  # 14.4 GiB, CI's validated 0.9 -> 0.85


def test_no_mixed_reserves_nothing():
    assert ModelRunner._kv_budget_extra_reserve(_runner(False), _TOTAL) == 0


def test_disagg_adds_its_reserve_to_the_mixed_one():
    """The override composes with the base reserve instead of replacing it."""
    runner = RapidServeModelRunner.__new__(RapidServeModelRunner)
    runner.config = SimpleNamespace(enable_mixed_prefill_decode=True)
    disagg = 4 * int(_TOTAL * 0.02)
    assert runner._kv_budget_extra_reserve(_TOTAL) == int(_TOTAL * 0.05) + disagg
    runner.config.enable_mixed_prefill_decode = False
    assert runner._kv_budget_extra_reserve(_TOTAL) == disagg
