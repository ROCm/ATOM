# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_MIXED_ALLOC_HEADROOM_GB caps the allocator below the device, not at the budget."""

import logging
from types import SimpleNamespace

import pytest
import torch

# `model_runner` imports aiter at module scope; same convention as
# test_mixed_spec_decode.py.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine.model_runner import ModelRunner

_GiB = 1 << 30
_TOTAL = 288 * _GiB
_NON_TORCH = 15 * _GiB
_BUDGET = int(0.9 * _TOTAL)


def _cap(monkeypatch, *, headroom, mixed=True, non_torch=_NON_TORCH):
    monkeypatch.setenv("ATOM_MIXED_ALLOC_HEADROOM_GB", str(headroom))
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda f: calls.append(f)
    )
    runner = SimpleNamespace(config=SimpleNamespace(enable_mixed_prefill_decode=mixed))
    ModelRunner._cap_allocator_below_device(runner, _TOTAL, non_torch, _BUDGET)
    return calls


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_MIXED_ALLOC_HEADROOM_GB", raising=False)
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda f: calls.append(f)
    )
    runner = SimpleNamespace(config=SimpleNamespace(enable_mixed_prefill_decode=True))
    ModelRunner._cap_allocator_below_device(runner, _TOTAL, _NON_TORCH, _BUDGET)
    assert calls == []


def test_caps_at_device_minus_non_torch_minus_headroom(monkeypatch):
    calls = _cap(monkeypatch, headroom=8)
    assert calls == [pytest.approx((288 - 15 - 8) / 288)]


def test_cap_sits_above_the_budget(monkeypatch):
    """The budget-sized cap OOMed a fitting allocation; this one must not."""
    (fraction,) = _cap(monkeypatch, headroom=8)
    assert fraction * _TOTAL > _BUDGET


def test_without_mixed_nothing_is_capped(monkeypatch):
    assert _cap(monkeypatch, headroom=8, mixed=False) == []


def test_a_headroom_that_would_cap_at_or_below_the_budget_is_refused(
    monkeypatch, caplog
):
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert _cap(monkeypatch, headroom=20) == []
    assert "not capping" in caplog.text
