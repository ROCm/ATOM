# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which kernel runs the KDA gating update of Kimi-K3 decode and spec verify.

aiter's gfx1250 kernel for the lower-bounded gate over an fp32 state pool on
gfx1250, ATOM's kernel otherwise. AITER, Triton and einops are stood in for
where they are not installed, so these tests run on the CPU job.
"""

import importlib.util
import inspect
import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock

import pytest
from aiter_stub import atom_without_aiter


@contextmanager
def _einops_where_missing():
    """model.py imports einops at the top, and the CPU job does not install it."""
    if "einops" in sys.modules or importlib.util.find_spec("einops") is not None:
        yield
        return
    stand_in = types.ModuleType("einops")
    stand_in.rearrange = MagicMock(name="einops.rearrange")
    sys.modules["einops"] = stand_in
    try:
        yield
    finally:
        sys.modules.pop("einops", None)


@pytest.fixture(scope="module")
def model():
    with atom_without_aiter() as load, _einops_where_missing():
        yield load("atom.models.kimi_k3.model")


@pytest.fixture
def decide(model, monkeypatch):
    """`_use_aiter_kda_gating` for an arch and state pool dtype, decided afresh."""
    monkeypatch.setattr(model, "logger", Mock())

    def decide(lower_bound=-5.0, gfx="gfx1250", ssm_dtype="fp32"):
        monkeypatch.setattr(model, "get_gfx", lambda: gfx)
        monkeypatch.setenv("ATOM_GDN_SSM_DTYPE", ssm_dtype)
        model._use_aiter_kda_gating.cache_clear()
        return model._use_aiter_kda_gating(lower_bound)

    yield decide
    model._use_aiter_kda_gating.cache_clear()


@pytest.mark.parametrize(
    "case, on_aiter",
    [
        ({}, True),
        ({"gfx": "gfx950"}, False),
        ({"lower_bound": None}, False),
        ({"ssm_dtype": "fp16"}, False),
    ],
    ids=["gfx1250-lower-bound-fp32-state", "gfx950", "no-lower-bound", "fp16-state"],
)
def test_route(decide, case, on_aiter):
    assert decide(**case) is on_aiter


def test_route_is_decided_and_logged_once(model, decide, monkeypatch):
    assert decide() is True
    monkeypatch.setattr(model, "get_gfx", lambda: "gfx950")
    assert model._use_aiter_kda_gating(-5.0) is True
    model.logger.info.assert_called_once()


def test_routed_call_that_aiter_declines_raises(model, monkeypatch):
    monkeypatch.setattr(model, "aiter_kda_gating_update", lambda **kwargs: None)
    args = dict.fromkeys(inspect.signature(model._kda_gating_update).parameters)
    args["on_aiter"] = True
    with pytest.raises(RuntimeError, match="declined a call routed to it"):
        model._kda_gating_update(**args)
