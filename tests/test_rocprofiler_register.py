# SPDX-License-Identifier: MIT
# Tests for atom/rocprofiler_register.py -- pure stdlib, no GPU.
#
# drop_inherited_registration_for_children() removes an inherited
# ROCPROFILER_REGISTER_LIBRARY claim in the parent before it spawns GPU
# workers, so each worker registers its own kineto client and torch.profiler
# captures GPU events (PyTorch 2.12+/rocprofiler-sdk). These tests cover the
# three branches: drop, opt-out, and the non-ROCm no-op.

import os

import pytest

from atom import rocprofiler_register as rr

_VAR = "ROCPROFILER_REGISTER_LIBRARY"
_KEEP = "ATOM_KEEP_INHERITED_ROCPROFILER_REGISTER"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_VAR, raising=False)
    monkeypatch.delenv(_KEEP, raising=False)


def test_drops_inherited_claim_on_rocm(monkeypatch):
    monkeypatch.setattr(rr, "torch_built_with_hip", lambda: True)
    monkeypatch.setenv(_VAR, "/opt/rocm/lib/librocprofiler-sdk.so")

    rr.drop_inherited_registration_for_children()

    assert _VAR not in os.environ


def test_noop_when_torch_is_not_rocm(monkeypatch):
    monkeypatch.setattr(rr, "torch_built_with_hip", lambda: False)
    monkeypatch.setenv(_VAR, "/some/lib.so")

    rr.drop_inherited_registration_for_children()

    assert os.environ[_VAR] == "/some/lib.so"


def test_opt_out_keeps_claim(monkeypatch):
    monkeypatch.setattr(rr, "torch_built_with_hip", lambda: True)
    monkeypatch.setenv(_VAR, "/some/lib.so")
    monkeypatch.setenv(_KEEP, "1")

    rr.drop_inherited_registration_for_children()

    assert os.environ[_VAR] == "/some/lib.so"


@pytest.mark.parametrize("value", ["0", "", "false", "no", "off", "off "])
def test_falsey_opt_out_still_drops(monkeypatch, value):
    # The opt-out goes through envs._flag_env, so "off"/"0"/empty read as off
    # (the flag is NOT set) and the claim is still dropped.
    monkeypatch.setattr(rr, "torch_built_with_hip", lambda: True)
    monkeypatch.setenv(_VAR, "/some/lib.so")
    monkeypatch.setenv(_KEEP, value)

    rr.drop_inherited_registration_for_children()

    assert _VAR not in os.environ


def test_pop_is_safe_when_var_absent(monkeypatch):
    monkeypatch.setattr(rr, "torch_built_with_hip", lambda: True)

    # No ROCPROFILER_REGISTER_LIBRARY in the environment: must not raise.
    rr.drop_inherited_registration_for_children()

    assert _VAR not in os.environ
