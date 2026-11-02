# SPDX-License-Identifier: MIT
# Tests for the MLA decode query-head floor (ATOM_MLA_MIN_HEADS).

import pytest

from atom.model_ops.mla_decode_heads import MLA_MIN_HEADS, mla_decode_min_heads
from atom.utils import envs


def _floor(monkeypatch, value, kv="fp8", sparse=False, dcp=1):
    if value is None:
        monkeypatch.delenv("ATOM_MLA_MIN_HEADS", raising=False)
    else:
        monkeypatch.setenv("ATOM_MLA_MIN_HEADS", value)
    return mla_decode_min_heads(kv, sparse, dcp)


def test_env_default_is_16(monkeypatch):
    monkeypatch.delenv("ATOM_MLA_MIN_HEADS", raising=False)
    assert envs.ATOM_MLA_MIN_HEADS == 16 == MLA_MIN_HEADS


@pytest.mark.parametrize("kv", ["fp8", "fp8_e4m3", "auto", "bf16"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("dcp", [1, 2, 8])
def test_flag_off_is_always_16(monkeypatch, kv, sparse, dcp):
    assert _floor(monkeypatch, None, kv, sparse, dcp) == 16
    assert _floor(monkeypatch, "16", kv, sparse, dcp) == 16


def test_12_applies_to_plain_fp8_decode(monkeypatch):
    assert _floor(monkeypatch, "12", "fp8") == 12
    assert _floor(monkeypatch, "12", "fp8_e4m3") == 12


@pytest.mark.parametrize(
    "kv, sparse, dcp",
    [("auto", False, 1), ("bf16", False, 1), ("fp8", True, 1), ("fp8", False, 2)],
)
def test_12_keeps_16_outside_its_domain(monkeypatch, kv, sparse, dcp):
    assert _floor(monkeypatch, "12", kv, sparse, dcp) == 16


@pytest.mark.parametrize("bad", ["8", "13", "32", "0"])
def test_other_values_are_refused(monkeypatch, bad):
    with pytest.raises(ValueError, match="ATOM_MLA_MIN_HEADS"):
        _floor(monkeypatch, bad)
