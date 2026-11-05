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


@pytest.mark.parametrize(
    "kv_layer, kv_builder, sparse, dcp",
    [
        ("fp8", "fp8_e4m3", False, 1),  # Kimi-K3 TP8 decode: the layer normalises
        ("auto", "auto", True, 1),  # the dtype to "fp8", the builder passes the
        ("fp8", "fp8", False, 8),  # raw config string -- both must still agree.
    ],
)
def test_layer_and_builder_floors_agree(monkeypatch, kv_layer, kv_builder, sparse, dcp):
    # The whole point of the shared helper: MLAAttention and
    # AiterMLAMetadataBuilder must land on the same floor for one decode, even
    # though they spell the KV dtype differently. If they diverge the layer's
    # padded heads and the builder's work descriptors disagree.
    monkeypatch.setenv("ATOM_MLA_MIN_HEADS", "12")
    assert mla_decode_min_heads(kv_layer, sparse, dcp) == mla_decode_min_heads(
        kv_builder, sparse, dcp
    )


@pytest.mark.parametrize(
    "num_heads, expected", [(12, 12), (16, 16), (8, 12), (128, 128)]
)
def test_padded_heads_with_flag_12(monkeypatch, num_heads, expected):
    # The call sites apply max(num_heads, floor): a rank with fewer than 12
    # heads pads up to 12, and a rank already at or above 16 is untouched.
    monkeypatch.setenv("ATOM_MLA_MIN_HEADS", "12")
    floor = mla_decode_min_heads("fp8", is_sparse=False, dcp_world_size=1)
    assert max(num_heads, floor) == expected
