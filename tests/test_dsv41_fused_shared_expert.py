# SPDX-License-Identifier: MIT
"""CPU checks for the fused shared-expert scale kwargs and the width gate."""

import logging

import atom.models.dsv41_fused_shared_expert as v4


def test_native_mx_scale_kwargs():
    assert v4.fused_scale_kwargs(32) == {
        "quant_block_size": 32,
        "scale_dtype_fmt": "ue8m0",
    }
    assert v4.fused_scale_kwargs(None) == {"transpose_scale": True}


def test_width_gate(monkeypatch):
    monkeypatch.delenv("ATOM_V4_USE_TRITON_FUSION", raising=False)
    assert v4.shared_expert_fused_act_enabled(2304) is False

    monkeypatch.setenv("ATOM_V4_USE_TRITON_FUSION", "1")
    v4._fused_width_fallback_logged = False
    # V4.1-Flash: unsharded 2304 and TP2 1152 pass; TP4 576 and TP8 288 do not.
    assert v4.shared_expert_fused_act_enabled(2304) is True
    assert v4.shared_expert_fused_act_enabled(1152) is True
    assert v4.shared_expert_fused_act_enabled(128) is True
    assert v4.shared_expert_fused_act_enabled(576) is False
    assert v4.shared_expert_fused_act_enabled(288) is False
    assert v4.shared_expert_fused_act_enabled(64) is False


def test_width_fallback_logs_once(monkeypatch, caplog):
    monkeypatch.setenv("ATOM_V4_USE_TRITON_FUSION", "1")
    v4._fused_width_fallback_logged = False
    with caplog.at_level(logging.INFO, logger="atom.models.dsv41_fused_shared_expert"):
        assert v4.shared_expert_fused_act_enabled(576) is False
        assert v4.shared_expert_fused_act_enabled(288) is False
    assert caplog.text.count("Using the unfused path") == 1
    assert "576" in caplog.text
