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
    assert v4.shared_expert_fused_act_enabled(2304, native_mx=True) is False

    monkeypatch.setenv("ATOM_V4_USE_TRITON_FUSION", "1")
    v4._fused_width_fallback_logged = False
    # Native MX w2 (V4.1-Flash): whole 32-column groups, so every TP up to 8.
    for width in (2304, 1152, 576, 288, 32):
        assert v4.shared_expert_fused_act_enabled(width, native_mx=True) is True
    assert v4.shared_expert_fused_act_enabled(48, native_mx=True) is False
    assert v4.shared_expert_fused_act_enabled(16, native_mx=True) is False
    # Blockscale w2: whole 128-column groups.
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
    assert "multiple of 128" in caplog.text
