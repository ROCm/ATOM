# SPDX-License-Identifier: MIT
"""CPU checks for the fused shared-expert scale kwargs and the width gate.

Load the helper by path. ``import atom.models.deepseek_v4`` runs the model
package, which this check does not need.
"""

import importlib.util
import logging
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[1]
    / "atom/models/deepseek_v4/shared_expert_fusion.py"
)
_spec = importlib.util.spec_from_file_location(
    "atom.models.deepseek_v4.shared_expert_fusion", _PATH
)
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)


def test_native_mx_scale_kwargs():
    assert v4.fused_scale_kwargs(32) == {
        "quant_block_size": 32,
        "scale_dtype_fmt": "ue8m0",
    }
    assert v4.fused_scale_kwargs(None) == {"transpose_scale": True}


def test_width_gate(monkeypatch):
    monkeypatch.delenv("ATOM_V4_USE_TRITON_FUSION", raising=False)
    assert v4.shared_expert_fused_act_enabled(2304, 32) is False  # flag off

    monkeypatch.setenv("ATOM_V4_USE_TRITON_FUSION", "1")
    v4._log_unfused_width.cache_clear()
    # A native MX w2 always fuses: LinearBase rejects input_size % 32 != 0, so
    # the per-rank intermediate is always a whole number of 32-wide groups.
    for width in (2304, 1152, 576, 288, 128, 64, 32):
        assert v4.shared_expert_fused_act_enabled(width, 32) is True
    # Blockscale w2 (128-wide groups): TP4 576 and TP8 288 stay unfused.
    assert v4.shared_expert_fused_act_enabled(2304, None) is True
    assert v4.shared_expert_fused_act_enabled(1152, None) is True
    assert v4.shared_expert_fused_act_enabled(576, None) is False
    assert v4.shared_expert_fused_act_enabled(288, None) is False
    assert v4.shared_expert_fused_act_enabled(64, None) is False


def test_width_fallback_logs_once(monkeypatch, caplog):
    monkeypatch.setenv("ATOM_V4_USE_TRITON_FUSION", "1")
    v4._log_unfused_width.cache_clear()
    with caplog.at_level(logging.INFO, logger=v4.__name__):
        # Only a blockscale w2 can reject a width; TP4/TP8 widths log once each.
        assert v4.shared_expert_fused_act_enabled(576, None) is False
        assert v4.shared_expert_fused_act_enabled(576, None) is False
        assert v4.shared_expert_fused_act_enabled(288, None) is False
    # One log per width: repeated 576 does not log again.
    assert caplog.text.count("Using the unfused path") == 2
    assert "multiple of 128" in caplog.text
    assert "576" in caplog.text
    assert "288" in caplog.text
