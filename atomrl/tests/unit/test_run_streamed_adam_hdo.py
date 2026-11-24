# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU-only behavior checks for the streamed Adam HDO integration script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "integration" / "run_streamed_adam_hdo.py"


def test_streamed_adam_hdo_reports_exact_no_gpu_skip(monkeypatch, capsys) -> None:
    spec = importlib.util.spec_from_file_location("run_streamed_adam_hdo", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == "STREAMED_ADAM_HDO_SKIPPED_NO_GPU\n"


def test_streamed_adam_hdo_requires_optimizer_completion_event() -> None:
    source = _SCRIPT.read_text()

    assert "torch.cuda.Event()" not in source
    assert "completion_event.record(" not in source
    assert "optimizer._lumen_streamed_adam_last_h2d_event" in source
    assert "last_h2d_event.query()" in source


def test_streamed_adam_hdo_checks_public_state_mapping() -> None:
    source = _SCRIPT.read_text()

    assert "set(optimizer.state) == {gpu_parameter}" in source
    assert "cpu_parameter not in optimizer.state" in source
    assert "public_state = optimizer.state[gpu_parameter]" in source
    assert '"master_param"' in source
    assert 'public_state["master_param"] is cpu_parameter' in source
    assert 'public_state["step"] is streamed_state["step"]' in source
    assert 'public_state["exp_avg"] is streamed_state["exp_avg"]' in source
    assert 'public_state["exp_avg_sq"] is streamed_state["exp_avg_sq"]' in source
    assert '_assert_close(public_state["step"], reference_state["step"])' in source
    assert (
        '_assert_close(public_state["exp_avg"], reference_state["exp_avg"])' in source
    )
    assert (
        '_assert_close(public_state["exp_avg_sq"], reference_state["exp_avg_sq"])'
        in source
    )
    assert '_assert_close(public_state["master_param"], reference_parameter)' in source
