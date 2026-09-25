# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_STEP_TIMING_LOG_S: per-step-kind forward wall-time summary."""

import logging
from types import SimpleNamespace

from atom.model_engine import step_timing
from atom.model_engine.step_timing import StepTimingLog


def _batch(prefill, decode):
    return SimpleNamespace(total_seqs_num_prefill=prefill, total_seqs_num_decode=decode)


def test_steps_are_binned_by_kind_and_logged_once_per_interval(monkeypatch, caplog):
    now = [100.0]
    monkeypatch.setattr(step_timing.time, "monotonic", lambda: now[0])
    log = StepTimingLog(10.0)
    with caplog.at_level(logging.INFO, logger="atom"):
        log.record(_batch(1, 0), 0.050, "Engine Core")
        log.record(_batch(0, 4), 0.020, "Engine Core")
        log.record(_batch(1, 3), 0.030, "Engine Core")
        assert not caplog.records  # interval not reached yet
        assert sorted(log.samples) == ["decode", "mixed", "prefill"]
        now[0] = 110.0
        log.record(_batch(0, 4), 0.040, "Engine Core")
    (record,) = caplog.records
    msg = record.getMessage()
    assert msg.startswith("[step-timing] Engine Core ")
    assert "decode: n=2" in msg and "mixed: n=1" in msg and "prefill: n=1" in msg
    assert not log.samples  # a new window starts after each summary
