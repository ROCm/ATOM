# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging

from atom.kv_transfer.disaggregation import pd_timing


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_PD_TIMING_SAMPLE", raising=False)
    assert not pd_timing.enabled()
    assert not pd_timing.sampled(0)


def test_samples_every_nth_transfer_id(monkeypatch):
    monkeypatch.setenv("ATOM_PD_TIMING_SAMPLE", "4")
    assert pd_timing.enabled()
    assert [t for t in range(10) if pd_timing.sampled(t)] == [0, 4, 8]
    assert pd_timing.sampled("12")
    assert not pd_timing.sampled(None)
    assert not pd_timing.sampled("not-an-id")


def test_log_line(caplog):
    with caplog.at_level(logging.INFO, logger="atom"):
        pd_timing.log("D_wdone", 8, stage=3, ok=1)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert msg.startswith("[PD-TIMING] ev=D_wdone tid=8 t=")
    assert msg.endswith(" stage=3 ok=1")
