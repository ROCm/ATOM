# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

from atomrl.utils.metrics import MetricsTracker


def test_update_and_mean() -> None:
    m = MetricsTracker()
    m.update("loss", 1.0)
    m.update("loss", 3.0)
    assert m.get_mean("loss") == 2.0


def test_reset() -> None:
    m = MetricsTracker()
    m.update("x", 5.0)
    m.reset()
    assert m.get_mean("x") == 0.0
