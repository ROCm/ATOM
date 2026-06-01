# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("atom")
_START = time.perf_counter()


def offload_trace_enabled() -> bool:
    return os.environ.get("OFFLOAD_TRACE_E2E", "0").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def offload_trace(event: str, **fields) -> None:
    if not offload_trace_enabled():
        return
    now = time.perf_counter()
    parts = [f"{key}={value}" for key, value in fields.items()]
    logger.info(
        "[OFFLOAD-TRACE] t=%.6f dt_ms=%.3f event=%s %s",
        now,
        (now - _START) * 1000,
        event,
        " ".join(parts),
    )
