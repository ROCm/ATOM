# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in per-request P/D latency probes (``ATOM_PD_TIMING_SAMPLE``).

Each probe is one ``[PD-TIMING]`` line with a wall-clock timestamp. Both
sides sample by the producer's request id (the consumer's ``transfer_id``),
so a sampled request is traced end to end across the two engines.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time

from atom.utils import envs

logger = logging.getLogger("atom")

_phases = threading.local()


def enabled() -> bool:
    return envs.ATOM_PD_TIMING_SAMPLE > 0


def sampled(transfer_id) -> bool:
    n = envs.ATOM_PD_TIMING_SAMPLE
    if n <= 0 or transfer_id is None:
        return False
    try:
        return int(transfer_id) % n == 0
    except (TypeError, ValueError):
        return False


def log(event: str, transfer_id, **fields) -> None:
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    logger.info(
        "[PD-TIMING] ev=%s tid=%s t=%.3f%s", event, transfer_id, time.time(), extra
    )


def phases_begin(timed: bool) -> None:
    """Start (or, untimed, stop) summing ``phase`` time on this thread."""
    _phases.acc = {} if timed else None


@contextlib.contextmanager
def phase(name: str):
    acc = getattr(_phases, "acc", None)
    if acc is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t0


def phases_end() -> dict[str, str]:
    """This thread's summed phases in ms, as ``log`` fields; stops summing."""
    acc = getattr(_phases, "acc", None) or {}
    _phases.acc = None
    return {f"{k}_ms": f"{v * 1e3:.1f}" for k, v in acc.items()}
