# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in per-request P/D latency probes (``ATOM_PD_TIMING_SAMPLE``).

Each probe is one ``[PD-TIMING]`` line with a wall-clock timestamp. Both
sides sample by the producer's request id (the consumer's ``transfer_id``),
so a sampled request is traced end to end across the two engines.
"""

from __future__ import annotations

import logging
import time

from atom.utils import envs

logger = logging.getLogger("atom")


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
