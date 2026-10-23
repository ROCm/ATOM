# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whether the runner thread computes on a High-priority stream.

The decision and the stream construction live apart so the decision can be
exercised on a machine with no GPU build: importing the runner pulls in aiter,
which resolves the chip architecture through rocminfo at import time. Kept free
of heavy imports for that reason.
"""

import logging

from atom.utils import envs

logger = logging.getLogger("atom")


def high_priority_compute_stream_enabled(config) -> bool:
    """Whether `ATOM_HIGH_PRIORITY_COMPUTE_STREAM` applies to this config.

    TBO overrides the flag. UBatchWrapper computes on the runner's stream, and
    a builder's tbo_comm_stream_priority (High on V4.1) puts comm ahead of the
    other ubatch's compute. ROCm has a single level above Normal, so a High
    compute stream would erase that ordering and share comm's queue pool.
    """
    if not envs.ATOM_HIGH_PRIORITY_COMPUTE_STREAM:
        return False
    if config.enable_tbo or config.enable_tbo_decode:
        logger.warning(
            "ATOM_HIGH_PRIORITY_COMPUTE_STREAM is ignored with TBO enabled: "
            "the compute stream would match the TBO comm stream's priority."
        )
        return False
    return True
