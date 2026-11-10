# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lightweight state-transfer capabilities, independent of byte codecs."""

from typing import Any


def replays_state(transfer_tensors: Any) -> bool:
    """Whether prefix hits rebuild per-request state instead of restoring it."""
    builder = getattr(transfer_tensors, "state_backend", None)
    state_transfer = getattr(builder, "state_transfer", None)
    return callable(state_transfer) and bool(
        getattr(state_transfer(), "replays", False)
    )
