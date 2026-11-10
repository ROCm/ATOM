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


def publishes_native_state(transfer_tensors: Any) -> bool:
    """Validate and detect the reusable PAGE-backed native-state contract."""

    spec = getattr(transfer_tensors, "paged_state_checkpoint_spec", None)
    copy = getattr(transfer_tensors, "execute_paged_state_copies", None)
    if spec is None and copy is None:
        return False
    if spec is None:
        raise ValueError(
            "execute_paged_state_copies requires paged_state_checkpoint_spec"
        )
    if not callable(copy):
        raise TypeError(
            "paged_state_checkpoint_spec requires a callable execute_paged_state_copies"
        )
    return True


def copies_paged_state(transfer_tensors: Any) -> bool:
    """Whether the backend keeps PAGE-backed state checkpoints.

    This is the predicate `BlockManager` builds `paged_state_checkpoints` from,
    and that coordinator is what selects the native-state scheduler.
    """

    builder = getattr(transfer_tensors, "state_backend", None)
    state_transfer = getattr(builder, "state_transfer", None)
    return callable(state_transfer) and bool(state_transfer().copies)
