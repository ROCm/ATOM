# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Admission to AITER's registered-buffer all-gather, for the row gathers that
prefer it over RCCL."""

import torch

from atom.utils import envs


def usable_ca_comm(group):
    """``group``'s custom all-reduce communicator, None if absent or disabled."""
    ca = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    return None if ca is None or ca.disabled else ca


def custom_gather_ok(
    payload: torch.Tensor, group, max_bytes: int | None = None
) -> bool:
    """Whether AITER's registered-buffer all-gather takes ``payload``, and, with
    ``max_bytes``, whether the payload is small enough to prefer it. Reads only
    the env, the group and the payload's shape, so every rank agrees."""
    if max_bytes is not None and payload.numel() * payload.element_size() > max_bytes:
        return False
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca = usable_ca_comm(group)
    return ca is not None and bool(ca.should_custom_ag(payload))
