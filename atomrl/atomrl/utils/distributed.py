# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Thin wrappers around ``torch.distributed`` for optional multi-process runs."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

logger = logging.getLogger(__name__)


def get_rank() -> int:
    """Return the distributed rank, or ``0`` if not initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return int(dist.get_rank())


def get_world_size() -> int:
    """Return the distributed world size, or ``1`` if not initialized."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def all_gather_tensors(
    tensor: Tensor, group: dist.ProcessGroup | None = None
) -> list[Tensor]:
    """All-gather a tensor from every rank into a list ordered by rank."""
    if not dist.is_available() or not dist.is_initialized():
        return [tensor.detach().clone()]

    world = dist.get_world_size()
    gathered: list[Tensor] = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor, group=group)
    return gathered


def broadcast_state_dict(
    state_dict: dict[str, Any],
    src_rank: int = 0,
    group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """Broadcast a picklable ``state_dict`` object from ``src_rank``.

    Uses ``torch.distributed`` object broadcast when initialized; otherwise
    returns a shallow copy of the input on all "ranks" (single process).
    """
    if not dist.is_available() or not dist.is_initialized():
        return dict(state_dict)

    obj_list: list[dict[str, Any] | None]
    if get_rank() == src_rank:
        obj_list = [state_dict]
    else:
        obj_list = [None]
    dist.broadcast_object_list(obj_list, src=src_rank, group=group)
    out = obj_list[0]
    if out is None:
        raise RuntimeError(
            "broadcast_state_dict failed: missing payload on non-src rank"
        )
    logger.debug("broadcast_state_dict: src_rank=%d keys=%d", src_rank, len(out))
    return out
