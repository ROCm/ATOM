# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Co-location utilities for placing actor + rollout on the same GPUs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_fused_worker_cls(
    worker_map: dict[str, type],
    name: str = "FusedWorker",
) -> type:
    """Create a worker that fuses multiple worker roles into one actor.

    Exposed methods are prefixed as ``{prefix}_{method}``, e.g.
    ``actor_train_step`` / ``rollout_generate``.
    """
    if not worker_map:
        raise ValueError("worker_map must not be empty for fused workers.")

    class _FusedWorker:
        def __init__(self, rank: int, world_size: int, **kwargs: Any) -> None:
            self._workers = {
                prefix: worker_cls(rank=rank, world_size=world_size, **kwargs)
                for prefix, worker_cls in worker_map.items()
            }

        def __getattr__(self, attr: str) -> Any:
            if "_" not in attr:
                raise AttributeError(attr)
            prefix, method = attr.split("_", 1)
            worker = self._workers.get(prefix)
            if worker is None:
                raise AttributeError(attr)
            target = getattr(worker, method, None)
            if target is None:
                raise AttributeError(attr)
            return target

        def health_check(self) -> dict[str, bool]:
            return {prefix: True for prefix in self._workers}

    _FusedWorker.__name__ = name
    _FusedWorker.__qualname__ = name
    return _FusedWorker


def create_colocated_worker_cls(
    cls_a: type,
    cls_b: type,
    name: str = "ColocatedWorker",
) -> type:
    """Create a Ray actor class that hosts two worker instances on the same GPU.

    This avoids cross-node weight transfer by co-locating training (actor)
    and inference (rollout) workers on the same GPU set.
    """

    return create_fused_worker_cls({"a": cls_a, "b": cls_b}, name=name)
