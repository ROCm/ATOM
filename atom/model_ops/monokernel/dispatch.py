# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Construction-time mode selection for native model MonoKernels."""

from __future__ import annotations

from atom.model_ops.monokernel.config import GLM5_GRAPH_BATCHES, glm5_tp_config
from atom.mono.runtime.consensus import MonoUnsupported


def tp_uniform_local_validation(
    error: Exception | None,
    *,
    group,
    world_size: int,
    context: str,
) -> None:
    """Make a local adapter validation result uniform across the TP group."""

    local = None if error is None else f"{type(error).__name__}: {error}"
    if world_size == 1:
        reports = [local]
    else:
        import torch.distributed as dist

        reports = [None] * world_size
        dist.all_gather_object(reports, local, group=group)
    failed = [
        (rank, report) for rank, report in enumerate(reports) if report is not None
    ]
    if failed:
        detail = "; ".join(f"rank {rank}: {report}" for rank, report in failed)
        raise MonoUnsupported(f"{context}: {detail}")


def glm52_native_config(
    *, samples: int, tp_size: int, kv_cache_dtype: str, mtp: bool, query_length: int
):
    """Return the measured scaled-FP4 GLM-5.2 shard geometry, or ``None``."""

    if tp_size != 4 or kv_cache_dtype != "fp8" or not mtp:
        return None
    if (
        query_length != 5
        or samples % query_length
        or samples // query_length not in GLM5_GRAPH_BATCHES
    ):
        return None
    return glm5_tp_config(tp_size)
