# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Retiring the NCCL watchdog's pending works before a CUDA graph capture."""

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _world


def drain_nccl_works() -> None:
    """Wait until every NCCL group's watchdog has retired the works it tracks.

    The watchdog polls the end event of each eager collective. A capture that
    records a collective joins its NCCL stream into the capture, and a poll
    landing meanwhile fails with `hipErrorCapturedEvent` and invalidates the
    capture. Finishing the device is not enough: the watchdog drops a work only
    on its next poll. A group with nothing pending returns at once.
    """
    if not dist.is_initialized():
        return
    torch.cuda.synchronize()
    for group, (backend, _) in list(_world.pg_map.items()):
        # Only ProcessGroupNCCL keeps a watchdog list; gloo refuses the call.
        if backend == "nccl":
            group._wait_for_pending_works()
