# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Startup allocation of a paged KV pool that can be exported through HIP IPC."""

import os

import torch


def allocate_shared_kv_buffer(size: int, device: torch.device) -> torch.Tensor:
    """Use native storage for the shared pool, then restore inference settings.

    Mooncake registers an IPC handle even for RDMA in supported CI images.
    HIP IPC cannot export expandable-segment storage. Called only during pool
    initialization, before request/offload threads use this pool; changing the
    process-wide allocator during serving would be unsafe.
    """
    conf = next(
        (
            os.environ[name]
            for name in (
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_HIP_ALLOC_CONF",
            )
            if os.environ.get(name)
        ),
        "",
    )
    fields = conf.split(",")
    expandable = any(
        field.replace(" ", "").lower() == "expandable_segments:true" for field in fields
    )
    if not expandable:
        return torch.zeros(size, dtype=torch.uint8, device=device)
    native = ",".join(
        [
            field
            for field in fields
            if field and field.partition(":")[0].strip() != "expandable_segments"
        ]
        + ["expandable_segments:False"]
    )
    torch.cuda.memory._set_allocator_settings(native)
    try:
        return torch.zeros(size, dtype=torch.uint8, device=device)
    finally:
        torch.cuda.memory._set_allocator_settings(conf)
