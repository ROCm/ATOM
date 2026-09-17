# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Arithmetic behind the KV cache budget.

Kept apart from `model_runner` so it can be tested without a GPU: that module
imports AITER kernels, which a non-GPU runner cannot load.
"""

from __future__ import annotations


def own_non_torch_bytes(
    total: int, free: int, torch_reserved: int, baseline: int | None
) -> int:
    """Device memory this engine holds outside the torch allocator.

    Device-used minus torch-reserved is the only reading available for it, and
    on a shared card it over-reports by whatever every other process holds.
    `gpu_memory_utilization` is this engine's share of the device rather than
    what is left after everyone else, so subtracting a peer's memory from that
    share charges the same bytes twice.

    It also does not stay still. A colocated RL trainer allocates after the
    rollout engine has been sized, so re-sizing on a sleep/wake cycle reads the
    trainer's memory as engine overhead. Measured on Qwen3-30B-A3B at
    utilization 0.45: 33556 blocks at startup against 8316 on the first wake,
    and the decode graphs captured against the first pool then fault when
    recaptured against the second at a 22528-token context.

    `baseline` is the reading taken while this engine was being sized, when the
    number was still its alone; `None` on the first call. Once established it is
    what gets charged, in either direction: our own out-of-allocator buffers are
    allocated during startup and not freed afterwards, so a later reading that
    differs differs because of someone else, and a pool that keeps its size
    across a sleep/wake cycle is the whole point. Where nothing else shares the
    card -- every single-tenant deployment -- the two readings are equal and
    this returns exactly what it always did.
    """
    if baseline is not None:
        return baseline
    return max((total - free) - torch_reserved, 0)
