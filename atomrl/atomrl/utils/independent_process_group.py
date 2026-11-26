# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Create an isolated torch.distributed process group.

The normal ``torch.distributed.new_group`` API can only add ranks from the
default world.  Weight synchronization connects one Megatron actor to separate
vLLM worker processes, so it needs an independent world.  This follows the
multi-main-process-group pattern used by MILES, without importing MILES at
runtime.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
from packaging.version import parse
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


def init_independent_process_group(
    *,
    backend: str | Backend,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int,
    rank: int,
    store: Store | None = None,
    group_name: str,
    pg_options: Any | None = None,
):
    """Create a process group whose rank space is independent of WORLD."""
    if store is not None and init_method is not None:
        raise ValueError("store and init_method are mutually exclusive")
    if world_size <= 0 or rank < 0:
        raise ValueError(
            f"invalid independent group rank={rank}, world_size={world_size}"
        )
    if store is None:
        if init_method is None:
            raise ValueError("init_method is required when store is not supplied")
        iterator = rendezvous(
            init_method,
            rank,
            world_size,
            timeout=timeout or default_pg_timeout,
        )
        store, rank, world_size = next(iterator)
        store.set_timeout(timeout or default_pg_timeout)
        store = PrefixStore(group_name, store)

    backend_obj = Backend(backend)
    options_name = (
        "backend_options"
        if parse(torch.__version__.split("+", 1)[0]) >= parse("2.6")
        else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend_obj,
        store,
        group_name=group_name,
        **{options_name: pg_options},
        timeout=timeout or timedelta(seconds=600),
    )
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
