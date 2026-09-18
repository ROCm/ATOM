# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Create a torch.distributed process group with its own rank space.

``torch.distributed.new_group`` can only draw ranks from the default world, but
RLHF weight sync joins one *trainer* process to this engine's worker processes --
two separate worlds that share no default group. So the receiving side needs a
group whose rank space is independent of WORLD.

Vendored rather than imported from the RL framework on purpose: ATOM must run
without that framework installed, and a rollout container that only has ATOM
would otherwise fail at group construction rather than at import.

Follows the multi-main-process-group pattern MILES uses, without depending on
MILES. It reaches into ``torch.distributed`` internals because there is no public
API for this; the version guard below is the price of that.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


def _options_keyword() -> str:
    """The name ``_new_process_group_helper`` gives its options argument.

    Renamed from ``pg_options`` to ``backend_options`` in torch 2.6. Parsed off
    the local version rather than hardcoded, so a container upgrade does not
    silently pass the option under a name the helper ignores.
    """
    base = torch.__version__.split("+", 1)[0]
    try:
        major, minor = (int(part) for part in base.split(".")[:2])
    except ValueError:
        # Unparseable version (a nightly or a fork tag): assume current naming.
        return "backend_options"
    return "backend_options" if (major, minor) >= (2, 6) else "pg_options"


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
    """Build a process group whose ranks are independent of the default world.

    Args:
        backend: ``"nccl"`` on ROCm too -- RCCL backs it.
        init_method: A rendezvous URL, e.g. ``tcp://host:port``. Mutually
            exclusive with *store*.
        timeout: Collective timeout. Weight sync moves tens of GB, so the
            default here is deliberately long.
        world_size: Total ranks, counting the trainer.
        rank: This process's rank in that space.
        store: A pre-built store, if the caller already has one.
        group_name: Namespaces the store, so two concurrent groups sharing a
            rendezvous endpoint cannot read each other's keys.
        pg_options: Backend-specific options, passed under whichever keyword
            the local torch expects.
    """
    if store is not None and init_method is not None:
        raise ValueError("store and init_method are mutually exclusive")
    if world_size <= 0 or rank < 0:
        raise ValueError(
            f"invalid independent group rank={rank}, world_size={world_size}"
        )
    if rank >= world_size:
        raise ValueError(
            f"rank {rank} is outside the world it is joining (world_size={world_size})"
        )

    if store is None:
        if init_method is None:
            raise ValueError("init_method is required when store is not supplied")
        iterator = rendezvous(
            init_method, rank, world_size, timeout=timeout or default_pg_timeout
        )
        store, rank, world_size = next(iterator)
        store.set_timeout(timeout or default_pg_timeout)
        # Without the prefix, two groups rendezvousing on one endpoint collide
        # on store keys and one of them hangs.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend(backend),
        store,
        group_name=group_name,
        **{_options_keyword(): pg_options},
        timeout=timeout or timedelta(seconds=600),
    )
    # The helper leaves the global rank map unset for a group built outside the
    # default world; collectives read it, so fill it in. Identity because this
    # group's rank space *is* its own.
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
