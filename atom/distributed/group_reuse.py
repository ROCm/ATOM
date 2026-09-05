# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deduplicate parallel groups whose rank sets coincide.

``GroupCoordinator.__init__`` builds, for every group with ``world_size > 1``:

* ``device_group`` -- ``torch.distributed.new_group(backend="nccl")``
* ``pynccl_comm``  -- a second, independent RCCL communicator over the *same* ranks
* ``ca_comm`` / ``qr_comm`` -- CustomAllreduce / QuickAllReduce IPC buffers

When several logical groups (TP/DCP/EP at ``tp == dcp == world_size``, plus the
degenerate single-rank PCP/PP/DP) span the *same* set of ranks, we hand every one
of them back the **same** ``GroupCoordinator`` instance instead of building a fresh
one -- so a single set of RCCL communicators is shared outright.

Trade-off: aliased groups issue their collectives on one communicator, so those
collectives serialize against each other. That is already true for TP/DCP/EP, whose
collectives are issued in sequence from the forward pass. Groups that must stay
isolated -- notably eplb's migration group, which uses ``new_group`` directly so its
isend/irecv cannot cross-match in-flight forward ops -- do not go through
``init_model_parallel_group`` and are therefore untouched.
"""

import contextlib
import logging

from atom.utils import envs

logger = logging.getLogger("atom")


@contextlib.contextmanager
def reuse_identical_rank_groups():
    """Make ``init_model_parallel_group`` hand back the *same* instance for a repeated rank set.

    Scoped to the ``init_dist_env`` call: groups built later (eplb migration) keep
    their own communicators. A no-op unless ``ATOM_REUSE_COMM_GROUPS`` is set
    (default on).
    """
    if not envs.ATOM_REUSE_COMM_GROUPS:
        yield
        return

    from aiter.dist import parallel_state as ps

    built: dict[tuple[tuple[int, ...], ...], ps.GroupCoordinator] = {}
    original = ps.init_model_parallel_group

    def init_or_alias(
        group_ranks,
        local_rank,
        backend,
        use_device_communicator=True,
        use_message_queue_broadcaster=False,
        group_name=None,
    ):
        key = tuple(tuple(r) for r in group_ranks)
        source = built.get(key)
        if (
            source is not None
            and source.use_device_communicator == use_device_communicator
        ):
            # Reuse the whole GroupCoordinator instance -- the caller's global
            # (e.g. _DCP/_EP) ends up pointing at the same object as the source
            # (e.g. _TP), so they share one set of communicators outright.
            if (
                use_message_queue_broadcaster
                and source.world_size > 1
                and source.mq_broadcaster is None
            ):
                # The broadcaster is a shared-memory queue over the gloo
                # cpu_group, so it costs no VRAM. TP asks for one and is built
                # first, so the source normally already has it; only build one
                # here if this rank set's source somehow lacks it.
                from aiter.dist.shm_broadcast import MessageQueue

                source.mq_broadcaster = MessageQueue.create_from_process_group(
                    source.cpu_group, 1 << 22, 6
                )
            logger.info(
                "group_reuse: %s reuses the group instance of %s (ranks=%s)",
                group_name,
                source.unique_name,
                list(key[0]) if len(key) == 1 else key,
            )
            return source

        group = original(
            group_ranks,
            local_rank,
            backend,
            use_device_communicator,
            use_message_queue_broadcaster,
            group_name,
        )
        built[key] = group
        return group

    ps.init_model_parallel_group = init_or_alias
    try:
        yield
    finally:
        ps.init_model_parallel_group = original
