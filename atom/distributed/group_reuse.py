# SPDX-License-Identifier: MIT
# Share one set of RCCL communicators among parallel groups spanning identical ranks.

"""Deduplicate parallel groups whose rank sets coincide.

``GroupCoordinator.__init__`` builds, for every group with ``world_size > 1``:

* ``device_group`` -- ``torch.distributed.new_group(backend="nccl")``
* ``pynccl_comm``  -- a second, independent RCCL communicator over the *same* ranks
* ``ca_comm`` / ``qr_comm`` -- CustomAllreduce / QuickAllReduce IPC buffers

On MI355X (8x288GB) those cost 2.40GB, 2.40GB and 2.00GB respectively, i.e. 6.8GB
per group. When ``tensor_parallel_size == decode_context_parallel_size == world_size``
the TP, DCP and EP groups all span ranks ``{0..N-1}`` yet each builds its own set,
and PCP/PP/DP degenerate to single-rank groups that still pay 0.54GB apiece for a
communicator no message ever crosses. Measured on 8xMI355X with ``-tp 8 -dcp 8``:
12.61GB for the six groups, of which 7.75GB is duplication.

That matters because the KV cache is sized as the *residual* of the memory budget
(see ``ModelRunner.get_num_blocks``): whatever the communicators hold is subtracted
from ``available_for_kv`` one-for-one.

The reuse trick is the one ATOM already applies to the vLLM plugin path in
``atom/plugin/vllm/tp_group_reuse.py`` (PR #1804): skip ``GroupCoordinator.__init__``,
which is what allocates, and graft the fields of an already-built group instead.

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

# Every attribute GroupCoordinator.__init__ assigns; an alias must carry all of
# them or callers reaching past the collective API will find a half-built object.
_COORD_FIELDS = (
    "rank",
    "local_rank",
    "ranks",
    "world_size",
    "rank_in_group",
    "cpu_group",
    "device_group",
    "device",
    "use_device_communicator",
    "device_communicator",
    "mq_broadcaster",
)


@contextlib.contextmanager
def reuse_identical_rank_groups():
    """Make ``init_model_parallel_group`` hand back an alias for a repeated rank set.

    Scoped to the ``init_dist_env`` call: groups built later (eplb migration) keep
    their own communicators. A no-op unless ``ATOM_REUSE_COMM_GROUPS`` is set.
    """
    if not envs.ATOM_REUSE_COMM_GROUPS:
        yield
        return

    from aiter.dist import parallel_state as ps

    class _AliasGroup(ps.GroupCoordinator):
        """A GroupCoordinator sharing another one's communicators.

        Deliberately does not call ``super().__init__`` -- that is the allocating
        path. Only ``unique_name`` is its own, so ``_register_group`` and any
        name-keyed lookup still resolve to distinct entries.
        """

        def __init__(self, source: "ps.GroupCoordinator", group_name: str | None):
            self.unique_name = ps._get_unique_name(group_name or "anonymous")
            ps._register_group(self)
            for field in _COORD_FIELDS:
                setattr(self, field, getattr(source, field))

    built: dict[tuple[tuple[int, ...], ...], "ps.GroupCoordinator"] = {}
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
            alias = _AliasGroup(source, group_name)
            # The broadcaster is a shared-memory queue built over the gloo
            # cpu_group, so it costs no VRAM and there is no reason to make two
            # groups drive one queue. TP is the group that asks for one, and it is
            # also the first built -- refusing to alias over it would strand the
            # largest duplicate.
            if use_message_queue_broadcaster and alias.world_size > 1:
                from aiter.dist.shm_broadcast import MessageQueue

                alias.mq_broadcaster = MessageQueue.create_from_process_group(
                    alias.cpu_group, 1 << 22, 6
                )
            logger.info(
                "group_reuse: %s reuses the communicators of %s (ranks=%s)",
                group_name,
                source.unique_name,
                list(key[0]) if len(key) == 1 else key,
            )
            return alias

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
