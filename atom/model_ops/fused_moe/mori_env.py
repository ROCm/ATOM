# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoRI environment for multi-node EP (M-MOE).

MoRI reads these once, when its shared-memory heap is created, which happens
inside the first collective. They therefore have to be in the environment
before ``init_dist_env`` runs -- setting them later is a no-op that looks like
it worked.

Split into a pure planner and a thin applier so the decisions are testable
without a process to mutate.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("atom")

# InterNodeV1 stages through five buffers (ShmemBufsInterNodeV1) where the
# intra-node path uses one, so the default heap that serves a single node is
# not enough once the group spans a boundary.
_MULTINODE_SHMEM_HEAP = "8G"

_LOOPBACK_IFNAMES = frozenset({"lo", "lo0", "localhost"})


def plan_mori_env(
    *, nnodes: int, current: dict[str, str] | None = None
) -> dict[str, str]:
    """Variables to add for this topology, given what is already exported.

    Never overrides an existing value: an operator who set one did so against a
    specific fabric, and this has less information than they do. Returns only
    the additions, so the caller can log exactly what it changed.
    """
    env = os.environ if current is None else current
    if nnodes <= 1:
        return {}

    additions: dict[str, str] = {}
    if not env.get("MORI_SHMEM_HEAP_SIZE"):
        additions["MORI_SHMEM_HEAP_SIZE"] = _MULTINODE_SHMEM_HEAP
    if not env.get("MORI_EP_LAUNCH_CONFIG_MODE"):
        additions["MORI_EP_LAUNCH_CONFIG_MODE"] = "AUTO"
    return additions


def check_mori_env(*, nnodes: int, current: dict[str, str] | None = None) -> None:
    """Reject settings that cannot work across nodes.

    Only the interface name is checked. It is the one that is both commonly
    wrong -- a single-node run needs ``lo`` and that value tends to survive
    into a multi-node launch script -- and silent when wrong: peers on other
    hosts are simply never reached, so the run hangs in the first collective
    with nothing logged.
    """
    if nnodes <= 1:
        return
    env = os.environ if current is None else current
    ifname = env.get("MORI_SOCKET_IFNAME", "")
    if ifname.lower() in _LOOPBACK_IFNAMES:
        raise ValueError(
            f"MORI_SOCKET_IFNAME={ifname!r} cannot reach {nnodes} nodes.\n"
            f"  Why: loopback resolves to this host, so MoRI's bootstrap never "
            f"sees the other nodes' peers and the first collective hangs with "
            f"nothing in the log. Single-node runs do need this value, which "
            f"is how it survives into a multi-node launcher.\n"
            f"  Fix: set MORI_SOCKET_IFNAME to the interface that carries "
            f"inter-node traffic (`ip -o addr show` on the node)."
        )


def apply_mori_env(*, nnodes: int) -> dict[str, str]:
    """Validate, then export. Returns what it set, for the startup summary."""
    check_mori_env(nnodes=nnodes)
    additions = plan_mori_env(nnodes=nnodes)
    for key, value in additions.items():
        os.environ[key] = value
    if additions:
        logger.info(
            "[wideep] MoRI env for %d nodes: %s",
            nnodes,
            " ".join(f"{k}={v}" for k, v in sorted(additions.items())),
        )
    return additions
